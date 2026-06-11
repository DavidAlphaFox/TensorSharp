// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;
using TensorSharp;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    public class Gemma3VisionEncoder : IDisposable
    {
        private readonly Dictionary<string, Tensor> _weights = new();
        private readonly Dictionary<string, Tensor> _transposedWeights = new();
        private readonly IAllocator _allocator;
        private readonly bool _useNativeAttention;
        // Cooperative GpuComputeLock yielding (see Gemma4VisionEncoder).
        private ModelBase _hostModel;
        // 中文：设置宿主模型引用，用于在编码过程中协作式让出 GPU 计算锁。
        public void SetHostModel(ModelBase model) => _hostModel = model;

        private readonly int _imageSize;
        private readonly int _patchSize;
        private readonly int _hiddenSize;
        private readonly int _intermediateSize;
        private readonly int _numHeads;
        private readonly int _blockCount;
        private readonly float _eps;
        private readonly int _projectionDim;
        private readonly int _tokensPerImage;

        public int ProjectionDim => _projectionDim;
        public int TokensPerImage => _tokensPerImage;

        // 中文：构造函数，从 mmproj GGUF 文件读取视觉编码器超参数（图像/patch 尺寸、隐藏维度、头数、层数等）并加载权重。
        public Gemma3VisionEncoder(string mmProjPath, IAllocator allocator)
        {
            _allocator = allocator;
            _useNativeAttention = allocator is GgmlAllocator;
            var gguf = new GgufFile(mmProjPath);

            _imageSize = (int)gguf.GetUint32("clip.vision.image_size", 896);
            _patchSize = (int)gguf.GetUint32("clip.vision.patch_size", 14);
            _hiddenSize = (int)gguf.GetUint32("clip.vision.embedding_length", 1152);
            _intermediateSize = (int)gguf.GetUint32("clip.vision.feed_forward_length", 4304);
            _numHeads = (int)gguf.GetUint32("clip.vision.attention.head_count", 16);
            _blockCount = (int)gguf.GetUint32("clip.vision.block_count", 27);
            _eps = gguf.GetFloat32("clip.vision.attention.layer_norm_epsilon", 1e-6f);
            _projectionDim = (int)gguf.GetUint32("clip.vision.projection_dim", 2560);
            _tokensPerImage = 256;

            Console.WriteLine($"Vision encoder: imageSize={_imageSize}, patchSize={_patchSize}, " +
                $"hidden={_hiddenSize}, intermediate={_intermediateSize}, heads={_numHeads}, " +
                $"blocks={_blockCount}, projDim={_projectionDim}");

            LoadWeights(gguf);
            gguf.Dispose();
        }

        // 中文：从 GGUF 文件读取所有视觉权重张量，按需反量化为 Float32，并将 GGUF 维度顺序反转为 TensorSharp 形状后存入权重字典。
        private void LoadWeights(GgufFile gguf)
        {
            Console.Write("Loading vision encoder weights...");
            int count = 0;
            foreach (var kv in gguf.Tensors)
            {
                var info = kv.Value;
                byte[] raw = gguf.ReadTensorData(info);

                long numElements = info.NumElements;
                float[] f32 = new float[numElements];

                if (info.Type == GgmlTensorType.F32)
                {
                    Buffer.BlockCopy(raw, 0, f32, 0, raw.Length);
                }
                else
                {
                    NativeDequant.DequantizeToFloat32((int)info.Type, raw, 0, f32, 0, numElements);
                }

                long[] ggufShape = new long[info.Shape.Length];
                for (int i = 0; i < info.Shape.Length; i++)
                    ggufShape[i] = (long)info.Shape[i];

                long[] tsShape = new long[ggufShape.Length];
                for (int i = 0; i < ggufShape.Length; i++)
                    tsShape[i] = ggufShape[ggufShape.Length - 1 - i];

                var tensor = new Tensor(_allocator, DType.Float32, tsShape);
                tensor.SetElementsAsFloat(f32);
                _weights[info.Name] = tensor;
                count++;
            }
            Console.WriteLine($" done ({count} tensors)");
        }

        /// <summary>
        /// Encode an image into vision embeddings ready to be injected into the text model.
        /// Input: pixelValues float array of shape [channels * imageSize * imageSize] normalized.
        /// Output: Tensor of shape [tokensPerImage, projectionDim].
        /// </summary>
        // 中文：完整前向流程，将归一化像素编码为视觉嵌入：patch 嵌入→位置嵌入→逐层 Transformer 编码→后置 LayerNorm→多模态投影。
        public unsafe Tensor Encode(float[] pixelValues)
        {
            int numPatches = (_imageSize / _patchSize) * (_imageSize / _patchSize);
            int patchesPerSide = _imageSize / _patchSize;
            int headDim = _hiddenSize / _numHeads;

            bool debug = Environment.GetEnvironmentVariable("DUMP_VISION") == "1";

            var hidden = PatchEmbed(pixelValues, patchesPerSide);
            if (debug) DumpTensor(hidden, "After PatchEmbed", numPatches);

            AddPositionEmbedding(hidden);
            if (debug) DumpTensor(hidden, "After PosEmbed", numPatches);

            for (int i = 0; i < _blockCount; i++)
            {
                Console.Write($"\r  Vision encoder block {i + 1}/{_blockCount}...");
                hidden = EncoderBlock(hidden, i, numPatches, headDim);
                if (debug && (i == 0 || i == _blockCount - 1))
                    DumpTensor(hidden, $"After block {i}", numPatches);
                // Yield GpuComputeLock between encoder blocks (see
                // Gemma4VisionEncoder).
                _hostModel?.YieldGpuComputeLock();
            }
            Console.WriteLine(" done");

            var postNormed = LayerNormOp(hidden, "v.post_ln.weight", "v.post_ln.bias");
            hidden.Dispose();
            if (debug) DumpTensor(postNormed, "After PostLN", numPatches);

            var projected = MultiModalProject(postNormed, patchesPerSide, numPatches);
            postNormed.Dispose();
            if (debug) DumpTensor(projected, "Final projected", (int)projected.Sizes[0]);

            return projected;
        }

        /// <summary>
        /// Conv2D patch embedding: [3, imageSize, imageSize] -> [numPatches, hiddenSize]
        /// Uses the v.patch_embd.weight [patchSize, patchSize, 3, hiddenSize] convolution kernel.
        /// </summary>
        // 中文：以 Conv2D 卷积核手工实现 patch 嵌入，将图像逐 patch 卷积投影为 [numPatches, hiddenSize] 的 token 序列。
        private unsafe Tensor PatchEmbed(float[] pixelValues, int patchesPerSide)
        {
            int numPatches = patchesPerSide * patchesPerSide;
            var result = new Tensor(_allocator, DType.Float32, numPatches, _hiddenSize);
            float* dst = GetFloatPtr(result);

            var convWeight = _weights["v.patch_embd.weight"];
            float* wPtr = GetFloatPtr(convWeight);
            float* biasPtr = _weights.ContainsKey("v.patch_embd.bias")
                ? GetFloatPtr(_weights["v.patch_embd.bias"]) : null;

            int C = 3;
            int P = _patchSize;

            for (int py = 0; py < patchesPerSide; py++)
            {
                for (int px = 0; px < patchesPerSide; px++)
                {
                    int patchIdx = py * patchesPerSide + px;
                    float* outPatch = dst + patchIdx * _hiddenSize;

                    for (int f = 0; f < _hiddenSize; f++)
                    {
                        float sum = biasPtr != null ? biasPtr[f] : 0f;

                        for (int c = 0; c < C; c++)
                        {
                            for (int ky = 0; ky < P; ky++)
                            {
                                for (int kx = 0; kx < P; kx++)
                                {
                                    int imgY = py * P + ky;
                                    int imgX = px * P + kx;
                                    float pixel = pixelValues[c * _imageSize * _imageSize + imgY * _imageSize + imgX];

                                    int wIdx = f * C * P * P + c * P * P + ky * P + kx;
                                    sum += pixel * wPtr[wIdx];
                                }
                            }
                        }

                        outPatch[f] = sum;
                    }
                }
            }

            return result;
        }

        // 中文：将可学习的位置嵌入逐元素加到 patch token 序列上（原地相加）。
        private void AddPositionEmbedding(Tensor hidden)
        {
            var posEmbd = _weights["v.position_embd.weight"];
            Ops.Add(hidden, hidden, posEmbd);
        }

        // 中文：单个 Transformer 编码层，执行 pre-LN 自注意力残差与 pre-LN MLP 残差（LN1→注意力→残差→LN2→MLP→残差）。
        private Tensor EncoderBlock(Tensor hidden, int blockIdx, int numPatches, int headDim)
        {
            string prefix = $"v.blk.{blockIdx}";

            using var ln1 = LayerNormOp(hidden, $"{prefix}.ln1.weight", $"{prefix}.ln1.bias");

            using var attnOut = VisionSelfAttention(ln1, prefix, numPatches, headDim);

            Ops.Add(attnOut, attnOut, hidden);
            hidden.Dispose();

            using var ln2 = LayerNormOp(attnOut, $"{prefix}.ln2.weight", $"{prefix}.ln2.bias");

            using var mlpOut = VisionMLP(ln2, prefix);

            var result = new Tensor(_allocator, DType.Float32, attnOut.Sizes);
            Ops.Add(result, attnOut, mlpOut);

            return result;
        }

        // 中文：视觉多头自注意力，计算 Q/K/V 并做缩放点积注意力（原生路径或手工 batched matmul+softmax），最后经输出投影。
        private Tensor VisionSelfAttention(Tensor input, string prefix, int numPatches, int headDim)
        {
            using var q = LinearForwardWithBias(input, $"{prefix}.attn_q.weight", $"{prefix}.attn_q.bias");
            using var k = LinearForwardWithBias(input, $"{prefix}.attn_k.weight", $"{prefix}.attn_k.bias");
            using var v = LinearForwardWithBias(input, $"{prefix}.attn_v.weight", $"{prefix}.attn_v.bias");

            float scale = 1f / MathF.Sqrt(headDim);

            if (_useNativeAttention)
            {
                using var q4 = q.View(1, numPatches, _numHeads, headDim);
                using var k4 = k.View(1, numPatches, _numHeads, headDim);
                using var v4 = v.View(1, numPatches, _numHeads, headDim);
                using var attn4 = Ops.ScaledDotProductAttention(null, q4, k4, v4, null, scale);
                using var flat = attn4.View(numPatches, _hiddenSize);
                return LinearForwardWithBias(flat, $"{prefix}.attn_out.weight", $"{prefix}.attn_out.bias");
            }

            using var qReshaped = q.View(numPatches, _numHeads, headDim);
            using var kReshaped = k.View(numPatches, _numHeads, headDim);
            using var vReshaped = v.View(numPatches, _numHeads, headDim);

            using var qT0 = qReshaped.Transpose(0, 1);
            using var kT0 = kReshaped.Transpose(0, 1);
            using var vT0 = vReshaped.Transpose(0, 1);
            using var qHeads = Ops.NewContiguous(qT0);
            using var kHeads = Ops.NewContiguous(kT0);
            using var vHeads = Ops.NewContiguous(vT0);

            using var kT = kHeads.Transpose(1, 2);
            var scores = new Tensor(_allocator, DType.Float32, _numHeads, numPatches, numPatches);
            Ops.AddmmBatch(scores, 0, scores, scale, qHeads, kT);
            Ops.Softmax(scores, scores);

            var attnOutput = new Tensor(_allocator, DType.Float32, _numHeads, numPatches, headDim);
            Ops.AddmmBatch(attnOutput, 0, attnOutput, 1.0f, scores, vHeads);
            scores.Dispose();

            using var transposed = attnOutput.Transpose(0, 1);
            using var contiguous = Ops.NewContiguous(transposed);
            using var flatContig = contiguous.View(numPatches, _hiddenSize);
            attnOutput.Dispose();

            return LinearForwardWithBias(flatContig, $"{prefix}.attn_out.weight", $"{prefix}.attn_out.bias");
        }

        // 中文：视觉前馈网络（MLP），先经 ffn_down 升维、GELU 激活，再经 ffn_up 投影回隐藏维度。
        private Tensor VisionMLP(Tensor input, string prefix)
        {
            using var fc1Out = LinearForwardWithBias(input, $"{prefix}.ffn_down.weight", $"{prefix}.ffn_down.bias");
            Ops.GELU(fc1Out, fc1Out);
            return LinearForwardWithBias(fc1Out, $"{prefix}.ffn_up.weight", $"{prefix}.ffn_up.bias");
        }

        /// <summary>
        /// Multi-modal projector: vision output → text space.
        /// Steps: reshape to 2D grid → average pool → RMSNorm → linear projection.
        /// </summary>
        // 中文：多模态投影器，将视觉输出重排为 2D 网格做平均池化下采样，再经 RMSNorm 与线性投影映射到文本嵌入空间。
        private unsafe Tensor MultiModalProject(Tensor visionOutput, int patchesPerSide, int numPatches)
        {
            int kernelSize = patchesPerSide / (int)MathF.Sqrt(_tokensPerImage);
            int pooledSide = patchesPerSide / kernelSize;
            int pooledTokens = pooledSide * pooledSide;

            var pooled = new Tensor(_allocator, DType.Float32, pooledTokens, _hiddenSize);
            float* srcPtr = GetFloatPtr(visionOutput);
            float* dstPtr = GetFloatPtr(pooled);

            for (int py = 0; py < pooledSide; py++)
            {
                for (int px = 0; px < pooledSide; px++)
                {
                    int outIdx = py * pooledSide + px;
                    float* outRow = dstPtr + outIdx * _hiddenSize;

                    for (int d = 0; d < _hiddenSize; d++)
                        outRow[d] = 0;

                    int count = 0;
                    for (int ky = 0; ky < kernelSize; ky++)
                    {
                        for (int kx = 0; kx < kernelSize; kx++)
                        {
                            int srcY = py * kernelSize + ky;
                            int srcX = px * kernelSize + kx;
                            int srcIdx = srcY * patchesPerSide + srcX;
                            float* srcRow = srcPtr + srcIdx * _hiddenSize;
                            for (int d = 0; d < _hiddenSize; d++)
                                outRow[d] += srcRow[d];
                            count++;
                        }
                    }

                    float invCount = 1f / count;
                    for (int d = 0; d < _hiddenSize; d++)
                        outRow[d] *= invCount;
                }
            }

            using var normed = RMSNormOp(pooled, "mm.soft_emb_norm.weight");
            pooled.Dispose();

            var projected = LinearProjection(normed, "mm.input_projection.weight");
            return projected;
        }

        /// <summary>
        /// Linear projection for mm.input_projection: y = x @ W (no bias, no transpose).
        /// The mm.input_projection.weight is stored as [projDim, hiddenSize] in GGUF.
        /// After loading, TensorSharp shape is [hiddenSize, projDim] (reversed).
        /// In Ollama, this weight is transposed before Mulmat (GGML convention), which
        /// effectively computes y = x @ W where W has TensorSharp shape [hiddenSize, projDim].
        /// </summary>
        // 中文：无偏置线性投影 y = x @ W（不转置权重），用于多模态输入投影。
        private Tensor LinearProjection(Tensor input, string weightName)
        {
            var weight = _weights[weightName];
            int seqLen = (int)input.Sizes[0];
            int outDim = (int)weight.Sizes[1];

            var result = new Tensor(_allocator, DType.Float32, seqLen, outDim);
            Ops.Addmm(result, 0, result, 1.0f, input, weight);
            return result;
        }

        // 中文：带偏置的线性层前向，使用缓存的转置权重做矩阵乘，必要时先令输入连续，最后加偏置。
        private unsafe Tensor LinearForwardWithBias(Tensor input, string weightName, string biasName)
        {
            var weight = _weights[weightName];
            int seqLen = (int)input.Sizes[0];
            int outDim = (int)weight.Sizes[0];

            var result = new Tensor(_allocator, DType.Float32, seqLen, outDim);

            Tensor contiguousInput = input.IsContiguous() ? null : Ops.NewContiguous(input);
            Tensor src = contiguousInput ?? input;
            Ops.Addmm(result, 0, result, 1.0f, src, GetOrCreateTransposedWeight(weightName));

            contiguousInput?.Dispose();

            if (_weights.TryGetValue(biasName, out var bias))
                Ops.Add(result, result, bias);

            return result;
        }

        // 中文：按权重名取出 gamma/beta 参数，对输入执行带偏置的 LayerNorm 归一化。
        private Tensor LayerNormOp(Tensor input, string weightName, string biasName)
        {
            _weights.TryGetValue(biasName, out var bias);
            return Ops.LayerNorm(null, input, _weights[weightName], bias, _eps);
        }

        // 中文：对输入执行无偏置的 RMSNorm 归一化（多模态投影前使用）。
        private Tensor RMSNormOp(Tensor input, string weightName)
        {
            return Ops.RMSNorm(null, input, _weights[weightName], null, _eps);
        }

        // 中文：调试辅助，打印张量首行/末行的前若干元素及其 L2 范数，用于核对中间激活值。
        private unsafe void DumpTensor(Tensor t, string label, int numRows)
        {
            float* ptr = GetFloatPtr(t);
            int dim = (int)t.Sizes[1];
            Console.Write($"\n  {label} [{numRows}x{dim}]: row0=[");
            for (int i = 0; i < Math.Min(5, dim); i++)
                Console.Write($"{ptr[i]:F6}{(i < 4 ? ", " : "")}");
            Console.Write($"] last_row=[");
            float* lastRow = ptr + (numRows - 1) * dim;
            for (int i = 0; i < Math.Min(5, dim); i++)
                Console.Write($"{lastRow[i]:F6}{(i < 4 ? ", " : "")}");
            float norm0 = 0, normLast = 0;
            for (int i = 0; i < dim; i++) { norm0 += ptr[i] * ptr[i]; normLast += lastRow[i] * lastRow[i]; }
            Console.WriteLine($"] norm0={MathF.Sqrt(norm0):F4} normLast={MathF.Sqrt(normLast):F4}");
        }

        // 中文：获取张量底层 Float32 数据的原始指针，用于手工逐元素计算。
        private static unsafe float* GetFloatPtr(Tensor t) =>
            TensorComputePrimitives.GetFloatPointer(t);

        // 中文：惰性获取并缓存权重的连续转置版本，避免每次线性层前向重复转置。
        private Tensor GetOrCreateTransposedWeight(string weightName)
        {
            if (_transposedWeights.TryGetValue(weightName, out var transposed))
                return transposed;

            using var weightViewT = _weights[weightName].Transpose();
            transposed = Ops.NewContiguous(weightViewT);
            _transposedWeights[weightName] = transposed;
            return transposed;
        }

        // 中文：释放所有缓存的转置权重与原始权重张量并清空字典，回收资源。
        public void Dispose()
        {
            foreach (var w in _transposedWeights.Values)
                w.Dispose();
            _transposedWeights.Clear();
            foreach (var w in _weights.Values)
                w.Dispose();
            _weights.Clear();
        }
    }
}

