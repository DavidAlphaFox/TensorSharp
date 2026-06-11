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
using System.Diagnostics;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.MLX;

namespace TensorSharp.Models
{
    /// <summary>
    /// Mistral 3 model architecture.
    /// Key features:
    /// - Standard LLaMA-like transformer with SiLU-gated MLP (SwiGLU)
    /// - GPT-J (norm) style RoPE with YaRN scaling for extended context
    /// - Position-dependent Q scaling: q *= (1 + beta * log(1 + floor(pos / orig_ctx)))
    /// - No QK-norm (unlike Qwen3/Gemma3)
    /// - Supports multimodal (vision) via separate Pixtral vision encoder
    /// </summary>
    public partial class Mistral3Model : ModelBase
    {
        // Bound the MLX lazy-graph depth across the per-layer dispatch loop.
        // Default matches Qwen35 (16). Override via TS_MLX_EVAL_EVERY_N_LAYERS.
        private static readonly int MlxEvalEveryNLayers = ResolveMlxEvalEveryNLayers();
        // 中文：从环境变量 TS_MLX_EVAL_EVERY_N_LAYERS 解析 MLX 每隔多少层强制求值一次，默认 16，用于限制 MLX 惰性计算图深度。
        private static int ResolveMlxEvalEveryNLayers()
        {
            string env = Environment.GetEnvironmentVariable("TS_MLX_EVAL_EVERY_N_LAYERS");
            if (!string.IsNullOrWhiteSpace(env) && int.TryParse(env, out int v) && v > 0)
                return v;
            return 16;
        }

        private Tensor[] _kvCacheK;
        private Tensor[] _kvCacheV;

        private string[][] _layerWeightNames;
        private float[] _ropeFreqs;
        private int _ropeDim;
        private int _attnKeyLen;
        private int _attnValLen;

        // YaRN scaling parameters
        private float _ropeScalingBeta;
        private int _ropeOrigCtx;
        private float _ropeExtFactor;
        private float _ropeBetaFast;
        private float _ropeBetaSlow;
        private float _ropeMscale;
        private float _ropeMscaleAllDim;
        private string _ropeType;

        // Vision support
        private Mistral3VisionEncoder _visionEncoder;
        private List<(Tensor embeddings, int position)> _pendingVisionEmbeddingsList = new();

        // 中文：构造函数，从 GGUF 文件加载 Mistral3 模型，解析基础配置与 YaRN/RoPE 参数，载入并融合权重、初始化 KV 缓存并预计算常量。
        public Mistral3Model(string ggufPath, BackendType backend)
            : base(ggufPath, backend)
        {
            string arch = _gguf.GetString("general.architecture") ?? "mistral3";
            Config = new ModelConfig { Architecture = arch };
            ParseBaseConfig();

            _attnKeyLen = Config.KeyLength > 0 ? Config.KeyLength : Config.HeadDim;
            _attnValLen = Config.ValueLength > 0 ? Config.ValueLength : _attnKeyLen;
            _ropeDim = (int)_gguf.GetUint32($"{arch}.rope.dimension_count", (uint)_attnKeyLen);

            // YaRN parameters
            _ropeType = _gguf.GetString($"{arch}.rope.scaling.type", "");
            _ropeScalingBeta = _gguf.GetFloat32($"{arch}.attention.temperature_scale",
                               _gguf.GetFloat32($"{arch}.rope.scaling_beta", 0.1f));
            _ropeOrigCtx = (int)_gguf.GetUint32($"{arch}.rope.scaling.original_context_length", 0);
            Config.OriginalContextLength = _ropeOrigCtx;
            _ropeExtFactor = _gguf.GetFloat32($"{arch}.rope.scaling.extrapolation_factor", 1.0f);
            _ropeBetaFast = _gguf.GetFloat32($"{arch}.rope.scaling.yarn_beta_fast",
                            _gguf.GetFloat32($"{arch}.rope.scaling.beta_fast", 32.0f));
            _ropeBetaSlow = _gguf.GetFloat32($"{arch}.rope.scaling.yarn_beta_slow",
                            _gguf.GetFloat32($"{arch}.rope.scaling.beta_slow", 1.0f));
            _ropeMscale = _gguf.GetFloat32($"{arch}.rope.scaling.mscale", 0f);
            _ropeMscaleAllDim = _gguf.GetFloat32($"{arch}.rope.scaling.mscale_all_dim", 0f);

            Console.WriteLine($"Model: {arch}, Layers={Config.NumLayers}, Hidden={Config.HiddenSize}, " +
                $"Heads={Config.NumHeads}, KVHeads={Config.NumKVHeads}, KeyLen={_attnKeyLen}, " +
                $"ValLen={_attnValLen}, Vocab={Config.VocabSize}");
            Console.WriteLine($"RoPE base={Config.RopeBase}, scale={Config.RopeScale}, type={_ropeType}, " +
                $"dim={_ropeDim}, origCtx={_ropeOrigCtx}");
            if (_ropeType == "yarn")
                Console.WriteLine($"YaRN beta={_ropeScalingBeta}, betaFast={_ropeBetaFast}, " +
                    $"betaSlow={_ropeBetaSlow}, extFactor={_ropeExtFactor}");

            ParseTokenizer();
            LoadWeights();
            FuseQKVWeights();
            FuseGateUpWeights();
            PrepareCudaQuantizedWeightsForInference();

            int maxContextLength = ResolveConfiguredContextLength();
            int initialCacheLength = ResolveInitialCacheAllocationLength(maxContextLength);
            if (initialCacheLength < maxContextLength)
                Console.WriteLine($"Initial {_backend} KV cache allocation: {initialCacheLength} tokens (grows on demand up to {maxContextLength}).");
            InitKVCache(initialCacheLength, maxContextLength);
            PrecomputeConstants();
        }

        // 中文：将每层的 Q/K/V 三个投影权重（量化或浮点）拼接融合为单个 attn_qkv 权重，以减少推理时的矩阵乘次数。
        private unsafe void FuseQKVWeights()
        {
            int fused = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                string qName = $"blk.{l}.attn_q.weight";
                string kName = $"blk.{l}.attn_k.weight";
                string vName = $"blk.{l}.attn_v.weight";
                string qkvName = $"blk.{l}.attn_qkv.weight";

                if (_quantWeights.TryGetValue(qName, out var qw) &&
                    _quantWeights.TryGetValue(kName, out var kw) &&
                    _quantWeights.TryGetValue(vName, out var vw) &&
                    qw.GgmlType == kw.GgmlType && kw.GgmlType == vw.GgmlType &&
                    qw.Ne0 == kw.Ne0 && kw.Ne0 == vw.Ne0)
                {
                    if (!TryCreateFusedQuantizedWeight(out QuantizedWeight fusedWeight, qw, kw, vw))
                        continue;

                    _quantWeights[qkvName] = fusedWeight;
                    _quantWeights.Remove(qName); qw.Dispose();
                    _quantWeights.Remove(kName); kw.Dispose();
                    _quantWeights.Remove(vName); vw.Dispose();
                    fused++;
                }
                else if (_weights.TryGetValue(qName, out var qf) &&
                         _weights.TryGetValue(kName, out var kf) &&
                         _weights.TryGetValue(vName, out var vf))
                {
                    int qDim = (int)qf.Sizes[0], kDim = (int)kf.Sizes[0], vDim = (int)vf.Sizes[0];
                    int inDim = (int)qf.Sizes[1];
                    var fusedTensor = new Tensor(_allocator, DType.Float32, qDim + kDim + vDim, inDim);
                    using (var s0 = fusedTensor.Narrow(0, 0, qDim)) Ops.Copy(s0, qf);
                    using (var s1 = fusedTensor.Narrow(0, qDim, kDim)) Ops.Copy(s1, kf);
                    using (var s2 = fusedTensor.Narrow(0, qDim + kDim, vDim)) Ops.Copy(s2, vf);
                    _weights[qkvName] = fusedTensor;
                    _weights.Remove(qName); qf.Dispose();
                    _weights.Remove(kName); kf.Dispose();
                    _weights.Remove(vName); vf.Dispose();
                    fused++;
                }
            }
            if (fused > 0)
                Console.WriteLine($"  Fused projections: {fused} QKV");
        }

        private bool[] _layerQkvFused;

        // 中文：预计算各层权重名称表（区分是否已融合 QKV）与 RoPE 频率数组，必要时应用 YaRN 频率修正。
        private void PrecomputeConstants()
        {
            int numLayers = Config.NumLayers;
            _layerQkvFused = new bool[numLayers];

            _layerWeightNames = new string[numLayers][];
            for (int l = 0; l < numLayers; l++)
            {
                string p = $"blk.{l}.";
                bool fused = _quantWeights.ContainsKey(p + "attn_qkv.weight") ||
                             _weights.ContainsKey(p + "attn_qkv.weight");
                _layerQkvFused[l] = fused;

                if (fused)
                {
                    _layerWeightNames[l] = new[]
                    {
                        p + "attn_norm.weight",      // 0
                        p + "attn_qkv.weight",        // 1
                        p + "attn_output.weight",     // 2
                        p + "ffn_norm.weight",         // 3
                        p + "ffn_gate_up.weight",      // 4
                        p + "ffn_down.weight",         // 5
                    };
                }
                else
                {
                    _layerWeightNames[l] = new[]
                    {
                        p + "attn_norm.weight",      // 0
                        p + "attn_q.weight",          // 1
                        p + "attn_k.weight",          // 2
                        p + "attn_v.weight",          // 3
                        p + "attn_output.weight",     // 4
                        p + "ffn_norm.weight",         // 5
                        p + "ffn_gate_up.weight",      // 6
                        p + "ffn_down.weight",         // 7
                    };
                }
            }

            int halfDim = _ropeDim / 2;
            float freqScale = 1.0f / Config.RopeScale;
            _ropeFreqs = new float[halfDim];
            for (int i = 0; i < halfDim; i++)
                _ropeFreqs[i] = freqScale / MathF.Pow(Config.RopeBase, (2.0f * i) / _ropeDim);

            if (_ropeType == "yarn" && _ropeOrigCtx > 0)
                ApplyYarnFreqCorrection(_ropeFreqs, halfDim);
        }

        /// <summary>
        /// Apply YaRN frequency correction to precomputed RoPE frequencies for decode path.
        /// Interpolates between extrapolated and interpolated frequencies based on
        /// whether each frequency band is within the "slow" or "fast" rotation range.
        /// </summary>
        // 中文：对预计算的 RoPE 频率应用 YaRN 修正，按波长在高频外推、低频内插与中间平滑混合之间选择，用于扩展上下文长度。
        private void ApplyYarnFreqCorrection(float[] freqs, int halfDim)
        {
            float lowFreqWavelen = (float)(_ropeOrigCtx / _ropeBetaSlow);
            float highFreqWavelen = (float)(_ropeOrigCtx / _ropeBetaFast);

            for (int i = 0; i < halfDim; i++)
            {
                float origFreq = 1.0f / MathF.Pow(Config.RopeBase, (2.0f * i) / _ropeDim);
                float wavelen = 2.0f * MathF.PI / origFreq;

                if (wavelen < highFreqWavelen)
                {
                    // High frequency: use original frequency (extrapolation)
                    freqs[i] = origFreq;
                }
                else if (wavelen > lowFreqWavelen)
                {
                    // Low frequency: use interpolated frequency
                    freqs[i] = origFreq / Config.RopeScale;
                }
                else
                {
                    // Intermediate: smooth blend between interpolated and extrapolated
                    float smooth = (lowFreqWavelen / wavelen - 1.0f) /
                                   (lowFreqWavelen / highFreqWavelen - 1.0f);
                    float interpFreq = origFreq / Config.RopeScale;
                    freqs[i] = (1.0f - smooth) * interpFreq + smooth * origFreq;
                }
            }
        }

        private int _kvCacheCapacity;

        // 中文：初始化每层的 K/V 缓存张量，按初始序列长度分配容量并记录最大上下文长度与 KV 数据类型。
        private void InitKVCache(int initialSeqLen, int maxSeqLen)
        {
            _maxContextLength = maxSeqLen;
            _kvCacheCapacity = initialSeqLen;
            int numKVHeads = Config.NumKVHeads;
            ApplyModelAlignedKvCacheDefault(_quantWeights);
            DType kvDtype = _kvCacheDtype.ToDType();
            _kvCacheK = new Tensor[Config.NumLayers];
            _kvCacheV = new Tensor[Config.NumLayers];
            for (int l = 0; l < Config.NumLayers; l++)
            {
                _kvCacheK[l] = new Tensor(_allocator, kvDtype, numKVHeads, initialSeqLen, _attnKeyLen);
                _kvCacheV[l] = new Tensor(_allocator, kvDtype, numKVHeads, initialSeqLen, _attnValLen);
                InitializeCacheTensor(_kvCacheK[l]);
                InitializeCacheTensor(_kvCacheV[l]);
            }
            _cacheSeqLen = 0;
        }

        // 中文：确保 KV 缓存容量满足所需序列长度，不足时按倍增策略扩容并拷贝已有缓存内容，超过最大上下文则抛异常。
        private void EnsureCacheCapacity(int requiredSeqLen)
        {
            if (requiredSeqLen <= _kvCacheCapacity)
                return;
            if (requiredSeqLen > _maxContextLength)
                throw new InvalidOperationException($"Requested sequence length {requiredSeqLen} exceeds configured max context {_maxContextLength}.");

            int newCapacity = Math.Max(_kvCacheCapacity, 1);
            while (newCapacity < requiredSeqLen)
                newCapacity = Math.Min(_maxContextLength, newCapacity * 2);

            int numKVHeads = Config.NumKVHeads;
            DType kvDtype = _kvCacheDtype.ToDType();
            for (int l = 0; l < Config.NumLayers; l++)
            {
                var newK = new Tensor(_allocator, kvDtype, numKVHeads, newCapacity, _attnKeyLen);
                var newV = new Tensor(_allocator, kvDtype, numKVHeads, newCapacity, _attnValLen);
                InitializeCacheTensor(newK);
                InitializeCacheTensor(newV);

                if (_cacheSeqLen > 0)
                {
                    using var srcK = _kvCacheK[l].Narrow(1, 0, _cacheSeqLen);
                    using var dstK = newK.Narrow(1, 0, _cacheSeqLen);
                    Ops.Copy(dstK, srcK);

                    using var srcV = _kvCacheV[l].Narrow(1, 0, _cacheSeqLen);
                    using var dstV = newV.Narrow(1, 0, _cacheSeqLen);
                    Ops.Copy(dstV, srcV);
                }

                _kvCacheK[l].Dispose();
                _kvCacheV[l].Dispose();
                _kvCacheK[l] = newK;
                _kvCacheV[l] = newV;
            }

            _kvCacheCapacity = newCapacity;
            Console.WriteLine($"Expanded Mistral3 attention cache to {newCapacity} tokens.");
        }

        // 中文：重置所有层的 KV 缓存张量并将缓存序列长度与各项性能计时/前向计数归零。
        public override void ResetKVCache()
        {
            for (int l = 0; l < Config.NumLayers; l++)
            {
                ResetCacheTensor(_kvCacheK[l]);
                ResetCacheTensor(_kvCacheV[l]);
            }
            _cacheSeqLen = 0;
            _linearTicks = _attnTicks = _normTicks = _embTicks = _lmHeadTicks = _logitsCopyTicks = 0;
            _forwardCount = 0;
            _forwardSw.Reset();
        }

        // 中文：将 KV 缓存截断到指定 token 数（调用基类逻辑），并使各层缓存张量的设备端副本失效以便后续重新同步。
        public override void TruncateKVCache(int tokenCount)
        {
            base.TruncateKVCache(tokenCount);
            for (int l = 0; l < Config.NumLayers; l++)
            {
                InvalidateTensorDeviceCache(_kvCacheK[l]);
                InvalidateTensorDeviceCache(_kvCacheV[l]);
            }
        }

        public override bool SupportsKVStateSnapshot => _kvCacheK != null && _kvCacheV != null;

        public override string KVStateFingerprint =>
            $"mistral3|arch={Config.Architecture}|L={Config.NumLayers}|H={Config.NumHeads}|KV={Config.NumKVHeads}|kL={_attnKeyLen}|vL={_attnValLen}|dtype={_kvCacheDtype.ToShortString()}";

        // 中文：计算指定 token 数量对应的 KV 缓存块字节大小，供快照导出/导入分配缓冲区使用。
        public override long ComputeKVBlockByteSize(int tokenCount)
            => KvBlockTransfer.ComputeBlockByteSize(_kvCacheK, _kvCacheV, tokenCount);

        // 中文：从 KV 缓存中提取指定范围的 token 块到目标字节缓冲区（用于状态快照），不支持快照时返回 false。
        public override bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            if (!SupportsKVStateSnapshot)
                return false;
            return KvBlockTransfer.Extract(
                _allocator, _kvCacheK, _kvCacheV, _cacheSeqLen,
                startToken, tokenCount, destination);
        }

        // 中文：将外部字节缓冲区中的 KV 块注入到缓存的指定位置（用于恢复状态），扩容缓存、更新序列长度并使设备端缓存失效。
        public override bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source)
        {
            if (!SupportsKVStateSnapshot)
                return false;
            EnsureCacheCapacity(destToken + tokenCount);
            if (!KvBlockTransfer.Inject(
                    _allocator, _kvCacheK, _kvCacheV, _cacheSeqLen,
                    destToken, tokenCount, source))
            {
                return false;
            }
            _cacheSeqLen = destToken + tokenCount;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                InvalidateTensorDeviceCache(_kvCacheK[l]);
                InvalidateTensorDeviceCache(_kvCacheV[l]);
            }
            return true;
        }

        // Vision support
        // 中文：加载多模态视觉编码器（Pixtral mmproj 文件）并将本模型设为其宿主，以支持图像输入。
        public void LoadVisionEncoder(string mmProjPath)
        {
            _visionEncoder = new Mistral3VisionEncoder(mmProjPath, _allocator);
            _visionEncoder.SetHostModel(this);
        }

        // 中文：登记一组待注入的视觉嵌入及其插入位置，等待下次 Forward 时注入到隐藏状态序列中。
        public void SetVisionEmbeddings(Tensor embeddings, int insertPosition)
        {
            _pendingVisionEmbeddingsList.Add((embeddings, insertPosition));
        }

        public Mistral3VisionEncoder VisionEncoder => _visionEncoder;

        // Chunk size for ForwardRefill: long prompts are processed in this-many-token
        // chunks so the per-layer attention-score allocation stays bounded
        // (~numHeads × chunkLen × totalKvLen × kvDtype). Past ~2048 the score tensor
        // can run into hundreds of MB on long contexts and thrash the MLX memory pool.
        // Override with TS_PREFILL_CHUNK when tuning.
        // 中文：从环境变量 TS_PREFILL_CHUNK 解析预填充分块大小，默认 2048，用于限制长提示词处理时注意力分数张量的内存占用。
        private static int ResolvePrefillChunkSize()
        {
            string env = Environment.GetEnvironmentVariable("TS_PREFILL_CHUNK");
            if (!string.IsNullOrEmpty(env) && int.TryParse(env, out int v) && v > 0)
                return v;
            return 2048;
        }

        // 中文：批量预填充入口，将长 token 序列按块预填充以约束内存（多模态或短序列则走单次 Forward），最后对末尾 token 调用 Forward 返回 logits。
        public override float[] ForwardRefill(int[] tokens)
        {
            if (tokens == null || tokens.Length <= 1)
                return Forward(tokens);

            // Multimodal embeddings carry absolute insert positions within the
            // current Forward call's hidden tensor, so chunked prefill would
            // need to remap them per-chunk. Skip chunking when any are pending
            // and let the single-call path handle injection.
            bool hasMultimodal = _pendingVisionEmbeddingsList.Count > 0;
            int chunkSize = ResolvePrefillChunkSize();
            int lastIdx = tokens.Length - 1;

            if (hasMultimodal || tokens.Length <= chunkSize)
                return Forward(tokens);

            for (int pos = 0; pos < lastIdx; pos += chunkSize)
            {
                int chunkLen = Math.Min(chunkSize, lastIdx - pos);
                var chunk = new int[chunkLen];
                Array.Copy(tokens, pos, chunk, 0, chunkLen);
                PrefillWithoutLogits(chunk);
            }
            return Forward(new[] { tokens[lastIdx] });
        }

        // 中文：对一块 token 执行前向计算并写入 KV 缓存，但不计算输出 logits（仅用于预填充以建立缓存上下文）。
        private void PrefillWithoutLogits(int[] tokens)
        {
            if (tokens == null || tokens.Length == 0)
                return;

            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;

            EnsureCacheCapacity(startPos + seqLen);

            long t1 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t1;

            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                hidden = TransformerBlock(hidden, layer, seqLen, startPos);
                if (_backend == BackendType.Mlx && (layer + 1) % MlxEvalEveryNLayers == 0
                    && layer + 1 != Config.NumLayers && hidden != null)
                {
                    MlxFusedOps.TryAsyncEvaluate(hidden);
                }
            }

            hidden.Dispose();
            _cacheSeqLen += seqLen;
            _forwardSw.Stop();
        }

        // 中文：核心前向计算：嵌入 token、注入待处理视觉嵌入、逐层 Transformer 计算、输出归一化并经 lm_head 投影为最末位置的 logits。
        public override float[] Forward(int[] tokens)
        {
            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;

            EnsureCacheCapacity(startPos + seqLen);

            long t1 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t1;

            if (_pendingVisionEmbeddingsList.Count > 0)
            {
                foreach (var (embeddings, position) in _pendingVisionEmbeddingsList)
                {
                    InjectVisionEmbeddings(hidden, embeddings, position, startPos);
                    embeddings.Dispose();
                }
                _pendingVisionEmbeddingsList.Clear();
            }

            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                hidden = TransformerBlock(hidden, layer, seqLen, startPos);
                if (_backend == BackendType.Mlx && (layer + 1) % MlxEvalEveryNLayers == 0
                    && layer + 1 != Config.NumLayers && hidden != null)
                {
                    MlxFusedOps.TryAsyncEvaluate(hidden);
                }
            }

            Tensor normed = RMSNormOp(hidden, "output_norm.weight");
            hidden.Dispose();

            Tensor lastHidden;
            if (seqLen > 1)
            {
                using var narrowed = normed.Narrow(0, seqLen - 1, 1);
                lastHidden = Ops.NewContiguous(narrowed);
            }
            else
            {
                lastHidden = normed.CopyRef();
            }
            normed.Dispose();

            long t2 = Stopwatch.GetTimestamp();
            Tensor logitsTensor = LinearForward(lastHidden, "output.weight");
            if (logitsTensor == null)
                logitsTensor = LinearForward(lastHidden, "token_embd.weight");
            _lmHeadTicks += Stopwatch.GetTimestamp() - t2;
            lastHidden.Dispose();

            long t3 = Stopwatch.GetTimestamp();
            _logitsBuffer = TensorToFloatArray(logitsTensor);
            _logitsCopyTicks += Stopwatch.GetTimestamp() - t3;
            logitsTensor.Dispose();

            _cacheSeqLen += seqLen;
            _forwardCount++;
            _forwardSw.Stop();
            return _logitsBuffer;
        }

        // 中文：将视觉编码器产生的图像嵌入按 token 逐行内存拷贝覆盖到隐藏状态张量的指定插入位置，实现多模态嵌入注入。
        private unsafe void InjectVisionEmbeddings(Tensor hidden, Tensor visionEmbeddings, int insertPos, int startPos)
        {
            int numVisionTokens = (int)visionEmbeddings.Sizes[0];
            int dim = Config.HiddenSize;
            float* hPtr = GetFloatPtr(hidden);
            float* vPtr = GetFloatPtr(visionEmbeddings);

            for (int t = 0; t < numVisionTokens; t++)
            {
                float* dst = hPtr + (long)(insertPos + t) * dim;
                float* src = vPtr + (long)t * dim;
                Buffer.MemoryCopy(src, dst, dim * sizeof(float), dim * sizeof(float));
            }

            // insertPos is the offset within the current prefill chunk; startPos
            // (the cached sequence length) makes the absolute position explicit so
            // the log reads monotonically across chunked multimodal prefill.
            Console.WriteLine($"Injected {numVisionTokens} vision tokens at chunk-offset {insertPos} (absolute position {startPos + insertPos})");
        }

        // 中文：单个 Transformer 层的前向计算：注意力前 RMSNorm + 注意力 + 残差，再 FFN 前 RMSNorm + SwiGLU FFN + 残差（尽量使用 MLX 融合的 Add+RMSNorm）。
        private Tensor TransformerBlock(Tensor hidden, int layer, int seqLen, int startPos)
        {
            string[] wn = _layerWeightNames[layer];

            bool fused = _layerQkvFused[layer];
            int normIdx = 0;
            int ffnNormIdx = fused ? 3 : 5;
            int gateUpIdx = fused ? 4 : 6;
            int downIdx = fused ? 5 : 7;

            Tensor normed = RMSNormOp(hidden, wn[normIdx]);
            Tensor attnOut = Attention(normed, layer, wn, seqLen, startPos);
            normed.Dispose();

            // Fused (hidden += attnOut; normed2 = RmsNorm(hidden, ffnNormW))
            // saves one MLX dispatch per residual stage. Falls through to
            // separate Add + RMSNorm if the fused MLX op isn't available
            // (e.g. non-MLX backend or unsupported shape).
            Tensor normed2 = null;
            if (_backend == BackendType.Mlx && _weights.TryGetValue(wn[ffnNormIdx], out var ffnNormW))
            {
                normed2 = new Tensor(_allocator, DType.Float32, hidden.Sizes[0], hidden.Sizes[1]);
                if (!MlxFusedOps.TryAddRmsNorm(hidden, attnOut, ffnNormW, Config.Eps, normed2))
                {
                    normed2.Dispose();
                    normed2 = null;
                }
            }
            if (normed2 == null)
            {
                Ops.Add(hidden, hidden, attnOut);
                normed2 = RMSNormOp(hidden, wn[ffnNormIdx]);
            }
            attnOut.Dispose();

            Tensor ffnOut = FFN(normed2, wn[gateUpIdx], wn[downIdx], seqLen);
            normed2.Dispose();

            Ops.Add(hidden, hidden, ffnOut);
            ffnOut.Dispose();

            return hidden;
        }

        // 中文：多头注意力计算：投影出 Q/K/V，应用 RoPE 与 YaRN 位置相关 Q 缩放，写入 KV 缓存，区分单 token 解码与多 token 预填充两条路径计算缩放点积注意力并输出投影。
        private Tensor Attention(Tensor input, int layer, string[] wn, int seqLen, int startPos)
        {
            int numHeads = Config.NumHeads;
            int numKVHeads = Config.NumKVHeads;
            int headDim = _attnKeyLen;
            int qDim = numHeads * headDim;
            int kDim = numKVHeads * headDim;
            int totalSeqLen = startPos + seqLen;
            float scale = 1.0f / MathF.Sqrt(headDim);

            Tensor qTensor, kTensor, vTensor;

            bool layerFused = _layerQkvFused[layer];
            if (layerFused)
            {
                Tensor qkvFused = LinearForward(input, wn[1]);
                if (seqLen == 1)
                {
                    qTensor = qkvFused.Narrow(1, 0, qDim);
                    kTensor = qkvFused.Narrow(1, qDim, kDim);
                    vTensor = qkvFused.Narrow(1, qDim + kDim, kDim);
                    qkvFused.Dispose();
                }
                else
                {
                    using (var qView = qkvFused.Narrow(1, 0, qDim))
                        qTensor = Ops.NewContiguous(qView);
                    using (var kView = qkvFused.Narrow(1, qDim, kDim))
                        kTensor = Ops.NewContiguous(kView);
                    using (var vView = qkvFused.Narrow(1, qDim + kDim, kDim))
                        vTensor = Ops.NewContiguous(vView);
                    qkvFused.Dispose();
                }
            }
            else
            {
                qTensor = LinearForward(input, wn[1]);  // attn_q
                kTensor = LinearForward(input, wn[2]);  // attn_k
                vTensor = LinearForward(input, wn[3]);  // attn_v
            }

            if (seqLen == 1)
            {
                ApplyRoPEDecode(qTensor, numHeads, headDim, startPos);
                ApplyRoPEDecode(kTensor, numKVHeads, headDim, startPos);

                // Position-dependent Q scaling for YaRN
                if (_ropeOrigCtx > 0)
                    ApplyPositionScale(qTensor, numHeads * headDim, startPos);
            }
            else
            {
                qTensor = ApplyRoPEPrefill(qTensor, numHeads, headDim, seqLen, startPos);
                kTensor = ApplyRoPEPrefill(kTensor, numKVHeads, headDim, seqLen, startPos);

                // Position-dependent Q scaling for YaRN
                if (_ropeOrigCtx > 0)
                    ApplyPositionScalePrefill(qTensor, numHeads, headDim, seqLen, startPos);
            }

            long t0 = Stopwatch.GetTimestamp();

            if (seqLen == 1)
            {
                CopyToCacheDecode(_kvCacheK[layer], kTensor, _kvCacheV[layer], vTensor,
                    numKVHeads, headDim, startPos);
                kTensor.Dispose();
                vTensor.Dispose();

                var attnResult = new Tensor(_allocator, DType.Float32, 1, numHeads * headDim);

                // MLX path: keep K/V on device and run attention via
                // mlx_fast_sdpa. Avoids the per-layer device→host copy of the
                // KV cache that AttentionDecodePureCS triggers via
                // GetHalfPointer (multi-GB transfer per token for long-
                // context configs).
                bool attnOk = false;
                if (_backend == BackendType.Mlx)
                {
                    attnOk = MlxFusedOps.TryDecodeAttention(
                        attnResult, qTensor, _kvCacheK[layer], _kvCacheV[layer],
                        numHeads, numKVHeads, headDim,
                        0, totalSeqLen, _kvCacheCapacity, false, scale);
                }
                if (!attnOk)
                {
                    AttentionDecodePureCS(qTensor, _kvCacheK[layer], _kvCacheV[layer],
                        attnResult, numHeads, numKVHeads, headDim, totalSeqLen, scale);
                }
                qTensor.Dispose();

                _attnTicks += Stopwatch.GetTimestamp() - t0;

                int outputIdx = layerFused ? 2 : 4;
                Tensor decodeOut = LinearForward(attnResult, wn[outputIdx]);
                attnResult.Dispose();
                return decodeOut;
            }

            Tensor qHeads = ReshapeToHeads(qTensor, numHeads, seqLen, headDim);
            qTensor.Dispose();
            Tensor kHeads = ReshapeToHeads(kTensor, numKVHeads, seqLen, headDim);
            kTensor.Dispose();
            Tensor vHeads = ReshapeToHeads(vTensor, numKVHeads, seqLen, _attnValLen);
            vTensor.Dispose();

            CopyToCache(_kvCacheK[layer], kHeads, startPos, seqLen);
            CopyToCache(_kvCacheV[layer], vHeads, startPos, seqLen);
            kHeads.Dispose();
            vHeads.Dispose();

            int groupSize = numHeads / numKVHeads;
            Tensor kExpanded = ExpandKVHeads(_kvCacheK[layer], groupSize, totalSeqLen);
            Tensor vExpanded = ExpandKVHeads(_kvCacheV[layer], groupSize, totalSeqLen);

            using var kT = kExpanded.Transpose(1, 2);
            var scores = new Tensor(_allocator, DType.Float32, numHeads, seqLen, totalSeqLen);
            Ops.AddmmBatch(scores, 0, scores, scale, qHeads, kT);
            qHeads.Dispose();
            kExpanded.Dispose();

            // Fused causal-mask + softmax on GPU. Replaces AddCausalMask + Softmax
            // (two separate ops) with one Metal kernel.
            if (IsGgmlBackend)
            {
                GgmlBasicOps.AttentionSoftmaxWithSinks(
                    scores, sinks: null,
                    numHeads: numHeads, seqLen: seqLen, kvLen: totalSeqLen,
                    maskStartPos: startPos, slidingWindow: 0, scale: 1.0f);
            }
            else
            {
                Ops.AddCausalMask(scores, seqLen, startPos, float.NegativeInfinity);
                Ops.Softmax(scores, scores);
            }

            var attnOut = new Tensor(_allocator, DType.Float32, numHeads, seqLen, _attnValLen);
            Ops.AddmmBatch(attnOut, 0, attnOut, 1.0f, scores, vExpanded);
            scores.Dispose();
            vExpanded.Dispose();

            Tensor flatOutput = ReshapeFromHeads(attnOut, numHeads, seqLen, _attnValLen);
            attnOut.Dispose();

            _attnTicks += Stopwatch.GetTimestamp() - t0;

            int outIdx = layerFused ? 2 : 4;
            Tensor output = LinearForward(flatOutput, wn[outIdx]);
            flatOutput.Dispose();

            return output;
        }

        /// <summary>
        /// GPT-J (norm) style RoPE: pairs adjacent elements (x[2i], x[2i+1]).
        /// Uses precomputed YaRN-corrected frequencies for decode.
        /// </summary>
        // 中文：解码（单 token）路径的 RoPE：用预计算的 YaRN 修正频率对每个头的相邻元素对 (x[2i], x[2i+1]) 做旋转位置编码。
        private unsafe void ApplyRoPEDecode(Tensor data, int numHeads, int headDim, int position)
        {
            int halfDim = _ropeDim / 2;
            float* ptr = GetFloatPtr(data);

            float* cosTable = stackalloc float[halfDim];
            float* sinTable = stackalloc float[halfDim];
            for (int i = 0; i < halfDim; i++)
            {
                float theta = position * _ropeFreqs[i];
                cosTable[i] = MathF.Cos(theta);
                sinTable[i] = MathF.Sin(theta);
            }

            for (int h = 0; h < numHeads; h++)
            {
                float* head = ptr + h * headDim;
                for (int i = 0; i < halfDim; i++)
                {
                    float x0 = head[2 * i];
                    float x1 = head[2 * i + 1];
                    head[2 * i] = x0 * cosTable[i] - x1 * sinTable[i];
                    head[2 * i + 1] = x0 * sinTable[i] + x1 * cosTable[i];
                }
            }
        }

        // 中文：预填充（多 token）路径的 RoPE：为每个位置构造位置索引并调用张量算子 RoPEEx（含 YaRN 参数）批量施加旋转位置编码。
        private Tensor ApplyRoPEPrefill(Tensor data, int numHeads, int headDim, int seqLen, int startPos)
        {
            int totalRows = seqLen * numHeads;
            int[] positions = new int[totalRows];
            for (int s = 0; s < seqLen; s++)
                for (int h = 0; h < numHeads; h++)
                    positions[s * numHeads + h] = startPos + s;
            using var posTensor = CreateIntTensor(positions, totalRows);

            using var reshaped = data.View(1, seqLen, numHeads, headDim);
            Tensor result = Ops.RoPEEx(
                null, reshaped, posTensor, _ropeDim, 0, _ropeOrigCtx,
                Config.RopeBase, 1.0f / Config.RopeScale,
                _ropeType == "yarn" ? _ropeExtFactor : 0f,
                ComputeAttnFactor(),
                _ropeType == "yarn" ? _ropeBetaFast : 0f,
                _ropeType == "yarn" ? _ropeBetaSlow : 0f);

            data.Dispose();

            Tensor flat = result.View(seqLen, numHeads * headDim);
            result.Dispose();
            return flat;
        }

        // 中文：计算 YaRN 注意力缩放因子（mscale），当 mscale 相关参数均非零时返回随 RopeScale 衰减的因子，否则返回 1。
        private float ComputeAttnFactor()
        {
            if (_ropeMscale != 0 && _ropeMscaleAllDim != 0)
                return 1.0f / (0.1f * MathF.Log(Config.RopeScale) + 1.0f);
            return 1.0f;
        }

        /// <summary>
        /// Position-dependent Q scaling for YaRN:
        /// q *= (1 + beta * log(1 + floor(pos / orig_ctx)))
        /// </summary>
        // 中文：解码路径的 YaRN 位置相关 Q 缩放：按公式 q *= (1 + beta·log(1 + floor(pos/orig_ctx))) 对整个 Q 张量整体缩放。
        private unsafe void ApplyPositionScale(Tensor qTensor, int totalQDim, int position)
        {
            float interval = MathF.Floor((float)position / _ropeOrigCtx);
            float posScale = 1.0f + _ropeScalingBeta * MathF.Log(1.0f + interval);
            if (MathF.Abs(posScale - 1.0f) < 1e-7f)
                return;

            float* ptr = GetFloatPtr(qTensor);
            VecScale(ptr, posScale, totalQDim);
        }

        // 中文：预填充路径的 YaRN 位置相关 Q 缩放：逐个序列位置按其绝对位置计算缩放系数并对该行的 Q 子向量缩放。
        private unsafe void ApplyPositionScalePrefill(Tensor qTensor, int numHeads, int headDim,
            int seqLen, int startPos)
        {
            float* ptr = GetFloatPtr(qTensor);
            int stride = numHeads * headDim;

            for (int s = 0; s < seqLen; s++)
            {
                int pos = startPos + s;
                float interval = MathF.Floor((float)pos / _ropeOrigCtx);
                float posScale = 1.0f + _ropeScalingBeta * MathF.Log(1.0f + interval);
                if (MathF.Abs(posScale - 1.0f) < 1e-7f)
                    continue;
                VecScale(ptr + (long)s * stride, posScale, stride);
            }
        }

        // Native batch decode is not used for Mistral 3 because YaRN applies
        // per-dimension frequency correction that the generic TransformerLayerDecode
        // API cannot express. The C# decode path uses GGML-backed matmul/attention
        // and only adds a lightweight C# RoPE kernel.

        // 中文：释放资源：销毁视觉编码器、待注入视觉嵌入及所有层的 K/V 缓存张量，最后调用基类释放。
        public override void Dispose()
        {
            _visionEncoder?.Dispose();
            foreach (var (embeddings, _) in _pendingVisionEmbeddingsList)
                embeddings?.Dispose();
            _pendingVisionEmbeddingsList.Clear();

            if (_kvCacheK != null)
                foreach (var t in _kvCacheK) t?.Dispose();
            if (_kvCacheV != null)
                foreach (var t in _kvCacheV) t?.Dispose();

            base.Dispose();
        }
    }
}
