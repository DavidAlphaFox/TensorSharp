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
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.MLX;

namespace TensorSharp.Models
{
    /// <summary>
    /// Qwen3.5 hybrid model: alternates GatedDeltaNet (recurrent) and FullAttention layers.
    /// Every Nth layer is full attention (determined by full_attention_interval), the rest are recurrent.
    /// Full attention layers: gated Q (Q+gate interleaved), QK-norm, sigmoid-gated output.
    /// Recurrent layers: SSM conv1d, gated delta net with recurrent state.
    /// Both layer types use post_attention_norm before FFN. FFN is either a dense SwiGLU
    /// (gate+up + down) or a Mixture-of-Experts SwiGLU (router + top-K SwiGLU experts +
    /// optional shared SwiGLU expert gated by sigmoid), depending on which weights are present.
    /// </summary>
    public partial class Qwen35Model : ModelBase
    {
        private bool[] _isRecurrent;
        private int _fullAttentionInterval;

        // MoE configuration (qwen35moe / qwen3next variants)
        private int _numExperts;
        private int _numExpertsUsed;
        private int _expertFfnLength;
        private int _sharedExpertFfnLength;
        private bool _normTopKProb;
        private bool[] _isMoeLayer;
        private bool[] _hasSharedExperts;
        private bool[] _hasSharedExpertGate;

        // Per-layer cached expert weight key strings (avoids string interpolation in the hot loop)
        private string[][] _expertGateKeys;
        private string[][] _expertUpKeys;
        private string[][] _expertDownKeys;

        // Per-layer pre-cached QuantizedWeight references (eliminates dictionary lookups in hot loops)
        private QuantizedWeight[][] _expertGateQW;

        // Per-layer cached references for MoE router and shared experts. These avoid
        // dictionary lookups on the hot decode path that happen once per token per layer.
        private QuantizedWeight[] _ffnGateInpQW;
        private Tensor[] _ffnGateInpF32;
        private QuantizedWeight[] _ffnGateShexpQW;
        private Tensor[] _ffnGateShexpF32;
        private QuantizedWeight[] _ffnUpShexpQW;
        private Tensor[] _ffnUpShexpF32;
        private QuantizedWeight[] _ffnDownShexpQW;
        private Tensor[] _ffnDownShexpF32;
        private Tensor[] _ffnGateInpShexpVec;
        private QuantizedWeight[][] _expertUpQW;
        private QuantizedWeight[][] _expertDownQW;
        private Tensor[][] _expertGateF32;
        private Tensor[][] _expertUpF32;
        private Tensor[][] _expertDownF32;

        // Per-layer stacked-along-experts weight handles for the fused MoE
        // prefill kernel. Each entry references the original 3D GGUF tensor
        // (zero-cost view for mmap'd models). Null when stacked weights are
        // unavailable for that layer (e.g. non-mmap models with mismatched
        // layouts), in which case we fall back to the per-expert batched path.
        private StackedExpertWeights[] _layerStackedGate;
        private StackedExpertWeights[] _layerStackedUp;
        private StackedExpertWeights[] _layerStackedDown;

        // Pre-allocated MoE work tensors (reused per token)
        private Tensor _moeTokenInput;     // [1, hiddenSize]
        private Tensor _moeGateBuf;        // [1, expertFfnLength]
        private Tensor _moeUpBuf;          // [1, expertFfnLength]
        private Tensor _moeDownBuf;        // [1, hiddenSize]
        private Tensor _moeBatchedResult;  // [1, hiddenSize] - accumulated batched MoE output

        // Pre-allocated MoE routing buffers
        private float[] _moeProbs;
        private int[] _moeTopExperts;
        private float[] _moeRouteW;
        // Pre-allocated MoE pointer arrays for the batched SwiGLU GGML kernel
        private IntPtr[] _moeGatePtrs;
        private IntPtr[] _moeUpPtrs;
        private IntPtr[] _moeDownPtrs;

        // Pre-allocated FullAttention decode work tensors (decode hot path; shape is fixed by config).
        // Pre-allocating these once eliminates ~3 small tensor allocations per FullAttention layer
        // per decode token.
        private Tensor _attnDecodeQBuf;    // [1, numHeads * headDim]   (deinterleaved Q)
        private Tensor _attnDecodeGBuf;    // [1, numHeads * headDim]   (deinterleaved gate)
        private Tensor _attnDecodeOutBuf;  // [1, numHeads * headDim]   (attention output)
        // [1, qFullDim + 2*kvDim] - reused fused QKV output for FullAttention decode.
        private Tensor _attnDecodeQkvBuf;
        // [1, 2 * intermediateSize] - reused fused gate_up output for dense FFN decode.
        private Tensor _ffnDecodeGateUpBuf;
        // Minimum total sequence length at which the standalone GPU flash attention decode
        // kernel becomes worthwhile. Below this, per-call Metal command buffer setup dominates
        // the savings vs. the CPU SIMD path. Tuned empirically on Apple M-series. Once the
        // fully-fused attention layer kernel is available we prefer it instead (see
        // <see cref="TryFusedAttnLayerDecode"/>) which avoids the standalone path entirely.
        private const int FlashAttnDecodeMinSeqLen = 2048;

        // Minimum total sequence length at which the fully-fused per-layer attention decode
        // kernel beats the existing FusedRmsNormMatMulQuant + CPU-SIMD attention + FusedMatMulQuantAdd
        // path. The fused kernel folds 6 small CPU-side ops into a single Metal graph dispatch,
        // but flash_attn_ext on Metal has a fixed setup cost that only amortises when the cached
        // sequence is long enough. Below this threshold we keep the existing decode path.
        // Set FUSED_ATTN_LAYER_MIN_SEQ_LEN=N to override at runtime for benchmarking.
        private static readonly int FusedAttnLayerDecodeMinSeqLen = ResolveFusedAttnLayerMinSeqLen();
        private static readonly int MlxFlashAttnDecodeMinSeqLen = ResolveMlxFlashAttnDecodeMinSeqLen();
        private static readonly int MlxEvalEveryNLayers = ResolveMlxEvalEveryNLayers();
        private static readonly bool MlxEvalDecodeLayerBoundaries =
            string.Equals(Environment.GetEnvironmentVariable("TS_MLX_EVAL_DECODE_LAYER_BOUNDARIES"), "1", StringComparison.Ordinal);
        private static readonly bool MlxEvalFinalLogits =
            string.Equals(Environment.GetEnvironmentVariable("TS_MLX_EVAL_FINAL_LOGITS"), "1", StringComparison.Ordinal);
        // Pipelined greedy decode state. _pipelineNextInputDevice holds the
        // input embedding for the NEXT decode step, computed on-device from
        // the previous step's argmax. When set, SubmitNextGreedyDecodeStep
        // uses it as the forward's input instead of calling Embedding(host
        // int). This lets the inference loop queue step N+1's forward
        // BEFORE syncing step N's token, so the LM-head + sync wait at the
        // end of step N overlaps with step N+1's first kernels.
        private Tensor _pipelineNextInputDevice;

        // Batched-MoE decode scratch buffers. Hold one row per active
        // expert (K = numExpertsUsed) so all K experts' matmuls can be
        // dispatched in a single Metal kernel.
        // _moeBatchedGate / _moeBatchedUp: [K, intermediate]
        // _moeBatchedDown: [K, hidden]
        // _moeBatchedExpertIndices: [K] int32 (device-side topK)
        // _moeBatchedRouteWeights: [1, K] float32 (device-side routing
        //     weights — used as the LHS of the final matmul that turns
        //     [K, hidden] expert outputs into a single [1, hidden]).
        // All allocated lazily on first use; reused across all calls.
        private Tensor _moeBatchedGate;
        private Tensor _moeBatchedUp;
        private Tensor _moeBatchedDown;
        private Tensor _moeBatchedExpertIndices;
        private Tensor _moeBatchedRouteWeights;
        // Cache keys for the per-layer stacked weight uploads. Use the
        // StackedExpertWeights.Data pointer as the unique identifier.
        //
        // Batched MoE decode: issue 1 Metal dispatch per (gate/up/down)
        // instead of K per-expert dispatches. Default on; set
        // TS_MLX_BATCHED_MOE_DECODE=0 to revert to the legacy per-expert
        // sequence. Uploads the full stacked expert weights as a SEPARATE
        // MLX array per (layer, kind), roughly doubling MLX-tracked
        // weight memory (per-expert arrays are still retained for the
        // prefill path); only set =0 on memory-constrained machines.
        private static readonly bool MlxBatchedMoeDecode =
            !string.Equals(Environment.GetEnvironmentVariable("TS_MLX_BATCHED_MOE_DECODE"), "0", StringComparison.Ordinal);

        // Fused (gate matmul + up matmul + SiLUMul) Metal kernel for the
        // batched MoE decode path. On by default; set
        // TS_MLX_MOE_FUSED_GATE_UP_SILU=0 to fall back to the legacy
        // 3-dispatch sequence for A/B comparison. Saves ~2 MLX dispatches
        // per MoE layer per decode token plus the [K, intermediate]
        // gate/up materialization round-trip.
        private static readonly bool MlxMoeFusedGateUpSiluDisabled =
            string.Equals(Environment.GetEnvironmentVariable("TS_MLX_MOE_FUSED_GATE_UP_SILU"), "0", StringComparison.Ordinal);

        // Device-router for MoE decode: compute top-K + softmax on the
        // device, skipping the per-MoE-layer host sync on routerLogits.
        // Eliminates ~60 GetFloatPtr round trips per decode token on
        // Qwen3.6-35B-A3B (60 MoE layers). On by default; set
        // TS_MLX_DEVICE_ROUTER=0 to disable. Falls back to host routing
        // automatically when prerequisites are missing (non-greedy
        // router, no batched MoE decode, or a shared-expert gate
        // requiring a host-side VecDot).
        private static readonly bool MlxDeviceRouter =
            !string.Equals(Environment.GetEnvironmentVariable("TS_MLX_DEVICE_ROUTER"), "0", StringComparison.Ordinal);

        // 中文：解析融合注意力层解码 kernel 启用的最小序列长度阈值（可由环境变量覆盖，默认 1）。
        private static int ResolveFusedAttnLayerMinSeqLen()
        {
            string env = Environment.GetEnvironmentVariable("FUSED_ATTN_LAYER_MIN_SEQ_LEN");
            if (!string.IsNullOrWhiteSpace(env) && int.TryParse(env, out int v) && v > 0)
                return v;
            // The fused per-layer attention decode kernel folds 6 small
            // CPU-side ops into a single Metal graph dispatch. omlx/vllm
            // both use the equivalent (mx.fast.scaled_dot_product_attention)
            // unconditionally — the per-call setup cost is amortised even
            // at small KV lengths once the rest of the model is on-device.
            return 1;
        }

        // 中文：解析 MLX flash-attention 解码 kernel 启用的最小序列长度阈值（可由环境变量覆盖，默认 1）。
        private static int ResolveMlxFlashAttnDecodeMinSeqLen()
        {
            string env = Environment.GetEnvironmentVariable("TS_MLX_FLASH_ATTN_DECODE_MIN_SEQ_LEN");
            if (!string.IsNullOrWhiteSpace(env) && int.TryParse(env, out int v) && v > 0)
                return v;
            // Always prefer the Metal flash-attention kernel for MLX
            // decode — see omlx/mlx-lm which dispatches
            // mx.fast.scaled_dot_product_attention for every step.
            return 1;
        }

        // 中文：解析 MLX 每隔多少层触发一次图求值（eval）的间隔（可由环境变量覆盖，默认 16）。
        private static int ResolveMlxEvalEveryNLayers()
        {
            string env = Environment.GetEnvironmentVariable("TS_MLX_EVAL_EVERY_N_LAYERS");
            if (!string.IsNullOrWhiteSpace(env) && int.TryParse(env, out int v) && v >= 0)
                return v;
            // Measured 11.6 tok/s @ N=8 vs 11.8 tok/s @ N=16 on Qwen3.6-35B-
            // A3B IQ2_XXS decode (100-token median of 3 with 3 warmups). The
            // larger interval gives MLX more graph to schedule between syncs;
            // smaller intervals (4) regress to 10.8 tok/s as sync count grows
            // and larger (32) regress to 11.1 tok/s as the graph becomes
            // unwieldy.
            return 16;
        }

        // Pre-cached layer prefix and weight name strings (avoids string interpolation in hot loops)
        private string[] _layerPrefix;
        private string[] _attnNormKey;
        private string[] _postAttnNormKey;
        private string[] _attnQkvKey;
        private string[] _attnQKey;
        private string[] _attnKKey;
        private string[] _attnVKey;
        private string[] _attnQNormKey;
        private string[] _attnKNormKey;
        private string[] _attnOutputKey;
        private string[] _ffnGateUpKey;
        private string[] _ffnDownKey;
        private string[] _ffnGateInpKey;
        private string[] _ffnGateShexpKey;
        private string[] _ffnUpShexpKey;
        private string[] _ffnDownShexpKey;
        private string[] _ffnGateInpShexpKey;

        // (GDN-only key arrays and weight tensors live in Qwen35Model.GatedDeltaNet.cs)

        // Pre-resolved weight references for FullAttention linear projections.
        // QK-norm weights are also cached here as F32 tensors (always F32 in GGUF).
        private QuantizedWeight[] _attnQkvQW;
        private Tensor[] _attnQkvF32;
        private QuantizedWeight[] _attnQQW;
        private Tensor[] _attnQF32;
        private QuantizedWeight[] _attnKQW;
        private Tensor[] _attnKF32;
        private QuantizedWeight[] _attnVQW;
        private Tensor[] _attnVF32;
        private QuantizedWeight[] _attnOutputQW;
        private Tensor[] _attnOutputF32;
        private Tensor[] _attnQNormW;
        private Tensor[] _attnKNormW;

        // Pre-resolved attention norm weights (always F32).
        private Tensor[] _attnNormW;
        private Tensor[] _postAttnNormW;

        // Pre-resolved final norm and LM head weights (called once per forward step).
        private Tensor _finalNormW;
        private QuantizedWeight _lmHeadQW;
        private Tensor _lmHeadF32;

        // Pre-resolved weight references for non-MoE FFN paths.
        private QuantizedWeight[] _ffnGateUpQW;
        private Tensor[] _ffnGateUpF32;
        private QuantizedWeight[] _ffnDownQW;
        private Tensor[] _ffnDownF32;

        // Full attention KV cache (only for attention layers)
        private Tensor[] _kvCacheK;
        private Tensor[] _kvCacheV;
        private MlxFusedOps.AttentionKvCache[] _mlxAttentionCache;
        private int _kvCacheCapacity;

        // GatedDeltaNet recurrent state, dimensions and projection weights live in
        // Qwen35Model.GatedDeltaNet.cs (also a partial of this class).
        private int _ropeDimCount;

        // MRoPE sections
        private int[] _mropeSections;

        // Pre-computed RoPE frequency table
        private float[] _ropeFreqs;

        // Cached RoPE position tensors (reused across attention layers in one forward pass)
        private Tensor _cachedRoPEPosQ, _cachedRoPEPosK;
        private int _cachedRoPEPosSeqLen, _cachedRoPEPosStartPos = -1;

        // Per-axis (T,H,W) positions for the next prefill chunk, packed flat as
        // [T0,H0,W0, T1,H1,W1, ...]. Populated by the multimodal injector just
        // before Forward() (via SetMRoPEPositions) when the request is a
        // Qwen3.5 vision prompt. Forward() consumes and clears the field at
        // the start of each call; missing/null means use plain scalar RoPE.
        // Per-pair modality assignment (which (T,H,W) axis drives which rotary
        // pair) is precomputed once in PrecomputeMRoPEInterleavedIds from
        // _mropeSections — vLLM mrope_interleaved.py's get_mrope_interleaved_id_list.
        private int[] _pendingMRoPEPositions;
        private int[] _mropeInterleavedIds; // length rotary_dim/2; values ∈ {0=T,1=H,2=W}

        // (GDN scratch buffers, chunked-prefill staging, and timing counters live in
        // Qwen35Model.GatedDeltaNet.cs.)

        // Vision encoder
        public Qwen35VisionEncoder VisionEncoder { get; private set; }
        private List<(Tensor embeddings, int position)> _visionEmbeddingsList = new();

        // Detailed prefill-only profiling counters. Set the QWEN35_PREFILL_PROFILE
        // environment variable to "1" to populate these so PrintTimingStats prints a
        // fine-grained breakdown of where the prefill seconds go. Off by default to
        // avoid the timer overhead in normal runs.
        private static readonly bool _profilePrefillStages =
            string.Equals(Environment.GetEnvironmentVariable("QWEN35_PREFILL_PROFILE"), "1", StringComparison.Ordinal);

        // QWEN35_DECODE_PROFILE=1: bucket the per-decode-token time into
        // attention/MoE/norm/lm-head/sync so we can attack the dominant
        // bucket. Adds a Stopwatch.GetTimestamp per layer; ~0.05 μs each,
        // negligible at decode rates.
        private static readonly bool _profileDecodeStages =
            string.Equals(Environment.GetEnvironmentVariable("QWEN35_DECODE_PROFILE"), "1", StringComparison.Ordinal);

        // Set QWEN35_DISABLE_FUSED_FFN=1 to disable the fully fused dense FFN graph
        // dispatch in FFNCachedFused (useful for A/B benchmarking against the legacy
        // 3-dispatch path: FusedRmsNormMatMul + SiLUMul + FusedMatMulQuantAdd).
        private static readonly bool _useFusedFfnPrefill =
            !string.Equals(Environment.GetEnvironmentVariable("QWEN35_DISABLE_FUSED_FFN"), "1", StringComparison.Ordinal);
        private long _decodeAttnBlockTicks;
        private long _decodeRecBlockTicks;
        private long _decodeForwardCount;
        private long _prefillEmbedTicks;
        private long _prefillAttnBlockTicks;
        private long _prefillRecBlockTicks;
        private long _prefillFinalLmHeadTicks;
        private long _prefillAttnQkvTicks;
        private long _prefillAttnDeinterleaveTicks;
        private long _prefillAttnQknormTicks;
        private long _prefillAttnRopeTicks;
        private long _prefillAttnReshapeTicks;
        private long _prefillAttnCacheCopyTicks;
        private long _prefillAttnExpandKvTicks;
        private long _prefillAttnComputeTicks;
        private long _prefillAttnGateTicks;
        private long _prefillAttnOutputTicks;
        private long _prefillAttnFfnTicks;
        private long _prefillRecInputProjTicks;
        private long _prefillRecCoreTicks;
        private long _prefillRecOutputTicks;
        private long _prefillRecFfnTicks;
        private long _mlxEvalBoundaryTicks;
        private long _mlxCacheEvalTicks;
        private int _prefillTokenCount;

        // 中文：构造函数。从 GGUF 读取架构与超参数（层数/头数/MoE/MRoPE/SSM 等），判定各层是循环（GatedDeltaNet）还是全注意力，加载并融合权重、检测 MoE 层、初始化缓存与 RoPE 表，完成模型构建。
        public Qwen35Model(string ggufPath, BackendType backend)
            : base(ggufPath, backend)
        {
            string arch = _gguf.GetString("general.architecture") ?? "qwen35";
            Config = new ModelConfig { Architecture = arch };
            ParseBaseConfig();

            // Qwen3.5 uses per-layer KV head counts; get the first non-zero value
            if (_gguf.Metadata.TryGetValue($"{arch}.attention.head_count_kv", out var hckvVal))
                Config.NumKVHeads = Convert.ToInt32(hckvVal);
            else
                Config.NumKVHeads = Config.NumHeads;

            // SSM config (GDN-specific dimensions are parsed inside the GDN partial)
            ParseGdnConfig(arch);
            _fullAttentionInterval = (int)_gguf.GetUint32($"{arch}.full_attention_interval", 4);

            // MRoPE sections
            var sections = _gguf.GetInt32Array($"{arch}.rope.dimension_sections");
            _mropeSections = sections;
            _ropeDimCount = (int)_gguf.GetUint32($"{arch}.rope.dimension_count", 0);

            // MoE configuration: present for qwen35moe / qwen3next variants
            _numExperts = (int)_gguf.GetUint32($"{arch}.expert_count", 0);
            _numExpertsUsed = (int)_gguf.GetUint32($"{arch}.expert_used_count", 0);
            _expertFfnLength = (int)_gguf.GetUint32($"{arch}.expert_feed_forward_length", 0);
            _sharedExpertFfnLength = (int)_gguf.GetUint32($"{arch}.expert_shared_feed_forward_length", (uint)_expertFfnLength);
            // qwen3 MoE renormalizes the selected top-K probabilities by default.
            _normTopKProb = true;
            Config.NumExperts = _numExperts;
            Config.NumExpertsUsed = _numExpertsUsed;

            // Determine which layers are recurrent (GatedDeltaNet) vs full-attention.
            // Prefer an explicit GGUF `layer_types` array if present (vLLM reads
            // `config.layer_types[i]` ∈ {"linear_attention","full_attention"} at
            // qwen3_5.py:230; future fine-tunes can ship a non-default pattern).
            // Fall back to the period-`full_attention_interval` pattern that stock
            // 27B/35B-A3B GGUFs use (no layer_types array shipped).
            _isRecurrent = new bool[Config.NumLayers];
            var layerTypes = _gguf.GetStringArray($"{arch}.layer_types");
            if (layerTypes != null && layerTypes.Length == Config.NumLayers)
            {
                for (int i = 0; i < Config.NumLayers; i++)
                    _isRecurrent[i] = string.Equals(layerTypes[i], "linear_attention", StringComparison.OrdinalIgnoreCase);
            }
            else
            {
                for (int i = 0; i < Config.NumLayers; i++)
                    _isRecurrent[i] = (i + 1) % _fullAttentionInterval != 0;
            }

            ParseTokenizer();

            Console.WriteLine($"Model: {arch}, Layers={Config.NumLayers}, Hidden={Config.HiddenSize}, " +
                $"Heads={Config.NumHeads}, KVHeads={Config.NumKVHeads}, HeadDim={Config.HeadDim}, Vocab={Config.VocabSize}");
            Console.WriteLine($"RoPE base={Config.RopeBase}, scale={Config.RopeScale}, eps={Config.Eps}");
            Console.WriteLine($"SSM: dInner={_ssmDInner}, dState={_ssmDState}, nGroup={_ssmNGroup}, " +
                $"dtRank={_ssmDtRank}, convKernel={_convKernel}, fullAttnInterval={_fullAttentionInterval}");
            Console.WriteLine($"MRoPE sections: [{string.Join(", ", _mropeSections ?? Array.Empty<int>())}]");

            int attnCount = 0, recCount = 0;
            for (int i = 0; i < Config.NumLayers; i++)
            {
                if (_isRecurrent[i]) recCount++; else attnCount++;
            }
            Console.WriteLine($"Layer types: {attnCount} full attention, {recCount} recurrent (GatedDeltaNet)");

            if (_numExperts > 0)
                Console.WriteLine($"MoE: experts={_numExperts}, used={_numExpertsUsed}, " +
                    $"expertFFN={_expertFfnLength}, sharedFFN={_sharedExpertFfnLength}");

            LoadWeights();
            FuseAttentionProjectionWeights();
            FuseRecurrentInputWeights();
            FuseGateUpWeights();
            DetectMoeLayers();
            BuildLayerKeys();
            InitMoeBuffers();
            PrepareCudaQuantizedWeightsForInference();
            int maxContextLength = ResolveConfiguredContextLength();
            int initialCacheLength = ResolveInitialCacheAllocationLength(maxContextLength);
            if (initialCacheLength < maxContextLength)
                Console.WriteLine($"Initial {_backend} KV cache allocation: {initialCacheLength} tokens (grows on demand up to {maxContextLength}).");
            InitCaches(initialCacheLength, maxContextLength);
            PrecomputeRoPE();
            InitGDNBuffers();
            CacheRecurrentWeights();
        }

        // 中文：将每个全注意力层的 Q/K/V 投影权重融合为单个 attn_qkv 权重，减少 hot 路径上的矩阵乘次数。
        private unsafe void FuseAttentionProjectionWeights()
        {
            int fused = 0;
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                if (_isRecurrent[layer])
                    continue;

                string prefix = $"blk.{layer}.";
                if (TryFuseWeights(prefix + "attn_qkv.weight",
                    prefix + "attn_q.weight",
                    prefix + "attn_k.weight",
                    prefix + "attn_v.weight"))
                {
                    fused++;
                }
            }

            if (fused > 0)
                Console.WriteLine($"  Fused projections: {fused} Q+K+V");
        }

        // 中文：将每个循环（GatedDeltaNet）层的输入投影权重（qkv/gate/beta/alpha）融合成一个 ssm_in_proj 权重包。
        private unsafe void FuseRecurrentInputWeights()
        {
            int fused = 0;
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                if (!_isRecurrent[layer])
                    continue;

                string prefix = $"blk.{layer}.";
                if (TryFuseWeights(prefix + "ssm_in_proj.weight",
                    prefix + "attn_qkv.weight",
                    prefix + "attn_gate.weight",
                    prefix + "ssm_beta.weight",
                    prefix + "ssm_alpha.weight"))
                {
                    fused++;
                }
            }

            if (fused > 0)
                Console.WriteLine($"  Fused projections: {fused} recurrent input packs");
        }

        // 中文：尝试把多个权重沿输出维拼接成一个融合权重（量化或 F32 两条路径），成功后替换字典并释放原权重；形状/类型不匹配时返回 false。
        private unsafe bool TryFuseWeights(string fusedName, params string[] weightNames)
        {
            if (weightNames == null || weightNames.Length < 2)
                return false;

            var quantWeights = new QuantizedWeight[weightNames.Length];
            bool allQuant = true;
            for (int i = 0; i < weightNames.Length; i++)
            {
                if (!_quantWeights.TryGetValue(weightNames[i], out quantWeights[i]))
                {
                    allQuant = false;
                    break;
                }
            }

            if (allQuant)
            {
                var first = quantWeights[0];
                long totalBytes = 0;
                long totalNe1 = 0;
                for (int i = 0; i < quantWeights.Length; i++)
                {
                    var qw = quantWeights[i];
                    if (qw.GgmlType != first.GgmlType || qw.Ne0 != first.Ne0)
                        return false;

                    totalBytes += qw.RawBytes;
                    totalNe1 += qw.Ne1;
                }

                if (!TryCreateFusedQuantizedWeight(out QuantizedWeight fusedWeight, quantWeights))
                    return false;

                _quantWeights[fusedName] = fusedWeight;
                for (int i = 0; i < weightNames.Length; i++)
                {
                    var name = weightNames[i];
                    var qw = quantWeights[i];
                    _quantWeights.Remove(name);
                    qw.Dispose();
                }

                return true;
            }

            var floatWeights = new Tensor[weightNames.Length];
            for (int i = 0; i < weightNames.Length; i++)
            {
                if (!_weights.TryGetValue(weightNames[i], out floatWeights[i]))
                    return false;
            }

            int inDim = (int)floatWeights[0].Sizes[1];
            int totalOutDim = 0;
            for (int i = 0; i < floatWeights.Length; i++)
            {
                var w = floatWeights[i];
                if ((int)w.Sizes[1] != inDim)
                    return false;

                totalOutDim += (int)w.Sizes[0];
            }

            var fusedTensor = new Tensor(_allocator, DType.Float32, totalOutDim, inDim);
            int outOffset = 0;
            for (int i = 0; i < floatWeights.Length; i++)
            {
                var w = floatWeights[i];
                int outDim = (int)w.Sizes[0];
                using var slice = fusedTensor.Narrow(0, outOffset, outDim);
                Ops.Copy(slice, w);
                outOffset += outDim;
            }

            _weights[fusedName] = fusedTensor;
            for (int i = 0; i < weightNames.Length; i++)
            {
                var name = weightNames[i];
                var w = floatWeights[i];
                _weights.Remove(name);
                w.Dispose();
            }

            return true;
        }

        // 中文：通过检测各层是否存在 ffn_gate_inp 等权重，标记哪些层是 MoE 层、是否含共享专家及共享专家门控。
        private void DetectMoeLayers()
        {
            int numLayers = Config.NumLayers;
            _isMoeLayer = new bool[numLayers];
            _hasSharedExperts = new bool[numLayers];
            _hasSharedExpertGate = new bool[numLayers];

            int moeCount = 0, sharedCount = 0, sharedGateCount = 0;
            for (int l = 0; l < numLayers; l++)
            {
                string prefix = $"blk.{l}.";
                _isMoeLayer[l] = WeightExists(prefix + "ffn_gate_inp.weight");
                if (_isMoeLayer[l])
                {
                    moeCount++;
                    _hasSharedExperts[l] = WeightExists(prefix + "ffn_up_shexp.weight") &&
                                           WeightExists(prefix + "ffn_down_shexp.weight");
                    if (_hasSharedExperts[l])
                        sharedCount++;
                    _hasSharedExpertGate[l] = _weights.ContainsKey(prefix + "ffn_gate_inp_shexp.weight");
                    if (_hasSharedExpertGate[l])
                        sharedGateCount++;
                }
            }

            if (moeCount > 0)
                Console.WriteLine($"  MoE layers: {moeCount}/{numLayers} (shared experts: {sharedCount}, gated shared: {sharedGateCount})");
        }

        // 中文：判断给定名称的权重是否存在（量化权重字典或 F32 权重字典任一命中）。
        private bool WeightExists(string name) =>
            _quantWeights.ContainsKey(name) || _weights.ContainsKey(name);

        // 中文：初始化 MoE 相关缓冲：预缓存各层每个专家及路由/共享专家的权重键与权重引用、堆叠专家权重句柄，并预分配 MoE 与解码热路径复用的工作张量。
        private void InitMoeBuffers()
        {
            if (_numExperts <= 0)
                return;

            _moeProbs = new float[_numExperts];
            _moeTopExperts = new int[_numExpertsUsed];
            _moeRouteW = new float[_numExpertsUsed];

            int numLayers = Config.NumLayers;
            _expertGateKeys = new string[numLayers][];
            _expertUpKeys = new string[numLayers][];
            _expertDownKeys = new string[numLayers][];
            _expertGateQW = new QuantizedWeight[numLayers][];
            _expertUpQW = new QuantizedWeight[numLayers][];
            _expertDownQW = new QuantizedWeight[numLayers][];
            _expertGateF32 = new Tensor[numLayers][];
            _expertUpF32 = new Tensor[numLayers][];
            _expertDownF32 = new Tensor[numLayers][];

            _ffnGateInpQW = new QuantizedWeight[numLayers];
            _ffnGateInpF32 = new Tensor[numLayers];
            _ffnGateShexpQW = new QuantizedWeight[numLayers];
            _ffnGateShexpF32 = new Tensor[numLayers];
            _ffnUpShexpQW = new QuantizedWeight[numLayers];
            _ffnUpShexpF32 = new Tensor[numLayers];
            _ffnDownShexpQW = new QuantizedWeight[numLayers];
            _ffnDownShexpF32 = new Tensor[numLayers];
            _ffnGateInpShexpVec = new Tensor[numLayers];
            _layerStackedGate = new StackedExpertWeights[numLayers];
            _layerStackedUp = new StackedExpertWeights[numLayers];
            _layerStackedDown = new StackedExpertWeights[numLayers];

            for (int l = 0; l < numLayers; l++)
            {
                if (!_isMoeLayer[l])
                    continue;
                _expertGateKeys[l] = new string[_numExperts];
                _expertUpKeys[l] = new string[_numExperts];
                _expertDownKeys[l] = new string[_numExperts];
                _expertGateQW[l] = new QuantizedWeight[_numExperts];
                _expertUpQW[l] = new QuantizedWeight[_numExperts];
                _expertDownQW[l] = new QuantizedWeight[_numExperts];
                _expertGateF32[l] = new Tensor[_numExperts];
                _expertUpF32[l] = new Tensor[_numExperts];
                _expertDownF32[l] = new Tensor[_numExperts];
                string p = _layerPrefix[l];
                for (int e = 0; e < _numExperts; e++)
                {
                    _expertGateKeys[l][e] = p + "ffn_gate_exps." + e + ".weight";
                    _expertUpKeys[l][e] = p + "ffn_up_exps." + e + ".weight";
                    _expertDownKeys[l][e] = p + "ffn_down_exps." + e + ".weight";

                    _quantWeights.TryGetValue(_expertGateKeys[l][e], out _expertGateQW[l][e]);
                    _quantWeights.TryGetValue(_expertUpKeys[l][e], out _expertUpQW[l][e]);
                    _quantWeights.TryGetValue(_expertDownKeys[l][e], out _expertDownQW[l][e]);
                    _weights.TryGetValue(_expertGateKeys[l][e], out _expertGateF32[l][e]);
                    _weights.TryGetValue(_expertUpKeys[l][e], out _expertUpF32[l][e]);
                    _weights.TryGetValue(_expertDownKeys[l][e], out _expertDownF32[l][e]);
                }

                // Look up the original 3D stacked-along-experts views populated
                // by ModelBase.LoadWeights. Used by the batched MoE decode path
                // (MlxBatchedMoeDecode) to upload all experts as one MLX array
                // per (layer, kind) and dispatch a single matmul instead of K
                // per-expert matmuls.
                _stackedExpertWeights.TryGetValue(p + "ffn_gate_exps.weight", out _layerStackedGate[l]);
                _stackedExpertWeights.TryGetValue(p + "ffn_up_exps.weight", out _layerStackedUp[l]);
                _stackedExpertWeights.TryGetValue(p + "ffn_down_exps.weight", out _layerStackedDown[l]);

                // Cache router and shared expert weights for this layer.
                _quantWeights.TryGetValue(_ffnGateInpKey[l], out _ffnGateInpQW[l]);
                _weights.TryGetValue(_ffnGateInpKey[l], out _ffnGateInpF32[l]);

                if (_hasSharedExperts != null && _hasSharedExperts[l])
                {
                    _quantWeights.TryGetValue(_ffnGateShexpKey[l], out _ffnGateShexpQW[l]);
                    _weights.TryGetValue(_ffnGateShexpKey[l], out _ffnGateShexpF32[l]);
                    _quantWeights.TryGetValue(_ffnUpShexpKey[l], out _ffnUpShexpQW[l]);
                    _weights.TryGetValue(_ffnUpShexpKey[l], out _ffnUpShexpF32[l]);
                    _quantWeights.TryGetValue(_ffnDownShexpKey[l], out _ffnDownShexpQW[l]);
                    _weights.TryGetValue(_ffnDownShexpKey[l], out _ffnDownShexpF32[l]);

                    if (_hasSharedExpertGate != null && _hasSharedExpertGate[l])
                        _weights.TryGetValue(_ffnGateInpShexpKey[l], out _ffnGateInpShexpVec[l]);
                }
            }

            // Pre-allocate scratch tensors reused across MoE expert calls. This avoids the
            // hot-loop allocation of [1, expertFFN] / [1, hidden] tensors per token per expert.
            if (_expertFfnLength > 0)
            {
                _moeTokenInput = new Tensor(_allocator, DType.Float32, 1, Config.HiddenSize);
                _moeGateBuf = new Tensor(_allocator, DType.Float32, 1, _expertFfnLength);
                _moeUpBuf = new Tensor(_allocator, DType.Float32, 1, _expertFfnLength);
                _moeDownBuf = new Tensor(_allocator, DType.Float32, 1, Config.HiddenSize);
                _moeBatchedResult = new Tensor(_allocator, DType.Float32, 1, Config.HiddenSize);
                _moeGatePtrs = new IntPtr[_numExpertsUsed];
                _moeUpPtrs = new IntPtr[_numExpertsUsed];
                _moeDownPtrs = new IntPtr[_numExpertsUsed];
            }

            // Pre-allocate decode-only FullAttention work tensors. Used only when seqLen == 1.
            int qHeadDim = Config.NumHeads * Config.HeadDim;
            if (qHeadDim > 0)
            {
                _attnDecodeQBuf = new Tensor(_allocator, DType.Float32, 1, qHeadDim);
                _attnDecodeGBuf = new Tensor(_allocator, DType.Float32, 1, qHeadDim);
                _attnDecodeOutBuf = new Tensor(_allocator, DType.Float32, 1, qHeadDim);

                // Pre-allocate fused QKV (Q+gate interleaved + K + V) and FFN gate_up buffers.
                // Eliminates 2 tensor allocations per attention layer per decode token.
                int qFullDim = Config.NumHeads * Config.HeadDim * 2;
                int kvDim = Config.NumKVHeads * Config.HeadDim;
                if (qFullDim > 0 && kvDim > 0)
                    _attnDecodeQkvBuf = new Tensor(_allocator, DType.Float32, 1, qFullDim + 2 * kvDim);
                if (Config.IntermediateSize > 0)
                    _ffnDecodeGateUpBuf = new Tensor(_allocator, DType.Float32, 1, 2 * Config.IntermediateSize);
            }
        }

        /// <summary>
        /// Pre-cache layer-prefix and weight name strings for every layer so the hot Forward()
        /// path never executes string interpolation. This eliminates dozens of allocations per
        /// layer per forward step and removes hundreds of dictionary string-hash lookups.
        /// </summary>
        // 中文：为每一层预先拼好层前缀与各权重名称字符串，避免 Forward 热路径上的字符串插值与重复字典哈希查找。
        private void BuildLayerKeys()
        {
            int n = Config.NumLayers;
            _layerPrefix = new string[n];
            _attnNormKey = new string[n];
            _postAttnNormKey = new string[n];
            _attnQkvKey = new string[n];
            _attnQKey = new string[n];
            _attnKKey = new string[n];
            _attnVKey = new string[n];
            _attnQNormKey = new string[n];
            _attnKNormKey = new string[n];
            _attnOutputKey = new string[n];
            _ffnGateUpKey = new string[n];
            _ffnDownKey = new string[n];
            _ffnGateInpKey = new string[n];
            _ffnGateShexpKey = new string[n];
            _ffnUpShexpKey = new string[n];
            _ffnDownShexpKey = new string[n];
            _ffnGateInpShexpKey = new string[n];
            InitGdnLayerKeyArrays(n);

            for (int l = 0; l < n; l++)
            {
                string p = $"blk.{l}.";
                _layerPrefix[l] = p;
                _attnNormKey[l] = p + "attn_norm.weight";
                _postAttnNormKey[l] = p + "post_attention_norm.weight";
                _attnQkvKey[l] = p + "attn_qkv.weight";
                _attnQKey[l] = p + "attn_q.weight";
                _attnKKey[l] = p + "attn_k.weight";
                _attnVKey[l] = p + "attn_v.weight";
                _attnQNormKey[l] = p + "attn_q_norm.weight";
                _attnKNormKey[l] = p + "attn_k_norm.weight";
                _attnOutputKey[l] = p + "attn_output.weight";
                _ffnGateUpKey[l] = p + "ffn_gate_up.weight";
                _ffnDownKey[l] = p + "ffn_down.weight";
                _ffnGateInpKey[l] = p + "ffn_gate_inp.weight";
                _ffnGateShexpKey[l] = p + "ffn_gate_shexp.weight";
                _ffnUpShexpKey[l] = p + "ffn_up_shexp.weight";
                _ffnDownShexpKey[l] = p + "ffn_down_shexp.weight";
                _ffnGateInpShexpKey[l] = p + "ffn_gate_inp_shexp.weight";
                SetGdnLayerKeys(l, p);
            }
        }

        /// <summary>
        /// Pre-resolve recurrent layer constant tensors and pre-compute a transposed conv1d
        /// weight layout that is friendly for SIMD vectorization across channels.
        /// </summary>
        // 中文：预解析并缓存各层的注意力/FFN/归一化权重以及最终归一化与 LM head 权重引用，循环层另行缓存其专用权重，消除 hot 路径上的字典查找。
        private unsafe void CacheRecurrentWeights()
        {
            int n = Config.NumLayers;
            InitGdnWeightArrays(n);
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;

            _attnQkvQW = new QuantizedWeight[n];
            _attnQkvF32 = new Tensor[n];
            _attnQQW = new QuantizedWeight[n];
            _attnQF32 = new Tensor[n];
            _attnKQW = new QuantizedWeight[n];
            _attnKF32 = new Tensor[n];
            _attnVQW = new QuantizedWeight[n];
            _attnVF32 = new Tensor[n];
            _attnOutputQW = new QuantizedWeight[n];
            _attnOutputF32 = new Tensor[n];
            _attnQNormW = new Tensor[n];
            _attnKNormW = new Tensor[n];
            _attnNormW = new Tensor[n];
            _postAttnNormW = new Tensor[n];

            _ffnGateUpQW = new QuantizedWeight[n];
            _ffnGateUpF32 = new Tensor[n];
            _ffnDownQW = new QuantizedWeight[n];
            _ffnDownF32 = new Tensor[n];

            // Final layer norm + LM head (vocab projection). Tied embedding weights fall
            // back to token_embd.weight when output.weight is not present.
            _weights.TryGetValue("output_norm.weight", out _finalNormW);
            if (!_quantWeights.TryGetValue("output.weight", out _lmHeadQW))
                _quantWeights.TryGetValue("token_embd.weight", out _lmHeadQW);
            if (!_weights.TryGetValue("output.weight", out _lmHeadF32))
                _weights.TryGetValue("token_embd.weight", out _lmHeadF32);

            for (int l = 0; l < n; l++)
            {
                // Norms / FFN are present for every layer (recurrent and attention).
                _weights.TryGetValue(_attnNormKey[l], out _attnNormW[l]);
                _weights.TryGetValue(_postAttnNormKey[l], out _postAttnNormW[l]);
                _quantWeights.TryGetValue(_ffnGateUpKey[l], out _ffnGateUpQW[l]);
                _weights.TryGetValue(_ffnGateUpKey[l], out _ffnGateUpF32[l]);
                _quantWeights.TryGetValue(_ffnDownKey[l], out _ffnDownQW[l]);
                _weights.TryGetValue(_ffnDownKey[l], out _ffnDownF32[l]);

                if (_isRecurrent[l])
                {
                    CacheRecurrentLayerWeights(l, qkvDim);
                }
                else
                {
                    _quantWeights.TryGetValue(_attnQkvKey[l], out _attnQkvQW[l]);
                    _weights.TryGetValue(_attnQkvKey[l], out _attnQkvF32[l]);
                    _quantWeights.TryGetValue(_attnQKey[l], out _attnQQW[l]);
                    _weights.TryGetValue(_attnQKey[l], out _attnQF32[l]);
                    _quantWeights.TryGetValue(_attnKKey[l], out _attnKQW[l]);
                    _weights.TryGetValue(_attnKKey[l], out _attnKF32[l]);
                    _quantWeights.TryGetValue(_attnVKey[l], out _attnVQW[l]);
                    _weights.TryGetValue(_attnVKey[l], out _attnVF32[l]);
                    _quantWeights.TryGetValue(_attnOutputKey[l], out _attnOutputQW[l]);
                    _weights.TryGetValue(_attnOutputKey[l], out _attnOutputF32[l]);
                    _weights.TryGetValue(_attnQNormKey[l], out _attnQNormW[l]);
                    _weights.TryGetValue(_attnKNormKey[l], out _attnKNormW[l]);
                }
            }
        }

        // 中文：初始化各层的 KV 缓存（全注意力层）与 GatedDeltaNet 循环状态缓存（循环层），按配置的 KV 缓存 dtype 分配并清零。
        private void InitCaches(int initialSeqLen, int maxSeqLen)
        {
            _maxContextLength = maxSeqLen;
            _kvCacheCapacity = initialSeqLen;
            int numLayers = Config.NumLayers;
            _kvCacheK = new Tensor[numLayers];
            _kvCacheV = new Tensor[numLayers];
            _mlxAttentionCache = _backend == BackendType.Mlx
                ? new MlxFusedOps.AttentionKvCache[numLayers]
                : null;

            InitGdnCacheArrays(numLayers);
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;

            ApplyModelAlignedKvCacheDefault(_quantWeights);
            DType kvDtype = _kvCacheDtype.ToDType();
            for (int l = 0; l < numLayers; l++)
            {
                if (!_isRecurrent[l])
                {
                    _kvCacheK[l] = new Tensor(_allocator, kvDtype, Config.NumKVHeads, initialSeqLen, Config.HeadDim);
                    _kvCacheV[l] = new Tensor(_allocator, kvDtype, Config.NumKVHeads, initialSeqLen, Config.HeadDim);
                    InitializeCacheTensor(_kvCacheK[l]);
                    InitializeCacheTensor(_kvCacheV[l]);
                    if (_mlxAttentionCache != null)
                        _mlxAttentionCache[l] = new MlxFusedOps.AttentionKvCache();
                }
                else
                {
                    InitGdnLayerCache(l, qkvDim);
                }
            }
            _cacheSeqLen = 0;
        }

        // 中文：当所需序列长度超过当前 KV 缓存容量时按倍增策略扩容（上限为最大上下文长度），并把已有缓存内容拷贝到新缓存。
        private void EnsureCacheCapacity(int requiredSeqLen)
        {
            if (requiredSeqLen <= _kvCacheCapacity)
                return;
            if (requiredSeqLen > _maxContextLength)
                throw new InvalidOperationException($"Requested sequence length {requiredSeqLen} exceeds configured max context {_maxContextLength}.");

            int newCapacity = Math.Max(_kvCacheCapacity, 1);
            while (newCapacity < requiredSeqLen)
                newCapacity = Math.Min(_maxContextLength, newCapacity * 2);

            DType kvDtype = _kvCacheDtype.ToDType();
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_isRecurrent[l])
                    continue;

                var newK = new Tensor(_allocator, kvDtype, Config.NumKVHeads, newCapacity, Config.HeadDim);
                var newV = new Tensor(_allocator, kvDtype, Config.NumKVHeads, newCapacity, Config.HeadDim);
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
            Console.WriteLine($"Expanded Qwen3.5 attention cache to {newCapacity} tokens.");
        }

        // 中文：预计算 RoPE 各维度的旋转频率表（按 RopeBase 与 RopeScale 生成 inv_freq）。
        private void PrecomputeRoPE()
        {
            int headDim = Config.HeadDim;
            int ropeDim = _ropeDimCount > 0 ? _ropeDimCount : headDim;
            int halfDim = ropeDim / 2;
            float freqScale = 1.0f / Config.RopeScale;
            _ropeFreqs = new float[halfDim];
            for (int i = 0; i < halfDim; i++)
                _ropeFreqs[i] = freqScale / MathF.Pow(Config.RopeBase, (2.0f * i) / ropeDim);
        }

        // 中文：重置所有层的 KV 缓存与 GDN 循环状态、清零缓存序列长度及各项计时计数器，使模型回到全新会话状态。
        public override void ResetKVCache()
        {
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!_isRecurrent[l])
                {
                    ResetCacheTensor(_kvCacheK[l]);
                    ResetCacheTensor(_kvCacheV[l]);
                    _mlxAttentionCache?[l]?.Reset();
                }
                else
                {
                    ResetGdnLayerCache(l);
                }
            }
            _cacheSeqLen = 0;
            _linearTicks = _attnTicks = _normTicks = _embTicks = _lmHeadTicks = _logitsCopyTicks = 0;
            _mlxEvalBoundaryTicks = 0;
            _mlxCacheEvalTicks = 0;
            _forwardCount = 0;
            ResetGdnTimingCounters();
            _forwardSw.Reset();

            _prefillEmbedTicks = _prefillAttnBlockTicks = _prefillRecBlockTicks = _prefillFinalLmHeadTicks = 0;
            _prefillAttnQkvTicks = _prefillAttnDeinterleaveTicks = _prefillAttnQknormTicks = 0;
            _prefillAttnRopeTicks = _prefillAttnReshapeTicks = _prefillAttnCacheCopyTicks = 0;
            _prefillAttnExpandKvTicks = _prefillAttnComputeTicks = _prefillAttnGateTicks = 0;
            _prefillAttnOutputTicks = _prefillAttnFfnTicks = 0;
            _prefillRecInputProjTicks = _prefillRecCoreTicks = _prefillRecOutputTicks = _prefillRecFfnTicks = 0;
            _prefillTokenCount = 0;
        }

        public override bool SupportsKVCacheTruncation => false;

        // Per-block snapshot for Qwen 3.5 (mix of attention layers and GDN
        // recurrent layers). Each block bundles:
        //   * For every attention layer L: K bytes for [start,start+B), V bytes
        //     for [start,start+B). Uses the same byte layout as the simpler
        //     models so a single helper can capture it.
        //   * For every GDN layer L: a snapshot of the layer's running state at
        //     the END of this block (convState ring buffer + writeIdx +
        //     deltaState tensor bytes). The state is "as of after the block's
        //     final token" because Capture runs after each prefill chunk lands.
        // Recurrent state isn't decomposable into per-position slices, so this
        // model relies on the per-chunk capture path
        // (<see cref="RequiresPerBlockCapture"/>).
        public override bool RequiresPerBlockCapture => true;

        public override bool SupportsKVStateSnapshot => _kvCacheK != null && _kvCacheV != null;

        public override string KVStateFingerprint =>
            $"qwen35|arch={Config.Architecture}|L={Config.NumLayers}|H={Config.NumHeads}|KV={Config.NumKVHeads}|D={Config.HeadDim}|gdnK={_headKDim}|gdnV={_headVDim}|nKHead={_numKHeads}|nVHead={_numVHeads}|convKern={_convKernel}|dtype={_kvCacheDtype.ToShortString()}";

        // 中文：计算一个含 tokenCount 个 token 的 KV 状态块的总字节数（全注意力层按 K/V 字节累加，循环层按 GDN 状态字节累加）。
        public override long ComputeKVBlockByteSize(int tokenCount)
        {
            if (tokenCount <= 0 || _kvCacheK == null || _isRecurrent == null) return 0;
            long total = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!_isRecurrent[l])
                {
                    if (_kvCacheK[l] == null || _kvCacheV[l] == null) return 0;
                    total += AttentionLayerBlockBytes(_kvCacheK[l], tokenCount);
                    total += AttentionLayerBlockBytes(_kvCacheV[l], tokenCount);
                }
                else
                {
                    total += GdnLayerStateBytes(l);
                }
            }
            return total;
        }

        // 中文：将各层从 startToken 起 tokenCount 个 token 的 KV 状态（注意力层 K/V + 循环层 GDN 状态）序列化导出到目标字节缓冲。
        public override bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            if (!SupportsKVStateSnapshot) return false;
            long expected = ComputeKVBlockByteSize(tokenCount);
            if (destination.Length != expected) return false;
            int offset = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!_isRecurrent[l])
                {
                    if (startToken + tokenCount > _cacheSeqLen) return false;
                    if (!CopyAttentionOut(_kvCacheK[l], startToken, tokenCount, destination[offset..], out int wK))
                        return false;
                    offset += wK;
                    if (!CopyAttentionOut(_kvCacheV[l], startToken, tokenCount, destination[offset..], out int wV))
                        return false;
                    offset += wV;
                }
                else
                {
                    if (!CopyGdnStateOut(l, destination[offset..], out int wG))
                        return false;
                    offset += wG;
                }
            }
            return offset == destination.Length;
        }

        // 中文：将外部序列化的 KV 状态块注入回各层缓存（必须紧接当前缓存末尾），扩容后写入并使设备端缓存视图失效以便下次重新加载。
        public override bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source)
        {
            if (!SupportsKVStateSnapshot) return false;
            if (destToken != _cacheSeqLen) return false;
            long expected = ComputeKVBlockByteSize(tokenCount);
            if (source.Length != expected) return false;

            EnsureCacheCapacity(destToken + tokenCount);
            int offset = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!_isRecurrent[l])
                {
                    if (!CopyAttentionIn(_kvCacheK[l], destToken, tokenCount, source[offset..], out int rK))
                        return false;
                    offset += rK;
                    if (!CopyAttentionIn(_kvCacheV[l], destToken, tokenCount, source[offset..], out int rV))
                        return false;
                    offset += rV;
                }
                else
                {
                    if (!CopyGdnStateIn(l, source[offset..], out int rG))
                        return false;
                    offset += rG;
                }
            }
            _cacheSeqLen = destToken + tokenCount;

            // Invalidate any device-cached views so the next forward refills them
            // from the freshly-written host buffers.
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_isRecurrent[l]) continue;
                InvalidateTensorDeviceCache(_kvCacheK[l]);
                InvalidateTensorDeviceCache(_kvCacheV[l]);
            }
            return true;
        }

        // 中文：计算某注意力层缓存张量中 tokenCount 个 token 所占的字节数（KV 头数 × token 数 × 每行字节）。
        private static long AttentionLayerBlockBytes(Tensor cacheTensor, int tokenCount)
        {
            long numKVHeads = cacheTensor.Sizes[0];
            long capacity = cacheTensor.Sizes[1];
            long rowBytes = cacheTensor.Storage.ByteLength / (numKVHeads * capacity);
            return numKVHeads * tokenCount * rowBytes;
        }

        // 中文：计算某 GDN 循环层完整状态的字节数（conv 状态字节 + writeIdx(4 字节) + deltaState 张量字节）。
        private long GdnLayerStateBytes(int layer)
        {
            // convState bytes + writeIdx (4 bytes) + deltaState tensor bytes.
            long convBytes = (long)_convState[layer].Length * sizeof(float);
            long deltaBytes = _deltaStateTensor[layer].Storage.ByteLength;
            return convBytes + sizeof(int) + deltaBytes;
        }

        // 中文：把注意力缓存张量中 [startToken, startToken+tokenCount) 的字节按每头拷贝导出到目标缓冲（先确保主机可读）。
        private static bool CopyAttentionOut(Tensor cacheTensor, int startToken, int tokenCount, Span<byte> destination, out int written)
        {
            cacheTensor.Storage.EnsureHostReadable();
            long numKVHeads = cacheTensor.Sizes[0];
            long capacity = cacheTensor.Sizes[1];
            long rowBytes = cacheTensor.Storage.ByteLength / (numKVHeads * capacity);
            long blockBytes = numKVHeads * tokenCount * rowBytes;
            if (destination.Length < blockBytes) { written = 0; return false; }
            IntPtr basePtr = cacheTensor.Storage.PtrAtElement(0);
            unsafe
            {
                byte* src = (byte*)basePtr;
                fixed (byte* dst = destination)
                {
                    long perHead = tokenCount * rowBytes;
                    for (long h = 0; h < numKVHeads; h++)
                    {
                        long s = (h * capacity + startToken) * rowBytes;
                        long d = h * perHead;
                        Buffer.MemoryCopy(src + s, dst + d, destination.Length - d, perHead);
                    }
                }
            }
            written = (int)blockBytes;
            return true;
        }

        // 中文：把源字节按每头写回注意力缓存张量 [destToken, destToken+tokenCount) 区间（导入的逆操作）。
        private static bool CopyAttentionIn(Tensor cacheTensor, int destToken, int tokenCount, ReadOnlySpan<byte> source, out int read)
        {
            cacheTensor.Storage.EnsureHostReadable();
            long numKVHeads = cacheTensor.Sizes[0];
            long capacity = cacheTensor.Sizes[1];
            if (destToken + tokenCount > capacity) { read = 0; return false; }
            long rowBytes = cacheTensor.Storage.ByteLength / (numKVHeads * capacity);
            long blockBytes = numKVHeads * tokenCount * rowBytes;
            if (source.Length < blockBytes) { read = 0; return false; }
            IntPtr basePtr = cacheTensor.Storage.PtrAtElement(0);
            unsafe
            {
                byte* dst = (byte*)basePtr;
                fixed (byte* srcBase = source)
                {
                    long perHead = tokenCount * rowBytes;
                    for (long h = 0; h < numKVHeads; h++)
                    {
                        long d = (h * capacity + destToken) * rowBytes;
                        long s = h * perHead;
                        Buffer.MemoryCopy(srcBase + s, dst + d, cacheTensor.Storage.ByteLength - d, perHead);
                    }
                }
            }
            read = (int)blockBytes;
            return true;
        }

        // 中文：把某 GDN 层的完整循环状态（convState 原始浮点字节 + writeIdx + deltaState 字节）序列化导出到目标缓冲。
        private bool CopyGdnStateOut(int layer, Span<byte> destination, out int written)
        {
            written = 0;
            SyncCudaGdnConvStateToHost(layer);
            float[] conv = _convState[layer];
            int convBytes = conv.Length * sizeof(float);
            Tensor delta = _deltaStateTensor[layer];
            long deltaBytes = delta.Storage.ByteLength;
            long total = (long)convBytes + sizeof(int) + deltaBytes;
            if (destination.Length < total) return false;

            // convState as raw float bytes.
            MemoryMarshal.AsBytes(conv.AsSpan()).CopyTo(destination[..convBytes]);
            // writeIdx as int.
            BitConverter.TryWriteBytes(destination.Slice(convBytes, sizeof(int)), _convStateWriteIdx[layer]);
            // deltaState bytes via storage pointer (host-resident on every backend).
            delta.Storage.EnsureHostReadable();
            IntPtr deltaBase = delta.Storage.PtrAtElement(0);
            unsafe
            {
                fixed (byte* dst = destination[(convBytes + sizeof(int))..])
                {
                    Buffer.MemoryCopy((void*)deltaBase, dst, deltaBytes, deltaBytes);
                }
            }
            written = (int)total;
            return true;
        }

        // 中文：把源字节恢复回某 GDN 层的 convState/writeIdx/deltaState 状态，并重置该层 MLX 缓存的 scratch 索引（导出的逆操作）。
        private bool CopyGdnStateIn(int layer, ReadOnlySpan<byte> source, out int read)
        {
            read = 0;
            float[] conv = _convState[layer];
            int convBytes = conv.Length * sizeof(float);
            Tensor delta = _deltaStateTensor[layer];
            long deltaBytes = delta.Storage.ByteLength;
            long total = (long)convBytes + sizeof(int) + deltaBytes;
            if (source.Length < total) return false;

            source[..convBytes].CopyTo(MemoryMarshal.AsBytes(conv.AsSpan()));
            _convStateWriteIdx[layer] = BitConverter.ToInt32(source.Slice(convBytes, sizeof(int)));
            SyncCudaGdnConvStateFromHost(layer);

            delta.Storage.EnsureHostReadable();
            IntPtr deltaBase = delta.Storage.PtrAtElement(0);
            unsafe
            {
                fixed (byte* src = source[(convBytes + sizeof(int))..])
                {
                    Buffer.MemoryCopy(src, (void*)deltaBase, deltaBytes, deltaBytes);
                }
            }
            // MLX cache reflects the host-side bytes lazily on next use; resetting
            // its scratch indices is enough for correctness.
            _mlxGdnCache?[layer]?.Reset();
            read = (int)total;
            return true;
        }

        // Chunk size for ForwardRefill: long prompts are processed in this-many-token
        // chunks so the per-layer attention-score allocation stays bounded.
        // Override with TS_PREFILL_CHUNK when tuning.
        // 中文：解析 ForwardRefill 的 prefill 分块大小（长提示按此长度分块处理以限制每层注意力分数的内存占用，可由环境变量覆盖，默认 2048）。
        private static int ResolvePrefillChunkSize()
        {
            string env = Environment.GetEnvironmentVariable("TS_PREFILL_CHUNK");
            if (!string.IsNullOrEmpty(env) && int.TryParse(env, out int v) && v > 0)
                return v;
            return 2048;
        }

        // 中文：批量预填充（prefill）入口：长提示按块逐段喂入 PrefillWithoutLogits 构建 KV 缓存，最后只对末 token 调用 Forward 取 logits；含多模态嵌入或短提示时退化为单次 Forward。
        public override float[] ForwardRefill(int[] tokens)
        {
            if (tokens == null || tokens.Length <= 1)
                return Forward(tokens);

            // Multimodal embeddings carry positions relative to the current
            // Forward call's hidden tensor; chunked prefill would need to
            // remap them per-chunk. Fall back to single Forward when any are pending.
            bool hasMultimodal = _visionEmbeddingsList.Count > 0;
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

        // 中文：仅前向传播一段 token 以填充 KV 缓存而不计算 logits（嵌入后逐层跑注意力/循环块并推进缓存序列长度），用于分块 prefill。
        private void PrefillWithoutLogits(int[] tokens)
        {
            if (tokens == null || tokens.Length == 0)
                return;

            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            EnsureCacheCapacity(startPos + seqLen);
            bool profilePrefill = _profilePrefillStages && seqLen > 1;
            if (profilePrefill)
                _prefillTokenCount += seqLen;

            long t1 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            long embEnd = Stopwatch.GetTimestamp();
            _embTicks += embEnd - t1;
            if (profilePrefill)
                _prefillEmbedTicks += embEnd - t1;

            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                long blkStart = profilePrefill ? Stopwatch.GetTimestamp() : 0;
                if (_isRecurrent[layer])
                    hidden = RecurrentBlock(hidden, layer, seqLen, startPos);
                else
                    hidden = AttentionBlock(hidden, layer, seqLen, startPos);
                TryEvaluateMlxLayerBoundary(hidden, layer, seqLen);
                if (profilePrefill)
                {
                    long elapsed = Stopwatch.GetTimestamp() - blkStart;
                    if (_isRecurrent[layer]) _prefillRecBlockTicks += elapsed;
                    else _prefillAttnBlockTicks += elapsed;
                }
            }

            hidden.Dispose();
            _cacheSeqLen += seqLen;
            _forwardSw.Stop();
        }

        // 中文：核心前向传播：嵌入并注入视觉嵌入后逐层跑全注意力/循环块，对末 token 做最终归一化 + LM head 得到词表 logits 并拷贝到主机缓冲返回。
        public override float[] Forward(int[] tokens)
        {
            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            EnsureCacheCapacity(startPos + seqLen);
            bool profilePrefill = _profilePrefillStages && seqLen > 1;
            if (profilePrefill)
                _prefillTokenCount += seqLen;

            long t1 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            long embEnd = Stopwatch.GetTimestamp();
            _embTicks += embEnd - t1;
            if (profilePrefill)
                _prefillEmbedTicks += embEnd - t1;

            // Inject any vision embeddings queued for this Forward call. The
            // queued positions are already expressed relative to the current
            // tokens slice (see ModelMultimodalInjector.QueuePreparedVisionEmbeddings),
            // so this works for both a fresh prefill (startPos == 0) and a
            // multi-turn refill that re-uses the prior KV cache (startPos > 0).
            // The latter is the path taken when a user uploads a second image
            // in the same chat session - gating on startPos == 0 here used to
            // silently drop the second image's vision embeddings and turn its
            // <|image_pad|> tokens into garbage.
            if (_visionEmbeddingsList.Count > 0)
                InjectVisionEmbeddings(hidden, seqLen);

            bool profileDecode = _profileDecodeStages && seqLen == 1;
            if (profileDecode)
                _decodeForwardCount++;
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                long blkStart = (profilePrefill || profileDecode) ? Stopwatch.GetTimestamp() : 0;
                if (_isRecurrent[layer])
                    hidden = RecurrentBlock(hidden, layer, seqLen, startPos);
                else
                    hidden = AttentionBlock(hidden, layer, seqLen, startPos);
                TryEvaluateMlxLayerBoundary(hidden, layer, seqLen);
                if (profilePrefill)
                {
                    long elapsed = Stopwatch.GetTimestamp() - blkStart;
                    if (_isRecurrent[layer]) _prefillRecBlockTicks += elapsed;
                    else _prefillAttnBlockTicks += elapsed;
                }
                else if (profileDecode)
                {
                    long elapsed = Stopwatch.GetTimestamp() - blkStart;
                    if (_isRecurrent[layer]) _decodeRecBlockTicks += elapsed;
                    else _decodeAttnBlockTicks += elapsed;
                }
            }

            // Pick out the last token's hidden state BEFORE the final norm so we can
            // fuse final-norm + LM-head matmul into a single GGML kernel for decode.
            Tensor lastHiddenRaw;
            if (seqLen > 1)
            {
                using var narrowed = hidden.Narrow(0, seqLen - 1, 1);
                lastHiddenRaw = Ops.NewContiguous(narrowed);
            }
            else
            {
                lastHiddenRaw = hidden.CopyRef();
            }
            hidden.Dispose();

            long t2 = Stopwatch.GetTimestamp();
            Tensor logitsTensor;
            if (IsGgmlBackend && _lmHeadQW != null && _finalNormW != null
                && lastHiddenRaw.DimensionCount == 2)
            {
                logitsTensor = FusedNormLinear(lastHiddenRaw, _finalNormW, _lmHeadQW, _lmHeadF32);
            }
            else
            {
                Tensor lastNormed = RMSNormOpCached(lastHiddenRaw, _finalNormW);
                logitsTensor = LinearForwardCached(lastNormed, _lmHeadQW, _lmHeadF32);
                lastNormed.Dispose();
            }
            lastHiddenRaw.Dispose();
            if (_backend == BackendType.Mlx && MlxEvalFinalLogits)
                MlxFusedOps.TryEvaluate(logitsTensor);
            long lmHeadEnd = Stopwatch.GetTimestamp();
            _lmHeadTicks += lmHeadEnd - t2;
            if (profilePrefill)
                _prefillFinalLmHeadTicks += lmHeadEnd - t2;

            long t3 = Stopwatch.GetTimestamp();
            if (_logitsBuffer == null || _logitsBuffer.Length != Config.VocabSize)
                _logitsBuffer = new float[Config.VocabSize];
            unsafe
            {
                float* src = GetFloatPtr(logitsTensor);
                fixed (float* dst = _logitsBuffer)
                    Buffer.MemoryCopy(src, dst, Config.VocabSize * 4, Config.VocabSize * 4);
            }
            _logitsCopyTicks += Stopwatch.GetTimestamp() - t3;
            logitsTensor.Dispose();

            _cacheSeqLen += seqLen;
            _forwardCount++;
            // Drop the MRoPE positions staged by the injector so the next
            // Forward (e.g. a decode token, or a different sequence under
            // the engine's per-seq KV swap) starts from a clean slate.
            _pendingMRoPEPositions = null;
            _forwardSw.Stop();
            return _logitsBuffer;
        }

        // Pipelined greedy decode is implemented only on the MLX path
        // (uses MLX argmax + device-side embedding lookup); other backends
        // fall through to the standard Forward + host-side sample loop.
        public override bool SupportsPipelinedGreedy =>
            _backend == BackendType.Mlx
            && _lmHeadQW != null
            && _finalNormW != null
            && _quantWeights.ContainsKey("token_embd.weight");

        // ===== Pipelined greedy decode =====
        //
        // Standard Forward(int[] tokens) issues all 60 layers + the LM head
        // and then host-syncs the 200K-float logits tensor before returning.
        // The sync drains all queued MLX kernels, costing ~8 ms/token on MLX
        // (Qwen3.6-35B-A3B IQ2_XXS) — pure GPU-idle wait from the host's
        // perspective.
        //
        // The pipelined API lets the CLI inference loop kick off forward N+1
        // BEFORE syncing forward N's token. The trick: compute argmax on
        // device, look up the next embedding on device, and return the
        // resulting [1] int32 device tensor as a deferred handle. The
        // caller queues the next step and only THEN host-reads the previous
        // step's token.
        //
        // Greedy only: top-K / temperature sampling still needs the full
        // logits on host. Gated by TS_MLX_PIPELINED_DECODE=1 (off by default).

        // Run one pipelined decode step. Returns a [1] int32 device tensor
        // holding the predicted next token. The caller is responsible for
        // syncing it to host (via Tensor.GetElementsAsInt) and disposing.
        // If firstTokenForBegin is non-null this is the first call after
        // prefill: we look up the embedding for that host int. Otherwise
        // we use the cached _pipelineNextInputDevice from the previous
        // call. Each call queues the next embedding lookup (via on-device
        // argmax) so the next call can run without any host work besides
        // building the layer graph.
        // 中文：执行一步流水线化贪心解码（MLX 专用）：跑完所有层 + LM head，在设备上做 argmax 得到 [1] int32 下一 token，并预先在设备上算好下一步输入嵌入以重叠后续前向，返回该设备 token 句柄。
        public override Tensor SubmitGreedyDecodeStep(int? firstTokenForBegin)
        {
            _forwardSw.Start();
            int seqLen = 1;
            int startPos = _cacheSeqLen;
            EnsureCacheCapacity(startPos + seqLen);

            Tensor inputHidden;
            if (firstTokenForBegin.HasValue)
            {
                // Begin path: upload host int and look up embedding on host.
                // Subsequent calls go through the device lookup path.
                if (_pipelineNextInputDevice != null)
                {
                    _pipelineNextInputDevice.Dispose();
                    _pipelineNextInputDevice = null;
                }
                long embT0 = Stopwatch.GetTimestamp();
                inputHidden = Embedding(new[] { firstTokenForBegin.Value });
                _embTicks += Stopwatch.GetTimestamp() - embT0;
            }
            else if (_pipelineNextInputDevice != null)
            {
                inputHidden = _pipelineNextInputDevice;
                _pipelineNextInputDevice = null;
            }
            else
            {
                throw new InvalidOperationException(
                    "SubmitGreedyDecodeStep: no cached input embedding and no firstTokenForBegin provided.");
            }

            // Run the layer stack. Same path as Forward.
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                if (_isRecurrent[layer])
                    inputHidden = RecurrentBlock(inputHidden, layer, seqLen, startPos);
                else
                    inputHidden = AttentionBlock(inputHidden, layer, seqLen, startPos);
                TryEvaluateMlxLayerBoundary(inputHidden, layer, seqLen);
            }

            // Final norm + LM head.
            long lmT0 = Stopwatch.GetTimestamp();
            Tensor lastNormed = RMSNormOpCached(inputHidden, _finalNormW);
            inputHidden.Dispose();
            Tensor logitsTensor = LinearForwardCached(lastNormed, _lmHeadQW, _lmHeadF32);
            lastNormed.Dispose();
            _lmHeadTicks += Stopwatch.GetTimestamp() - lmT0;

            // Device argmax → [1] int32. Falls back to host argmax if MLX
            // path fails (e.g. non-MLX backend or unsupported dtype).
            var deviceToken = new Tensor(_allocator, DType.Int32, 1);
            if (!MlxFusedOps.TryArgMaxLastAxis(deviceToken, logitsTensor))
            {
                // Host fallback. This forces a sync but only fires when MLX
                // argmax isn't available; the loop will still work, just
                // without the pipelining benefit.
                unsafe
                {
                    float* src = GetFloatPtr(logitsTensor);
                    int maxIdx = 0;
                    float maxVal = src[0];
                    int vocab = (int)logitsTensor.ElementCount();
                    for (int i = 1; i < vocab; i++)
                    {
                        float v = src[i];
                        if (v > maxVal) { maxVal = v; maxIdx = i; }
                    }
                    deviceToken.SetElementsAsInt(new[] { maxIdx });
                }
            }
            logitsTensor.Dispose();

            // Pre-compute the input embedding for the NEXT call so the
            // caller can submit the next forward without any host work
            // related to the predicted token.
            _pipelineNextInputDevice = new Tensor(_allocator, DType.Float32, 1, Config.HiddenSize);
            if (!TryComputeNextInputEmbedding(_pipelineNextInputDevice, deviceToken))
            {
                // If device embedding lookup failed, sync the token and use
                // the host path. This is the fallback for backends without
                // device get_rows or unsupported quant types.
                int hostTok = deviceToken.GetElementsAsInt(1)[0];
                _pipelineNextInputDevice.Dispose();
                _pipelineNextInputDevice = Embedding(new[] { hostTok });
            }
            else if (_backend == BackendType.Mlx)
            {
                // Kick the queued kernels (argmax + embedding lookup) so
                // they start executing on Metal before the host issues the
                // next forward step. The host sync of `deviceToken` will
                // also drain this implicitly, but the explicit async-eval
                // gives MLX a chance to schedule the work earlier.
                MlxFusedOps.TryAsyncEvaluate(_pipelineNextInputDevice);
            }

            _cacheSeqLen += seqLen;
            _forwardCount++;
            _pendingMRoPEPositions = null;
            _forwardSw.Stop();

            return deviceToken;
        }

        // Look up token_embd row for a [1] int32 device token, writing the
        // [1, hidden] result into outEmb. Reuses the existing MLX quantized
        // get_rows path (MlxQuantizedOps.TryGetRowsQuantizedToFloat32) for
        // quantized embedding tables; otherwise returns false so the caller
        // falls back to the host path.
        // 中文：在设备上（MLX）为 [1] int32 token 查 token_embd 行得到 [1, hidden] 输入嵌入写入 outEmb；不支持时返回 false 让调用方回退到主机路径。
        private bool TryComputeNextInputEmbedding(Tensor outEmb, Tensor deviceTokenInt)
        {
            if (_backend != BackendType.Mlx)
                return false;

            if (_quantWeights.TryGetValue("token_embd.weight", out var qw))
            {
                return MlxQuantizedOps.TryGetRowsQuantizedToFloat32(
                    outEmb,
                    qw.CacheKey,
                    qw.Data,
                    qw.GgmlType,
                    qw.Ne0,
                    qw.Ne1,
                    qw.RawBytes,
                    deviceTokenInt);
            }

            // F32 token_embd path: use Ops.IndexSelect if supported on MLX.
            // Most quantized models won't hit this — left as a TODO so we
            // fall back to host path until/unless someone tests it.
            return false;
        }

        // Release any pending pipelined-decode state. Call at end of a
        // generation run so resources don't linger.
        // 中文：释放流水线贪心解码的挂起状态（预算的下一步设备输入嵌入），在一次生成结束时调用以免资源滞留。
        public override void ResetPipelinedGreedyState()
        {
            _pipelineNextInputDevice?.Dispose();
            _pipelineNextInputDevice = null;
        }

        // 中文：在 MLX 后端按设定间隔（每 N 层或最后一层）触发隐藏态与对应层缓存状态的图求值，把惰性计算图分段刷新以平衡同步开销与调度深度。
        private void TryEvaluateMlxLayerBoundary(Tensor hidden, int layer, int seqLen)
        {
            if (_backend != BackendType.Mlx || MlxEvalEveryNLayers <= 0)
            {
                return;
            }

            if (seqLen == 1 && !MlxEvalDecodeLayerBoundaries)
            {
                if ((layer + 1) % MlxEvalEveryNLayers == 0 || layer + 1 == Config.NumLayers)
                {
                    int firstDecodeLayer = Math.Max(0, layer - MlxEvalEveryNLayers + 1);
                    TryEvaluateMlxLayerCacheState(firstDecodeLayer, layer);
                }
                return;
            }

            if (hidden == null
                || ((layer + 1) % MlxEvalEveryNLayers != 0 && layer + 1 != Config.NumLayers))
            {
                return;
            }

            long start = Stopwatch.GetTimestamp();
            // Async at intermediate boundaries; the LM head/RMSNorm at the end
            // of forward will drain the graph (or the sampler's host read does).
            // Last layer keeps a sync eval as a safety net.
            bool isLastLayer = (layer + 1) == Config.NumLayers;
            bool evaluated = isLastLayer
                ? MlxFusedOps.TryEvaluate(hidden)
                : MlxFusedOps.TryAsyncEvaluate(hidden);
            if (evaluated)
                _mlxEvalBoundaryTicks += Stopwatch.GetTimestamp() - start;

            int firstLayer = Math.Max(0, layer - MlxEvalEveryNLayers + 1);
            TryEvaluateMlxLayerCacheState(firstLayer, layer);
        }

        // 中文：在 MLX 后端对 [firstLayer, lastLayer] 区间各层的 KV/GDN 缓存状态触发图求值，使累积的缓存写入提前落盘。
        private void TryEvaluateMlxLayerCacheState(int firstLayer, int lastLayer)
        {
            if (_backend != BackendType.Mlx)
                return;

            long start = Stopwatch.GetTimestamp();
            bool evaluated = false;
            for (int l = firstLayer; l <= lastLayer && l < Config.NumLayers; l++)
            {
                if (l < 0)
                    continue;

                if (_isRecurrent[l])
                    evaluated |= _mlxGdnCache?[l]?.TryEvaluateState() == true;
                else
                    evaluated |= _mlxAttentionCache?[l]?.TryEvaluateState() == true;
            }

            if (evaluated)
                _mlxCacheEvalTicks += Stopwatch.GetTimestamp() - start;
        }

        #region Full Attention Block

        /// <summary>
        /// Full attention with gated Q, QK-norm, sigmoid-gated output, and post-attention norm.
        /// Q projection outputs 2x: [Q, gate] interleaved per head.
        /// </summary>
        // 中文：单个全注意力 Transformer 块：注意力（含残差与输出投影，可走融合解码/融合 outproj+FFN 路径）+ 后注意力归一化 + FFN（稠密 SwiGLU 或 MoE），就地更新并返回隐藏态。
        private Tensor AttentionBlock(Tensor hidden, int layer, int seqLen, int startPos)
        {
            // Decode fast path (long context): fold the entire attention block (norm + QKV +
            // QK-norm + RoPE + KV cache append + flash attention + sigmoid-gated mix + output
            // projection + residual add) into one fused GGML graph dispatch. This pays off once
            // the per-layer CPU attention cost (which scales with the cached sequence length)
            // exceeds the cost of building the fused graph + the GPU flash-attn dispatch. For
            // short contexts the optimised CPU SIMD attention + 2 small GGML dispatches is
            // already faster, so we keep the existing path.
            int totalSeqLen = startPos + seqLen;
            bool fusedDecodeApplied = false;
            if (seqLen == 1 && totalSeqLen >= FusedAttnLayerDecodeMinSeqLen
                && TryFusedAttnLayerDecode(hidden, layer, startPos))
            {
                fusedDecodeApplied = true;
            }

            // Fuse:
            //   hidden = hidden + attn_out_proj(attention(rms_norm(hidden)))
            // into the FullAttention call: input norm + QKV is fused inside FullAttention, and
            // when the GGML backend is available the output projection is also fused with the
            // residual add (FusedMatMulQuantAdd) so the residual lives entirely on the GPU.
            // Fused outproj+FFN for attention layers: when the fused attention layer
            // decode is NOT used and the layer is dense FFN (not MoE), fuse the attention
            // output projection + residual + FFN into one GPU dispatch.
            bool canFuseAttnOutFFN = !fusedDecodeApplied && IsGgmlBackend
                && !(_isMoeLayer != null && _isMoeLayer[layer])
                && _attnOutputQW[layer] != null
                && _postAttnNormW[layer] != null
                && _ffnGateUpQW[layer] != null
                && _ffnDownQW[layer] != null;

            Tensor attnOut;
            if (canFuseAttnOutFFN)
                attnOut = FullAttention(hidden, _attnNormW[layer], layer, seqLen, startPos,
                    residual: null, skipOutputProj: true);
            else if (fusedDecodeApplied)
                attnOut = null;
            else
                attnOut = FullAttention(hidden, _attnNormW[layer], layer, seqLen, startPos, residual: hidden);

            bool profilePrefill = _profilePrefillStages && seqLen > 1;
            long ffnStart = profilePrefill ? Stopwatch.GetTimestamp() : 0;

            if (canFuseAttnOutFFN && attnOut != null)
            {
                int intermSize = Config.IntermediateSize;
                int halfDim = intermSize > 0 ? intermSize : (int)(_ffnGateUpQW[layer].Ne1 / 2);

                if (halfDim > 0 && hidden.DimensionCount == 2 && attnOut.DimensionCount == 2
                    && hidden.Sizes[0] == attnOut.Sizes[0])
                {
                    try
                    {
                        long t0 = Stopwatch.GetTimestamp();
                        GgmlBasicOps.FusedOutProjFFN(
                            hidden, attnOut,
                            _attnOutputQW[layer].CacheKey, _attnOutputQW[layer].GgmlType,
                            _attnOutputQW[layer].Ne0, _attnOutputQW[layer].Ne1, _attnOutputQW[layer].RawBytes,
                            _postAttnNormW[layer], Config.Eps,
                            _ffnGateUpQW[layer].CacheKey, _ffnGateUpQW[layer].GgmlType,
                            _ffnGateUpQW[layer].Ne0, _ffnGateUpQW[layer].Ne1, _ffnGateUpQW[layer].RawBytes,
                            _ffnDownQW[layer].CacheKey, _ffnDownQW[layer].GgmlType,
                            _ffnDownQW[layer].Ne0, _ffnDownQW[layer].Ne1, _ffnDownQW[layer].RawBytes,
                            halfDim);
                        _linearTicks += Stopwatch.GetTimestamp() - t0;
                        attnOut.Dispose();
                        if (profilePrefill)
                            _prefillAttnFfnTicks += Stopwatch.GetTimestamp() - ffnStart;
                        return hidden;
                    }
                    catch { /* fall through */ }
                }

                // Fallback: separate output proj + residual.
                if (TryLinearAddInto(hidden, attnOut, _attnOutputQW[layer]))
                    attnOut.Dispose();
                else
                {
                    var o = LinearForwardCached(attnOut, _attnOutputQW[layer], _attnOutputF32[layer]);
                    attnOut.Dispose();
                    Ops.Add(hidden, hidden, o);
                    o.Dispose();
                }
                attnOut = null;
            }

            if (attnOut != null)
            {
                Ops.Add(hidden, hidden, attnOut);
                attnOut.Dispose();
            }

            Tensor ffnOut;
            if (_isMoeLayer != null && _isMoeLayer[layer])
            {
                Tensor normed2;
                bool ownsNormed2 = true;
                if (seqLen == 1 && IsGgmlBackend && _moeTokenInput != null
                    && _moeTokenInput.Sizes[1] == Config.HiddenSize
                    && _postAttnNormW[layer] != null)
                {
                    RMSNormToBufferCpu(_moeTokenInput, hidden, _postAttnNormW[layer], Config.HiddenSize, Config.Eps);
                    normed2 = _moeTokenInput;
                    ownsNormed2 = false;
                }
                else
                    normed2 = RMSNormOpCached(hidden, _postAttnNormW[layer]);

                if (seqLen == 1 && TryMoEResidualDecode(hidden, normed2, layer))
                    ffnOut = null;
                else
                    ffnOut = MoEForward(normed2, layer, seqLen);
                if (ownsNormed2)
                    normed2.Dispose();
            }
            else
            {
                ffnOut = FFNCachedFused(hidden, _postAttnNormW[layer], layer, seqLen);
            }

            if (ffnOut != null)
            {
                Ops.Add(hidden, hidden, ffnOut);
                ffnOut.Dispose();
            }

            if (profilePrefill)
                _prefillAttnFfnTicks += Stopwatch.GetTimestamp() - ffnStart;

            return hidden;
        }

        // 中文：完整带门控注意力实现：输入归一化 + 融合 QKV → 拆分 Q/gate → QK-norm → RoPE/MRoPE → 写入 KV 缓存 → flash/SIMD 注意力 → sigmoid 门控混合 → 输出投影（可与残差融合），返回输出张量或在已就地加残差时返回 null。
        private unsafe Tensor FullAttention(Tensor input, Tensor inputNormW, int layer, int seqLen, int startPos,
            Tensor residual = null, bool skipOutputProj = false)
        {
            int numHeads = Config.NumHeads;
            int numKVHeads = Config.NumKVHeads;
            int headDim = Config.HeadDim;
            int qFullDim = numHeads * headDim * 2;
            int kvDim = numKVHeads * headDim;
            int totalSeqLen = startPos + seqLen;
            bool profilePrefill = _profilePrefillStages && seqLen > 1;
            long stageStart = profilePrefill ? Stopwatch.GetTimestamp() : 0;

            // Fused norm + QKV when the fused-QKV weight is available; otherwise we have to
            // produce the normalized input separately for the three independent projections.
            Tensor qFull;
            Tensor kTensor;
            Tensor vTensor;
            Tensor normedInput = null;
            Tensor fusedQkv;
            bool ownsFusedQkv = true;
            if (inputNormW != null && _attnQkvQW[layer] != null)
            {
                // Decode hot path: write fused norm+QKV into the pre-allocated buffer to
                // avoid a tensor allocation per layer per token. Falls back to the regular
                // allocating path for prefill (multi-row) or when the buffer doesn't fit.
                if (seqLen == 1 && _attnDecodeQkvBuf != null
                    && _attnDecodeQkvBuf.Sizes[1] == qFullDim + 2 * kvDim)
                {
                    fusedQkv = TryFusedNormLinearInto(_attnDecodeQkvBuf, input, inputNormW, _attnQkvQW[layer]);
                    if (fusedQkv != null)
                        ownsFusedQkv = false;
                    else
                        fusedQkv = FusedNormLinear(input, inputNormW, _attnQkvQW[layer], _attnQkvF32[layer]);
                }
                else
                {
                    fusedQkv = FusedNormLinear(input, inputNormW, _attnQkvQW[layer], _attnQkvF32[layer]);
                }
            }
            else
            {
                normedInput = inputNormW != null ? RMSNormOpCached(input, inputNormW) : input.CopyRef();
                fusedQkv = LinearForwardCached(normedInput, _attnQkvQW[layer], _attnQkvF32[layer]);
            }
            if (fusedQkv != null)
            {
                if (seqLen == 1)
                {
                    qFull = fusedQkv.Narrow(1, 0, qFullDim);
                    kTensor = fusedQkv.Narrow(1, qFullDim, kvDim);
                    vTensor = fusedQkv.Narrow(1, qFullDim + kvDim, kvDim);
                }
                else
                {
                    using (var qView = fusedQkv.Narrow(1, 0, qFullDim))
                        qFull = Ops.NewContiguous(qView);
                    using (var kView = fusedQkv.Narrow(1, qFullDim, kvDim))
                        kTensor = Ops.NewContiguous(kView);
                    using (var vView = fusedQkv.Narrow(1, qFullDim + kvDim, kvDim))
                        vTensor = Ops.NewContiguous(vView);
                }
                if (ownsFusedQkv)
                    fusedQkv.Dispose();
            }
            else
            {
                if (normedInput == null)
                    normedInput = inputNormW != null ? RMSNormOpCached(input, inputNormW) : input.CopyRef();

                // Q projection outputs [seqLen, numHeads * headDim * 2] (Q + gate interleaved per head).
                qFull = LinearForwardCached(normedInput, _attnQQW[layer], _attnQF32[layer]);
                kTensor = LinearForwardCached(normedInput, _attnKQW[layer], _attnKF32[layer]);
                vTensor = LinearForwardCached(normedInput, _attnVQW[layer], _attnVF32[layer]);
            }
            normedInput?.Dispose();
            if (profilePrefill) { long now = Stopwatch.GetTimestamp(); _prefillAttnQkvTicks += now - stageStart; stageStart = now; }

            Tensor qTensor, gateTensor;
            bool ownsQGateBuffers;
            DeinterleaveQGate(qFull, seqLen, numHeads, headDim, out qTensor, out gateTensor, out ownsQGateBuffers);
            qFull.Dispose();
            if (profilePrefill) { long now = Stopwatch.GetTimestamp(); _prefillAttnDeinterleaveTicks += now - stageStart; stageStart = now; }

            qTensor = ApplyQKNormCached(qTensor, _attnQNormW[layer], numHeads, seqLen);
            kTensor = ApplyQKNormCached(kTensor, _attnKNormW[layer], numKVHeads, seqLen);
            if (profilePrefill) { long now = Stopwatch.GetTimestamp(); _prefillAttnQknormTicks += now - stageStart; stageStart = now; }

            // RoPE - decode path applies Q and K together so the cos/sin table is computed once.
            // If the multimodal injector has staged per-axis MRoPE positions for
            // this prefill (vision prompt on Qwen3.5), route through ApplyMRoPEPrefill
            // so image-region rotary dims get the right (T,H,W) angle interleaving.
            // Text-only prompts and post-image decode tokens use the scalar path.
            bool useMRoPE = _pendingMRoPEPositions != null && _pendingMRoPEPositions.Length >= 3 * seqLen;
            if (useMRoPE)
            {
                qTensor = ApplyMRoPEPrefill(qTensor, numHeads, seqLen, _pendingMRoPEPositions);
                kTensor = ApplyMRoPEPrefill(kTensor, numKVHeads, seqLen, _pendingMRoPEPositions);
            }
            else if (seqLen == 1)
            {
                if (_backend == BackendType.Mlx)
                {
                    qTensor = ApplyRoPEPrefill(qTensor, numHeads, seqLen, startPos);
                    kTensor = ApplyRoPEPrefill(kTensor, numKVHeads, seqLen, startPos);
                }
                else
                {
                    ApplyRoPEDecodeQKInPlace(qTensor, kTensor, numHeads, numKVHeads, startPos);
                }
            }
            else
            {
                qTensor = ApplyRoPEPrefill(qTensor, numHeads, seqLen, startPos);
                kTensor = ApplyRoPEPrefill(kTensor, numKVHeads, seqLen, startPos);
            }
            if (profilePrefill) { long now = Stopwatch.GetTimestamp(); _prefillAttnRopeTicks += now - stageStart; stageStart = now; }

            long t0 = Stopwatch.GetTimestamp();

            // Attention computation
            float attentionScale = 1.0f / MathF.Sqrt(headDim);
            Tensor attnOutput = null;

            if (seqLen == 1)
            {
                // Reuse the pre-allocated decode output buffer to avoid a per-token allocation.
                attnOutput = _attnDecodeOutBuf != null
                    ? _attnDecodeOutBuf
                    : new Tensor(_allocator, DType.Float32, 1, numHeads * headDim);

                int maxSeqLen = (int)_kvCacheK[layer].Sizes[1];

                // Fast path: fuse KV cache append + flash attention into a single device graph.
                // Only worth the GPU dispatch overhead once the cache is large enough that
                // CPU SIMD attention starts to dominate. Below the threshold the per-layer
                // per-token Metal command buffer setup costs more than the saved compute.
                bool flashOk = false;
                bool cacheCopied = false;
                if (_backend == BackendType.Mlx
                    && headDim == 256
                    && _mlxAttentionCache?[layer] != null
                    && _mlxAttentionCache[layer].Length == startPos)
                {
                    using Tensor qHeads = qTensor.View(numHeads, 1, headDim);
                    using Tensor kHeads = kTensor.View(numKVHeads, 1, headDim);
                    using Tensor vHeads = vTensor.View(numKVHeads, 1, headDim);
                    flashOk = _mlxAttentionCache[layer].TryAttentionHeadDim256(
                        attnOutput,
                        qHeads,
                        kHeads,
                        vHeads,
                        numHeads,
                        numKVHeads,
                        1,
                        startPos,
                        causal: true);
                    cacheCopied = flashOk;
                }
                else if (IsGgmlBackend && totalSeqLen >= FlashAttnDecodeMinSeqLen)
                {
                    flashOk = TryFlashAttnDecode(qTensor, kTensor, vTensor,
                        _kvCacheK[layer], _kvCacheV[layer], attnOutput,
                        numHeads, numKVHeads, headDim, maxSeqLen, startPos, attentionScale);
                }
                else if (_backend == BackendType.Mlx && totalSeqLen >= MlxFlashAttnDecodeMinSeqLen)
                {
                    CopyToCacheDecode(_kvCacheK[layer], kTensor, _kvCacheV[layer], vTensor,
                        numKVHeads, headDim, startPos);
                    cacheCopied = true;
                    flashOk = MlxFusedOps.TryDecodeAttention(
                        attnOutput, qTensor, _kvCacheK[layer], _kvCacheV[layer],
                        numHeads, numKVHeads, headDim,
                        0, totalSeqLen, maxSeqLen, false, attentionScale);
                }

                if (!flashOk)
                {
                    if (!cacheCopied)
                        CopyToCacheDecode(_kvCacheK[layer], kTensor, _kvCacheV[layer], vTensor,
                            numKVHeads, headDim, startPos);
                    AttentionDecodePureCS(qTensor, _kvCacheK[layer], _kvCacheV[layer],
                        attnOutput, numHeads, numKVHeads, headDim, totalSeqLen, attentionScale);
                }

                kTensor.Dispose();
                vTensor.Dispose();
                if (ownsQGateBuffers)
                    qTensor.Dispose();
            }
            else
            {
                if (profilePrefill) { long now = Stopwatch.GetTimestamp(); _prefillAttnReshapeTicks += now - stageStart; stageStart = now; }

                Tensor kHeads = ReshapeToHeads(kTensor, numKVHeads, seqLen, headDim);
                Tensor vHeads = ReshapeToHeads(vTensor, numKVHeads, seqLen, headDim);

                if (profilePrefill) { long now2 = Stopwatch.GetTimestamp(); _prefillAttnExpandKvTicks += now2 - stageStart; stageStart = now2; }

                // Fused GPU attention: Q*K^T → causal mask → softmax → *V in one
                // GGML graph dispatch, eliminating ExpandKVHeads + 5 separate ops.
                //
                // FusedPrefillAttention is F32-only on the native side. The continuation
                // branch (startPos > 0) reads K/V directly from the persistent KV cache,
                // which may be Float16 when the caller picked a quantized cache dtype
                // (e.g. on IQ2_XXS / Q4_K models where ApplyModelAlignedKvCacheDefault
                // selects F16 to halve cache memory). Reading F16 bytes as F32 silently
                // produces garbage that propagates through softmax+V into the residual,
                // ultimately surfacing as a NaN router probability in the next MoE layer
                // and an out-of-range expert index that crashes inside the native
                // ggml_mul_mat_id dispatch. Restrict the continuation fast path to F32
                // caches and fall back to ExpandKVHeads (which handles F16) otherwise.
                bool usedFusedAttn = false;
                bool usedMlxAttentionCache = false;
                bool kvCacheIsF32 = _kvCacheK[layer].ElementType == DType.Float32
                    && _kvCacheV[layer].ElementType == DType.Float32;
                bool canFuseContinuation = kvCacheIsF32 || startPos == 0;
                if (_backend == BackendType.Mlx
                    && headDim == 256
                    && _mlxAttentionCache?[layer] != null
                    && _mlxAttentionCache[layer].Length == startPos)
                {
                    Tensor qHeadsForAttn = null;
                    try
                    {
                        attnOutput = new Tensor(_allocator, DType.Float32, seqLen, numHeads * headDim);
                        qHeadsForAttn = ReshapeToHeads(qTensor, numHeads, seqLen, headDim);
                        if (_mlxAttentionCache[layer].TryAttentionHeadDim256(
                            attnOutput,
                            qHeadsForAttn,
                            kHeads,
                            vHeads,
                            numHeads,
                            numKVHeads,
                            seqLen,
                            startPos,
                            causal: true))
                        {
                            usedFusedAttn = true;
                            usedMlxAttentionCache = true;
                            qTensor.Dispose();
                            kTensor.Dispose();
                            vTensor.Dispose();
                        }
                        else
                        {
                            attnOutput.Dispose();
                            attnOutput = null;
                        }
                    }
                    finally
                    {
                        qHeadsForAttn?.Dispose();
                    }
                }

                bool tryMlxPrefillAttention = _backend == BackendType.Mlx
                    && !usedFusedAttn
                    && (headDim <= 128
                        || string.Equals(Environment.GetEnvironmentVariable("TS_MLX_CHUNKED_VECTOR_PREFILL"), "1", StringComparison.Ordinal));
                if (tryMlxPrefillAttention)
                {
                    Tensor qHeadsForAttn = null;
                    try
                    {
                        attnOutput = new Tensor(_allocator, DType.Float32, seqLen, numHeads * headDim);
                        qHeadsForAttn = ReshapeToHeads(qTensor, numHeads, seqLen, headDim);
                        if (MlxFusedOps.TryPrefillAttention(
                            attnOutput, qHeadsForAttn, kHeads, vHeads,
                            numHeads, numKVHeads, headDim,
                            seqLen, totalSeqLen,
                            startPos, 0, attentionScale))
                        {
                            usedFusedAttn = true;
                            qTensor.Dispose();
                            kTensor.Dispose();
                            vTensor.Dispose();
                        }
                        else
                        {
                            attnOutput.Dispose();
                            attnOutput = null;
                        }
                    }
                    finally
                    {
                        qHeadsForAttn?.Dispose();
                    }
                }

                // Write to KV cache after the MLX prefill-attention attempt. The
                // default MLX cache copy uses host pointers for short prompts; doing
                // that before attention dirties K/V and forces a re-upload.
                if (!usedMlxAttentionCache)
                {
                    CopyToCache(_kvCacheK[layer], kHeads, startPos, seqLen);
                    CopyToCache(_kvCacheV[layer], vHeads, startPos, seqLen);
                }
                if (profilePrefill) { long now = Stopwatch.GetTimestamp(); _prefillAttnCacheCopyTicks += now - stageStart; stageStart = now; }

                if (!usedFusedAttn && IsGgmlBackend && canFuseContinuation)
                {
                    try
                    {
                        attnOutput = new Tensor(_allocator, DType.Float32, seqLen, numHeads * headDim);

                        if (startPos == 0)
                        {
                            // Initial prefill: pass flat Q/K/V directly (inputFormat=1).
                            // The GPU kernel does reshape+permute as free graph ops,
                            // eliminating the ReshapeToHeads copies for Q.
                            GgmlBasicOps.FusedPrefillAttention(
                                qTensor, kTensor, vTensor, attnOutput,
                                numHeads, numKVHeads, headDim,
                                seqLen, seqLen,
                                0, 0, attentionScale, inputFormat: 1);
                        }
                        else
                        {
                            // Continuation: KV cache is head-first format (inputFormat=0),
                            // but Q is flat (inputFormat=1). Use head-first path with cache.
                            Tensor qHeadsForAttn = ReshapeToHeads(qTensor, numHeads, seqLen, headDim);
                            GgmlBasicOps.FusedPrefillAttention(
                                qHeadsForAttn, _kvCacheK[layer], _kvCacheV[layer], attnOutput,
                                numHeads, numKVHeads, headDim,
                                seqLen, totalSeqLen,
                                startPos, 0, attentionScale, inputFormat: 0);
                            qHeadsForAttn.Dispose();
                        }
                        usedFusedAttn = true;
                        qTensor.Dispose();
                        kTensor.Dispose();
                        vTensor.Dispose();
                    }
                    catch
                    {
                        attnOutput?.Dispose();
                        attnOutput = null;
                    }
                }
                kHeads.Dispose();
                vHeads.Dispose();

                if (!usedFusedAttn)
                {
                    Tensor qHeads = ReshapeToHeads(qTensor, numHeads, seqLen, headDim);
                    qTensor.Dispose();
                    kTensor.Dispose();
                    vTensor.Dispose();

                    int groupSize = numHeads / numKVHeads;
                    Tensor kExpanded = ExpandKVHeads(_kvCacheK[layer], groupSize, totalSeqLen);
                    Tensor vExpanded = ExpandKVHeads(_kvCacheV[layer], groupSize, totalSeqLen);

                    using var kT = kExpanded.Transpose(1, 2);
                    var scores = new Tensor(_allocator, DType.Float32, numHeads, seqLen, totalSeqLen);
                    Ops.AddmmBatch(scores, 0, scores, attentionScale, qHeads, kT);
                    qHeads?.Dispose();
                    kExpanded.Dispose();

                    // Fused causal-mask + softmax on GPU. Replaces AddCausalMask + Softmax
                    // (two separate ops) with one Metal kernel. No sinks for Qwen3.5
                    // dense attention.
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

                    var attnOut = new Tensor(_allocator, DType.Float32, numHeads, seqLen, headDim);
                    Ops.AddmmBatch(attnOut, 0, attnOut, 1.0f, scores, vExpanded);
                    scores.Dispose();
                    vExpanded.Dispose();

                    attnOutput = ReshapeFromHeads(attnOut, numHeads, seqLen, headDim);
                    attnOut.Dispose();
                }
                if (profilePrefill) { long now = Stopwatch.GetTimestamp(); _prefillAttnComputeTicks += now - stageStart; stageStart = now; }
            }

            // Decode hot path: do the sigmoid-gated mix on CPU. The data is tiny
            // (single row, numHeads * headDim floats) so the GPU dispatch overhead
            // dominates. Eliminates one Metal command buffer per attention layer.
            if (seqLen == 1 && _backend != BackendType.Mlx && attnOutput != null && gateTensor != null
                && attnOutput.ElementType == DType.Float32 && gateTensor.ElementType == DType.Float32)
                ApplySigmoidGateCpu(attnOutput, gateTensor);
            else
                ApplySigmoidGate(attnOutput, gateTensor);
            if (ownsQGateBuffers)
                gateTensor.Dispose();
            if (profilePrefill) { long now = Stopwatch.GetTimestamp(); _prefillAttnGateTicks += now - stageStart; stageStart = now; }

            _attnTicks += Stopwatch.GetTimestamp() - t0;

            bool ownsAttnOutput = !(seqLen == 1 && _attnDecodeOutBuf != null && ReferenceEquals(attnOutput, _attnDecodeOutBuf));

            // When skipOutputProj is set, return the raw attention output (post sigmoid gate)
            // so the caller can fuse output_proj + FFN into one dispatch.
            if (skipOutputProj)
            {
                Tensor rawOut = ownsAttnOutput ? attnOutput : Ops.NewContiguous(attnOutput);
                if (profilePrefill) _prefillAttnOutputTicks += Stopwatch.GetTimestamp() - stageStart;
                _attnTicks += Stopwatch.GetTimestamp() - t0;
                return rawOut;
            }

            // Fast path: fuse output projection with the residual add (eliminates the
            // intermediate output tensor and one GPU sync). Only valid for matching shapes.
            if (residual != null
                && _attnOutputQW[layer] != null
                && residual.DimensionCount == 2
                && attnOutput.DimensionCount == 2
                && residual.Sizes[0] == attnOutput.Sizes[0]
                && TryLinearAddInto(residual, attnOutput, _attnOutputQW[layer]))
            {
                if (ownsAttnOutput)
                    attnOutput.Dispose();
                if (profilePrefill) _prefillAttnOutputTicks += Stopwatch.GetTimestamp() - stageStart;
                return null;
            }

            Tensor output = LinearForwardCached(attnOutput, _attnOutputQW[layer], _attnOutputF32[layer]);
            if (ownsAttnOutput)
                attnOutput.Dispose();
            if (profilePrefill) _prefillAttnOutputTicks += Stopwatch.GetTimestamp() - stageStart;
            return output;
        }

        // 中文：对注意力输出施加 sigmoid 门控混合：attn = attn * sigmoid(gate)（GPU 路径）。
        private void ApplySigmoidGate(Tensor attn, Tensor gate)
        {
            Ops.SigmoidMul(attn, attn, gate);
        }

        /// <summary>
        /// CPU SIMD sigmoid-gated mix for the single-row decode path:
        /// <c>attn[i] = attn[i] * sigmoid(gate[i])</c>. The tensors are small enough that
        /// going through the GPU just incurs Metal dispatch overhead with no compute win.
        /// </summary>
        // 中文：单行解码热路径的 CPU 版 sigmoid 门控混合 attn[i] *= sigmoid(gate[i])，数据量小以避免 GPU 调度开销。
        private unsafe void ApplySigmoidGateCpu(Tensor attn, Tensor gate)
        {
            float* aPtr = GetFloatPtr(attn);
            float* gPtr = GetFloatPtr(gate);
            int n = (int)attn.ElementCount();

            int vLen = Vector<float>.Count;
            int i = 0;
            // Sigmoid via 1/(1+exp(-x)) is not directly vectorizable in System.Numerics
            // (no element-wise exp). Process scalar but keep the multiply vectorized.
            // Going scalar here is still ~5x faster than a GPU dispatch for ~4KB of data.
            for (; i < n; i++)
            {
                float g = gPtr[i];
                float sig = 1.0f / (1.0f + MathF.Exp(-g));
                aPtr[i] = aPtr[i] * sig;
            }

            InvalidateTensorDeviceCache(attn);
        }

        /// <summary>
        /// Single-pass deinterleave of the gated Q projection into separate Q and gate tensors.
        /// The fused Q projection produces [seqLen, numHeads, 2*headDim] where each head holds
        /// Q followed by gate. The original implementation called Narrow + NewContiguous twice,
        /// each of which allocates and copies. Doing the split with explicit Buffer.MemoryCopy
        /// avoids the intermediate strided views and pays a single contiguous copy per slice.
        /// </summary>
        // 中文：把门控 Q 投影 [seqLen, numHeads, 2*headDim]（每头 Q 后接 gate）一次性拆分为独立的 Q 与 gate 张量，按后端走 GPU 视图或 CPU memcpy 路径。
        private unsafe void DeinterleaveQGate(Tensor qFull, int seqLen, int numHeads, int headDim,
            out Tensor qTensor, out Tensor gateTensor, out bool ownsBuffers)
        {
            int dimPerHead = headDim * 2;
            int totalPerToken = numHeads * headDim;

            // GPU deinterleave (Strided View + Contiguous) keeps everything on
            // device. The CPU path below is faster per-call for tiny tensors
            // but forces 3 GetFloatPtr syncs per attention layer; on MLX decode
            // that's 180 round trips per token. Always prefer the GPU path for
            // MLX seqLen==1; for prefill keep the historical headDim==256 gate
            // (parallel CPU SIMD pays off for the larger per-token work).
            bool mlxGpuDeinterleave = _backend == BackendType.Mlx
                && qFull.Storage is MlxStorage
                && (seqLen == 1
                    || headDim == 256
                    || string.Equals(Environment.GetEnvironmentVariable("TS_MLX_QWEN_GPU_DEINTERLEAVE"), "1", StringComparison.Ordinal));
            if (mlxGpuDeinterleave)
            {
                using Tensor qGate = qFull.View(seqLen, numHeads, 2, headDim);
                using Tensor qView = qGate.Select(2, 0);
                using Tensor gateView = qGate.Select(2, 1);
                Tensor qContiguous = Ops.NewContiguous(qView);
                Tensor gateContiguous = Ops.NewContiguous(gateView);
                qTensor = qContiguous.View(seqLen, totalPerToken);
                gateTensor = gateContiguous.View(seqLen, totalPerToken);
                qContiguous.Dispose();
                gateContiguous.Dispose();
                ownsBuffers = true;
                return;
            }

            // Decode hot path: re-use the pre-allocated decode buffers'
            // STORAGE but hand back a fresh Tensor object that AddRef'd it.
            // Returning the bare _attnDecodeQBuf reference was a footgun:
            // any unguarded qTensor.Dispose() in the caller would drop the
            // storage's refcount to 0 and free the model's persistent
            // decode buffer, then the next attention layer's
            // DeinterleaveQGate would crash with a NullReferenceException
            // when GetFloatPtr returned the now-null buffer pointer (the
            // Qwen35 batched ForwardBatch hit exactly this on every
            // decode token). The CopyRef'd handle owns its own refcount
            // increment; caller can freely Dispose it and the underlying
            // storage stays alive as long as _attnDecodeQBuf does.
            if (seqLen == 1 && _attnDecodeQBuf != null && _attnDecodeGBuf != null
                && _attnDecodeQBuf.Sizes[1] == totalPerToken)
            {
                qTensor = _attnDecodeQBuf.CopyRef();
                gateTensor = _attnDecodeGBuf.CopyRef();
                ownsBuffers = true;
            }
            else
            {
                qTensor = new Tensor(_allocator, DType.Float32, seqLen, totalPerToken);
                gateTensor = new Tensor(_allocator, DType.Float32, seqLen, totalPerToken);
                ownsBuffers = true;
            }

            float* src = GetFloatPtr(qFull);
            float* qDst = GetFloatPtr(qTensor);
            float* gDst = GetFloatPtr(gateTensor);
            if (src == null || qDst == null || gDst == null)
            {
                throw new InvalidOperationException(
                    $"DeinterleaveQGate: null storage pointer "
                    + $"(src={(IntPtr)src:x}, qDst={(IntPtr)qDst:x}, gDst={(IntPtr)gDst:x}). "
                    + $"seqLen={seqLen}, numHeads={numHeads}, headDim={headDim}. "
                    + $"qFull.Storage={qFull.Storage?.GetType().Name}, "
                    + $"qTensor.Storage={qTensor.Storage?.GetType().Name}, "
                    + $"gateTensor.Storage={gateTensor.Storage?.GetType().Name}.");
            }
            int headBytes = headDim * sizeof(float);

            if (seqLen > 1)
            {
                // Prefill path: parallelize across tokens. Each token's deinterleave
                // is independent, so this scales linearly with CPU core count.
                long srcL = (long)src, qDstL = (long)qDst, gDstL = (long)gDst;
                int capturedNumHeads = numHeads;
                int capturedDimPerHead = dimPerHead;
                int capturedTotalPerToken = totalPerToken;
                int capturedHeadDim = headDim;
                int capturedHeadBytes = headBytes;

                Parallel.For(0, seqLen, s =>
                {
                    float* srcRow = (float*)srcL + (long)s * capturedNumHeads * capturedDimPerHead;
                    float* qRow = (float*)qDstL + (long)s * capturedTotalPerToken;
                    float* gRow = (float*)gDstL + (long)s * capturedTotalPerToken;
                    for (int h = 0; h < capturedNumHeads; h++)
                    {
                        float* srcHead = srcRow + h * capturedDimPerHead;
                        Buffer.MemoryCopy(srcHead, qRow + h * capturedHeadDim, capturedHeadBytes, capturedHeadBytes);
                        Buffer.MemoryCopy(srcHead + capturedHeadDim, gRow + h * capturedHeadDim, capturedHeadBytes, capturedHeadBytes);
                    }
                });
            }
            else
            {
                float* srcRow = src;
                float* qRow = qDst;
                float* gRow = gDst;
                for (int h = 0; h < numHeads; h++)
                {
                    float* srcHead = srcRow + h * dimPerHead;
                    Buffer.MemoryCopy(srcHead, qRow + h * headDim, headBytes, headBytes);
                    Buffer.MemoryCopy(srcHead + headDim, gRow + h * headDim, headBytes, headBytes);
                }
            }

            InvalidateTensorDeviceCache(qTensor);
            InvalidateTensorDeviceCache(gateTensor);
        }

        /// <summary>
        /// FFN with pre-resolved weight references, mirroring <see cref="ModelBase.FFN"/>
        /// but skipping the dictionary lookup. SwiGLU on a fused gate+up projection.
        /// </summary>
        // 中文：稠密 FFN（使用预解析权重引用）：对融合 gate_up 投影做 SwiGLU（SiLU(gate)*up）后经 down 投影返回结果。
        private Tensor FFNCached(Tensor input, int layer, int seqLen)
        {
            int intermSize = Config.IntermediateSize;
            Tensor gateUp = LinearForwardCached(input, _ffnGateUpQW[layer], _ffnGateUpF32[layer]);
            int halfDim = intermSize > 0 ? intermSize : (int)(gateUp.Sizes[1] / 2);

            Tensor gate, up;
            if (seqLen == 1)
            {
                gate = gateUp.Narrow(1, 0, halfDim);
                up = gateUp.Narrow(1, halfDim, halfDim);
            }
            else
            {
                using (var gView = gateUp.Narrow(1, 0, halfDim))
                    gate = Ops.NewContiguous(gView);
                using (var uView = gateUp.Narrow(1, halfDim, halfDim))
                    up = Ops.NewContiguous(uView);
            }
            gateUp.Dispose();

            Ops.SiLUMul(gate, gate, up);
            up.Dispose();

            Tensor down = LinearForwardCached(gate, _ffnDownQW[layer], _ffnDownF32[layer]);
            gate.Dispose();
            return down;
        }

        /// <summary>
        /// Dense FFN with fused post-attn-norm + gate_up projection AND fused down + residual add.
        /// Reduces 4 GPU dispatches (norm, gateUp, down, add) to 2, plus the SiLU*Mul activation.
        /// Returns null when the down+residual fusion succeeded (residual already updated in-place);
        /// otherwise returns the down output for the caller to add manually.
        /// </summary>
        // 中文：稠密 FFN 的融合版：把后注意力归一化 + gate_up 投影、以及 down + 残差加分别融合，prefill 时可整块塞进单次 GGML 图；成功融合残差加时返回 null，否则返回 down 输出供调用方手动相加。
        private Tensor FFNCachedFused(Tensor residual, Tensor postNormW, int layer, int seqLen)
        {
            int intermSize = Config.IntermediateSize;

            // Prefill fast path: collapse the entire dense SwiGLU FFN
            //   residual += down_W^T @ ( silu(gate_part) * up_part )
            // into a single GGML graph dispatch. Replaces 3 separate dispatches
            // (FusedRmsNormMatMulQuant + SiLUMulSplit + FusedMatMulQuantAdd)
            // with one, removing the matched graph builds, allocations and
            // host<->backend syncs. This is the dominant prefill cost on the
            // hybrid GatedDeltaNet+Attention Qwen35 architecture. Set
            // QWEN35_DISABLE_FUSED_FFN=1 to bypass the fast path for A/B
            // comparison or debugging.
            if (seqLen > 1
                && IsGgmlBackend
                && _useFusedFfnPrefill
                && postNormW != null
                && _ffnGateUpQW[layer] != null
                && _ffnDownQW[layer] != null
                && residual.DimensionCount == 2
                && residual.Sizes[0] == seqLen
                && residual.Sizes[1] == _ffnGateUpQW[layer].Ne0)
            {
                int halfDimFused = intermSize > 0 ? intermSize : (int)(_ffnGateUpQW[layer].Ne1 / 2);
                if (halfDimFused > 0 && _ffnGateUpQW[layer].Ne1 == 2L * halfDimFused
                    && _ffnDownQW[layer].Ne0 == halfDimFused
                    && _ffnDownQW[layer].Ne1 == residual.Sizes[1])
                {
                    long t0 = Stopwatch.GetTimestamp();
                    GgmlBasicOps.FusedFFNSwiGLUQuant(residual, residual, postNormW, Config.Eps,
                        _ffnGateUpQW[layer].CacheKey, _ffnGateUpQW[layer].GgmlType,
                        _ffnGateUpQW[layer].Ne0, _ffnGateUpQW[layer].Ne1, _ffnGateUpQW[layer].RawBytes,
                        _ffnDownQW[layer].CacheKey, _ffnDownQW[layer].GgmlType,
                        _ffnDownQW[layer].Ne0, _ffnDownQW[layer].Ne1, _ffnDownQW[layer].RawBytes,
                        halfDimFused);
                    _linearTicks += Stopwatch.GetTimestamp() - t0;
                    return null;
                }
            }

            // Fused norm + gate_up projection. Decode reuses a pre-allocated [1, 2*intermSize]
            // buffer to avoid one tensor allocation per layer per token.
            Tensor gateUp = null;
            bool ownsGateUp = true;
            if (postNormW != null && _ffnGateUpQW[layer] != null && IsGgmlBackend)
            {
                if (seqLen == 1 && _ffnDecodeGateUpBuf != null
                    && _ffnDecodeGateUpBuf.Sizes[1] == _ffnGateUpQW[layer].Ne1)
                {
                    gateUp = TryFusedNormLinearInto(_ffnDecodeGateUpBuf, residual, postNormW, _ffnGateUpQW[layer]);
                    if (gateUp != null)
                        ownsGateUp = false;
                }
                if (gateUp == null)
                    gateUp = FusedNormLinear(residual, postNormW, _ffnGateUpQW[layer], _ffnGateUpF32[layer]);
            }

            if (gateUp == null)
            {
                // Fallback: explicit norm then gate_up.
                Tensor normed = RMSNormOpCached(residual, postNormW);
                gateUp = LinearForwardCached(normed, _ffnGateUpQW[layer], _ffnGateUpF32[layer]);
                normed.Dispose();
            }

            int halfDim = intermSize > 0 ? intermSize : (int)(gateUp.Sizes[1] / 2);

            Tensor gate;
            if (seqLen == 1)
            {
                Tensor gView = gateUp.Narrow(1, 0, halfDim);
                Tensor uView = gateUp.Narrow(1, halfDim, halfDim);
                if (IsGgmlBackend)
                    SiLUMulInPlaceCpu(gView, uView);
                else
                    Ops.SiLUMul(gView, gView, uView);
                uView.Dispose();
                gate = gView;
            }
            else if ((IsGgmlBackend || _backend == BackendType.Mlx) && ownsGateUp)
            {
                // Fused split path: silu(gate_up[:, :H]) * gate_up[:, H:] without the
                // two large contiguous half-copies used by the legacy split path.
                gate = new Tensor(gateUp.Allocator, DType.Float32, gateUp.Sizes[0], halfDim);
                Ops.SiLUMulSplit(gate, gateUp, halfDim);
            }
            else
            {
                using (var gView = gateUp.Narrow(1, 0, halfDim))
                    gate = Ops.NewContiguous(gView);
                using (var uView = gateUp.Narrow(1, halfDim, halfDim))
                {
                    Ops.SiLUMul(gate, gate, uView);
                }
            }
            if (ownsGateUp)
                gateUp.Dispose();

            // Fused down + residual add.
            if (residual.DimensionCount == 2
                && gate.DimensionCount == 2
                && residual.Sizes[0] == gate.Sizes[0]
                && _ffnDownQW[layer] != null
                && TryLinearAddInto(residual, gate, _ffnDownQW[layer]))
            {
                gate.Dispose();
                return null;
            }

            Tensor down = LinearForwardCached(gate, _ffnDownQW[layer], _ffnDownF32[layer]);
            gate.Dispose();
            return down;
        }

        /// <summary>
        /// RMSNorm with a pre-resolved alpha tensor, avoiding the dictionary lookup that
        /// <see cref="ModelBase.RMSNormOp"/> performs per call. The arithmetic is identical.
        /// </summary>
        // 中文：使用预解析 alpha 权重的 RMSNorm，避免每次字典查找，运算与 ModelBase.RMSNormOp 一致。
        private Tensor RMSNormOpCached(Tensor input, Tensor alpha)
        {
            long t0 = Stopwatch.GetTimestamp();
            int rows = (int)input.Sizes[0];
            int dim = (int)(input.ElementCount() / rows);

            Tensor input2d = input.Sizes.Length != 2 ? input.View(rows, dim) : null;
            Tensor src = input2d ?? input;

            Tensor result = Ops.RMSNorm(null, src, alpha, null, Config.Eps);

            input2d?.Dispose();
            _normTicks += Stopwatch.GetTimestamp() - t0;
            return result;
        }

        /// <summary>
        /// Fused RMSNorm + quantized matmul in a single GGML kernel dispatch.
        /// Equivalent to: matmul(rms_norm(input, normW, eps), qW), but reduces 2 dispatches to 1
        /// and skips materialising the intermediate normalized tensor on the GPU.
        /// Falls back to the unfused path when the GGML backend or quant weight is unavailable.
        /// </summary>
        // 中文：把 RMSNorm 与量化矩阵乘融合到单次 kernel（GGML/MLX），等价 matmul(rms_norm(input), qW)；后端或权重不可用时回退到显式 norm + linear。
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Tensor FusedNormLinear(Tensor input, Tensor normW, QuantizedWeight qw, Tensor wF32)
        {
            // Fused path: needs GGML backend, a quantized weight, and a 2D input view.
            if (IsGgmlBackend && qw != null && normW != null && input.DimensionCount == 2)
            {
                long t0 = Stopwatch.GetTimestamp();
                int seqLen = (int)input.Sizes[0];
                int outDim = (int)qw.Ne1;
                Tensor result = new Tensor(_allocator, DType.Float32, seqLen, outDim);
                GgmlBasicOps.FusedRmsNormMatMulQuant(result, input, normW, Config.Eps,
                    qw.CacheKey, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes);
                _linearTicks += Stopwatch.GetTimestamp() - t0;
                return result;
            }

            if (_backend == BackendType.Mlx && qw != null && normW != null && input.DimensionCount == 2)
            {
                long t0 = Stopwatch.GetTimestamp();
                int seqLen = (int)input.Sizes[0];
                int outDim = (int)qw.Ne1;
                Tensor result = new Tensor(_allocator, DType.Float32, seqLen, outDim);
                if (MlxQuantizedOps.TryRmsNormAddmmQuantizedToFloat32(
                    result,
                    input,
                    normW,
                    Config.Eps,
                    qw.EnsureDeviceCacheKey(),
                    qw.Data,
                    qw.GgmlType,
                    qw.Ne0,
                    qw.Ne1,
                    qw.RawBytes))
                {
                    _linearTicks += Stopwatch.GetTimestamp() - t0;
                    return result;
                }

                result.Dispose();
            }

            // Fallback: explicit norm + linear.
            Tensor normed = RMSNormOpCached(input, normW);
            Tensor projected = LinearForwardCached(normed, qw, wF32);
            normed.Dispose();
            return projected;
        }

        /// <summary>
        /// Variant of <see cref="FusedNormLinear"/> that writes into a caller-supplied
        /// pre-allocated output buffer instead of allocating a new tensor each call.
        /// Returns the result buffer if the fused fast path executed, or null when the
        /// fallback path (explicit norm + linear) had to be used; in that case the
        /// caller will see the standard <see cref="FusedNormLinear"/> result.
        /// </summary>
        // 中文：FusedNormLinear 的写入预分配缓冲版：融合快路径成功则写入并返回 output，否则返回 null 让调用方走普通分配路径。
        private Tensor TryFusedNormLinearInto(Tensor output, Tensor input, Tensor normW, QuantizedWeight qw)
        {
            if (qw == null || normW == null
                || input.DimensionCount != 2 || output == null
                || output.DimensionCount != 2 || output.Sizes[1] != qw.Ne1
                || output.Sizes[0] != input.Sizes[0])
                return null;

            if (_backend == BackendType.Mlx)
            {
                long tMlx = Stopwatch.GetTimestamp();
                if (MlxQuantizedOps.TryRmsNormAddmmQuantizedToFloat32(
                    output,
                    input,
                    normW,
                    Config.Eps,
                    qw.EnsureDeviceCacheKey(),
                    qw.Data,
                    qw.GgmlType,
                    qw.Ne0,
                    qw.Ne1,
                    qw.RawBytes))
                {
                    _linearTicks += Stopwatch.GetTimestamp() - tMlx;
                    return output;
                }

                return null;
            }

            if (!IsGgmlBackend)
                return null;

            long t0 = Stopwatch.GetTimestamp();
            GgmlBasicOps.FusedRmsNormMatMulQuant(output, input, normW, Config.Eps,
                qw.CacheKey, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes);
            _linearTicks += Stopwatch.GetTimestamp() - t0;
            return output;
        }

        /// <summary>
        /// Fused output projection + residual add: residual += matmul(input, qW) in one dispatch.
        /// Equivalent to LinearForwardCached + Ops.Add but avoids the intermediate output tensor
        /// and one GPU sync. Returns true if the fused path executed; the caller must do its own
        /// add otherwise.
        /// </summary>
        // 中文：把输出投影矩阵乘与残差加融合为单次 dispatch：residual += matmul(input, qW)；融合执行返回 true，否则返回 false 让调用方自行相加。
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool TryLinearAddInto(Tensor residual, Tensor input, QuantizedWeight qw)
        {
            if (qw == null || input.DimensionCount != 2 || residual.DimensionCount != 2)
                return false;

            if (_backend == BackendType.Mlx)
            {
                long tMlx = Stopwatch.GetTimestamp();
                bool ok = MlxQuantizedOps.TryAddmmQuantizedAddToFloat32(
                    residual,
                    input,
                    qw.EnsureDeviceCacheKey(),
                    qw.Data,
                    qw.GgmlType,
                    qw.Ne0,
                    qw.Ne1,
                    qw.RawBytes);
                if (ok)
                    _linearTicks += Stopwatch.GetTimestamp() - tMlx;
                return ok;
            }

            if (!IsGgmlBackend)
                return false;

            long t0 = Stopwatch.GetTimestamp();
            GgmlBasicOps.FusedMatMulQuantAdd(residual, input,
                qw.CacheKey, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes);
            _linearTicks += Stopwatch.GetTimestamp() - t0;
            return true;
        }

        /// <summary>
        /// Single-token flash attention decode. Combines the KV cache append and the
        /// scaled-dot-product attention into a single GGML graph that runs on the device,
        /// replacing the host-side <see cref="ModelBase.AttentionDecodePureCS"/> path. The
        /// kernel writes the new K/V vectors directly into the persistent KV cache (zero-copy
        /// host-pointer binding) and produces the attention output for the new query.
        ///
        /// Returns true if the device kernel was used; false to indicate the caller should
        /// fall back to the CPU SIMD attention path. Only safe for the GGML backend.
        /// </summary>
        // 中文：单 token flash-attention 解码（仅 GGML）：把 KV 缓存追加与缩放点积注意力融合到一次设备图，K/V 直接零拷贝写入持久 KV 缓存；成功返回 true，否则回退 CPU SIMD 注意力。
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool TryFlashAttnDecode(Tensor q, Tensor k, Tensor v,
            Tensor kCache, Tensor vCache, Tensor output,
            int numHeads, int numKVHeads, int headDim,
            int maxSeqLen, int position, float scale)
        {
            if (!IsGgmlBackend || q == null || k == null || v == null || kCache == null || vCache == null || output == null)
                return false;
            if (q.ElementType != DType.Float32 || k.ElementType != DType.Float32 || v.ElementType != DType.Float32 ||
                output.ElementType != DType.Float32)
                return false;
            if (kCache.ElementType != DType.Float32 && kCache.ElementType != DType.Float16)
                return false;
            if (vCache.ElementType != kCache.ElementType)
                return false;

            try
            {
                GgmlBasicOps.FlashAttnDecode(q, k, v, kCache, vCache, output,
                    numHeads, numKVHeads, headDim, maxSeqLen, position, scale);

                // The kernel writes the output through the host pointer (unified memory on
                // Apple Silicon), but downstream GGML ops still need to know the buffer is
                // host-fresh so any cached device mirror is reloaded.
                InvalidateTensorDeviceCache(output);
                return true;
            }
            catch (Exception)
            {
                return false;
            }
        }

        /// <summary>
        /// Single-token Qwen3.5 attention layer decode in one fused GGML graph. Performs the
        /// entire FullAttention block (input RMSNorm, fused QKV, deinterleave Q/gate, per-head
        /// QK norm, RoPE, KV cache append, flash attention, sigmoid-gated mix, output projection,
        /// residual add) in a single dispatch. Eliminates ~2 standalone GGML calls plus the
        /// CPU/GPU sync overhead between the QKV and output kernels.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        /// <summary>
        /// Fused per-layer Qwen3.5 attention prefill. Folds the entire attention
        /// block (input RMSNorm + fused QKV + Q/K norm + RoPE + KV-cache append
        /// + causal-masked softmax + attention + sigmoid-gated mix + output
        /// projection + residual add) into ONE ggml_cgraph dispatch per layer,
        /// writing the residual back into the caller's `hidden` buffer in place.
        ///
        /// Returns false (and does NOT touch `hidden`) when:
        ///  - any required quantized weight isn't loaded for this layer
        ///    (the kernel currently requires the QKV / O quantized CacheKey),
        ///  - the layer is a gated-delta-net layer (not handled here),
        ///  - the KV cache dtype isn't F32 / F16, or
        ///  - the layer is shared (KV donor mapping isn't handled here yet).
        /// Caller falls back to the legacy multi-dispatch FullAttention path.
        /// </summary>
        // 中文：把整个全注意力层的 prefill（归一化+融合QKV+QK norm+RoPE+KV缓存追加+因果softmax注意力+sigmoid门控+输出投影+残差加）折叠进一次 GGML 图就地更新 hidden；前置条件不满足时返回 false 回退到多次 dispatch 路径。
        private unsafe bool TryFusedAttnLayerPrefill(Tensor hidden, int layer, int seqLen, int startPos)
        {
            if (!IsGgmlBackend) return false;
            if (hidden == null || hidden.DimensionCount != 2 || hidden.ElementType != DType.Float32)
                return false;

            // Skip gated-delta-net (recurrent) layers — they go through their
            // own kernel path entirely.
            if (_isRecurrent != null && _isRecurrent[layer]) return false;

            // Fused kernels bake scalar RoPE into the graph. Per-axis MRoPE
            // angles can't be expressed without a kernel update, so when
            // multimodal positions are pending fall back to the legacy
            // multi-dispatch path which routes through ApplyMRoPEPrefill.
            if (_pendingMRoPEPositions != null) return false;

            QuantizedWeight qkv = _attnQkvQW[layer];
            QuantizedWeight oOut = _attnOutputQW[layer];
            Tensor attnNorm = _attnNormW[layer];
            Tensor qNorm = _attnQNormW[layer];
            Tensor kNorm = _attnKNormW[layer];
            Tensor kCache = _kvCacheK[layer];
            Tensor vCache = _kvCacheV[layer];

            if (qkv == null || oOut == null || attnNorm == null || qNorm == null || kNorm == null
                || kCache == null || vCache == null)
                return false;
            if (kCache.ElementType != DType.Float32 && kCache.ElementType != DType.Float16)
                return false;
            if (vCache.ElementType != kCache.ElementType)
                return false;

            int headDim = Config.HeadDim;
            int numHeads = Config.NumHeads;
            int numKVHeads = Config.NumKVHeads;
            int maxSeqLen = (int)kCache.Sizes[1];
            int ropeDims = _ropeDimCount > 0 ? _ropeDimCount : headDim;
            const int ropeMode = 2; // NeoX, matches the legacy ApplyRoPEPrefill
            float ropeFreqScale = 1.0f / Config.RopeScale;

            int kvCacheTypeId = (kCache.ElementType == DType.Float16) ? 1 : 0;

            try
            {
                long t0 = Stopwatch.GetTimestamp();
                GgmlBasicOps.Qwen35AttentionLayerPrefill(
                    (IntPtr)GetFloatPtr(hidden), Config.HiddenSize, seqLen,
                    (IntPtr)GetFloatPtr(attnNorm),
                    qkv.CacheKey, qkv.GgmlType, qkv.Ne0, qkv.Ne1, qkv.RawBytes,
                    (IntPtr)GetFloatPtr(qNorm), (IntPtr)GetFloatPtr(kNorm),
                    oOut.CacheKey, oOut.GgmlType, oOut.Ne0, oOut.Ne1, oOut.RawBytes,
                    TensorComputePrimitives.GetStoragePointer(kCache),
                    TensorComputePrimitives.GetStoragePointer(vCache),
                    numHeads, numKVHeads, headDim,
                    maxSeqLen, startPos,
                    Config.RopeBase, ropeFreqScale, ropeDims, ropeMode,
                    kvCacheTypeId, Config.Eps);
                _attnTicks += Stopwatch.GetTimestamp() - t0;

                InvalidateTensorDeviceCache(hidden);
                InvalidateTensorDeviceCache(kCache);
                InvalidateTensorDeviceCache(vCache);
                return true;
            }
            catch
            {
                return false;
            }
        }

        // 中文：单 token 的全注意力层解码融合 kernel（仅 GGML）：把整层注意力块（归一化+融合QKV+拆分+QK norm+RoPE+KV缓存追加+flash注意力+sigmoid门控+输出投影+残差加）折叠进一次 GGML 图就地更新 residual；前置条件不满足返回 false。
        private bool TryFusedAttnLayerDecode(Tensor residual, int layer, int position)
        {
            if (!IsGgmlBackend)
                return false;
            if (residual == null || residual.DimensionCount != 2 || residual.ElementType != DType.Float32)
                return false;
            if (residual.Sizes[0] != 1)
                return false;

            QuantizedWeight qkv = _attnQkvQW[layer];
            QuantizedWeight oOut = _attnOutputQW[layer];
            Tensor attnNorm = _attnNormW[layer];
            Tensor qNorm = _attnQNormW[layer];
            Tensor kNorm = _attnKNormW[layer];
            Tensor kCache = _kvCacheK[layer];
            Tensor vCache = _kvCacheV[layer];

            if (qkv == null || oOut == null || attnNorm == null || qNorm == null || kNorm == null
                || kCache == null || vCache == null)
                return false;
            if (kCache.ElementType != DType.Float32 && kCache.ElementType != DType.Float16)
                return false;
            if (vCache.ElementType != kCache.ElementType)
                return false;

            int headDim = Config.HeadDim;
            int numHeads = Config.NumHeads;
            int numKVHeads = Config.NumKVHeads;
            int maxSeqLen = (int)kCache.Sizes[1];

            // The fused kernel uses NeoX RoPE (rope_mode = 2), matching the standalone path.
            const int ropeMode = 2;
            float ropeFreqScale = 1.0f / Config.RopeScale;

            try
            {
                long t0 = Stopwatch.GetTimestamp();
                GgmlBasicOps.Qwen35AttentionLayerDecode(
                    residual,
                    attnNorm,
                    qkv.CacheKey, qkv.GgmlType, qkv.Ne0, qkv.Ne1, qkv.RawBytes,
                    qNorm, kNorm, headDim,
                    oOut.CacheKey, oOut.GgmlType, oOut.Ne0, oOut.Ne1, oOut.RawBytes,
                    kCache, vCache,
                    numHeads, numKVHeads,
                    maxSeqLen, position,
                    Config.Eps, Config.RopeBase, ropeFreqScale, ropeMode);
                _attnTicks += Stopwatch.GetTimestamp() - t0;

                // Output is written through the host pointer (unified memory). Downstream
                // GGML ops need a fresh device mirror so invalidate the cache for the residual
                // and the KV cache slabs that we just appended into.
                InvalidateTensorDeviceCache(residual);
                InvalidateTensorDeviceCache(kCache);
                InvalidateTensorDeviceCache(vCache);
                return true;
            }
            catch (Exception)
            {
                return false;
            }
        }

        // 中文：对每个注意力头的 Q 或 K 向量按 headDim 做 RMSNorm（QK-norm），解码用 CPU 就地路径、MLX/prefill 走设备就地路径，返回原张量。
        private Tensor ApplyQKNormCached(Tensor data, Tensor alpha, int numHeads, int seqLen)
        {
            int headDim = Config.HeadDim;

            // Decode (seqLen==1): the CPU path is faster per call for tiny
            // tensors, but on MLX every GetFloatPtr forces an mlx_eval +
            // device→host copy. With 4 syncs/attn-layer × 60 layers that's a
            // lot of round trips, so stay on device for MLX. Other backends
            // keep the CPU SIMD fast path which saves 2 GPU dispatches.
            if (seqLen == 1 && _backend != BackendType.Mlx)
            {
                RMSNormInPlaceCpu(data, alpha, numHeads, headDim, Config.Eps);
                return data;
            }

            // In-place RMSNorm: row-independent normalization works regardless
            // of row order, avoiding one tensor allocation+copy per Q/K per layer.
            using var reshaped = data.View(seqLen * numHeads, headDim);
            Ops.RMSNorm(reshaped, reshaped, alpha, null, Config.Eps);
            return data;
        }

        /// <summary>
        /// Fused decode-path RoPE for Q and K at the same position. Computes the
        /// cos/sin table once (since both share <paramref name="position"/>) and applies
        /// the rotation in place to both tensors. The inner loop is SIMD-vectorized
        /// when <c>halfDim</c> is divisible by the hardware vector width, with a scalar
        /// tail for the remainder.
        /// </summary>
        // 中文：解码路径下对同一 position 的 Q 与 K 一次性应用 RoPE：只算一次 cos/sin 表并对两者就地旋转（内层 SIMD 向量化）。
        private unsafe void ApplyRoPEDecodeQKInPlace(Tensor qData, Tensor kData,
            int numQHeads, int numKHeads, int position)
        {
            int headDim = Config.HeadDim;
            int ropeDim = _ropeDimCount > 0 ? _ropeDimCount : headDim;
            int halfDim = ropeDim / 2;

            float* cosTable = stackalloc float[halfDim];
            float* sinTable = stackalloc float[halfDim];
            for (int i = 0; i < halfDim; i++)
            {
                float theta = position * _ropeFreqs[i];
                cosTable[i] = MathF.Cos(theta);
                sinTable[i] = MathF.Sin(theta);
            }

            float* qPtr = GetFloatPtr(qData);
            float* kPtr = GetFloatPtr(kData);
            ApplyRoPERotationInPlace(qPtr, numQHeads, headDim, halfDim, cosTable, sinTable);
            ApplyRoPERotationInPlace(kPtr, numKHeads, headDim, halfDim, cosTable, sinTable);

            InvalidateTensorDeviceCache(qData);
            InvalidateTensorDeviceCache(kData);
        }

        // 中文：用给定 cos/sin 表对每个头的向量做 NeoX 式成对就地旋转（[head[i], head[i+halfDim]] 旋转），主循环 SIMD 向量化、尾部标量处理。
        private static unsafe void ApplyRoPERotationInPlace(float* ptr, int numHeads,
            int headDim, int halfDim, float* cosTable, float* sinTable)
        {
            int vecSz = Vector<float>.Count;
            int vecEnd = halfDim - (halfDim % vecSz);

            for (int h = 0; h < numHeads; h++)
            {
                float* head = ptr + h * headDim;
                float* hi = head + halfDim;

                int i = 0;
                for (; i < vecEnd; i += vecSz)
                {
                    var x0 = LdVecLocal(head + i);
                    var x1 = LdVecLocal(hi + i);
                    var c = LdVecLocal(cosTable + i);
                    var s = LdVecLocal(sinTable + i);
                    StVecLocal(head + i, x0 * c - x1 * s);
                    StVecLocal(hi + i, x0 * s + x1 * c);
                }
                for (; i < halfDim; i++)
                {
                    float x0 = head[i];
                    float x1 = hi[i];
                    head[i] = x0 * cosTable[i] - x1 * sinTable[i];
                    hi[i] = x0 * sinTable[i] + x1 * cosTable[i];
                }
            }
        }

        // 中文：prefill 路径下对 Q 或 K 应用标量 RoPE：按 (seqLen,startPos) 构建并跨层缓存位置张量，再调用 Ops.RoPEEx 就地旋转，返回原张量。
        private Tensor ApplyRoPEPrefill(Tensor data, int numHeads, int seqLen, int startPos)
        {
            int headDim = Config.HeadDim;
            int ropeDim = _ropeDimCount > 0 ? _ropeDimCount : headDim;

            // Cache position tensors across attention layers in the same forward pass.
            // All attention layers share (seqLen, startPos); only numHeads differs (Q vs K).
            Tensor posTensor;
            bool isQ = (numHeads == Config.NumHeads);

            if (_cachedRoPEPosSeqLen == seqLen && _cachedRoPEPosStartPos == startPos)
            {
                posTensor = isQ ? _cachedRoPEPosQ : _cachedRoPEPosK;
                if (posTensor == null || (int)posTensor.Sizes[0] != seqLen * numHeads)
                    posTensor = null;
            }
            else
            {
                _cachedRoPEPosQ?.Dispose();
                _cachedRoPEPosK?.Dispose();
                _cachedRoPEPosQ = null;
                _cachedRoPEPosK = null;
                _cachedRoPEPosSeqLen = seqLen;
                _cachedRoPEPosStartPos = startPos;
                posTensor = null;
            }

            if (posTensor == null)
            {
                int totalRows = seqLen * numHeads;
                int[] positions = new int[totalRows];
                for (int s = 0; s < seqLen; s++)
                    for (int h = 0; h < numHeads; h++)
                        positions[s * numHeads + h] = startPos + s;
                posTensor = CreateIntTensor(positions, totalRows);
                if (isQ) _cachedRoPEPosQ = posTensor;
                else _cachedRoPEPosK = posTensor;
            }

            // In-place RoPE: avoids allocating a new tensor per call
            using var reshaped = data.View(1, seqLen, numHeads, headDim);
            Ops.RoPEEx(reshaped, reshaped, posTensor, ropeDim, 2, 0,
                Config.RopeBase, 1.0f / Config.RopeScale,
                0.0f, 1.0f, 0.0f, 0.0f);
            return data;
        }

        /// <summary>Set the per-axis (T,H,W) positions for the next prefill
        /// call. Length must equal 3 * seqLen of the upcoming Forward(). Called
        /// by ModelMultimodalInjector.QueuePromptEmbeddingsForSlice when an
        /// image is in the prompt slice.</summary>
        // 中文：为下一次 prefill 设置按 (T,H,W) 轴展平的 MRoPE 位置数组（多模态视觉提示用），由多模态注入器在 Forward 前调用。
        public void SetMRoPEPositions(int[] flatThw)
        {
            _pendingMRoPEPositions = flatThw;
        }

        /// <summary>Build the per-pair modality assignment from `_mropeSections`
        /// using vLLM's get_mrope_interleaved_id_list algorithm (see
        /// mrope_interleaved.py:138-185). Result: int[rotary_dim/2] where
        /// each entry ∈ {0=T, 1=H, 2=W}. Called lazily on first MRoPE-prefill.</summary>
        // 中文：依据 _mropeSections 用 vLLM 的交错算法预计算每个旋转对的模态归属（int[rotary_dim/2]，值 ∈ {0=T,1=H,2=W}，并强制末位为 T），首次 MRoPE prefill 时惰性调用。
        private void PrecomputeMRoPEInterleavedIds()
        {
            int ropeDim = _ropeDimCount > 0 ? _ropeDimCount : Config.HeadDim;
            int pairs = ropeDim / 2;
            int[] sec = _mropeSections;
            // Qwen3.5 GGUFs ship [11,11,10,0]; the trailing 0 is video-time
            // padding. Collapse to length-3 (T,H,W) and apply force_last like
            // vLLM does for len-3 sections.
            int a = sec != null && sec.Length > 0 ? sec[0] : pairs / 3;
            int b = sec != null && sec.Length > 1 ? sec[1] : pairs / 3;
            int c = sec != null && sec.Length > 2 ? sec[2] : pairs - 2 * (pairs / 3);
            if (a + b + c != pairs)
            {
                Console.WriteLine($"[qwen35-mrope] section sum {a + b + c} != rotary_dim/2 {pairs}; " +
                                  $"falling back to T-only (axis=0) - mrope_section may be wrong");
                _mropeInterleavedIds = new int[pairs]; // all zeros = all-T = standard RoPE
                return;
            }
            // force_last=true: last pair must be T (axis 0). Python's
            // mrope_interleaved.py decrements `a` BEFORE building the counts
            // dict, so the placement-fraction score uses `a-1` as the T
            // denominator (not `a`). We track that as `aTotal` below.
            int aTotal = a - 1, bTotal = b, cTotal = c;
            int aRem = aTotal, bRem = bTotal, cRem = cTotal;
            int[] ids = new int[pairs];
            int last = -1;
            for (int i = 0; i < pairs - 1; i++)
            {
                // Candidates: remaining > 0 and != last; if all candidates == last, relax.
                int bestK = -1;
                double bestScore = double.MaxValue;
                for (int k = 0; k < 3; k++)
                {
                    int rem = k == 0 ? aRem : k == 1 ? bRem : cRem;
                    if (rem <= 0) continue;
                    if (k == last) continue;
                    int total = k == 0 ? aTotal : k == 1 ? bTotal : cTotal;
                    int placed = total - rem;
                    double score = (double)placed / total;
                    if (score < bestScore || (score == bestScore && k < bestK))
                    {
                        bestScore = score;
                        bestK = k;
                    }
                }
                if (bestK < 0)
                {
                    // All candidates blocked by "!= last" - relax.
                    for (int k = 0; k < 3; k++)
                    {
                        int rem = k == 0 ? aRem : k == 1 ? bRem : cRem;
                        if (rem <= 0) continue;
                        int total = k == 0 ? aTotal : k == 1 ? bTotal : cTotal;
                        int placed = total - rem;
                        double score = (double)placed / total;
                        if (score < bestScore || (score == bestScore && k < bestK))
                        {
                            bestScore = score;
                            bestK = k;
                        }
                    }
                }
                ids[i] = bestK;
                if (bestK == 0) aRem--;
                else if (bestK == 1) bRem--;
                else cRem--;
                last = bestK;
            }
            ids[pairs - 1] = 0; // force_last
            _mropeInterleavedIds = ids;
        }

        /// <summary>Per-axis (interleaved MRoPE) variant of ApplyRoPEPrefill.
        /// Q/K go in, get rotated in place by NeoX-style pair rotation where
        /// pair i uses _pendingMRoPEPositions[3*token + _mropeInterleavedIds[i]]
        /// as the position instead of a single token position. Returns the
        /// original tensor (modified). Falls back silently if positions or ids
        /// aren't set; the caller should have checked _pendingMRoPEPositions.
        ///
        /// Costs one host download + one host upload per call (Q and K each).
        /// Acceptable for prefill since it's already the slow path; not used in
        /// decode.</summary>
        // 中文：ApplyRoPEPrefill 的多模态交错变体：每个旋转对按其模态归属取 (T,H,W) 中对应位置算角度做 NeoX 旋转，GGML 后端走 ggml_rope_multi 在设备完成、否则走主机下载/上传路径，返回原张量。
        private Tensor ApplyMRoPEPrefill(Tensor data, int numHeads, int seqLen, int[] mropePositions)
        {
            int headDim = Config.HeadDim;
            int ropeDim = _ropeDimCount > 0 ? _ropeDimCount : headDim;
            int pairs = ropeDim / 2;
            if (_mropeInterleavedIds == null || _mropeInterleavedIds.Length != pairs)
                PrecomputeMRoPEInterleavedIds();
            int[] modality = _mropeInterleavedIds;

            // On the GGML backend, route through the native ggml_rope_multi
            // kernel which runs entirely on-device (no per-layer host download
            // / upload that the managed path below incurs). Requires shape
            // [headDim, numHeads, seqLen, 1] in GGML order (= C# View(1, seqLen,
            // numHeads, headDim)) and positions laid out per-axis-concatenated
            // as [T0..T_n, H0..H_n, W0..W_n, 0..0] of length 4*seqLen.
            // Mode 40 = GGML_ROPE_TYPE_IMROPE; sections come straight from the
            // GGUF (Qwen3.5 ships [11,11,10,0]).
            if (data.Storage is TensorSharp.GGML.GgmlStorage && _mropeSections != null && _mropeSections.Length >= 4)
            {
                int[] flatThw = new int[4 * seqLen];
                for (int t = 0; t < seqLen; t++)
                {
                    flatThw[t]            = mropePositions[3 * t + 0]; // T axis
                    flatThw[seqLen + t]   = mropePositions[3 * t + 1]; // H axis
                    flatThw[2*seqLen + t] = mropePositions[3 * t + 2]; // W axis
                    // 4th axis: 0 (no video / fourth section is 0 for Qwen3.5).
                }
                using var positionsTensor = CreateIntTensor(flatThw, flatThw.Length);
                using var reshaped = data.View(1, seqLen, numHeads, headDim);
                const int GGML_ROPE_TYPE_IMROPE = 40;
                int[] sec = new int[4] {
                    _mropeSections[0], _mropeSections[1], _mropeSections[2], _mropeSections[3]
                };
                TensorSharp.GGML.GgmlBasicOps.RoPEMRoPE(
                    reshaped, reshaped, positionsTensor, sec,
                    ropeDim, GGML_ROPE_TYPE_IMROPE,
                    /*originalContextLength*/ 0,
                    Config.RopeBase, 1.0f / Config.RopeScale);
                return data;
            }

            int total = seqLen * numHeads * headDim;
            float[] buf = data.GetElementsAsFloat(total);

            // NeoX rotation: dim pair (i, i + pairs) rotates by theta.
            // Per-pair angle = positions[token, modality[i]] * freq[i].
            // freq[i] = (1/RopeScale) / RopeBase^(2i/ropeDim) (matches _ropeFreqs).
            float[] freqs = _ropeFreqs;
            int strideToken = numHeads * headDim;
            int strideHead = headDim;
            for (int t = 0; t < seqLen; t++)
            {
                int p0 = mropePositions[3 * t + 0];
                int p1 = mropePositions[3 * t + 1];
                int p2 = mropePositions[3 * t + 2];
                for (int h = 0; h < numHeads; h++)
                {
                    int baseOff = t * strideToken + h * strideHead;
                    for (int i = 0; i < pairs; i++)
                    {
                        int mod = modality[i];
                        int pos = mod == 0 ? p0 : mod == 1 ? p1 : p2;
                        float theta = pos * freqs[i];
                        float c = MathF.Cos(theta);
                        float s = MathF.Sin(theta);
                        float x0 = buf[baseOff + i];
                        float x1 = buf[baseOff + i + pairs];
                        buf[baseOff + i]         = x0 * c - x1 * s;
                        buf[baseOff + i + pairs] = x0 * s + x1 * c;
                    }
                }
            }

            // Upload back. Use the same trick as elsewhere - create a contiguous
            // fresh tensor and copy via the helper that backends understand.
            long[] shape = data.Sizes.ToArray();
            using var refreshed = CreateFloatTensor(buf, shape);
            Ops.Copy(data, refreshed);
            return data;
        }

        #endregion


        #region Mixture-of-Experts FFN

        /// <summary>
        /// Mixture-of-Experts SwiGLU FFN as used by qwen35moe / qwen3next.
        /// 1. router_logits = input @ ffn_gate_inp ; probs = softmax(router_logits)
        /// 2. select top-K experts per token, optionally renormalize selected probs
        /// 3. expert SwiGLU: SiLU(input @ gate_e) * (input @ up_e) @ down_e
        /// 4. weighted sum across selected experts
        /// 5. optional shared expert SwiGLU gated by sigmoid(input . ffn_gate_inp_shexp)
        ///
        /// Optimizations vs the original implementation:
        /// - Pre-cached `QuantizedWeight` / F32 expert tensors avoid per-token dictionary lookups.
        /// - Reuses pre-allocated decode-path tensors (`_moeTokenInput`, `_moeGateBuf`, `_moeUpBuf`,
        ///   `_moeDownBuf`) so the seqLen=1 hot path performs no per-expert tensor allocation.
        /// - Falls back to the previous per-token allocation strategy for prefill (seqLen > 1)
        ///   where reuse would constrain parallelism between token slices.
        /// </summary>
        /// <summary>
        /// Single-token MoE that fuses (a) routed expert SwiGLU, (b) optional shared expert
        /// SwiGLU, and (c) the residual add into one GGML graph dispatch via the
        /// MoEExpertsSwiGLUResidual kernel. Returns true if the fully fused path executed and
        /// `residual` already contains the updated value; false to fall back to MoEForward.
        /// </summary>
        // 中文：单 token MoE 解码融合路径（仅 GGML）：CPU 上算路由 logits 的 top-K 与（归一化）softmax 权重，再把路由专家 SwiGLU、可选共享专家及残差加全部融合进一次 MoEExpertsSwiGLUResidual 图；成功返回 true 并就地更新 residual。
        private unsafe bool TryMoEResidualDecode(Tensor residual, Tensor input, int layer)
        {
            if (!IsGgmlBackend
                || _moeBatchedResult == null
                || _moeGatePtrs == null
                || _expertGateQW == null || _expertUpQW == null || _expertDownQW == null
                || _expertGateQW[layer] == null || _expertUpQW[layer] == null || _expertDownQW[layer] == null)
                return false;

            int hiddenSize = Config.HiddenSize;

            Tensor routerLogits = LinearForwardCached(input, _ffnGateInpQW[layer], _ffnGateInpF32[layer]);
            if (routerLogits == null)
                return false;

            // Routing: skip the full-vocab softmax dispatch and operate on the raw logits.
            // Softmax is monotonic so top-K of softmax equals top-K of logits; for the
            // normalized-top-K case (the common path) the top-K weights are softmax(top-K
            // logits), which we compute on CPU. Eliminates one Metal dispatch per MoE layer.
            float* logits = GetFloatPtr(routerLogits);
            int[] topExperts = _moeTopExperts;
            float[] routeW = _moeRouteW;
            SelectTopKInPlace(logits, _numExperts, _numExpertsUsed, topExperts);

            if (_normTopKProb)
            {
                float maxLogit = float.NegativeInfinity;
                for (int k = 0; k < _numExpertsUsed; k++)
                {
                    float v = logits[topExperts[k]];
                    if (v > maxLogit) maxLogit = v;
                }
                float wSum = 0f;
                for (int k = 0; k < _numExpertsUsed; k++)
                {
                    float w = MathF.Exp(logits[topExperts[k]] - maxLogit);
                    routeW[k] = w;
                    wSum += w;
                }
                if (wSum > 0f)
                {
                    float inv = 1.0f / wSum;
                    for (int k = 0; k < _numExpertsUsed; k++)
                        routeW[k] *= inv;
                }
            }
            else
            {
                // Unnormalised case: weights must be the absolute softmax probabilities,
                // which need the full denominator (sum over all experts). Compute it on CPU.
                float maxLogit = float.NegativeInfinity;
                for (int i = 0; i < _numExperts; i++)
                    if (logits[i] > maxLogit) maxLogit = logits[i];
                float denom = 0f;
                for (int i = 0; i < _numExperts; i++)
                    denom += MathF.Exp(logits[i] - maxLogit);
                float invDenom = denom > 0f ? 1.0f / denom : 0f;
                for (int k = 0; k < _numExpertsUsed; k++)
                    routeW[k] = MathF.Exp(logits[topExperts[k]] - maxLogit) * invDenom;
            }
            routerLogits.Dispose();

            QuantizedWeight gQW0 = null, uQW0 = null, dQW0 = null;
            for (int k = 0; k < _numExpertsUsed; k++)
            {
                int e = topExperts[k];
                var g = _expertGateQW[layer][e];
                var u = _expertUpQW[layer][e];
                var d = _expertDownQW[layer][e];
                if (g == null || u == null || d == null)
                    return false;
                if (gQW0 == null) { gQW0 = g; uQW0 = u; dQW0 = d; }

                if (g.GgmlType != gQW0.GgmlType || g.Ne0 != gQW0.Ne0 || g.Ne1 != gQW0.Ne1 ||
                    u.GgmlType != uQW0.GgmlType || u.Ne0 != uQW0.Ne0 || u.Ne1 != uQW0.Ne1 ||
                    d.GgmlType != dQW0.GgmlType || d.Ne0 != dQW0.Ne0 || d.Ne1 != dQW0.Ne1)
                    return false;

                _moeGatePtrs[k] = g.CacheKey;
                _moeUpPtrs[k] = u.CacheKey;
                _moeDownPtrs[k] = d.CacheKey;
            }

            // Optional shared expert weights & gate scalar.
            bool useShared = false;
            IntPtr sgPtr = IntPtr.Zero, suPtr = IntPtr.Zero, sdPtr = IntPtr.Zero;
            int sgType = 0, suType = 0, sdType = 0;
            long sgNe0 = 0, sgNe1 = 0, sgBytes = 0;
            long suNe0 = 0, suNe1 = 0, suBytes = 0;
            long sdNe0 = 0, sdNe1 = 0, sdBytes = 0;
            float sharedScalar = 0f;

            if (_hasSharedExperts != null && _hasSharedExperts[layer]
                && _ffnGateShexpQW[layer] != null
                && _ffnUpShexpQW[layer] != null
                && _ffnDownShexpQW[layer] != null)
            {
                var sg = _ffnGateShexpQW[layer];
                var su = _ffnUpShexpQW[layer];
                var sd = _ffnDownShexpQW[layer];
                sgPtr = sg.CacheKey; suPtr = su.CacheKey; sdPtr = sd.CacheKey;
                sgType = sg.GgmlType; suType = su.GgmlType; sdType = sd.GgmlType;
                sgNe0 = sg.Ne0; sgNe1 = sg.Ne1; sgBytes = sg.RawBytes;
                suNe0 = su.Ne0; suNe1 = su.Ne1; suBytes = su.RawBytes;
                sdNe0 = sd.Ne0; sdNe1 = sd.Ne1; sdBytes = sd.RawBytes;

                sharedScalar = 1.0f;
                if (_hasSharedExpertGate != null && _hasSharedExpertGate[layer])
                {
                    var gateInpVec = _ffnGateInpShexpVec[layer];
                    if (gateInpVec != null)
                    {
                        float* tokenRow = GetFloatPtr(input);
                        float* gateInpPtr = GetFloatPtr(gateInpVec);
                        int n = Math.Min((int)gateInpVec.ElementCount(), hiddenSize);
                        sharedScalar = SigmoidScalar(VecDot(tokenRow, gateInpPtr, n));
                    }
                }
                useShared = true;
            }

            long t0exp = Stopwatch.GetTimestamp();
            GgmlBasicOps.MoEExpertsSwiGLUResidual(
                residual, input,
                _numExpertsUsed,
                _moeGatePtrs, _moeUpPtrs, _moeDownPtrs,
                gQW0.GgmlType, gQW0.Ne0, gQW0.Ne1, gQW0.RawBytes,
                uQW0.GgmlType, uQW0.Ne0, uQW0.Ne1, uQW0.RawBytes,
                dQW0.GgmlType, dQW0.Ne0, dQW0.Ne1, dQW0.RawBytes,
                routeW,
                useShared,
                sgPtr, suPtr, sdPtr,
                sgType, sgNe0, sgNe1, sgBytes,
                suType, suNe0, suNe1, suBytes,
                sdType, sdNe0, sdNe1, sdBytes,
                sharedScalar);
            _linearTicks += Stopwatch.GetTimestamp() - t0exp;

            InvalidateTensorDeviceCache(residual);
            return true;
        }

        /// <summary>
        /// MoE decode with pre-computed router logits (from the fused outproj+norm+router kernel).
        /// Skips the router projection dispatch since logits are already available.
        /// </summary>
        // 中文：与 TryMoEResidualDecode 相同的 MoE 解码融合路径，但路由 logits 已由上游融合 kernel 算好直接传入，省去本层的路由投影 dispatch。
        private unsafe bool TryMoEResidualDecodeWithRouter(Tensor residual, Tensor input, Tensor routerLogits, int layer)
        {
            if (!IsGgmlBackend || _moeBatchedResult == null || _moeGatePtrs == null
                || _expertGateQW == null || _expertUpQW == null || _expertDownQW == null
                || _expertGateQW[layer] == null || _expertUpQW[layer] == null || _expertDownQW[layer] == null)
                return false;

            int hiddenSize = Config.HiddenSize;
            float* logits = GetFloatPtr(routerLogits);
            int[] topExperts = _moeTopExperts;
            float[] routeW = _moeRouteW;
            SelectTopKInPlace(logits, _numExperts, _numExpertsUsed, topExperts);

            if (_normTopKProb)
            {
                float maxLogit = float.NegativeInfinity;
                for (int k = 0; k < _numExpertsUsed; k++)
                    if (logits[topExperts[k]] > maxLogit) maxLogit = logits[topExperts[k]];
                float wSum = 0f;
                for (int k = 0; k < _numExpertsUsed; k++)
                {
                    float w = MathF.Exp(logits[topExperts[k]] - maxLogit);
                    routeW[k] = w; wSum += w;
                }
                if (wSum > 0f) { float inv = 1.0f / wSum; for (int k = 0; k < _numExpertsUsed; k++) routeW[k] *= inv; }
            }
            else
            {
                float maxLogit = float.NegativeInfinity;
                for (int i = 0; i < _numExperts; i++) if (logits[i] > maxLogit) maxLogit = logits[i];
                float denom = 0f;
                for (int i = 0; i < _numExperts; i++) denom += MathF.Exp(logits[i] - maxLogit);
                float invDenom = denom > 0f ? 1.0f / denom : 0f;
                for (int k = 0; k < _numExpertsUsed; k++) routeW[k] = MathF.Exp(logits[topExperts[k]] - maxLogit) * invDenom;
            }

            QuantizedWeight gQW0 = null, uQW0 = null, dQW0 = null;
            for (int k = 0; k < _numExpertsUsed; k++)
            {
                int e = topExperts[k];
                var g = _expertGateQW[layer][e]; var u = _expertUpQW[layer][e]; var d = _expertDownQW[layer][e];
                if (g == null || u == null || d == null) return false;
                if (gQW0 == null) { gQW0 = g; uQW0 = u; dQW0 = d; }
                if (g.GgmlType != gQW0.GgmlType || g.Ne0 != gQW0.Ne0 || g.Ne1 != gQW0.Ne1 ||
                    u.GgmlType != uQW0.GgmlType || u.Ne0 != uQW0.Ne0 || u.Ne1 != uQW0.Ne1 ||
                    d.GgmlType != dQW0.GgmlType || d.Ne0 != dQW0.Ne0 || d.Ne1 != dQW0.Ne1) return false;
                _moeGatePtrs[k] = g.CacheKey; _moeUpPtrs[k] = u.CacheKey; _moeDownPtrs[k] = d.CacheKey;
            }

            bool useShared = false;
            IntPtr sgPtr = IntPtr.Zero, suPtr = IntPtr.Zero, sdPtr = IntPtr.Zero;
            int sgType = 0, suType = 0, sdType = 0;
            long sgNe0 = 0, sgNe1 = 0, sgBytes = 0, suNe0 = 0, suNe1 = 0, suBytes = 0, sdNe0 = 0, sdNe1 = 0, sdBytes = 0;
            float sharedScalar = 0f;

            if (_hasSharedExperts != null && _hasSharedExperts[layer]
                && _ffnGateShexpQW[layer] != null && _ffnUpShexpQW[layer] != null && _ffnDownShexpQW[layer] != null)
            {
                var sg = _ffnGateShexpQW[layer]; var su = _ffnUpShexpQW[layer]; var sd = _ffnDownShexpQW[layer];
                sgPtr = sg.CacheKey; suPtr = su.CacheKey; sdPtr = sd.CacheKey;
                sgType = sg.GgmlType; suType = su.GgmlType; sdType = sd.GgmlType;
                sgNe0 = sg.Ne0; sgNe1 = sg.Ne1; sgBytes = sg.RawBytes;
                suNe0 = su.Ne0; suNe1 = su.Ne1; suBytes = su.RawBytes;
                sdNe0 = sd.Ne0; sdNe1 = sd.Ne1; sdBytes = sd.RawBytes;
                sharedScalar = 1.0f;
                if (_hasSharedExpertGate != null && _hasSharedExpertGate[layer])
                {
                    var gateInpVec = _ffnGateInpShexpVec[layer];
                    if (gateInpVec != null)
                    {
                        float* tokenRow = GetFloatPtr(input);
                        float* gateInpPtr = GetFloatPtr(gateInpVec);
                        int n = Math.Min((int)gateInpVec.ElementCount(), hiddenSize);
                        sharedScalar = SigmoidScalar(VecDot(tokenRow, gateInpPtr, n));
                    }
                }
                useShared = true;
            }

            long t0exp = Stopwatch.GetTimestamp();
            GgmlBasicOps.MoEExpertsSwiGLUResidual(residual, input,
                _numExpertsUsed, _moeGatePtrs, _moeUpPtrs, _moeDownPtrs,
                gQW0.GgmlType, gQW0.Ne0, gQW0.Ne1, gQW0.RawBytes,
                uQW0.GgmlType, uQW0.Ne0, uQW0.Ne1, uQW0.RawBytes,
                dQW0.GgmlType, dQW0.Ne0, dQW0.Ne1, dQW0.RawBytes,
                routeW, useShared, sgPtr, suPtr, sdPtr,
                sgType, sgNe0, sgNe1, sgBytes, suType, suNe0, suNe1, suBytes,
                sdType, sdNe0, sdNe1, sdBytes, sharedScalar);
            _linearTicks += Stopwatch.GetTimestamp() - t0exp;
            InvalidateTensorDeviceCache(residual);
            return true;
        }

        // 中文：MLX 的 MoE prefill 批处理路径：对所有 token 路由并按专家分组，把同一专家的 token 行聚集成批做 gate/up/SiLUMul/down 矩阵乘，再按路由权重 scatter-add 回输出，最后叠加共享专家；不满足条件返回 false。
        private unsafe bool TryMoEPrefillBatchedMlx(
            Tensor input,
            Tensor output,
            float* routePtr,
            bool routeRowsAreLogits,
            Tensor sharedDownAll,
            Tensor sharedGateInpVec,
            int layer,
            int seqLen,
            int hiddenSize)
        {
            if (_backend != BackendType.Mlx
                || seqLen <= 1
                || _numExperts <= 0
                || _numExpertsUsed <= 0
                || _expertGateQW == null
                || _expertUpQW == null
                || _expertDownQW == null
                || _expertGateQW[layer] == null
                || _expertUpQW[layer] == null
                || _expertDownQW[layer] == null)
            {
                return false;
            }

            int nUsed = _numExpertsUsed;
            int totalRoutes = checked(seqLen * nUsed);
            int[] selectedExperts = new int[totalRoutes];
            float[] selectedWeights = new float[totalRoutes];
            int[] expertCounts = new int[_numExperts];
            int[] tokTop = new int[nUsed];
            float[] tokW = new float[nUsed];

            for (int s = 0; s < seqLen; s++)
            {
                float* routeRow = routePtr + (long)s * _numExperts;
                SelectTopKRouteWeights(routeRow, routeRowsAreLogits, tokTop, tokW);

                int routeOffset = s * nUsed;
                for (int k = 0; k < nUsed; k++)
                {
                    int expert = tokTop[k];
                    selectedExperts[routeOffset + k] = expert;
                    selectedWeights[routeOffset + k] = tokW[k];
                    expertCounts[expert]++;
                }
            }

            int[] expertOffsets = new int[_numExperts + 1];
            for (int e = 0; e < _numExperts; e++)
                expertOffsets[e + 1] = expertOffsets[e] + expertCounts[e];

            int[] cursors = new int[_numExperts];
            Array.Copy(expertOffsets, cursors, _numExperts);
            int[] routedTokenRows = new int[totalRoutes];
            float[] routedWeights = new float[totalRoutes];
            for (int s = 0; s < seqLen; s++)
            {
                int routeOffset = s * nUsed;
                for (int k = 0; k < nUsed; k++)
                {
                    int expert = selectedExperts[routeOffset + k];
                    int dst = cursors[expert]++;
                    routedTokenRows[dst] = s;
                    routedWeights[dst] = selectedWeights[routeOffset + k];
                }
            }

            // Batched per-layer expert dispatch. Uploads routedTokenRows /
            // routedWeights once, gathers all routes into a single
            // [totalRoutes, hidden] tensor, then narrow-views that buffer
            // per expert. Per-expert work reduces to: narrow → matmul
            // (gate/up) → SiLUMul → matmul (down) → scatter-add with
            // narrow-views of the layer-wide indices/weights tensors.
            // Each narrow is a free view (shares storage, just adjusts
            // offset/sizes).
            Tensor rowIndicesAll = null;
            Tensor routeWeightsAll = null;
            Tensor batchInputAll = null;
            try
            {
                rowIndicesAll = CreateIntTensor(routedTokenRows, totalRoutes);
                routeWeightsAll = CreateFloatTensor(routedWeights, totalRoutes);
                batchInputAll = new Tensor(_allocator, DType.Float32, totalRoutes, hiddenSize);
                if (!MlxFusedOps.TryGatherRows(batchInputAll, input, rowIndicesAll))
                    return false;

                for (int e = 0; e < _numExperts; e++)
                {
                    int start = expertOffsets[e];
                    int batchSize = expertOffsets[e + 1] - start;
                    if (batchSize == 0)
                        continue;

                    Tensor batchInput = null;
                    Tensor rowIndices = null;
                    Tensor routeWeights = null;
                    Tensor gate = null;
                    Tensor up = null;
                    Tensor down = null;
                    try
                    {
                        batchInput = batchInputAll.Narrow(0, start, batchSize);
                        rowIndices = rowIndicesAll.Narrow(0, start, batchSize);
                        routeWeights = routeWeightsAll.Narrow(0, start, batchSize);

                        gate = ExpertLinearForwardAlloc(batchInput, layer, e, kind: 0);
                        up = ExpertLinearForwardAlloc(batchInput, layer, e, kind: 1);
                        if (gate == null || up == null)
                            return false;

                        Ops.SiLUMul(gate, gate, up);

                        down = ExpertLinearForwardAlloc(gate, layer, e, kind: 2);
                        if (down == null)
                            return false;

                        if (!MlxFusedOps.TryScatterAddWeightedRows(output, down, rowIndices, routeWeights))
                            return false;
                    }
                    finally
                    {
                        down?.Dispose();
                        up?.Dispose();
                        gate?.Dispose();
                        // Narrow views: dispose to release their tensor
                        // wrappers; the underlying storage stays alive via
                        // batchInputAll / rowIndicesAll / routeWeightsAll.
                        batchInput?.Dispose();
                        rowIndices?.Dispose();
                        routeWeights?.Dispose();
                    }
                }

                if (!TryAddSharedExpertMlx(output, input, sharedDownAll, sharedGateInpVec, seqLen, hiddenSize))
                    return false;

                return true;
            }
            catch (NotSupportedException)
            {
                return false;
            }
            catch (InvalidOperationException)
            {
                return false;
            }
            finally
            {
                batchInputAll?.Dispose();
                routeWeightsAll?.Dispose();
                rowIndicesAll?.Dispose();
            }
        }

        // 中文：MLX 路径把共享专家输出叠加到 output：无门控时直接相加，有门控向量时按 sigmoid(input·gateInp) 逐 token 缩放后相加；前置不满足返回 false。
        private bool TryAddSharedExpertMlx(Tensor output, Tensor input, Tensor sharedDownAll, Tensor sharedGateInpVec, int seqLen, int hiddenSize)
        {
            if (sharedDownAll == null)
                return true;

            if (sharedGateInpVec == null)
            {
                Ops.Add(output, output, sharedDownAll);
                return true;
            }

            if (sharedGateInpVec.ElementType != DType.Float32
                || sharedGateInpVec.ElementCount() != hiddenSize
                || sharedGateInpVec.Storage is not MlxStorage)
            {
                return false;
            }

            using var gateMatrix = sharedGateInpVec.View(hiddenSize, 1);
            using var gateLogits = new Tensor(_allocator, DType.Float32, seqLen, 1);
            Ops.Addmm(gateLogits, 0, gateLogits, 1.0f, input, gateMatrix);
            Ops.Sigmoid(gateLogits, gateLogits);

            using var gatedShared = new Tensor(_allocator, DType.Float32, seqLen, hiddenSize);
            Ops.Mul(gatedShared, sharedDownAll, gateLogits);
            Ops.Add(output, output, gatedShared);
            return true;
        }

        // 中文：通用 MoE SwiGLU FFN（qwen35moe/qwen3next）：算路由 logits → top-K 专家与（归一化）权重 → 各专家 SwiGLU 加权求和 → 可选共享专家（可门控），按后端/序列长度选择设备路由、批处理或逐 token 路径，返回 FFN 输出张量。
        private unsafe Tensor MoEForward(Tensor input, int layer, int seqLen)
        {
            int hiddenSize = Config.HiddenSize;

            Tensor routerLogits = LinearForwardCached(input, _ffnGateInpQW[layer], _ffnGateInpF32[layer]);
            if (routerLogits == null)
                throw new InvalidOperationException($"Missing MoE router weight for layer {layer}: {_ffnGateInpKey[layer]}");

            bool routeRowsAreLogits = _normTopKProb;

            // Device-routing fast path: skip the per-MoE-layer routerData
            // host sync entirely. Compute top-K + softmax on the device and
            // feed the resulting [K] int32 + [1, K] float32 device tensors
            // directly into the batched MoE matmul.
            //
            // Requires: greedy router, MLX decode-on-device + batched MoE
            // available, no shared-expert gate (the gate scalar still uses
            // host VecDot today). Falls through to the host path otherwise.
            bool hasSharedGate = _hasSharedExperts != null && _hasSharedExperts[layer]
                && _hasSharedExpertGate != null && _hasSharedExpertGate[layer]
                && _ffnGateInpShexpVec?[layer] != null;
            if (MlxDeviceRouter
                && _backend == BackendType.Mlx
                && seqLen == 1
                && routeRowsAreLogits
                && MlxBatchedMoeDecode
                && !hasSharedGate
                && _layerStackedGate != null && _layerStackedGate[layer] != null
                && _moeGateBuf != null)
            {
                Tensor? maybe = TryMoEForwardDeviceRouter(input, routerLogits, layer);
                if (maybe != null)
                {
                    return maybe;
                }
                // Fell through: continue with the original host-routing path.
                // routerLogits is still alive.
            }

            Tensor routerData;
            if (routeRowsAreLogits)
            {
                routerData = routerLogits;
            }
            else
            {
                routerData = Ops.Softmax(null, routerLogits);
                routerLogits.Dispose();
            }

            float* routePtr = GetFloatPtr(routerData);

            var output = new Tensor(_allocator, DType.Float32, seqLen, hiddenSize);
            float* outputPtr = null;
            // MLX decode (seqLen == 1) keeps expert accumulation on-device
            // (one Ops.Fill then per-expert weighted-add via
            // TryAddScaledInPlace) and syncs once at the layer boundary
            // instead of forcing a host round-trip per expert.
            bool mlxDecodeOnDevice = _backend == BackendType.Mlx
                && seqLen == 1;
            if (_backend == BackendType.Mlx && (seqLen > 1 || mlxDecodeOnDevice))
            {
                Ops.Fill(output, 0f);
            }
            else
            {
                outputPtr = GetFloatPtr(output);
                VecZero(outputPtr, seqLen * hiddenSize);
            }

            // For prefill, batch the shared expert over all tokens up-front.
            Tensor sharedDownAll = null;
            float* sharedGateInpPtr = null;
            int sharedGateInpDim = hiddenSize;

            if (_hasSharedExperts != null && _hasSharedExperts[layer])
            {
                Tensor sharedGate = LinearForwardCached(input, _ffnGateShexpQW[layer], _ffnGateShexpF32[layer]);
                Tensor sharedUp = LinearForwardCached(input, _ffnUpShexpQW[layer], _ffnUpShexpF32[layer]);
                if (sharedGate != null && sharedUp != null)
                {
                    Ops.SiLUMul(sharedGate, sharedGate, sharedUp);
                    sharedDownAll = LinearForwardCached(sharedGate, _ffnDownShexpQW[layer], _ffnDownShexpF32[layer]);
                }
                sharedUp?.Dispose();
                sharedGate?.Dispose();

                if (_hasSharedExpertGate != null && _hasSharedExpertGate[layer])
                {
                    var gateInpVec = _ffnGateInpShexpVec[layer];
                    if (gateInpVec != null)
                    {
                        sharedGateInpPtr = GetFloatPtr(gateInpVec);
                        sharedGateInpDim = (int)gateInpVec.ElementCount();
                    }
                }
            }

            if (_backend == BackendType.Mlx
                && TryMoEPrefillBatchedMlx(input, output, routePtr, routeRowsAreLogits, sharedDownAll, _ffnGateInpShexpVec?[layer], layer, seqLen, hiddenSize))
            {
                sharedDownAll?.Dispose();
                routerData.Dispose();
                return output;
            }

            // For the mlxDecodeOnDevice path we already zeroed `output` on
            // device above; re-syncing it to host here would force a
            // device→host copy and discard that work. Likewise `input` only
            // needs a host pointer if a downstream code path (shared expert
            // gate scalar / legacy outRow accumulate) actually reads it.
            float* inputPtr = null;
            if (!mlxDecodeOnDevice)
            {
                outputPtr = GetFloatPtr(output);
                VecZero(outputPtr, seqLen * hiddenSize);
                inputPtr = GetFloatPtr(input);
            }
            else if (sharedDownAll != null && sharedGateInpPtr != null)
            {
                // Shared expert with a gate vector needs the input row on
                // host for the dot product. Only sync here, never above.
                inputPtr = GetFloatPtr(input);
            }
            float[] routeW = _moeRouteW;
            int[] topExperts = _moeTopExperts;

            // Reuse scratch buffers for both decode and prefill. For prefill (seqLen > 1),
            // instead of Narrow + NewContiguous per token (which allocates and copies), we
            // reuse a single [1, hiddenSize] tensor and copy the row data into it with
            // Buffer.MemoryCopy. This eliminates O(seqLen * numMoELayers) allocations.
            bool useReusedBuffers = _moeGateBuf != null;

            // For prefill: reuse a single row tensor that we populate via memcpy per token.
            Tensor prefillRowBuf = null;
            if (seqLen > 1 && useReusedBuffers && _moeTokenInput != null
                && _moeTokenInput.Sizes[1] == hiddenSize)
            {
                prefillRowBuf = _moeTokenInput;
            }

            // Prefill batched-by-expert path: group tokens by expert assignment
            // and run batched matmuls instead of per-token dispatches.
            // Converts seqLen*numExpertsUsed individual matmuls → numExperts batched matmuls.
            if (seqLen > 1 && _expertGateQW != null && _expertGateQW[layer] != null)
            {
                // 1. Route all tokens and group by expert
                var expertBatches = new List<(int tokenIdx, float weight)>[_numExperts];
                for (int i = 0; i < _numExperts; i++)
                    expertBatches[i] = new List<(int, float)>();

                for (int s = 0; s < seqLen; s++)
                {
                    float* routeRow = routePtr + (long)s * _numExperts;
                    int[] tokTop = new int[_numExpertsUsed];
                    float[] tokW = new float[_numExpertsUsed];
                    SelectTopKRouteWeights(routeRow, routeRowsAreLogits, tokTop, tokW);
                    for (int k = 0; k < _numExpertsUsed; k++)
                        expertBatches[tokTop[k]].Add((s, tokW[k]));
                }

                // 2. Process each expert with batched tokens
                int rowBytes = hiddenSize * sizeof(float);
                for (int e = 0; e < _numExperts; e++)
                {
                    var batch = expertBatches[e];
                    if (batch.Count == 0) continue;
                    int batchSize = batch.Count;

                    // Gather input rows
                    var batchInput = new Tensor(_allocator, DType.Float32, batchSize, hiddenSize);
                    float* batchPtr = GetFloatPtr(batchInput);
                    for (int b = 0; b < batchSize; b++)
                        Buffer.MemoryCopy(inputPtr + (long)batch[b].tokenIdx * hiddenSize,
                            batchPtr + (long)b * hiddenSize, rowBytes, rowBytes);

                    // Batched expert FFN: gate → up → SiLU*up → down
                    Tensor gate = ExpertLinearForwardAlloc(batchInput, layer, e, kind: 0);
                    Tensor up = ExpertLinearForwardAlloc(batchInput, layer, e, kind: 1);
                    batchInput.Dispose();
                    if (gate == null || up == null) { gate?.Dispose(); up?.Dispose(); continue; }

                    Ops.SiLUMul(gate, gate, up);
                    up.Dispose();

                    Tensor down = ExpertLinearForwardAlloc(gate, layer, e, kind: 2);
                    gate.Dispose();
                    if (down == null) continue;

                    // Scatter-accumulate with routing weights
                    float* downPtr = GetFloatPtr(down);
                    for (int b = 0; b < batchSize; b++)
                    {
                        float w = batch[b].weight;
                        float* src = downPtr + (long)b * hiddenSize;
                        float* dst = outputPtr + (long)batch[b].tokenIdx * hiddenSize;
                        VecScaleAdd(dst, src, w, hiddenSize);
                    }
                    down.Dispose();
                }

                // 3. Add shared expert contribution (already computed in batch)
                if (sharedDownAll != null)
                {
                    for (int s = 0; s < seqLen; s++)
                    {
                        float gateScalar = 1.0f;
                        if (sharedGateInpPtr != null)
                        {
                            float* tokenRow = inputPtr + (long)s * hiddenSize;
                            int n = Math.Min(sharedGateInpDim, hiddenSize);
                            gateScalar = SigmoidScalar(VecDot(tokenRow, sharedGateInpPtr, n));
                        }
                        float* sharedPtr = GetFloatPtr(sharedDownAll) + (long)s * hiddenSize;
                        VecScaleAdd(outputPtr + (long)s * hiddenSize, sharedPtr, gateScalar, hiddenSize);
                    }
                }
            }
            else
            {
            // Original token-by-token path (decode and fallback)
            for (int s = 0; s < seqLen; s++)
            {
                float* routeRow = routePtr + (long)s * _numExperts;
                SelectTopKRouteWeights(routeRow, routeRowsAreLogits, topExperts, routeW);

                Tensor tokenInput;
                bool disposeTokenInput;
                if (seqLen == 1)
                {
                    tokenInput = input;
                    disposeTokenInput = false;
                }
                else if (prefillRowBuf != null)
                {
                    float* srcRow = inputPtr + (long)s * hiddenSize;
                    float* dstRow = GetFloatPtr(prefillRowBuf);
                    long bytes = (long)hiddenSize * sizeof(float);
                    Buffer.MemoryCopy(srcRow, dstRow, bytes, bytes);
                    InvalidateTensorDeviceCache(prefillRowBuf);
                    tokenInput = prefillRowBuf;
                    disposeTokenInput = false;
                }
                else
                {
                    using var rowView = input.Narrow(0, s, 1);
                    tokenInput = Ops.NewContiguous(rowView);
                    disposeTokenInput = true;
                }

                // For MLX decode (seqLen=1), keep accumulation on device so
                // the kernels for all 8 active experts queue up without a
                // host sync between them. The legacy host-side accumulation
                // is still used for other backends and for fallback.
                if (mlxDecodeOnDevice && useReusedBuffers)
                {
                    RunMoEExpertsReusedMlxOnDevice(output, tokenInput, layer, topExperts, routeW);

                    if (sharedDownAll != null)
                    {
                        float gateScalar = 1.0f;
                        if (sharedGateInpPtr != null && inputPtr != null)
                        {
                            // seqLen==1 in this branch, so the input row is
                            // just inputPtr (no s*hiddenSize offset).
                            int n = Math.Min(sharedGateInpDim, hiddenSize);
                            gateScalar = SigmoidScalar(VecDot(inputPtr, sharedGateInpPtr, n));
                        }
                        AddScaledTensorMlx(output, sharedDownAll, gateScalar);
                    }
                }
                else
                {
                    float* outRow = outputPtr + (long)s * hiddenSize;

                    if (useReusedBuffers)
                    {
                        RunMoEExpertsReused(tokenInput, layer, topExperts, routeW, outRow, hiddenSize);
                    }
                    else
                    {
                        RunMoEExpertsAllocating(tokenInput, layer, topExperts, routeW, outRow, hiddenSize);
                    }

                    if (sharedDownAll != null)
                    {
                        float gateScalar = 1.0f;
                        if (sharedGateInpPtr != null)
                        {
                            float* tokenRow = inputPtr + (long)s * hiddenSize;
                            int n = Math.Min(sharedGateInpDim, hiddenSize);
                            gateScalar = SigmoidScalar(VecDot(tokenRow, sharedGateInpPtr, n));
                        }
                        float* sharedPtr = GetFloatPtr(sharedDownAll) + (long)s * hiddenSize;
                        VecScaleAdd(outRow, sharedPtr, gateScalar, hiddenSize);
                    }
                }

                if (disposeTokenInput)
                    tokenInput.Dispose();
            }
            } // end of token-by-token else

            sharedDownAll?.Dispose();
            routerData.Dispose();

            InvalidateTensorDeviceCache(output);
            return output;
        }

        // MLX-only decode helper: keep all expert accumulation on the GPU.
        // The legacy RunMoEExpertsReused path issues 3 matmuls + SiLUMul per
        // expert and then calls GetFloatPtr+VecScaleAdd, which forces an
        // mlx_eval+device→host sync after every expert. With 8 active
        // experts × 60 MoE layers that's ~480 round trips per decode token.
        //
        // Two on-device variants:
        //  - Batched: a single custom Metal kernel processes all K active
        //    experts' gate (then up, then down) matmuls in one Metal
        //    dispatch each. Requires a per-quant-type batched MoE kernel
        //    (currently only IQ2_XXS). Saves K-1 dispatches per matmul:
        //    on Qwen3.5 with K=8 that's ~21 dispatches saved per MoE
        //    layer (was 24 individual matmuls + 8 SiLUMul + 8 AddScaled →
        //    now 3 batched matmuls + 1 SiLUMul + 1 routeW@down matmul).
        //  - Sequential (fallback): per-expert matmul loop with the
        //    fused AddScaled accumulator. Used when no batched kernel
        //    is available for the layer's quant type.
        // 中文：MLX 解码专用：让全部激活专家的累加保持在 GPU 上以避免逐专家主机同步——优先走批处理 kernel，否则按专家循环用复用 scratch 做 gate/up/SiLUMul/down 并 AddScaled 累加到 output。
        private void RunMoEExpertsReusedMlxOnDevice(Tensor output, Tensor tokenInput, int layer,
            int[] topExperts, float[] routeW)
        {
            if (MlxBatchedMoeDecode && TryRunMoEExpertsBatchedMlx(output, tokenInput, layer, topExperts, routeW))
                return;

            for (int k = 0; k < _numExpertsUsed; k++)
            {
                int e = topExperts[k];
                if (!ExpertLinearForwardInto(_moeGateBuf, tokenInput, layer, e, kind: 0))
                    continue;
                if (!ExpertLinearForwardInto(_moeUpBuf, tokenInput, layer, e, kind: 1))
                    continue;

                Ops.SiLUMul(_moeGateBuf, _moeGateBuf, _moeUpBuf);

                if (!ExpertLinearForwardInto(_moeDownBuf, _moeGateBuf, layer, e, kind: 2))
                    continue;

                AddScaledTensorMlx(output, _moeDownBuf, routeW[k]);
            }
        }

        // Decode-only MoEForward that does the routing on-device. Skips
        // the per-MoE-layer host sync on routerLogits/routerData by:
        //   1. Running the top-K + softmax of routerLogits on device →
        //      _moeBatchedExpertIndices [K] int32, _moeBatchedRouteWeights
        //      [1, K] float32.
        //   2. Running the batched MoE matmul using those device tensors.
        //   3. Adding the shared expert contribution if applicable
        //      (no host-side gate scalar — only used when there's no
        //      shared-expert gate, asserted by the caller).
        // Returns null on precondition failure so the caller can fall
        // through to the host-routing path.
        // 中文：解码版 MoEForward 的设备端路由路径：在设备上算 top-K+softmax，直接用得到的设备索引/权重张量跑批处理 MoE 矩阵乘并叠加共享专家，省去对 routerLogits 的主机同步；前置不满足返回 null 让调用方回退主机路由。
        private Tensor? TryMoEForwardDeviceRouter(Tensor input, Tensor routerLogits, int layer)
        {
            int hiddenSize = Config.HiddenSize;
            int K = _numExpertsUsed;

            // Lazy-allocate the batched scratch buffers. Same buffers as
            // TryRunMoEExpertsBatchedMlx so reuse the existing shapes /
            // disposal hooks. We need the indices + routeWeights tensors
            // before the batched MoE runs; the gate/up/down buffers are
            // also needed and have the same lifecycle.
            var gateW = _layerStackedGate[layer];
            if (gateW == null) return null;
            int intermediate = (int)gateW.PerExpertNe1;

            if (_moeBatchedGate == null
                || _moeBatchedGate.Sizes[0] != K || _moeBatchedGate.Sizes[1] != intermediate)
            {
                _moeBatchedGate?.Dispose();
                _moeBatchedUp?.Dispose();
                _moeBatchedDown?.Dispose();
                _moeBatchedExpertIndices?.Dispose();
                _moeBatchedRouteWeights?.Dispose();
                _moeBatchedGate = new Tensor(_allocator, DType.Float32, K, intermediate);
                _moeBatchedUp = new Tensor(_allocator, DType.Float32, K, intermediate);
                _moeBatchedDown = new Tensor(_allocator, DType.Float32, K, hiddenSize);
                _moeBatchedExpertIndices = new Tensor(_allocator, DType.Int32, K);
                _moeBatchedRouteWeights = new Tensor(_allocator, DType.Float32, 1, K);
            }

            // 1. Device-side top-K + softmax on routerLogits.
            if (!MlxFusedOps.TryMoeRouterTopKSoftmax(
                    routerLogits, _moeBatchedExpertIndices, _moeBatchedRouteWeights))
            {
                return null;
            }

            // 2. Allocate output, zero on device.
            var output = new Tensor(_allocator, DType.Float32, 1, hiddenSize);
            Ops.Fill(output, 0f);

            // 3. Pre-compute shared expert (no shared gate path here).
            Tensor sharedDownAll = null;
            if (_hasSharedExperts != null && _hasSharedExperts[layer])
            {
                Tensor sharedGate = LinearForwardCached(input, _ffnGateShexpQW[layer], _ffnGateShexpF32[layer]);
                Tensor sharedUp = LinearForwardCached(input, _ffnUpShexpQW[layer], _ffnUpShexpF32[layer]);
                if (sharedGate != null && sharedUp != null)
                {
                    Ops.SiLUMul(sharedGate, sharedGate, sharedUp);
                    sharedDownAll = LinearForwardCached(sharedGate, _ffnDownShexpQW[layer], _ffnDownShexpF32[layer]);
                }
                sharedUp?.Dispose();
                sharedGate?.Dispose();
            }

            try
            {
                // 4. Batched MoE matmul using the device tensors we just
                //    computed. Note we bypass TryRunMoEExpertsBatchedMlx's
                //    SetElementsAsInt/SetElementsAsFloat (since the tensors
                //    are already populated) by inlining a smaller variant.
                if (!RunBatchedMoeMatmulFromDevice(output, input, layer))
                {
                    output.Dispose();
                    sharedDownAll?.Dispose();
                    return null;
                }

                // 5. Shared expert add (no host scalar — the gate-less
                //    path just adds the full shared down).
                if (sharedDownAll != null)
                {
                    Ops.Add(output, output, sharedDownAll);
                }
                routerLogits.Dispose();
                return output;
            }
            finally
            {
                sharedDownAll?.Dispose();
            }
        }

        // Inner helper: runs the 3 batched matmuls + SiLUMul + routeW@down
        // chain assuming _moeBatchedExpertIndices and _moeBatchedRouteWeights
        // are already populated (typically by the device-router path).
        // 中文：内层辅助：假设设备端专家索引/路由权重已就绪，跑 3 个批处理矩阵乘 + SiLUMul + routeW@down 链并把加权专家输出累加进 output（供设备路由路径调用）。
        private bool RunBatchedMoeMatmulFromDevice(Tensor output, Tensor tokenInput, int layer)
        {
            var gateW = _layerStackedGate[layer];
            var upW = _layerStackedUp[layer];
            var downW = _layerStackedDown[layer];
            if (gateW == null || upW == null || downW == null) return false;
            if (!MlxQuantizedOps.SupportsBatchedMoeMatmul(gateW.GgmlType)) return false;
            try
            {
                if (!MlxQuantizedOps.TryMoeMatmulBatched(
                        _moeBatchedGate, tokenInput, _moeBatchedExpertIndices,
                        gateW.Data, gateW.Data, gateW.GgmlType,
                        gateW.PerExpertNe0, gateW.PerExpertNe1, gateW.NumExperts, gateW.TotalRawBytes,
                        sharedInput: true))
                    return false;
                if (!MlxQuantizedOps.TryMoeMatmulBatched(
                        _moeBatchedUp, tokenInput, _moeBatchedExpertIndices,
                        upW.Data, upW.Data, upW.GgmlType,
                        upW.PerExpertNe0, upW.PerExpertNe1, upW.NumExperts, upW.TotalRawBytes,
                        sharedInput: true))
                    return false;
                Ops.SiLUMul(_moeBatchedGate, _moeBatchedGate, _moeBatchedUp);
                if (!MlxQuantizedOps.TryMoeMatmulBatched(
                        _moeBatchedDown, _moeBatchedGate, _moeBatchedExpertIndices,
                        downW.Data, downW.Data, downW.GgmlType,
                        downW.PerExpertNe0, downW.PerExpertNe1, downW.NumExperts, downW.TotalRawBytes,
                        sharedInput: false))
                    return false;
                Ops.Addmm(output, 1.0f, output, 1.0f, _moeBatchedRouteWeights, _moeBatchedDown);
                return true;
            }
            catch
            {
                return false;
            }
        }

        // Batched-MoE decode. Requires:
        //  - Stacked expert weights for gate/up/down on this layer
        //  - A batched-MoE matmul kernel for the weights' quant type
        //  - All experts share the same (quant type, dimensions)
        // Returns false if any precondition fails so the caller falls back
        // to the sequential path.
        // 中文：MLX 批处理 MoE 解码：上传 top-K 索引与路由权重，用堆叠专家权重一次 Metal dispatch 完成 gate/up（可融合 SiLU）/down 批处理矩阵乘，再以 routeW@down 加权累加进 output；缺少堆叠权重或 kernel 不支持时返回 false 回退顺序路径。
        private bool TryRunMoEExpertsBatchedMlx(Tensor output, Tensor tokenInput, int layer,
            int[] topExperts, float[] routeW)
        {
            // Need stacked weights for all three projections.
            if (_layerStackedGate == null || _layerStackedUp == null || _layerStackedDown == null)
                return false;
            var gateW = _layerStackedGate[layer];
            var upW = _layerStackedUp[layer];
            var downW = _layerStackedDown[layer];
            if (gateW == null || upW == null || downW == null)
                return false;
            if (!MlxQuantizedOps.SupportsBatchedMoeMatmul(gateW.GgmlType)) return false;
            if (gateW.GgmlType != upW.GgmlType || upW.GgmlType != downW.GgmlType) return false;
            // gate/up share input dim (hidden) and output dim (intermediate)
            if (gateW.PerExpertNe0 != upW.PerExpertNe0 || gateW.PerExpertNe1 != upW.PerExpertNe1) return false;
            // down: input dim = intermediate, output dim = hidden
            if (downW.PerExpertNe0 != gateW.PerExpertNe1 || downW.PerExpertNe1 != gateW.PerExpertNe0) return false;

            int K = _numExpertsUsed;
            int hiddenSize = (int)gateW.PerExpertNe0;
            int intermediate = (int)gateW.PerExpertNe1;
            if (tokenInput.DimensionCount != 2 || tokenInput.Sizes[0] != 1 || tokenInput.Sizes[1] != hiddenSize)
                return false;
            if (output.DimensionCount != 2 || output.Sizes[0] != 1 || output.Sizes[1] != hiddenSize)
                return false;

            // Lazy-init the batched scratch buffers.
            if (_moeBatchedGate == null
                || _moeBatchedGate.Sizes[0] != K || _moeBatchedGate.Sizes[1] != intermediate)
            {
                _moeBatchedGate?.Dispose();
                _moeBatchedUp?.Dispose();
                _moeBatchedDown?.Dispose();
                _moeBatchedExpertIndices?.Dispose();
                _moeBatchedRouteWeights?.Dispose();
                _moeBatchedGate = new Tensor(_allocator, DType.Float32, K, intermediate);
                _moeBatchedUp = new Tensor(_allocator, DType.Float32, K, intermediate);
                _moeBatchedDown = new Tensor(_allocator, DType.Float32, K, hiddenSize);
                _moeBatchedExpertIndices = new Tensor(_allocator, DType.Int32, K);
                _moeBatchedRouteWeights = new Tensor(_allocator, DType.Float32, 1, K);
            }

            // Upload topK expert indices and routing weights to device.
            // These are tiny (K ints + K floats) so the upload cost is
            // negligible compared to the per-MoE-layer GPU work we save.
            _moeBatchedExpertIndices.SetElementsAsInt(topExperts);
            _moeBatchedRouteWeights.SetElementsAsFloat(routeW);

            try
            {
                // 1+2+3 fused: produce _moeBatchedGate = silu(gate @ x) * (up @ x)
                // in ONE Metal kernel. Falls back to the 3-dispatch sequence
                // if the fused kernel isn't available (e.g. quant type
                // doesn't support it yet) or fails. Gated by
                // TS_MLX_MOE_FUSED_GATE_UP_SILU=0 to disable for A/B.
                bool fusedOk = false;
                if (!MlxMoeFusedGateUpSiluDisabled)
                {
                    fusedOk = MlxQuantizedOps.TryMoeFusedGateUpSilu(
                        _moeBatchedGate, tokenInput, _moeBatchedExpertIndices,
                        gateW.Data, gateW.Data, gateW.TotalRawBytes,
                        upW.Data, upW.Data, upW.TotalRawBytes,
                        gateW.GgmlType,
                        gateW.PerExpertNe0, gateW.PerExpertNe1, gateW.NumExperts);
                }
                if (!fusedOk)
                {
                    // 1. Batched gate matmul: [1, hidden] @ stackedGate → [K, intermediate]
                    if (!MlxQuantizedOps.TryMoeMatmulBatched(
                            _moeBatchedGate, tokenInput, _moeBatchedExpertIndices,
                            gateW.Data, gateW.Data, gateW.GgmlType,
                            gateW.PerExpertNe0, gateW.PerExpertNe1, gateW.NumExperts, gateW.TotalRawBytes,
                            sharedInput: true))
                        return false;

                    // 2. Batched up matmul: same input → [K, intermediate]
                    if (!MlxQuantizedOps.TryMoeMatmulBatched(
                            _moeBatchedUp, tokenInput, _moeBatchedExpertIndices,
                            upW.Data, upW.Data, upW.GgmlType,
                            upW.PerExpertNe0, upW.PerExpertNe1, upW.NumExperts, upW.TotalRawBytes,
                            sharedInput: true))
                        return false;

                    // 3. SiLUMul element-wise on [K, intermediate]: one kernel.
                    Ops.SiLUMul(_moeBatchedGate, _moeBatchedGate, _moeBatchedUp);
                }

                // 4. Batched down matmul: [K, intermediate] @ stackedDown → [K, hidden]
                if (!MlxQuantizedOps.TryMoeMatmulBatched(
                        _moeBatchedDown, _moeBatchedGate, _moeBatchedExpertIndices,
                        downW.Data, downW.Data, downW.GgmlType,
                        downW.PerExpertNe0, downW.PerExpertNe1, downW.NumExperts, downW.TotalRawBytes,
                        sharedInput: false))
                    return false;

                // 5. Weighted sum: routeW[1, K] @ down[K, hidden] = [1, hidden]
                //    Then accumulate into output. We do this as one Addmm:
                //    output = 1.0 * output + 1.0 * (routeW @ down)
                Ops.Addmm(output, 1.0f, output, 1.0f, _moeBatchedRouteWeights, _moeBatchedDown);
                return true;
            }
            catch
            {
                return false;
            }
        }

        // output += scalar * src, all on device, no host round trip. With the
        // mlx_compile path enabled this is ONE fused Metal kernel via
        // MlxFusedOps.TryAddScaledInPlace; otherwise we fall back to
        // mulv + addt (2 kernels). The fallback variant is destructive on
        // `src` — _moeDownBuf is rewritten next iteration by the next
        // expert's down matmul anyway, so this is safe in the decode loop.
        // 中文：MLX 设备端就地累加 output += scalar * src（优先单融合 kernel，否则回退 mul+add 两步，回退会破坏 src），无主机往返。
        private void AddScaledTensorMlx(Tensor output, Tensor src, float scalar)
        {
            if (scalar == 0f)
                return;

            if (MlxFusedOps.TryAddScaledInPlace(output, src, scalar))
                return;

            // Eager fallback: 2 MLX kernels (mulv + addt).
            Ops.Mul(src, src, scalar);
            Ops.Add(output, output, src);
        }

        // 中文：用复用 scratch 张量跑选中专家的 SwiGLU 并按路由权重累加到 outRow：优先把全部专家的 gate/up/silu*mul/down 融合进单次 GGML 批处理图，否则逐专家 dispatch 回退。
        private unsafe void RunMoEExpertsReused(Tensor tokenInput, int layer,
            int[] topExperts, float[] routeW, float* outRow, int hiddenSize)
        {
            // Fast path: batched SwiGLU MoE in a single GGML graph (decode only).
            // Combines 4*N = 32 dispatches (gate/up/silu*mul/down for 8 experts) into one
            // graph submission, dramatically reducing GPU dispatch overhead on Metal/CUDA.
            if (IsGgmlBackend
                && _moeBatchedResult != null
                && _moeGatePtrs != null
                && _expertGateQW != null && _expertUpQW != null && _expertDownQW != null
                && _expertGateQW[layer] != null && _expertUpQW[layer] != null && _expertDownQW[layer] != null)
            {
                bool allQuantized = true;
                QuantizedWeight gQW0 = null, uQW0 = null, dQW0 = null;
                for (int k = 0; k < _numExpertsUsed; k++)
                {
                    int e = topExperts[k];
                    var g = _expertGateQW[layer][e];
                    var u = _expertUpQW[layer][e];
                    var d = _expertDownQW[layer][e];
                    if (g == null || u == null || d == null)
                    {
                        allQuantized = false;
                        break;
                    }
                    if (gQW0 == null) { gQW0 = g; uQW0 = u; dQW0 = d; }

                    // All experts in a layer must share dtype + shape so the batched kernel can
                    // bind them as identically-shaped tensors. The Qwen3 GGUF format guarantees
                    // this, but we keep a defensive check for forward compatibility.
                    if (g.GgmlType != gQW0.GgmlType || g.Ne0 != gQW0.Ne0 || g.Ne1 != gQW0.Ne1 ||
                        u.GgmlType != uQW0.GgmlType || u.Ne0 != uQW0.Ne0 || u.Ne1 != uQW0.Ne1 ||
                        d.GgmlType != dQW0.GgmlType || d.Ne0 != dQW0.Ne0 || d.Ne1 != dQW0.Ne1)
                    {
                        allQuantized = false;
                        break;
                    }

                    _moeGatePtrs[k] = g.CacheKey;
                    _moeUpPtrs[k] = u.CacheKey;
                    _moeDownPtrs[k] = d.CacheKey;
                }

                if (allQuantized)
                {
                    long t0exp = Stopwatch.GetTimestamp();
                    GgmlBasicOps.MoEExpertsSwiGLUForward(
                        _moeBatchedResult, tokenInput,
                        _numExpertsUsed,
                        _moeGatePtrs, _moeUpPtrs, _moeDownPtrs,
                        gQW0.GgmlType, gQW0.Ne0, gQW0.Ne1, gQW0.RawBytes,
                        uQW0.GgmlType, uQW0.Ne0, uQW0.Ne1, uQW0.RawBytes,
                        dQW0.GgmlType, dQW0.Ne0, dQW0.Ne1, dQW0.RawBytes,
                        routeW);
                    _linearTicks += Stopwatch.GetTimestamp() - t0exp;

                    InvalidateTensorDeviceCache(_moeBatchedResult);
                    float* batchedPtr = GetFloatPtr(_moeBatchedResult);
                    VecScaleAdd(outRow, batchedPtr, 1.0f, hiddenSize);
                    return;
                }
            }

            // Fallback: per-expert dispatch using reusable scratch tensors.
            for (int k = 0; k < _numExpertsUsed; k++)
            {
                int e = topExperts[k];
                if (!ExpertLinearForwardInto(_moeGateBuf, tokenInput, layer, e, kind: 0))
                    continue;
                if (!ExpertLinearForwardInto(_moeUpBuf, tokenInput, layer, e, kind: 1))
                    continue;

                Ops.SiLUMul(_moeGateBuf, _moeGateBuf, _moeUpBuf);

                if (!ExpertLinearForwardInto(_moeDownBuf, _moeGateBuf, layer, e, kind: 2))
                    continue;

                float w = routeW[k];
                float* downPtr = GetFloatPtr(_moeDownBuf);
                VecScaleAdd(outRow, downPtr, w, hiddenSize);
            }
        }

        // 中文：逐专家分配临时张量跑 SwiGLU（gate/up/SiLUMul/down）并按路由权重累加到 outRow 的 MoE 回退实现（不复用 scratch）。
        private unsafe void RunMoEExpertsAllocating(Tensor tokenInput, int layer,
            int[] topExperts, float[] routeW, float* outRow, int hiddenSize)
        {
            for (int k = 0; k < _numExpertsUsed; k++)
            {
                int e = topExperts[k];
                Tensor gate = ExpertLinearForwardAlloc(tokenInput, layer, e, kind: 0);
                Tensor up = ExpertLinearForwardAlloc(tokenInput, layer, e, kind: 1);
                if (gate == null || up == null)
                {
                    gate?.Dispose();
                    up?.Dispose();
                    continue;
                }

                Ops.SiLUMul(gate, gate, up);
                up.Dispose();

                Tensor down = ExpertLinearForwardAlloc(gate, layer, e, kind: 2);
                gate.Dispose();
                if (down == null)
                    continue;

                float w = routeW[k];
                float* downPtr = GetFloatPtr(down);
                VecScaleAdd(outRow, downPtr, w, hiddenSize);
                down.Dispose();
            }
        }

        /// <summary>
        /// Linear forward that allocates the result tensor, using cached expert weight
        /// references (kind: 0=gate, 1=up, 2=down). Used by prefill where the output rows
        /// vary per token. Returns null if the weight is missing.
        /// </summary>
        // 中文：用缓存的专家权重（kind: 0=gate,1=up,2=down）做线性前向并分配返回结果张量（量化走 AddmmQuant，F32 走转置 Addmm），权重缺失返回 null；prefill 用。
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Tensor ExpertLinearForwardAlloc(Tensor input, int layer, int expert, int kind)
        {
            long t0 = Stopwatch.GetTimestamp();
            QuantizedWeight qw = kind == 0 ? _expertGateQW[layer][expert]
                                : kind == 1 ? _expertUpQW[layer][expert]
                                            : _expertDownQW[layer][expert];

            if (qw != null)
            {
                Tensor result = new Tensor(_allocator, DType.Float32, input.Sizes[0], qw.Ne1);
                if (IsGgmlBackend)
                    GgmlBasicOps.AddmmQuant(result, input, qw.CacheKey, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes);
                else
                    AddmmQuantManaged(result, input, qw);
                _linearTicks += Stopwatch.GetTimestamp() - t0;
                return result;
            }

            Tensor w = kind == 0 ? _expertGateF32[layer][expert]
                     : kind == 1 ? _expertUpF32[layer][expert]
                                 : _expertDownF32[layer][expert];
            if (w != null)
            {
                Tensor result = new Tensor(_allocator, DType.Float32, input.Sizes[0], w.Sizes[0]);
                using var wT = w.Transpose();
                Ops.Addmm(result, 0, result, 1.0f, input, wT);
                _linearTicks += Stopwatch.GetTimestamp() - t0;
                return result;
            }

            return null;
        }

        /// <summary>
        /// LinearForward variant that takes pre-resolved weight references, eliminating the
        /// dictionary lookup that dominates the hot decode path for layer-shared weights
        /// such as MoE routers and shared expert projections.
        /// </summary>
        // 中文：使用预解析权重引用（量化或 F32）的线性前向并分配返回结果，消除 hot 路径字典查找；两者皆空返回 null。
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Tensor LinearForwardCached(Tensor input, QuantizedWeight qw, Tensor wF32)
        {
            long t0 = Stopwatch.GetTimestamp();
            Tensor result;

            if (qw != null)
            {
                int seqLen = (int)input.Sizes[0];
                int outDim = (int)qw.Ne1;
                result = new Tensor(_allocator, DType.Float32, seqLen, outDim);
                if (IsGgmlBackend)
                    GgmlBasicOps.AddmmQuant(result, input, qw.CacheKey, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes);
                else
                    AddmmQuantManaged(result, input, qw);
            }
            else if (wF32 != null)
            {
                int outDimF32 = (int)wF32.Sizes[0];
                int seqLenF32 = (int)input.Sizes[0];
                using var wT = wF32.Transpose();
                result = new Tensor(_allocator, DType.Float32, seqLenF32, outDimF32);
                Ops.Addmm(result, 0, result, 1.0f, input, wT);
            }
            else
            {
                return null;
            }

            _linearTicks += Stopwatch.GetTimestamp() - t0;
            return result;
        }

        /// <summary>
        /// Linear forward into a pre-allocated result tensor using cached expert weight
        /// references (kind: 0=gate, 1=up, 2=down). Returns false if the weight is missing.
        /// </summary>
        // 中文：用缓存的专家权重（kind: 0=gate,1=up,2=down）做线性前向并写入预分配的 result 张量，成功返回 true，权重缺失返回 false。
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool ExpertLinearForwardInto(Tensor result, Tensor input, int layer, int expert, int kind)
        {
            long t0 = Stopwatch.GetTimestamp();
            QuantizedWeight qw = kind == 0 ? _expertGateQW[layer][expert]
                                : kind == 1 ? _expertUpQW[layer][expert]
                                            : _expertDownQW[layer][expert];

            if (qw != null)
            {
                if (IsGgmlBackend)
                    GgmlBasicOps.AddmmQuant(result, input, qw.CacheKey, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes);
                else
                    AddmmQuantManaged(result, input, qw);
                _linearTicks += Stopwatch.GetTimestamp() - t0;
                return true;
            }

            Tensor w = kind == 0 ? _expertGateF32[layer][expert]
                     : kind == 1 ? _expertUpF32[layer][expert]
                                 : _expertDownF32[layer][expert];
            if (w != null)
            {
                using var wT = w.Transpose();
                Ops.Addmm(result, 0, result, 1.0f, input, wT);
                _linearTicks += Stopwatch.GetTimestamp() - t0;
                return true;
            }

            return false;
        }

        // 中文：在原地从 n 个值中选出前 k 大的下标（委托给 TensorComputePrimitives 实现），用于 MoE 路由选 top-K 专家。
        private static unsafe void SelectTopKInPlace(float* values, int n, int k, int[] indices) =>
            TensorComputePrimitives.SelectTopKInPlace(values, n, k, indices);

        // 中文：为一行路由数据选 top-K 专家并计算其路由权重：输入是 logits 时按归一化或全 softmax 求权重，输入是概率时直接取（必要时归一化），结果写入 topExperts/routeWeights。
        private unsafe void SelectTopKRouteWeights(float* routeRow, bool routeRowIsLogits, int[] topExperts, float[] routeWeights)
        {
            SelectTopKInPlace(routeRow, _numExperts, _numExpertsUsed, topExperts);

            if (routeRowIsLogits)
            {
                if (_normTopKProb)
                {
                    float maxLogit = float.NegativeInfinity;
                    for (int k = 0; k < _numExpertsUsed; k++)
                    {
                        float v = routeRow[topExperts[k]];
                        if (v > maxLogit)
                            maxLogit = v;
                    }

                    float sum = 0f;
                    for (int k = 0; k < _numExpertsUsed; k++)
                    {
                        float w = MathF.Exp(routeRow[topExperts[k]] - maxLogit);
                        routeWeights[k] = w;
                        sum += w;
                    }

                    if (sum > 0f)
                    {
                        float inv = 1.0f / sum;
                        for (int k = 0; k < _numExpertsUsed; k++)
                            routeWeights[k] *= inv;
                    }
                    return;
                }

                float fullMax = float.NegativeInfinity;
                for (int i = 0; i < _numExperts; i++)
                    if (routeRow[i] > fullMax)
                        fullMax = routeRow[i];

                float denom = 0f;
                for (int i = 0; i < _numExperts; i++)
                    denom += MathF.Exp(routeRow[i] - fullMax);

                float invDenom = denom > 0f ? 1.0f / denom : 0f;
                for (int k = 0; k < _numExpertsUsed; k++)
                    routeWeights[k] = MathF.Exp(routeRow[topExperts[k]] - fullMax) * invDenom;
                return;
            }

            float wSum = 0f;
            for (int k = 0; k < _numExpertsUsed; k++)
            {
                routeWeights[k] = routeRow[topExperts[k]];
                wSum += routeWeights[k];
            }

            if (_normTopKProb && wSum > 0f)
            {
                float inv = 1.0f / wSum;
                for (int k = 0; k < _numExpertsUsed; k++)
                    routeWeights[k] *= inv;
            }
        }

        #endregion

        #region Vision Support

        // 中文：从 mmproj 文件加载 Qwen3.5 视觉编码器并将本模型设为其宿主，启用多模态（图像）输入。
        public void LoadVisionEncoder(string mmProjPath)
        {
            VisionEncoder = new Qwen35VisionEncoder(mmProjPath, _allocator);
            VisionEncoder.SetHostModel(this);
        }

        // 中文：将一批视觉嵌入及其起始位置入队，等下一次 Forward 时注入到对应 image_pad token 位置。
        public void SetVisionEmbeddings(Tensor visionEmbeddings, int startPosition)
        {
            _visionEmbeddingsList.Add((visionEmbeddings, startPosition));
        }

        /// <summary>
        /// Inject vision embeddings into text embeddings at the image_pad token positions.
        /// </summary>
        // 中文：把已入队的各批视觉嵌入按其起始位置 memcpy 覆盖到文本嵌入张量对应的 image_pad token 行（维度/越界不符则跳过并告警），随后清空队列并释放视觉张量。
        private unsafe void InjectVisionEmbeddings(Tensor textEmbeddings, int seqLen)
        {
            if (_visionEmbeddingsList.Count == 0)
                return;

            float* textPtr = GetFloatPtr(textEmbeddings);
            int dim = Config.HiddenSize;
            foreach (var (visionEmbeddings, startPos) in _visionEmbeddingsList)
            {
                if (visionEmbeddings == null || startPos < 0)
                    continue;

                int numVisionTokens = (int)visionEmbeddings.Sizes[0];
                int projDim = (int)visionEmbeddings.Sizes[1];

                if (projDim != dim)
                {
                    Console.WriteLine($"Warning: Vision embedding dim ({projDim}) != text hidden dim ({dim}). Skipping injection.");
                    visionEmbeddings.Dispose();
                    continue;
                }

                if (startPos + numVisionTokens > seqLen)
                {
                    Console.WriteLine($"Warning: Vision tokens ({numVisionTokens}) exceed sequence at position {startPos}. Skipping.");
                    visionEmbeddings.Dispose();
                    continue;
                }

                float* visPtr = GetFloatPtr(visionEmbeddings);
                int bytes = numVisionTokens * dim * sizeof(float);
                Buffer.MemoryCopy(visPtr, textPtr + startPos * dim, bytes, bytes);

                Console.WriteLine($"  Injected {numVisionTokens} vision embeddings at position {startPos}");
                visionEmbeddings.Dispose();
            }

            _visionEmbeddingsList.Clear();
        }

        #endregion

        // 中文：打印推理计时统计：MLX 后端打印 linear/attention/norm/eval 等各阶段毫秒占比，其它后端走基类统计，并附带 GDN、prefill、decode 各阶段的细分计时。
        public override void PrintTimingStats()
        {
            if (_backend == BackendType.Mlx && _forwardCount > 0)
            {
                double totalMs = _forwardSw.Elapsed.TotalMilliseconds;
                double msPerTick = 1000.0 / Stopwatch.Frequency;
                double linearMs = _linearTicks * msPerTick;
                double attnMs = _attnTicks * msPerTick;
                double normMs = _normTicks * msPerTick;
                double embMs = _embTicks * msPerTick;
                double lmHeadMs = _lmHeadTicks * msPerTick;
                double logitsCopyMs = _logitsCopyTicks * msPerTick;
                double evalMs = _mlxEvalBoundaryTicks * msPerTick;
                double cacheEvalMs = _mlxCacheEvalTicks * msPerTick;
                double otherMs = totalMs - linearMs - attnMs - normMs - evalMs - cacheEvalMs;

                Console.WriteLine($"Timing ({_forwardCount} forward calls, {totalMs:F0} ms total, {totalMs / _forwardCount:F0} ms/token):");
                Console.WriteLine($"  Linear graph build: {linearMs:F0} ms ({100 * linearMs / totalMs:F1}%)");
                Console.WriteLine($"  Attention CPU/ops: {attnMs:F0} ms ({100 * attnMs / totalMs:F1}%)");
                Console.WriteLine($"  Norm:              {normMs:F0} ms ({100 * normMs / totalMs:F1}%)");
                Console.WriteLine($"  MLX graph eval:    {evalMs:F0} ms ({100 * evalMs / totalMs:F1}%, interval={MlxEvalEveryNLayers})");
                Console.WriteLine($"  MLX cache eval:    {cacheEvalMs:F0} ms ({100 * cacheEvalMs / totalMs:F1}%)");
                Console.WriteLine($"  (LM head:          {lmHeadMs:F0} ms, included in Linear/eval)");
                Console.WriteLine($"  (Embedding:        {embMs:F0} ms, in Other)");
                Console.WriteLine($"  (Logits copy:      {logitsCopyMs:F0} ms, in Other/eval)");
                Console.WriteLine($"  Other:             {otherMs:F0} ms ({100 * otherMs / totalMs:F1}%)");
            }
            else
            {
                base.PrintTimingStats();
            }

            if (_backend != BackendType.Mlx && _mlxEvalBoundaryTicks != 0)
            {
                double ms = _mlxEvalBoundaryTicks * 1000.0 / Stopwatch.Frequency;
                Console.WriteLine($"  (MLX layer eval: {ms:F0} ms, interval={MlxEvalEveryNLayers})");
            }
            if (_backend != BackendType.Mlx && _mlxCacheEvalTicks != 0)
            {
                double ms = _mlxCacheEvalTicks * 1000.0 / Stopwatch.Frequency;
                Console.WriteLine($"  (MLX cache eval: {ms:F0} ms, interval={MlxEvalEveryNLayers})");
            }
            PrintGdnTimingStats();
            PrintPrefillStageStats();
            PrintDecodeStageStats();
        }

        // 中文：当启用解码分阶段 profiling 时，打印每 token 的注意力块与循环块解码耗时（需 QWEN35_DECODE_PROFILE=1）。
        private void PrintDecodeStageStats()
        {
            if (!_profileDecodeStages || _decodeForwardCount == 0)
                return;
            double msPerTick = 1000.0 / Stopwatch.Frequency;
            double attnMs = _decodeAttnBlockTicks * msPerTick;
            double recMs = _decodeRecBlockTicks * msPerTick;
            double cnt = _decodeForwardCount;
            Console.WriteLine($"Decode stage breakdown ({_decodeForwardCount} decode forwards):");
            Console.WriteLine($"  Attention blocks: {attnMs:F0} ms total ({attnMs / cnt:F2} ms/token)");
            Console.WriteLine($"  Recurrent blocks: {recMs:F0} ms total ({recMs / cnt:F2} ms/token)");
        }

        // 中文：当启用 prefill 分阶段 profiling 时，打印嵌入、注意力块（QKV/拆分/QK-norm/RoPE/缓存/注意力计算/门控/输出/FFN 等细项）、循环块及最终 norm+LM head 的耗时占比（需 QWEN35_PREFILL_PROFILE=1）。
        private void PrintPrefillStageStats()
        {
            if (!_profilePrefillStages || _prefillTokenCount == 0)
                return;

            double msPerTick = 1000.0 / Stopwatch.Frequency;
            double embMs = _prefillEmbedTicks * msPerTick;
            double attnMs = _prefillAttnBlockTicks * msPerTick;
            double recMs = _prefillRecBlockTicks * msPerTick;
            double lmHeadMs = _prefillFinalLmHeadTicks * msPerTick;

            double attnQkvMs = _prefillAttnQkvTicks * msPerTick;
            double attnDeintMs = _prefillAttnDeinterleaveTicks * msPerTick;
            double attnQknMs = _prefillAttnQknormTicks * msPerTick;
            double attnRopeMs = _prefillAttnRopeTicks * msPerTick;
            double attnReshapeMs = _prefillAttnReshapeTicks * msPerTick;
            double attnCacheCopyMs = _prefillAttnCacheCopyTicks * msPerTick;
            double attnExpandKvMs = _prefillAttnExpandKvTicks * msPerTick;
            double attnComputeMs = _prefillAttnComputeTicks * msPerTick;
            double attnGateMs = _prefillAttnGateTicks * msPerTick;
            double attnOutputMs = _prefillAttnOutputTicks * msPerTick;
            double attnFfnMs = _prefillAttnFfnTicks * msPerTick;

            double recInputMs = _prefillRecInputProjTicks * msPerTick;
            double recCoreMs = _prefillRecCoreTicks * msPerTick;
            double recOutputMs = _prefillRecOutputTicks * msPerTick;
            double recFfnMs = _prefillRecFfnTicks * msPerTick;

            double total = embMs + attnMs + recMs + lmHeadMs;

            Console.WriteLine($"Prefill profile ({_prefillTokenCount} tokens, {total:F0} ms total):");
            Console.WriteLine($"  Embedding:                  {embMs,8:F0} ms ({100 * embMs / total,5:F1}%)");
            Console.WriteLine($"  Attention block (8 layers): {attnMs,8:F0} ms ({100 * attnMs / total,5:F1}%)");
            Console.WriteLine($"    QKV proj:                 {attnQkvMs,8:F0} ms");
            Console.WriteLine($"    Deinterleave Q/gate:      {attnDeintMs,8:F0} ms");
            Console.WriteLine($"    QK-norm:                  {attnQknMs,8:F0} ms");
            Console.WriteLine($"    RoPE (Q+K):               {attnRopeMs,8:F0} ms");
            Console.WriteLine($"    Reshape to heads:         {attnReshapeMs,8:F0} ms");
            Console.WriteLine($"    Cache copy (K,V):         {attnCacheCopyMs,8:F0} ms");
            Console.WriteLine($"    Expand KV heads:          {attnExpandKvMs,8:F0} ms");
            Console.WriteLine($"    Attention compute:        {attnComputeMs,8:F0} ms (QK^T + softmax + V)");
            Console.WriteLine($"    Sigmoid gate:             {attnGateMs,8:F0} ms");
            Console.WriteLine($"    Output proj:              {attnOutputMs,8:F0} ms");
            Console.WriteLine($"    FFN (norm+gate_up+down):  {attnFfnMs,8:F0} ms");
            Console.WriteLine($"  Recurrent block (24 layers):{recMs,8:F0} ms ({100 * recMs / total,5:F1}%)");
            Console.WriteLine($"    Input proj (norm+pack):   {recInputMs,8:F0} ms");
            Console.WriteLine($"    GDN core (conv+chunked):  {recCoreMs,8:F0} ms");
            Console.WriteLine($"    Output proj:              {recOutputMs,8:F0} ms");
            Console.WriteLine($"    FFN (norm+gate_up+down):  {recFfnMs,8:F0} ms");
            Console.WriteLine($"  Final norm + LM head:       {lmHeadMs,8:F0} ms ({100 * lmHeadMs / total,5:F1}%)");
        }

        // 中文：释放模型资源：视觉编码器与挂起视觉嵌入、各层 KV/MLX 注意力缓存、GDN 状态、所有 MoE 与注意力解码 scratch 张量，最后调用基类释放。
        public override void Dispose()
        {
            VisionEncoder?.Dispose();
            foreach (var (visionEmbeddings, _) in _visionEmbeddingsList)
                visionEmbeddings?.Dispose();
            _visionEmbeddingsList.Clear();

            if (_kvCacheK != null)
                foreach (var t in _kvCacheK) t?.Dispose();
            if (_kvCacheV != null)
                foreach (var t in _kvCacheV) t?.Dispose();
            if (_mlxAttentionCache != null)
                foreach (var cache in _mlxAttentionCache) cache?.Dispose();

            DisposeGdnState();

            _moeTokenInput?.Dispose();
            _moeGateBuf?.Dispose();
            _moeUpBuf?.Dispose();
            _moeDownBuf?.Dispose();
            _moeBatchedResult?.Dispose();
            _moeBatchedGate?.Dispose();
            _moeBatchedUp?.Dispose();
            _moeBatchedDown?.Dispose();
            _moeBatchedExpertIndices?.Dispose();
            _moeBatchedRouteWeights?.Dispose();

            _attnDecodeQBuf?.Dispose();
            _attnDecodeGBuf?.Dispose();
            _attnDecodeOutBuf?.Dispose();
            _attnDecodeQkvBuf?.Dispose();
            _ffnDecodeGateUpBuf?.Dispose();

            base.Dispose();
        }
    }
}
