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
using System.Diagnostics.CodeAnalysis;

// ───────────────────────────────────────────────────────────────────────────
// 【文件说明】经典（非分页）KV 缓存的状态记录器。
// 【主要类型】KVCache：记录模型 KV 张量当前持有的「规范 token 序列」及最近一次前向产生的
//             下一 token logits，是「模型正算到哪里」的唯一真相来源。它只在托管内存中保存
//             token 序列与 logits 缓冲，真正的 K/V 激活仍存放在模型各层张量里。
// ───────────────────────────────────────────────────────────────────────────
namespace TensorSharp.Runtime
{
    /// <summary>
    /// Tracks the canonical sequence of tokens currently held in the model's K/V tensors,
    /// plus the optional next-token logits produced by the most recent forward call.
    ///
    /// This object is the single source of truth for "what is the model in the middle of?".
    /// It mirrors the per-layer K/V tensor state inside the model so that the orchestrator
    /// can decide, for any new input prompt, how many leading tokens are already cached.
    ///
    /// Design invariants:
    ///   1. Every token in <see cref="Tokens"/> has been forwarded through the model exactly
    ///      once (in order). The model's internal `_cacheSeqLen` always equals
    ///      <see cref="Count"/>.
    ///   2. <see cref="NextLogits"/> is non-null iff a forward call recorded its output
    ///      logits via <see cref="RecordAppend"/> with <c>nextLogits</c> set, and no
    ///      subsequent <see cref="TruncateTo"/> / <see cref="Reset"/> has invalidated them.
    ///   3. The cache is owned by the orchestrator (not the model). Resetting / truncating
    ///      the cache must always be paired with the corresponding model call.
    ///
    /// The object itself stores ONLY the token sequence and logits buffer in managed memory.
    /// The actual K/V activations stay in the model's per-layer tensors which live in the
    /// model's allocator (CPU pinned memory for CPU/Metal backends, GPU device memory for
    /// the GGML CUDA backend - the allocator is selected at model construction time and the
    /// cache helper itself never moves K/V data between host and device).
    /// </summary>
    public sealed class KVCache
    {
        private readonly List<int> _tokens = new();
        private float[]? _nextLogits;

        /// <summary>The number of tokens currently held in the model's KV state.</summary>
        public int Count => _tokens.Count;

        /// <summary>Read-only view of the cached token sequence.</summary>
        public IReadOnlyList<int> Tokens => _tokens;

        /// <summary>
        /// Logits produced after the most recent token in <see cref="Tokens"/> was forwarded.
        /// Null if no logits were recorded for the current state (e.g. immediately after
        /// truncation, or if the most recent <see cref="RecordAppend"/> didn't supply them).
        /// </summary>
        public float[]? NextLogits => _nextLogits;

        /// <summary>True if the cache contains no tokens.</summary>
        public bool IsEmpty => _tokens.Count == 0;

        /// <summary>
        /// Length of the longest common prefix between the cached tokens and
        /// <paramref name="other"/>. Returns 0 for empty / null inputs. Returns
        /// <see cref="Count"/> when <paramref name="other"/> starts with the entire cache.
        /// </summary>
        // 中文：返回缓存 token 序列与 other 的最长公共前缀长度，用于判断可复用的缓存量。
        public int CommonPrefixLength(IReadOnlyList<int> other)
        {
            if (other == null || other.Count == 0 || _tokens.Count == 0)
                return 0;

            int max = Math.Min(_tokens.Count, other.Count);
            int i = 0;
            for (; i < max; i++)
            {
                if (_tokens[i] != other[i])
                    break;
            }
            return i;
        }

        /// <summary>
        /// True if the cached tokens are an exact prefix of <paramref name="input"/>
        /// (or equal to it). False when <paramref name="input"/> is shorter than the cache
        /// or when any token differs.
        /// </summary>
        // 中文：判断缓存的 token 序列是否恰为 input 的前缀（或与之相等）。
        public bool IsPrefixOf(IReadOnlyList<int> input)
        {
            if (input == null || input.Count < _tokens.Count)
                return false;

            for (int i = 0; i < _tokens.Count; i++)
            {
                if (_tokens[i] != input[i])
                    return false;
            }
            return true;
        }

        /// <summary>
        /// Returns true when the cached token sequence is identical to <paramref name="input"/>
        /// AND logits were recorded for the position right after it.
        /// In that case <paramref name="logits"/> receives a fresh copy of the cached logits.
        /// </summary>
        // 中文：当缓存序列与 input 完全一致且已记录其后续 logits 时，输出该 logits 的副本并返回 true。
        public bool TryGetExactMatchLogits(IReadOnlyList<int> input, [NotNullWhen(true)] out float[]? logits)
        {
            logits = null;
            if (_nextLogits == null || input == null || input.Count != _tokens.Count)
                return false;

            for (int i = 0; i < _tokens.Count; i++)
            {
                if (_tokens[i] != input[i])
                    return false;
            }

            logits = (float[])_nextLogits.Clone();
            return true;
        }

        /// <summary>
        /// Drop everything and return to the empty state. The caller is responsible for
        /// also calling <see cref="IModelArchitecture.ResetKVCache"/> on the model.
        /// </summary>
        // 中文：清空全部 token 与缓存 logits，使缓存回到空状态（模型侧需调用方另行重置）。
        public void Reset()
        {
            _tokens.Clear();
            _nextLogits = null;
        }

        /// <summary>
        /// Keep only the first <paramref name="length"/> tokens. Cached logits are
        /// invalidated. The caller is responsible for also calling
        /// <see cref="IModelArchitecture.TruncateKVCache"/> on the model.
        /// </summary>
        // 中文：仅保留前 length 个 token 并作废缓存 logits（模型侧 KV 缓存需调用方同步截断）。
        public void TruncateTo(int length)
        {
            if (length < 0)
                throw new ArgumentOutOfRangeException(nameof(length), "Truncation length cannot be negative.");
            if (length > _tokens.Count)
                throw new ArgumentOutOfRangeException(nameof(length), "Truncation length exceeds cached token count.");

            if (length < _tokens.Count)
            {
                _tokens.RemoveRange(length, _tokens.Count - length);
                _nextLogits = null;
            }
        }

        /// <summary>
        /// Record that the model has just forwarded <paramref name="newTokens"/> on top of
        /// the current cached prefix. Updates the cache to reflect what is now in the
        /// model's KV state. <paramref name="nextLogits"/> are the logits returned by the
        /// final forward call (logits for the next token to be generated). May be null when
        /// the caller does not need to cache them (e.g. mid-prefill chunks).
        /// </summary>
        // 中文：记录模型已在当前前缀之上前向了 newTokens，追加这些 token 并更新缓存的下一步 logits。
        public void RecordAppend(IReadOnlyList<int> newTokens, float[]? nextLogits)
        {
            if (newTokens == null)
                return;

            for (int i = 0; i < newTokens.Count; i++)
                _tokens.Add(newTokens[i]);

            _nextLogits = nextLogits;
        }

        /// <summary>
        /// Convenience overload for appending a single token (typical decode step).
        /// </summary>
        // 中文：追加单个 token 的便捷重载（典型 decode 步），并更新缓存的下一步 logits。
        public void RecordAppend(int token, float[]? nextLogits)
        {
            _tokens.Add(token);
            _nextLogits = nextLogits;
        }

        /// <summary>
        /// Plan the operations required to bring the model's KV state from its current
        /// contents to one that contains <paramref name="inputTokens"/> as a prefix
        /// (so that the next-token logits at position <c>inputTokens.Count</c> are available).
        ///
        /// The result describes what the orchestrator must do, but does not modify any
        /// state itself; the orchestrator is responsible for applying the plan to both the
        /// model and to this cache (via <see cref="TruncateTo"/> / <see cref="RecordAppend"/>).
        ///
        /// <paramref name="supportsTruncation"/> models that report <c>false</c> can only
        /// reuse the cache when the entire current cache is a prefix of the new input.
        /// </summary>
        // 中文：根据当前缓存与目标 inputTokens 规划复用方案（精确命中/部分复用/重置），仅生成计划不改状态。
        public ReusePlan PlanReuse(IReadOnlyList<int> inputTokens, bool supportsTruncation)
        {
            if (inputTokens == null || inputTokens.Count == 0)
                return ReusePlan.Reset(0);

            // Exact match: nothing to forward, hopefully cached logits are available.
            if (TryGetExactMatchLogits(inputTokens, out float[]? cachedLogits))
                return ReusePlan.ExactMatch(cachedLogits);

            int common = CommonPrefixLength(inputTokens);

            // For non-truncatable models (recurrent state): only reuse if the cache is a
            // prefix of the new input.
            if (!supportsTruncation && common < _tokens.Count)
                return ReusePlan.Reset(inputTokens.Count);

            // We always need at least one token in the forward to compute fresh logits for
            // the next step. If the input matches the cache for all but its last position,
            // back the prefix off by one to leave a token to forward.
            if (common == inputTokens.Count)
                common = Math.Max(0, inputTokens.Count - 1);

            int suffixLength = inputTokens.Count - common;

            if (common == 0)
                return ReusePlan.Reset(inputTokens.Count);

            return ReusePlan.Reuse(common, suffixLength);
        }
    }

    /// <summary>
    /// Outcome of <see cref="KVCache.PlanReuse"/>. Describes the work the orchestrator
    /// must do for the next forward call.
    /// </summary>
    public readonly struct ReusePlan
    {
        public ReusePlanKind Kind { get; }

        /// <summary>
        /// Number of tokens to keep from the existing KV cache (also the position at which
        /// the model should append the next forward). 0 when the cache is being reset.
        /// </summary>
        public int ReusedPrefixLength { get; }

        /// <summary>
        /// Number of new tokens to forward through the model on the next call.
        /// 0 when <see cref="Kind"/> is <see cref="ReusePlanKind.ExactMatch"/>.
        /// </summary>
        public int TokensToForward { get; }

        /// <summary>
        /// Pre-computed logits when <see cref="Kind"/> is <see cref="ReusePlanKind.ExactMatch"/>;
        /// otherwise null.
        /// </summary>
        public float[]? CachedLogits { get; }

        // 中文：私有构造，统一设置复用方案的种类、可复用前缀长度、待前向 token 数与缓存 logits。
        private ReusePlan(ReusePlanKind kind, int reusedPrefix, int tokensToForward, float[]? cachedLogits)
        {
            Kind = kind;
            ReusedPrefixLength = reusedPrefix;
            TokensToForward = tokensToForward;
            CachedLogits = cachedLogits;
        }

        // 中文：构造「精确命中」方案，无需前向，直接携带缓存的 logits。
        public static ReusePlan ExactMatch(float[]? cachedLogits)
            => new(ReusePlanKind.ExactMatch, 0, 0, cachedLogits);

        // 中文：构造「部分复用」方案，保留指定前缀并前向其余 token。
        public static ReusePlan Reuse(int reusedPrefix, int tokensToForward)
            => new(ReusePlanKind.PartialReuse, reusedPrefix, tokensToForward, null);

        // 中文：构造「重置」方案，清空缓存并从头前向全部输入 token。
        public static ReusePlan Reset(int tokensToForward)
            => new(ReusePlanKind.Reset, 0, tokensToForward, null);
    }

    public enum ReusePlanKind
    {
        /// <summary>
        /// The cached tokens already match the input exactly and the cached
        /// next-token logits are valid. No forward call is needed.
        /// </summary>
        ExactMatch,
        /// <summary>
        /// Truncate the model's KV cache to <see cref="ReusePlan.ReusedPrefixLength"/> and
        /// forward the next <see cref="ReusePlan.TokensToForward"/> tokens.
        /// </summary>
        PartialReuse,
        /// <summary>
        /// Reset the model's KV cache and forward all <see cref="ReusePlan.TokensToForward"/>
        /// input tokens from scratch.
        /// </summary>
        Reset,
    }
}
