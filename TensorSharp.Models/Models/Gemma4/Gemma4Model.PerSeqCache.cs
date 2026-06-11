// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Per-request KV-cache holders for the per-sequence fused-decode path.
//
// Problem this solves: with N>=2 concurrent requests the continuous-batching
// engine routed every step through the op-by-op batched paged forward
// (ForwardBatchCore). Each of that path's ~20 Ops.* dispatches per layer does
// a host round-trip + ggml_backend_synchronize on the Metal backend (activation
// tensors bind device-local, so the per-op sync cannot be deferred even with
// async compute). The result: hundreds of Metal queue drains per token, the GPU
// sits idle between dispatches (measured ~30% utilisation), and aggregate
// throughput at N=2 falls BELOW the single-stream rate.
//
// Fix: give each in-flight request its OWN set of KV-cache tensors and switch
// the model between them with a cheap pointer swap (no byte-level extract/inject,
// no sliding-window-cache wrap corruption). The engine then runs each scheduled
// sequence through the proven single-graph fused Forward (NativeGemma4ModelDecode
// for decode), which keeps the GPU saturated — one fused decode graph per token
// per sequence instead of ~840 tiny serialized dispatches for the whole batch.
//
// The single-request (N==1) path is untouched: it keeps using the model's
// primary cache and the engine's live-cache continuation / prefix-cache reuse.
// RestorePrimaryCache() reinstates the primary cache before any N==1 step that
// follows a multi-sequence (fused) episode.
using System;
using System.Collections.Generic;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    public partial class Gemma4Model
    {
        private sealed class Gemma4KvCacheHolder
        {
            public Tensor[] K;
            public Tensor[] V;
            public int[] Sizes;
            public int GlobalCapacity;
            public int SeqLen;
            public bool HostDirty;
        }

        // Per-request fused-decode cache holders, keyed by RequestId.
        private Dictionary<string, Gemma4KvCacheHolder> _fusedHolders;
        // RequestId whose holder is currently checked out into the active
        // _kvCacheK/_kvCacheV fields, or null when the primary cache is active.
        private string _activeFusedKey;
        // Snapshot of the primary cache, saved while a fused holder is checked
        // out so RestorePrimaryCache() can reinstate it for the N==1 path.
        private Gemma4KvCacheHolder _primaryHolder;

        /// <summary>The per-sequence fused forward is the path the engine
        /// dispatches for concurrent (N&gt;=2) requests: each request decodes
        /// through its own KV-cache holder (swapped in with a cheap pointer
        /// flip) instead of the round-robin per-step KV extract/inject swap.
        /// It is wired up for any GGML-backed Gemma 4 whose single-token
        /// <c>Forward</c> runs as a (near-)single GPU graph:
        ///   * dense models use the model-wide fused decode kernel
        ///     (<c>NativeGemma4ModelDecode</c>, gated by
        ///     <c>_canUseFusedFullModelDecode</c>);
        ///   * MoE models (<c>gemma-4-26B-A4B</c> etc.) can't use that
        ///     model-wide kernel, but their per-layer fused MoE-decode kernel
        ///     (<c>TryFusedMoELayerDecode</c>) plus the fused per-layer kernel
        ///     for the dense majority keeps each token's <c>Forward</c> down to
        ///     ~one dispatch per layer — far fewer than the op-by-op batched
        ///     paged path, and crucially the batched path can't run MoE at all
        ///     (<c>ForwardBatch</c> throws on MoE layers), so without this the
        ///     engine falls back to the serial KV-swap path and concurrent
        ///     requests decode round-robin.
        /// Both write to the active <c>_kvCacheK</c>/<c>_kvCacheV</c>, which the
        /// per-request holders swap (with <see cref="RefreshDecodeArraysKvCache"/>
        /// repointing the fused-decode pointer arrays), so MoE and dense share
        /// the exact same per-request-cache machinery below.</summary>
        // 中文：是否支持按序列融合前向（并发 N>=2 时每请求用各自 KV 缓存做单图融合解码）；要求 GGML 后端且支持全模型融合解码或为 MoE 模型。
        public bool SupportsPerSequenceFusedForward =>
            IsGgmlBackend && (_canUseFusedFullModelDecode || _numExperts > 0);

        // 中文：判断指定 requestId 是否已存在对应的按序列融合 KV 缓存 holder。
        public bool HasFusedSequenceCache(string requestId)
            => requestId != null && _fusedHolders != null && _fusedHolders.ContainsKey(requestId);

        // 中文：把当前活动 KV 缓存字段（K/V 张量、大小、容量、序列长度、脏标志）快照打包成一个 holder 返回（零拷贝，仅引用）。
        private Gemma4KvCacheHolder SnapshotActiveCache() => new Gemma4KvCacheHolder
        {
            K = _kvCacheK,
            V = _kvCacheV,
            Sizes = _kvCacheSize,
            GlobalCapacity = _kvCacheGlobalCapacity,
            SeqLen = _cacheSeqLen,
            HostDirty = _kvCacheHostDirty,
        };

        // 中文：将给定 holder 的 K/V 等状态装载回模型活动缓存字段，并刷新融合解码内核所用的 K/V 指针数组。
        private void LoadCacheHolder(Gemma4KvCacheHolder h)
        {
            _kvCacheK = h.K;
            _kvCacheV = h.V;
            _kvCacheSize = h.Sizes;
            _kvCacheGlobalCapacity = h.GlobalCapacity;
            _cacheSeqLen = h.SeqLen;
            _kvCacheHostDirty = h.HostDirty;
            // The fused-decode kernels read raw K/V cache pointers cached in
            // _decodeArrays; repoint them at the just-bound holder's tensors.
            RefreshDecodeArraysKvCache();
        }

        // 中文：分配一套全新的空 KV 缓存张量并包装为 holder（SeqLen=0），用于首次见到某请求时创建其专属缓存。
        private Gemma4KvCacheHolder CreateFreshHolder()
        {
            AllocateKvCacheArrays(_initialGlobalCacheLength,
                out var k, out var v, out var sizes, out _);
            return new Gemma4KvCacheHolder
            {
                K = k,
                V = v,
                Sizes = sizes,
                GlobalCapacity = _initialGlobalCacheLength,
                SeqLen = 0,
                HostDirty = false,
            };
        }

        /// <summary>Make <paramref name="requestId"/>'s KV cache the model's
        /// active cache, creating an empty one the first time the request is
        /// seen. Cheap: just swaps tensor-array references and refreshes the
        /// fused-decode pointer arrays. Returns true when the cache was freshly
        /// created, so the caller knows to inject any prefix-cache-reused prefix
        /// (NumComputedTokens &gt; 0 at admission) before the first Forward.</summary>
        // 中文：将指定请求的 KV 缓存切换为模型活动缓存（首次出现则新建空缓存）；先快照保存当前活动缓存再装载目标 holder，返回是否为新建（提示调用方注入前缀缓存）。
        public bool BindSequenceCache(string requestId)
        {
            if (string.IsNullOrEmpty(requestId))
                throw new ArgumentException("RequestId required", nameof(requestId));
            _fusedHolders ??= new Dictionary<string, Gemma4KvCacheHolder>(StringComparer.Ordinal);

            if (string.Equals(_activeFusedKey, requestId, StringComparison.Ordinal))
                return false; // already active

            // Save whatever cache is currently checked out so its (possibly
            // grown) tensors aren't lost when we repoint the active fields.
            if (_activeFusedKey == null)
                _primaryHolder = SnapshotActiveCache();
            else
                _fusedHolders[_activeFusedKey] = SnapshotActiveCache();

            bool fresh;
            if (_fusedHolders.TryGetValue(requestId, out var holder))
            {
                fresh = false;
            }
            else
            {
                holder = CreateFreshHolder();
                _fusedHolders[requestId] = holder;
                fresh = true;
            }
            LoadCacheHolder(holder);
            _activeFusedKey = requestId;
            return fresh;
        }

        /// <summary>Transition the single in-flight N==1 owner (whose live state
        /// is in the primary cache) into the fused path without copying any KV
        /// bytes: hand the live primary arrays to the owner's holder and give the
        /// primary a fresh empty allocation for later N==1 use. Called by the
        /// executor on the first multi-sequence step when a prior owner exists,
        /// so that owner's history is preserved as its own per-request cache.</summary>
        // 中文：零拷贝地把当前 N==1 主缓存（含 owner 活动 K/V）收编为该请求的融合 holder，并给主缓存重新分配一套空张量供后续 N==1 使用；仅在主缓存当前活动时生效。
        public void AdoptPrimaryCacheToFused(string requestId)
        {
            if (string.IsNullOrEmpty(requestId)) return;
            _fusedHolders ??= new Dictionary<string, Gemma4KvCacheHolder>(StringComparer.Ordinal);

            // Only meaningful when the primary cache is the one currently active
            // (i.e. the N==1 owner ran most recently). If a fused holder is
            // already checked out there is nothing to adopt.
            if (_activeFusedKey != null)
                return;
            if (_fusedHolders.ContainsKey(requestId))
                return;

            // The active fields hold the primary cache with the owner's live K/V.
            // Move those arrays into the owner's holder (zero copy).
            var holder = SnapshotActiveCache();
            _fusedHolders[requestId] = holder;
            _activeFusedKey = requestId; // owner's holder is now checked out

            // Give the primary a fresh empty allocation so a future N==1 step for
            // a never-fused request doesn't reset the adopted holder's tensors.
            AllocateKvCacheArrays(_initialGlobalCacheLength,
                out var k, out var v, out var sizes, out _);
            _primaryHolder = new Gemma4KvCacheHolder
            {
                K = k,
                V = v,
                Sizes = sizes,
                GlobalCapacity = _initialGlobalCacheLength,
                SeqLen = 0,
                HostDirty = false,
            };
        }

        /// <summary>Reinstate the primary cache as the model's active cache.
        /// Invoked by the executor before an N==1 step that follows a fused
        /// episode so the legacy single-sequence path (which resets/injects the
        /// active cache in place) never clobbers a concurrent request's holder.
        /// No-op when the primary cache is already active.</summary>
        // 中文：在 N==1 步骤前把主缓存重新设为活动缓存；先保存当前检出的融合 holder 再换回主缓存快照，避免单序列路径就地重置时破坏并发请求的 holder。主缓存已活动时为空操作。
        public void RestorePrimaryCache()
        {
            if (_activeFusedKey == null)
                return;
            // Save the checked-out fused holder, then swap the primary back in.
            _fusedHolders[_activeFusedKey] = SnapshotActiveCache();
            _activeFusedKey = null;
            if (_primaryHolder != null)
            {
                LoadCacheHolder(_primaryHolder);
                _primaryHolder = null;
            }
        }

        /// <summary>Release a finished/aborted request's per-request cache. The
        /// engine calls this from InferenceEngine when a sequence leaves the
        /// scheduler. Frees the holder's tensors (after restoring the primary if
        /// the released holder happened to be the active one).</summary>
        // 中文：释放已完成/中止请求的按序列缓存；若该 holder 正被检出则先换回主缓存避免活动字段悬空，然后从字典移除并销毁其张量。
        public void OnSequenceReleased(string requestId)
        {
            if (_fusedHolders == null || string.IsNullOrEmpty(requestId))
                return;
            if (!_fusedHolders.TryGetValue(requestId, out var holder))
                return;

            if (string.Equals(_activeFusedKey, requestId, StringComparison.Ordinal))
            {
                // The released sequence's cache is the one currently checked out.
                // Swap the primary back in so the active fields don't dangle.
                _activeFusedKey = null;
                if (_primaryHolder != null)
                {
                    LoadCacheHolder(_primaryHolder);
                    _primaryHolder = null;
                }
            }

            _fusedHolders.Remove(requestId);
            DisposeHolder(holder);
        }

        // 中文：销毁 holder 的逐层 K/V 张量；用 HashSet 去重避免重复释放，并跳过别名其它层的 donor 层。
        private void DisposeHolder(Gemma4KvCacheHolder holder)
        {
            if (holder?.K == null) return;
            var disposed = new HashSet<Tensor>();
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_kvDonorMap.ContainsKey(l)) continue; // donor layers alias another layer
                if (holder.K[l] != null && disposed.Add(holder.K[l])) holder.K[l].Dispose();
                if (holder.V != null && holder.V[l] != null && disposed.Add(holder.V[l])) holder.V[l].Dispose();
            }
        }

        /// <summary>Free every per-request fused cache holder (and the saved
        /// primary snapshot). Called on model dispose. Does not touch the
        /// currently-active arrays (those are the model's _kvCacheK, disposed by
        /// the normal cache teardown).</summary>
        // 中文：模型销毁时释放所有按序列融合 holder 及主缓存快照；跳过与活动 _kvCacheK 共享数组的 holder（由主缓存拆解负责释放），最后清空字典与活动键。
        private void DisposeAllFusedHolders()
        {
            if (_fusedHolders != null)
            {
                foreach (var kv in _fusedHolders)
                {
                    // Skip the active holder; its arrays are _kvCacheK and are
                    // disposed by the model's main cache teardown.
                    if (string.Equals(kv.Key, _activeFusedKey, StringComparison.Ordinal))
                        continue;
                    DisposeHolder(kv.Value);
                }
                _fusedHolders.Clear();
                _fusedHolders = null;
            }
            if (_primaryHolder != null)
            {
                // If a fused holder is active, the primary snapshot owns distinct
                // arrays that must be freed; if the primary is active it shares
                // _kvCacheK and is freed by the main teardown.
                if (_activeFusedKey != null)
                    DisposeHolder(_primaryHolder);
                _primaryHolder = null;
            }
            _activeFusedKey = null;
        }
    }
}
