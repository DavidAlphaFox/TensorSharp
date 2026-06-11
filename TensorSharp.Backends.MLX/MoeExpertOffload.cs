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
using System.Runtime.InteropServices;

namespace TensorSharp.MLX
{
    /// <summary>
    /// Per-process configuration for MoE expert weight offload to SSD-backed mmap.
    /// Enable with <c>TS_MLX_EXPERT_OFFLOAD_MB=&lt;n&gt;</c>. When set to a positive
    /// value:
    ///   - <see cref="ModelBase.PrepareMlxQuantizedWeightsForInference"/> SKIPS the
    ///     eager device upload for tensors whose name matches <c>*_exps.*</c> and
    ///     for the stacked-experts views populated alongside them; their host data
    ///     (the GGUF mmap pointer) is kept alive so MLX can lazily upload an
    ///     expert when it is first routed in a matmul call.
    ///   - <see cref="MlxQuantizedOps"/> maintains an LRU over those "offloadable"
    ///     cache entries; when their resident byte total exceeds the configured
    ///     ceiling, the oldest entries are evicted (their MLX arrays are freed via
    ///     the FIFO-ordered worker, so any kernel currently using them completes
    ///     before the free runs).
    /// Non-expert weights (attention/embedding/lm_head) are NEVER offloaded — they
    /// are small in aggregate, hot on every forward, and remain permanently
    /// device-resident under the existing preload path.
    /// </summary>
    public static class MoeExpertOffload
    {
        private const string EnvVarMb = "TS_MLX_EXPERT_OFFLOAD_MB";
        private static readonly long _maxCacheBytes = ParseLimit();
        private static readonly HashSet<IntPtr> _offloadableKeys = new();
        private static readonly object _sync = new();

        /// <summary>True when <c>TS_MLX_EXPERT_OFFLOAD_MB</c> is set to a positive value.</summary>
        public static bool IsEnabled => _maxCacheBytes > 0;

        /// <summary>
        /// Maximum total raw-bytes of MLX-resident offloadable (expert) weights
        /// before LRU eviction kicks in. 0 disables the mechanism entirely; any
        /// positive value enables offload AND caps cached expert bytes at that
        /// value. The metric uses each weight's <c>RawBytes</c> (GGUF block byte
        /// count) as a proxy for MLX-side residency.
        /// </summary>
        public static long MaxCacheBytes => _maxCacheBytes;

        /// <summary>
        /// Per the GGUF / llama.cpp naming convention, all MoE expert weight
        /// tensors carry an <c>_exps.</c> infix (e.g. <c>blk.0.ffn_gate_exps.5.weight</c>
        /// and the stacked <c>blk.0.ffn_gate_exps.weight</c>). Non-expert
        /// weights never carry that infix.
        /// </summary>
        // 中文：依据 GGUF 命名约定，判断权重名是否含 "_exps." 中缀，即是否为 MoE 专家权重。
        public static bool IsExpertWeightName(string name)
            => !string.IsNullOrEmpty(name) && name.IndexOf("_exps.", StringComparison.Ordinal) >= 0;

        /// <summary>
        /// Register a cache key as an eligible offload target. Called once per
        /// expert weight (and once per stacked-experts view) during model
        /// preparation. Subsequent <see cref="MlxQuantizedOps"/> cache lookups
        /// keyed by this pointer participate in the LRU and may be evicted; all
        /// other entries are pinned in the cache forever.
        /// </summary>
        // 中文：将缓存键登记为可卸载目标，使其参与 LRU 并可被淘汰（线程安全）。
        public static void RegisterOffloadable(IntPtr cacheKey)
        {
            if (cacheKey == IntPtr.Zero)
                return;
            lock (_sync)
            {
                _offloadableKeys.Add(cacheKey);
            }
        }

        // 中文：判断给定缓存键是否已登记为可卸载（线程安全）。
        public static bool IsOffloadable(IntPtr cacheKey)
        {
            if (cacheKey == IntPtr.Zero)
                return false;
            lock (_sync)
            {
                return _offloadableKeys.Contains(cacheKey);
            }
        }

        // 中文：清空所有已登记的可卸载缓存键（线程安全）。
        public static void Clear()
        {
            lock (_sync)
            {
                _offloadableKeys.Clear();
            }
        }

        /// <summary>
        /// Hint to the OS that the given file-backed mmap region is no longer
        /// needed. Used by the MLX offload path to actively release expert
        /// weight pages: the baseline (non-offload) preload path achieves this
        /// implicitly by calling <c>ReleaseHostData</c> after upload (which
        /// drops Metal's claim on the buffer and madvises DONTNEED), but in
        /// offload mode we keep the host pointer alive for re-upload — so we
        /// have to call this explicitly on registration and on cache eviction
        /// to match baseline's eviction behaviour. No-op on Windows.
        /// The range is rounded outward to whole page boundaries (16 KB on
        /// Apple Silicon); for GGUF tensors aligned on 32-byte block
        /// boundaries the rounding may overlap adjacent tensors, which is
        /// fine — they're also file-backed and will page back in on next
        /// touch.
        /// </summary>
        // 中文：将文件映射区域按页边界对齐后 madvise(DONTNEED)，提示操作系统释放专家权重页（Windows 上为空操作）。
        public static unsafe void AdvisePagesNotNeeded(IntPtr data, long byteCount)
        {
            if (data == IntPtr.Zero || byteCount <= 0)
                return;
            if (!OperatingSystem.IsMacOS() && !OperatingSystem.IsLinux())
                return;

            long pageSize = Environment.SystemPageSize;
            long address = data.ToInt64();
            long pageMask = ~(pageSize - 1);
            long alignedAddress = address & pageMask;
            long prefixBytes = address - alignedAddress;
            ulong length = checked((ulong)(byteCount + prefixBytes));
            ulong roundedLength = (length + (ulong)pageSize - 1) & ~((ulong)pageSize - 1);

            try
            {
                _ = madvise((void*)alignedAddress, (nuint)roundedLength, MadvDontNeed);
            }
            catch (DllNotFoundException)
            {
            }
            catch (EntryPointNotFoundException)
            {
            }
        }

        private const int MadvDontNeed = 4;

        // 中文：P/Invoke 绑定到 libc 的 madvise，用于向内核传递内存使用建议。
        [DllImport("libc", SetLastError = true, EntryPoint = "madvise")]
        private static extern unsafe int madvise(void* addr, nuint len, int advice);

        // 中文：解析环境变量 TS_MLX_EXPERT_OFFLOAD_MB，将正整数 MB 值换算为字节上限，非法或非正值返回 0（禁用）。
        private static long ParseLimit()
        {
            string value = Environment.GetEnvironmentVariable(EnvVarMb);
            if (!long.TryParse(value, out long mb) || mb <= 0)
                return 0;
            return mb * 1024L * 1024L;
        }
    }
}
