// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

// ──────【文件说明】──────
// 文件：KvBlockTransfer.cs
// 用途：为分页 KV 缓存（Paged KV Cache）提供跨后端的字节级块提取与注入操作。
//       支持 F32、F16、Q8_0（GGUF 对称量化）等多种存储格式，
//       通过直接操作主机内存指针并逐行 memcpy 实现高效的 token 范围迁移。
// 主要类型：KvBlockTransfer（internal static class）
// 参考：分页注意力机制 (Paged Attention): https://arxiv.org/abs/2309.06180
//       Q8_0 对称量化格式: 参考 GGUF 规范
// ────────────────────────

using System;
using System.Collections.Generic;
using TensorSharp;

namespace TensorSharp.Models
{
    /// <summary>
    /// Backend-portable byte serialization helper for per-layer K/V cache tensors of
    /// shape <c>(numKVHeads, capacity, headDim)</c>. Used by the paged KV cache
    /// machinery to capture and restore block-aligned token ranges across sessions.
    ///
    /// We deliberately do NOT route through <see cref="Ops.Copy"/> because the GGML
    /// op registry only services F32 tensors, while the KV cache routinely lives in
    /// F16 or Q8_0 storage. Instead we drain pending async compute via
    /// <see cref="Storage.EnsureHostReadable"/>, take the storage's base host
    /// pointer (every backend in the repo - GGML, CUDA, MLX, CPU - exposes a host
    /// mirror through <see cref="Storage.PtrAtElement"/> at index 0) and then walk
    /// the (head, position) rows with a per-row <c>memcpy</c>. For Q8_0 we use the
    /// block-bytes-per-row figure derived from <see cref="Storage.ByteLength"/>
    /// instead of <c>ElementType.Size()</c>, which is the only safe way to address
    /// Q8_0 at offsets other than zero.
    /// </summary>
    internal static class KvBlockTransfer
    {
        /// <summary>
        /// Total bytes required for one block of <paramref name="tokenCount"/> tokens
        /// across every <c>(K, V)</c> pair in <paramref name="kCache"/> /
        /// <paramref name="vCache"/>. Storage objects shared across layers (e.g.
        /// Gemma 4's donor/alias map) are counted exactly once - that's what makes
        /// it safe to extract/inject without double-writing the same bytes.
        /// </summary>
        // 中文：计算指定 token 数量对应的 KV 缓存块所需总字节数，共享存储只计一次
        public static long ComputeBlockByteSize(Tensor[] kCache, Tensor[] vCache, int tokenCount)
        {
            if (kCache == null || vCache == null || kCache.Length != vCache.Length || tokenCount <= 0)
                return 0;

            long total = 0;
            foreach (var t in EnumerateUniqueStorageTensors(kCache, vCache))
                total += PerLayerBlockBytes(t, tokenCount);
            return total;
        }

        // 中文：将指定 token 范围内的 KV 缓存数据提取（序列化）到目标字节缓冲区
        public static bool Extract(
            IAllocator allocator,
            Tensor[] kCache,
            Tensor[] vCache,
            int currentSeqLen,
            int startToken,
            int tokenCount,
            Span<byte> destination)
        {
            if (!ValidateArgs(kCache, vCache, tokenCount))
                return false;
            if (startToken < 0 || startToken + tokenCount > currentSeqLen)
                return false;
            long expected = ComputeBlockByteSize(kCache, vCache, tokenCount);
            if (destination.Length != expected)
                return false;

            int offset = 0;
            foreach (var t in EnumerateUniqueStorageTensors(kCache, vCache))
            {
                if (!CopyOneOut(t, startToken, tokenCount, destination[offset..], out int written))
                    return false;
                offset += written;
            }
            return true;
        }

        // 中文：将字节缓冲区中的 KV 缓存数据注入（反序列化）到指定目标 token 位置，仅允许顺序追加
        public static bool Inject(
            IAllocator allocator,
            Tensor[] kCache,
            Tensor[] vCache,
            int currentSeqLen,
            int destToken,
            int tokenCount,
            ReadOnlySpan<byte> source)
        {
            if (!ValidateArgs(kCache, vCache, tokenCount))
                return false;
            // We only permit appending in order. Higher-level code resets the cache
            // before restoration and accumulates blocks from position 0 onward.
            if (destToken != currentSeqLen)
                return false;
            long expected = ComputeBlockByteSize(kCache, vCache, tokenCount);
            if (source.Length != expected)
                return false;

            int offset = 0;
            foreach (var t in EnumerateUniqueStorageTensors(kCache, vCache))
            {
                long capacity = t.Sizes[1];
                if (destToken + tokenCount > capacity)
                    return false;
                if (!CopyOneIn(t, destToken, tokenCount, source[offset..], out int read))
                    return false;
                offset += read;
            }
            return true;
        }

        /// <summary>
        /// Walk K[0], V[0], K[1], V[1], ... but yield each unique <see cref="Storage"/>
        /// at most once. The deterministic order matters because the captured byte
        /// blob is laid out in the same order the enumerator visits tensors; flipping
        /// the order would silently corrupt restoration.
        /// </summary>
        // 中文：按 K[0],V[0],K[1],V[1]... 顺序遍历所有层，去重后逐一返回唯一 Storage 对应的张量
        private static IEnumerable<Tensor> EnumerateUniqueStorageTensors(Tensor[] kCache, Tensor[] vCache)
        {
            var seen = new HashSet<Storage>(ReferenceEqualityComparer.Instance);
            for (int l = 0; l < kCache.Length; l++)
            {
                Tensor k = kCache[l];
                if (k != null && seen.Add(k.Storage))
                    yield return k;
                Tensor v = vCache[l];
                if (v != null && seen.Add(v.Storage))
                    yield return v;
            }
        }

        // 中文：验证 KV 缓存张量数组的合法性，包括非空、维度、偏移量及跨层数据类型一致性检查
        private static bool ValidateArgs(Tensor[] kCache, Tensor[] vCache, int tokenCount)
        {
            if (kCache == null || vCache == null || kCache.Length == 0)
                return false;
            if (kCache.Length != vCache.Length)
                return false;
            if (tokenCount <= 0)
                return false;
            DType firstDtype = DType.Float32;
            bool first = true;
            for (int l = 0; l < kCache.Length; l++)
            {
                if (kCache[l] == null || vCache[l] == null)
                    return false;
                if (kCache[l].Sizes.Length != 3 || vCache[l].Sizes.Length != 3)
                    return false;
                if (kCache[l].StorageOffset != 0 || vCache[l].StorageOffset != 0)
                    return false;
                if (kCache[l].Storage.ElementType != vCache[l].Storage.ElementType)
                    return false;
                if (first)
                {
                    firstDtype = kCache[l].Storage.ElementType;
                    first = false;
                }
                else if (kCache[l].Storage.ElementType != firstDtype)
                {
                    return false; // mixed dtypes across layers would invalidate the fingerprint
                }
            }
            return true;
        }

        // 中文：计算单个张量在指定 tokenCount 下的块字节数，适配 Q8_0 等非标准量化格式的行步长
        private static long PerLayerBlockBytes(Tensor tensor, int tokenCount)
        {
            // Use the actual on-storage row stride - this gracefully handles Q8_0
            // (where headDim/32 * 34 bytes per row != headDim * 1 byte/elem).
            return RowBytes(tensor) * tensor.Sizes[0] * tokenCount;
        }

        // 中文：从存储总字节数和张量尺寸推导每个 (head, position) 行的字节数，兼容 F32/F16/Q8_0
        private static long RowBytes(Tensor tensor)
        {
            // Bytes per (head, position) row, derived from the storage byte length
            // and the tensor extents. Works for F32 / F16 / Q8_0 alike provided the
            // tensor is contiguous at offset 0 (validated upstream).
            long numKVHeads = tensor.Sizes[0];
            long capacity = tensor.Sizes[1];
            return tensor.Storage.ByteLength / (numKVHeads * capacity);
        }

        // 中文：将单个缓存张量中指定 token 范围的数据按头逐行 memcpy 拷贝到目标缓冲区（提取方向）
        private static bool CopyOneOut(Tensor cacheTensor, int startToken, int tokenCount, Span<byte> destination, out int bytesWritten)
        {
            cacheTensor.Storage.EnsureHostReadable();

            long numKVHeads = cacheTensor.Sizes[0];
            long capacity = cacheTensor.Sizes[1];
            long rowBytes = RowBytes(cacheTensor);
            long blockBytes = numKVHeads * tokenCount * rowBytes;
            if (destination.Length < blockBytes)
            {
                bytesWritten = 0;
                return false;
            }

            IntPtr storageBase = cacheTensor.Storage.PtrAtElement(0);
            if (storageBase == IntPtr.Zero)
            {
                bytesWritten = 0;
                return false;
            }

            unsafe
            {
                byte* src = (byte*)storageBase;
                fixed (byte* dstBase = destination)
                {
                    long perHead = tokenCount * rowBytes;
                    for (long h = 0; h < numKVHeads; h++)
                    {
                        long srcOffset = (h * capacity + startToken) * rowBytes;
                        long dstOffset = h * perHead;
                        Buffer.MemoryCopy(
                            src + srcOffset,
                            dstBase + dstOffset,
                            destination.Length - dstOffset,
                            perHead);
                    }
                }
            }

            bytesWritten = (int)blockBytes;
            return true;
        }

        // 中文：将源缓冲区中的数据按头逐行 memcpy 写入单个缓存张量的指定目标 token 位置（注入方向）
        private static bool CopyOneIn(Tensor cacheTensor, int destToken, int tokenCount, ReadOnlySpan<byte> source, out int bytesRead)
        {
            cacheTensor.Storage.EnsureHostReadable();

            long numKVHeads = cacheTensor.Sizes[0];
            long capacity = cacheTensor.Sizes[1];
            long rowBytes = RowBytes(cacheTensor);
            long blockBytes = numKVHeads * tokenCount * rowBytes;
            if (source.Length < blockBytes)
            {
                bytesRead = 0;
                return false;
            }

            IntPtr storageBase = cacheTensor.Storage.PtrAtElement(0);
            if (storageBase == IntPtr.Zero)
            {
                bytesRead = 0;
                return false;
            }

            unsafe
            {
                byte* dst = (byte*)storageBase;
                fixed (byte* srcBase = source)
                {
                    long perHead = tokenCount * rowBytes;
                    for (long h = 0; h < numKVHeads; h++)
                    {
                        long dstOffset = (h * capacity + destToken) * rowBytes;
                        long srcOffset = h * perHead;
                        Buffer.MemoryCopy(
                            srcBase + srcOffset,
                            dst + dstOffset,
                            cacheTensor.Storage.ByteLength - dstOffset,
                            perHead);
                    }
                }
            }

            bytesRead = (int)blockBytes;
            return true;
        }
    }
}
