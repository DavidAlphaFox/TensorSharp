// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Runtime.CompilerServices;

namespace TensorSharp.Runtime.Paged
{
    /// <summary>
    /// Physical byte storage for the paged KV pool: <c>numBlocks</c> slabs of
    /// <c>blockByteSize</c> bytes each. Indexed by <see cref="KvBlock.Id"/>.
    ///
    /// The block layout is whatever the model's <c>TryExtractKVBlock</c>
    /// produces - this class just owns the bytes and a sticky "dirty" flag per
    /// slot. Storage is in managed memory; for the GGML/CUDA path the bytes are
    /// shuttled into device-resident KV tensors by the model layer at
    /// inject time.
    /// </summary>
    public sealed class PagedKvStorage : IDisposable
    {
        private readonly byte[][] _slabs;
        private readonly long _blockByteSize;
        private readonly int _numBlocks;
        private bool _disposed;

        // 中文：构造分页 KV 字节存储，记录块数与每块字节数，slab 数组延迟分配。
        public PagedKvStorage(int numBlocks, long blockByteSize)
        {
            if (numBlocks <= 0) throw new ArgumentOutOfRangeException(nameof(numBlocks));
            if (blockByteSize <= 0) throw new ArgumentOutOfRangeException(nameof(blockByteSize));

            _numBlocks = numBlocks;
            _blockByteSize = blockByteSize;
            _slabs = new byte[numBlocks][];
            // Lazy slab allocation: many configurations over-provision blocks for
            // worst-case context but only the active prefix actually gets used.
        }

        public int NumBlocks => _numBlocks;
        public long BlockByteSize => _blockByteSize;
        public long ReservedBytes => (long)_numBlocks * _blockByteSize;

        /// <summary>Get a writable span for block <paramref name="blockId"/>. Allocates
        /// the slab on first access. The returned span is exactly <see cref="BlockByteSize"/>
        /// long.</summary>
        // 中文：返回指定块的可写字节 span，首次访问时分配其 slab。
        public Span<byte> GetSpan(int blockId)
        {
            EnsureSlab(blockId);
            return _slabs[blockId].AsSpan();
        }

        /// <summary>Read-only view of block <paramref name="blockId"/>.</summary>
        // 中文：返回指定块的只读字节 span，首次访问时分配其 slab。
        public ReadOnlySpan<byte> GetReadOnlySpan(int blockId)
        {
            EnsureSlab(blockId);
            return _slabs[blockId].AsSpan();
        }

        /// <summary>Drop the slab for block <paramref name="blockId"/> back to the GC.
        /// Used when the block is being evicted from the cache and the bytes will
        /// not be reused.</summary>
        // 中文：将指定块的 slab 置空交还 GC（块被驱逐且字节不再复用时调用）。
        public void ReleaseSlab(int blockId)
        {
            if ((uint)blockId >= (uint)_numBlocks) return;
            _slabs[blockId] = null;
        }

        // 中文：校验块 id 范围并在 slab 尚未分配时按块字节数惰性分配。
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void EnsureSlab(int blockId)
        {
            if ((uint)blockId >= (uint)_numBlocks)
                throw new ArgumentOutOfRangeException(nameof(blockId), $"Block id {blockId} out of range [0,{_numBlocks}).");
            if (_slabs[blockId] == null)
                _slabs[blockId] = new byte[_blockByteSize];
        }

        // 中文：释放所有 slab 引用以交还 GC，幂等。
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            for (int i = 0; i < _slabs.Length; i++)
                _slabs[i] = null;
        }
    }
}
