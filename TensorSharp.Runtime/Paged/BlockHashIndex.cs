// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Collections.Generic;

namespace TensorSharp.Runtime.Paged
{
    /// <summary>
    /// Maps content hashes (<see cref="KvBlockHash"/>) to physical <see cref="KvBlock"/>s.
    /// When a sequence's prompt prefix produces a hash already in this index, the
    /// sequence can adopt the existing block (incrementing its ref count) instead
    /// of recomputing its K/V. Pure analogue of vLLM's
    /// <c>cached_block_hash_to_block</c>.
    ///
    /// At most one block per hash is stored. If a second block with the same
    /// content arrives (rare, can happen when two sequences race to the same
    /// prefix) the index keeps the first - the redundant block is left
    /// un-indexed and will simply be reclaimed when its sequence finishes.
    /// </summary>
    public sealed class BlockHashIndex
    {
        private readonly Dictionary<KvBlockHash, KvBlock> _index = new();

        public int Count => _index.Count;

        // 中文：将内容哈希到物理块的映射注册到索引（同一哈希只保留首个块）。
        public void Register(KvBlockHash hash, KvBlock block)
        {
            _index.TryAdd(hash, block);
        }

        // 中文：按内容哈希查找已索引的物理块（前缀缓存命中），未命中返回 false。
        public bool TryGet(KvBlockHash hash, out KvBlock block)
        {
            return _index.TryGetValue(hash, out block);
        }

        // 中文：从索引中移除指定哈希（块被驱逐或复用时调用）。
        public void Unregister(KvBlockHash hash)
        {
            _index.Remove(hash);
        }

        // 中文：清空整个哈希索引。
        public void Clear()
        {
            _index.Clear();
        }
    }
}
