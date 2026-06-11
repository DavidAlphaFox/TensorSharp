// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Runtime.Paged
{
    /// <summary>
    /// Metadata for a single physical block in the paged KV-cache pool. Mirrors
    /// vLLM's <c>KVCacheBlock</c>: a fixed-size slab of bytes that holds
    /// <c>blockSize</c> tokens worth of K/V state across all model layers,
    /// referenced by zero or more <see cref="SequenceState"/> instances.
    ///
    /// A block is in one of three states:
    ///   * Live and owned: <c>RefCount &gt; 0</c>. The block holds the K/V for some
    ///     active sequence(s). Cannot be evicted.
    ///   * Cached and free: <c>RefCount == 0 &amp;&amp; ContentHash.HasValue</c>. The block
    ///     is in the free queue (LRU end) and also indexed by content hash so a
    ///     future sequence with the same prefix can adopt it for free.
    ///   * Empty and free: <c>RefCount == 0 &amp;&amp; ContentHash == null</c>. The block
    ///     has never held content (or its hash was evicted). It is at the front of
    ///     the free queue and will be handed out first.
    ///
    /// Mutation is the <see cref="BlockPool"/>'s responsibility; this type is just
    /// the storage. Bytes for the actual K/V payload live in
    /// <see cref="PagedKvStorage"/> keyed by <see cref="Id"/>.
    /// </summary>
    public sealed class KvBlock
    {
        /// <summary>Stable physical id (0..numBlocks-1). Indexes into
        /// <see cref="PagedKvStorage"/> for the actual bytes.</summary>
        public int Id { get; }

        /// <summary>How many <see cref="SequenceState"/> objects currently reference
        /// this block via their <see cref="BlockTable"/>.</summary>
        public int RefCount { get; internal set; }

        /// <summary>When non-null this block is full (every slot occupied) and the
        /// hash uniquely identifies its content for prefix caching. When null the
        /// block is either being written into (not yet full) or has been evicted
        /// from the prefix cache.</summary>
        public KvBlockHash? ContentHash { get; internal set; }

        /// <summary>Number of valid tokens written into this block (0..blockSize).
        /// Only full blocks (Used==blockSize) can be hashed into the prefix cache.</summary>
        public int Used { get; internal set; }

        /// <summary>Doubly-linked-list pointers for the free queue. Maintained by
        /// <see cref="FreeBlockQueue"/>. When the block is allocated both pointers
        /// are null.</summary>
        internal KvBlock PrevFree;
        internal KvBlock NextFree;

        // 中文：以稳定物理 id 构造空块，引用计数与已用槽位归零、无内容哈希。
        public KvBlock(int id)
        {
            Id = id;
            RefCount = 0;
            ContentHash = null;
            Used = 0;
        }

        /// <summary>True when the block is in the free queue.</summary>
        public bool IsFree => RefCount == 0;

        // 中文：返回包含 id、引用计数、已用槽位与是否有哈希的调试字符串。
        public override string ToString()
            => $"KvBlock(id={Id}, refs={RefCount}, used={Used}, hash={(ContentHash.HasValue ? "yes" : "no")})";
    }

    /// <summary>
    /// Doubly-linked list of free <see cref="KvBlock"/>s in LRU eviction order.
    /// Front = least-recently-used (handed out first); back = most-recently-used.
    /// O(1) push/pop/remove via per-node pointers. Mirrors vLLM's
    /// <c>FreeKVCacheBlockQueue</c>.
    /// </summary>
    public sealed class FreeBlockQueue
    {
        private readonly KvBlock _head;   // sentinel
        private readonly KvBlock _tail;   // sentinel
        private int _count;

        // 中文：构造空闲块双向链表，建立头尾哨兵节点并初始化计数为零。
        public FreeBlockQueue()
        {
            _head = new KvBlock(-1);
            _tail = new KvBlock(-1);
            _head.NextFree = _tail;
            _tail.PrevFree = _head;
            _count = 0;
        }

        public int Count => _count;

        /// <summary>Append to the tail (most-recently-freed end).</summary>
        // 中文：将块追加到队尾（最近释放端）；若块已在某队列则抛异常。
        public void Enqueue(KvBlock block)
        {
            if (block.PrevFree != null || block.NextFree != null)
                throw new InvalidOperationException($"Block {block.Id} is already on a free queue.");

            block.PrevFree = _tail.PrevFree;
            block.NextFree = _tail;
            _tail.PrevFree.NextFree = block;
            _tail.PrevFree = block;
            _count++;
        }

        /// <summary>Pop from the head (least-recently-used).</summary>
        // 中文：从队首（最久未用端）弹出一个块，队列为空返回 null。
        public KvBlock Dequeue()
        {
            if (_count == 0)
                return null;
            KvBlock first = _head.NextFree;
            Remove(first);
            return first;
        }

        /// <summary>Remove the given block from anywhere in the queue. The block
        /// must currently be in this queue; otherwise behavior is undefined.</summary>
        // 中文：O(1) 从队列任意位置摘除指定块并清空其前后指针。
        public void Remove(KvBlock block)
        {
            if (block.PrevFree == null && block.NextFree == null)
                return; // not in queue
            block.PrevFree.NextFree = block.NextFree;
            block.NextFree.PrevFree = block.PrevFree;
            block.PrevFree = null;
            block.NextFree = null;
            _count--;
        }

        /// <summary>Peek the first block without removing.</summary>
        // 中文：查看队首块但不移除，队列为空返回 null。
        public KvBlock PeekFront()
        {
            if (_count == 0) return null;
            return _head.NextFree;
        }
    }
}
