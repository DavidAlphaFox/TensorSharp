// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

namespace TensorSharp.Runtime.Paged
{
    /// <summary>
    /// Per-sequence ordered list of physical <see cref="KvBlock"/>s holding that
    /// sequence's K/V state. Index 0 is positions <c>[0..blockSize)</c>, index 1
    /// is <c>[blockSize..2*blockSize)</c>, etc.
    ///
    /// Mirrors vLLM's per-row <c>BlockTable</c> entry but represented as a list
    /// of object references rather than an integer ID tensor - the C# executor
    /// resolves the IDs at injection time when it needs them as a tensor.
    /// </summary>
    public sealed class BlockTable
    {
        private readonly List<KvBlock> _blocks = new(8);
        private int _numTokens;
        private readonly int _blockSize;

        // 中文：构造块表，记录每块容纳的 token 数（块大小）。
        public BlockTable(int blockSize)
        {
            _blockSize = blockSize;
        }

        public int BlockSize => _blockSize;
        public int NumBlocks => _blocks.Count;

        /// <summary>Total tokens currently committed to this sequence's KV cache
        /// (i.e. positions whose K/V has been computed and stored).</summary>
        public int NumTokens => _numTokens;

        public IReadOnlyList<KvBlock> Blocks => _blocks;

        /// <summary>The physical block that covers position <paramref name="tokenPos"/>.
        /// </summary>
        // 中文：返回覆盖指定 token 位置的物理块，越界或未分配则抛异常。
        public KvBlock GetBlockAt(int tokenPos)
        {
            if (tokenPos < 0) throw new ArgumentOutOfRangeException(nameof(tokenPos));
            int blockIdx = tokenPos / _blockSize;
            if (blockIdx >= _blocks.Count)
                throw new ArgumentOutOfRangeException(nameof(tokenPos),
                    $"Position {tokenPos} not yet allocated ({_blocks.Count} blocks).");
            return _blocks[blockIdx];
        }

        /// <summary>Slot index within the physical block for position
        /// <paramref name="tokenPos"/>. Combined with the block's storage span
        /// this gives the byte offset for that token's K/V.</summary>
        // 中文：返回指定 token 位置在其物理块内的槽位下标（位置对块大小取余）。
        public int GetSlotInBlock(int tokenPos)
        {
            return tokenPos % _blockSize;
        }

        // 中文：在序列尾部追加一个新分配的物理块。
        public void AppendBlock(KvBlock block)
        {
            _blocks.Add(block);
        }

        /// <summary>Mark <paramref name="newTokens"/> additional tokens as
        /// committed. Called by the executor after each forward.</summary>
        // 中文：推进已提交 token 计数；若所需块数超过已分配块数则抛异常。
        public void AdvanceTokens(int newTokens)
        {
            _numTokens += newTokens;
            int neededBlocks = (_numTokens + _blockSize - 1) / _blockSize;
            if (neededBlocks > _blocks.Count)
                throw new InvalidOperationException(
                    $"AdvanceTokens({newTokens}) wants {neededBlocks} blocks but only {_blocks.Count} are allocated.");
        }

        /// <summary>Truncate the sequence back to <paramref name="newTokenCount"/>.
        /// Returns blocks that should be freed (those whose first position was
        /// past <paramref name="newTokenCount"/>). The returned blocks are
        /// removed from this table. The caller is responsible for calling
        /// <see cref="BlockPool.Free(System.Collections.Generic.IReadOnlyList{KvBlock})"/>.</summary>
        // 中文：将序列截断回指定 token 数，移除并返回多余的尾部块供调用方释放。
        public List<KvBlock> TruncateTo(int newTokenCount)
        {
            if (newTokenCount < 0 || newTokenCount > _numTokens)
                throw new ArgumentOutOfRangeException(nameof(newTokenCount));

            _numTokens = newTokenCount;
            int neededBlocks = (newTokenCount + _blockSize - 1) / _blockSize;
            var freed = new List<KvBlock>();
            while (_blocks.Count > neededBlocks)
            {
                freed.Add(_blocks[_blocks.Count - 1]);
                _blocks.RemoveAt(_blocks.Count - 1);
            }
            // The trailing partial block now has a smaller "used" count, but
            // we don't change Used here because the pool's block ref counts
            // are unaffected. The owner can re-write into the freed slots.
            return freed;
        }

        /// <summary>Reset the committed-token counter to zero while KEEPING the
        /// allocated physical blocks. Used by the live-cache-continuation fallback:
        /// when the model's live KV cache turns out to be unusable at execution time
        /// the sequence must re-prefill from position 0, but the blocks already
        /// reserved for it (sized for the would-be reused prefix) stay so the
        /// re-prefill can write into them without reallocating.</summary>
        // 中文：将已提交 token 计数清零但保留已分配的物理块（活缓存续接回退时重新预填用）。
        public void ResetTokensKeepingBlocks()
        {
            _numTokens = 0;
        }

        /// <summary>Drop ALL blocks. Returned to the caller for freeing.</summary>
        // 中文：清空所有块并将其返回给调用方释放，同时把 token 计数清零。
        public List<KvBlock> Clear()
        {
            var all = new List<KvBlock>(_blocks);
            _blocks.Clear();
            _numTokens = 0;
            return all;
        }

        /// <summary>Capacity (in tokens) currently allocated, including the
        /// partial trailing block. Tokens beyond this require allocating new
        /// blocks before they can be forwarded.</summary>
        public int CapacityTokens => _blocks.Count * _blockSize;

        /// <summary>How many tokens can still be written into the currently-
        /// allocated blocks without growing.</summary>
        public int FreeSlotsInCurrentBlocks => CapacityTokens - _numTokens;
    }
}
