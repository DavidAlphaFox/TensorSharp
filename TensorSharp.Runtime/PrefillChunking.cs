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
// 文件：PrefillChunking.cs
// 用途：提供大提示词的分块预填充（chunked prefill）策略，避免超长序列一次性进入模型
//       导致注意力分数张量（[numHeads, seqLen, seqLen]）内存爆炸。
//       服务端管道与 CLI 交互会话共享同一策略，确保两个前端的分块阈值保持一致。
// 主要类型：PrefillChunking（静态工具类）
// ────────────────────────

using System;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Shared chunked-prefill policy used by both the long-running server pipeline
    /// and the CLI interactive session. Keeping the chunk-size selection in one
    /// place ensures the two front-ends never drift apart on what counts as "too
    /// large to feed through the model in a single forward call".
    ///
    /// The chunk size caps the height of the [numHeads, seqLen, seqLen] attention
    /// score tensor at <c>numHeads * chunkSize * (chunkSize + prevContext)</c> so
    /// 32K-token prompts no longer try to allocate gigabytes of scores in one go.
    /// </summary>
    public static class PrefillChunking
    {
        /// <summary>
        /// Maximum tokens fed through the model in a single ForwardRefill call.
        /// GgmlCuda has wider kernel buffers and tolerates a larger chunk; other
        /// backends use a more conservative cap that still keeps the score tensor
        /// bounded under typical (16 head, 128 head-dim) configurations.
        /// </summary>
        // 中文：根据后端类型计算单次前向传播允许的最大 token 分块大小；GgmlCuda 上限为 5120，其余后端为 2048
        public static int ResolveChunkSize(BackendType backend, int tokenCount)
        {
            if (tokenCount <= 0)
                return 0;

            return backend == BackendType.GgmlCuda
                ? Math.Min(tokenCount, 5120)
                : Math.Min(tokenCount, 2048);
        }

        /// <summary>
        /// Convenience wrapper around <see cref="ResolveChunkSize"/> that returns true
        /// when the prompt would actually be split (so callers can avoid the chunked
        /// code path's per-chunk array copies for short prompts).
        /// </summary>
        // 中文：判断当前提示词是否需要分块处理，并通过 out 参数返回实际分块大小；短提示词直接返回 false 以跳过分块开销
        public static bool ShouldChunk(BackendType backend, int tokenCount, out int chunkSize)
        {
            chunkSize = ResolveChunkSize(backend, tokenCount);
            return chunkSize > 0 && chunkSize < tokenCount;
        }
    }
}
