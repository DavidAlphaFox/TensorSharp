// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;

namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>
    /// Client-facing handle to an in-flight request. Streams sampled tokens
    /// through <see cref="Tokens"/> and exposes finalization metadata via
    /// <see cref="Completion"/>.
    ///
    /// Producers (the engine worker thread) publish via the internal
    /// PublishToken / CompleteFinished / CompleteWithError methods. Consumers
    /// read via the standard async channel pattern.
    /// </summary>
    public sealed class InferenceRequestHandle
    {
        private readonly Channel<int> _tokens = Channel.CreateUnbounded<int>(
            new UnboundedChannelOptions { SingleReader = true, SingleWriter = true });
        private readonly TaskCompletionSource<InferenceCompletion> _completionTcs =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
        private readonly CancellationTokenRegistration _ctReg;
        private int _publishedTokens;

        public string RequestId => Sequence.RequestId;
        public SequenceState Sequence { get; }
        public ChannelReader<int> Tokens => _tokens.Reader;
        public Task<InferenceCompletion> Completion => _completionTcs.Task;
        public DateTime SubmittedAt => Sequence.SubmittedAt;

        // 中文：构造请求句柄，保存序列状态并注册取消令牌以在取消时中止该请求。
        internal InferenceRequestHandle(SequenceState seq, InferenceEngine engine, CancellationToken ct)
        {
            Sequence = seq;
            _ctReg = ct.Register(() => engine.Abort(seq.RequestId));
        }

        // 中文：将新采样出的 token 写入通道供消费者读取，并递增已发布 token 计数。
        internal void PublishToken(int tokenId)
        {
            // Channel is unbounded; should never fail in practice.
            _tokens.Writer.TryWrite(tokenId);
            Interlocked.Increment(ref _publishedTokens);
        }

        // 中文：正常完成请求：关闭 token 通道并以序列的最终状态与统计信息填充完成 future。
        internal void CompleteFinished()
        {
            _tokens.Writer.TryComplete();
            _ctReg.Dispose();
            var completion = new InferenceCompletion
            {
                Status = Sequence.Status,
                FinishReason = Sequence.FinishReason,
                OutputTokenCount = Sequence.OutputTokens.Count,
                PromptTokenCount = Sequence.PromptTokens.Count,
                PrefixCacheReusedTokens = Sequence.PrefixCacheReusedTokens,
                FirstTokenAt = Sequence.FirstTokenAt,
                SubmittedAt = Sequence.SubmittedAt,
            };
            _completionTcs.TrySetResult(completion);
        }

        // 中文：以异常终止请求：用错误关闭 token 通道并将完成 future 置为异常。
        internal void CompleteWithError(Exception ex)
        {
            _tokens.Writer.TryComplete(ex);
            _ctReg.Dispose();
            _completionTcs.TrySetException(ex);
        }

        // 中文：因客户端取消而中止请求：关闭通道并以 aborted 状态填充完成 future。
        internal void CompleteAborted()
        {
            _tokens.Writer.TryComplete();
            _ctReg.Dispose();
            var completion = new InferenceCompletion
            {
                Status = SequenceStatus.FinishedAborted,
                FinishReason = "aborted",
                OutputTokenCount = Sequence.OutputTokens.Count,
                PromptTokenCount = Sequence.PromptTokens.Count,
                PrefixCacheReusedTokens = Sequence.PrefixCacheReusedTokens,
                FirstTokenAt = Sequence.FirstTokenAt,
                SubmittedAt = Sequence.SubmittedAt,
            };
            _completionTcs.TrySetResult(completion);
        }
    }

    public sealed class InferenceCompletion
    {
        public SequenceStatus Status { get; init; }
        public string FinishReason { get; init; }
        public int PromptTokenCount { get; init; }
        public int OutputTokenCount { get; init; }
        public int PrefixCacheReusedTokens { get; init; }
        public DateTime? FirstTokenAt { get; init; }
        public DateTime SubmittedAt { get; init; }
    }
}
