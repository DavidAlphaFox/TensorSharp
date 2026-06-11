// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// DEPRECATED no-op shim. The InferenceQueue used to serialize all requests
// against a single model instance so concurrent callers wouldn't corrupt the
// shared KV state. With the continuous-batching engine the real concurrency
// boundary lives inside <see cref="TensorSharp.Runtime.Scheduling.InferenceEngine"/>,
// which already supports multiple in-flight sequences. This shim grants
// tickets immediately so existing adapter code keeps compiling without
// changes; once all adapters are migrated, this file (and the explicit
// queue-position chunks they emit) can be deleted.
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace TensorSharp.Server
{
    public class InferenceQueue
    {
        private long _totalProcessed;

        // 中文：无参构造，使用空日志器委托给主构造函数。
        public InferenceQueue() : this(NullLogger<InferenceQueue>.Instance) { }
        // 中文：主构造函数——本类已是空操作 shim，故忽略传入的日志器。
        public InferenceQueue(ILogger<InferenceQueue> logger) { _ = logger; }

        public int PendingCount => 0;
        public long TotalProcessed => Interlocked.Read(ref _totalProcessed);
        public bool IsBusy => false;

        // 中文：入队——不做任何串行化，仅累加处理计数并立即发放一张兼容性票据。
        public QueueTicket Enqueue(CancellationToken ct, string requestId = null)
        {
            // No serialization: the engine handles concurrent requests.
            // We still hand out a ticket so callers can dispose it for
            // backwards-compatible counting.
            Interlocked.Increment(ref _totalProcessed);
            return new QueueTicket(ct, requestId);
        }

        // 中文：返回队列状态快照（恒为不忙、零挂起，仅累计处理数有效）。
        public QueueStatus GetStatus()
        {
            return new QueueStatus
            {
                Busy = false,
                PendingRequests = 0,
                TotalProcessed = TotalProcessed,
                CurrentRequestId = null
            };
        }

        // 中文：释放票据——空操作，保留以兼容旧调用方。
        internal void Release(QueueTicket _) { }
        // 中文：移除已取消票据——空操作，保留以兼容旧调用方。
        internal void RemoveCancelled(QueueTicket _) { }
    }

    public class QueueStatus
    {
        public bool Busy { get; set; }
        public int PendingRequests { get; set; }
        public long TotalProcessed { get; set; }
        public string CurrentRequestId { get; set; }
    }

    /// <summary>
    /// Vestigial ticket that grants immediately. Kept only because adapters
    /// still reference its <see cref="Position"/> / <see cref="PendingCount"/>
    /// API when emitting queue-status events. With continuous batching
    /// queue position is meaningless, so we always report position 0.
    /// </summary>
    public class QueueTicket : IDisposable
    {
        private readonly CancellationTokenRegistration _ctReg;
        private bool _disposed;

        public string RequestId { get; }
        public int Position => 0;
        public bool IsReady => true;
        public bool IsCancelled => false;
        internal System.Collections.Generic.LinkedListNode<QueueTicket> Node { get; set; }

        // 中文：构造票据——记录请求 id 并注册一个空的取消回调以便后续释放。
        internal QueueTicket(CancellationToken ct, string requestId)
        {
            RequestId = requestId;
            _ctReg = ct.Register(() => { });
        }

        // 中文：等待（带超时）——票据立即就绪，直接返回已完成任务。
        public Task WaitAsync(TimeSpan timeout) => Task.CompletedTask;
        // 中文：等待就绪——票据立即就绪，直接返回已完成任务。
        public Task WaitUntilReadyAsync() => Task.CompletedTask;

        // 中文：释放票据——幂等地注销取消回调注册。
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _ctReg.Dispose();
        }
    }
}
