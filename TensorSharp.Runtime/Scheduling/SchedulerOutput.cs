// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Collections.Generic;

namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>
    /// One per-sequence decision from the scheduler for a single step.
    /// Mirrors vLLM's per-row entries in <c>SchedulerOutput.num_scheduled_tokens</c>
    /// + <c>scheduled_new_reqs</c>.
    /// </summary>
    public sealed class ScheduledSequenceWork
    {
        // 中文：构造单序列调度决策，记录序列、本步调度的 token 数及是否为新准入/预填充。
        public ScheduledSequenceWork(
            SequenceState seq,
            int numScheduledTokens,
            bool isNewAdmission,
            bool isPrefill)
        {
            Sequence = seq;
            NumScheduledTokens = numScheduledTokens;
            IsNewAdmission = isNewAdmission;
            IsPrefill = isPrefill;
        }

        public SequenceState Sequence { get; }

        /// <summary>How many tokens of this sequence to forward this step. For
        /// a decode step this is 1. For prefill it can be up to the unprocessed
        /// prompt length (or smaller if chunked).</summary>
        public int NumScheduledTokens { get; }

        /// <summary>True iff this is the first step the sequence appears in
        /// (we just transitioned from Waiting/Preempted to Running).</summary>
        public bool IsNewAdmission { get; }

        /// <summary>True when <see cref="NumScheduledTokens"/> &gt; 1; informational
        /// flag passed to the executor.</summary>
        public bool IsPrefill { get; }

        /// <summary>Position offset where this step's tokens land in the
        /// sequence's logical timeline. Equal to
        /// <see cref="SequenceState.NumComputedTokens"/> at scheduling time.</summary>
        public int StartPosition => Sequence.NumComputedTokens;

        // 中文：返回该调度项的可读摘要（请求 ID、token 数、预填充/解码、是否新准入）。
        public override string ToString()
            => $"Work({Sequence.RequestId} +{NumScheduledTokens} {(IsPrefill ? "prefill" : "decode")}{(IsNewAdmission ? " new" : "")})";
    }

    /// <summary>
    /// What the scheduler picked for one step. Consumed by the executor.
    /// Mirrors vLLM's <c>SchedulerOutput</c>.
    /// </summary>
    public sealed class SchedulerOutput
    {
        // 中文：构造空的单步调度输出，初始化已调度工作、被抢占与已完成请求 ID 列表。
        public SchedulerOutput()
        {
            ScheduledWork = new List<ScheduledSequenceWork>();
            PreemptedRequestIds = new List<string>();
            FinishedRequestIds = new List<string>();
        }

        public List<ScheduledSequenceWork> ScheduledWork { get; }
        public List<string> PreemptedRequestIds { get; }
        public List<string> FinishedRequestIds { get; }

        /// <summary>Sum of <see cref="ScheduledSequenceWork.NumScheduledTokens"/>
        /// across all scheduled work. Used for token-budget telemetry.</summary>
        // 中文：累加所有已调度工作项的 token 数，得到本步的总调度 token 量。
        public int TotalScheduledTokens
        {
            get
            {
                int s = 0;
                for (int i = 0; i < ScheduledWork.Count; i++)
                    s += ScheduledWork[i].NumScheduledTokens;
                return s;
            }
        }

        public bool IsEmpty => ScheduledWork.Count == 0 && FinishedRequestIds.Count == 0;
    }
}
