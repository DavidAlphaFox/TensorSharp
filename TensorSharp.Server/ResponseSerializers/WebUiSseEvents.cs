// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Collections.Generic;
using System.Linq;
using TensorSharp.Models;

namespace TensorSharp.Server.ResponseSerializers
{
    /// <summary>
    /// Anonymous-typed payload builders for the Web UI's chat SSE protocol.
    /// The shapes are deliberately minimal so the JS UI can keep using
    /// <c>JSON.parse</c> + <c>typeof</c> checks instead of formal schemas.
    /// </summary>
    internal static class WebUiSseEvents
    {
        // 中文：构建排队进度事件，携带当前排队位置与待处理数量。
        public static object QueueProgress(int position, int pending) => new
        {
            queue_position = position,
            queue_pending = pending,
        };

        // 中文：构建单个 token 文本事件。
        public static object Token(string token) => new { token };

        // 中文：构建思考内容事件。
        public static object Thinking(string thinking) => new { thinking };

        // 中文：构建工具调用事件，将工具名与参数映射为列表。
        public static object ToolCalls(IReadOnlyList<ToolCall> toolCalls) => new
        {
            tool_calls = toolCalls.Select(tc => (object)new
            {
                name = tc.Name,
                arguments = tc.Arguments,
            }).ToList(),
        };

        // 中文：构建完成事件，汇总 token 数、耗时、速率、中止/错误状态、会话 ID 及 KV 缓存复用统计。
        public static object Done(
            int tokenCount,
            double elapsedSeconds,
            double tokPerSec,
            bool aborted,
            string error,
            string sessionId,
            int promptTokens,
            int kvCacheReusedTokens) => new
        {
            done = true,
            tokenCount,
            elapsed = elapsedSeconds,
            tokPerSec,
            aborted,
            error,
            sessionId,
            promptTokens,
            kvReusedTokens = kvCacheReusedTokens,
            kvReusePercent = promptTokens > 0 ? 100.0 * kvCacheReusedTokens / promptTokens : 0.0,
        };
    }
}
