// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using TensorSharp.Models;

namespace TensorSharp.Server.ResponseSerializers
{
    /// <summary>
    /// Builders for the response shapes used by the Ollama-compatible
    /// endpoints. Returns anonymous-typed objects which the JSON serializer
    /// emits as Ollama clients expect (notably: stable property ordering and
    /// ISO-8601 <c>created_at</c> timestamps).
    /// </summary>
    internal static class OllamaResponseFactory
    {
        // 中文：构建 generate 接口的排队进度块，告知客户端当前排队位置与待处理数量。
        public static object QueueGenerateChunk(string model, int position, int pending) => new
        {
            model,
            created_at = TimestampNow(),
            response = "",
            done = false,
            queue_position = position,
            queue_pending = pending,
        };

        // 中文：构建 generate 流式响应中的单个 token 文本块（done=false）。
        public static object GenerateTokenChunk(string model, string piece) => new
        {
            model,
            created_at = TimestampNow(),
            response = piece,
            done = false,
        };

        // 中文：构建 generate 流式响应的终结块，附带 token 计数、耗时与 KV 缓存命中统计（done=true）。
        public static object GenerateFinalChunk(
            string model,
            int promptTokens,
            int evalTokens,
            int kvCacheReusedTokens,
            long totalNs,
            long promptNs,
            long evalNs) => new
        {
            model,
            created_at = TimestampNow(),
            response = "",
            done = true,
            done_reason = "stop",
            total_duration = totalNs,
            prompt_eval_count = promptTokens,
            prompt_eval_duration = promptNs,
            eval_count = evalTokens,
            eval_duration = evalNs,
            prompt_cache_hit_tokens = kvCacheReusedTokens,
            prompt_cache_hit_ratio = ComputeRatio(kvCacheReusedTokens, promptTokens),
        };

        // 中文：构建 generate 接口的错误终结块（done_reason=error）。
        public static object GenerateError(string model, string error) => new
        {
            model,
            created_at = TimestampNow(),
            response = "",
            done = true,
            done_reason = "error",
            error,
        };

        // 中文：构建 generate 接口的非流式完整响应，一次性返回全部内容与统计数据。
        public static object GenerateNonStreamingResponse(
            string model,
            string content,
            int promptTokens,
            int evalTokens,
            int kvCacheReusedTokens,
            long totalNs,
            long promptNs,
            long evalNs) => new
        {
            model,
            created_at = TimestampNow(),
            response = content,
            done = true,
            done_reason = "stop",
            total_duration = totalNs,
            prompt_eval_count = promptTokens,
            prompt_eval_duration = promptNs,
            eval_count = evalTokens,
            eval_duration = evalNs,
            prompt_cache_hit_tokens = kvCacheReusedTokens,
            prompt_cache_hit_ratio = ComputeRatio(kvCacheReusedTokens, promptTokens),
        };

        // 中文：构建 chat 接口的排队进度块，message 为空并附带排队位置与待处理数量。
        public static object QueueChatChunk(string model, int position, int pending) => new
        {
            model,
            created_at = TimestampNow(),
            message = new { role = "assistant", content = "" },
            done = false,
            queue_position = position,
            queue_pending = pending,
        };

        // 中文：构建 chat 流式响应中的原始 token 块，将片段填入 assistant 消息的 content。
        public static object ChatRawTokenChunk(string model, string piece) => new
        {
            model,
            created_at = TimestampNow(),
            message = new { role = "assistant", content = piece },
            done = false,
        };

        // 中文：构建 chat 流式响应中已解析的块，分别填入正文 content 与思考 thinking 字段。
        public static object ChatParsedChunk(string model, string contentChunk, string thinkingChunk) => new
        {
            model,
            created_at = TimestampNow(),
            message = new { role = "assistant", content = contentChunk, thinking = thinkingChunk },
            done = false,
        };

        // 中文：构建 chat 原始流式响应的终结块，附带 token 计数、耗时与 KV 缓存统计（done=true）。
        public static object ChatRawFinalChunk(
            string model,
            int promptTokens,
            int evalTokens,
            int kvCacheReusedTokens,
            long totalNs,
            long promptNs,
            long evalNs) => new
        {
            model,
            created_at = TimestampNow(),
            message = new { role = "assistant", content = "" },
            done = true,
            done_reason = "stop",
            total_duration = totalNs,
            prompt_eval_count = promptTokens,
            prompt_eval_duration = promptNs,
            eval_count = evalTokens,
            eval_duration = evalNs,
            prompt_cache_hit_tokens = kvCacheReusedTokens,
            prompt_cache_hit_ratio = ComputeRatio(kvCacheReusedTokens, promptTokens),
        };

        // 中文：构建 chat 已解析流式响应的终结块，含工具调用列表，并据此决定 done_reason 为 tool_calls 或 stop。
        public static object ChatParsedFinalChunk(
            string model,
            IReadOnlyList<ToolCall> collectedToolCalls,
            int promptTokens,
            int evalTokens,
            int kvCacheReusedTokens,
            long totalNs,
            long promptNs,
            long evalNs)
        {
            var toolCallsJson = ConvertToolCalls(collectedToolCalls);
            return new
            {
                model,
                created_at = TimestampNow(),
                message = new
                {
                    role = "assistant",
                    content = "",
                    tool_calls = toolCallsJson,
                },
                done = true,
                done_reason = collectedToolCalls != null ? "tool_calls" : "stop",
                total_duration = totalNs,
                prompt_eval_count = promptTokens,
                prompt_eval_duration = promptNs,
                eval_count = evalTokens,
                eval_duration = evalNs,
                prompt_cache_hit_tokens = kvCacheReusedTokens,
                prompt_cache_hit_ratio = ComputeRatio(kvCacheReusedTokens, promptTokens),
            };
        }

        // 中文：构建 chat 接口的错误终结块（done_reason=error）。
        public static object ChatErrorChunk(string model, string error) => new
        {
            model,
            created_at = TimestampNow(),
            message = new { role = "assistant", content = "" },
            done = true,
            done_reason = "error",
            error,
        };

        // 中文：构建 chat 接口的非流式完整响应，包装传入的 message 并附带 token 与耗时统计。
        public static object ChatNonStreamingResponse(
            string model,
            object message,
            string doneReason,
            int promptTokens,
            int evalTokens,
            int kvCacheReusedTokens,
            long totalNs,
            long promptNs,
            long evalNs) => new
        {
            model,
            created_at = TimestampNow(),
            message,
            done = true,
            done_reason = doneReason,
            total_duration = totalNs,
            prompt_eval_count = promptTokens,
            prompt_eval_duration = promptNs,
            eval_count = evalTokens,
            eval_duration = evalNs,
            prompt_cache_hit_tokens = kvCacheReusedTokens,
            prompt_cache_hit_ratio = ComputeRatio(kvCacheReusedTokens, promptTokens),
        };

        // 中文：构建非流式 chat 的 assistant 消息对象，含正文、思考内容与转换后的工具调用。
        public static object ChatNonStreamingMessage(string content, string thinking, IReadOnlyList<ToolCall> toolCalls) => new
        {
            role = "assistant",
            content = content ?? "",
            thinking,
            tool_calls = ConvertToolCalls(toolCalls),
        };

        // 中文：构建仅含正文的简单 assistant 消息对象。
        public static object ChatPlainMessage(string content) => new
        {
            role = "assistant",
            content,
        };

        // 中文：将内部 ToolCall 列表转换为 Ollama 工具调用的匿名对象数组，空列表返回 null。
        private static IReadOnlyList<object> ConvertToolCalls(IReadOnlyList<ToolCall> toolCalls)
        {
            if (toolCalls == null || toolCalls.Count == 0)
                return null;

            var result = new object[toolCalls.Count];
            for (int i = 0; i < toolCalls.Count; i++)
            {
                var tc = toolCalls[i];
                result[i] = new
                {
                    function = new { name = tc.Name, arguments = tc.Arguments },
                };
            }
            return result;
        }

        // 中文：返回当前 UTC 时间的 ISO-8601 字符串，作为 created_at 时间戳。
        private static string TimestampNow() => DateTime.UtcNow.ToString("o");

        /// <summary>
        /// Fraction of the prompt that was served from the prior turn's KV cache.
        /// Returns 0.0 when the prompt is empty so consumers can render the value
        /// uniformly without special-casing the no-tokens path.
        /// </summary>
        // 中文：计算 KV 缓存命中占 prompt 的比例，prompt 为空时返回 0.0。
        private static double ComputeRatio(int reused, int total)
        {
            return total > 0 ? (double)reused / total : 0.0;
        }
    }
}
