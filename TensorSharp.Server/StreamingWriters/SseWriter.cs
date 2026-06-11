// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;

namespace TensorSharp.Server.StreamingWriters
{
    /// <summary>
    /// Tiny helpers for writing Server-Sent Events. Centralised so we never
    /// have <c>"data: " + json + "\n\n"</c> inlined across handlers and so
    /// flushes are always paired with the writes that need to be visible to
    /// the client immediately.
    /// </summary>
    internal static class SseWriter
    {
        // 中文：设置 SSE 流的响应头（text/event-stream、禁用缓存、保持连接）。
        public static void ApplyHeaders(HttpResponse response)
        {
            response.ContentType = "text/event-stream";
            response.Headers["Cache-Control"] = "no-cache";
            response.Headers["Connection"] = "keep-alive";
        }

        /// <summary>
        /// Serialise <paramref name="payload"/> as a single <c>data:</c> SSE
        /// frame and flush it. Honours <paramref name="cancellationToken"/> so
        /// callers can abort quickly when the client disappears.
        /// </summary>
        // 中文：将载荷序列化为单个 data: SSE 帧写入并刷新，遵从取消令牌以便客户端断开时快速中止。
        public static async Task WriteEventAsync(
            HttpResponse response,
            object payload,
            CancellationToken cancellationToken,
            JsonSerializerOptions jsonOptions = null)
        {
            string json = JsonSerializer.Serialize(payload, jsonOptions);
            await response.WriteAsync($"data: {json}\n\n", cancellationToken);
            await response.Body.FlushAsync(cancellationToken);
        }

        /// <summary>
        /// Same as <see cref="WriteEventAsync(HttpResponse, object, CancellationToken, JsonSerializerOptions)"/>
        /// but without a cancellation token; used by best-effort "final flush"
        /// paths that must not throw on already-cancelled requests.
        /// </summary>
        // 中文：无取消令牌版本的 SSE 帧写入，用于不应在请求已取消时抛异常的尽力而为收尾刷新。
        public static async Task WriteEventAsync(
            HttpResponse response,
            object payload,
            JsonSerializerOptions jsonOptions = null)
        {
            string json = JsonSerializer.Serialize(payload, jsonOptions);
            await response.WriteAsync($"data: {json}\n\n");
            await response.Body.FlushAsync();
        }

        /// <summary>
        /// Write the literal <c>data: [DONE]</c> sentinel used by the OpenAI
        /// streaming chat-completions API.
        /// </summary>
        // 中文：写入 OpenAI 流式接口使用的 data: [DONE] 结束哨兵。
        public static Task WriteDoneSentinelAsync(HttpResponse response, CancellationToken cancellationToken)
        {
            return response.WriteAsync("data: [DONE]\n\n", cancellationToken);
        }
    }
}
