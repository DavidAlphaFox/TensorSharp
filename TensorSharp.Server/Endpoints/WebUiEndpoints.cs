// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using TensorSharp.Server.ProtocolAdapters;

namespace TensorSharp.Server.Endpoints
{
    /// <summary>
    /// Routes for the bundled Web UI: queue status, model state + reload, and
    /// the SSE chat stream. Kept as pure routing - all behaviour lives in
    /// <see cref="WebUiAdapter"/>.
    /// </summary>
    internal static class WebUiEndpoints
    {
        // 中文：注册 Web UI 路由：队列状态、模型列表/加载，以及 SSE 聊天流，逻辑均委托给 WebUiAdapter。
        public static IEndpointRouteBuilder MapWebUiEndpoints(this IEndpointRouteBuilder endpoints)
        {
            endpoints.MapGet("/api/queue/status", (WebUiAdapter adapter) => adapter.GetQueueStatus());
            endpoints.MapGet("/api/models", (WebUiAdapter adapter) => adapter.GetModels());
            endpoints.MapPost("/api/models/load",
                (HttpContext ctx, HttpRequest req, WebUiAdapter adapter) => adapter.LoadModelAsync(ctx, req));
            endpoints.MapPost("/api/chat",
                (HttpContext ctx, WebUiAdapter adapter) => adapter.ChatStreamAsync(ctx));
            return endpoints;
        }
    }
}
