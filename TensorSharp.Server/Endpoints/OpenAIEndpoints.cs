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
    /// Routes for the OpenAI-compatible chat-completions surface.
    /// </summary>
    internal static class OpenAIEndpoints
    {
        // 中文：注册 OpenAI 兼容路由：POST /v1/chat/completions 聊天补全与 GET /v1/models 模型列表。
        public static IEndpointRouteBuilder MapOpenAIEndpoints(this IEndpointRouteBuilder endpoints)
        {
            endpoints.MapPost("/v1/chat/completions",
                (HttpContext ctx, OpenAIChatAdapter adapter) => adapter.ChatCompletionsAsync(ctx));
            endpoints.MapGet("/v1/models",
                (OpenAIChatAdapter adapter) => adapter.ListModels());
            return endpoints;
        }
    }
}
