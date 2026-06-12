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
// 文件：PromptRenderer.cs
// 用途：提供基于 GGUF 聊天模板的提示词渲染实现，将对话消息列表与模板合并为模型输入字符串。
// 主要类型：GgufPromptRenderer —— 实现 IPromptRenderer 接口，调用 ChatTemplate 完成渲染。
// ──────────────────────────

using System.Collections.Generic;

namespace TensorSharp.Runtime
{
    public sealed class GgufPromptRenderer : IPromptRenderer
    {
        // 中文：根据 GGUF 聊天模板将消息列表渲染为最终的提示词字符串，支持工具调用与思考模式
        public string Render(
            string template,
            List<ChatMessage> messages,
            bool addGenerationPrompt = true,
            string? architecture = null,
            List<ToolFunction>? tools = null,
            bool enableThinking = false)
        {
            return ChatTemplate.RenderFromGgufTemplate(
                template,
                messages,
                addGenerationPrompt,
                architecture,
                tools,
                enableThinking);
        }
    }
}

