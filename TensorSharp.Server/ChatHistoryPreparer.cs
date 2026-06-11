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
using System.Linq;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace TensorSharp.Server
{
    internal static class ChatHistoryPreparer
    {
        // 中文：准备推理历史的便捷重载，使用空日志器。
        public static List<ChatMessage> PrepareHistoryForInference(List<ChatMessage> history, string arch)
            => PrepareHistoryForInference(history, arch, NullLogger.Instance);

        // 中文：逐条规范化历史消息（如视频抽帧降采样），仅在有改动时复制列表，否则原样返回。
        public static List<ChatMessage> PrepareHistoryForInference(List<ChatMessage> history, string arch, ILogger logger)
        {
            if (history == null || history.Count == 0)
                return history;

            List<ChatMessage> prepared = null;
            for (int i = 0; i < history.Count; i++)
            {
                var normalized = NormalizeMessageForInference(history[i], arch, logger);
                if (ReferenceEquals(normalized, history[i]))
                    continue;

                prepared ??= new List<ChatMessage>(history);
                prepared[i] = normalized;
            }

            return prepared ?? history;
        }

        // 中文：将上一轮跟踪历史中匹配前缀的助手原始输出 token 复用到本轮消息上，以便提示渲染器跨轮重用助手 token。
        public static List<ChatMessage> AugmentWithCachedRawTokens(List<ChatMessage> incoming, IReadOnlyList<ChatMessage> trackedHistory)
        {
            if (incoming == null)
                return null;

            int matchUntil = 0;
            if (trackedHistory != null)
            {
                int max = Math.Min(incoming.Count, trackedHistory.Count);
                for (int i = 0; i < max; i++)
                {
                    ChatMessage src = incoming[i];
                    ChatMessage tracked = trackedHistory[i];

                    if (src.Role != tracked.Role)
                        break;

                    // Compare on Content for non-assistant roles only. Assistant content can be
                    // legitimately altered by the streaming output parser between turns.
                    if (src.Role != "assistant"
                        && !string.Equals(src.Content ?? string.Empty, tracked.Content ?? string.Empty, StringComparison.Ordinal))
                        break;

                    matchUntil = i + 1;
                }
            }

            var result = new List<ChatMessage>(incoming.Count);
            for (int i = 0; i < incoming.Count; i++)
            {
                ChatMessage src = incoming[i];

                bool useTracked = trackedHistory != null
                    && i < matchUntil
                    && trackedHistory[i].Role == "assistant"
                    && trackedHistory[i].RawOutputTokens != null
                    && trackedHistory[i].RawOutputTokens.Count > 0
                    && (src.RawOutputTokens == null || src.RawOutputTokens.Count == 0);

                if (useTracked)
                {
                    result.Add(new ChatMessage
                    {
                        Role = src.Role,
                        Content = src.Content,
                        ImagePaths = src.ImagePaths,
                        AudioPaths = src.AudioPaths,
                        TextFilePaths = src.TextFilePaths,
                        IsVideo = src.IsVideo,
                        ToolCalls = src.ToolCalls,
                        Thinking = src.Thinking,
                        RawOutputTokens = trackedHistory[i].RawOutputTokens,
                    });
                }
                else
                {
                    result.Add(src);
                }
            }
            return result;
        }

        // 中文：用本轮输入历史加上新生成的助手消息（含原始 token）重建会话跟踪历史，供下一轮复用。
        public static void UpdateTrackedHistory(
            List<ChatMessage> trackedHistory,
            List<ChatMessage> incomingHistory,
            string assistantText,
            List<int> generatedTokens)
        {
            trackedHistory.Clear();
            if (incomingHistory != null)
            {
                for (int i = 0; i < incomingHistory.Count; i++)
                    trackedHistory.Add(CloneShallow(incomingHistory[i]));
            }

            trackedHistory.Add(new ChatMessage
            {
                Role = "assistant",
                Content = assistantText,
                RawOutputTokens = generatedTokens,
            });
        }

        // 中文：判断单条消息是否含图像或音频等多模态内容。
        public static bool HasMultimodalContent(ChatMessage msg)
        {
            if (msg == null) return false;
            return (msg.ImagePaths != null && msg.ImagePaths.Count > 0) ||
                   (msg.AudioPaths != null && msg.AudioPaths.Count > 0);
        }

        // 中文：判断整段历史中是否有任一消息含多模态内容。
        public static bool HasMultimodalContent(List<ChatMessage> history)
        {
            if (history == null || history.Count == 0)
                return false;

            return history.Any(HasMultimodalContent);
        }

        // 中文：按提示顺序收集历史中所有非空图像路径。
        public static List<string> GetImagePathsInPromptOrder(List<ChatMessage> history)
        {
            var imagePaths = new List<string>();
            if (history == null)
                return imagePaths;

            foreach (var msg in history)
            {
                if (msg.ImagePaths == null)
                    continue;

                foreach (var path in msg.ImagePaths)
                {
                    if (!string.IsNullOrEmpty(path))
                        imagePaths.Add(path);
                }
            }

            return imagePaths;
        }

        // 中文：规范化单条消息——对 gemma4 视频消息按配置上限做均匀抽帧降采样，否则原样返回。
        private static ChatMessage NormalizeMessageForInference(ChatMessage msg, string arch, ILogger logger)
        {
            int maxVideoFrames = MediaHelper.GetConfiguredMaxVideoFrames();
            // maxVideoFrames <= 0 means "no cap" (pure time-based extraction); leave history untouched.
            if (arch != "gemma4" || maxVideoFrames <= 0 || !msg.IsVideo || msg.ImagePaths == null || msg.ImagePaths.Count <= maxVideoFrames)
                return msg;

            var sampled = MediaHelper.SelectEvenlySpacedIndices(msg.ImagePaths.Count, maxVideoFrames)
                .Select(i => msg.ImagePaths[i])
                .ToList();

            (logger ?? NullLogger.Instance).LogInformation(LogEventIds.VideoFrameDownsample,
                "video.downsample originalFrames={OriginalFrames} sampledFrames={SampledFrames} architecture={Architecture}",
                msg.ImagePaths.Count, sampled.Count, arch);

            return new ChatMessage
            {
                Role = msg.Role,
                Content = msg.Content,
                ImagePaths = sampled,
                AudioPaths = msg.AudioPaths != null ? new List<string>(msg.AudioPaths) : null,
                TextFilePaths = msg.TextFilePaths != null ? new List<string>(msg.TextFilePaths) : null,
                IsVideo = msg.IsVideo,
                ToolCalls = msg.ToolCalls,
                Thinking = msg.Thinking,
                RawOutputTokens = msg.RawOutputTokens,
            };
        }

        // 中文：浅复制一条聊天消息（各字段引用直接拷贝）。
        private static ChatMessage CloneShallow(ChatMessage src)
        {
            return new ChatMessage
            {
                Role = src.Role,
                Content = src.Content,
                ImagePaths = src.ImagePaths,
                AudioPaths = src.AudioPaths,
                TextFilePaths = src.TextFilePaths,
                IsVideo = src.IsVideo,
                ToolCalls = src.ToolCalls,
                Thinking = src.Thinking,
                RawOutputTokens = src.RawOutputTokens,
            };
        }
    }
}
