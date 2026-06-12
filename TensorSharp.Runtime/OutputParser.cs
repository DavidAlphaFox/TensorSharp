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
// 文件：OutputParser.cs
// 用途：定义流式输出解析器，用于从 LLM 模型生成的 token 流中实时提取
//       思考内容（thinking）、正文内容（content）以及工具调用（tool calls）。
// 主要类型：
//   - ToolFunction / ToolParameter：描述向模型提供的工具函数定义
//   - ToolCall / ParsedOutput：表示从模型输出中提取的工具调用与解析结果
//   - IOutputParser：流式解析器接口
//   - Qwen3OutputParser：Qwen3 格式解析器（<think>...</think> + <tool_call>...</tool_call>）
//   - Qwen35OutputParser：Qwen3.5 格式解析器（继承 Qwen3，始终启用思考模式）
//   - Gemma4OutputParser：Gemma4 格式解析器（<|channel>thought...</channel|> 思考 + <|tool_call>...</tool_call|> 工具）
//   - HarmonyOutputParser：GPT-OSS/Harmony 格式解析器（<|start|>/<|end|> 消息帧，<|channel|> 频道分发）
//   - PassthroughOutputParser：直通解析器（不做任何标签解析）
//   - OutputParserFactory：根据模型架构名称创建对应解析器的工厂类
// ────────────────────────

using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Represents a tool function definition provided to the model.
    /// </summary>
    public class ToolFunction
    {
        public string Name { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public Dictionary<string, ToolParameter> Parameters { get; set; } = new();
        public List<string> Required { get; set; } = new();
    }

    public class ToolParameter
    {
        public string Type { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public List<string> Enum { get; set; } = new();
    }

    /// <summary>
    /// Represents a tool call extracted from model output.
    /// </summary>
    public class ToolCall
    {
        public string Name { get; set; } = string.Empty;
        public Dictionary<string, object> Arguments { get; set; } = new();
        public int Index { get; set; }

        // 中文：将工具调用序列化为 "函数名(JSON参数)" 的可读字符串形式
        public override string ToString()
        {
            string args = Arguments != null ? JsonSerializer.Serialize(Arguments) : "{}";
            return $"{Name}({args})";
        }
    }

    /// <summary>
    /// Parsed output from a model generation step.
    /// </summary>
    public class ParsedOutput
    {
        public string Content { get; set; } = "";
        public string Thinking { get; set; } = "";
        public List<ToolCall>? ToolCalls { get; set; }
    }

    /// <summary>
    /// Streaming parser that extracts thinking content, regular content, and tool calls
    /// from model output. Handles model-specific tag formats.
    /// </summary>
    public interface IOutputParser : IOutputProtocolParser
    {
    }

    // ========================================================================
    // Qwen3 Parser: <think>...</think> for thinking, <tool_call>...</tool_call>
    // ========================================================================

    public class Qwen3OutputParser : IOutputParser
    {
        private enum State { CollectingThinking, ThinkingDone, CollectingContent, CollectingTool }

        private State _state;
        private readonly StringBuilder _buffer = new();
        private bool _stripLeadingThinkTag;
        private int _callIndex;

        public bool HasThinkingSupport => true;
        public bool HasToolSupport => true;
        public bool AlwaysRequired => false;

        // 中文：初始化解析器状态，根据是否启用思考模式决定起始解析状态
        public void Init(bool enableThinking, List<ToolFunction> tools)
        {
            _buffer.Clear();
            _callIndex = 0;
            if (enableThinking)
            {
                _state = State.CollectingThinking;
                _stripLeadingThinkTag = true;
            }
            else
            {
                _state = State.CollectingContent;
                _stripLeadingThinkTag = false;
            }
        }

        // 中文：流式接收新 token 文本，驱动状态机解析思考内容、正文内容及工具调用，返回本次增量解析结果
        public ParsedOutput Add(string text, bool done)
        {
            _buffer.Append(text);
            var result = new ParsedOutput();
            var thinkingSb = new StringBuilder();
            var contentSb = new StringBuilder();
            var toolCalls = new List<ToolCall>();

            bool keepParsing = true;
            while (keepParsing)
            {
                keepParsing = false;
                string buf = _buffer.ToString();

                switch (_state)
                {
                    case State.CollectingThinking:
                        if (_stripLeadingThinkTag)
                        {
                            string trimmed = buf.TrimStart();
                            if (trimmed.StartsWith("<think>"))
                            {
                                buf = trimmed.Substring(7).TrimStart();
                                _buffer.Clear();
                                _buffer.Append(buf);
                                _stripLeadingThinkTag = false;
                                keepParsing = buf.Length > 0;
                                break;
                            }
                            if ("<think>".StartsWith(trimmed) && !done)
                                break;
                            _stripLeadingThinkTag = false;
                        }

                        int closeIdx = buf.IndexOf("</think>", StringComparison.Ordinal);
                        int toolIdx = buf.IndexOf("<tool_call>", StringComparison.Ordinal);

                        if (toolIdx >= 0 && (closeIdx < 0 || toolIdx < closeIdx))
                        {
                            string before = buf.Substring(0, toolIdx).TrimEnd();
                            string after = buf.Substring(toolIdx + 11).TrimStart();
                            _buffer.Clear();
                            _buffer.Append(after);
                            if (before.Length > 0) thinkingSb.Append(before);
                            _state = State.CollectingTool;
                            keepParsing = true;
                        }
                        else if (closeIdx >= 0)
                        {
                            string thinking = buf.Substring(0, closeIdx).TrimEnd();
                            string after = buf.Substring(closeIdx + 8).TrimStart();
                            _buffer.Clear();
                            _buffer.Append(after);
                            if (thinking.Length > 0) thinkingSb.Append(thinking);
                            _state = after.Length > 0 ? State.CollectingContent : State.ThinkingDone;
                            keepParsing = after.Length > 0;
                        }
                        else if (done)
                        {
                            if (buf.Length > 0) thinkingSb.Append(buf);
                            _buffer.Clear();
                        }
                        else
                        {
                            int hold = HoldBackForPartialTag(buf, "</think>", "<tool_call>");
                            if (hold > 0)
                            {
                                string emit = buf.Substring(0, buf.Length - hold);
                                if (emit.Length > 0) thinkingSb.Append(emit);
                                _buffer.Clear();
                                _buffer.Append(buf.Substring(buf.Length - hold));
                            }
                            else
                            {
                                thinkingSb.Append(buf);
                                _buffer.Clear();
                            }
                        }
                        break;

                    case State.ThinkingDone:
                        string td = buf.TrimStart();
                        _buffer.Clear();
                        if (td.Length > 0)
                        {
                            _buffer.Append(td);
                            _state = State.CollectingContent;
                            keepParsing = true;
                        }
                        break;

                    case State.CollectingContent:
                        int tcIdx = buf.IndexOf("<tool_call>", StringComparison.Ordinal);
                        if (tcIdx >= 0)
                        {
                            string before = buf.Substring(0, tcIdx).TrimEnd();
                            string after = buf.Substring(tcIdx + 11).TrimStart();
                            _buffer.Clear();
                            _buffer.Append(after);
                            if (before.Length > 0) contentSb.Append(before);
                            _state = State.CollectingTool;
                            keepParsing = true;
                        }
                        else if (done)
                        {
                            if (buf.Length > 0) contentSb.Append(buf);
                            _buffer.Clear();
                        }
                        else
                        {
                            int hold = HoldBackForPartialTag(buf, "<tool_call>");
                            if (hold > 0)
                            {
                                string emit = buf.Substring(0, buf.Length - hold);
                                if (emit.Length > 0) contentSb.Append(emit);
                                _buffer.Clear();
                                _buffer.Append(buf.Substring(buf.Length - hold));
                            }
                            else
                            {
                                contentSb.Append(buf);
                                _buffer.Clear();
                            }
                        }
                        break;

                    case State.CollectingTool:
                        int endIdx = buf.IndexOf("</tool_call>", StringComparison.Ordinal);
                        if (endIdx >= 0)
                        {
                            string raw = buf.Substring(0, endIdx);
                            string after = buf.Substring(endIdx + 12).TrimStart();
                            _buffer.Clear();
                            _buffer.Append(after);
                            var tc = ParseQwen3ToolCall(raw);
                            if (tc != null) toolCalls.Add(tc);
                            _state = State.CollectingContent;
                            keepParsing = after.Length > 0;
                        }
                        else if (done && buf.Length > 0)
                        {
                            var tc = ParseQwen3ToolCall(buf);
                            if (tc != null) toolCalls.Add(tc);
                            _buffer.Clear();
                            _state = State.CollectingContent;
                        }
                        break;
                }
            }

            result.Content = contentSb.ToString();
            result.Thinking = thinkingSb.ToString();
            result.ToolCalls = toolCalls.Count > 0 ? toolCalls : null;
            return result;
        }

        // 中文：将 Qwen3 格式的工具调用原始 JSON 字符串解析为 ToolCall 对象，提取函数名和参数字典
        private ToolCall? ParseQwen3ToolCall(string raw)
        {
            raw = raw.Trim();
            if (raw.Length == 0) return null;
            try
            {
                using var doc = JsonDocument.Parse(raw);
                var root = doc.RootElement;
                string? name = root.GetProperty("name").GetString();
                if (string.IsNullOrEmpty(name)) return null;

                var args = new Dictionary<string, object>();
                if (root.TryGetProperty("arguments", out var argsEl) && argsEl.ValueKind == JsonValueKind.Object)
                {
                    foreach (var prop in argsEl.EnumerateObject())
                        args[prop.Name] = JsonElementToObject(prop.Value);
                }
                return new ToolCall { Name = name, Arguments = args, Index = _callIndex++ };
            }
            catch
            {
                return null;
            }
        }

        // 中文：计算缓冲区尾部与给定标签集合之间的最大前缀重叠长度，用于流式场景中防止标签被截断拆分而导致漏判
        private static int HoldBackForPartialTag(string buf, params string[] tags)
        {
            int maxOverlap = 0;
            foreach (var tag in tags)
            {
                int max = Math.Min(tag.Length, buf.Length);
                for (int i = max; i > 0; i--)
                {
                    if (buf.EndsWith(tag.Substring(0, i), StringComparison.Ordinal))
                    {
                        maxOverlap = Math.Max(maxOverlap, i);
                        break;
                    }
                }
            }
            return maxOverlap;
        }

        // 中文：将 JsonElement 递归转换为对应的 CLR 对象（字符串、数值、布尔、字典、列表等）
        internal static object JsonElementToObject(JsonElement el)
        {
            return el.ValueKind switch
            {
                JsonValueKind.String => el.GetString() ?? string.Empty,
                JsonValueKind.Number => el.TryGetInt64(out long l) ? (object)l : el.GetDouble(),
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                JsonValueKind.Null => null!,
                JsonValueKind.Object => JsonElementToDict(el),
                JsonValueKind.Array => JsonElementToList(el),
                _ => el.GetRawText()
            };
        }

        // 中文：将 JSON Object 类型的 JsonElement 转换为 Dictionary<string, object>
        private static Dictionary<string, object> JsonElementToDict(JsonElement el)
        {
            var d = new Dictionary<string, object>();
            foreach (var p in el.EnumerateObject())
                d[p.Name] = JsonElementToObject(p.Value);
            return d;
        }

        // 中文：将 JSON Array 类型的 JsonElement 转换为 List<object>
        private static List<object> JsonElementToList(JsonElement el)
        {
            var list = new List<object>();
            foreach (var item in el.EnumerateArray())
                list.Add(JsonElementToObject(item));
            return list;
        }
    }

    // ========================================================================
    // Qwen3.5 Parser: same tags as Qwen3, always starts in thinking mode
    // ========================================================================

    public class Qwen35OutputParser : Qwen3OutputParser
    {
    }

    // ========================================================================
    // Gemma4 Parser: <|channel>thought\n...<channel|> for thinking,
    //                <|tool_call>call:NAME{args}<tool_call|> for tool calls
    // ========================================================================

    public class Gemma4OutputParser : IOutputParser
    {
        private enum State { CollectingContent, CollectingThinking, CollectingToolCall }

        private State _state;
        private readonly StringBuilder _buffer = new();
        private bool _thinkingEnabled;
        private bool _needsChannelNameStrip;

        public bool HasThinkingSupport => true;
        public bool HasToolSupport => true;
        public bool AlwaysRequired => true;

        // 中文：初始化 Gemma4 解析器，清空缓冲区并设置思考模式开关，始终从正文收集状态开始
        public void Init(bool enableThinking, List<ToolFunction> tools)
        {
            _buffer.Clear();
            _thinkingEnabled = enableThinking;
            _needsChannelNameStrip = false;
            _state = State.CollectingContent;
        }

        // 中文：流式接收新 token，驱动 Gemma4 格式状态机，解析 <|channel> 思考块与 <|tool_call> 工具调用块
        public ParsedOutput Add(string text, bool done)
        {
            _buffer.Append(text);
            var result = new ParsedOutput();
            var thinkingSb = new StringBuilder();
            var contentSb = new StringBuilder();
            var toolCalls = new List<ToolCall>();

            bool keepParsing = true;
            while (keepParsing)
            {
                keepParsing = false;
                string buf = _buffer.ToString();
                if (buf.Length == 0) break;

                switch (_state)
                {
                    case State.CollectingContent:
                        int chIdx = buf.IndexOf("<|channel>", StringComparison.Ordinal);
                        int tcIdx = buf.IndexOf("<|tool_call>", StringComparison.Ordinal);

                        if (chIdx >= 0 && (tcIdx < 0 || chIdx < tcIdx))
                        {
                            string before = buf.Substring(0, chIdx).TrimEnd();
                            string after = buf.Substring(chIdx + 10);
                            _buffer.Clear();
                            _buffer.Append(after);
                            if (before.Length > 0) contentSb.Append(before);
                            _state = State.CollectingThinking;
                            _needsChannelNameStrip = true;
                            keepParsing = true;
                        }
                        else if (tcIdx >= 0)
                        {
                            string before = buf.Substring(0, tcIdx).TrimEnd();
                            string after = buf.Substring(tcIdx + 12);
                            _buffer.Clear();
                            _buffer.Append(after);
                            if (before.Length > 0) contentSb.Append(before);
                            _state = State.CollectingToolCall;
                            keepParsing = true;
                        }
                        else if (!done)
                        {
                            int hold = HoldBack(buf, "<|channel>", "<|tool_call>");
                            if (hold > 0)
                            {
                                string emit = buf.Substring(0, buf.Length - hold);
                                if (emit.Length > 0) contentSb.Append(emit);
                                _buffer.Clear();
                                _buffer.Append(buf.Substring(buf.Length - hold));
                            }
                            else
                            {
                                contentSb.Append(buf);
                                _buffer.Clear();
                            }
                        }
                        else
                        {
                            if (buf.Length > 0) contentSb.Append(buf);
                            _buffer.Clear();
                        }
                        break;

                    case State.CollectingThinking:
                        if (_needsChannelNameStrip)
                        {
                            if (buf.StartsWith("thought\n"))
                            {
                                buf = buf.Substring(8);
                                _buffer.Clear();
                                _buffer.Append(buf);
                                _needsChannelNameStrip = false;
                                keepParsing = buf.Length > 0;
                                break;
                            }
                            if (!done && ("thought\n".StartsWith(buf) || buf.StartsWith("thought")))
                                break;
                            _needsChannelNameStrip = false;
                        }

                        int closeIdx = buf.IndexOf("<channel|>", StringComparison.Ordinal);
                        if (closeIdx >= 0)
                        {
                            string thinking = buf.Substring(0, closeIdx).TrimEnd();
                            string after = buf.Substring(closeIdx + 10).TrimStart();
                            _buffer.Clear();
                            _buffer.Append(after);
                            if (thinking.Length > 0 && _thinkingEnabled) thinkingSb.Append(thinking);
                            _state = State.CollectingContent;
                            keepParsing = after.Length > 0;
                        }
                        else if (!done)
                        {
                            int hold = HoldBack(buf, "<channel|>");
                            if (hold > 0)
                            {
                                string emit = buf.Substring(0, buf.Length - hold);
                                if (emit.Length > 0 && _thinkingEnabled) thinkingSb.Append(emit);
                                _buffer.Clear();
                                _buffer.Append(buf.Substring(buf.Length - hold));
                            }
                            else
                            {
                                if (_thinkingEnabled) thinkingSb.Append(buf);
                                _buffer.Clear();
                            }
                        }
                        else
                        {
                            if (buf.Length > 0 && _thinkingEnabled) thinkingSb.Append(buf);
                            _buffer.Clear();
                        }
                        break;

                    case State.CollectingToolCall:
                        int endIdx = buf.IndexOf("<tool_call|>", StringComparison.Ordinal);
                        if (endIdx >= 0)
                        {
                            string raw = buf.Substring(0, endIdx);
                            string after = buf.Substring(endIdx + 12).TrimStart();
                            _buffer.Clear();
                            _buffer.Append(after);
                            var tc = ParseGemma4ToolCall(raw);
                            if (tc != null) toolCalls.Add(tc);
                            _state = State.CollectingContent;
                            keepParsing = after.Length > 0;
                        }
                        else if (done && buf.Length > 0)
                        {
                            var tc = ParseGemma4ToolCall(buf);
                            if (tc != null) toolCalls.Add(tc);
                            _buffer.Clear();
                            _state = State.CollectingContent;
                        }
                        break;
                }
            }

            result.Content = contentSb.ToString();
            result.Thinking = thinkingSb.ToString();
            result.ToolCalls = toolCalls.Count > 0 ? toolCalls : null;
            return result;
        }

        private static readonly Regex GemmaQuotedStringRe = new(@"<\|""\|>(.*?)<\|""\|>", RegexOptions.Singleline);
        private static readonly Regex GemmaBareKeyRe = new(@"([,{])(\w+):");

        // 中文：解析 Gemma4 格式的工具调用内容（call:NAME{args}），先提取函数名，再将参数转换为标准 JSON 后反序列化
        private static ToolCall? ParseGemma4ToolCall(string content)
        {
            content = content.Trim();
            if (!content.StartsWith("call:")) return null;
            content = content.Substring(5);

            int braceIdx = content.IndexOf('{');
            if (braceIdx < 0) return null;

            string name = content.Substring(0, braceIdx).Trim();
            string argsStr = content.Substring(braceIdx);

            string json = Gemma4ArgsToJson(argsStr);
            try
            {
                using var doc = JsonDocument.Parse(json);
                var args = new Dictionary<string, object>();
                foreach (var prop in doc.RootElement.EnumerateObject())
                    args[prop.Name] = Qwen3OutputParser.JsonElementToObject(prop.Value);
                return new ToolCall { Name = name, Arguments = args };
            }
            catch
            {
                return null;
            }
        }

        // 中文：将 Gemma4 特有的非标准参数格式（<|"|> 引号标记 + 裸键名）转换为标准 JSON 字符串，以便后续反序列化
        internal static string Gemma4ArgsToJson(string s)
        {
            var quotedStrings = new List<string>();
            string text = GemmaQuotedStringRe.Replace(s, m =>
            {
                quotedStrings.Add(m.Groups[1].Value);
                return "\x00" + (char)(quotedStrings.Count - 1) + "\x00";
            });

            text = GemmaBareKeyRe.Replace(text, "$1\"$2\":");

            for (int i = 0; i < quotedStrings.Count; i++)
            {
                string escaped = JsonSerializer.Serialize(quotedStrings[i]);
                text = text.Replace("\x00" + (char)i + "\x00", escaped);
            }

            return text;
        }

        // 中文：计算缓冲区尾部与给定标签集合之间的最大前缀重叠长度，防止流式输出中标签被截断拆分
        private static int HoldBack(string buf, params string[] tags)
        {
            int maxOverlap = 0;
            foreach (var tag in tags)
            {
                int max = Math.Min(tag.Length, buf.Length);
                for (int i = max; i > 0; i--)
                {
                    if (buf.EndsWith(tag.Substring(0, i), StringComparison.Ordinal))
                    {
                        maxOverlap = Math.Max(maxOverlap, i);
                        break;
                    }
                }
            }
            return maxOverlap;
        }
    }

    // ========================================================================
    // GPT OSS / Harmony Parser
    // Uses <|start|>...<|end|> message framing with <|message|> header end,
    // <|channel|>analysis for thinking, <|channel|>final for content
    // ========================================================================

    public class HarmonyOutputParser : IOutputParser
    {
        private enum HState { LookingForStart, ParsingHeader, ParsingContent }

        private HState _state;
        private readonly StringBuilder _buffer = new();
        private readonly StringBuilder _toolArgs = new();
        private string? _currentChannel;
        private string? _currentRecipient;
        private int _callIndex;

        private const string MsgStartTag = "<|start|>";
        private const string MsgEndTag = "<|end|>";
        private const string CallTag = "<|call|>";
        private const string ReturnTag = "<|return|>";
        private const string HeaderEndTag = "<|message|>";
        private const string ChannelTag = "<|channel|>";
        private const string FunctionPrefix = "functions.";

        // Tags that terminate a content message during generation.
        private static readonly string[] EndTags = { MsgEndTag, CallTag, ReturnTag };
        // Tags whose partial suffixes must be held back while streaming content.
        private static readonly string[] HoldTags = { MsgEndTag, CallTag, ReturnTag, MsgStartTag };

        public bool HasThinkingSupport => true;
        public bool HasToolSupport => true;
        public bool AlwaysRequired => true;

        // 中文：初始化 Harmony 解析器，预置 "<|start|>assistant" 到缓冲区以跳过提示中已有的起始标记，重置所有状态
        public void Init(bool enableThinking, List<ToolFunction> tools)
        {
            _buffer.Clear();
            _toolArgs.Clear();
            _state = HState.LookingForStart;
            _currentChannel = null;
            _currentRecipient = null;
            _callIndex = 0;

            // The prompt's generation marker is "<|start|>assistant", so the
            // model's first emitted token is "<|channel|>". Prime the buffer so
            // the parser is already past the start tag.
            _buffer.Append("<|start|>assistant");
        }

        // 中文：流式接收新 token，驱动 Harmony 消息帧状态机（查找起始标记 → 解析消息头 → 解析正文），提取思考/正文/工具调用
        public ParsedOutput Add(string text, bool done)
        {
            _buffer.Append(text);
            var result = new ParsedOutput();
            var contentSb = new StringBuilder();
            var thinkingSb = new StringBuilder();
            var toolCalls = new List<ToolCall>();

            bool keepParsing = true;
            while (keepParsing)
            {
                keepParsing = false;
                string buf = _buffer.ToString();
                if (buf.Length == 0) break;

                switch (_state)
                {
                    case HState.LookingForStart:
                        int startIdx = buf.IndexOf(MsgStartTag, StringComparison.Ordinal);
                        if (startIdx >= 0)
                        {
                            string after = buf.Substring(startIdx + MsgStartTag.Length);
                            _buffer.Clear();
                            _buffer.Append(after);
                            _state = HState.ParsingHeader;
                            keepParsing = true;
                        }
                        else if (!done)
                        {
                            int hold = HoldBack(buf, MsgStartTag);
                            if (hold > 0)
                            {
                                _buffer.Clear();
                                _buffer.Append(buf.Substring(buf.Length - hold));
                            }
                        }
                        break;

                    case HState.ParsingHeader:
                        int headerEnd = buf.IndexOf(HeaderEndTag, StringComparison.Ordinal);
                        if (headerEnd >= 0)
                        {
                            string header = buf.Substring(0, headerEnd);
                            string after = buf.Substring(headerEnd + HeaderEndTag.Length);
                            _buffer.Clear();
                            _buffer.Append(after);

                            ParseHeader(header);

                            _state = HState.ParsingContent;
                            keepParsing = after.Length > 0;
                        }
                        else if (!done)
                        {
                            int hold = HoldBack(buf, HeaderEndTag);
                            if (hold > 0 && hold < buf.Length)
                            {
                                _buffer.Clear();
                                _buffer.Append(buf.Substring(buf.Length - hold));
                            }
                        }
                        break;

                    case HState.ParsingContent:
                        int endIdx = FindEarliestEndTag(buf, out int tagLen);
                        if (endIdx >= 0)
                        {
                            string content = buf.Substring(0, endIdx);
                            string after = buf.Substring(endIdx + tagLen);
                            _buffer.Clear();
                            _buffer.Append(after);

                            EmitContent(content, contentSb, thinkingSb);
                            FinalizeMessage(toolCalls);
                            _state = HState.LookingForStart;
                            keepParsing = after.Length > 0;
                        }
                        else if (!done)
                        {
                            int hold = HoldBack(buf, HoldTags);
                            if (hold > 0)
                            {
                                string emit = buf.Substring(0, buf.Length - hold);
                                if (emit.Length > 0) EmitContent(emit, contentSb, thinkingSb);
                                _buffer.Clear();
                                _buffer.Append(buf.Substring(buf.Length - hold));
                            }
                            else
                            {
                                EmitContent(buf, contentSb, thinkingSb);
                                _buffer.Clear();
                            }
                        }
                        else
                        {
                            if (buf.Length > 0) EmitContent(buf, contentSb, thinkingSb);
                            FinalizeMessage(toolCalls);
                            _buffer.Clear();
                        }
                        break;
                }
            }

            result.Content = contentSb.ToString();
            result.Thinking = thinkingSb.ToString();
            if (toolCalls.Count > 0)
                result.ToolCalls = toolCalls;
            return result;
        }

        /// <summary>
        /// Parse a message header (the text between &lt;|start|&gt; and &lt;|message|&gt;)
        /// to extract the channel and, for tool calls, the "to=functions.NAME" recipient.
        /// Handles both header orderings (recipient before or after the channel tag).
        /// </summary>
        // 中文：从消息头文本中提取 <|channel|> 频道名称和 to=functions.NAME 收件人信息，支持两种字段顺序
        private void ParseHeader(string header)
        {
            int chIdx = header.IndexOf(ChannelTag, StringComparison.Ordinal);
            if (chIdx >= 0)
            {
                string channelPart = header.Substring(chIdx + ChannelTag.Length);
                int spaceIdx = channelPart.IndexOfAny(new[] { ' ', '\t', '\n', '\r' });
                _currentChannel = spaceIdx >= 0 ? channelPart.Substring(0, spaceIdx) : channelPart;
            }
            else
            {
                _currentChannel = "final";
            }

            _currentRecipient = null;
            int toIdx = header.IndexOf("to=", StringComparison.Ordinal);
            if (toIdx >= 0)
            {
                string rest = header.Substring(toIdx + 3);
                int end = 0;
                while (end < rest.Length && !char.IsWhiteSpace(rest[end]) && rest[end] != '<')
                    end++;
                if (end > 0)
                    _currentRecipient = rest.Substring(0, end);
            }
        }

        // 中文：根据当前频道将内容分发到思考缓冲区、正文缓冲区或工具参数缓冲区
        private void EmitContent(string content, StringBuilder contentSb, StringBuilder thinkingSb)
        {
            if (content.Length == 0) return;
            if (IsToolCall())
                _toolArgs.Append(content);
            else if (_currentChannel == "analysis")
                thinkingSb.Append(content);
            else
                contentSb.Append(content);
        }

        /// <summary>Finalize the current message: emit a tool call if it targeted functions.*.</summary>
        // 中文：完成当前消息处理，若收件人为 functions.* 则构建工具调用对象并加入结果列表，然后清空工具参数缓冲区
        private void FinalizeMessage(List<ToolCall> toolCalls)
        {
            if (IsToolCall())
            {
                var tc = BuildToolCall();
                if (tc != null)
                    toolCalls.Add(tc);
            }
            _toolArgs.Clear();
            _currentRecipient = null;
        }

        // 中文：判断当前消息是否为工具调用（即收件人以 "functions." 前缀开头）
        private bool IsToolCall() =>
            _currentRecipient != null && _currentRecipient.StartsWith(FunctionPrefix, StringComparison.Ordinal);

        // 中文：从收件人名称和工具参数缓冲区构建 ToolCall 对象，尝试将参数解析为 JSON，若解析失败则保留空参数字典
        private ToolCall? BuildToolCall()
        {
            string name = _currentRecipient!.Substring(FunctionPrefix.Length);
            if (string.IsNullOrEmpty(name)) return null;

            var args = new Dictionary<string, object>();
            string raw = _toolArgs.ToString().Trim();
            if (raw.Length > 0)
            {
                try
                {
                    using var doc = JsonDocument.Parse(raw);
                    if (doc.RootElement.ValueKind == JsonValueKind.Object)
                    {
                        foreach (var prop in doc.RootElement.EnumerateObject())
                            args[prop.Name] = Qwen3OutputParser.JsonElementToObject(prop.Value);
                    }
                }
                catch
                {
                    // Malformed JSON: surface the call with no parsed arguments
                    // rather than dropping it entirely.
                }
            }
            return new ToolCall { Name = name, Arguments = args, Index = _callIndex++ };
        }

        /// <summary>Find the earliest message-terminating tag in the buffer.</summary>
        // 中文：在缓冲区中线性搜索所有终止标签，返回最早出现的标签起始位置及其长度
        private static int FindEarliestEndTag(string buf, out int tagLen)
        {
            int best = -1;
            tagLen = 0;
            foreach (var tag in EndTags)
            {
                int idx = buf.IndexOf(tag, StringComparison.Ordinal);
                if (idx >= 0 && (best < 0 || idx < best))
                {
                    best = idx;
                    tagLen = tag.Length;
                }
            }
            return best;
        }

        // 中文：计算缓冲区尾部与给定标签集合之间的最大前缀重叠长度，用于 Harmony 格式的流式标签边界保护
        private static int HoldBack(string buf, params string[] tags)
        {
            int maxOverlap = 0;
            foreach (var tag in tags)
            {
                int max = Math.Min(tag.Length, buf.Length);
                for (int i = max; i > 0; i--)
                {
                    if (buf.EndsWith(tag.Substring(0, i), StringComparison.Ordinal))
                    {
                        maxOverlap = Math.Max(maxOverlap, i);
                        break;
                    }
                }
            }
            return maxOverlap;
        }
    }

    // ========================================================================
    // Passthrough parser (no thinking/tool parsing)
    // ========================================================================

    public class PassthroughOutputParser : IOutputParser
    {
        public bool HasThinkingSupport => false;
        public bool HasToolSupport => false;
        public bool AlwaysRequired => false;

        // 中文：直通解析器无需初始化操作，忽略所有参数
        public void Init(bool enableThinking, List<ToolFunction> tools) { }

        // 中文：直接将输入文本作为正文内容返回，不做任何标签解析
        public ParsedOutput Add(string text, bool done)
        {
            return new ParsedOutput { Content = text };
        }
    }

    // ========================================================================
    // Factory
    // ========================================================================

    public static class OutputParserFactory
    {
        // 中文：根据模型架构名称创建对应的输出解析器实例，未知架构默认返回直通解析器
        public static IOutputParser Create(string architecture)
        {
            return architecture switch
            {
                "gemma4" => new Gemma4OutputParser(),
                "qwen3" => new Qwen3OutputParser(),
                "qwen35" or "qwen35moe" or "qwen3next" or "qwen3vl" or "qwen3vlmoe" => new Qwen35OutputParser(),
                "gptoss" or "gpt-oss" => new HarmonyOutputParser(),
                "nemotron_h" or "nemotron_h_moe" => new Qwen3OutputParser(),
                _ => new PassthroughOutputParser()
            };
        }

        // 中文：判断给定架构是否要求解析器始终启用（即 AlwaysRequired 为 true 的架构）
        public static bool IsAlwaysRequired(string architecture)
        {
            return architecture is "gptoss" or "gpt-oss" or "gemma4";
        }
    }
}
