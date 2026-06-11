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
using System.IO;
using System.Linq;
using System.Threading;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

// ───────────────────────────────────────────────────────────────────────────
// 【文件说明】模型服务——服务端的对外业务门面。
// 【主要类型】ModelService：管理模型的加载 / 卸载与生命周期，持有会话与生成流水线，
//             向各协议适配器（Ollama / OpenAI / WebUI）提供统一的会话级推理调用入口。
// ───────────────────────────────────────────────────────────────────────────
namespace TensorSharp.Server
{
    public class ModelService : IDisposable
    {
        private readonly ModelLifecycleService _lifecycle;
        private readonly ChatSession _intrinsicSession;
        private readonly InferenceEngineHost _engineHost;
        private readonly ChatGenerationPipeline _generation;

        // 中文：无参构造函数，使用空日志记录器委托给主构造函数。
        public ModelService()
            : this(NullLogger<ModelService>.Instance)
        {
        }

        // 中文：主构造函数，装配提示渲染、生命周期服务、内置会话、引擎宿主与生成流水线等核心依赖。
        public ModelService(ILogger<ModelService> logger)
        {
            logger ??= NullLogger<ModelService>.Instance;

            var promptRenderer = new GgufPromptRenderer();
            var kvCacheRenderer = new KVCachePromptRenderer(promptRenderer);
            var telemetry = new InferenceTelemetry(logger);

            _lifecycle = new ModelLifecycleService(logger);
            _intrinsicSession = new ChatSession("__svc_intrinsic__");
            _engineHost = new InferenceEngineHost(_lifecycle, logger);
            _generation = new ChatGenerationPipeline(_lifecycle, _engineHost, kvCacheRenderer, telemetry, logger);
        }

        /// <summary>The internal lifecycle service. Exposed so the
        /// <see cref="InferenceEngineHost"/> can hook into model load/unload
        /// transitions; do not call this from other code paths.</summary>
        // 中文：暴露内部生命周期服务，仅供引擎宿主挂接模型加载/卸载状态转换之用。
        internal ModelLifecycleService LifecycleService => _lifecycle;

        public bool IsLoaded => _lifecycle.IsLoaded;
        public string LoadedModelName => _lifecycle.LoadedModelName;
        public string LoadedModelPath => _lifecycle.LoadedModelPath;
        public string LoadedMmProjName => _lifecycle.LoadedMmProjName;
        public string LoadedMmProjPath => _lifecycle.LoadedMmProjPath;
        public string LoadedBackend => _lifecycle.LoadedBackend;
        public string Architecture => _lifecycle.Architecture;
        public ModelBase Model => _lifecycle.Model;

        /// <summary>
        /// Legacy compatibility shim. The engine owns KV state, so no server
        /// session is ever active in the model.
        /// </summary>
        // 中文：遗留兼容属性，引擎自持KV状态，模型中从不存在活动会话，故恒返回null。
        public ChatSession ActiveSession => null;

        /// <summary>
        /// Legacy compatibility shim. Server-side session KV bookkeeping was
        /// removed; callers receive an isolated empty cache that is never used
        /// by inference.
        /// </summary>
        // 中文：遗留兼容属性，返回一个永不被推理使用的独立空缓存。
        public KVCache KVCache => new();

        /// <summary>
        /// Snapshot of the intrinsic compatibility session's tracked history.
        /// Session-aware requests use the explicit <see cref="ChatSession"/>
        /// instance passed to the generation methods.
        /// </summary>
        // 中文：返回内置兼容会话所跟踪历史的只读快照。
        public IReadOnlyList<ChatMessage> TrackedHistory => _intrinsicSession.TrackedHistory.AsReadOnly();

        // 中文：判断指定名称的模型是否已被加载，委托给生命周期服务。
        public bool IsModelAlreadyLoaded(string modelName)
        {
            return _lifecycle.IsModelAlreadyLoaded(modelName);
        }

        /// <summary>Engine host exposed for adapters that submit requests
        /// directly to the engine (e.g. multi-turn streaming clients that
        /// want to manage their own session bookkeeping).</summary>
        // 中文：暴露推理引擎宿主，供需自管会话记账、直接向引擎提交请求的适配器使用。
        public InferenceEngineHost EngineHost => _engineHost;

        // 中文：加载模型——先重置引擎再清空内置会话历史，最后委托生命周期服务执行实际加载，避免工作线程与模型释放竞争。
        public void LoadModel(string modelPath, string mmProjPath, string backendStr)
        {
            // Tear down the per-model engine BEFORE the model is unloaded so
            // the engine's worker thread doesn't race the model disposal.
            _engineHost.Reset();
            _intrinsicSession.TrackedHistory.Clear();
            _lifecycle.LoadModel(modelPath, mmProjPath, backendStr);
        }

        /// <summary>
        /// Legacy compatibility shim for older callers. There is no
        /// service-owned KV cache to invalidate; this clears only the intrinsic
        /// tracked history used by non-session-aware overloads.
        /// </summary>
        // 中文：遗留兼容方法，仅清空内置会话的跟踪历史，并无服务级KV缓存可作废。
        public void InvalidateKVCache()
        {
            _intrinsicSession.TrackedHistory.Clear();
        }

        /// <summary>
        /// Reset the given session's tracked conversation history. Engine-owned
        /// KV blocks are request-scoped and are not reset through this API.
        /// </summary>
        // 中文：重置指定会话的跟踪历史并更新其最后使用时间（引擎KV块按请求作用域不在此处理）。
        public void ResetSession(ChatSession session)
        {
            if (session == null)
                return;
            session.TrackedHistory.Clear();
            session.LastUsedAt = DateTime.UtcNow;
        }

        /// <summary>
        /// Dispose the given session and release its tracked history. The
        /// inference engine releases KV blocks independently of session
        /// disposal.
        /// </summary>
        // 中文：释放指定会话并清理其跟踪历史（引擎KV块独立于会话释放）。
        public void DisposeSession(ChatSession session)
        {
            session?.Dispose();
        }

        /// <summary>
        /// Stream chat inference tokens. Must be called within the InferenceQueue to prevent concurrent access.
        /// </summary>
        // 中文：流式聊天推理的非会话感知重载，转发至使用内置会话的会话感知重载。
        public IAsyncEnumerable<string> ChatStreamAsync(
            List<ChatMessage> history,
            int maxTokens,
            CancellationToken cancellationToken,
            SamplingConfig samplingConfig = null,
            List<ToolFunction> tools = null,
            bool enableThinking = false)
        {
            return ChatStreamAsync(_intrinsicSession, history, maxTokens, cancellationToken, samplingConfig, tools, enableThinking);
        }

        /// <summary>
        /// Stream chat inference tokens using the given <paramref name="session"/>'s
        /// tracked history. Must be called within the InferenceQueue.
        /// </summary>
        // 中文：会话感知的流式聊天推理，使用给定会话的跟踪历史，委托给生成流水线产出token流。
        public IAsyncEnumerable<string> ChatStreamAsync(
            ChatSession session,
            List<ChatMessage> history,
            int maxTokens,
            CancellationToken cancellationToken,
            SamplingConfig samplingConfig = null,
            List<ToolFunction> tools = null,
            bool enableThinking = false)
        {
            return _generation.ChatStreamAsync(session, history, maxTokens, cancellationToken, samplingConfig, tools, enableThinking);
        }

        /// <summary>
        /// Stream chat inference tokens with timing metrics. Must be called within the InferenceQueue.
        /// </summary>
        // 中文：带计时指标的流式聊天推理非会话感知重载，转发至使用内置会话的重载。
        public IAsyncEnumerable<(string piece, bool done, int promptTokens, int evalTokens, int kvCacheReusedTokens, long totalNs, long promptNs, long evalNs)>
            ChatStreamWithMetricsAsync(
                List<ChatMessage> history,
                int maxTokens,
                CancellationToken cancellationToken,
                SamplingConfig samplingConfig = null,
                List<ToolFunction> tools = null,
                bool enableThinking = false)
        {
            return ChatStreamWithMetricsAsync(_intrinsicSession, history, maxTokens, cancellationToken, samplingConfig, tools, enableThinking);
        }

        /// <summary>
        /// Session-aware overload of
        /// <see cref="ChatStreamWithMetricsAsync(List{ChatMessage}, int, CancellationToken, SamplingConfig, List{ToolFunction}, bool)"/>.
        /// </summary>
        // 中文：带计时指标的会话感知流式聊天推理，委托给生成流水线产出附带指标的token流。
        public IAsyncEnumerable<(string piece, bool done, int promptTokens, int evalTokens, int kvCacheReusedTokens, long totalNs, long promptNs, long evalNs)>
            ChatStreamWithMetricsAsync(
                ChatSession session,
                List<ChatMessage> history,
                int maxTokens,
                CancellationToken cancellationToken,
                SamplingConfig samplingConfig = null,
                List<ToolFunction> tools = null,
                bool enableThinking = false)
        {
            return _generation.ChatStreamWithMetricsAsync(session, history, maxTokens, cancellationToken, samplingConfig, tools, enableThinking);
        }

        /// <summary>
        /// Stream generate tokens. Must be called within the InferenceQueue to prevent concurrent access.
        /// Intended for one-shot completions and does not update session history.
        /// </summary>
        // 中文：一次性补全的流式生成非会话感知重载，转发至使用内置会话的重载，不更新会话历史。
        public IAsyncEnumerable<(string piece, bool done, int promptTokens, int evalTokens, int kvCacheReusedTokens, long totalNs, long promptNs, long evalNs)>
            GenerateStreamAsync(
                string prompt,
                List<string> imagePaths,
                int maxTokens,
                CancellationToken cancellationToken,
                SamplingConfig samplingConfig = null)
        {
            return GenerateStreamAsync(_intrinsicSession, prompt, imagePaths, maxTokens, cancellationToken, samplingConfig);
        }

        /// <summary>
        /// Session-aware streaming generate. Generate requests are treated as
        /// one-shot prompts and do not update tracked chat history.
        /// </summary>
        // 中文：会话感知的一次性流式生成，委托给生成流水线，将提示视为一次性输入且不更新跟踪历史。
        public IAsyncEnumerable<(string piece, bool done, int promptTokens, int evalTokens, int kvCacheReusedTokens, long totalNs, long promptNs, long evalNs)>
            GenerateStreamAsync(
                ChatSession session,
                string prompt,
                List<string> imagePaths,
                int maxTokens,
                CancellationToken cancellationToken,
                SamplingConfig samplingConfig = null)
        {
            return _generation.GenerateStreamAsync(session, prompt, imagePaths, maxTokens, cancellationToken, samplingConfig);
        }

        /// <summary>
        /// Instance-friendly shim that augments against the intrinsic compatibility
        /// session's tracked history. Prefer the static overload that takes an
        /// explicit tracked history for deterministic testing.
        /// </summary>
        // 中文：实例便捷封装，针对内置会话的跟踪历史增补缓存的原始token。
        internal List<ChatMessage> AugmentWithCachedRawTokens(List<ChatMessage> incoming)
        {
            return AugmentWithCachedRawTokens(incoming, _intrinsicSession.TrackedHistory);
        }

        // 中文：根据后端类型与token数解析预填充分块大小。
        internal static int ResolvePrefillChunkSize(BackendType backend, int tokenCount)
            => PrefillChunking.ResolveChunkSize(backend, tokenCount);

        // 中文：用给定跟踪历史中缓存的原始token增补传入消息列表（静态版本，便于确定性测试）。
        internal static List<ChatMessage> AugmentWithCachedRawTokens(
            List<ChatMessage> incoming,
            IReadOnlyList<ChatMessage> trackedHistory)
            => ChatHistoryPreparer.AugmentWithCachedRawTokens(incoming, trackedHistory);

        // 中文：按指定架构将聊天历史整理为可供推理使用的形式。
        internal static List<ChatMessage> PrepareHistoryForInference(List<ChatMessage> history, string arch)
            => ChatHistoryPreparer.PrepareHistoryForInference(history, arch);

        // 中文：带日志记录器的整理推理历史重载，便于诊断输出。
        internal static List<ChatMessage> PrepareHistoryForInference(List<ChatMessage> history, string arch, ILogger logger)
            => ChatHistoryPreparer.PrepareHistoryForInference(history, arch, logger);

        // 中文：判断单条消息是否包含多模态内容。
        internal static bool HasMultimodalContent(ChatMessage msg)
            => ChatHistoryPreparer.HasMultimodalContent(msg);

        // 中文：判断整段历史中是否包含多模态内容。
        internal static bool HasMultimodalContent(List<ChatMessage> history)
            => ChatHistoryPreparer.HasMultimodalContent(history);

        // 中文：按提示中出现顺序提取历史里的图像路径列表。
        internal static List<string> GetImagePathsInPromptOrder(List<ChatMessage> history)
            => ChatHistoryPreparer.GetImagePathsInPromptOrder(history);

        // 中文：将消息列表序列化为便于日志记录的字符串。
        internal static string SerializeMessagesForLog(List<ChatMessage> messages)
            => InferenceTelemetry.SerializeMessagesForLog(messages);

        // 中文：将消息中的上传内容序列化为便于日志记录的字符串。
        internal static string SerializeUploadsForLog(ChatMessage message)
            => InferenceTelemetry.SerializeUploadsForLog(message);

        // 中文：扫描目录中的*.gguf模型文件（排除mmproj投影文件）并按名称排序返回。
        public List<string> ScanModels(string directory)
        {
            if (!Directory.Exists(directory)) return new List<string>();
            return Directory.GetFiles(directory, "*.gguf")
                .Select(Path.GetFileName)
                .Where(f => !IsMmProjFile(f))
                .OrderBy(f => f)
                .ToList();
        }

        // 中文：扫描目录中的mmproj多模态投影文件并按名称排序返回。
        public List<string> ScanMmProjModels(string directory)
        {
            if (!Directory.Exists(directory)) return new List<string>();
            return Directory.GetFiles(directory, "*.gguf")
                .Select(Path.GetFileName)
                .Where(IsMmProjFile)
                .OrderBy(f => f)
                .ToList();
        }

        // 中文：释放资源——依次销毁引擎宿主、生命周期服务与内置会话。
        public void Dispose()
        {
            _engineHost.Dispose();
            _lifecycle.Dispose();
            _intrinsicSession.Dispose();
        }

        // 中文：通过文件名是否包含"mmproj"判断其为多模态投影文件。
        private static bool IsMmProjFile(string fileName)
        {
            return fileName.IndexOf("mmproj", StringComparison.OrdinalIgnoreCase) >= 0;
        }
    }
}
