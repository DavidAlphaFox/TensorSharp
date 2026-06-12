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
// 文件：ModelMultimodalInjector.cs
// 用途：多模态嵌入注入器，负责将图像/音频输入经由各模型专属视觉/音频编码器编码后，
//       以嵌入张量形式注入到 LLM 提示词 token 序列中，支持并发请求隔离（per-requestId bucket）
//       以及 Qwen3.5 多轴旋转位置编码（MRoPE）位置表的构建与切片下发。
//       支持的模型系列：Gemma4、Gemma3、Qwen3.5、Mistral3、Nemotron。
// ────────────────────────

using System;
using System.Collections.Generic;
using System.IO;
using TensorSharp;

namespace TensorSharp.Models
{
    internal sealed class ModelMultimodalInjector : IMultimodalInjector, IDisposable
    {
        private readonly ModelBase _model;
        private readonly Dictionary<string, CachedEmbedding> _visionCache = new(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, CachedEmbedding> _audioCache = new(StringComparer.OrdinalIgnoreCase);

        // Per-request buckets. "" is the default bucket used by direct
        // single-threaded callers (for example InteractiveSession);
        // engine-path callers pass a unique requestId so concurrent requests
        // don't clobber each other's prepared embeddings. Mutations are guarded
        // by _bucketLock because the engine's per-seq Forward (driven by the
        // worker thread inside BatchExecutor) can race the request-thread's
        // ProcessPromptTokens for a different request.
        private readonly object _bucketLock = new();
        private readonly Dictionary<string, List<PreparedEmbeddingSpan>> _visionByRequest = new();
        private readonly Dictionary<string, List<PreparedEmbeddingSpan>> _audioByRequest = new();

        // Per-request flat [T0,H0,W0, T1,H1,W1, ...] position table for
        // interleaved MRoPE-using models (Qwen3.5). Populated by Process*History
        // when an image is in the prompt, then sliced and pushed to the model
        // alongside vision-embedding queueing so prefill RoPE can apply the
        // right per-axis positions to the right rotary dims. Null/missing entry
        // means the request is text-only and standard scalar RoPE is fine.
        private readonly Dictionary<string, int[]> _mropePositionsByRequest = new();

        // "Live view" into the bucket for the request currently being processed
        // by ProcessPromptTokens. The model-specific Process*History helpers
        // append to these. Reset to the default bucket between requests.
        private List<PreparedEmbeddingSpan> _preparedVisionEmbeddings;
        private List<PreparedEmbeddingSpan> _preparedAudioEmbeddings;
        // RequestId currently being processed. Set at ProcessPromptTokens entry
        // so the model-specific Process*History helpers (which don't take a
        // requestId arg) can attach per-request state like the MRoPE position
        // table to the right bucket. Single-threaded under the chat pipeline's
        // GpuComputeLock so no races.
        private string _currentRequestId;

        private sealed class CachedEmbedding : IDisposable
        {
            // 中文：构造缓存嵌入条目，记录文件路径、大小、修改时间戳及编码后的嵌入张量
            public CachedEmbedding(
                string fullPath,
                long fileSize,
                long lastWriteUtcTicks,
                Tensor embeddings,
                int tokenCount,
                int extra0 = 0,
                int extra1 = 0)
            {
                FullPath = fullPath;
                FileSize = fileSize;
                LastWriteUtcTicks = lastWriteUtcTicks;
                Embeddings = embeddings;
                TokenCount = tokenCount;
                Extra0 = extra0;
                Extra1 = extra1;
            }

            public string FullPath { get; }
            public long FileSize { get; }
            public long LastWriteUtcTicks { get; }
            public Tensor Embeddings { get; }
            public int TokenCount { get; }
            public int Extra0 { get; }
            public int Extra1 { get; }

            // 中文：通过文件大小和最后修改时间戳判断缓存是否仍然有效
            public bool Matches(long fileSize, long lastWriteUtcTicks) =>
                FileSize == fileSize && LastWriteUtcTicks == lastWriteUtcTicks;

            // 中文：释放持有的嵌入张量资源
            public void Dispose()
            {
                Embeddings?.Dispose();
            }
        }

        private sealed class PreparedEmbeddingSpan
        {
            // 中文：构造已准备好的嵌入区间，记录缓存条目、插入位置及对应的提示词 token 范围
            public PreparedEmbeddingSpan(
                CachedEmbedding cacheEntry,
                int insertPosition,
                int promptTokenStart,
                int promptTokenEndExclusive)
            {
                CacheEntry = cacheEntry;
                InsertPosition = insertPosition;
                PromptTokenStart = promptTokenStart;
                PromptTokenEndExclusive = promptTokenEndExclusive;
            }

            public CachedEmbedding CacheEntry { get; }
            public int InsertPosition { get; set; }
            public int PromptTokenStart { get; set; }
            public int PromptTokenEndExclusive { get; set; }
            public int EndPosition => InsertPosition + CacheEntry.TokenCount;
        }

        // 中文：构造注入器，初始化默认请求桶（空字符串 key）的视觉与音频嵌入列表
        public ModelMultimodalInjector(ModelBase model)
        {
            _model = model;
            _preparedVisionEmbeddings = GetOrCreateBucket(_visionByRequest, "");
            _preparedAudioEmbeddings = GetOrCreateBucket(_audioByRequest, "");
        }

        // 中文：将 null requestId 规范化为空字符串，用于 dictionary 键统一查找
        private static string NormalizeRequestId(string requestId) => requestId ?? "";

        // 中文：线程安全地获取或创建指定 requestId 对应的嵌入区间列表桶
        private List<PreparedEmbeddingSpan> GetOrCreateBucket(
            Dictionary<string, List<PreparedEmbeddingSpan>> buckets, string requestId)
        {
            lock (_bucketLock)
            {
                if (!buckets.TryGetValue(requestId, out var list))
                {
                    list = new List<PreparedEmbeddingSpan>();
                    buckets[requestId] = list;
                }
                return list;
            }
        }

        // 中文：根据模型类型加载对应的视觉编码器（和音频编码器）投影权重文件
        public void LoadProjectors(string mmProjPath)
        {
            if (string.IsNullOrWhiteSpace(mmProjPath))
                return;

            switch (_model)
            {
                case Gemma4Model g4:
                    g4.LoadVisionEncoder(mmProjPath);
                    g4.LoadAudioEncoder(mmProjPath);
                    break;
                case Gemma3Model g3:
                    g3.LoadVisionEncoder(mmProjPath);
                    break;
                case Qwen35Model q35:
                    q35.LoadVisionEncoder(mmProjPath);
                    break;
                case Mistral3Model m3:
                    m3.LoadVisionEncoder(mmProjPath);
                    break;
                case NemotronModel nem:
                    nem.LoadVisionEncoder(mmProjPath);
                    break;
            }
        }

        // 中文：处理对话历史中的多模态输入，将图像/音频占位符 token 展开并准备对应嵌入区间
        public List<int> ProcessPromptTokens(List<ChatMessage> history, List<int> inputTokens, string requestId = null)
        {
            string key = NormalizeRequestId(requestId);
            _preparedVisionEmbeddings = GetOrCreateBucket(_visionByRequest, key);
            _preparedAudioEmbeddings = GetOrCreateBucket(_audioByRequest, key);
            _currentRequestId = key;

            _preparedVisionEmbeddings.Clear();
            _preparedAudioEmbeddings.Clear();

            if (history == null || history.Count == 0 || inputTokens == null || inputTokens.Count == 0)
                return inputTokens;

            if (_model is Gemma4Model g4)
                return ProcessGemma4History(g4, history, inputTokens);
            if (_model is Gemma3Model g3)
                return ProcessGemma3History(g3, history, inputTokens);
            if (_model is Qwen35Model q35)
                return ProcessQwen35History(q35, history, inputTokens);
            if (_model is Mistral3Model m3)
                return ProcessMistral3History(m3, history, inputTokens);
            if (_model is NemotronModel nem)
                return ProcessNemotronHistory(nem, history, inputTokens);

            return inputTokens;
        }

        // 中文：将可复用前缀长度向前截断，确保不切断任何图像/音频嵌入区间
        public int ClampReusablePrefix(int reusablePrefixTokenCount, string requestId = null)
        {
            string key = NormalizeRequestId(requestId);
            var visionBucket = GetOrCreateBucket(_visionByRequest, key);
            var audioBucket = GetOrCreateBucket(_audioByRequest, key);
            int clamped = ClampReusablePrefix(reusablePrefixTokenCount, visionBucket);
            clamped = ClampReusablePrefix(clamped, audioBucket);
            return clamped;
        }

        // 中文：将起始裁剪长度向后扩展，确保不在嵌入区间中途裁断
        public int ClampTrimStart(int trimStartTokenCount, string requestId = null)
        {
            string key = NormalizeRequestId(requestId);
            var visionBucket = GetOrCreateBucket(_visionByRequest, key);
            var audioBucket = GetOrCreateBucket(_audioByRequest, key);
            int clamped = ClampTrimStart(trimStartTokenCount, visionBucket);
            clamped = ClampTrimStart(clamped, audioBucket);
            return clamped;
        }

        // 中文：对指定请求的已准备嵌入区间执行起始裁剪，并更新各区间的偏移量
        public void TrimPreparedPrompt(int trimStartTokenCount, string requestId = null)
        {
            string key = NormalizeRequestId(requestId);
            TrimPreparedPrompt(GetOrCreateBucket(_visionByRequest, key), trimStartTokenCount);
            TrimPreparedPrompt(GetOrCreateBucket(_audioByRequest, key), trimStartTokenCount);
        }

        // 中文：将当前请求的全部视觉和音频嵌入推送到模型（跳过已被可复用前缀覆盖的区间）
        public bool QueuePromptEmbeddings(int reusablePrefixTokenCount, string requestId = null)
        {
            string key = NormalizeRequestId(requestId);
            var visionBucket = GetOrCreateBucket(_visionByRequest, key);
            var audioBucket = GetOrCreateBucket(_audioByRequest, key);
            bool queued = QueuePreparedVisionEmbeddings(visionBucket, reusablePrefixTokenCount);
            queued |= QueuePreparedAudioEmbeddings(audioBucket, reusablePrefixTokenCount);
            return queued;
        }

        // 中文：将与指定 token 切片范围重叠的嵌入行推送到模型，同时下发 MRoPE 位置切片（用于 Qwen3.5 分页推理）
        public bool QueuePromptEmbeddingsForSlice(int promptStartToken, int tokenCount, string requestId = null)
        {
            if (tokenCount <= 0)
                return false;
            if (promptStartToken < 0)
                throw new ArgumentOutOfRangeException(nameof(promptStartToken));

            long promptEndToken = (long)promptStartToken + tokenCount;
            if (promptEndToken > int.MaxValue)
                promptEndToken = int.MaxValue;

            string key = NormalizeRequestId(requestId);
            var visionBucket = GetOrCreateBucket(_visionByRequest, key);
            var audioBucket = GetOrCreateBucket(_audioByRequest, key);
            bool queued = QueuePreparedVisionEmbeddingsForSlice(visionBucket, promptStartToken, (int)promptEndToken);
            queued |= QueuePreparedAudioEmbeddingsForSlice(audioBucket, promptStartToken, (int)promptEndToken);

            // Also push the matching slice of MRoPE positions onto the model
            // so the upcoming Forward call can apply interleaved per-axis
            // rotations to image-region rotary dims. Text-only requests skip
            // this (TryGet returns null) and the model uses scalar positions.
            int[] mropeSlice = TryGetMRoPEPositionsForSlice(requestId, promptStartToken, tokenCount);
            if (mropeSlice != null && _model is Qwen35Model q35)
            {
                q35.SetMRoPEPositions(mropeSlice);
                queued = true;
            }
            return queued;
        }

        /// <summary>Store the flat (T,H,W) position table for a request. Length
        /// must equal 3 * promptTokenCount. Pass null to clear.</summary>
        // 中文：存储或清除指定请求的多轴旋转位置编码（MRoPE）位置表，长度须为 3 * promptTokenCount
        internal void SetMRoPEPositions(string requestId, int[] flatThw)
        {
            string key = NormalizeRequestId(requestId);
            lock (_bucketLock)
            {
                if (flatThw == null) _mropePositionsByRequest.Remove(key);
                else _mropePositionsByRequest[key] = flatThw;
            }
        }

        /// <summary>Slice the request's MRoPE position table for the prompt range
        /// [promptStartToken, promptStartToken + tokenCount). Returns null if the
        /// request has no MRoPE positions (text-only request).</summary>
        // 中文：从请求的 MRoPE 全量位置表中切取指定 token 范围的子数组，纯文本请求返回 null
        internal int[] TryGetMRoPEPositionsForSlice(string requestId, int promptStartToken, int tokenCount)
        {
            if (tokenCount <= 0) return null;
            string key = NormalizeRequestId(requestId);
            int[] full;
            lock (_bucketLock)
            {
                if (!_mropePositionsByRequest.TryGetValue(key, out full) || full == null)
                    return null;
            }
            int total = full.Length / 3;
            if (promptStartToken >= total) return null;
            int end = Math.Min(promptStartToken + tokenCount, total);
            int len = end - promptStartToken;
            if (len <= 0) return null;
            int[] slice = new int[len * 3];
            Buffer.BlockCopy(full, promptStartToken * 3 * sizeof(int), slice, 0, len * 3 * sizeof(int));
            return slice;
        }

        // 中文：检查指定请求是否仍有未消费的视觉或音频嵌入等待推送
        public bool HasPendingEmbeddings(string requestId)
        {
            string key = NormalizeRequestId(requestId);
            lock (_bucketLock)
            {
                if (_visionByRequest.TryGetValue(key, out var vision) && vision.Count > 0)
                    return true;
                if (_audioByRequest.TryGetValue(key, out var audio) && audio.Count > 0)
                    return true;
                return false;
            }
        }

        // 中文：清除指定请求的所有已准备嵌入状态及 MRoPE 位置表，并释放非默认请求的桶条目
        public void ClearPreparedPromptState(string requestId)
        {
            string key = NormalizeRequestId(requestId);
            lock (_bucketLock)
            {
                if (_visionByRequest.TryGetValue(key, out var vision))
                    vision.Clear();
                if (_audioByRequest.TryGetValue(key, out var audio))
                    audio.Clear();
                _mropePositionsByRequest.Remove(key);
                if (key.Length > 0)
                {
                    // Drop the buckets entirely so a finished request doesn't leak
                    // dictionary entries. The default bucket ("") stays around.
                    _visionByRequest.Remove(key);
                    _audioByRequest.Remove(key);
                }
            }
        }

        // 中文：处理 Gemma4 模型的对话历史，展开图像/音频占位符并准备对应嵌入区间
        private List<int> ProcessGemma4History(Gemma4Model model, List<ChatMessage> history, List<int> inputTokens)
        {
            int imageStartId = _model.Tokenizer.LookupToken("<|image>");
            int imageEndId = _model.Tokenizer.LookupToken("<image|>");
            if (imageStartId < 0) imageStartId = 255999;
            if (imageEndId < 0) imageEndId = 256000;

            int audioStartId = _model.Tokenizer.LookupToken("<|audio>");
            int audioEndId = _model.Tokenizer.LookupToken("<audio|>");

            // The gemma4uv unified embedder declares its own image_mean / image_std
            // (mean=0, std=1 -> [0,1]); the gemma4v SigLIP path keeps the legacy
            // [-1,1] normalization.
            var imageProcessor = model.VisionEncoder != null
                ? (model.VisionEncoder.IsUnified
                    ? new Gemma4ImageProcessor(imageMean: model.VisionEncoder.ImageMean,
                        imageStd: model.VisionEncoder.ImageStd)
                    : new Gemma4ImageProcessor())
                : null;
            int searchFrom = 0;

            foreach (var message in history)
            {
                if (message.ImagePaths != null && model.VisionEncoder != null)
                {
                    foreach (var imagePath in message.ImagePaths)
                    {
                        CachedEmbedding cached = GetOrCreateGemma4VisionEmbedding(model, imageProcessor, imagePath);
                        int tokenPosition = FindTokenPosition(inputTokens, imageStartId, searchFrom);

                        if (tokenPosition >= 0)
                        {
                            inputTokens = ExpandSingleTokenPlaceholder(inputTokens, tokenPosition, imageStartId, cached.TokenCount, imageEndId);
                            _preparedVisionEmbeddings.Add(new PreparedEmbeddingSpan(
                                cached,
                                tokenPosition + 1,
                                tokenPosition,
                                tokenPosition + cached.TokenCount + 2));
                            searchFrom = tokenPosition + cached.TokenCount + 2;
                        }
                    }
                }

                if (message.AudioPaths != null && model.AudioEncoder != null && audioStartId >= 0 && audioEndId >= 0)
                {
                    foreach (var audioPath in message.AudioPaths)
                    {
                        CachedEmbedding cached = GetOrCreateGemma4AudioEmbedding(model, audioPath);
                        int tokenPosition = FindTokenPosition(inputTokens, audioStartId, searchFrom);

                        if (tokenPosition >= 0)
                        {
                            inputTokens = ExpandSingleTokenPlaceholder(inputTokens, tokenPosition, audioStartId, cached.TokenCount, audioEndId);
                            _preparedAudioEmbeddings.Add(new PreparedEmbeddingSpan(
                                cached,
                                tokenPosition + 1,
                                tokenPosition,
                                tokenPosition + cached.TokenCount + 2));
                            searchFrom = tokenPosition + cached.TokenCount + 2;
                        }
                    }
                }
            }

            return inputTokens;
        }

        // 中文：处理 Gemma3 模型的对话历史，将图像 token 序列展开为 SigLIP patch token 并准备嵌入区间
        private List<int> ProcessGemma3History(Gemma3Model model, List<ChatMessage> history, List<int> inputTokens)
        {
            if (model.VisionEncoder == null)
                return inputTokens;

            var imagePaths = GetImagePathsInPromptOrder(history);
            if (imagePaths.Count == 0)
                return inputTokens;

            var processor = new Gemma3ImageProcessor();
            int startId = _model.Tokenizer.LookupToken("<start_of_image>");
            if (startId < 0) startId = Gemma3ImageProcessor.StartOfImageToken;
            int endId = Gemma3ImageProcessor.EndOfImageToken;
            int newlineId = Gemma3ImageProcessor.NewlineNewlineToken;
            int padId = Gemma3ImageProcessor.PadToken;

            inputTokens = ChatTemplate.ExpandGemma3ImageTokens(
                inputTokens,
                startId,
                endId,
                newlineId,
                padId,
                processor.TokensPerImage);

            int searchFrom = 0;
            foreach (var imagePath in imagePaths)
            {
                CachedEmbedding cached = GetOrCreateGemma3VisionEmbedding(model, processor, imagePath);
                int tokenStart = FindGemma3ImageInsertPosition(inputTokens, startId, padId, searchFrom);

                if (tokenStart >= 0)
                {
                    _preparedVisionEmbeddings.Add(new PreparedEmbeddingSpan(
                        cached,
                        tokenStart,
                        tokenStart - 2,
                        tokenStart + cached.TokenCount + 2));
                    searchFrom = tokenStart + processor.TokensPerImage + 2;
                }
            }

            return inputTokens;
        }

        // 中文：处理 Qwen3.5 模型的对话历史，展开图像 pad token 并构建多轴旋转位置编码（MRoPE）位置表
        // 算法说明：MRoPE 为图像 token 分配三维位置 (T, H, W)，文本 token 三轴相同（标量位置），
        //           图像 token 按 patch 网格坐标分配 H/W 轴，T 轴固定为图像起始文本位置，
        //           图像结束后文本标量跳至 max(H, W) + imgBase，避免位置重叠。
        private List<int> ProcessQwen35History(Qwen35Model model, List<ChatMessage> history, List<int> inputTokens)
        {
            if (model.VisionEncoder == null)
                return inputTokens;

            var imagePaths = GetImagePathsInPromptOrder(history);
            if (imagePaths.Count == 0)
                return inputTokens;

            int imagePadId = _model.Tokenizer.LookupToken("<|image_pad|>");
            if (imagePadId < 0)
                return inputTokens;

            var processor = new Qwen35ImageProcessor(model.VisionEncoder.PatchSize, model.VisionEncoder.SpatialMergeSize);
            var cachedEmbeddings = new CachedEmbedding[imagePaths.Count];
            var tokenCounts = new int[imagePaths.Count];
            for (int i = 0; i < imagePaths.Count; i++)
            {
                cachedEmbeddings[i] = GetOrCreateQwen35VisionEmbedding(model, processor, imagePaths[i]);
                tokenCounts[i] = cachedEmbeddings[i].TokenCount;
            }

            inputTokens = ChatTemplate.ExpandImageTokens(inputTokens, imagePadId, tokenCounts);

            // Build the per-token (T,H,W) MRoPE position table for the entire
            // expanded prompt. vLLM Qwen3.5 (MRotaryEmbedding.get_input_positions)
            // assigns positions like this:
            //  - text tokens get (k, k, k) where k is the running scalar position
            //  - image tokens at merged grid coords (h, w) get
            //      (text_pos, text_pos + h, text_pos + w)
            //    where text_pos is the running scalar at the image's start;
            //    after the image, the running scalar resumes at
            //      max(H, W) + text_pos
            //    so the next text token gets (next_k, next_k, next_k) with no
            //    overlap. For static images T axis stays at text_pos.
            int total = inputTokens.Count;
            int[] thw = new int[3 * total];
            int searchFrom = 0;
            int textPos = 0;
            int writeIdx = 0;
            int imgIdx = 0;
            while (writeIdx < total)
            {
                int imgStart = (imgIdx < imagePaths.Count)
                    ? FindTokenPosition(inputTokens, imagePadId, searchFrom)
                    : -1;
                int textEnd = imgStart >= 0 ? imgStart : total;

                // text run [writeIdx, textEnd) - collapse all three axes
                for (int t = writeIdx; t < textEnd; t++)
                {
                    thw[3 * t + 0] = textPos;
                    thw[3 * t + 1] = textPos;
                    thw[3 * t + 2] = textPos;
                    textPos++;
                }

                if (imgStart < 0) break;

                int mergedH = cachedEmbeddings[imgIdx].Extra0;
                int mergedW = cachedEmbeddings[imgIdx].Extra1;
                int imgTokenCount = tokenCounts[imgIdx];
                if (mergedH * mergedW != imgTokenCount)
                {
                    Console.WriteLine($"[qwen35-mrope] image {imgIdx} grid {mergedH}x{mergedW}={mergedH * mergedW} " +
                                      $"≠ token count {imgTokenCount}; falling back to text-only positions");
                    for (int t = imgStart; t < imgStart + imgTokenCount; t++)
                    {
                        thw[3 * t + 0] = textPos;
                        thw[3 * t + 1] = textPos;
                        thw[3 * t + 2] = textPos;
                        textPos++;
                    }
                }
                else
                {
                    int imgBase = textPos;
                    for (int h = 0; h < mergedH; h++)
                    {
                        for (int w = 0; w < mergedW; w++)
                        {
                            int t = imgStart + h * mergedW + w;
                            thw[3 * t + 0] = imgBase;        // T axis: constant for a single image
                            thw[3 * t + 1] = imgBase + h;    // H axis
                            thw[3 * t + 2] = imgBase + w;    // W axis
                        }
                    }
                    // After the image, the running scalar jumps past the
                    // image's max H/W so subsequent text tokens don't alias
                    // image positions.
                    textPos = imgBase + Math.Max(mergedH, mergedW);
                }

                _preparedVisionEmbeddings.Add(new PreparedEmbeddingSpan(
                    cachedEmbeddings[imgIdx],
                    imgStart,
                    imgStart,
                    imgStart + imgTokenCount));

                writeIdx = imgStart + imgTokenCount;
                searchFrom = imgStart + imgTokenCount;
                imgIdx++;
            }

            // Stash on the injector so QueuePromptEmbeddingsForSlice can push
            // the right slice into the model just before each Forward call.
            string key = NormalizeRequestId(_currentRequestId);
            lock (_bucketLock)
            {
                _mropePositionsByRequest[key] = thw;
            }

            return inputTokens;
        }

        // 中文：处理 Mistral3 模型的对话历史，将图像展开为行列格式的 patch token 网格并准备嵌入区间
        private List<int> ProcessMistral3History(Mistral3Model model, List<ChatMessage> history, List<int> inputTokens)
        {
            if (model.VisionEncoder == null)
                return inputTokens;

            var imagePaths = GetImagePathsInPromptOrder(history);
            if (imagePaths.Count == 0)
                return inputTokens;

            var processor = new Mistral3ImageProcessor(
                model.VisionEncoder.ImageSize,
                model.VisionEncoder.PatchSize);

            int searchFrom = 0;
            foreach (var imagePath in imagePaths)
            {
                CachedEmbedding cached = GetOrCreateMistral3VisionEmbedding(model, processor, imagePath);
                int numRows = cached.Extra0;
                int numCols = cached.Extra1;

                int tokenPosition = FindTokenPosition(inputTokens, Mistral3ImageProcessor.ImgTokenId, searchFrom);
                if (tokenPosition < 0)
                    continue;

                var expanded = new List<int>(inputTokens.Count + numRows * numCols + numRows);
                for (int i = 0; i < tokenPosition; i++)
                    expanded.Add(inputTokens[i]);

                for (int row = 0; row < numRows; row++)
                {
                    for (int col = 0; col < numCols; col++)
                        expanded.Add(Mistral3ImageProcessor.ImgTokenId);

                    expanded.Add(row == numRows - 1
                        ? Mistral3ImageProcessor.ImgEndTokenId
                        : Mistral3ImageProcessor.ImgBreakTokenId);
                }

                for (int i = tokenPosition + 1; i < inputTokens.Count; i++)
                    expanded.Add(inputTokens[i]);

                _preparedVisionEmbeddings.Add(new PreparedEmbeddingSpan(
                    cached,
                    tokenPosition,
                    tokenPosition,
                    tokenPosition + numRows * numCols + numRows));

                inputTokens = expanded;
                searchFrom = tokenPosition + numRows * numCols + numRows;
            }

            return inputTokens;
        }

        // 中文：处理 Nemotron 模型的对话历史，将图像占位符展开为分块 tile 拼接嵌入并准备区间
        private List<int> ProcessNemotronHistory(NemotronModel model, List<ChatMessage> history, List<int> inputTokens)
        {
            if (model.VisionEncoder == null)
                return inputTokens;

            int imageTokenId = _model.Tokenizer.LookupToken("<image>");
            int imageStartId = _model.Tokenizer.LookupToken("<img>");
            int imageEndId = _model.Tokenizer.LookupToken("</img>");
            if (imageTokenId < 0) imageTokenId = 18;
            if (imageStartId < 0) imageStartId = 19;
            if (imageEndId < 0) imageEndId = 20;

            int searchFrom = 0;
            foreach (var message in history)
            {
                if (message.ImagePaths != null && message.ImagePaths.Count > 0)
                {
                    foreach (var imagePath in message.ImagePaths)
                    {
                        if (string.IsNullOrEmpty(imagePath))
                            continue;

                        CachedEmbedding cached = GetOrCreateNemotronVisionEmbedding(model, imagePath);
                        int tokenPosition = FindTokenPosition(inputTokens, imageTokenId, searchFrom);
                        if (tokenPosition < 0)
                            continue;

                        inputTokens = ExpandSingleTokenPlaceholder(
                            inputTokens, tokenPosition, imageStartId, cached.TokenCount, imageEndId);

                        // Insertion point is right after the start sentinel token.
                        _preparedVisionEmbeddings.Add(new PreparedEmbeddingSpan(
                            cached,
                            tokenPosition + 1,
                            tokenPosition,
                            tokenPosition + cached.TokenCount + 2));

                        searchFrom = tokenPosition + cached.TokenCount + 2;
                    }
                }

                // Audio path: the chat template emits a `<so_embedding>` per uploaded
                // audio file so the model "sees" the modality, but real inference is
                // gated on a Parakeet audio mmproj that this distribution does not ship.
                // The test data still gets preprocessed in the CLI for verification.
            }

            return inputTokens;
        }

        // 中文：获取或创建 Nemotron 模型的视觉嵌入缓存，对图像分块编码后拼接为单一张量
        private CachedEmbedding GetOrCreateNemotronVisionEmbedding(NemotronModel model, string imagePath)
        {
            return GetOrCreateCachedEmbedding(_visionCache, imagePath, fullPath =>
            {
                var processor = model.ImageProcessor;
                var tiles = processor.ProcessImage(fullPath);
                if (tiles.Count == 0)
                    throw new InvalidOperationException($"Image '{fullPath}' produced zero vision tiles.");

                // Encode each tile and concatenate into a single [totalTokens, hidden] tensor
                // so a single PreparedEmbeddingSpan covers the whole image.
                var tileEmbeddings = new Tensor[tiles.Count];
                int totalTokens = 0;
                int hidden = 0;
                try
                {
                    for (int i = 0; i < tiles.Count; i++)
                    {
                        var tile = tiles[i];
                        tileEmbeddings[i] = model.VisionEncoder.Encode(tile.Pixels, tile.Width, tile.Height);
                        totalTokens += (int)tileEmbeddings[i].Sizes[0];
                        if (i == 0)
                            hidden = (int)tileEmbeddings[i].Sizes[1];
                    }

                    var concatenated = new Tensor(tileEmbeddings[0].Allocator, DType.Float32, totalTokens, hidden);
                    int offset = 0;
                    for (int i = 0; i < tileEmbeddings.Length; i++)
                    {
                        int rows = (int)tileEmbeddings[i].Sizes[0];
                        using var slice = concatenated.Narrow(0, offset, rows);
                        Ops.Copy(slice, tileEmbeddings[i]);
                        offset += rows;
                    }

                    return CreateCachedEmbedding(fullPath, concatenated);
                }
                finally
                {
                    foreach (var t in tileEmbeddings) t?.Dispose();
                }
            });
        }

        // 中文：获取或创建 Gemma4 模型的视觉嵌入缓存（支持统一嵌入器和 SigLIP 两条路径）
        private CachedEmbedding GetOrCreateGemma4VisionEmbedding(
            Gemma4Model model,
            Gemma4ImageProcessor processor,
            string imagePath)
        {
            return GetOrCreateCachedEmbedding(_visionCache, imagePath, fullPath =>
            {
                var (pixels, imageWidth, imageHeight) = processor.ProcessImage(fullPath);
                Tensor embeddings = model.VisionEncoder.Encode(pixels, imageWidth, imageHeight);
                return CreateCachedEmbedding(fullPath, embeddings);
            });
        }

        // 中文：获取或创建 Gemma4 模型的音频嵌入缓存，支持无编码器直接投影和 mel 频谱两种路径
        private CachedEmbedding GetOrCreateGemma4AudioEmbedding(Gemma4Model model, string audioPath)
        {
            return GetOrCreateCachedEmbedding(_audioCache, audioPath, fullPath =>
            {
                float[] samples = Gemma4AudioPreprocessor.DecodeAudioFile(fullPath);

                // Gemma 4 "unified" models (projector_type "gemma4ua", e.g.
                // gemma-4-12b) are encoder-free: the raw waveform is chunked into
                // 640-sample frames and projected directly, with no mel
                // spectrogram or conformer encoder.
                if (model.AudioEncoder.IsEncoderFree)
                {
                    Tensor rawEmbeddings = model.AudioEncoder.EncodeRawWaveform(samples);
                    return CreateCachedEmbedding(fullPath, rawEmbeddings);
                }

                if (samples.Length % 128 != 0)
                {
                    int padded = samples.Length + (128 - samples.Length % 128);
                    Array.Resize(ref samples, padded);
                }

                var (melData, numFrames) = Gemma4AudioPreprocessor.ComputeMelSpectrogram(samples);
                if (melData == null || numFrames == 0)
                    throw new InvalidOperationException($"Audio file '{fullPath}' did not produce a valid mel spectrogram.");

                Tensor embeddings = model.AudioEncoder.Encode(melData, numFrames);
                return CreateCachedEmbedding(fullPath, embeddings);
            });
        }

        // 中文：获取或创建 Gemma3 模型的视觉嵌入缓存，使用 SigLIP 视觉编码器对图像进行编码
        private CachedEmbedding GetOrCreateGemma3VisionEmbedding(
            Gemma3Model model,
            Gemma3ImageProcessor processor,
            string imagePath)
        {
            return GetOrCreateCachedEmbedding(_visionCache, imagePath, fullPath =>
            {
                float[] pixels = processor.ProcessImage(fullPath);
                Tensor embeddings = model.VisionEncoder.Encode(pixels);
                return CreateCachedEmbedding(fullPath, embeddings);
            });
        }

        // 中文：获取或创建 Qwen3.5 模型的视觉嵌入缓存，记录合并后的网格尺寸（mergedH, mergedW）用于 MRoPE
        private CachedEmbedding GetOrCreateQwen35VisionEmbedding(
            Qwen35Model model,
            Qwen35ImageProcessor processor,
            string imagePath)
        {
            return GetOrCreateCachedEmbedding(_visionCache, imagePath, fullPath =>
            {
                var (pixels, resizedHeight, resizedWidth) = processor.ProcessImage(fullPath);
                Tensor embeddings = model.VisionEncoder.Encode(pixels, resizedHeight, resizedWidth);
                int mergedH = resizedHeight / processor.PatchSize / processor.MergeSize;
                int mergedW = resizedWidth / processor.PatchSize / processor.MergeSize;
                return CreateCachedEmbedding(fullPath, embeddings, mergedH, mergedW);
            });
        }

        // 中文：获取或创建 Mistral3 模型的视觉嵌入缓存，记录 patch 网格行列数用于 token 展开
        private CachedEmbedding GetOrCreateMistral3VisionEmbedding(
            Mistral3Model model,
            Mistral3ImageProcessor processor,
            string imagePath)
        {
            return GetOrCreateCachedEmbedding(_visionCache, imagePath, fullPath =>
            {
                var (pixels, imageWidth, imageHeight) = processor.ProcessImage(fullPath);
                Tensor embeddings = model.VisionEncoder.Encode(pixels, imageWidth, imageHeight);
                int numRows = imageHeight / model.VisionEncoder.PatchSize / model.VisionEncoder.SpatialMergeSize;
                int numCols = imageWidth / model.VisionEncoder.PatchSize / model.VisionEncoder.SpatialMergeSize;
                return CreateCachedEmbedding(fullPath, embeddings, numRows, numCols);
            });
        }

        // 中文：通用缓存获取/创建逻辑，基于文件大小和修改时间判断缓存有效性，过期则重新编码
        private CachedEmbedding GetOrCreateCachedEmbedding(
            Dictionary<string, CachedEmbedding> cache,
            string path,
            Func<string, CachedEmbedding> factory)
        {
            string fullPath = NormalizePath(path);
            GetMediaVersion(fullPath, out long fileSize, out long lastWriteUtcTicks);

            if (cache.TryGetValue(fullPath, out var cached) && cached.Matches(fileSize, lastWriteUtcTicks))
                return cached;

            cached?.Dispose();
            CachedEmbedding fresh = factory(fullPath);
            cache[fullPath] = fresh;
            return fresh;
        }

        // 中文：创建 CachedEmbedding 对象，从文件系统读取版本信息并封装嵌入张量
        private static CachedEmbedding CreateCachedEmbedding(string fullPath, Tensor embeddings, int extra0 = 0, int extra1 = 0)
        {
            GetMediaVersion(fullPath, out long fileSize, out long lastWriteUtcTicks);
            return new CachedEmbedding(
                fullPath,
                fileSize,
                lastWriteUtcTicks,
                embeddings,
                (int)embeddings.Sizes[0],
                extra0,
                extra1);
        }

        // 中文：将已准备的视觉嵌入推送到模型，跳过完全在可复用前缀范围内的嵌入区间
        private bool QueuePreparedVisionEmbeddings(List<PreparedEmbeddingSpan> bucket, int reusablePrefixTokenCount)
        {
            if (bucket.Count == 0)
                return false;

            bool queued = false;

            switch (_model)
            {
                case Gemma4Model g4:
                    foreach (var span in bucket)
                    {
                        if (span.EndPosition <= reusablePrefixTokenCount)
                            continue;

                        g4.SetVisionEmbeddings(CloneTensor(span.CacheEntry.Embeddings), span.InsertPosition - reusablePrefixTokenCount);
                        queued = true;
                    }
                    break;
                case Gemma3Model g3:
                    foreach (var span in bucket)
                    {
                        if (span.EndPosition <= reusablePrefixTokenCount)
                            continue;

                        g3.SetVisionEmbeddings(CloneTensor(span.CacheEntry.Embeddings), span.InsertPosition - reusablePrefixTokenCount);
                        queued = true;
                    }
                    break;
                case Qwen35Model q35:
                    foreach (var span in bucket)
                    {
                        if (span.EndPosition <= reusablePrefixTokenCount)
                            continue;

                        q35.SetVisionEmbeddings(CloneTensor(span.CacheEntry.Embeddings), span.InsertPosition - reusablePrefixTokenCount);
                        queued = true;
                    }
                    break;
                case Mistral3Model m3:
                    foreach (var span in bucket)
                    {
                        if (span.EndPosition <= reusablePrefixTokenCount)
                            continue;

                        m3.SetVisionEmbeddings(CloneTensor(span.CacheEntry.Embeddings), span.InsertPosition - reusablePrefixTokenCount);
                        queued = true;
                    }
                    break;
                case NemotronModel nem:
                    foreach (var span in bucket)
                    {
                        if (span.EndPosition <= reusablePrefixTokenCount)
                            continue;

                        nem.SetVisionEmbeddings(CloneTensor(span.CacheEntry.Embeddings), span.InsertPosition - reusablePrefixTokenCount);
                        queued = true;
                    }
                    break;
            }

            return queued;
        }

        // 中文：将已准备的音频嵌入推送到 Gemma4 模型，跳过完全在可复用前缀范围内的嵌入区间
        private bool QueuePreparedAudioEmbeddings(List<PreparedEmbeddingSpan> bucket, int reusablePrefixTokenCount)
        {
            if (bucket.Count == 0 || _model is not Gemma4Model g4)
                return false;

            bool queued = false;
            foreach (var span in bucket)
            {
                if (span.EndPosition <= reusablePrefixTokenCount)
                    continue;

                g4.SetAudioEmbeddings(CloneTensor(span.CacheEntry.Embeddings), span.InsertPosition - reusablePrefixTokenCount);
                queued = true;
            }

            return queued;
        }

        // 中文：将与指定 token 切片范围重叠的视觉嵌入行推送到模型（用于分块 prefill 场景）
        private bool QueuePreparedVisionEmbeddingsForSlice(List<PreparedEmbeddingSpan> bucket, int promptStartToken, int promptEndToken)
        {
            if (bucket.Count == 0)
                return false;

            bool queued = false;

            switch (_model)
            {
                case Gemma4Model g4:
                    foreach (var span in bucket)
                    {
                        if (!TryCloneOverlappingEmbeddingRows(span, promptStartToken, promptEndToken,
                                out Tensor embeddings, out int insertPosition))
                            continue;

                        g4.SetVisionEmbeddings(embeddings, insertPosition);
                        queued = true;
                    }
                    break;
                case Gemma3Model g3:
                    foreach (var span in bucket)
                    {
                        if (!TryCloneOverlappingEmbeddingRows(span, promptStartToken, promptEndToken,
                                out Tensor embeddings, out int insertPosition))
                            continue;

                        g3.SetVisionEmbeddings(embeddings, insertPosition);
                        queued = true;
                    }
                    break;
                case Qwen35Model q35:
                    foreach (var span in bucket)
                    {
                        if (!TryCloneOverlappingEmbeddingRows(span, promptStartToken, promptEndToken,
                                out Tensor embeddings, out int insertPosition))
                            continue;

                        q35.SetVisionEmbeddings(embeddings, insertPosition);
                        queued = true;
                    }
                    break;
                case Mistral3Model m3:
                    foreach (var span in bucket)
                    {
                        if (!TryCloneOverlappingEmbeddingRows(span, promptStartToken, promptEndToken,
                                out Tensor embeddings, out int insertPosition))
                            continue;

                        m3.SetVisionEmbeddings(embeddings, insertPosition);
                        queued = true;
                    }
                    break;
                case NemotronModel nem:
                    foreach (var span in bucket)
                    {
                        if (!TryCloneOverlappingEmbeddingRows(span, promptStartToken, promptEndToken,
                                out Tensor embeddings, out int insertPosition))
                            continue;

                        nem.SetVisionEmbeddings(embeddings, insertPosition);
                        queued = true;
                    }
                    break;
            }

            return queued;
        }

        // 中文：将与指定 token 切片范围重叠的音频嵌入行推送到 Gemma4 模型（用于分块 prefill 场景）
        private bool QueuePreparedAudioEmbeddingsForSlice(List<PreparedEmbeddingSpan> bucket, int promptStartToken, int promptEndToken)
        {
            if (bucket.Count == 0 || _model is not Gemma4Model g4)
                return false;

            bool queued = false;
            foreach (var span in bucket)
            {
                if (!TryCloneOverlappingEmbeddingRows(span, promptStartToken, promptEndToken,
                        out Tensor embeddings, out int insertPosition))
                    continue;

                g4.SetAudioEmbeddings(embeddings, insertPosition);
                queued = true;
            }

            return queued;
        }

        // 中文：将可复用前缀长度截断至不切断任何嵌入区间的最大安全值（内部静态重载）
        private static int ClampReusablePrefix(int prefixTokenCount, List<PreparedEmbeddingSpan> spans)
        {
            if (prefixTokenCount <= 0 || spans.Count == 0)
                return prefixTokenCount;

            int clamped = prefixTokenCount;
            foreach (var span in spans)
            {
                if (clamped > span.InsertPosition && clamped < span.EndPosition)
                    clamped = Math.Min(clamped, span.InsertPosition);
            }

            return clamped;
        }

        // 中文：将起始裁剪长度扩展至不在嵌入区间中途截断的最小安全值（内部静态重载）
        private static int ClampTrimStart(int trimStartTokenCount, List<PreparedEmbeddingSpan> spans)
        {
            if (trimStartTokenCount <= 0 || spans.Count == 0)
                return trimStartTokenCount;

            int clamped = trimStartTokenCount;
            foreach (var span in spans)
            {
                if (clamped > span.PromptTokenStart && clamped < span.PromptTokenEndExclusive)
                    clamped = Math.Max(clamped, span.PromptTokenEndExclusive);
            }

            return clamped;
        }

        // 中文：从已准备的嵌入区间列表中移除完全在裁剪范围内的区间，并更新剩余区间的偏移量
        private static void TrimPreparedPrompt(List<PreparedEmbeddingSpan> spans, int trimStartTokenCount)
        {
            if (trimStartTokenCount <= 0 || spans.Count == 0)
                return;

            for (int i = spans.Count - 1; i >= 0; i--)
            {
                PreparedEmbeddingSpan span = spans[i];
                if (span.PromptTokenEndExclusive <= trimStartTokenCount)
                {
                    spans.RemoveAt(i);
                    continue;
                }

                span.InsertPosition -= trimStartTokenCount;
                span.PromptTokenStart -= trimStartTokenCount;
                span.PromptTokenEndExclusive -= trimStartTokenCount;
            }
        }

        // 中文：清空所有请求的视觉和音频嵌入桶（不释放缓存张量，仅清空列表）
        private void ClearAllPreparedPromptState()
        {
            lock (_bucketLock)
            {
                foreach (var bucket in _visionByRequest.Values) bucket.Clear();
                foreach (var bucket in _audioByRequest.Values) bucket.Clear();
            }
        }

        // 中文：读取媒体文件的版本信息（文件大小与最后修改时间），文件不存在时返回哨兵值
        private static void GetMediaVersion(string fullPath, out long fileSize, out long lastWriteUtcTicks)
        {
            if (File.Exists(fullPath))
            {
                var fileInfo = new FileInfo(fullPath);
                fileSize = fileInfo.Length;
                lastWriteUtcTicks = fileInfo.LastWriteTimeUtc.Ticks;
                return;
            }

            fileSize = -1;
            lastWriteUtcTicks = 0;
        }

        // 中文：将输入路径规范化为绝对路径，空白路径返回空字符串
        private static string NormalizePath(string path)
        {
            if (string.IsNullOrWhiteSpace(path))
                return path ?? string.Empty;

            return Path.GetFullPath(path);
        }

        // 中文：深拷贝张量，在相同设备/分配器上创建新张量并复制数据
        private static Tensor CloneTensor(Tensor source)
        {
            var clone = new Tensor(source.Allocator, source.ElementType, source.Sizes);
            Ops.Copy(clone, source);
            return clone;
        }

        // 中文：计算嵌入区间与 prompt 切片的重叠部分，克隆重叠行并返回切片内的插入位置
        private static bool TryCloneOverlappingEmbeddingRows(
            PreparedEmbeddingSpan span,
            int promptStartToken,
            int promptEndToken,
            out Tensor embeddings,
            out int insertPosition)
        {
            embeddings = null;
            insertPosition = 0;

            int overlapStart = Math.Max(promptStartToken, span.InsertPosition);
            int overlapEnd = Math.Min(promptEndToken, span.EndPosition);
            if (overlapStart >= overlapEnd)
                return false;

            int sourceStart = overlapStart - span.InsertPosition;
            int rowCount = overlapEnd - overlapStart;
            insertPosition = overlapStart - promptStartToken;
            embeddings = CloneTensorRows(span.CacheEntry.Embeddings, sourceStart, rowCount);
            return true;
        }

        // 中文：克隆张量的指定行范围，若覆盖全部行则直接调用 CloneTensor 避免额外切片开销
        private static Tensor CloneTensorRows(Tensor source, int startRow, int rowCount)
        {
            if (startRow == 0 && rowCount == source.Sizes[0])
                return CloneTensor(source);

            using var rows = source.Narrow(0, startRow, rowCount);
            var clone = new Tensor(source.Allocator, source.ElementType, rows.Sizes);
            Ops.Copy(clone, rows);
            return clone;
        }

        // 中文：按对话顺序收集所有消息中的图像路径列表，过滤空路径
        private static List<string> GetImagePathsInPromptOrder(List<ChatMessage> history)
        {
            var imagePaths = new List<string>();
            if (history == null)
                return imagePaths;

            foreach (var message in history)
            {
                if (message.ImagePaths == null)
                    continue;

                foreach (var path in message.ImagePaths)
                {
                    if (!string.IsNullOrEmpty(path))
                        imagePaths.Add(path);
                }
            }

            return imagePaths;
        }

        // 中文：将单个占位符 token 展开为 [startToken, 0*N, endToken] 形式的 token 序列
        private static List<int> ExpandSingleTokenPlaceholder(
            List<int> inputTokens, int tokenPosition, int startTokenId, int expandedTokenCount, int endTokenId)
        {
            var expanded = new List<int>(inputTokens.Count + expandedTokenCount + 1);
            for (int i = 0; i < tokenPosition; i++)
                expanded.Add(inputTokens[i]);
            expanded.Add(startTokenId);
            for (int i = 0; i < expandedTokenCount; i++)
                expanded.Add(0);
            expanded.Add(endTokenId);
            for (int i = tokenPosition + 1; i < inputTokens.Count; i++)
                expanded.Add(inputTokens[i]);
            return expanded;
        }

        // 中文：从指定起始位置线性扫描 token 列表，返回目标 token 的首个匹配位置，未找到返回 -1
        private static int FindTokenPosition(List<int> tokens, int tokenId, int searchFrom)
        {
            for (int i = Math.Max(0, searchFrom); i < tokens.Count; i++)
            {
                if (tokens[i] == tokenId)
                    return i;
            }

            return -1;
        }

        // 中文：查找 Gemma3 图像的嵌入插入位置（紧跟 startToken 后的第一个 padToken 位置）
        private static int FindGemma3ImageInsertPosition(List<int> tokens, int startTokenId, int padTokenId, int searchFrom)
        {
            for (int i = Math.Max(0, searchFrom); i + 1 < tokens.Count; i++)
            {
                if (tokens[i] == startTokenId && tokens[i + 1] == padTokenId)
                    return i + 1;
            }

            return -1;
        }

        // 中文：释放所有已缓存的视觉和音频嵌入张量，并清空缓存字典
        public void Dispose()
        {
            ClearAllPreparedPromptState();

            foreach (var cached in _visionCache.Values)
                cached.Dispose();
            _visionCache.Clear();

            foreach (var cached in _audioCache.Values)
                cached.Dispose();
            _audioCache.Clear();
        }
    }
}
