// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Threading;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Server
{
    /// <summary>
    /// Owner of the per-model <see cref="InferenceEngine"/>. Lifecycle-bound to
    /// <see cref="ModelLifecycleService"/>: the engine is constructed lazily on
    /// first access (after a model has been loaded) and rebuilt whenever the
    /// model's KV-state fingerprint changes (i.e. on model swap). Disposing
    /// this service tears down the engine, which joins its worker thread and
    /// frees the paged KV block pool.
    ///
    /// This service is the public substitute for the legacy
    /// <see cref="InferenceQueue"/>: submission is non-blocking, multiple
    /// requests run concurrently (with iteration-level fairness), and the
    /// paged KV pool / continuous-batching scheduler / per-block prefix cache
    /// all live behind this single entry point. Adapters that haven't yet
    /// dropped queue-status chunks still take <see cref="InferenceQueue"/>
    /// tickets, but those tickets grant immediately so the engine remains the
    /// only real concurrency boundary.
    /// </summary>
    public sealed class InferenceEngineHost : IDisposable
    {
        private readonly ModelLifecycleService _lifecycle;
        private readonly ILogger _logger;
        private readonly object _gate = new();
        private InferenceEngine _engine;
        private string _fingerprint;
        private bool _disposed;

        // 中文：构造函数，绑定模型生命周期服务与日志记录器（生命周期为空时抛出异常）。
        internal InferenceEngineHost(ModelLifecycleService lifecycle, ILogger logger)
        {
            _lifecycle = lifecycle ?? throw new ArgumentNullException(nameof(lifecycle));
            _logger = logger ?? NullLogger.Instance;
        }

        /// <summary>Get the engine for the currently-loaded model, constructing
        /// it if it hasn't been built yet (or rebuilding it if the model has
        /// changed). Returns null when no model is loaded or when the model
        /// supports neither the KV-state snapshot contract nor the batched
        /// paged-attention contract.
        ///
        /// Models that implement <see cref="IBatchedPagedModel"/> serve
        /// parallel requests via <c>ForwardBatch</c> and don't need to swap
        /// KV state between sequences, so they qualify even when
        /// <see cref="ModelBase.SupportsKVStateSnapshot"/> reports false.</summary>
        // 中文：获取当前模型的推理引擎，按KV状态指纹懒构建/重建；模型不支持相关契约或未加载时返回null。
        public InferenceEngine TryGetEngine()
        {
            var model = _lifecycle.Model;
            if (model == null) return null;
            if (!model.SupportsKVStateSnapshot && model is not IBatchedPagedModel) return null;

            string fp = model.KVStateFingerprint ?? string.Empty;
            lock (_gate)
            {
                if (_disposed) return null;
                if (_engine != null && string.Equals(_fingerprint, fp, StringComparison.Ordinal))
                    return _engine;

                _engine?.Dispose();
                var cfg = SchedulerConfig.FromEnvironment();
                _engine = new InferenceEngine(model, cfg, _logger);
                _fingerprint = fp;
                _logger.LogInformation(
                    "InferenceEngine constructed for fingerprint {Fingerprint} (blocks={NumBlocks}, blockSize={BlockSize}, maxBatched={MaxBatched})",
                    fp, cfg.NumBlocks, cfg.BlockSize, cfg.MaxNumBatchedTokens);
                return _engine;
            }
        }

        /// <summary>Drop the engine (if any). Called by <see cref="ModelLifecycleService"/>
        /// when the model is unloaded so we don't hold onto a stale block pool.</summary>
        // 中文：在模型卸载时销毁并清空当前引擎及其指纹，避免持有陈旧的KV块池。
        public void Reset()
        {
            lock (_gate)
            {
                _engine?.Dispose();
                _engine = null;
                _fingerprint = null;
            }
        }

        // 中文：释放宿主——加锁标记已释放并销毁引擎（连带终结工作线程、释放分页KV块池）。
        public void Dispose()
        {
            lock (_gate)
            {
                if (_disposed) return;
                _disposed = true;
                _engine?.Dispose();
                _engine = null;
            }
        }
    }
}
