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
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Logging;

namespace TensorSharp.Server
{
    /// <summary>
    /// Thread-safe registry of <see cref="ChatSession"/> instances. The Web UI creates
    /// one session per chat; the Ollama and OpenAI compatibility endpoints share a
    /// single built-in "default" session that survives the lifetime of the server.
    ///
    /// The manager is intentionally a thin wrapper around a concurrent dictionary so
    /// that session lookup is lock-free on the inference hot path. Sessions track
    /// conversation history only; the inference engine owns all KV blocks.
    /// </summary>
    public sealed class SessionManager
    {
        public const string DefaultSessionId = "__default__";

        private readonly ConcurrentDictionary<string, ChatSession> _sessions = new(StringComparer.Ordinal);
        private readonly ILogger<SessionManager> _logger;

        // 中文：无参构造函数，使用空日志记录器委托给主构造函数。
        public SessionManager()
            : this(NullLogger<SessionManager>.Instance)
        {
        }

        // 中文：主构造函数，注入日志记录器并预先创建内置的默认会话。
        public SessionManager(ILogger<SessionManager> logger)
        {
            _logger = logger ?? NullLogger<SessionManager>.Instance;
            _sessions[DefaultSessionId] = new ChatSession(DefaultSessionId);
            _logger.LogDebug(LogEventIds.SessionCreated,
                "Default session {SessionId} created", DefaultSessionId);
        }

        /// <summary>
        /// Shared session used by stateless API clients (Ollama / OpenAI compatible
        /// endpoints). Never removed from the registry so raw assistant-token
        /// history remains available across requests for those clients.
        /// </summary>
        public ChatSession DefaultSession => _sessions[DefaultSessionId];

        /// <summary>Snapshot of the current session ids (for diagnostics / tests).</summary>
        public IReadOnlyList<string> SessionIds => _sessions.Keys.ToArray();

        public int SessionCount => _sessions.Count;

        /// <summary>
        /// Create a new session with a freshly-generated id. The returned session is
        /// already registered and can be looked up via <see cref="GetSession"/>.
        /// </summary>
        // 中文：创建并注册一个带新生成id的会话，循环重试直至成功加入注册表后返回。
        public ChatSession CreateSession()
        {
            while (true)
            {
                var session = new ChatSession();
                if (_sessions.TryAdd(session.Id, session))
                {
                    _logger.LogInformation(LogEventIds.SessionCreated,
                        "Created chat session {SessionId} (total sessions={SessionCount})",
                        session.Id, _sessions.Count);
                    return session;
                }
            }
        }

        /// <summary>
        /// Look up the session with the given id. Returns the default session when
        /// <paramref name="id"/> is null/empty (so endpoints can treat "no session"
        /// as "stateless client"). Returns null when the id is provided but no such
        /// session exists.
        /// </summary>
        // 中文：按id查找会话；id为空时返回默认会话，id无对应会话时返回null。
        public ChatSession GetSession(string id)
        {
            if (string.IsNullOrWhiteSpace(id))
                return DefaultSession;

            return _sessions.TryGetValue(id, out var session) ? session : null;
        }

        /// <summary>
        /// Remove the session from the registry and return it (without disposing it
        /// yet). The caller is expected to call <see cref="ModelService.DisposeSession"/>
        /// on the returned instance so the public session-lifecycle surface stays
        /// centralized.
        ///
        /// The default session cannot be removed; this method returns null for it.
        /// </summary>
        // 中文：从注册表移除指定会话并返回（不在此处释放）；默认会话不可移除，无匹配时返回null。
        public ChatSession TryRemove(string id)
        {
            if (string.IsNullOrWhiteSpace(id))
                return null;
            if (string.Equals(id, DefaultSessionId, StringComparison.Ordinal))
            {
                _logger.LogWarning(LogEventIds.SessionRemoved,
                    "Refusing to remove default session {SessionId}", DefaultSessionId);
                return null;
            }

            if (_sessions.TryRemove(id, out var session))
            {
                _logger.LogInformation(LogEventIds.SessionRemoved,
                    "Removed chat session {SessionId} (remaining sessions={SessionCount})",
                    id, _sessions.Count);
                return session;
            }

            _logger.LogDebug(LogEventIds.SessionRemoved,
                "TryRemove({SessionId}) found no matching session", id);
            return null;
        }
    }
}
