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
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Disk-backed second tier for the paged KV cache. Blocks evicted from the
    /// in-memory tier flow here (asynchronously, on a background writer thread)
    /// and are re-read synchronously on lookup miss. The on-disk layout is a
    /// hash-partitioned tree (<c>{root}/{hash[0:2]}/{hash}.kvb</c>) - this avoids
    /// pathological single-directory performance on platforms like APFS / NTFS.
    ///
    /// Format (little-endian):
    ///   bytes 0..3   : magic 0x544B564B ("TKVK" - TensorSharp KV-block)
    ///   bytes 4..7   : format version (currently 1)
    ///   bytes 8..15  : payload byte length
    ///   bytes 16..23 : 8-byte fingerprint hash (for cross-model collision rejection)
    ///   bytes 24..   : raw payload
    /// </summary>
    internal sealed class SsdKvBlockTier : IDisposable
    {
        private const uint Magic = 0x544B564Bu; // "TKVK"
        private const int FormatVersion = 1;
        private const int HeaderSize = 24;

        private readonly string _rootDir;
        private readonly long _maxBytes;
        private readonly ulong _fingerprintHash;
        private readonly ILogger _logger;
        private readonly object _gate = new();
        private readonly LinkedList<DiskEntry> _lru = new();
        private readonly Dictionary<KvBlockHash, LinkedListNode<DiskEntry>> _index = new();
        private long _residentBytes;

        private readonly BlockingCollection<WriteJob> _writeQueue;
        private readonly Thread _writerThread;
        private volatile bool _disposed;

        // 中文：构造 SSD 溢出层，校验参数、计算指纹哈希、创建根目录、重建已有文件索引并启动后台写线程。
        public SsdKvBlockTier(string rootDir, long maxBytes, string fingerprint, ILogger logger = null)
        {
            if (string.IsNullOrWhiteSpace(rootDir))
                throw new ArgumentException("Root directory must be a non-empty path.", nameof(rootDir));
            if (maxBytes <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxBytes));

            _rootDir = rootDir;
            _maxBytes = maxBytes;
            _fingerprintHash = StableFingerprintHash(fingerprint ?? string.Empty);
            _logger = logger ?? NullLogger.Instance;

            Directory.CreateDirectory(_rootDir);
            ReindexExistingFiles();

            _writeQueue = new BlockingCollection<WriteJob>(boundedCapacity: 256);
            _writerThread = new Thread(WriterLoop)
            {
                Name = "TensorSharp KV SSD writer",
                IsBackground = true,
            };
            _writerThread.Start();
        }

        public long ResidentBytes
        {
            get { lock (_gate) return _residentBytes; }
        }

        public int Count
        {
            get { lock (_gate) return _index.Count; }
        }

        // 中文：按哈希同步读取磁盘上的 KV 块，命中则更新 LRU 顺序并校验头部后返回原始负载。
        public bool TryRead(KvBlockHash hash, out byte[] payload)
        {
            string path = PathFor(hash);
            lock (_gate)
            {
                if (!_index.TryGetValue(hash, out var node))
                {
                    payload = null;
                    return false;
                }
                _lru.Remove(node);
                _lru.AddFirst(node);
            }

            try
            {
                using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
                Span<byte> header = stackalloc byte[HeaderSize];
                if (ReadExact(fs, header) != HeaderSize)
                {
                    payload = null;
                    return false;
                }

                uint magic = BinaryPrimitives.ReadUInt32LittleEndian(header);
                int version = BinaryPrimitives.ReadInt32LittleEndian(header[4..]);
                long payloadLength = BinaryPrimitives.ReadInt64LittleEndian(header[8..]);
                ulong fingerprint = BinaryPrimitives.ReadUInt64LittleEndian(header[16..]);

                if (magic != Magic || version != FormatVersion || fingerprint != _fingerprintHash || payloadLength <= 0 || payloadLength > int.MaxValue)
                {
                    payload = null;
                    return false;
                }

                payload = new byte[payloadLength];
                int read = ReadExact(fs, payload);
                if (read != payloadLength)
                {
                    payload = null;
                    return false;
                }
                return true;
            }
            catch (IOException ex)
            {
                _logger.LogWarning(ex, "Failed to read SSD KV cache block {Hash}", hash);
                payload = null;
                return false;
            }
        }

        // 中文：将一个 KV 块异步入队，交由后台写线程落盘（空负载或已释放时直接忽略）。
        public void EnqueueWrite(KvBlockHash hash, byte[] payload)
        {
            if (payload == null || payload.Length == 0)
                return;
            if (_disposed)
                return;
            try
            {
                _writeQueue.Add(new WriteJob(hash, payload));
            }
            catch (InvalidOperationException)
            {
                // Queue completed during shutdown - drop write silently.
            }
        }

        // 中文：清空溢出层，删除所有已落盘文件并重置索引、LRU 链表与驻留字节计数。
        public void Clear()
        {
            lock (_gate)
            {
                foreach (var entry in _index.Values)
                {
                    try { File.Delete(PathFor(entry.Value.Hash)); }
                    catch { /* best effort */ }
                }
                _index.Clear();
                _lru.Clear();
                _residentBytes = 0;
            }
        }

        // 中文：释放资源，停止接收新写入、等待后台写线程退出并销毁写队列。
        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            try { _writeQueue.CompleteAdding(); }
            catch { /* already completed */ }
            try { _writerThread.Join(TimeSpan.FromSeconds(5)); }
            catch { /* best effort */ }
            _writeQueue.Dispose();
        }

        // 中文：后台写线程主循环，持续从队列取出写任务并逐个落盘，异常仅记日志不中断。
        private void WriterLoop()
        {
            try
            {
                foreach (var job in _writeQueue.GetConsumingEnumerable())
                {
                    try
                    {
                        WriteBlock(job.Hash, job.Payload);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "Failed to spill KV block {Hash} to SSD tier", job.Hash);
                    }
                }
            }
            catch (ObjectDisposedException)
            {
                // Queue disposed during shutdown.
            }
        }

        // 中文：将单个 KV 块原子写入磁盘（先写 .tmp 再 move），并更新索引/LRU、按容量上限淘汰旧块。
        private void WriteBlock(KvBlockHash hash, byte[] payload)
        {
            string path = PathFor(hash);
            string dir = Path.GetDirectoryName(path);
            if (!Directory.Exists(dir))
                Directory.CreateDirectory(dir);

            string tempPath = path + ".tmp";
            try
            {
                using (var fs = new FileStream(tempPath, FileMode.Create, FileAccess.Write, FileShare.None))
                {
                    Span<byte> header = stackalloc byte[HeaderSize];
                    BinaryPrimitives.WriteUInt32LittleEndian(header, Magic);
                    BinaryPrimitives.WriteInt32LittleEndian(header[4..], FormatVersion);
                    BinaryPrimitives.WriteInt64LittleEndian(header[8..], payload.LongLength);
                    BinaryPrimitives.WriteUInt64LittleEndian(header[16..], _fingerprintHash);
                    fs.Write(header);
                    fs.Write(payload, 0, payload.Length);
                }
                if (File.Exists(path))
                    File.Delete(path);
                File.Move(tempPath, path);
            }
            catch
            {
                try { if (File.Exists(tempPath)) File.Delete(tempPath); }
                catch { /* best effort */ }
                throw;
            }

            long entryBytes = payload.LongLength + HeaderSize;
            List<KvBlockHash> evicted = null;
            lock (_gate)
            {
                if (_index.TryGetValue(hash, out var existing))
                {
                    _residentBytes -= existing.Value.SizeBytes;
                    _lru.Remove(existing);
                    _index.Remove(hash);
                }

                var entry = new DiskEntry(hash, entryBytes);
                var node = _lru.AddFirst(entry);
                _index[hash] = node;
                _residentBytes += entryBytes;

                while (_residentBytes > _maxBytes && _lru.Last != null && _lru.Last != node)
                {
                    var victim = _lru.Last;
                    _lru.RemoveLast();
                    _index.Remove(victim.Value.Hash);
                    _residentBytes -= victim.Value.SizeBytes;
                    evicted ??= new List<KvBlockHash>();
                    evicted.Add(victim.Value.Hash);
                }
            }

            if (evicted != null)
            {
                foreach (var victimHash in evicted)
                {
                    try { File.Delete(PathFor(victimHash)); }
                    catch (Exception ex)
                    {
                        _logger.LogDebug(ex, "Failed to delete evicted SSD KV block {Hash}", victimHash);
                    }
                }
            }
        }

        // 中文：启动时扫描根目录下已有的 .kvb 文件，校验文件名与头部合法后重建内存索引与 LRU。
        private void ReindexExistingFiles()
        {
            if (!Directory.Exists(_rootDir))
                return;

            foreach (string subDir in Directory.EnumerateDirectories(_rootDir))
            {
                foreach (string file in Directory.EnumerateFiles(subDir, "*.kvb"))
                {
                    KvBlockHash hash;
                    try
                    {
                        string stem = Path.GetFileNameWithoutExtension(file);
                        if (stem.Length != 32)
                            continue;
                        byte[] bytes = Convert.FromHexString(stem);
                        if (bytes.Length != 16)
                            continue;
                        hash = KvBlockHash.FromBytes(bytes);
                    }
                    catch
                    {
                        continue;
                    }

                    long length;
                    try { length = new FileInfo(file).Length; }
                    catch { continue; }
                    if (length < HeaderSize)
                        continue;

                    // Validate header so leftovers from a different model don't poison the index.
                    if (!ValidateHeader(file))
                        continue;

                    var entry = new DiskEntry(hash, length);
                    var node = _lru.AddLast(entry);
                    _index[hash] = node;
                    _residentBytes += length;
                }
            }
        }

        // 中文：读取文件头并校验 magic、版本与指纹哈希，确保不是其它模型遗留的块。
        private bool ValidateHeader(string file)
        {
            try
            {
                using var fs = new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.Read);
                Span<byte> header = stackalloc byte[HeaderSize];
                if (ReadExact(fs, header) != HeaderSize)
                    return false;
                uint magic = BinaryPrimitives.ReadUInt32LittleEndian(header);
                int version = BinaryPrimitives.ReadInt32LittleEndian(header[4..]);
                ulong fingerprint = BinaryPrimitives.ReadUInt64LittleEndian(header[16..]);
                return magic == Magic && version == FormatVersion && fingerprint == _fingerprintHash;
            }
            catch
            {
                return false;
            }
        }

        // 中文：循环读取流直到填满目标缓冲或到达流尾，返回实际读取的字节数。
        private static int ReadExact(Stream s, Span<byte> dest)
        {
            int read = 0;
            while (read < dest.Length)
            {
                int n = s.Read(dest[read..]);
                if (n <= 0)
                    break;
                read += n;
            }
            return read;
        }

        // 中文：根据哈希的十六进制串计算块文件路径（按前两位做哈希分区子目录）。
        private string PathFor(KvBlockHash hash)
        {
            string hex = hash.ToHexString();
            return Path.Combine(_rootDir, hex.Substring(0, 2), hex + ".kvb");
        }

        // 中文：用 FNV-1a 算法对指纹字符串计算稳定哈希，避免每次读取都做 SHA-256。
        private static ulong StableFingerprintHash(string fingerprint)
        {
            // Lightweight FNV-1a so we don't pay SHA-256 cost on every read.
            const ulong offsetBasis = 14695981039346656037UL;
            const ulong prime = 1099511628211UL;
            ulong hash = offsetBasis;
            foreach (char c in fingerprint)
            {
                hash ^= c;
                hash *= prime;
            }
            return hash;
        }

        private readonly record struct WriteJob(KvBlockHash Hash, byte[] Payload);
        private readonly record struct DiskEntry(KvBlockHash Hash, long SizeBytes);
    }
}
