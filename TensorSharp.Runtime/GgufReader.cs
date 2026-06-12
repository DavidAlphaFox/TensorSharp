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
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using System.Text;

// ───────────────────────────────────────────────────────────────────────────
// 【文件说明】GGUF 模型文件解析器——模型加载的入口。
// 【主要类型】GgufReader（及 GgufValueType 等）：通过内存映射读取 GGUF 文件的元数据
//             （架构、超参、分词器等）与张量目录，为上层构建具体模型提供权重与配置。
// ───────────────────────────────────────────────────────────────────────────
namespace TensorSharp.Runtime
{
    // 中文：GGUF 元数据键值对的值类型枚举，对应 GGUF 规范中定义的所有标量与聚合类型
    public enum GgufValueType : uint
    {
        Uint8 = 0, Int8 = 1, Uint16 = 2, Int16 = 3,
        Uint32 = 4, Int32 = 5, Float32 = 6, Bool = 7,
        String = 8, Array = 9, Uint64 = 10, Int64 = 11, Float64 = 12
    }

    // 中文：GGML 张量量化格式枚举；各格式的块大小与字节布局由 GetBlockSize/GetTypeSize 给出。
    //       Q4_K/Q5_K/Q6_K 采用 k-量化方案（超级块 + 子块缩放），参考：https://github.com/ggerganov/llama.cpp/pull/1684
    //       IQ2_XXS/IQ2_XS/IQ3_XXS 采用重要性量化（importance-quant），参考：https://github.com/ggerganov/llama.cpp/pull/4773
    //       MXFP4 采用 MX Microscaling 浮点格式，参考：https://arxiv.org/abs/2310.10537
    public enum GgmlTensorType : uint
    {
        F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3,
        Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
        Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
        IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ1_S = 19,
        IQ4_NL = 20, IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23,
        I8 = 24, I16 = 25, I32 = 26, I64 = 27, F64 = 28,
        IQ1_M = 29, BF16 = 30,
        TQ1_0 = 34, TQ2_0 = 35,
        MXFP4 = 39,
    }

    // 中文：描述单个张量在 GGUF 文件中的元信息（名称、形状、量化类型、数据偏移）
    public class GgufTensorInfo
    {
        public string Name { get; set; } = string.Empty;
        public ulong[] Shape { get; set; } = Array.Empty<ulong>();
        public GgmlTensorType Type { get; set; }
        public ulong Offset { get; set; }

        // 中文：计算张量的总元素数（各维度之积）
        public long NumElements
        {
            get
            {
                long n = 1;
                foreach (var d in Shape) n *= (long)d;
                return n;
            }
        }
    }

    // 中文：GGUF 文件的主读取类，封装内存映射、元数据解析与张量数据读取能力
    public class GgufFile : IDisposable
    {
        public uint Version { get; private set; }
        public Dictionary<string, object> Metadata { get; } = new();
        public Dictionary<string, GgufTensorInfo> Tensors { get; } = new();
        public long DataOffset { get; private set; }

        private FileStream _stream;
        private string _path;
        private MemoryMappedFile? _mappedFile;
        private MemoryMappedViewAccessor? _mappedView;
        private unsafe byte* _mappedBase;
        private bool _mappedPointerAcquired;
        private unsafe byte* _lockedBase;
        private ulong _lockedLength;

        // 中文：构造函数，以只读方式打开指定路径的 GGUF 文件并立即解析文件头与张量目录
        public GgufFile(string path)
        {
            _path = path;
            _stream = File.OpenRead(path);
            Parse();
        }

        /// <summary>
        /// Pins the GGUF mmap region in physical RAM via mlock(2). This
        /// prevents the kernel from evicting model-weight pages between
        /// inference passes (which would otherwise force the next forward
        /// to page-fault every weight back from SSD/swap). Best-effort:
        /// silently no-ops on failure (e.g. when the process memlock
        /// rlimit is too low, or the kernel rejects the wire request).
        /// Idempotent — safe to call multiple times.
        /// </summary>
        // 中文：通过 mlock(2) 将内存映射区域锁定在物理内存中，防止推理过程中权重被换出；
        //       优先一次性锁定整个区域，失败时按 256 MB 分块重试，整体为尽力而为（失败静默）
        public unsafe bool TryLockMappedRegion()
        {
            if (_lockedBase != null)
                return true;
            EnsureMappedView();
            if (_mappedBase == null)
                return false;
            try
            {
                long capacity = _mappedView!.Capacity;
                if (capacity <= 0)
                    return false;
                ulong len = (ulong)capacity;

                // First try a single mlock for the whole region. macOS XNU
                // sometimes returns EAGAIN when asked to wire many GB at
                // once even though the global limit allows it — split into
                // chunks and try again. 256 MB chunks are large enough to
                // amortise syscall overhead and small enough to avoid the
                // single-call rejection.
                int rc = mlock(_mappedBase, (nuint)len);
                if (rc == 0)
                {
                    _lockedBase = _mappedBase;
                    _lockedLength = len;
                    return true;
                }
                LastLockError = Marshal.GetLastWin32Error();

                const ulong chunk = 256UL * 1024 * 1024;
                ulong locked = 0;
                while (locked < len)
                {
                    ulong remaining = len - locked;
                    ulong step = remaining < chunk ? remaining : chunk;
                    int rcChunk = mlock(_mappedBase + locked, (nuint)step);
                    if (rcChunk != 0)
                    {
                        LastLockError = Marshal.GetLastWin32Error();
                        if (locked > 0)
                            _ = munlock(_mappedBase, (nuint)locked);
                        return false;
                    }
                    locked += step;
                }

                _lockedBase = _mappedBase;
                _lockedLength = len;
                LastLockError = 0;
                return true;
            }
            catch (DllNotFoundException) { return false; }
            catch (EntryPointNotFoundException) { return false; }
        }

        public int LastLockError { get; private set; }

        [DllImport("libc", SetLastError = true, EntryPoint = "mlock")]
        private static extern unsafe int mlock(void* addr, nuint len);

        [DllImport("libc", SetLastError = true, EntryPoint = "munlock")]
        private static extern unsafe int munlock(void* addr, nuint len);

        // 中文：解析 GGUF 文件头——校验魔数与版本，顺序读取所有键值元数据与张量目录，
        //       并依据对齐要求计算张量数据区的起始偏移 DataOffset
        private void Parse()
        {
            using var reader = new BinaryReader(_stream, Encoding.UTF8, leaveOpen: true);

            uint magic = reader.ReadUInt32();
            if (magic != 0x46554747) // "GGUF" in little-endian
                throw new InvalidDataException($"Not a GGUF file (magic: 0x{magic:X8})");

            Version = reader.ReadUInt32();
            if (Version < 2)
                throw new NotSupportedException($"GGUF version {Version} not supported");

            ulong tensorCount = reader.ReadUInt64();
            ulong kvCount = reader.ReadUInt64();

            for (ulong i = 0; i < kvCount; i++)
            {
                string key = ReadString(reader);
                var valType = (GgufValueType)reader.ReadUInt32();
                object value = ReadValue(reader, valType);
                Metadata[key] = value;
            }

            for (ulong i = 0; i < tensorCount; i++)
            {
                var info = new GgufTensorInfo();
                info.Name = ReadString(reader);
                uint dims = reader.ReadUInt32();
                info.Shape = new ulong[dims];
                for (uint d = 0; d < dims; d++)
                    info.Shape[d] = reader.ReadUInt64();
                info.Type = (GgmlTensorType)reader.ReadUInt32();
                info.Offset = reader.ReadUInt64();
                Tensors[info.Name] = info;
            }

            long pos = _stream.Position;
            int alignment = 32;
            if (Metadata.TryGetValue("general.alignment", out var a))
                alignment = Convert.ToInt32(a);
            DataOffset = pos + (alignment - pos % alignment) % alignment;
        }

        // 中文：从元数据中读取字符串类型的键值，键不存在时返回 defaultValue
        public string? GetString(string key, string? defaultValue = null)
        {
            if (!Metadata.TryGetValue(key, out var v)) return defaultValue;
            return v as string ?? defaultValue;
        }

        // 中文：从元数据中读取 uint32 类型的键值，兼容 int[]/uint[] 数组首元素形式，键不存在时返回 defaultValue
        public uint GetUint32(string key, uint defaultValue = 0)
        {
            if (!Metadata.TryGetValue(key, out var v)) return defaultValue;
            if (v is int[] ia && ia.Length > 0) return (uint)ia[0];
            if (v is uint[] ua && ua.Length > 0) return ua[0];
            return Convert.ToUInt32(v);
        }

        // 中文：从元数据中读取 float32 类型的键值，兼容 float[] 数组首元素形式，键不存在时返回 defaultValue
        public float GetFloat32(string key, float defaultValue = 0f)
        {
            if (!Metadata.TryGetValue(key, out var v)) return defaultValue;
            if (v is float[] fa && fa.Length > 0) return fa[0];
            return Convert.ToSingle(v);
        }

        // 中文：从元数据中读取布尔类型的键值，键不存在时返回 defaultValue
        public bool GetBool(string key, bool defaultValue = false)
        {
            if (!Metadata.TryGetValue(key, out var v)) return defaultValue;
            return Convert.ToBoolean(v);
        }

        // 中文：从元数据中读取字符串数组类型的键值，键不存在或类型不匹配时返回 null
        public string[]? GetStringArray(string key)
        {
            if (!Metadata.TryGetValue(key, out var v)) return null;
            if (v is string[] sa) return sa;
            return null;
        }

        // 中文：从元数据中读取 float 数组类型的键值，键不存在或类型不匹配时返回 null
        public float[]? GetFloatArray(string key)
        {
            if (!Metadata.TryGetValue(key, out var v)) return null;
            if (v is float[] fa) return fa;
            return null;
        }

        // 中文：从元数据中读取 int32 数组类型的键值，兼容 uint[] 自动转换，键不存在时返回 null
        public int[]? GetInt32Array(string key)
        {
            if (!Metadata.TryGetValue(key, out var v)) return null;
            if (v is int[] ia) return ia;
            if (v is uint[] ua)
            {
                var result = new int[ua.Length];
                for (int i = 0; i < ua.Length; i++) result[i] = (int)ua[i];
                return result;
            }
            return null;
        }

        // 中文：从元数据中读取 bool 数组类型的键值，键不存在或类型不匹配时返回 null
        public bool[]? GetBoolArray(string key)
        {
            if (!Metadata.TryGetValue(key, out var v)) return null;
            if (v is bool[] ba) return ba;
            return null;
        }

        // 中文：从元数据中读取 uint32 数组类型的键值，兼容 int[] 自动转换，键不存在时返回 null
        public uint[]? GetUint32Array(string key)
        {
            if (!Metadata.TryGetValue(key, out var v)) return null;
            if (v is uint[] ua) return ua;
            if (v is int[] ia)
            {
                var result = new uint[ia.Length];
                for (int i = 0; i < ia.Length; i++) result[i] = (uint)ia[i];
                return result;
            }
            return null;
        }

        // 中文：将指定张量的原始字节数据读入新分配的托管字节数组并返回
        public byte[] ReadTensorData(GgufTensorInfo tensorInfo)
        {
            long byteCount = GetTensorByteCount(tensorInfo);
            byte[] data = new byte[byteCount];
            _stream.Seek(DataOffset + (long)tensorInfo.Offset, SeekOrigin.Begin);
            _stream.ReadExactly(data, 0, data.Length);
            return data;
        }

        /// <summary>
        /// Read F32 tensor data directly into a float array in chunks (for tensors > 2GB raw bytes).
        /// </summary>
        // 中文：以 16 MB 分块方式将 F32 张量数据直接读入托管 float 数组，适用于原始字节数超过 2 GB 的超大张量
        public unsafe void ReadTensorDataToFloat32(GgufTensorInfo tensorInfo, float[] dest, long numElements)
        {
            long totalBytes = numElements * 4;
            _stream.Seek(DataOffset + (long)tensorInfo.Offset, SeekOrigin.Begin);
            const int chunkBytes = 16 * 1024 * 1024;
            byte[] buffer = new byte[chunkBytes];
            long bytesRead = 0;

            fixed (float* destBase = dest)
            {
                while (bytesRead < totalBytes)
                {
                    int toRead = (int)Math.Min(totalBytes - bytesRead, chunkBytes);
                    _stream.ReadExactly(buffer, 0, toRead);
                    fixed (byte* srcPtr = buffer)
                    {
                        Buffer.MemoryCopy(srcPtr, (byte*)destBase + bytesRead,
                            totalBytes - bytesRead, toRead);
                    }
                    bytesRead += toRead;
                }
            }
        }

        /// <summary>
        /// Read F32 tensor data directly into native memory pointed to by dest (for tensors > 2G elements).
        /// </summary>
        // 中文：以 16 MB 分块方式将 F32 张量数据读入由 dest 指向的原生内存，适用于元素数超过 2G 的超大张量
        public unsafe void ReadTensorDataToFloat32Native(GgufTensorInfo tensorInfo, IntPtr dest, long numElements)
        {
            long totalBytes = numElements * 4;
            _stream.Seek(DataOffset + (long)tensorInfo.Offset, SeekOrigin.Begin);
            const int chunkBytes = 16 * 1024 * 1024;
            byte[] buffer = new byte[chunkBytes];
            long bytesRead = 0;
            byte* destPtr = (byte*)dest;

            while (bytesRead < totalBytes)
            {
                int toRead = (int)Math.Min(totalBytes - bytesRead, chunkBytes);
                _stream.ReadExactly(buffer, 0, toRead);
                System.Runtime.InteropServices.Marshal.Copy(buffer, 0, (IntPtr)(destPtr + bytesRead), toRead);
                bytesRead += toRead;
            }
        }

        /// <summary>
        /// Read tensor data directly into pre-allocated native memory (for tensors > 2GB).
        /// </summary>
        // 中文：以最大 8 MB 分块方式将任意格式张量的原始字节数据读入预分配的原生内存，适用于超过 2 GB 的张量
        public unsafe void ReadTensorDataToNative(GgufTensorInfo tensorInfo, IntPtr dest, long byteCount)
        {
            _stream.Seek(DataOffset + (long)tensorInfo.Offset, SeekOrigin.Begin);
            byte[] buffer = new byte[Math.Min(byteCount, 8 * 1024 * 1024)];
            long remaining = byteCount;
            byte* destPtr = (byte*)dest.ToPointer();
            while (remaining > 0)
            {
                int toRead = (int)Math.Min(remaining, buffer.Length);
                _stream.ReadExactly(buffer, 0, toRead);
                System.Runtime.InteropServices.Marshal.Copy(buffer, 0, (IntPtr)destPtr, toRead);
                destPtr += toRead;
                remaining -= toRead;
            }
        }

        // 中文：通过内存映射获取指定张量数据在进程地址空间中的直接指针，避免数据拷贝；
        //       若内存映射不可用则返回 false
        public unsafe bool TryGetTensorDataPointer(GgufTensorInfo tensorInfo, out IntPtr dataPtr)
        {
            dataPtr = IntPtr.Zero;
            if (tensorInfo == null)
                return false;

            EnsureMappedView();
            if (_mappedBase == null)
                return false;

            dataPtr = (IntPtr)(_mappedBase + DataOffset + (long)tensorInfo.Offset);
            return true;
        }

        // 中文：根据张量的形状和量化类型计算其在文件中占用的总字节数
        public long GetTensorByteCount(GgufTensorInfo tensorInfo)
        {
            long ne0 = (long)tensorInfo.Shape[0];
            long rows = 1;
            for (int i = 1; i < tensorInfo.Shape.Length; i++)
                rows *= (long)tensorInfo.Shape[i];

            long rowBytes = GetRowBytes(tensorInfo.Type, ne0);
            return rowBytes * rows;
        }

        // 中文：计算指定量化类型下每行（沿最内层维度）所占字节数，公式为 (ne0 / blockSize) * typeSize
        private static long GetRowBytes(GgmlTensorType type, long ne0)
        {
            long blockSize = GetBlockSize(type);
            long typeSize = GetTypeSize(type);
            return (ne0 / blockSize) * typeSize;
        }

        // 中文：返回各量化格式的量化块大小（每块包含的元素数）；
        //       标量格式块大小为 1，Q4_0/Q8_0 等经典格式为 32，k-quant/IQ 系列等超级块格式为 256
        public static long GetBlockSize(GgmlTensorType type)
        {
            switch (type)
            {
                case GgmlTensorType.F32:
                case GgmlTensorType.F16:
                case GgmlTensorType.BF16:
                case GgmlTensorType.I8:
                case GgmlTensorType.I16:
                case GgmlTensorType.I32:
                case GgmlTensorType.I64:
                case GgmlTensorType.F64:
                    return 1;
                case GgmlTensorType.Q4_0:
                case GgmlTensorType.Q4_1:
                case GgmlTensorType.Q5_0:
                case GgmlTensorType.Q5_1:
                case GgmlTensorType.Q8_0:
                case GgmlTensorType.Q8_1:
                case GgmlTensorType.IQ4_NL:
                case GgmlTensorType.MXFP4:
                    return 32;
                default:
                    return 256;
            }
        }

        // 中文：返回各量化格式每个量化块在文件中占用的字节数（类型大小）。
        //       Q8_0 对称量化：每块存储 1 个 fp16 缩放因子 + 32 个 int8 权重，共 34 字节，参考 GGUF 规范。
        //       Q4_K/Q5_K/Q6_K k-量化：超级块包含子块缩放因子与量化码，参考 https://github.com/ggerganov/llama.cpp/pull/1684
        //       IQ2_XXS/IQ2_XS/IQ3_XXS 重要性量化：基于向量码本的极低比特量化，参考 https://github.com/ggerganov/llama.cpp/pull/4773
        //       MXFP4 MX Microscaling：每块含 1 字节共享指数 + 32 个 4-bit 尾数，参考 https://arxiv.org/abs/2310.10537
        public static long GetTypeSize(GgmlTensorType type)
        {
            switch (type)
            {
                case GgmlTensorType.F32: return 4;
                case GgmlTensorType.F16: return 2;
                case GgmlTensorType.BF16: return 2;
                case GgmlTensorType.Q4_0: return 2 + 32 / 2;
                case GgmlTensorType.Q4_1: return 2 + 2 + 32 / 2;
                case GgmlTensorType.Q5_0: return 2 + 4 + 32 / 2;
                case GgmlTensorType.Q5_1: return 2 + 2 + 4 + 32 / 2;
                case GgmlTensorType.Q8_0: return 2 + 32;
                case GgmlTensorType.Q8_1: return 2 + 2 + 32;
                case GgmlTensorType.Q2_K: return 256 / 16 + 256 / 4 + 2 + 2;
                case GgmlTensorType.Q3_K: return 256 / 8 + 256 / 4 + 12 + 2;
                case GgmlTensorType.Q4_K: return 2 + 2 + 12 + 256 / 2;
                case GgmlTensorType.Q5_K: return 2 + 2 + 12 + 256 / 8 + 256 / 2;
                case GgmlTensorType.Q6_K: return 256 / 2 + 256 / 4 + 256 / 16 + 2;
                case GgmlTensorType.Q8_K: return 4 + 256 + 2 * 256 / 16;
                case GgmlTensorType.IQ2_XXS: return 2 + 256 / 8 * 2;           // 66
                case GgmlTensorType.IQ2_XS: return 2 + 256 / 8 * 2 + 256 / 32; // 74
                case GgmlTensorType.IQ3_XXS: return 2 + 3 * (256 / 8);         // 98
                case GgmlTensorType.IQ1_S: return 2 + 256 / 8 + 256 / 16;      // 50
                case GgmlTensorType.IQ4_NL: return 2 + 32 / 2;                 // 18
                case GgmlTensorType.IQ3_S: return 2 + 13 * (256 / 32) + 256 / 64; // 110
                case GgmlTensorType.IQ2_S: return 2 + 256 / 4 + 256 / 16;      // 82
                case GgmlTensorType.IQ4_XS: return 2 + 2 + 256 / 64 + 256 / 2; // 136
                case GgmlTensorType.IQ1_M: return 256 / 8 + 256 / 16 + 256 / 32; // 56
                case GgmlTensorType.TQ1_0: return 2 + 256 / 64 + (256 - 4 * 256 / 64) / 5; // 54
                case GgmlTensorType.TQ2_0: return 2 + 256 / 4;                 // 66
                case GgmlTensorType.MXFP4: return 1 + 32 / 2;                  // 17
                case GgmlTensorType.I8: return 1;
                case GgmlTensorType.I16: return 2;
                case GgmlTensorType.I32: return 4;
                case GgmlTensorType.I64: return 8;
                case GgmlTensorType.F64: return 8;
                default:
                    throw new NotSupportedException($"Unknown GGML tensor type: {type}");
            }
        }

        // 中文：从二进制流中读取 GGUF 格式的字符串（先读 uint64 长度，再读对应字节数的 UTF-8 内容）
        private string ReadString(BinaryReader reader)
        {
            ulong len = reader.ReadUInt64();
            byte[] bytes = reader.ReadBytes((int)len);
            return Encoding.UTF8.GetString(bytes);
        }

        // 中文：根据给定的值类型枚举从二进制流中读取对应的标量或数组值并以 object 形式返回
        private object ReadValue(BinaryReader reader, GgufValueType type)
        {
            switch (type)
            {
                case GgufValueType.Uint8: return reader.ReadByte();
                case GgufValueType.Int8: return reader.ReadSByte();
                case GgufValueType.Uint16: return reader.ReadUInt16();
                case GgufValueType.Int16: return reader.ReadInt16();
                case GgufValueType.Uint32: return reader.ReadUInt32();
                case GgufValueType.Int32: return reader.ReadInt32();
                case GgufValueType.Float32: return reader.ReadSingle();
                case GgufValueType.Bool: return reader.ReadByte() != 0;
                case GgufValueType.String: return ReadString(reader);
                case GgufValueType.Uint64: return reader.ReadUInt64();
                case GgufValueType.Int64: return reader.ReadInt64();
                case GgufValueType.Float64: return reader.ReadDouble();
                case GgufValueType.Array: return ReadArray(reader);
                default:
                    throw new NotSupportedException($"Unknown GGUF value type: {type}");
            }
        }

        // 中文：读取 GGUF 数组元数据（先读元素类型与数量，再逐元素解析）并以强类型数组形式返回
        private object ReadArray(BinaryReader reader)
        {
            var elemType = (GgufValueType)reader.ReadUInt32();
            ulong count = reader.ReadUInt64();

            switch (elemType)
            {
                case GgufValueType.Uint32:
                {
                    var arr = new uint[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadUInt32();
                    return arr;
                }
                case GgufValueType.Int32:
                {
                    var arr = new int[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadInt32();
                    return arr;
                }
                case GgufValueType.Float32:
                {
                    var arr = new float[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadSingle();
                    return arr;
                }
                case GgufValueType.String:
                {
                    var arr = new string[count];
                    for (ulong i = 0; i < count; i++) arr[i] = ReadString(reader);
                    return arr;
                }
                case GgufValueType.Uint8:
                {
                    var arr = new byte[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadByte();
                    return arr;
                }
                case GgufValueType.Int8:
                {
                    var arr = new sbyte[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadSByte();
                    return arr;
                }
                case GgufValueType.Uint16:
                {
                    var arr = new ushort[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadUInt16();
                    return arr;
                }
                case GgufValueType.Int16:
                {
                    var arr = new short[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadInt16();
                    return arr;
                }
                case GgufValueType.Uint64:
                {
                    var arr = new ulong[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadUInt64();
                    return arr;
                }
                case GgufValueType.Int64:
                {
                    var arr = new long[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadInt64();
                    return arr;
                }
                case GgufValueType.Float64:
                {
                    var arr = new double[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadDouble();
                    return arr;
                }
                case GgufValueType.Bool:
                {
                    var arr = new bool[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadByte() != 0;
                    return arr;
                }
                default:
                    throw new NotSupportedException($"Unknown array element type: {elemType}");
            }
        }

        // 中文：释放所有非托管资源：先解锁 mlock 区域，再释放内存映射指针、视图、文件映射及文件流
        public unsafe void Dispose()
        {
            if (_lockedBase != null)
            {
                try { _ = munlock(_lockedBase, (nuint)_lockedLength); } catch { }
                _lockedBase = null;
                _lockedLength = 0;
            }
            if (_mappedPointerAcquired && _mappedView != null)
            {
                _mappedView.SafeMemoryMappedViewHandle.ReleasePointer();
                _mappedPointerAcquired = false;
                _mappedBase = null;
            }

            _mappedView?.Dispose();
            _mappedView = null;
            _mappedFile?.Dispose();
            _mappedFile = null;
            _stream?.Dispose();
            _stream = null!;
        }

        // 中文：懒加载内存映射视图并获取基地址指针；已映射则直接返回，否则创建只读内存映射并固定指针
        private unsafe void EnsureMappedView()
        {
            if (_mappedBase != null)
                return;

            _mappedFile ??= MemoryMappedFile.CreateFromFile(_path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            _mappedView ??= _mappedFile.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);

            byte* viewPtr = null;
            _mappedView.SafeMemoryMappedViewHandle.AcquirePointer(ref viewPtr);
            viewPtr += _mappedView.PointerOffset;
            _mappedBase = viewPtr;
            _mappedPointerAcquired = true;
        }
    }
}
