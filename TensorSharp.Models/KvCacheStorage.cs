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
// 文件：KvCacheStorage.cs
// 用途：定义 KV 缓存（Key/Value Cache）的数据类型枚举 KvCacheDtype（F32/F16/Q8_0），
//       提供各数据类型对应的字节大小、GGML 类型 ID 查询、以及进程级别配置管理类 KvCacheDtypeConfig。
// 主要类型：
//   - KvCacheDtype         : KV 缓存精度枚举，支持 F32（全精度）、F16（半精度）、Q8_0（对称 8 位量化）
//   - KvCacheDtypeExtensions : KvCacheDtype 的扩展方法集合
//   - KvCacheDtypeConfig   : 进程级 KV 缓存类型配置，支持环境变量 KV_CACHE_DTYPE 或命令行参数设置
// Q8_0 量化原理：每 32 个元素为一个 block，每个元素用 1 个 int8 存储，外加 1 个 F16 缩放因子（scale），
//   共 34 字节/block，约 1.0625 字节/元素。符合 GGUF 规范中的对称量化格式。
// ────────────────────────

using System;
using System.Buffers;
using TensorSharp;

namespace TensorSharp.Models
{
    /// <summary>
    /// Storage element type for the per-layer K/V cache tensors. Selected once per
    /// model load (env var <c>KV_CACHE_DTYPE</c> or per-CLI flag) and threaded into
    /// every model's <c>InitKVCache</c>, <c>CopyToCache*</c>, and attention helpers.
    ///
    /// <c>F32</c> is the historical default: each cache element occupies four bytes
    /// and all attention kernels can read it directly.
    ///
    /// <c>F16</c> halves the resident KV-cache size (e.g. Gemma 4 E4B drops from
    /// ~4136 MB to ~2068 MB at the full 131K context). Quality is essentially
    /// identical: post-RoPE K/V values live in the [-10, 10] range that F16
    /// represents with ~10 bits of mantissa.
    ///
    /// <c>Q8_0</c> is the "model-aligned" choice for Q8_0 / Q4_K / IQ4_XS GGUF
    /// files. The cache stores values in 32-element blocks of one int8 quant per
    /// element plus a single F16 scale, for ~1.0625 bytes/elem total. That is
    /// another 2x compression over F16 (e.g. Gemma 4 E4B drops to ~1097 MB) and
    /// matches the precision of the model weights themselves, so the dominant
    /// matmul / flash-attention kernels can avoid an F16↔F32 reconvert path.
    /// Q8_0 is only valid when the entire cache I/O path stays in native GGML
    /// kernels - models that fall back to the C# managed attention helpers
    /// (which expect to walk the cache as a flat F32/F16 buffer) will reject
    /// Q8_0 at <c>InitKVCache</c> time.
    /// </summary>
    public enum KvCacheDtype
    {
        F32 = 0,
        F16 = 1,
        Q8_0 = 2,
    }

    public static class KvCacheDtypeExtensions
    {
        // ggml type ids - must match the enum in
        // ExternalProjects/ggml/include/ggml.h.
        private const int GGML_TYPE_F32  = 0;
        private const int GGML_TYPE_F16  = 1;
        private const int GGML_TYPE_Q8_0 = 8;

        // 中文：返回指定 KV 缓存数据类型每个元素占用的字节数（Q8_0 取下界 1，实际按 32 元素 block 对齐）
        public static int ElementBytes(this KvCacheDtype dtype) => dtype switch
        {
            KvCacheDtype.F32 => 4,
            KvCacheDtype.F16 => 2,
            // Lower-bound for budget logging; actual storage rounds up to the
            // 32-element block boundary (34 bytes per block).
            KvCacheDtype.Q8_0 => 1,
            _ => throw new ArgumentOutOfRangeException(nameof(dtype)),
        };

        /// <summary>
        /// Bytes consumed by a contiguous Q8_0 cache of the given length (32-element
        /// blocks of 34 bytes each). For F32/F16 this is just elementCount*Size.
        /// </summary>
        // 中文：计算给定元素数量下指定数据类型所需的字节总量；Q8_0 按每 32 元素 34 字节对齐（GGUF 对称量化规范）
        public static long ByteLengthFor(this KvCacheDtype dtype, long elementCount) => dtype switch
        {
            KvCacheDtype.F32 => elementCount * 4,
            KvCacheDtype.F16 => elementCount * 2,
            KvCacheDtype.Q8_0 => DTypeExtensions.Q8_0Bytes(elementCount),
            _ => throw new ArgumentOutOfRangeException(nameof(dtype)),
        };

        // 中文：将 KvCacheDtype 枚举值转换为框架内部通用的 DType 枚举
        public static DType ToDType(this KvCacheDtype dtype) => dtype switch
        {
            KvCacheDtype.F32 => DType.Float32,
            KvCacheDtype.F16 => DType.Float16,
            KvCacheDtype.Q8_0 => DType.Q8_0,
            _ => throw new ArgumentOutOfRangeException(nameof(dtype)),
        };

        // 中文：返回数据类型的短字符串表示（如 "f32"/"f16"/"q8_0"），用于日志输出或命令行显示
        public static string ToShortString(this KvCacheDtype dtype) => dtype switch
        {
            KvCacheDtype.F32 => "f32",
            KvCacheDtype.F16 => "f16",
            KvCacheDtype.Q8_0 => "q8_0",
            _ => throw new ArgumentOutOfRangeException(nameof(dtype)),
        };

        // 中文：返回与 GGML 原生库对应的类型 ID（须与 ggml.h 中的枚举值保持一致）
        public static int GgmlType(this KvCacheDtype dtype) => dtype switch
        {
            KvCacheDtype.F32 => GGML_TYPE_F32,
            KvCacheDtype.F16 => GGML_TYPE_F16,
            KvCacheDtype.Q8_0 => GGML_TYPE_Q8_0,
            _ => throw new ArgumentOutOfRangeException(nameof(dtype)),
        };

        /// <summary>
        /// Block-quantized cache types cannot be read element-by-element from C#
        /// managed code without an explicit dequantize pass. Models that need to
        /// walk the cache directly (e.g. for SWA prev-window gather) must reject
        /// these dtypes during initialization.
        /// </summary>
        // 中文：判断该数据类型是否为块量化格式（如 Q8_0），块量化类型无法被 C# 托管代码逐元素直读，需先反量化
        public static bool IsBlockQuantized(this KvCacheDtype dtype) => dtype switch
        {
            KvCacheDtype.Q8_0 => true,
            _ => false,
        };
    }

    /// <summary>
    /// Process-wide configuration for the KV-cache storage type. The CLI / server
    /// front-end calls <see cref="ConfigureFromEnvironment"/> once at start-up;
    /// each model picks the value up via <see cref="Current"/> when it allocates
    /// its per-layer K/V tensors.
    /// </summary>
    public static class KvCacheDtypeConfig
    {
        private static KvCacheDtype _current = KvCacheDtype.F32;
        private static bool _explicitlySet;

        public static KvCacheDtype Current => _current;
        public static bool IsExplicitlySet => _explicitlySet;

        // 中文：显式设置进程级 KV 缓存数据类型，并标记为用户主动配置（后续自动推断将跳过）
        public static void Set(KvCacheDtype dtype)
        {
            _current = dtype;
            _explicitlySet = true;
        }

        /// <summary>
        /// Apply a model-aligned default cache dtype if the user hasn't explicitly
        /// chosen one. Models whose dominant weight tier is below F32 (Q8_0,
        /// Q4_K, IQ4_XS, F16, etc.) get an F16 cache: K/V values fit losslessly
        /// inside F16's ~10-bit mantissa for the [-10, 10] post-RoPE range, so
        /// outputs are byte-identical to the F32 baseline while halving cache
        /// memory and bandwidth. Pure F32 native models keep the F32 cache.
        /// Callers can opt in to the more aggressive Q8_0 cache via
        /// <c>--kv-cache-dtype q8_0</c>.
        /// </summary>
        // 中文：根据模型权重的主导 GGML 类型自动推断 KV 缓存默认精度：非 F32 模型默认使用 F16 缓存以节省内存
        public static void ApplyModelDtypeDefault(int dominantGgmlType)
        {
            if (_explicitlySet) return;
            // 0 = GGML_TYPE_F32. Anything else (F16, BF16, Q8_0, Q4_K, ...) is
            // already operating below F32 precision so an F16 cache adds no
            // measurable error.
            if (dominantGgmlType != 0)
                _current = KvCacheDtype.F16;
        }

        // 中文：将字符串（如 "f32"/"f16"/"q8_0" 及常见别名）解析为 KvCacheDtype 枚举值，解析失败返回 false
        public static bool TryParse(string value, out KvCacheDtype dtype)
        {
            dtype = KvCacheDtype.F32;
            if (string.IsNullOrWhiteSpace(value)) return false;
            switch (value.Trim().ToLowerInvariant())
            {
                case "f32":
                case "float32":
                case "fp32":
                    dtype = KvCacheDtype.F32;
                    return true;
                case "f16":
                case "float16":
                case "fp16":
                case "half":
                    dtype = KvCacheDtype.F16;
                    return true;
                case "q8_0":
                case "q8":
                case "int8":
                    dtype = KvCacheDtype.Q8_0;
                    return true;
                default:
                    return false;
            }
        }

        /// <summary>
        /// Read the <c>KV_CACHE_DTYPE</c> environment variable (if any) and apply
        /// it as the process-wide default. Unrecognized values are ignored.
        /// </summary>
        // 中文：读取环境变量 KV_CACHE_DTYPE 并将其解析后应用为进程级 KV 缓存类型，无效值静默忽略
        public static void ConfigureFromEnvironment()
        {
            string value = Environment.GetEnvironmentVariable("KV_CACHE_DTYPE");
            if (TryParse(value, out KvCacheDtype dtype))
                Set(dtype);
        }
    }
}
