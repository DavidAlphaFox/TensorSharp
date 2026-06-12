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
// 文件：NativeDequant.cs
// 用途：提供统一的量化张量反量化入口，优先调用原生 GGML 实现，
//       当原生库不可用（DllNotFoundException/EntryPointNotFoundException）时
//       自动回退到托管（Managed）实现，确保跨平台兼容性。
// 主要类型：NativeDequant（内部静态工具类）
// 支持格式：GGUF 规范定义的全部量化类型，包括
//   Q4_K/Q5_K/Q6_K（k-quant，https://github.com/ggerganov/llama.cpp/pull/1684）、
//   IQ2_XXS/IQ2_XS/IQ3_XXS（I-quant，https://github.com/ggerganov/llama.cpp/pull/4773）、
//   Q8_0 对称量化（参考 GGUF 规范）等。
// ────────────────────────

using System;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    internal static class NativeDequant
    {
        // 中文：将字节数组中的量化数据反量化为 float32，优先使用原生实现，失败时回退到托管实现
        public static void DequantizeToFloat32(int ggmlType, byte[] src, int srcOffset, float[] dst, int dstOffset, long numElements)
        {
            try
            {
                GgmlGgufTensorDequant.DequantizeToFloat32(ggmlType, src, srcOffset, dst, dstOffset, numElements);
            }
            catch (Exception ex) when (ShouldUseManagedFallback(ex))
            {
                ManagedQuantizedOps.DequantizeToFloat32(ggmlType, src, srcOffset, dst, dstOffset, numElements);
            }
        }

        // 中文：将非托管指针指向的量化数据反量化为 float32 托管数组，优先使用原生实现，失败时回退到托管实现
        public static void DequantizeToFloat32(int ggmlType, IntPtr src, float[] dst, int dstOffset, long numElements)
        {
            try
            {
                GgmlGgufTensorDequant.DequantizeToFloat32(ggmlType, src, dst, dstOffset, numElements);
            }
            catch (Exception ex) when (ShouldUseManagedFallback(ex))
            {
                ManagedQuantizedOps.DequantizeToFloat32(ggmlType, src, dst, dstOffset, numElements);
            }
        }

        // 中文：在两个非托管内存指针之间执行全原生反量化（src→dst），适用于零拷贝场景，失败时回退到托管实现
        public static void DequantizeToFloat32Native(int ggmlType, IntPtr src, IntPtr dst, long numElements)
        {
            try
            {
                GgmlGgufTensorDequant.DequantizeToFloat32Native(ggmlType, src, dst, numElements);
            }
            catch (Exception ex) when (ShouldUseManagedFallback(ex))
            {
                ManagedQuantizedOps.DequantizeToFloat32Native(ggmlType, src, dst, numElements);
            }
        }

        // 中文：计算给定量化类型一行（ne 个元素）所占的字节数；回退路径按 GGUF 块大小对齐校验后计算
        public static long RowSize(int ggmlType, long ne)
        {
            try
            {
                return GgmlGgufTensorDequant.GetRowSizeBytes(ggmlType, ne);
            }
            catch (Exception ex) when (ShouldUseManagedFallback(ex))
            {
                var type = (GgmlTensorType)ggmlType;
                long blockSize = GgufFile.GetBlockSize(type);
                if (ne % blockSize != 0)
                    throw new NotSupportedException($"Tensor type {type} requires row length aligned to {blockSize}, got {ne}.");

                return (ne / blockSize) * GgufFile.GetTypeSize(type);
            }
        }

        // 中文：判断捕获到的异常是否应触发托管回退（原生库缺失或入口点未找到时返回 true，支持递归展开 TypeInitializationException）
        private static bool ShouldUseManagedFallback(Exception ex)
        {
            if (ex is DllNotFoundException or EntryPointNotFoundException)
                return true;

            if (ex is TypeInitializationException tie && tie.InnerException != null)
                return ShouldUseManagedFallback(tie.InnerException);

            return false;
        }
    }
}
