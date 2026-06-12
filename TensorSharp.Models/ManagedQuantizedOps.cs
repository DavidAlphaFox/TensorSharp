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
using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

// ───────────────────────────────────────────────────────────────────────────
// 【文件说明】纯托管（C# / SIMD）量化算子实现。
// 【主要类型】ManagedQuantizedOps：在不反量化为 FP32 的前提下，直接对 GGUF 量化权重
//             （Q4_0 / Q4_K / Q8_0 / MXFP4 / IQ2_XXS 等）做矩阵乘等运算，借助硬件内建
//             指令（AVX 等）加速；作为无原生库时 CPU 后端的量化计算路径。
// ───────────────────────────────────────────────────────────────────────────
namespace TensorSharp.Models
{
    internal static class ManagedQuantizedOps
    {
        private const int QK4_0 = 32;
        private const int QK4_1 = 32;
        private const int QK5_0 = 32;
        private const int QK5_1 = 32;
        private const int QK8_0 = 32;
        private const int QK8_1 = 32;
        private const int QK4_NL = 32;
        private const int QK_MXFP4 = 32;
        private const int QK_K = 256;
        private const int K_SCALE_SIZE = 12;
        private const int Q4_0BlockBytes = 2 + QK4_0 / 2;
        private const int Q4_1BlockBytes = 4 + QK4_1 / 2;
        private const int Q5_0BlockBytes = 2 + 4 + QK5_0 / 2;
        private const int Q5_1BlockBytes = 4 + 4 + QK5_1 / 2;
        private const int Q8_0BlockBytes = 2 + QK8_0;
        private const int Q8_1BlockBytes = 4 + QK8_1;
        private const int Q4_KBlockBytes = 4 + K_SCALE_SIZE + QK_K / 2;
        private const int Q5_KBlockBytes = 4 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2;
        private const int Q6_KBlockBytes = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;
        private const int Q8_KBlockBytes = 4 + QK_K + 2 * (QK_K / 16);

        private static readonly sbyte[] Iq4NlValues =
        {
            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
        };

        private static readonly sbyte[] Mxfp4Values =
        {
            0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
        };

        // 中文：判断指定 GGML 张量类型是否可在 CPU 上以量化格式（不先解压到 FP32）直接存储和计算
        public static bool SupportsCpuQuantizedStorage(GgmlTensorType type)
        {
            return type switch
            {
                GgmlTensorType.F16 => true,
                GgmlTensorType.BF16 => true,
                GgmlTensorType.Q4_0 => true,
                GgmlTensorType.Q4_1 => true,
                GgmlTensorType.Q5_0 => true,
                GgmlTensorType.Q5_1 => true,
                GgmlTensorType.Q8_0 => true,
                GgmlTensorType.Q8_1 => true,
                GgmlTensorType.Q4_K => true,
                GgmlTensorType.Q5_K => true,
                GgmlTensorType.Q6_K => true,
                GgmlTensorType.IQ4_NL => true,
                GgmlTensorType.MXFP4 => true,
                _ => false,
            };
        }

        // 中文：判断指定 GGML 张量类型是否支持反量化到 FP32（包含整数类型和浮点类型）
        public static bool SupportsDequantization(GgmlTensorType type)
        {
            return type switch
            {
                GgmlTensorType.F32 => true,
                GgmlTensorType.F16 => true,
                GgmlTensorType.BF16 => true,
                GgmlTensorType.I8 => true,
                GgmlTensorType.I16 => true,
                GgmlTensorType.I32 => true,
                GgmlTensorType.I64 => true,
                GgmlTensorType.F64 => true,
                _ => SupportsCpuQuantizedStorage(type),
            };
        }

        // 中文：计算给定量化类型一行（ne 个元素）所需的字节数，要求行长度对齐到块大小
        public static long RowSize(int ggmlType, long ne)
        {
            var type = (GgmlTensorType)ggmlType;
            if (!SupportsDequantization(type))
                throw new NotSupportedException($"Pure C# backend does not support GGUF tensor type {type}.");

            long blockSize = GgufFile.GetBlockSize(type);
            if (ne % blockSize != 0)
                throw new NotSupportedException($"Tensor type {type} requires row length aligned to {blockSize}, got {ne}.");

            return (ne / blockSize) * GgufFile.GetTypeSize(type);
        }

        // 中文：将托管字节数组中的量化数据反量化为 FP32，写入目标 float 数组（托管重载）
        public static unsafe void DequantizeToFloat32(int ggmlType, byte[] src, int srcOffset, float[] dst, int dstOffset, long numElements)
        {
            var type = (GgmlTensorType)ggmlType;
            if (!SupportsDequantization(type))
                throw new NotSupportedException($"Pure C# backend does not support GGUF tensor type {type}.");

            fixed (byte* srcBase = src)
            fixed (float* dstBase = dst)
            {
                DequantizeToFloat32(type, srcBase + srcOffset, dstBase + dstOffset, numElements);
            }
        }

        // 中文：将非托管指针指向的量化数据反量化为 FP32，写入目标托管 float 数组（IntPtr 源重载）
        public static unsafe void DequantizeToFloat32(int ggmlType, IntPtr src, float[] dst, int dstOffset, long numElements)
        {
            var type = (GgmlTensorType)ggmlType;
            if (!SupportsDequantization(type))
                throw new NotSupportedException($"Pure C# backend does not support GGUF tensor type {type}.");

            fixed (float* dstBase = dst)
            {
                DequantizeToFloat32(type, (byte*)src.ToPointer(), dstBase + dstOffset, numElements);
            }
        }

        // 中文：将非托管指针指向的量化数据反量化为 FP32，输出写入非托管目标内存（全指针重载，适合 P/Invoke 场景）
        public static unsafe void DequantizeToFloat32Native(int ggmlType, IntPtr src, IntPtr dst, long numElements)
        {
            var type = (GgmlTensorType)ggmlType;
            if (!SupportsDequantization(type))
                throw new NotSupportedException($"Pure C# backend does not support GGUF tensor type {type}.");

            DequantizeToFloat32(type, (byte*)src.ToPointer(), (float*)dst.ToPointer(), numElements);
        }

        // 中文：将非托管指针指向的量化行数据反量化为 FP32，直接写入裸 float 指针（供内部快速路径调用）
        public static unsafe void DequantizeRowToFloat32(int ggmlType, IntPtr src, float* dst, long numElements)
        {
            var type = (GgmlTensorType)ggmlType;
            if (!SupportsDequantization(type))
                throw new NotSupportedException($"Pure C# backend does not support GGUF tensor type {type}.");

            DequantizeToFloat32(type, (byte*)src.ToPointer(), dst, numElements);
        }

        // 中文：对量化权重行与多行 FP32 激活向量做批量点积，结果累加到 outputs 数组（托管数组重载）
        public static unsafe void DotRowBatchToFloat32(int ggmlType, byte[] src, int srcOffset,
            float[] inputs, int inputOffset, int inputRowStride, int rowCount, long numElements,
            float[] outputs, int outputOffset)
        {
            var type = (GgmlTensorType)ggmlType;
            if (!SupportsDequantization(type))
                throw new NotSupportedException($"Pure C# backend does not support GGUF tensor type {type}.");

            fixed (byte* srcBase = src)
            fixed (float* inputBase = inputs)
            fixed (float* outputBase = outputs)
            {
                DotRowBatchToFloat32(
                    ggmlType,
                    (IntPtr)(srcBase + srcOffset),
                    inputBase + inputOffset,
                    inputRowStride,
                    rowCount,
                    numElements,
                    outputBase + outputOffset);
            }
        }

        // 中文：对量化权重行与多行 FP32 激活向量做批量点积，核心实现：先按块反量化到栈上暂存区，再调用 TensorPrimitives.Dot 累加
        public static unsafe void DotRowBatchToFloat32(int ggmlType, IntPtr src, float* inputs,
            int inputRowStride, int rowCount, long numElements, float* outputs)
        {
            var type = (GgmlTensorType)ggmlType;
            if (!SupportsDequantization(type))
                throw new NotSupportedException($"Pure C# backend does not support GGUF tensor type {type}.");
            if (rowCount < 1)
                throw new ArgumentOutOfRangeException(nameof(rowCount));
            if (inputRowStride < numElements)
                throw new ArgumentOutOfRangeException(nameof(inputRowStride));

            long blockSize = GgufFile.GetBlockSize(type);
            if (numElements % blockSize != 0)
                throw new NotSupportedException($"Tensor type {type} requires row length aligned to {blockSize}, got {numElements}.");

            for (int row = 0; row < rowCount; row++)
                outputs[row] = 0.0f;

            if (type == GgmlTensorType.F32)
            {
                float* weight = (float*)src.ToPointer();
                for (int row = 0; row < rowCount; row++)
                    outputs[row] = DotFloat(inputs + (long)row * inputRowStride, weight, (int)numElements);
                return;
            }

            float* scratch = stackalloc float[QK_K];
            byte* chunkPtr = (byte*)src.ToPointer();
            long elementOffset = 0;

            while (elementOffset < numElements)
            {
                int chunkElements = GetDotChunkSize(type, numElements - elementOffset);
                DequantizeToFloat32(type, chunkPtr, scratch, chunkElements);

                float* inputChunk = inputs + elementOffset;
                for (int row = 0; row < rowCount; row++)
                {
                    outputs[row] += DotFloat(inputChunk + (long)row * inputRowStride, scratch, chunkElements);
                }

                chunkPtr += GetDotChunkBytes(type, chunkElements);
                elementOffset += chunkElements;
            }
        }

        // 中文：尝试直接对量化权重矩阵（非托管指针）做 addmm 矩阵乘并累加到 FP32 输出，激活先量化为匹配格式再做整数点积；
        //       算法原理：将 FP32 激活动态量化为 Q8_0/Q8_1/Q8_K，与权重逐块做整数内积后乘以联合缩放因子，
        //       避免全量反量化为 FP32，显著降低内存带宽；支持 Parallel.For 列并行加速（outDim >= 128 时启用）。
        //       参考 GGUF 规范及 k-quantization：https://github.com/ggerganov/llama.cpp/pull/1684
        public static unsafe bool TryAddmmQuantizedToFloat32(
            int ggmlType,
            IntPtr weights,
            long ne0,
            long ne1,
            float* input,
            int inputRowStride,
            int rowCount,
            float* output,
            int outputRowStride)
        {
            var type = (GgmlTensorType)ggmlType;
            if (ne0 > int.MaxValue || ne1 > int.MaxValue)
                return false;

            if (!TryGetDirectMatMulPlan(type, (int)ne0, out ActivationQuantKind activationKind, out int activationRowBytes))
                return false;

            if (weights == IntPtr.Zero)
                throw new ArgumentException("Quantized weights pointer cannot be null.", nameof(weights));
            if (inputRowStride < ne0)
                throw new ArgumentOutOfRangeException(nameof(inputRowStride));
            if (outputRowStride < ne1)
                throw new ArgumentOutOfRangeException(nameof(outputRowStride));

            long totalActivationBytes = (long)rowCount * activationRowBytes;
            if (totalActivationBytes > int.MaxValue)
                return false;

            byte[] rented = ArrayPool<byte>.Shared.Rent((int)totalActivationBytes);
            try
            {
                fixed (byte* activationBase = rented)
                {
                    for (int row = 0; row < rowCount; row++)
                    {
                        byte* dst = activationBase + (long)row * activationRowBytes;
                        float* src = input + (long)row * inputRowStride;
                        QuantizeActivation(src, dst, (int)ne0, activationKind);
                    }

                    byte* weightBase = (byte*)weights.ToPointer();
                    int weightRowBytes = (int)RowSize(ggmlType, ne0);
                    int outDim = (int)ne1;
                    int inDim = (int)ne0;
                    nint activationAddress = (nint)activationBase;
                    nint weightAddress = (nint)weightBase;
                    nint outputAddress = (nint)output;

                    void ComputeColumnRange(int startCol, int endCol)
                    {
                        byte* activationPtr = (byte*)activationAddress;
                        byte* weightPtr = (byte*)weightAddress;
                        float* outputPtr = (float*)outputAddress;

                        for (int col = startCol; col < endCol; col++)
                        {
                            byte* weightRow = weightPtr + (long)col * weightRowBytes;
                            for (int row = 0; row < rowCount; row++)
                            {
                                byte* activationRow = activationPtr + (long)row * activationRowBytes;
                                outputPtr[(long)row * outputRowStride + col] =
                                    DotQuantized(type, weightRow, activationRow, inDim);
                            }
                        }
                    }

                    bool useParallel = outDim >= 128 && (long)rowCount * outDim >= 512 && Environment.ProcessorCount > 1;
                    if (useParallel)
                    {
                        Parallel.For(0, outDim, col => ComputeColumnRange(col, col + 1));
                    }
                    else
                    {
                        ComputeColumnRange(0, outDim);
                    }
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(rented);
            }

            return true;
        }

        // 中文：TryAddmmQuantizedToFloat32 的托管数组重载，将数组偏移转换为裸指针后委托给指针版本
        public static unsafe bool TryAddmmQuantizedToFloat32(
            int ggmlType,
            byte[] weights,
            int weightsOffset,
            long ne0,
            long ne1,
            float[] input,
            int inputOffset,
            int inputRowStride,
            int rowCount,
            float[] output,
            int outputOffset,
            int outputRowStride)
        {
            if (weights == null)
                throw new ArgumentNullException(nameof(weights));
            if (input == null)
                throw new ArgumentNullException(nameof(input));
            if (output == null)
                throw new ArgumentNullException(nameof(output));

            fixed (byte* weightPtr = weights)
            fixed (float* inputPtr = input)
            fixed (float* outputPtr = output)
            {
                return TryAddmmQuantizedToFloat32(
                    ggmlType,
                    (IntPtr)(weightPtr + weightsOffset),
                    ne0,
                    ne1,
                    inputPtr + inputOffset,
                    inputRowStride,
                    rowCount,
                    outputPtr + outputOffset,
                    outputRowStride);
            }
        }

        private enum ActivationQuantKind
        {
            Q8_0,
            Q8_1,
            Q8_K,
        }

        // 中文：根据权重量化类型选择对应的激活量化策略（Q8_0/Q8_1/Q8_K），并计算量化后每行的字节数
        private static bool TryGetDirectMatMulPlan(
            GgmlTensorType type,
            int elementCount,
            out ActivationQuantKind activationKind,
            out int activationRowBytes)
        {
            activationKind = default;
            activationRowBytes = 0;

            switch (type)
            {
                case GgmlTensorType.Q4_0:
                case GgmlTensorType.Q5_0:
                case GgmlTensorType.Q8_0:
                case GgmlTensorType.Q8_1:
                    if (elementCount % QK8_0 != 0)
                        return false;
                    activationKind = ActivationQuantKind.Q8_0;
                    activationRowBytes = elementCount / QK8_0 * Q8_0BlockBytes;
                    return true;

                case GgmlTensorType.Q4_1:
                case GgmlTensorType.Q5_1:
                    if (elementCount % QK8_1 != 0)
                        return false;
                    activationKind = ActivationQuantKind.Q8_1;
                    activationRowBytes = elementCount / QK8_1 * Q8_1BlockBytes;
                    return true;

                case GgmlTensorType.Q4_K:
                case GgmlTensorType.Q5_K:
                case GgmlTensorType.Q6_K:
                    if (elementCount % QK_K != 0)
                        return false;
                    activationKind = ActivationQuantKind.Q8_K;
                    activationRowBytes = elementCount / QK_K * Q8_KBlockBytes;
                    return true;

                default:
                    return false;
            }
        }

        // 中文：将 FP32 激活向量按指定量化策略（Q8_0/Q8_1/Q8_K）量化写入目标字节缓冲区
        private static unsafe void QuantizeActivation(float* src, byte* dst, int elementCount, ActivationQuantKind kind)
        {
            switch (kind)
            {
                case ActivationQuantKind.Q8_0:
                    QuantizeF32ToQ8_0(src, dst, elementCount);
                    return;
                case ActivationQuantKind.Q8_1:
                    QuantizeF32ToQ8_1(src, dst, elementCount);
                    return;
                case ActivationQuantKind.Q8_K:
                    QuantizeF32ToQ8_K(src, dst, elementCount);
                    return;
                default:
                    throw new ArgumentOutOfRangeException(nameof(kind), kind, null);
            }
        }

        // 中文：根据权重量化类型分派到对应的整数向量点积函数（VecDotXxx），返回单行点积结果
        private static unsafe float DotQuantized(GgmlTensorType type, byte* weightRow, byte* activationRow, int elementCount)
        {
            return type switch
            {
                GgmlTensorType.Q4_0 => VecDotQ4_0Q8_0(weightRow, activationRow, elementCount / QK4_0),
                GgmlTensorType.Q4_1 => VecDotQ4_1Q8_1(weightRow, activationRow, elementCount / QK4_1),
                GgmlTensorType.Q5_0 => VecDotQ5_0Q8_0(weightRow, activationRow, elementCount / QK5_0),
                GgmlTensorType.Q5_1 => VecDotQ5_1Q8_1(weightRow, activationRow, elementCount / QK5_1),
                GgmlTensorType.Q8_0 => VecDotQ8_0Q8_0(weightRow, activationRow, elementCount / QK8_0),
                GgmlTensorType.Q8_1 => VecDotQ8_1Q8_0(weightRow, activationRow, elementCount / QK8_1),
                GgmlTensorType.Q4_K => VecDotQ4_KQ8_K(weightRow, activationRow, elementCount / QK_K),
                GgmlTensorType.Q5_K => VecDotQ5_KQ8_K(weightRow, activationRow, elementCount / QK_K),
                GgmlTensorType.Q6_K => VecDotQ6_KQ8_K(weightRow, activationRow, elementCount / QK_K),
                _ => throw new NotSupportedException($"Direct managed quantized matmul does not support {type}."),
            };
        }

        // 中文：统一入口：按张量类型将量化字节流反量化为 FP32 数组，分派到各格式专用实现
        private static unsafe void DequantizeToFloat32(GgmlTensorType type, byte* src, float* dst, long numElements)
        {
            switch (type)
            {
                case GgmlTensorType.F32:
                    Buffer.MemoryCopy(src, dst, numElements * sizeof(float), numElements * sizeof(float));
                    return;
                case GgmlTensorType.F16:
                    DequantizeF16(src, dst, numElements);
                    return;
                case GgmlTensorType.BF16:
                    DequantizeBf16(src, dst, numElements);
                    return;
                case GgmlTensorType.I8:
                    DequantizeI8(src, dst, numElements);
                    return;
                case GgmlTensorType.I16:
                    DequantizeI16(src, dst, numElements);
                    return;
                case GgmlTensorType.I32:
                    DequantizeI32(src, dst, numElements);
                    return;
                case GgmlTensorType.I64:
                    DequantizeI64(src, dst, numElements);
                    return;
                case GgmlTensorType.F64:
                    DequantizeF64(src, dst, numElements);
                    return;
                case GgmlTensorType.Q4_0:
                    DequantizeQ40(src, dst, numElements);
                    return;
                case GgmlTensorType.Q4_1:
                    DequantizeQ41(src, dst, numElements);
                    return;
                case GgmlTensorType.Q5_0:
                    DequantizeQ50(src, dst, numElements);
                    return;
                case GgmlTensorType.Q5_1:
                    DequantizeQ51(src, dst, numElements);
                    return;
                case GgmlTensorType.Q8_0:
                    DequantizeQ80(src, dst, numElements);
                    return;
                case GgmlTensorType.Q8_1:
                    DequantizeQ81(src, dst, numElements);
                    return;
                case GgmlTensorType.Q4_K:
                    DequantizeQ4K(src, dst, numElements);
                    return;
                case GgmlTensorType.Q5_K:
                    DequantizeQ5K(src, dst, numElements);
                    return;
                case GgmlTensorType.Q6_K:
                    DequantizeQ6K(src, dst, numElements);
                    return;
                case GgmlTensorType.IQ4_NL:
                    DequantizeIq4Nl(src, dst, numElements);
                    return;
                case GgmlTensorType.MXFP4:
                    DequantizeMxfp4(src, dst, numElements);
                    return;
                default:
                    throw new NotSupportedException($"Pure C# backend does not support GGUF tensor type {type}.");
            }
        }

        // 中文：将 IEEE 754 FP16（半精度）字节流逐元素转换为 FP32
        private static unsafe void DequantizeF16(byte* src, float* dst, long numElements)
        {
            for (long i = 0; i < numElements; i++)
                dst[i] = HalfToSingle(ReadUInt16(src + i * 2));
        }

        // 中文：将 BF16（脑浮点）字节流逐元素转换为 FP32，原理：BF16 低 16 位为 0，高位与 FP32 对齐，直接左移 16 位即得 FP32 位模式
        private static unsafe void DequantizeBf16(byte* src, float* dst, long numElements)
        {
            for (long i = 0; i < numElements; i++)
            {
                uint bits = (uint)ReadUInt16(src + i * 2) << 16;
                dst[i] = BitConverter.Int32BitsToSingle((int)bits);
            }
        }

        // 中文：将有符号 8 位整数（I8）字节流逐元素转换为 FP32
        private static unsafe void DequantizeI8(byte* src, float* dst, long numElements)
        {
            for (long i = 0; i < numElements; i++)
                dst[i] = ((sbyte*)src)[i];
        }

        // 中文：将有符号 16 位整数（I16）字节流逐元素转换为 FP32
        private static unsafe void DequantizeI16(byte* src, float* dst, long numElements)
        {
            for (long i = 0; i < numElements; i++)
                dst[i] = (short)ReadUInt16(src + i * 2);
        }

        // 中文：将有符号 32 位整数（I32）字节流逐元素转换为 FP32
        private static unsafe void DequantizeI32(byte* src, float* dst, long numElements)
        {
            for (long i = 0; i < numElements; i++)
                dst[i] = ReadInt32(src + i * 4);
        }

        // 中文：将有符号 64 位整数（I64）字节流逐元素转换为 FP32
        private static unsafe void DequantizeI64(byte* src, float* dst, long numElements)
        {
            for (long i = 0; i < numElements; i++)
                dst[i] = ReadInt64(src + i * 8);
        }

        // 中文：将 FP64（双精度）字节流逐元素截断转换为 FP32
        private static unsafe void DequantizeF64(byte* src, float* dst, long numElements)
        {
            for (long i = 0; i < numElements; i++)
                dst[i] = (float)ReadDouble(src + i * 8);
        }

        // 中文：Q4_0 格式反量化：每块 32 个元素，格式为 [scale(FP16) | 16字节nibbles]，
        //       每个 nibble 值域 [0,15]，减去 8 得到有符号值 [-8,7]，再乘以缩放因子 d。
        //       参考 GGUF 规范（对称零点量化）：https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
        private static unsafe void DequantizeQ40(byte* src, float* dst, long numElements)
        {
            if (numElements % QK4_0 != 0)
                throw new NotSupportedException($"Q4_0 requires {QK4_0}-element alignment, got {numElements}.");

            int nb = (int)(numElements / QK4_0);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * (2 + QK4_0 / 2);
                float d = HalfToSingle(ReadUInt16(block));
                byte* qs = block + 2;
                float* y = dst + i * QK4_0;
                for (int j = 0; j < QK4_0 / 2; j++)
                {
                    int x0 = (qs[j] & 0x0F) - 8;
                    int x1 = (qs[j] >> 4) - 8;
                    y[j] = x0 * d;
                    y[j + QK4_0 / 2] = x1 * d;
                }
            }
        }

        // 中文：Q4_1 格式反量化：每块 32 个元素，格式为 [scale(FP16) | min(FP16) | 16字节nibbles]，
        //       nibble 值域 [0,15]（无符号），公式：x = q * d + m，存在非零偏移 m（最小值补偿）。
        //       参考 GGUF 规范（非对称量化）：https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
        private static unsafe void DequantizeQ41(byte* src, float* dst, long numElements)
        {
            if (numElements % QK4_1 != 0)
                throw new NotSupportedException($"Q4_1 requires {QK4_1}-element alignment, got {numElements}.");

            int nb = (int)(numElements / QK4_1);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * (4 + QK4_1 / 2);
                float d = HalfToSingle(ReadUInt16(block));
                float m = HalfToSingle(ReadUInt16(block + 2));
                byte* qs = block + 4;
                float* y = dst + i * QK4_1;
                for (int j = 0; j < QK4_1 / 2; j++)
                {
                    int x0 = qs[j] & 0x0F;
                    int x1 = qs[j] >> 4;
                    y[j] = x0 * d + m;
                    y[j + QK4_1 / 2] = x1 * d + m;
                }
            }
        }

        // 中文：Q5_0 格式反量化：在 Q4_0 基础上每个元素额外存储 1 位高位（qh），
        //       每块格式为 [scale(FP16) | qh(4字节) | 16字节nibbles]，有效值域 [-16,15]（对称），公式：x = q5 * d。
        //       参考 GGUF 规范：https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
        private static unsafe void DequantizeQ50(byte* src, float* dst, long numElements)
        {
            if (numElements % QK5_0 != 0)
                throw new NotSupportedException($"Q5_0 requires {QK5_0}-element alignment, got {numElements}.");

            int blockBytes = 2 + 4 + QK5_0 / 2;
            int nb = (int)(numElements / QK5_0);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * blockBytes;
                float d = HalfToSingle(ReadUInt16(block));
                uint qh = ReadUInt32(block + 2);
                byte* qs = block + 6;
                float* y = dst + i * QK5_0;
                for (int j = 0; j < QK5_0 / 2; j++)
                {
                    int xh0 = (int)(((qh >> j) << 4) & 0x10);
                    int xh1 = (int)((qh >> (j + 12)) & 0x10);
                    int x0 = ((qs[j] & 0x0F) | xh0) - 16;
                    int x1 = ((qs[j] >> 4) | xh1) - 16;
                    y[j] = x0 * d;
                    y[j + QK5_0 / 2] = x1 * d;
                }
            }
        }

        // 中文：Q5_1 格式反量化：在 Q4_1 基础上每个元素额外存储 1 位高位（qh），
        //       格式为 [scale(FP16) | min(FP16) | qh(4字节) | 16字节nibbles]，非对称，公式：x = q5 * d + m。
        //       参考 GGUF 规范：https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
        private static unsafe void DequantizeQ51(byte* src, float* dst, long numElements)
        {
            if (numElements % QK5_1 != 0)
                throw new NotSupportedException($"Q5_1 requires {QK5_1}-element alignment, got {numElements}.");

            int blockBytes = 4 + 4 + QK5_1 / 2;
            int nb = (int)(numElements / QK5_1);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * blockBytes;
                float d = HalfToSingle(ReadUInt16(block));
                float m = HalfToSingle(ReadUInt16(block + 2));
                uint qh = ReadUInt32(block + 4);
                byte* qs = block + 8;
                float* y = dst + i * QK5_1;
                for (int j = 0; j < QK5_1 / 2; j++)
                {
                    int xh0 = (int)(((qh >> j) << 4) & 0x10);
                    int xh1 = (int)((qh >> (j + 12)) & 0x10);
                    int x0 = (qs[j] & 0x0F) | xh0;
                    int x1 = (qs[j] >> 4) | xh1;
                    y[j] = x0 * d + m;
                    y[j + QK5_1 / 2] = x1 * d + m;
                }
            }
        }

        // 中文：Q8_0 格式反量化（对称 8 位量化）：每块 32 个有符号字节，格式为 [scale(FP16) | 32字节 int8]，
        //       公式：x = q * d；d = max(|x|) / 127，参考 GGUF 规范对称量化定义。
        private static unsafe void DequantizeQ80(byte* src, float* dst, long numElements)
        {
            if (numElements % QK8_0 != 0)
                throw new NotSupportedException($"Q8_0 requires {QK8_0}-element alignment, got {numElements}.");

            int blockBytes = 2 + QK8_0;
            int nb = (int)(numElements / QK8_0);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * blockBytes;
                float d = HalfToSingle(ReadUInt16(block));
                sbyte* qs = (sbyte*)(block + 2);
                float* y = dst + i * QK8_0;
                for (int j = 0; j < QK8_0; j++)
                    y[j] = qs[j] * d;
            }
        }

        // 中文：Q8_1 格式反量化：在 Q8_0 基础上额外存储整数和 s（用于与 Q4_1 做快速点积），
        //       格式为 [scale(FP16) | sum(FP16) | 32字节 int8]，公式：x = q * d。
        private static unsafe void DequantizeQ81(byte* src, float* dst, long numElements)
        {
            if (numElements % QK8_1 != 0)
                throw new NotSupportedException($"Q8_1 requires {QK8_1}-element alignment, got {numElements}.");

            int blockBytes = 4 + QK8_1;
            int nb = (int)(numElements / QK8_1);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * blockBytes;
                float d = HalfToSingle(ReadUInt16(block));
                sbyte* qs = (sbyte*)(block + 4);
                float* y = dst + i * QK8_1;
                for (int j = 0; j < QK8_1; j++)
                    y[j] = qs[j] * d;
            }
        }

        // 中文：Q4_K 格式反量化（k-quantization 4 位，超级块 256 元素）：
        //       每个超级块含全局缩放因子 d/min（FP16）、12 字节紧凑 6 位尺度表（8组 scale/min 各 6 位）、
        //       128 字节 nibbles；公式：x = d * sc * q4 - min * mn，实现 GGUF k-quant 规范。
        //       参考 k-quantization PR：https://github.com/ggerganov/llama.cpp/pull/1684
        private static unsafe void DequantizeQ4K(byte* src, float* dst, long numElements)
        {
            if (numElements % QK_K != 0)
                throw new NotSupportedException($"Q4_K requires {QK_K}-element alignment, got {numElements}.");

            int blockBytes = 4 + K_SCALE_SIZE + QK_K / 2;
            int nb = (int)(numElements / QK_K);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * blockBytes;
                float d = HalfToSingle(ReadUInt16(block));
                float min = HalfToSingle(ReadUInt16(block + 2));
                byte* scales = block + 4;
                byte* q = block + 4 + K_SCALE_SIZE;
                float* y = dst + i * QK_K;
                int isIdx = 0;
                for (int j = 0; j < QK_K; j += 64)
                {
                    GetScaleMinK4(isIdx, scales, out byte sc1, out byte m1q);
                    GetScaleMinK4(isIdx + 1, scales, out byte sc2, out byte m2q);
                    float d1 = d * sc1;
                    float d2 = d * sc2;
                    float m1 = min * m1q;
                    float m2 = min * m2q;
                    for (int l = 0; l < 32; l++)
                        y[j + l] = d1 * (q[l] & 0x0F) - m1;
                    for (int l = 0; l < 32; l++)
                        y[j + l + 32] = d2 * (q[l] >> 4) - m2;
                    q += 32;
                    isIdx += 2;
                }
            }
        }

        // 中文：Q5_K 格式反量化（k-quantization 5 位，超级块 256 元素）：
        //       在 Q4_K 基础上每元素额外存储 1 位高位（qh，32字节）；
        //       公式：x = d * sc * (lo4 | hi1<<4) - min * mn，实现 5 位精度的 k-quant。
        //       参考 k-quantization PR：https://github.com/ggerganov/llama.cpp/pull/1684
        private static unsafe void DequantizeQ5K(byte* src, float* dst, long numElements)
        {
            if (numElements % QK_K != 0)
                throw new NotSupportedException($"Q5_K requires {QK_K}-element alignment, got {numElements}.");

            int blockBytes = 4 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2;
            int nb = (int)(numElements / QK_K);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * blockBytes;
                float d = HalfToSingle(ReadUInt16(block));
                float min = HalfToSingle(ReadUInt16(block + 2));
                byte* scales = block + 4;
                byte* qh = block + 4 + K_SCALE_SIZE;
                byte* ql = qh + QK_K / 8;
                float* y = dst + i * QK_K;
                int isIdx = 0;
                byte u1 = 1;
                byte u2 = 2;
                for (int j = 0; j < QK_K; j += 64)
                {
                    GetScaleMinK4(isIdx, scales, out byte sc1, out byte m1q);
                    GetScaleMinK4(isIdx + 1, scales, out byte sc2, out byte m2q);
                    float d1 = d * sc1;
                    float d2 = d * sc2;
                    float m1 = min * m1q;
                    float m2 = min * m2q;
                    for (int l = 0; l < 32; l++)
                        y[j + l] = d1 * ((ql[l] & 0x0F) + ((qh[l] & u1) != 0 ? 16 : 0)) - m1;
                    for (int l = 0; l < 32; l++)
                        y[j + l + 32] = d2 * ((ql[l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0)) - m2;
                    ql += 32;
                    isIdx += 2;
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
        }

        // 中文：Q6_K 格式反量化（k-quantization 6 位，超级块 256 元素）：
        //       低 4 位存于 ql（128字节），高 2 位存于 qh（64字节），16 个有符号 int8 子块 scale；
        //       公式：q6 = (lo4 | hi2<<4) - 32，x = d * scale[sub] * q6，实现 6 位精度 k-quant。
        //       参考 k-quantization PR：https://github.com/ggerganov/llama.cpp/pull/1684
        private static unsafe void DequantizeQ6K(byte* src, float* dst, long numElements)
        {
            if (numElements % QK_K != 0)
                throw new NotSupportedException($"Q6_K requires {QK_K}-element alignment, got {numElements}.");

            int blockBytes = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;
            int nb = (int)(numElements / QK_K);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * blockBytes;
                byte* ql = block;
                byte* qh = ql + QK_K / 2;
                sbyte* scales = (sbyte*)(qh + QK_K / 4);
                float d = HalfToSingle(ReadUInt16((byte*)(scales + QK_K / 16)));
                float* y = dst + i * QK_K;

                for (int n = 0; n < QK_K; n += 128)
                {
                    for (int l = 0; l < 32; l++)
                    {
                        int isIdx = l / 16;
                        sbyte q1 = (sbyte)(((ql[l] & 0x0F) | (((qh[l] >> 0) & 0x03) << 4)) - 32);
                        sbyte q2 = (sbyte)(((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 0x03) << 4)) - 32);
                        sbyte q3 = (sbyte)(((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) - 32);
                        sbyte q4 = (sbyte)(((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) - 32);
                        y[n + l] = d * scales[isIdx] * q1;
                        y[n + l + 32] = d * scales[isIdx + 2] * q2;
                        y[n + l + 64] = d * scales[isIdx + 4] * q3;
                        y[n + l + 96] = d * scales[isIdx + 6] * q4;
                    }

                    ql += 64;
                    qh += 32;
                    scales += 8;
                }
            }
        }

        // 中文：IQ4_NL 格式反量化（非线性 4 位量化）：
        //       16 个量化值由非均匀查找表 Iq4NlValues 给出（值域 [-127,113]，非等间距），
        //       公式：x = d * table[q4]；非线性量化点分布更接近正态分布，减小均方误差。
        //       参考 I-quant 系列 PR：https://github.com/ggerganov/llama.cpp/pull/4773
        private static unsafe void DequantizeIq4Nl(byte* src, float* dst, long numElements)
        {
            if (numElements % QK4_NL != 0)
                throw new NotSupportedException($"IQ4_NL requires {QK4_NL}-element alignment, got {numElements}.");

            int blockBytes = 2 + QK4_NL / 2;
            int nb = (int)(numElements / QK4_NL);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * blockBytes;
                float d = HalfToSingle(ReadUInt16(block));
                byte* qs = block + 2;
                float* y = dst + i * QK4_NL;
                for (int j = 0; j < QK4_NL / 2; j++)
                {
                    y[j] = d * Iq4NlValues[qs[j] & 0x0F];
                    y[j + QK4_NL / 2] = d * Iq4NlValues[qs[j] >> 4];
                }
            }
        }

        // 中文：MXFP4 格式反量化（MX Microscaling 4 位浮点量化）：
        //       每块 32 元素，1 字节 E8M0 共享指数（无尾数的 8 位指数格式）作为缩放因子，
        //       16 个 4 位有符号浮点值由 Mxfp4Values 查找表映射；
        //       E8M0 转 FP32：scale = 2^(exp-1)；公式：x = scale * table[q4]。
        //       参考 MX Microscaling 规范：https://arxiv.org/abs/2310.10537
        private static unsafe void DequantizeMxfp4(byte* src, float* dst, long numElements)
        {
            if (numElements % QK_MXFP4 != 0)
                throw new NotSupportedException($"MXFP4 requires {QK_MXFP4}-element alignment, got {numElements}.");

            int blockBytes = 1 + QK_MXFP4 / 2;
            int nb = (int)(numElements / QK_MXFP4);
            for (int i = 0; i < nb; i++)
            {
                byte* block = src + i * blockBytes;
                float d = E8M0ToFp32Half(block[0]);
                byte* qs = block + 1;
                float* y = dst + i * QK_MXFP4;
                for (int j = 0; j < QK_MXFP4 / 2; j++)
                {
                    y[j] = d * Mxfp4Values[qs[j] & 0x0F];
                    y[j + QK_MXFP4 / 2] = d * Mxfp4Values[qs[j] >> 4];
                }
            }
        }

        // 中文：将 FP32 激活向量量化为 Q8_0 格式（对称 8 位量化）：
        //       每块 32 元素，scale = max(|x|) / 127，量化公式：q = clamp(round(x / scale), -127, 127)；
        //       参考 GGUF Q8_0 对称量化规范。
        private static unsafe void QuantizeF32ToQ8_0(float* src, byte* dst, int elementCount)
        {
            int blockCount = elementCount / QK8_0;
            for (int block = 0; block < blockCount; block++)
            {
                float* blockSrc = src + block * QK8_0;
                byte* blockDst = dst + block * Q8_0BlockBytes;
                float maxAbs = MaxAbs(blockSrc, QK8_0);
                float scale = maxAbs / 127.0f;
                WriteHalf(blockDst, scale);

                sbyte* qs = (sbyte*)(blockDst + 2);
                if (scale == 0.0f)
                {
                    Unsafe.InitBlockUnaligned(qs, 0, QK8_0);
                    continue;
                }

                float invScale = 1.0f / scale;
                for (int i = 0; i < QK8_0; i++)
                    qs[i] = ClampToInt8(MathF.Round(blockSrc[i] * invScale));
            }
        }

        // 中文：将 FP32 激活向量量化为 Q8_1 格式（非对称 8 位量化，额外存储量化整数和 s = scale * sum(q)）：
        //       s 用于与 Q4_1 权重做快速点积时加速计算偏置项 m4 * s8，避免逐元素乘法。
        private static unsafe void QuantizeF32ToQ8_1(float* src, byte* dst, int elementCount)
        {
            int blockCount = elementCount / QK8_1;
            for (int block = 0; block < blockCount; block++)
            {
                float* blockSrc = src + block * QK8_1;
                byte* blockDst = dst + block * Q8_1BlockBytes;
                float maxAbs = MaxAbs(blockSrc, QK8_1);
                float scale = maxAbs / 127.0f;
                WriteHalf(blockDst, scale);

                sbyte* qs = (sbyte*)(blockDst + 4);
                int sum = 0;
                if (scale != 0.0f)
                {
                    float invScale = 1.0f / scale;
                    for (int i = 0; i < QK8_1; i++)
                    {
                        sbyte q = ClampToInt8(MathF.Round(blockSrc[i] * invScale));
                        qs[i] = q;
                        sum += q;
                    }
                }
                else
                {
                    Unsafe.InitBlockUnaligned(qs, 0, QK8_1);
                }

                WriteHalf(blockDst + 2, scale * sum);
            }
        }

        // 中文：将 FP32 激活向量量化为 Q8_K 格式（k-quant 用激活量化，超级块 256 元素）：
        //       scale 以 FP32 存储（而非 FP16），每 16 元素一组记录子块整数和 bsums（short），
        //       供 Q4_K/Q5_K/Q6_K 的点积函数利用 bsums 快速计算偏置项，节省乘法运算。
        //       参考 k-quantization PR：https://github.com/ggerganov/llama.cpp/pull/1684
        private static unsafe void QuantizeF32ToQ8_K(float* src, byte* dst, int elementCount)
        {
            int blockCount = elementCount / QK_K;
            for (int block = 0; block < blockCount; block++)
            {
                float* blockSrc = src + block * QK_K;
                byte* blockDst = dst + block * Q8_KBlockBytes;
                float maxAbs = MaxAbs(blockSrc, QK_K);
                float scale = maxAbs / 127.0f;
                Unsafe.WriteUnaligned(blockDst, scale);

                sbyte* qs = (sbyte*)(blockDst + 4);
                short* bsums = (short*)(blockDst + 4 + QK_K);
                if (scale == 0.0f)
                {
                    Unsafe.InitBlockUnaligned(qs, 0, QK_K);
                    Unsafe.InitBlockUnaligned(bsums, 0, QK_K / 16 * sizeof(short));
                    continue;
                }

                float invScale = 1.0f / scale;
                for (int group = 0; group < QK_K / 16; group++)
                {
                    int sum = 0;
                    int offset = group * 16;
                    for (int i = 0; i < 16; i++)
                    {
                        sbyte q = ClampToInt8(MathF.Round(blockSrc[offset + i] * invScale));
                        qs[offset + i] = q;
                        sum += q;
                    }

                    bsums[group] = (short)sum;
                }
            }
        }

        // 中文：将 float 值以 FP16（半精度）格式写入非对齐字节指针
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void WriteHalf(byte* dst, float value)
        {
            Unsafe.WriteUnaligned(dst, BitConverter.HalfToUInt16Bits((System.Half)value));
        }

        // 中文：将浮点值四舍五入并钳位到 [-127, 127] 范围的有符号 8 位整数（量化截断辅助函数）
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static sbyte ClampToInt8(float value)
        {
            int rounded = (int)value;
            if (rounded > 127) return 127;
            if (rounded < -127) return -127;
            return (sbyte)rounded;
        }

        // 中文：计算浮点数组中绝对值的最大值，优先使用 AVX-512 向量化（16 路并行），
        //       次选 System.Numerics.Vector<float> 自适应 SIMD，兜底标量循环
        private static unsafe float MaxAbs(float* src, int length)
        {
            if (Avx512F.IsSupported && length >= 16)
            {
                Vector512<float> max = Vector512<float>.Zero;
                int i = 0;
                for (; i <= length - 16; i += 16)
                    max = Avx512F.Max(max, Vector512.Abs(Avx512F.LoadVector512(src + i)));

                float result = HorizontalMax(max);
                for (; i < length; i++)
                {
                    float abs = MathF.Abs(src[i]);
                    if (abs > result) result = abs;
                }

                return result;
            }

            int vectorSize = Vector<float>.Count;
            Vector<float> maxVec = Vector<float>.Zero;
            int j = 0;
            for (; j <= length - vectorSize; j += vectorSize)
                maxVec = Vector.Max(maxVec, Vector.Abs(LoadVec(src + j)));

            float maxAbs = 0.0f;
            for (int lane = 0; lane < Vector<float>.Count; lane++)
                if (maxVec[lane] > maxAbs) maxAbs = maxVec[lane];

            for (; j < length; j++)
            {
                float abs = MathF.Abs(src[j]);
                if (abs > maxAbs) maxAbs = abs;
            }

            return maxAbs;
        }

        // 中文：Q4_0 × Q8_0 整数向量点积（对称 4 位权重 × 对称 8 位激活）：
        //       逐块计算 sum(q4_signed * q8)，再乘以联合缩放因子 d4 * d8，
        //       q4 有符号值 = nibble - 8 ∈ [-8, 7]；参考 GGUF 规范对称量化内积。
        private static unsafe float VecDotQ4_0Q8_0(byte* q4, byte* q8, int blockCount)
        {
            float sum = 0.0f;
            for (int block = 0; block < blockCount; block++)
            {
                byte* q4Block = q4 + block * Q4_0BlockBytes;
                byte* q8Block = q8 + block * Q8_0BlockBytes;
                float d4 = HalfToSingle(ReadUInt16(q4Block));
                float d8 = HalfToSingle(ReadUInt16(q8Block));
                byte* qs = q4Block + 2;
                sbyte* qx = (sbyte*)(q8Block + 2);

                int isum = 0;
                for (int i = 0; i < QK4_0 / 2; i++)
                {
                    int low = (qs[i] & 0x0F) - 8;
                    int high = (qs[i] >> 4) - 8;
                    isum += low * qx[i] + high * qx[i + QK4_0 / 2];
                }

                sum += d4 * d8 * isum;
            }

            return sum;
        }

        // 中文：Q4_1 × Q8_1 整数向量点积（非对称 4 位权重 × 非对称 8 位激活）：
        //       公式：dot = d4*d8*sum(q4*q8) + m4*s8，其中 s8 = d8*sum(q8) 预存于 Q8_1 块头，
        //       利用 s8 避免逐元素计算偏置项，降低乘法开销。
        private static unsafe float VecDotQ4_1Q8_1(byte* q4, byte* q8, int blockCount)
        {
            float sum = 0.0f;
            for (int block = 0; block < blockCount; block++)
            {
                byte* q4Block = q4 + block * Q4_1BlockBytes;
                byte* q8Block = q8 + block * Q8_1BlockBytes;
                float d4 = HalfToSingle(ReadUInt16(q4Block));
                float m4 = HalfToSingle(ReadUInt16(q4Block + 2));
                float d8 = HalfToSingle(ReadUInt16(q8Block));
                float s8 = HalfToSingle(ReadUInt16(q8Block + 2));
                byte* qs = q4Block + 4;
                sbyte* qx = (sbyte*)(q8Block + 4);

                int isum = 0;
                for (int i = 0; i < QK4_1 / 2; i++)
                    isum += (qs[i] & 0x0F) * qx[i] + (qs[i] >> 4) * qx[i + QK4_1 / 2];

                sum += d4 * d8 * isum + m4 * s8;
            }

            return sum;
        }

        // 中文：Q5_0 × Q8_0 整数向量点积（对称 5 位权重 × 对称 8 位激活）：
        //       q5 = (lo4 | hi1<<4) - 16 ∈ [-16, 15]，高位 qh 以位掩码方式嵌入；
        //       公式：dot = d5 * d8 * sum(q5 * q8)。
        private static unsafe float VecDotQ5_0Q8_0(byte* q5, byte* q8, int blockCount)
        {
            float sum = 0.0f;
            for (int block = 0; block < blockCount; block++)
            {
                byte* q5Block = q5 + block * Q5_0BlockBytes;
                byte* q8Block = q8 + block * Q8_0BlockBytes;
                float d5 = HalfToSingle(ReadUInt16(q5Block));
                float d8 = HalfToSingle(ReadUInt16(q8Block));
                uint qh = ReadUInt32(q5Block + 2);
                byte* qs = q5Block + 6;
                sbyte* qx = (sbyte*)(q8Block + 2);

                int isum = 0;
                for (int i = 0; i < QK5_0 / 2; i++)
                {
                    int xh0 = (int)(((qh >> i) << 4) & 0x10);
                    int xh1 = (int)((qh >> (i + 12)) & 0x10);
                    int x0 = ((qs[i] & 0x0F) | xh0) - 16;
                    int x1 = ((qs[i] >> 4) | xh1) - 16;
                    isum += x0 * qx[i] + x1 * qx[i + QK5_0 / 2];
                }

                sum += d5 * d8 * isum;
            }

            return sum;
        }

        // 中文：Q5_1 × Q8_1 整数向量点积（非对称 5 位权重 × 非对称 8 位激活）：
        //       公式：dot = d5*d8*sum(q5*q8) + m5*s8，q5 = (lo4 | hi1<<4) ∈ [0, 31]（无符号）。
        private static unsafe float VecDotQ5_1Q8_1(byte* q5, byte* q8, int blockCount)
        {
            float sum = 0.0f;
            for (int block = 0; block < blockCount; block++)
            {
                byte* q5Block = q5 + block * Q5_1BlockBytes;
                byte* q8Block = q8 + block * Q8_1BlockBytes;
                float d5 = HalfToSingle(ReadUInt16(q5Block));
                float m5 = HalfToSingle(ReadUInt16(q5Block + 2));
                uint qh = ReadUInt32(q5Block + 4);
                byte* qs = q5Block + 8;
                float d8 = HalfToSingle(ReadUInt16(q8Block));
                float s8 = HalfToSingle(ReadUInt16(q8Block + 2));
                sbyte* qx = (sbyte*)(q8Block + 4);

                int isum = 0;
                for (int i = 0; i < QK5_1 / 2; i++)
                {
                    int xh0 = (int)(((qh >> i) << 4) & 0x10);
                    int xh1 = (int)((qh >> (i + 12)) & 0x10);
                    int x0 = (qs[i] & 0x0F) | xh0;
                    int x1 = (qs[i] >> 4) | xh1;
                    isum += x0 * qx[i] + x1 * qx[i + QK5_1 / 2];
                }

                sum += d5 * d8 * isum + m5 * s8;
            }

            return sum;
        }

        // 中文：Q8_0 × Q8_0 对称 8 位向量点积，自动分派到 AVX-512 / AVX2 / 标量三条路径
        private static unsafe float VecDotQ8_0Q8_0(byte* q8w, byte* q8x, int blockCount)
        {
            if (Avx512F.IsSupported && Avx512BW.IsSupported)
                return VecDotQ8_0Q8_0Avx512(q8w, q8x, blockCount);
            if (Avx2.IsSupported)
                return VecDotQ8_0Q8_0Avx2(q8w, q8x, blockCount);

            float sum = 0.0f;
            for (int block = 0; block < blockCount; block++)
            {
                byte* wb = q8w + block * Q8_0BlockBytes;
                byte* xb = q8x + block * Q8_0BlockBytes;
                float dw = HalfToSingle(ReadUInt16(wb));
                float dx = HalfToSingle(ReadUInt16(xb));
                sbyte* qw = (sbyte*)(wb + 2);
                sbyte* qx = (sbyte*)(xb + 2);

                int isum = 0;
                for (int i = 0; i < QK8_0; i++)
                    isum += qw[i] * qx[i];
                sum += dw * dx * isum;
            }

            return sum;
        }

        // 中文：Q8_1 × Q8_0 向量点积（Q8_1 权重跳过其存储的 sum 字段，直接与 Q8_0 激活做整数内积）
        private static unsafe float VecDotQ8_1Q8_0(byte* q8w, byte* q8x, int blockCount)
        {
            float sum = 0.0f;
            for (int block = 0; block < blockCount; block++)
            {
                byte* wb = q8w + block * Q8_1BlockBytes;
                byte* xb = q8x + block * Q8_0BlockBytes;
                float dw = HalfToSingle(ReadUInt16(wb));
                float dx = HalfToSingle(ReadUInt16(xb));
                sbyte* qw = (sbyte*)(wb + 4);
                sbyte* qx = (sbyte*)(xb + 2);

                int isum = 0;
                for (int i = 0; i < QK8_1; i++)
                    isum += qw[i] * qx[i];
                sum += dw * dx * isum;
            }

            return sum;
        }

        // 中文：Q8_0 × Q8_0 AVX-512 加速向量点积：
        //       每块一次性将 32 个 int8 扩展为 int16（512位），两两相乘后水平累加到 int32，
        //       再转 FP32 使用 FMA 累加，充分利用 AVX-512BW 的 vpmaddwd 指令。
        private static unsafe float VecDotQ8_0Q8_0Avx512(byte* q8w, byte* q8x, int blockCount)
        {
            Vector512<float> acc = Vector512<float>.Zero;
            Vector512<short> ones = Vector512.Create((short)1);

            for (int block = 0; block < blockCount; block++)
            {
                byte* wb = q8w + block * Q8_0BlockBytes;
                byte* xb = q8x + block * Q8_0BlockBytes;
                float scale = HalfToSingle(ReadUInt16(wb)) * HalfToSingle(ReadUInt16(xb));

                Vector256<sbyte> qwBytes = Unsafe.ReadUnaligned<Vector256<sbyte>>(wb + 2);
                Vector256<sbyte> qxBytes = Unsafe.ReadUnaligned<Vector256<sbyte>>(xb + 2);
                Vector512<short> qw = Avx512BW.ConvertToVector512Int16(qwBytes);
                Vector512<short> qx = Avx512BW.ConvertToVector512Int16(qxBytes);
                Vector512<short> products = Avx512BW.MultiplyLow(qw, qx);
                Vector512<int> pairSums = Avx512BW.MultiplyAddAdjacent(products, ones);
                Vector512<float> dotParts = Avx512F.ConvertToVector512Single(pairSums);

                acc = Avx512F.FusedMultiplyAdd(Vector512.Create(scale), dotParts, acc);
            }

            return HorizontalSum(acc);
        }

        // 中文：Q8_0 × Q8_0 AVX2 加速向量点积：
        //       使用 vpsignb 将权重符号应用到激活（等效于 |w| * sign(w)*x），再用 vpmaddubsw + vpmaddwd
        //       完成 uint8×int8 乘加，最终 FMA 累加到 float256 寄存器。
        private static unsafe float VecDotQ8_0Q8_0Avx2(byte* q8w, byte* q8x, int blockCount)
        {
            Vector256<float> acc = Vector256<float>.Zero;
            Vector256<short> ones = Vector256.Create((short)1);

            for (int block = 0; block < blockCount; block++)
            {
                byte* wb = q8w + block * Q8_0BlockBytes;
                byte* xb = q8x + block * Q8_0BlockBytes;
                float scale = HalfToSingle(ReadUInt16(wb)) * HalfToSingle(ReadUInt16(xb));

                Vector256<sbyte> qw = Unsafe.ReadUnaligned<Vector256<sbyte>>(wb + 2);
                Vector256<sbyte> qx = Unsafe.ReadUnaligned<Vector256<sbyte>>(xb + 2);
                Vector256<sbyte> absW = Avx2.Sign(qw, qw);
                Vector256<sbyte> signedX = Avx2.Sign(qx, qw);
                Vector256<short> prod = Avx2.MultiplyAddAdjacent(absW.AsByte(), signedX);
                Vector256<int> pairSums = Avx2.MultiplyAddAdjacent(prod, ones);
                Vector256<float> dotParts = Avx.ConvertToVector256Single(pairSums);
                acc = Fma.IsSupported
                    ? Fma.MultiplyAdd(Vector256.Create(scale), dotParts, acc)
                    : Avx.Add(acc, Avx.Multiply(Vector256.Create(scale), dotParts));
            }

            return HorizontalSum(acc);
        }

        // 中文：Q4_K × Q8_K 超级块向量点积（k-quantization 4 位权重 × k-quant 8 位激活）：
        //       公式：dot = sum_j{ d8 * (d4 * sc[j] * sum(q4*q8) - dmin * mn[j] * bsum[j]) }，
        //       利用 Q8_K 预存的 bsums（每 16 元素整数和）快速计算偏置项，避免逐元素乘法。
        //       参考 k-quantization PR：https://github.com/ggerganov/llama.cpp/pull/1684
        private static unsafe float VecDotQ4_KQ8_K(byte* q4k, byte* q8k, int superBlockCount)
        {
            float sum = 0.0f;
            byte* scBuf = stackalloc byte[8];
            byte* mnBuf = stackalloc byte[8];

            for (int block = 0; block < superBlockCount; block++)
            {
                float d4 = HalfToSingle(ReadUInt16(q4k));
                float dmin = HalfToSingle(ReadUInt16(q4k + 2));
                UnpackQ4Q5Scales(q4k + 4, scBuf, mnBuf);
                byte* qs = q4k + 16;
                float d8 = ReadSingle(q8k);
                sbyte* q8Values = (sbyte*)(q8k + 4);
                short* bsums = (short*)(q8k + 4 + QK_K);

                for (int j = 0; j < 8; j++)
                {
                    int pairIndex = j / 2;
                    bool highNibble = (j & 1) != 0;
                    sbyte* q8Vals = q8Values + j * 32;
                    int prodSum = 0;
                    for (int i = 0; i < 32; i++)
                    {
                        int raw = qs[pairIndex * 32 + i];
                        int q = highNibble ? raw >> 4 : raw & 0x0F;
                        prodSum += q * q8Vals[i];
                    }

                    int q8Sum = bsums[j * 2] + bsums[j * 2 + 1];
                    sum += d8 * (d4 * scBuf[j] * prodSum - dmin * mnBuf[j] * q8Sum);
                }

                q4k += Q4_KBlockBytes;
                q8k += Q8_KBlockBytes;
            }

            return sum;
        }

        // 中文：Q5_K × Q8_K 超级块向量点积（k-quantization 5 位权重 × k-quant 8 位激活）：
        //       在 Q4_K 点积基础上，每元素额外从 qh 中取第 5 位（lo4 | bit5<<4），提升精度；
        //       公式同 Q4_K，参考 k-quantization PR：https://github.com/ggerganov/llama.cpp/pull/1684
        private static unsafe float VecDotQ5_KQ8_K(byte* q5k, byte* q8k, int superBlockCount)
        {
            float sum = 0.0f;
            byte* scBuf = stackalloc byte[8];
            byte* mnBuf = stackalloc byte[8];

            for (int block = 0; block < superBlockCount; block++)
            {
                float d5 = HalfToSingle(ReadUInt16(q5k));
                float dmin = HalfToSingle(ReadUInt16(q5k + 2));
                UnpackQ4Q5Scales(q5k + 4, scBuf, mnBuf);
                byte* qh = q5k + 16;
                byte* qs = q5k + 48;
                float d8 = ReadSingle(q8k);
                sbyte* q8Values = (sbyte*)(q8k + 4);
                short* bsums = (short*)(q8k + 4 + QK_K);

                for (int j = 0; j < 8; j++)
                {
                    int pairIndex = j / 2;
                    bool highNibble = (j & 1) != 0;
                    sbyte* q8Vals = q8Values + j * 32;
                    int prodSum = 0;
                    for (int i = 0; i < 32; i++)
                    {
                        int raw = qs[pairIndex * 32 + i];
                        int lo4 = highNibble ? raw >> 4 : raw & 0x0F;
                        int bit5 = (qh[i] >> j) & 1;
                        prodSum += (lo4 | (bit5 << 4)) * q8Vals[i];
                    }

                    int q8Sum = bsums[j * 2] + bsums[j * 2 + 1];
                    sum += d8 * (d5 * scBuf[j] * prodSum - dmin * mnBuf[j] * q8Sum);
                }

                q5k += Q5_KBlockBytes;
                q8k += Q8_KBlockBytes;
            }

            return sum;
        }

        // 中文：Q6_K × Q8_K 超级块向量点积（k-quantization 6 位权重 × k-quant 8 位激活）：
        //       q6 = (lo4 | hi2<<4) - 32 ∈ [-32, 31]，16 个子块各有独立 scale；
        //       公式：dot = sum_{sub}{ d6 * d8 * scales[sub] * sum(q6 * q8) }；
        //       参考 k-quantization PR：https://github.com/ggerganov/llama.cpp/pull/1684
        private static unsafe float VecDotQ6_KQ8_K(byte* q6k, byte* q8k, int superBlockCount)
        {
            float sum = 0.0f;

            for (int block = 0; block < superBlockCount; block++)
            {
                byte* ql = q6k;
                byte* qh = q6k + QK_K / 2;
                sbyte* scales = (sbyte*)(q6k + QK_K / 2 + QK_K / 4);
                float d6 = HalfToSingle(ReadUInt16((byte*)(scales + QK_K / 16)));
                float d8 = ReadSingle(q8k);
                sbyte* q8Values = (sbyte*)(q8k + 4);
                float scaleBase = d6 * d8;

                for (int sub = 0; sub < 16; sub++)
                {
                    float scale = scaleBase * scales[sub];
                    sbyte* q8Vals = q8Values + sub * 16;
                    int half = sub / 8;
                    int sh = sub % 8;
                    int qlOffset = half * 64 + (sh % 4) * 16;
                    bool isUpper = sh >= 4;
                    int qhOffset = half * 32 + (sh % 2) * 16;
                    int qhShift = (sh / 2) * 2;

                    int isum = 0;
                    for (int i = 0; i < 16; i++)
                    {
                        int lo4 = isUpper ? (ql[qlOffset + i] >> 4) & 0x0F : ql[qlOffset + i] & 0x0F;
                        int hi2 = (qh[qhOffset + i] >> qhShift) & 0x03;
                        int q6 = (lo4 | (hi2 << 4)) - 32;
                        isum += q6 * q8Vals[i];
                    }

                    sum += scale * isum;
                }

                q6k += Q6_KBlockBytes;
                q8k += Q8_KBlockBytes;
            }

            return sum;
        }

        // 中文：将 K-quant 紧凑 12 字节 scales/mins 字节流解包为 8 个独立的 scale 和 min 值
        private static unsafe void UnpackQ4Q5Scales(byte* packed, byte* scales, byte* mins)
        {
            for (int i = 0; i < 8; i++)
                GetScaleMinK4(i, packed, out scales[i], out mins[i]);
        }

        // 中文：根据量化类型计算每次点积分块处理的元素数量（以 QK_K 或单块大小为上限）
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int GetDotChunkSize(GgmlTensorType type, long remaining)
        {
            return type switch
            {
                GgmlTensorType.F16 or GgmlTensorType.BF16 or
                GgmlTensorType.I8 or GgmlTensorType.I16 or GgmlTensorType.I32 or
                GgmlTensorType.I64 or GgmlTensorType.F64 => (int)Math.Min(remaining, QK_K),
                _ => (int)Math.Min(remaining, GgufFile.GetBlockSize(type)),
            };
        }

        // 中文：根据量化类型和分块元素数计算该分块的字节大小（用于推进源指针）
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int GetDotChunkBytes(GgmlTensorType type, int chunkElements)
        {
            return type switch
            {
                GgmlTensorType.F32 => chunkElements * sizeof(float),
                GgmlTensorType.F16 or GgmlTensorType.BF16 => chunkElements * sizeof(ushort),
                GgmlTensorType.I8 => chunkElements,
                GgmlTensorType.I16 => chunkElements * sizeof(short),
                GgmlTensorType.I32 => chunkElements * sizeof(int),
                GgmlTensorType.I64 or GgmlTensorType.F64 => chunkElements * sizeof(long),
                _ => (int)GgufFile.GetTypeSize(type),
            };
        }

        // 中文：从非对齐指针加载一个 System.Numerics.Vector<float>（平台自适应 SIMD 宽度）
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector<float> LoadVec(float* ptr) => Unsafe.ReadUnaligned<Vector<float>>(ref *(byte*)ptr);

        // 中文：调用 TensorPrimitives.Dot 计算两个 FP32 向量的内积（硬件加速路径由运行时选择）
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe float DotFloat(float* lhs, float* rhs, int length)
        {
            return TensorPrimitives.Dot(
                new ReadOnlySpan<float>(lhs, length),
                new ReadOnlySpan<float>(rhs, length));
        }

        // 中文：从非对齐字节指针读取 UInt16（小端序）
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe ushort ReadUInt16(byte* p) => Unsafe.ReadUnaligned<ushort>(ref *p);

        // 中文：从非对齐字节指针读取 UInt32（小端序）
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe uint ReadUInt32(byte* p) => Unsafe.ReadUnaligned<uint>(ref *p);

        // 中文：从非对齐字节指针读取 Int32（小端序）
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe int ReadInt32(byte* p) => Unsafe.ReadUnaligned<int>(ref *p);

        // 中文：从非对齐字节指针读取 Int64（小端序）
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe long ReadInt64(byte* p) => Unsafe.ReadUnaligned<long>(ref *p);

        // 中文：从非对齐字节指针读取 Double（小端序）
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe double ReadDouble(byte* p) => Unsafe.ReadUnaligned<double>(ref *p);

        // 中文：从非对齐字节指针读取 Single（float，小端序）
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe float ReadSingle(byte* p) => Unsafe.ReadUnaligned<float>(ref *p);

        // 中文：将 IEEE 754 FP16 位模式（UInt16）转换为 FP32（System.Half 中间转换）
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float HalfToSingle(ushort value) => (float)BitConverter.UInt16BitsToHalf(value);

        // 中文：对 AVX2 Vector256<float> 做水平求和，将 8 个 lane 的值累加为标量
        private static unsafe float HorizontalSum(Vector256<float> v)
        {
            float* tmp = stackalloc float[8];
            Avx.Store(tmp, v);
            float sum = 0.0f;
            for (int i = 0; i < 8; i++)
                sum += tmp[i];
            return sum;
        }

        // 中文：对 AVX-512 Vector512<float> 做水平求和，将 16 个 lane 的值累加为标量
        private static unsafe float HorizontalSum(Vector512<float> v)
        {
            float* tmp = stackalloc float[16];
            Avx512F.Store(tmp, v);
            float sum = 0.0f;
            for (int i = 0; i < 16; i++)
                sum += tmp[i];
            return sum;
        }

        // 中文：对 AVX-512 Vector512<float> 做水平最大值归约，返回 16 个 lane 中的最大值
        private static unsafe float HorizontalMax(Vector512<float> v)
        {
            float* tmp = stackalloc float[16];
            Avx512F.Store(tmp, v);
            float max = tmp[0];
            for (int i = 1; i < 16; i++)
                if (tmp[i] > max) max = tmp[i];
            return max;
        }

        // 中文：将 MXFP4 格式的 E8M0 共享指数字节转换为 FP32 缩放因子：
        //       E8M0 为纯指数格式（无尾数、无符号位），scale = 2^(value-1)；
        //       特殊处理 value<2 的小值以避免下溢；参考 MX 规范：https://arxiv.org/abs/2310.10537
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float E8M0ToFp32Half(byte value)
        {
            uint bits = value < 2 ? 0x00200000u << value : ((uint)value - 1u) << 23;
            return BitConverter.Int32BitsToSingle((int)bits);
        }

        // 中文：从 K-quant 紧凑 12 字节尺度数组中解包第 j 个 scale（d）和 min（m）值（各 6 位）：
        //       前 4 组（j<4）：低 6 位存于 q[j] 和 q[j+4]；后 4 组：高 2 位借用前一字节高位拼接。
        //       参考 k-quantization PR：https://github.com/ggerganov/llama.cpp/pull/1684
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void GetScaleMinK4(int j, byte* q, out byte d, out byte m)
        {
            if (j < 4)
            {
                d = (byte)(q[j] & 63);
                m = (byte)(q[j + 4] & 63);
                return;
            }

            d = (byte)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
            m = (byte)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
        }
    }
}
