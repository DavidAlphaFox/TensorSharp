// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using AdvUtils;
using System;
using System.Data;
using System.Reflection;
using System.Runtime.InteropServices;
using TensorSharp.Core;

namespace TensorSharp.Cpu
{
    [OpsClass]
    public class CpuBasicOps
    {
        public CpuBasicOps()
        {
        }


        // 中文：点积/矩阵乘法分发，根据 lhs/rhs 的维度选择向量点积、矩阵乘向量或矩阵乘矩阵。
        [RegisterOpStorageType("dot", typeof(CpuStorage))]
        public static Tensor Dot(Tensor result, Tensor lhs, Tensor rhs)
        {
            if (lhs.DimensionCount == 1 && rhs.DimensionCount == 1)
            {
                return MatrixMultiplication.Dot(result, lhs, rhs);
            }
            else if (lhs.DimensionCount == 2 && rhs.DimensionCount == 1)
            {
                return MatrixMultiplication.Mul_M_V(result, lhs, rhs);
            }
            else if (lhs.DimensionCount == 2 && rhs.DimensionCount == 2)
            {
                return MatrixMultiplication.Mul_M_M(result, lhs, rhs);
            }
            else
            {
                throw new NotSupportedException(message: string.Format("Multiplication of {0}D with {1}D tensor is not supported"));
            }
        }

        // 中文：矩阵乘加运算 result = beta*src + alpha*(m1·m2)，对二维矩阵执行 GEMM。
        [RegisterOpStorageType("addmm", typeof(CpuStorage))]
        public static Tensor Addmm(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
        {
            try
            {
                if (src.ElementType != m1.ElementType || src.ElementType != m2.ElementType || (result != null && result.ElementType != src.ElementType))
                {
                    throw new InvalidOperationException($"All tensors must have the same element type  src = '{src.ElementType}', m1 = '{m1.ElementType}', m2 = '{m2.ElementType}' result = '{result.ElementType}'");
                }

                if (result != null && !(result.Storage is CpuStorage))
                {
                    throw new ArgumentException("result must be a CPU tensor", nameof(result));
                }

                if (!(m1.Storage is CpuStorage))
                {
                    throw new ArgumentException("m1 must be a CPU tensor", nameof(m1));
                }

                if (!(m2.Storage is CpuStorage))
                {
                    throw new ArgumentException("m2 must be a CPU tensor", nameof(m2));
                }

                if (src.DimensionCount != 2)
                {
                    throw new ArgumentException("src must be a matrix", nameof(src));
                }

                if (m1.DimensionCount != 2)
                {
                    throw new ArgumentException("m1 must be a matrix", nameof(m1));
                }

                if (m2.DimensionCount != 2)
                {
                    throw new ArgumentException("m2 must be a matrix", nameof(m2));
                }

                if (src.Sizes[0] != m1.Sizes[0] || src.Sizes[1] != m2.Sizes[1] || m1.Sizes[1] != m2.Sizes[0])
                {
                    throw new InvalidOperationException("Size mismatch");
                }

                Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);

                if (writeTarget != src)
                {
                    Ops.Copy(writeTarget, src);
                }


                MatrixMultiplication.Gemm(alpha, m1, m2, beta, writeTarget);


                return writeTarget;
            }
            catch (Exception err)
            {
                Logger.WriteLine(Logger.Level.err, $"Exception = '{err.Message}'.");
                Logger.WriteLine(Logger.Level.debug, $"Call stack = '{err.StackTrace}'");

                throw;
            }
        }

        // 中文：批量矩阵乘加 result = beta*src + alpha*(m1·m2)，对三维[batch,M,N]张量执行批量 GEMM。
        [RegisterOpStorageType("addmmbatch", typeof(CpuStorage))]
        public static Tensor AddmmBatch(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
        {
            if (src.ElementType != m1.ElementType || src.ElementType != m2.ElementType || (result != null && result.ElementType != src.ElementType))
            {
                throw new InvalidOperationException($"All tensors must have the same element type  src = '{src.ElementType}', m1 = '{m1.ElementType}', m2 = '{m2.ElementType}' result = '{result.ElementType}'");
            }

            if (result != null && !(result.Storage is CpuStorage))
            {
                throw new ArgumentException("result must be a CPU tensor", nameof(result));
            }

            if (!(m1.Storage is CpuStorage))
            {
                throw new ArgumentException("m1 must be a CPU tensor", nameof(m1));
            }

            if (!(m2.Storage is CpuStorage))
            {
                throw new ArgumentException("m2 must be a CPU tensor", nameof(m2));
            }

            if (src.DimensionCount != 3)
            {
                throw new ArgumentException("src must be a 3D batch tensor [batch, M, N]", nameof(src));
            }

            if (m1.DimensionCount != 3)
            {
                throw new ArgumentException("m1 must be a 3D batch tensor [batch, M, K]", nameof(m1));
            }

            if (m2.DimensionCount != 3)
            {
                throw new ArgumentException("m2 must be a 3D batch tensor [batch, K, N]", nameof(m2));
            }

            if (src.Sizes[1] != m1.Sizes[1] || src.Sizes[2] != m2.Sizes[2] || m1.Sizes[2] != m2.Sizes[1])
            {
                throw new InvalidOperationException($"Size mismatch, srcSize0 = {src.Sizes[0]}, m1Size0 = {m1.Sizes[0]}, srcSize1 = {src.Sizes[1]}, m2Size1 = {m2.Sizes[1]}, m1Size1 = '{m1.Sizes[1]}', m2Size0 = '{m2.Sizes[0]}'");
            }

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);

            if (writeTarget != src)
            {
                Ops.Copy(writeTarget, src);
            }

            MatrixMultiplication.GemmBatch(alpha, m1, m2, beta, writeTarget);

            return writeTarget;
        }

        // 中文：MoE 专家路由矩阵乘，按 ids 选择对应专家权重与每个 token 输入做矩阵乘。
        [RegisterOpStorageType("mulmatid", typeof(CpuStorage))]
        public Tensor MulmatID(Tensor result, Tensor expertWeights, Tensor input, Tensor ids)
        {
            if (expertWeights.DimensionCount != 3 || input.DimensionCount != 3 || ids.DimensionCount != 2)
            {
                throw new NotSupportedException("mulmatid expects expertWeights/input to be 3D and ids to be 2D.");
            }

            long tokens = input.Sizes[0];
            long expertUsed = ids.Sizes[1];
            long rows = expertWeights.Sizes[1];
            long cols = expertWeights.Sizes[2];
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, expertWeights.Allocator, DType.Float32, false, tokens, expertUsed, rows);

            bool useIntIds = ids.ElementType == DType.Int32;
            int[] idsInt = useIntIds ? ids.GetElementsAsInt((int)ids.ElementCount()) : Array.Empty<int>();
            for (int token = 0; token < tokens; token++)
            {
                for (int expertSlot = 0; expertSlot < expertUsed; expertSlot++)
                {
                    int expertId = useIntIds
                        ? idsInt[token * (int)expertUsed + expertSlot]
                        : (int)ids.GetElementAsFloat(token, expertSlot);

                    int inputExpertSlot = expertSlot % (int)input.Sizes[1];
                    for (int row = 0; row < rows; row++)
                    {
                        float acc = 0.0f;
                        for (int col = 0; col < cols; col++)
                        {
                            acc += expertWeights.GetElementAsFloat(expertId, row, col) * input.GetElementAsFloat(token, inputExpertSlot, col);
                        }

                        writeTarget.SetElementAsFloat(acc, token, expertSlot, row);
                    }
                }
            }

            return writeTarget;
        }

        // 中文：MoE 专家偏置加法，按 ids 选择对应专家偏置加到 src 对应位置上。
        [RegisterOpStorageType("addid", typeof(CpuStorage))]
        public Tensor AddID(Tensor result, Tensor src, Tensor bias, Tensor ids)
        {
            if (src.DimensionCount != 3 || bias.DimensionCount != 2 || ids.DimensionCount != 2)
            {
                throw new NotSupportedException("addid expects src to be 3D, bias to be 2D, and ids to be 2D.");
            }

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            Ops.Copy(writeTarget, src);

            bool useIntIds = ids.ElementType == DType.Int32;
            int[] idsInt = useIntIds ? ids.GetElementsAsInt((int)ids.ElementCount()) : Array.Empty<int>();
            for (int token = 0; token < src.Sizes[0]; token++)
            {
                for (int expertSlot = 0; expertSlot < src.Sizes[1]; expertSlot++)
                {
                    int expertId = useIntIds
                        ? idsInt[token * (int)ids.Sizes[1] + expertSlot]
                        : (int)ids.GetElementAsFloat(token, expertSlot);

                    for (int row = 0; row < src.Sizes[2]; row++)
                    {
                        float updated = writeTarget.GetElementAsFloat(token, expertSlot, row) + bias.GetElementAsFloat(expertId, row);
                        writeTarget.SetElementAsFloat(updated, token, expertSlot, row);
                    }
                }
            }

            return writeTarget;
        }



        private static bool UseCpuOpsNative => RuntimeInformation.IsOSPlatform(OSPlatform.Windows);

        private readonly MethodInfo abs_func = NativeWrapper.GetMethod("TS_Abs");
        // 中文：逐元素绝对值 |x|。
        [RegisterOpStorageType("abs", typeof(CpuStorage))]
        public Tensor Abs(Tensor result, Tensor src)
        {
            if (UseCpuOpsNative) return NativeWrapper.InvokeNullableResultElementwise(abs_func, result, src);
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            TensorApplyCPU.Abs(writeTarget, src);
            return writeTarget;
        }

        private readonly MethodInfo neg_func = NativeWrapper.GetMethod("TS_Neg");
        // 中文：逐元素取负 -x。
        [RegisterOpStorageType("neg", typeof(CpuStorage))]
        public Tensor Neg(Tensor result, Tensor src)
        {
            if (UseCpuOpsNative) return NativeWrapper.InvokeNullableResultElementwise(neg_func, result, src);
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            TensorApplyCPU.Neg(writeTarget, src);
            return writeTarget;
        }

        private readonly MethodInfo sign_func = NativeWrapper.GetMethod("TS_Sign");
        // 中文：逐元素符号函数 sign(x)。
        [RegisterOpStorageType("sign", typeof(CpuStorage))]
        public Tensor Sign(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(sign_func, result, src); }


        private readonly MethodInfo sqrt_func = NativeWrapper.GetMethod("TS_Sqrt");
        // 中文：逐元素平方根 sqrt(x)。
        [RegisterOpStorageType("sqrt", typeof(CpuStorage))]
        public Tensor Sqrt(Tensor result, Tensor src)
        {
            if (UseCpuOpsNative) return NativeWrapper.InvokeNullableResultElementwise(sqrt_func, result, src);
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            TensorApplyCPU.Sqrt(writeTarget, src);
            return writeTarget;
        }


        // 中文：逐元素平方根倒数 1/sqrt(x)。
        [RegisterOpStorageType("rsqrt", typeof(CpuStorage))]
        public Tensor Rsqrt(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            TensorApplyCPU.Rsqrt(writeTarget, src);

            return writeTarget;
        }

        // 中文：逐元素自然指数 exp(x)。
        [RegisterOpStorageType("exp", typeof(CpuStorage))]
        public Tensor Exp(Tensor result, Tensor src) 
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            TensorApplyCPU.Exp(writeTarget, src);

            return writeTarget;
        }

        // 中文：逐元素自然对数 log(x)。
        [RegisterOpStorageType("log", typeof(CpuStorage))]
        public Tensor Log(Tensor result, Tensor src) 
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            TensorApplyCPU.Log(writeTarget, src);

            return writeTarget;
        }

        private readonly MethodInfo log1p_func = NativeWrapper.GetMethod("TS_Log1p");
        // 中文：逐元素 log(1+x)，提升小数值精度。
        [RegisterOpStorageType("log1p", typeof(CpuStorage))]
        public Tensor Log1p(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(log1p_func, result, src); }

        private readonly MethodInfo floor_func = NativeWrapper.GetMethod("TS_Floor");
        // 中文：逐元素向下取整 floor(x)。
        [RegisterOpStorageType("floor", typeof(CpuStorage))]
        public Tensor Floor(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(floor_func, result, src); }

        private readonly MethodInfo ceil_func = NativeWrapper.GetMethod("TS_Ceil");
        // 中文：逐元素向上取整 ceil(x)。
        [RegisterOpStorageType("ceil", typeof(CpuStorage))]
        public Tensor Ceil(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(ceil_func, result, src); }

        private readonly MethodInfo round_func = NativeWrapper.GetMethod("TS_Round");
        // 中文：逐元素四舍五入取整 round(x)。
        [RegisterOpStorageType("round", typeof(CpuStorage))]
        public Tensor Round(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(round_func, result, src); }

        private readonly MethodInfo trunc_func = NativeWrapper.GetMethod("TS_Trunc");
        // 中文：逐元素截断取整 trunc(x)（向零取整）。
        [RegisterOpStorageType("trunc", typeof(CpuStorage))]
        public Tensor Trunc(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(trunc_func, result, src); }

        private readonly MethodInfo frac_func = NativeWrapper.GetMethod("TS_Frac");
        // 中文：逐元素取小数部分 frac(x)=x-trunc(x)。
        [RegisterOpStorageType("frac", typeof(CpuStorage))]
        public Tensor Frac(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(frac_func, result, src); }


        // 中文：ReLU 激活，逐元素 max(0,x)。
        [RegisterOpStorageType("relu", typeof(CpuStorage))]
        public Tensor Relu(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            TensorApplyCPU.Relu(writeTarget, src);

            return writeTarget;
        }


        private readonly MethodInfo sin_func = NativeWrapper.GetMethod("TS_Sin");
        // 中文：逐元素正弦 sin(x)。
        [RegisterOpStorageType("sin", typeof(CpuStorage))]
        public Tensor Sin(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(sin_func, result, src); }

        private readonly MethodInfo cos_func = NativeWrapper.GetMethod("TS_Cos");
        // 中文：逐元素余弦 cos(x)。
        [RegisterOpStorageType("cos", typeof(CpuStorage))]
        public Tensor Cos(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(cos_func, result, src); }

        private readonly MethodInfo tan_func = NativeWrapper.GetMethod("TS_Tan");
        // 中文：逐元素正切 tan(x)。
        [RegisterOpStorageType("tan", typeof(CpuStorage))]
        public Tensor Tan(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(tan_func, result, src); }


        private readonly MethodInfo asin_func = NativeWrapper.GetMethod("TS_Asin");
        // 中文：逐元素反正弦 asin(x)。
        [RegisterOpStorageType("asin", typeof(CpuStorage))]
        public Tensor Asin(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(asin_func, result, src); }

        private readonly MethodInfo acos_func = NativeWrapper.GetMethod("TS_Acos");
        // 中文：逐元素反余弦 acos(x)。
        [RegisterOpStorageType("acos", typeof(CpuStorage))]
        public Tensor Acos(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(acos_func, result, src); }

        private readonly MethodInfo atan_func = NativeWrapper.GetMethod("TS_Atan");
        // 中文：逐元素反正切 atan(x)。
        [RegisterOpStorageType("atan", typeof(CpuStorage))]
        public Tensor Atan(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(atan_func, result, src); }


        private readonly MethodInfo sinh_func = NativeWrapper.GetMethod("TS_Sinh");
        // 中文：逐元素双曲正弦 sinh(x)。
        [RegisterOpStorageType("sinh", typeof(CpuStorage))]
        public Tensor Sinh(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(sinh_func, result, src); }

        private readonly MethodInfo cosh_func = NativeWrapper.GetMethod("TS_Cosh");
        // 中文：逐元素双曲余弦 cosh(x)。
        [RegisterOpStorageType("cosh", typeof(CpuStorage))]
        public Tensor Cosh(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(cosh_func, result, src); }

        // 中文：Tanh 激活，逐元素双曲正切 tanh(x)。
        [RegisterOpStorageType("tanh", typeof(CpuStorage))]
        public Tensor Tanh(Tensor result, Tensor src) 
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            TensorApplyCPU.Tanh(writeTarget, src);

            return writeTarget;
        }


        // 中文：Sigmoid 激活，逐元素 1/(1+exp(-x))。
        [RegisterOpStorageType("sigmoid", typeof(CpuStorage))]
        public Tensor Sigmoid(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            TensorApplyCPU.Sigmoid(writeTarget, src);

            return writeTarget;
        }


        // 中文：Tanh 反向梯度，由前向输出 resW 与上游梯度 resG 计算梯度。
        [RegisterOpStorageType("tanhD", typeof(CpuStorage))]
        public Tensor TanhD(Tensor result, Tensor resW, Tensor resG)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, resW, false, resW.Sizes);
            TensorApplyCPU.TanhD(writeTarget, resW, resG);

            return writeTarget;
        }


        // 中文：Sigmoid 反向梯度，由前向输出 resW 与上游梯度 resG 计算梯度。
        [RegisterOpStorageType("sigmoidD", typeof(CpuStorage))]
        public Tensor SigmoidD(Tensor result, Tensor resW, Tensor resG)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, resW, false, resW.Sizes);
            TensorApplyCPU.SigmoidD(writeTarget, resW, resG);

            return writeTarget;
        }


        // 中文：Sigmoid 反向梯度并累加到 t 上（result = t + sigmoid'梯度）。
        [RegisterOpStorageType("addsigmoidD", typeof(CpuStorage))]
        public Tensor AddSigmoidD(Tensor result, Tensor t, Tensor resW, Tensor resG)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, resW, false, resW.Sizes);
            TensorApplyCPU.AddSigmoidD(writeTarget, t, resW, resG);

            return writeTarget;
        }


        private readonly MethodInfo add3_func = NativeWrapper.GetMethod("TS_Add3");
        // 中文：三张量逐元素相加 x+y+z。
        [RegisterOpStorageType("add3", typeof(CpuStorage))]
        public Tensor Add3(Tensor result, Tensor x, Tensor y, Tensor z) { return NativeWrapper.InvokeNullableResultElementwise(add3_func, result, x, y, z); }

        private readonly MethodInfo add4_func = NativeWrapper.GetMethod("TS_Add4");
        // 中文：四张量逐元素相加 x+y+z+w。
        [RegisterOpStorageType("add4", typeof(CpuStorage))]
        public Tensor Add4(Tensor result, Tensor x, Tensor y, Tensor z, Tensor w) { return NativeWrapper.InvokeNullableResultElementwise(add4_func, result, x, y, z, w); }


        // 中文：逐元素融合乘加 x + y*z。
        [RegisterOpStorageType("addmul", typeof(CpuStorage))]
        public Tensor AddMul(Tensor result, Tensor x, Tensor y, Tensor z)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, x, false, x.Sizes);
            TensorApplyCPU.AddMul(writeTarget, x, y, z);

            return writeTarget;
        }

        // 中文：逐元素融合乘除加 x + y/z。
        [RegisterOpStorageType("adddiv", typeof(CpuStorage))]
        public Tensor AddDiv(Tensor result, Tensor x, Tensor y, Tensor z)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, x, false, x.Sizes);
            TensorApplyCPU.AddDiv(writeTarget, x, y, z);

            return writeTarget;
        }

        // 中文：构建源-目标注意力掩码，依据源/目标实际长度填充有效值与掩码值。
        [RegisterOpStorageType("buildsrctgtmask", typeof(CpuStorage))]
        public Tensor BuildSrcTgtMask(Tensor result, Tensor srcOriginalLengths, Tensor tgtOriginalLengths, int srcPaddedSeqLen, int tgtPaddedSeqLen, float value, float maskedValue)
        {

            int ndim = result.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(result.Sizes, result.Strides);
            long cols = result.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;

            TensorApplyCPU.BuildSrcTgtMask(result, srcOriginalLengths, tgtOriginalLengths, (int)rows, (int)cols, tgtPaddedSeqLen, value, maskedValue);
            return result;
        }



        // 中文：构建自注意力填充掩码，依据序列实际长度区分有效位与填充位。
        [RegisterOpStorageType("buildselfmask", typeof(CpuStorage))]
        public Tensor BuildSelfMask(Tensor result, Tensor originalLengths, int paddedSeqLen, float value, float maskedValue)
        {
            int ndim = result.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(result.Sizes, result.Strides);
            long cols = result.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;

            TensorApplyCPU.BuildSelfMask(result, originalLengths, (int)rows, (int)cols, paddedSeqLen, value, maskedValue);
            return result;
        }


        // 中文：构建自注意力下三角因果掩码并结合序列长度，用于自回归解码。
        [RegisterOpStorageType("buildselftrimask", typeof(CpuStorage))]
        public Tensor BuildSelfTriMask(Tensor result, Tensor originalLengths, int paddedSeqLen, float value, float maskedValue)
        {
            int ndim = result.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(result.Sizes, result.Strides);
            long cols = result.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;

            TensorApplyCPU.BuildSelfTriMask(result, originalLengths, (int)rows, (int)cols, paddedSeqLen, value, maskedValue);
            return result;
        }


        // 中文：构建下三角因果掩码（不依赖长度），用于自回归注意力。
        [RegisterOpStorageType("buildtrimask", typeof(CpuStorage))]
        public Tensor BuildTriMask(Tensor result, float value, float maskedValue)
        {
            int ndim = result.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(result.Sizes, result.Strides);
            long cols = result.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;

            TensorApplyCPU.BuildTriMask(result, (int)rows, (int)cols, value, maskedValue);
            return result;
        }


        // 中文：逐元素乘加标量版 x + y*z（z 为标量）。
        [RegisterOpStorageType("addmulv", typeof(CpuStorage))]
        public Tensor AddMulV(Tensor result, Tensor x, Tensor y, float z)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, x, false, x.Sizes);
            TensorApplyCPU.AddMulV(writeTarget, x, y, z);
            return writeTarget;

        }

        private readonly MethodInfo maskfill_func = NativeWrapper.GetMethod("TS_MaskFill");
        // 中文：按 mask 掩码填充，将被掩码位置替换为 defValue。
        [RegisterOpStorageType("maskfill", typeof(CpuStorage))]
        public Tensor MaskFill(Tensor result, Tensor t, Tensor mask, float defValue) { return NativeWrapper.InvokeNullableResultElementwise(maskfill_func, result, t, mask, defValue); }


        private readonly MethodInfo atan2_func = NativeWrapper.GetMethod("TS_Atan2");
        // 中文：逐元素二参数反正切 atan2(y,x)。
        [RegisterOpStorageType("atan2", typeof(CpuStorage))]
        public Tensor Atan2(Tensor result, Tensor srcY, Tensor srcX) { return NativeWrapper.InvokeNullableResultElementwise(atan2_func, result, srcY, srcX); }


        // 中文：逐元素幂运算 x^value（指数为标量）。
        [RegisterOpStorageType("pow", typeof(CpuStorage))]
        public Tensor Pow(Tensor result, Tensor src, float value)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            TensorApplyCPU.Pow(writeTarget, src, value);

            return writeTarget;

        }

        private readonly MethodInfo tpow_func = NativeWrapper.GetMethod("TS_Tpow");
        // 中文：逐元素以标量为底的幂运算 value^x。
        [RegisterOpStorageType("tpow", typeof(CpuStorage))]
        public Tensor Tpow(Tensor result, float value, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(tpow_func, result, value, src); }

        private readonly MethodInfo lerp_func = NativeWrapper.GetMethod("TS_Lerp");
        // 中文：逐元素线性插值 srcA + weight*(srcB-srcA)。
        [RegisterOpStorageType("lerp", typeof(CpuStorage))]
        public Tensor Lerp(Tensor result, Tensor srcA, Tensor srcB, float weight) { return NativeWrapper.InvokeNullableResultElementwise(lerp_func, result, srcA, srcB, weight); }

        // private readonly MethodInfo clamp_func = NativeWrapper.GetMethod("TS_Clamp");
        // 中文：逐元素截断到区间 [min,max]。
        [RegisterOpStorageType("clamp", typeof(CpuStorage))]
        public Tensor Clamp(Tensor result, Tensor src, float min, float max)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            TensorApplyCPU.Clamp(writeTarget, src, min, max);

            return writeTarget;

            //return NativeWrapper.InvokeNullableResultElementwise(clamp_func, result, src, min, max);
        }


        // 中文：逐元素融合 srcX*srcY + srcZ*srcW。
        [RegisterOpStorageType("mulmuladd", typeof(CpuStorage))]
        public Tensor MulMulAdd(Tensor result, Tensor srcX, Tensor srcY, Tensor srcZ, Tensor srcW)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, srcX, false, srcX.Sizes);
            TensorApplyCPU.MulMulAdd(writeTarget, srcX, srcY, srcZ, srcW);

            return writeTarget;
        }


        // 中文：逐元素 tanh(srcX + srcY)，先相加再做双曲正切。
        [RegisterOpStorageType("addtanh", typeof(CpuStorage))]
        public Tensor AddTanh(Tensor result, Tensor srcX, Tensor srcY)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, srcX, false, srcX.Sizes);
            TensorApplyCPU.AddTanh(writeTarget, srcX, srcY);

            return writeTarget;
        }


        private readonly MethodInfo addtanh3_func = NativeWrapper.GetMethod("TS_AddTanh3");
        // 中文：逐元素 tanh(srcX + srcY + srcZ)，三项相加再做双曲正切。
        [RegisterOpStorageType("addtanh3", typeof(CpuStorage))]
        public Tensor AddTanh3(Tensor result, Tensor srcX, Tensor srcY, Tensor srcZ) { return NativeWrapper.InvokeNullableResultElementwise(addtanh3_func, result, srcX, srcY, srcZ); }


        // 中文：Tanh 反向梯度并累加（result = srcX + tanh'梯度，由 srcY/srcZ 计算）。
        [RegisterOpStorageType("addtanhD", typeof(CpuStorage))]
        public Tensor AddTanhD(Tensor result, Tensor srcX, Tensor srcY, Tensor srcZ)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, srcX, false, srcX.Sizes);
            TensorApplyCPU.AddTanhD(writeTarget, srcX, srcY, srcZ);

            return writeTarget;
        }

        // 中文：SiLU/Swish 激活，逐元素 x*sigmoid(x)。
        [RegisterOpStorageType("SiLU", typeof(CpuStorage))]
        public Tensor SiLU(Tensor result, Tensor srcW)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, srcW, false, srcW.Sizes);
            TensorApplyCPU.SiLU(writeTarget, srcW);
            return writeTarget;
        }

        // 中文：GELU 激活，逐元素高斯误差线性单元。
        [RegisterOpStorageType("GELU", typeof(CpuStorage))]
        public Tensor GELU(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            TensorApplyCPU.GELU(writeTarget, src);
            return writeTarget;
        }

        // 中文：GELU 门控乘法，GELU(gate)*up，用于 GLU 类前馈网络。
        [RegisterOpStorageType("GELUMul", typeof(CpuStorage))]
        public Tensor GELUMul(Tensor result, Tensor gate, Tensor up)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, gate, false, gate.Sizes);
            TensorApplyCPU.GELUMul(writeTarget, gate, up);
            return writeTarget;
        }

        // 中文：SiLU 门控乘法，SiLU(gate)*up，用于 SwiGLU 前馈网络。
        [RegisterOpStorageType("SiLUMul", typeof(CpuStorage))]
        public Tensor SiLUMul(Tensor result, Tensor gate, Tensor up)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, gate, false, gate.Sizes);
            TensorApplyCPU.SiLUMul(writeTarget, gate, up);
            return writeTarget;
        }

        // 中文：Sigmoid 门控乘法，x*sigmoid(gate)。
        [RegisterOpStorageType("SigmoidMul", typeof(CpuStorage))]
        public Tensor SigmoidMul(Tensor result, Tensor x, Tensor gate)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, x, false, x.Sizes);
            TensorApplyCPU.SigmoidMul(writeTarget, x, gate);
            return writeTarget;
        }


        // 中文：SiLU 反向梯度并累加（result = srcG + SiLU'梯度，由 srcW/resG 计算）。
        [RegisterOpStorageType("AddSiLUD", typeof(CpuStorage))]
        public Tensor AddSiLUD(Tensor result, Tensor srcG, Tensor srcW, Tensor resG)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, srcW, false, srcW.Sizes);
            TensorApplyCPU.AddSiLUD(writeTarget, srcG, srcW, resG);

            return writeTarget;
        }

        // 中文：SiLU 反向梯度，由前向输入 srcW 与上游梯度 resG 计算梯度。
        [RegisterOpStorageType("SiLUD", typeof(CpuStorage))]
        public Tensor SiLUD(Tensor result, Tensor srcW, Tensor resG)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, srcW, false, srcW.Sizes);
            TensorApplyCPU.SiLUD(writeTarget, srcW, resG);

            return writeTarget;
        }


        // 中文：LeakyReLU 激活，负半轴保留小斜率。
        [RegisterOpStorageType("LeakyReLU", typeof(CpuStorage))]
        public Tensor LeakyReLU(Tensor result, Tensor srcW)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, srcW, false, srcW.Sizes);
            TensorApplyCPU.LeakyReLU(writeTarget, srcW);

            return writeTarget;
        }


        // 中文：LeakyReLU 反向梯度并累加（result = srcG + LeakyReLU'梯度）。
        [RegisterOpStorageType("AddLeakyReLUD", typeof(CpuStorage))]
        public Tensor AddLeakyReLUD(Tensor result, Tensor srcG, Tensor srcW, Tensor resG)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, srcW, false, srcW.Sizes);
            TensorApplyCPU.AddLeakyReLUD(writeTarget, srcG, srcW, resG);

            return writeTarget;
        }

        // 中文：LeakyReLU 反向梯度，由前向输入 srcW 与上游梯度 resG 计算。
        [RegisterOpStorageType("LeakyReLUD", typeof(CpuStorage))]
        public Tensor LeakyReLUD(Tensor result, Tensor srcW, Tensor resG)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, srcW, false, srcW.Sizes);
            TensorApplyCPU.LeakyReLUD(writeTarget, srcW, resG);

            return writeTarget;
        }

        // 中文：ReLU 反向梯度并累加（result = src + ReLU'梯度，由 w/g 计算）。
        [RegisterOpStorageType("addrelud", typeof(CpuStorage))]
        public Tensor AddReluD(Tensor result, Tensor src, Tensor w, Tensor g) 
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, w, false, w.Sizes);
            TensorApplyCPU.AddReluD(writeTarget, src, w, g);

            return writeTarget;
        }

        // 中文：ReLU 反向梯度，由前向输出 w 与上游梯度 g 计算。
        [RegisterOpStorageType("relud", typeof(CpuStorage))]
        public Tensor ReluD(Tensor result, Tensor w, Tensor g)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, w, false, w.Sizes);
            TensorApplyCPU.ReluD(writeTarget, w, g);

            return writeTarget;
        }

        // 中文：张量加标量 lhs + rhs（rhs 为标量）。
        [RegisterOpStorageType("addv", typeof(CpuStorage))]
        public Tensor Add(Tensor result, Tensor lhs, float rhs)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Sizes);
            TensorApplyCPU.Add(writeTarget, lhs, rhs);

            return writeTarget;
        }

        private readonly MethodInfo sub_func = NativeWrapper.GetMethod("TS_Sub");
        // 中文：张量减标量 lhs - rhs（rhs 为标量）。
        [RegisterOpStorageType("subv", typeof(CpuStorage))]
        public Tensor Sub(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(sub_func, result, lhs, rhs); }


        // 中文：标量减张量 lhs - rhs（lhs 为标量），逐元素反向相减。
        [RegisterOpStorageType("rsubv", typeof(CpuStorage))]
        public Tensor Sub(Tensor result, float lhs, Tensor rhs)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, rhs, false, rhs.Sizes);
            TensorApplyCPU.RSub(writeTarget, lhs, rhs);

            return writeTarget;
        }


        // 中文：张量乘标量 lhs * rhs（rhs 为标量）。
        [RegisterOpStorageType("mulv", typeof(CpuStorage))]
        public Tensor Mul(Tensor result, Tensor lhs, float rhs)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Sizes);
            TensorApplyCPU.Mul(writeTarget, lhs, rhs);

            return writeTarget;
        }

       // private readonly MethodInfo div_func = NativeWrapper.GetMethod("TS_Div");
        // 中文：张量除标量 lhs / rhs（rhs 为标量）。
        [RegisterOpStorageType("divv", typeof(CpuStorage))]
        public Tensor Div(Tensor result, Tensor lhs, float rhs)
        {

            //return NativeWrapper.InvokeNullableResultElementwise(div_func, result, lhs, rhs);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Sizes);
            TensorApplyCPU.Div(writeTarget, lhs, rhs);

            return writeTarget;
        }

        private readonly MethodInfo rdiv_func = NativeWrapper.GetMethod("TS_Rdiv");
        // 中文：标量除张量 lhs / rhs（lhs 为标量），逐元素反向相除。
        [RegisterOpStorageType("rdivv", typeof(CpuStorage))]
        public Tensor Div(Tensor result, float lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(rdiv_func, result, rhs, lhs); }

        private readonly MethodInfo mod_func = NativeWrapper.GetMethod("TS_Mod");
        // 中文：张量对标量取模 lhs % rhs（rhs 为标量）。
        [RegisterOpStorageType("modv", typeof(CpuStorage))]
        public Tensor Mod(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(mod_func, result, lhs, rhs); }


        private readonly MethodInfo gtValue_func = NativeWrapper.GetMethod("TS_gtValue");
        // 中文：逐元素与标量比较 lhs > rhs，输出 0/1 掩码。
        [RegisterOpStorageType("gtValue", typeof(CpuStorage))]
        public Tensor GreaterThan(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(gtValue_func, result, lhs, rhs); }

        private readonly MethodInfo ltValue_func = NativeWrapper.GetMethod("TS_gtValue");
        // 中文：逐元素与标量比较 lhs < rhs，输出 0/1 掩码。
        [RegisterOpStorageType("ltValue", typeof(CpuStorage))]
        public Tensor LessThan(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(ltValue_func, result, lhs, rhs); }

        private readonly MethodInfo geValue_func = NativeWrapper.GetMethod("TS_gtValue");
        // 中文：逐元素与标量比较 lhs >= rhs，输出 0/1 掩码。
        [RegisterOpStorageType("geValue", typeof(CpuStorage))]
        public Tensor GreaterOrEqual(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(geValue_func, result, lhs, rhs); }

        private readonly MethodInfo leValue_func = NativeWrapper.GetMethod("TS_gtValue");
        // 中文：逐元素与标量比较 lhs <= rhs，输出 0/1 掩码。
        [RegisterOpStorageType("leValue", typeof(CpuStorage))]
        public Tensor LessOrEqual(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(leValue_func, result, lhs, rhs); }

        private readonly MethodInfo eqValue_func = NativeWrapper.GetMethod("TS_gtValue");
        // 中文：逐元素与标量比较 lhs == rhs，输出 0/1 掩码。
        [RegisterOpStorageType("eqValue", typeof(CpuStorage))]
        public Tensor EqualTo(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(eqValue_func, result, lhs, rhs); }

        private readonly MethodInfo neValue_func = NativeWrapper.GetMethod("TS_gtValue");
        // 中文：逐元素与标量比较 lhs != rhs，输出 0/1 掩码。
        [RegisterOpStorageType("neValue", typeof(CpuStorage))]
        public Tensor NotEqual(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(neValue_func, result, lhs, rhs); }


        // 中文：两张量逐元素相加 lhs + rhs。
        [RegisterOpStorageType("addt", typeof(CpuStorage))]
        public Tensor Add(Tensor result, Tensor lhs, Tensor rhs)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Sizes);
            TensorApplyCPU.Add(writeTarget, lhs, rhs);

            return writeTarget;
        }

        // 中文：原地累加 result += rhs（就地写回 result）。
        [RegisterOpStorageType("atomicadd", typeof(CpuStorage))]
        public Tensor AtomicAdd(Tensor result, Tensor rhs)
        {
            TensorApplyCPU.Add(result, result, rhs);
            return result;
        }

        // 中文：两张量逐元素相减 lhs - rhs。
        [RegisterOpStorageType("subt", typeof(CpuStorage))]
        public Tensor Sub(Tensor result, Tensor lhs, Tensor rhs)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Sizes);
            TensorApplyCPU.Sub(writeTarget, lhs, rhs);
            return writeTarget;
        }

        // 中文：两张量逐元素相乘 lhs * rhs（Hadamard 积）。
        [RegisterOpStorageType("mult", typeof(CpuStorage))]
        public Tensor Mul(Tensor result, Tensor lhs, Tensor rhs)        
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Sizes);
            TensorApplyCPU.Mul(writeTarget, lhs, rhs);
            return writeTarget;
        }

        // 中文：两张量逐元素相除 lhs / rhs。
        [RegisterOpStorageType("divt", typeof(CpuStorage))]
        public Tensor Div(Tensor result, Tensor lhs, Tensor rhs)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Sizes);
            TensorApplyCPU.Div(writeTarget, lhs, rhs);

            return writeTarget;
        }

        private readonly MethodInfo cmod_func = NativeWrapper.GetMethod("TS_CMod");
        // 中文：两张量逐元素取模 lhs % rhs。
        [RegisterOpStorageType("modt", typeof(CpuStorage))]
        public Tensor Mod(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(cmod_func, result, lhs, rhs); }


        private readonly MethodInfo gtTensor_func = NativeWrapper.GetMethod("TS_gtTensor");
        // 中文：两张量逐元素比较 lhs > rhs，输出 0/1 掩码。
        [RegisterOpStorageType("gtTensor", typeof(CpuStorage))]
        public Tensor GreaterThan(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(gtTensor_func, result, lhs, rhs); }

        private readonly MethodInfo ltTensor_func = NativeWrapper.GetMethod("TS_ltTensor");
        // 中文：两张量逐元素比较 lhs < rhs，输出 0/1 掩码（注册名沿用源码）。
        [RegisterOpStorageType("gtTensor", typeof(CpuStorage))]
        public Tensor LessThan(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(ltTensor_func, result, lhs, rhs); }

        private readonly MethodInfo geTensor_func = NativeWrapper.GetMethod("TS_geTensor");
        // 中文：两张量逐元素比较 lhs >= rhs，输出 0/1 掩码。
        [RegisterOpStorageType("geTensor", typeof(CpuStorage))]
        public Tensor GreaterOrEqual(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(geTensor_func, result, lhs, rhs); }

        private readonly MethodInfo leTensor_func = NativeWrapper.GetMethod("TS_leTensor");
        // 中文：两张量逐元素比较 lhs <= rhs，输出 0/1 掩码。
        [RegisterOpStorageType("leTensor", typeof(CpuStorage))]
        public Tensor LessOrEqual(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(leTensor_func, result, lhs, rhs); }

        private readonly MethodInfo eqTensor_func = NativeWrapper.GetMethod("TS_eqTensor");
        // 中文：两张量逐元素比较 lhs == rhs，输出 0/1 掩码。
        [RegisterOpStorageType("eqTensor", typeof(CpuStorage))]
        public Tensor EqualTo(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(eqTensor_func, result, lhs, rhs); }

        private readonly MethodInfo neTensor_func = NativeWrapper.GetMethod("TS_neTensor");
        // 中文：两张量逐元素比较 lhs != rhs，输出 0/1 掩码。
        [RegisterOpStorageType("neTensor", typeof(CpuStorage))]
        public Tensor NotEqual(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(neTensor_func, result, lhs, rhs); }


        // 中文：沿指定维度求和归约。
        [RegisterOpStorageType("sum", typeof(CpuStorage))]
        public Tensor Sum(Tensor result, Tensor src, int dimension)
        {
            Tensor writeTarget = NativeWrapper.CreateResultDimensionwise(result, src, dimension);
            TensorApplyCPU.Sum(writeTarget, src, dimension);

            return writeTarget;
        }

        // 中文：沿指定维度求平均归约。
        [RegisterOpStorageType("mean", typeof(CpuStorage))]
        public Tensor Mean(Tensor result, Tensor src, int dimension)
        {
            Tensor writeTarget = NativeWrapper.CreateResultDimensionwise(result, src, dimension);
            TensorApplyCPU.Mean(writeTarget, src, dimension);

            return writeTarget;
        }


        private readonly MethodInfo prod_func = NativeWrapper.GetMethod("TS_Prod");
        // 中文：沿指定维度求连乘积归约。
        [RegisterOpStorageType("prod", typeof(CpuStorage))]
        public Tensor Prod(Tensor result, Tensor src, int dimension) { return NativeWrapper.InvokeNullableResultDimensionwise(prod_func, result, src, dimension); }

        private readonly MethodInfo min_func = NativeWrapper.GetMethod("TS_Min");
        // 中文：沿指定维度求最小值归约。
        [RegisterOpStorageType("min", typeof(CpuStorage))]
        public Tensor Min(Tensor result, Tensor src, int dimension) { return NativeWrapper.InvokeNullableResultDimensionwise(min_func, result, src, dimension); }


        // 中文：沿指定维度求最大值归约。
        [RegisterOpStorageType("max", typeof(CpuStorage))]
        public Tensor Max(Tensor result, Tensor src, int dimension)
        {
            Tensor writeTarget = NativeWrapper.CreateResultDimensionwise(result, src, dimension);
            TensorApplyCPU.Max(writeTarget, src, dimension);

            return writeTarget;
        }




        private readonly MethodInfo argmin_func = NativeWrapper.GetMethod("TS_Argmin");
        // 中文：沿指定维度求最小值所在索引（argmin）。
        [RegisterOpStorageType("argmin", typeof(CpuStorage))]
        public Tensor Argmin(Tensor result, Tensor src, int dimension) { return NativeWrapper.InvokeNullableResultDimensionwise(argmin_func, result, src, dimension); }


        // 中文：沿指定维度求最大值所在索引（argmax）。
        [RegisterOpStorageType("argmax", typeof(CpuStorage))]
        public Tensor Argmax(Tensor result, Tensor src, int dimension)
        {
            Tensor writeTarget = NativeWrapper.CreateResultDimensionwise(result, src, dimension);
            TensorApplyCPU.Argmax(writeTarget, src, dimension);

            return writeTarget;

        }

        private readonly MethodInfo norm_func = NativeWrapper.GetMethod("TS_Norm");
        // 中文：沿指定维度求 p 范数（p=value）。
        [RegisterOpStorageType("norm", typeof(CpuStorage))]
        public Tensor Norm(Tensor result, Tensor src, int dimension, float value) { return NativeWrapper.InvokeNullableResultDimensionwise(norm_func, result, src, dimension, value); }

        private readonly MethodInfo std_func = NativeWrapper.GetMethod("TS_Std");
        // 中文：沿指定维度求标准差（normByN 控制除以 N 还是 N-1）。
        [RegisterOpStorageType("std", typeof(CpuStorage))]
        public Tensor Std(Tensor result, Tensor src, int dimension, bool normByN) { return NativeWrapper.InvokeNullableResultDimensionwise(std_func, result, src, dimension, normByN); }

        private readonly MethodInfo var_func = NativeWrapper.GetMethod("TS_Var");
        // 中文：沿指定维度求方差（normByN 控制除以 N 还是 N-1）。
        [RegisterOpStorageType("var", typeof(CpuStorage))]
        public Tensor Var(Tensor result, Tensor src, int dimension, bool normByN) { return NativeWrapper.InvokeNullableResultDimensionwise(var_func, result, src, dimension, normByN); }



        private readonly MethodInfo sumall_func = NativeWrapper.GetMethod("TS_SumAll");
        // 中文：对全部元素求和，输出标量。
        [RegisterOpStorageType("sumall", typeof(CpuStorage))]
        public Tensor SumAll(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(sumall_func, writeTarget, src);
            return writeTarget;
        }

        private readonly MethodInfo prodall_func = NativeWrapper.GetMethod("TS_ProdAll");
        // 中文：对全部元素求连乘积，输出标量。
        [RegisterOpStorageType("prodall", typeof(CpuStorage))]
        public Tensor ProdAll(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(prodall_func, writeTarget, src);
            return writeTarget;
        }

        private readonly MethodInfo minall_func = NativeWrapper.GetMethod("TS_MinAll");
        // 中文：对全部元素求最小值，输出标量（注册名沿用源码）。
        [RegisterOpStorageType("prodall", typeof(CpuStorage))]
        public Tensor MinAll(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(minall_func, writeTarget, src);
            return writeTarget;
        }

        private readonly MethodInfo maxall_func = NativeWrapper.GetMethod("TS_MaxAll");
        // 中文：对全部元素求最大值，输出标量。
        [RegisterOpStorageType("maxall", typeof(CpuStorage))]
        public Tensor MaxAll(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(maxall_func, writeTarget, src);
            return writeTarget;
        }


        private readonly MethodInfo meanall_func = NativeWrapper.GetMethod("TS_MeanAll");
        // 中文：对全部元素求平均，输出标量。
        [RegisterOpStorageType("meanall", typeof(CpuStorage))]
        public Tensor MeanAll(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(meanall_func, writeTarget, src);
            return writeTarget;
        }

        private readonly MethodInfo varall_func = NativeWrapper.GetMethod("TS_VarAll");
        // 中文：对全部元素求方差，输出标量。
        [RegisterOpStorageType("varall", typeof(CpuStorage))]
        public Tensor VarAll(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(varall_func, writeTarget, src);
            return writeTarget;
        }

        private readonly MethodInfo stdall_func = NativeWrapper.GetMethod("TS_StdAll");
        // 中文：对全部元素求标准差，输出标量。
        [RegisterOpStorageType("stdall", typeof(CpuStorage))]
        public Tensor StdAll(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(stdall_func, writeTarget, src);
            return writeTarget;
        }



        // 中文：LayerNorm 层归一化前向，对最后一维归一化并施加 gamma/beta 仿射。
        [RegisterOpStorageType("layernorm", typeof(CpuStorage))]
        public Tensor LayerNorm(Tensor result, Tensor src, Tensor gamma_, Tensor beta_, float eps)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            TensorApplyCPU.LayerNorm(writeTarget, src, gamma_, beta_, eps, (int)src.Sizes[0], (int)src.Sizes[1]);
            return writeTarget;
        }

        // 中文：LayerNorm 反向，计算输入梯度及 gamma/beta 的梯度。
        [RegisterOpStorageType("layernormgrad", typeof(CpuStorage))]
        public Tensor LayerNormGrad(Tensor result, Tensor gradGamma_, Tensor gradBeta_, Tensor adj_, Tensor y_, Tensor x_, Tensor gamma_, Tensor beta_, float eps)
        {
            try
            {
                Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, adj_, false, adj_.Sizes);
                TensorApplyCPU.LayerNormGrad(writeTarget, gradGamma_, gradBeta_, adj_, y_, x_, gamma_, beta_, (int)adj_.Sizes[0], (int)adj_.Sizes[1], eps);

                return writeTarget;
            }
            catch (Exception err)
            {
                Logger.WriteLine(Logger.Level.err, ConsoleColor.Red, $"{nameof(LayerNormGrad)} exception: '{err.Message}', CallStack:'{err.StackTrace}'");
                throw;
            }
        }

        // 中文：RMSNorm 前向，按均方根归一化并施加 gamma/beta 仿射。
        [RegisterOpStorageType("rmsnorm", typeof(CpuStorage))]
        public Tensor RMSNorm(Tensor result, Tensor src, Tensor gamma_, Tensor beta_, float eps)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            TensorApplyCPU.RMSNorm(writeTarget, src, gamma_, beta_, eps, (int)src.Sizes[0], (int)src.Sizes[1]);
            return writeTarget;
        }

        // 中文：RMSNorm 反向，计算输入梯度及 gamma/beta 的梯度。
        [RegisterOpStorageType("rmsnormgrad", typeof(CpuStorage))]
        public Tensor RMSNormGrad(Tensor result, Tensor gradGamma_, Tensor gradBeta_, Tensor adj_, Tensor y_, Tensor x_, Tensor gamma_, Tensor beta_, float eps)
        {
            try
            {
                Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, adj_, false, adj_.Sizes);
                TensorApplyCPU.RMSNormGrad(writeTarget, gradGamma_, gradBeta_, adj_, y_, x_, gamma_, beta_, (int)adj_.Sizes[0], (int)adj_.Sizes[1], eps);

                return writeTarget;
            }
            catch (Exception err)
            {
                Logger.WriteLine(Logger.Level.err, ConsoleColor.Red, $"{nameof(RMSNormGrad)} exception: '{err.Message}', CallStack:'{err.StackTrace}'");
                throw;
            }
        }

        private readonly MethodInfo addlayerNorm_func = NativeWrapper.GetMethod("TS_AddLayerNorm");
        // 中文：残差相加后做 LayerNorm，前向计算 LayerNorm(src1+src2)。
        [RegisterOpStorageType("addlayernorm", typeof(CpuStorage))]
        public Tensor AddLayerNorm(Tensor result, Tensor src1, Tensor src2, Tensor gamma_, Tensor beta_, float eps)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src1, false, src1.Sizes);
            NativeWrapper.InvokeTypeMatch(addlayerNorm_func, writeTarget, src1, src2, gamma_, beta_, eps, (int)src1.Sizes[0], (int)src1.Sizes[1]);
            return writeTarget;
        }

        private readonly MethodInfo addlayerNormGrad_func = NativeWrapper.GetMethod("TS_AddLayerNormGrad");
        // 中文：残差 LayerNorm 反向，分别回传到两个残差输入 x1/x2 及 gamma/beta 梯度。
        [RegisterOpStorageType("addlayernormgrad", typeof(CpuStorage))]
        public void AddLayerNormGrad(Tensor result1, Tensor result2, Tensor gradGamma_, Tensor gradBeta_, Tensor adj_, Tensor y_, Tensor x1_, Tensor x2_, Tensor gamma_, Tensor beta_, float eps)
        {
            Tensor writeTarget1 = TensorResultBuilder.GetWriteTarget(result1, adj_, false, adj_.Sizes);
            Tensor writeTarget2 = TensorResultBuilder.GetWriteTarget(result2, adj_, false, adj_.Sizes);
            NativeWrapper.InvokeTypeMatch(addlayerNormGrad_func, writeTarget1, writeTarget2, gradGamma_, gradBeta_, adj_, y_, x1_, x2_, gamma_, beta_, (int)adj_.Sizes[0], (int)adj_.Sizes[1], eps);
        }

        // 中文：按 indice 行索引收集 src 的行（embedding 查表），isAdd 时累加而非覆盖。
        [RegisterOpStorageType("indexselect", typeof(CpuStorage))]
        public Tensor IndexSelect(Tensor result, Tensor src, Tensor indice, bool isAdd)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, new long[] { indice.Sizes[0], src.Sizes[1] });

            int ndim = writeTarget.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(writeTarget.Sizes, writeTarget.Strides);
            long cols = writeTarget.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;

            TensorApplyCPU.IndexSelect(writeTarget, src, indice, (int)rows, (int)cols, isAdd);
            return writeTarget;
        }


        // 中文：IndexSelect 反向，将上游梯度 adj 按 indice 散加回 grad 对应行。
        [RegisterOpStorageType("indexselectgrad", typeof(CpuStorage))]
        public Tensor IndexSelectGrad(Tensor grad, Tensor adj, Tensor indice)
        {
            if (grad == null)
            {
                throw new ArgumentNullException($"Tensor grad should not be null.");
            }

            int ndim = adj.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(adj.Sizes, adj.Strides);
            long cols = adj.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;
            TensorApplyCPU.IndexSelectGrad(grad, adj, indice, (int)rows, (int)cols);
            return grad;
        }

        // 中文：沿指定维度逐元素重复 repeats 次（repeat_interleave）。
        [RegisterOpStorageType("repeat_interleave", typeof(CpuStorage))]
        public Tensor RepeatInterleave(Tensor result, Tensor src, int repeats, int dim)
        {
            if (dim < 0 || dim >= src.DimensionCount)
                throw new ArgumentOutOfRangeException(nameof(dim));
            if (repeats < 1)
                throw new ArgumentOutOfRangeException(nameof(repeats));

            long[] resultSizes = src.Sizes.ToArray();
            resultSizes[dim] *= repeats;

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, resultSizes);

            using var contiguousSrc = Ops.AsContiguous(src);

            int sliceCount = 1;
            for (int d = 0; d < dim; d++)
                sliceCount *= (int)contiguousSrc.Sizes[d];

            int innerSize = 1;
            for (int d = dim + 1; d < contiguousSrc.DimensionCount; d++)
                innerSize *= (int)contiguousSrc.Sizes[d];

            int dimSize = (int)contiguousSrc.Sizes[dim];
            int sliceSize = innerSize;

            unsafe
            {
                float* srcPtr = (float*)CpuNativeHelpers.GetBufferStart(contiguousSrc);
                float* dstPtr = (float*)CpuNativeHelpers.GetBufferStart(writeTarget);

                for (int outer = 0; outer < sliceCount; outer++)
                {
                    float* srcBatch = srcPtr + outer * dimSize * sliceSize;
                    float* dstBatch = dstPtr + outer * dimSize * repeats * sliceSize;
                    TensorApplyCPU.RepeatInterleave(dstBatch, srcBatch, dimSize, repeats, sliceSize);
                }
            }

            return writeTarget;
        }

        // 中文：就地为注意力分数加因果掩码，将未来位置置为 maskedValue。
        [RegisterOpStorageType("add_causal_mask", typeof(CpuStorage))]
        public void AddCausalMask(Tensor tensor, int seqLen, int startPos, float maskedValue)
        {
            if (!tensor.IsContiguous())
                throw new InvalidOperationException("AddCausalMask requires a contiguous tensor.");

            int ndim = tensor.DimensionCount;
            long cols = tensor.Sizes[ndim - 1];
            long totalElements = tensor.ElementCount();
            long totalRows = totalElements / cols;

            unsafe
            {
                float* ptr = (float*)CpuNativeHelpers.GetBufferStart(tensor);
                TensorApplyCPU.AddCausalMask(ptr, (int)totalRows, (int)cols, seqLen, startPos, maskedValue);
            }
        }

        // 中文：沿最后一维取前 k 大的值及其索引（Top-K）。
        [RegisterOpStorageType("topK", typeof(CpuStorage))]
        public void TopK(Tensor outVal, Tensor outIdx, Tensor src, int k)
        {
            int ndim = src.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(src.Sizes, src.Strides);
            long cols = src.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;

            TensorApplyCPU.TopK(outVal, outIdx, src, k, (int)rows, (int)cols);
        }

        // 中文：RoPE 旋转位置编码前向，按行偏移对 Q/K 施加旋转。
        [RegisterOpStorageType("rope", typeof(CpuStorage))]
        public Tensor RoPE(Tensor result, Tensor src, int seqLen, int rowOffset)
        {
            int ndim = src.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(src.Sizes, src.Strides);
            long cols = src.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            TensorApplyCPU.RoPE(writeTarget, src, (int)rows, (int)cols, seqLen, rowOffset);
            return writeTarget;
        }

        // 中文：扩展 RoPE，支持显式位置、频率缩放与 YaRN 外推参数，可选累加到 result。
        [RegisterOpStorageType("rope_ex", typeof(CpuStorage))]
        public Tensor RoPEEx(Tensor result, Tensor src, Tensor positions, int ropeDim, int mode, int originalContextLength, float freqBase, float freqScale, float extFactor, float attnFactor, float betaFast, float betaSlow, bool addToResult, bool invertPositions)
        {
            if (positions == null)
            {
                throw new ArgumentNullException(nameof(positions));
            }

            int ndim = src.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(src.Sizes, src.Strides);
            long cols = src.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;
            if (positions.ElementCount() != rows)
            {
                throw new InvalidOperationException("rope_ex expects one explicit position value per logical row.");
            }

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            TensorApplyCPU.RoPEEx(writeTarget, src, positions, (int)rows, (int)cols, ropeDim, mode, freqBase, freqScale, originalContextLength, extFactor, attnFactor, betaFast, betaSlow);

            if (addToResult && result != null)
            {
                Ops.Add(writeTarget, writeTarget, result);
            }

            return writeTarget;
        }

        // 中文：RoPE 反向，对上游梯度施加逆旋转得到输入梯度。
        [RegisterOpStorageType("ropegrad", typeof(CpuStorage))]
        public Tensor RoPEGrad(Tensor grad_, Tensor adj_, int seqLen, int rowOffset)
        {
            int ndim = adj_.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(adj_.Sizes, adj_.Strides);
            long cols = adj_.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(grad_, adj_, true, adj_.Sizes);
            TensorApplyCPU.RoPEGrad(writeTarget, adj_, (int)rows, (int)cols, seqLen, rowOffset);


            return writeTarget;
        }

        // 中文：检测张量是否含 NaN/Inf 等损坏值，返回布尔结果。
        [RegisterOpStorageType("iscorrupted", typeof(CpuStorage))]
        public bool IsCorrupted(Tensor src)
        {
            int ndim = src.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(src.Sizes, src.Strides);
            long cols = src.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;
            return TensorApplyCPU.IsCorrupted(src, (int)rows, (int)cols);
        }


        // 中文：Softmax 前向，对最后一维做数值稳定的归一化指数。
        [RegisterOpStorageType("softmax", typeof(CpuStorage))]
        public Tensor Softmax(Tensor result, Tensor src)
        {
            int ndim = src.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(src.Sizes, src.Strides);
            long cols = src.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            TensorApplyCPU.Softmax(writeTarget, src, (int)rows, (int)cols);
            return writeTarget;
        }

        // 中文：Softmax 反向，由前向输出 val 与上游梯度 adj 计算输入梯度（addGrad 控制累加）。
        [RegisterOpStorageType("softmaxgrad", typeof(CpuStorage))]
        public Tensor SoftmaxGrad(Tensor grad_, Tensor adj_, Tensor val_, bool addGrad = true)
        {
            int ndim = adj_.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(adj_.Sizes, adj_.Strides);
            long cols = adj_.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(grad_, adj_, true, adj_.Sizes);
            TensorApplyCPU.SoftmaxGrad(writeTarget, adj_, val_, (int)rows, (int)cols, addGrad);


            return writeTarget;
        }


        private readonly MethodInfo rmsProp_func = NativeWrapper.GetMethod("TS_RMSProp");
        // 中文：RMSProp 优化器原地更新权重 tw，使用梯度 tg 与缓存 tc。
        [RegisterOpStorageType("rmsprop", typeof(CpuStorage))]
        public Tensor RMSProp(Tensor tw, Tensor tg, Tensor tc, float gradNormFactor, float step_size, float clipval, float regc, float decay_rate, float eps)
        {
            NativeWrapper.InvokeTypeMatch(rmsProp_func, tw, tg, tc, (int)tw.Sizes[0], (int)tw.Sizes[1], gradNormFactor, step_size, clipval, regc, decay_rate, eps);
            return tw;
        }

        // 中文：Adam 优化器原地更新权重 tw，维护一阶矩 tm 与二阶矩 tv。
        [RegisterOpStorageType("adam", typeof(CpuStorage))]
        public Tensor Adam(Tensor tw, Tensor tg, Tensor tv, Tensor tm, float gradNormFactor, float step_size, float clipval, float regc, float decay_rate_v, float decay_rate_m, int iter, float eps)
        {
            TensorApplyCPU.Adam(tw, tg, tv, tm, (int)tw.Sizes[0], (int)tw.Sizes[1], gradNormFactor, step_size, clipval, regc, decay_rate_v, decay_rate_m, iter, eps);
            return tw;
        }

        private readonly MethodInfo normall_func = NativeWrapper.GetMethod("TS_NormAll");
        // 中文：对全部元素求 p 范数（p=value），输出标量。
        [RegisterOpStorageType("normall", typeof(CpuStorage))]
        public Tensor NormAll(Tensor result, Tensor src, float value)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(normall_func, writeTarget, src, value);
            return writeTarget;
        }
    }
}
