// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
﻿using System;
using System.Runtime.InteropServices;

namespace TensorSharp.Cpu
{
    public enum CpuDType : int
    {
        Float32 = 0,
        Float16 = 1,
        Float64 = 2,
        Int32 = 3,
        UInt8 = 4,
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct TensorRef64
    {
        public IntPtr buffer;
        public IntPtr sizes;
        public IntPtr strides;
        public int dimCount;
        public CpuDType elementType;
    }


    public static class CpuOpsNative
    {
        private const string dll = "CpuOps.dll";
        private const CallingConvention cc = CallingConvention.Cdecl;

        // 中文：绑定原生 TS_GetLastError，获取最近一次原生调用的错误信息字符串指针。
        [DllImport(dll, CallingConvention = cc)]
        public static extern IntPtr TS_GetLastError();

        // 中文：绑定原生 TS_Copy，将 src 张量逐元素拷贝到 result。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Copy(IntPtr result, IntPtr src);

        // 中文：绑定原生 TS_Abs，逐元素求绝对值。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Abs(IntPtr result, IntPtr src);
        // 中文：绑定原生 TS_Neg，逐元素取负。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Neg(IntPtr result, IntPtr src);
        // 中文：绑定原生 TS_Sign，逐元素取符号。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Sign(IntPtr result, IntPtr src);


        // 中文：绑定原生 TS_Sqrt，逐元素求平方根。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Sqrt(IntPtr result, IntPtr src);

        // 中文：绑定原生 TS_Log1p，逐元素计算 log(1+x)。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Log1p(IntPtr result, IntPtr src);
        // 中文：绑定原生 TS_Floor，逐元素向下取整。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Floor(IntPtr result, IntPtr src);
        // 中文：绑定原生 TS_Ceil，逐元素向上取整。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Ceil(IntPtr result, IntPtr src);
        // 中文：绑定原生 TS_Round，逐元素四舍五入。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Round(IntPtr result, IntPtr src);
        // 中文：绑定原生 TS_Trunc，逐元素截断取整。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Trunc(IntPtr result, IntPtr src);
        // 中文：绑定原生 TS_Frac，逐元素取小数部分。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Frac(IntPtr result, IntPtr src);

        // 中文：绑定原生 TS_Sin，逐元素求正弦。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Sin(IntPtr result, IntPtr src);
        // 中文：绑定原生 TS_Cos，逐元素求余弦。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Cos(IntPtr result, IntPtr src);
        // 中文：绑定原生 TS_Tan，逐元素求正切。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Tan(IntPtr result, IntPtr src);
        // 中文：绑定原生 TS_Asin，逐元素求反正弦。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Asin(IntPtr result, IntPtr src);
        // 中文：绑定原生 TS_Acos，逐元素求反余弦。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Acos(IntPtr result, IntPtr src);
        // 中文：绑定原生 TS_Atan，逐元素求反正切。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Atan(IntPtr result, IntPtr src);
        // 中文：绑定原生 TS_Sinh，逐元素求双曲正弦。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Sinh(IntPtr result, IntPtr src);
        // 中文：绑定原生 TS_Cosh，逐元素求双曲余弦。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Cosh(IntPtr result, IntPtr src);

        // 中文：绑定原生 TS_Add3，逐元素计算三张量之和 x+y+z。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Add3(IntPtr result, IntPtr x, IntPtr y, IntPtr z);
        // 中文：绑定原生 TS_Add4，逐元素计算四张量之和 x+y+z+w。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Add4(IntPtr result, IntPtr x, IntPtr y, IntPtr z, IntPtr w);


        // 中文：绑定原生 TS_MaskFill，按掩码将 t 中对应位置填为默认值。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_MaskFill(IntPtr result, IntPtr t, IntPtr mask, float defValue);


        // 中文：绑定原生 TS_Atan2，逐元素计算 atan2(srcY, srcX)。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Atan2(IntPtr result, IntPtr srcY, IntPtr srcX);
        // 中文：绑定原生 TS_Tpow，逐元素计算 value 的 src 次幂。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Tpow(IntPtr result, float value, IntPtr src);
        // 中文：绑定原生 TS_Lerp，逐元素在 srcA 与 srcB 间按权重线性插值。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Lerp(IntPtr result, IntPtr srcA, IntPtr srcB, float weight);
        //[DllImport(dll, CallingConvention = cc)] public static extern int TS_Clamp(IntPtr result, IntPtr src, float min, float max);

        // 中文：绑定原生 TS_AddTanh3，逐元素计算 tanh(x+y+z)。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_AddTanh3(IntPtr result, IntPtr srcX, IntPtr srcY, IntPtr srcZ);
        // 中文：绑定原生 TS_Add，逐元素将张量加上标量 rhs。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Add(IntPtr result, IntPtr lhs, float rhs);
        // 中文：绑定原生 TS_Sub，逐元素将张量减去标量 rhs。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Sub(IntPtr result, IntPtr lhs, float rhs);
        //[DllImport(dll, CallingConvention = cc)] public static extern int TS_Div(IntPtr result, IntPtr lhs, float rhs);
        // 中文：绑定原生 TS_Rdiv，逐元素计算标量 rhs 除以张量（反向除法）。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Rdiv(IntPtr result, IntPtr lhs, float rhs);
        // 中文：绑定原生 TS_Mod，逐元素对标量 rhs 取模。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Mod(IntPtr result, IntPtr lhs, float rhs);

        // 中文：绑定原生 TS_gtValue，逐元素与标量比较是否大于。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_gtValue(IntPtr result, IntPtr lhs, float rhs);
        // 中文：绑定原生 TS_ltValue，逐元素与标量比较是否小于。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_ltValue(IntPtr result, IntPtr lhs, float rhs);
        // 中文：绑定原生 TS_geValue，逐元素与标量比较是否大于等于。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_geValue(IntPtr result, IntPtr lhs, float rhs);
        // 中文：绑定原生 TS_leValue，逐元素与标量比较是否小于等于。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_leValue(IntPtr result, IntPtr lhs, float rhs);
        // 中文：绑定原生 TS_eqValue，逐元素与标量比较是否相等。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_eqValue(IntPtr result, IntPtr lhs, float rhs);
        // 中文：绑定原生 TS_neValue，逐元素与标量比较是否不等。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_neValue(IntPtr result, IntPtr lhs, float rhs);

        //[DllImport(dll, CallingConvention = cc)] public static extern int TS_CDiv(IntPtr result, IntPtr lhs, IntPtr rhs);
        // 中文：绑定原生 TS_CMod，逐元素对两张量取模。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_CMod(IntPtr result, IntPtr lhs, IntPtr rhs);

        // 中文：绑定原生 TS_gtTensor，逐元素比较两张量是否大于。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_gtTensor(IntPtr result, IntPtr lhs, IntPtr rhs);
        // 中文：绑定原生 TS_ltTensor，逐元素比较两张量是否小于。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_ltTensor(IntPtr result, IntPtr lhs, IntPtr rhs);
        // 中文：绑定原生 TS_geTensor，逐元素比较两张量是否大于等于。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_geTensor(IntPtr result, IntPtr lhs, IntPtr rhs);
        // 中文：绑定原生 TS_leTensor，逐元素比较两张量是否小于等于。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_leTensor(IntPtr result, IntPtr lhs, IntPtr rhs);
        // 中文：绑定原生 TS_eqTensor，逐元素比较两张量是否相等。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_eqTensor(IntPtr result, IntPtr lhs, IntPtr rhs);
        // 中文：绑定原生 TS_neTensor，逐元素比较两张量是否不等。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_neTensor(IntPtr result, IntPtr lhs, IntPtr rhs);


        // 中文：绑定原生 TS_Sum，沿指定维度求和。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Sum(IntPtr result, IntPtr src, int dimension);
        // 中文：绑定原生 TS_Prod，沿指定维度求积。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Prod(IntPtr result, IntPtr src, int dimension);
        // 中文：绑定原生 TS_Min，沿指定维度求最小值。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Min(IntPtr result, IntPtr src, int dimension);

        // 中文：绑定原生 TS_Argmin，沿指定维度求最小值索引。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Argmin(IntPtr result, IntPtr src, int dimension);

        // 中文：绑定原生 TS_Norm，沿指定维度求 p 范数。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Norm(IntPtr result, IntPtr src, int dimension, float value);
        // 中文：绑定原生 TS_Std，沿指定维度求标准差。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Std(IntPtr result, IntPtr src, int dimension, bool normByN);
        // 中文：绑定原生 TS_Var，沿指定维度求方差。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Var(IntPtr result, IntPtr src, int dimension, bool normByN);

        // 中文：绑定原生 TS_SumAll，对全部元素求和。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_SumAll(IntPtr result, IntPtr src);
        // 中文：绑定原生 TS_ProdAll，对全部元素求积。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_ProdAll(IntPtr result, IntPtr src);
        // 中文：绑定原生 TS_MinAll，对全部元素求最小值。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_MinAll(IntPtr result, IntPtr src);
        // 中文：绑定原生 TS_MaxAll，对全部元素求最大值。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_MaxAll(IntPtr result, IntPtr src);

        // 中文：绑定原生 TS_MeanAll，对全部元素求均值。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_MeanAll(IntPtr result, IntPtr src);
        // 中文：绑定原生 TS_VarAll，对全部元素求方差。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_VarAll(IntPtr result, IntPtr src);
        // 中文：绑定原生 TS_StdAll，对全部元素求标准差。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_StdAll(IntPtr result, IntPtr src);
        // 中文：绑定原生 TS_NormAll，对全部元素求 p 范数。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_NormAll(IntPtr result, IntPtr src, float value);


        // 中文：绑定原生 TS_NewRNG，创建一个原生随机数生成器实例。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_NewRNG(out IntPtr rng);
        // 中文：绑定原生 TS_DeleteRNG，销毁随机数生成器实例。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_DeleteRNG(IntPtr rng);
        // 中文：绑定原生 TS_SetRNGSeed，为随机数生成器设置种子。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_SetRNGSeed(IntPtr rng, int newSeed);

        // 中文：绑定原生 TS_RandomUniform，用均匀分布随机数填充张量。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_RandomUniform(IntPtr rng, IntPtr result, float min, float max);
        // 中文：绑定原生 TS_RandomNormal，用正态分布随机数填充张量。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_RandomNormal(IntPtr rng, IntPtr result, float mean, float stdv);
        // 中文：绑定原生 TS_RandomExponential，用指数分布随机数填充张量。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_RandomExponential(IntPtr rng, IntPtr result, float lambda);
        // 中文：绑定原生 TS_RandomCauchy，用柯西分布随机数填充张量。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_RandomCauchy(IntPtr rng, IntPtr result, float median, float sigma);
        // 中文：绑定原生 TS_RandomLogNormal，用对数正态分布随机数填充张量。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_RandomLogNormal(IntPtr rng, IntPtr result, float mean, float stdv);
        // 中文：绑定原生 TS_RandomGeometric，用几何分布随机数填充张量。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_RandomGeometric(IntPtr rng, IntPtr result, float p);
        // 中文：绑定原生 TS_RandomBernoulli，用伯努利分布随机数填充张量。
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_RandomBernoulli(IntPtr rng, IntPtr result, float p);


        // 中文：绑定原生 TS_Unfolded_Acc，卷积反向 im2col 的累加（将展开列累加回输入）。
        [DllImport(dll, CallingConvention = cc)]
        public static extern int TS_Unfolded_Acc(IntPtr finput, IntPtr input, int kW, int kH, int dW, int dH, int padW, int padH, int nInputPlane, int inputWidth, int inputHeight, int outputWidth, int outputHeight);
        // 中文：绑定原生 TS_Unfolded_Copy，卷积前向 im2col 展开（将输入图像展开为列矩阵）。
        [DllImport(dll, CallingConvention = cc)]
        public static extern int TS_Unfolded_Copy(IntPtr finput, IntPtr input, int kW, int kH, int dW, int dH, int padW, int padH, int nInputPlane, int inputWidth, int inputHeight, int outputWidth, int outputHeight);


        // 中文：绑定原生 TS_AddLayerNorm，对两输入之和执行层归一化（含 gamma/beta 缩放偏置）前向。
        [DllImport(dll, CallingConvention = cc)]
        public static extern int TS_AddLayerNorm(IntPtr out_, IntPtr in1_, IntPtr in2_, IntPtr gamma_, IntPtr beta_, float eps, int rows, int cols);

        // 中文：绑定原生 TS_AddLayerNormGrad，AddLayerNorm 的反向，计算输入与 gamma/beta 的梯度。
        [DllImport(dll, CallingConvention = cc)]
        public static extern int TS_AddLayerNormGrad(IntPtr result1, IntPtr result2, IntPtr gradGamma_, IntPtr gradBeta_, IntPtr adj_, IntPtr y_, IntPtr x1_, IntPtr x2_, IntPtr gamma_, IntPtr beta_, int rows, int cols, float eps);


        // 中文：绑定原生 TS_RMSProp，执行 RMSProp 优化器的权重更新。
        [DllImport(dll, CallingConvention = cc)]
        public static extern int TS_RMSProp(IntPtr tw, IntPtr tg, IntPtr tc, int rows, int cols, int batchSize, float step_size, float clipval, float regc, float decay_rate, float eps);

        // 中文：绑定原生 TS_SpatialMaxPooling_updateOutput_frame，单帧空间最大池化前向，同时输出最大值索引。
        [DllImport(dll, CallingConvention = cc)]
        public static extern int TS_SpatialMaxPooling_updateOutput_frame(IntPtr input_p, IntPtr output_p, IntPtr ind_p, long nslices, long iwidth, long iheight, long owidth, long oheight, int kW, int kH, int dW, int dH, int padW, int padH);

        // 中文：绑定原生 TS_SpatialMaxPooling_updateGradInput_frame，单帧空间最大池化反向，按索引回传梯度。
        [DllImport(dll, CallingConvention = cc)]
        public static extern int TS_SpatialMaxPooling_updateGradInput_frame(IntPtr gradInput, IntPtr gradOutput, IntPtr ind, long nslices, long iwidth, long iheight, long owidth, long oheight, int dW, int dH);

      //  [DllImport(dll, CallingConvention = cc)] public static extern int TS_ScatterFill(IntPtr result, float value, int dim, IntPtr indices);
    }
}
