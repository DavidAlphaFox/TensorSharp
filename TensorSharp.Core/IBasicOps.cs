// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
﻿// ───────────────────────────────────────────────────────────────────────────
// 【文件说明】后端张量算子接口的历史定义（当前以注释形式保留，作为算子清单参考）。
// 【说明】实际算子的注册与分发由 OpRegistry + 特性注解完成，各后端按约束匹配实现；
//         未实现的算子统一回退到 CPU，从而保证各后端结果一致。
// ───────────────────────────────────────────────────────────────────────────
namespace TensorSharp
{
    /*
    public interface IBasicOps
    {
        // 中文：分配并返回一个与 src 数据相同但内存连续的新张量
        Tensor NewContiguous(Tensor src);

        // 中文：若 src 已连续则直接返回，否则返回连续副本
        Tensor AsContiguous(Tensor src);

        // 中文：计算两个一维张量的点积，返回标量张量
        Tensor Dot(Tensor result, Tensor lhs, Tensor rhs);

        // result = (alpha * m1 * m2) + (beta * src)
        // 中文：矩阵乘加融合算子，计算 result = alpha * m1 × m2 + beta * src（GEMM 形式）
        Tensor Addmm(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2);

        //
        // Element-wise operators
        //

        // 中文：将 src 张量的全部元素逐一复制到 result 张量
        void Copy(Tensor result, Tensor src);
        // 中文：将 result 张量的所有元素填充为常量 value
        void Fill(Tensor result, float value);

        // 中文：沿指定维度拼接多个张量，输出到 result
        Tensor Concat(Tensor result, int dimension, params Tensor[] inputs);

        // 中文：逐元素计算绝对值 |src|
        Tensor Abs(Tensor result, Tensor src);
        // 中文：逐元素取反 -src
        Tensor Neg(Tensor result, Tensor src);
        // 中文：逐元素符号函数，返回 -1、0 或 1
        Tensor Sign(Tensor result, Tensor src);

        // 中文：逐元素平方根 sqrt(src)
        Tensor Sqrt(Tensor result, Tensor src);
        // 中文：逐元素自然指数 e^src
        Tensor Exp(Tensor result, Tensor src);
        // 中文：逐元素自然对数 ln(src)
        Tensor Log(Tensor result, Tensor src);
        // 中文：逐元素计算 ln(1 + src)，对接近零的值更精确
        Tensor Log1p(Tensor result, Tensor src);
        // 中文：逐元素向下取整 floor(src)
        Tensor Floor(Tensor result, Tensor src);
        // 中文：逐元素向上取整 ceil(src)
        Tensor Ceil(Tensor result, Tensor src);
        // 中文：逐元素四舍五入 round(src)
        Tensor Round(Tensor result, Tensor src);
        // 中文：逐元素截断小数部分（向零取整）trunc(src)
        Tensor Trunc(Tensor result, Tensor src);
        // 中文：逐元素取小数部分 src - trunc(src)
        Tensor Frac(Tensor result, Tensor src);

        // 中文：逐元素正弦函数 sin(src)
        Tensor Sin(Tensor result, Tensor src);
        // 中文：逐元素余弦函数 cos(src)
        Tensor Cos(Tensor result, Tensor src);
        // 中文：逐元素正切函数 tan(src)
        Tensor Tan(Tensor result, Tensor src);
        // 中文：逐元素反正弦函数 arcsin(src)
        Tensor Asin(Tensor result, Tensor src);
        // 中文：逐元素反余弦函数 arccos(src)
        Tensor Acos(Tensor result, Tensor src);
        // 中文：逐元素反正切函数 arctan(src)
        Tensor Atan(Tensor result, Tensor src);
        // 中文：逐元素双曲正弦函数 sinh(src)
        Tensor Sinh(Tensor result, Tensor src);
        // 中文：逐元素双曲余弦函数 cosh(src)
        Tensor Cosh(Tensor result, Tensor src);
        // 中文：逐元素双曲正切函数 tanh(src)，常用于激活函数
        Tensor Tanh(Tensor result, Tensor src);

        // 中文：逐元素 Sigmoid 激活函数 1/(1+e^{-src})
        Tensor Sigmoid(Tensor result, Tensor src);

        // 中文：逐元素四象限反正切 atan2(srcY, srcX)
        Tensor Atan2(Tensor result, Tensor srcY, Tensor srcX);
        // 中文：逐元素幂运算 src^value
        Tensor Pow(Tensor result, Tensor src, float value);
        // 中文：逐元素反向幂运算 value^src
        Tensor Tpow(Tensor result, float value, Tensor src);
        // 中文：逐元素线性插值 srcA + weight * (srcB - srcA)
        Tensor Lerp(Tensor result, Tensor srcA, Tensor srcB, float weight);
        // 中文：逐元素值域裁剪，将 src 中每个元素限制在 [min, max] 范围内
        Tensor Clamp(Tensor result, Tensor src, float min, float max);

        // 中文：张量与标量相加 rhs + lhs（标量版）
        Tensor Add(Tensor result, Tensor rhs, float lhs);
        // 中文：张量减标量 rhs - lhs（标量版）
        Tensor Sub(Tensor result, Tensor rhs, float lhs);
        // 中文：标量减张量 rhs - lhs（标量在左）
        Tensor Sub(Tensor result, float rhs, Tensor lhs);
        // 中文：张量与标量逐元素相乘 rhs * lhs（标量版）
        Tensor Mul(Tensor result, Tensor rhs, float lhs);
        // 中文：张量除以标量 rhs / lhs（标量版）
        Tensor Div(Tensor result, Tensor rhs, float lhs);
        // 中文：标量除以张量 rhs / lhs（标量在左）
        Tensor Div(Tensor result, float rhs, Tensor lhs);
        // 中文：张量对标量取模 rhs % lhs（标量版）
        Tensor Mod(Tensor result, Tensor rhs, float lhs);

        // 中文：逐元素比较 rhs > lhs，结果为 0 或 1（标量版）
        Tensor GreaterThan(Tensor result, Tensor rhs, float lhs);
        // 中文：逐元素比较 rhs < lhs，结果为 0 或 1（标量版）
        Tensor LessThan(Tensor result, Tensor rhs, float lhs);
        // 中文：逐元素比较 rhs >= lhs，结果为 0 或 1（标量版）
        Tensor GreaterOrEqual(Tensor result, Tensor rhs, float lhs);
        // 中文：逐元素比较 rhs <= lhs，结果为 0 或 1（标量版）
        Tensor LessOrEqual(Tensor result, Tensor rhs, float lhs);
        // 中文：逐元素判等 rhs == lhs，结果为 0 或 1（标量版）
        Tensor EqualTo(Tensor result, Tensor rhs, float lhs);
        // 中文：逐元素判不等 rhs != lhs，结果为 0 或 1（标量版）
        Tensor NotEqual(Tensor result, Tensor rhs, float lhs);


        // 中文：逐元素张量相加 rhs + lhs（张量版）
        Tensor Add(Tensor result, Tensor rhs, Tensor lhs);
        // 中文：逐元素张量相减 rhs - lhs（张量版）
        Tensor Sub(Tensor result, Tensor rhs, Tensor lhs);
        // 中文：逐元素张量相乘 rhs * lhs（Hadamard 积，张量版）
        Tensor Mul(Tensor result, Tensor rhs, Tensor lhs);
        // 中文：逐元素张量相除 rhs / lhs（张量版）
        Tensor Div(Tensor result, Tensor rhs, Tensor lhs);
        // 中文：逐元素张量取模 rhs % lhs（张量版）
        Tensor Mod(Tensor result, Tensor rhs, Tensor lhs);

        // 中文：逐元素比较 rhs > lhs，结果为 0 或 1（张量版）
        Tensor GreaterThan(Tensor result, Tensor rhs, Tensor lhs);
        // 中文：逐元素比较 rhs < lhs，结果为 0 或 1（张量版）
        Tensor LessThan(Tensor result, Tensor rhs, Tensor lhs);
        // 中文：逐元素比较 rhs >= lhs，结果为 0 或 1（张量版）
        Tensor GreaterOrEqual(Tensor result, Tensor rhs, Tensor lhs);
        // 中文：逐元素比较 rhs <= lhs，结果为 0 或 1（张量版）
        Tensor LessOrEqual(Tensor result, Tensor rhs, Tensor lhs);
        // 中文：逐元素判等 rhs == lhs，结果为 0 或 1（张量版）
        Tensor EqualTo(Tensor result, Tensor rhs, Tensor lhs);
        // 中文：逐元素判不等 rhs != lhs，结果为 0 或 1（张量版）
        Tensor NotEqual(Tensor result, Tensor rhs, Tensor lhs);


        //
        // Dimension-wise operators
        //

        // 中文：沿指定维度对张量求和，保留该维度（归约）
        Tensor Sum(Tensor result, Tensor src, int dimension);
        // 中文：沿指定维度对张量求连乘积（归约）
        Tensor Prod(Tensor result, Tensor src, int dimension);
        // 中文：沿指定维度求最小值（归约）
        Tensor Min(Tensor result, Tensor src, int dimension);
        // 中文：沿指定维度求最大值（归约）
        Tensor Max(Tensor result, Tensor src, int dimension);

        // 中文：沿指定维度求最大值的索引位置（argmax 归约）
        Tensor Argmax(Tensor result, Tensor src, int dimension);


        // 中文：沿指定维度求均值（归约）
        Tensor Mean(Tensor result, Tensor src, int dimension);
        // 中文：沿指定维度求 L-value 范数（归约）
        Tensor Norm(Tensor result, Tensor src, int dimension, float value);
        // 中文：沿指定维度求标准差，normByN 为 true 时除以 N，否则除以 N-1
        Tensor Std(Tensor result, Tensor src, int dimension, bool normByN);
        // 中文：沿指定维度求方差，normByN 为 true 时除以 N，否则除以 N-1
        Tensor Var(Tensor result, Tensor src, int dimension, bool normByN);

        //
        // Full-tensor operators
        //

        // 中文：对整个张量所有元素求和，输出到 result 张量
        Tensor SumAll(Tensor result, Tensor src);
        // 中文：对整个张量所有元素求连乘积，输出到 result 张量
        Tensor ProdAll(Tensor result, Tensor src);
        // 中文：对整个张量所有元素求最小值，输出到 result 张量
        Tensor MinAll(Tensor result, Tensor src);
        // 中文：对整个张量所有元素求最大值，输出到 result 张量
        Tensor MaxAll(Tensor result, Tensor src);

        // 中文：对整个张量所有元素求均值，输出到 result 张量
        Tensor MeanAll(Tensor result, Tensor src);
        // 中文：对整个张量所有元素求方差，输出到 result 张量
        Tensor VarAll(Tensor result, Tensor src);
        // 中文：对整个张量所有元素求标准差，输出到 result 张量
        Tensor StdAll(Tensor result, Tensor src);
        // 中文：对整个张量所有元素求 L-value 范数，输出到 result 张量
        Tensor NormAll(Tensor result, Tensor src, float value);

        // 中文：对整个张量所有元素求和，直接返回 float 标量
        float SumAll(Tensor src);
        // 中文：对整个张量所有元素求连乘积，直接返回 float 标量
        float ProdAll(Tensor src);
        // 中文：对整个张量所有元素求最小值，直接返回 float 标量
        float MinAll(Tensor src);
        // 中文：对整个张量所有元素求最大值，直接返回 float 标量
        float MaxAll(Tensor src);

        // 中文：对整个张量所有元素求均值，直接返回 float 标量
        float MeanAll(Tensor src);
        // 中文：对整个张量所有元素求方差，直接返回 float 标量
        float VarAll(Tensor src);
        // 中文：对整个张量所有元素求标准差，直接返回 float 标量
        float StdAll(Tensor src);
        // 中文：对整个张量所有元素求 L-value 范数，直接返回 float 标量
        float NormAll(Tensor src, float value);
    }*/
}
