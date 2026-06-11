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

namespace TensorSharp.Expression
{
    public class SVar
    {
        private readonly SExpression expression;


        // 中文：构造函数，用给定的标量表达式包装成一个标量变量。
        public SVar(SExpression expression)
        {
            this.expression = expression;
        }


        // 中文：触发求值，返回该标量变量计算出的 float 结果。
        public float Evaluate()
        {
            return expression.Evaluate();
        }

        public SExpression Expression => expression;


        // 中文：隐式转换，把 float 常量包装为常量标量变量，便于参与表达式构建。
        public static implicit operator SVar(float value) { return new SVar(new ConstScalarExpression(value)); }

        // 中文：一元负号运算符，构建对标量取负的延迟表达式。
        public static SVar operator -(SVar src) { return new SVar(new UnaryScalarExpression(src.expression, val => -val)); }

        // 中文：加法运算符，构建两个标量相加的延迟表达式。
        public static SVar operator +(SVar lhs, SVar rhs) { return new SVar(new BinaryScalarExpression(lhs.expression, rhs.expression, (l, r) => l + r)); }
        // 中文：减法运算符，构建两个标量相减的延迟表达式。
        public static SVar operator -(SVar lhs, SVar rhs) { return new SVar(new BinaryScalarExpression(lhs.expression, rhs.expression, (l, r) => l - r)); }
        // 中文：乘法运算符，构建两个标量相乘的延迟表达式。
        public static SVar operator *(SVar lhs, SVar rhs) { return new SVar(new BinaryScalarExpression(lhs.expression, rhs.expression, (l, r) => l * r)); }
        // 中文：除法运算符，构建两个标量相除的延迟表达式。
        public static SVar operator /(SVar lhs, SVar rhs) { return new SVar(new BinaryScalarExpression(lhs.expression, rhs.expression, (l, r) => l / r)); }
        // 中文：取模运算符，构建两个标量求余的延迟表达式。
        public static SVar operator %(SVar lhs, SVar rhs) { return new SVar(new BinaryScalarExpression(lhs.expression, rhs.expression, (l, r) => l % r)); }


        // 中文：构建求绝对值的延迟标量表达式。
        public SVar Abs() { return new SVar(new UnaryScalarExpression(expression, val => Math.Abs(val))); }
        // 中文：构建求符号（-1/0/1）的延迟标量表达式。
        public SVar Sign() { return new SVar(new UnaryScalarExpression(expression, val => Math.Sign(val))); }

        // 中文：构建求平方根的延迟标量表达式。
        public SVar Sqrt() { return new SVar(new UnaryScalarExpression(expression, val => (float)Math.Sqrt(val))); }
        // 中文：构建求自然指数 e^x 的延迟标量表达式。
        public SVar Exp() { return new SVar(new UnaryScalarExpression(expression, val => (float)Math.Exp(val))); }
        // 中文：构建求自然对数的延迟标量表达式。
        public SVar Log() { return new SVar(new UnaryScalarExpression(expression, val => (float)Math.Log(val))); }
        // 中文：构建向下取整的延迟标量表达式。
        public SVar Floor() { return new SVar(new UnaryScalarExpression(expression, val => (float)Math.Floor(val))); }
        // 中文：构建向上取整的延迟标量表达式。
        public SVar Ceil() { return new SVar(new UnaryScalarExpression(expression, val => (float)Math.Ceiling(val))); }
        // 中文：构建四舍五入取整的延迟标量表达式。
        public SVar Round() { return new SVar(new UnaryScalarExpression(expression, val => (float)Math.Round(val))); }
        // 中文：构建向零截断取整的延迟标量表达式。
        public SVar Trunc() { return new SVar(new UnaryScalarExpression(expression, val => (float)Math.Truncate(val))); }


        // 中文：构建求正弦的延迟标量表达式。
        public SVar Sin() { return new SVar(new UnaryScalarExpression(expression, val => (float)Math.Sin(val))); }
        // 中文：构建求余弦的延迟标量表达式。
        public SVar Cos() { return new SVar(new UnaryScalarExpression(expression, val => (float)Math.Cos(val))); }
        // 中文：构建求正切的延迟标量表达式。
        public SVar Tan() { return new SVar(new UnaryScalarExpression(expression, val => (float)Math.Tan(val))); }

        // 中文：构建求反正弦的延迟标量表达式。
        public SVar Asin() { return new SVar(new UnaryScalarExpression(expression, val => (float)Math.Asin(val))); }
        // 中文：构建求反余弦的延迟标量表达式。
        public SVar Acos() { return new SVar(new UnaryScalarExpression(expression, val => (float)Math.Acos(val))); }
        // 中文：构建求反正切的延迟标量表达式。
        public SVar Atan() { return new SVar(new UnaryScalarExpression(expression, val => (float)Math.Atan(val))); }

        // 中文：构建求双曲正弦的延迟标量表达式。
        public SVar Sinh() { return new SVar(new UnaryScalarExpression(expression, val => (float)Math.Sinh(val))); }
        // 中文：构建求双曲余弦的延迟标量表达式。
        public SVar Cosh() { return new SVar(new UnaryScalarExpression(expression, val => (float)Math.Cosh(val))); }
        // 中文：构建求双曲正切的延迟标量表达式。
        public SVar Tanh() { return new SVar(new UnaryScalarExpression(expression, val => (float)Math.Tanh(val))); }


        // 中文：构建求幂 x^y 的延迟标量表达式。
        public SVar Pow(SVar y) { return new SVar(new BinaryScalarExpression(expression, y.expression, (xVal, yVal) => (float)Math.Pow(xVal, yVal))); }
        // 中文：构建将标量限制在 [min, max] 区间内的延迟标量表达式。
        public SVar Clamp(SVar min, SVar max) { return new SVar(new DelegateScalarExpression(() => ClampFloat(expression.Evaluate(), min.expression.Evaluate(), max.expression.Evaluate()))); }

        // public TVar Pow(TVar y) { return new TVar(new BinaryScalarTensorExpression(this.Expression, y.Expression, Ops.Tpow)); }


        // 中文：构建由 (y, x) 求二参数反正切 atan2 的延迟标量表达式。
        public static SVar Atan2(SVar y, SVar x) { return new SVar(new DelegateScalarExpression(() => (float)Math.Atan2(y.Evaluate(), x.Evaluate()))); }
        // 中文：构建在 a 与 b 之间按 weight 线性插值的延迟标量表达式。
        public static SVar Lerp(SVar a, SVar b, SVar weight) { return new SVar(new DelegateScalarExpression(() => LerpFloat(a.Evaluate(), b.Evaluate(), weight.Evaluate()))); }


        // 中文：线性插值辅助方法，计算 a + weight * (b - a)。
        private static float LerpFloat(float a, float b, float weight)
        {
            return a + weight * (b - a);
        }

        // 中文：钳位辅助方法，将 value 限制在 [min, max] 区间内并返回。
        private static float ClampFloat(float value, float min, float max)
        {
            if (value < min)
            {
                return min;
            }
            else if (value > max)
            {
                return max;
            }
            else
            {
                return value;
            }
        }
    }
}
