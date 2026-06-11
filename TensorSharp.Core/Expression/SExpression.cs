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
    public abstract class SExpression
    {
        // 中文：抽象求值方法，由各派生标量表达式实现以返回 float 结果。
        public abstract float Evaluate();
    }


    public class ConstScalarExpression : SExpression
    {
        private readonly float value;

        // 中文：构造函数，保存一个常量浮点值。
        public ConstScalarExpression(float value)
        {
            this.value = value;
        }

        // 中文：求值，直接返回保存的常量值。
        public override float Evaluate()
        {
            return value;
        }
    }

    public class DelegateScalarExpression : SExpression
    {
        private readonly Func<float> evaluate;

        // 中文：构造函数，保存一个用于求值的委托。
        public DelegateScalarExpression(Func<float> evaluate)
        {
            this.evaluate = evaluate;
        }

        // 中文：求值，调用保存的委托并返回其结果。
        public override float Evaluate()
        {
            return evaluate();
        }
    }

    public class UnaryScalarExpression : SExpression
    {
        private readonly SExpression src;
        private readonly Func<float, float> evaluate;


        // 中文：构造函数，保存源表达式与一元运算委托。
        public UnaryScalarExpression(SExpression src, Func<float, float> evaluate)
        {
            this.src = src;
            this.evaluate = evaluate;
        }

        // 中文：求值，先求源表达式再套用一元运算委托返回结果。
        public override float Evaluate()
        {
            return evaluate(src.Evaluate());
        }
    }

    public class BinaryScalarExpression : SExpression
    {
        private readonly SExpression left;
        private readonly SExpression right;
        private readonly Func<float, float, float> evaluate;


        // 中文：构造函数，保存左右子表达式与二元运算委托。
        public BinaryScalarExpression(SExpression left, SExpression right, Func<float, float, float> evaluate)
        {
            this.left = left;
            this.right = right;
            this.evaluate = evaluate;
        }

        // 中文：求值，分别求左右子表达式再套用二元运算委托返回结果。
        public override float Evaluate()
        {
            return evaluate(left.Evaluate(), right.Evaluate());
        }
    }
}
