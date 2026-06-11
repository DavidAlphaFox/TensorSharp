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
    public abstract class TExpression
    {
        public bool IsValidLvalue { get; private set; }

        // 中文：基类构造函数，标记该张量表达式是否可作为写入目标（左值）。
        public TExpression(bool isValidLvalue = false)
        {
            IsValidLvalue = isValidLvalue;
        }

        // 中文：抽象求值方法，将表达式求值为张量，可选写入 writeTarget。
        public abstract Tensor Evaluate(Tensor writeTarget);
    }

    public class ViewExpression : TExpression
    {
        private readonly TExpression src;
        private readonly Func<Tensor, Tensor> evaluate;

        // 中文：构造函数，保存源表达式与视图变换委托，并继承源的左值性。
        public ViewExpression(TExpression src, Func<Tensor, Tensor> evaluate)
            : base(src.IsValidLvalue)
        {
            this.src = src;
            this.evaluate = evaluate;
        }

        // 中文：求值，求源张量后施加视图变换；视图操作不能直接写入其它张量。
        public override Tensor Evaluate(Tensor writeTarget)
        {
            if (writeTarget != null)
            {
                throw new InvalidOperationException("Cannot Select directly into another tensor");
            }

            using (Tensor s = src.Evaluate(null))
            {
                return evaluate(s);
            }
        }
    }

    public class FromArrayExpression : TExpression
    {
        private readonly IAllocator allocator;
        private readonly Array array;

        // 中文：构造函数，保存分配器与 .NET 数组源数据。
        public FromArrayExpression(IAllocator allocator, Array array)
            : base(false)
        {
            this.allocator = allocator;
            this.array = array;
        }

        // 中文：求值，将数组数据拷贝到 writeTarget 或新建张量并返回。
        public override Tensor Evaluate(Tensor writeTarget)
        {
            if (writeTarget != null)
            {
                writeTarget.CopyFrom(array);
                return writeTarget;
            }
            else
            {
                return Tensor.FromArray(allocator, array);
            }
        }
    }

    public class AsTypeExpression : TExpression
    {
        private readonly TExpression src;
        private readonly DType type;

        // 中文：构造函数，保存源表达式与目标元素数据类型。
        public AsTypeExpression(TExpression src, DType type)
        {
            this.src = src;
            this.type = type;
        }

        // 中文：求值，将源张量按目标数据类型拷贝（类型转换）并返回。
        public override Tensor Evaluate(Tensor writeTarget)
        {
            using (Tensor srcVal = src.Evaluate(null))
            {
                if (writeTarget == null)
                {
                    writeTarget = new Tensor(srcVal.Allocator, type, srcVal.Sizes);
                }

                Ops.Copy(writeTarget, srcVal);
                return writeTarget;
            }
        }
    }

    public class ToDeviceExpression : TExpression
    {
        private readonly TExpression src;
        private readonly IAllocator allocator;

        // 中文：构造函数，保存源表达式与目标设备的分配器。
        public ToDeviceExpression(TExpression src, IAllocator allocator)
        {
            this.src = src;
            this.allocator = allocator;
        }

        // 中文：求值，将源张量拷贝到目标设备（分配器）对应的张量并返回。
        public override Tensor Evaluate(Tensor writeTarget)
        {
            using (Tensor srcVal = src.Evaluate(null))
            {
                if (writeTarget == null)
                {
                    writeTarget = new Tensor(allocator, srcVal.ElementType, srcVal.Sizes);
                }

                Ops.Copy(writeTarget, srcVal);
                return writeTarget;
            }
        }
    }

    public class ScatterFillExpression : TExpression
    {
        private readonly TExpression src;
        private readonly TExpression indices;
        private readonly SVar value;
        private readonly int dimension;


        // 中文：构造函数，保存源张量、填充标量值、维度与索引表达式。
        public ScatterFillExpression(TExpression src, SVar value, int dimension, TExpression indices)
        {
            this.src = src;
            this.value = value;
            this.dimension = dimension;
            this.indices = indices;
        }

        // 中文：求值，复制源张量后在指定维度按索引散布填充标量值并返回。
        public override Tensor Evaluate(Tensor writeTarget)
        {
            using (Tensor s = src.Evaluate(null))
            using (Tensor i = indices.Evaluate(null))
            {
                if (!writeTarget.Equals(s))
                {
                    Ops.Copy(writeTarget, s);
                }
                Ops.ScatterFill(writeTarget, value.Evaluate(), dimension, i);
            }

            return writeTarget;
        }
    }

    public class FillExpression : TExpression
    {
        private readonly IAllocator allocator;
        private readonly DType elementType;
        private readonly long[] sizes;
        private readonly Action<Tensor> fillAction;


        // 中文：构造函数，保存分配器、元素类型、尺寸与填充动作委托。
        public FillExpression(IAllocator allocator, DType elementType, long[] sizes, Action<Tensor> fillAction)
        {
            this.allocator = allocator;
            this.elementType = elementType;
            this.sizes = sizes;
            this.fillAction = fillAction;
        }

        // 中文：求值，按给定尺寸创建张量（如需）并执行填充动作后返回。
        public override Tensor Evaluate(Tensor writeTarget)
        {
            if (writeTarget == null)
            {
                writeTarget = new Tensor(allocator, elementType, sizes);
            }

            fillAction(writeTarget);

            return writeTarget;
        }
    }

    public class AddmmExpression : TExpression
    {
        private readonly TExpression src, m1, m2;
        private readonly float alpha, beta;

        // 中文：构造函数，保存 beta、源张量、alpha 及两个待相乘矩阵表达式。
        public AddmmExpression(float beta, TExpression src, float alpha, TExpression m1, TExpression m2)
        {
            this.beta = beta;
            this.src = src;
            this.alpha = alpha;
            this.m1 = m1;
            this.m2 = m2;
        }

        // 中文：求值，计算 beta*src + alpha*(m1*m2) 矩阵乘加并返回。
        public override Tensor Evaluate(Tensor writeTarget)
        {
            using (Tensor s = src.Evaluate(null))
            using (Tensor m1Val = m1.Evaluate(null))
            using (Tensor m2Val = m2.Evaluate(null))
            {
                return Ops.Addmm(writeTarget, beta, s, alpha, m1Val, m2Val);
            }
        }
    }


    public class TensorValueExpression : TExpression
    {
        private readonly Tensor value;

        // 中文：构造函数，包装一个已有张量值，并标记为合法左值。
        public TensorValueExpression(Tensor value)
            : base(true)
        {
            this.value = value;
        }

        // 中文：求值，无目标时返回张量引用副本，否则拷贝到 writeTarget 并返回。
        public override Tensor Evaluate(Tensor writeTarget)
        {
            if (writeTarget == null)
            {
                return value.CopyRef();
            }
            else
            {
                Ops.Copy(writeTarget, value);
                return writeTarget;
            }
        }
    }

    public class BinaryTensorTensorExpression : TExpression
    {
        private readonly TExpression left, right;
        private readonly Func<Tensor, Tensor, Tensor, Tensor> evaluate;

        // 中文：构造函数，保存左右张量表达式与二元张量运算委托。
        public BinaryTensorTensorExpression(TExpression left, TExpression right, Func<Tensor, Tensor, Tensor, Tensor> evaluate)
        {
            this.left = left;
            this.right = right;
            this.evaluate = evaluate;
        }

        // 中文：求值，分别求左右张量后对其施加二元张量运算并返回。
        public override Tensor Evaluate(Tensor writeTarget)
        {
            using (Tensor lhs = left.Evaluate(null))
            using (Tensor rhs = right.Evaluate(null))
            {
                return evaluate(writeTarget, lhs, rhs);
            }
        }
    }

    public class UnaryTensorExpression : TExpression
    {
        private readonly TExpression src;
        private readonly Func<Tensor, Tensor, Tensor> evaluate;

        // 中文：构造函数，保存源张量表达式与一元张量运算委托。
        public UnaryTensorExpression(TExpression src, Func<Tensor, Tensor, Tensor> evaluate)
        {
            this.src = src;
            this.evaluate = evaluate;
        }

        // 中文：求值，求源张量后对其施加一元张量运算并返回。
        public override Tensor Evaluate(Tensor writeTarget)
        {
            using (Tensor s = src.Evaluate(null))
            {
                return evaluate(writeTarget, s);
            }
        }
    }

    public class BinaryScalarTensorExpression : TExpression
    {
        public readonly SExpression left;
        public readonly TExpression right;
        public readonly Func<Tensor, float, Tensor, Tensor> evaluate;

        // 中文：构造函数，保存左侧标量表达式、右侧张量表达式与运算委托。
        public BinaryScalarTensorExpression(SExpression left, TExpression right, Func<Tensor, float, Tensor, Tensor> evaluate)
        {
            this.left = left;
            this.right = right;
            this.evaluate = evaluate;
        }

        // 中文：求值，以左标量与右张量为操作数施加标量-张量运算并返回。
        public override Tensor Evaluate(Tensor writeTarget)
        {
            using (Tensor rhs = right.Evaluate(null))
            {
                return evaluate(writeTarget, left.Evaluate(), rhs);
            }
        }
    }

    public class BinaryTensorScalarExpression : TExpression
    {
        public readonly TExpression left;
        public readonly SExpression right;
        public readonly Func<Tensor, Tensor, float, Tensor> evaluate;

        // 中文：构造函数，保存左侧张量表达式、右侧标量表达式与运算委托。
        public BinaryTensorScalarExpression(TExpression left, SExpression right, Func<Tensor, Tensor, float, Tensor> evaluate)
        {
            this.left = left;
            this.right = right;
            this.evaluate = evaluate;
        }

        // 中文：求值，以左张量与右标量为操作数施加张量-标量运算并返回。
        public override Tensor Evaluate(Tensor writeTarget)
        {
            using (Tensor lhs = left.Evaluate(null))
            {
                return evaluate(writeTarget, lhs, right.Evaluate());
            }
        }
    }
}
