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

namespace TensorSharp
{
    public abstract class OpConstraint
    {
        // 中文：抽象方法，判断给定实参是否满足该约束。
        public abstract bool SatisfiedFor(object[] args);
    }

    public class ArgCountConstraint : OpConstraint
    {
        private readonly int argCount;

        // 中文：构造函数，记录期望的参数个数。
        public ArgCountConstraint(int argCount) { this.argCount = argCount; }

        // 中文：判断实参个数是否等于期望值。
        public override bool SatisfiedFor(object[] args)
        {
            return args.Length == argCount;
        }
    }

    public class ArgTypeConstraint : OpConstraint
    {
        private readonly int argIndex;
        private readonly Type requiredType;

        // 中文：构造函数，记录受约束的参数下标及要求的类型。
        public ArgTypeConstraint(int argIndex, Type requiredType)
        {
            this.argIndex = argIndex;
            this.requiredType = requiredType;
        }

        // 中文：判断指定下标的实参类型是否可赋给要求的类型。
        public override bool SatisfiedFor(object[] args)
        {
            return requiredType.IsAssignableFrom(args[argIndex].GetType());
        }
    }

    public class ArgStorageTypeConstraint : OpConstraint
    {
        private readonly int argIndex;
        private readonly Type requiredType;
        private readonly bool allowNull;

        // 中文：构造函数，记录参数下标、要求的 Storage 类型及是否允许为 null。
        public ArgStorageTypeConstraint(int argIndex, Type requiredType, bool allowNull = true)
        {
            this.argIndex = argIndex;
            this.requiredType = requiredType;
            this.allowNull = allowNull;
        }

        // 中文：判断指定 Tensor 参数的 Storage 类型是否匹配（按 allowNull 处理空值）。
        public override bool SatisfiedFor(object[] args)
        {
            if (allowNull && args[argIndex] == null)
            {
                return true;
            }
            else if (!allowNull && args[argIndex] == null)
            {
                return false;
            }

            Storage argStorage = ((Tensor)args[argIndex]).Storage;
            return argStorage.GetType() == requiredType;
        }
    }
}
