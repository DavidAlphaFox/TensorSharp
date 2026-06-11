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
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.ExceptionServices;

namespace TensorSharp
{
    [AttributeUsage(AttributeTargets.Class)]
    public class OpsClassAttribute : Attribute
    {
    }

    [AttributeUsage(AttributeTargets.Method)]
    public abstract class RegisterOp : Attribute
    {
        public string OpName { get; private set; }

        // 中文：构造函数，记录待注册算子的操作名。
        public RegisterOp(string opName)
        {
            OpName = opName;
        }

        // 中文：以反射调用已注册方法，并将内部异常原样向外抛出（保留原始堆栈）。
        protected static object InvokeRegisteredMethod(object instance, MethodInfo method, object[] args)
        {
            try
            {
                return method.Invoke(instance, args);
            }
            catch (TargetInvocationException ex) when (ex.InnerException != null)
            {
                ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
                throw;
            }
        }

        // 中文：抽象方法，由子类实现将指定方法连同约束注册到 OpRegistry。
        public abstract void DoRegister(object instance, MethodInfo method, IEnumerable<OpConstraint> paramConstraints);
    }

    /// <summary>
    /// Register a method where the only constraint is that the argument counts match.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method)]
    public class RegisterOpArgCount : RegisterOp
    {
        // 中文：构造函数，按操作名创建仅校验参数个数的注册特性。
        public RegisterOpArgCount(string opName) : base(opName)
        {
        }

        // 中文：以“参数约束 + 参数个数约束”将方法注册到 OpRegistry。
        public override void DoRegister(object instance, MethodInfo method, IEnumerable<OpConstraint> paramConstraints)
        {
            List<OpConstraint> constraints = new List<OpConstraint>();
            constraints.AddRange(paramConstraints);
            constraints.Add(new ArgCountConstraint(method.GetParameters().Length));

            OpRegistry.Register(OpName, args => InvokeRegisteredMethod(instance, method, args), constraints);
        }
    }


    // 中文：注册算子特性，附加“参数个数 + Tensor 参数须为指定 Storage 类型”的约束。
    [AttributeUsage(AttributeTargets.Method)]
    public class RegisterOpStorageType : RegisterOp
    {
        private readonly Type storageType;

        // 中文：构造函数，记录操作名及 Tensor 参数所要求的 Storage 类型。
        public RegisterOpStorageType(string opName, Type storageType) : base(opName)
        {
            this.storageType = storageType;
        }

        // 中文：附加参数个数约束，并为每个 Tensor 参数加上 Storage 类型约束后注册方法。
        public override void DoRegister(object instance, MethodInfo method, IEnumerable<OpConstraint> paramConstraints)
        {
            List<OpConstraint> constraints = new List<OpConstraint>();
            constraints.AddRange(paramConstraints);
            constraints.Add(new ArgCountConstraint(method.GetParameters().Length));

            ParameterInfo[] methodParams = method.GetParameters();
            for (int i = 0; i < methodParams.Length; ++i)
            {
                if (methodParams[i].ParameterType == typeof(Tensor))
                {
                    constraints.Add(new ArgStorageTypeConstraint(i, storageType));
                }
            }

            OpRegistry.Register(OpName, args => InvokeRegisteredMethod(instance, method, args), constraints);
        }
    }




    [AttributeUsage(AttributeTargets.Parameter)]
    public abstract class ArgConstraintAttribute : Attribute
    {
        // 中文：默认构造函数。
        public ArgConstraintAttribute()
        {
        }

        // 中文：抽象方法，由子类返回该参数对应的约束集合。
        public abstract IEnumerable<OpConstraint> GetConstraints(ParameterInfo parameter, object instance);
    }

    [AttributeUsage(AttributeTargets.Parameter)]
    public class OpArgStorageType : ArgConstraintAttribute
    {
        private readonly Type storageType;

        // 中文：构造函数，记录该参数所要求的 Storage 类型。
        public OpArgStorageType(Type storageType)
        {
            this.storageType = storageType;
        }

        // 中文：生成针对该参数位置的 Storage 类型约束。
        public override IEnumerable<OpConstraint> GetConstraints(ParameterInfo parameter, object instance)
        {
            yield return new ArgStorageTypeConstraint(parameter.Position, storageType);
        }
    }
}
