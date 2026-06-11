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
using System.Linq;
using System.Reflection;

// ───────────────────────────────────────────────────────────────────────────
// 【文件说明】算子注册与分发中心——后端无关计算的关键枢纽。
// 【主要类型】OpRegistry：通过反射扫描各后端用特性标注的算子，按操作名与约束
//             （设备 / 数据类型等）匹配并调用；某后端缺失的算子可回退到 CPU。
// ───────────────────────────────────────────────────────────────────────────
namespace TensorSharp
{
    public delegate object OpHandler(object[] args);

    public static class OpRegistry
    {
        private class OpInstance
        {
            public OpHandler handler;
            public IEnumerable<OpConstraint> constraints;
        }

        private static readonly Dictionary<string, List<OpInstance>> opInstances = new Dictionary<string, List<OpInstance>>();
        // Remember which assemblies have been registered to avoid accidental double-registering
        private static readonly HashSet<Assembly> registeredAssemblies = new HashSet<Assembly>();

        // 中文：静态构造函数，初始化时自动注册本程序集中的 CPU 算子。
        static OpRegistry()
        {
            // Register CPU ops from this assembly
            RegisterAssembly(Assembly.GetExecutingAssembly());
        }

        // 中文：将一个算子处理器及其约束按操作名登记到注册表中。
        public static void Register(string opName, OpHandler handler, IEnumerable<OpConstraint> constraints)
        {
            OpInstance newInstance = new OpInstance() { handler = handler, constraints = constraints };

            if (opInstances.TryGetValue(opName, out List<OpInstance> instanceList))
            {
                instanceList.Add(newInstance);
            }
            else
            {
                instanceList = new List<OpInstance>
                {
                    newInstance
                };
                opInstances.Add(opName, instanceList);
            }
        }

        // 中文：按操作名查找首个满足全部约束的处理器并调用，无匹配则抛异常。
        public static object Invoke(string opName, params object[] args)
        {
            if (opInstances.TryGetValue(opName, out List<OpInstance> instanceList))
            {
                foreach (OpInstance instance in instanceList)
                {
                    if (instance.constraints.All(x => x.SatisfiedFor(args)))
                    {
                        return instance.handler.Invoke(args);
                    }
                }

                throw new ApplicationException("None of the registered handlers match the arguments for " + opName);
            }
            else
            {
                throw new ApplicationException("No handlers have been registered for op " + opName);
            }
        }

        // 中文：反射扫描程序集中带 OpsClass 特性的类型，注册其 RegisterOp 标注的算子方法（去重防止重复注册）。
        public static void RegisterAssembly(Assembly assembly)
        {
            if (!registeredAssemblies.Contains(assembly))
            {
                registeredAssemblies.Add(assembly);

                IEnumerable<Type> types = assembly.TypesWithAttribute<OpsClassAttribute>(false)
                    .Select(x => x.Item1);

                foreach (Type type in types)
                {
                    object instance = Activator.CreateInstance(type);

                    IEnumerable<Tuple<MethodInfo, IEnumerable<RegisterOp>>> methods = type.MethodsWithAttribute<RegisterOp>(false);
                    foreach (Tuple<MethodInfo, IEnumerable<RegisterOp>> method in methods)
                    {
                        IEnumerable<OpConstraint> paramConstraints = GetParameterConstraints(method.Item1, instance);
                        foreach (RegisterOp attribute in method.Item2)
                        {
                            attribute.DoRegister(instance, method.Item1, paramConstraints);
                        }
                    }
                }
            }
        }

        // 中文：收集方法各参数上 ArgConstraint 特性产生的约束并汇总返回。
        private static IEnumerable<OpConstraint> GetParameterConstraints(MethodInfo method, object instance)
        {
            IEnumerable<OpConstraint> result = Enumerable.Empty<OpConstraint>();
            foreach (Tuple<ParameterInfo, IEnumerable<ArgConstraintAttribute>> parameter in method.ParametersWithAttribute<ArgConstraintAttribute>(false))
            {
                foreach (ArgConstraintAttribute attribute in parameter.Item2)
                {
                    result = Enumerable.Concat(result, attribute.GetConstraints(parameter.Item1, instance));
                }
            }

            return result;
        }
    }
}
