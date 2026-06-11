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
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using TensorSharp.Core;

namespace TensorSharp.Cpu
{
    public static class NativeWrapper
    {
        // 中文：按名称反射获取 CpuOpsNative 中的公共静态方法信息。
        public static MethodInfo GetMethod(string name)
        {
            return typeof(CpuOpsNative).GetMethod(name, BindingFlags.Public | BindingFlags.Static);
        }

        // 中文：逐元素运算入口，必要时按另一参数张量尺寸创建结果张量后调用原生方法。
        public static Tensor InvokeNullableResultElementwise(MethodInfo method, params object[] args)
        {
            Tensor resultTensor;
            if (args[0] == null)
            {
                Tensor otherTensor = args.OfType<Tensor>().First();
                resultTensor = TensorResultBuilder.GetWriteTarget(null, otherTensor, false, otherTensor.Sizes);
            }
            else
            {
                Tensor resultSrc = (Tensor)args[0];
                Tensor otherTensor = args.OfType<Tensor>().Skip(1).First();
                resultTensor = TensorResultBuilder.GetWriteTarget(resultSrc, otherTensor, false, otherTensor.Sizes);
            }

            args[0] = resultTensor;
            InvokeTypeMatch(method, args);
            return resultTensor;
        }

        // 中文：按维度规约运算入口，校验维度并以缩减后尺寸创建结果张量后调用原生方法。
        public static Tensor InvokeNullableResultDimensionwise(MethodInfo method, Tensor result, Tensor src, int dimension, params object[] extraArgs)
        {
            if (dimension < 0 || dimension >= src.Sizes.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(dimension));
            }

            long[] desiredSize = src.Sizes.ToArray();
            desiredSize[dimension] = 1;
            Tensor resultTensor = TensorResultBuilder.GetWriteTarget(result, src, false, desiredSize);

            List<object> finalArgs = new List<object>(extraArgs.Length + 3)
            {
                resultTensor,
                src,
                dimension
            };
            finalArgs.AddRange(extraArgs);
            InvokeTypeMatch(method, finalArgs.ToArray());
            return resultTensor;
        }


        // 中文：仅创建按维度规约的结果张量（指定维度尺寸置 1），不执行运算。
        public static Tensor CreateResultDimensionwise(Tensor result, Tensor src, int dimension)
        {
            if (dimension < 0 || dimension >= src.Sizes.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(dimension));
            }

            long[] desiredSize = src.Sizes.ToArray();
            desiredSize[dimension] = 1;
            Tensor resultTensor = TensorResultBuilder.GetWriteTarget(result, src, false, desiredSize);

            return resultTensor;
        }

        // 中文：校验所有张量参数元素类型一致后再调用原生方法。
        public static void InvokeTypeMatch(MethodInfo method, params object[] args)
        {
            IEnumerable<Tensor> tensors = args.OfType<Tensor>();
            if (tensors.Any())
            {
                DType elemType = tensors.First().ElementType;
                if (!tensors.All(x => x.ElementType == elemType))
                {
                    string allTypes = string.Join(", ", tensors.Select(x => x.ElementType));
                    throw new InvalidOperationException("All tensors must have the same argument types. Given: " + allTypes);
                }
            }

            Invoke(method, args);
        }


        // 中文：为张量构建原生 TensorRef64 结构指针，返回可释放对象用于回收非托管内存。
        public static IDisposable BuildTensorRefPtr(Tensor tensor, out IntPtr tensorRefPtr)
        {
            TensorRef64 tensorRef = NativeWrapper.AllocTensorRef(tensor);
            IntPtr tensorPtr = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(TensorRef64)));
            Marshal.StructureToPtr(tensorRef, tensorPtr, false);

            tensorRefPtr = tensorPtr;

            return new DelegateDisposable(() =>
            {
                Marshal.FreeHGlobal(tensorPtr);
                NativeWrapper.FreeTensorRef(tensorRef);
            });
        }

        // 中文：将张量参数转换为原生 TensorRef64 指针后反射调用原生方法，结束后释放并检查返回码。
        public static void Invoke(MethodInfo method, params object[] args)
        {
            List<TensorRef64> freeListTensor = new List<TensorRef64>();
            List<IntPtr> freeListPtr = new List<IntPtr>();

            try
            {
                for (int i = 0; i < args.Length; ++i)
                {
                    if (args[i] is Tensor tensor)
                    {
                        if (!(tensor.Storage is CpuStorage))
                        {
                            throw new InvalidOperationException("Argument " + i + " is not a Cpu tensor");
                        }

                        TensorRef64 tensorRef = AllocTensorRef(tensor);
                        IntPtr tensorPtr = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(TensorRef64)));
                        Marshal.StructureToPtr(tensorRef, tensorPtr, false);

                        args[i] = tensorPtr;

                        freeListTensor.Add(tensorRef);
                        freeListPtr.Add(tensorPtr);
                    }
                }

                //return method.Invoke(null, args);
                int result = (int)method.Invoke(null, args);
                if (result != 0)
                {
                    throw new ApplicationException(GetLastError());
                }
            }
            finally
            {
                foreach (TensorRef64 tensorRef in freeListTensor)
                {
                    FreeTensorRef(tensorRef);
                }

                foreach (IntPtr tensorPtr in freeListPtr)
                {
                    Marshal.FreeHGlobal(tensorPtr);
                }
            }
        }

        // 中文：检查原生调用返回码，非零时抛出携带原生错误信息的异常。
        public static void CheckResult(int result)
        {
            if (result != 0)
            {
                throw new ApplicationException(GetLastError());
            }
        }

        // 中文：从原生层获取并转换最近一次错误信息字符串。
        private static string GetLastError()
        {
            IntPtr strPtr = CpuOpsNative.TS_GetLastError();
            return Marshal.PtrToStringAnsi(strPtr);
        }


        // 中文：根据张量构建 TensorRef64（缓冲区指针、维度、尺寸、步幅、元素类型）。
        public static TensorRef64 AllocTensorRef(Tensor tensor)
        {
            TensorRef64 tensorRef = new TensorRef64
            {
                buffer = CpuNativeHelpers.GetBufferStart(tensor),
                dimCount = tensor.Sizes.Length,
                sizes = AllocArray(tensor.Sizes),
                strides = AllocArray(tensor.Strides),
                elementType = (CpuDType)tensor.ElementType
            };
            return tensorRef;
        }

        // 中文：在非托管堆上分配并拷贝 long 数组，返回其指针。
        private static IntPtr AllocArray(ReadOnlySpan<long> data)
        {
            IntPtr result = Marshal.AllocHGlobal(sizeof(long) * data.Length);
            long[] copy = data.ToArray();
            Marshal.Copy(copy, 0, result, copy.Length);
            return result;
        }

        // 中文：释放 TensorRef64 中尺寸与步幅数组占用的非托管内存。
        public static void FreeTensorRef(TensorRef64 tensorRef)
        {
            Marshal.FreeHGlobal(tensorRef.sizes);
            Marshal.FreeHGlobal(tensorRef.strides);
        }
    }
}
