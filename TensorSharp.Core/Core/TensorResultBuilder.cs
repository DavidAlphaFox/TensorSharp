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

namespace TensorSharp.Core
{
    public static class TensorResultBuilder
    {
        // If a maybeResult is null, a new tensor will be constructed using the device id and element type of newTemplate
        // 中文：以模板张量的分配器与元素类型为依据，获取写入目标（long[] 尺寸重载）。
        public static Tensor GetWriteTarget(Tensor maybeResult, Tensor newTemplate, bool requireContiguous, params long[] requiredSizes)
        {
            return GetWriteTarget(maybeResult, newTemplate.Allocator, newTemplate.ElementType, requireContiguous, (ReadOnlySpan<long>)requiredSizes);
        }

        // 中文：以模板张量的分配器与元素类型为依据，获取写入目标（ReadOnlySpan 尺寸重载）。
        public static Tensor GetWriteTarget(Tensor maybeResult, Tensor newTemplate, bool requireContiguous, ReadOnlySpan<long> requiredSizes)
        {
            return GetWriteTarget(maybeResult, newTemplate.Allocator, newTemplate.ElementType, requireContiguous, requiredSizes);
        }

        // 中文：指定分配器与元素类型获取写入目标（long[] 尺寸重载，转发到 Span 版本）。
        public static Tensor GetWriteTarget(Tensor maybeResult, IAllocator allocatorForNew, DType elementTypeForNew, bool requireContiguous, params long[] requiredSizes)
        {
            return GetWriteTarget(maybeResult, allocatorForNew, elementTypeForNew, requireContiguous, (ReadOnlySpan<long>)requiredSizes);
        }

        // 中文：核心实现——已有结果张量则校验尺寸/连续性后复用，否则按要求新建张量。
        public static Tensor GetWriteTarget(Tensor maybeResult, IAllocator allocatorForNew, DType elementTypeForNew, bool requireContiguous, ReadOnlySpan<long> requiredSizes)
        {
            if (maybeResult != null)
            {
                if (!MatchesRequirements(maybeResult, requireContiguous, requiredSizes))
                {
                    string message = string.Format("output tensor does not match requirements. Tensor must have sizes {0}{1}",
                        TensorDimensionHelpers.FormatDimensions(requiredSizes, ", "),
                        requireContiguous ? "; and must be contiguous. " : ". ");

                    message += $"Tensor's actual shape is '{TensorDimensionHelpers.FormatDimensions(maybeResult.Sizes, ", ")}' and contiguous = '{maybeResult.IsContiguous()}'";

                    throw new InvalidOperationException(message);
                }
                return maybeResult;
            }
            else
            {
                return new Tensor(allocatorForNew, elementTypeForNew, requiredSizes);
            }
        }

        // 中文：判断张量是否满足连续性要求且尺寸与期望一致。
        private static bool MatchesRequirements(Tensor tensor, bool requireContiguous, ReadOnlySpan<long> requiredSizes)
        {
            if (requireContiguous && !tensor.IsContiguous())
            {
                return false;
            }

            return ArrayEqual(tensor.Sizes, requiredSizes);
        }

        // 中文：逐元素比较两个跨度是否长度相同且完全相等。
        public static bool ArrayEqual<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b)
            where T : IEquatable<T>
        {
            if (a.Length != b.Length)
            {
                return false;
            }

            for (int i = 0; i < a.Length; ++i)
            {
                if (!a[i].Equals(b[i]))
                {
                    return false;
                }
            }

            return true;
        }

        // 中文：逐元素比较两个跨度是否相等，但忽略指定下标处的差异。
        public static bool ArrayEqualExcept<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b, int ignoreIndex)
            where T : IEquatable<T>
        {
            if (a.Length != b.Length)
            {
                return false;
            }

            for (int i = 0; i < a.Length; ++i)
            {
                if (i == ignoreIndex)
                {
                    continue;
                }

                if (!a[i].Equals(b[i]))
                {
                    return false;
                }
            }

            return true;
        }
    }
}
