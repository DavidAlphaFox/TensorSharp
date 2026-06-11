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
using System.Linq;
using System.Text;

namespace TensorSharp
{
    internal static class TensorFormatting
    {
        // 中文：生成由 count 个字符 c 组成的字符串。
        private static string RepeatChar(char c, int count)
        {
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < count; ++i)
            {
                builder.Append(c);
            }
            return builder.ToString();
        }

        // 中文：构造指定宽度的整数对齐格式串。
        private static string GetIntFormat(int length)
        {
            string padding = RepeatChar('#', length - 1);
            return string.Format(" {0}0;-{0}0", padding);
        }

        // 中文：构造指定宽度、保留四位小数的浮点对齐格式串。
        private static string GetFloatFormat(int length)
        {
            string padding = RepeatChar('#', length - 1);
            return string.Format(" {0}0.0000;-{0}0.0000", padding);
        }

        // 中文：构造指定宽度的科学计数法格式串。
        private static string GetScientificFormat(int length)
        {
            int padCount = length - 6;
            string padding = RepeatChar('0', padCount);
            return string.Format(" {0}.0000e+00;-0.{0}e+00", padding);
        }




        // 中文：判断张量元素是否全为整数值（用于选择整数显示格式）。
        private static bool IsIntOnly(Storage storage, Tensor tensor)
        {
            // HACK this is a hacky way of iterating over the elements of the tensor.
            // if the tensor has holes, this will incorrectly include those elements
            // in the iteration.
            long minOffset = tensor.StorageOffset;
            long maxOffset = minOffset + TensorDimensionHelpers.GetStorageSize(tensor.Sizes, tensor.Strides) - 1;
            for (long i = minOffset; i <= maxOffset; ++i)
            {
                double value = Convert.ToDouble((object)storage.GetElementAsFloat(i));
                if (value != Math.Ceiling(value))
                {
                    return false;
                }
            }

            return true;
        }

        // 中文：返回张量元素的最小值与最大值的绝对值，用于决定格式与缩放。
        private static Tuple<double, double> AbsMinMax(Storage storage, Tensor tensor)
        {
            if (storage.ElementCount == 0)
            {
                return Tuple.Create(0.0, 0.0);
            }

            double min = storage.GetElementAsFloat(0);
            double max = storage.GetElementAsFloat(0);

            // HACK this is a hacky way of iterating over the elements of the tensor.
            // if the tensor has holes, this will incorrectly include those elements
            // in the iteration.
            long minOffset = tensor.StorageOffset;
            long maxOffset = minOffset + TensorDimensionHelpers.GetStorageSize(tensor.Sizes, tensor.Strides) - 1;

            for (long i = minOffset; i <= maxOffset; ++i)
            {
                float item = storage.GetElementAsFloat(i);
                if (item < min)
                {
                    min = item;
                }

                if (item > max)
                {
                    max = item;
                }
            }

            return Tuple.Create(Math.Abs(min), Math.Abs(max));
        }

        private enum FormatType
        {
            Int,
            Scientific,
            Float,
        }
        // 中文：根据数值范围与整数模式，决定格式类型、缩放因子与字段宽度。
        private static Tuple<FormatType, double, int> GetFormatSize(Tuple<double, double> minMax, bool intMode)
        {
            int expMin = minMax.Item1 != 0 ?
                    (int)Math.Floor(Math.Log10(minMax.Item1)) + 1 :
                    1;
            int expMax = minMax.Item2 != 0 ?
                    (int)Math.Floor(Math.Log10(minMax.Item2)) + 1 :
                    1;

            if (intMode)
            {
                if (expMax > 9)
                {
                    return Tuple.Create(FormatType.Scientific, 1.0, 11);
                }
                else
                {
                    return Tuple.Create(FormatType.Int, 1.0, expMax + 1);
                }
            }
            else
            {
                if (expMax - expMin > 4)
                {
                    int sz = Math.Abs(expMax) > 99 || Math.Abs(expMin) > 99 ?
                        12 : 11;
                    return Tuple.Create(FormatType.Scientific, 1.0, sz);
                }
                else
                {
                    if (expMax > 5 || expMax < 0)
                    {
                        return Tuple.Create(FormatType.Float,
                            Math.Pow(10, expMax - 1), 7);
                    }
                    else
                    {
                        return Tuple.Create(FormatType.Float, 1.0,
                            expMax == 0 ? 7 : expMax + 6);
                    }
                }
            }
        }

        // 中文：按格式类型与宽度构造对应的格式串。
        private static string BuildFormatString(FormatType type, int size)
        {
            switch (type)
            {
                case FormatType.Int: return GetIntFormat(size);
                case FormatType.Float: return GetFloatFormat(size);
                case FormatType.Scientific: return GetScientificFormat(size);
                default: throw new InvalidOperationException("Invalid format type " + type);
            }
        }

        // 中文：综合判断并返回打印格式串、缩放因子与字段宽度。
        private static Tuple<string, double, int> GetStorageFormat(Storage storage, Tensor tensor)
        {
            if (storage.ElementCount == 0)
            {
                return Tuple.Create("", 1.0, 0);
            }

            bool intMode = IsIntOnly(storage, tensor);
            Tuple<double, double> minMax = AbsMinMax(storage, tensor);

            Tuple<FormatType, double, int> formatSize = GetFormatSize(minMax, intMode);
            string formatString = BuildFormatString(formatSize.Item1, formatSize.Item3);

            return Tuple.Create("{0:" + formatString + "}", formatSize.Item2, formatSize.Item3);
        }

        // 中文：生成描述张量类型、各维尺寸与所在位置的摘要字符串。
        public static string FormatTensorTypeAndSize(Tensor tensor)
        {
            StringBuilder result = new StringBuilder();
            result
                .Append("[")
                .Append(tensor.ElementType)
                .Append(" tensor");

            if (tensor.DimensionCount == 0)
            {
                result.Append(" with no dimension");
            }
            else
            {
                result
                .Append(" of size ")
                .Append(tensor.Sizes[0]);

                for (int i = 1; i < tensor.DimensionCount; ++i)
                {
                    result.Append("x").Append(tensor.Sizes[i]);
                }
            }

            result.Append(" on ").Append(tensor.Storage.LocationDescription());
            result.Append("]");
            return result.ToString();
        }

        // 中文：将一维张量逐元素格式化输出（必要时带缩放因子）。
        private static void FormatVector(StringBuilder builder, Tensor tensor)
        {
            Tuple<string, double, int> storageFormat = GetStorageFormat(tensor.Storage, tensor);
            string format = storageFormat.Item1;
            double scale = storageFormat.Item2;

            if (scale != 1)
            {
                builder.AppendLine(scale + " *");
                for (int i = 0; i < tensor.Sizes[0]; ++i)
                {
                    double value = Convert.ToDouble((object)tensor.GetElementAsFloat(i)) / scale;
                    builder.AppendLine(string.Format(format, value));
                }
            }
            else
            {
                for (int i = 0; i < tensor.Sizes[0]; ++i)
                {
                    double value = Convert.ToDouble((object)tensor.GetElementAsFloat(i));
                    builder.AppendLine(string.Format(format, value));
                }
            }
        }

        // 中文：将二维张量按列分块、对齐缩进地格式化为矩阵输出。
        private static void FormatMatrix(StringBuilder builder, Tensor tensor, string indent)
        {
            Tuple<string, double, int> storageFormat = GetStorageFormat(tensor.Storage, tensor);
            string format = storageFormat.Item1;
            double scale = storageFormat.Item2;
            int sz = storageFormat.Item3;

            builder.Append(indent);

            int nColumnPerLine = (int)Math.Floor((80 - indent.Length) / (double)(sz + 1));
            long firstColumn = 0;
            long lastColumn;
            while (firstColumn < tensor.Sizes[1])
            {
                if (firstColumn + nColumnPerLine - 2 < tensor.Sizes[1])
                {
                    lastColumn = firstColumn + nColumnPerLine - 2;
                }
                else
                {
                    lastColumn = tensor.Sizes[1] - 1;
                }

                if (nColumnPerLine < tensor.Sizes[1])
                {
                    if (firstColumn != 1)
                    {
                        builder.AppendLine();
                    }
                    builder.Append("Columns ").Append(firstColumn).Append(" to ").Append(lastColumn).AppendLine();
                }

                if (scale != 1)
                {
                    builder.Append(scale).AppendLine(" *");
                }

                for (long l = 0; l < tensor.Sizes[0]; ++l)
                {
                    using (Tensor row = tensor.Select(0, l))
                    {
                        for (long c = firstColumn; c <= lastColumn; ++c)
                        {
                            double value = Convert.ToDouble((object)row.GetElementAsFloat(c)) / scale;
                            builder.Append(string.Format(format, value));
                            if (c == lastColumn)
                            {
                                builder.AppendLine();
                                if (l != tensor.Sizes[0])
                                {
                                    builder.Append(scale != 1 ? indent + " " : indent);
                                }
                            }
                            else
                            {
                                builder.Append(' ');
                            }
                        }
                    }
                }
                firstColumn = lastColumn + 1;
            }
        }

        // 中文：将三维及以上张量按高维索引遍历，逐个二维切片格式化输出。
        private static void FormatTensor(StringBuilder builder, Tensor tensor)
        {
            int startingLength = builder.Length;
            long[] counter = Enumerable.Repeat((long)0, tensor.DimensionCount - 2).ToArray();
            bool finished = false;
            counter[0] = -1;
            while (true)
            {
                for (int i = 0; i < tensor.DimensionCount - 2; ++i)
                {
                    counter[i]++;
                    if (counter[i] >= tensor.Sizes[i])
                    {
                        if (i == tensor.DimensionCount - 3)
                        {
                            finished = true;
                            break;
                        }
                        counter[i] = 1;
                    }
                    else
                    {
                        break;
                    }
                }

                if (finished)
                {
                    break;
                }

                if (builder.Length - startingLength > 1)
                {
                    builder.AppendLine();
                }

                builder.Append('(');
                Tensor tensorCopy = tensor.CopyRef();
                for (int i = 0; i < tensor.DimensionCount - 2; ++i)
                {
                    Tensor newCopy = tensorCopy.Select(0, counter[i]);
                    tensorCopy.Dispose();
                    tensorCopy = newCopy;
                    builder.Append(counter[i]).Append(',');
                }

                builder.AppendLine(".,.) = ");
                FormatMatrix(builder, tensorCopy, " ");

                tensorCopy.Dispose();
            }
        }

        // 中文：按维度数选择相应格式化方式，返回张量的完整文本表示。
        public static string Format(Tensor tensor)
        {
            StringBuilder result = new StringBuilder();
            if (tensor.DimensionCount == 0)
            {
            }
            else if (tensor.DimensionCount == 1)
            {
                FormatVector(result, tensor);
            }
            else if (tensor.DimensionCount == 2)
            {
                FormatMatrix(result, tensor, "");
            }
            else
            {
                FormatTensor(result, tensor);
            }

            result.AppendLine(FormatTensorTypeAndSize(tensor));
            return result.ToString();
        }
    }
}
