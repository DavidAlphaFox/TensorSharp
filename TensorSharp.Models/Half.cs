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
using System.Runtime.InteropServices;

// ───────────────────────────────────────────────────────────────────────────
// 【文件说明】IEEE 754 半精度浮点（F16）软件转换辅助结构体。
// 【主要类型】half：16 位浮点封装，提供与 float 的隐式互转，用于不支持硬件 F16 指令的路径。
// ───────────────────────────────────────────────────────────────────────────
namespace TensorSharp.Models
{
    [StructLayout(LayoutKind.Sequential)]
#pragma warning disable CS8981 // Type names should be PascalCase
    public struct half
#pragma warning restore CS8981
    {
        public ushort x;

        // 中文：从 F32 构造 half 值，调用 FloatToHalf 软件转换
        public half(float value)
        {
            x = FloatToHalf(value);
        }

        // 中文：half → float 隐式转换操作符，通过 HalfToFloat 还原 F32 值
        public static implicit operator float(half h)
        {
            return HalfToFloat(h.x);
        }

        // 中文：float → half 隐式转换操作符
        public static implicit operator half(float f)
        {
            return new half(f);
        }

        // 中文：F32 → F16 软件编码：提取符号/指数/尾数，截断指数范围后拼装 16 位表示
        private static ushort FloatToHalf(float value)
        {
            int bits = System.BitConverter.SingleToInt32Bits(value);
            int sign = (bits >> 16) & 0x8000;
            int exponent = ((bits >> 23) & 0xFF) - 127 + 15;
            int mantissa = bits & 0x7FFFFF;

            if (exponent <= 0)
                return (ushort)sign;
            if (exponent > 30)
                return (ushort)(sign | 0x7C00);

            return (ushort)(sign | (exponent << 10) | (mantissa >> 13));
        }

        // 中文：F16 → F32 软件解码：分别处理次正规数、无穷/NaN 和正规数三种情形
        private static float HalfToFloat(ushort value)
        {
            int sign = (value >> 15) & 1;
            int exponent = (value >> 10) & 0x1F;
            int mantissa = value & 0x3FF;

            if (exponent == 0)
            {
                if (mantissa == 0) return sign == 0 ? 0f : -0f;
                float result = mantissa / 1024f * (1f / 16384f);
                return sign == 0 ? result : -result;
            }
            if (exponent == 31)
            {
                return mantissa == 0
                    ? (sign == 0 ? float.PositiveInfinity : float.NegativeInfinity)
                    : float.NaN;
            }

            float val = (1f + mantissa / 1024f) * MathF.Pow(2, exponent - 15);
            return sign == 0 ? val : -val;
        }
    }
}

