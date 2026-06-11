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

namespace TensorSharp
{
    [StructLayout(LayoutKind.Sequential)]
#pragma warning disable CS8981 // Type names should be PascalCase
    public struct half
#pragma warning restore CS8981
    {
        public ushort x;

        // 中文：由 float 构造 half，内部存储为 16 位编码。
        public half(float value)
        {
            x = FloatToHalf(value);
        }

        // 中文：half 到 float 的隐式转换运算符。
        public static implicit operator float(half h)
        {
            return HalfToFloat(h.x);
        }

        // 中文：float 到 half 的隐式转换运算符。
        public static implicit operator half(float f)
        {
            return new half(f);
        }

        // 中文：按 IEEE 754 位运算将 float 编码为 16 位半精度（处理下溢与上溢饱和）。
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

        // 中文：将 16 位半精度编码解码为 float（处理非规格化数、Inf 与 NaN）。
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
