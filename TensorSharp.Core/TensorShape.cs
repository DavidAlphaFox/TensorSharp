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
    public readonly ref struct TensorShape
    {
        private readonly ReadOnlySpan<long> sizes;

        // 中文：由各维尺寸构造张量形状。
        public TensorShape(ReadOnlySpan<long> sizes)
        {
            this.sizes = sizes;
        }

        public int Rank => sizes.Length;
        public int Length => sizes.Length;
        public ReadOnlySpan<long> Sizes => sizes;
        public long ElementCount => TensorDimensionHelpers.ElementCount(sizes);

        public long this[int dimension] => sizes[dimension];

        // 中文：判断本形状各维尺寸是否与另一序列逐元素相等。
        public bool SequenceEqual(ReadOnlySpan<long> other)
        {
            return sizes.SequenceEqual(other);
        }

        // 中文：将各维尺寸复制为 long 数组返回。
        public long[] ToArray()
        {
            return sizes.ToArray();
        }

        // 中文：以 "x" 连接各维尺寸返回形状的字符串表示（如 2x3x4）。
        public override string ToString()
        {
            return TensorDimensionHelpers.FormatDimensions(sizes, "x");
        }
    }
}
