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
    public readonly ref struct TensorView
    {
        private readonly ReadOnlySpan<long> sizes;
        private readonly ReadOnlySpan<long> strides;

        // 中文：构造张量视图，校验尺寸与步幅长度一致并记录存储偏移。
        public TensorView(ReadOnlySpan<long> sizes, ReadOnlySpan<long> strides, long storageOffset)
        {
            if (sizes.Length != strides.Length)
            {
                throw new ArgumentException("sizes and strides must have the same length");
            }

            this.sizes = sizes;
            this.strides = strides;
            StorageOffset = storageOffset;
        }

        public int DimensionCount => sizes.Length;
        public TensorShape Shape => new TensorShape(sizes);
        public ReadOnlySpan<long> Sizes => sizes;
        public ReadOnlySpan<long> Strides => strides;
        public long StorageOffset { get; }

        // 中文：返回指定维度的尺寸。
        public long Size(int dimension)
        {
            return sizes[dimension];
        }

        // 中文：返回指定维度的步幅。
        public long Stride(int dimension)
        {
            return strides[dimension];
        }

        // 中文：根据尺寸与步幅计算所需的底层存储元素数。
        public long GetStorageSize()
        {
            return TensorDimensionHelpers.GetStorageSize(sizes, strides);
        }
    }
}
