// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
// ───────────────────────────────────────────────────────────────────────────
// 【文件说明】内存分配器接口与 BLAS 后端枚举。
// 【主要类型】IAllocator：各计算后端通过它分配 Storage（张量内存）；
//             BlasEnum：可选的底层线性代数库（DotNet / MKL / CUDA 等）。
// ───────────────────────────────────────────────────────────────────────────
namespace TensorSharp
{
    public enum BlasEnum
    {
        DotNet,
        MKL,
        CUDA,
        GGML_METAL,
        GGML_CPU,
        MLX
    }


    public interface IAllocator
    {
        BlasEnum BlasEnum { get; }
        int DeviceId { get; }
        Storage Allocate(DType elementType, long elementCount);

        float GetAllocatedMemoryRatio();
    }
}
