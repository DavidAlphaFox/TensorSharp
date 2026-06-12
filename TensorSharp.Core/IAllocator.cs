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
    // 中文：底层线性代数后端的枚举，用于选择张量运算所使用的计算库或硬件加速器
    public enum BlasEnum
    {
        // 中文：使用 .NET 内置数学库（纯托管代码，跨平台兼容）
        DotNet,
        // 中文：使用 Intel Math Kernel Library 加速 CPU 矩阵运算
        MKL,
        // 中文：使用 NVIDIA CUDA GPU 加速并行计算
        CUDA,
        // 中文：使用 Apple Metal GPU 加速（通过 GGML 后端）
        GGML_METAL,
        // 中文：使用 GGML CPU 后端执行量化推理
        GGML_CPU,
        // 中文：使用 Apple MLX 框架进行 Apple Silicon 加速推理
        MLX
    }


    // 中文：内存分配器接口，定义各计算后端分配张量存储空间的统一契约
    public interface IAllocator
    {
        // 中文：获取当前分配器所使用的底层线性代数后端类型
        BlasEnum BlasEnum { get; }
        // 中文：获取当前分配器绑定的计算设备编号（如 GPU 索引，CPU 为 0）
        int DeviceId { get; }
        // 中文：按指定元素类型与数量分配张量底层存储对象（Storage）
        Storage Allocate(DType elementType, long elementCount);

        // 中文：获取当前已分配内存占设备总内存的比率（取值范围 0.0～1.0）
        float GetAllocatedMemoryRatio();
    }
}
