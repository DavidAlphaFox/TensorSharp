// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

// ──────【文件说明】──────
// 文件：BackendExecutionPlan.cs
// 用途：定义后端执行计划的具体实现，封装模型推理所使用的后端类型（CPU/Metal/CUDA 等），
//       并提供判断是否使用 GGML 后端以及权重是否应量化存储的辅助逻辑。
// 主要类型：BackendExecutionPlan（实现 IBackendExecutionPlan 接口）
// ────────────────────────

namespace TensorSharp.Models
{
    internal sealed class BackendExecutionPlan : IBackendExecutionPlan
    {
        // 中文：构造函数，根据指定的后端类型初始化执行计划
        public BackendExecutionPlan(BackendType backendType)
        {
            BackendType = backendType;
        }

        public BackendType BackendType { get; }

        // 中文：判断当前执行计划是否使用 GGML 系列后端（包括 GgmlCpu、GgmlMetal、GgmlCuda）
        public bool UsesGgmlBackend =>
            BackendType == BackendType.GgmlCpu ||
            BackendType == BackendType.GgmlMetal ||
            BackendType == BackendType.GgmlCuda;

        // 中文：根据后端类型与张量信息判断该权重是否应以量化格式存储，委托给 ModelBase 统一处理
        public bool ShouldStoreWeightQuantized(GgufTensorInfo info)
        {
            return ModelBase.ShouldStoreWeightQuantized(BackendType, info);
        }
    }
}

