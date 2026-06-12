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
// 文件：ModelPrimitives.cs
// 用途：定义 TensorSharp.Runtime 中模型推理所需的基础类型，包括
//       BackendType（后端类型枚举）和 ModelConfig（模型架构配置类）。
//       这两个类型被框架中的各个推理后端（CPU / CUDA / MLX / GGML 系列）
//       共同依赖，用于描述语言模型的结构超参数。
// ──────────────────────

namespace TensorSharp.Runtime
{
    // 中文：推理后端类型枚举，标识当前运行时所使用的硬件加速方案
    public enum BackendType
    {
        // 中文：纯 CPU 后端（跨平台通用实现）
        Cpu,
        // 中文：NVIDIA CUDA GPU 后端
        Cuda,
        // 中文：Apple MLX 框架后端（适用于 Apple Silicon）
        Mlx,
        // 中文：基于 GGML 库的 CPU 后端
        GgmlCpu,
        // 中文：基于 GGML 库的 Apple Metal GPU 后端
        GgmlMetal,
        // 中文：基于 GGML 库的 CUDA GPU 后端
        GgmlCuda,
    }

    // 中文：模型架构配置类，封装 LLM 的结构超参数，供各推理后端统一读取
    public class ModelConfig
    {
        // 中文：模型架构名称（如 "llama"、"mistral" 等），用于选择对应的推理图
        public string Architecture { get; set; } = string.Empty;
        // 中文：隐藏层维度大小（即 d_model），决定嵌入向量与前馈网络的宽度
        public int HiddenSize { get; set; }
        // 中文：注意力头数量（Query 头数），影响多头自注意力并行度
        public int NumHeads { get; set; }
        // 中文：键值头数量，用于分组查询注意力（GQA）机制，小于 NumHeads 时启用 GQA
        public int NumKVHeads { get; set; }
        // 中文：每个注意力头的 Key 向量长度，为 0 时回退到 HeadDim 计算值
        public int KeyLength { get; set; }
        // 中文：每个注意力头的 Value 向量长度，为 0 时回退到 HeadDim 计算值
        public int ValueLength { get; set; }
        // 中文：RMSNorm / LayerNorm 中的数值稳定性 epsilon 值
        public float Eps { get; set; }
        // 中文：RoPE（旋转位置编码）基频，常见值为 10000，扩展上下文时可调大
        public float RopeBase { get; set; }
        // 中文：RoPE 频率缩放因子，默认为 1.0；长上下文微调时通常设为大于 1 的值
        public float RopeScale { get; set; } = 1f;
        // 中文：Transformer 解码器层数（即模型深度）
        public int NumLayers { get; set; }
        // 中文：词汇表大小，决定 Embedding 矩阵与 LM Head 输出维度
        public int VocabSize { get; set; }
        // 中文：前馈网络（FFN / MLP）中间层维度，MoE 模型中为单个专家的中间层大小
        public int IntermediateSize { get; set; }
        // 中文：对话模板字符串，用于将多轮消息格式化为模型输入（Jinja2 或自定义格式）
        public string ChatTemplate { get; set; } = string.Empty;

        // 中文：MoE（混合专家）模型中专家总数；非 MoE 模型设为 0
        public int NumExperts { get; set; }
        // 中文：MoE 推理时每个 token 实际激活（路由到）的专家数量
        public int NumExpertsUsed { get; set; }
        // 中文：滑动窗口注意力的窗口大小（Mistral/Qwen 等模型使用），0 表示全局注意力
        public int SlidingWindow { get; set; }
        // 中文：是否使用循环 KV Cache（配合滑动窗口注意力，节省显存）
        public bool UsesCircularKvCache { get; set; }
        // 中文：模型训练时的原始上下文长度，用于 RoPE 外推缩放计算的基准值
        public int OriginalContextLength { get; set; }

        // 中文：计算单个注意力头的维度：优先取 KeyLength，其次 ValueLength，最后回退为 HiddenSize / NumHeads
        public int HeadDim => KeyLength > 0 ? KeyLength : (ValueLength > 0 ? ValueLength : HiddenSize / NumHeads);
    }
}
