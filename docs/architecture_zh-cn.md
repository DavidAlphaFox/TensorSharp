# TensorSharp 项目架构总览

[English](../README.md) | [中文](architecture_zh-cn.md)

> 本文是 TensorSharp 代码库的整体架构参考，面向希望快速理解项目分层、模块职责与
> 一次推理完整链路的开发者。逐个模型架构的实现细节见
> [各模型架构卡片](models/README_zh-cn.md)；分页 KV 与连续批处理的深入说明见
> [分页注意力与连续批处理](PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md)。

## 一、项目定位

TensorSharp 是一个**纯 C# 实现的大语言模型（LLM）本地推理引擎**，加载 GGUF
模型文件，提供：

- **命令行程序（CLI）** —— 一次性生成与交互式 REPL；
- **Web 服务** —— 浏览器聊天界面，以及 Ollama / OpenAI 兼容的 HTTP API；
- **多后端计算** —— 纯 C# CPU、直连 CUDA/cuBLAS、Apple Silicon MLX/Metal、
  以及 GGML（CPU / Metal / CUDA）原生后端。

关键特性：连续批处理 + 分页 KV 缓存（vLLM 式）、多模态（图像 / 视频 / 音频）输入、
工具调用、思考 / 推理模式，以及免反量化的原生量化矩阵乘。

- 版本：`2.8.6`（见 `Directory.Build.props`）
- 运行时：.NET 10，`Nullable` 开启，允许 `unsafe`
- 许可证：BSD-3-Clause

## 二、解决方案结构（11 个项目）

| 项目 | 角色 | 关键内容 |
|---|---|---|
| `TensorSharp.Core` | 张量与算子基础 | `Tensor` / `Storage` / `IAllocator` / `OpRegistry` / `DType` / 纯 C# CPU 算子 |
| `TensorSharp.Runtime` | 推理运行时 | 调度器、分页 KV、分词器、GGUF 读取、模板、采样、结构化输出 |
| `TensorSharp.Models` | 模型实现 | `ModelBase` 基类、各模型架构、托管量化算子、多模态注入 |
| `TensorSharp.Backends.Cuda` | CUDA 后端 | 直连 PTX 内核 + cuBLAS |
| `TensorSharp.Backends.MLX` | MLX 后端 | Apple Silicon Metal（含 P/Invoke 原生桥接） |
| `TensorSharp.Backends.GGML` | GGML 后端 | GGML CPU / Metal / CUDA 的托管封装 |
| `TensorSharp.GGML.Native` | GGML 原生层 | C++ 算子（`ggml_ops_*.cpp`），CMake 构建，首次编译自动生成 |
| `TensorSharp.Server` | Web 服务 | ASP.NET Core 入口、端点、协议适配器、生成流水线 |
| `TensorSharp.Cli` | 命令行 | 参数解析、模型加载、一次性 / 交互式运行 |
| `TensorSharp.TestMatrix` | 测试 / 评测 | 模型 × 后端 × 特性 × 环境变量 矩阵扫描 |
| `InferenceWeb.Tests` | 单元 / 集成测试 | 各子系统的回归测试与基准 |
| `AdvUtils` | 通用工具 | 日志等辅助 |

## 三、分层架构

```text
┌─────────────────────────────────────────────────────────────────────┐
│  接入层      TensorSharp.Cli（命令行 / REPL）   TensorSharp.Server     │
│              Program.cs / InteractiveSession      （ASP.NET Core）     │
├─────────────────────────────────────────────────────────────────────┤
│  服务 / 协议  ProtocolAdapters：Ollama / OpenAI / WebUI                 │
│              Endpoints · ChatGenerationPipeline · ModelService         │
├─────────────────────────────────────────────────────────────────────┤
│  运行时       TensorSharp.Runtime                                      │
│   ├─ Scheduling：ContinuousBatchScheduler / BatchExecutor /            │
│   │              InferenceEngine（vLLM 式迭代级调度）                   │
│   ├─ Paged：分页 KV 缓存（BlockPool / BlockTable / 块哈希前缀复用）     │
│   ├─ 分词器：BpeTokenizer / SentencePieceTokenizer · GgufReader        │
│   └─ 模板 / 采样 / 结构化输出：Jinja2Template / ChatTemplate /          │
│      TokenSampler / SamplingConfig / StructuredOutputs                 │
├─────────────────────────────────────────────────────────────────────┤
│  模型层       TensorSharp.Models                                       │
│   ├─ ModelBase（统一前向 / 解码基类）                                  │
│   ├─ Models/：Gemma3 Gemma4 Qwen3 Qwen35 GptOss Nemotron Mistral3      │
│   │           （含视觉 / 音频编码器、批量前向）                         │
│   ├─ ManagedQuantizedOps（量化矩阵乘，免反量化）                       │
│   └─ ModelMultimodalInjector（多模态注入）                            │
├─────────────────────────────────────────────────────────────────────┤
│  计算后端     IBasicOps 抽象 + OpRegistry 算子注册分发                  │
│   ├─ TensorSharp.Core：纯 C# CPU（TensorApplyCPU / 矩阵乘 / BLAS）      │
│   ├─ Backends.Cuda：直连 CUDA/cuBLAS（PTX 内核）                       │
│   ├─ Backends.MLX： Apple Silicon Metal                                │
│   └─ Backends.GGML： GGML CPU / Metal / CUDA（P/Invoke 原生桥接）       │
├─────────────────────────────────────────────────────────────────────┤
│  原生层       TensorSharp.GGML.Native（C++ ggml_ops_*.cpp）            │
│              CMake 构建，首次编译自动生成                              │
└─────────────────────────────────────────────────────────────────────┘
```

## 四、核心设计要点

### 1. 张量与存储

- `Tensor`（`TensorSharp.Core/Tensor.cs`）是基本数据结构：由 **形状 `sizes`、
  步长 `strides`、底层存储 `Storage`、存储偏移** 组成，支持视图 / 切片而不拷贝。
- `Storage` 是抽象内存块，各后端派生实现；通过 `RefCounted` **引用计数**统一管理
  生命周期，引用归零时调用 `Destroy()` 释放（CPU 固定内存或 GPU 显存）。
- `IAllocator` 负责按 `DType` 分配 `Storage`；`DType` 枚举覆盖 FP32 / FP16 及各类
  GGUF 量化格式。

### 2. 后端无关的算子分发

- 算子不通过虚函数表，而是通过 `OpRegistry`（`TensorSharp.Core/OpRegistry.cs`）：
  各后端以特性注解声明自己实现的算子，注册表按 **操作名 + 约束（设备 / 数据类型）**
  反射匹配并调用。
- **回退保证**：任何后端未实现的算子都会回退到 CPU 实现，因此各后端输出保持一致，
  这也是「每个后端对未实现 op 回退 CPU」承诺的实现基础。

### 3. 模型层

- `ModelBase`（`TensorSharp.Models/ModelBase.cs`，体量很大）是所有架构的统一基类，
  封装通用前向 / 解码、各层张量与权重管理、采样接入。
- 每个模型族用 **partial class** 拆分：主体（`XxxModel.cs`）+ 批量前向
  （`XxxModel.BatchedForward.cs`）+ 视觉 / 音频编码器 + 图像 / 音频预处理器。
- `QuantizedWeight` 持有原生内存指针的量化权重；`ManagedQuantizedOps` 在
  **不反量化为 FP32** 的前提下直接做量化矩阵乘（Q4_K / Q8_0 / MXFP4 / IQ2_XXS 等）。

### 4. 连续批处理与分页 KV

- `InferenceEngine` 是引擎门面，独占一个工作线程跑「步进循环」，对外提供
  `SubmitRequest` / 逐 token 输出句柄。
- `ContinuousBatchScheduler` 做 vLLM 式 **迭代级调度**：维护等待队列与运行集合，
  每步决定下一次前向算哪些序列，分配 KV 块、命中前缀缓存、块耗尽时抢占低优先级序列。
- `PagedKvCacheManager` + `Paged/*`（`BlockPool` / `BlockTable` / `BlockHashIndex`）
  实现分页 KV 池与 **块哈希前缀共享**：新请求若与历史共享前缀，prefill 成本从 O(n)
  降为 O(块大小 × 后缀)。
- `BatchExecutor` 真正驱动模型前向：维护「任一时刻模型 KV 张量只持有一个序列状态」
  的不变式，并在支持批量分页注意力的模型上让多序列共享 KV。

### 5. 三套 API 协议

- `TensorSharp.Server/ProtocolAdapters` 下的 `OllamaAdapter` / `OpenAIChatAdapter`
  / `WebUiAdapter` 把三种协议适配到同一条 `ChatGenerationPipeline`。
- `ChatGenerationPipeline` 串联 **提示词渲染 → 推理引擎 → 输出解析 → 流式返回**；
  KV 状态生命周期由引擎持有，会话层只作历史容器。

## 五、一次推理的完整链路

```text
GGUF 文件 ──(GgufReader 解析元数据/张量)──▶ 构建具体模型(ModelBase 派生)
   │
   ▼
请求(CLI / Ollama / OpenAI / WebUI)
   │  ProtocolAdapter 归一化
   ▼
ChatTemplate / Jinja2Template 渲染提示词  ──(多模态时)──▶ ModelMultimodalInjector 注入图像/音频 embedding
   │
   ▼
InferenceEngine.SubmitRequest
   │  ContinuousBatchScheduler 调度 + PagedKvCacheManager 分配/复用 KV 块
   ▼
BatchExecutor → 模型前向(各后端算子, 量化免反量化)
   │
   ▼
TokenSampler 采样(SamplingConfig) ──▶ OutputParser / StructuredOutputs 解析(工具调用/思考/JSON)
   │
   ▼
逐 token 流式返回(SSE / Ollama / OpenAI 响应)
```

## 六、计算后端一览

| 后端 | 标志 | 适用硬件 | 说明 |
|---|---|---|---|
| 纯 C# CPU | `cpu` | 任意 / 调试 | 无原生依赖，结果基准 |
| GGML CPU | `ggml_cpu` | 任意 | 原生内核，比纯 C# CPU 快 |
| GGML Metal | `ggml_metal` | Apple Silicon | macOS 默认 |
| MLX | `mlx` | Apple Silicon | 另一条 Apple GPU 路径 |
| GGML CUDA | `ggml_cuda` | NVIDIA | 最常测试的 NVIDIA 路径 |
| 直连 CUDA | `cuda` | NVIDIA | 直接 PTX/cuBLAS，实验性 |

所有后端对未实现的算子统一回退 CPU，输出保持一致。

## 七、相关文档

- [算子分派与计算后端](compute_backends_ops_zh-cn.md)
- [各模型架构卡片](models/README_zh-cn.md)
- [分页注意力与连续批处理](PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md)
- [环境变量特性矩阵](env_var_feature_matrix_zh-cn.md)
- [推理基准矩阵](inference_benchmark_matrix_zh-cn.md)
- [项目 README（中文）](../README_zh-cn.md)
