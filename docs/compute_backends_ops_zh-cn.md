# TensorSharp 算子分派与计算后端

[English](../README.md#compute-backends) | [中文](compute_backends_ops_zh-cn.md)

> 本文讲清楚 TensorSharp 的**张量算子是如何被分派到各计算后端、以及各后端如何加载与调用其原生算子**的。
> 项目整体分层见 [项目架构总览](architecture_zh-cn.md)；分页 KV 与连续批处理见
> [分页注意力与连续批处理](PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md)。

## 一、总览：一套分派，多种后端

TensorSharp 的所有计算最终都作用于 `Tensor`，但具体由哪个后端执行，取决于该张量的
**存储类型（Storage）**。上层代码永远只调用后端无关的门面 `Ops.*`，由 `OpRegistry`
按存储类型把调用路由到对应后端的算子实现；任何后端未实现的算子都**回退到 CPU**，
从而保证所有后端输出一致。

```text
            上层模型/运行时代码
                   │  只调用门面 API
                   ▼
            Ops.Xxx(...)           （TensorSharp.Core/Ops.cs，单行包装）
                   │  OpRegistry.Invoke("xxx", ...)
                   ▼
        ┌──────── OpRegistry ────────┐   按「算子名 + 参数张量的 Storage 类型」匹配
        │  反射收集 [RegisterOpStorageType] │
        ▼          ▼          ▼          ▼          ▼
   CpuBasicOps  CudaBasicOps  GgmlBasicOps  MlxBasicOps  …
   (CpuStorage) (CudaStorage) (GgmlStorage) (MlxStorage)
        │          │            │            │
        │          │            │            │  各后端内部：能则走原生/GPU，否则回退 CPU
        ▼          ▼            ▼            ▼
   纯 C# 计算   PTX 内核+cuBLAS  原生共享库    MLX C API
              (CUDA Driver API) (P/Invoke)   (P/Invoke)
```

### 1. 关键构件（均在 `TensorSharp.Core`）

| 构件 | 文件 | 职责 |
|---|---|---|
| `Tensor` | `Tensor.cs` | 多维数组：形状/步长/存储/偏移 |
| `Storage` | `Storage.cs` | 底层内存抽象，各后端派生（`CpuStorage`/`CudaStorage`/`GgmlStorage`/`MlxStorage`） |
| `Ops` | `Ops.cs` | 对外**门面 API**，每个方法只是 `OpRegistry.Invoke("名", ...)` 的单行包装 |
| `OpRegistry` | `OpRegistry.cs` | 算子注册表：反射收集各后端算子，按名+存储类型分派 |
| `[RegisterOpStorageType]` | `OpRegistryAttributes.cs` | 标注某方法实现了「某算子 + 某存储类型」 |

### 2. 分派的三步

```csharp
// ① 门面（后端无关）
public static void Fill(Tensor result, float value)
    => OpRegistry.Invoke("fill", result, value);

// ② 注册（启动时反射登记一次）
[RegisterOpStorageType("fill", typeof(CudaStorage))]
public static void Fill(Tensor result, float value) { ... }   // CUDA 版
[RegisterOpStorageType("fill", typeof(GgmlStorage))]
public static void Fill(Tensor result, float value) { ... }   // GGML 版

// ③ 匹配（运行时）
//   OpRegistry 看参数张量的 Storage 是 CudaStorage 还是 GgmlStorage…，选对应实现
```

每个后端在启动时调用自己的 `XxxBackend.Register()` → `OpRegistry.RegisterAssembly(程序集)`
完成注册（如 `CudaBackend.Register()`、GGML 在 `GgmlContext` 构造时注册）。

### 3. 统一的「先原生、后回退」结构

每个后端算子都是这个形状——**能用原生/GPU 就用，否则回退 CPU**：

```csharp
[RegisterOpStorageType("fill", typeof(CudaStorage))]
public static void Fill(Tensor result, float value)
{
    if (CudaKernelOps.TryFill(result, value)) return;          // 满足条件 → GPU 内核
    CudaCpuFallback.InvokeVoid("fill", result, result, value); // 否则 → 拷回 CPU 用 OpRegistry 算
}
```

这正是 README「每个后端对未实现的算子都回退 CPU，输出在所有后端保持一致」的实现基础。

---

## 二、CPU 后端（纯 C#）

`TensorSharp.Core` 内置：`CpuBasicOps`（注册算子）+ `TensorApplyCPU`（逐元素计算引擎，
通过跨步迭代对每个元素套用一元/二元/三元运算与归约）+ `MatrixMultiplication`
（含从 BLAS 翻译来的 `SGEMM`/`DGEMM` 参考实现，可选 OpenBLAS 原生加速）。无原生依赖，
是所有后端回退的目标，也是结果正确性的基准。

---

## 三、CUDA 后端：PTX 内核 + 运行时 JIT

项目 `TensorSharp.Backends.Cuda`。两类原生算子：**自研 PTX 内核** + **cuBLAS**。

### 1. 构建期：`.cu` → PTX（不是 cubin）

`csproj` 用 `nvcc -ptx -arch=<自动探测架构>` 把 `native/kernels/tensorsharp_kernels.cu`
编译成 **PTX 文本**（`native/ptx/tensorsharp_kernels.ptx`），拷到输出目录的 `cuda_kernels/`。
- 架构由 `native/resolve-cuda-arch.{sh,ps1}` 自动探测。
- **没有 nvcc 时构建仍成功**，只是不产生 PTX → 这些算子全部走 CPU 回退。

> PTX 是 NVIDIA 的**虚拟指令集（中间表示）**，不是最终机器码。运行时由 GPU **驱动 JIT**
> 编成当前显卡的 SASS。发布 PTX 而非 cubin 是为了「一份内核跨多种 GPU」的可移植性。

### 2. 加载期：CUDA Driver API 装载 PTX

不走高层 Runtime，而是直接用 Driver API（`Interop/CudaDriverApi.cs` 的 `[DllImport]`）：

```text
LocatePtxPath() 找到 .ptx
  → CudaModule.LoadFromBytes() 调 cuModuleLoadData()      （驱动 JIT：PTX → SASS）
  → CudaKernels 构造时对每个内核 module.GetFunction("ts_fill_f32") 解析并缓存函数句柄
```

内核实例挂在分配器上：`CudaAllocator` 构造时 `CudaKernels.TryCreate()`，存为 `allocator.Kernels`
（PTX 缺失则为 `null`）。

### 3. 运行期：分派 → 启动内核

```csharp
// CudaKernelOps.TryFill 内部
TryGetContiguous(result, out var storage, out var ptr, out var count);  // 取设备指针，要求连续/F32|F16
allocator.Context.MakeCurrent();                                        // 绑定 CUDA 上下文
kernels.LaunchFillF32(ptr, count, value, allocator.Stream.Handle);      // → cuLaunchKernel（异步，走 stream）
storage.MarkDeviceModified();                                           // 标记设备数据已变（惰性主机同步）
```

`Launch()` 统一封装 `cuLaunchKernel`，网格大小 `Grid(count)=⌈count/BlockSize⌉`。矩阵乘走
`CudaBlas`（`cublasSgemm`/`SgemmStridedBatched`）；量化权重走 `CudaQuantizedOps`，在设备上
**免反量化矩阵乘**并缓存已上传权重。

### 4. 回退

`TryFill` 在「PTX 未加载 / 类型不支持 / 非连续 / 超 int 上限」时返回 `false`，
`CudaCpuFallback` 把 CUDA 张量拷回主机，用 `OpRegistry` 调 CPU 版算完再写回设备。

---

## 四、GGML 后端：预编译共享库 + P/Invoke

项目 `TensorSharp.Backends.GGML` + 原生 `TensorSharp.GGML.Native`。与 CUDA 后端**本质不同**：
不是 PTX/JIT，而是**预编译的共享库 + P/Invoke**。

### 1. 构建期：CMake → 单个共享库 `GgmlOps`

`TensorSharp.GGML.Native/`（`ggml_ops_*.cpp` 等十余个 C++ 文件 + 内嵌 ggml）由 CMake 编成：

| 平台 | 产物 | 构建脚本 |
|---|---|---|
| Linux | `libGgmlOps.so` | `build-linux.sh` |
| macOS | `libGgmlOps.dylib` | `build-macos.sh` |
| Windows | `GgmlOps.dll` | `build-windows.ps1` |

- **一个库内含三套后端**：CPU（恒有）+ Metal（仅 macOS）+ CUDA（`--cuda` 编，需
  `TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON`）。运行时由参数选用。
- csproj 用 **stamp 文件做 MSBuild 级增量**：源码/配置没变则跳过原生构建。

### 2. 加载期：P/Invoke + 自定义 DllImport 解析器

`GgmlNative.cs` 用 `[DllImport("GgmlOps")]` 声明全部原生函数（86 个 extern）。库定位靠
静态构造函数注册的解析器：

```csharp
static GgmlNative()
    => NativeLibrary.SetDllImportResolver(typeof(GgmlNative).Assembly, ImportResolver);

private static IntPtr ImportResolver(string libraryName, ...)
{
    foreach (string candidate in GetCandidatePaths(assembly))   // 输出目录/程序集目录/仓库 build 目录
        if (File.Exists(candidate) && NativeLibrary.TryLoad(candidate, out IntPtr handle))
            return handle;
    return IntPtr.Zero;
}
```

### 3. 初始化与后端选择

`GgmlContext` 构造时（每模型一个）：

```csharp
MemoryPool = new GgmlMemoryPool(backendType);    // 原生内存池
MemoryPool.EnsureInitialBlocks();
GgmlNative.EnsureAvailable(backendType);          // 调 TSGgml_IsBackendAvailable 校验所选后端
OpRegistry.RegisterAssembly(...);                 // 注册 GGML 算子
GgmlNative.SetAsyncCompute(Metal && ...);         // Metal 默认惰性异步
```

### 4. 运行期：构造张量视图，零拷贝跨边界

```csharp
// GgmlBasicOps.Addmm（已 [RegisterOpStorageType("addmm", typeof(GgmlStorage))]）
TryCreateStandardView(writeTarget, out GgmlTensorView2D resultView);  // 取原生指针+维度+步长
TryCreateRawView(m1, out var m1View);
TryCreateRawView(m2, out var m2View);
GgmlNative.Addmm(resultView, srcView, m1View, m2View, beta, alpha);

// GgmlNative.Addmm → extern，返回 int 状态码
public static void Addmm(GgmlTensorView2D result, ...)
    => CheckResult(TSGgml_AddmmF32(result, src, m1, m2, beta, alpha), "addmm");
```

传过去的是轻量结构体 `GgmlTensorView2D { 数据指针 Data, 维度 Dim0/Dim1, 步长 Stride0/Stride1,
字节数 RawBytes }`——**直接传 GGML 存储的原生指针，零拷贝**。`CheckResult` 检查返回码，
非 0 取 `TSGgml_GetLastError()` 抛异常。量化权重走 `AddmmQuant(...,m2Data,m2GgmlType,...)`，
把 GGUF 量化字节 + ggml 类型 id 传给原生，**免反量化**。

### 5. 原生侧：逐算子驱动

每个 `TSGgml_*` 内部：把传入指针/维度包装成 `ggml_tensor` → 构建一张**很小的 ggml 计算图**
→ 在所选后端（CPU/Metal/CUDA）上 `ggml_backend_graph_compute` → 结果写回 `result` 指针。
即 **C# 一次驱动一个算子**。

Metal 上为省去每算子命令缓冲同步（约 30–100µs），默认**异步惰性提交**（`SetAsyncCompute`）：
算子提交后立即返回，**只在主机真正读数据时才同步**——这也是 `Storage.EnsureHostReadable()`
（`TensorComputePrimitives.GetFloatPointer` 会调）存在的原因。`TS_GGML_ASYNC_COMPUTE=0` 可关闭。

---

## 五、MLX 后端（Apple Silicon）

项目 `TensorSharp.Backends.MLX`。与 GGML 类似走 **P/Invoke**，`MlxNative.cs` 绑定 MLX C API
（102 个 DllImport：数组创建/释放、matmul、量化矩阵乘、融合快算子、Metal 自定义内核、
图编译/eval 等）。MLX 调用经 `MlxWorker` 串行化；采用**惰性计算图 + `eval` 求值**模型
（`MlxCompiledOps` 编译并缓存图）。同样有 `MlxCpuFallback`。

---

## 六、后端横向对比

| 维度 | CPU（纯 C#） | CUDA | MLX | GGML |
|---|---|---|---|---|
| 原生产物 | 无 | PTX（`-ptx`） | 预编译动态库 | 预编译动态库 `GgmlOps` |
| 构建工具 | — | `nvcc` | `cmake`/clang | `cmake` |
| 加载方式 | — | Driver API `cuModuleLoadData`（驱动 JIT） | P/Invoke `NativeLibrary` | P/Invoke + DllImport 解析器 |
| 调用粒度 | 逐元素 | 逐内核 `cuLaunchKernel` | 惰性图 + `eval` | 逐算子建小 ggml 图 |
| 矩阵乘 | SGEMM/OpenBLAS | cuBLAS | MLX matmul | ggml mul_mat |
| 量化矩阵乘 | `ManagedQuantizedOps` | `CudaQuantizedOps` | `MlxQuantizedOps` | `TSGgml_*Quant*` |
| 标志 | `cpu` | `cuda` | `mlx` | `ggml_cpu`/`ggml_metal`/`ggml_cuda` |

**共同点**（所有后端一致）：
1. 上层只调 `Ops.*`，由 `OpRegistry` 按 **Storage 类型**分派；
2. 算子用 `[RegisterOpStorageType("名", typeof(XxxStorage))]` 注册；
3. 未实现/不满足条件的算子**回退 CPU**，输出在所有后端保持一致。

**核心差异一句话**：CUDA = PTX + 运行时 JIT（Driver API）；GGML/MLX = 预编译库 + P/Invoke；
但**上层分派与 CPU 回退机制完全相同**。

---

## 七、相关源文件索引

| 关注点 | 文件 |
|---|---|
| 算子门面 / 注册 / 分派 | `TensorSharp.Core/Ops.cs`、`OpRegistry.cs`、`OpRegistryAttributes.cs` |
| CPU 计算引擎 | `TensorSharp.Core/TensorApplyCPU.cs`、`Cpu/CpuBasicOps.cs`、`Cpu/MatrixMultiplication.cs` |
| CUDA 分派/内核/加载 | `TensorSharp.Backends.Cuda/CudaBasicOps.cs`、`CudaKernelOps.cs`、`CudaKernels.cs`、`CudaModule.cs`、`Interop/CudaDriverApi.cs` |
| CUDA 原生内核 | `TensorSharp.Backends.Cuda/native/kernels/tensorsharp_kernels.cu` |
| GGML 分派/绑定/上下文 | `TensorSharp.Backends.GGML/GgmlBasicOps.cs`、`GgmlNative.cs`、`GgmlContext.cs`、`GgmlMemoryPool.cs` |
| GGML 原生算子 | `TensorSharp.GGML.Native/ggml_ops_*.cpp` |
| MLX 分派/绑定 | `TensorSharp.Backends.MLX/MlxBasicOps.cs`、`MlxNative.cs`、`MlxWorker.cs` |
