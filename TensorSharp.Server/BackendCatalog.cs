using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Cuda;
using TensorSharp.GGML;
using TensorSharp.MLX;

namespace TensorSharp.Server
{
    internal sealed record BackendOption(string Value, string Label);

    internal static class BackendCatalog
    {
        // TensorSharp.Server should always expose the two CPU choices distinctly:
        // `ggml_cpu` is the native GGML CPU backend, while `cpu` is the pure C# backend.
        private static readonly BackendDescriptor[] BackendDescriptors =
        {
            new("mlx", "MLX Metal (GPU)", null, AlwaysAvailable: false),
            new("cuda", "CUDA (cuBLAS GPU)", null, AlwaysAvailable: false),
            new("ggml_metal", "GGML Metal (GPU)", GgmlBackendType.Metal, AlwaysAvailable: false),
            new("ggml_cuda", "GGML CUDA (GPU)", GgmlBackendType.Cuda, AlwaysAvailable: false),
            new("ggml_cpu", "GGML CPU", GgmlBackendType.Cpu, AlwaysAvailable: true),
            new("cpu", "CPU (Pure C#)", GgmlBackendType.Cpu, AlwaysAvailable: true),
        };

        // 中文：根据各后端可用性探测（MLX/CUDA/GGML）过滤出当前环境支持的后端选项列表。
        internal static IReadOnlyList<BackendOption> GetSupportedBackends(
            Func<GgmlBackendType, bool> isGgmlBackendAvailable = null,
            Func<bool> isCudaBackendAvailable = null,
            Func<bool> isMlxBackendAvailable = null)
        {
            isGgmlBackendAvailable ??= IsGgmlBackendAvailable;
            isCudaBackendAvailable ??= CudaBackend.IsAvailable;
            isMlxBackendAvailable ??= MlxBackend.IsAvailable;

            return BackendDescriptors
                .Where(descriptor => descriptor.AlwaysAvailable ||
                    (string.Equals(descriptor.Value, "mlx", StringComparison.OrdinalIgnoreCase)
                        ? isMlxBackendAvailable()
                        : descriptor.GgmlBackendType.HasValue
                        ? isGgmlBackendAvailable(descriptor.GgmlBackendType.Value)
                        : isCudaBackendAvailable()))
                .Select(descriptor => new BackendOption(descriptor.Value, descriptor.Label))
                .ToArray();
        }

        // 中文：解析默认后端——若配置后端受支持则采用，否则回退到首个受支持后端。
        internal static string ResolveDefaultBackend(string configuredBackend, IReadOnlyList<BackendOption> supportedBackends)
        {
            string canonicalBackend = Canonicalize(configuredBackend);
            if (!string.IsNullOrEmpty(canonicalBackend) &&
                supportedBackends.Any(backend => string.Equals(backend.Value, canonicalBackend, StringComparison.OrdinalIgnoreCase)))
            {
                return canonicalBackend;
            }

            return supportedBackends.FirstOrDefault()?.Value ?? canonicalBackend ?? configuredBackend;
        }

        // 中文：将后端名各种别名规范化为标准取值，空白返回 null，未知值原样小写返回。
        internal static string Canonicalize(string backend)
        {
            if (string.IsNullOrWhiteSpace(backend))
                return null;

            return backend.Trim().ToLowerInvariant() switch
            {
                "mlx" or "mlx_metal" or "mlx-metal" => "mlx",
                "cuda" or "direct_cuda" or "direct-cuda" => "cuda",
                "ggml_cuda" or "ggml-cuda" => "ggml_cuda",
                "ggml_metal" => "ggml_metal",
                "ggml_cpu" => "ggml_cpu",
                "cpu" => "cpu",
                var value => value,
            };
        }

        // 中文：将 BackendType 枚举映射为对应的后端字符串取值。
        internal static string ToBackendValue(BackendType backendType)
        {
            return backendType switch
            {
                BackendType.Mlx => "mlx",
                BackendType.Cuda => "cuda",
                BackendType.GgmlMetal => "ggml_metal",
                BackendType.GgmlCuda => "ggml_cuda",
                BackendType.GgmlCpu => "ggml_cpu",
                BackendType.Cpu => "cpu",
                _ => null,
            };
        }

        // 中文：轻量探测指定 GGML 后端是否可用（仅做编译标志/平台检查，不真正初始化设备），异常时视为不可用。
        private static bool IsGgmlBackendAvailable(GgmlBackendType backendType)
        {
            try
            {
                // Backend discovery runs at web-app startup, so it must not spin up
                // any GGML device — otherwise picking a non-GGML backend (MLX,
                // direct CUDA) would still trigger `ggml_metal_device_init` / etc.
                // logs at startup. CanInitializeBackend is a lightweight compile-flag
                // + platform check; the real GGML init is deferred until a GGML
                // backend is actually selected.
                return GgmlBasicOps.CanInitializeBackend(backendType);
            }
            catch
            {
                return false;
            }
        }

        private sealed record BackendDescriptor(string Value, string Label, GgmlBackendType? GgmlBackendType, bool AlwaysAvailable);
    }
}



