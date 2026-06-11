using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;

namespace TensorSharp.Cuda.Interop
{
    internal static class CudaLibraryResolver
    {
        private static int registered;

        // 中文：注册本程序集的原生库解析器（只执行一次），并在 Windows 上补充 CUDA 路径
        public static void Register()
        {
            if (Interlocked.Exchange(ref registered, 1) != 0)
                return;

            NativeLibrary.SetDllImportResolver(typeof(CudaLibraryResolver).Assembly, Resolve);
            EnsureWindowsCudaPath();
        }

        // 中文：自定义 DllImport 解析逻辑，按平台尝试加载对应的 cuda 驱动与 cublas 库
        private static IntPtr Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
        {
            if (libraryName == "cuda")
            {
                string driverName = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
                if (NativeLibrary.TryLoad(driverName, out IntPtr cudaHandle))
                    return cudaHandle;
            }

            if (libraryName == "cublas")
            {
                foreach (string candidate in GetCublasCandidates())
                {
                    if (NativeLibrary.TryLoad(candidate, out IntPtr cublasHandle))
                        return cublasHandle;
                }
            }

            return IntPtr.Zero;
        }

        // 中文：按平台与优先级返回 cuBLAS 库文件的候选名称列表
        private static IEnumerable<string> GetCublasCandidates()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                yield return "cublas64_13.dll";
                yield return "cublas64_12.dll";
                yield return "cublas64_11.dll";
                yield break;
            }

            // Prefer versioned runtime libraries; unversioned symlinks can point
            // at stubs or a different toolkit than the native GGML bridge uses.
            yield return "libcublas.so.12";
            yield return "libcublas.so.13";
            yield return "libcublas.so.11";
            yield return "libcublas.so";
        }

        // 中文：在 Windows 上将 CUDA 的 bin 目录追加到 PATH 环境变量，便于加载原生库
        private static void EnsureWindowsCudaPath()
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return;

            string currentPath = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
            var existing = new HashSet<string>(
                currentPath.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries),
                StringComparer.OrdinalIgnoreCase);

            string[] additions = EnumerateCudaBinDirectories()
                .Where(path => Directory.Exists(path) && !existing.Contains(path))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToArray();

            if (additions.Length == 0)
                return;

            Environment.SetEnvironmentVariable("PATH", string.Join(Path.PathSeparator, additions.Concat(new[] { currentPath })));
        }

        // 中文：枚举可能的 CUDA bin 目录（来自环境变量及默认安装路径）
        private static IEnumerable<string> EnumerateCudaBinDirectories()
        {
            foreach (string variableName in new[] { "CUDA_PATH", "CUDA_HOME" })
            {
                string root = Environment.GetEnvironmentVariable(variableName);
                if (!string.IsNullOrWhiteSpace(root))
                    yield return Path.Combine(root, "bin");
            }

            string programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
            string cudaRoot = Path.Combine(programFiles, "NVIDIA GPU Computing Toolkit", "CUDA");
            if (!Directory.Exists(cudaRoot))
                yield break;

            foreach (string versionDir in Directory.EnumerateDirectories(cudaRoot, "v*").OrderByDescending(path => path))
                yield return Path.Combine(versionDir, "bin");
        }
    }
}
