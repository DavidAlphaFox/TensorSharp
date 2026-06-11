using System;
using System.Collections.Generic;
using System.IO;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    internal sealed class CudaModule : IDisposable
    {
        private readonly Dictionary<string, IntPtr> functions = new Dictionary<string, IntPtr>(StringComparer.Ordinal);
        private IntPtr module;

        // 中文：私有构造函数，保存已加载的 CUDA 模块句柄。
        private CudaModule(IntPtr module)
        {
            this.module = module;
        }

        // 中文：从文件读取 PTX/cubin 字节并加载为 CUDA 模块。
        public static CudaModule LoadFromFile(string path)
        {
            if (path == null)
                throw new ArgumentNullException(nameof(path));

            byte[] bytes = File.ReadAllBytes(path);
            return LoadFromBytes(bytes);
        }

        // 中文：从内存字节加载 CUDA 模块，必要时补零终止符后调用 cuModuleLoadData。
        public static unsafe CudaModule LoadFromBytes(byte[] ptxBytes)
        {
            if (ptxBytes == null)
                throw new ArgumentNullException(nameof(ptxBytes));

            byte[] terminated = ptxBytes;
            if (terminated.Length == 0 || terminated[terminated.Length - 1] != 0)
            {
                terminated = new byte[ptxBytes.Length + 1];
                Buffer.BlockCopy(ptxBytes, 0, terminated, 0, ptxBytes.Length);
            }

            fixed (byte* ptx = terminated)
            {
                CudaDriverApi.cuModuleLoadData(out IntPtr module, (IntPtr)ptx).ThrowOnError();
                return new CudaModule(module);
            }
        }

        // 中文：按名称获取模块中的核函数句柄，结果缓存以避免重复查询。
        public IntPtr GetFunction(string name)
        {
            if (!functions.TryGetValue(name, out IntPtr function))
            {
                CudaDriverApi.cuModuleGetFunction(out function, module, name).ThrowOnError();
                functions.Add(name, function);
            }

            return function;
        }

        // 中文：卸载 CUDA 模块、清空函数缓存并置空句柄。
        public void Dispose()
        {
            IntPtr current = module;
            if (current != IntPtr.Zero)
            {
                module = IntPtr.Zero;
                functions.Clear();
                CudaDriverApi.cuModuleUnload(current);
            }
        }
    }
}
