using System;
using System.Threading;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed class CudaCublasHandle : IDisposable
    {
        private IntPtr handle;

        // 中文：私有构造函数，保存底层 cuBLAS 句柄。
        private CudaCublasHandle(IntPtr handle)
        {
            this.handle = handle;
        }

        public IntPtr Handle => handle;

        // 中文：创建 cuBLAS 句柄并启用 Tensor Core 数学模式，失败时销毁句柄。
        public static CudaCublasHandle Create()
        {
            CublasApi.cublasCreate(out IntPtr handle).ThrowOnCublasError();
            try
            {
                CublasApi.cublasSetMathMode(handle, CublasApi.CUBLAS_TENSOR_OP_MATH).ThrowOnCublasError();
                return new CudaCublasHandle(handle);
            }
            catch
            {
                CublasApi.cublasDestroy(handle);
                throw;
            }
        }

        // 中文：将 cuBLAS 句柄绑定到指定 CUDA 流，已释放则抛异常。
        public void SetStream(IntPtr stream)
        {
            if (handle == IntPtr.Zero)
                throw new ObjectDisposedException(nameof(CudaCublasHandle));

            CublasApi.cublasSetStream(handle, stream).ThrowOnCublasError();
        }

        // 中文：销毁 cuBLAS 句柄（仅执行一次）。
        public void Dispose()
        {
            IntPtr nativeHandle = Interlocked.Exchange(ref handle, IntPtr.Zero);
            if (nativeHandle != IntPtr.Zero)
                CublasApi.cublasDestroy(nativeHandle);
        }
    }
}
