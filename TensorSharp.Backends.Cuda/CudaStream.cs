using System;
using System.Threading;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed class CudaStream : IDisposable
    {
        private IntPtr stream;

        // 中文：私有构造函数，保存底层 CUDA 流句柄。
        private CudaStream(IntPtr stream)
        {
            this.stream = stream;
        }

        public IntPtr Handle => stream;

        // 中文：创建一个新的 CUDA 流并封装为 CudaStream。
        public static CudaStream Create()
        {
            CudaDriverApi.cuStreamCreate(out IntPtr stream, 0).ThrowOnError();
            return new CudaStream(stream);
        }

        // 中文：阻塞等待该流上所有操作完成，已释放则抛异常。
        public void Synchronize()
        {
            if (stream == IntPtr.Zero)
                throw new ObjectDisposedException(nameof(CudaStream));

            CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();
        }

        // 中文：销毁 CUDA 流（仅执行一次）。
        public void Dispose()
        {
            IntPtr handle = Interlocked.Exchange(ref stream, IntPtr.Zero);
            if (handle != IntPtr.Zero)
                CudaDriverApi.cuStreamDestroy(handle);
        }
    }
}
