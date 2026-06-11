using System;
using System.Threading;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed class CudaContext : IDisposable
    {
        private IntPtr context;
        private readonly int device;

        // 中文：私有构造函数，保存底层上下文句柄、逻辑设备 ID 与驱动设备号。
        private CudaContext(IntPtr context, int deviceId, int device)
        {
            this.context = context;
            DeviceId = deviceId;
            this.device = device;
        }

        public int DeviceId { get; }

        public IntPtr Handle => context;

        // 中文：初始化驱动并为指定设备保留主上下文（primary context）后设为当前，创建 CudaContext 实例。
        public static CudaContext Create(int deviceId)
        {
            CudaLibraryResolver.Register();
            CudaDriverApi.cuInit(0).ThrowOnError();
            CudaDriverApi.cuDeviceGet(out int device, deviceId).ThrowOnError();

            // cuBLAS and the CUDA runtime use the device primary context. Retaining
            // it keeps the direct backend compatible with GGML CUDA probing and WSL.
            CudaDriverApi.cuDevicePrimaryCtxRetain(out IntPtr context, device).ThrowOnError();

            try
            {
                CudaDriverApi.cuCtxSetCurrent(context).ThrowOnError();
            }
            catch
            {
                CudaDriverApi.cuDevicePrimaryCtxRelease(device);
                throw;
            }

            return new CudaContext(context, deviceId, device);
        }

        // 中文：将本上下文设为当前线程的活动 CUDA 上下文，已释放则抛异常。
        public void MakeCurrent()
        {
            if (context == IntPtr.Zero)
                throw new ObjectDisposedException(nameof(CudaContext));

            CudaDriverApi.cuCtxSetCurrent(context).ThrowOnError();
        }

        // 中文：释放上下文，若其为当前上下文则先清空，再释放设备主上下文（仅执行一次）。
        public void Dispose()
        {
            IntPtr handle = Interlocked.Exchange(ref context, IntPtr.Zero);
            if (handle == IntPtr.Zero)
                return;

            if (CudaDriverApi.cuCtxGetCurrent(out IntPtr current) == 0 && current == handle)
                CudaDriverApi.cuCtxSetCurrent(IntPtr.Zero);

            CudaDriverApi.cuDevicePrimaryCtxRelease(device);
        }
    }
}
