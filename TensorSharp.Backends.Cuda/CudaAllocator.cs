using System;
using System.Collections.Generic;
using System.Threading;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    [Serializable]
    public sealed class CudaAllocator : IAllocator, IDisposable
    {
        private int disposed;
        private readonly object poolSync = new object();
        private readonly Dictionary<long, Stack<IntPtr>> devicePool = new Dictionary<long, Stack<IntPtr>>();
        private readonly long maxCachedBytes;
        private readonly long maxCachedBlockBytes;
        private long cachedBytes;
        private bool poolEnabled;

        // 中文：构造函数，注册 CUDA 后端并初始化指定设备的上下文、流、cuBLAS 句柄与内核，配置内存池参数。
        public CudaAllocator(int deviceId = 0)
        {
            CudaBackend.Register();
            poolEnabled = !string.Equals(Environment.GetEnvironmentVariable("TENSORSHARP_CUDA_POOL"), "0", StringComparison.Ordinal);
            maxCachedBytes = ReadPoolLimit("TENSORSHARP_CUDA_POOL_MAX_MB", 512L) * 1024L * 1024L;
            maxCachedBlockBytes = ReadPoolLimit("TENSORSHARP_CUDA_POOL_MAX_BLOCK_MB", 256L) * 1024L * 1024L;

            CudaContext context = null;
            CudaStream stream = null;
            CudaCublasHandle blas = null;
            CudaKernels kernels = null;

            try
            {
                context = CudaContext.Create(deviceId);
                context.MakeCurrent();
                stream = CudaStream.Create();
                blas = CudaCublasHandle.Create();
                blas.SetStream(stream.Handle);
                kernels = CudaKernels.TryCreate();
            }
            catch
            {
                kernels?.Dispose();
                blas?.Dispose();
                stream?.Dispose();
                context?.Dispose();
                throw;
            }

            Context = context;
            Stream = stream;
            Blas = blas;
            Kernels = kernels;
            DeviceId = deviceId;
        }

        public BlasEnum BlasEnum => BlasEnum.CUDA;

        public int DeviceId { get; }

        internal CudaContext Context { get; }

        internal CudaStream Stream { get; }

        internal CudaCublasHandle Blas { get; }

        internal CudaKernels Kernels { get; }

        // 中文：分配一块指定元素类型与数量的 CUDA 存储。
        public Storage Allocate(DType elementType, long elementCount)
        {
            return new CudaStorage(this, elementType, elementCount);
        }

        // 中文：从内存池借用设备显存，命中则复用，否则按对齐大小调用 cuMemAlloc 新分配。
        internal IntPtr RentDeviceMemory(long requestedBytes, out long allocationBytes)
        {
            ThrowIfDisposed();
            allocationBytes = RoundAllocationSize(Math.Max(requestedBytes, 1));

            if (poolEnabled)
            {
                lock (poolSync)
                {
                    if (devicePool.TryGetValue(allocationBytes, out Stack<IntPtr> stack) && stack.Count > 0)
                    {
                        cachedBytes -= allocationBytes;
                        return stack.Pop();
                    }
                }
            }

            Context.MakeCurrent();
            CudaDriverApi.cuMemAlloc(out IntPtr ptr, new UIntPtr((ulong)allocationBytes)).ThrowOnError();
            return ptr;
        }

        // 中文：归还设备显存，符合池化条件则缓存以备复用，否则调用 cuMemFree 释放。
        internal void ReturnDeviceMemory(IntPtr ptr, long allocationBytes)
        {
            if (ptr == IntPtr.Zero)
                return;

            if (poolEnabled && allocationBytes > 0 && allocationBytes <= maxCachedBlockBytes)
            {
                lock (poolSync)
                {
                    if (cachedBytes + allocationBytes <= maxCachedBytes)
                    {
                        if (!devicePool.TryGetValue(allocationBytes, out Stack<IntPtr> stack))
                        {
                            stack = new Stack<IntPtr>();
                            devicePool[allocationBytes] = stack;
                        }

                        stack.Push(ptr);
                        cachedBytes += allocationBytes;
                        return;
                    }
                }
            }

            Context.MakeCurrent();
            CudaDriverApi.cuMemFree(ptr);
        }

        // 中文：查询设备显存占用比例（已用显存 / 总显存）。
        public float GetAllocatedMemoryRatio()
        {
            Context.MakeCurrent();
            CudaDriverApi.cuMemGetInfo(out UIntPtr free, out UIntPtr total).ThrowOnError();
            ulong totalBytes = total.ToUInt64();
            if (totalBytes == 0)
                return 0.0f;

            ulong freeBytes = free.ToUInt64();
            return (float)(1.0 - (double)freeBytes / totalBytes);
        }

        // 中文：使当前上下文生效并同步流，等待所有已提交操作完成。
        public void Synchronize()
        {
            Context.MakeCurrent();
            Stream.Synchronize();
        }

        // 中文：释放分配器，同步流、清空缓存显存并销毁内核、cuBLAS、流与上下文（仅执行一次）。
        public void Dispose()
        {
            if (Interlocked.Exchange(ref disposed, 1) != 0)
                return;

            Context.MakeCurrent();
            Stream.Synchronize();
            FreeCachedDeviceMemory();
            Kernels?.Dispose();
            Blas.Dispose();
            Stream.Dispose();
            Context.Dispose();
        }

        // 中文：释放内存池中所有缓存的设备显存块并清零计数。
        private void FreeCachedDeviceMemory()
        {
            lock (poolSync)
            {
                foreach (Stack<IntPtr> stack in devicePool.Values)
                {
                    while (stack.Count > 0)
                    {
                        IntPtr ptr = stack.Pop();
                        if (ptr != IntPtr.Zero)
                            CudaDriverApi.cuMemFree(ptr);
                    }
                }

                devicePool.Clear();
                cachedBytes = 0;
            }
        }

        // 中文：若分配器已释放则抛出 ObjectDisposedException。
        private void ThrowIfDisposed()
        {
            if (Volatile.Read(ref disposed) != 0)
                throw new ObjectDisposedException(nameof(CudaAllocator));
        }

        // 中文：将请求字节数对齐到分配粒度（小块向上取 2 的幂，大块按 1MB 对齐）以提高池复用率。
        private static long RoundAllocationSize(long bytes)
        {
            const long smallMax = 1L << 20;
            if (bytes <= 256)
                return 256;

            if (bytes <= smallMax)
            {
                long size = 256;
                while (size < bytes)
                    size <<= 1;
                return size;
            }

            const long largeAlignment = 1L << 20;
            return ((bytes + largeAlignment - 1) / largeAlignment) * largeAlignment;
        }

        // 中文：从环境变量读取内存池上限（MB），解析失败则返回默认值。
        private static long ReadPoolLimit(string name, long defaultMb)
        {
            string value = Environment.GetEnvironmentVariable(name);
            if (long.TryParse(value, out long parsed) && parsed >= 0)
                return parsed;
            return defaultMb;
        }
    }
}
