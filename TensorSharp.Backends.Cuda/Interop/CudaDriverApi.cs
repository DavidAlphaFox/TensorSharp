using System;
using System.Runtime.InteropServices;

namespace TensorSharp.Cuda.Interop
{
    internal static class CudaDriverApi
    {
        private const string LibName = "cuda";

        // 中文：绑定 cuInit，初始化 CUDA Driver API（使用前必须调用一次）
        [DllImport(LibName)]
        public static extern int cuInit(uint flags);

        // 中文：绑定 cuDeviceGet，按序号获取对应的 CUDA 设备句柄
        [DllImport(LibName)]
        public static extern int cuDeviceGet(out int device, int ordinal);

        // 中文：绑定 cuDeviceGetCount，获取系统中可用 CUDA 设备的数量
        [DllImport(LibName)]
        public static extern int cuDeviceGetCount(out int count);

        // 中文：绑定 cuDeviceGetName，获取指定设备的名称字符串
        [DllImport(LibName)]
        public static extern int cuDeviceGetName(byte[] name, int len, int device);

        // 中文：绑定 cuDeviceTotalMem_v2，查询指定设备的显存总容量（字节）
        [DllImport(LibName, EntryPoint = "cuDeviceTotalMem_v2")]
        public static extern int cuDeviceTotalMem(out UIntPtr bytes, int device);

        // 中文：绑定 cuMemGetInfo_v2，查询当前设备的空闲显存与总显存大小
        [DllImport(LibName, EntryPoint = "cuMemGetInfo_v2")]
        public static extern int cuMemGetInfo(out UIntPtr free, out UIntPtr total);

        // 中文：绑定 cuDeviceGetAttribute，查询设备的某项属性（如计算能力、SM 数量等）
        [DllImport(LibName)]
        public static extern int cuDeviceGetAttribute(out int value, int attribute, int device);

        // 中文：绑定 cuCtxCreate_v2，为指定设备创建一个 CUDA 上下文并设为当前
        [DllImport(LibName, EntryPoint = "cuCtxCreate_v2")]
        public static extern int cuCtxCreate(out IntPtr ctx, uint flags, int device);

        // 中文：绑定 cuCtxDestroy_v2，销毁指定的 CUDA 上下文并释放其资源
        [DllImport(LibName, EntryPoint = "cuCtxDestroy_v2")]
        public static extern int cuCtxDestroy(IntPtr ctx);

        // 中文：绑定 cuCtxSetCurrent，将指定上下文绑定为当前线程的活动 CUDA 上下文
        [DllImport(LibName)]
        public static extern int cuCtxSetCurrent(IntPtr ctx);

        // 中文：绑定 cuCtxGetCurrent，获取当前线程正在使用的 CUDA 上下文
        [DllImport(LibName)]
        public static extern int cuCtxGetCurrent(out IntPtr ctx);

        // 中文：绑定 cuDevicePrimaryCtxRetain，获取并保留指定设备的主上下文（引用计数加一）
        [DllImport(LibName)]
        public static extern int cuDevicePrimaryCtxRetain(out IntPtr ctx, int device);

        // 中文：绑定 cuDevicePrimaryCtxRelease，释放指定设备主上下文的引用（引用计数减一）
        [DllImport(LibName)]
        public static extern int cuDevicePrimaryCtxRelease(int device);

        // 中文：绑定 cuMemAlloc_v2，在设备显存上分配指定字节数的内存
        [DllImport(LibName, EntryPoint = "cuMemAlloc_v2")]
        public static extern int cuMemAlloc(out IntPtr devicePtr, UIntPtr byteSize);

        // 中文：绑定 cuMemFree_v2，释放先前在设备显存上分配的内存
        [DllImport(LibName, EntryPoint = "cuMemFree_v2")]
        public static extern int cuMemFree(IntPtr devicePtr);

        // 中文：绑定 cuMemcpyHtoD_v2，将数据从主机内存同步拷贝到设备显存
        [DllImport(LibName, EntryPoint = "cuMemcpyHtoD_v2")]
        public static extern int cuMemcpyHtoD(IntPtr dstDevice, IntPtr srcHost, UIntPtr byteCount);

        // 中文：绑定 cuMemcpyHtoDAsync_v2，将数据从主机内存异步（指定流）拷贝到设备显存
        [DllImport(LibName, EntryPoint = "cuMemcpyHtoDAsync_v2")]
        public static extern int cuMemcpyHtoDAsync(IntPtr dstDevice, IntPtr srcHost, UIntPtr byteCount, IntPtr stream);

        // 中文：绑定 cuMemcpyDtoH_v2，将数据从设备显存同步拷贝回主机内存
        [DllImport(LibName, EntryPoint = "cuMemcpyDtoH_v2")]
        public static extern int cuMemcpyDtoH(IntPtr dstHost, IntPtr srcDevice, UIntPtr byteCount);

        // 中文：绑定 cuMemcpyDtoHAsync_v2，将数据从设备显存异步（指定流）拷贝回主机内存
        [DllImport(LibName, EntryPoint = "cuMemcpyDtoHAsync_v2")]
        public static extern int cuMemcpyDtoHAsync(IntPtr dstHost, IntPtr srcDevice, UIntPtr byteCount, IntPtr stream);

        // 中文：绑定 cuMemcpyDtoD_v2，在设备显存之间同步拷贝数据
        [DllImport(LibName, EntryPoint = "cuMemcpyDtoD_v2")]
        public static extern int cuMemcpyDtoD(IntPtr dstDevice, IntPtr srcDevice, UIntPtr byteCount);

        // 中文：绑定 cuMemcpyDtoDAsync_v2，在设备显存之间异步（指定流）拷贝数据
        [DllImport(LibName, EntryPoint = "cuMemcpyDtoDAsync_v2")]
        public static extern int cuMemcpyDtoDAsync(IntPtr dstDevice, IntPtr srcDevice, UIntPtr byteCount, IntPtr stream);

        // 中文：绑定 cuMemsetD8_v2，将设备显存按字节填充为指定的 8 位值
        [DllImport(LibName, EntryPoint = "cuMemsetD8_v2")]
        public static extern int cuMemsetD8(IntPtr dstDevice, byte value, UIntPtr count);

        // 中文：绑定 cuModuleLoadData，从内存中的 PTX/cubin 镜像加载 CUDA 模块
        [DllImport(LibName)]
        public static extern int cuModuleLoadData(out IntPtr module, IntPtr image);

        // 中文：绑定 cuModuleGetFunction，按名称从已加载模块中获取内核函数句柄
        [DllImport(LibName)]
        public static extern int cuModuleGetFunction(out IntPtr function, IntPtr module, string name);

        // 中文：绑定 cuModuleUnload，卸载已加载的 CUDA 模块并释放资源
        [DllImport(LibName)]
        public static extern int cuModuleUnload(IntPtr module);

        // 中文：绑定 cuLaunchKernel，按指定网格/线程块维度在 GPU 上启动内核函数
        [DllImport(LibName)]
        public static extern int cuLaunchKernel(
            IntPtr function,
            uint gridDimX,
            uint gridDimY,
            uint gridDimZ,
            uint blockDimX,
            uint blockDimY,
            uint blockDimZ,
            uint sharedMemBytes,
            IntPtr stream,
            IntPtr kernelParams,
            IntPtr extra);

        // 中文：绑定 cuStreamCreate，创建一个新的 CUDA 流用于异步操作排队
        [DllImport(LibName)]
        public static extern int cuStreamCreate(out IntPtr stream, uint flags);

        // 中文：绑定 cuStreamDestroy_v2，销毁指定的 CUDA 流
        [DllImport(LibName, EntryPoint = "cuStreamDestroy_v2")]
        public static extern int cuStreamDestroy(IntPtr stream);

        // 中文：绑定 cuStreamSynchronize，阻塞等待指定流中的所有任务执行完成
        [DllImport(LibName)]
        public static extern int cuStreamSynchronize(IntPtr stream);

        // 中文：绑定 cuGetErrorString，将 CUDA 错误码转换为可读的错误描述字符串
        [DllImport(LibName)]
        public static extern int cuGetErrorString(int error, out IntPtr str);

        public const int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
        public const int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;
        public const int CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;
    }
}
