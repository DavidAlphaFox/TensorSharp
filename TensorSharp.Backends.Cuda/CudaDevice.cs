using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed class CudaDevice
    {
        // 中文：私有构造函数，保存设备序号、名称、总显存、计算能力及 SM 数量等属性。
        private CudaDevice(int ordinal, string name, long totalMemoryBytes, int ccMajor, int ccMinor, int multiprocessorCount)
        {
            Ordinal = ordinal;
            Name = name;
            TotalMemoryBytes = totalMemoryBytes;
            ComputeCapabilityMajor = ccMajor;
            ComputeCapabilityMinor = ccMinor;
            MultiprocessorCount = multiprocessorCount;
        }

        public int Ordinal { get; }
        public string Name { get; }
        public long TotalMemoryBytes { get; }
        public int ComputeCapabilityMajor { get; }
        public int ComputeCapabilityMinor { get; }
        public int MultiprocessorCount { get; }

        public string TotalMemoryFormatted => $"{TotalMemoryBytes / (1024.0 * 1024 * 1024):F1} GB";
        public string ComputeCapability => $"{ComputeCapabilityMajor}.{ComputeCapabilityMinor}";

        // 中文：检测系统是否可用 CUDA：尝试加载驱动库并探测设备数量，任何异常视为不可用。
        public static bool IsAvailable()
        {
            try
            {
                string driverName = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
                if (!NativeLibrary.TryLoad(driverName, out IntPtr handle))
                    return false;

                NativeLibrary.Free(handle);
                return ProbeDeviceCount() > 0;
            }
            catch
            {
                return false;
            }
        }

        // 中文：初始化驱动并返回可用 CUDA 设备数量。
        public static int GetDeviceCount()
        {
            CudaLibraryResolver.Register();
            CudaDriverApi.cuInit(0).ThrowOnError();
            CudaDriverApi.cuDeviceGetCount(out int count).ThrowOnError();
            return count;
        }

        // 中文：查询指定序号设备的名称、总显存、计算能力与 SM 数量，构造 CudaDevice 对象。
        public static CudaDevice GetDevice(int ordinal)
        {
            CudaLibraryResolver.Register();
            CudaDriverApi.cuInit(0).ThrowOnError();
            CudaDriverApi.cuDeviceGet(out int device, ordinal).ThrowOnError();

            byte[] nameBuffer = new byte[256];
            CudaDriverApi.cuDeviceGetName(nameBuffer, nameBuffer.Length, device).ThrowOnError();
            int nullIndex = Array.IndexOf(nameBuffer, (byte)0);
            string name = Encoding.ASCII.GetString(nameBuffer, 0, nullIndex >= 0 ? nullIndex : nameBuffer.Length).Trim();

            CudaDriverApi.cuDeviceTotalMem(out UIntPtr totalMemory, device).ThrowOnError();
            CudaDriverApi.cuDeviceGetAttribute(out int ccMajor, CudaDriverApi.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device).ThrowOnError();
            CudaDriverApi.cuDeviceGetAttribute(out int ccMinor, CudaDriverApi.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device).ThrowOnError();
            CudaDriverApi.cuDeviceGetAttribute(out int smCount, CudaDriverApi.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device).ThrowOnError();

            return new CudaDevice(ordinal, name, checked((long)totalMemory.ToUInt64()), ccMajor, ccMinor, smCount);
        }

        // 中文：返回设备的可读描述字符串（序号、名称、显存、计算能力、SM 数）。
        public override string ToString()
        {
            return $"GPU {Ordinal}: {Name} ({TotalMemoryFormatted}, sm_{ComputeCapabilityMajor}{ComputeCapabilityMinor}, {MultiprocessorCount} SMs)";
        }

        // 中文：禁止内联地探测设备数量（隔离驱动调用以便 IsAvailable 安全捕获异常）。
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static int ProbeDeviceCount()
        {
            CudaLibraryResolver.Register();
            CudaDriverApi.cuInit(0).ThrowOnError();
            CudaDriverApi.cuDeviceGetCount(out int count).ThrowOnError();
            return count;
        }
    }
}
