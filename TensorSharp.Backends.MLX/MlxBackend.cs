using System;
using System.Reflection;
using System.Threading;

namespace TensorSharp.MLX
{
    public static class MlxBackend
    {
        private static int registered;

        // 中文：返回 MLX 原生库是否已加载可用。
        public static bool IsAvailable()
        {
            return MlxNative.IsAvailable();
        }

        // 中文：注册本程序集算子与 MLX 回退算子，原子标志保证只执行一次。
        public static void Register()
        {
            if (Interlocked.Exchange(ref registered, 1) != 0)
                return;

            OpRegistry.RegisterAssembly(Assembly.GetExecutingAssembly());
            MlxFallbackOps.Register();
        }

        // 中文：确保 MLX 后端可用并初始化指定 GPU 设备，不可用时抛出带安装指引的异常。
        public static void EnsureAvailable(int deviceId = 0)
        {
            if (!IsAvailable())
            {
                throw new PlatformNotSupportedException(
                    "MLX backend is not available. Install or copy libmlxc into the app directory, " +
                    "set TENSORSHARP_MLX_LIBRARY to the libmlxc path, or set TENSORSHARP_MLX_LIBRARY_DIR to a directory containing libmlxc.");
            }

            MlxNative.EnsureGpuDevice(deviceId);
        }

        // 中文：获取当前 MLX 显存快照（委托给原生层）。
        public static MlxMemorySnapshot GetMemorySnapshot()
        {
            return MlxNative.GetMemorySnapshot();
        }

        // 中文：清空 MLX 原生侧的内存缓存。
        public static void ClearCache()
        {
            MlxNative.ClearCache();
        }
    }
}
