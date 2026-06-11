using System.Reflection;

namespace TensorSharp.Cuda
{
    public static class CudaBackend
    {
        private static int registered;

        // 中文：将当前程序集中的 CUDA 算子注册到 OpRegistry（线程安全，仅执行一次）。
        public static void Register()
        {
            if (System.Threading.Interlocked.Exchange(ref registered, 1) != 0)
                return;

            OpRegistry.RegisterAssembly(Assembly.GetExecutingAssembly());
        }

        // 中文：返回当前系统是否存在可用的 CUDA 设备。
        public static bool IsAvailable() => CudaDevice.IsAvailable();
    }
}
