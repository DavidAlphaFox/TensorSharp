using System;

namespace TensorSharp.MLX
{
    [Serializable]
    public sealed class MlxAllocator : IAllocator, IDisposable
    {
        private bool disposed;

        // 中文：构造 MLX 分配器，注册后端、确保指定设备可用并记录设备 ID。
        public MlxAllocator(int deviceId = 0)
        {
            MlxBackend.Register();
            MlxBackend.EnsureAvailable(deviceId);
            DeviceId = deviceId;
        }

        public BlasEnum BlasEnum => BlasEnum.MLX;

        public int DeviceId { get; }

        // 中文：分配指定元素类型与数量的 MLX 存储；已释放则抛异常。
        public Storage Allocate(DType elementType, long elementCount)
        {
            ThrowIfDisposed();
            return new MlxStorage(this, elementType, elementCount);
        }

        // 中文：返回已分配内存占比，MLX 统一内存下恒为 0。
        public float GetAllocatedMemoryRatio()
        {
            return 0.0f;
        }

        // 中文：获取当前 MLX 显存快照（委托给后端）。
        public MlxMemorySnapshot GetMemorySnapshot()
        {
            return MlxBackend.GetMemorySnapshot();
        }

        // 中文：若分配器已释放则抛出 ObjectDisposedException。
        private void ThrowIfDisposed()
        {
            if (disposed)
                throw new ObjectDisposedException(nameof(MlxAllocator));
        }

        // 中文：幂等释放，清理本设备的量化算子缓存并清空后端缓存。
        public void Dispose()
        {
            if (disposed)
                return;

            disposed = true;
            MlxQuantizedOps.ClearDeviceCache(DeviceId);
            MlxBackend.ClearCache();
        }
    }
}
