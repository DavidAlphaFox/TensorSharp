using System;
using System.Runtime.InteropServices;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    [Serializable]
    public sealed unsafe class CudaStorage : Storage
    {
        private readonly object sync = new object();
        private IntPtr hostBuffer;
        private IntPtr deviceBuffer;
        private long deviceAllocationBytes;
        private bool hostDirty;
        private bool deviceDirty;

        // 中文：构造函数，校验参数并通过分配器在 CUDA 设备上租用一块设备显存。
        public CudaStorage(CudaAllocator allocator, DType elementType, long elementCount)
            : base(allocator, elementType, elementCount)
        {
            AllocatorImpl = allocator ?? throw new ArgumentNullException(nameof(allocator));
            if (ByteLength < 0)
                throw new ArgumentOutOfRangeException(nameof(elementCount));

            AllocatorImpl.Context.MakeCurrent();
            deviceBuffer = AllocatorImpl.RentDeviceMemory(ByteLength, out deviceAllocationBytes);
        }

        internal CudaAllocator AllocatorImpl { get; }

        internal IntPtr DeviceBuffer => deviceBuffer;

        public int DeviceId => AllocatorImpl.DeviceId;

        // 中文：销毁存储，线程安全地归还设备显存并释放主机镜像缓冲区。
        protected override void Destroy()
        {
            lock (sync)
            {
                if (deviceBuffer != IntPtr.Zero)
                {
                    AllocatorImpl.Context.MakeCurrent();
                    AllocatorImpl.ReturnDeviceMemory(deviceBuffer, deviceAllocationBytes);
                    deviceBuffer = IntPtr.Zero;
                    deviceAllocationBytes = 0;
                }

                if (hostBuffer != IntPtr.Zero)
                {
                    NativeMemory.AlignedFree(hostBuffer.ToPointer());
                    hostBuffer = IntPtr.Zero;
                }
            }
        }

        // 中文：返回存储位置描述，格式为 "CUDA:设备号"。
        public override string LocationDescription()
        {
            return $"CUDA:{DeviceId}";
        }

        // 中文：返回指定元素的主机指针，先从设备同步数据并标记主机已写脏，供原始指针读写。
        public override IntPtr PtrAtElement(long index)
        {
            ThrowIfDisposed();
            ValidateElementRange(index, 0);

            // Existing TensorSharp model code may mutate through raw pointers. Treat
            // every pointer checkout as a possible host-side write so the next CUDA
            // dispatch refreshes device memory before using this storage as input.
            SyncHostFromDevice();
            hostDirty = true;
            return HostPtrAtElementUnchecked(index);
        }

        // 中文：返回指定元素在设备显存中的指针（按字节偏移定位）。
        internal IntPtr DevicePtrAtElement(long index)
        {
            ThrowIfDisposed();
            ValidateElementRange(index, 0);
            return AddBytes(deviceBuffer, checked(index * ElementType.Size()));
        }

        // 中文：若主机数据为脏，则异步将主机缓冲区拷贝到设备，确保设备显存为最新。
        internal void EnsureDeviceCurrent()
        {
            ThrowIfDisposed();
            if (ByteLength == 0)
                return;

            lock (sync)
            {
                if (!hostDirty)
                    return;

                AllocatorImpl.Context.MakeCurrent();
                CudaDriverApi.cuMemcpyHtoDAsync(
                    deviceBuffer,
                    hostBuffer,
                    new UIntPtr((ulong)ByteLength),
                    AllocatorImpl.Stream.Handle).ThrowOnError();
                hostDirty = false;
                deviceDirty = false;
            }
        }

        // 中文：标记设备数据已被内核修改（设置 deviceDirty、清除 hostDirty）。
        internal void MarkDeviceModified()
        {
            ThrowIfDisposed();
            deviceDirty = true;
            hostDirty = false;
        }

        // 中文：若设备数据为脏，则将设备显存拷回主机镜像并同步流，确保主机数据为最新。
        internal void SyncHostFromDevice()
        {
            ThrowIfDisposed();
            if (ByteLength == 0)
                return;

            lock (sync)
            {
                if (!deviceDirty)
                    return;

                EnsureHostBuffer();
                AllocatorImpl.Context.MakeCurrent();
                CudaDriverApi.cuMemcpyDtoHAsync(
                    hostBuffer,
                    deviceBuffer,
                    new UIntPtr((ulong)ByteLength),
                    AllocatorImpl.Stream.Handle).ThrowOnError();
                AllocatorImpl.Stream.Synchronize();
                deviceDirty = false;
                hostDirty = false;
            }
        }

        // 中文：从源存储整体拷贝设备显存，要求两者字节长度相等。
        internal void CopyDeviceFrom(CudaStorage src)
        {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (src.ByteLength != ByteLength)
                throw new ArgumentException("CUDA device copy requires equal byte lengths.", nameof(src));

            CopyDeviceFrom(src, 0, 0, ByteLength);
        }

        // 中文：按字节偏移与长度做设备到设备拷贝，同分配器走异步流拷贝、跨分配器先同步再同步拷贝。
        internal void CopyDeviceFrom(CudaStorage src, long destinationByteOffset, long sourceByteOffset, long byteCount)
        {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (destinationByteOffset < 0 || sourceByteOffset < 0 || byteCount < 0 ||
                destinationByteOffset + byteCount > ByteLength ||
                sourceByteOffset + byteCount > src.ByteLength)
            {
                throw new ArgumentOutOfRangeException(nameof(byteCount));
            }

            if (byteCount == 0)
                return;

            src.EnsureDeviceCurrent();
            AllocatorImpl.Context.MakeCurrent();
            IntPtr dst = AddBytes(deviceBuffer, destinationByteOffset);
            IntPtr source = AddBytes(src.deviceBuffer, sourceByteOffset);
            if (ReferenceEquals(AllocatorImpl, src.AllocatorImpl))
            {
                CudaDriverApi.cuMemcpyDtoDAsync(
                    dst,
                    source,
                    new UIntPtr((ulong)byteCount),
                    AllocatorImpl.Stream.Handle).ThrowOnError();
            }
            else
            {
                src.SynchronizeDeviceWork();
                CudaDriverApi.cuMemcpyDtoD(dst, source, new UIntPtr((ulong)byteCount)).ThrowOnError();
            }

            MarkDeviceModified();
        }

        // 中文：等待该存储所属 CUDA 流上的所有工作完成。
        internal void SynchronizeDeviceWork()
        {
            ThrowIfDisposed();
            AllocatorImpl.Context.MakeCurrent();
            AllocatorImpl.Stream.Synchronize();
        }

        // 中文：从设备同步后，将指定区间的 Int32 元素读取到托管 int 数组返回。
        public override int[] GetElementsAsInt(long index, int length)
        {
            SyncHostFromDevice();
            if (ElementType != DType.Int32)
                throw new NotSupportedException("Element type " + ElementType + " not supported");

            ValidateElementRange(index, length);
            int[] array = new int[length];
            int* source = (int*)HostPtrAtElementUnchecked(index).ToPointer();
            for (int i = 0; i < length; i++)
                array[i] = source[i];

            return array;
        }

        // 中文：将 int 数组写入指定区间的主机镜像，并标记主机已写脏。
        public override void SetElementsAsInt(long index, int[] value)
        {
            if (value == null)
                throw new ArgumentNullException(nameof(value));
            if (ElementType != DType.Int32)
                throw new NotSupportedException("Element type " + ElementType + " not supported");

            ValidateElementRange(index, value.Length);
            EnsureHostBuffer();
            int* target = (int*)HostPtrAtElementUnchecked(index).ToPointer();
            for (int i = 0; i < value.Length; i++)
                target[i] = value[i];

            hostDirty = true;
            deviceDirty = false;
        }

        // 中文：从设备同步后，按元素类型读取单个元素并转换为 float 返回。
        public override float GetElementAsFloat(long index)
        {
            SyncHostFromDevice();
            ValidateElementRange(index, 1);

            return ElementType switch
            {
                DType.Float32 => ((float*)hostBuffer.ToPointer())[index],
                DType.Float64 => (float)((double*)hostBuffer.ToPointer())[index],
                DType.Int32 => ((int*)hostBuffer.ToPointer())[index],
                DType.UInt8 => ((byte*)hostBuffer.ToPointer())[index],
                _ => throw new NotSupportedException("Element type " + ElementType + " not supported"),
            };
        }

        // 中文：从设备同步后，将指定区间的 Float32 元素读取到托管 float 数组返回。
        public override float[] GetElementsAsFloat(long index, int length)
        {
            SyncHostFromDevice();
            if (ElementType != DType.Float32)
                throw new NotSupportedException("Element type " + ElementType + " not supported");

            ValidateElementRange(index, length);
            float[] array = new float[length];
            float* source = (float*)HostPtrAtElementUnchecked(index).ToPointer();
            for (int i = 0; i < length; i++)
                array[i] = source[i];

            return array;
        }

        // 中文：将单个 float 值按元素类型转换后写入主机镜像，并标记主机已写脏。
        public override void SetElementAsFloat(long index, float value)
        {
            ValidateElementRange(index, 1);
            EnsureHostBuffer();
            switch (ElementType)
            {
                case DType.Float32:
                    ((float*)hostBuffer.ToPointer())[index] = value;
                    break;
                case DType.Float64:
                    ((double*)hostBuffer.ToPointer())[index] = value;
                    break;
                case DType.Int32:
                    ((int*)hostBuffer.ToPointer())[index] = (int)value;
                    break;
                case DType.UInt8:
                    ((byte*)hostBuffer.ToPointer())[index] = (byte)value;
                    break;
                default:
                    throw new NotSupportedException("Element type " + ElementType + " not supported");
            }

            hostDirty = true;
            deviceDirty = false;
        }

        // 中文：将 float 数组写入指定区间的主机镜像，并标记主机已写脏。
        public override void SetElementsAsFloat(long index, float[] value)
        {
            if (value == null)
                throw new ArgumentNullException(nameof(value));
            if (ElementType != DType.Float32)
                throw new NotSupportedException("Element type " + ElementType + " not supported");

            ValidateElementRange(index, value.Length);
            EnsureHostBuffer();
            float* target = (float*)HostPtrAtElementUnchecked(index).ToPointer();
            for (int i = 0; i < value.Length; i++)
                target[i] = value[i];

            hostDirty = true;
            deviceDirty = false;
        }

        // 中文：CUDA 存储暂不支持 half 类型主机写入，直接抛出不支持异常。
        public override void SetElementsAsHalf(long index, half[] value)
        {
            throw new NotSupportedException("CUDA storage currently supports TensorSharp Float32/Float64/Int32/UInt8 host access.");
        }

        // 中文：从外部指针按字节拷贝数据到本存储的主机镜像，并标记主机已写脏。
        public override void CopyToStorage(long storageIndex, IntPtr src, long byteCount)
        {
            if (src == IntPtr.Zero && byteCount > 0)
                throw new ArgumentNullException(nameof(src));

            ValidateByteRange(storageIndex, byteCount);
            EnsureHostBuffer();
            Buffer.MemoryCopy(src.ToPointer(), HostPtrAtElementUnchecked(storageIndex).ToPointer(), byteCount, byteCount);
            hostDirty = true;
            deviceDirty = false;
        }

        // 中文：从设备同步后，将本存储主机镜像按字节拷贝到外部目标指针。
        public override void CopyFromStorage(IntPtr dst, long storageIndex, long byteCount)
        {
            if (dst == IntPtr.Zero && byteCount > 0)
                throw new ArgumentNullException(nameof(dst));

            SyncHostFromDevice();
            ValidateByteRange(storageIndex, byteCount);
            Buffer.MemoryCopy(HostPtrAtElementUnchecked(storageIndex).ToPointer(), dst.ToPointer(), byteCount, byteCount);
        }

        // 中文：确保主机缓冲区已分配后，返回指定元素的主机指针（不做范围校验）。
        private IntPtr HostPtrAtElementUnchecked(long index)
        {
            EnsureHostBuffer();
            return AddBytes(hostBuffer, checked(index * ElementType.Size()));
        }

        // 中文：惰性分配 64 字节对齐的主机镜像缓冲区，分配失败则抛出内存不足异常。
        private void EnsureHostBuffer()
        {
            if (hostBuffer != IntPtr.Zero)
                return;

            long allocationSize = Math.Max(ByteLength, 1);
            hostBuffer = (IntPtr)NativeMemory.AlignedAlloc((nuint)allocationSize, 64);
            if (hostBuffer == IntPtr.Zero)
                throw new OutOfMemoryException($"Failed to allocate {allocationSize} bytes of CUDA host mirror memory.");
        }

        // 中文：在指针基址上加上字节偏移量，返回新指针。
        private static IntPtr AddBytes(IntPtr pointer, long byteOffset)
        {
            return new IntPtr(pointer.ToInt64() + byteOffset);
        }

        // 中文：校验元素索引与长度是否在合法范围内，越界则抛出异常。
        private void ValidateElementRange(long index, long length)
        {
            if (index < 0 || length < 0 || index + length > ElementCount)
                throw new ArgumentOutOfRangeException(nameof(index));
        }

        // 中文：校验按字节计的访问范围是否在合法区间内，越界则抛出异常。
        private void ValidateByteRange(long storageIndex, long byteCount)
        {
            long byteOffset = checked(storageIndex * ElementType.Size());
            if (storageIndex < 0 || byteCount < 0 || byteOffset + byteCount > ByteLength)
                throw new ArgumentOutOfRangeException(nameof(storageIndex));
        }

        // 中文：若设备缓冲区已释放，则抛出对象已释放异常。
        private void ThrowIfDisposed()
        {
            if (deviceBuffer == IntPtr.Zero)
                throw new ObjectDisposedException(nameof(CudaStorage));
        }
    }
}
