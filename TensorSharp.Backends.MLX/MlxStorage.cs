using System;
using System.Runtime.InteropServices;

namespace TensorSharp.MLX
{
    [Serializable]
    public sealed unsafe class MlxStorage : Storage
    {
        private readonly object sync = new();
        private IntPtr buffer;
        private MlxNative.MlxArray deviceArray;
        private bool hostDirty = true;
        private bool deviceDirty;

        // 中文：构造 MLX 数组存储，绑定分配器与元素类型并校验字节长度。
        public MlxStorage(MlxAllocator allocator, DType elementType, long elementCount)
            : base(allocator, elementType, elementCount)
        {
            AllocatorImpl = allocator ?? throw new ArgumentNullException(nameof(allocator));
            if (ByteLength < 0)
                throw new ArgumentOutOfRangeException(nameof(elementCount));
        }

        internal MlxAllocator AllocatorImpl { get; }

        public int DeviceId => AllocatorImpl.DeviceId;

        // 中文：销毁存储，释放主机对齐内存与设备端 MLX 数组。
        protected override void Destroy()
        {
            if (buffer != IntPtr.Zero)
            {
                NativeMemory.AlignedFree(buffer.ToPointer());
                buffer = IntPtr.Zero;
            }

            if (deviceArray.IsValid)
            {
                MlxNative.FreeArray(deviceArray);
                deviceArray = default;
            }
        }

        // 中文：返回存储位置描述字符串（形如 MLX:设备号）。
        public override string LocationDescription()
        {
            return $"MLX:{DeviceId}";
        }

        // 中文：取指定元素的主机指针，先同步设备数据并标记主机为脏。
        public override IntPtr PtrAtElement(long index)
        {
            ThrowIfDestroyed();
            ValidateElementRange(index, 0);
            EnsureHostReadable();
            hostDirty = true;
            return AddBytes(buffer, checked(index * ElementType.Size()));
        }

        // 中文：确保主机缓冲区可读，必要时把设备端 MLX 数组拷回主机。
        public override void EnsureHostReadable()
        {
            ThrowIfDestroyed();
            lock (sync)
            {
                EnsureHostBufferAllocated();
                if (!deviceDirty || !deviceArray.IsValid || ByteLength == 0)
                    return;

                MlxNative.CopyArrayToHost(deviceArray, ElementType, buffer, ByteLength);
                deviceDirty = false;
                hostDirty = false;
            }
        }

        // 中文：按张量的形状/步长/偏移创建设备数组的跨步视图。
        internal MlxNative.MlxArray CreateArrayView(Tensor tensor)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
            if (!ReferenceEquals(tensor.Storage, this))
                throw new ArgumentException("Tensor is not backed by this MLX storage.", nameof(tensor));

            EnsureDeviceCurrent();
            return MlxNative.AsStrided(deviceArray, ToIntArray(tensor.Sizes), ToLongArray(tensor.Strides), tensor.StorageOffset);
        }

        // 中文：用新数组替换设备端存储，reshape 为一维扁平并标记设备为脏。
        internal void ReplaceDeviceArray(MlxNative.MlxArray array)
        {
            if (!array.IsValid)
                throw new ArgumentException("MLX array is empty.", nameof(array));
            if (ElementCount > int.MaxValue)
                throw new NotSupportedException("MLX storage arrays larger than Int32.MaxValue elements are not supported yet.");

            // Phase 4 attempted to skip the reshape when the incoming array
            // was already row-contiguous (and stored the multi-dim array
            // directly as the storage's deviceArray). That broke
            // <see cref="UpdateDeviceSlice"/>, which calls 1D
            // <c>SliceUpdate</c> on <c>deviceArray</c> — mlx errors out
            // with "Invalid number of indices or strides for array with
            // dimension 2" when the deviceArray is multi-dim. Keep the
            // invariant: <c>deviceArray</c> is always stored as 1D flat,
            // and reshape on entry. The reshape on a row-contiguous array
            // is internally a metadata-only no-op in MLX, so the cost is
            // limited to the mlx_reshape C call itself (~5 µs).
            MlxNative.MlxArray flat = MlxNative.Reshape(array, new[] { (int)ElementCount });
            MlxNative.FreeArray(array);

            lock (sync)
            {
                if (deviceArray.IsValid)
                    MlxNative.FreeArray(deviceArray);

                deviceArray = flat;
                hostDirty = false;
                deviceDirty = true;
            }
        }

        // 中文：用更新数组就地写入设备存储的连续切片（SliceUpdate）。
        internal void UpdateDeviceSlice(Tensor tensor, MlxNative.MlxArray update)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
            if (!ReferenceEquals(tensor.Storage, this))
                throw new ArgumentException("Tensor is not backed by this MLX storage.", nameof(tensor));
            if (!update.IsValid)
                throw new ArgumentException("MLX update array is empty.", nameof(update));
            if (!tensor.IsContiguous())
                throw new NotSupportedException("MLX slice updates require contiguous tensor views.");
            if (tensor.StorageOffset < 0 || tensor.StorageOffset > int.MaxValue)
                throw new NotSupportedException("MLX slice offsets larger than Int32.MaxValue are not supported yet.");
            if (tensor.ElementCount() > int.MaxValue)
                throw new NotSupportedException("MLX slice lengths larger than Int32.MaxValue are not supported yet.");

            EnsureDeviceCurrent();
            MlxNative.MlxArray flatUpdate = default;
            MlxNative.MlxArray updatedStorage = default;
            try
            {
                int length = (int)tensor.ElementCount();
                int start = (int)tensor.StorageOffset;
                flatUpdate = MlxNative.Reshape(update, new[] { length });
                updatedStorage = MlxNative.SliceUpdate(deviceArray, flatUpdate, start, start + length);

                lock (sync)
                {
                    if (deviceArray.IsValid)
                        MlxNative.FreeArray(deviceArray);

                    deviceArray = updatedStorage;
                    updatedStorage = default;
                    hostDirty = false;
                    deviceDirty = true;
                }
            }
            finally
            {
                MlxNative.FreeArray(flatUpdate);
                MlxNative.FreeArray(updatedStorage);
            }
        }

        // 中文：确保设备端数组为最新，必要时从主机缓冲区上载或新建零数组。
        internal void EnsureDeviceCurrent()
        {
            ThrowIfDestroyed();
            if (ElementCount > int.MaxValue)
                throw new NotSupportedException("MLX storage arrays larger than Int32.MaxValue elements are not supported yet.");

            lock (sync)
            {
                if (deviceArray.IsValid && !hostDirty)
                    return;

                if (deviceArray.IsValid)
                {
                    MlxNative.FreeArray(deviceArray);
                    deviceArray = default;
                }

                if (buffer != IntPtr.Zero)
                    deviceArray = MlxNative.NewArrayFromHost(buffer, new[] { (int)ElementCount }, ElementType);
                else
                    deviceArray = MlxNative.Full(new[] { (int)ElementCount }, 0f, ElementType);
                hostDirty = false;
                deviceDirty = false;
            }
        }

        // 中文：从主机缓冲区按 Int32 读取一段元素到数组。
        public override int[] GetElementsAsInt(long index, int length)
        {
            ValidateElementRange(index, length);
            if (ElementType != DType.Int32)
                throw new NotSupportedException("Element type " + ElementType + " not supported");

            int[] result = new int[length];
            EnsureHostReadable();
            int* src = (int*)AddBytes(buffer, checked(index * ElementType.Size()));
            for (int i = 0; i < length; i++)
                result[i] = src[i];
            return result;
        }

        // 中文：把 Int32 数组写入主机缓冲区指定区间。
        public override void SetElementsAsInt(long index, int[] value)
        {
            if (value == null)
                throw new ArgumentNullException(nameof(value));
            ValidateElementRange(index, value.Length);
            if (ElementType != DType.Int32)
                throw new NotSupportedException("Element type " + ElementType + " not supported");

            int* dst = (int*)PtrAtElement(index);
            for (int i = 0; i < value.Length; i++)
                dst[i] = value[i];
        }

        // 中文：按元素类型读取单个元素并转换为 float。
        public override float GetElementAsFloat(long index)
        {
            ValidateElementRange(index, 1);
            EnsureHostReadable();
            return ElementType switch
            {
                DType.Float32 => ((float*)buffer)[index],
                DType.Float64 => (float)((double*)buffer)[index],
                DType.Float16 => (float)((half*)buffer)[index],
                DType.Int32 => ((int*)buffer)[index],
                DType.UInt8 => ((byte*)buffer)[index],
                _ => throw new NotSupportedException("Element type " + ElementType + " not supported"),
            };
        }

        // 中文：逐元素读取一段并以 float 数组返回。
        public override float[] GetElementsAsFloat(long index, int length)
        {
            ValidateElementRange(index, length);
            float[] result = new float[length];
            for (int i = 0; i < length; i++)
                result[i] = GetElementAsFloat(index + i);
            return result;
        }

        // 中文：将 float 值按元素类型转换后写入单个元素并标记主机为脏。
        public override void SetElementAsFloat(long index, float value)
        {
            ValidateElementRange(index, 1);
            EnsureHostReadable();
            switch (ElementType)
            {
                case DType.Float32:
                    ((float*)buffer)[index] = value;
                    break;
                case DType.Float64:
                    ((double*)buffer)[index] = value;
                    break;
                case DType.Float16:
                    ((half*)buffer)[index] = value;
                    break;
                case DType.Int32:
                    ((int*)buffer)[index] = (int)value;
                    break;
                case DType.UInt8:
                    ((byte*)buffer)[index] = (byte)value;
                    break;
                default:
                    throw new NotSupportedException("Element type " + ElementType + " not supported");
            }
            hostDirty = true;
        }

        // 中文：逐元素把 float 数组写入主机缓冲区指定区间。
        public override void SetElementsAsFloat(long index, float[] value)
        {
            if (value == null)
                throw new ArgumentNullException(nameof(value));
            ValidateElementRange(index, value.Length);
            for (int i = 0; i < value.Length; i++)
                SetElementAsFloat(index + i, value[i]);
        }

        // 中文：把 half 数组写入主机缓冲区（仅限 Float16）并标记主机为脏。
        public override void SetElementsAsHalf(long index, half[] value)
        {
            if (value == null)
                throw new ArgumentNullException(nameof(value));
            ValidateElementRange(index, value.Length);
            if (ElementType != DType.Float16)
                throw new NotSupportedException("Element type " + ElementType + " not supported");

            half* dst = (half*)PtrAtElement(index);
            for (int i = 0; i < value.Length; i++)
                dst[i] = value[i];
            hostDirty = true;
        }

        // 中文：从外部指针按字节拷入主机缓冲区并标记主机为脏。
        public override void CopyToStorage(long storageIndex, IntPtr src, long byteCount)
        {
            if (src == IntPtr.Zero && byteCount > 0)
                throw new ArgumentNullException(nameof(src));
            ValidateByteRange(storageIndex, byteCount);
            EnsureHostReadable();
            Buffer.MemoryCopy(src.ToPointer(), AddBytes(buffer, checked(storageIndex * ElementType.Size())).ToPointer(), byteCount, byteCount);
            hostDirty = true;
        }

        // 中文：从主机缓冲区按字节拷出到外部指针。
        public override void CopyFromStorage(IntPtr dst, long storageIndex, long byteCount)
        {
            if (dst == IntPtr.Zero && byteCount > 0)
                throw new ArgumentNullException(nameof(dst));
            ValidateByteRange(storageIndex, byteCount);
            EnsureHostReadable();
            Buffer.MemoryCopy(AddBytes(buffer, checked(storageIndex * ElementType.Size())).ToPointer(), dst.ToPointer(), byteCount, byteCount);
        }

        // 中文：把 long 跨度转换为 int 数组，超出 Int32 范围则抛异常。
        private static int[] ToIntArray(ReadOnlySpan<long> values)
        {
            int[] result = new int[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                if (values[i] > int.MaxValue)
                    throw new NotSupportedException("MLX tensor dimensions larger than Int32.MaxValue are not supported yet.");
                result[i] = (int)values[i];
            }

            return result;
        }

        // 中文：把 long 跨度复制为 long 数组。
        private static long[] ToLongArray(ReadOnlySpan<long> values)
        {
            long[] result = new long[values.Length];
            values.CopyTo(result);
            return result;
        }

        // 中文：按需分配并清零 64 字节对齐的主机缓冲区。
        private void EnsureHostBufferAllocated()
        {
            if (buffer != IntPtr.Zero)
                return;

            nuint allocationBytes = (nuint)Math.Max(ByteLength, 1);
            buffer = (IntPtr)NativeMemory.AlignedAlloc(allocationBytes, 64);
            NativeMemory.Clear(buffer.ToPointer(), allocationBytes);
        }

        // 中文：校验元素索引与长度是否落在合法范围内。
        private void ValidateElementRange(long index, long length)
        {
            if (index < 0 || length < 0 || index + length > ElementCount)
                throw new ArgumentOutOfRangeException(nameof(index));
        }

        // 中文：校验字节偏移与字节数是否落在缓冲区范围内。
        private void ValidateByteRange(long storageIndex, long byteCount)
        {
            if (byteCount < 0)
                throw new ArgumentOutOfRangeException(nameof(byteCount));

            long byteOffset = checked(storageIndex * ElementType.Size());
            if (byteOffset < 0 || byteOffset + byteCount > ByteLength)
                throw new ArgumentOutOfRangeException(nameof(storageIndex));
        }

        // 中文：在指针上偏移指定字节数返回新指针。
        private static IntPtr AddBytes(IntPtr ptr, long bytes)
        {
            return new IntPtr(ptr.ToInt64() + bytes);
        }
    }
}
