// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using TensorSharp;

namespace TensorSharp.GGML
{
    [Serializable]
    public class GgmlStorage : Storage
    {
        private IntPtr buffer;

        // 中文：构造函数，校验上下文并从 GGML 内存池按字节长度分配底层缓冲区。
        public GgmlStorage(GgmlAllocator allocator, GgmlContext context, DType elementType, long elementCount)
            : base(allocator, elementType, elementCount)
        {
            Context = context ?? throw new ArgumentNullException(nameof(context));
            buffer = context.MemoryPool.Allocate(ByteLength);
        }

        public GgmlContext Context { get; }

        public int DeviceId => ((GgmlAllocator)Allocator).DeviceId;

        // 中文：销毁存储，将缓冲区归还 GGML 内存池并置空指针，避免重复释放。
        protected override void Destroy()
        {
            if (buffer != IntPtr.Zero)
            {
                // Note: under async compute, a freshly disposed pool block may be
                // reused for the next allocation while a previous GPU op is still
                // writing to it. That's safe in two ways:
                //   1) If the next GPU op uses the recycled block via zero-copy
                //      bind, Metal's command queue is FIFO so the previous op's
                //      writes complete before the next op's reads/writes begin.
                //   2) If host code writes to the recycled block via
                //      TensorComputePrimitives.GetFloatPointer, EnsureHostReadable()
                //      drains pending work first.
                Context.MemoryPool.Free(buffer, ByteLength);
                buffer = IntPtr.Zero;
            }
        }

        // 中文：返回存储位置描述字符串，格式为 GGML:设备号。
        public override string LocationDescription()
        {
            return $"GGML:{DeviceId}";
        }

        /// <summary>
        /// Drain any pending GGML async compute targeting this storage's host
        /// memory before the caller reads/writes it. Cheap when no work is pending
        /// (single atomic check on the C++ side); when there is pending work it
        /// performs one ggml_backend_synchronize to wait for the Metal command
        /// queue to drain.
        ///
        /// This is what makes the lazy-sync optimisation (see
        /// <see cref="GgmlBasicOps.SetAsyncCompute"/>) safe: the per-op GPU
        /// kernels return without waiting on the Metal command buffer, so by the
        /// time host code reaches <see cref="TensorComputePrimitives.GetFloatPointer"/>
        /// the GPU may still be writing to this storage's bytes; this call
        /// guarantees we synchronize before returning the raw pointer.
        /// </summary>
        // 中文：在主机读写前排空 GGML 待处理的异步计算，保证主机内存数据已就绪。
        public override void EnsureHostReadable()
        {
            GgmlBasicOps.HostReadBarrier();
        }

        // 中文：返回指定元素索引处的缓冲区指针；Q8_0 块量化类型只允许索引为 0。
        public override IntPtr PtrAtElement(long index)
        {
            // Block-quantized types (Q8_0) cannot be addressed at element granularity.
            // Native kernels always pass index = 0 (the buffer base) so we honour that;
            // anything else is a bug in the caller.
            if (ElementType == DType.Q8_0)
            {
                if (index != 0)
                    throw new NotSupportedException(
                        $"Q8_0 storage does not support element-level addressing (index={index}).");
                return buffer;
            }
            return new IntPtr(buffer.ToInt64() + (index * ElementType.Size()));
        }

        // 中文：从 Int32 缓冲区读取 length 个整数到托管数组返回。
        public override int[] GetElementsAsInt(long index, int length)
        {
            EnsureHostReadable();
            unsafe
            {
                if (ElementType == DType.Int32)
                {
                    int* p = (int*)buffer.ToPointer();
                    int[] array = new int[length];

                    for (int i = 0; i < length; i++)
                    {
                        array[i] = *(p + index + i);
                    }

                    return array;
                }

                throw new NotSupportedException("Element type " + ElementType + " not supported");
            }
        }

        // 中文：按元素类型（F32/F16/F64/Int32/UInt8）读取单个元素并转换为 float 返回。
        public override float GetElementAsFloat(long index)
        {
            EnsureHostReadable();
            unsafe
            {
                if (ElementType == DType.Float32)
                {
                    return ((float*)buffer.ToPointer())[index];
                }
                else if (ElementType == DType.Float16)
                {
                    return (float)System.BitConverter.UInt16BitsToHalf(((ushort*)buffer.ToPointer())[index]);
                }
                else if (ElementType == DType.Float64)
                {
                    return (float)((double*)buffer.ToPointer())[index];
                }
                else if (ElementType == DType.Int32)
                {
                    return ((int*)buffer.ToPointer())[index];
                }
                else if (ElementType == DType.UInt8)
                {
                    return ((byte*)buffer.ToPointer())[index];
                }

                throw new NotSupportedException("Element type " + ElementType + " not supported");
            }
        }

        // 中文：从 Float32 缓冲区读取 length 个浮点数到托管数组返回。
        public override float[] GetElementsAsFloat(long index, int length)
        {
            EnsureHostReadable();
            unsafe
            {
                if (ElementType == DType.Float32)
                {
                    float* p = (float*)buffer.ToPointer();
                    float[] array = new float[length];

                    for (int i = 0; i < length; i++)
                    {
                        array[i] = *(p + index + i);
                    }

                    return array;
                }

                throw new NotSupportedException("Element type " + ElementType + " not supported");
            }
        }

        // 中文：按元素类型将 float 值转换后写入指定索引处的缓冲区元素。
        public override void SetElementAsFloat(long index, float value)
        {
            EnsureHostReadable();
            unsafe
            {
                if (ElementType == DType.Float32)
                {
                    ((float*)buffer.ToPointer())[index] = value;
                }
                else if (ElementType == DType.Float16)
                {
                    ((ushort*)buffer.ToPointer())[index] = System.BitConverter.HalfToUInt16Bits((System.Half)value);
                }
                else if (ElementType == DType.Float64)
                {
                    ((double*)buffer.ToPointer())[index] = value;
                }
                else if (ElementType == DType.Int32)
                {
                    ((int*)buffer.ToPointer())[index] = (int)value;
                }
                else if (ElementType == DType.UInt8)
                {
                    ((byte*)buffer.ToPointer())[index] = (byte)value;
                }
                else
                {
                    throw new NotSupportedException("Element type " + ElementType + " not supported");
                }
            }
        }

        // 中文：将整数数组批量写入 Int32 缓冲区的指定起始索引处。
        public override void SetElementsAsInt(long index, int[] value)
        {
            EnsureHostReadable();
            unsafe
            {
                if (ElementType == DType.Int32)
                {
                    for (int i = 0; i < value.Length; i++)
                    {
                        ((int*)buffer.ToPointer())[index + i] = value[i];
                    }
                }
                else
                {
                    throw new NotSupportedException("Element type " + ElementType + " not supported");
                }
            }
        }

        // 中文：将浮点数组批量写入 Float32 缓冲区的指定起始索引处。
        public override void SetElementsAsFloat(long index, float[] value)
        {
            EnsureHostReadable();
            unsafe
            {
                if (ElementType == DType.Float32)
                {
                    for (int i = 0; i < value.Length; i++)
                    {
                        ((float*)buffer.ToPointer())[index + i] = value[i];
                    }
                }
                else
                {
                    throw new NotSupportedException("Element type " + ElementType + " not supported");
                }
            }
        }

        // 中文：写入半精度数组，GGML 后端暂不支持，直接抛出不支持异常。
        public override void SetElementsAsHalf(long index, half[] value)
        {
            throw new NotSupportedException("GGML backends currently support Float32 tensors only. Disable AMP to use this backend.");
        }

        // 中文：将外部源指针的数据按字节拷贝进本存储，拷贝前先排空异步 GPU 写入。
        public override void CopyToStorage(long storageIndex, IntPtr src, long byteCount)
        {
            // Drain pending async GPU writes before this CPU memcpy targets the
            // backing store, otherwise the write could race with an in-flight
            // zero-copy GPU write to the same host memory.
            EnsureHostReadable();
            IntPtr dstPtr = PtrAtElement(storageIndex);
            unsafe
            {
                Buffer.MemoryCopy(src.ToPointer(), dstPtr.ToPointer(), byteCount, byteCount);
            }
        }

        // 中文：将本存储的数据按字节拷贝到外部目标指针，拷贝前先排空异步 GPU 写入。
        public override void CopyFromStorage(IntPtr dst, long storageIndex, long byteCount)
        {
            // Drain pending async GPU writes so the CPU memcpy reads complete data.
            EnsureHostReadable();
            IntPtr srcPtr = PtrAtElement(storageIndex);
            unsafe
            {
                Buffer.MemoryCopy(srcPtr.ToPointer(), dst.ToPointer(), byteCount, byteCount);
            }
        }
    }
}
