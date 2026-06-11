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
using System.Runtime.InteropServices;

namespace TensorSharp.Cpu
{
    public class CpuStorage : Storage
    {
        public IntPtr buffer;


        // 中文：构造函数，按字节长度在非托管堆上分配 CPU 存储缓冲区。
        public CpuStorage(IAllocator allocator, DType ElementType, long elementCount)
            : base(allocator, ElementType, elementCount)
        {
            buffer = Marshal.AllocHGlobal(new IntPtr(ByteLength));
        }

        // 中文：释放非托管缓冲区内存并将指针置零。
        protected override void Destroy()
        {
            Marshal.FreeHGlobal(buffer);
            buffer = IntPtr.Zero;
        }

        // 中文：返回存储位置描述，固定为 "CPU"。
        public override string LocationDescription()
        {
            return "CPU";
        }

        // 中文：计算并返回指定元素索引对应的非托管内存指针。
        public override IntPtr PtrAtElement(long index)
        {
            return new IntPtr(buffer.ToInt64() + (index * ElementType.Size()));
        }

        // 中文：从缓冲区读取一段 Int32 元素并以 int 数组返回（仅支持 Int32 类型）。
        public override int[] GetElementsAsInt(long index, int length)
        {
            unsafe
            {
                if (ElementType == DType.Int32)
                {
                    int* p = ((int*)buffer.ToPointer());
                    int[] array = new int[length];

                    for (int i = 0; i < length; i++)
                    {
                        array[i] = *(p + i);
                    }
                    return array;
                }
                else
                {
                    throw new NotSupportedException("Element type " + ElementType + " not supported");
                }
            }
        }

        // 中文：按当前元素类型读取单个元素并转换为 float 返回。
        public override float GetElementAsFloat(long index)
        {
            unsafe
            {
                if (ElementType == DType.Float32)
                {
                    return ((float*)buffer.ToPointer())[index];
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
                else
                {
                    throw new NotSupportedException("Element type " + ElementType + " not supported");
                }
            }
        }

        // 中文：从缓冲区读取一段 Float32 元素并以 float 数组返回（仅支持 Float32 类型）。
        public override float[] GetElementsAsFloat(long index, int length)
        {
            unsafe
            {
                if (ElementType == DType.Float32)
                {
                    float* p = ((float*)buffer.ToPointer());
                    float[] array = new float[length];

                    for (int i = 0; i < length; i++)
                    {
                        array[i] = *(p + i);
                    }
                    return array;
                }
                else
                {
                    throw new NotSupportedException("Element type " + ElementType + " not supported");
                }
            }
        }

        // 中文：将 float 值按当前元素类型转换后写入指定索引位置。
        public override void SetElementAsFloat(long index, float value)
        {
            unsafe
            {
                if (ElementType == DType.Float32)
                {
                    ((float*)buffer.ToPointer())[index] = value;
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

        // 中文：将一个 int 数组批量写入缓冲区（仅支持 Int32 类型）。
        public override void SetElementsAsInt(long index, int[] value)
        {
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

        // 中文：将一个 float 数组批量写入缓冲区（仅支持 Float32 类型）。
        public override void SetElementsAsFloat(long index, float[] value)
        {
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

        // 中文：写入 half 半精度数组，CPU 端尚未实现，调用将抛出异常。
        public override void SetElementsAsHalf(long index, half[] value)
        {
            throw new NotImplementedException($"SetElementsAsHalf has not been implemented for CPUs yet.");
        }

        // 中文：将外部源指针的若干字节拷贝到本存储指定位置。
        public override void CopyToStorage(long storageIndex, IntPtr src, long byteCount)
        {
            IntPtr dstPtr = PtrAtElement(storageIndex);
            unsafe
            {
                Buffer.MemoryCopy(src.ToPointer(), dstPtr.ToPointer(), byteCount, byteCount);
            }
        }

        // 中文：将本存储指定位置的若干字节拷贝到外部目标指针。
        public override void CopyFromStorage(IntPtr dst, long storageIndex, long byteCount)
        {
            IntPtr srcPtr = PtrAtElement(storageIndex);
            unsafe
            {
                Buffer.MemoryCopy(srcPtr.ToPointer(), dst.ToPointer(), byteCount, byteCount);
            }
        }
    }
}
