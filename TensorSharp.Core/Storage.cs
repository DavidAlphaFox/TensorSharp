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

// ───────────────────────────────────────────────────────────────────────────
// 【文件说明】张量底层内存的抽象基类。
// 【主要类型】Storage：继承自引用计数基类 RefCounted，封装一块连续内存；
//             各计算后端（CPU / CUDA / MLX / GGML）各自派生实现具体的存储与读写。
// ───────────────────────────────────────────────────────────────────────────
namespace TensorSharp
{
    [Serializable]
    public abstract class Storage : RefCounted
    {
        // 中文：构造存储对象，记录分配器、元素类型与元素数量。
        public Storage(IAllocator allocator, DType elementType, long elementCount)
        {
            Allocator = allocator;
            ElementType = elementType;
            ElementCount = elementCount;
        }

        /// <summary>
        /// Gets a reference to the allocator that constructed this Storage object.
        /// </summary>
        public IAllocator Allocator { get; private set; }

        public DType ElementType { get; private set; }
        public long ElementCount { get; private set; }

        // Block-quantized types (Q8_0) round their byte length up to the block
        // boundary (32-element blocks of 34 bytes each). Linear types stay at
        // ElementCount * ElementType.Size() as before.
        public long ByteLength => ElementType.ByteLengthFor(ElementCount);

        // 中文：判断当前是否为该存储的唯一持有者（引用计数为 1）。
        public bool IsOwnerExclusive()
        {
            return GetCurrentRefCount() == 1;
        }

        // 中文：返回指定元素索引处的原生内存指针。
        public abstract IntPtr PtrAtElement(long index);


        // 中文：从指定位置读取 length 个元素并以 int 数组返回。
        public abstract int[] GetElementsAsInt(long index, int length);
        // 中文：将 int 数组写入指定位置。
        public abstract void SetElementsAsInt(long index, int[] value);


        // 中文：返回该存储所在位置的描述（如 CPU / 某号 GPU）。
        public abstract string LocationDescription();

        // 中文：读取指定索引处的单个元素并转为 float。
        public abstract float GetElementAsFloat(long index);
        // 中文：从指定位置读取 length 个元素并以 float 数组返回。
        public abstract float[] GetElementsAsFloat(long index, int length);
        // 中文：以 float 写入指定索引处的单个元素。
        public abstract void SetElementAsFloat(long index, float value);
        // 中文：将 float 数组写入指定位置。
        public abstract void SetElementsAsFloat(long index, float[] value);

        // 中文：将 half（半精度）数组写入指定位置。
        public abstract void SetElementsAsHalf(long index, half[] value);

        // 中文：从原生源指针拷贝 byteCount 字节到本存储的指定位置。
        public abstract void CopyToStorage(long storageIndex, IntPtr src, long byteCount);
        // 中文：从本存储的指定位置拷贝 byteCount 字节到目标指针。
        public abstract void CopyFromStorage(IntPtr dst, long storageIndex, long byteCount);

        /// <summary>
        /// Hook that backends with deferred GPU compute (currently the GGML/Metal
        /// backend in async mode) override to drain any in-flight work before host
        /// code reads or writes this storage's bytes directly. Default: no-op.
        ///
        /// Called from <see cref="TensorComputePrimitives.GetFloatPointer"/> /
        /// <see cref="TensorComputePrimitives.GetHalfPointer"/>, which are the
        /// gateways for raw-pointer host access on tensor data. Native op code
        /// goes through <see cref="PtrAtElement"/> directly and intentionally
        /// skips this hook so that op chaining stays asynchronous.
        /// </summary>
        // 中文：在主机直接读写存储字节前，确保异步后端的在途计算已完成（默认空实现）。
        public virtual void EnsureHostReadable()
        {
        }

    }
}
