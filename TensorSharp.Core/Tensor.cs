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
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using TensorSharp.Core;

// ───────────────────────────────────────────────────────────────────────────
// 【文件说明】张量核心类型——整个框架的基本数据结构。
// 【主要类型】Tensor：多维数组，由 形状(sizes)、步长(strides)、底层存储(Storage)
//             和存储偏移组成，并通过引用计数管理生命周期；所有计算最终作用于它。
// ───────────────────────────────────────────────────────────────────────────
namespace TensorSharp
{
    [Serializable]
    public class Tensor : IDisposable
    {
        private readonly long[] sizes;
        private readonly long[] strides;
        private readonly Storage storage;
        private readonly long storageOffset;
        private readonly long elementCount;

        private int isDisposed;


        /// <summary>
        /// Construct a new tensor, using the given allocator to construct a storage. The new tensor
        /// will be contiguous in memory. The tensor's elements will not be initialized.
        /// </summary>
        /// <param name="allocator"></param>
        /// <param name="elementType"></param>
        /// <param name="sizes"></param>
        // 中文：构造连续张量，按给定形状用分配器新建存储（元素不初始化）。
        public Tensor(IAllocator allocator, DType elementType, params long[] sizes)
            : this(allocator, elementType, (ReadOnlySpan<long>)sizes)
        {
        }

        // 中文：构造连续张量，由形状推导连续步长再委托主构造函数。
        public Tensor(IAllocator allocator, DType elementType, ReadOnlySpan<long> sizes)
            : this(allocator, elementType, sizes, TensorDimensionHelpers.GetContiguousStride(sizes))
        {
        }

        // 中文：以指定形状与步长新建张量并分配存储（数组重载）。
        public Tensor(IAllocator allocator, DType elementType, long[] sizes, long[] strides)
            : this(allocator, elementType, (ReadOnlySpan<long>)sizes, (ReadOnlySpan<long>)strides)
        {
        }

        // 中文：核心构造函数，校验形状与步长长度一致并按存储大小分配新存储。
        public Tensor(IAllocator allocator, DType elementType, ReadOnlySpan<long> sizes, ReadOnlySpan<long> strides)
        {
            if (sizes.Length != strides.Length)
            {
                throw new ArgumentException("sizes and strides must have the same length");
            }

            this.sizes = sizes.ToArray();
            this.strides = strides.ToArray();
            elementCount = TensorDimensionHelpers.ElementCount(sizes);
            storageOffset = 0;
            storage = allocator.Allocate(elementType, TensorDimensionHelpers.GetStorageSize(sizes, strides));
        }

        // 中文：在已有存储上构造张量视图（拷贝形状/步长元数据）。
        public Tensor(long[] sizes, long[] strides, Storage storage, long storageOffset)
            : this(sizes, strides, storage, storageOffset, copyMetadata: true)
        {
        }

        // 中文：在已有存储上构造张量视图（由 span 转数组，无需再次拷贝）。
        public Tensor(ReadOnlySpan<long> sizes, ReadOnlySpan<long> strides, Storage storage, long storageOffset)
            : this(sizes.ToArray(), strides.ToArray(), storage, storageOffset, copyMetadata: false)
        {
        }

        // 中文：私有底层构造函数，绑定既有存储、按需克隆元数据并对存储增引用计数。
        private Tensor(long[] sizes, long[] strides, Storage storage, long storageOffset, bool copyMetadata)
        {
            if (sizes.Length != strides.Length)
            {
                throw new ArgumentException("sizes and strides must have the same length");
            }

            this.sizes = copyMetadata ? (long[])sizes.Clone() : sizes;
            this.strides = copyMetadata ? (long[])strides.Clone() : strides;
            this.storage = storage;
            this.storageOffset = storageOffset;
            elementCount = TensorDimensionHelpers.ElementCount(this.sizes);

            this.storage.AddRef();
        }

        //~Tensor()
        //{
        //    if (!isDisposed)
        //    {
        //        Dispose();
        //    }
        //}

        // 中文：返回张量的类型与尺寸的字符串描述。
        public override string ToString()
        {
            return TensorFormatting.FormatTensorTypeAndSize(this);
        }

        // 中文：释放张量，原子置位防重入并对底层存储减引用计数。
        public void Dispose()
        {
            if (Interlocked.Exchange(ref isDisposed, 1) == 0)
            {
                storage.Release();
            }
        }

        // 中文：判等，要求存储引用、偏移、形状与步长全部相同。
        public override bool Equals(object obj)
        {
            Tensor o = obj as Tensor;
            if (o == null)
            {
                return false;
            }

            return
                Object.ReferenceEquals(storage, o.storage) &&
                storageOffset == o.storageOffset &&
                TensorResultBuilder.ArrayEqual((ReadOnlySpan<long>)sizes, o.sizes) &&
                TensorResultBuilder.ArrayEqual((ReadOnlySpan<long>)strides, o.strides);
        }

        // 中文：基于存储、偏移、各维形状与步长计算哈希码。
        public override int GetHashCode()
        {
            HashCode hash = new HashCode();
            hash.Add(storage);
            hash.Add(storageOffset);

            for (int i = 0; i < sizes.Length; ++i)
            {
                hash.Add(sizes[i]);
            }

            for (int i = 0; i < strides.Length; ++i)
            {
                hash.Add(strides[i]);
            }

            return hash.ToHashCode();
        }

        public DType ElementType => storage.ElementType;
        public ReadOnlySpan<long> Sizes => sizes;
        public ReadOnlySpan<long> Strides => strides;
        public ReadOnlyMemory<long> SizesMemory => sizes;
        public ReadOnlyMemory<long> StridesMemory => strides;
        public TensorShape Shape => new TensorShape(sizes);
        public TensorView Layout => new TensorView(sizes, strides, storageOffset);
        public Storage Storage => storage;
        public long StorageOffset => storageOffset;
        public IAllocator Allocator => storage.Allocator;

        public int DimensionCount => sizes.Length;



        // 中文：返回容纳该张量所需的底层存储元素数量。
        public long GetStorageSize()
        {
            return TensorDimensionHelpers.GetStorageSize(sizes, strides);
        }

        /// <summary>
        /// Returns a new Tensor object which points to the same storage as this,
        /// incrementing the refcount of the storage object.
        /// </summary>
        // 中文：返回共享同一存储的新张量并递增存储引用计数。
        public Tensor CopyRef()
        {
            return new Tensor(sizes, strides, storage, storageOffset, copyMetadata: false);
        }

        // 中文：判断当前是否为底层存储的唯一持有者。
        public bool IsOwnerExclusive()
        {
            return storage.IsOwnerExclusive();
        }

        // 中文：返回张量内容的完整格式化字符串。
        public string Format()
        {
            return TensorFormatting.Format(this);
        }

        // 中文：返回张量的元素总数。
        public long ElementCount()
        {
            return elementCount;
        }

        // 中文：判断张量在内存中是否连续（按步长从末维逐维校验）。
        public bool IsContiguous()
        {
            long z = 1;
            for (int d = sizes.Length - 1; d >= 0; d--)
            {
                if (sizes[d] != 1)
                {
                    if (strides[d] == z)
                    {
                        z *= sizes[d];
                    }
                    else
                    {
                        return false;
                    }
                }
            }
            return true;
        }


        // 中文：判断与另一张量的形状是否完全相同。
        public bool IsSameSizeAs(Tensor other)
        {
            return Core.TensorResultBuilder.ArrayEqual((ReadOnlySpan<long>)sizes, other.sizes);
        }

        /// <summary>
        /// Note: this does not check whether indices are in range
        /// </summary>
        /// <param name="indices"></param>
        /// <returns></returns>
        // 中文：按多维索引读取单个元素并转为 float（校验索引数量与范围）。
        public float GetElementAsFloat(params long[] indices)
        {
            if (indices.Length != DimensionCount)
            {
                throw new ArgumentException($"Number of indices must equal number of tensor dimensions. Tensor dim = '{DimensionCount}' and input indices length is '{indices.Length}'");
            }

            for (int i = 0; i < indices.Length; ++i)
            {
                if (indices[i] < 0 || indices[i] >= Sizes[i])
                {
                    throw new ArgumentException("Index " + i + " with value " + indices[i] + " is out of range");
                }
            }

            long offset = 0;
            for (int i = 0; i < indices.Length; ++i)
            {
                offset += indices[i] * strides[i];
            }

            return storage.GetElementAsFloat(storageOffset + offset);
        }

        // 中文：从存储偏移处连续读取 length 个元素为 float 数组。
        public float[] GetElementsAsFloat(int length)
        {
            return storage.GetElementsAsFloat(storageOffset, length);
        }

        /// <summary>
        /// Note: this does not check whether indices are in range
        /// </summary>
        /// <param name="indices"></param>
        /// <returns></returns>
        // 中文：按多维索引将单个元素写入为 float（校验索引数量与范围）。
        public void SetElementAsFloat(float value, params long[] indices)
        {
            if (indices.Length != DimensionCount)
            {
                throw new ArgumentException("Number of indices must equal number of tensor dimensions");
            }

            for (int i = 0; i < indices.Length; ++i)
            {
                if (indices[i] < 0 || indices[i] >= Sizes[i])
                {
                    throw new ArgumentException("Index " + i + " with value " + indices[i] + " is out of range");
                }
            }

            long offset = 0;
            for (int i = 0; i < indices.Length; ++i)
            {
                offset += indices[i] * strides[i];
            }

            storage.SetElementAsFloat(storageOffset + offset, value);
        }

        // 中文：从存储偏移处连续写入一组 float 元素。
        public void SetElementsAsFloat(float[] value)
        {
            storage.SetElementsAsFloat(storageOffset, value);
        }

        // 中文：从存储偏移处连续写入一组 half（半精度）元素。
        public void SetElementsAsHalf(half[] value)
        {
            storage.SetElementsAsHalf(storageOffset, value);
        }

        // 中文：从指定多维索引位置起连续写入一组 float（校验索引数量与范围）。
        public void SetElementsAsFloat(float[] value, params long[] indices)
        {
            if (indices.Length != DimensionCount)
            {
                throw new ArgumentException("Number of indices must equal number of tensor dimensions");
            }

            for (int i = 0; i < indices.Length; ++i)
            {
                if (indices[i] < 0 || indices[i] >= Sizes[i])
                {
                    throw new ArgumentException("Index " + i + " with value " + indices[i] + " is out of range");
                }
            }

            long offset = 0;
            for (int i = 0; i < indices.Length; ++i)
            {
                offset += indices[i] * strides[i];
            }

            storage.SetElementsAsFloat(storageOffset + offset, value);
        }


        // 中文：将形状各维度拼接为空格分隔的字符串（用于错误提示）。
        private static string PrintSizes(ReadOnlySpan<long> sizes)
        {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < sizes.Length; ++i)
            {
                sb.Append(sizes[i]);
                sb.Append(" ");
            }

            return sb.ToString();
        }

        // 中文：从存储偏移处连续读取 length 个元素为 int 数组。
        public int[] GetElementsAsInt(int length)
        {
            return storage.GetElementsAsInt(storageOffset, length);
        }

        // 中文：从存储偏移处连续写入一组 int 元素。
        public void SetElementsAsInt(int[] value)
        {
            storage.SetElementsAsInt(storageOffset, value);
        }

        // 中文：以新形状返回共享存储的视图（数组重载）。
        public Tensor View(params long[] sizes)
        {
            return View((ReadOnlySpan<long>)sizes);
        }

        // 中文：在不改变数据的前提下重塑形状，要求张量连续且元素总数不变。
        public Tensor View(ReadOnlySpan<long> sizes)
        {
            if (!IsContiguous())
            {
                throw new InvalidOperationException("Cannot use View on a non-contiguous tensor");
            }

            if (ElementCount() != TensorDimensionHelpers.ElementCount(sizes))
            {
                throw new InvalidOperationException($"Output tensor must have the same number of elements as the input. Size = {PrintSizes(this.sizes)}, New Size = {PrintSizes(sizes)}");
            }

            long[] newSizes = sizes.ToArray();
            return new Tensor(newSizes, TensorDimensionHelpers.GetContiguousStride(newSizes), storage, storageOffset, copyMetadata: false);
        }

        // 中文：在指定维度上截取 [startIndex, startIndex+size) 的连续切片视图。
        public Tensor Narrow(int dimension, long startIndex, long size)
        {
            if (dimension < 0 || dimension >= DimensionCount)
            {
                throw new ArgumentOutOfRangeException("dimension");
            }

            if (startIndex < 0 || startIndex >= sizes[dimension])
            {
                throw new ArgumentOutOfRangeException("startIndex", $"startIndex = '{startIndex}', sizes[dimension] = '{sizes[dimension]}', dimension = '{dimension}', size = '{size}'");
            }

            if (size <= 0 || startIndex + size > sizes[dimension])
            {
                throw new ArgumentOutOfRangeException("size", $"startIndex = '{startIndex}', sizes[dimension] = '{sizes[dimension]}', dimension = '{dimension}', size = '{size}'");
            }

            long newOffset = storageOffset + startIndex * strides[dimension];
            long[] newSizes = (long[])sizes.Clone();
            newSizes[dimension] = size;

            return new Tensor(newSizes, strides, storage, newOffset, copyMetadata: false);
        }

        // 中文：在指定维度上取 index 处切片，返回降一维的视图。
        public Tensor Select(int dimension, long index)
        {
            if (DimensionCount == 1)
            {
                throw new InvalidOperationException("Select requires 2 or more dimensions");
            }

            if (dimension < 0 || dimension >= DimensionCount)
            {
                throw new ArgumentOutOfRangeException("dimension");
            }

            if (index < 0 || index >= sizes[dimension])
            {
                throw new ArgumentOutOfRangeException("index");
            }

            long newOffset = storageOffset + index * strides[dimension];
            long[] newSizes = ArrayRemove(sizes, dimension);
            long[] newStrides = ArrayRemove(strides, dimension);

            return new Tensor(newSizes, newStrides, storage, newOffset, copyMetadata: false);
        }


        // 中文：二维张量转置（交换 0、1 两维）。
        public Tensor Transpose()
        {
            if (DimensionCount != 2)
            {
                throw new InvalidOperationException("Parameterless Transpose is only valid on 2d tensors");
            }

            return Transpose(0, 1);
        }

        // 中文：交换指定两个维度的形状与步长，返回转置视图。
        public Tensor Transpose(int dimension1, int dimension2)
        {
            if (dimension1 < 0 || dimension1 >= DimensionCount)
            {
                throw new ArgumentOutOfRangeException("dimension1");
            }

            if (dimension2 < 0 || dimension2 >= DimensionCount)
            {
                throw new ArgumentOutOfRangeException("dimension2");
            }

            if (dimension1 == dimension2)
            {
                return CopyRef();
            }

            long[] newSizes = (long[])sizes.Clone();
            long[] newStrides = (long[])strides.Clone();
            ArraySwap(newSizes, dimension1, dimension2);
            ArraySwap(newStrides, dimension1, dimension2);
            return new Tensor(newSizes, newStrides, storage, storageOffset, copyMetadata: false);
        }

        // 中文：按给定排列重排所有维度，内部分解为一系列两两交换实现。
        public Tensor Permute(params int[] dims)
        {
            if (dims.Length != DimensionCount)
            {
                throw new InvalidOperationException("The number of permutation indices must equal the number of tensor dimensions");
            }

            Tensor result = CopyRef();
            int[] permutation = (int[])dims.Clone();
            foreach (Tuple<int, int> swap in SwapsForPermutation(permutation))
            {
                Tensor resultOld = result;
                result = result.Transpose(swap.Item1, swap.Item2);
                resultOld.Dispose();
            }

            return result;
        }

        /// <summary>
        /// Expand one or more singleton dimensions (dimensions with size 1) by using a stride of 0
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="sizes"></param>
        /// <returns></returns>
        // 中文：将 size 为 1 的维度广播到目标尺寸（数组重载）。
        public Tensor Expand(params long[] newSizes)
        {
            return Expand((ReadOnlySpan<long>)newSizes);
        }

        // 中文：广播扩展，仅对 size 为 1 的维度令步长为 0 以共享数据。
        public Tensor Expand(ReadOnlySpan<long> newSizes)
        {
            if (newSizes.Length != DimensionCount)
            {
                throw new InvalidOperationException($"number of elements of newSizes must match the dimension count of tensor. New dimension = '{newSizes.Length}', Current dimension = '{DimensionCount}', New tensor shape = '{TensorDimensionHelpers.FormatDimensions(newSizes)}', current tensor shape = '{TensorDimensionHelpers.FormatDimensions(Sizes)}'");
            }

            long[] newStrides = (long[])strides.Clone();
            long[] targetSizes = newSizes.ToArray();
            for (int i = 0; i < newSizes.Length; ++i)
            {
                if (newSizes[i] <= 0)
                {
                    throw new ArgumentException(
                        $"Expand: target size at dim {i} must be positive (PyTorch/ggml-style broadcast); got {newSizes[i]}.");
                }

                if (newSizes[i] != sizes[i])
                {
                    if (sizes[i] != 1)
                    {
                        throw new InvalidOperationException(
                            $"Expand: can only broadcast size-1 dimensions; dim {i} has size {sizes[i]} but target is {newSizes[i]}.");
                    }

                    newStrides[i] = 0;
                }
            }

            return new Tensor(targetSizes, newStrides, storage, storageOffset, copyMetadata: false);
        }


        /// <summary>
        /// Return a new tensor where **all** singleton dimensions have been removed
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        // 中文：移除所有 size 为 1 的维度，返回降维视图。
        public Tensor Squeeze()
        {
            int newDimensionCount = 0;
            for (int i = 0; i < sizes.Length; ++i)
            {
                if (sizes[i] != 1)
                {
                    newDimensionCount++;
                }
            }

            long[] newSizes = new long[newDimensionCount];
            long[] newStrides = new long[newDimensionCount];
            int targetIndex = 0;
            for (int i = 0; i < sizes.Length; ++i)
            {
                if (sizes[i] != 1)
                {
                    newSizes[targetIndex] = sizes[i];
                    newStrides[targetIndex] = strides[i];
                    targetIndex++;
                }
            }

            return new Tensor(newSizes, newStrides, storage, storageOffset, copyMetadata: false);
        }


        /// <summary>
        /// Return a new tensor where the given singleton dimension has been removed
        /// </summary>
        /// <param name="dimension"></param>
        /// <returns></returns>
        // 中文：移除指定的单一（size 为 1）维度，返回降维视图。
        public Tensor Squeeze(int dimension)
        {
            if (DimensionCount == 1)
            {
                throw new InvalidOperationException("Squeeze requires 2 or more dimensions");
            }

            if (dimension < 0 || dimension >= DimensionCount)
            {
                throw new ArgumentOutOfRangeException("dimension");
            }

            long[] newSizes = ArrayRemove(sizes, dimension);
            long[] newStrides = ArrayRemove(strides, dimension);

            return new Tensor(newSizes, newStrides, storage, storageOffset, copyMetadata: false);
        }




        /// <summary>
        /// Returns a tensor which contains all slices of size size in the given dimension. The step between two slices is given by step.
        /// The result tensor has an additional dimension of size size.
        /// </summary>
        /// <param name="dimension"></param>
        /// <param name="size"></param>
        /// <param name="step"></param>
        /// <returns></returns>
        // 中文：在指定维度上以 size 窗口、step 步长滑窗展开，新增一维存放窗口元素。
        public Tensor Unfold(int dimension, long size, long step)
        {
            if (DimensionCount == 0)
            {
                throw new InvalidOperationException("Cannot unfold an empty tensor");
            }

            if (dimension < 0 || dimension >= DimensionCount)
            {
                throw new ArgumentOutOfRangeException("dimension is out of range", "dimension");
            }

            if (size > sizes[dimension])
            {
                throw new ArgumentOutOfRangeException("size cannot be larger than the size of dimension", "size");
            }

            if (step <= 0)
            {
                throw new ArgumentOutOfRangeException("step must be at least 1", "step");
            }

            long[] newSize = new long[DimensionCount + 1];
            long[] newStrides = new long[DimensionCount + 1];
            Array.Copy(sizes, newSize, DimensionCount);
            Array.Copy(strides, newStrides, DimensionCount);

            newSize[DimensionCount] = size;
            newStrides[DimensionCount] = strides[dimension];

            newSize[dimension] = (sizes[dimension] - size) / step + 1;
            newStrides[dimension] = step * strides[dimension];

            return new Tensor(newSize, newStrides, Storage, StorageOffset, copyMetadata: false);
        }


        // Pad array by prepending with 1 until its length equals newSize
        // 中文：在数组前端补 1 直至长度达到 newSize。
        private static long[] Pad1Prepend(long[] array, int newSize)
        {
            long[] result = new long[newSize];

            // Fill new extra elements with 1
            for (int i = 0; i < newSize - array.Length; ++i)
            {
                result[i] = 1;
            }

            // Copy array to the last array.Length elements of result
            Array.Copy(array, 0, result, newSize - array.Length, array.Length);

            return result;
        }

        // Prepend singleton dimensions until DimensionCount equals newDimCount
        // 中文：在前端补充单一维度，使维度数达到 newDimCount，返回相应视图。
        private Tensor PadToDimCount(int newDimCount)
        {
            long[] newSizes = Pad1Prepend(sizes, newDimCount);

            long[] newStrides = TensorDimensionHelpers.GetContiguousStride(newSizes);
            Array.Copy(strides, 0, newStrides, newStrides.Length - strides.Length, strides.Length);

            return new Tensor(newSizes, newStrides, storage, storageOffset, copyMetadata: false);
        }

        // 中文：按各维重复次数平铺张量，借助 Unfold/Expand/Copy 生成新张量。
        public Tensor RepeatTensor(params long[] repetitions)
        {
            if (repetitions.Length < DimensionCount)
            {
                throw new InvalidOperationException("repetitions must be at least the same length as the number of tensor dimensions");
            }

            for (int i = 0; i < repetitions.Length; ++i)
            {
                if (repetitions[i] < 1)
                {
                    throw new InvalidOperationException("All dimensions must be repeated at least once");
                }
            }

            Tensor paddedSrc = PadToDimCount(repetitions.Length);
            long[] resultSize = new long[repetitions.Length];
            for (int i = 0; i < repetitions.Length; ++i)
            {
                resultSize[i] = paddedSrc.sizes[i] * repetitions[i];
            }

            Tensor result = new Tensor(Allocator, ElementType, resultSize);

            Tensor urTensor = result.CopyRef();
            for (int i = 0; i < paddedSrc.DimensionCount; ++i)
            {
                Tensor oldUrTensor = urTensor;
                urTensor = urTensor.Unfold(i, paddedSrc.Sizes[i], paddedSrc.Sizes[i]);
                oldUrTensor.Dispose();
            }

            Tensor paddedSrc2 = paddedSrc.PadToDimCount(urTensor.DimensionCount);
            Tensor expandedSrc = paddedSrc2.Expand(urTensor.Sizes);
            Ops.Copy(urTensor, expandedSrc);

            paddedSrc.Dispose();
            paddedSrc2.Dispose();
            urTensor.Dispose();
            expandedSrc.Dispose();

            /*
            var sizesWritten = (long[])this.sizes.Clone();
            using (var subResult = result.GetRegion(Enumerable.Repeat((long)0, DimensionCount).ToArray(), sizesWritten))
            {
                Ops.Copy(subResult, this);
            }

            for (int i = 0; i < repetitions.Length; ++i)
            {
                if (repetitions[i] == 1) continue;

                sizesWritten[i] *= repetitions[i];
                using (var subResultSrc = result.GetRegion(Enumerable.Repeat((long)0, DimensionCount).ToArray(), this.sizes))
                using (var subResultTgt = result.GetRegion(Enumerable.Repeat((long)0, DimensionCount).ToArray(), this.sizes))
                {
                    Ops.Copy(subResultTgt, subResultSrc);
                }
            }*/

            return result;
        }

        /*
            private Tensor GetRegion(long[] dimensionStarts, long[] dimensionSizes)
        {
            var result = this.CopyRef();
            for (int i = 0; i < dimensionStarts.Length; ++i)
            {
                var resultOld = result;
                result.Narrow(i, dimensionStarts[i], dimensionSizes[i]);
                resultOld.Dispose();
            }
            return result;
        }*/


        // 中文：从 CLR 数组按字节拷贝数据到张量存储（要求连续、元素数与类型匹配）。
        public void CopyFrom(Array array)
        {
            DType elementType = DTypeBuilder.FromCLRType(array.GetType().GetElementType());

            if (!IsContiguous())
            {
                throw new InvalidOperationException("Tensor must be contiguous to copy from CLR array");
            }

            if (ElementCount() != array.LongLength)
            {
                throw new InvalidOperationException("Tensor and array must have the same number of elements");
            }

            if (ElementType != elementType)
            {
                throw new InvalidOperationException("Tensor and array must have the same element types");
            }

            GCHandle handle = GCHandle.Alloc(array, GCHandleType.Pinned);
            try
            {
                int length = Buffer.ByteLength(array);
                Storage.CopyToStorage(StorageOffset, handle.AddrOfPinnedObject(), length);
            }
            finally
            {
                handle.Free();
            }
        }


        // 中文：将张量存储按字节拷贝到 CLR 数组（要求连续、元素数与类型匹配）。
        public void CopyToArray(Array array)
        {
            DType elementType = DTypeBuilder.FromCLRType(array.GetType().GetElementType());

            if (!IsContiguous())
            {
                throw new InvalidOperationException("Tensor must be contiguous to copy from CLR array");
            }

            if (ElementCount() != array.LongLength)
            {
                throw new InvalidOperationException("Tensor and array must have the same number of elements");
            }

            if (ElementType != elementType)
            {
                throw new InvalidOperationException("Tensor and array must have the same element types");
            }

            GCHandle handle = GCHandle.Alloc(array, GCHandleType.Pinned);
            try
            {
                int length = Buffer.ByteLength(array);
                Storage.CopyFromStorage(handle.AddrOfPinnedObject(), StorageOffset, length);
            }
            finally
            {
                handle.Free();
            }
        }


        // 中文：由 CLR 多维数组创建同形状的新张量并拷入数据（假定无平台填充）。
        public static Tensor FromArray(IAllocator allocator, Array array)
        {
            // From the CLI spec(section 8.9.1):
            // Array elements shall be laid out within the array object in row - major order
            // (i.e., the elements associated with the rightmost array dimension shall be laid out contiguously from lowest to highest index).
            // The actual storage allocated for each array element can include platform - specific padding.

            // This is already in the order we want - and here we will (potentially incorrectly) assume that there is no
            // 'platform-specific padding'. This appears to be a reasonable assumption on both CLR and Mono.
            // Assuming no platform-specific padding allows us to use memcpy instead of iterating and copying each element

            DType elementType = DTypeBuilder.FromCLRType(array.GetType().GetElementType());

            long[] dimSizes = new long[array.Rank];
            for (int i = 0; i < dimSizes.Length; ++i)
            {
                dimSizes[i] = array.GetLength(i);
            }

            Tensor result = new Tensor(allocator, elementType, dimSizes);
            result.CopyFrom(array);
            return result;
        }


        // 中文：交换数组中两个下标处的元素。
        private static void ArraySwap<T>(T[] array, int index1, int index2)
        {
            T temp = array[index1];
            array[index1] = array[index2];
            array[index2] = temp;
        }

        // Return a copy of an array, but with the item at index removed
        // 中文：返回去掉指定下标元素后的数组副本。
        private static T[] ArrayRemove<T>(T[] source, long index)
        {
            T[] result = new T[source.Length - 1];
            for (int i = 0; i < result.Length; ++i)
            {
                if (i < index)
                {
                    result[i] = source[i];
                }
                else
                {
                    result[i] = source[i + 1];
                }
            }
            return result;
        }

        // Convert a permutation into a sequence of swap operations.
        // perm must contain a permuation of the indices [0, perm.Length)
        // The returned tuples indicate pairs of indices that should be swapped. The swaps
        // must be performed in the given order.
        // 中文：将一个排列分解为按序执行的下标交换对序列。
        private static IEnumerable<Tuple<int, int>> SwapsForPermutation(int[] perm)
        {
            int j;
            for (int i = 0; i < perm.Length; ++i)
            {
                int p = perm[i];
                if (p != i && p != -1)
                {
                    j = i;
                    do
                    {
                        if (perm[j] < 0 || perm[j] >= perm.Length)
                        {
                            throw new InvalidOperationException("Invalid permutation");
                        }

                        yield return Tuple.Create(j, perm[j]);


                        int jOld = j;
                        j = perm[j];
                        perm[jOld] = -1;
                    } while (perm[j] != i);
                    perm[j] = j;
                }
            }
        }


        // 中文：将张量序列化写入流。
        public void Serialize(System.IO.Stream stream)
        {
            TensorSerialization.Serialize(this, stream);
        }

        // 中文：从流反序列化并用分配器重建张量。
        public static Tensor Deserialize(IAllocator allocator, System.IO.Stream stream)
        {
            return TensorSerialization.Deserialize(allocator, stream);
        }
    }
}
