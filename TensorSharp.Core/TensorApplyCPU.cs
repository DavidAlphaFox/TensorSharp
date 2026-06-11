// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Xml;
using System.Threading.Tasks;
using TensorSharp.Cpu;

namespace TensorSharp
{
	public class TensorApplyCPU
	{
        private const int ParallelWorkThreshold = 1 << 15;

        #region Tensor iteration methods
        unsafe public delegate void Apply1KernelFunction(float* x);
		unsafe public delegate void Apply2KernelFunction(float* x, float* y);
		unsafe public delegate void Apply3KernelFunction(float* x, float* y, float* z);
		unsafe public delegate void Apply4KernelFunction(float* x, float* y, float* z, float* k);
		unsafe public delegate void Apply5KernelFunction(float* x, float* y, float* z, float* k, float* l);
		unsafe public delegate void ApplyDim2KernelFuncton(float* x, long sizeX, long stridesX, float* y, long sizeY, long stridesY);
		unsafe public delegate void ApplyDim3KernelFuncton(float* x, long sizeX, long stridesX, float* y, long sizeY, long stridesY, float* z, long sizeZ, long stridesZ);

        // 中文：尝试获取连续 Float32 张量的裸指针与元素数，用于走快速路径（失败则返回 false）。
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        unsafe private static bool TryGetContiguousFloat(Tensor tensor, out float* ptr, out int length)
        {
            if (tensor.ElementType == DType.Float32 && tensor.IsContiguous() && tensor.ElementCount() <= int.MaxValue)
            {
                ptr = (float*)CpuNativeHelpers.GetBufferStart(tensor);
                length = (int)tensor.ElementCount();
                return true;
            }

            ptr = null;
            length = 0;
            return false;
        }

        // 中文：尝试把连续 Float32 张量按行（末维为列）拆出指针、行数、列数，用于按行向量化快速路径。
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        unsafe private static bool TryGetContiguousRows(Tensor tensor, out float* ptr, out int rows, out int cols)
        {
            if (tensor.ElementType == DType.Float32 && tensor.IsContiguous() && tensor.ElementCount() <= int.MaxValue && tensor.Sizes[^1] <= int.MaxValue)
            {
                cols = (int)tensor.Sizes[^1];
                int elementCount = (int)tensor.ElementCount();
                if (cols > 0 && elementCount % cols == 0)
                {
                    ptr = (float*)CpuNativeHelpers.GetBufferStart(tensor);
                    rows = elementCount / cols;
                    return true;
                }
            }

            ptr = null;
            rows = 0;
            cols = 0;
            return false;
        }

        // 中文：根据外层与内层工作量之积是否超过阈值，判断是否值得并行化。
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool ShouldParallelize(int outerWork, int innerWork)
        {
            return outerWork > 1 && (long)outerWork * innerWork >= ParallelWorkThreshold;
        }

        // 中文：对连续缓冲逐元素计算 SiLU（x*sigmoid(x)），借助 TensorPrimitives 向量化，处理原地与非原地两种情形。
        unsafe private static void SiLUContiguous(float* resultPtr, float* srcPtr, int length)
        {
            ReadOnlySpan<float> input = new ReadOnlySpan<float>(srcPtr, length);
            Span<float> output = new Span<float>(resultPtr, length);

            if (resultPtr != srcPtr)
            {
                TensorPrimitives.Sigmoid(input, output);
                TensorPrimitives.Multiply(input, output, output);
                return;
            }

            float[] rented = ArrayPool<float>.Shared.Rent(length);
            try
            {
                Span<float> sigmoid = rented.AsSpan(0, length);
                TensorPrimitives.Sigmoid(input, sigmoid);
                TensorPrimitives.Multiply(input, sigmoid, output);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(rented);
            }
        }

        // 中文：连续缓冲上计算 SiLU(gate)*up（SwiGLU 融合），向量化处理原地与非原地两种情形。
        unsafe private static void SiLUMulContiguous(float* resultPtr, float* gatePtr, float* upPtr, int length)
        {
            ReadOnlySpan<float> gate = new ReadOnlySpan<float>(gatePtr, length);
            ReadOnlySpan<float> up = new ReadOnlySpan<float>(upPtr, length);
            Span<float> output = new Span<float>(resultPtr, length);

            if (resultPtr != gatePtr && resultPtr != upPtr)
            {
                TensorPrimitives.Sigmoid(gate, output);
                MultiplySiLUGateUp(gate, up, output, output);
                return;
            }

            float[] rented = ArrayPool<float>.Shared.Rent(length);
            try
            {
                Span<float> tmp = rented.AsSpan(0, length);
                TensorPrimitives.Sigmoid(gate, tmp);
                MultiplySiLUGateUp(gate, up, tmp, output);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(rented);
            }
        }

        // 中文：逐元素计算 gate*sigmoid*up 并写入 output，使用 Vector<float> 向量化加尾部标量收尾。
        private static void MultiplySiLUGateUp(
            ReadOnlySpan<float> gate,
            ReadOnlySpan<float> up,
            ReadOnlySpan<float> sigmoid,
            Span<float> output)
        {
            int vectorSize = Vector<float>.Count;
            int i = 0;
            for (; i <= output.Length - vectorSize; i += vectorSize)
            {
                Vector<float> value =
                    new Vector<float>(gate.Slice(i)) *
                    new Vector<float>(sigmoid.Slice(i)) *
                    new Vector<float>(up.Slice(i));
                value.CopyTo(output.Slice(i));
            }

            for (; i < output.Length; i++)
            {
                output[i] = gate[i] * sigmoid[i] * up[i];
            }
        }

        // 中文：连续缓冲上计算 x*sigmoid(gate)，向量化处理原地与非原地两种情形。
        unsafe private static void SigmoidMulContiguous(float* resultPtr, float* xPtr, float* gatePtr, int length)
        {
            ReadOnlySpan<float> x = new ReadOnlySpan<float>(xPtr, length);
            ReadOnlySpan<float> gate = new ReadOnlySpan<float>(gatePtr, length);
            Span<float> output = new Span<float>(resultPtr, length);

            if (resultPtr != xPtr && resultPtr != gatePtr)
            {
                TensorPrimitives.Sigmoid(gate, output);
                TensorPrimitives.Multiply(x, output, output);
                return;
            }

            float[] rented = ArrayPool<float>.Shared.Rent(length);
            try
            {
                Span<float> tmp = rented.AsSpan(0, length);
                TensorPrimitives.Sigmoid(gate, tmp);
                TensorPrimitives.Multiply(x, tmp, output);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(rented);
            }
        }

        // 中文：从裸指针处非对齐地读取一个 Vector<float>。
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        unsafe private static Vector<float> LoadVec(float* ptr)
        {
            return Unsafe.ReadUnaligned<Vector<float>>(ref *(byte*)ptr);
        }

        // 中文：向裸指针处非对齐地写入一个 Vector<float>。
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        unsafe private static void StoreVec(float* ptr, Vector<float> value)
        {
            Unsafe.WriteUnaligned(ref *(byte*)ptr, value);
        }

        // 中文：对两段连续缓冲做向量化点积（累加 lhs*rhs），尾部用标量收尾。
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        unsafe private static float DotContiguous(float* lhs, float* rhs, int length)
        {
            int vectorSize = Vector<float>.Count;
            Vector<float> acc = Vector<float>.Zero;
            int i = 0;

            for (; i <= length - vectorSize; i += vectorSize)
            {
                acc += LoadVec(lhs + i) * LoadVec(rhs + i);
            }

            float sum = Vector.Sum(acc);
            for (; i < length; i++)
            {
                sum += lhs[i] * rhs[i];
            }

            return sum;
        }

		// 中文：对单个张量做跨步迭代，逐元素调用一元核函数 func。
		unsafe static void Apply1(Tensor tensor1, Apply1KernelFunction func)
		{
			float* buffer1 = (float*)CpuNativeHelpers.GetBufferStart(tensor1);

			TensorIterState tensor1Iter = new TensorIterState(buffer1, tensor1.DimensionCount, tensor1.SizesMemory, tensor1.StridesMemory);

			do
			{
				for (; !tensor1Iter.ReachedBlockEnd(); tensor1Iter.BlockStep())
				{
					func(tensor1Iter.data);
				}

			} while (tensor1Iter.NextBlock());
		}


		// 中文：对两个张量并行跨步迭代，逐元素调用二元核函数 func（支持按 step 分块）。
		unsafe static void Apply2(Tensor tensor1, Tensor tensor2, Apply2KernelFunction func, int step = 1)
		{
			float* buffer1 = (float*)CpuNativeHelpers.GetBufferStart(tensor1);
			float* buffer2 = (float*)CpuNativeHelpers.GetBufferStart(tensor2);

			TensorIterState tensor1Iter = new TensorIterState(buffer1, tensor1.DimensionCount, tensor1.SizesMemory, tensor1.StridesMemory, step);
			TensorIterState tensor2Iter = new TensorIterState(buffer2, tensor2.DimensionCount, tensor2.SizesMemory, tensor2.StridesMemory, step);

			do
			{
				for (; !tensor1Iter.ReachedBlockEnd() && !tensor2Iter.ReachedBlockEnd(); tensor1Iter.BlockStep(), tensor2Iter.BlockStep())
				{
					func(tensor1Iter.data, tensor2Iter.data);
				}

			} while (tensor1Iter.NextBlock() && tensor2Iter.NextBlock());
		}


		// 中文：对三个张量并行跨步迭代，逐元素调用三元核函数 func。
		unsafe static void Apply3(Tensor tensor1, Tensor tensor2, Tensor tensor3, Apply3KernelFunction func, int step = 1)
		{
			float* buffer1 = (float*)CpuNativeHelpers.GetBufferStart(tensor1);
			float* buffer2 = (float*)CpuNativeHelpers.GetBufferStart(tensor2);
			float* buffer3 = (float*)CpuNativeHelpers.GetBufferStart(tensor3);

			TensorIterState tensor1Iter = new TensorIterState(buffer1, tensor1.DimensionCount, tensor1.SizesMemory, tensor1.StridesMemory, step);
			TensorIterState tensor2Iter = new TensorIterState(buffer2, tensor2.DimensionCount, tensor2.SizesMemory, tensor2.StridesMemory, step);
			TensorIterState tensor3Iter = new TensorIterState(buffer3, tensor3.DimensionCount, tensor3.SizesMemory, tensor3.StridesMemory, step);

			do
			{
				for (; !tensor1Iter.ReachedBlockEnd() && !tensor2Iter.ReachedBlockEnd() && !tensor3Iter.ReachedBlockEnd();
						tensor1Iter.BlockStep(), tensor2Iter.BlockStep(), tensor3Iter.BlockStep())
				{
					func(tensor1Iter.data, tensor2Iter.data, tensor3Iter.data);
				}

			} while (tensor1Iter.NextBlock() && tensor2Iter.NextBlock() && tensor3Iter.NextBlock());
		}


		// 中文：对四个张量并行跨步迭代，逐元素调用四元核函数 func。
		unsafe static void Apply4(Tensor tensor1, Tensor tensor2, Tensor tensor3, Tensor tensor4, Apply4KernelFunction func)
		{
			float* buffer1 = (float*)CpuNativeHelpers.GetBufferStart(tensor1);
			float* buffer2 = (float*)CpuNativeHelpers.GetBufferStart(tensor2);
			float* buffer3 = (float*)CpuNativeHelpers.GetBufferStart(tensor3);
			float* buffer4 = (float*)CpuNativeHelpers.GetBufferStart(tensor4);

			TensorIterState tensor1Iter = new TensorIterState(buffer1, tensor1.DimensionCount, tensor1.SizesMemory, tensor1.StridesMemory);
			TensorIterState tensor2Iter = new TensorIterState(buffer2, tensor2.DimensionCount, tensor2.SizesMemory, tensor2.StridesMemory);
			TensorIterState tensor3Iter = new TensorIterState(buffer3, tensor3.DimensionCount, tensor3.SizesMemory, tensor3.StridesMemory);
			TensorIterState tensor4Iter = new TensorIterState(buffer4, tensor4.DimensionCount, tensor4.SizesMemory, tensor4.StridesMemory);

			do
			{
				for (; !tensor1Iter.ReachedBlockEnd() && !tensor2Iter.ReachedBlockEnd() && !tensor3Iter.ReachedBlockEnd() && !tensor4Iter.ReachedBlockEnd();
					tensor1Iter.BlockStep(), tensor2Iter.BlockStep(), tensor3Iter.BlockStep(), tensor4Iter.BlockStep())
				{
					func(tensor1Iter.data, tensor2Iter.data, tensor3Iter.data, tensor4Iter.data);
				}

			} while (tensor1Iter.NextBlock() && tensor2Iter.NextBlock() && tensor3Iter.NextBlock() && tensor4Iter.NextBlock());
		}


		// 中文：对五个张量并行跨步迭代，逐元素调用五元核函数 func。
		unsafe static void Apply5(Tensor tensor1, Tensor tensor2, Tensor tensor3, Tensor tensor4, Tensor tensor5, Apply5KernelFunction func, int step = 1)
		{
			float* buffer1 = (float*)CpuNativeHelpers.GetBufferStart(tensor1);
			float* buffer2 = (float*)CpuNativeHelpers.GetBufferStart(tensor2);
			float* buffer3 = (float*)CpuNativeHelpers.GetBufferStart(tensor3);
			float* buffer4 = (float*)CpuNativeHelpers.GetBufferStart(tensor4);
			float* buffer5 = (float*)CpuNativeHelpers.GetBufferStart(tensor5);


			TensorIterState tensor1Iter = new TensorIterState(buffer1, tensor1.DimensionCount, tensor1.SizesMemory, tensor1.StridesMemory, step);
			TensorIterState tensor2Iter = new TensorIterState(buffer2, tensor2.DimensionCount, tensor2.SizesMemory, tensor2.StridesMemory, step);
			TensorIterState tensor3Iter = new TensorIterState(buffer3, tensor3.DimensionCount, tensor3.SizesMemory, tensor3.StridesMemory, step);
			TensorIterState tensor4Iter = new TensorIterState(buffer4, tensor4.DimensionCount, tensor4.SizesMemory, tensor4.StridesMemory, step);
			TensorIterState tensor5Iter = new TensorIterState(buffer5, tensor5.DimensionCount, tensor5.SizesMemory, tensor5.StridesMemory, step);

			do
			{
				for (; !tensor1Iter.ReachedBlockEnd() && !tensor2Iter.ReachedBlockEnd() && !tensor3Iter.ReachedBlockEnd() && !tensor4Iter.ReachedBlockEnd() && !tensor5Iter.ReachedBlockEnd();
					tensor1Iter.BlockStep(), tensor2Iter.BlockStep(), tensor3Iter.BlockStep(), tensor4Iter.BlockStep(), tensor5Iter.BlockStep())
				{
					func(tensor1Iter.data, tensor2Iter.data, tensor3Iter.data, tensor4Iter.data, tensor5Iter.data);
				}

			} while (tensor1Iter.NextBlock() && tensor2Iter.NextBlock() && tensor3Iter.NextBlock() && tensor4Iter.NextBlock() && tensor5Iter.NextBlock());
		}


		// 中文：沿指定维 iterationDim 对两个张量按维迭代，对每个切片调用核函数（传入该维的 size/stride）。
		unsafe static void ApplyDim2(Tensor tensor1, Tensor tensor2, int iterationDim, ApplyDim2KernelFuncton func)
		{
			float* buffer1 = (float*)CpuNativeHelpers.GetBufferStart(tensor1);
			float* buffer2 = (float*)CpuNativeHelpers.GetBufferStart(tensor2);

			TensorDimIterState tensor1Iter = new TensorDimIterState(buffer1, tensor1.DimensionCount, tensor1.SizesMemory, tensor1.StridesMemory, iterationDim);
			TensorDimIterState tensor2Iter = new TensorDimIterState(buffer2, tensor2.DimensionCount, tensor2.SizesMemory, tensor2.StridesMemory, iterationDim);

			do
			{
				func(tensor1Iter.data, tensor1Iter.size, tensor1Iter.stride,
					tensor2Iter.data, tensor2Iter.size, tensor2Iter.stride);

			} while (tensor1Iter.NextBlock() && tensor2Iter.NextBlock());
		}




		// 中文：沿指定维 iterationDim 对三个张量按维迭代，对每个切片调用核函数（传入各自的 size/stride）。
		unsafe static void ApplyDim3(Tensor tensor1, Tensor tensor2, Tensor tensor3, int iterationDim, ApplyDim3KernelFuncton func)
		{
			float* buffer1 = (float*)CpuNativeHelpers.GetBufferStart(tensor1);
			float* buffer2 = (float*)CpuNativeHelpers.GetBufferStart(tensor2);
			float* buffer3 = (float*)CpuNativeHelpers.GetBufferStart(tensor3);

			TensorDimIterState tensor1Iter = new TensorDimIterState(buffer1, tensor1.DimensionCount, tensor1.SizesMemory, tensor1.StridesMemory, iterationDim);
			TensorDimIterState tensor2Iter = new TensorDimIterState(buffer2, tensor2.DimensionCount, tensor2.SizesMemory, tensor2.StridesMemory, iterationDim);
			TensorDimIterState tensor3Iter = new TensorDimIterState(buffer3, tensor3.DimensionCount, tensor3.SizesMemory, tensor3.StridesMemory, iterationDim);

			do
			{
				func(tensor1Iter.data, tensor1Iter.size, tensor1Iter.stride,
					tensor2Iter.data, tensor2Iter.size, tensor2Iter.stride,
					tensor3Iter.data, tensor3Iter.size, tensor3Iter.stride);

			} while (tensor1Iter.NextBlock() && tensor2Iter.NextBlock() && tensor3Iter.NextBlock());
		}


        #endregion

        // True iff all three tensors are 2D contiguous float32 with matching
        // outer shape (i.e. a typical [N,D] gather/scatter layout). The fast
        // paths below bypass the generic TensorDimIterState walk: they iterate
        // rows in the outer loop (cache-friendly writes) and detect the
        // embedding-style "all indices in a row equal" pattern to emit a
        // single row memcpy instead of D scalar copies.
        // 中文：判断 result/src/indices 是否均为 2D 连续 Float32 且外形匹配，若是则取出裸指针与 N/D 维度走 gather/scatter 快速路径。
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe bool TryGetGatherScatter2DFast(
            Tensor result, Tensor src, Tensor indices, int dim,
            out float* rPtr, out float* sPtr, out float* iPtr,
            out long N, out long D, out long otherDim)
        {
            rPtr = sPtr = iPtr = null;
            N = D = otherDim = 0;

            if (result == null || src == null || indices == null) return false;
            if (result.DimensionCount != 2 || src.DimensionCount != 2 || indices.DimensionCount != 2) return false;
            if (dim != 0 && dim != 1) return false;
            if (result.ElementType != DType.Float32 || src.ElementType != DType.Float32) return false;
            if (indices.ElementType != DType.Float32) return false;
            if (!result.IsContiguous() || !src.IsContiguous() || !indices.IsContiguous()) return false;

            // Shape preconditions match the public API: result.shape == indices.shape,
            // and src/result agree on every dim except `dim`.
            if (result.Sizes[0] != indices.Sizes[0] || result.Sizes[1] != indices.Sizes[1]) return false;
            int otherIdx = dim == 0 ? 1 : 0;
            if (src.Sizes[otherIdx] != result.Sizes[otherIdx]) return false;

            N = result.Sizes[0];
            D = result.Sizes[1];
            otherDim = src.Sizes[dim];

            rPtr = (float*)CpuNativeHelpers.GetBufferStart(result);
            sPtr = (float*)CpuNativeHelpers.GetBufferStart(src);
            iPtr = (float*)CpuNativeHelpers.GetBufferStart(indices);
            return true;
        }

        // 中文：沿 dim 按 indices 收集（gather）元素到 result，含 2D 连续快速路径与嵌入式整行 memcpy 优化。
        unsafe public static void Gather(Tensor result, Tensor src, int dim, Tensor indices)
		{
            if (TryGetGatherScatter2DFast(result, src, indices, dim,
                out float* rPtr, out float* sPtr, out float* iPtr,
                out long N, out long D, out long sSize))
            {
                if (dim == 0)
                {
                    // result[i, j] = src[indices[i, j], j]
                    for (long i = 0; i < N; i++)
                    {
                        float* indRow = iPtr + i * D;
                        float* resRow = rPtr + i * D;
                        long firstIdx = (long)indRow[0];
                        if (firstIdx < 0 || firstIdx >= sSize)
                            throw new IndexOutOfRangeException($"Invalid index in gather. Idx = '{firstIdx}', sSize = '{sSize}'");

                        // Detect embedding-style uniform-row indices and emit
                        // a single row memcpy.
                        bool uniform = true;
                        for (long j = 1; j < D; j++)
                        {
                            if ((long)indRow[j] != firstIdx) { uniform = false; break; }
                        }
                        if (uniform)
                        {
                            long bytes = D * sizeof(float);
                            Buffer.MemoryCopy(sPtr + firstIdx * D, resRow, bytes, bytes);
                        }
                        else
                        {
                            for (long j = 0; j < D; j++)
                            {
                                long idx = (long)indRow[j];
                                if (idx < 0 || idx >= sSize)
                                    throw new IndexOutOfRangeException($"Invalid index in gather. Idx = '{idx}', sSize = '{sSize}'");
                                resRow[j] = sPtr[idx * D + j];
                            }
                        }
                    }
                }
                else // dim == 1: result[i, j] = src[i, indices[i, j]]
                {
                    long srcCols = sSize;
                    for (long i = 0; i < N; i++)
                    {
                        float* srcRow = sPtr + i * srcCols;
                        float* indRow = iPtr + i * D;
                        float* resRow = rPtr + i * D;
                        for (long j = 0; j < D; j++)
                        {
                            long idx = (long)indRow[j];
                            if (idx < 0 || idx >= srcCols)
                                throw new IndexOutOfRangeException($"Invalid index in gather. Idx = '{idx}', sSize = '{srcCols}'");
                            resRow[j] = srcRow[idx];
                        }
                    }
                }
                return;
            }

			unsafe void func(float* rData, long rSize, long rStride,
				float* sData, long sSize2, long sStride,
				float* iData, long iSize, long iStride)
			{
				for (int i = 0; i < iSize; ++i)
				{
					long idx = (long)*(iData + i * iStride);
					if (idx < 0 || idx >= sSize2) { throw new IndexOutOfRangeException($"Invalid index in gather. Idx = '{idx}', sSize = '{sSize2}'"); }

					*(rData + i * rStride) = sData[idx * sStride];
				}
			}

			ApplyDim3(result, src, indices, dim, func);
		}



		// 中文：沿 dim 按 indices 把 src 元素散布（scatter）写入 result，含 2D 连续快速路径与整行 memcpy 优化。
		unsafe public static void Scatter(Tensor result, Tensor src, int dim, Tensor indices)
		{
            if (TryGetGatherScatter2DFast(result, src, indices, dim,
                out float* rPtr, out float* sPtr, out float* iPtr,
                out long N, out long D, out long rSize))
            {
                if (dim == 0)
                {
                    // result[indices[i, j], j] = src[i, j]
                    for (long i = 0; i < N; i++)
                    {
                        float* indRow = iPtr + i * D;
                        float* srcRow = sPtr + i * D;
                        long firstIdx = (long)indRow[0];
                        if (firstIdx < 0 || firstIdx >= rSize)
                            throw new IndexOutOfRangeException($"Invalid index in scatter. Idx = '{firstIdx}', rSize = '{rSize}'");

                        bool uniform = true;
                        for (long j = 1; j < D; j++)
                        {
                            if ((long)indRow[j] != firstIdx) { uniform = false; break; }
                        }
                        if (uniform)
                        {
                            long bytes = D * sizeof(float);
                            Buffer.MemoryCopy(srcRow, rPtr + firstIdx * D, bytes, bytes);
                        }
                        else
                        {
                            for (long j = 0; j < D; j++)
                            {
                                long idx = (long)indRow[j];
                                if (idx < 0 || idx >= rSize)
                                    throw new IndexOutOfRangeException($"Invalid index in scatter. Idx = '{idx}', rSize = '{rSize}'");
                                rPtr[idx * D + j] = srcRow[j];
                            }
                        }
                    }
                }
                else // dim == 1: result[i, indices[i, j]] = src[i, j]
                {
                    long resCols = rSize;
                    for (long i = 0; i < N; i++)
                    {
                        float* resRow = rPtr + i * resCols;
                        float* indRow = iPtr + i * D;
                        float* srcRow = sPtr + i * D;
                        for (long j = 0; j < D; j++)
                        {
                            long idx = (long)indRow[j];
                            if (idx < 0 || idx >= resCols)
                                throw new IndexOutOfRangeException($"Invalid index in scatter. Idx = '{idx}', rSize = '{resCols}'");
                            resRow[idx] = srcRow[j];
                        }
                    }
                }
                return;
            }

			unsafe void func(float* rData, long rSize2, long rStride,
				float* sData, long sSize, long sStride,
				float* iData, long iSize, long iStride)
			{

				for (int i = 0; i < iSize; ++i)
				{
					long idx = (long)*(iData + i * iStride);
					if (idx < 0 || idx >= rSize2) { throw new IndexOutOfRangeException($"Invalid index in scatter. Idx = '{idx}', rSize = '{rSize2}'"); }

					rData[idx * rStride] = *(sData + i * sStride);
				}

			}

			ApplyDim3(result, src, indices, dim, func);
		}

		// 中文：沿 dim 按 indices 把 src 元素累加散布（scatter-add）到 result，含 2D 连续快速路径与向量化整行累加。
		unsafe public static void ScatterAdd(Tensor result, Tensor src, int dim, Tensor indices)
		{
            if (TryGetGatherScatter2DFast(result, src, indices, dim,
                out float* rPtr, out float* sPtr, out float* iPtr,
                out long N, out long D, out long rSize))
            {
                if (dim == 0)
                {
                    // result[indices[i, j], j] += src[i, j]
                    for (long i = 0; i < N; i++)
                    {
                        float* indRow = iPtr + i * D;
                        float* srcRow = sPtr + i * D;
                        long firstIdx = (long)indRow[0];
                        if (firstIdx < 0 || firstIdx >= rSize)
                            throw new IndexOutOfRangeException($"Invalid index in scatter. Idx = '{firstIdx}', rSize = '{rSize}'");

                        bool uniform = true;
                        for (long j = 1; j < D; j++)
                        {
                            if ((long)indRow[j] != firstIdx) { uniform = false; break; }
                        }
                        if (uniform)
                        {
                            // Vectorized add for the destination row.
                            int vectorSize = Vector<float>.Count;
                            float* dst = rPtr + firstIdx * D;
                            long j = 0;
                            for (; j <= D - vectorSize; j += vectorSize)
                            {
                                Vector<float> a = LoadVec(dst + j);
                                Vector<float> b = LoadVec(srcRow + j);
                                StoreVec(dst + j, a + b);
                            }
                            for (; j < D; j++)
                                dst[j] += srcRow[j];
                        }
                        else
                        {
                            for (long j = 0; j < D; j++)
                            {
                                long idx = (long)indRow[j];
                                if (idx < 0 || idx >= rSize)
                                    throw new IndexOutOfRangeException($"Invalid index in scatter. Idx = '{idx}', rSize = '{rSize}'");
                                rPtr[idx * D + j] += srcRow[j];
                            }
                        }
                    }
                }
                else // dim == 1: result[i, indices[i, j]] += src[i, j]
                {
                    long resCols = rSize;
                    for (long i = 0; i < N; i++)
                    {
                        float* resRow = rPtr + i * resCols;
                        float* indRow = iPtr + i * D;
                        float* srcRow = sPtr + i * D;
                        for (long j = 0; j < D; j++)
                        {
                            long idx = (long)indRow[j];
                            if (idx < 0 || idx >= resCols)
                                throw new IndexOutOfRangeException($"Invalid index in scatter. Idx = '{idx}', rSize = '{resCols}'");
                            resRow[idx] += srcRow[j];
                        }
                    }
                }
                return;
            }

			unsafe void func(float* rData, long rSize2, long rStride,
				float* sData, long sSize, long sStride,
				float* iData, long iSize, long iStride)
			{

				for (int i = 0; i < iSize; ++i)
				{
					long idx = (long)*(iData + i * iStride);
					if (idx < 0 || idx >= rSize2) { throw new IndexOutOfRangeException($"Invalid index in scatter. Idx = '{idx}', rSize = '{rSize2}'"); }

					rData[idx * rStride] += *(sData + i * sStride);
				}

			}

			ApplyDim3(result, src, indices, dim, func);
		}

		// 中文：沿 dim 按 indices 指定位置把 result 填充为常量 value（无 src），含 2D 连续快速路径。
		unsafe public static void ScatterFill(Tensor result, float value, int dim, Tensor indices)
		{
            // 2D-contig fast path for ScatterFill (no src tensor).
            if (result != null && indices != null
                && result.DimensionCount == 2 && indices.DimensionCount == 2
                && (dim == 0 || dim == 1)
                && result.ElementType == DType.Float32 && indices.ElementType == DType.Float32
                && result.IsContiguous() && indices.IsContiguous()
                && indices.Sizes[1 - dim] == result.Sizes[1 - dim])
            {
                long N = indices.Sizes[0];
                long D = indices.Sizes[1];
                long otherDim = result.Sizes[dim];

                float* rPtrFast = (float*)CpuNativeHelpers.GetBufferStart(result);
                float* iPtrFast = (float*)CpuNativeHelpers.GetBufferStart(indices);

                if (dim == 0)
                {
                    long resCols = result.Sizes[1];
                    for (long i = 0; i < N; i++)
                    {
                        float* indRow = iPtrFast + i * D;
                        for (long j = 0; j < D; j++)
                        {
                            long idx = (long)indRow[j];
                            if (idx < 0 || idx >= otherDim)
                                throw new IndexOutOfRangeException($"Invalid index in ScatterFill. Idx = '{idx}', rSize = '{otherDim}'");
                            rPtrFast[idx * resCols + j] = value;
                        }
                    }
                }
                else // dim == 1
                {
                    long resCols = result.Sizes[1];
                    for (long i = 0; i < N; i++)
                    {
                        float* resRow = rPtrFast + i * resCols;
                        float* indRow = iPtrFast + i * D;
                        for (long j = 0; j < D; j++)
                        {
                            long idx = (long)indRow[j];
                            if (idx < 0 || idx >= resCols)
                                throw new IndexOutOfRangeException($"Invalid index in ScatterFill. Idx = '{idx}', rSize = '{resCols}'");
                            resRow[idx] = value;
                        }
                    }
                }
                return;
            }

			unsafe void func(float* rData, long rSize, long rStride, float* iData, long iSize, long iStride)
			{
				for (int i = 0; i < iSize; ++i)
				{
					long idx = (long)*(iData + i * iStride);
					if (idx < 0 || idx >= rSize) { throw new IndexOutOfRangeException($"Invalid index in ScatterFill. Idx = '{idx}', rSize = '{rSize}'"); }

					rData[idx * rStride] = value;
				}

			}

			ApplyDim2(result, indices, dim, func);
		}



		// 中文：把整个张量填充为常量 value，含 Float32 连续、Float16、Q8_0（仅置零）等多种类型快速路径。
		unsafe public static void Fill(Tensor result, float value)
		{
            if (TryGetContiguousFloat(result, out float* resultPtr, out int length))
            {
                new Span<float>(resultPtr, length).Fill(value);
                return;
            }

            // The generic Apply1 path treats every element as 4 bytes, so
            // routing a Float16 (or block-quantized Q8_0) tensor through it
            // walks past the storage buffer and surfaces as an
            // AccessViolationException during KV-cache zero-init. Mirror the
            // GGML backend's contiguous F16/Q8_0 fast paths so non-F32 caches
            // can be filled safely on the managed CPU backend.
            if (result.ElementType == DType.Float16 && IsContiguousNonNarrowed(result))
            {
                ushort halfBits = BitConverter.HalfToUInt16Bits((System.Half)value);
                ushort* halfBuffer = (ushort*)CpuNativeHelpers.GetBufferStart(result);
                long elementCount = result.ElementCount();
                if (halfBits == 0)
                {
                    long offset = 0;
                    while (offset < elementCount)
                    {
                        int slice = (int)Math.Min(elementCount - offset, int.MaxValue);
                        new Span<ushort>(halfBuffer + offset, slice).Clear();
                        offset += slice;
                    }
                }
                else
                {
                    for (long i = 0; i < elementCount; i++)
                        halfBuffer[i] = halfBits;
                }
                return;
            }

            if (result.ElementType == DType.Q8_0 && IsContiguousNonNarrowed(result))
            {
                if (value != 0f)
                    throw new NotSupportedException("Fill on Q8_0 tensors only supports value=0 (cache reset).");
                long byteLength = DTypeExtensions.Q8_0Bytes(result.ElementCount());
                byte* byteBuffer = (byte*)CpuNativeHelpers.GetBufferStart(result);
                long offset = 0;
                while (offset < byteLength)
                {
                    int slice = (int)Math.Min(byteLength - offset, int.MaxValue);
                    new Span<byte>(byteBuffer + offset, slice).Clear();
                    offset += slice;
                }
                return;
            }

            if (result.ElementType != DType.Float32)
                throw new NotSupportedException(
                    $"Fill on {result.ElementType} tensors requires a contiguous, non-narrowed layout.");

			unsafe void func(float* r)
			{
				*r = value;
			}

			Apply1(result, func);
		}

        // 中文：检查张量是否为零偏移、行主序紧密连续（未被切片窄化）的布局。
        private static bool IsContiguousNonNarrowed(Tensor t)
        {
            if (t.StorageOffset != 0) return false;
            long expected = 1;
            for (int d = t.DimensionCount - 1; d >= 0; d--)
            {
                if (t.Strides[d] != expected) return false;
                expected *= t.Sizes[d];
            }
            return expected == t.ElementCount();
        }


		// 中文：逐元素把 src 截断到 [min,max] 区间写入 result。
		unsafe public static void Clamp(Tensor result, Tensor src, float min, float max)
		{
			unsafe void func(float* r, float* s)
			{
				*r = clamp(*s, min, max);
			}
			Apply2(result, src, func);
		}


		// 中文：把 src 拷贝到 result，含连续同类型整块 memcpy、末维向量化与通用跨步逐元素三种路径。
		unsafe public static void Copy(Tensor result, Tensor src)
		{
            if (result.IsContiguous() && src.IsContiguous() &&
                result.ElementType == src.ElementType &&
                result.ElementCount() == src.ElementCount())
            {
                long byteCount = result.ElementCount() * result.ElementType.Size();
                if (byteCount <= int.MaxValue)
                {
                    byte* srcBytes = (byte*)CpuNativeHelpers.GetBufferStart(src);
                    byte* resultBytes = (byte*)CpuNativeHelpers.GetBufferStart(result);
                    new ReadOnlySpan<byte>(srcBytes, (int)byteCount).CopyTo(new Span<byte>(resultBytes, (int)byteCount));
                }
                else
                {
                    Buffer.MemoryCopy(
                        CpuNativeHelpers.GetBufferStart(src).ToPointer(),
                        CpuNativeHelpers.GetBufferStart(result).ToPointer(),
                        byteCount,
                        byteCount);
                }
                return;
            }

			int vectorSize = Vector<float>.Count;
			if (result.Strides[^1] == 1 && src.Strides[^1] == 1 && result.Sizes[^1] % vectorSize == 0)
			{
				unsafe void funcVec(float* r, float* s)
				{
					Span<float> spanR = new Span<float>(r, vectorSize);
					Span<float> spanS = new Span<float>(s, vectorSize);

					Vector<float> vecS = new Vector<float>(spanS);
					vecS.CopyTo(spanR);
				}

				Apply2(result, src, funcVec, vectorSize);
			}
			else
			{
				unsafe void func(float* r, float* s)
				{
					*r = *s;
				}
				Apply2(result, src, func);
			}
		}


		// 中文：沿指定维归约求和，将每个切片元素累加写入 result。
		unsafe public static void Sum(Tensor result, Tensor src, int dimension)
		{
			unsafe void func(float* r, long rSize, long rStride, float* s, long sSize, long sStride)
			{
				float sum = 0.0f;
				for (long i = 0; i < sSize; ++i)
				{
					sum += s[i * sStride];
				}
				*r = sum;
			}
			ApplyDim2(result, src, dimension, func);
		}


		// 中文：沿指定维归约求均值（和除以该维长度）写入 result。
		unsafe public static void Mean(Tensor result, Tensor src, int dimension)
		{
			unsafe void func(float* r, long rSize, long rStride, float* s, long sSize, long sStride)
			{
				float sum = 0.0f;
				for (long i = 0; i < sSize; ++i)
				{
					sum += s[i * sStride];
				}
				*r = sum / sSize;
			}
			ApplyDim2(result, src, dimension, func);
		}


		// 中文：沿指定维求最大值的下标（argmax），把索引写入 resultIndices。
		unsafe public static void Argmax(Tensor resultIndices, Tensor src, int dimension)
		{

			unsafe void func(float* rIndVal, long rIndSize, long rIndStride,
				float* s, long sSize, long sStride)
			{
				float value = s[0];
				float index = 0;
				for (long i = 1; i < sSize; ++i)
				{
					float currentVal = s[i * sStride];
					if (currentVal > value)
					{
						value = currentVal;
						index = (float)i;
					}
				}
				*rIndVal = index;
			}

			ApplyDim2(resultIndices, src, dimension, func);
		}

		// 中文：沿指定维归约求最大值写入 result。
		unsafe public static void Max(Tensor result, Tensor src, int dimension)
		{
			unsafe void func(float* r, long rSize, long rStride, float* s, long sSize, long sStride)
			{
				float value = s[0];
				for (long i = 1; i < sSize; ++i)
				{
					value = Math.Max(value, s[i * sStride]);
				}
				*r = value;
			}

			ApplyDim2(result, src, dimension, func);
		}


		// 中文：逐元素张量相加 result=lhs+rhs，含连续 SIMD 快速路径与末维向量化/通用路径。
		unsafe public static void Add(Tensor result, Tensor lhs, Tensor rhs)
		{
            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(lhs, out float* lhsPtr, out int lhsLength) &&
                TryGetContiguousFloat(rhs, out float* rhsPtr, out int rhsLength) &&
                length == lhsLength && length == rhsLength)
            {
                int simdWidth = Vector<float>.Count;
                int i = 0;
                for (; i <= length - simdWidth; i += simdWidth)
                {
                    StoreVec(resultPtr + i, LoadVec(lhsPtr + i) + LoadVec(rhsPtr + i));
                }

                for (; i < length; i++)
                {
                    resultPtr[i] = lhsPtr[i] + rhsPtr[i];
                }

                return;
            }

			int vectorSize = Vector<float>.Count;
			if (result.Strides[^1] == 1 && lhs.Strides[^1] == 1 && rhs.Strides[^1] == 1 && result.Sizes[^1] % vectorSize == 0)
			{
				unsafe void funcVec(float* r, float* left, float* right)
				{
					Span<float> spanR = new Span<float>(r, vectorSize);
					Span<float> spanLeft = new Span<float>(left, vectorSize);
					Span<float> spanRight = new Span<float>(right, vectorSize);

					Vector<float> vecLeft = new Vector<float>(spanLeft);
					Vector<float> vecRight = new Vector<float>(spanRight);

					Vector<float> vecR = vecLeft + vecRight;
					vecR.CopyTo(spanR);

				}

				Apply3(result, lhs, rhs, funcVec, vectorSize);
			}
			else
			{
				unsafe void func(float* r, float* left, float* right)
				{
					*r = add(*left, *right);
				}

				Apply3(result, lhs, rhs, func);
			}
		}


		// 中文：逐元素张量相减 result=lhs-rhs，含连续 SIMD 快速路径与末维向量化/通用路径。
		unsafe public static void Sub(Tensor result, Tensor lhs, Tensor rhs)
		{
            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(lhs, out float* lhsPtr, out int lhsLength) &&
                TryGetContiguousFloat(rhs, out float* rhsPtr, out int rhsLength) &&
                length == lhsLength && length == rhsLength)
            {
                int simdWidth = Vector<float>.Count;
                int i = 0;
                for (; i <= length - simdWidth; i += simdWidth)
                {
                    StoreVec(resultPtr + i, LoadVec(lhsPtr + i) - LoadVec(rhsPtr + i));
                }

                for (; i < length; i++)
                {
                    resultPtr[i] = lhsPtr[i] - rhsPtr[i];
                }

                return;
            }

			int vectorSize = Vector<float>.Count;
			if (result.Strides[^1] == 1 && lhs.Strides[^1] == 1 && rhs.Strides[^1] == 1 && result.Sizes[^1] % vectorSize == 0)
			{
				unsafe void funcVec(float* r, float* left, float* right)
				{
					Span<float> spanR = new Span<float>(r, vectorSize);
					Span<float> spanLeft = new Span<float>(left, vectorSize);
					Span<float> spanRight = new Span<float>(right, vectorSize);

					Vector<float> vecLeft = new Vector<float>(spanLeft);
					Vector<float> vecRight = new Vector<float>(spanRight);

					Vector<float> vecR = vecLeft - vecRight;
					vecR.CopyTo(spanR);

				}

				Apply3(result, lhs, rhs, funcVec, vectorSize);
			}
			else
			{
				unsafe void func(float* r, float* left, float* right)
				{
					*r = *left - *right;
				}

				Apply3(result, lhs, rhs, func);
			}
		}

		// 中文：逐元素张量加标量 result=src+value，含连续 SIMD 快速路径与末维向量化/通用路径。
		unsafe public static void Add(Tensor result, Tensor src, float value)
		{
            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(src, out float* srcPtr, out int srcLength) &&
                length == srcLength)
            {
                int simdWidth = Vector<float>.Count;
                Vector<float> vecValue = new Vector<float>(value);
                int i = 0;
                for (; i <= length - simdWidth; i += simdWidth)
                {
                    StoreVec(resultPtr + i, LoadVec(srcPtr + i) + vecValue);
                }

                for (; i < length; i++)
                {
                    resultPtr[i] = srcPtr[i] + value;
                }

                return;
            }

			int vectorSize = Vector<float>.Count;
			if (result.Strides[^1] == 1 && src.Strides[^1] == 1 && result.Sizes[^1] % vectorSize == 0)
			{
				Vector<float> vecV = new Vector<float>(value);
				unsafe void funcVec(float* r, float* s)
				{
					Span<float> spanR = new Span<float>(r, vectorSize);
					Span<float> spanS = new Span<float>(s, vectorSize);

					Vector<float> vecS = new Vector<float>(spanS);

					Vector<float> vecR = vecS + vecV;
					vecR.CopyTo(spanR);

				}

				Apply2(result, src, funcVec, vectorSize);
			}
			else
			{
				unsafe void func(float* r, float* s)
				{
					*r = add(*s, value);
				}

				Apply2(result, src, func);
			}
		}


		// 中文：逐元素求幂 result=src^value。
		unsafe public static void Pow(Tensor result, Tensor src, float value)
		{
				unsafe void func(float* r, float* s)
				{
					*r = (float)Math.Pow(*s, value);
				}

				Apply2(result, src, func);			
		}


		// 中文：逐元素标量减张量 result=value-src（反向减法），含连续 SIMD 快速路径与末维向量化/通用路径。
		unsafe public static void RSub(Tensor result, float value, Tensor src)
		{
            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(src, out float* srcPtr, out int srcLength) &&
                length == srcLength)
            {
                int simdWidth = Vector<float>.Count;
                Vector<float> vecValue = new Vector<float>(value);
                int i = 0;
                for (; i <= length - simdWidth; i += simdWidth)
                {
                    StoreVec(resultPtr + i, vecValue - LoadVec(srcPtr + i));
                }

                for (; i < length; i++)
                {
                    resultPtr[i] = value - srcPtr[i];
                }

                return;
            }

			int vectorSize = Vector<float>.Count;
			if (result.Strides[^1] == 1 && src.Strides[^1] == 1 && result.Sizes[^1] % vectorSize == 0)
			{
				Vector<float> vecV = new Vector<float>(value);
				unsafe void funcVec(float* r, float* s)
				{
					Span<float> spanR = new Span<float>(r, vectorSize);
					Span<float> spanS = new Span<float>(s, vectorSize);

					Vector<float> vecS = new Vector<float>(spanS);

					Vector<float> vecR = vecV - vecS;
					vecR.CopyTo(spanR);

				}

				Apply2(result, src, funcVec, vectorSize);
			}
			else
			{
				unsafe void func(float* r, float* s)
				{
					*r = value - *s;
				}

				Apply2(result, src, func);
			}
		}



		// 中文：逐元素张量乘标量 result=src*value，含连续 SIMD 快速路径与末维向量化/通用路径。
		unsafe public static void Mul(Tensor result, Tensor src, float value)
		{
            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(src, out float* srcPtr, out int srcLength) &&
                length == srcLength)
            {
                int simdWidth = Vector<float>.Count;
                Vector<float> vecValue = new Vector<float>(value);
                int i = 0;
                for (; i <= length - simdWidth; i += simdWidth)
                {
                    StoreVec(resultPtr + i, LoadVec(srcPtr + i) * vecValue);
                }

                for (; i < length; i++)
                {
                    resultPtr[i] = srcPtr[i] * value;
                }

                return;
            }

			int vectorSize = Vector<float>.Count;
			if (result.Strides[^1] == 1 && src.Strides[^1] == 1 && result.Sizes[^1] % vectorSize == 0)
			{
				Vector<float> vecV = new Vector<float>(value);
				unsafe void funcVec(float* r, float* s)
				{
					Span<float> spanR = new Span<float>(r, vectorSize);
					Span<float> spanS = new Span<float>(s, vectorSize);

					Vector<float> vecS = new Vector<float>(spanS);

					Vector<float> vecR = vecS * vecV;
					vecR.CopyTo(spanR);
				}

				Apply2(result, src, funcVec, vectorSize);
			}
			else
			{
				unsafe void func(float* r, float* s)
				{
					*r = mul(*s, value);
				}

				Apply2(result, src, func);
			}
		}


		// 中文：逐元素张量除以标量 result=lhs/rhs，含连续 SIMD 快速路径与末维向量化/通用路径。
		unsafe public static void Div(Tensor result, Tensor lhs, float rhs)
		{
            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(lhs, out float* lhsPtr, out int lhsLength) &&
                length == lhsLength)
            {
                int simdWidth = Vector<float>.Count;
                Vector<float> vecValue = new Vector<float>(rhs);
                int i = 0;
                for (; i <= length - simdWidth; i += simdWidth)
                {
                    StoreVec(resultPtr + i, LoadVec(lhsPtr + i) / vecValue);
                }

                for (; i < length; i++)
                {
                    resultPtr[i] = lhsPtr[i] / rhs;
                }

                return;
            }

			int vectorSize = Vector<float>.Count;
			if (result.Strides[^1] == 1 && lhs.Strides[^1] == 1 && result.Sizes[^1] % vectorSize == 0)
			{
				Vector<float> vecV = new Vector<float>(rhs);
				unsafe void funcVec(float* r, float* s)
				{
					Span<float> spanR = new Span<float>(r, vectorSize);
					Span<float> spanS = new Span<float>(s, vectorSize);

					Vector<float> vecS = new Vector<float>(spanS);

					Vector<float> vecR = vecS / vecV;
					vecR.CopyTo(spanR);
				}

				Apply2(result, lhs, funcVec, vectorSize);
			}
			else
			{
				unsafe void func(float* r, float* s)
				{
					*r = div(*s, rhs);
				}

				Apply2(result, lhs, func);
			}
		}

		// 中文：逐元素张量相乘 result=lhs*rhs，含连续 SIMD 快速路径与末维向量化/通用路径。
		unsafe public static void Mul(Tensor result, Tensor lhs, Tensor rhs)
		{
            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(lhs, out float* lhsPtr, out int lhsLength) &&
                TryGetContiguousFloat(rhs, out float* rhsPtr, out int rhsLength) &&
                length == lhsLength && length == rhsLength)
            {
                int simdWidth = Vector<float>.Count;
                int i = 0;
                for (; i <= length - simdWidth; i += simdWidth)
                {
                    StoreVec(resultPtr + i, LoadVec(lhsPtr + i) * LoadVec(rhsPtr + i));
                }

                for (; i < length; i++)
                {
                    resultPtr[i] = lhsPtr[i] * rhsPtr[i];
                }

                return;
            }

			int vectorSize = Vector<float>.Count;
			if (result.Strides[^1] == 1 && lhs.Strides[^1] == 1 && rhs.Strides[^1] == 1 && result.Sizes[^1] % vectorSize == 0)
			{
				unsafe void funcVec(float* r, float* left, float* right)
				{
					Span<float> spanR = new Span<float>(r, vectorSize);
					Span<float> spanLeft = new Span<float>(left, vectorSize);
					Span<float> spanRight = new Span<float>(right, vectorSize);

					Vector<float> vecLeft = new Vector<float>(spanLeft);
					Vector<float> vecRight = new Vector<float>(spanRight);

					Vector<float> vecR = vecLeft * vecRight;
					vecR.CopyTo(spanR);

				}

				Apply3(result, lhs, rhs, funcVec, vectorSize);
			}
			else
			{
				unsafe void func(float* r, float* left, float* right)
				{
					*r = mul(*left, *right);
				}

				Apply3(result, lhs, rhs, func);
			}
		}


		// 中文：逐元素张量相除 result=lhs/rhs，含末维向量化与通用跨步路径。
		unsafe public static void Div(Tensor result, Tensor lhs, Tensor rhs)
		{
			int vectorSize = Vector<float>.Count;
			if (result.Strides[^1] == 1 && lhs.Strides[^1] == 1 && rhs.Strides[^1] == 1 && result.Sizes[^1] % vectorSize == 0)
			{
				unsafe void funcVec(float* r, float* left, float* right)
				{
					Span<float> spanR = new Span<float>(r, vectorSize);
					Span<float> spanLeft = new Span<float>(left, vectorSize);
					Span<float> spanRight = new Span<float>(right, vectorSize);

					Vector<float> vecLeft = new Vector<float>(spanLeft);
					Vector<float> vecRight = new Vector<float>(spanRight);

					Vector<float> vecR = vecLeft / vecRight;
					vecR.CopyTo(spanR);

				}

				Apply3(result, lhs, rhs, funcVec, vectorSize);
			}
			else
			{
				unsafe void func(float* r, float* left, float* right)
				{
					*r = div(*left, *right);
				}

				Apply3(result, lhs, rhs, func);
			}
		}

		// 中文：逐元素 ReLU（max(x,0)），含末维向量化与通用路径。
		unsafe static public void Relu(Tensor result, Tensor src)
		{
			int vectorSize = Vector<float>.Count;
			if (result.Strides[^1] == 1 && src.Strides[^1] == 1 && result.Sizes[^1] % vectorSize == 0)
			{
				unsafe void funcVec(float* r, float* s)
				{
					Span<float> spanR = new Span<float>(r, vectorSize);
					Span<float> spanS = new Span<float>(s, vectorSize);

					Vector<float> vecS = new Vector<float>(spanS);

					Vector<float> vecR = Vector.Max(vecS, Vector<float>.Zero);
					vecR.CopyTo(spanR);
				}

				Apply2(result, src, funcVec, vectorSize);
			}
			else
			{
				unsafe void func(float* r, float* s)
				{
					*r = relu(*s);
				};

				Apply2(result, src, func);
			}
		}


		// 中文：逐元素取绝对值。
		unsafe static public void Abs(Tensor result, Tensor src)
		{
			unsafe void func(float* r, float* s)
			{
				*r = Math.Abs(*s);
			}
			Apply2(result, src, func);
		}

		// 中文：逐元素取负。
		unsafe static public void Neg(Tensor result, Tensor src)
		{
			unsafe void func(float* r, float* s)
			{
				*r = -(*s);
			}
			Apply2(result, src, func);
		}

		// 中文：逐元素求平方根。
		unsafe static public void Sqrt(Tensor result, Tensor src)
		{
			unsafe void func(float* r, float* s)
			{
				*r = (float)Math.Sqrt(*s);
			}
			Apply2(result, src, func);
		}

		// 中文：逐元素求平方根的倒数（1/sqrt），含末维向量化与通用路径。
		unsafe static public void Rsqrt(Tensor result, Tensor src)
		{
			int vectorSize = Vector<float>.Count;
			if (result.Strides[^1] == 1 && src.Strides[^1] == 1 && result.Sizes[^1] % vectorSize == 0)
			{
				unsafe void funcVec(float* r, float* s)
				{
					Span<float> spanR = new Span<float>(r, vectorSize);
					Span<float> spanS = new Span<float>(s, vectorSize);

					Vector<float> vecS = new Vector<float>(spanS);

					Vector<float> vecR = Vector<float>.One / Vector.SquareRoot(vecS);
					vecR.CopyTo(spanR);
				}

				Apply2(result, src, funcVec, vectorSize);
			}
			else
			{
				unsafe void func(float* r, float* s)
				{
					*r = (float)(1.0 / Math.Sqrt(*s));
				};

				Apply2(result, src, func);
			}
		}



		// 中文：逐元素 Sigmoid，连续时走 TensorPrimitives 向量化快速路径，否则通用跨步。
		unsafe static public void Sigmoid(Tensor result, Tensor src)
		{
            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(src, out float* srcPtr, out int srcLength) &&
                length == srcLength)
            {
                TensorPrimitives.Sigmoid(new ReadOnlySpan<float>(srcPtr, length), new Span<float>(resultPtr, length));
                return;
            }

			unsafe void func(float* r, float* s)
			{
				*r = sigmoid(*s);
			};

			Apply2(result, src, func);
		}


		// 中文：逐元素计算 Sigmoid 反向梯度并加到 t 上（四元 apply）。
		unsafe static public void AddSigmoidD(Tensor result, Tensor t, Tensor resW, Tensor resG)
		{
			unsafe void func(float* r, float* x, float* y, float* z)
			{
				*r = addSigmoidD(*x, *y, *z);
			}

			Apply4(result, t, resW, resG, func);
		}



		// 中文：逐元素计算 Sigmoid 反向梯度（result=sigmoidD(resW,resG)），三元 apply。
		unsafe static public void SigmoidD(Tensor result, Tensor resW, Tensor resG)
		{
			unsafe void func(float* r, float* x, float* y)
			{
				*r = sigmoidD(*x, *y);
			}

			Apply3(result, resW, resG, func);
		}



		// 中文：逐元素 Tanh，连续时走标量循环快速路径，否则通用跨步。
		unsafe static public void Tanh(Tensor result, Tensor src)
		{
            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(src, out float* srcPtr, out int srcLength) &&
                length == srcLength)
            {
                for (int i = 0; i < length; i++)
                {
                    resultPtr[i] = MathF.Tanh(srcPtr[i]);
                }

                return;
            }

			unsafe void func(float* r, float* s)
			{
				*r = MathF.Tanh(*s);
			};

			Apply2(result, src, func);
		}


		// 中文：逐元素自然对数 log，连续时走标量循环快速路径，否则通用跨步。
		unsafe static public void Log(Tensor result, Tensor src)
		{
            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(src, out float* srcPtr, out int srcLength) &&
                length == srcLength)
            {
                for (int i = 0; i < length; i++)
                {
                    resultPtr[i] = MathF.Log(srcPtr[i]);
                }

                return;
            }

			unsafe void func(float* r, float* s)
			{
				*r = MathF.Log(*s);
			};

			Apply2(result, src, func);
		}

        // 中文：逐元素自然指数 exp，连续时走 TensorPrimitives 向量化快速路径，否则通用跨步。
        unsafe static public void Exp(Tensor result, Tensor src)
        {
            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(src, out float* srcPtr, out int srcLength) &&
                length == srcLength)
            {
                TensorPrimitives.Exp(new ReadOnlySpan<float>(srcPtr, length), new Span<float>(resultPtr, length));
                return;
            }

            unsafe void func(float* r, float* s)
            {
                *r = MathF.Exp(*s);
            };

            Apply2(result, src, func);
        }

        // 中文：逐元素计算 Tanh 反向梯度（result=tanhD(resW,resG)），三元 apply。
        unsafe static public void TanhD(Tensor result, Tensor resW, Tensor resG)
		{
			unsafe void func(float* r, float* x, float* y)
			{
				*r = tanhD(*x, *y);
			}

			Apply3(result, resW, resG, func);
		}



		// 中文：逐元素计算 tanh(x+y)（result=addtanh(x,y)），三元 apply。
		unsafe static public void AddTanh(Tensor result, Tensor srcX, Tensor srcY)
		{
			unsafe void func(float* r, float* x, float* y)
			{
				*r = addtanh(*x, *y);
			}

			Apply3(result, srcX, srcY, func);
		}


		// 中文：逐元素计算 AddTanh 的反向梯度（result=addtanhD(x,y,z)），四元 apply。
		unsafe static public void AddTanhD(Tensor result, Tensor srcX, Tensor srcY, Tensor srcZ)
		{
			unsafe void func(float* r, float* x, float* y, float* z)
			{
				*r = addtanhD(*x, *y, *z);

			}
			Apply4(result, srcX, srcY, srcZ, func);
		}

		// 中文：逐元素计算 ReLU 反向梯度（result=relud(srcW,resG)），三元 apply。
		unsafe static public void ReluD(Tensor result, Tensor srcW, Tensor resG)
		{
			unsafe void func(float* r, float* y, float* x)
			{
				*r = relud(*y, *x);
			}

			Apply3(result, srcW, resG, func);
		}

		// 中文：逐元素计算 ReLU 反向梯度并累加（result=addrelud(x,w,g)），四元 apply。
		unsafe static public void AddReluD(Tensor result, Tensor srcX, Tensor srcW, Tensor srcG)
		{
			unsafe void func(float* r, float*x, float* w, float* g)
			{
				*r = addrelud(*x, *w, *g);
			}

			Apply4(result, srcX, srcW, srcG, func);
		}

		// 中文：逐元素 SiLU（x*sigmoid(x)），连续时走向量化快速路径，否则通用跨步。
		unsafe static public void SiLU(Tensor result, Tensor src)
		{
            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(src, out float* srcPtr, out int srcLength) &&
                length == srcLength)
            {
                SiLUContiguous(resultPtr, srcPtr, length);
                return;
            }

			unsafe void func(float* r, float* s)
			{
				*r = SiLU(*s);
			};

			Apply2(result, src, func);
		}

		// 中文：逐元素 GELU 激活，连续时走标量循环快速路径，否则通用跨步。
		unsafe static public void GELU(Tensor result, Tensor src)
		{
            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(src, out float* srcPtr, out int srcLength) &&
                length == srcLength)
            {
                for (int i = 0; i < length; i++)
                {
                    resultPtr[i] = GELU(srcPtr[i]);
                }

                return;
            }

			unsafe void func(float* r, float* s)
			{
				*r = GELU(*s);
			};
			Apply2(result, src, func);
		}

		// 中文：逐元素计算 GELU(gate)*up（GeGLU 融合），含连续快速路径与通用三元 apply。
		unsafe static public void GELUMul(Tensor result, Tensor gate, Tensor up)
		{
            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(gate, out float* gatePtr, out int gateLength) &&
                TryGetContiguousFloat(up, out float* upPtr, out int upLength) &&
                length == gateLength && length == upLength)
            {
                for (int i = 0; i < length; i++)
                {
                    resultPtr[i] = GELU(gatePtr[i]) * upPtr[i];
                }

                return;
            }

			unsafe void func(float* r, float* g, float* u)
			{
				*r = GELU(*g) * *u;
			}
			Apply3(result, gate, up, func);
		}

		// 中文：逐元素计算 SiLU(gate)*up（SwiGLU 融合），含连续向量化快速路径与通用三元 apply。
		unsafe static public void SiLUMul(Tensor result, Tensor gate, Tensor up)
		{
            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(gate, out float* gatePtr, out int gateLength) &&
                TryGetContiguousFloat(up, out float* upPtr, out int upLength) &&
                length == gateLength && length == upLength)
            {
                SiLUMulContiguous(resultPtr, gatePtr, upPtr, length);
                return;
            }

			unsafe void func(float* r, float* g, float* u)
			{
				*r = SiLU(*g) * *u;
			}
			Apply3(result, gate, up, func);
		}

		// 中文：逐元素计算 x*sigmoid(gate)，含连续向量化快速路径与通用三元 apply。
		unsafe static public void SigmoidMul(Tensor result, Tensor x, Tensor gate)
		{
            if (TryGetContiguousFloat(result, out float* resultPtr, out int length) &&
                TryGetContiguousFloat(x, out float* xPtr, out int xLength) &&
                TryGetContiguousFloat(gate, out float* gatePtr, out int gateLength) &&
                length == xLength && length == gateLength)
            {
                SigmoidMulContiguous(resultPtr, xPtr, gatePtr, length);
                return;
            }

			unsafe void func(float* r, float* a, float* b)
			{
				float sig = 1.0f / (1.0f + MathF.Exp(-*b));
				*r = *a * sig;
			}
			Apply3(result, x, gate, func);
		}


		// 中文：逐元素计算 SiLU 反向梯度（result=SiLUD(srcW,resG)），三元 apply。
		unsafe static public void SiLUD(Tensor result, Tensor srcW, Tensor resG)
		{
			unsafe void func(float* r, float* y, float* x)
			{
				*r = SiLUD(*y, *x);
			}

			Apply3(result, srcW, resG, func);
		}

		// 中文：逐元素计算 SiLU 反向梯度并累加（result=AddSiLUD(x,w,g)），四元 apply。
		unsafe static public void AddSiLUD(Tensor result, Tensor srcG, Tensor srcW, Tensor resG)
		{
			unsafe void func(float* r, float* x, float* w, float* g)
			{
				*r = AddSiLUD(*x, *w, *g);
			}

			Apply4(result, srcG, srcW, resG, func);
		}



        // 中文：逐元素 LeakyReLU 激活，二元 apply。
        unsafe static public void LeakyReLU(Tensor result, Tensor src)
        {
            unsafe void func(float* r, float* s)
            {
                *r = LeakyReLU(*s);
            };

            Apply2(result, src, func);
        }


        // 中文：逐元素计算 LeakyReLU 反向梯度（result=LeakyReLUD(srcW,resG)），三元 apply。
        unsafe static public void LeakyReLUD(Tensor result, Tensor srcW, Tensor resG)
        {
            unsafe void func(float* r, float* y, float* x)
            {
                *r = LeakyReLUD(*y, *x);
            }

            Apply3(result, srcW, resG, func);
        }

        // 中文：逐元素计算 LeakyReLU 反向梯度并累加（result=AddLeakyReLUD(x,w,g)），四元 apply。
        unsafe static public void AddLeakyReLUD(Tensor result, Tensor srcG, Tensor srcW, Tensor resG)
        {
            unsafe void func(float* r, float* x, float* w, float* g)
            {
                *r = AddLeakyReLUD(*x, *w, *g);
            }

            Apply4(result, srcG, srcW, resG, func);
        }


        // 中文：逐元素计算 result=x+y*val（带标量系数的乘加），三元 apply。
        unsafe static public void AddMulV(Tensor result, Tensor srcX, Tensor srcY, float val)
		{
			unsafe void func(float* r, float* x, float* y)
			{
				*r = *x + (*y * val);
			}

			Apply3(result, srcX, srcY, func);
		}


		// 中文：逐元素计算 result=x*y+z*w（两积之和），含末维向量化与通用五元 apply。
		unsafe static public void MulMulAdd(Tensor result, Tensor srcX, Tensor srcY, Tensor srcZ, Tensor srcW)
		{
			int vectorSize = Vector<float>.Count;
			if (result.Strides[^1] == 1 && srcX.Strides[^1] == 1 && srcY.Strides[^1] == 1 && srcZ.Strides[^1] == 1 && srcW.Strides[^1] == 1 && result.Sizes[^1] % vectorSize == 0)
			{
				unsafe void funcVec(float* r, float* x, float* y, float* z, float* w)
				{
					Span<float> spanR = new Span<float>(r, vectorSize);
					Span<float> spanX = new Span<float>(x, vectorSize);
					Span<float> spanY = new Span<float>(y, vectorSize);
					Span<float> spanZ = new Span<float>(z, vectorSize);
					Span<float> spanW = new Span<float>(w, vectorSize);

					Vector<float> vecX = new Vector<float>(spanX);
					Vector<float> vecY = new Vector<float>(spanY);
					Vector<float> vecZ = new Vector<float>(spanZ);
					Vector<float> vecW = new Vector<float>(spanW);

					Vector<float> vecR = vecX * vecY + vecZ * vecW;
					vecR.CopyTo(spanR);
				}

				Apply5(result, srcX, srcY, srcZ, srcW, funcVec, vectorSize);
			}
			else
			{
				unsafe void func(float* r, float* x, float* y, float* z, float* w)
				{
					*r = mulmuladd(*x, *y, *z, *w);
				}

				Apply5(result, srcX, srcY, srcZ, srcW, func);
			}
		}




		// 中文：逐元素计算 result=x+y*z（乘加），四元 apply。
		unsafe static public void AddMul(Tensor result, Tensor srcX, Tensor srcY, Tensor srcZ)
		{
			unsafe void func(float* r, float* x, float* y, float* z)
			{
				*r = addmul(*x, *y, *z);
			}

			Apply4(result, srcX, srcY, srcZ, func);
		}


		// 中文：逐元素计算 result=x+y/z（加除），四元 apply。
		unsafe static public void AddDiv(Tensor result, Tensor srcX, Tensor srcY, Tensor srcZ)
		{
			unsafe void func(float* r, float* x, float* y, float* z)
			{
				*r = adddiv(*x, *y, *z);
			}

			Apply4(result, srcX, srcY, srcZ, func);
		}


		// 中文：按各样本原始序列长度构建自注意力 padding 掩码（有效位填 value，越界位填 maskedValue）。
		unsafe static public void BuildSelfMask(Tensor result, Tensor originalLengths, int rows, int cols, int paddedSeqLen, float value, float maskedValue)
		{
			float* ptResult = (float*)CpuNativeHelpers.GetBufferStart(result);
			float* ptOriginalLengths = (float*)CpuNativeHelpers.GetBufferStart(originalLengths);

			for (int j = 0; j < rows; j++)
			{
				float* resultRow = ptResult + j * cols;
				int batchIdx = j / paddedSeqLen;
				int seqIdxInBatch = j % paddedSeqLen;

				for (int id = 0; id < cols; id++)
				{
					int originalLength = (int)ptOriginalLengths[batchIdx];
					if (id < originalLength && seqIdxInBatch < originalLength)
					{
						resultRow[id] = value;
					}
					else
					{
						resultRow[id] = maskedValue;
					}
				}
			}
		}



		// 中文：把每个切片连续重复 repeats 次写入 dst（repeat_interleave），按切片可并行 memcpy。
		unsafe static public void RepeatInterleave(float* dst, float* src, int sliceCount, int repeats, int sliceSize)
		{
            long sliceBytes = (long)sliceSize * sizeof(float);

            void CopySlice(int i)
            {
                float* srcSlice = src + i * sliceSize;
                float* dstSlice = dst + i * repeats * sliceSize;
                for (int r = 0; r < repeats; r++)
                {
                    Buffer.MemoryCopy(srcSlice, dstSlice + r * sliceSize, sliceBytes, sliceBytes);
                }
            }

            if (ShouldParallelize(sliceCount, repeats * sliceSize))
            {
                Parallel.For(0, sliceCount, CopySlice);
            }
            else
            {
                for (int i = 0; i < sliceCount; i++)
                {
                    CopySlice(i);
                }
            }
		}

	// 中文：对每行将未来位置（超过因果阈值的列）加上/填为 maskedValue，实现因果（causal）掩码，可并行。
	unsafe static public void AddCausalMask(float* data, int totalRows, int cols, int seqLen, int startPos, float maskedValue)
	{
        void MaskRow(int row)
        {
            int t = row % seqLen;
            int threshold = startPos + t;
            int sStart = Math.Max(0, threshold + 1);
            if (sStart >= cols)
            {
                return;
            }

            float* rowPtr = data + row * cols;
            if (float.IsNegativeInfinity(maskedValue))
            {
                new Span<float>(rowPtr + sStart, cols - sStart).Fill(float.NegativeInfinity);
                return;
            }

            for (int s = sStart; s < cols; s++)
            {
                rowPtr[s] += maskedValue;
            }
        }

        if (ShouldParallelize(totalRows, cols))
        {
            Parallel.For(0, totalRows, MaskRow);
        }
        else
        {
            for (int row = 0; row < totalRows; row++)
            {
                MaskRow(row);
            }
        }
	}

		// 中文：构建下三角因果掩码（id<=j 填 value，其余填 maskedValue）。
		unsafe static public void BuildTriMask(Tensor result, int rows, int cols, float value, float maskedValue)
		{
			float* ptResult = (float*)CpuNativeHelpers.GetBufferStart(result);

			for (int j = 0; j < rows; j++)
			{
				float* resultRow = ptResult + j * cols;
				for (int id = 0; id < cols; id++)
				{

					if (id <= j)
					{
						resultRow[id] = value;
					}
					else
					{
						resultRow[id] = maskedValue;
					}
				}
			}
		}



		// 中文：构建结合 padding 长度与下三角因果约束的自注意力掩码。
		unsafe static public void BuildSelfTriMask(Tensor result, Tensor originalLengths, int rows, int cols, int paddedSeqLen, float value, float maskedValue)
		{
			float* ptResult = (float*)CpuNativeHelpers.GetBufferStart(result);
			float* ptOriginalLengths = (float*)CpuNativeHelpers.GetBufferStart(originalLengths);

			for (int j = 0; j < rows; j++)
			{
				float* resultRow = ptResult + j * cols;
				int batchIdx = j / paddedSeqLen;
				int seqIdxInBatch = j % paddedSeqLen;

				for (int id = 0; id < cols; id++)
				{
					int originalLength = (int)ptOriginalLengths[batchIdx];
					if (id < originalLength && seqIdxInBatch < originalLength && id <= seqIdxInBatch)
					{
						resultRow[id] = value;
					}
					else
					{
						resultRow[id] = maskedValue;
					}

				}
			}
		}



		// 中文：构建源-目标交叉注意力掩码（按源/目标各自原始长度判定有效位）。
		unsafe static public void BuildSrcTgtMask(Tensor result, Tensor srcOriginalLengths, Tensor tgtOriginalLengths, int rows, int cols, int tgtPaddedSeqLen, float value, float maskedValue)
		{
			float* ptResult = (float*)CpuNativeHelpers.GetBufferStart(result);
			float* ptSrcOriginalLengths = (float*)CpuNativeHelpers.GetBufferStart(srcOriginalLengths);
			float* ptTgtOriginalLengths = (float*)CpuNativeHelpers.GetBufferStart(tgtOriginalLengths);

			for (int j = 0; j < rows; j++)
			{
				float* resultRow = ptResult + j * cols;
				int batchIdx = j / tgtPaddedSeqLen;
				int seqIdxInBatch = j % tgtPaddedSeqLen;

				for (int id = 0; id < cols; id++)
				{
					int srcOriginalLength = (int)ptSrcOriginalLengths[batchIdx];
					int tgtOriginalLength = (int)ptTgtOriginalLengths[batchIdx];

					if (id < srcOriginalLength && seqIdxInBatch < tgtOriginalLength)
					{
						resultRow[id] = value;
					}
					else
					{
						resultRow[id] = maskedValue;
					}
				}
			}
		}


		// 中文：TopK 辅助——把 (data,idx) 插入降序排列的长度为 k 的值/索引数组中，挤掉最小者。
		unsafe static public void replace_smaller(float* array, float* arrayIdx, int k, float data, float idx)
		{
			if (data < array[k - 1])
				return;
			for (int j = k - 2; j >= 0; j--)
			{
				if (data > array[j])
				{
					array[j + 1] = array[j];
					arrayIdx[j + 1] = arrayIdx[j];
				}
				else
				{
					array[j + 1] = data;
					arrayIdx[j + 1] = idx;
					return;
				}
			}
			array[0] = data;
			arrayIdx[0] = idx;
		}

		// 中文：逐行求 TopK 的值与索引，借助 replace_smaller 维护每行降序的前 k 名。
		unsafe static public void TopK(Tensor outVal, Tensor outIdx, Tensor inVal, int k, int rows, int cols)
		{
			float* pOutVal = (float*)CpuNativeHelpers.GetBufferStart(outVal);
			float* pOutIdx = (float*)CpuNativeHelpers.GetBufferStart(outIdx);
			float* pInVal = (float*)CpuNativeHelpers.GetBufferStart(inVal);


			for (int j = 0; j < rows; ++j)
			{
				float* outputRow = pOutVal + j * k;
				float* outputIdxRow = pOutIdx + j * k;
				float* inputRow = pInVal + j * cols;

				for (int i = 0; i < k; ++i)
				{
					outputRow[i] = -1.70141e+38f;
					outputIdxRow[i] = -1.70141e+38f;

				}

				for (int i = 0; i < cols; i++)
				{
					replace_smaller(outputRow, outputIdxRow, k, inputRow[i], i);
				}
			}
		}

		// 中文：对每行按位置应用旋转位置编码（RoPE），对相邻维度对做旋转变换。
		unsafe static public void RoPE(Tensor tOut, Tensor tIn, int rows, int cols, int seqLen, int rowOffset)
		{
			float* result = (float*)CpuNativeHelpers.GetBufferStart(tOut);
			float* src = (float*)CpuNativeHelpers.GetBufferStart(tIn);

			for (int j = 0; j < rows; ++j)
			{
				float* resultRow = result + j * cols;
				float* srcRow = src + j * cols;
				int m = (j % seqLen) + rowOffset;

				for (int id = 0; id < cols; id++)
				{
					int i = id / 2;
					float theta = (float)Math.Pow(500000.0, -2.0 * i / cols);
					float theta_m = theta * m;
					float cos_theta_m = (float)Math.Cos(theta_m);
					float sin_theta_m = (float)Math.Sin(theta_m);

					if (id % 2 == 0)
					{
						resultRow[id] = srcRow[id] * cos_theta_m - srcRow[id + 1] * sin_theta_m;
					}
					else
					{
						resultRow[id] = srcRow[id] * cos_theta_m + srcRow[id - 1] * sin_theta_m;
					}

				}
			}
		}

        // 中文：按张量元素类型（Int32 或 Float）读取第 index 个位置值并返回 int。
        unsafe static private int ReadPosition(Tensor positions, int index)
        {
            return positions.ElementType switch
            {
                DType.Int32 => ((int*)CpuNativeHelpers.GetBufferStart(positions))[index],
                _ => (int)((float*)CpuNativeHelpers.GetBufferStart(positions))[index],
            };
        }

        // 中文：扩展版 RoPE，支持显式位置张量、NeoX 排布、YaRN 频率缩放与外推，可并行逐行应用旋转。
        unsafe static public void RoPEEx(Tensor tOut, Tensor tIn, Tensor positions, int rows, int cols, int ropeDim, int mode, float freqBase, float freqScale,
            int nCtxOrig = 0, float extFactor = 0.0f, float attnFactor = 1.0f, float betaFast = 0.0f, float betaSlow = 0.0f)
        {
            const int GGML_ROPE_TYPE_NEOX = 2;

            float* result = (float*)CpuNativeHelpers.GetBufferStart(tOut);
            float* src = (float*)CpuNativeHelpers.GetBufferStart(tIn);
            bool isNeoX = (mode & GGML_ROPE_TYPE_NEOX) != 0;
            int activeRopeDim = Math.Min(ropeDim, cols);
            int pairCount = activeRopeDim / 2;
            long rowBytes = (long)cols * sizeof(float);

            if (pairCount <= 0)
            {
                if (result != src)
                {
                    Buffer.MemoryCopy(src, result, rowBytes * rows, rowBytes * rows);
                }

                return;
            }

            bool useYarn = extFactor != 0.0f;

            float corrDimLow = 0, corrDimHigh = 0;
            float mscale = attnFactor;
            if (useYarn)
            {
                YarnCorrDims(activeRopeDim, nCtxOrig, freqBase, betaFast, betaSlow, out corrDimLow, out corrDimHigh);
                mscale *= 1.0f + 0.1f * MathF.Log(1.0f / freqScale);
            }

            float[] invFreqBuffer = ArrayPool<float>.Shared.Rent(pairCount);
            try
            {
                for (int i = 0; i < pairCount; i++)
                {
                    invFreqBuffer[i] = MathF.Pow(freqBase, -2.0f * i / activeRopeDim);
                }

                bool useIntPositions = positions.ElementType == DType.Int32;
                int* positionInts = useIntPositions ? (int*)CpuNativeHelpers.GetBufferStart(positions) : null;
                float* positionFloats = useIntPositions ? null : (float*)CpuNativeHelpers.GetBufferStart(positions);

                void ApplyRow(int row)
                {
                    float* resultRow = result + row * cols;
                    float* srcRow = src + row * cols;
                    if (resultRow != srcRow)
                    {
                        Buffer.MemoryCopy(srcRow, resultRow, rowBytes, rowBytes);
                    }

                    int position = useIntPositions ? positionInts[row] : (int)positionFloats[row];

                    if (isNeoX)
                    {
                        int half = pairCount;
                        for (int i = 0; i < half; ++i)
                        {
                            float thetaExtrap = position * invFreqBuffer[i];
                            float cosTheta, sinTheta;
                            if (useYarn)
                            {
                                YarnRoPE(thetaExtrap, freqScale, corrDimLow, corrDimHigh, i, extFactor, mscale, out cosTheta, out sinTheta);
                            }
                            else
                            {
                                float angle = thetaExtrap * freqScale;
                                cosTheta = MathF.Cos(angle);
                                sinTheta = MathF.Sin(angle);
                            }
                            float left = srcRow[i];
                            float right = srcRow[i + half];
                            resultRow[i] = left * cosTheta - right * sinTheta;
                            resultRow[i + half] = right * cosTheta + left * sinTheta;
                        }
                    }
                    else
                    {
                        for (int i = 0, pair = 0; i < pairCount; ++i, pair += 2)
                        {
                            float thetaExtrap = position * invFreqBuffer[i];
                            float cosTheta, sinTheta;
                            if (useYarn)
                            {
                                YarnRoPE(thetaExtrap, freqScale, corrDimLow, corrDimHigh, i, extFactor, mscale, out cosTheta, out sinTheta);
                            }
                            else
                            {
                                float angle = thetaExtrap * freqScale;
                                cosTheta = MathF.Cos(angle);
                                sinTheta = MathF.Sin(angle);
                            }
                            float left = srcRow[pair];
                            float right = srcRow[pair + 1];
                            resultRow[pair] = left * cosTheta - right * sinTheta;
                            resultRow[pair + 1] = right * cosTheta + left * sinTheta;
                        }
                    }
                }

                if (ShouldParallelize(rows, activeRopeDim))
                {
                    Parallel.For(0, rows, ApplyRow);
                }
                else
                {
                    for (int row = 0; row < rows; ++row)
                    {
                        ApplyRow(row);
                    }
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(invFreqBuffer);
            }
        }

        // 中文：YaRN 辅助——按给定旋转圈数计算对应的修正维度位置。
        static private float YarnCorrDim(int nDims, int nCtxOrig, float nRot, float freqBase)
        {
            return nDims * MathF.Log(nCtxOrig / (nRot * 2.0f * MathF.PI)) / (2.0f * MathF.Log(freqBase));
        }

        // 中文：YaRN 辅助——根据 betaFast/betaSlow 计算插值斜坡的低/高修正维度边界。
        static private void YarnCorrDims(int nDims, int nCtxOrig, float freqBase, float betaFast, float betaSlow, out float low, out float high)
        {
            if (betaFast == 0.0f && betaSlow == 0.0f)
            {
                low = float.MaxValue;
                high = (float)(nDims / 2 - 1);
            }
            else
            {
                low = MathF.Max(0, MathF.Floor(YarnCorrDim(nDims, nCtxOrig, betaFast, freqBase)));
                high = MathF.Min(nDims / 2 - 1, MathF.Ceiling(YarnCorrDim(nDims, nCtxOrig, betaSlow, freqBase)));
            }
        }

        // 中文：YaRN 辅助——按斜坡混合外推/内插角并乘以注意力缩放，输出该维度的 cos/sin。
        static private void YarnRoPE(float thetaExtrap, float freqScale, float corrDimLow, float corrDimHigh, int i0, float extFactor, float mscale,
            out float cosTheta, out float sinTheta)
        {
            float thetaInterp = freqScale * thetaExtrap;
            float rampY = ((float)i0 - corrDimLow) / MathF.Max(0.001f, corrDimHigh - corrDimLow);
            float rampMix = (1.0f - MathF.Min(1.0f, MathF.Max(0.0f, rampY))) * extFactor;
            float theta = thetaInterp * (1.0f - rampMix) + thetaExtrap * rampMix;
            cosTheta = MathF.Cos(theta) * mscale;
            sinTheta = MathF.Sin(theta) * mscale;
        }


		// 中文：RoPE 的反向传播，对每行相邻维度对用反向旋转把上游梯度累加到 grad。
		unsafe static public void RoPEGrad(Tensor tOut, Tensor tIn, int rows, int cols, int seqLen, int rowOffset)
		{
			float* grad = (float*)CpuNativeHelpers.GetBufferStart(tOut);
			float* adj = (float*)CpuNativeHelpers.GetBufferStart(tIn);

			for (int j = 0; j < rows; j++)
			{
				float* gradRow = grad + j * cols;
				float* adjRow = adj + j * cols;
				int m = (j % seqLen) + rowOffset;

				for (int id = 0; id < cols; id++)
				{
					int i = id / 2;
					float theta = (float)Math.Pow(500000.0, -2.0 * i / cols);
					float theta_m = theta * m;
					float cos_theta_m = (float)Math.Cos(theta_m);
					float sin_theta_m = (float)Math.Sin(theta_m);

					if (id % 2 == 0)
					{
						gradRow[id] += (adjRow[id] * cos_theta_m + adjRow[id + 1] * sin_theta_m);
					}
					else
					{
						gradRow[id] += (adjRow[id] * cos_theta_m - adjRow[id - 1] * sin_theta_m);
					}
				}
			}
		}

		// 中文：扫描张量是否含非有限值（NaN/Inf），存在则返回 true。
		unsafe static public bool IsCorrupted(Tensor tIn, int rows, int cols)
		{
            float* pIn = (float*)CpuNativeHelpers.GetBufferStart(tIn);

            for (int j = 0; j < rows; ++j)
            {
                float* sp = pIn + j * cols;
                for (int i = 0; i < cols; ++i)
                {
					if (float.IsFinite(sp[i]) == false)
					{
						return true;
					}
                }
            }
			return false;
        }

		// 中文：逐行做数值稳定的 Softmax（减最大值后 exp 再归一化），含连续行的向量化与可并行快速路径。
		unsafe static public void Softmax(Tensor tOut, Tensor tIn, int rows, int cols)
		{
            if (TryGetContiguousRows(tOut, out float* contiguousOut, out int outRows, out int outCols) &&
                TryGetContiguousRows(tIn, out float* contiguousIn, out int inRows, out int inCols) &&
                rows == outRows && rows == inRows && cols == outCols && cols == inCols)
            {
                void ComputeRow(int row)
                {
                    float* so = contiguousOut + row * cols;
                    float* sp = contiguousIn + row * cols;

                    int vectorSize = Vector<float>.Count;
                    Vector<float> vecMax = new Vector<float>(float.NegativeInfinity);
                    int i = 0;
                    for (; i <= cols - vectorSize; i += vectorSize)
                    {
                        vecMax = Vector.Max(vecMax, LoadVec(sp + i));
                    }

                    float max = float.NegativeInfinity;
                    for (int lane = 0; lane < vectorSize; lane++)
                    {
                        max = MathF.Max(max, vecMax[lane]);
                    }

                    for (; i < cols; ++i)
                    {
                        max = MathF.Max(max, sp[i]);
                    }

                    float sum = 0.0f;
                    for (i = 0; i < cols; ++i)
                    {
                        float ex = MathF.Exp(sp[i] - max);
                        so[i] = ex;
                        sum += ex;
                    }

                    float invSum = 1.0f / sum;
                    Vector<float> vecInvSum = new Vector<float>(invSum);
                    i = 0;
                    for (; i <= cols - vectorSize; i += vectorSize)
                    {
                        StoreVec(so + i, LoadVec(so + i) * vecInvSum);
                    }

                    for (; i < cols; i++)
                    {
                        so[i] *= invSum;
                    }
                }

                if (ShouldParallelize(rows, cols))
                {
                    Parallel.For(0, rows, ComputeRow);
                }
                else
                {
                    for (int row = 0; row < rows; ++row)
                    {
                        ComputeRow(row);
                    }
                }

                return;
            }

			float* pOut = (float*)CpuNativeHelpers.GetBufferStart(tOut);
			float* pIn = (float*)CpuNativeHelpers.GetBufferStart(tIn);

			for (int j = 0; j < rows; ++j)
			{
				float* so = pOut + j * cols;
				float* sp = pIn + j * cols;

				// Match GGML softmax: max starts at -inf so leading masked (-inf) logits behave like ggml_soft_max.
				float max = float.NegativeInfinity;
				for (int i = 0; i < cols; ++i)
				{
					max = Math.Max(max, sp[i]);
				}

				float sum = 0.0f;
				for (int i = 0; i < cols; ++i)
				{
					float ex = (float)Math.Exp(sp[i] - max);
					so[i] = ex;
					sum += ex;
				}

				Span<float> spanSO = new Span<float>(so, cols);
				int vectorSize = Vector<float>.Count;
				int k = 0;
				Vector<float> vecSum = new Vector<float>(sum);
				for (k = 0; k < cols - vectorSize; k += vectorSize)
				{
					Vector<float> vecSO = new Vector<float>(spanSO.Slice(k));
					vecSO /= vecSum;

					vecSO.CopyTo(spanSO.Slice(k));
				}
				for (; k < cols; k++)
				{
					so[k] /= sum;
				}
			}
		}


		// 中文：逐行计算 Softmax 反向梯度（grad=val*(adj-Σval*adj)），可选累加或覆盖到 grad。
		unsafe static public void SoftmaxGrad(Tensor grad_, Tensor adj_, Tensor val_, int rows, int cols, bool addGrad)
		{

			float* grad = (float*)CpuNativeHelpers.GetBufferStart(grad_);
			float* adj = (float*)CpuNativeHelpers.GetBufferStart(adj_);
			float* val = (float*)CpuNativeHelpers.GetBufferStart(val_);

			for (int j = 0; j < rows; ++j)
			{
				float* gradRow = grad + j * cols;
				float* adjRow = adj + j * cols;
				float* valRow = val + j * cols;

				float sum = 0.0f;
				for (int i = 0; i < cols; ++i)
				{
					sum += valRow[i] * adjRow[i];
				}

				for (int i = 0; i < cols; ++i)
				{
					if (addGrad)
					{
						gradRow[i] += valRow[i] * (adjRow[i] - sum);
					}
					else
					{
						gradRow[i] = valRow[i] * (adjRow[i] - sum);
					}
				}
			}
		}



		// 中文：按行索引从 src 选取行写入/累加到 result（嵌入查表），含连续行的可并行向量化快速路径。
		unsafe static public void IndexSelect(Tensor result_, Tensor src_, Tensor indice_, int rows, int cols, bool isAdd)
		{
            if (TryGetContiguousRows(result_, out float* contiguousResult, out int resultRows, out int resultCols) &&
                TryGetContiguousRows(src_, out float* contiguousSrc, out int srcRows, out int srcCols) &&
                resultRows == rows && resultCols == cols && srcCols == cols &&
                indice_.IsContiguous() && indice_.ElementCount() == rows)
            {
                void* contiguousIndice = (void*)CpuNativeHelpers.GetBufferStart(indice_);
                bool contiguousInt32 = indice_.ElementType == DType.Int32;
                long rowBytes = (long)cols * sizeof(float);

                void CopyRow(int row)
                {
                    int srcIdx = contiguousInt32 ? ((int*)contiguousIndice)[row] : (int)((float*)contiguousIndice)[row];
                    if (srcIdx < 0)
                    {
                        return;
                    }

                    if ((uint)srcIdx >= (uint)srcRows)
                    {
                        throw new IndexOutOfRangeException($"Invalid index in index_select. Idx = '{srcIdx}', srcRows = '{srcRows}'");
                    }

                    float* resultRow = contiguousResult + row * cols;
                    float* srcRow = contiguousSrc + srcIdx * cols;
                    if (!isAdd)
                    {
                        Buffer.MemoryCopy(srcRow, resultRow, rowBytes, rowBytes);
                        return;
                    }

                    int vectorSize = Vector<float>.Count;
                    int i = 0;
                    for (; i <= cols - vectorSize; i += vectorSize)
                    {
                        StoreVec(resultRow + i, LoadVec(resultRow + i) + LoadVec(srcRow + i));
                    }

                    for (; i < cols; ++i)
                    {
                        resultRow[i] += srcRow[i];
                    }
                }

                if (ShouldParallelize(rows, cols))
                {
                    Parallel.For(0, rows, CopyRow);
                }
                else
                {
                    for (int row = 0; row < rows; row++)
                    {
                        CopyRow(row);
                    }
                }

                return;
            }

			float* result = (float*)CpuNativeHelpers.GetBufferStart(result_);
			float* src = (float*)CpuNativeHelpers.GetBufferStart(src_);
			void* indice = (void*)CpuNativeHelpers.GetBufferStart(indice_);
			bool isInt32 = indice_.ElementType == DType.Int32;

			for (int j = 0; j < rows; j++)
			{
				int srcIdx = isInt32 ? ((int*)indice)[j] : (int)((float*)indice)[j];
				if (srcIdx >= 0)
				{
					float* resultRow = result + j * cols;
					float* srcRow = src + srcIdx * cols;

					for (int i = 0; i < cols; ++i)
					{
						if (isAdd == false)
						{
							resultRow[i] = srcRow[i];
						}
						else
						{
							resultRow[i] += srcRow[i];
						}
					}
				}
			}
		}


		// 中文：IndexSelect 的反向，把上游梯度 adj 按索引累加回对应 grad 行（嵌入梯度散布）。
		unsafe static public void IndexSelectGrad(Tensor grad_, Tensor adj_, Tensor indice_, int rows, int cols)
		{
			float* grad = (float*)CpuNativeHelpers.GetBufferStart(grad_);
			float* adj = (float*)CpuNativeHelpers.GetBufferStart(adj_);
			float* indice = (float*)CpuNativeHelpers.GetBufferStart(indice_);

			for (int j = 0; j < rows; j++)
			{
				int gradIdx = (int)indice[j];
				if (gradIdx >= 0)
				{
					float* adjRow = adj + j * cols;
					float* gradRow = grad + gradIdx * cols;

					for (int i = 0; i < cols; ++i)
					{
						gradRow[i] += adjRow[i];
					}
				}
			}
		}



		// 中文：逐行做 LayerNorm（减均值除标准差后用 gamma 缩放、beta 平移），均值/方差用向量化累加。
		unsafe static public void LayerNorm(Tensor out_,
			Tensor in_,
			Tensor gamma_,
			Tensor beta_,
			float eps,
			int rows,
			int cols)
		{
			float* outPtr = (float*)CpuNativeHelpers.GetBufferStart(out_);
			float* inPtr = (float*)CpuNativeHelpers.GetBufferStart(in_);
			float* alpha = (float*)CpuNativeHelpers.GetBufferStart(gamma_);
			float* beta = (beta_ != null) ? (float*)CpuNativeHelpers.GetBufferStart(beta_) : null;

			for (int j = 0; j < rows; ++j)
			{
				float* so = outPtr + j * cols;
				float* sp = inPtr + j * cols;

				Span<float> spanSP = new Span<float>(sp, cols);

				float sum = 0.0f;
				int vectorSize = Vector<float>.Count;
				Vector<float> vecAdded = Vector<float>.Zero;
				int i = 0;

				for (i = 0; i < cols - vectorSize; i += vectorSize)
				{
					Vector<float> vecSp = new Vector<float>(spanSP.Slice(i));
					vecAdded += vecSp;
				}
				sum = Vector.Dot(vecAdded, Vector<float>.One);
				for (; i < cols; i++)
				{
					sum += sp[i];
				}

				float mean = sum / cols;
				float sqSum = 0.0f;

				Vector<float> vecMean = new Vector<float>(mean);
				for (i = 0; i < cols - vectorSize; i += vectorSize)
				{
					Vector<float> vecSp = new Vector<float>(spanSP.Slice(i));
					Vector<float> vecEx = vecSp - vecMean;
					sqSum += Vector.Dot(vecEx, vecEx);
				}
				for (; i < cols; ++i)
				{
					float ex = sp[i] - mean;
					sqSum += ex * ex;
				}

				float sigma = (float)Math.Sqrt(eps + sqSum / cols);

				Span<float> spanSO = new Span<float>(so, cols);
				Span<float> spanAlpha = new Span<float>(alpha, cols);
				bool hasBeta = beta != null;
				Span<float> spanBeta = hasBeta ? new Span<float>(beta, cols) : Span<float>.Empty;
				Vector<float> vecSigma = new Vector<float>(sigma);

				for (i = 0; i < cols - vectorSize; i += vectorSize)
				{
					Vector<float> vecSp = new Vector<float>(spanSP.Slice(i));
					Vector<float> vecAlpha = new Vector<float>(spanAlpha.Slice(i));

					Vector<float> vecT = vecAlpha * ((vecSp - vecMean) / vecSigma);

					if (hasBeta)
					{
						Vector<float> vecBeta = new Vector<float>(spanBeta.Slice(i));
						vecT += vecBeta;
					}

					vecT.CopyTo(spanSO.Slice(i));
				}
				for (; i < cols; ++i)
				{
					float t = alpha[i] * ((sp[i] - mean) / sigma);
					if (beta != null)
					{
						t += beta[i];
					}

					so[i] = t;
				}


			}
		}



		// 中文：逐行做 RMSNorm（按均方根归一化后用 gamma 缩放、可选 beta 偏置），含连续行可并行向量化快速路径。
		unsafe static public void RMSNorm(Tensor out_,
			Tensor in_,
			Tensor gamma_,
			Tensor beta_,
			float eps,
			int rows,
			int cols)
		{
            if (TryGetContiguousRows(out_, out float* contiguousOut, out int outRows, out int outCols) &&
                TryGetContiguousRows(in_, out float* contiguousIn, out int inRows, out int inCols) &&
                TryGetContiguousFloat(gamma_, out float* gammaPtr, out int gammaLength) &&
                rows == outRows && rows == inRows && cols == outCols && cols == inCols && gammaLength == cols &&
                (beta_ == null || (TryGetContiguousFloat(beta_, out _, out int betaLength) && betaLength == cols)))
            {
                float* betaPtr = beta_ != null ? (float*)CpuNativeHelpers.GetBufferStart(beta_) : null;
                bool hasBiasFast = betaPtr != null;
                float colsAsFloat = cols;
                int vectorSize = Vector<float>.Count;

                void ApplyRow(int row)
                {
                    float* yRow = contiguousOut + row * cols;
                    float* xRow = contiguousIn + row * cols;

                    Vector<float> acc = Vector<float>.Zero;
                    int i = 0;
                    for (; i <= cols - vectorSize; i += vectorSize)
                    {
                        Vector<float> vx = LoadVec(xRow + i);
                        acc += vx * vx;
                    }

                    float sqSum = Vector.Sum(acc);
                    for (; i < cols; i++)
                    {
                        sqSum += xRow[i] * xRow[i];
                    }

                    float invRms = 1.0f / MathF.Sqrt(sqSum / colsAsFloat + eps);
                    Vector<float> vecInvRms = new Vector<float>(invRms);

                    i = 0;
                    if (hasBiasFast)
                    {
                        for (; i <= cols - vectorSize; i += vectorSize)
                        {
                            Vector<float> value = LoadVec(xRow + i) * vecInvRms * LoadVec(gammaPtr + i) + LoadVec(betaPtr + i);
                            StoreVec(yRow + i, value);
                        }

                        for (; i < cols; i++)
                        {
                            yRow[i] = gammaPtr[i] * (xRow[i] * invRms) + betaPtr[i];
                        }
                    }
                    else
                    {
                        for (; i <= cols - vectorSize; i += vectorSize)
                        {
                            Vector<float> value = LoadVec(xRow + i) * vecInvRms * LoadVec(gammaPtr + i);
                            StoreVec(yRow + i, value);
                        }

                        for (; i < cols; i++)
                        {
                            yRow[i] = gammaPtr[i] * (xRow[i] * invRms);
                        }
                    }
                }

                if (ShouldParallelize(rows, cols))
                {
                    Parallel.For(0, rows, ApplyRow);
                }
                else
                {
                    for (int row = 0; row < rows; row++)
                    {
                        ApplyRow(row);
                    }
                }

                return;
            }

			float* outPtr = (float*)CpuNativeHelpers.GetBufferStart(out_);
			float* inPtr = (float*)CpuNativeHelpers.GetBufferStart(in_);
			float* gamma = (float*)CpuNativeHelpers.GetBufferStart(gamma_);
			float* beta = (beta_ != null) ? (float*)CpuNativeHelpers.GetBufferStart(beta_) : null;
			bool bias = (beta_ != null);

            float N = cols;
			for (int j = 0; j < rows; j++)
			{
				float* yRow = outPtr + j * cols;
				float* xRow = inPtr + j * cols;

				float _sqSum = 0;

				for (int id = 0; id < cols; id++)
				{
					float xv = (float)xRow[id];

					_sqSum += xv * xv;

				}

				float rms = (float)Math.Sqrt(_sqSum / N + eps); 
				for (int id = 0; id < cols; id++)
				{

					float gammav = gamma[id];
					float xv = xRow[id];
					float betav = bias ? beta[id] : 0.0f;
					float rmsNorm = xv / rms;
					float y = gammav * rmsNorm + betav;
					yRow[id] = y;

				}
			}


		}

        // 中文：LayerNorm 的反向，逐行计算并累加对输入 x、gamma、beta 的梯度（分有/无 beta 两种分支）。
        unsafe static public void LayerNormGrad(Tensor gradX_,
			Tensor gradGamma_,
			Tensor gradBeta_,
			Tensor adj_,
			Tensor y_,
			Tensor x_,
			Tensor gamma_,
			Tensor beta_,
			int rows,
			int cols,
			float eps)
		{
			float* gradX = (float*)CpuNativeHelpers.GetBufferStart(gradX_);
			float* gradGamma = (float*)CpuNativeHelpers.GetBufferStart(gradGamma_);
			float* gradBeta = gradBeta_ != null ? (float*)CpuNativeHelpers.GetBufferStart(gradBeta_) : null;
			float* adj = (float*)CpuNativeHelpers.GetBufferStart(adj_);
			float* y = (float*)CpuNativeHelpers.GetBufferStart(y_);
			float* x = (float*)CpuNativeHelpers.GetBufferStart(x_);
			float* gamma = (float*)CpuNativeHelpers.GetBufferStart(gamma_);
			float* beta = beta_ != null ? (float*)CpuNativeHelpers.GetBufferStart(beta_) : null;

			if (beta != null)
			{
				for (int j = 0; j < rows; ++j)
				{
					float* xRow = x + j * cols;
					float* yRow = y + j * cols;
					float* adjRow = adj + j * cols;
					float* gradXRow = gradX + j * cols;

					float sum_x = 0.0f;
					float sum_adj = 0.0f;
					float sum_adj_x = 0.0f;
					float sum_sqr = 0.0f;

					for (int i = 0; i < cols; ++i)
					{
						sum_x += xRow[i];
						sum_adj_x += adjRow[i] * (yRow[i] - (beta != null ? beta[i] : 0.0f)) / gamma[i];
						sum_adj += adjRow[i];
					}

					float mean = sum_x / cols;
					for (int i = 0; i < cols; ++i)
					{
						float ex = xRow[i] - mean;
						sum_sqr += ex * ex;
					}

					float sigma = (float)Math.Sqrt(eps + sum_sqr / cols);
					for (int i = 0; i < cols; ++i)
					{
						float grad_x = 0.0f;
						float x_hat = (yRow[i] - beta[i]) / gamma[i];
						grad_x += cols * adjRow[i];
						grad_x -= sum_adj;
						grad_x -= sum_adj_x * x_hat;
						grad_x /= cols * sigma;

						gradXRow[i] += gamma[i] * grad_x;
						gradGamma[i] += adjRow[i] * x_hat;
						gradBeta[i] += adjRow[i];
					}
				}
			}
			else
			{
				for (int j = 0; j < rows; ++j)
				{
					float* xRow = x + j * cols;
					float* yRow = y + j * cols;
					float* adjRow = adj + j * cols;
					float* gradXRow = gradX + j * cols;

					float sum_x = 0.0f;
					float sum_adj = 0.0f;
					float sum_adj_x = 0.0f;
					float sum_sqr = 0.0f;

					for (int i = 0; i < cols; ++i)
					{
						sum_x += xRow[i];
						sum_adj_x += adjRow[i] * (yRow[i] - (beta != null ? beta[i] : 0.0f)) / gamma[i];
						sum_adj += adjRow[i];
					}

					float mean = sum_x / cols;

					for (int i = 0; i < cols; ++i)
					{
						float ex = xRow[i] - mean;
						sum_sqr += ex * ex;
					}

					float sigma = (float)Math.Sqrt(eps + sum_sqr / cols);

					for (int i = 0; i < cols; ++i)
					{
						float grad_x = 0.0f;
						float x_hat = yRow[i] / gamma[i];
						grad_x += cols * adjRow[i];
						grad_x -= sum_adj;
						grad_x -= sum_adj_x * x_hat;
						grad_x /= cols * sigma;

						gradXRow[i] += gamma[i] * grad_x;
						gradGamma[i] += adjRow[i] * x_hat;
					}
				}
			}
		}


		// 中文：RMSNorm 的反向，逐行按 RMS 雅可比计算并累加对 x、gamma、beta 的梯度，含梯度裁剪与 NaN 置零。
		unsafe static public void RMSNormGrad(Tensor gradX_,
			Tensor gradGamma_,
			Tensor gradBeta_,
			Tensor adj_,
			Tensor y_,
			Tensor x_,
			Tensor gamma_,
			Tensor beta_,
			int rows,
			int cols,
			float eps)
		{
			float* gradX = (float*)CpuNativeHelpers.GetBufferStart(gradX_);
			float* gradGamma = (float*)CpuNativeHelpers.GetBufferStart(gradGamma_);
			float* gradBeta = (gradBeta_ != null) ? (float*)CpuNativeHelpers.GetBufferStart(gradBeta_) : null;
            float* adj = (float*)CpuNativeHelpers.GetBufferStart(adj_);
			float* y = (float*)CpuNativeHelpers.GetBufferStart(y_);
			float* x = (float*)CpuNativeHelpers.GetBufferStart(x_);
			float* gamma = (float*)CpuNativeHelpers.GetBufferStart(gamma_);
            float* beta = (beta_ != null) ? (float*)CpuNativeHelpers.GetBufferStart(beta_) : null;
			bool bias = (beta_ != null);

            float N = cols;
			for (int j = 0; j < rows; j++)
			{
				float* xRow = x + j * cols;
				float* yRow = y + j * cols;
				float* adjRow = adj + j * cols;

				float sum_adj_r = (float)0.0f;
				float sum_sqr = (float)0.0f;

				for (int id = 0; id < cols; id++)
				{

					float xv = xRow[id];
					float yv = yRow[id];
					float betav = bias ? beta[id] : 0.0f;
					float gammav = (float)gamma[id];
					float adjv = adjRow[id];
					float rv = (yv - betav) / gammav; // go back to RMSNorm(x) from scaled and shifted version for accumulation

					sum_adj_r += adjv * rv;
					sum_sqr += xv * xv;

				}

				float rms = (float)Math.Sqrt(sum_sqr / N + eps);

				// Jacobian of RMS norm
				// J = [ \frac{1}{N * rms} (N\delta_{ij} - RN_i RN_j) ]_{ij}
				// J * a = dC/dx_i = ( N a_i - RN_i \sum_j RN_j a_j ) / (N * rms)

				for (int id = 0; id < cols; id++)
				{

					float xv = xRow[id];
					float gammav = (float)gamma[id];
					float adjv = adjRow[id];
					float rmsNorm = xv / rms;

					float gradNorm = N * adjv - rmsNorm * sum_adj_r;
					gradNorm /= N * rms;

					float gradXv = gammav * gradNorm;

					// Keep RMSN gradient between [-1000, 1000] for TensorOps, this currently used for making values fit into fp16. This wil also clip inf. 
					// @TODO: to be fixed and removed.
					float sign = Math.Sign(gradXv); //functional::Ops<AccType>::sgn(gradXv);
					float cutoff = (float)1000.0f; // @TODO: expose this somehow as an option? or better: make obsolete.
					gradXv = Math.Abs(gradXv) > cutoff ? sign * cutoff : gradXv; // if gradXv is NaN the value return is NaN too because NaN > value is false.

					// @TODO: frankly, this is embarrasing and should rather be removed or optional? It does help for low precision computation though. Maybe turn into option?
					gradXv = float.IsNaN(gradXv) ? 0.0f : gradXv; // turn NaN into 0.

					float* gradXRow = gradX + j * cols;
					gradXRow[id] += (float)(gradXv);

					gradGamma[id] += (float)(adjv * rmsNorm);
					if (bias)
					{
						gradBeta[id] += adjRow[id];
					}
                }
			}
		}


        // 中文：逐元素执行 Adam 优化器一步更新（梯度归一化裁剪、一/二阶矩估计与偏差修正后更新权重并清零梯度）。
        unsafe static public void Adam(Tensor tw, Tensor tg, Tensor tv, Tensor tm, int rows, int cols, float gradNormFactor, float step_size, float clipval, float regc, float decay_rate_v, float decay_rate_m, int iter, float eps)
		{
			float* w = (float*)CpuNativeHelpers.GetBufferStart(tw);
			float* g = (float*)CpuNativeHelpers.GetBufferStart(tg);
			float* v = (float*)CpuNativeHelpers.GetBufferStart(tv);
			float* m = (float*)CpuNativeHelpers.GetBufferStart(tm);

			for (int j = 0; j < rows; j++)
			{
				float* sw = w + j * cols;
				float* sg = g + j * cols;
				float* sv = v + j * cols;
				float* sm = m + j * cols;

				for (int i = 0; i < cols; i++)
				{
					if (sg[i] != 0.0)
					{
						float g2 = sg[i] * gradNormFactor;

						if (g2 > clipval)
						{
							g2 = clipval;
						}
						if (g2 < -clipval)
						{
							g2 = -clipval;
						}

						sm[i] = sm[i] * decay_rate_m + (1.0f - decay_rate_m) * g2;
						sv[i] = sv[i] * decay_rate_v + (1.0f - decay_rate_v) * g2 * g2;

						double m_cap = sm[i] / (1.0 - Math.Pow(decay_rate_m, iter));
						double v_cap = sv[i] / (1.0 - Math.Pow(decay_rate_v, iter));

						sw[i] -= (float)(step_size * m_cap / (Math.Sqrt(v_cap) + eps));

						sg[i] = 0;
					}
				}
			}
		}


		#region Internal operations


		// 中文：标量 ReLU（负数取 0）。
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float relu(float w)
		{
			if (w < 0.0f)
				return 0.0f;
			return w;

		}

		// 中文：标量 ReLU 反向梯度（w>0 透传 g，否则 0）。
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float relud(float w, float g)
		{
			if (w > 0.0f)
				return g;
			return 0.0f;
		}

		// 中文：标量 ReLU 反向梯度并累加到 t（w>0 加 g，否则不变）。
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float addrelud(float t, float w, float g)
		{
			if (w > 0.0f)
				return t + g;
			return t;
		}

        // 中文：标量 LeakyReLU（负数乘 0.01）。
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static float LeakyReLU(float w)
        {
            if (w < 0.0f)
                return 0.01f * w;
            return w;

        }

        // 中文：标量 LeakyReLU 反向梯度（w>=0 透传 g，否则 0.01*g）。
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static float LeakyReLUD(float w, float g)
        {
            if (w >= 0.0f)
                return g;
            return 0.01f * g;
        }

        // 中文：标量 LeakyReLU 反向梯度并累加到 t。
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static float AddLeakyReLUD(float t, float w, float g)
        {
            if (w >= 0.0f)
                return t + g;
            return t + 0.01f * g;
        }


        // 中文：标量 SiLU（w*sigmoid(w)）。
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float SiLU(float w)
		{
			return w / (1.0f + (float)Math.Exp(-w));
		}

		// 中文：标量 GELU（tanh 近似形式）。
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float GELU(float x)
		{
			return 0.5f * x * (1.0f + (float)Math.Tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
		}

		// 中文：标量 SiLU 反向梯度（导数 sig*(1+w*(1-sig)) 乘上游 resG）。
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float SiLUD(float w, float resG)
		{
			float sig = 1.0f / (1.0f + (float)Math.Exp(-w));
			float grad = sig * (1.0f + w * (1.0f - sig));
			return resG * grad;
		}

		// 中文：标量 SiLU 反向梯度并累加到 t。
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float AddSiLUD(float t, float w, float resG)
		{
			float sig = 1.0f / (1.0f + (float)Math.Exp(-w));
			float grad = sig * (1.0f + w * (1.0f - sig));
			return t + resG * grad;
		}

		// 中文：标量加法 x+y。
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float add(float x, float y)
		{
			return x + y;
		}

		// 中文：标量乘法 x*y。
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float mul(float x, float y)
		{
			return x * y;
		}


		// 中文：标量除法 x/y。
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float div(float x, float y)
		{
			return x / y;
		}

		// 中文：标量 Sigmoid（1/(1+e^-x)）。
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float sigmoid(float x)
		{
			return (float)(1.0 / (1.0 + Math.Exp(-x)));
		}


		// 中文：标量 Sigmoid 反向梯度（resW*(1-resW)*resG，resW 为前向输出）。
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float sigmoidD(float resW, float resG)
		{
			return resW * (1.0f - resW) * resG;
		}


		// 中文：标量 Sigmoid 反向梯度并累加到 t。
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float addSigmoidD(float t, float resW, float resG)
		{
			return t + resW * (1.0f - resW) * resG;
		}



		// 中文：标量两积之和 x*y+z*w。
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float mulmuladd(float x, float y, float z, float w)
		{
			return x * y + z * w;
		}

		// 中文：标量截断到 [min,max] 区间。
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float clamp(float val, float min, float max)
		{
			if (val < min)
				return min;
			if (val > max)
				return max;
			return val;
		}

		// 中文：标量乘加 x+y*z。
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float addmul(float x, float y, float z)
		{
			return x + y * z;
		}

		// 中文：标量加除 x+y/z。
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float adddiv(float x, float y, float z)
		{
			return x + y / z;
		}


		// 中文：标量 tanh(x+y)。
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float addtanh(float x, float y)
		{
			return (float)Math.Tanh(x + y);
		}

		// 中文：标量 Tanh 反向梯度并累加到 t（(1-resW^2)*resG）。
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float addtanhD(float t, float resW, float resG)
		{
			return t + (1.0f - resW * resW) * resG;
		}

		// 中文：标量 Tanh 反向梯度（(1-resW^2)*resG，resW 为前向输出）。
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float tanhD(float resW, float resG)
		{
			return (1.0f - resW * resW) * resG;
		}

		#endregion
	}
}
