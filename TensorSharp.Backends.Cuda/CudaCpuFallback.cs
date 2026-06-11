using System;
using System.Collections.Generic;
using TensorSharp.Cpu;

namespace TensorSharp.Cuda
{
    internal static class CudaCpuFallback
    {
        private static readonly CpuAllocator CpuAllocator = new CpuAllocator(BlasEnum.DotNet);

        // 中文：将算子回退到 CPU 执行并返回张量结果，再把结果拷回 CUDA 张量。
        public static Tensor InvokeTensor(string opName, Tensor resultTensor, params object[] args)
        {
            object returnValue = Invoke(opName, args, out Dictionary<Tensor, Tensor> mappedTensors);

            try
            {
                if (resultTensor != null)
                {
                    Tensor cpuResult = mappedTensors[resultTensor];
                    CopyLogical(resultTensor, cpuResult);
                    return resultTensor;
                }

                if (returnValue is Tensor cpuReturn)
                {
                    Tensor cudaReturn = CreateCudaLike(cpuReturn, args);
                    CopyLogical(cudaReturn, cpuReturn);
                    return cudaReturn;
                }

                return null;
            }
            finally
            {
                DisposeMapped(mappedTensors, returnValue as Tensor);
            }
        }

        // 中文：将无返回值算子回退到 CPU 执行，并把被修改张量的结果拷回 CUDA 张量。
        public static void InvokeVoid(string opName, Tensor modifiedTensor, params object[] args)
        {
            object returnValue = Invoke(opName, args, out Dictionary<Tensor, Tensor> mappedTensors);
            try
            {
                if (modifiedTensor != null)
                    CopyLogical(modifiedTensor, mappedTensors[modifiedTensor]);
            }
            finally
            {
                DisposeMapped(mappedTensors, returnValue as Tensor);
            }
        }

        // 中文：将参数中的 CUDA 张量映射为 CPU 张量后，通过 OpRegistry 在 CPU 上调用指定算子。
        private static object Invoke(string opName, object[] args, out Dictionary<Tensor, Tensor> mappedTensors)
        {
            mappedTensors = new Dictionary<Tensor, Tensor>(ReferenceEqualityComparer.Instance);
            object[] cpuArgs = new object[args.Length];

            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] is Tensor tensor)
                {
                    if (!mappedTensors.TryGetValue(tensor, out Tensor cpuTensor))
                    {
                        cpuTensor = ToCpuTensor(tensor);
                        mappedTensors.Add(tensor, cpuTensor);
                    }

                    cpuArgs[i] = cpuTensor;
                }
                else
                {
                    cpuArgs[i] = args[i];
                }
            }

            return OpRegistry.Invoke(opName, cpuArgs);
        }

        // 中文：创建与源张量同形状的 CPU 张量并将数据逻辑拷贝过去。
        private static Tensor ToCpuTensor(Tensor source)
        {
            Tensor cpu = new Tensor(CpuAllocator, source.ElementType, source.Sizes);
            CopyLogical(cpu, source);
            return cpu;
        }

        // 中文：依据原始参数中的 CUDA 分配器，创建与源张量同形状的 CUDA 张量。
        private static Tensor CreateCudaLike(Tensor source, object[] originalArgs)
        {
            IAllocator allocator = null;
            foreach (object arg in originalArgs)
            {
                if (arg is Tensor tensor && tensor.Storage is CudaStorage)
                {
                    allocator = tensor.Allocator;
                    break;
                }
            }

            allocator ??= new CudaAllocator();
            return new Tensor(allocator, source.ElementType, source.Sizes);
        }

        // 中文：在校验形状一致后，按逻辑索引将源张量元素逐一拷贝到目标张量。
        internal static void CopyLogical(Tensor destination, Tensor source)
        {
            if (destination.ElementCount() != source.ElementCount())
                throw new InvalidOperationException("Source and destination tensors must have the same number of elements.");

            if (destination.DimensionCount != source.DimensionCount)
                throw new InvalidOperationException("Source and destination tensors must have the same rank.");

            for (int i = 0; i < source.DimensionCount; i++)
            {
                if (destination.Sizes[i] != source.Sizes[i])
                    throw new InvalidOperationException("Source and destination tensors must have the same shape.");
            }

            if (source.DimensionCount == 0)
            {
                CopyElement(destination, destination.StorageOffset, source, source.StorageOffset);
                return;
            }

            CopyRecursive(destination, source, 0, destination.StorageOffset, source.StorageOffset);
        }

        // 中文：按步幅递归遍历各维度，逐元素完成逻辑拷贝。
        private static void CopyRecursive(Tensor destination, Tensor source, int dimension, long destinationOffset, long sourceOffset)
        {
            if (dimension == source.DimensionCount)
            {
                CopyElement(destination, destinationOffset, source, sourceOffset);
                return;
            }

            long size = source.Sizes[dimension];
            long sourceStride = source.Strides[dimension];
            long destinationStride = destination.Strides[dimension];
            for (long i = 0; i < size; i++)
            {
                CopyRecursive(
                    destination,
                    source,
                    dimension + 1,
                    destinationOffset + i * destinationStride,
                    sourceOffset + i * sourceStride);
            }
        }

        // 中文：将单个元素按 float 形式从源位置拷贝到目标位置，不支持的类型抛异常。
        private static void CopyElement(Tensor destination, long destinationOffset, Tensor source, long sourceOffset)
        {
            if (destination.ElementType != source.ElementType)
                throw new InvalidOperationException("Source and destination tensors must have the same element type.");

            switch (source.ElementType)
            {
                case DType.Float32:
                case DType.Float64:
                case DType.Int32:
                case DType.UInt8:
                    destination.Storage.SetElementAsFloat(destinationOffset, source.Storage.GetElementAsFloat(sourceOffset));
                    break;
                default:
                    throw new NotSupportedException($"CUDA CPU fallback does not support {source.ElementType} tensors.");
            }
        }

        // 中文：释放回退过程中创建的所有临时 CPU 张量（含返回张量），并去重避免重复释放。
        private static void DisposeMapped(Dictionary<Tensor, Tensor> mappedTensors, Tensor returnedTensor)
        {
            var disposed = new HashSet<Tensor>(ReferenceEqualityComparer.Instance);
            foreach (Tensor tensor in mappedTensors.Values)
            {
                if (disposed.Add(tensor))
                    tensor.Dispose();
            }

            if (returnedTensor != null && disposed.Add(returnedTensor))
                returnedTensor.Dispose();
        }
    }
}
