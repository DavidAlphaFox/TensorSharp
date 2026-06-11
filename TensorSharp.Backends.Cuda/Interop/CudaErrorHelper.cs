using System;
using System.Runtime.InteropServices;

namespace TensorSharp.Cuda.Interop
{
    internal static class CudaErrorHelper
    {
        // 中文：检查 CUDA Driver 返回码，非 0 时查询错误描述并抛出 CudaException
        public static void ThrowOnError(this int result)
        {
            if (result == 0)
                return;

            string message = "Unknown CUDA error";
            if (CudaDriverApi.cuGetErrorString(result, out IntPtr strPtr) == 0 && strPtr != IntPtr.Zero)
                message = Marshal.PtrToStringAnsi(strPtr) ?? message;

            throw new CudaException(result, message);
        }

        // 中文：检查 cuBLAS 返回码，非 0 时将状态码映射为可读信息并抛出 CudaException
        public static void ThrowOnCublasError(this int result)
        {
            if (result == 0)
                return;

            string message = result switch
            {
                1 => "CUBLAS_STATUS_NOT_INITIALIZED",
                3 => "CUBLAS_STATUS_ALLOC_FAILED",
                7 => "CUBLAS_STATUS_INVALID_VALUE",
                8 => "CUBLAS_STATUS_ARCH_MISMATCH",
                11 => "CUBLAS_STATUS_MAPPING_ERROR",
                13 => "CUBLAS_STATUS_EXECUTION_FAILED",
                14 => "CUBLAS_STATUS_INTERNAL_ERROR",
                15 => "CUBLAS_STATUS_NOT_SUPPORTED",
                _ => "Unknown cuBLAS error",
            };

            throw new CudaException(result, $"cuBLAS: {message}");
        }
    }
}
