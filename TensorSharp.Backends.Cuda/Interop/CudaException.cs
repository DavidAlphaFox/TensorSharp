using System;

namespace TensorSharp.Cuda.Interop
{
    // 中文：表示 CUDA/cuBLAS 原生调用失败的异常类型，携带错误码与描述信息
    public sealed class CudaException : Exception
    {
        // 中文：构造异常，记录错误码并生成包含错误码与描述的异常消息
        public CudaException(int errorCode, string message)
            : base($"CUDA error {errorCode}: {message}")
        {
            ErrorCode = errorCode;
        }

        public int ErrorCode { get; }
    }
}
