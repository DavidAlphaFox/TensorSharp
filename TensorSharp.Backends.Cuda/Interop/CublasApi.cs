using System;
using System.Runtime.InteropServices;

namespace TensorSharp.Cuda.Interop
{
    internal static class CublasApi
    {
        private const string LibName = "cublas";

        // 中文：绑定 cublasCreate_v2，创建并初始化一个 cuBLAS 库句柄
        [DllImport(LibName, EntryPoint = "cublasCreate_v2")]
        public static extern int cublasCreate(out IntPtr handle);

        // 中文：绑定 cublasDestroy_v2，销毁 cuBLAS 句柄并释放其占用的资源
        [DllImport(LibName, EntryPoint = "cublasDestroy_v2")]
        public static extern int cublasDestroy(IntPtr handle);

        // 中文：绑定 cublasSetStream_v2，为该 cuBLAS 句柄设置后续运算所用的 CUDA 流
        [DllImport(LibName, EntryPoint = "cublasSetStream_v2")]
        public static extern int cublasSetStream(IntPtr handle, IntPtr stream);

        // 中文：绑定 cublasSetMathMode，设置数学运算模式（如启用 Tensor Core 加速）
        [DllImport(LibName)]
        public static extern int cublasSetMathMode(IntPtr handle, int mode);

        // 中文：绑定 cublasSgemm_v2，执行单精度浮点矩阵乘法 C = alpha*op(A)*op(B) + beta*C
        [DllImport(LibName, EntryPoint = "cublasSgemm_v2")]
        public static extern int cublasSgemm(
            IntPtr handle,
            int transa,
            int transb,
            int m,
            int n,
            int k,
            ref float alpha,
            IntPtr a,
            int lda,
            IntPtr b,
            int ldb,
            ref float beta,
            IntPtr c,
            int ldc);

        // 中文：绑定 cublasSgemmStridedBatched，按固定步长批量执行多组单精度矩阵乘法
        [DllImport(LibName)]
        public static extern int cublasSgemmStridedBatched(
            IntPtr handle,
            int transa,
            int transb,
            int m,
            int n,
            int k,
            ref float alpha,
            IntPtr a,
            int lda,
            long strideA,
            IntPtr b,
            int ldb,
            long strideB,
            ref float beta,
            IntPtr c,
            int ldc,
            long strideC,
            int batchCount);

        public const int CUBLAS_OP_N = 0;
        public const int CUBLAS_OP_T = 1;
        public const int CUBLAS_TENSOR_OP_MATH = 1;
    }
}
