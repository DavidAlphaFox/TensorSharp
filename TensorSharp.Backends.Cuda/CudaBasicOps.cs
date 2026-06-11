using System;
using TensorSharp.Core;

namespace TensorSharp.Cuda
{
    [OpsClass]
    public sealed class CudaBasicOps
    {
        // 中文：用标量值填充张量，优先走 CUDA 内核，失败时回退到 CPU 实现。
        [RegisterOpStorageType("fill", typeof(CudaStorage))]
        public static void Fill(Tensor result, float value)
        {
            if (result == null)
                throw new ArgumentNullException(nameof(result));

            if (CudaKernelOps.TryFill(result, value))
                return;

            CudaCpuFallback.InvokeVoid("fill", result, result, value);
        }

        // 中文：将源张量按逻辑拷贝到目标张量，优先 CUDA 拷贝，失败时回退 CPU 逐元素拷贝。
        [RegisterOpStorageType("copy", typeof(CudaStorage))]
        public static void Copy(Tensor result, Tensor src)
        {
            if (CudaKernelOps.TryCopy(result, src))
                return;

            CudaCpuFallback.CopyLogical(result, src);
        }

        // 中文：计算两个向量的点积，当前直接走 CPU 回退实现。
        [RegisterOpStorageType("dot", typeof(CudaStorage))]
        public static Tensor Dot(Tensor result, Tensor lhs, Tensor rhs)
        {
            return CudaCpuFallback.InvokeTensor("dot", result, result, lhs, rhs);
        }

        // 中文：矩阵乘加 beta*src + alpha*(m1·m2)，优先用 cuBLAS，失败时回退 CPU。
        [RegisterOpStorageType("addmm", typeof(CudaStorage))]
        public static Tensor Addmm(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            if (CudaBlas.TryAddmm(writeTarget, beta, src, alpha, m1, m2))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor("addmm", result, result, beta, src, alpha, m1, m2);
        }

        // 中文：批量矩阵乘加，对成批矩阵执行 beta*src + alpha*(m1·m2)，优先 cuBLAS 批处理。
        [RegisterOpStorageType("addmmbatch", typeof(CudaStorage))]
        public static Tensor AddmmBatch(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            if (CudaBlas.TryAddmmBatch(writeTarget, beta, src, alpha, m1, m2))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor("addmmbatch", result, result, beta, src, alpha, m1, m2);
        }

        // 中文：MoE 专家权重按 id 索引的矩阵乘（按专家选择权重再乘输入），走 CPU 回退。
        [RegisterOpStorageType("mulmatid", typeof(CudaStorage))]
        public static Tensor MulmatID(Tensor result, Tensor expertWeights, Tensor input, Tensor ids)
        {
            return CudaCpuFallback.InvokeTensor("mulmatid", result, result, expertWeights, input, ids);
        }

        // 中文：按 id 索引选择 bias 并加到源张量上（MoE 偏置加法），走 CPU 回退。
        [RegisterOpStorageType("addid", typeof(CudaStorage))]
        public static Tensor AddID(Tensor result, Tensor src, Tensor bias, Tensor ids)
        {
            return CudaCpuFallback.InvokeTensor("addid", result, result, src, bias, ids);
        }

        // 中文：逐元素取绝对值。
        [RegisterOpStorageType("abs", typeof(CudaStorage))]
        public static Tensor Abs(Tensor result, Tensor src) => Unary("abs", result, src);

        // 中文：逐元素取相反数（取负）。
        [RegisterOpStorageType("neg", typeof(CudaStorage))]
        public static Tensor Neg(Tensor result, Tensor src) => Unary("neg", result, src);

        // 中文：逐元素取符号（正/负/零返回 +1/-1/0）。
        [RegisterOpStorageType("sign", typeof(CudaStorage))]
        public static Tensor Sign(Tensor result, Tensor src) => Unary("sign", result, src);

        // 中文：逐元素求平方根。
        [RegisterOpStorageType("sqrt", typeof(CudaStorage))]
        public static Tensor Sqrt(Tensor result, Tensor src) => Unary("sqrt", result, src);

        // 中文：逐元素求平方根的倒数（1/sqrt(x)）。
        [RegisterOpStorageType("rsqrt", typeof(CudaStorage))]
        public static Tensor Rsqrt(Tensor result, Tensor src) => Unary("rsqrt", result, src);

        // 中文：逐元素求自然指数（e^x）。
        [RegisterOpStorageType("exp", typeof(CudaStorage))]
        public static Tensor Exp(Tensor result, Tensor src) => Unary("exp", result, src);

        // 中文：逐元素求自然对数（ln x）。
        [RegisterOpStorageType("log", typeof(CudaStorage))]
        public static Tensor Log(Tensor result, Tensor src) => Unary("log", result, src);

        // 中文：逐元素求 ln(1+x)，对接近 0 的输入更数值稳定。
        [RegisterOpStorageType("log1p", typeof(CudaStorage))]
        public static Tensor Log1p(Tensor result, Tensor src) => Unary("log1p", result, src);

        // 中文：逐元素向下取整。
        [RegisterOpStorageType("floor", typeof(CudaStorage))]
        public static Tensor Floor(Tensor result, Tensor src) => Unary("floor", result, src);

        // 中文：逐元素向上取整。
        [RegisterOpStorageType("ceil", typeof(CudaStorage))]
        public static Tensor Ceil(Tensor result, Tensor src) => Unary("ceil", result, src);

        // 中文：逐元素四舍五入到最近整数。
        [RegisterOpStorageType("round", typeof(CudaStorage))]
        public static Tensor Round(Tensor result, Tensor src) => Unary("round", result, src);

        // 中文：逐元素截断取整（向零取整，舍去小数部分）。
        [RegisterOpStorageType("trunc", typeof(CudaStorage))]
        public static Tensor Trunc(Tensor result, Tensor src) => Unary("trunc", result, src);

        // 中文：逐元素取小数部分（x 减去其截断整数）。
        [RegisterOpStorageType("frac", typeof(CudaStorage))]
        public static Tensor Frac(Tensor result, Tensor src) => Unary("frac", result, src);

        // 中文：ReLU 激活，逐元素取 max(0, x)，优先 CUDA 内核。
        [RegisterOpStorageType("relu", typeof(CudaStorage))]
        public static Tensor Relu(Tensor result, Tensor src) => Unary("relu", result, src, CudaUnaryOp.Relu);

        // 中文：逐元素求正弦 sin(x)。
        [RegisterOpStorageType("sin", typeof(CudaStorage))]
        public static Tensor Sin(Tensor result, Tensor src) => Unary("sin", result, src);

        // 中文：逐元素求余弦 cos(x)。
        [RegisterOpStorageType("cos", typeof(CudaStorage))]
        public static Tensor Cos(Tensor result, Tensor src) => Unary("cos", result, src);

        // 中文：逐元素求正切 tan(x)。
        [RegisterOpStorageType("tan", typeof(CudaStorage))]
        public static Tensor Tan(Tensor result, Tensor src) => Unary("tan", result, src);

        // 中文：逐元素求反正弦 asin(x)。
        [RegisterOpStorageType("asin", typeof(CudaStorage))]
        public static Tensor Asin(Tensor result, Tensor src) => Unary("asin", result, src);

        // 中文：逐元素求反余弦 acos(x)。
        [RegisterOpStorageType("acos", typeof(CudaStorage))]
        public static Tensor Acos(Tensor result, Tensor src) => Unary("acos", result, src);

        // 中文：逐元素求反正切 atan(x)。
        [RegisterOpStorageType("atan", typeof(CudaStorage))]
        public static Tensor Atan(Tensor result, Tensor src) => Unary("atan", result, src);

        // 中文：逐元素求双曲正弦 sinh(x)。
        [RegisterOpStorageType("sinh", typeof(CudaStorage))]
        public static Tensor Sinh(Tensor result, Tensor src) => Unary("sinh", result, src);

        // 中文：逐元素求双曲余弦 cosh(x)。
        [RegisterOpStorageType("cosh", typeof(CudaStorage))]
        public static Tensor Cosh(Tensor result, Tensor src) => Unary("cosh", result, src);

        // 中文：逐元素求双曲正切 tanh(x)（亦作激活函数），优先 CUDA 内核。
        [RegisterOpStorageType("tanh", typeof(CudaStorage))]
        public static Tensor Tanh(Tensor result, Tensor src) => Unary("tanh", result, src, CudaUnaryOp.Tanh);

        // 中文：Sigmoid 激活，逐元素求 1/(1+e^-x)，优先 CUDA 内核。
        [RegisterOpStorageType("sigmoid", typeof(CudaStorage))]
        public static Tensor Sigmoid(Tensor result, Tensor src) => Unary("sigmoid", result, src, CudaUnaryOp.Sigmoid);

        // 中文：SiLU/Swish 激活，逐元素求 x*sigmoid(x)，优先 CUDA 内核。
        [RegisterOpStorageType("SiLU", typeof(CudaStorage))]
        public static Tensor SiLU(Tensor result, Tensor src) => Unary("SiLU", result, src, CudaUnaryOp.SiLU);

        // 中文：GELU 高斯误差线性单元激活，逐元素计算，优先 CUDA 内核。
        [RegisterOpStorageType("GELU", typeof(CudaStorage))]
        public static Tensor GELU(Tensor result, Tensor src) => Unary("GELU", result, src, CudaUnaryOp.GELU);

        // 中文：张量逐元素加上一个标量值（lhs + rhs）。
        [RegisterOpStorageType("addv", typeof(CudaStorage))]
        public static Tensor AddValue(Tensor result, Tensor lhs, float rhs) => Scalar("addv", result, lhs, rhs, CudaScalarOp.Add);

        // 中文：张量逐元素减去一个标量值（lhs - rhs）。
        [RegisterOpStorageType("subv", typeof(CudaStorage))]
        public static Tensor SubValue(Tensor result, Tensor lhs, float rhs) => Scalar("subv", result, lhs, rhs, CudaScalarOp.Sub);

        // 中文：反向标量减法，标量减张量（lhs - rhs，标量在前）。
        [RegisterOpStorageType("rsubv", typeof(CudaStorage))]
        public static Tensor RSubValue(Tensor result, float lhs, Tensor rhs)
        {
            return Scalar("rsubv", result, rhs, lhs, CudaScalarOp.ReverseSub);
        }

        // 中文：张量逐元素乘以一个标量值（lhs * rhs）。
        [RegisterOpStorageType("mulv", typeof(CudaStorage))]
        public static Tensor MulValue(Tensor result, Tensor lhs, float rhs) => Scalar("mulv", result, lhs, rhs, CudaScalarOp.Mul);

        // 中文：张量逐元素除以一个标量值（lhs / rhs）。
        [RegisterOpStorageType("divv", typeof(CudaStorage))]
        public static Tensor DivValue(Tensor result, Tensor lhs, float rhs) => Scalar("divv", result, lhs, rhs, CudaScalarOp.Div);

        // 中文：反向标量除法，标量除以张量（lhs / rhs，标量在前）。
        [RegisterOpStorageType("rdivv", typeof(CudaStorage))]
        public static Tensor RDivValue(Tensor result, float lhs, Tensor rhs)
        {
            return Scalar("rdivv", result, rhs, lhs, CudaScalarOp.ReverseDiv);
        }

        // 中文：张量逐元素对标量取模（lhs % rhs），走 CPU 回退实现。
        [RegisterOpStorageType("modv", typeof(CudaStorage))]
        public static Tensor ModValue(Tensor result, Tensor lhs, float rhs) => ScalarFallback("modv", result, lhs, rhs);

        // 中文：两个张量逐元素相加（lhs + rhs）。
        [RegisterOpStorageType("addt", typeof(CudaStorage))]
        public static Tensor AddTensor(Tensor result, Tensor lhs, Tensor rhs) => Binary("addt", result, lhs, rhs, CudaBinaryOp.Add);

        // 中文：两个张量逐元素相减（lhs - rhs）。
        [RegisterOpStorageType("subt", typeof(CudaStorage))]
        public static Tensor SubTensor(Tensor result, Tensor lhs, Tensor rhs) => Binary("subt", result, lhs, rhs, CudaBinaryOp.Sub);

        // 中文：两个张量逐元素相乘（Hadamard 积，lhs * rhs）。
        [RegisterOpStorageType("mult", typeof(CudaStorage))]
        public static Tensor MulTensor(Tensor result, Tensor lhs, Tensor rhs) => Binary("mult", result, lhs, rhs, CudaBinaryOp.Mul);

        // 中文：两个张量逐元素相除（lhs / rhs）。
        [RegisterOpStorageType("divt", typeof(CudaStorage))]
        public static Tensor DivTensor(Tensor result, Tensor lhs, Tensor rhs) => Binary("divt", result, lhs, rhs, CudaBinaryOp.Div);

        // 中文：两个张量逐元素取模（lhs % rhs），走 CPU 回退实现。
        [RegisterOpStorageType("modt", typeof(CudaStorage))]
        public static Tensor ModTensor(Tensor result, Tensor lhs, Tensor rhs) => BinaryFallback("modt", result, lhs, rhs);

        // 中文：融合乘加，逐元素计算 x + y*z，优先 CUDA 三元内核。
        [RegisterOpStorageType("addmul", typeof(CudaStorage))]
        public static Tensor AddMul(Tensor result, Tensor x, Tensor y, Tensor z)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, x, false, x.Sizes);
            if (CudaKernelOps.TryTernary(writeTarget, x, y, z, CudaTernaryOp.AddMul))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor("addmul", writeTarget, writeTarget, x, y, z);
        }

        // 中文：融合加除，逐元素计算 x + y/z，优先 CUDA 三元内核。
        [RegisterOpStorageType("adddiv", typeof(CudaStorage))]
        public static Tensor AddDiv(Tensor result, Tensor x, Tensor y, Tensor z)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, x, false, x.Sizes);
            if (CudaKernelOps.TryTernary(writeTarget, x, y, z, CudaTernaryOp.AddDiv))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor("adddiv", writeTarget, writeTarget, x, y, z);
        }

        // 中文：融合标量乘加，逐元素计算 x + y*z（z 为标量），优先 CUDA 内核。
        [RegisterOpStorageType("addmulv", typeof(CudaStorage))]
        public static Tensor AddMulV(Tensor result, Tensor x, Tensor y, float z)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, x, false, x.Sizes);
            if (CudaKernelOps.TryAddMulScalar(writeTarget, x, y, z))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor("addmulv", writeTarget, writeTarget, x, y, z);
        }

        // 中文：四元融合运算，逐元素计算 x*y + z*w，优先 CUDA 内核。
        [RegisterOpStorageType("mulmuladd", typeof(CudaStorage))]
        public static Tensor MulMulAdd(Tensor result, Tensor x, Tensor y, Tensor z, Tensor w)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, x, false, x.Sizes);
            if (CudaKernelOps.TryMulMulAdd(writeTarget, x, y, z, w))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor("mulmuladd", writeTarget, writeTarget, x, y, z, w);
        }

        // 中文：SwiGLU 融合激活，逐元素计算 SiLU(gate)*up，优先 CUDA 二元激活内核。
        [RegisterOpStorageType("SiLUMul", typeof(CudaStorage))]
        public static Tensor SiLUMul(Tensor result, Tensor gate, Tensor up)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, gate, false, gate.Sizes);
            if (CudaKernelOps.TryBinaryActivation(writeTarget, gate, up, CudaBinaryActivationOp.SiLUMul))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor("SiLUMul", writeTarget, writeTarget, gate, up);
        }

        // 中文：将 gate-up 拼接张量按 halfDim 切分后做 SwiGLU，即 SiLU(gate)*up，失败时回退切分计算。
        [RegisterOpStorageType("SiLUMulSplit", typeof(CudaStorage))]
        public static Tensor SiLUMulSplit(Tensor result, Tensor gateUp, int halfDim)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, gateUp.Allocator, DType.Float32, false, gateUp.Sizes[0], halfDim);
            if (CudaKernelOps.TrySiLUMulSplit(writeTarget, gateUp, halfDim))
                return writeTarget;

            using (Tensor gate = gateUp.Narrow(1, 0, halfDim))
            using (Tensor up = gateUp.Narrow(1, halfDim, halfDim))
            {
                Ops.Copy(writeTarget, gate);
                Ops.SiLUMul(writeTarget, writeTarget, up);
            }

            return writeTarget;
        }

        // 中文：GeGLU 融合激活，逐元素计算 GELU(gate)*up，优先 CUDA 二元激活内核。
        [RegisterOpStorageType("GELUMul", typeof(CudaStorage))]
        public static Tensor GELUMul(Tensor result, Tensor gate, Tensor up)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, gate, false, gate.Sizes);
            if (CudaKernelOps.TryBinaryActivation(writeTarget, gate, up, CudaBinaryActivationOp.GELUMul))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor("GELUMul", writeTarget, writeTarget, gate, up);
        }

        // 中文：融合门控激活，逐元素计算 x*Sigmoid(gate)，优先 CUDA 二元激活内核。
        [RegisterOpStorageType("SigmoidMul", typeof(CudaStorage))]
        public static Tensor SigmoidMul(Tensor result, Tensor x, Tensor gate)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, x, false, x.Sizes);
            if (CudaKernelOps.TryBinaryActivation(writeTarget, x, gate, CudaBinaryActivationOp.SigmoidMul))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor("SigmoidMul", writeTarget, writeTarget, x, gate);
        }

        // 中文：沿指定维度做求和归约。
        [RegisterOpStorageType("sum", typeof(CudaStorage))]
        public static Tensor Sum(Tensor result, Tensor src, int dimension) => Reduce("sum", result, src, dimension);

        // 中文：沿指定维度做求平均归约。
        [RegisterOpStorageType("mean", typeof(CudaStorage))]
        public static Tensor Mean(Tensor result, Tensor src, int dimension) => Reduce("mean", result, src, dimension);

        // 中文：沿指定维度做连乘归约。
        [RegisterOpStorageType("prod", typeof(CudaStorage))]
        public static Tensor Prod(Tensor result, Tensor src, int dimension) => Reduce("prod", result, src, dimension);

        // 中文：沿指定维度求最小值归约。
        [RegisterOpStorageType("min", typeof(CudaStorage))]
        public static Tensor Min(Tensor result, Tensor src, int dimension) => Reduce("min", result, src, dimension);

        // 中文：沿指定维度求最大值归约。
        [RegisterOpStorageType("max", typeof(CudaStorage))]
        public static Tensor Max(Tensor result, Tensor src, int dimension) => Reduce("max", result, src, dimension);

        // 中文：沿指定维度求最小值所在索引（argmin）归约。
        [RegisterOpStorageType("argmin", typeof(CudaStorage))]
        public static Tensor Argmin(Tensor result, Tensor src, int dimension) => Reduce("argmin", result, src, dimension);

        // 中文：沿指定维度求最大值所在索引（argmax）归约。
        [RegisterOpStorageType("argmax", typeof(CudaStorage))]
        public static Tensor Argmax(Tensor result, Tensor src, int dimension) => Reduce("argmax", result, src, dimension);

        // 中文：对张量做 Softmax 归一化，优先 CUDA 内核，失败时回退 CPU。
        [RegisterOpStorageType("softmax", typeof(CudaStorage))]
        public static Tensor Softmax(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            if (CudaKernelOps.TrySoftmax(writeTarget, src))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor("softmax", writeTarget, writeTarget, src);
        }

        // 中文：Softmax 反向传播，计算其梯度，走 CPU 回退实现。
        [RegisterOpStorageType("softmaxgrad", typeof(CudaStorage))]
        public static Tensor SoftmaxGrad(Tensor grad, Tensor adj, Tensor val, bool addGrad)
        {
            return CudaCpuFallback.InvokeTensor("softmaxgrad", grad, grad, adj, val, addGrad);
        }

        // 中文：缩放点积注意力，按 query/key/value 与掩码计算注意力输出，优先 CUDA 内核。
        [RegisterOpStorageType("scaled_dot_product_attention", typeof(CudaStorage))]
        public static Tensor ScaledDotProductAttention(Tensor result, Tensor query, Tensor key, Tensor value, Tensor mask, float scale)
        {
            long[] outputSizes = new long[] { query.Sizes[0], query.Sizes[1], query.Sizes[2], value.Sizes[3] };
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, query.Allocator, query.ElementType, false, outputSizes);
            if (CudaKernelOps.TryScaledDotProductAttention(writeTarget, query, key, value, mask, scale))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor("scaled_dot_product_attention", writeTarget, writeTarget, query, key, value, mask, scale);
        }

        // 中文：按索引张量从源张量按行选取（如 embedding 查表），isAdd 控制累加，优先 CUDA 内核。
        [RegisterOpStorageType("indexselect", typeof(CudaStorage))]
        public static Tensor IndexSelect(Tensor result, Tensor src, Tensor indice, bool isAdd)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, indice.Sizes[0], src.Sizes[1]);
            if (CudaKernelOps.TryIndexSelect(writeTarget, src, indice, isAdd))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor("indexselect", writeTarget, writeTarget, src, indice, isAdd);
        }

        // 中文：IndexSelect 的反向传播，将梯度按索引散回，走 CPU 回退实现。
        [RegisterOpStorageType("indexselectgrad", typeof(CudaStorage))]
        public static Tensor IndexSelectGrad(Tensor grad, Tensor adj, Tensor indice)
        {
            return CudaCpuFallback.InvokeTensor("indexselectgrad", grad, grad, adj, indice);
        }

        // 中文：沿指定维度对元素逐个重复 repeats 次（repeat_interleave），走 CPU 回退实现。
        [RegisterOpStorageType("repeat_interleave", typeof(CudaStorage))]
        public static Tensor RepeatInterleave(Tensor result, Tensor src, int repeats, int dim)
        {
            return CudaCpuFallback.InvokeTensor("repeat_interleave", result, result, src, repeats, dim);
        }

        // 中文：为注意力分数原地叠加因果（下三角）掩码，将未来位置置为掩码值，优先 CUDA 内核。
        [RegisterOpStorageType("add_causal_mask", typeof(CudaStorage))]
        public static void AddCausalMask(Tensor tensor, int seqLen, int startPos, float maskedValue)
        {
            if (CudaKernelOps.TryAddCausalMask(tensor, seqLen, startPos, maskedValue))
                return;

            CudaCpuFallback.InvokeVoid("add_causal_mask", tensor, tensor, seqLen, startPos, maskedValue);
        }

        // 中文：取张量中前 k 个最大值及其索引（Top-K），走 CPU 回退实现。
        [RegisterOpStorageType("topK", typeof(CudaStorage))]
        public static Tensor TopK(Tensor outVal, Tensor outIdx, Tensor inVal, int k)
        {
            return CudaCpuFallback.InvokeTensor("topK", outVal, outVal, outIdx, inVal, k);
        }

        // 中文：应用旋转位置编码（RoPE），优先 CUDA 内核，失败时回退 CPU。
        [RegisterOpStorageType("rope", typeof(CudaStorage))]
        public static Tensor RoPE(Tensor result, Tensor src, int seqLen, int rowOffset)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            if (CudaKernelOps.TryRoPE(writeTarget, src, seqLen, rowOffset))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor("rope", writeTarget, writeTarget, src, seqLen, rowOffset);
        }

        // 中文：RoPE 旋转位置编码的反向传播，走 CPU 回退实现。
        [RegisterOpStorageType("ropegrad", typeof(CudaStorage))]
        public static Tensor RoPEGrad(Tensor grad, Tensor adj, int seqLen, int rowOffset)
        {
            return CudaCpuFallback.InvokeTensor("ropegrad", grad, grad, adj, seqLen, rowOffset);
        }

        // 中文：扩展版 RoPE，支持显式位置、NTK/YaRN 频率缩放等参数（freqBase/scale/extFactor 等），优先 CUDA 内核。
        [RegisterOpStorageType("rope_ex", typeof(CudaStorage))]
        public static Tensor RoPEEx(
            Tensor result,
            Tensor src,
            Tensor positions,
            int ropeDim,
            int mode,
            int originalContextLength,
            float freqBase,
            float freqScale,
            float extFactor,
            float attnFactor,
            float betaFast,
            float betaSlow,
            bool addToResult,
            bool invertPositions)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            if (CudaKernelOps.TryRoPEEx(
                writeTarget,
                src,
                positions,
                ropeDim,
                mode,
                originalContextLength,
                freqBase,
                freqScale,
                extFactor,
                attnFactor,
                betaFast,
                betaSlow,
                addToResult))
            {
                return writeTarget;
            }

            return CudaCpuFallback.InvokeTensor(
                "rope_ex",
                writeTarget,
                writeTarget,
                src,
                positions,
                ropeDim,
                mode,
                originalContextLength,
                freqBase,
                freqScale,
                extFactor,
                attnFactor,
                betaFast,
                betaSlow,
                addToResult,
                invertPositions);
        }

        // 中文：层归一化（LayerNorm），用 alpha 缩放、beta 偏移、eps 防止除零，走 CPU 回退实现。
        [RegisterOpStorageType("layernorm", typeof(CudaStorage))]
        public static Tensor LayerNorm(Tensor result, Tensor src, Tensor alpha, Tensor beta, float eps)
        {
            return CudaCpuFallback.InvokeTensor("layernorm", result, result, src, alpha, beta, eps);
        }

        // 中文：RMS 归一化（RMSNorm），用 alpha/beta 缩放偏移、eps 防止除零，优先 CUDA 内核。
        [RegisterOpStorageType("rmsnorm", typeof(CudaStorage))]
        public static Tensor RMSNorm(Tensor result, Tensor src, Tensor alpha, Tensor beta, float eps)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            if (CudaKernelOps.TryRMSNorm(writeTarget, src, alpha, beta, eps))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor("rmsnorm", writeTarget, writeTarget, src, alpha, beta, eps);
        }

        // 中文：先做残差相加（src1+src2）再做 LayerNorm 的融合算子，走 CPU 回退实现。
        [RegisterOpStorageType("addlayernorm", typeof(CudaStorage))]
        public static Tensor AddLayerNorm(Tensor result, Tensor src1, Tensor src2, Tensor alpha, Tensor beta, float eps)
        {
            return CudaCpuFallback.InvokeTensor("addlayernorm", result, result, src1, src2, alpha, beta, eps);
        }

        // 中文：沿指定维度按 indices 收集元素（gather），走 CPU 回退实现。
        [RegisterOpStorageType("gather", typeof(CudaStorage))]
        public static Tensor Gather(Tensor result, Tensor src, int dim, Tensor indices)
        {
            return CudaCpuFallback.InvokeTensor("gather", result, result, src, dim, indices);
        }

        // 中文：沿指定维度按 indices 将源值散布写入目标（scatter），走 CPU 回退实现。
        [RegisterOpStorageType("scatter", typeof(CudaStorage))]
        public static Tensor Scatter(Tensor result, Tensor src, int dim, Tensor indices)
        {
            return CudaCpuFallback.InvokeTensor("scatter", result, result, src, dim, indices);
        }

        // 中文：沿指定维度按 indices 将源值累加散布到目标（scatter_add），走 CPU 回退实现。
        [RegisterOpStorageType("scatter_add", typeof(CudaStorage))]
        public static Tensor ScatterAdd(Tensor result, Tensor src, int dim, Tensor indices)
        {
            return CudaCpuFallback.InvokeTensor("scatter_add", result, result, src, dim, indices);
        }

        // 中文：沿指定维度按 indices 将标量值散布填充到目标位置（scatter_fill），走 CPU 回退实现。
        [RegisterOpStorageType("scatter_fill", typeof(CudaStorage))]
        public static Tensor ScatterFill(Tensor result, float value, int dim, Tensor indices)
        {
            return CudaCpuFallback.InvokeTensor("scatter_fill", result, result, value, dim, indices);
        }

        // 中文：将 float32 张量转换为 float16（half）精度，走 CPU 回退实现。
        [RegisterOpStorageType("float2half", typeof(CudaStorage))]
        public static Tensor Float2Half(Tensor result, Tensor src)
        {
            return CudaCpuFallback.InvokeTensor("float2half", result, result, src);
        }

        // 中文：将 float16（half）张量转换为 float32 精度，走 CPU 回退实现。
        [RegisterOpStorageType("half2float", typeof(CudaStorage))]
        public static Tensor Half2Float(Tensor result, Tensor src)
        {
            return CudaCpuFallback.InvokeTensor("half2float", result, result, src);
        }

        // 中文：将 rhs 原子地累加到 result 上（atomic add），走 CPU 回退实现。
        [RegisterOpStorageType("atomicadd", typeof(CudaStorage))]
        public static Tensor AtomicAdd(Tensor result, Tensor rhs)
        {
            return CudaCpuFallback.InvokeTensor("atomicadd", result, result, rhs);
        }

        // 中文：一元算子的 CPU 回退分发助手，直接调用 CPU 实现执行命名运算。
        private static Tensor Unary(string opName, Tensor result, Tensor src)
        {
            return CudaCpuFallback.InvokeTensor(opName, result, result, src);
        }

        // 中文：一元算子分发助手，先尝试 CUDA 内核执行 cudaOp，失败时回退 CPU。
        private static Tensor Unary(string opName, Tensor result, Tensor src, CudaUnaryOp cudaOp)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            if (CudaKernelOps.TryUnary(writeTarget, src, cudaOp))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor(opName, writeTarget, writeTarget, src);
        }

        // 中文：二元算子分发助手，先尝试 CUDA 内核执行 cudaOp，失败时回退 CPU。
        private static Tensor Binary(string opName, Tensor result, Tensor lhs, Tensor rhs, CudaBinaryOp cudaOp)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Sizes);
            if (CudaKernelOps.TryBinary(writeTarget, lhs, rhs, cudaOp))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor(opName, writeTarget, writeTarget, lhs, rhs);
        }

        // 中文：张量-标量算子分发助手，先尝试 CUDA 内核执行 cudaOp，失败时回退 CPU。
        private static Tensor Scalar(string opName, Tensor result, Tensor lhs, float rhs, CudaScalarOp cudaOp)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Sizes);
            if (CudaKernelOps.TryScalar(writeTarget, lhs, rhs, cudaOp))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor(opName, writeTarget, writeTarget, lhs, rhs);
        }

        // 中文：二元算子的纯 CPU 回退助手，直接调用 CPU 实现执行命名运算。
        private static Tensor BinaryFallback(string opName, Tensor result, Tensor lhs, Tensor rhs)
        {
            return CudaCpuFallback.InvokeTensor(opName, result, result, lhs, rhs);
        }

        // 中文：张量-标量算子的纯 CPU 回退助手，直接调用 CPU 实现执行命名运算。
        private static Tensor ScalarFallback(string opName, Tensor result, Tensor lhs, float rhs)
        {
            return CudaCpuFallback.InvokeTensor(opName, result, result, lhs, rhs);
        }

        // 中文：归约算子的 CPU 回退分发助手，沿 dimension 维度执行命名归约运算。
        private static Tensor Reduce(string opName, Tensor result, Tensor src, int dimension)
        {
            return CudaCpuFallback.InvokeTensor(opName, result, result, src, dimension);
        }
    }
}
