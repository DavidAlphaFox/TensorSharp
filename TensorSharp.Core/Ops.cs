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

using TensorSharp.Core;

namespace TensorSharp
{
    public static class Ops
    {
        /// <summary>
        /// Logit offset for invalid attention positions before softmax. Matches GGML
        /// <c>ggml_compute_forward_diag_mask_inf</c> (-INFINITY) and Ollama/llama.cpp causal masking.
        /// </summary>
        public const float AttentionMaskMaskedLogit = float.NegativeInfinity;

        // 中文：将源张量复制为一块全新的连续内存张量并返回。
        public static Tensor NewContiguous(Tensor src)
        {
            Tensor result = new Tensor(src.Allocator, src.ElementType, src.Sizes);
            Copy(result, src);
            return result;
        }

        // 中文：若张量已连续则返回其引用，否则复制为连续张量（懒连续化）。
        public static Tensor AsContiguous(Tensor src)
        {
            if (src.IsContiguous())
            {
                return src.CopyRef();
            }
            else
            {
                return NewContiguous(src);
            }
        }

        // 中文：沿指定维度把多个输入张量拼接到 result 中。
        public static Tensor Concat(Tensor result, int dimension, params Tensor[] inputs)
        {
            return TensorConcatenation.Concat(result, dimension, inputs);
        }

        // 中文：将源张量元素逐一复制到目标张量。
        public static void Copy(Tensor result, Tensor src) { OpRegistry.Invoke("copy", result, src); }
        // 中文：用标量值填充整个张量。
        public static void Fill(Tensor result, float value) { OpRegistry.Invoke("fill", result, value); }

        // 中文：计算两个张量的点积（内积）。
        public static Tensor Dot(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("dot", result, lhs, rhs); }
        // 中文：矩阵乘加 result = beta*src + alpha*(m1×m2)。
        public static Tensor Addmm(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2) { return (Tensor)OpRegistry.Invoke("addmm", result, beta, src, alpha, m1, m2); }

        // 中文：批量版矩阵乘加 result = beta*src + alpha*(m1×m2)，按批次进行。
        public static Tensor AddmmBatch(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2) { return (Tensor)OpRegistry.Invoke("addmmbatch", result, beta, src, alpha, m1, m2); }
        // 中文：MoE 专家矩阵乘，按 ids 为每个 token 选择对应专家权重与输入相乘。
        public static Tensor MulmatID(Tensor result, Tensor expertWeights, Tensor input, Tensor ids) { return (Tensor)OpRegistry.Invoke("mulmatid", result, expertWeights, input, ids); }
        // 中文：MoE 按 ids 为每个 token 加上对应专家的偏置。
        public static Tensor AddID(Tensor result, Tensor src, Tensor bias, Tensor ids) { return (Tensor)OpRegistry.Invoke("addid", result, src, bias, ids); }

        // 中文：逐元素取绝对值。
        public static Tensor Abs(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("abs", result, src); }
        // 中文：逐元素取相反数（取负）。
        public static Tensor Neg(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("neg", result, src); }
        // 中文：逐元素取符号（-1/0/1）。
        public static Tensor Sign(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("sign", result, src); }



        // 中文：逐元素 SiLU（Swish）激活函数。
        public static Tensor SiLU(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("SiLU", result, src); }
        // 中文：逐元素 GELU 激活函数。
        public static Tensor GELU(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("GELU", result, src); }

        // 中文：SwiGLU 门控：SiLU(gate) 与 up 逐元素相乘。
        public static Tensor SiLUMul(Tensor result, Tensor gate, Tensor up) { return (Tensor)OpRegistry.Invoke("SiLUMul", result, gate, up); }
        // 中文：SwiGLU 融合版，将 gateUp 按 halfDim 拆分为 gate 与 up 后做 SiLU 门控相乘。
        public static Tensor SiLUMulSplit(Tensor result, Tensor gateUp, int halfDim) { return (Tensor)OpRegistry.Invoke("SiLUMulSplit", result, gateUp, halfDim); }
        // 中文：GeGLU 门控：GELU(gate) 与 up 逐元素相乘。
        public static Tensor GELUMul(Tensor result, Tensor gate, Tensor up) { return (Tensor)OpRegistry.Invoke("GELUMul", result, gate, up); }
        // 中文：Sigmoid 门控：x 与 Sigmoid(gate) 逐元素相乘。
        public static Tensor SigmoidMul(Tensor result, Tensor x, Tensor gate) { return (Tensor)OpRegistry.Invoke("SigmoidMul", result, x, gate); }

        // 中文：SiLU 反向传播，依据权重 srcW 与上游梯度 resG 计算输入梯度。
        public static Tensor SiLUD(Tensor result, Tensor srcW, Tensor resG) { return (Tensor)OpRegistry.Invoke("SiLUD", result, srcW, resG); }

        // 中文：SiLU 反向传播并将结果累加到已有梯度 srcG 上。
        public static Tensor AddSiLUD(Tensor result, Tensor srcG, Tensor srcW, Tensor resG) { return (Tensor)OpRegistry.Invoke("AddSiLUD", result, srcG, srcW, resG); }



        // 中文：将 float32 张量转换为 float16（半精度）。
        public static Tensor Float2Half(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("float2half", result, src); }
        // 中文：将 float16（半精度）张量转换为 float32。
        public static Tensor Half2Float(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("half2float", result, src); }


        // 中文：逐元素 ReLU 激活函数。
        public static Tensor Relu(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("relu", result, src); }

        // 中文：ReLU 反向传播，依据权重 w 与上游梯度 g 计算输入梯度。
        public static Tensor ReluD(Tensor result, Tensor w, Tensor g) { return (Tensor)OpRegistry.Invoke("relud", result, w, g); }

        // 中文：ReLU 反向传播并将结果累加到已有梯度 t 上。
        public static Tensor AddReluD(Tensor result, Tensor t, Tensor w, Tensor g) { return (Tensor)OpRegistry.Invoke("addrelud", result, t, w, g); }



        // 中文：逐元素 LeakyReLU 激活函数。
        public static Tensor LeakyReLU(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("LeakyReLU", result, src); }

        // 中文：LeakyReLU 反向传播，依据权重 w 与上游梯度 g 计算输入梯度。
        public static Tensor LeakyReLUD(Tensor result, Tensor w, Tensor g) { return (Tensor)OpRegistry.Invoke("LeakyReLUD", result, w, g); }

        // 中文：LeakyReLU 反向传播并将结果累加到已有梯度 t 上。
        public static Tensor AddLeakyReLUD(Tensor result, Tensor t, Tensor w, Tensor g) { return (Tensor)OpRegistry.Invoke("AddLeakyReLUD", result, t, w, g); }


        // 中文：逐元素平方根。
        public static Tensor Sqrt(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("sqrt", result, src); }

        // 中文：逐元素平方根的倒数（1/sqrt）。
        public static Tensor Rsqrt(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("rsqrt", result, src); }

        // 中文：逐元素自然指数 e^x。
        public static Tensor Exp(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("exp", result, src); }
        // 中文：逐元素自然对数 ln(x)。
        public static Tensor Log(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("log", result, src); }
        // 中文：逐元素 ln(1+x)，对小 x 更稳定。
        public static Tensor Log1p(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("log1p", result, src); }
        // 中文：逐元素向下取整。
        public static Tensor Floor(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("floor", result, src); }
        // 中文：逐元素向上取整。
        public static Tensor Ceil(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("ceil", result, src); }
        // 中文：逐元素四舍五入取整。
        public static Tensor Round(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("round", result, src); }
        // 中文：逐元素向零截断取整。
        public static Tensor Trunc(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("trunc", result, src); }
        // 中文：逐元素取小数部分。
        public static Tensor Frac(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("frac", result, src); }

        // 中文：逐元素正弦 sin。
        public static Tensor Sin(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("sin", result, src); }
        // 中文：逐元素余弦 cos。
        public static Tensor Cos(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("cos", result, src); }
        // 中文：逐元素正切 tan。
        public static Tensor Tan(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("tan", result, src); }

        // 中文：逐元素反正弦 arcsin。
        public static Tensor Asin(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("asin", result, src); }
        // 中文：逐元素反余弦 arccos。
        public static Tensor Acos(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("acos", result, src); }
        // 中文：逐元素反正切 arctan。
        public static Tensor Atan(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("atan", result, src); }

        // 中文：逐元素双曲正弦 sinh。
        public static Tensor Sinh(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("sinh", result, src); }
        // 中文：逐元素双曲余弦 cosh。
        public static Tensor Cosh(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("cosh", result, src); }
        // 中文：逐元素双曲正切 tanh（也作激活函数）。
        public static Tensor Tanh(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("tanh", result, src); }

        // 中文：逐元素 Sigmoid 激活函数。
        public static Tensor Sigmoid(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("sigmoid", result, src); }

        // 中文：Sigmoid 反向传播并将结果累加到已有梯度 t 上。
        public static Tensor AddSigmoidD(Tensor result, Tensor t, Tensor resW, Tensor resG) { return (Tensor)OpRegistry.Invoke("addsigmoidD", result, t, resW, resG); }

        // 中文：Tanh 反向传播并将结果累加到已有梯度 t 上。
        public static Tensor AddTanhD(Tensor result, Tensor t, Tensor resW, Tensor resG) { return (Tensor)OpRegistry.Invoke("addtanhD", result, t, resW, resG); }


        // 中文：Sigmoid 反向传播，依据输出 resW 与上游梯度 resG 计算输入梯度。
        public static Tensor SigmoidD(Tensor result, Tensor resW, Tensor resG) { return (Tensor)OpRegistry.Invoke("sigmoidD", result, resW, resG); }

        // 中文：Tanh 反向传播，依据输出 resW 与上游梯度 resG 计算输入梯度。
        public static Tensor TanhD(Tensor result, Tensor resW, Tensor resG) { return (Tensor)OpRegistry.Invoke("tanhD", result, resW, resG); }


        // 中文：融合运算 tanh(x+y)，两张量相加后取双曲正切。
        public static Tensor AddTanh(Tensor result, Tensor x, Tensor y) { return (Tensor)OpRegistry.Invoke("addtanh", result, x, y); }
        // 中文：融合运算 tanh(x+y+z)，三张量相加后取双曲正切。
        public static Tensor AddTanh3(Tensor result, Tensor x, Tensor y, Tensor z) { return (Tensor)OpRegistry.Invoke("addtanh3", result, x, y, z); }

        // 中文：融合运算 result = x*y + z*w（两次逐元素相乘再相加）。
        public static Tensor MulMulAdd(Tensor result, Tensor x, Tensor y, Tensor z, Tensor w) { return (Tensor)OpRegistry.Invoke("mulmuladd", result, x, y, z, w); }

        // 中文：融合运算 result = x + y*z（张量逐元素乘加）。
        public static Tensor AddMul(Tensor result, Tensor x, Tensor y, Tensor z) { return (Tensor)OpRegistry.Invoke("addmul", result, x, y, z); }
        // 中文：融合运算 result = x + y*z，其中 z 为标量。
        public static Tensor AddMulV(Tensor result, Tensor x, Tensor y, float z) { return (Tensor)OpRegistry.Invoke("addmulv", result, x, y, z); }

        // 中文：融合运算 result = x + y/z（张量逐元素除后相加）。
        public static Tensor AddDiv(Tensor result, Tensor x, Tensor y, Tensor z) { return (Tensor)OpRegistry.Invoke("adddiv", result, x, y, z); }



        // 中文：按掩码填充，mask 为真的位置写入 defValue，否则保留 t 的值。
        public static Tensor MaskFill(Tensor result, Tensor t, Tensor mask, float defValue) { return (Tensor)OpRegistry.Invoke("maskfill", result, t, mask, defValue); }

        // 中文：逐元素 atan2(srcY, srcX)，由坐标求方位角。
        public static Tensor Atan2(Tensor result, Tensor srcY, Tensor srcX) { return (Tensor)OpRegistry.Invoke("atan2", result, srcY, srcX); }
        // 中文：逐元素幂运算 src^value（指数为标量）。
        public static Tensor Pow(Tensor result, Tensor src, float value) { return (Tensor)OpRegistry.Invoke("pow", result, src, value); }
        // 中文：逐元素幂运算 value^src（底数为标量，指数为张量）。
        public static Tensor Tpow(Tensor result, float value, Tensor src) { return (Tensor)OpRegistry.Invoke("tpow", result, value, src); }
        // 中文：按权重在 srcA 与 srcB 之间做线性插值。
        public static Tensor Lerp(Tensor result, Tensor srcA, Tensor srcB, float weight) { return (Tensor)OpRegistry.Invoke("lerp", result, srcA, srcB); }
        // 中文：逐元素裁剪到 [min, max] 区间。
        public static Tensor Clamp(Tensor result, Tensor src, float min, float max) { return (Tensor)OpRegistry.Invoke("clamp", result, src, min, max); }

        // 中文：张量加标量 result = lhs + rhs。
        public static Tensor Add(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("addv", result, lhs, rhs); }
        // 中文：张量减标量 result = lhs - rhs。
        public static Tensor Sub(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("subv", result, lhs, rhs); }
        // 中文：标量减张量 result = lhs - rhs（标量在左）。
        public static Tensor Sub(Tensor result, float lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("rsubv", result, lhs, rhs); }
        // 中文：张量乘标量 result = lhs * rhs。
        public static Tensor Mul(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("mulv", result, lhs, rhs); }
        // 中文：张量除标量 result = lhs / rhs。
        public static Tensor Div(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("divv", result, lhs, rhs); }
        // 中文：标量除张量 result = lhs / rhs（标量在左）。
        public static Tensor Div(Tensor result, float lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("rdivv", result, lhs, rhs); }
        // 中文：张量对标量取模 result = lhs % rhs。
        public static Tensor Mod(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("modv", result, lhs, rhs); }

        // 中文：张量与标量逐元素比较 lhs > rhs，返回布尔/0-1 掩码。
        public static Tensor GreaterThan(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("gtValue", result, lhs, rhs); }
        // 中文：张量与标量逐元素比较 lhs < rhs，返回布尔/0-1 掩码。
        public static Tensor LessThan(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("ltValue", result, lhs, rhs); }
        // 中文：张量与标量逐元素比较 lhs >= rhs，返回布尔/0-1 掩码。
        public static Tensor GreaterOrEqual(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("geValue", result, lhs, rhs); }
        // 中文：张量与标量逐元素比较 lhs <= rhs，返回布尔/0-1 掩码。
        public static Tensor LessOrEqual(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("leValue", result, lhs, rhs); }
        // 中文：张量与标量逐元素比较 lhs == rhs，返回布尔/0-1 掩码。
        public static Tensor EqualTo(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("eqValue", result, lhs, rhs); }
        // 中文：张量与标量逐元素比较 lhs != rhs，返回布尔/0-1 掩码。
        public static Tensor NotEqual(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("neValue", result, lhs, rhs); }

        // 中文：两张量逐元素相加 result = lhs + rhs。
        public static Tensor Add(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("addt", result, lhs, rhs); }
        // 中文：两张量逐元素相减 result = lhs - rhs。
        public static Tensor Sub(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("subt", result, lhs, rhs); }
        // 中文：两张量逐元素相乘 result = lhs * rhs。
        public static Tensor Mul(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("mult", result, lhs, rhs); }
        // 中文：两张量逐元素相除 result = lhs / rhs。
        public static Tensor Div(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("divt", result, lhs, rhs); }
        // 中文：两张量逐元素取模 result = lhs % rhs。
        public static Tensor Mod(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("modt", result, lhs, rhs); }


        // 中文：原子累加，将 rhs 线程安全地累加到 result 上。
        public static Tensor AtomicAdd(Tensor result, Tensor rhs) { return (Tensor)OpRegistry.Invoke("atomicadd", result, rhs); }

        // 中文：两张量逐元素比较 lhs > rhs，返回布尔/0-1 掩码。
        public static Tensor GreaterThan(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("gtTensor", result, lhs, rhs); }
        // 中文：两张量逐元素比较 lhs < rhs，返回布尔/0-1 掩码。
        public static Tensor LessThan(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("ltTensor", result, lhs, rhs); }
        // 中文：两张量逐元素比较 lhs >= rhs，返回布尔/0-1 掩码。
        public static Tensor GreaterOrEqual(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("geTensor", result, lhs, rhs); }
        // 中文：两张量逐元素比较 lhs <= rhs，返回布尔/0-1 掩码。
        public static Tensor LessOrEqual(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("leTensor", result, lhs, rhs); }
        // 中文：两张量逐元素比较 lhs == rhs，返回布尔/0-1 掩码。
        public static Tensor EqualTo(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("eqTensor", result, lhs, rhs); }
        // 中文：两张量逐元素比较 lhs != rhs，返回布尔/0-1 掩码。
        public static Tensor NotEqual(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("neTensor", result, lhs, rhs); }


        // 中文：沿指定维度求和（降维归约）。
        public static Tensor Sum(Tensor result, Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("sum", result, src, dimension); }
        // 中文：沿指定维度求连乘积（降维归约）。
        public static Tensor Prod(Tensor result, Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("prod", result, src, dimension); }
        // 中文：沿指定维度求最小值（降维归约）。
        public static Tensor Min(Tensor result, Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("min", result, src, dimension); }
        // 中文：沿指定维度求最大值（降维归约）。
        public static Tensor Max(Tensor result, Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("max", result, src, dimension); }
        // 中文：沿指定维度求最小值所在的索引。
        public static Tensor Argmin(Tensor result, Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("argmin", result, src, dimension); }
        // 中文：沿指定维度求最大值所在的索引。
        public static Tensor Argmax(Tensor result, Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("argmax", result, src, dimension); }

        // 中文：沿指定维度求均值（降维归约）。
        public static Tensor Mean(Tensor result, Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("mean", result, src, dimension); }
        // 中文：沿指定维度求 p 范数（p = value）。
        public static Tensor Norm(Tensor result, Tensor src, int dimension, float value) { return (Tensor)OpRegistry.Invoke("norm", result, src, dimension, value); }
        // 中文：沿指定维度求标准差，normByN 决定除以 N 还是 N-1。
        public static Tensor Std(Tensor result, Tensor src, int dimension, bool normByN) { return (Tensor)OpRegistry.Invoke("std", result, src, dimension, normByN); }
        // 中文：沿指定维度求方差，normByN 决定除以 N 还是 N-1。
        public static Tensor Var(Tensor result, Tensor src, int dimension, bool normByN) { return (Tensor)OpRegistry.Invoke("var", result, src, dimension, normByN); }

        // 中文：检查张量是否含 NaN/Inf 等损坏数据，返回布尔结果。
        public static bool IsCorrupted(Tensor src) { return (bool)OpRegistry.Invoke("iscorrupted", src); }
        // 中文：沿最后一维做 Softmax 归一化。
        public static Tensor Softmax(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("softmax", result, src); }
        // 中文：Softmax 反向传播，依据前向输出 val 与上游梯度 adj 计算梯度，addGrad 控制是否累加。
        public static Tensor SoftmaxGrad(Tensor grad, Tensor adj, Tensor val, bool addGrad = true) { return (Tensor)OpRegistry.Invoke("softmaxgrad", grad, adj, val, addGrad); }


        // 中文：按索引选取行/切片（嵌入查表），isAdd 为真时累加到 result。
        public static Tensor IndexSelect(Tensor result, Tensor src, Tensor indice, bool isAdd = false) { return (Tensor)OpRegistry.Invoke("indexselect", result, src, indice, isAdd); }
        // 中文：IndexSelect 的反向传播，按索引将上游梯度 adj 散回到 grad。
        public static Tensor IndexSelectGrad(Tensor grad, Tensor adj, Tensor indice) { return (Tensor)OpRegistry.Invoke("indexselectgrad", grad, adj, indice); }

        /// <summary>
        /// Repeat each slice along <paramref name="dim"/> <paramref name="repeats"/> times consecutively.
        /// Input must be contiguous. Result shape is the same as src except dimension <paramref name="dim"/>
        /// is multiplied by <paramref name="repeats"/>.
        /// </summary>
        // 中文：沿 dim 维把每个切片连续重复 repeats 次（要求输入连续）。
        public static Tensor RepeatInterleave(Tensor result, Tensor src, int repeats, int dim) { return (Tensor)OpRegistry.Invoke("repeat_interleave", result, src, repeats, dim); }

        /// <summary>
        /// In-place causal mask: for each logical row t (with period <paramref name="seqLen"/> across outer dims),
        /// add <paramref name="maskedValue"/> to all positions s where s &gt; startPos + t.
        /// The tensor's last dimension is the key-sequence dimension.
        /// </summary>
        // 中文：就地加因果掩码，对未来位置加 maskedValue 实现自回归注意力屏蔽。
        public static void AddCausalMask(Tensor tensor, int seqLen, int startPos, float maskedValue) { OpRegistry.Invoke("add_causal_mask", tensor, seqLen, startPos, maskedValue); }

        // 中文：按行索引选取行（IndexSelect 的便捷封装，等价于嵌入查表）。
        public static Tensor Rows(Tensor result, Tensor src, Tensor indices) { return IndexSelect(result, src, indices, false); }
        // 中文：按行索引写回行（Scatter 在第 0 维的便捷封装）。
        public static Tensor SetRows(Tensor result, Tensor src, Tensor indices) { return Scatter(result, src, 0, indices); }


        // 中文：旋转位置编码 RoPE 前向，对 src 按序列位置施加旋转。
        public static Tensor RoPE(Tensor result, Tensor src, int seqLen, int rowOffset) { return (Tensor)OpRegistry.Invoke("rope", result, src, seqLen, rowOffset); }
        // 中文：RoPE 反向传播，按上游梯度 adj 计算输入梯度。
        public static Tensor RoPEGrad(Tensor grad, Tensor adj, int seqLen, int rowOffset) { return (Tensor)OpRegistry.Invoke("ropegrad", grad, adj, seqLen, rowOffset); }
        // 中文：扩展版 RoPE，支持显式位置、NTK/YaRN 缩放等参数（freqBase、extFactor 等）。
        public static Tensor RoPEEx(Tensor result, Tensor src, Tensor positions, int ropeDim, int mode, int originalContextLength, float freqBase, float freqScale, float extFactor = 0.0f, float attnFactor = 1.0f, float betaFast = 0.0f, float betaSlow = 0.0f, bool addToResult = false, bool invertPositions = false)
        {
            return (Tensor)OpRegistry.Invoke("rope_ex", result, src, positions, ropeDim, mode, originalContextLength, freqBase, freqScale, extFactor, attnFactor, betaFast, betaSlow, addToResult, invertPositions);
        }


        // 中文：构建源-目标注意力掩码，按原始长度区分有效/填充位置（用于编码器-解码器交叉注意力）。
        public static Tensor BuildSrcTgtMask(Tensor result, Tensor srcOriginalLengths, Tensor tgtOriginalLengths, int srcPaddedSeqLength, int tgtPaddedSeqLength, float value, float maskedValue)
        {
            return (Tensor)OpRegistry.Invoke("buildsrctgtmask", result, srcOriginalLengths, tgtOriginalLengths, srcPaddedSeqLength, tgtPaddedSeqLength, value, maskedValue);
        }

        // 中文：构建自注意力填充掩码，按原始长度屏蔽 padding 位置。
        public static Tensor BuildSelfMask(Tensor result, Tensor originalLengths, int paddedSeqLength, float value, float maskedValue)
        {
            return (Tensor)OpRegistry.Invoke("buildselfmask", result, originalLengths, paddedSeqLength, value, maskedValue);
        }

        // 中文：构建自注意力下三角因果掩码，并同时按原始长度屏蔽 padding 位置。
        public static Tensor BuildSelfTriMask(Tensor result, Tensor originalLengths, int paddedSeqLength, float value, float maskedValue)
        {
            return (Tensor)OpRegistry.Invoke("buildselftrimask", result, originalLengths, paddedSeqLength, value, maskedValue);
        }

        // 中文：构建纯下三角因果掩码（不考虑 padding 长度）。
        public static Tensor BuildTriMask(Tensor result, float value, float maskedValue)
        {
            return (Tensor)OpRegistry.Invoke("buildtrimask", result, value, maskedValue);
        }


        // 中文：取前 k 大元素，分别输出值 outVal 与索引 outIdx。
        public static Tensor TopK(Tensor outVal, Tensor outIdx, Tensor inVal, int k)
        {
            return (Tensor)OpRegistry.Invoke("topK", outVal, outIdx, inVal, k);
        }

        // 中文：层归一化 LayerNorm，使用增益 alpha 与偏置 beta，eps 防止除零。
        public static Tensor LayerNorm(Tensor result, Tensor src, Tensor alpha, Tensor beta, float eps = 1e-09f) { return (Tensor)OpRegistry.Invoke("layernorm", result, src, alpha, beta, eps); }
        // 中文：LayerNorm 反向传播，计算输入及 alpha/beta 的梯度。
        public static Tensor LayerNormGrad(Tensor outGrad, Tensor alphaGrad, Tensor betaGrad, Tensor inGrad, Tensor y, Tensor x, Tensor alpha, Tensor beta, float eps = 1e-09f) 
        { 
            return (Tensor)OpRegistry.Invoke("layernormgrad", outGrad, alphaGrad, betaGrad, inGrad, y, x, alpha, beta, eps);
        }

        // 中文：FlashAttention 前向，由 Q/K/V 计算注意力输出 O 与对数和 L。
        public static Tensor FlashAttention(Tensor O, Tensor L, Tensor Q, Tensor K, Tensor V, int q_start_offset = 0)
        {
            return (Tensor)OpRegistry.Invoke("flashattention", O, L, Q, K, V, q_start_offset);
        }

        // 中文：缩放点积注意力 SDPA，含可选 mask 与缩放因子 scale。
        public static Tensor ScaledDotProductAttention(Tensor result, Tensor query, Tensor key, Tensor value, Tensor mask, float scale)
        {
            return (Tensor)OpRegistry.Invoke("scaled_dot_product_attention", result, query, key, value, mask, scale);
        }

        // 中文：FlashAttention 反向传播，计算 Q/K/V 的梯度 dQ/dK/dV。
        public static void FlashAttentionGrad(Tensor Q, Tensor K, Tensor V, Tensor O, Tensor dO, Tensor L, Tensor dQ, Tensor dK, Tensor dV)
        {
            OpRegistry.Invoke("flashattentiongrad", Q, K, V, O, dO, L, dQ, dK, dV);
        }

        // 中文：RMS 归一化 RMSNorm，使用增益 alpha 与偏置 beta，eps 防止除零。
        public static Tensor RMSNorm(Tensor result, Tensor src, Tensor alpha, Tensor beta, float eps = 1e-09f) { return (Tensor)OpRegistry.Invoke("rmsnorm", result, src, alpha, beta, eps); }
        // 中文：RMSNorm 反向传播，计算输入及 alpha/beta 的梯度。
        public static Tensor RMSNormGrad(Tensor outGrad, Tensor alphaGrad, Tensor betaGrad, Tensor inGrad, Tensor y, Tensor x, Tensor alpha, Tensor beta, float eps = 1e-09f)
        {
            return (Tensor)OpRegistry.Invoke("rmsnormgrad", outGrad, alphaGrad, betaGrad, inGrad, y, x, alpha, beta, eps);
        }

        // 中文：先将 src1 与 src2 相加（残差连接）再做 LayerNorm 的融合算子。
        public static Tensor AddLayerNorm(Tensor result, Tensor src1, Tensor src2, Tensor alpha, Tensor beta, float eps = 1e-09f) { return (Tensor)OpRegistry.Invoke("addlayernorm", result, src1, src2, alpha, beta, eps); }
        // 中文：AddLayerNorm 反向传播，计算两个输入 x1/x2 及 alpha/beta 的梯度。
        public static Tensor AddLayerNormGrad(Tensor out1Grad, Tensor out2Grad, Tensor alphaGrad, Tensor betaGrad, Tensor inGrad, Tensor y, Tensor x1, Tensor x2, Tensor alpha, Tensor beta, float eps = 1e-09f) { return (Tensor)OpRegistry.Invoke("addlayernormgrad", out1Grad, out2Grad, alphaGrad, betaGrad, inGrad, y, x1, x2, alpha, beta, eps); }

        // 中文：Adam 优化器更新一步，依据梯度与一二阶矩 m/v 更新权重。
        public static Tensor Adam(Tensor weight, Tensor gradient, Tensor v, Tensor m, float gradNormFactor, float step_size, float clipval, float regc, float decay_rate_v, float decay_rate_m, int iter, float eps)
        {
            return (Tensor)OpRegistry.Invoke("adam", weight, gradient, v, m, gradNormFactor, step_size, clipval, regc, decay_rate_v, decay_rate_m, iter, eps);
        }

        // 中文：RMSProp 优化器更新一步，依据梯度与平方均值缓存 cache 更新权重。
        public static Tensor RMSProp(Tensor weight, Tensor gradient, Tensor cache, float gradNormFactor, float step_size, float clipval, float regc, float decay_rate, float eps)
        {
            return (Tensor)OpRegistry.Invoke("rmsprop", weight, gradient, cache, gradNormFactor, step_size, clipval, regc, decay_rate, eps);
        }

        // 中文：对所有元素求和，结果以 1 元素张量返回。
        public static Tensor SumAll(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("sumall", result, src); }
        // 中文：对所有元素求连乘积，结果以 1 元素张量返回。
        public static Tensor ProdAll(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("prodall", result, src); }
        // 中文：对所有元素求最小值，结果以 1 元素张量返回。
        public static Tensor MinAll(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("minall", result, src); }
        // 中文：对所有元素求最大值，结果以 1 元素张量返回。
        public static Tensor MaxAll(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("maxall", result, src); }

        // 中文：对所有元素求均值，结果以 1 元素张量返回。
        public static Tensor MeanAll(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("meanall", result, src); }
        // 中文：对所有元素求 p 范数（p = value），结果以 1 元素张量返回。
        public static Tensor NormAll(Tensor result, Tensor src, float value) { return (Tensor)OpRegistry.Invoke("normall", result, src, value); }
        // 中文：对所有元素求标准差，结果以 1 元素张量返回。
        public static Tensor StdAll(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("stdall", result, src); }
        // 中文：对所有元素求方差，结果以 1 元素张量返回。
        public static Tensor VarAll(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("varall", result, src); }


        // 中文：对所有元素求和并取出标量 float 返回（重载：直接返回数值）。
        public static float SumAll(Tensor src) { using (Tensor resultTensor = SumAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        // 中文：对所有元素求连乘积并取出标量 float 返回（重载：直接返回数值）。
        public static float ProdAll(Tensor src) { using (Tensor resultTensor = ProdAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        // 中文：对所有元素求最小值并取出标量 float 返回（重载：直接返回数值）。
        public static float MinAll(Tensor src) { using (Tensor resultTensor = MinAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        // 中文：对所有元素求最大值并取出标量 float 返回（重载：直接返回数值）。
        public static float MaxAll(Tensor src) { using (Tensor resultTensor = MaxAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }

        // 中文：对所有元素求均值并取出标量 float 返回（重载：直接返回数值）。
        public static float MeanAll(Tensor src) { using (Tensor resultTensor = MeanAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        // 中文：对所有元素求方差并取出标量 float 返回（重载：直接返回数值）。
        public static float VarAll(Tensor src) { using (Tensor resultTensor = VarAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        // 中文：对所有元素求标准差并取出标量 float 返回（重载：直接返回数值）。
        public static float StdAll(Tensor src) { using (Tensor resultTensor = StdAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        // 中文：对所有元素求 p 范数并取出标量 float 返回（重载：直接返回数值）。
        public static float NormAll(Tensor src, float value) { using (Tensor resultTensor = NormAll(null, src, value)) { return resultTensor.GetElementAsFloat(0); } }


     //   public static Tensor IndexSelect(Tensor result, Tensor src, int dim, Tensor indices) { return (Tensor)OpRegistry.Invoke("index_select", result, src, dim, indices); }
        // 中文：沿 dim 维按 indices 聚集元素（gather）。
        public static Tensor Gather(Tensor result, Tensor src, int dim, Tensor indices) { return (Tensor)OpRegistry.Invoke("gather", result, src, dim, indices); }
        // 中文：沿 dim 维按 indices 将 src 散布写入 result（scatter，覆盖写）。
        public static Tensor Scatter(Tensor result, Tensor src, int dim, Tensor indices) { return (Tensor)OpRegistry.Invoke("scatter", result, src, dim, indices); }


        // 中文：沿 dim 维按 indices 将 src 累加散布到 result（scatter-add）。
        public static Tensor ScatterAdd(Tensor result, Tensor src, int dim, Tensor indices) { return (Tensor)OpRegistry.Invoke("scatter_add", result, src, dim, indices); }

        // 中文：沿 dim 维按 indices 将标量 value 散布写入 result（scatter 填充）。
        public static Tensor ScatterFill(Tensor result, float value, int dim, Tensor indices) { return (Tensor)OpRegistry.Invoke("scatter_fill", result, value, dim, indices); }


        // 中文：从随机数发生器取下一个种子，src 为空时返回 null。
        public static int? GetSeed(RandomGenerator src)
        {
            return src == null ? (int?)null : src.NextSeed();
        }

        // 中文：用 [min, max] 均匀分布随机数填充张量。
        public static void RandomUniform(Tensor result, RandomGenerator seedSource, float min, float max) { OpRegistry.Invoke("random_uniform", result, GetSeed(seedSource), min, max); }
        // 中文：用指定均值与标准差的正态分布随机数填充张量。
        public static void RandomNormal(Tensor result, RandomGenerator seedSource, float mean, float stdv) { OpRegistry.Invoke("random_normal", result, GetSeed(seedSource), mean, stdv); }
        // 中文：用参数 lambda 的指数分布随机数填充张量。
        public static void RandomExponential(Tensor result, RandomGenerator seedSource, float lambda) { OpRegistry.Invoke("random_exponential", result, GetSeed(seedSource), lambda); }
        // 中文：用指定中位数与尺度 sigma 的柯西分布随机数填充张量。
        public static void RandomCauchy(Tensor result, RandomGenerator seedSource, float median, float sigma) { OpRegistry.Invoke("random_cauchy", result, GetSeed(seedSource), median, sigma); }
        // 中文：用指定参数的对数正态分布随机数填充张量。
        public static void RandomLogNormal(Tensor result, RandomGenerator seedSource, float mean, float stdv) { OpRegistry.Invoke("random_lognormal", result, GetSeed(seedSource), mean, stdv); }
        // 中文：用成功概率 p 的几何分布随机数填充张量。
        public static void RandomGeometric(Tensor result, RandomGenerator seedSource, float p) { OpRegistry.Invoke("random_geometric", result, GetSeed(seedSource), p); }
        // 中文：用成功概率 p 的伯努利分布（0/1）随机数填充张量。
        public static void RandomBernoulli(Tensor result, RandomGenerator seedSource, float p) { OpRegistry.Invoke("random_bernoulli", result, GetSeed(seedSource), p); }
    }
}
