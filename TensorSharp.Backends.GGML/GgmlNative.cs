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
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;

namespace TensorSharp.GGML
{

public enum GgmlBackendType
{
    Metal = 1,
    Cpu = 2,
    Cuda = 3,
}

    [StructLayout(LayoutKind.Sequential)]
    internal readonly struct GgmlTensorView2D
    {
        public readonly IntPtr Data;
        public readonly int Dim0;
        public readonly int Dim1;
        public readonly int Stride0;
        public readonly int Stride1;
        public readonly long RawBytes;

        public GgmlTensorView2D(IntPtr data, int dim0, int dim1, int stride0, int stride1, long rawBytes)
        {
            Data = data;
            Dim0 = dim0;
            Dim1 = dim1;
            Stride0 = stride0;
            Stride1 = stride1;
            RawBytes = rawBytes;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    internal readonly struct GgmlTensorView3D
    {
        public readonly IntPtr Data;
        public readonly int Dim0;
        public readonly int Dim1;
        public readonly int Dim2;
        public readonly int Stride0;
        public readonly int Stride1;
        public readonly int Stride2;
        public readonly long RawBytes;

        public GgmlTensorView3D(IntPtr data, int dim0, int dim1, int dim2, int stride0, int stride1, int stride2, long rawBytes)
        {
            Data = data;
            Dim0 = dim0;
            Dim1 = dim1;
            Dim2 = dim2;
            Stride0 = stride0;
            Stride1 = stride1;
            Stride2 = stride2;
            RawBytes = rawBytes;
        }
    }

[StructLayout(LayoutKind.Sequential)]
internal readonly struct GgmlTensorView4D
{
    public readonly IntPtr Data;
    public readonly int Ne0;
    public readonly int Ne1;
    public readonly int Ne2;
    public readonly int Ne3;
    public readonly long Nb1;
    public readonly long Nb2;
    public readonly long Nb3;
    public readonly long RawBytes;

    public GgmlTensorView4D(IntPtr data, int ne0, int ne1, int ne2, int ne3, long nb1, long nb2, long nb3, long rawBytes)
    {
        Data = data;
        Ne0 = ne0;
        Ne1 = ne1;
        Ne2 = ne2;
        Ne3 = ne3;
        Nb1 = nb1;
        Nb2 = nb2;
        Nb3 = nb3;
        RawBytes = rawBytes;
    }
}

[StructLayout(LayoutKind.Sequential)]
internal readonly struct GgmlContiguousTensor
{
    public readonly IntPtr Data;
    public readonly long ElementCount;
    public readonly int ElementType;

    public GgmlContiguousTensor(IntPtr data, long elementCount, DType elementType)
    {
        Data = data;
        ElementCount = elementCount;
        ElementType = (int)elementType;
    }
}

[StructLayout(LayoutKind.Sequential)]
internal readonly struct GgmlQuantizedWeight
{
    public readonly IntPtr Data;
    public readonly int GgmlType;
    public readonly long Ne0;
    public readonly long Ne1;
    public readonly long RawBytes;

    public GgmlQuantizedWeight(IntPtr data, int ggmlType, long ne0, long ne1, long rawBytes)
    {
        Data = data;
        GgmlType = ggmlType;
        Ne0 = ne0;
        Ne1 = ne1;
        RawBytes = rawBytes;
    }
}

// Descriptor for the fused single-layer Gemma 4 MoE decode kernel
// (TSGgml_Gemma4MoELayerDecode). Field order/types MUST match the native
// TSGgmlGemma4MoELayerDesc struct EXACTLY: all 8-byte fields (pointers then
// int64) first, then 4-byte fields (int32 then float). StructBytes is a
// sizeof() sanity check the native side validates before use.
[StructLayout(LayoutKind.Sequential)]
public struct Gemma4MoELayerDecodeArgs
{
    // pointers (24)
    public IntPtr Hidden;
    public IntPtr AttnNormW;
    public IntPtr QkvW;
    public IntPtr KW;
    public IntPtr VW;
    public IntPtr QNormW;
    public IntPtr KNormW;
    public IntPtr OW;
    public IntPtr PostAttnNormW;
    public IntPtr KCache;
    public IntPtr VCache;
    public IntPtr FreqFactors;
    public IntPtr FfnNormW;
    public IntPtr GuW;
    public IntPtr DownW;
    public IntPtr PostFfwNorm1W;
    public IntPtr GateInpW;
    public IntPtr GateInpScale;
    public IntPtr PreFfwNorm2W;
    public IntPtr GateUpExps;
    public IntPtr DownExps;
    public IntPtr DownExpsScale;
    public IntPtr PostFfwNorm2W;
    public IntPtr PostFfwNormW;

    // int64 weight shapes (24)
    public long QkvNe0, QkvNe1, QkvBytes;
    public long KNe0, KNe1, KBytes;
    public long VNe0, VNe1, VBytes;
    public long ONe0, ONe1, OBytes;
    public long GuNe0, GuNe1, GuBytes;
    public long DownNe0, DownNe1, DownBytes;
    public long GueNe0, GueNe1, GueBytes;
    public long DeNe0, DeNe1, DeBytes;

    // int32 scalars / shapes (24)
    public int StructBytes;
    public int HiddenSize;
    public int NumHeads;
    public int NumKvHeads;
    public int HeadDim;
    public int CacheSize;
    public int IsLocal;
    public int IsShared;
    public int SlidingWindow;
    public int Position;
    public int RopeNDims;
    public int KvCacheType;
    public int NumExperts;
    public int NumExpertsUsed;
    public int FreqFactorsLen;
    public int QkvType;
    public int KType;
    public int VType;
    public int OType;
    public int GuType;
    public int DownType;
    public int GueType;
    public int DeType;
    public int SeparateQkv;

    // float scalars (4)
    public float Eps;
    public float RopeBase;
    public float InvSqrtHidden;
    public float LayerOutputScale;
}

internal enum GgmlUnaryOp
{
    Neg = 1,
    Exp = 2,
    Log = 3,
    Sqrt = 4,
    Relu = 5,
    Sigmoid = 6,
    Tanh = 7,
    SiLU = 8,
    Step = 9,
    Abs = 10,
    Sign = 11,
    GELU = 12,
}

internal enum GgmlFusedActMulOp
{
    SiLUMul = 1,
    GELUMul = 2,
    SigmoidMul = 3,
}

internal enum GgmlBinaryTensorOp
{
    Add = 1,
    Sub = 2,
    Mul = 3,
    Div = 4,
}

internal enum GgmlBinaryScalarOp
{
    Add = 1,
    Sub = 2,
    ReverseSub = 3,
    Mul = 4,
    Div = 5,
    ReverseDiv = 6,
}

internal enum GgmlActivationGradOp
{
    Relu = 1,
    Sigmoid = 2,
    Tanh = 3,
    SiLU = 4,
}

internal enum GgmlNormOp
{
    LayerNorm = 1,
    RmsNorm = 2,
}

internal enum GgmlReductionOp
{
    Sum = 1,
    Mean = 2,
}

internal enum GgmlIndexReductionOp
{
    Argmin = 1,
    Argmax = 2,
}

    internal static class GgmlNative
    {
        private const string DllName = "GgmlOps";
        private const CallingConvention CallingConventionType = CallingConvention.Cdecl;
        private static int s_windowsDependencySearchPathsInitialized;

        // 中文：静态构造函数，注册原生库的 DllImport 解析器。
        static GgmlNative()
        {
            NativeLibrary.SetDllImportResolver(typeof(GgmlNative).Assembly, ImportResolver);
        }

        // 中文：获取原生层最近一次错误的描述字符串指针。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern IntPtr TSGgml_GetLastError();

        // 中文：查询当前环境是否支持 Metal 后端。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_IsMetalAvailable();

        // 中文：检测指定后端类型（Metal/CPU/CUDA）能否被成功初始化。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_CanInitializeBackend(int backendType);

        // 中文：查询指定后端类型当前是否可用。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_IsBackendAvailable(int backendType);

        // 中文：F32 的 addmm 运算 result = beta*src + alpha*(m1·m2)（带偏置的矩阵乘）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_AddmmF32(
            GgmlTensorView2D result,
            GgmlTensorView2D src,
            GgmlTensorView2D m1,
            GgmlTensorView2D m2,
            float beta,
            float alpha);

        // 中文：输入为 F32、权重 m2 为量化格式的矩阵乘 result = m1·m2。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_AddmmQuantF32(
            GgmlTensorView2D result,
            GgmlTensorView2D m1,
            IntPtr m2Data,
            int m2GgmlType,
            long m2Ne0,
            long m2Ne1,
            long m2RawBytes);

        // 中文：融合算子，先对输入做 RMSNorm 再与量化权重做矩阵乘。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_FusedRmsNormMatMulQuantF32(
            GgmlTensorView2D result,
            GgmlTensorView2D input,
            IntPtr normWeightData,
            int normWeightCount,
            float eps,
            IntPtr m2Data,
            int m2GgmlType,
            long m2Ne0,
            long m2Ne1,
            long m2RawBytes);

        // 中文：融合算子，量化矩阵乘后再加上残差 residual += input·m2。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_FusedMatMulQuantAddF32(
            GgmlTensorView2D residual,
            GgmlTensorView2D input,
            IntPtr m2Data,
            int m2GgmlType,
            long m2Ne0,
            long m2Ne1,
            long m2RawBytes);

        // 中文：融合的 SwiGLU 前馈网络（RMSNorm + gate_up + SwiGLU + down + 残差，量化权重）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_FusedFFNSwiGLUQuantF32(
            GgmlTensorView2D residual,
            GgmlTensorView2D input,
            IntPtr normWeightData,
            int normWeightCount,
            float eps,
            IntPtr gateUpData,
            int gateUpGgmlType,
            long gateUpNe0,
            long gateUpNe1,
            long gateUpRawBytes,
            IntPtr downData,
            int downGgmlType,
            long downNe0,
            long downNe1,
            long downRawBytes,
            int halfDim);

        // 中文：融合算子，输出投影 + 残差 + 归一化 + MoE 路由器打分（量化权重）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_FusedOutProjNormRouterQuantF32(
            GgmlTensorView2D residual, GgmlTensorView2D input,
            IntPtr outProjData, int outProjType, long outNe0, long outNe1, long outBytes,
            IntPtr normData, int normCount, float eps,
            GgmlTensorView2D normedOut,
            IntPtr routerData, int routerType, long routerNe0, long routerNe1, long routerBytes,
            GgmlTensorView2D routerOut);

        // 中文：视觉编码器的融合 MLP（LayerNorm + 上投影 + 激活 + 下投影，含偏置）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_FusedVisionMLPF32(
            GgmlTensorView2D hidden,
            IntPtr lnW, IntPtr lnB, int lnDim, float eps,
            IntPtr upW, int upNe0, int upNe1, long upBytes,
            IntPtr upB, int upBDim,
            IntPtr downW, int downNe0, int downNe1, long downBytes,
            IntPtr downB, int downBDim);

        // 中文：融合算子，输出投影 + 残差 + FFN 归一化 + SwiGLU 前馈（量化权重）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_FusedOutProjFFNQuantF32(
            GgmlTensorView2D residual, GgmlTensorView2D input,
            IntPtr outProjData, int outProjType, long outNe0, long outNe1, long outRawBytes,
            IntPtr ffnNormData, int ffnNormCount, float eps,
            IntPtr guData, int guType, long guNe0, long guNe1, long guRawBytes,
            IntPtr dnData, int dnType, long dnNe0, long dnNe1, long dnRawBytes,
            int halfDim);

        // 中文：视觉编码器的融合自注意力（LayerNorm + QKV + RoPE + 注意力 + 输出投影）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_FusedVisionAttentionF32(
            GgmlTensorView2D hidden,
            IntPtr lnW, IntPtr lnB, int lnDim, float eps,
            IntPtr qkvW, int qkvNe0, int qkvNe1, long qkvBytes,
            IntPtr qkvB, int qkvBDim,
            IntPtr outW, int outNe0, int outNe1, long outBytes,
            IntPtr outB, int outBDim,
            IntPtr cosTable, IntPtr sinTable,
            int numPatches, int numHeads, int headDim, int halfDim,
            float attnScale);

        // 中文：按索引从量化矩阵中取行（get_rows），输出 F32（如词嵌入查表）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_GetRowsQuantF32(
            GgmlTensorView2D result,
            IntPtr srcData,
            int srcGgmlType,
            long srcNe0,
            long srcNe1,
            long srcRawBytes,
            GgmlContiguousTensor indices);

        // 中文：MoE 多专家前向（up + down 投影并按路由权重加权聚合，量化权重）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_MoEExpertsForwardF32(
            GgmlTensorView2D result,
            GgmlTensorView2D input,
            int numExperts,
            IntPtr[] upDataPtrs,
            IntPtr[] downDataPtrs,
            int upGgmlType,
            long upNe0,
            long upNe1,
            long upRawBytesEach,
            int downGgmlType,
            long downNe0,
            long downNe1,
            long downRawBytesEach,
            float[] routeWeights);

        // 中文：MoE 多专家 SwiGLU 前向（gate + up + SwiGLU + down 并按路由权重聚合）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_MoEExpertsSwiGLUForwardF32(
            GgmlTensorView2D result,
            GgmlTensorView2D input,
            int numExperts,
            IntPtr[] gateDataPtrs,
            IntPtr[] upDataPtrs,
            IntPtr[] downDataPtrs,
            int gateGgmlType,
            long gateNe0,
            long gateNe1,
            long gateRawBytesEach,
            int upGgmlType,
            long upNe0,
            long upNe1,
            long upRawBytesEach,
            int downGgmlType,
            long downNe0,
            long downNe1,
            long downRawBytesEach,
            float[] routeWeights);

        // 中文：MoE 多专家 SwiGLU 前向并加残差，可选共享专家分支。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_MoEExpertsSwiGLUResidualF32(
            GgmlTensorView2D residual,
            GgmlTensorView2D input,
            int numExperts,
            IntPtr[] gateDataPtrs,
            IntPtr[] upDataPtrs,
            IntPtr[] downDataPtrs,
            int gateGgmlType,
            long gateNe0,
            long gateNe1,
            long gateRawBytesEach,
            int upGgmlType,
            long upNe0,
            long upNe1,
            long upRawBytesEach,
            int downGgmlType,
            long downNe0,
            long downNe1,
            long downRawBytesEach,
            float[] routeWeights,
            int useShared,
            IntPtr sharedGateData,
            IntPtr sharedUpData,
            IntPtr sharedDownData,
            int sharedGateGgmlType,
            long sharedGateNe0,
            long sharedGateNe1,
            long sharedGateRawBytes,
            int sharedUpGgmlType,
            long sharedUpNe0,
            long sharedUpNe1,
            long sharedUpRawBytes,
            int sharedDownGgmlType,
            long sharedDownNe0,
            long sharedDownNe1,
            long sharedDownRawBytes,
            float sharedScalar);

        // 中文：批量量化矩阵乘，按权重偏移数组分段对多组权重做 addmm。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_AddmmQuantBatchF32(
            GgmlTensorView2D result,
            GgmlTensorView2D m1,
            IntPtr m2Data,
            int m2GgmlType,
            long m2Ne0,
            long m2RawBytes,
            int batchCount,
            long[] weightOffsets,
            long[] weightNe1Arr);

        // 中文：3D 张量的批量 addmm，result = beta*src + alpha*(m1·m2)（按批维并行）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_AddmmBatchF32(
            GgmlTensorView3D result,
            GgmlTensorView3D src,
            GgmlTensorView3D m1,
            GgmlTensorView3D m2,
            float beta,
            float alpha);

        // 中文：按专家 id 选择权重的索引矩阵乘 mul_mat_id（MoE 专家分发）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_MulMatIdF32(
            GgmlTensorView3D result,
            GgmlTensorView3D expertWeights,
            GgmlTensorView3D input,
            GgmlContiguousTensor ids,
            int idsRows,
            int idsCols);

        // 中文：按专家 id 选择偏置并逐元素相加 add_id（MoE 专家偏置）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_AddIdF32(
            GgmlTensorView3D result,
            GgmlTensorView3D src,
            GgmlTensorView2D bias,
            GgmlContiguousTensor ids,
            int idsRows,
            int idsCols);

        // 中文：沿最后一维做归约（求和/均值），op 指定归约类型。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_ReduceLastDimF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D src);

        // 中文：沿最后一维做索引归约（argmin/argmax），返回极值的下标。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_IndexReductionF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D src);

        // 中文：对 4D 张量沿最后一维做 softmax 归一化。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_SoftmaxF32(
            GgmlTensorView4D result,
            GgmlTensorView4D src);

        // In-place softmax with causal+SWA mask and optional attention sinks.
        // Replaces the GptOss CPU softmax-with-sinks loop. See native side:
        // attention_softmax_with_sinks_f32_impl in ggml_ops_norm_attn.cpp.
        // 中文：原地注意力 softmax，支持因果+滑动窗口掩码及可选 attention sinks。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_AttentionSoftmaxWithSinksF32(
            GgmlTensorView3D scores,
            IntPtr sinksData,         // float* [num_heads], or IntPtr.Zero for no sinks
            int numHeads,
            int seqLen,
            int kvLen,
            int maskStartPos,
            int slidingWindow,
            float scale);

        // Fused MoE FFN prefill (mul_mat_id-based).
        // Collapses an entire layer's MoE forward (gate + up + SwiGLU + down +
        // expert weighting + aggregation) into one GGML graph dispatch.
        // See native side: TSGgml_MoEFFNPrefillSwiGLUQuantF32 in ggml_ops_moe.cpp.
        // 中文：MoE FFN 预填充（基于 mul_mat_id 将整层 MoE 前向融合为单次计算图调度）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_MoEFFNPrefillSwiGLUQuantF32(
            IntPtr hiddenIn,
            IntPtr hiddenOut,
            int seqLen,
            int hiddenDim,
            int nFf,
            int numExperts,
            int nUsed,
            IntPtr selectedExperts,    // int32* [seqLen, nUsed]
            IntPtr routingWeights,     // float* [seqLen, nUsed]
            IntPtr gateData, int gateType, long gateNe0, long gateNe1, long gateTotalBytes,
            IntPtr upData,   int upType,   long upNe0,   long upNe1,   long upTotalBytes,
            IntPtr downData, int downType, long downNe0, long downNe1, long downTotalBytes,
            IntPtr gateBias,           // optional float* [biasDim, numExperts] (biasDim = nFf or 2*nFf for fused gate_up); IntPtr.Zero to skip
            IntPtr upBias,             // optional, only valid when up_data != null
            IntPtr downBias,           // optional float* [hiddenDim, numExperts]
            int activationType,        // 0 = SwiGLU split, 1 = SwiGLU OAI, 2 = GEGLU split, 3 = ReLU-squared
            float oaiAlpha,
            float oaiLimit);

        // Gemma 4 MoE GEGLU + post_norm + residual add fused kernel.
        // Computes residual_in_out += rms_norm(moe_ffn(hidden_in), eps) * post_norm_w
        // in a single GGML graph dispatch. Mirrors the existing
        // TSGgml_MoEFFNPrefillSwiGLUQuantF32 ABI but adds the residual buffer,
        // the post_ffw_norm_2 weight, and an RMSNorm epsilon.
        // See native side: TSGgml_Gemma4MoEGEGLUResidualF32 in ggml_ops_moe.cpp.
        // 中文：Gemma 4 的 MoE GEGLU 前向 + post_norm + 残差相加，融合为单次计算图调度。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_Gemma4MoEGEGLUResidualF32(
            IntPtr hiddenIn,
            IntPtr residualInOut,      // float* [seqLen, hiddenDim] - dense FFN result; kernel adds normed MoE output to it in place
            IntPtr postNormW,          // float* [hiddenDim] - post_ffw_norm_2.weight
            float postNormEps,
            int seqLen,
            int hiddenDim,
            int nFf,
            int numExperts,
            int nUsed,
            IntPtr selectedExperts,
            IntPtr routingWeights,
            IntPtr gateData, int gateType, long gateNe0, long gateNe1, long gateTotalBytes,
            IntPtr upData,   int upType,   long upNe0,   long upNe1,   long upTotalBytes,
            IntPtr downData, int downType, long downNe0, long downNe1, long downTotalBytes,
            IntPtr gateBias,
            IntPtr upBias,
            IntPtr downBias,
            int activationType,
            float oaiAlpha,
            float oaiLimit);

        // 中文：缩放点积注意力（SDPA），可选掩码与缩放因子。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_ScaledDotProductAttentionF32(
            GgmlTensorView4D result,
            GgmlTensorView4D query,
            GgmlTensorView4D key,
            GgmlTensorView4D value,
            GgmlTensorView4D mask,
            int hasMask,
            float scale);

        // 中文：softmax 的反向梯度计算，可选累加到已有梯度。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_SoftmaxGradF32(
            GgmlTensorView4D result,
            GgmlTensorView4D adj,
            GgmlTensorView4D val,
            int addGrad);

        // 中文：交叉熵损失前向，支持标签平滑，输出标量损失值。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_CrossEntropyLossF32(
            out float lossValue,
            GgmlTensorView4D probs,
            GgmlContiguousTensor targetIndices,
            float smooth,
            float labelSmooth);

        // 中文：交叉熵损失的反向梯度计算，可选累加到已有梯度。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_CrossEntropyLossBackwardF32(
            GgmlTensorView4D grad,
            GgmlTensorView4D probs,
            GgmlContiguousTensor targetIndices,
            float lossGradient,
            float smooth,
            float labelSmooth,
            int addGrad);

        // 中文：Adam 优化器原地更新权重，含一阶/二阶动量、梯度裁剪与权重衰减。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_AdamF32(
            GgmlContiguousTensor weight,
            GgmlContiguousTensor gradient,
            GgmlContiguousTensor v,
            GgmlContiguousTensor m,
            float gradNormFactor,
            float stepSize,
            float clipValue,
            float regc,
            float decayRateV,
            float decayRateM,
            int iter,
            float eps);

        // 中文：单步解码的完整 Transformer 层前向（注意力 + KV cache + FFN，量化权重）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_TransformerLayerDecode(
            IntPtr hiddenData, int hiddenSize,
            IntPtr attnNormData,
            IntPtr qkvData, int qkvType, long qkvNe0, long qkvNe1, long qkvBytes,
            IntPtr qNormData, IntPtr kNormData, int headDim,
            IntPtr oData, int oType, long oNe0, long oNe1, long oBytes,
            IntPtr ffnNormData,
            IntPtr guData, int guType, long guNe0, long guNe1, long guBytes,
            IntPtr downData, int downType, long downNe0, long downNe1, long downBytes,
            IntPtr kCacheData, IntPtr vCacheData,
            int numHeads, int numKvHeads,
            int maxSeqLen, int position,
            float eps, float ropeBase, float ropeFreqScale,
            int intermediateSize, int ropeMode,
            int kvCacheType);

        // 中文：Gemma 4 单层预填充前向（注意力 + RoPE + SWA + PLE + FFN，写入 KV cache）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_Gemma4LayerPrefill(
            IntPtr hiddenData, int hiddenSize, int seqLen,
            IntPtr attnNormW,
            IntPtr qkvW, int qkvType, long qkvNe0, long qkvNe1, long qkvBytes,
            IntPtr qNormW, IntPtr kNormW,
            IntPtr oW, int oType, long oNe0, long oNe1, long oBytes,
            IntPtr postAttnNormW,
            IntPtr ffnNormW,
            IntPtr guW, int guType, long guNe0, long guNe1, long guBytes,
            IntPtr downW, int downType, long downNe0, long downNe1, long downBytes,
            IntPtr postFfnNormW,
            IntPtr kCacheData, IntPtr vCacheData,
            int numHeads, int kvHeads, int headDim,
            int cacheSize, int startPos,
            int isLocal, int slidingWindow,
            float ropeBase, int ropeDims,
            IntPtr ropeFreqFactors, int freqFactorsLen,
            float layerScalar, float eps,
            IntPtr swaPrevK, IntPtr swaPrevV, int prevWindowLen,
            IntPtr pleInputData, int pleDim,
            IntPtr pleGateW, int pleGateType, long pleGateNe0, long pleGateNe1, long pleGateBytes,
            IntPtr pleProjW, int pleProjType, long pleProjNe0, long pleProjNe1, long pleProjBytes,
            IntPtr plePostNormW,
            IntPtr freshKOut, IntPtr freshVOut,
            int isShared,
            IntPtr donorK, IntPtr donorV, int donorKvLen,
            int kvCacheType);

        // 中文：Gemma4LayerPrefill 的托管封装，调用原生层并检查返回结果。
        public static void Gemma4LayerPrefill(
            IntPtr hiddenData, int hiddenSize, int seqLen,
            IntPtr attnNormW,
            IntPtr qkvW, int qkvType, long qkvNe0, long qkvNe1, long qkvBytes,
            IntPtr qNormW, IntPtr kNormW,
            IntPtr oW, int oType, long oNe0, long oNe1, long oBytes,
            IntPtr postAttnNormW,
            IntPtr ffnNormW,
            IntPtr guW, int guType, long guNe0, long guNe1, long guBytes,
            IntPtr downW, int downType, long downNe0, long downNe1, long downBytes,
            IntPtr postFfnNormW,
            IntPtr kCacheData, IntPtr vCacheData,
            int numHeads, int kvHeads, int headDim,
            int cacheSize, int startPos,
            int isLocal, int slidingWindow,
            float ropeBase, int ropeDims,
            IntPtr ropeFreqFactors, int freqFactorsLen,
            float layerScalar, float eps,
            IntPtr swaPrevK, IntPtr swaPrevV, int prevWindowLen,
            IntPtr pleInputData, int pleDim,
            IntPtr pleGateW, int pleGateType, long pleGateNe0, long pleGateNe1, long pleGateBytes,
            IntPtr pleProjW, int pleProjType, long pleProjNe0, long pleProjNe1, long pleProjBytes,
            IntPtr plePostNormW,
            IntPtr freshKOut, IntPtr freshVOut,
            int isShared,
            IntPtr donorK, IntPtr donorV, int donorKvLen,
            int kvCacheType = 0)
        {
            CheckResult(TSGgml_Gemma4LayerPrefill(
                hiddenData, hiddenSize, seqLen,
                attnNormW,
                qkvW, qkvType, qkvNe0, qkvNe1, qkvBytes,
                qNormW, kNormW,
                oW, oType, oNe0, oNe1, oBytes,
                postAttnNormW,
                ffnNormW,
                guW, guType, guNe0, guNe1, guBytes,
                downW, downType, downNe0, downNe1, downBytes,
                postFfnNormW,
                kCacheData, vCacheData,
                numHeads, kvHeads, headDim,
                cacheSize, startPos,
                isLocal, slidingWindow,
                ropeBase, ropeDims,
                ropeFreqFactors, freqFactorsLen,
                layerScalar, eps,
                swaPrevK, swaPrevV, prevWindowLen,
                pleInputData, pleDim,
                pleGateW, pleGateType, pleGateNe0, pleGateNe1, pleGateBytes,
                pleProjW, pleProjType, pleProjNe0, pleProjNe1, pleProjBytes,
                plePostNormW,
                freshKOut, freshVOut,
                isShared,
                donorK, donorV, donorKvLen,
                kvCacheType), "gemma4_layer_prefill");
        }

        // 中文：预填充阶段的融合注意力（含因果/滑动窗口掩码与缩放）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_FusedPrefillAttentionF32(
            IntPtr qData, IntPtr kData, IntPtr vData, IntPtr outData,
            int numHeads, int numKvHeads, int headDim,
            int seqLen, int kvLen,
            int maskStartPos, int slidingWindow,
            float scale, int inputFormat);

        // 中文：单步解码的 Flash Attention，读取并更新 KV cache。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_FlashAttnDecodeF32(
            IntPtr qData, IntPtr kData, IntPtr vData,
            IntPtr kCacheData, IntPtr vCacheData,
            IntPtr outData,
            int numHeads, int numKvHeads, int headDim,
            int maxSeqLen, int position,
            float scale, int kvCacheType);

        // 中文：分页 KV cache 的注意力前向（PagedAttention，按块表索引访问 KV）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_PagedAttentionForward(
            IntPtr qData,
            IntPtr pagedKData,
            IntPtr pagedVData,
            IntPtr outData,
            IntPtr queryStartLoc,
            IntPtr seqLens,
            IntPtr positions,
            IntPtr blockTableFlat,
            IntPtr blockTableOffsets,
            int numSeqs,
            int numTokens,
            int numHeads,
            int numKvHeads,
            int headDim,
            int blockSize,
            int slidingWindow,
            float scale);

        // 中文：带 attention sinks 的分页注意力前向。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_PagedAttentionForwardWithSinks(
            IntPtr qData,
            IntPtr pagedKData,
            IntPtr pagedVData,
            IntPtr outData,
            IntPtr queryStartLoc,
            IntPtr seqLens,
            IntPtr positions,
            IntPtr blockTableFlat,
            IntPtr blockTableOffsets,
            int numSeqs,
            int numTokens,
            int numHeads,
            int numKvHeads,
            int headDim,
            int blockSize,
            int slidingWindow,
            float scale,
            IntPtr sinksData);          // [numHeads] F32 or IntPtr.Zero

        // GPU-resident variant: qData and outData point to existing backend
        // (Tensor storage) buffers, so the kernel can zero-copy bind them
        // instead of round-tripping through host arrays + ggml_backend_synchronize.
        // Eliminates the per-layer queue drain that GetElementsAsFloat would
        // otherwise force.
        // 中文：GPU 常驻版分页注意力前向，q/out 直接绑定后端缓冲区以实现零拷贝。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_PagedAttentionForwardDevice(
            IntPtr qData,
            IntPtr pagedKData,
            IntPtr pagedVData,
            IntPtr outData,
            IntPtr queryStartLoc,
            IntPtr seqLens,
            IntPtr positions,
            IntPtr blockTableFlat,
            IntPtr blockTableOffsets,
            int numSeqs,
            int numTokens,
            int numHeads,
            int numKvHeads,
            int headDim,
            int blockSize,
            int slidingWindow,
            float scale);

        // 中文：GPU 常驻且带 attention sinks 的分页注意力前向。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_PagedAttentionForwardDeviceWithSinks(
            IntPtr qData,
            IntPtr pagedKData,
            IntPtr pagedVData,
            IntPtr outData,
            IntPtr queryStartLoc,
            IntPtr seqLens,
            IntPtr positions,
            IntPtr blockTableFlat,
            IntPtr blockTableOffsets,
            int numSeqs,
            int numTokens,
            int numHeads,
            int numKvHeads,
            int headDim,
            int blockSize,
            int slidingWindow,
            float scale,
            IntPtr sinksData);

        // 中文：Qwen3.5 单步解码的注意力层前向（QK 归一化 + RoPE + KV cache）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_Qwen35AttentionLayerDecode(
            IntPtr residualData, int hiddenSize,
            IntPtr attnNormData,
            IntPtr qkvData, int qkvType, long qkvNe0, long qkvNe1, long qkvBytes,
            IntPtr qNormData, IntPtr kNormData, int headDim,
            IntPtr oData, int oType, long oNe0, long oNe1, long oBytes,
            IntPtr kCacheData, IntPtr vCacheData,
            int numHeads, int numKvHeads,
            int maxSeqLen, int position,
            float eps, float ropeBase, float ropeFreqScale,
            int ropeMode, int kvCacheType);

        // 中文：GPT-OSS 注意力层预填充前向（含 SWA/sinks、可融合或分离的 QKV 与 RoPE）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_GptOssAttentionLayerPrefill(
            IntPtr hiddenData, int hiddenSize, int seqLen,
            IntPtr attnNormW,
            IntPtr qkvW, int qkvType, long qkvNe0, long qkvNe1, long qkvBytes,
            IntPtr qkvB,
            int isQkvFused,
            IntPtr kW, int kType, long kNe0, long kNe1, long kBytes,
            IntPtr kB,
            IntPtr vW, int vType, long vNe0, long vNe1, long vBytes,
            IntPtr vB,
            IntPtr oW, int oType, long oNe0, long oNe1, long oBytes,
            IntPtr oB,
            IntPtr kCacheData, IntPtr vCacheData,
            int numHeads, int kvHeads, int headDim,
            int cacheSize, int startPos,
            int isSwa, int slidingWindow,
            IntPtr sinksData,
            float ropeBase, float ropeFreqScale, int ropeDims,
            int originalContextLength,
            int kvCacheType,
            float eps);

        // 中文：Qwen3.5 注意力层预填充前向（QK 归一化 + RoPE + 写入 KV cache）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_Qwen35AttentionLayerPrefill(
            IntPtr hiddenData, int hiddenSize, int seqLen,
            IntPtr attnNormW,
            IntPtr qkvW, int qkvType, long qkvNe0, long qkvNe1, long qkvBytes,
            IntPtr qNormW, IntPtr kNormW,
            IntPtr oW, int oType, long oNe0, long oNe1, long oBytes,
            IntPtr kCacheData, IntPtr vCacheData,
            int numHeads, int kvHeads, int headDim,
            int cacheSize, int startPos,
            float ropeBase, float ropeFreqScale, int ropeDims,
            int ropeMode,
            int kvCacheType,
            float eps);

        // 中文：Qwen35AttentionLayerPrefill 的托管封装，调用原生层并检查返回结果。
        public static void Qwen35AttentionLayerPrefill(
            IntPtr hiddenData, int hiddenSize, int seqLen,
            IntPtr attnNormW,
            IntPtr qkvW, int qkvType, long qkvNe0, long qkvNe1, long qkvBytes,
            IntPtr qNormW, IntPtr kNormW,
            IntPtr oW, int oType, long oNe0, long oNe1, long oBytes,
            IntPtr kCacheData, IntPtr vCacheData,
            int numHeads, int kvHeads, int headDim,
            int cacheSize, int startPos,
            float ropeBase, float ropeFreqScale, int ropeDims,
            int ropeMode,
            int kvCacheType,
            float eps)
        {
            CheckResult(TSGgml_Qwen35AttentionLayerPrefill(
                hiddenData, hiddenSize, seqLen,
                attnNormW,
                qkvW, qkvType, qkvNe0, qkvNe1, qkvBytes,
                qNormW, kNormW,
                oW, oType, oNe0, oNe1, oBytes,
                kCacheData, vCacheData,
                numHeads, kvHeads, headDim,
                cacheSize, startPos,
                ropeBase, ropeFreqScale, ropeDims,
                ropeMode, kvCacheType, eps), "qwen35_attention_layer_prefill");
        }

        // 中文：GptOssAttentionLayerPrefill 的托管封装，调用原生层并检查返回结果。
        public static void GptOssAttentionLayerPrefill(
            IntPtr hiddenData, int hiddenSize, int seqLen,
            IntPtr attnNormW,
            IntPtr qkvW, int qkvType, long qkvNe0, long qkvNe1, long qkvBytes,
            IntPtr qkvB,
            int isQkvFused,
            IntPtr kW, int kType, long kNe0, long kNe1, long kBytes,
            IntPtr kB,
            IntPtr vW, int vType, long vNe0, long vNe1, long vBytes,
            IntPtr vB,
            IntPtr oW, int oType, long oNe0, long oNe1, long oBytes,
            IntPtr oB,
            IntPtr kCacheData, IntPtr vCacheData,
            int numHeads, int kvHeads, int headDim,
            int cacheSize, int startPos,
            int isSwa, int slidingWindow,
            IntPtr sinksData,
            float ropeBase, float ropeFreqScale, int ropeDims,
            int originalContextLength,
            int kvCacheType,
            float eps)
        {
            CheckResult(TSGgml_GptOssAttentionLayerPrefill(
                hiddenData, hiddenSize, seqLen,
                attnNormW,
                qkvW, qkvType, qkvNe0, qkvNe1, qkvBytes,
                qkvB,
                isQkvFused,
                kW, kType, kNe0, kNe1, kBytes,
                kB,
                vW, vType, vNe0, vNe1, vBytes,
                vB,
                oW, oType, oNe0, oNe1, oBytes,
                oB,
                kCacheData, vCacheData,
                numHeads, kvHeads, headDim,
                cacheSize, startPos,
                isSwa, slidingWindow,
                sinksData,
                ropeBase, ropeFreqScale, ropeDims,
                originalContextLength,
                kvCacheType,
                eps), "gpt_oss_attention_layer_prefill");
        }

        // 中文：整模型单步解码，将所有 Transformer 层作为一张计算图一次性前向。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_TransformerModelDecode(
            IntPtr hiddenData, int hiddenSize, int numLayers,
            IntPtr[] attnNormArr, IntPtr[] qkvArr, IntPtr[] qNormArr, IntPtr[] kNormArr,
            IntPtr[] oArr, IntPtr[] ffnNormArr, IntPtr[] guArr, IntPtr[] downArr,
            IntPtr[] kCacheArr, IntPtr[] vCacheArr,
            int qkvType, long qkvNe0, long qkvNe1, long qkvBytes,
            int oType, long oNe0, long oNe1, long oBytes,
            int guType, long guNe0, long guNe1, long guBytes,
            int downType, long downNe0, long downNe1, long downBytes,
            int headDim, int numHeads, int numKvHeads,
            int maxSeqLen, int position,
            float eps, float ropeBase, float ropeFreqScale,
            int intermediateSize, int ropeMode,
            int kvCacheType);

        // 中文：Gemma 4 整模型单步解码，按层数组传入权重并融合为一张计算图前向。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_Gemma4ModelDecode(
            IntPtr hiddenData, int hiddenSize, int numLayers,
            IntPtr[] attnNormArr, IntPtr[] qkvArr, IntPtr[] qNormArr, IntPtr[] kNormArr,
            IntPtr[] oArr, IntPtr[] postAttnNormArr,
            IntPtr[] ffnNormArr, IntPtr[] guArr, IntPtr[] downArr, IntPtr[] postFfnNormArr,
            IntPtr[] kCacheArr, IntPtr[] vCacheArr,
            int[] headDimArr, int[] kvHeadsArr, int[] cacheSizeArr, int[] isLocalArr,
            int[] kvSourceArr,
            float[] ropeBaseArr, float[] layerScalarArr,
            int[] qkvTypeArr, long[] qkvNe0Arr, long[] qkvNe1Arr, long[] qkvBytesArr,
            int[] oTypeArr, long[] oNe0Arr, long[] oNe1Arr, long[] oBytesArr,
            int[] guTypeArr, long[] guNe0Arr, long[] guNe1Arr, long[] guBytesArr,
            int[] downTypeArr, long[] downNe0Arr, long[] downNe1Arr, long[] downBytesArr,
            int numHeads, int position,
            float eps, int slidingWindow,
            IntPtr ropeFreqFactors, int ropeFreqFactorsLen,
            int[] ropeNDimsArr,
            IntPtr pleData, int pleDim,
            IntPtr[] pleGateArr, int[] pleGateTypeArr, long[] pleGateNe0Arr, long[] pleGateNe1Arr, long[] pleGateBytesArr,
            IntPtr[] pleProjArr, int[] pleProjTypeArr, long[] pleProjNe0Arr, long[] pleProjNe1Arr, long[] pleProjBytesArr,
            IntPtr[] plePostNormArr,
            int kvCacheType,
            IntPtr[] kArr, int[] kTypeArr, long[] kNe0Arr, long[] kNe1Arr, long[] kBytesArr,
            IntPtr[] vArr, int[] vTypeArr, long[] vNe0Arr, long[] vNe1Arr, long[] vBytesArr);

        // 中文：Gemma 4 MoE 单层单步解码，参数通过描述符结构体一次性传入。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_Gemma4MoELayerDecode(in Gemma4MoELayerDecodeArgs desc);

        // 中文：Gemma4MoELayerDecode 的托管封装，调用原生层并检查返回结果。
        public static void Gemma4MoELayerDecode(in Gemma4MoELayerDecodeArgs desc)
        {
            CheckResult(TSGgml_Gemma4MoELayerDecode(in desc), nameof(TSGgml_Gemma4MoELayerDecode));
        }

        // Model-wide MoE decode: the whole transformer as one graph/token.
        // `layers` is one Gemma4MoELayerDecodeArgs per layer (blittable, marshalled
        // as a contiguous TSGgmlGemma4MoELayerDesc array). hidden/position come from
        // the explicit params; the per-element Hidden/Position fields are ignored.
        // 中文：Gemma 4 MoE 整模型单步解码，按层描述符数组将整个 Transformer 作为一张计算图前向。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_Gemma4MoEModelDecode(
            [In] Gemma4MoELayerDecodeArgs[] layers, int numLayers,
            IntPtr hidden, int hiddenSize, int position);

        // 中文：Gemma4MoEModelDecode 的托管封装，调用原生层并检查返回结果。
        public static void Gemma4MoEModelDecode(Gemma4MoELayerDecodeArgs[] layers, int numLayers, IntPtr hidden, int hiddenSize, int position)
        {
            CheckResult(TSGgml_Gemma4MoEModelDecode(layers, numLayers, hidden, hiddenSize, position), nameof(TSGgml_Gemma4MoEModelDecode));
        }

        // 中文：门控 DeltaNet 线性注意力的分块（chunked）前向，更新循环状态。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_GatedDeltaNetChunkedF32(
            GgmlTensorView3D q,
            GgmlTensorView3D k,
            GgmlTensorView3D v,
            GgmlTensorView3D z,
            GgmlTensorView2D alpha,
            GgmlTensorView2D beta,
            GgmlTensorView3D state,
            GgmlTensorView3D gatedOut,
            IntPtr dtBiasData,
            IntPtr aLogData,
            IntPtr ssmNormWData,
            int chunkSize,
            float eps);

        // Mirrors NemoMamba2BatchedSeqDesc in ggml_ops_mamba2.cpp; same 32-byte
        // POD layout on 64-bit (two ints, two padding ints, two pointers).
        // 中文：Nemotron Mamba2 批量单步推理（含因果卷积 + SSM 扫描 + 归一化）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_NemotronMamba2BatchedStepF32(
            int numSeqs,
            [In, Out] NemoMamba2BatchedSeqDesc[] seqs,
            int numTokens,
            IntPtr packedBatched,
            int dInProjTotal,
            int dInner,
            int dState,
            int nHead,
            int headDim,
            int nGroup,
            int dConv,
            IntPtr convWt,
            IntPtr convBias,
            IntPtr dtBias,
            IntPtr aLog,
            IntPtr dData,
            IntPtr ssmNormW,
            float eps,
            IntPtr outBatched);

        // 中文：门控 DeltaNet 批量单步推理，按序列描述符更新各自的循环状态。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_GatedDeltaNetBatchedStepF32(
            int numSeqs,
            [In, Out] GdnBatchedSeqDesc[] seqs,
            int numTokens,
            IntPtr packedBatched,
            int packedDim,
            int qkvDim,
            int qkDim,
            int vDim,
            int zDim,
            int numKHeads,
            int numVHeads,
            int headKDim,
            int headVDim,
            int convKernel,
            int ssmDInner,
            IntPtr convWt,
            IntPtr dtBias,
            IntPtr aLog,
            IntPtr ssmNormW,
            float eps,
            IntPtr gatedOut);

        // 中文：Nemotron Mamba2 预填充前向，初始化卷积/SSM 状态并输出隐藏态。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_NemotronMamba2PrefillF32(
            GgmlTensorView2D projected,
            GgmlTensorView2D hiddenOut,
            IntPtr convStateData,
            int convStateElements,
            IntPtr ssmStateData,
            int ssmStateElements,
            IntPtr convWeightData,
            IntPtr convBiasData,
            IntPtr dtBiasData,
            IntPtr aData,
            IntPtr dData,
            IntPtr ssmNormData,
            int dInner,
            int dState,
            int nHead,
            int headDim,
            int nGroup,
            int dConv,
            float eps);

        // 中文：Nemotron Mamba2 单步解码前向，按状态键复用并可选初始化/回传状态。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_NemotronMamba2DecodeF32(
            ulong stateKey,
            GgmlTensorView2D projected,
            GgmlTensorView2D hiddenOut,
            IntPtr convStateData,
            int convStateElements,
            IntPtr ssmStateData,
            int ssmStateElements,
            int initializeState,
            int downloadState,
            IntPtr convWeightData,
            IntPtr convBiasData,
            IntPtr dtBiasData,
            IntPtr aData,
            IntPtr dData,
            IntPtr ssmNormData,
            int dInner,
            int dState,
            int nHead,
            int headDim,
            int nGroup,
            int dConv,
            float eps);

        // 中文：清除指定模型键缓存的 Mamba2 解码状态。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_NemotronMamba2DecodeClear(ulong modelKey);

        // 中文：分配对齐的原生内存块并返回指针。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern IntPtr TSGgml_AlignedAlloc(UIntPtr size);

        // 中文：释放由 TSGgml_AlignedAlloc 分配的对齐内存。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_AlignedFree(IntPtr ptr);

        // 中文：清空主机端缓冲区缓存。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_ClearHostBufferCache();

        // 中文：关闭并释放 GGML 后端的全部资源。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_Shutdown();

        // 中文：使指定主机缓冲区缓存失效（强制下次重新同步）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_InvalidateHostBuffer(IntPtr ptr);

        // 中文：将指定主机缓冲区与后端设备内存同步。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_SyncHostBuffer(IntPtr ptr, long byteCount);

        // Async dispatch (deferred ggml_backend_synchronize). When enabled, per-op
        // kernels return without waiting on the Metal command buffer; subsequent ops
        // chain through the Metal command queue, and host-side reads must call
        // TSGgml_HostReadBarrier first to drain pending GPU work. See
        // GgmlStorage.EnsureHostReadable for the C# entry point that triggers this.
        // 中文：开启/关闭异步计算（延迟后端同步）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_SetAsyncCompute(int enabled);

        // 中文：查询当前异步计算是否启用。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_GetAsyncCompute();

        // 中文：主机读屏障，排空挂起的 GPU 计算以确保主机端读取数据有效。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_HostReadBarrier();

        // 中文：将量化权重预加载到设备缓存，后续算子可按 cacheKey 复用。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_PreloadQuantizedWeight(IntPtr cacheKey, IntPtr hostData, int ggmlType, long ne0, long ne1, long rawBytes);

        // 中文：注册一个可卸载（可在显存不足时换出）的缓冲区键。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_RegisterOffloadable(IntPtr key);

        // 中文：设置可卸载缓冲区占用的显存预算上限（字节）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_SetOffloadableBudget(long bytes);

        // 中文：清除所有可卸载缓冲区的卸载状态记录。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_ClearOffloadableState();

        // 中文：返回指定量化类型下一行（ne 个元素）所占的字节数。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern UIntPtr TSGgml_RowSize(int ggmlType, long ne);

        // 中文：将量化数据反量化为 F32 写入目标缓冲区。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_DequantizeToF32(int ggmlType, IntPtr src, long numElements, IntPtr dst);

        // 中文：F32 张量逐元素拷贝（支持跨步的视图复制）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_CopyF32(
            GgmlTensorView4D result,
            GgmlTensorView4D src);

        // 中文：逐元素一元运算（如 Neg/Exp/Relu/SiLU/GELU 等，由 op 指定）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_UnaryF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D src);

        // 中文：两张量逐元素二元运算（加/减/乘/除，支持广播）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_BinaryTensorF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D lhs,
            GgmlTensorView4D rhs);

        // 中文：融合的激活后逐元素相乘（如 SiLU(a)*b 等门控运算）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_FusedActMulF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D a,
            GgmlTensorView4D b);

        // 中文：对 gate_up 拼接张量按 halfDim 切分后做激活并相乘（SwiGLU/GEGLU 门控）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_FusedActMulSplitF32(
            int op,
            GgmlTensorView2D result,
            GgmlTensorView2D gateUp,
            int halfDim);

        // 中文：张量与标量的逐元素运算（加/减/乘/除及其反向变体）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_BinaryScalarF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D src,
            float scalar);

        // 中文：激活函数的反向梯度计算（Relu/Sigmoid/Tanh/SiLU），可累加。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_ActivationGradF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D src,
            GgmlTensorView4D grad,
            GgmlTensorView4D accumulation,
            int hasAccumulation);

        // 中文：归一化前向（LayerNorm 或 RMSNorm），带缩放 gamma 与可选偏置 beta。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_NormF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D src,
            GgmlTensorView4D gamma,
            GgmlTensorView4D beta,
            int hasBeta,
            float eps);

        // 中文：归一化的反向梯度计算，输出输入梯度及 gamma/beta 梯度。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_NormGradF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D gradGamma,
            GgmlTensorView4D gradBeta,
            GgmlTensorView4D adj,
            GgmlTensorView4D x,
            GgmlTensorView4D gamma,
            int hasGradBeta,
            float eps);

        // 中文：按索引选取行（index_select），可累加到结果。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_IndexSelectF32(
            GgmlTensorView2D result,
            GgmlTensorView2D src,
            GgmlContiguousTensor indices,
            int addToResult);

        // 中文：index_select 的反向梯度，将上游梯度按索引散射回输入梯度。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_IndexSelectGradF32(
            GgmlTensorView2D grad,
            GgmlTensorView2D adj,
            GgmlContiguousTensor indices);

        // 中文：基础旋转位置编码 RoPE，可累加并可反转位置（用于反向）。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_RoPEF32(
            GgmlTensorView4D result,
            GgmlTensorView4D src,
            int seqLen,
            int rowOffset,
            int addToResult,
            int invertPositions);

        // 中文：扩展版 RoPE，支持显式位置、NEOX/GLM 模式与 YaRN 频率缩放参数。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_RoPEExF32(
            GgmlTensorView4D result,
            GgmlTensorView4D src,
            GgmlContiguousTensor positions,
            int ropeDim,
            int mode,
            int originalContextLength,
            float freqBase,
            float freqScale,
            float extFactor,
            float attnFactor,
            float betaFast,
            float betaSlow,
            int addToResult,
            int invertPositions);

        // 中文：多模态 RoPE（mRoPE），按 sect 段划分维度做多维位置编码。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_RoPEMRoPEF32(
            GgmlTensorView4D result,
            GgmlTensorView4D src,
            GgmlContiguousTensor positions,
            int ropeDim,
            int mode,
            int sect0, int sect1, int sect2, int sect3,
            int originalContextLength,
            float freqBase,
            float freqScale,
            float extFactor,
            float attnFactor,
            float betaFast,
            float betaSlow);

        // 中文：带每维频率因子（freq_factors）的扩展 RoPE，用于长上下文频率缩放。
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_RoPEExFreqFactorsF32(
            GgmlTensorView4D result,
            GgmlTensorView4D src,
            GgmlContiguousTensor positions,
            int ropeDim,
            int mode,
            int originalContextLength,
            float freqBase,
            float freqScale,
            float extFactor,
            float attnFactor,
            float betaFast,
            float betaSlow,
            int addToResult,
            int invertPositions,
            IntPtr freqFactors,
            int freqFactorsLen);

        // 中文：确保指定后端可用，不可用或加载失败时抛出明确异常。
        public static void EnsureAvailable(GgmlBackendType backendType)
        {
            if (backendType == GgmlBackendType.Metal && !OperatingSystem.IsMacOS())
            {
                throw new PlatformNotSupportedException("The GGML Metal backend is available on macOS only.");
            }

            if (backendType == GgmlBackendType.Cuda && !IsCudaPlatformSupported())
            {
                throw new PlatformNotSupportedException("The GGML CUDA backend is supported on Windows and Linux only.");
            }

            try
            {
                if (TSGgml_IsBackendAvailable((int)backendType) == 0)
                {
                    string backendName = backendType switch
                    {
                        GgmlBackendType.Metal => "ggml-metal",
                        GgmlBackendType.Cuda => "ggml-cuda",
                        _ => "ggml-cpu",
                    };
                    throw new InvalidOperationException($"Failed to initialize {backendName}. {GetBackendAvailabilityHint(backendType)}");
                }
            }
            catch (DllNotFoundException ex)
            {
                throw new InvalidOperationException("Failed to load the native GGML bridge. Build `TensorSharp.GGML.Native` first.", ex);
            }
            catch (EntryPointNotFoundException ex)
            {
                throw new InvalidOperationException("The native GGML bridge is out of date. Rebuild `TensorSharp.GGML.Native`.", ex);
            }
        }

        // 中文：判断指定后端能否初始化，返回布尔值（不抛异常）。
        public static bool CanInitialize(GgmlBackendType backendType)
        {
            if (backendType == GgmlBackendType.Metal && !OperatingSystem.IsMacOS())
            {
                return false;
            }

            if (backendType == GgmlBackendType.Cuda && !IsCudaPlatformSupported())
            {
                return false;
            }

            try
            {
                return TSGgml_CanInitializeBackend((int)backendType) != 0;
            }
            catch (DllNotFoundException)
            {
                return false;
            }
            catch (EntryPointNotFoundException)
            {
                return false;
            }
        }

        // 中文：Addmm 的托管封装（F32 带偏置矩阵乘）。
        public static void Addmm(GgmlTensorView2D result, GgmlTensorView2D src, GgmlTensorView2D m1, GgmlTensorView2D m2, float beta, float alpha)
        {
            CheckResult(TSGgml_AddmmF32(result, src, m1, m2, beta, alpha), "addmm");
        }

        // 中文：AddmmQuant 的托管封装（量化权重矩阵乘）。
        public static void AddmmQuant(GgmlTensorView2D result, GgmlTensorView2D m1, IntPtr m2Data, int m2GgmlType, long m2Ne0, long m2Ne1, long m2RawBytes)
        {
            CheckResult(TSGgml_AddmmQuantF32(result, m1, m2Data, m2GgmlType, m2Ne0, m2Ne1, m2RawBytes), "addmm_quant");
        }

        // 中文：FusedRmsNormMatMulQuant 的托管封装（RMSNorm + 量化矩阵乘）。
        public static void FusedRmsNormMatMulQuant(
            GgmlTensorView2D result, GgmlTensorView2D input,
            IntPtr normWeightData, int normWeightCount, float eps,
            IntPtr m2Data, int m2GgmlType, long m2Ne0, long m2Ne1, long m2RawBytes)
        {
            CheckResult(TSGgml_FusedRmsNormMatMulQuantF32(
                result, input, normWeightData, normWeightCount, eps,
                m2Data, m2GgmlType, m2Ne0, m2Ne1, m2RawBytes), "fused_rms_norm_matmul_quant");
        }

        // 中文：FusedMatMulQuantAdd 的托管封装（量化矩阵乘 + 残差相加）。
        public static void FusedMatMulQuantAdd(
            GgmlTensorView2D residual, GgmlTensorView2D input,
            IntPtr m2Data, int m2GgmlType, long m2Ne0, long m2Ne1, long m2RawBytes)
        {
            CheckResult(TSGgml_FusedMatMulQuantAddF32(
                residual, input, m2Data, m2GgmlType, m2Ne0, m2Ne1, m2RawBytes), "fused_matmul_quant_add");
        }

        // 中文：FusedFFNSwiGLUQuant 的托管封装（融合 SwiGLU 前馈网络）。
        public static void FusedFFNSwiGLUQuant(
            GgmlTensorView2D residual,
            GgmlTensorView2D input,
            IntPtr normWeightData,
            int normWeightCount,
            float eps,
            IntPtr gateUpData, int gateUpGgmlType, long gateUpNe0, long gateUpNe1, long gateUpRawBytes,
            IntPtr downData, int downGgmlType, long downNe0, long downNe1, long downRawBytes,
            int halfDim)
        {
            CheckResult(TSGgml_FusedFFNSwiGLUQuantF32(
                residual, input, normWeightData, normWeightCount, eps,
                gateUpData, gateUpGgmlType, gateUpNe0, gateUpNe1, gateUpRawBytes,
                downData, downGgmlType, downNe0, downNe1, downRawBytes,
                halfDim), "fused_ffn_swiglu_quant");
        }

        // 中文：FusedOutProjNormRouter 的托管封装（输出投影 + 归一化 + 路由打分）。
        public static void FusedOutProjNormRouter(
            GgmlTensorView2D residual, GgmlTensorView2D input,
            IntPtr outProjData, int outProjType, long outNe0, long outNe1, long outBytes,
            IntPtr normData, int normCount, float eps,
            GgmlTensorView2D normedOut,
            IntPtr routerData, int routerType, long routerNe0, long routerNe1, long routerBytes,
            GgmlTensorView2D routerOut)
        {
            CheckResult(TSGgml_FusedOutProjNormRouterQuantF32(residual, input,
                outProjData, outProjType, outNe0, outNe1, outBytes,
                normData, normCount, eps, normedOut,
                routerData, routerType, routerNe0, routerNe1, routerBytes,
                routerOut), "fused_outproj_norm_router");
        }

        // 中文：FusedOutProjFFN 的托管封装（输出投影 + FFN）。
        public static void FusedOutProjFFN(
            GgmlTensorView2D residual, GgmlTensorView2D input,
            IntPtr outProjData, int outProjType, long outNe0, long outNe1, long outRawBytes,
            IntPtr ffnNormData, int ffnNormCount, float eps,
            IntPtr guData, int guType, long guNe0, long guNe1, long guRawBytes,
            IntPtr dnData, int dnType, long dnNe0, long dnNe1, long dnRawBytes,
            int halfDim)
        {
            CheckResult(TSGgml_FusedOutProjFFNQuantF32(residual, input,
                outProjData, outProjType, outNe0, outNe1, outRawBytes,
                ffnNormData, ffnNormCount, eps,
                guData, guType, guNe0, guNe1, guRawBytes,
                dnData, dnType, dnNe0, dnNe1, dnRawBytes,
                halfDim), "fused_outproj_ffn");
        }

        // 中文：FusedVisionMLP 的托管封装（视觉 MLP）。
        public static void FusedVisionMLP(
            GgmlTensorView2D hidden,
            IntPtr lnW, IntPtr lnB, int lnDim, float eps,
            IntPtr upW, int upNe0, int upNe1, long upBytes,
            IntPtr upB, int upBDim,
            IntPtr downW, int downNe0, int downNe1, long downBytes,
            IntPtr downB, int downBDim)
        {
            CheckResult(TSGgml_FusedVisionMLPF32(hidden,
                lnW, lnB, lnDim, eps,
                upW, upNe0, upNe1, upBytes, upB, upBDim,
                downW, downNe0, downNe1, downBytes, downB, downBDim), "fused_vision_mlp");
        }

        // 中文：FusedVisionAttention 的托管封装（视觉自注意力）。
        public static void FusedVisionAttention(
            GgmlTensorView2D hidden,
            IntPtr lnW, IntPtr lnB, int lnDim, float eps,
            IntPtr qkvW, int qkvNe0, int qkvNe1, long qkvBytes,
            IntPtr qkvB, int qkvBDim,
            IntPtr outW, int outNe0, int outNe1, long outBytes,
            IntPtr outB, int outBDim,
            IntPtr cosTable, IntPtr sinTable,
            int numPatches, int numHeads, int headDim, int halfDim,
            float attnScale)
        {
            CheckResult(TSGgml_FusedVisionAttentionF32(hidden,
                lnW, lnB, lnDim, eps,
                qkvW, qkvNe0, qkvNe1, qkvBytes, qkvB, qkvBDim,
                outW, outNe0, outNe1, outBytes, outB, outBDim,
                cosTable, sinTable, numPatches, numHeads, headDim, halfDim,
                attnScale), "fused_vision_attention");
        }

        // 中文：GetRowsQuant 的托管封装（量化矩阵按索引取行）。
        public static void GetRowsQuant(GgmlTensorView2D result, IntPtr srcData, int srcGgmlType, long srcNe0, long srcNe1, long srcRawBytes, GgmlContiguousTensor indices)
        {
            CheckResult(TSGgml_GetRowsQuantF32(result, srcData, srcGgmlType, srcNe0, srcNe1, srcRawBytes, indices), "get_rows_quant");
        }

        // 中文：MoEExpertsForward 的托管封装（MoE 多专家前向）。
        public static void MoEExpertsForward(GgmlTensorView2D result, GgmlTensorView2D input,
            int numExperts, IntPtr[] upDataPtrs, IntPtr[] downDataPtrs,
            int upGgmlType, long upNe0, long upNe1, long upRawBytesEach,
            int downGgmlType, long downNe0, long downNe1, long downRawBytesEach,
            float[] routeWeights)
        {
            CheckResult(TSGgml_MoEExpertsForwardF32(result, input, numExperts,
                upDataPtrs, downDataPtrs,
                upGgmlType, upNe0, upNe1, upRawBytesEach,
                downGgmlType, downNe0, downNe1, downRawBytesEach,
                routeWeights), "moe_experts_forward");
        }

        // 中文：MoEExpertsSwiGLUForward 的托管封装（MoE 多专家 SwiGLU 前向）。
        public static void MoEExpertsSwiGLUForward(GgmlTensorView2D result, GgmlTensorView2D input,
            int numExperts,
            IntPtr[] gateDataPtrs, IntPtr[] upDataPtrs, IntPtr[] downDataPtrs,
            int gateGgmlType, long gateNe0, long gateNe1, long gateRawBytesEach,
            int upGgmlType, long upNe0, long upNe1, long upRawBytesEach,
            int downGgmlType, long downNe0, long downNe1, long downRawBytesEach,
            float[] routeWeights)
        {
            CheckResult(TSGgml_MoEExpertsSwiGLUForwardF32(result, input, numExperts,
                gateDataPtrs, upDataPtrs, downDataPtrs,
                gateGgmlType, gateNe0, gateNe1, gateRawBytesEach,
                upGgmlType, upNe0, upNe1, upRawBytesEach,
                downGgmlType, downNe0, downNe1, downRawBytesEach,
                routeWeights), "moe_experts_swiglu_forward");
        }

        // 中文：MoEExpertsSwiGLUResidual 的托管封装（MoE SwiGLU 前向 + 残差，可选共享专家）。
        public static void MoEExpertsSwiGLUResidual(GgmlTensorView2D residual, GgmlTensorView2D input,
            int numExperts,
            IntPtr[] gateDataPtrs, IntPtr[] upDataPtrs, IntPtr[] downDataPtrs,
            int gateGgmlType, long gateNe0, long gateNe1, long gateRawBytesEach,
            int upGgmlType, long upNe0, long upNe1, long upRawBytesEach,
            int downGgmlType, long downNe0, long downNe1, long downRawBytesEach,
            float[] routeWeights,
            bool useShared,
            IntPtr sharedGateData, IntPtr sharedUpData, IntPtr sharedDownData,
            int sharedGateGgmlType, long sharedGateNe0, long sharedGateNe1, long sharedGateRawBytes,
            int sharedUpGgmlType, long sharedUpNe0, long sharedUpNe1, long sharedUpRawBytes,
            int sharedDownGgmlType, long sharedDownNe0, long sharedDownNe1, long sharedDownRawBytes,
            float sharedScalar)
        {
            CheckResult(TSGgml_MoEExpertsSwiGLUResidualF32(residual, input, numExperts,
                gateDataPtrs, upDataPtrs, downDataPtrs,
                gateGgmlType, gateNe0, gateNe1, gateRawBytesEach,
                upGgmlType, upNe0, upNe1, upRawBytesEach,
                downGgmlType, downNe0, downNe1, downRawBytesEach,
                routeWeights,
                useShared ? 1 : 0,
                sharedGateData, sharedUpData, sharedDownData,
                sharedGateGgmlType, sharedGateNe0, sharedGateNe1, sharedGateRawBytes,
                sharedUpGgmlType, sharedUpNe0, sharedUpNe1, sharedUpRawBytes,
                sharedDownGgmlType, sharedDownNe0, sharedDownNe1, sharedDownRawBytes,
                sharedScalar), "moe_experts_swiglu_residual");
        }

        // 中文：AddmmQuantBatch 的托管封装（批量量化矩阵乘）。
        public static void AddmmQuantBatch(GgmlTensorView2D result, GgmlTensorView2D m1, IntPtr m2Data, int m2GgmlType, long m2Ne0, long m2RawBytes,
            int batchCount, long[] weightOffsets, long[] weightNe1Arr)
        {
            CheckResult(TSGgml_AddmmQuantBatchF32(result, m1, m2Data, m2GgmlType, m2Ne0, m2RawBytes, batchCount, weightOffsets, weightNe1Arr), "addmm_quant_batch");
        }

        // 中文：AddmmBatch 的托管封装（3D 批量 addmm）。
        public static void AddmmBatch(GgmlTensorView3D result, GgmlTensorView3D src, GgmlTensorView3D m1, GgmlTensorView3D m2, float beta, float alpha)
        {
            CheckResult(TSGgml_AddmmBatchF32(result, src, m1, m2, beta, alpha), "addmmbatch");
        }

        // 中文：MulMatId 的托管封装（按专家 id 的索引矩阵乘）。
        public static void MulMatId(GgmlTensorView3D result, GgmlTensorView3D expertWeights, GgmlTensorView3D input, GgmlContiguousTensor ids, int idsRows, int idsCols)
        {
            CheckResult(TSGgml_MulMatIdF32(result, expertWeights, input, ids, idsRows, idsCols), "mulmatid");
        }

        // 中文：AddId 的托管封装（按专家 id 的索引偏置相加）。
        public static void AddId(GgmlTensorView3D result, GgmlTensorView3D src, GgmlTensorView2D bias, GgmlContiguousTensor ids, int idsRows, int idsCols)
        {
            CheckResult(TSGgml_AddIdF32(result, src, bias, ids, idsRows, idsCols), "addid");
        }

        // 中文：ReduceLastDim 的托管封装（沿最后一维归约）。
        public static void ReduceLastDim(GgmlReductionOp op, GgmlTensorView4D result, GgmlTensorView4D src)
        {
            CheckResult(TSGgml_ReduceLastDimF32((int)op, result, src), op.ToString());
        }

        // 中文：IndexReduction 的托管封装（argmin/argmax 索引归约）。
        public static void IndexReduction(GgmlIndexReductionOp op, GgmlTensorView4D result, GgmlTensorView4D src)
        {
            CheckResult(TSGgml_IndexReductionF32((int)op, result, src), op.ToString());
        }

        // 中文：Softmax 的托管封装。
        public static void Softmax(GgmlTensorView4D result, GgmlTensorView4D src)
        {
            CheckResult(TSGgml_SoftmaxF32(result, src), "softmax");
        }

        /// <summary>
        /// In-place softmax with causal+SWA mask and optional attention sinks.
        /// scores layout is [numHeads, seqLen, kvLen]. sinksData may be IntPtr.Zero
        /// when no sinks are needed; slidingWindow &lt;= 0 disables the SWA mask.
        ///
        /// Replaces three separate ops in the GptOss attention path: AddCausalMask
        /// (GPU) + ApplySWAMask (CPU) + ApplySoftmaxWithSinks (CPU). The CPU
        /// softmax-with-sinks loop dominated GptOss prefill (~76% of total time
        /// on pp2048) because it walked ~6 billion elements through MathF.Exp on
        /// a single thread; folding it into one Metal kernel collapses that.
        /// </summary>
        // 中文：AttentionSoftmaxWithSinks 的托管封装（原地因果+SWA 掩码 softmax，含 sinks）。
        public static void AttentionSoftmaxWithSinks(
            GgmlTensorView3D scores,
            IntPtr sinksData,
            int numHeads,
            int seqLen,
            int kvLen,
            int maskStartPos,
            int slidingWindow,
            float scale)
        {
            CheckResult(TSGgml_AttentionSoftmaxWithSinksF32(
                scores, sinksData, numHeads, seqLen, kvLen,
                maskStartPos, slidingWindow, scale),
                "attention_softmax_with_sinks");
        }

        // 中文：MoEFFNPrefillSwiGLUQuant 的托管封装（整层 MoE FFN 预填充前向）。
        public static void MoEFFNPrefillSwiGLUQuant(
            IntPtr hiddenIn,
            IntPtr hiddenOut,
            int seqLen,
            int hiddenDim,
            int nFf,
            int numExperts,
            int nUsed,
            IntPtr selectedExperts,
            IntPtr routingWeights,
            IntPtr gateData, int gateType, long gateNe0, long gateNe1, long gateTotalBytes,
            IntPtr upData,   int upType,   long upNe0,   long upNe1,   long upTotalBytes,
            IntPtr downData, int downType, long downNe0, long downNe1, long downTotalBytes,
            IntPtr gateBias,
            IntPtr upBias,
            IntPtr downBias,
            int activationType,
            float oaiAlpha,
            float oaiLimit)
        {
            CheckResult(TSGgml_MoEFFNPrefillSwiGLUQuantF32(
                hiddenIn, hiddenOut, seqLen, hiddenDim, nFf,
                numExperts, nUsed, selectedExperts, routingWeights,
                gateData, gateType, gateNe0, gateNe1, gateTotalBytes,
                upData,   upType,   upNe0,   upNe1,   upTotalBytes,
                downData, downType, downNe0, downNe1, downTotalBytes,
                gateBias, upBias, downBias,
                activationType, oaiAlpha, oaiLimit),
                "moe_ffn_prefill_swiglu_quant");
        }

        // 中文：Gemma4MoEGEGLUResidual 的托管封装（MoE GEGLU + post_norm + 残差）。
        public static void Gemma4MoEGEGLUResidual(
            IntPtr hiddenIn,
            IntPtr residualInOut,
            IntPtr postNormW,
            float postNormEps,
            int seqLen,
            int hiddenDim,
            int nFf,
            int numExperts,
            int nUsed,
            IntPtr selectedExperts,
            IntPtr routingWeights,
            IntPtr gateData, int gateType, long gateNe0, long gateNe1, long gateTotalBytes,
            IntPtr upData,   int upType,   long upNe0,   long upNe1,   long upTotalBytes,
            IntPtr downData, int downType, long downNe0, long downNe1, long downTotalBytes,
            IntPtr gateBias,
            IntPtr upBias,
            IntPtr downBias,
            int activationType,
            float oaiAlpha,
            float oaiLimit)
        {
            CheckResult(TSGgml_Gemma4MoEGEGLUResidualF32(
                hiddenIn, residualInOut, postNormW, postNormEps,
                seqLen, hiddenDim, nFf,
                numExperts, nUsed, selectedExperts, routingWeights,
                gateData, gateType, gateNe0, gateNe1, gateTotalBytes,
                upData,   upType,   upNe0,   upNe1,   upTotalBytes,
                downData, downType, downNe0, downNe1, downTotalBytes,
                gateBias, upBias, downBias,
                activationType, oaiAlpha, oaiLimit),
                "gemma4_moe_geglu_residual");
        }

        // 中文：ScaledDotProductAttention 的托管封装（缩放点积注意力）。
        public static void ScaledDotProductAttention(GgmlTensorView4D result, GgmlTensorView4D query, GgmlTensorView4D key, GgmlTensorView4D value, GgmlTensorView4D mask, bool hasMask, float scale)
        {
            CheckResult(TSGgml_ScaledDotProductAttentionF32(result, query, key, value, mask, hasMask ? 1 : 0, scale), "scaled_dot_product_attention");
        }

        // 中文：SoftmaxGrad 的托管封装（softmax 反向梯度）。
        public static void SoftmaxGrad(GgmlTensorView4D result, GgmlTensorView4D adj, GgmlTensorView4D val, bool addGrad)
        {
            CheckResult(TSGgml_SoftmaxGradF32(result, adj, val, addGrad ? 1 : 0), "softmaxgrad");
        }

        // 中文：CrossEntropyLoss 的托管封装，返回标量损失值。
        public static float CrossEntropyLoss(GgmlTensorView4D probs, GgmlContiguousTensor targetIndices, float smooth, float labelSmooth)
        {
            CheckResult(TSGgml_CrossEntropyLossF32(out float lossValue, probs, targetIndices, smooth, labelSmooth), "crossentropyloss");
            return lossValue;
        }

        // 中文：CrossEntropyLossBackward 的托管封装（交叉熵反向梯度）。
        public static void CrossEntropyLossBackward(GgmlTensorView4D grad, GgmlTensorView4D probs, GgmlContiguousTensor targetIndices, float lossGradient, float smooth, float labelSmooth, bool addGrad)
        {
            CheckResult(TSGgml_CrossEntropyLossBackwardF32(grad, probs, targetIndices, lossGradient, smooth, labelSmooth, addGrad ? 1 : 0), "crossentropyloss_backward");
        }

        // 中文：Adam 的托管封装（Adam 优化器原地更新权重）。
        public static void Adam(
            GgmlContiguousTensor weight,
            GgmlContiguousTensor gradient,
            GgmlContiguousTensor v,
            GgmlContiguousTensor m,
            float gradNormFactor,
            float stepSize,
            float clipValue,
            float regc,
            float decayRateV,
            float decayRateM,
            int iter,
            float eps)
        {
            CheckResult(TSGgml_AdamF32(weight, gradient, v, m, gradNormFactor, stepSize, clipValue, regc, decayRateV, decayRateM, iter, eps), "adam");
        }

        // 中文：Copy 的托管封装（张量逐元素拷贝）。
        public static void Copy(GgmlTensorView4D result, GgmlTensorView4D src)
        {
            CheckResult(TSGgml_CopyF32(result, src), "copy");
        }

        // 中文：Unary 的托管封装（逐元素一元运算）。
        public static void Unary(GgmlUnaryOp op, GgmlTensorView4D result, GgmlTensorView4D src)
        {
            CheckResult(TSGgml_UnaryF32((int)op, result, src), op.ToString());
        }

        // 中文：BinaryTensor 的托管封装（两张量逐元素二元运算）。
        public static void BinaryTensor(GgmlBinaryTensorOp op, GgmlTensorView4D result, GgmlTensorView4D lhs, GgmlTensorView4D rhs)
        {
            CheckResult(TSGgml_BinaryTensorF32((int)op, result, lhs, rhs), op.ToString());
        }

        // 中文：FusedActMul 的托管封装（融合激活后相乘）。
        public static void FusedActMul(GgmlFusedActMulOp op, GgmlTensorView4D result, GgmlTensorView4D a, GgmlTensorView4D b)
        {
            CheckResult(TSGgml_FusedActMulF32((int)op, result, a, b), op.ToString());
        }

        // 中文：FusedActMulSplit 的托管封装（gate_up 切分后激活相乘）。
        public static void FusedActMulSplit(GgmlFusedActMulOp op, GgmlTensorView2D result, GgmlTensorView2D gateUp, int halfDim)
        {
            CheckResult(TSGgml_FusedActMulSplitF32((int)op, result, gateUp, halfDim), op.ToString() + "Split");
        }

        // 中文：BinaryScalar 的托管封装（张量与标量逐元素运算）。
        public static void BinaryScalar(GgmlBinaryScalarOp op, GgmlTensorView4D result, GgmlTensorView4D src, float scalar)
        {
            CheckResult(TSGgml_BinaryScalarF32((int)op, result, src, scalar), op.ToString());
        }

        // 中文：ActivationGrad 的托管封装（激活函数反向梯度）。
        public static void ActivationGrad(GgmlActivationGradOp op, GgmlTensorView4D result, GgmlTensorView4D src, GgmlTensorView4D grad, GgmlTensorView4D accumulation, bool hasAccumulation)
        {
            CheckResult(TSGgml_ActivationGradF32((int)op, result, src, grad, accumulation, hasAccumulation ? 1 : 0), $"{op}Grad");
        }

        // 中文：Norm 的托管封装（LayerNorm/RMSNorm 归一化）。
        public static void Norm(GgmlNormOp op, GgmlTensorView4D result, GgmlTensorView4D src, GgmlTensorView4D gamma, GgmlTensorView4D beta, bool hasBeta, float eps)
        {
            CheckResult(TSGgml_NormF32((int)op, result, src, gamma, beta, hasBeta ? 1 : 0, eps), op.ToString());
        }

        // 中文：NormGrad 的托管封装（归一化反向梯度）。
        public static void NormGrad(GgmlNormOp op, GgmlTensorView4D result, GgmlTensorView4D gradGamma, GgmlTensorView4D gradBeta, GgmlTensorView4D adj, GgmlTensorView4D x, GgmlTensorView4D gamma, bool hasGradBeta, float eps)
        {
            CheckResult(TSGgml_NormGradF32((int)op, result, gradGamma, gradBeta, adj, x, gamma, hasGradBeta ? 1 : 0, eps), $"{op}Grad");
        }

        // 中文：IndexSelect 的托管封装（按索引选取行）。
        public static void IndexSelect(GgmlTensorView2D result, GgmlTensorView2D src, GgmlContiguousTensor indices, bool addToResult)
        {
            CheckResult(TSGgml_IndexSelectF32(result, src, indices, addToResult ? 1 : 0), "indexselect");
        }

        // 中文：IndexSelectGrad 的托管封装（index_select 反向梯度）。
        public static void IndexSelectGrad(GgmlTensorView2D grad, GgmlTensorView2D adj, GgmlContiguousTensor indices)
        {
            CheckResult(TSGgml_IndexSelectGradF32(grad, adj, indices), "indexselectgrad");
        }

        // 中文：RoPE 的托管封装（基础旋转位置编码前向）。
        public static void RoPE(GgmlTensorView4D result, GgmlTensorView4D src, int seqLen, int rowOffset)
        {
            CheckResult(TSGgml_RoPEF32(result, src, seqLen, rowOffset, 0, 0), "rope");
        }

        // 中文：RoPEGrad 的托管封装（RoPE 反向梯度，反转位置）。
        public static void RoPEGrad(GgmlTensorView4D result, GgmlTensorView4D adj, int seqLen, int rowOffset)
        {
            CheckResult(TSGgml_RoPEF32(result, adj, seqLen, rowOffset, 1, 1), "ropegrad");
        }

        // 中文：RoPEEx 的托管封装（扩展 RoPE，含 YaRN 参数）。
        public static void RoPEEx(
            GgmlTensorView4D result,
            GgmlTensorView4D src,
            GgmlContiguousTensor positions,
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
            CheckResult(
                TSGgml_RoPEExF32(
                    result,
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
                    addToResult ? 1 : 0,
                    invertPositions ? 1 : 0),
                "rope_ex");
        }

        // 中文：RoPEMRoPE 的托管封装（多模态 mRoPE）。
        public static void RoPEMRoPE(
            GgmlTensorView4D result,
            GgmlTensorView4D src,
            GgmlContiguousTensor positions,
            int ropeDim,
            int mode,
            int sect0, int sect1, int sect2, int sect3,
            int originalContextLength,
            float freqBase,
            float freqScale,
            float extFactor,
            float attnFactor,
            float betaFast,
            float betaSlow)
        {
            CheckResult(
                TSGgml_RoPEMRoPEF32(
                    result, src, positions,
                    ropeDim, mode,
                    sect0, sect1, sect2, sect3,
                    originalContextLength,
                    freqBase, freqScale,
                    extFactor, attnFactor,
                    betaFast, betaSlow),
                "rope_mrope");
        }

        // 中文：RoPEExWithFreqFactors 的托管封装（带每维频率因子的扩展 RoPE）。
        public static void RoPEExWithFreqFactors(
            GgmlTensorView4D result,
            GgmlTensorView4D src,
            GgmlContiguousTensor positions,
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
            bool invertPositions,
            IntPtr freqFactors,
            int freqFactorsLen)
        {
            CheckResult(
                TSGgml_RoPEExFreqFactorsF32(
                    result,
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
                    addToResult ? 1 : 0,
                    invertPositions ? 1 : 0,
                    freqFactors,
                    freqFactorsLen),
                "rope_ex_ff");
        }

        // 中文：TransformerLayerDecode 的托管封装（单步解码完整 Transformer 层）。
        public static void TransformerLayerDecode(
            IntPtr hiddenData, int hiddenSize,
            IntPtr attnNormData,
            IntPtr qkvData, int qkvType, long qkvNe0, long qkvNe1, long qkvBytes,
            IntPtr qNormData, IntPtr kNormData, int headDim,
            IntPtr oData, int oType, long oNe0, long oNe1, long oBytes,
            IntPtr ffnNormData,
            IntPtr guData, int guType, long guNe0, long guNe1, long guBytes,
            IntPtr downData, int downType, long downNe0, long downNe1, long downBytes,
            IntPtr kCacheData, IntPtr vCacheData,
            int numHeads, int numKvHeads,
            int maxSeqLen, int position,
            float eps, float ropeBase, float ropeFreqScale,
            int intermediateSize, int ropeMode,
            int kvCacheType = 0)
        {
            CheckResult(TSGgml_TransformerLayerDecode(
                hiddenData, hiddenSize,
                attnNormData,
                qkvData, qkvType, qkvNe0, qkvNe1, qkvBytes,
                qNormData, kNormData, headDim,
                oData, oType, oNe0, oNe1, oBytes,
                ffnNormData,
                guData, guType, guNe0, guNe1, guBytes,
                downData, downType, downNe0, downNe1, downBytes,
                kCacheData, vCacheData,
                numHeads, numKvHeads,
                maxSeqLen, position,
                eps, ropeBase, ropeFreqScale,
                intermediateSize, ropeMode, kvCacheType), "transformer_layer_decode");
        }

        /// <summary>
        /// Single-token flash attention decode kernel. Appends the new K/V to the persistent
        /// KV cache at <paramref name="position"/>, then runs <c>ggml_flash_attn_ext</c> on the
        /// device against the populated portion of the cache. Q, K, V, and the output buffer
        /// must point to F32 contiguous memory in (heads, head_dim) row-major layout.
        /// </summary>
        // 中文：FusedPrefillAttention 的托管封装（预填充阶段融合注意力）。
        public static void FusedPrefillAttention(
            IntPtr qData, IntPtr kData, IntPtr vData, IntPtr outData,
            int numHeads, int numKvHeads, int headDim,
            int seqLen, int kvLen,
            int maskStartPos, int slidingWindow,
            float scale, int inputFormat = 0)
        {
            CheckResult(TSGgml_FusedPrefillAttentionF32(
                qData, kData, vData, outData,
                numHeads, numKvHeads, headDim,
                seqLen, kvLen,
                maskStartPos, slidingWindow, scale, inputFormat), "fused_prefill_attention");
        }

        // 中文：FlashAttnDecode 的托管封装（单步解码 Flash Attention）。
        public static void FlashAttnDecode(
            IntPtr qData, IntPtr kData, IntPtr vData,
            IntPtr kCacheData, IntPtr vCacheData,
            IntPtr outData,
            int numHeads, int numKvHeads, int headDim,
            int maxSeqLen, int position,
            float scale, int kvCacheType = 0)
        {
            CheckResult(TSGgml_FlashAttnDecodeF32(
                qData, kData, vData,
                kCacheData, vCacheData,
                outData,
                numHeads, numKvHeads, headDim,
                maxSeqLen, position, scale, kvCacheType), "flash_attn_decode");
        }

        /// <summary>
        /// Native batched paged attention via <c>ggml_flash_attn_ext</c>. For
        /// each sequence in the batch, the C++ side gathers K and V from the
        /// paged buffer (walking the per-sequence block table), then runs the
        /// backend's fused flash-attention kernel. One Metal/CUDA kernel per
        /// sequence per layer, with the gather inside the native side so we
        /// don't pay the managed↔native border crossing N×L times.
        /// </summary>
        /// <param name="qData">[numTokens, numHeads * headDim] row-major float[].</param>
        /// <param name="pagedKData">[numBlocks * blockSize, numKvHeads, headDim] row-major.</param>
        /// <param name="pagedVData">Same layout as pagedKData.</param>
        /// <param name="outData">[numTokens, numHeads * headDim] (writes back).</param>
        /// <param name="queryStartLoc">[numSeqs + 1] cumulative query offsets.</param>
        /// <param name="seqLens">[numSeqs] total context length per sequence.</param>
        /// <param name="positions">[numTokens] absolute position per query token (drives the causal mask).</param>
        /// <param name="blockTableFlat">Concatenated per-sequence block tables.</param>
        /// <param name="blockTableOffsets">[numSeqs] offset of each seq's table inside blockTableFlat.</param>
        // 中文：PagedAttentionForward 的托管封装，固定托管数组后调用原生分页注意力。
        public static unsafe void PagedAttentionForward(
            float[] qData,
            float[] pagedKData,
            float[] pagedVData,
            float[] outData,
            int[] queryStartLoc,
            int[] seqLens,
            int[] positions,
            int[] blockTableFlat,
            int[] blockTableOffsets,
            int numSeqs,
            int numTokens,
            int numHeads,
            int numKvHeads,
            int headDim,
            int blockSize,
            float scale,
            int slidingWindow = 0)
        {
            fixed (float* q = qData)
            fixed (float* kp = pagedKData)
            fixed (float* vp = pagedVData)
            fixed (float* o = outData)
            fixed (int* qsl = queryStartLoc)
            fixed (int* sl = seqLens)
            fixed (int* pos = positions)
            fixed (int* btf = blockTableFlat)
            fixed (int* bto = blockTableOffsets)
            {
                CheckResult(TSGgml_PagedAttentionForward(
                    (IntPtr)q, (IntPtr)kp, (IntPtr)vp, (IntPtr)o,
                    (IntPtr)qsl, (IntPtr)sl, (IntPtr)pos,
                    (IntPtr)btf, (IntPtr)bto,
                    numSeqs, numTokens, numHeads, numKvHeads, headDim,
                    blockSize, slidingWindow, scale), "paged_attention_forward");
            }
        }

        /// <summary>Native paged-attention forward with per-head attention
        /// sinks (gpt-oss style). Sinks is a [numHeads] F32 array; null
        /// degenerates to the regular paged attention. Goes through
        /// ggml_flash_attn_ext_add_sinks under the hood so the Metal/CUDA
        /// flash-attn kernel includes the sink as a virtual softmax position.</summary>
        // 中文：PagedAttentionForwardWithSinks 的托管封装（带 attention sinks 的分页注意力）。
        public static unsafe void PagedAttentionForwardWithSinks(
            float[] qData,
            float[] pagedKData,
            float[] pagedVData,
            float[] outData,
            int[] queryStartLoc,
            int[] seqLens,
            int[] positions,
            int[] blockTableFlat,
            int[] blockTableOffsets,
            int numSeqs,
            int numTokens,
            int numHeads,
            int numKvHeads,
            int headDim,
            int blockSize,
            float scale,
            int slidingWindow,
            float[] sinksData)
        {
            fixed (float* q = qData)
            fixed (float* kp = pagedKData)
            fixed (float* vp = pagedVData)
            fixed (float* o = outData)
            fixed (int* qsl = queryStartLoc)
            fixed (int* sl = seqLens)
            fixed (int* pos = positions)
            fixed (int* btf = blockTableFlat)
            fixed (int* bto = blockTableOffsets)
            fixed (float* sink = sinksData)
            {
                CheckResult(TSGgml_PagedAttentionForwardWithSinks(
                    (IntPtr)q, (IntPtr)kp, (IntPtr)vp, (IntPtr)o,
                    (IntPtr)qsl, (IntPtr)sl, (IntPtr)pos,
                    (IntPtr)btf, (IntPtr)bto,
                    numSeqs, numTokens, numHeads, numKvHeads, headDim,
                    blockSize, slidingWindow, scale,
                    sinksData != null ? (IntPtr)sink : IntPtr.Zero),
                    "paged_attention_forward_with_sinks");
            }
        }

        /// <summary>
        /// GPU-resident paged-attention forward. <paramref name="qData"/> and
        /// <paramref name="outData"/> point to backend-allocated buffers
        /// (typically <c>tensor.Storage.PtrAtElement(...)</c> on the Metal /
        /// CUDA backend). The kernel zero-copy binds Q's tensor and writes
        /// the attention output directly into the caller's output tensor —
        /// no host-side memcpy round-trip, no per-layer
        /// <c>ggml_backend_synchronize</c>. K/V paged storage is still passed
        /// as host arrays.
        /// </summary>
        // 中文：PagedAttentionForwardDevice 的托管封装（GPU 常驻分页注意力，零拷贝绑定 q/out）。
        public static unsafe void PagedAttentionForwardDevice(
            IntPtr qData,
            float[] pagedKData,
            float[] pagedVData,
            IntPtr outData,
            int[] queryStartLoc,
            int[] seqLens,
            int[] positions,
            int[] blockTableFlat,
            int[] blockTableOffsets,
            int numSeqs,
            int numTokens,
            int numHeads,
            int numKvHeads,
            int headDim,
            int blockSize,
            float scale,
            int slidingWindow = 0)
        {
            fixed (float* kp = pagedKData)
            fixed (float* vp = pagedVData)
            fixed (int* qsl = queryStartLoc)
            fixed (int* sl = seqLens)
            fixed (int* pos = positions)
            fixed (int* btf = blockTableFlat)
            fixed (int* bto = blockTableOffsets)
            {
                CheckResult(TSGgml_PagedAttentionForwardDevice(
                    qData, (IntPtr)kp, (IntPtr)vp, outData,
                    (IntPtr)qsl, (IntPtr)sl, (IntPtr)pos,
                    (IntPtr)btf, (IntPtr)bto,
                    numSeqs, numTokens, numHeads, numKvHeads, headDim,
                    blockSize, slidingWindow, scale),
                    "paged_attention_forward_device");
            }
        }

        /// <summary>GPU-resident paged-attention forward with per-head
        /// attention sinks. Pass <c>null</c> for <paramref name="sinksData"/>
        /// to match <see cref="PagedAttentionForwardDevice"/>.</summary>
        // 中文：PagedAttentionForwardDeviceWithSinks 的托管封装（GPU 常驻且带 sinks 的分页注意力）。
        public static unsafe void PagedAttentionForwardDeviceWithSinks(
            IntPtr qData,
            float[] pagedKData,
            float[] pagedVData,
            IntPtr outData,
            int[] queryStartLoc,
            int[] seqLens,
            int[] positions,
            int[] blockTableFlat,
            int[] blockTableOffsets,
            int numSeqs,
            int numTokens,
            int numHeads,
            int numKvHeads,
            int headDim,
            int blockSize,
            float scale,
            int slidingWindow,
            float[] sinksData)
        {
            fixed (float* kp = pagedKData)
            fixed (float* vp = pagedVData)
            fixed (int* qsl = queryStartLoc)
            fixed (int* sl = seqLens)
            fixed (int* pos = positions)
            fixed (int* btf = blockTableFlat)
            fixed (int* bto = blockTableOffsets)
            fixed (float* sink = sinksData)
            {
                CheckResult(TSGgml_PagedAttentionForwardDeviceWithSinks(
                    qData, (IntPtr)kp, (IntPtr)vp, outData,
                    (IntPtr)qsl, (IntPtr)sl, (IntPtr)pos,
                    (IntPtr)btf, (IntPtr)bto,
                    numSeqs, numTokens, numHeads, numKvHeads, headDim,
                    blockSize, slidingWindow, scale,
                    sinksData != null ? (IntPtr)sink : IntPtr.Zero),
                    "paged_attention_forward_device_with_sinks");
            }
        }

        // 中文：Qwen35AttentionLayerDecode 的托管封装（Qwen3.5 单步解码注意力层）。
        public static void Qwen35AttentionLayerDecode(
            IntPtr residualData, int hiddenSize,
            IntPtr attnNormData,
            IntPtr qkvData, int qkvType, long qkvNe0, long qkvNe1, long qkvBytes,
            IntPtr qNormData, IntPtr kNormData, int headDim,
            IntPtr oData, int oType, long oNe0, long oNe1, long oBytes,
            IntPtr kCacheData, IntPtr vCacheData,
            int numHeads, int numKvHeads,
            int maxSeqLen, int position,
            float eps, float ropeBase, float ropeFreqScale,
            int ropeMode, int kvCacheType = 0)
        {
            CheckResult(TSGgml_Qwen35AttentionLayerDecode(
                residualData, hiddenSize,
                attnNormData,
                qkvData, qkvType, qkvNe0, qkvNe1, qkvBytes,
                qNormData, kNormData, headDim,
                oData, oType, oNe0, oNe1, oBytes,
                kCacheData, vCacheData,
                numHeads, numKvHeads,
                maxSeqLen, position,
                eps, ropeBase, ropeFreqScale, ropeMode, kvCacheType), "qwen35_attention_layer_decode");
        }

        // 中文：TransformerModelDecode 的托管封装（整模型单步解码）。
        public static void TransformerModelDecode(
            IntPtr hiddenData, int hiddenSize, int numLayers,
            IntPtr[] attnNormArr, IntPtr[] qkvArr, IntPtr[] qNormArr, IntPtr[] kNormArr,
            IntPtr[] oArr, IntPtr[] ffnNormArr, IntPtr[] guArr, IntPtr[] downArr,
            IntPtr[] kCacheArr, IntPtr[] vCacheArr,
            int qkvType, long qkvNe0, long qkvNe1, long qkvBytes,
            int oType, long oNe0, long oNe1, long oBytes,
            int guType, long guNe0, long guNe1, long guBytes,
            int downType, long downNe0, long downNe1, long downBytes,
            int headDim, int numHeads, int numKvHeads,
            int maxSeqLen, int position,
            float eps, float ropeBase, float ropeFreqScale,
            int intermediateSize, int ropeMode,
            int kvCacheType = 0)
        {
            CheckResult(TSGgml_TransformerModelDecode(
                hiddenData, hiddenSize, numLayers,
                attnNormArr, qkvArr, qNormArr, kNormArr,
                oArr, ffnNormArr, guArr, downArr,
                kCacheArr, vCacheArr,
                qkvType, qkvNe0, qkvNe1, qkvBytes,
                oType, oNe0, oNe1, oBytes,
                guType, guNe0, guNe1, guBytes,
                downType, downNe0, downNe1, downBytes,
                headDim, numHeads, numKvHeads,
                maxSeqLen, position,
                eps, ropeBase, ropeFreqScale,
                intermediateSize, ropeMode, kvCacheType), "transformer_model_decode");
        }

        // 中文：Gemma4ModelDecode 的托管封装（Gemma 4 整模型单步解码）。
        public static void Gemma4ModelDecode(
            IntPtr hiddenData, int hiddenSize, int numLayers,
            IntPtr[] attnNormArr, IntPtr[] qkvArr, IntPtr[] qNormArr, IntPtr[] kNormArr,
            IntPtr[] oArr, IntPtr[] postAttnNormArr,
            IntPtr[] ffnNormArr, IntPtr[] guArr, IntPtr[] downArr, IntPtr[] postFfnNormArr,
            IntPtr[] kCacheArr, IntPtr[] vCacheArr,
            int[] headDimArr, int[] kvHeadsArr, int[] cacheSizeArr, int[] isLocalArr,
            int[] kvSourceArr,
            float[] ropeBaseArr, float[] layerScalarArr,
            int[] qkvTypeArr, long[] qkvNe0Arr, long[] qkvNe1Arr, long[] qkvBytesArr,
            int[] oTypeArr, long[] oNe0Arr, long[] oNe1Arr, long[] oBytesArr,
            int[] guTypeArr, long[] guNe0Arr, long[] guNe1Arr, long[] guBytesArr,
            int[] downTypeArr, long[] downNe0Arr, long[] downNe1Arr, long[] downBytesArr,
            int numHeads, int position,
            float eps, int slidingWindow,
            IntPtr ropeFreqFactors, int ropeFreqFactorsLen,
            int[] ropeNDimsArr,
            IntPtr pleData, int pleDim,
            IntPtr[] pleGateArr, int[] pleGateTypeArr, long[] pleGateNe0Arr, long[] pleGateNe1Arr, long[] pleGateBytesArr,
            IntPtr[] pleProjArr, int[] pleProjTypeArr, long[] pleProjNe0Arr, long[] pleProjNe1Arr, long[] pleProjBytesArr,
            IntPtr[] plePostNormArr,
            int kvCacheType = 0,
            IntPtr[] kArr = null, int[] kTypeArr = null, long[] kNe0Arr = null, long[] kNe1Arr = null, long[] kBytesArr = null,
            IntPtr[] vArr = null, int[] vTypeArr = null, long[] vNe0Arr = null, long[] vNe1Arr = null, long[] vBytesArr = null)
        {
            CheckResult(TSGgml_Gemma4ModelDecode(
                hiddenData, hiddenSize, numLayers,
                attnNormArr, qkvArr, qNormArr, kNormArr,
                oArr, postAttnNormArr,
                ffnNormArr, guArr, downArr, postFfnNormArr,
                kCacheArr, vCacheArr,
                headDimArr, kvHeadsArr, cacheSizeArr, isLocalArr,
                kvSourceArr,
                ropeBaseArr, layerScalarArr,
                qkvTypeArr, qkvNe0Arr, qkvNe1Arr, qkvBytesArr,
                oTypeArr, oNe0Arr, oNe1Arr, oBytesArr,
                guTypeArr, guNe0Arr, guNe1Arr, guBytesArr,
                downTypeArr, downNe0Arr, downNe1Arr, downBytesArr,
                numHeads, position,
                eps, slidingWindow,
                ropeFreqFactors, ropeFreqFactorsLen,
                ropeNDimsArr,
                pleData, pleDim,
                pleGateArr, pleGateTypeArr, pleGateNe0Arr, pleGateNe1Arr, pleGateBytesArr,
                pleProjArr, pleProjTypeArr, pleProjNe0Arr, pleProjNe1Arr, pleProjBytesArr,
                plePostNormArr, kvCacheType,
                kArr, kTypeArr, kNe0Arr, kNe1Arr, kBytesArr,
                vArr, vTypeArr, vNe0Arr, vNe1Arr, vBytesArr), "gemma4_model_decode");
        }

        // 中文：GatedDeltaNetChunked 的托管封装（门控 DeltaNet 分块前向）。
        public static void GatedDeltaNetChunked(
            GgmlTensorView3D q,
            GgmlTensorView3D k,
            GgmlTensorView3D v,
            GgmlTensorView3D z,
            GgmlTensorView2D alpha,
            GgmlTensorView2D beta,
            GgmlTensorView3D state,
            GgmlTensorView3D gatedOut,
            IntPtr dtBiasData,
            IntPtr aLogData,
            IntPtr ssmNormWData,
            int chunkSize,
            float eps)
        {
            CheckResult(TSGgml_GatedDeltaNetChunkedF32(
                q, k, v, z, alpha, beta, state, gatedOut,
                dtBiasData, aLogData, ssmNormWData,
                chunkSize, eps), "gated_delta_net_chunked");
        }

        // Batched per-token Nemotron Mamba2 step. Runs all (seq, token) pairs
        // for an active decode/prefill batch in one native call, indexing each
        // seq's persistent conv FIFO + SSM state via the seqs[] descriptors.
        // 中文：NemotronMamba2BatchedStep 的托管封装（Mamba2 批量单步推理）。
        public static void NemotronMamba2BatchedStep(
            NemoMamba2BatchedSeqDesc[] seqs,
            int numTokens,
            IntPtr packedBatched,
            int dInProjTotal,
            int dInner,
            int dState,
            int nHead,
            int headDim,
            int nGroup,
            int dConv,
            IntPtr convWt,
            IntPtr convBias,
            IntPtr dtBias,
            IntPtr aLog,
            IntPtr dData,
            IntPtr ssmNormW,
            float eps,
            IntPtr outBatched)
        {
            CheckResult(TSGgml_NemotronMamba2BatchedStepF32(
                seqs?.Length ?? 0, seqs, numTokens,
                packedBatched, dInProjTotal,
                dInner, dState, nHead, headDim, nGroup, dConv,
                convWt, convBias, dtBias, aLog, dData, ssmNormW,
                eps, outBatched),
                "nemotron_mamba2_batched_step");
        }

        // Batched per-token Qwen3.5 GDN step. Runs all (seq, token) pairs for
        // an active decode/prefill batch in one native call, swapping in the
        // matching per-slot conv ring + ssm state via the seqs[] descriptors.
        // The descriptors' ConvWriteIdx field is updated in place — caller
        // copies it back to its per-slot bookkeeping after the call returns.
        // 中文：GatedDeltaNetBatchedStep 的托管封装（Qwen3.5 GDN 批量单步推理）。
        public static void GatedDeltaNetBatchedStep(
            GdnBatchedSeqDesc[] seqs,
            int numTokens,
            IntPtr packedBatched,
            int packedDim,
            int qkvDim,
            int qkDim,
            int vDim,
            int zDim,
            int numKHeads,
            int numVHeads,
            int headKDim,
            int headVDim,
            int convKernel,
            int ssmDInner,
            IntPtr convWt,
            IntPtr dtBias,
            IntPtr aLog,
            IntPtr ssmNormW,
            float eps,
            IntPtr gatedOut)
        {
            CheckResult(TSGgml_GatedDeltaNetBatchedStepF32(
                seqs?.Length ?? 0, seqs, numTokens,
                packedBatched, packedDim, qkvDim, qkDim, vDim, zDim,
                numKHeads, numVHeads, headKDim, headVDim,
                convKernel, ssmDInner,
                convWt, dtBias, aLog, ssmNormW, eps, gatedOut),
                "gated_delta_net_batched_step");
        }

        // 中文：NemotronMamba2Prefill 的托管封装（Mamba2 预填充前向）。
        public static void NemotronMamba2Prefill(
            GgmlTensorView2D projected,
            GgmlTensorView2D hiddenOut,
            IntPtr convStateData,
            int convStateElements,
            IntPtr ssmStateData,
            int ssmStateElements,
            IntPtr convWeightData,
            IntPtr convBiasData,
            IntPtr dtBiasData,
            IntPtr aData,
            IntPtr dData,
            IntPtr ssmNormData,
            int dInner,
            int dState,
            int nHead,
            int headDim,
            int nGroup,
            int dConv,
            float eps)
        {
            CheckResult(TSGgml_NemotronMamba2PrefillF32(
                projected, hiddenOut,
                convStateData, convStateElements,
                ssmStateData, ssmStateElements,
                convWeightData, convBiasData, dtBiasData, aData, dData, ssmNormData,
                dInner, dState, nHead, headDim, nGroup, dConv, eps), "nemotron_mamba2_prefill");
        }

        // 中文：NemotronMamba2Decode 的托管封装（Mamba2 单步解码前向）。
        public static void NemotronMamba2Decode(
            ulong stateKey,
            GgmlTensorView2D projected,
            GgmlTensorView2D hiddenOut,
            IntPtr convStateData,
            int convStateElements,
            IntPtr ssmStateData,
            int ssmStateElements,
            bool initializeState,
            bool downloadState,
            IntPtr convWeightData,
            IntPtr convBiasData,
            IntPtr dtBiasData,
            IntPtr aData,
            IntPtr dData,
            IntPtr ssmNormData,
            int dInner,
            int dState,
            int nHead,
            int headDim,
            int nGroup,
            int dConv,
            float eps)
        {
            CheckResult(TSGgml_NemotronMamba2DecodeF32(
                stateKey, projected, hiddenOut,
                convStateData, convStateElements,
                ssmStateData, ssmStateElements,
                initializeState ? 1 : 0,
                downloadState ? 1 : 0,
                convWeightData, convBiasData, dtBiasData, aData, dData, ssmNormData,
                dInner, dState, nHead, headDim, nGroup, dConv, eps), "nemotron_mamba2_decode");
        }

        // 中文：NemotronMamba2DecodeClear 的托管封装（清除 Mamba2 解码缓存状态）。
        public static void NemotronMamba2DecodeClear(ulong modelKey)
        {
            TSGgml_NemotronMamba2DecodeClear(modelKey);
        }

        /// <summary>Allocate memory with 16 KB alignment (page-aligned for Metal host_ptr).</summary>
        // 中文：AlignedAlloc 的托管封装（分配对齐内存，失败抛 OOM）。
        public static IntPtr AlignedAlloc(long size)
        {
            IntPtr ptr = TSGgml_AlignedAlloc(new UIntPtr((ulong)size));
            if (ptr == IntPtr.Zero && size > 0)
                throw new OutOfMemoryException($"Failed to allocate {size} bytes of aligned memory.");
            return ptr;
        }

        /// <summary>Free memory allocated by AlignedAlloc.</summary>
        // 中文：AlignedFree 的托管封装（释放对齐内存）。
        public static void AlignedFree(IntPtr ptr)
        {
            TSGgml_AlignedFree(ptr);
        }

        /// <summary>Free all cached Metal host_ptr buffer objects.</summary>
        // 中文：ClearHostBufferCache 的托管封装（清空主机缓冲区缓存）。
        public static void ClearHostBufferCache()
        {
            TSGgml_ClearHostBufferCache();
        }

        /// <summary>
        /// Tear down the process-global GGML backend before the C runtime
        /// finalisers run. On macOS the ggml-metal device singleton asserts
        /// that its resource set is empty when its static destructor fires;
        /// if the backend, host-buffer cache, and preloaded-buffer cache
        /// outlive the .NET host the assertion aborts the process on exit.
        /// Hook this onto AppDomain.ProcessExit / ApplicationStopped.
        /// </summary>
        // 中文：Shutdown 的托管封装（关闭并释放 GGML 后端资源）。
        public static void Shutdown()
        {
            TSGgml_Shutdown();
        }

        // 中文：InvalidateHostBuffer 的托管封装（使主机缓冲区缓存失效）。
        public static void InvalidateHostBuffer(IntPtr ptr)
        {
            if (ptr != IntPtr.Zero)
                TSGgml_InvalidateHostBuffer(ptr);
        }

        // 中文：SyncHostBuffer 的托管封装（同步主机缓冲区与设备内存）。
        public static void SyncHostBuffer(IntPtr ptr, long byteCount)
        {
            if (ptr == IntPtr.Zero || byteCount <= 0)
                return;

            CheckResult(TSGgml_SyncHostBuffer(ptr, byteCount), "sync_host_buffer");
        }

        /// <summary>
        /// Enable lazy synchronization on the Metal backend. When on, per-op kernels
        /// return immediately after committing their command buffer instead of
        /// blocking on `[cmd_buf waitUntilCompleted]`. Subsequent ops chain through
        /// the Metal command queue, and host-side reads (via
        /// TensorComputePrimitives.GetFloatPointer / GetHalfPointer, which call
        /// Storage.EnsureHostReadable) drain pending work on demand.
        ///
        /// This mirrors llama.cpp's Metal backend: ggml_metal_graph_compute commits
        /// its command buffer and returns; only an explicit ggml_backend_synchronize
        /// blocks. For TensorSharp's per-op driving model, lazy sync collapses the
        /// per-op `[cmd_buf waitUntilCompleted]` round-trip overhead (~30-100 µs each
        /// on M-series Macs) that dominates prefill on long prompts.
        /// </summary>
        // 中文：SetAsyncCompute 的托管封装（开启/关闭异步计算）。
        public static void SetAsyncCompute(bool enabled)
        {
            TSGgml_SetAsyncCompute(enabled ? 1 : 0);
        }

        /// <summary>True if async compute is currently enabled on the GGML backend.</summary>
        // 中文：GetAsyncCompute 的托管封装（查询异步计算是否启用）。
        public static bool GetAsyncCompute()
        {
            return TSGgml_GetAsyncCompute() != 0;
        }

        /// <summary>
        /// Drain any GPU work that was deferred under async compute. Cheap when no
        /// work is pending (single atomic exchange on the C++ side); when work is
        /// pending it does one ggml_backend_synchronize on the Metal command queue.
        /// </summary>
        // 中文：HostReadBarrier 的托管封装（排空挂起的 GPU 计算）。
        public static void HostReadBarrier()
        {
            TSGgml_HostReadBarrier();
        }

        // 中文：PreloadQuantizedWeight 的托管封装（预加载量化权重到设备缓存）。
        public static void PreloadQuantizedWeight(IntPtr cacheKey, IntPtr hostData, int ggmlType, long ne0, long ne1, long rawBytes)
        {
            if (cacheKey == IntPtr.Zero || hostData == IntPtr.Zero || rawBytes <= 0)
                throw new ArgumentException("PreloadQuantizedWeight requires valid cache key, host data, and size.");

            CheckResult(TSGgml_PreloadQuantizedWeight(cacheKey, hostData, ggmlType, ne0, ne1, rawBytes), "preload_quantized_weight");
        }

        /// <summary>
        /// Mark a host data pointer as eligible for the MoE expert offload LRU.
        /// After registration, the GGML native cache touches an LRU on lookup
        /// hits for this pointer and evicts from the LRU tail when residency
        /// exceeds the budget configured by <see cref="SetOffloadableBudget"/>.
        /// Registration is sticky; call <see cref="ClearOffloadableState"/> on
        /// model unload to reset.
        /// </summary>
        // 中文：RegisterOffloadable 的托管封装（注册可卸载缓冲区到 LRU）。
        public static void RegisterOffloadable(IntPtr key)
        {
            if (key == IntPtr.Zero)
                return;
            TSGgml_RegisterOffloadable(key);
        }

        /// <summary>
        /// Configure the byte ceiling for the offloadable cache LRU. Zero
        /// disables eviction (registered entries still participate in the LRU
        /// but nothing is freed).
        /// </summary>
        // 中文：SetOffloadableBudget 的托管封装（设置可卸载缓存的显存预算上限）。
        public static void SetOffloadableBudget(long bytes)
        {
            TSGgml_SetOffloadableBudget(bytes > 0 ? bytes : 0);
        }

        /// <summary>
        /// Reset offloadable registrations, LRU state, and byte accounting.
        /// Does not touch the underlying CachedHostBuffer entries.
        /// </summary>
        // 中文：ClearOffloadableState 的托管封装（清除可卸载缓存的注册与 LRU 状态）。
        public static void ClearOffloadableState()
        {
            TSGgml_ClearOffloadableState();
        }

        /// <summary>Bytes for one row along ne[0]; 0 if type/shape invalid.</summary>
        // 中文：RowSizeBytesOrZero 的托管封装（返回一行字节数，类型无效时为 0）。
        internal static long RowSizeBytesOrZero(int ggmlType, long ne0)
        {
            return (long)TSGgml_RowSize(ggmlType, ne0).ToUInt64();
        }

        // 中文：将 GGUF 量化张量（托管字节数组）反量化为 F32（固定内存后调用原生层）。
        internal static void DequantizeGgufTensorToFloat32(int ggmlType, byte[] src, int srcOffset, float[] dst, int dstOffset, long numElements)
        {
            if (numElements < 0 || numElements > int.MaxValue)
            {
                throw new ArgumentOutOfRangeException(nameof(numElements));
            }

            int n = (int)numElements;
            if (srcOffset < 0 || dstOffset < 0 || checked(dstOffset + n) > dst.Length || srcOffset > src.Length)
            {
                throw new ArgumentException("Invalid src/dst range for dequantization.");
            }

            GCHandle hSrc = GCHandle.Alloc(src, GCHandleType.Pinned);
            GCHandle hDst = GCHandle.Alloc(dst, GCHandleType.Pinned);
            try
            {
                IntPtr pSrc = IntPtr.Add(hSrc.AddrOfPinnedObject(), srcOffset);
                IntPtr pDst = IntPtr.Add(hDst.AddrOfPinnedObject(), dstOffset * sizeof(float));
                int r = TSGgml_DequantizeToF32(ggmlType, pSrc, numElements, pDst);
                if (r == -1)
                {
                    throw new ArgumentException("Dequantization failed (invalid arguments).");
                }

                if (r == -2)
                {
                    throw new NotSupportedException(
                        $"GGML tensor type {ggmlType} cannot be dequantized to float32.");
                }
            }
            finally
            {
                if (hSrc.IsAllocated)
                {
                    hSrc.Free();
                }

                if (hDst.IsAllocated)
                {
                    hDst.Free();
                }
            }
        }

        // 中文：将 GGUF 量化张量（原生指针）反量化为 F32。
        internal static void DequantizeGgufTensorToFloat32Native(int ggmlType, IntPtr src, IntPtr dst, long numElements)
        {
            if (src == IntPtr.Zero || dst == IntPtr.Zero || numElements < 0)
            {
                throw new ArgumentException("Invalid src/dst pointers or element count for dequantization.");
            }

            int r = TSGgml_DequantizeToF32(ggmlType, src, numElements, dst);
            if (r == -1)
            {
                throw new ArgumentException("Dequantization failed (invalid arguments).");
            }

            if (r == -2)
            {
                throw new NotSupportedException(
                    $"GGML tensor type {ggmlType} cannot be dequantized to float32.");
            }
        }

        // 中文：检查原生调用返回值，非成功时抛出带最近错误信息的异常。
        private static void CheckResult(int result, string opName)
        {
            if (result != 0)
            {
                return;
            }

            throw new InvalidOperationException($"Native GGML {opName} failed. {GetLastErrorMessage("Unknown native GGML error.")}");
        }

        // 中文：DllImport 解析器，按候选路径查找并加载原生 GgmlOps 库。
        private static IntPtr ImportResolver(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
        {
            if (!string.Equals(libraryName, DllName, StringComparison.Ordinal))
            {
                return IntPtr.Zero;
            }

            EnsureWindowsNativeDependencySearchPaths();

            foreach (string candidate in GetCandidatePaths(assembly))
            {
                if (File.Exists(candidate) && NativeLibrary.TryLoad(candidate, out IntPtr handle))
                {
                    return handle;
                }
            }

            return IntPtr.Zero;
        }

        // 中文：枚举原生库的候选搜索路径（输出目录、程序集目录与仓库构建目录）。
        private static IEnumerable<string> GetCandidatePaths(Assembly assembly)
        {
            string baseDirectory = AppContext.BaseDirectory;
            string assemblyDirectory = Path.GetDirectoryName(assembly.Location) ?? baseDirectory;

            foreach (string fileName in GetCandidateFileNames())
            {
                yield return Path.Combine(baseDirectory, fileName);
                yield return Path.Combine(assemblyDirectory, fileName);
            }

            foreach (string root in EnumerateRepoRoots(baseDirectory))
            {
                foreach (string fileName in GetCandidateFileNames())
                {
                    yield return Path.Combine(root, "TensorSharp.GGML.Native", "build", fileName);
                    yield return Path.Combine(root, "TensorSharp.GGML.Native", "build", "Release", fileName);
                    yield return Path.Combine(root, "TensorSharp.GGML.Native", "build-windows", fileName);
                    yield return Path.Combine(root, "TensorSharp.GGML.Native", "build-windows", "Release", fileName);
                }
            }
        }

        // 中文：从起始目录向上枚举所有疑似仓库根目录。
        private static IEnumerable<string> EnumerateRepoRoots(string startDirectory)
        {
            DirectoryInfo current = new DirectoryInfo(startDirectory);
            while (current != null)
            {
                if (IsRepoRoot(current.FullName))
                {
                    yield return current.FullName;
                }

                current = current.Parent;
            }
        }

        // 中文：按操作系统返回原生库文件名（dll/dylib/so）。
        private static IEnumerable<string> GetCandidateFileNames()
        {
            yield return OperatingSystem.IsWindows() ? "GgmlOps.dll" :
                OperatingSystem.IsMacOS() ? "libGgmlOps.dylib" :
                "libGgmlOps.so";
        }

        // 中文：判断当前平台（Windows/Linux）是否支持 CUDA 后端。
        private static bool IsCudaPlatformSupported()
        {
            return OperatingSystem.IsWindows() || OperatingSystem.IsLinux();
        }

        // 中文：在 Windows 上将 CUDA 等原生依赖目录加入 PATH（仅初始化一次）。
        private static void EnsureWindowsNativeDependencySearchPaths()
        {
            if (!OperatingSystem.IsWindows())
                return;

            if (Interlocked.Exchange(ref s_windowsDependencySearchPathsInitialized, 1) != 0)
                return;

            string currentPath = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
            var existingEntries = new HashSet<string>(
                currentPath.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries),
                StringComparer.OrdinalIgnoreCase);

            var additions = EnumerateWindowsNativeDependencyDirectories()
                .Where(path => Directory.Exists(path) && !existingEntries.Contains(path))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToArray();

            if (additions.Length == 0)
                return;

            Environment.SetEnvironmentVariable(
                "PATH",
                string.Join(Path.PathSeparator, additions.Concat(new[] { currentPath })));
        }

        // 中文：枚举 Windows 上 CUDA 工具包的 bin 依赖目录。
        private static IEnumerable<string> EnumerateWindowsNativeDependencyDirectories()
        {
            foreach (string variableName in new[] { "CUDA_PATH", "CUDA_HOME" })
            {
                string root = Environment.GetEnvironmentVariable(variableName);
                if (!string.IsNullOrWhiteSpace(root))
                    yield return Path.Combine(root, "bin");
            }

            string programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
            string cudaRoot = Path.Combine(programFiles, "NVIDIA GPU Computing Toolkit", "CUDA");
            if (!Directory.Exists(cudaRoot))
                yield break;

            foreach (string versionDir in Directory.EnumerateDirectories(cudaRoot, "v*").OrderByDescending(path => path))
                yield return Path.Combine(versionDir, "bin");
        }

        // 中文：根据解决方案文件或 .git 目录判断给定路径是否为仓库根目录。
        private static bool IsRepoRoot(string path)
        {
            string[] markers =
            {
                "TensorSharp.slnx",
                "TensorSharp.sln",
                "Seq2SeqSharp.sln",
            };

            return markers.Any(marker => File.Exists(Path.Combine(path, marker)))
                || Directory.Exists(Path.Combine(path, ".git"));
        }

        // 中文：读取原生层最近错误信息并转为托管字符串，为空时返回回退文本。
        private static string GetLastErrorMessage(string fallback)
        {
            IntPtr errPtr = TSGgml_GetLastError();
            string message = errPtr == IntPtr.Zero ? null : Marshal.PtrToStringAnsi(errPtr);
            return string.IsNullOrWhiteSpace(message) ? fallback : message;
        }

        // 中文：生成后端不可用时的提示信息（含 CUDA 重新构建建议）。
        private static string GetBackendAvailabilityHint(GgmlBackendType backendType)
        {
            string defaultMessage = "Build the native GGML bridge and ensure the requested GGML backend is available.";
            string backendMessage = GetLastErrorMessage(defaultMessage);

            if (backendType == GgmlBackendType.Cuda && IsCudaPlatformSupported())
            {
                string rebuildHint = OperatingSystem.IsWindows()
                    ? "Rebuild the native GGML bridge with CUDA enabled, for example: `powershell -ExecutionPolicy Bypass -File TensorSharp.GGML.Native/build-windows.ps1 --cuda` or `set TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON` before `dotnet build`."
                    : "Rebuild the native GGML bridge with CUDA enabled, for example: `bash TensorSharp.GGML.Native/build-linux.sh --cuda` or `TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON dotnet build`.";

                if (string.IsNullOrWhiteSpace(backendMessage))
                    return rebuildHint;

                if (backendMessage.Contains("not available in this build", StringComparison.OrdinalIgnoreCase))
                    return $"{backendMessage} {rebuildHint}";
            }

            return backendMessage;
        }
    }
}
