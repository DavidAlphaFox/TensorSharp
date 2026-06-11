namespace TensorSharp.Cuda
{
    public static class CudaFusedOps
    {
        // 中文：尝试执行 GQA（分组查询注意力）预填充阶段的融合注意力内核，支持滑动窗口与掩码。
        public static bool TryGqaPrefillAttention(
            Tensor result,
            Tensor query,
            Tensor key,
            Tensor value,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int seqLen,
            int kvLen,
            int maskStart,
            int windowSize,
            float scale)
        {
            return CudaKernelOps.TryGqaPrefillAttention(
                result,
                query,
                key,
                value,
                numQHeads,
                numKVHeads,
                headDim,
                seqLen,
                kvLen,
                maskStart,
                windowSize,
                scale);
        }

        // 中文：尝试对注意力分数执行带 attention sink 的融合 Softmax（含缩放、掩码与滑动窗口）。
        public static bool TryAttentionSoftmaxWithSinks(
            Tensor scores,
            Tensor sinks,
            int numHeads,
            int seqLen,
            int kvLen,
            int maskStart,
            int windowSize,
            float scale)
        {
            return CudaKernelOps.TryAttentionSoftmaxWithSinks(
                scores,
                sinks,
                numHeads,
                seqLen,
                kvLen,
                maskStart,
                windowSize,
                scale);
        }

        // 中文：尝试从源张量按列偏移和宽度切片出指定列区间到结果张量。
        public static bool TrySliceColumns(Tensor result, Tensor src, int colOffset, int width)
        {
            return CudaKernelOps.TrySliceColumns(result, src, colOffset, width);
        }

        // 中文：尝试执行 GQA 解码阶段的融合注意力内核，从 KV 缓存（支持环形缓存）读取键值。
        public static bool TryGqaDecodeAttention(
            Tensor result,
            Tensor query,
            Tensor keyCache,
            Tensor valueCache,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int attendStart,
            int attendLen,
            int cacheSize,
            bool circular,
            float scale)
        {
            return CudaKernelOps.TryGqaDecodeAttention(
                result,
                query,
                keyCache,
                valueCache,
                numQHeads,
                numKVHeads,
                headDim,
                attendStart,
                attendLen,
                cacheSize,
                circular,
                scale);
        }

        // 中文：尝试执行带 attention sink 的 GQA 解码阶段融合注意力内核（基于 KV 缓存）。
        public static bool TryGqaDecodeAttentionWithSinks(
            Tensor result,
            Tensor query,
            Tensor keyCache,
            Tensor valueCache,
            Tensor sinks,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int attendStart,
            int attendLen,
            int cacheSize,
            bool circular,
            float scale)
        {
            return CudaKernelOps.TryGqaDecodeAttentionWithSinks(
                result,
                query,
                keyCache,
                valueCache,
                sinks,
                numQHeads,
                numKVHeads,
                headDim,
                attendStart,
                attendLen,
                cacheSize,
                circular,
                scale);
        }

        // 中文：尝试执行带 attention sink 的 GQA 预填充阶段融合注意力内核（基于 KV 缓存）。
        public static bool TryGqaPrefillAttentionWithSinks(
            Tensor result,
            Tensor query,
            Tensor keyCache,
            Tensor valueCache,
            Tensor sinks,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int seqLen,
            int kvLen,
            int cacheSize,
            int maskStart,
            int windowSize,
            float scale)
        {
            return CudaKernelOps.TryGqaPrefillAttentionWithSinks(
                result,
                query,
                keyCache,
                valueCache,
                sinks,
                numQHeads,
                numKVHeads,
                headDim,
                seqLen,
                kvLen,
                cacheSize,
                maskStart,
                windowSize,
                scale);
        }

        // 中文：尝试将偏置向量按行广播加到张量的每一行上。
        public static bool TryAddBiasRows(Tensor tensor, Tensor bias)
        {
            return CudaKernelOps.TryAddBiasRows(tensor, bias);
        }

        // 中文：尝试将扁平布局张量重排为 head-first（按头优先）布局。
        public static bool TryFlatToHeadFirst(Tensor result, Tensor src, int numHeads, int seqLen, int headDim)
        {
            return CudaKernelOps.TryFlatToHeadFirst(result, src, numHeads, seqLen, headDim);
        }

        // 中文：尝试从融合的 QKV 张量按列偏移拆分出某一部分并重排为 head-first 布局。
        public static bool TrySplitQkvToHeadFirst(Tensor result, Tensor qkv, int colOffset, int numHeads, int seqLen, int headDim)
        {
            return CudaKernelOps.TrySplitQkvToHeadFirst(result, qkv, colOffset, numHeads, seqLen, headDim);
        }

        // 中文：尝试将 head-first 布局张量从起始位置写入 KV 缓存（支持环形缓存）。
        public static bool TryCopyHeadFirstToCache(Tensor cache, Tensor src, int startPos, int seqLen, int cacheSize, bool circular)
        {
            return CudaKernelOps.TryCopyHeadFirstToCache(cache, src, startPos, seqLen, cacheSize, circular);
        }

        // 中文：尝试从环形 KV 缓存中按起始位置收集 head-first 布局的连续序列。
        public static bool TryGatherCircularHeadFirst(Tensor result, Tensor cache, int startPos, int seqLen, int cacheSize)
        {
            return CudaKernelOps.TryGatherCircularHeadFirst(result, cache, startPos, seqLen, cacheSize);
        }

        // 中文：尝试将两个 head-first 布局张量拼接到结果张量。
        public static bool TryConcatHeadFirst(Tensor result, Tensor a, Tensor b)
        {
            return CudaKernelOps.TryConcatHeadFirst(result, a, b);
        }

        // 中文：尝试对 head-first 布局数据原地施加 NeoX 风格的旋转位置编码（RoPE）。
        public static bool TryNeoXRoPEHeadFirst(Tensor data, Tensor cosTable, Tensor sinTable, int numHeads, int seqLen, int headDim, int ropeHalf)
        {
            return CudaKernelOps.TryNeoXRoPEHeadFirst(data, cosTable, sinTable, numHeads, seqLen, headDim, ropeHalf);
        }

        // 中文：尝试对拼接的 gate/up 张量执行 GELU 门控乘法（GeGLU）融合内核。
        public static bool TryGELUMulSplit(Tensor result, Tensor gateUp, int halfDim)
        {
            return CudaKernelOps.TryGELUMulSplit(result, gateUp, halfDim);
        }

        // 中文：尝试对拼接的 gate/up 张量执行 OpenAI 风格的 SwiGLU 门控融合内核（含 alpha 与裁剪 limit）。
        public static bool TrySwiGluOaiSplit(Tensor result, Tensor gateUp, int halfDim, float alpha, float limit)
        {
            return CudaKernelOps.TrySwiGluOaiSplit(result, gateUp, halfDim, alpha, limit);
        }

        // 中文：尝试执行 Qwen3.5 门控 DeltaNet 的融合内核（含卷积状态与 SSM 状态更新）。
        public static bool TryQwen35GatedDeltaNetPacked(
            Tensor result,
            Tensor packed,
            Tensor convState,
            Tensor ssmState,
            Tensor convWeight,
            Tensor dtBias,
            Tensor aLog,
            Tensor ssmNorm,
            int seqLen,
            int packedDim,
            int qkvDim,
            int qkDim,
            int vDim,
            int numKHeads,
            int numVHeads,
            int headKDim,
            int headVDim,
            int convKernel,
            int convWriteIdx,
            float eps)
        {
            return CudaKernelOps.TryQwen35GatedDeltaNetPacked(
                result,
                packed,
                convState,
                ssmState,
                convWeight,
                dtBias,
                aLog,
                ssmNorm,
                seqLen,
                packedDim,
                qkvDim,
                qkDim,
                vDim,
                numKHeads,
                numVHeads,
                headKDim,
                headVDim,
                convKernel,
                convWriteIdx,
                eps);
        }

        // 中文：尝试将 Qwen3.5 门控 DeltaNet 的 QKV、z、beta、alpha 输入打包为统一的 packed 张量。
        public static bool TryQwen35GatedDeltaNetPackInputs(
            Tensor packed,
            Tensor qkv,
            Tensor z,
            Tensor beta,
            Tensor alpha,
            int seqLen,
            int qkvDim,
            int zDim,
            int numVHeads,
            int packedDim)
        {
            return CudaKernelOps.TryQwen35GatedDeltaNetPackInputs(
                packed,
                qkv,
                z,
                beta,
                alpha,
                seqLen,
                qkvDim,
                zDim,
                numVHeads,
                packedDim);
        }
    }
}
