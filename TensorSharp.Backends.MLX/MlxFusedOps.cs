using System;
using TensorSharp.Core;

namespace TensorSharp.MLX
{
    public static class MlxFusedOps
    {
        private static readonly int Qwen35GdnPackedMinSeqLen = ResolveQwen35GdnPackedMinSeqLen();
        private static readonly bool Qwen35GdnPackedKernelsEnabled =
            string.Equals(Environment.GetEnvironmentVariable("TS_MLX_QWEN35_GDN_PACKED_KERNELS"), "1", StringComparison.Ordinal);

        // 中文：从环境变量解析 Qwen3.5 GDN 打包内核启用的最小序列长度，缺省/非法时返回 64。
        private static int ResolveQwen35GdnPackedMinSeqLen()
        {
            string env = Environment.GetEnvironmentVariable("TS_MLX_QWEN35_GDN_PACKED_MIN_SEQ_LEN");
            if (!string.IsNullOrWhiteSpace(env) && int.TryParse(env, out int parsed) && parsed > 0)
                return parsed;

            return 64;
        }

        // 中文：在设备上同步求值（Eval）该 MLX 张量的计算图，非 MLX 张量或异常时返回 false。
        public static bool TryEvaluate(Tensor tensor)
        {
            if (tensor == null || tensor.Storage is not MlxStorage)
                return false;

            MlxNative.MlxArray view = default;
            try
            {
                view = GetView(tensor);
                MlxNative.Eval(view);
                return true;
            }
            catch
            {
                return false;
            }
            finally
            {
                MlxNative.FreeArray(view);
            }
        }

        // Device-side argmax along the last axis. Writes a single int32 to
        // the output tensor — typically a [1]-shaped tensor that holds the
        // predicted next token in pipelined greedy decode. The host can sync
        // it via Tensor.GetElementsAsInt(1)[0] without bringing the full
        // [vocab] logits tensor across the device→host boundary.
        // 中文：在设备端对 logits 沿最后一维做 argmax，把预测的下一个 token 写入 int32 结果张量，避免整段 vocab 跨设备拷回主机。
        public static bool TryArgMaxLastAxis(Tensor result, Tensor logits)
        {
            if (result == null || logits == null) return false;
            if (result.Storage is not MlxStorage || logits.Storage is not MlxStorage)
                return false;
            if (result.ElementType != DType.Int32 || logits.ElementType != DType.Float32)
                return false;
            if (logits.DimensionCount < 1) return false;

            MlxNative.MlxArray logitsView = default;
            MlxNative.MlxArray argmax = default;
            MlxNative.MlxArray casted = default;
            try
            {
                logitsView = GetView(logits);
                argmax = MlxNative.ArgMaxAxis(logitsView, logits.DimensionCount - 1, keepDims: false);
                // mlx_argmax returns uint32; recast to int32 to match the
                // C# result tensor dtype.
                casted = MlxNative.Astype(argmax, DType.Int32);
                MlxNative.FreeArray(argmax);
                argmax = default;
                SetDeviceResult(result, casted);
                casted = default;
                return true;
            }
            catch
            {
                return false;
            }
            finally
            {
                MlxNative.FreeArray(logitsView);
                MlxNative.FreeArray(argmax);
                MlxNative.FreeArray(casted);
            }
        }

        // Device-side MoE router top-K + (normalized) softmax. Writes
        //   topIndices[K] = indices of the K largest logits along axis -1
        //   routeWeights[1, K] = softmax(top-K logits) — normalized within
        //                          the K selected experts
        // Used by Qwen3.5 MoE decode to skip the per-MoE-layer host sync
        // on routerLogits. Eliminates ~60 GetFloatPtr round trips per
        // decode token (Qwen3.6-35B-A3B has 60 MoE layers).
        //
        // Requires: routerLogits shape [1, numExperts], topIndices [K]
        // int32, routeWeights [1, K] float32, all MLX-backed.
        // 中文：在设备端完成 MoE 路由的 top-K 选取与归一化 softmax，输出选中专家索引及其权重，省去每 MoE 层的主机同步。
        public static bool TryMoeRouterTopKSoftmax(
            Tensor routerLogits,
            Tensor topIndices,
            Tensor routeWeights)
        {
            if (routerLogits == null || topIndices == null || routeWeights == null)
                return false;
            if (routerLogits.Storage is not MlxStorage
                || topIndices.Storage is not MlxStorage
                || routeWeights.Storage is not MlxStorage)
                return false;
            if (routerLogits.ElementType != DType.Float32
                || topIndices.ElementType != DType.Int32
                || routeWeights.ElementType != DType.Float32)
                return false;
            if (routerLogits.DimensionCount != 2 || routerLogits.Sizes[0] != 1)
                return false;
            if (topIndices.DimensionCount != 1) return false;
            if (routeWeights.DimensionCount != 2 || routeWeights.Sizes[0] != 1)
                return false;
            int K = checked((int)topIndices.Sizes[0]);
            int numExperts = checked((int)routerLogits.Sizes[1]);
            if (K <= 0 || K >= numExperts) return false;
            if (routeWeights.Sizes[1] != K) return false;

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxNative.MlxArray logitsView = default;
                MlxNative.MlxArray neg = default;
                MlxNative.MlxArray partIdx = default;
                MlxNative.MlxArray topIdx2d = default;
                MlxNative.MlxArray topIdx1d = default;
                MlxNative.MlxArray topIdx1dInt = default;
                MlxNative.MlxArray topLogits = default;
                MlxNative.MlxArray softmaxed = default;
                try
                {
                    logitsView = GetView(routerLogits);
                    // Negate so argpartition's "k smallest" gives top-K largest.
                    neg = MlxNative.Unary(MlxNative.MlxUnaryOp.Neg, logitsView);
                    // [1, numExperts] uint32 — first K positions are top-K largest.
                    partIdx = MlxNative.ArgPartitionAxis(neg, K - 1, axis: 1);
                    // Slice first K → [1, K]
                    topIdx2d = MlxNative.Slice(
                        partIdx,
                        starts: new[] { 0, 0 },
                        stops: new[] { 1, K },
                        strides: new[] { 1, 1 });
                    // Reshape to [K] for the batched MoE kernel.
                    topIdx1d = MlxNative.Reshape(topIdx2d, new[] { K });
                    // Cast uint32 → int32 (the batched kernel expects int32 indices).
                    topIdx1dInt = MlxNative.Astype(topIdx1d, DType.Int32);
                    // Gather the top-K logits and softmax them (normalized within
                    // the K selected experts — matches the Qwen3.5 _normTopKProb
                    // host-side semantics).
                    topLogits = MlxNative.TakeAxis(logitsView, topIdx1dInt, axis: 1);
                    softmaxed = MlxNative.SoftmaxLastAxis(topLogits);

                    SetDeviceResult(topIndices, topIdx1dInt);
                    topIdx1dInt = default;
                    SetDeviceResult(routeWeights, softmaxed);
                    softmaxed = default;
                    return true;
                }
                catch
                {
                    return false;
                }
                finally
                {
                    MlxNative.FreeArray(logitsView);
                    MlxNative.FreeArray(neg);
                    MlxNative.FreeArray(partIdx);
                    MlxNative.FreeArray(topIdx2d);
                    MlxNative.FreeArray(topIdx1d);
                    MlxNative.FreeArray(topIdx1dInt);
                    MlxNative.FreeArray(topLogits);
                    MlxNative.FreeArray(softmaxed);
                }
            });
        }

        // In-place fused output += scalar * src for MLX tensors. Replaces
        // the two-kernel mulv+addt chain with one fused Metal kernel
        // (via mlx_compile). Used by the MoE decode accumulator where
        // each saved kernel × 8 experts × 60 layers adds up.
        // Returns false when the caller should fall back to its own
        // eager path (e.g. non-MLX storage, compile disabled, errors).
        // 中文：以单个融合 Metal 内核就地执行 output += scalar * src，校验通过后委派给 worker 线程执行。
        public static bool TryAddScaledInPlace(Tensor output, Tensor src, float scalar)
        {
            if (MlxCompiledOps.Disabled) return false;
            if (output == null || src == null) return false;
            if (output.Storage is not MlxStorage outStorage || src.Storage is not MlxStorage)
                return false;
            if (output.ElementType != DType.Float32 || src.ElementType != DType.Float32)
                return false;
            if (scalar == 0f) return true;

            return MlxWorker.Shared.Invoke(() => RunAddScaledInPlace(output, src, scalar));
        }

        // 中文：TryAddScaledInPlace 的实际执行体，构建视图与标量、调用融合 AddScaled 内核并把结果写回 output。
        private static bool RunAddScaledInPlace(Tensor output, Tensor src, float scalar)
        {
            MlxNative.MlxArray outView = default;
            MlxNative.MlxArray srcView = default;
            MlxNative.MlxArray scalarArray = default;
            MlxNative.MlxArray result = default;
            try
            {
                outView = GetView(output);
                srcView = GetView(src);
                scalarArray = MlxNative.NewScalar(scalar);
                result = MlxCompiledOps.AddScaled(outView, srcView, scalarArray);
                SetDeviceResult(output, result);
                result = default;
                return true;
            }
            catch
            {
                return false;
            }
            finally
            {
                MlxNative.FreeArray(outView);
                MlxNative.FreeArray(srcView);
                MlxNative.FreeArray(scalarArray);
                MlxNative.FreeArray(result);
            }
        }

        // Async variant: schedules graph execution on Metal and returns
        // immediately. The next host read drains the queue. Use this at
        // layer boundaries so Metal command-buffer issue overlaps with
        // completion of earlier layers — mirrors ollama mlxrunner's
        // mlx.AsyncEval at each forward step.
        // 中文：异步调度该 MLX 张量的图执行后立即返回，用于在层边界让命令缓冲发射与前序层完成重叠。
        public static bool TryAsyncEvaluate(Tensor tensor)
        {
            if (tensor == null || tensor.Storage is not MlxStorage)
                return false;

            MlxNative.MlxArray view = default;
            try
            {
                view = GetView(tensor);
                MlxNative.AsyncEval(view);
                return true;
            }
            catch
            {
                return false;
            }
            finally
            {
                MlxNative.FreeArray(view);
            }
        }

        // 中文：将拥有完整存储的连续 MLX 张量物化为紧致连续数组并同步求值后写回，否则返回 false。
        public static bool TryMaterialize(Tensor tensor)
        {
            if (tensor == null
                || tensor.Storage is not MlxStorage
                || !tensor.IsContiguous()
                || tensor.StorageOffset != 0
                || tensor.ElementCount() != tensor.Storage.ElementCount)
            {
                return false;
            }

            MlxNative.MlxArray view = default;
            MlxNative.MlxArray contiguous = default;
            try
            {
                view = GetView(tensor);
                contiguous = MlxNative.Contiguous(view);
                MlxNative.Eval(contiguous);
                SetDeviceResult(tensor, contiguous);
                contiguous = default;
                return true;
            }
            catch
            {
                return false;
            }
            finally
            {
                MlxNative.FreeArray(view);
                MlxNative.FreeArray(contiguous);
            }
        }

        // 中文：按 int32 索引向量在设备端从 src 收集行（take + 连续化）写入 result，实现行级 gather。
        public static bool TryGatherRows(Tensor result, Tensor src, Tensor indices)
        {
            if (!CanUseResult(result)
                || !CanUseResult(src)
                || !CanUseInt32Vector(indices)
                || result.DimensionCount != 2
                || src.DimensionCount != 2
                || indices.DimensionCount != 1
                || result.Sizes[0] != indices.Sizes[0]
                || result.Sizes[1] != src.Sizes[1])
            {
                return false;
            }

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxNative.MlxArray srcView = default;
                MlxNative.MlxArray indicesView = default;
                MlxNative.MlxArray gathered = default;
                MlxNative.MlxArray contiguous = default;
                try
                {
                    srcView = GetView(src);
                    indicesView = GetView(indices);
                    gathered = MlxNative.TakeAxis(srcView, indicesView, 0);
                    contiguous = MlxNative.Contiguous(gathered);
                    SetDeviceResult(result, contiguous);
                    contiguous = default;
                    return true;
                }
                catch
                {
                    return false;
                }
                finally
                {
                    MlxNative.FreeArray(srcView);
                    MlxNative.FreeArray(indicesView);
                    MlxNative.FreeArray(gathered);
                    MlxNative.FreeArray(contiguous);
                }
            });
        }

        // 中文：按已排序去重的索引把加权后的 rows 散加（scatter-add）回 output 对应行，用于 MoE 专家结果合并而避免主机散射。
        public static bool TryScatterAddWeightedRows(Tensor output, Tensor rows, Tensor indices, Tensor weights)
        {
            // indices must be sorted and unique. Qwen35's per-expert prefill
            // grouping emits rows in token order, so this avoids a host scatter
            // while preserving all untouched output rows on the device.
            if (!CanUseResult(output)
                || !CanUseResult(rows)
                || !CanUseInt32Vector(indices)
                || !CanUseResult(weights)
                || output.DimensionCount != 2
                || rows.DimensionCount != 2
                || indices.DimensionCount != 1
                || weights.DimensionCount != 1
                || rows.Sizes[0] != indices.Sizes[0]
                || weights.Sizes[0] != indices.Sizes[0]
                || rows.Sizes[1] != output.Sizes[1])
            {
                return false;
            }

            int seqLen = checked((int)output.Sizes[0]);
            int batchSize = checked((int)rows.Sizes[0]);
            int hiddenDim = checked((int)output.Sizes[1]);
            if (batchSize <= 0)
                return true;

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxNative.MlxArray outputView = default;
                MlxNative.MlxArray rowsView = default;
                MlxNative.MlxArray indicesView = default;
                MlxNative.MlxArray weightsView = default;
                MlxNative.MlxArray scattered = default;
                try
                {
                    outputView = GetView(output);
                    rowsView = GetView(rows);
                    indicesView = GetView(indices);
                    weightsView = GetView(weights);
                    scattered = MlxNative.ScatterAddWeightedRows(outputView, rowsView, indicesView, weightsView, seqLen, batchSize, hiddenDim);
                    SetDeviceResult(output, scattered);
                    scattered = default;
                    return true;
                }
                catch
                {
                    return false;
                }
                finally
                {
                    MlxNative.FreeArray(outputView);
                    MlxNative.FreeArray(rowsView);
                    MlxNative.FreeArray(indicesView);
                    MlxNative.FreeArray(weightsView);
                    MlxNative.FreeArray(scattered);
                }
            });
        }

        /// <summary>
        /// Fused (residual += input; normedOut = RmsNorm(updated residual, normWeight)).
        /// On success, <paramref name="residual"/> is updated in place to
        /// (residual + input) and <paramref name="normedOut"/> is filled with
        /// RmsNorm of the updated residual using <paramref name="normWeight"/>.
        /// Saves one MLX dispatch vs the equivalent Ops.Add + Ops.RMSNorm pair
        /// in the pre-norm transformer block pattern.
        /// </summary>
        // 中文：融合执行 residual += input 并将其 RMSNorm 结果写入 normedOut，省去一次 MLX 调度（pre-norm 块模式）。
        public static bool TryAddRmsNorm(Tensor residual, Tensor input, Tensor normWeight, float eps, Tensor normedOut)
        {
            if (!CanUseResult(residual)
                || !CanUseResult(input)
                || !CanUseResult(normWeight)
                || !CanUseResult(normedOut)
                || residual.DimensionCount != 2
                || input.DimensionCount != 2
                || normWeight.DimensionCount != 1
                || normedOut.DimensionCount != 2
                || residual.Sizes[0] != input.Sizes[0]
                || residual.Sizes[1] != input.Sizes[1]
                || residual.Sizes[0] != normedOut.Sizes[0]
                || residual.Sizes[1] != normedOut.Sizes[1]
                || normWeight.Sizes[0] != residual.Sizes[1])
            {
                return false;
            }

            int rows = checked((int)residual.Sizes[0]);
            int hiddenDim = checked((int)residual.Sizes[1]);
            return MlxWorker.Shared.Invoke(() =>
            {
                MlxNative.MlxArray residualView = default;
                MlxNative.MlxArray inputView = default;
                MlxNative.MlxArray weightView = default;
                MlxNative.MlxArray updated = default;
                MlxNative.MlxArray normed = default;
                try
                {
                    residualView = GetView(residual);
                    inputView = GetView(input);
                    weightView = GetView(normWeight);
                    (updated, normed) = MlxNative.AddRmsNorm(residualView, inputView, weightView, eps, rows, hiddenDim);
                    SetDeviceResult(residual, updated);
                    updated = default;
                    SetDeviceResult(normedOut, normed);
                    normed = default;
                    return true;
                }
                catch
                {
                    return false;
                }
                finally
                {
                    MlxNative.FreeArray(residualView);
                    MlxNative.FreeArray(inputView);
                    MlxNative.FreeArray(weightView);
                    MlxNative.FreeArray(updated);
                    MlxNative.FreeArray(normed);
                }
            });
        }

        // 中文：融合执行 (input 的 RMSNorm) 与残差相加并就地写回 residual。
        public static bool TryRmsNormAddInPlace(Tensor residual, Tensor input, Tensor normWeight, float eps)
        {
            if (!CanUseResult(residual)
                || !CanUseResult(input)
                || !CanUseResult(normWeight)
                || residual.DimensionCount != 2
                || input.DimensionCount != 2
                || normWeight.DimensionCount != 1
                || residual.Sizes[0] != input.Sizes[0]
                || residual.Sizes[1] != input.Sizes[1]
                || normWeight.Sizes[0] != residual.Sizes[1])
            {
                return false;
            }

            int rows = checked((int)residual.Sizes[0]);
            int hiddenDim = checked((int)residual.Sizes[1]);
            return MlxWorker.Shared.Invoke(() =>
            {
                MlxNative.MlxArray residualView = default;
                MlxNative.MlxArray inputView = default;
                MlxNative.MlxArray weightView = default;
                MlxNative.MlxArray output = default;
                try
                {
                    residualView = GetView(residual);
                    inputView = GetView(input);
                    weightView = GetView(normWeight);
                    output = MlxNative.RmsNormAdd(residualView, inputView, weightView, eps, rows, hiddenDim);
                    SetDeviceResult(residual, output);
                    output = default;
                    return true;
                }
                catch
                {
                    return false;
                }
                finally
                {
                    MlxNative.FreeArray(residualView);
                    MlxNative.FreeArray(inputView);
                    MlxNative.FreeArray(weightView);
                    MlxNative.FreeArray(output);
                }
            });
        }

        // 中文：对拼接的 gateUp 张量做 GeGLU 融合：拆出 gate/up 两半，gate 过 GELU 后与 up 逐元素相乘写入 result。
        public static bool TryGeluMulSplit(Tensor result, Tensor gateUp, int halfDim)
        {
            if (!CanUseResult(result)
                || !CanUseResult(gateUp)
                || gateUp.DimensionCount != 2
                || result.DimensionCount != 2
                || halfDim <= 0
                || gateUp.Sizes[1] != 2L * halfDim
                || result.Sizes[0] != gateUp.Sizes[0]
                || result.Sizes[1] != halfDim)
            {
                return false;
            }

            int rows = checked((int)gateUp.Sizes[0]);
            MlxNative.MlxArray gateUpView = default;
            MlxNative.MlxArray output = default;
            try
            {
                gateUpView = GetView(gateUp);
                output = MlxNative.GeluMulSplit(gateUpView, rows, halfDim);
                SetDeviceResult(result, output);
                output = default;
                return true;
            }
            catch
            {
                return false;
            }
            finally
            {
                MlxNative.FreeArray(gateUpView);
                MlxNative.FreeArray(output);
            }
        }

        /// <summary>
        /// Update <c>cache[:, startPos : startPos + seqLen, :] = src</c> in a
        /// single MLX <c>slice_update</c> dispatch. Replaces the per-head
        /// <c>Ops.Copy(cacheHead.Narrow(...), srcHead)</c> loop in
        /// <c>ModelBase.TryCopyHeadFirstToCacheMlx</c>, which costs
        /// <c>kvHeads × (AsStrided + Contiguous + Reshape + SliceUpdate) = 8+
        /// MLX ops per KV-cache write per layer</c>. Decode dispatches this
        /// for K and V on every layer (84 ops/token × 42 layers = ~3.5K
        /// dispatches saved per token at typical Gemma 4 dims).
        ///
        /// <paramref name="cache"/> must be a 3D MLX-backed float tensor
        /// <c>[heads, cacheLen, headDim]</c> that owns its storage (no
        /// offset/strided view). <paramref name="src"/> must be a 3D
        /// MLX-backed float tensor <c>[heads, seqLen, headDim]</c>.
        /// </summary>
        // 中文：用单次 slice_update 把 src 写入 KV 缓存的 [:, startPos:startPos+seqLen, :] 区间，替代逐头拷贝循环。
        public static bool TryWriteKvCacheBlock(Tensor cache, Tensor src, int startPos, int seqLen)
        {
            if (cache == null || src == null)
                return false;
            if (cache.Storage is not MlxStorage cacheStorage || src.Storage is not MlxStorage)
                return false;
            if (cache.ElementType != DType.Float32 && cache.ElementType != DType.Float16)
                return false;
            if (src.ElementType != DType.Float32 && src.ElementType != DType.Float16)
                return false;
            if (cache.DimensionCount != 3 || src.DimensionCount != 3)
                return false;
            if (cache.Sizes[0] != src.Sizes[0] || cache.Sizes[2] != src.Sizes[2])
                return false;
            if (src.Sizes[1] != seqLen)
                return false;
            if (startPos < 0 || (long)startPos + seqLen > cache.Sizes[1])
                return false;
            // Storage must own the whole cache so we can ReplaceDeviceArray
            // with the slice_update result. Sub-views aren't supported here
            // (the caller would have to plumb its own bookkeeping for that
            // case, and KV caches don't use sub-views in practice).
            if (cache.StorageOffset != 0 || cache.Storage.ElementCount != cache.ElementCount())
                return false;

            int heads = (int)cache.Sizes[0];
            int headDim = (int)cache.Sizes[2];

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxNative.MlxArray cacheView = default;
                MlxNative.MlxArray srcView = default;
                MlxNative.MlxArray casted = default;
                MlxNative.MlxArray updated = default;
                try
                {
                    cacheView = GetView(cache);
                    srcView = GetView(src);
                    MlxNative.MlxArray slice = srcView;
                    if (src.ElementType != cache.ElementType)
                    {
                        casted = MlxNative.Astype(srcView, cache.ElementType);
                        slice = casted;
                    }
                    int[] starts = { 0, startPos, 0 };
                    int[] stops = { heads, startPos + seqLen, headDim };
                    int[] strides = { 1, 1, 1 };
                    updated = MlxNative.SliceUpdateMulti(cacheView, slice, starts, stops, strides);
                    cacheStorage.ReplaceDeviceArray(updated);
                    updated = default;
                    return true;
                }
                catch
                {
                    return false;
                }
                finally
                {
                    MlxNative.FreeArray(cacheView);
                    MlxNative.FreeArray(srcView);
                    MlxNative.FreeArray(casted);
                    MlxNative.FreeArray(updated);
                }
            });
        }

        /// <summary>
        /// Fused Gemma 4 decode-step QKV preprocessing. Reads the
        /// post-matmul <paramref name="qkv"/> [1, q_dim + k_dim + v_dim],
        /// then in one Metal kernel:
        ///   * splits Q / K / V,
        ///   * applies weighted RMSNorm to each head of Q and K,
        ///   * applies unweighted RMSNorm to each head of V,
        ///   * applies NeoX-style RoPE to Q and K (cos/sin already evaluated
        ///     at the target position).
        /// Outputs Q flat <c>[1, NumHeads * HeadDim]</c> and K / V head-first
        /// <c>[NumKVHeads, 1, HeadDim]</c> (the layout the cache slice_update
        /// expects). Replaces 5 separate MLX dispatches per layer.
        /// </summary>
        // 中文：Gemma 4 解码步 QKV 预处理融合内核：单核内拆分 QKV、对各头做 RMSNorm 并对 Q/K 施加 NeoX RoPE，输出指定布局。
        public static bool TryGemma4QkvPreprocessDecode(
            Tensor qOut,
            Tensor kOut,
            Tensor vOut,
            Tensor qkv,
            Tensor qNormWeight,
            Tensor kNormWeight,
            Tensor cosTable,
            Tensor sinTable,
            int numHeads,
            int numKVHeads,
            int headDim,
            int rotHalf,
            float eps)
        {
            if (qOut == null || kOut == null || vOut == null || qkv == null
                || qNormWeight == null || kNormWeight == null
                || cosTable == null || sinTable == null)
                return false;
            if (qOut.Storage is not MlxStorage || kOut.Storage is not MlxStorage || vOut.Storage is not MlxStorage)
                return false;
            if (qkv.Storage is not MlxStorage || qNormWeight.Storage is not MlxStorage
                || kNormWeight.Storage is not MlxStorage
                || cosTable.Storage is not MlxStorage || sinTable.Storage is not MlxStorage)
                return false;
            if (qOut.ElementType != DType.Float32 || kOut.ElementType != DType.Float32 || vOut.ElementType != DType.Float32
                || qkv.ElementType != DType.Float32
                || qNormWeight.ElementType != DType.Float32 || kNormWeight.ElementType != DType.Float32
                || cosTable.ElementType != DType.Float32 || sinTable.ElementType != DType.Float32)
                return false;
            if (numHeads <= 0 || numKVHeads <= 0 || numHeads % numKVHeads != 0
                || headDim <= 0 || headDim > 512
                || (headDim & (headDim - 1)) != 0
                || rotHalf <= 0 || rotHalf * 2 > headDim)
                return false;
            if (qkv.ElementCount() != (long)(numHeads + 2 * numKVHeads) * headDim)
                return false;
            if (qNormWeight.ElementCount() != headDim || kNormWeight.ElementCount() != headDim)
                return false;
            if (cosTable.ElementCount() != rotHalf || sinTable.ElementCount() != rotHalf)
                return false;
            if (qOut.ElementCount() != (long)numHeads * headDim) return false;
            if (kOut.ElementCount() != (long)numKVHeads * headDim) return false;
            if (vOut.ElementCount() != (long)numKVHeads * headDim) return false;

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxNative.MlxArray qkvView = default;
                MlxNative.MlxArray qNormView = default;
                MlxNative.MlxArray kNormView = default;
                MlxNative.MlxArray cosView = default;
                MlxNative.MlxArray sinView = default;
                MlxNative.MlxArray q = default;
                MlxNative.MlxArray k = default;
                MlxNative.MlxArray v = default;
                try
                {
                    qkvView = GetView(qkv);
                    qNormView = GetView(qNormWeight);
                    kNormView = GetView(kNormWeight);
                    cosView = GetView(cosTable);
                    sinView = GetView(sinTable);

                    MlxNative.Gemma4QkvPreprocessDecode(
                        qkvView, qNormView, kNormView, cosView, sinView,
                        numHeads, numKVHeads, headDim, rotHalf, eps,
                        out q, out k, out v);

                    SetDeviceResult(qOut, q); q = default;
                    SetDeviceResult(kOut, k); k = default;
                    SetDeviceResult(vOut, v); v = default;
                    return true;
                }
                catch
                {
                    return false;
                }
                finally
                {
                    MlxNative.FreeArray(qkvView);
                    MlxNative.FreeArray(qNormView);
                    MlxNative.FreeArray(kNormView);
                    MlxNative.FreeArray(cosView);
                    MlxNative.FreeArray(sinView);
                    MlxNative.FreeArray(q);
                    MlxNative.FreeArray(k);
                    MlxNative.FreeArray(v);
                }
            });
        }

        // 中文：把扁平的 [seqLen, numHeads*headDim] 张量（可带列偏移）重排为 head-first 的 [numHeads, seqLen, headDim]。
        public static bool TryFlatToHeadFirst(Tensor result, Tensor input, int numHeads, int seqLen, int headDim, int colOffset = 0)
        {
            if (!CanUseResult(result)
                || !CanUseResult(input)
                || input.DimensionCount != 2
                || result.DimensionCount != 3
                || numHeads <= 0
                || seqLen <= 0
                || headDim <= 0
                || colOffset < 0
                || input.Sizes[0] != seqLen
                || input.Sizes[1] > int.MaxValue
                || colOffset + (long)numHeads * headDim > input.Sizes[1]
                || result.Sizes[0] != numHeads
                || result.Sizes[1] != seqLen
                || result.Sizes[2] != headDim)
            {
                return false;
            }

            int sourceStride = checked((int)input.Sizes[1]);
            return MlxWorker.Shared.Invoke(() =>
            {
                MlxNative.MlxArray inputView = default;
                MlxNative.MlxArray output = default;
                try
                {
                    inputView = GetView(input);
                    output = MlxNative.FlatToHeadFirst(inputView, seqLen, numHeads, headDim, sourceStride, colOffset);
                    SetDeviceResult(result, output);
                    output = default;
                    return true;
                }
                catch
                {
                    return false;
                }
                finally
                {
                    MlxNative.FreeArray(inputView);
                    MlxNative.FreeArray(output);
                }
            });
        }

        // 中文：对扁平布局 [seqLen, numHeads*headDim] 的数据就地施加 NeoX 风格 RoPE 旋转位置编码。
        public static bool TryNeoXRoPEFlatInPlace(Tensor data, Tensor cosTable, Tensor sinTable,
            int numHeads, int seqLen, int headDim, int rotHalf)
        {
            if (!CanUseResult(data)
                || !CanUseResult(cosTable)
                || !CanUseResult(sinTable)
                || data.DimensionCount != 2
                || cosTable.DimensionCount != 1
                || sinTable.DimensionCount != 1
                || numHeads <= 0
                || seqLen <= 0
                || headDim <= 0
                || rotHalf <= 0
                || rotHalf * 2 > headDim
                || data.Sizes[0] != seqLen
                || data.Sizes[1] != (long)numHeads * headDim
                || cosTable.Sizes[0] != (long)seqLen * rotHalf
                || sinTable.Sizes[0] != (long)seqLen * rotHalf)
            {
                return false;
            }

            return TryNeoXRoPEInPlace(data, cosTable, sinTable, numHeads, seqLen, headDim, rotHalf, headFirst: false);
        }

        // 中文：对 head-first 布局 [numHeads, seqLen, headDim] 的数据就地施加 NeoX 风格 RoPE 旋转位置编码。
        public static bool TryNeoXRoPEHeadFirstInPlace(Tensor data, Tensor cosTable, Tensor sinTable,
            int numHeads, int seqLen, int headDim, int rotHalf)
        {
            if (!CanUseResult(data)
                || !CanUseResult(cosTable)
                || !CanUseResult(sinTable)
                || data.DimensionCount != 3
                || cosTable.DimensionCount != 1
                || sinTable.DimensionCount != 1
                || numHeads <= 0
                || seqLen <= 0
                || headDim <= 0
                || rotHalf <= 0
                || rotHalf * 2 > headDim
                || data.Sizes[0] != numHeads
                || data.Sizes[1] != seqLen
                || data.Sizes[2] != headDim
                || cosTable.Sizes[0] != (long)seqLen * rotHalf
                || sinTable.Sizes[0] != (long)seqLen * rotHalf)
            {
                return false;
            }

            return TryNeoXRoPEInPlace(data, cosTable, sinTable, numHeads, seqLen, headDim, rotHalf, headFirst: true);
        }

        // 中文：NeoX RoPE 的共用执行体，按 headFirst 标志调用底层内核并把旋转后的结果写回 data。
        private static bool TryNeoXRoPEInPlace(Tensor data, Tensor cosTable, Tensor sinTable,
            int numHeads, int seqLen, int headDim, int rotHalf, bool headFirst)
        {
            return MlxWorker.Shared.Invoke(() =>
            {
                MlxNative.MlxArray dataView = default;
                MlxNative.MlxArray cosView = default;
                MlxNative.MlxArray sinView = default;
                MlxNative.MlxArray output = default;
                try
                {
                    dataView = GetView(data);
                    cosView = GetView(cosTable);
                    sinView = GetView(sinTable);
                    output = MlxNative.NeoXRoPE(dataView, cosView, sinView, numHeads, seqLen, headDim, rotHalf, headFirst);
                    SetDeviceResult(data, output);
                    output = default;
                    return true;
                }
                catch
                {
                    return false;
                }
                finally
                {
                    MlxNative.FreeArray(dataView);
                    MlxNative.FreeArray(cosView);
                    MlxNative.FreeArray(sinView);
                    MlxNative.FreeArray(output);
                }
            });
        }

        public sealed class AttentionKvCache : IDisposable
        {
            private MlxNative.MlxArray kCache;
            private MlxNative.MlxArray vCache;
            private int length;

            public int Length => length;

            // 中文：物化并异步求值缓存中的 K/V 数组，限制图深度，返回是否有数组被求值。
            public bool TryEvaluateState()
            {
                try
                {
                    bool evaluated = Materialize(ref kCache);
                    evaluated |= Materialize(ref vCache);
                    return evaluated;
                }
                catch
                {
                    return false;
                }
            }

            // 中文：释放 K/V 缓存数组并把缓存长度清零，重置注意力 KV 缓存状态。
            public void Reset()
            {
                MlxNative.FreeArray(kCache);
                MlxNative.FreeArray(vCache);
                kCache = default;
                vCache = default;
                length = 0;
            }

            // 中文：headDim=256 的注意力：将本步 K/V 追加进缓存并对 Q 运行融合注意力内核，结果写入 result 并推进缓存长度。
            public bool TryAttentionHeadDim256(
                Tensor result,
                Tensor qHeads,
                Tensor kHeads,
                Tensor vHeads,
                int numHeads,
                int numKVHeads,
                int seqLen,
                int startPos,
                bool causal,
                float scale = 0.0625f)
            {
                if (result == null || qHeads == null || kHeads == null || vHeads == null)
                    return false;
                if (seqLen <= 0
                    || startPos < 0
                    || startPos != length
                    || numHeads <= 0
                    || numKVHeads <= 0
                    || numHeads % numKVHeads != 0
                    || !CanUseResult(result)
                    || !CanUseAttentionTensor(qHeads)
                    || !CanUseAttentionTensor(kHeads)
                    || !CanUseAttentionTensor(vHeads)
                    || qHeads.ElementType != DType.Float32
                    || kHeads.ElementType != DType.Float32
                    || vHeads.ElementType != DType.Float32
                    || qHeads.DimensionCount != 3
                    || kHeads.DimensionCount != 3
                    || vHeads.DimensionCount != 3
                    || qHeads.Sizes[0] != numHeads
                    || qHeads.Sizes[1] != seqLen
                    || qHeads.Sizes[2] != 256
                    || kHeads.Sizes[0] != numKVHeads
                    || kHeads.Sizes[1] != seqLen
                    || kHeads.Sizes[2] != 256
                    || vHeads.Sizes[0] != numKVHeads
                    || vHeads.Sizes[1] != seqLen
                    || vHeads.Sizes[2] != 256
                    || result.DimensionCount != 2
                    || result.Sizes[0] != seqLen
                    || result.Sizes[1] != (long)numHeads * 256)
                {
                    return false;
                }

                return MlxWorker.Shared.Invoke(() =>
                {
                    MlxNative.MlxArray qView = default;
                    MlxNative.MlxArray qCompact = default;
                    MlxNative.MlxArray nextK = default;
                    MlxNative.MlxArray nextV = default;
                    MlxNative.MlxArray attention = default;
                    try
                    {
                        qView = GetView(qHeads);
                        qCompact = qHeads.IsContiguous() ? qView : MlxNative.Contiguous(qView);
                        if (qHeads.IsContiguous())
                            qView = default;

                        if (!TryBuildUpdatedCache(kHeads, vHeads, out nextK, out nextV))
                            return false;

                        int nextLength = length + seqLen;
                        attention = MlxNative.HeadDim256Attention(
                            qCompact,
                            nextK,
                            nextV,
                            numHeads,
                            numKVHeads,
                            seqLen,
                            nextLength,
                            startPos,
                            causal,
                            scale);
                        SetDeviceResult(result, attention);
                        attention = default;

                        MlxNative.FreeArray(kCache);
                        MlxNative.FreeArray(vCache);
                        kCache = nextK;
                        vCache = nextV;
                        nextK = default;
                        nextV = default;
                        length = nextLength;
                        return true;
                    }
                    catch (Exception ex)
                    {
                        if (string.Equals(Environment.GetEnvironmentVariable("TS_MLX_GDN_NATIVE_DEBUG"), "1", StringComparison.Ordinal))
                            Console.WriteLine($"[MLX] Native Qwen35 GDN disabled for this call: {ex.Message}");
                        return false;
                    }
                    finally
                    {
                        MlxNative.FreeArray(qView);
                        MlxNative.FreeArray(qCompact);
                        MlxNative.FreeArray(nextK);
                        MlxNative.FreeArray(nextV);
                        MlxNative.FreeArray(attention);
                    }
                });
            }

            // 中文：将本步 K/V（连续化后）与已有缓存沿序列轴拼接，构造更新后的 K/V 缓存数组。
            private bool TryBuildUpdatedCache(Tensor kHeads, Tensor vHeads,
                out MlxNative.MlxArray nextK, out MlxNative.MlxArray nextV)
            {
                nextK = default;
                nextV = default;
                MlxNative.MlxArray kView = default;
                MlxNative.MlxArray vView = default;
                MlxNative.MlxArray kCompact = default;
                MlxNative.MlxArray vCompact = default;
                MlxNative.MlxArray concatK = default;
                MlxNative.MlxArray concatV = default;
                try
                {
                    kView = GetView(kHeads);
                    vView = GetView(vHeads);
                    kCompact = kHeads.IsContiguous() ? kView : MlxNative.Contiguous(kView);
                    vCompact = vHeads.IsContiguous() ? vView : MlxNative.Contiguous(vView);
                    if (kHeads.IsContiguous())
                        kView = default;
                    if (vHeads.IsContiguous())
                        vView = default;

                    if (length == 0)
                    {
                        nextK = kCompact;
                        nextV = vCompact;
                        kCompact = default;
                        vCompact = default;
                        return true;
                    }

                    if (!kCache.IsValid || !vCache.IsValid)
                        return false;

                    concatK = MlxNative.ConcatenateAxis(kCache, kCompact, 1);
                    concatV = MlxNative.ConcatenateAxis(vCache, vCompact, 1);
                    nextK = concatK;
                    nextV = concatV;
                    concatK = default;
                    concatV = default;
                    return true;
                }
                finally
                {
                    MlxNative.FreeArray(kView);
                    MlxNative.FreeArray(vView);
                    MlxNative.FreeArray(kCompact);
                    MlxNative.FreeArray(vCompact);
                    MlxNative.FreeArray(concatK);
                    MlxNative.FreeArray(concatV);
                }
            }

            // 中文：释放注意力 KV 缓存（委托给 Reset）。
            public void Dispose()
            {
                Reset();
            }
        }

        public sealed class GatedDeltaNetCache : IDisposable
        {
            private MlxNative.MlxArray convState;
            private MlxNative.MlxArray deltaState;
            private MlxNative.MlxArray onesKey;
            private MlxNative.MlxArray onesValue;
            private int keyDim;
            private int valueDim;

            // 中文：释放 GDN 的卷积状态与 delta 状态，重置门控 DeltaNet 缓存的循环状态。
            public void Reset()
            {
                MlxNative.FreeArray(convState);
                MlxNative.FreeArray(deltaState);
                convState = default;
                deltaState = default;
            }

            // 中文：物化并异步求值 GDN 的卷积状态与 delta 状态，限制图深度，返回是否有状态被求值。
            public bool TryEvaluateState()
            {
                try
                {
                    bool evaluated = Materialize(ref convState);
                    evaluated |= Materialize(ref deltaState);
                    return evaluated;
                }
                catch
                {
                    return false;
                }
            }

            // 中文：释放 GDN 全部缓存（循环状态及缓存的全 1 归一化权重数组）。
            public void Dispose()
            {
                Reset();
                MlxNative.FreeArray(onesKey);
                MlxNative.FreeArray(onesValue);
                onesKey = default;
                onesValue = default;
            }

            // 中文：Qwen3.5 GDN 门控 DeltaNet 的打包内核入口，校验张量形状/启用条件后委派 worker 执行打包预处理与 delta 递推。
            public bool TryRunQwen35Packed(
                Tensor result,
                Tensor packedRaw,
                Tensor convWeight,
                Tensor dtBias,
                Tensor aLog,
                Tensor normWeight,
                int seqLen,
                int packedDim,
                int qkvDim,
                int keyDim,
                int valueDim,
                int numKeyHeads,
                int numValueHeads,
                int headKeyDim,
                int headValueDim,
                int convKernel,
                float eps)
            {
                if (string.Equals(Environment.GetEnvironmentVariable("TS_MLX_GDN_NATIVE"), "0", StringComparison.Ordinal))
                    return false;
                if (seqLen != 1 && !(Qwen35GdnPackedKernelsEnabled && seqLen >= Qwen35GdnPackedMinSeqLen))
                    return false;
                if (!CanUseResult(result)
                    || !CanUseGdnTensor(packedRaw)
                    || !CanUseGdnTensor(convWeight)
                    || !CanUseGdnTensor(dtBias)
                    || !CanUseGdnTensor(aLog)
                    || !CanUseGdnTensor(normWeight)
                    || seqLen <= 0
                    || packedDim <= 0
                    || convKernel <= 1
                    || qkvDim <= 0
                    || keyDim <= 0
                    || valueDim <= 0
                    || packedDim < qkvDim + valueDim + numValueHeads * 2
                    || numKeyHeads <= 0
                    || numValueHeads <= 0
                    || headKeyDim <= 0
                    || headValueDim <= 0
                    || headKeyDim % 32 != 0
                    || numValueHeads % numKeyHeads != 0)
                {
                    return false;
                }
                if (packedRaw.DimensionCount != 2 || packedRaw.Sizes[0] != seqLen || packedRaw.Sizes[1] != packedDim)
                    return false;
                if (result.DimensionCount != 2 || result.Sizes[0] != seqLen || result.Sizes[1] != valueDim)
                    return false;

                return MlxWorker.Shared.Invoke(() => RunQwen35PackedInner(
                    result, packedRaw, convWeight, dtBias, aLog, normWeight,
                    seqLen, packedDim, qkvDim, keyDim, valueDim,
                    numKeyHeads, numValueHeads, headKeyDim, headValueDim,
                    convKernel, eps));
            }

            // 中文：打包 GDN 的实际执行体：打包预处理→Q/K 融合 RMSNorm+缩放→门控 delta 递推→后处理门控，更新卷积/delta 状态并写回结果。
            private bool RunQwen35PackedInner(
                Tensor result, Tensor packedRaw, Tensor convWeight, Tensor dtBias, Tensor aLog, Tensor normWeight,
                int seqLen, int packedDim, int qkvDim, int keyDim, int valueDim,
                int numKeyHeads, int numValueHeads, int headKeyDim, int headValueDim,
                int convKernel, float eps)
            {
                MlxNative.MlxArray packedView = default;
                MlxNative.MlxArray convWeightView = default;
                MlxNative.MlxArray dtBiasView = default;
                MlxNative.MlxArray aLogView = default;
                MlxNative.MlxArray normWeightView = default;
                MlxNative.MlxArray q = default;
                MlxNative.MlxArray k = default;
                MlxNative.MlxArray v = default;
                MlxNative.MlxArray qNorm = default;
                MlxNative.MlxArray kNorm = default;
                MlxNative.MlxArray qScaled = default;
                MlxNative.MlxArray kScaled = default;
                MlxNative.MlxArray gDecay = default;
                MlxNative.MlxArray betaSig = default;
                MlxNative.MlxArray zSilu = default;
                MlxNative.MlxArray nextConv = default;
                MlxNative.MlxArray deltaOut = default;
                MlxNative.MlxArray nextDelta = default;
                MlxNative.MlxArray gated = default;

                try
                {
                    bool channelMajor = convWeight.DimensionCount == 2
                        && convWeight.Sizes[0] == qkvDim
                        && convWeight.Sizes[1] == convKernel;
                    bool kernelMajor = convWeight.DimensionCount == 2
                        && convWeight.Sizes[0] == convKernel
                        && convWeight.Sizes[1] == qkvDim;
                    if (!channelMajor && !kernelMajor)
                        return false;

                    EnsureState(convKernel - 1, qkvDim, numValueHeads, headValueDim, headKeyDim);
                    EnsureNormWeights(headKeyDim, headValueDim);

                    packedView = GetView(packedRaw);
                    convWeightView = GetView(convWeight);
                    dtBiasView = GetView(dtBias);
                    aLogView = GetView(aLog);
                    normWeightView = GetView(normWeight);

                    MlxNative.Qwen35GdnPreprocessPacked(
                        packedView,
                        convState,
                        convWeightView,
                        dtBiasView,
                        aLogView,
                        seqLen,
                        packedDim,
                        qkvDim,
                        keyDim,
                        valueDim,
                        numKeyHeads,
                        numValueHeads,
                        headKeyDim,
                        headValueDim,
                        convKernel,
                        channelMajor,
                        out q,
                        out k,
                        out v,
                        out gDecay,
                        out betaSig,
                        out zSilu,
                        out nextConv);

                    // Q and K each go through FastRmsNorm(x, ones, eps=1e-6)
                    // followed by a scalar multiply. With mlx_compile enabled
                    // we fuse the norm + scalar mul into one kernel per Q/K,
                    // dropping 2 of the ~9 Metal dispatches that the decode-
                    // time GDN block issues. Falls back to the eager 2-kernel
                    // chain if compile is disabled.
                    if (!MlxCompiledOps.Disabled)
                    {
                        MlxNative.MlxArray qScale = default;
                        MlxNative.MlxArray kScale = default;
                        try
                        {
                            qScale = MlxNative.NewScalar(1.0f / headKeyDim);
                            kScale = MlxNative.NewScalar(1.0f / MathF.Sqrt(headKeyDim));
                            qScaled = MlxCompiledOps.RmsNormScaled(q, onesKey, qScale, 1e-6f);
                            kScaled = MlxCompiledOps.RmsNormScaled(k, onesKey, kScale, 1e-6f);
                        }
                        finally
                        {
                            MlxNative.FreeArray(qScale);
                            MlxNative.FreeArray(kScale);
                        }
                    }
                    else
                    {
                        qNorm = MlxNative.FastRmsNorm(q, onesKey, 1e-6f);
                        kNorm = MlxNative.FastRmsNorm(k, onesKey, 1e-6f);
                        qScaled = MulScalar(qNorm, 1.0f / headKeyDim);
                        kScaled = MulScalar(kNorm, 1.0f / MathF.Sqrt(headKeyDim));
                    }

                    MlxNative.GatedDelta(
                        qScaled,
                        kScaled,
                        v,
                        gDecay,
                        betaSig,
                        deltaState,
                        1,
                        seqLen,
                        numKeyHeads,
                        numValueHeads,
                        headKeyDim,
                        headValueDim,
                        out deltaOut,
                        out nextDelta);

                    gated = MlxNative.Qwen35GdnPostprocess(
                        deltaOut,
                        zSilu,
                        normWeightView,
                        seqLen,
                        valueDim,
                        numValueHeads,
                        headValueDim,
                        eps);
                    SetDeviceResult(result, gated);
                    gated = default;

                    MlxNative.FreeArray(convState);
                    MlxNative.FreeArray(deltaState);
                    convState = nextConv;
                    deltaState = nextDelta;
                    nextConv = default;
                    nextDelta = default;
                    return true;
                }
                catch (Exception ex)
                {
                    if (string.Equals(Environment.GetEnvironmentVariable("TS_MLX_GDN_NATIVE_DEBUG"), "1", StringComparison.Ordinal))
                        Console.WriteLine($"[MLX] Packed native Qwen35 GDN disabled for this call: {ex.Message}");
                    return false;
                }
                finally
                {
                    MlxNative.FreeArray(packedView);
                    MlxNative.FreeArray(convWeightView);
                    MlxNative.FreeArray(dtBiasView);
                    MlxNative.FreeArray(aLogView);
                    MlxNative.FreeArray(normWeightView);
                    MlxNative.FreeArray(q);
                    MlxNative.FreeArray(k);
                    MlxNative.FreeArray(v);
                    MlxNative.FreeArray(qNorm);
                    MlxNative.FreeArray(kNorm);
                    MlxNative.FreeArray(qScaled);
                    MlxNative.FreeArray(kScaled);
                    MlxNative.FreeArray(gDecay);
                    MlxNative.FreeArray(betaSig);
                    MlxNative.FreeArray(zSilu);
                    MlxNative.FreeArray(nextConv);
                    MlxNative.FreeArray(deltaOut);
                    MlxNative.FreeArray(nextDelta);
                    MlxNative.FreeArray(gated);
                }
            }

            // 中文：Qwen3.5 GDN 门控 DeltaNet 的非打包内核入口，校验各输入张量形状后委派 worker 执行完整 GDN 计算。
            public bool TryRunQwen35(
                Tensor result,
                Tensor qkvRaw,
                Tensor zRaw,
                Tensor betaRaw,
                Tensor alphaRaw,
                Tensor convWeight,
                Tensor dtBias,
                Tensor aLog,
                Tensor normWeight,
                int seqLen,
                int qkvDim,
                int keyDim,
                int valueDim,
                int numKeyHeads,
                int numValueHeads,
                int headKeyDim,
                int headValueDim,
                int convKernel,
                float eps)
            {
                if (string.Equals(Environment.GetEnvironmentVariable("TS_MLX_GDN_NATIVE"), "0", StringComparison.Ordinal))
                    return false;
                if (!CanUseResult(result)
                    || !CanUseGdnTensor(qkvRaw)
                    || !CanUseGdnTensor(zRaw)
                    || !CanUseGdnTensor(betaRaw)
                    || !CanUseGdnTensor(alphaRaw)
                    || !CanUseGdnTensor(convWeight)
                    || !CanUseGdnTensor(dtBias)
                    || !CanUseGdnTensor(aLog)
                    || !CanUseGdnTensor(normWeight)
                    || seqLen <= 0
                    || convKernel <= 1
                    || qkvDim <= 0
                    || keyDim <= 0
                    || valueDim <= 0
                    || numKeyHeads <= 0
                    || numValueHeads <= 0
                    || headKeyDim <= 0
                    || headValueDim <= 0
                    || headKeyDim % 32 != 0
                    || numValueHeads % numKeyHeads != 0)
                {
                    return false;
                }
                if (qkvRaw.DimensionCount != 2 || qkvRaw.Sizes[0] != seqLen || qkvRaw.Sizes[1] != qkvDim)
                    return false;
                if (zRaw.DimensionCount != 2 || zRaw.Sizes[0] != seqLen || zRaw.Sizes[1] != valueDim)
                    return false;
                if (betaRaw.DimensionCount != 2 || betaRaw.Sizes[0] != seqLen || betaRaw.Sizes[1] != numValueHeads)
                    return false;
                if (alphaRaw.DimensionCount != 2 || alphaRaw.Sizes[0] != seqLen || alphaRaw.Sizes[1] != numValueHeads)
                    return false;
                if (result.DimensionCount != 2 || result.Sizes[0] != seqLen || result.Sizes[1] != valueDim)
                    return false;

                return MlxWorker.Shared.Invoke(() => RunQwen35Inner(
                    result, qkvRaw, zRaw, betaRaw, alphaRaw,
                    convWeight, dtBias, aLog, normWeight,
                    seqLen, qkvDim, keyDim, valueDim,
                    numKeyHeads, numValueHeads, headKeyDim, headValueDim,
                    convKernel, eps));
            }

            // 中文：非打包 GDN 的实际执行体：深度卷积+SiLU→拆分 QKV→Q/K 归一化缩放→门控/beta 计算→门控 delta 递推→RMSNorm 与 z 门控，更新状态并写回结果（可选走打包快路径）。
            private bool RunQwen35Inner(
                Tensor result, Tensor qkvRaw, Tensor zRaw, Tensor betaRaw, Tensor alphaRaw,
                Tensor convWeight, Tensor dtBias, Tensor aLog, Tensor normWeight,
                int seqLen, int qkvDim, int keyDim, int valueDim,
                int numKeyHeads, int numValueHeads, int headKeyDim, int headValueDim,
                int convKernel, float eps)
            {
                MlxNative.MlxArray qkvView = default;
                MlxNative.MlxArray zView = default;
                MlxNative.MlxArray betaView = default;
                MlxNative.MlxArray alphaView = default;
                MlxNative.MlxArray convWeightView = default;
                MlxNative.MlxArray dtBiasView = default;
                MlxNative.MlxArray aLogView = default;
                MlxNative.MlxArray normWeightView = default;
                MlxNative.MlxArray qkv3 = default;
                MlxNative.MlxArray concat = default;
                MlxNative.MlxArray convOut3 = default;
                MlxNative.MlxArray convOut2 = default;
                MlxNative.MlxArray convSilu = default;
                MlxNative.MlxArray nextConv = default;
                MlxNative.MlxArray qSlice = default;
                MlxNative.MlxArray kSlice = default;
                MlxNative.MlxArray vSlice = default;
                MlxNative.MlxArray q = default;
                MlxNative.MlxArray k = default;
                MlxNative.MlxArray v = default;
                MlxNative.MlxArray qNorm = default;
                MlxNative.MlxArray kNorm = default;
                MlxNative.MlxArray qScaled = default;
                MlxNative.MlxArray kScaled = default;
                MlxNative.MlxArray alphaPlus = default;
                MlxNative.MlxArray softplus = default;
                MlxNative.MlxArray gLinear = default;
                MlxNative.MlxArray gDecay2 = default;
                MlxNative.MlxArray gDecay = default;
                MlxNative.MlxArray betaSig2 = default;
                MlxNative.MlxArray betaSig = default;
                MlxNative.MlxArray deltaOut = default;
                MlxNative.MlxArray nextDelta = default;
                MlxNative.MlxArray normed = default;
                MlxNative.MlxArray z4 = default;
                MlxNative.MlxArray zSilu = default;
                MlxNative.MlxArray gated4 = default;
                MlxNative.MlxArray gated2 = default;

                try
                {
                    EnsureState(convKernel - 1, qkvDim, numValueHeads, headValueDim, headKeyDim);
                    EnsureNormWeights(headKeyDim, headValueDim);

                    qkvView = GetView(qkvRaw);
                    zView = GetView(zRaw);
                    betaView = GetView(betaRaw);
                    alphaView = GetView(alphaRaw);
                    convWeightView = GetView(convWeight);
                    dtBiasView = GetView(dtBias);
                    aLogView = GetView(aLog);
                    normWeightView = GetView(normWeight);

                    if (Qwen35GdnPackedKernelsEnabled && seqLen >= Qwen35GdnPackedMinSeqLen)
                    {
                        try
                        {
                            bool channelMajor = convWeight.DimensionCount == 2
                                && convWeight.Sizes[0] == qkvDim
                                && convWeight.Sizes[1] == convKernel;
                            bool kernelMajor = convWeight.DimensionCount == 2
                                && convWeight.Sizes[0] == convKernel
                                && convWeight.Sizes[1] == qkvDim;
                            if (!channelMajor && !kernelMajor)
                                throw new NotSupportedException("Qwen35 MLX GDN conv weight must be [qkvDim, kernel] or [kernel, qkvDim].");

                            MlxNative.Qwen35GdnPreprocess(
                                qkvView,
                                zView,
                                betaView,
                                alphaView,
                                convState,
                                convWeightView,
                                dtBiasView,
                                aLogView,
                                seqLen,
                                qkvDim,
                                keyDim,
                                valueDim,
                                numKeyHeads,
                                numValueHeads,
                                headKeyDim,
                                headValueDim,
                                convKernel,
                                channelMajor,
                                out q,
                                out k,
                                out v,
                                out gDecay,
                                out betaSig,
                                out zSilu,
                                out nextConv);

                            qNorm = MlxNative.FastRmsNorm(q, onesKey, 1e-6f);
                            kNorm = MlxNative.FastRmsNorm(k, onesKey, 1e-6f);
                            qScaled = MulScalar(qNorm, 1.0f / headKeyDim);
                            kScaled = MulScalar(kNorm, 1.0f / MathF.Sqrt(headKeyDim));

                            MlxNative.GatedDelta(
                                qScaled,
                                kScaled,
                                v,
                                gDecay,
                                betaSig,
                                deltaState,
                                1,
                                seqLen,
                                numKeyHeads,
                                numValueHeads,
                                headKeyDim,
                                headValueDim,
                                out deltaOut,
                                out nextDelta);

                            gated2 = MlxNative.Qwen35GdnPostprocess(
                                deltaOut,
                                zSilu,
                                normWeightView,
                                seqLen,
                                valueDim,
                                numValueHeads,
                                headValueDim,
                                eps);
                            SetDeviceResult(result, gated2);
                            gated2 = default;

                            MlxNative.FreeArray(convState);
                            MlxNative.FreeArray(deltaState);
                            convState = nextConv;
                            deltaState = nextDelta;
                            nextConv = default;
                            nextDelta = default;
                            return true;
                        }
                        catch (Exception ex)
                        {
                            if (string.Equals(Environment.GetEnvironmentVariable("TS_MLX_GDN_NATIVE_DEBUG"), "1", StringComparison.Ordinal))
                                Console.WriteLine($"[MLX] Packed Qwen35 GDN kernels disabled for this call: {ex.Message}");
                            MlxNative.FreeArray(q);
                            MlxNative.FreeArray(k);
                            MlxNative.FreeArray(v);
                            MlxNative.FreeArray(qNorm);
                            MlxNative.FreeArray(kNorm);
                            MlxNative.FreeArray(qScaled);
                            MlxNative.FreeArray(kScaled);
                            MlxNative.FreeArray(gDecay);
                            MlxNative.FreeArray(betaSig);
                            MlxNative.FreeArray(zSilu);
                            MlxNative.FreeArray(deltaOut);
                            MlxNative.FreeArray(nextDelta);
                            MlxNative.FreeArray(gated2);
                            MlxNative.FreeArray(nextConv);
                            q = k = v = qNorm = kNorm = qScaled = kScaled = gDecay = betaSig = zSilu = deltaOut = nextDelta = gated2 = nextConv = default;
                        }
                    }

                    qkv3 = MlxNative.Reshape(qkvView, new[] { 1, seqLen, qkvDim });
                    concat = MlxNative.ConcatenateAxis(convState, qkv3, 1);
                    convOut3 = RunDepthwiseConv(concat, convWeightView, convWeight, seqLen, qkvDim, convKernel);
                    nextConv = MlxNative.Slice(
                        concat,
                        new[] { 0, seqLen, 0 },
                        new[] { 1, seqLen + convKernel - 1, qkvDim },
                        new[] { 1, 1, 1 });

                    convSilu = Silu(convOut3);
                    convOut2 = MlxNative.Reshape(convSilu, new[] { seqLen, qkvDim });

                    qSlice = MlxNative.Slice(convOut2, new[] { 0, 0 }, new[] { seqLen, keyDim }, new[] { 1, 1 });
                    kSlice = MlxNative.Slice(convOut2, new[] { 0, keyDim }, new[] { seqLen, 2 * keyDim }, new[] { 1, 1 });
                    vSlice = MlxNative.Slice(convOut2, new[] { 0, 2 * keyDim }, new[] { seqLen, 2 * keyDim + valueDim }, new[] { 1, 1 });
                    q = MlxNative.Reshape(qSlice, new[] { 1, seqLen, numKeyHeads, headKeyDim });
                    k = MlxNative.Reshape(kSlice, new[] { 1, seqLen, numKeyHeads, headKeyDim });
                    v = MlxNative.Reshape(vSlice, new[] { 1, seqLen, numValueHeads, headValueDim });

                    // Fused rms_norm + scalar mul for Q/K (see TryRunQwen35Packed
                    // for rationale). Saves 2 kernel launches per GDN layer.
                    if (!MlxCompiledOps.Disabled)
                    {
                        MlxNative.MlxArray qScale = default;
                        MlxNative.MlxArray kScale = default;
                        try
                        {
                            qScale = MlxNative.NewScalar(1.0f / headKeyDim);
                            kScale = MlxNative.NewScalar(1.0f / MathF.Sqrt(headKeyDim));
                            qScaled = MlxCompiledOps.RmsNormScaled(q, onesKey, qScale, 1e-6f);
                            kScaled = MlxCompiledOps.RmsNormScaled(k, onesKey, kScale, 1e-6f);
                        }
                        finally
                        {
                            MlxNative.FreeArray(qScale);
                            MlxNative.FreeArray(kScale);
                        }
                    }
                    else
                    {
                        qNorm = MlxNative.FastRmsNorm(q, onesKey, 1e-6f);
                        kNorm = MlxNative.FastRmsNorm(k, onesKey, 1e-6f);
                        qScaled = MulScalar(qNorm, 1.0f / headKeyDim);
                        kScaled = MulScalar(kNorm, 1.0f / MathF.Sqrt(headKeyDim));
                    }

                    alphaPlus = MlxNative.Binary(MlxNative.MlxBinaryOp.Add, alphaView, dtBiasView);
                    softplus = Softplus(alphaPlus);
                    gLinear = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, softplus, aLogView);
                    gDecay2 = MlxNative.Unary(MlxNative.MlxUnaryOp.Exp, gLinear);
                    gDecay = MlxNative.Reshape(gDecay2, new[] { 1, seqLen, numValueHeads });
                    betaSig2 = MlxNative.Unary(MlxNative.MlxUnaryOp.Sigmoid, betaView);
                    betaSig = MlxNative.Reshape(betaSig2, new[] { 1, seqLen, numValueHeads });

                    MlxNative.GatedDelta(
                        qScaled,
                        kScaled,
                        v,
                        gDecay,
                        betaSig,
                        deltaState,
                        1,
                        seqLen,
                        numKeyHeads,
                        numValueHeads,
                        headKeyDim,
                        headValueDim,
                        out deltaOut,
                        out nextDelta);

                    normed = MlxNative.FastRmsNorm(deltaOut, normWeightView, eps);
                    z4 = MlxNative.Reshape(zView, new[] { 1, seqLen, numValueHeads, headValueDim });
                    zSilu = Silu(z4);
                    gated4 = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, normed, zSilu);
                    gated2 = MlxNative.Reshape(gated4, new[] { seqLen, valueDim });
                    SetDeviceResult(result, gated2);
                    gated2 = default;

                    MlxNative.FreeArray(convState);
                    MlxNative.FreeArray(deltaState);
                    convState = nextConv;
                    deltaState = nextDelta;
                    nextConv = default;
                    nextDelta = default;
                    return true;
                }
                catch
                {
                    return false;
                }
                finally
                {
                    MlxNative.FreeArray(qkvView);
                    MlxNative.FreeArray(zView);
                    MlxNative.FreeArray(betaView);
                    MlxNative.FreeArray(alphaView);
                    MlxNative.FreeArray(convWeightView);
                    MlxNative.FreeArray(dtBiasView);
                    MlxNative.FreeArray(aLogView);
                    MlxNative.FreeArray(normWeightView);
                    MlxNative.FreeArray(qkv3);
                    MlxNative.FreeArray(concat);
                    MlxNative.FreeArray(convOut3);
                    MlxNative.FreeArray(convOut2);
                    MlxNative.FreeArray(convSilu);
                    MlxNative.FreeArray(nextConv);
                    MlxNative.FreeArray(qSlice);
                    MlxNative.FreeArray(kSlice);
                    MlxNative.FreeArray(vSlice);
                    MlxNative.FreeArray(q);
                    MlxNative.FreeArray(k);
                    MlxNative.FreeArray(v);
                    MlxNative.FreeArray(qNorm);
                    MlxNative.FreeArray(kNorm);
                    MlxNative.FreeArray(qScaled);
                    MlxNative.FreeArray(kScaled);
                    MlxNative.FreeArray(alphaPlus);
                    MlxNative.FreeArray(softplus);
                    MlxNative.FreeArray(gLinear);
                    MlxNative.FreeArray(gDecay2);
                    MlxNative.FreeArray(gDecay);
                    MlxNative.FreeArray(betaSig2);
                    MlxNative.FreeArray(betaSig);
                    MlxNative.FreeArray(deltaOut);
                    MlxNative.FreeArray(nextDelta);
                    MlxNative.FreeArray(normed);
                    MlxNative.FreeArray(z4);
                    MlxNative.FreeArray(zSilu);
                    MlxNative.FreeArray(gated4);
                    MlxNative.FreeArray(gated2);
                }
            }

            // 中文：按需惰性初始化 GDN 的卷积状态与 delta 状态为零张量（仅在尚未分配时）。
            private void EnsureState(int convTail, int qkvDim, int numValueHeads, int headValueDim, int headKeyDim)
            {
                if (!convState.IsValid)
                    convState = MlxNative.Full(new[] { 1, convTail, qkvDim }, 0.0f, DType.Float32);
                if (!deltaState.IsValid)
                    deltaState = MlxNative.Full(new[] { 1, numValueHeads, headValueDim, headKeyDim }, 0.0f, DType.Float32);
            }

            // 中文：按需缓存用于 Q/K RMSNorm 的全 1 权重数组，维度变化时重建。
            private void EnsureNormWeights(int keyDim, int valueDim)
            {
                if (!onesKey.IsValid || this.keyDim != keyDim)
                {
                    MlxNative.FreeArray(onesKey);
                    onesKey = MlxNative.Full(new[] { keyDim }, 1.0f, DType.Float32);
                    this.keyDim = keyDim;
                }
                if (!onesValue.IsValid || this.valueDim != valueDim)
                {
                    MlxNative.FreeArray(onesValue);
                    onesValue = MlxNative.Full(new[] { valueDim }, 1.0f, DType.Float32);
                    this.valueDim = valueDim;
                }
            }

            // 中文：以逐核位移加权求和的方式对拼接序列执行 GDN 的因果深度卷积，支持 channel-major/kernel-major 两种权重布局。
            private static MlxNative.MlxArray RunDepthwiseConv(
                MlxNative.MlxArray concat,
                MlxNative.MlxArray convWeight,
                Tensor convWeightTensor,
                int seqLen,
                int qkvDim,
                int convKernel)
            {
                MlxNative.MlxArray acc = default;
                try
                {
                    bool channelMajor = convWeightTensor.DimensionCount == 2
                        && convWeightTensor.Sizes[0] == qkvDim
                        && convWeightTensor.Sizes[1] == convKernel;
                    bool kernelMajor = convWeightTensor.DimensionCount == 2
                        && convWeightTensor.Sizes[0] == convKernel
                        && convWeightTensor.Sizes[1] == qkvDim;
                    if (!channelMajor && !kernelMajor)
                        throw new NotSupportedException("Qwen35 MLX GDN conv weight must be [qkvDim, kernel] or [kernel, qkvDim].");

                    for (int i = 0; i < convKernel; i++)
                    {
                        MlxNative.MlxArray seg = default;
                        MlxNative.MlxArray wi = default;
                        MlxNative.MlxArray wi3 = default;
                        MlxNative.MlxArray term = default;
                        MlxNative.MlxArray next = default;
                        try
                        {
                            seg = MlxNative.Slice(
                                concat,
                                new[] { 0, i, 0 },
                                new[] { 1, i + seqLen, qkvDim },
                                new[] { 1, 1, 1 });
                            wi = channelMajor
                                ? MlxNative.Slice(convWeight, new[] { 0, i }, new[] { qkvDim, i + 1 }, new[] { 1, 1 })
                                : MlxNative.Slice(convWeight, new[] { i, 0 }, new[] { i + 1, qkvDim }, new[] { 1, 1 });
                            wi3 = MlxNative.Reshape(wi, new[] { 1, 1, qkvDim });
                            term = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, seg, wi3);
                            if (!acc.IsValid)
                            {
                                acc = term;
                                term = default;
                            }
                            else
                            {
                                next = MlxNative.Binary(MlxNative.MlxBinaryOp.Add, acc, term);
                                MlxNative.FreeArray(acc);
                                acc = next;
                                next = default;
                            }
                        }
                        finally
                        {
                            MlxNative.FreeArray(seg);
                            MlxNative.FreeArray(wi);
                            MlxNative.FreeArray(wi3);
                            MlxNative.FreeArray(term);
                            MlxNative.FreeArray(next);
                        }
                    }

                    MlxNative.MlxArray result = acc;
                    acc = default;
                    return result;
                }
                finally
                {
                    MlxNative.FreeArray(acc);
                }
            }

            // 中文：计算 SiLU 激活 x * sigmoid(x)。
            private static MlxNative.MlxArray Silu(MlxNative.MlxArray input)
            {
                MlxNative.MlxArray sigmoid = default;
                try
                {
                    sigmoid = MlxNative.Unary(MlxNative.MlxUnaryOp.Sigmoid, input);
                    return MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, input, sigmoid);
                }
                finally
                {
                    MlxNative.FreeArray(sigmoid);
                }
            }

            // 中文：计算 Softplus 激活 log(1 + exp(x))。
            private static MlxNative.MlxArray Softplus(MlxNative.MlxArray input)
            {
                MlxNative.MlxArray exp = default;
                MlxNative.MlxArray one = default;
                MlxNative.MlxArray plusOne = default;
                try
                {
                    exp = MlxNative.Unary(MlxNative.MlxUnaryOp.Exp, input);
                    one = MlxNative.NewScalar(1.0f);
                    plusOne = MlxNative.Binary(MlxNative.MlxBinaryOp.Add, exp, one);
                    return MlxNative.Unary(MlxNative.MlxUnaryOp.Log, plusOne);
                }
                finally
                {
                    MlxNative.FreeArray(exp);
                    MlxNative.FreeArray(one);
                    MlxNative.FreeArray(plusOne);
                }
            }

            // 中文：将输入数组逐元素乘以标量值。
            private static MlxNative.MlxArray MulScalar(MlxNative.MlxArray input, float value)
            {
                MlxNative.MlxArray scalar = default;
                try
                {
                    scalar = MlxNative.NewScalar(value);
                    return MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, input, scalar);
                }
                finally
                {
                    MlxNative.FreeArray(scalar);
                }
            }
        }

        // 中文：将数组连续化并异步求值以约束计算图深度（不阻塞主机），就地替换为求值后的数组。
        private static bool Materialize(ref MlxNative.MlxArray array)
        {
            if (!array.IsValid)
                return false;

            MlxNative.MlxArray contiguous = default;
            try
            {
                contiguous = MlxNative.Contiguous(array);
                // AsyncEval bounds the graph depth (so we don't accumulate a
                // 60-layer attention chain by the end of the forward pass)
                // without blocking the host. The next op that depends on this
                // array — typically the next layer's attention or the host
                // sampler read — will drain implicitly. The previous code
                // used sync Eval here, costing ~1 ms × 7-8 cache evals per
                // decode token (one every MlxEvalEveryNLayers).
                MlxNative.AsyncEval(contiguous);
                MlxNative.FreeArray(array);
                array = contiguous;
                contiguous = default;
                return true;
            }
            finally
            {
                MlxNative.FreeArray(contiguous);
            }
        }

        // 中文：预填充阶段的因果注意力，优先走 head-first 融合 SDPA，失败时可选回退到分块向量化预填充注意力。
        public static bool TryPrefillAttention(
            Tensor result,
            Tensor qHeads,
            Tensor kHeads,
            Tensor vHeads,
            int numHeads,
            int numKVHeads,
            int headDim,
            int seqLen,
            int kvLen,
            int maskStart,
            int windowSize,
            float scale)
        {
            if (seqLen <= 0 || kvLen <= 0 || maskStart != 0 || kvLen != seqLen)
                return false;
            if (windowSize > 0 && kvLen > windowSize)
                return false;

            if (TryRunHeadFirstAttention(result, qHeads, kHeads, vHeads,
                numHeads, numKVHeads, seqLen, kvLen, headDim,
                seqLen == 1 ? string.Empty : "causal",
                scale))
            {
                return true;
            }

            if (maskStart == 0 && kvLen == seqLen)
            {
                if (!string.Equals(Environment.GetEnvironmentVariable("TS_MLX_CHUNKED_VECTOR_PREFILL"), "1", StringComparison.Ordinal))
                    return false;

                return TryRunChunkedVectorPrefillAttention(result, qHeads, kHeads, vHeads,
                    numHeads, numKVHeads, seqLen, headDim, scale);
            }

            return false;
        }

        /// <summary>
        /// Decode attention with per-head attention sinks (GPT-OSS family).
        /// Replaces the host-CPU <see cref="ModelBase.AttentionDecodePureCS"/>
        /// path for MLX backend so K/V cache stays on device. Sinks must
        /// be an MLX-backed <see cref="DType.Float32"/> tensor of shape
        /// [numHeads]. Sliding-window attention is supported via
        /// <paramref name="slidingWindow"/> &gt; 0; pass 0 for full causal.
        /// </summary>
        // 中文：带每头注意力 sink 的解码注意力（GPT-OSS 系列），支持滑动窗口，使 K/V 缓存保留在设备上。
        public static bool TryDecodeAttentionWithSinks(
            Tensor result,
            Tensor qFlat,
            Tensor kCache,
            Tensor vCache,
            Tensor sinks,
            int numHeads,
            int numKVHeads,
            int headDim,
            int cacheLen,
            int kvLen,
            int slidingWindow,
            float scale)
        {
            if (result == null || qFlat == null || kCache == null || vCache == null || sinks == null)
                return false;
            if (!CanUseAttentionTensor(qFlat) || !CanUseAttentionTensor(kCache)
                || !CanUseAttentionTensor(vCache) || !CanUseAttentionTensor(sinks))
                return false;
            if (qFlat.ElementCount() != (long)numHeads * headDim)
                return false;
            if (sinks.ElementType != DType.Float32
                || sinks.DimensionCount != 1
                || sinks.Sizes[0] != numHeads)
                return false;
            if (kvLen <= 0 || kvLen > cacheLen)
                return false;
            if (numHeads % numKVHeads != 0)
                return false;
            if (headDim > 256)   // matches kernel HeadDim cap
                return false;

            int attendStart = (slidingWindow > 0 && kvLen > slidingWindow)
                ? kvLen - slidingWindow
                : 0;
            int attendEnd = kvLen;

            MlxNative.MlxArray qView = default;
            MlxNative.MlxArray kView = default;
            MlxNative.MlxArray vView = default;
            MlxNative.MlxArray sinksView = default;
            MlxNative.MlxArray output = default;
            try
            {
                qView = GetView(qFlat);
                kView = GetView(kCache);
                vView = GetView(vCache);
                sinksView = GetView(sinks);

                output = MlxNative.DecodeAttentionWithSinks(
                    qView, kView, vView, sinksView,
                    numHeads, numKVHeads, headDim,
                    cacheLen, attendStart, attendEnd, scale);
                SetDeviceResult(result, output);
                output = default;
                return true;
            }
            catch
            {
                return false;
            }
            finally
            {
                MlxNative.FreeArray(qView);
                MlxNative.FreeArray(kView);
                MlxNative.FreeArray(vView);
                MlxNative.FreeArray(sinksView);
                MlxNative.FreeArray(output);
            }
        }

        // 中文：单步解码注意力，按 circular 标志对 KV 缓存做线性/环形切片（必要时拼接绕回段），整段子图打包进单次 worker 往返执行。
        public static bool TryDecodeAttention(
            Tensor result,
            Tensor qFlat,
            Tensor kCache,
            Tensor vCache,
            int numHeads,
            int numKVHeads,
            int headDim,
            int attendStart,
            int attendLen,
            int cacheLen,
            bool circular,
            float scale)
        {
            if (attendLen <= 0 || attendStart < 0 || cacheLen <= 0)
                return false;
            if (!CanUseAttentionTensor(qFlat) || !CanUseAttentionTensor(kCache) || !CanUseAttentionTensor(vCache))
                return false;
            if (qFlat.ElementCount() != (long)numHeads * headDim)
                return false;

            // Batch the entire decode-attention sub-graph (narrows + view +
            // TryRunHeadFirstAttention, which itself issues several MLX
            // ops) into one worker round-trip. Without this wrapper each
            // attention layer at decode pays ~15+ queue hand-offs.
            return MlxWorker.Shared.Invoke(() =>
            {
                Tensor kLogical = null;
                Tensor vLogical = null;
                try
                {
                    if (circular)
                    {
                        int firstSlot = attendStart % cacheLen;
                        if (firstSlot < 0)
                            firstSlot += cacheLen;

                        if (TryCircularDecodeAttention(result, qFlat, kCache, vCache,
                            numHeads, numKVHeads, headDim, firstSlot, attendLen, cacheLen, scale))
                        {
                            return true;
                        }

                        if (firstSlot + attendLen <= cacheLen)
                        {
                            kLogical = kCache.Narrow(1, firstSlot, attendLen);
                            vLogical = vCache.Narrow(1, firstSlot, attendLen);
                        }
                        else
                        {
                            int tailLen = cacheLen - firstSlot;
                            int headLen = attendLen - tailLen;
                            using var kTail = kCache.Narrow(1, firstSlot, tailLen);
                            using var kHead = kCache.Narrow(1, 0, headLen);
                            using var vTail = vCache.Narrow(1, firstSlot, tailLen);
                            using var vHead = vCache.Narrow(1, 0, headLen);
                            kLogical = Ops.Concat(null, 1, kTail, kHead);
                            vLogical = Ops.Concat(null, 1, vTail, vHead);
                        }
                    }
                    else
                    {
                        if (attendStart + attendLen > cacheLen)
                            return false;

                        // Phase 6g experiment: custom head_dim=512 kernel
                        // exists (TryDecodeAttentionHeadDim512) but is gated
                        // off by default — MLX's built-in fast_sdpa already
                        // uses simdgroup ops internally and beat the custom
                        // kernel by ~1 ms / tok on Gemma 4 E4B Q8_0 (42.5
                        // vs 41.6 ms / tok with it engaged). The custom
                        // kernel is kept for future experiments. Set
                        // TS_MLX_FUSED_HEADDIM512_SDPA=1 to engage.
                        if (headDim == 512 && attendStart == 0
                            && string.Equals(Environment.GetEnvironmentVariable("TS_MLX_FUSED_HEADDIM512_SDPA"), "1", StringComparison.Ordinal))
                        {
                            if (TryDecodeAttentionHeadDim512(result, qFlat, kCache, vCache,
                                numHeads, numKVHeads, headDim, attendLen, cacheLen, scale))
                            {
                                return true;
                            }
                        }

                        kLogical = kCache.Narrow(1, attendStart, attendLen);
                        vLogical = vCache.Narrow(1, attendStart, attendLen);
                    }

                    using var qHeads = qFlat.View(numHeads, 1, headDim);
                    return TryRunHeadFirstAttention(result, qHeads, kLogical, vLogical,
                        numHeads, numKVHeads, 1, attendLen, headDim, string.Empty, scale);
                }
                finally
                {
                    kLogical?.Dispose();
                    vLogical?.Dispose();
                }
            });
        }

        /// <summary>
        /// Decode-step attention for Gemma 4 global layers (head_dim = 512,
        /// non-circular). Calls the custom simdgroup-optimized Metal kernel
        /// when shapes match; falls through to MLX's built-in SDPA path
        /// otherwise. Drops in for the existing TryDecodeAttention non-
        /// circular branch when head_dim == 512.
        /// </summary>
        // 中文：Gemma 4 全局层（headDim=512、非环形）解码注意力，形状匹配时调用定制 simdgroup Metal 内核。
        public static bool TryDecodeAttentionHeadDim512(
            Tensor result,
            Tensor qFlat,
            Tensor kCache,
            Tensor vCache,
            int numHeads,
            int numKVHeads,
            int headDim,
            int attendLen,
            int cacheLen,
            float scale)
        {
            if (!CanUseResult(result)
                || !CanUseAttentionTensor(qFlat)
                || !CanUseAttentionTensor(kCache)
                || !CanUseAttentionTensor(vCache)
                || qFlat.ElementType != DType.Float32
                || !qFlat.IsContiguous()
                || !kCache.IsContiguous()
                || !vCache.IsContiguous()
                || headDim != 512
                || attendLen <= 0
                || attendLen > cacheLen
                || numHeads <= 0
                || numKVHeads <= 0
                || numHeads % numKVHeads != 0
                || result.DimensionCount != 2
                || result.Sizes[0] != 1
                || result.Sizes[1] != (long)numHeads * headDim
                || qFlat.ElementCount() != (long)numHeads * headDim
                || kCache.DimensionCount != 3
                || vCache.DimensionCount != 3
                || kCache.Sizes[0] != numKVHeads
                || vCache.Sizes[0] != numKVHeads
                || kCache.Sizes[1] != cacheLen
                || vCache.Sizes[1] != cacheLen
                || kCache.Sizes[2] != headDim
                || vCache.Sizes[2] != headDim)
            {
                return false;
            }

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxNative.MlxArray qView = default;
                MlxNative.MlxArray kView = default;
                MlxNative.MlxArray vView = default;
                MlxNative.MlxArray output = default;
                try
                {
                    qView = GetView(qFlat);
                    kView = GetView(kCache);
                    vView = GetView(vCache);
                    output = MlxNative.DecodeAttentionHeadDim512(
                        qView, kView, vView,
                        numHeads, numKVHeads, headDim, cacheLen, attendLen, scale);
                    SetDeviceResult(result, output);
                    output = default;
                    return true;
                }
                catch
                {
                    return false;
                }
                finally
                {
                    MlxNative.FreeArray(qView);
                    MlxNative.FreeArray(kView);
                    MlxNative.FreeArray(vView);
                    MlxNative.FreeArray(output);
                }
            });
        }

        // 中文：环形 KV 缓存的解码注意力，从 firstSlot 起按环形布局调用定制内核，避免显式绕回拼接。
        private static bool TryCircularDecodeAttention(
            Tensor result,
            Tensor qFlat,
            Tensor kCache,
            Tensor vCache,
            int numHeads,
            int numKVHeads,
            int headDim,
            int firstSlot,
            int attendLen,
            int cacheLen,
            float scale)
        {
            if (!CanUseResult(result)
                || !CanUseAttentionTensor(qFlat)
                || !CanUseAttentionTensor(kCache)
                || !CanUseAttentionTensor(vCache)
                || qFlat.ElementType != DType.Float32
                || !qFlat.IsContiguous()
                || !kCache.IsContiguous()
                || !vCache.IsContiguous()
                || headDim <= 0
                || headDim > 256
                || attendLen <= 0
                || attendLen > cacheLen
                || numHeads <= 0
                || numKVHeads <= 0
                || numHeads % numKVHeads != 0
                || result.DimensionCount != 2
                || result.Sizes[0] != 1
                || result.Sizes[1] != (long)numHeads * headDim
                || qFlat.ElementCount() != (long)numHeads * headDim
                || kCache.DimensionCount != 3
                || vCache.DimensionCount != 3
                || kCache.Sizes[0] != numKVHeads
                || vCache.Sizes[0] != numKVHeads
                || kCache.Sizes[1] != cacheLen
                || vCache.Sizes[1] != cacheLen
                || kCache.Sizes[2] != headDim
                || vCache.Sizes[2] != headDim)
            {
                return false;
            }

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxNative.MlxArray qView = default;
                MlxNative.MlxArray kView = default;
                MlxNative.MlxArray vView = default;
                MlxNative.MlxArray output = default;
                try
                {
                    qView = GetView(qFlat);
                    kView = GetView(kCache);
                    vView = GetView(vCache);
                    output = MlxNative.CircularDecodeAttention(
                        qView,
                        kView,
                        vView,
                        numHeads,
                        numKVHeads,
                        headDim,
                        cacheLen,
                        firstSlot,
                        attendLen,
                        scale);
                    SetDeviceResult(result, output);
                    output = default;
                    return true;
                }
                catch
                {
                    return false;
                }
                finally
                {
                    MlxNative.FreeArray(qView);
                    MlxNative.FreeArray(kView);
                    MlxNative.FreeArray(vView);
                    MlxNative.FreeArray(output);
                }
            });
        }

        // 中文：head-first 布局注意力的入口，校验快注意力可用性与各张量形状后委派 worker 执行内部计算。
        private static bool TryRunHeadFirstAttention(
            Tensor result,
            Tensor qHeads,
            Tensor kHeads,
            Tensor vHeads,
            int numHeads,
            int numKVHeads,
            int seqLen,
            int kvLen,
            int headDim,
            string maskMode,
            float scale)
        {
            if (!CanUseFastAttention(numHeads, numKVHeads, seqLen, kvLen, headDim))
                return false;

            if (!CanUseResult(result)
                || !CanUseAttentionTensor(qHeads)
                || !CanUseAttentionTensor(kHeads)
                || !CanUseAttentionTensor(vHeads)
                || qHeads.ElementType != DType.Float32
                || result.Sizes.Length != 2
                || result.Sizes[0] != seqLen
                || result.Sizes[1] != (long)numHeads * headDim
                || qHeads.Sizes.Length != 3
                || kHeads.Sizes.Length != 3
                || vHeads.Sizes.Length != 3
                || qHeads.Sizes[0] != numHeads
                || qHeads.Sizes[1] != seqLen
                || qHeads.Sizes[2] != headDim
                || kHeads.Sizes[0] != numKVHeads
                || kHeads.Sizes[1] != kvLen
                || kHeads.Sizes[2] != headDim
                || vHeads.Sizes[0] != numKVHeads
                || vHeads.Sizes[1] != kvLen
                || vHeads.Sizes[2] != headDim)
            {
                return false;
            }

            return MlxWorker.Shared.Invoke(() => RunHeadFirstAttentionInner(
                result, qHeads, kHeads, vHeads, numHeads, numKVHeads, seqLen, kvLen, headDim, maskMode, scale));
        }

        // 中文：head-first 注意力实际执行体：连续化输入，headDim=256 走定制内核否则走 MLX 快 SDPA，并把结果转回序列主序扁平写入 result。
        private static bool RunHeadFirstAttentionInner(
            Tensor result, Tensor qHeads, Tensor kHeads, Tensor vHeads,
            int numHeads, int numKVHeads, int seqLen, int kvLen, int headDim, string maskMode, float scale)
        {
            MlxNative.MlxArray qView = default;
            MlxNative.MlxArray kView = default;
            MlxNative.MlxArray vView = default;
            MlxNative.MlxArray qCompact = default;
            MlxNative.MlxArray kCompact = default;
            MlxNative.MlxArray vCompact = default;
            MlxNative.MlxArray q4 = default;
            MlxNative.MlxArray k4 = default;
            MlxNative.MlxArray v4 = default;
            MlxNative.MlxArray kF32 = default;
            MlxNative.MlxArray vF32 = default;
            MlxNative.MlxArray attention = default;
            MlxNative.MlxArray seqMajor = default;
            MlxNative.MlxArray contiguous = default;
            MlxNative.MlxArray flat = default;

            try
            {
                qView = GetView(qHeads);
                kView = GetView(kHeads);
                vView = GetView(vHeads);

                MlxNative.MlxArray qInput = qView;
                MlxNative.MlxArray kInputView = kView;
                MlxNative.MlxArray vInputView = vView;
                if (!qHeads.IsContiguous())
                {
                    qCompact = MlxNative.Contiguous(qView);
                    qInput = qCompact;
                }
                if (!kHeads.IsContiguous())
                {
                    kCompact = MlxNative.Contiguous(kView);
                    kInputView = kCompact;
                }
                if (!vHeads.IsContiguous())
                {
                    vCompact = MlxNative.Contiguous(vView);
                    vInputView = vCompact;
                }

                if (string.Equals(Environment.GetEnvironmentVariable("TS_MLX_HEAD_DIM256_ATTENTION"), "1", StringComparison.Ordinal)
                    && headDim == 256
                    && kHeads.ElementType == DType.Float32
                    && vHeads.ElementType == DType.Float32
                    && (string.IsNullOrEmpty(maskMode) || string.Equals(maskMode, "causal", StringComparison.Ordinal)))
                {
                    bool causal = string.Equals(maskMode, "causal", StringComparison.Ordinal);
                    int maskStart = causal ? kvLen - seqLen : 0;
                    attention = MlxNative.HeadDim256Attention(
                        qInput,
                        kInputView,
                        vInputView,
                        numHeads,
                        numKVHeads,
                        seqLen,
                        kvLen,
                        maskStart,
                        causal,
                        scale);
                    SetDeviceResult(result, attention);
                    attention = default;
                    return true;
                }

                q4 = MlxNative.Reshape(qInput, new[] { 1, numHeads, seqLen, headDim });
                k4 = MlxNative.Reshape(kInputView, new[] { 1, numKVHeads, kvLen, headDim });
                v4 = MlxNative.Reshape(vInputView, new[] { 1, numKVHeads, kvLen, headDim });

                MlxNative.MlxArray kInput = k4;
                MlxNative.MlxArray vInput = v4;
                if (kHeads.ElementType != DType.Float32)
                {
                    kF32 = MlxNative.Astype(k4, DType.Float32);
                    kInput = kF32;
                }
                if (vHeads.ElementType != DType.Float32)
                {
                    vF32 = MlxNative.Astype(v4, DType.Float32);
                    vInput = vF32;
                }

                attention = MlxNative.FastScaledDotProductAttention(
                    q4,
                    kInput,
                    vInput,
                    scale,
                    maskMode ?? string.Empty,
                    default);
                seqMajor = MlxNative.Transpose(attention, new[] { 0, 2, 1, 3 });
                contiguous = MlxNative.Contiguous(seqMajor);
                flat = MlxNative.Reshape(contiguous, new[] { seqLen, numHeads * headDim });
                SetDeviceResult(result, flat);
                flat = default;
                return true;
            }
            catch
            {
                return false;
            }
            finally
            {
                MlxNative.FreeArray(qView);
                MlxNative.FreeArray(kView);
                MlxNative.FreeArray(vView);
                MlxNative.FreeArray(qCompact);
                MlxNative.FreeArray(kCompact);
                MlxNative.FreeArray(vCompact);
                MlxNative.FreeArray(q4);
                MlxNative.FreeArray(k4);
                MlxNative.FreeArray(v4);
                MlxNative.FreeArray(kF32);
                MlxNative.FreeArray(vF32);
                MlxNative.FreeArray(attention);
                MlxNative.FreeArray(seqMajor);
                MlxNative.FreeArray(contiguous);
                MlxNative.FreeArray(flat);
            }
        }

        // 中文：分块向量化预填充注意力，按可向量化的块大小切分查询，逐块对前缀 K/V 运行因果注意力。
        private static bool TryRunChunkedVectorPrefillAttention(
            Tensor result,
            Tensor qHeads,
            Tensor kHeads,
            Tensor vHeads,
            int numHeads,
            int numKVHeads,
            int seqLen,
            int headDim,
            float scale)
        {
            int groupSize = numHeads / numKVHeads;
            int blockSize = MaxVectorQueryLen(groupSize);
            if (blockSize <= 0 || seqLen <= blockSize)
                return false;

            if (!CanUseResult(result)
                || !CanUseAttentionTensor(qHeads)
                || !CanUseAttentionTensor(kHeads)
                || !CanUseAttentionTensor(vHeads)
                || qHeads.ElementType != DType.Float32
                || headDim != 256
                || result.Sizes.Length != 2
                || result.Sizes[0] != seqLen
                || result.Sizes[1] != (long)numHeads * headDim)
            {
                return false;
            }

            try
            {
                for (int offset = 0; offset < seqLen; offset += blockSize)
                {
                    int blockLen = Math.Min(blockSize, seqLen - offset);
                    int prefixLen = offset + blockLen;

                    using Tensor outBlock = result.Narrow(0, offset, blockLen);
                    using Tensor qBlock = qHeads.Narrow(1, offset, blockLen);
                    using Tensor kPrefix = kHeads.Narrow(1, 0, prefixLen);
                    using Tensor vPrefix = vHeads.Narrow(1, 0, prefixLen);

                    if (!TryRunHeadFirstAttention(outBlock, qBlock, kPrefix, vPrefix,
                        numHeads, numKVHeads, blockLen, prefixLen, headDim,
                        blockLen == 1 ? string.Empty : "causal",
                        scale))
                    {
                        return false;
                    }
                }

                return true;
            }
            catch
            {
                return false;
            }
        }

        // 中文：判断给定的头数/序列长/headDim 是否落在 MLX 快 SDPA 内核支持范围内。
        private static bool CanUseFastAttention(int numHeads, int numKVHeads, int seqLen, int kvLen, int headDim)
        {
            if (seqLen <= 0 || kvLen <= 0 || seqLen > kvLen || numKVHeads <= 0 || numHeads % numKVHeads != 0)
                return false;

            // mx.fast.scaled_dot_product_attention dispatches to either the
            // vector (seqLen<=8) or tiled-matrix (seqLen>8) Metal kernel
            // depending on shape. Both paths now support head_dim ∈
            // {64, 80, 96, 128, 256} and (for the vector kernel) 512, with
            // arbitrary kv length. We mirror omlx and dispatch any head
            // dim within MLX's documented range; the caller's try/catch
            // around FastScaledDotProductAttention provides a safe
            // fallback if MLX rejects a shape at runtime.
            int maxHeadDim = MaxFastAttentionHeadDim();
            if (headDim <= 0 || headDim > Math.Max(512, maxHeadDim))
                return false;

            return true;
        }

        // 中文：根据 GQA 分组大小计算向量内核可处理的最大查询长度（min(8, 32/groupSize)）。
        private static int MaxVectorQueryLen(int groupSize)
        {
            if (groupSize <= 0)
                return 0;
            return Math.Min(8, 32 / groupSize);
        }

        // 中文：从环境变量读取快注意力允许的最大 headDim，缺省返回 256。
        private static int MaxFastAttentionHeadDim()
        {
            string value = Environment.GetEnvironmentVariable("TS_MLX_SDPA_MAX_HEAD_DIM");
            if (!string.IsNullOrWhiteSpace(value) && int.TryParse(value, out int parsed) && parsed > 0)
                return parsed;
            return 256;
        }

        // 中文：校验张量是否为可作结果使用的 MLX 后端、float32、连续张量。
        private static bool CanUseResult(Tensor tensor)
        {
            return tensor != null
                && tensor.Storage is MlxStorage
                && tensor.ElementType == DType.Float32
                && tensor.IsContiguous();
        }

        // 中文：校验张量是否为可用的 MLX 后端、int32、一维、连续向量（如索引）。
        private static bool CanUseInt32Vector(Tensor tensor)
        {
            return tensor != null
                && tensor.Storage is MlxStorage
                && tensor.ElementType == DType.Int32
                && tensor.DimensionCount == 1
                && tensor.IsContiguous();
        }

        // 中文：校验张量是否为注意力可用的 MLX 后端、float32 或 float16 张量。
        private static bool CanUseAttentionTensor(Tensor tensor)
        {
            return tensor != null
                && tensor.Storage is MlxStorage
                && (tensor.ElementType == DType.Float32 || tensor.ElementType == DType.Float16);
        }

        // 中文：校验张量是否为 GDN 内核可用的 MLX 后端、float32、连续张量。
        private static bool CanUseGdnTensor(Tensor tensor)
        {
            return tensor != null
                && tensor.Storage is MlxStorage
                && tensor.ElementType == DType.Float32
                && tensor.IsContiguous();
        }

        // 中文：从张量的 MLX 存储创建对应的 MLX 数组视图。
        private static MlxNative.MlxArray GetView(Tensor tensor)
        {
            return ((MlxStorage)tensor.Storage).CreateArrayView(tensor);
        }

        // 中文：把计算结果写回张量存储——整存储时直接替换设备数组，否则按切片更新并释放临时数组。
        private static void SetDeviceResult(Tensor tensor, MlxNative.MlxArray output)
        {
            MlxStorage storage = (MlxStorage)tensor.Storage;
            if (tensor.StorageOffset == 0 && tensor.Storage.ElementCount == tensor.ElementCount())
            {
                storage.ReplaceDeviceArray(output);
            }
            else
            {
                storage.UpdateDeviceSlice(tensor, output);
                MlxNative.FreeArray(output);
            }
        }
    }
}
