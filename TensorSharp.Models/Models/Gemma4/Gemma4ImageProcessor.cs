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
using System.IO;

namespace TensorSharp.Models
{
    public class Gemma4ImageProcessor
    {
        public int PatchSize { get; }
        public int NMerge { get; }

        private readonly int _minPixels;
        private readonly int _maxPixels;
        // When set (e.g. the gemma4uv unified embedder declares mean=0, std=1),
        // pixels are normalized as (pixel/255 - mean) / std instead of the legacy
        // SigLIP [-1, 1] mapping used by the gemma4v path.
        private readonly float[] _imageMean;
        private readonly float[] _imageStd;

        // 中文：构造图像处理器，依据 patch 尺寸、合并系数与 token 上下限计算像素范围，并配置可选的均值/标准差归一化参数。
        public Gemma4ImageProcessor(int patchSize = 16, int nMerge = 3, int minTokens = 40, int maxTokens = 280,
            float[] imageMean = null, float[] imageStd = null)
        {
            PatchSize = patchSize;
            NMerge = nMerge;
            int patchArea = patchSize * patchSize * nMerge * nMerge;
            _minPixels = minTokens * patchArea;
            _maxPixels = maxTokens * patchArea;
            _imageMean = imageMean != null && imageMean.Length >= 3 ? imageMean : null;
            _imageStd = imageStd != null && imageStd.Length >= 3 ? imageStd : null;
        }

        /// <summary>
        /// Process an image file into normalized pixel values in channel-first format [C, H, W].
        /// Mirrors llama.cpp's gemma4 vision preprocessing (mtmd_image_preprocessor_dyn_size):
        ///   1. Pick a canvas size via "smart resize" (calc_size_preserved_ratio) that keeps
        ///      min_pixels &lt;= W*H &lt;= max_pixels while aligning each side to patch*merge.
        ///   2. Resize the image into that canvas preserving aspect ratio (PAD_CEIL), centering
        ///      the content and letter-boxing the remainder with the padding colour (black).
        /// Normalization is (pixel/255 - mean) / std when the mmproj declares mean/std
        /// (gemma4uv unified embedder, mean=0 std=1), otherwise the legacy SigLIP [-1,1] map.
        /// Returns pixel data and the actual canvas dimensions.
        /// </summary>
        // 中文：读取并解码图像文件，按比例计算目标画布尺寸，做信箱填充与归一化，返回通道优先的像素及画布宽高。
        public (float[] pixels, int width, int height) ProcessImage(string imagePath)
        {
            byte[] fileBytes = File.ReadAllBytes(imagePath);
            int origWidth, origHeight;
            byte[] rgba = Gemma3ImageProcessor.DecodeImageToRGBA(fileBytes, out origWidth, out origHeight);

            int alignSize = PatchSize * NMerge;
            CalcSizePreservedRatio(origWidth, origHeight, alignSize, _minPixels, _maxPixels,
                out int targetW, out int targetH);

            float[] pixels = BuildLetterboxedNormalized(rgba, origWidth, origHeight, targetW, targetH);
            return (pixels, targetW, targetH);
        }

        /// <summary>
        /// Compute the resized canvas dimensions preserving aspect ratio so that
        /// <c>min_pixels &lt;= W*H &lt;= max_pixels</c>, with each side aligned to
        /// <paramref name="alignSize"/> (patch_size * n_merge). Direct port of
        /// llama.cpp <c>img_tool::calc_size_preserved_ratio(..., min_pixels, max_pixels)</c>
        /// (the "smart_resize" used by the Qwen/Gemma transformers processors).
        /// </summary>
        // 中文：保持长宽比计算目标画布尺寸，使总像素落在 [minPixels, maxPixels] 区间且各边对齐到 alignSize（smart_resize 移植）。
        internal static void CalcSizePreservedRatio(int width, int height, int alignSize,
            int minPixels, int maxPixels, out int targetW, out int targetH)
        {
            if (width <= 0 || height <= 0 || alignSize <= 0)
            {
                targetW = alignSize;
                targetH = alignSize;
                return;
            }

            // std::round rounds halves away from zero; match that exactly.
            int RoundBy(double x) => (int)Math.Round(x / alignSize, MidpointRounding.AwayFromZero) * alignSize;
            int CeilBy(double x) => (int)Math.Ceiling(x / alignSize) * alignSize;
            int FloorBy(double x) => (int)Math.Floor(x / alignSize) * alignSize;

            // Always align up (round) first.
            int hBar = Math.Max(alignSize, RoundBy(height));
            int wBar = Math.Max(alignSize, RoundBy(width));

            long area = (long)hBar * wBar;
            if (maxPixels > 0 && area > maxPixels)
            {
                double beta = Math.Sqrt((double)height * width / maxPixels);
                hBar = Math.Max(alignSize, FloorBy(height / beta));
                wBar = Math.Max(alignSize, FloorBy(width / beta));
            }
            else if (minPixels > 0 && area < minPixels)
            {
                double beta = Math.Sqrt((double)minPixels / ((double)height * width));
                hBar = CeilBy(height * beta);
                wBar = CeilBy(width * beta);
            }

            targetW = wBar;
            targetH = hBar;
        }

        /// <summary>
        /// Resize the source image into a <paramref name="targetW"/> x <paramref name="targetH"/>
        /// canvas preserving aspect ratio (llama.cpp PAD_CEIL): the content is scaled down/up by
        /// the smaller of the per-axis scales, centred in the canvas, and the surrounding border is
        /// filled with the (normalized) padding colour black. The result is channel-first [C,H,W].
        /// </summary>
        // 中文：按比例缩放内容并居中放入目标画布，周边以归一化黑色填充（信箱化），输出通道优先 [C,H,W] 归一化像素。
        private float[] BuildLetterboxedNormalized(byte[] rgba, int origW, int origH, int targetW, int targetH)
        {
            float scale = Math.Min((float)targetW / origW, (float)targetH / origH);
            int newW = Math.Min((int)Math.Ceiling(origW * (double)scale), targetW);
            int newH = Math.Min((int)Math.Ceiling(origH * (double)scale), targetH);
            newW = Math.Max(1, newW);
            newH = Math.Max(1, newH);
            int offsetX = (targetW - newW) / 2;
            int offsetY = (targetH - newH) / 2;

            bool hasMeanStd = _imageMean != null && _imageStd != null;

            // Resize the content preserving aspect ratio into [C, newH, newW].
            float[] content = hasMeanStd
                ? Gemma3ImageProcessor.ResizeRgbaToChannelFirstNormalized(rgba, origW, origH, newW, newH, _imageMean, _imageStd)
                : Gemma3ImageProcessor.ResizeRgbaToChannelFirstNormalized(rgba, origW, origH, newW, newH);

            int targetPixels = targetW * targetH;
            int contentPixels = newW * newH;
            float[] result = new float[3 * targetPixels];

            for (int c = 0; c < 3; c++)
            {
                // Normalized value of a black pad pixel (0/255 == 0) under the active scheme.
                float padVal = hasMeanStd ? (0f - _imageMean[c]) / _imageStd[c] : -1f;
                int cBase = c * targetPixels;

                for (int i = 0; i < targetPixels; i++)
                    result[cBase + i] = padVal;

                for (int y = 0; y < newH; y++)
                {
                    int srcRow = c * contentPixels + y * newW;
                    int dstRow = cBase + (y + offsetY) * targetW + offsetX;
                    Array.Copy(content, srcRow, result, dstRow, newW);
                }
            }

            return result;
        }

        // 中文：根据图像宽高计算切分 patch 并按 NMerge 合并后的输出 token 数量。
        public int ComputeOutputTokens(int imageWidth, int imageHeight)
        {
            int patchesX = imageWidth / PatchSize;
            int patchesY = imageHeight / PatchSize;
            int mergedX = patchesX / NMerge;
            int mergedY = patchesY / NMerge;
            return mergedX * mergedY;
        }
    }
}

