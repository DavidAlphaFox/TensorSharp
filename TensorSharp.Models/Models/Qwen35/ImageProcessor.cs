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
    public class Qwen35ImageProcessor
    {
        public int PatchSize { get; }
        public int MergeSize { get; }
        public int Factor { get; }
        public int ShortestEdge { get; }
        public int LongestEdge { get; }

        // 中文：构造函数，初始化 patch 尺寸、合并尺寸、对齐因子（patchSize*mergeSize）以及面积上下限。
        public Qwen35ImageProcessor(int patchSize = 14, int mergeSize = 2,
            int shortestEdge = 64 * 1024, int longestEdge = 2 * 1024 * 1024)
        {
            PatchSize = patchSize;
            MergeSize = mergeSize;
            Factor = patchSize * mergeSize;
            ShortestEdge = shortestEdge;
            LongestEdge = longestEdge;
        }

        // 中文：读取图片文件的宽高尺寸（委托给 Gemma3 图像处理器）。
        public static (int width, int height) ReadImageDimensions(string path)
        {
            return Gemma3ImageProcessor.ReadImageDimensions(path);
        }

        // 中文：智能缩放，将高宽对齐到 factor 的整数倍，并在超出面积上限或低于下限时按比例缩放回区间。
        public (int height, int width) SmartResize(int height, int width)
        {
            int factor = Factor;
            if (height < factor || width < factor)
                throw new ArgumentException($"Image too small: {height}x{width}, minimum {factor}x{factor}");

            int hBar = (int)Math.Round((double)height / factor, MidpointRounding.ToEven) * factor;
            int wBar = (int)Math.Round((double)width / factor, MidpointRounding.ToEven) * factor;

            if ((long)hBar * wBar > LongestEdge)
            {
                double beta = Math.Sqrt((double)height * width / LongestEdge);
                hBar = (int)Math.Floor(height / beta / factor) * factor;
                wBar = (int)Math.Floor(width / beta / factor) * factor;
            }
            else if ((long)hBar * wBar < ShortestEdge)
            {
                double beta = Math.Sqrt((double)ShortestEdge / (height * width));
                hBar = (int)Math.Ceiling(height * beta / factor) * factor;
                wBar = (int)Math.Ceiling(width * beta / factor) * factor;
            }

            return (hBar, wBar);
        }

        // 中文：根据原始高宽计算图像最终生成的视觉 token 数（缩放后网格按合并尺寸下采样的乘积）。
        public int ComputeImageTokenCount(int origHeight, int origWidth)
        {
            var (resizedH, resizedW) = SmartResize(origHeight, origWidth);
            int gridH = resizedH / PatchSize;
            int gridW = resizedW / PatchSize;
            return (gridH / MergeSize) * (gridW / MergeSize);
        }

        // 中文：从图片路径读取尺寸并计算其视觉 token 数。
        public int ComputeImageTokenCount(string imagePath)
        {
            var (width, height) = ReadImageDimensions(imagePath);
            return ComputeImageTokenCount(height, width);
        }

        // 中文：根据原始高宽返回缩放后的 patch 网格高宽（resized 尺寸除以 patchSize）。
        public (int gridHeight, int gridWidth) GetPatchGrid(int origHeight, int origWidth)
        {
            var (resizedH, resizedW) = SmartResize(origHeight, origWidth);
            return (resizedH / PatchSize, resizedW / PatchSize);
        }

        /// <summary>
        /// Full image processing pipeline: load, composite, resize, normalize to channel-first float array.
        /// Returns (normalizedPixels, resizedHeight, resizedWidth).
        /// </summary>
        // 中文：完整图像处理流水线：读取文件、解码为 RGBA、智能缩放、归一化为通道优先 float 数组并返回缩放后尺寸。
        public (float[] pixels, int resizedH, int resizedW) ProcessImage(string imagePath)
        {
            byte[] fileBytes = File.ReadAllBytes(imagePath);
            byte[] rgba = Gemma3ImageProcessor.DecodeImageToRGBA(fileBytes, out int origWidth, out int origHeight);

            var (resizedH, resizedW) = SmartResize(origHeight, origWidth);
            float[] pixels = Gemma3ImageProcessor.ResizeRgbaToChannelFirstNormalized(
                rgba, origWidth, origHeight, resizedW, resizedH);
            return (pixels, resizedH, resizedW);
        }
    }
}

