// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

// ──────【文件说明】──────
// 文件：MediaHelper.cs
// 用途：提供视频帧提取与 PNG 图像编码的辅助工具，用于将视频内容转换为
//       多模态 LLM 推理所需的图像帧序列。支持基于时间的均匀采样策略，
//       并内置零依赖的纯托管 PNG 编码器（含 zlib/Deflate 压缩与 CRC32 校验）。
// 主要类型：MediaHelper（静态工具类）
// ────────────────────────

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Runtime.InteropServices;
using OpenCvSharp;

namespace TensorSharp.Models
{
    public static class MediaHelper
    {
        /// <summary>
        /// Frames sampled per second of video when no explicit rate is supplied.
        /// Overridable via the <c>VIDEO_SAMPLE_FPS</c> environment variable.
        /// </summary>
        public const double DefaultVideoSampleFps = 1.0;

        /// <summary>
        /// Default upper bound on the number of extracted frames. <c>0</c> means
        /// "no cap": extraction is purely time-based at the sampling fps. Set the
        /// <c>VIDEO_MAX_FRAMES</c> environment variable to a positive value to
        /// bound long videos.
        /// </summary>
        public const int DefaultVideoMaxFrames = 0;

        // 中文：从环境变量 VIDEO_SAMPLE_FPS 读取视频帧采样率（帧/秒），无效时回退到默认值
        /// <summary>
        /// Resolves the sampling rate (frames per second of video) from the
        /// <c>VIDEO_SAMPLE_FPS</c> environment variable, falling back to
        /// <see cref="DefaultVideoSampleFps"/> when unset or invalid.
        /// </summary>
        public static double GetConfiguredVideoSampleFps(double fallback = DefaultVideoSampleFps)
        {
            string raw = Environment.GetEnvironmentVariable("VIDEO_SAMPLE_FPS");
            if (!string.IsNullOrWhiteSpace(raw) &&
                double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out double parsed) &&
                parsed > 0)
            {
                return parsed;
            }

            return fallback > 0 ? fallback : DefaultVideoSampleFps;
        }

        // 中文：从环境变量 VIDEO_MAX_FRAMES 读取最大提取帧数上限，返回 0 表示不限制
        /// <summary>
        /// Resolves the optional upper bound on extracted frames from the
        /// <c>VIDEO_MAX_FRAMES</c> environment variable. Returns <c>0</c> (no cap)
        /// when unset or invalid so that extraction stays purely time-based.
        /// </summary>
        public static int GetConfiguredMaxVideoFrames(int fallback = DefaultVideoMaxFrames)
        {
            string raw = Environment.GetEnvironmentVariable("VIDEO_MAX_FRAMES");
            if (!string.IsNullOrWhiteSpace(raw) &&
                int.TryParse(raw, out int parsed) &&
                parsed > 0)
            {
                return parsed;
            }

            return fallback > 0 ? fallback : 0;
        }

        // 中文：基于时间均匀采样从视频文件中提取帧，将每帧保存为临时 PNG 文件并返回路径列表；
        //       当设置了最大帧数上限时，对候选帧做二次均匀降采样以防止超长视频撑爆上下文窗口
        /// <summary>
        /// Extracts frames from a video using time-based sampling: one frame every
        /// <c>1 / fps</c> seconds (default 1 fps), so the frame count scales with the
        /// clip's duration. When <paramref name="maxFrames"/> resolves to a positive
        /// value (via the <c>VIDEO_MAX_FRAMES</c> env var or an explicit argument) it
        /// acts as an upper bound, evenly down-selecting; otherwise every sampled
        /// frame is kept.
        /// </summary>
        /// <param name="videoPath">Path to the source video file.</param>
        /// <param name="maxFrames">
        /// Optional cap on the number of frames. <c>&lt;= 0</c> resolves from
        /// <c>VIDEO_MAX_FRAMES</c> (default: no cap).
        /// </param>
        /// <param name="fps">
        /// Sampling rate in frames per second of video. <c>&lt;= 0</c> resolves from
        /// <c>VIDEO_SAMPLE_FPS</c> (default: 1 fps).
        /// </param>
        public static List<string> ExtractVideoFrames(string videoPath, int maxFrames = 0, double fps = 0.0)
        {
            if (maxFrames <= 0)
                maxFrames = GetConfiguredMaxVideoFrames();
            if (fps <= 0)
                fps = GetConfiguredVideoSampleFps();

            string tempDir = Path.Combine(Path.GetTempPath(), $"frames_{Guid.NewGuid():N}");
            Directory.CreateDirectory(tempDir);

            using var capture = new VideoCapture(videoPath);
            if (!capture.IsOpened())
                throw new Exception($"Failed to open video file: {videoPath}");

            double videoFps = capture.Get(VideoCaptureProperties.Fps);
            int totalFrames = (int)capture.Get(VideoCaptureProperties.FrameCount);
            if (videoFps <= 0 || totalFrames <= 0)
                throw new Exception($"Invalid video: fps={videoFps}, frames={totalFrames}");

            // Time-based candidate sampling: pick one frame every (videoFps / fps)
            // source frames, i.e. one frame per (1 / fps) seconds of wall-clock time.
            int frameInterval = Math.Max(1, (int)Math.Round(videoFps / fps));
            var candidateFrames = new List<int>();
            for (int frameIdx = 0; frameIdx < totalFrames; frameIdx += frameInterval)
                candidateFrames.Add(frameIdx);

            // Keep every time-sampled frame by default. VIDEO_MAX_FRAMES (when > 0)
            // is an optional upper bound that evenly down-selects to protect against
            // runaway frame counts / context blow-up on very long clips.
            List<int> selectedPositions;
            if (maxFrames > 0 && candidateFrames.Count > maxFrames)
            {
                selectedPositions = SelectEvenlySpacedIndices(candidateFrames.Count, maxFrames);
            }
            else
            {
                selectedPositions = new List<int>(candidateFrames.Count);
                for (int i = 0; i < candidateFrames.Count; i++)
                    selectedPositions.Add(i);
            }

            var frames = new List<string>();
            using var mat = new Mat();

            foreach (int pos in selectedPositions)
            {
                int frameIdx = candidateFrames[pos];

                capture.Set(VideoCaptureProperties.PosFrames, frameIdx);
                if (!capture.Read(mat) || mat.Empty())
                    break;

                string framePath = Path.Combine(tempDir, $"frame_{frames.Count + 1:D4}.png");
                SaveMatAsPng(mat, framePath);
                frames.Add(framePath);
            }

            return frames;
        }

        // 中文：在 [0, count) 范围内均匀选取 maxCount 个索引，用于对候选帧列表做二次降采样
        public static List<int> SelectEvenlySpacedIndices(int count, int maxCount)
        {
            var indices = new List<int>();
            if (count <= 0 || maxCount <= 0)
                return indices;

            if (count <= maxCount)
            {
                for (int i = 0; i < count; i++)
                    indices.Add(i);
                return indices;
            }

            if (maxCount == 1)
            {
                indices.Add(count / 2);
                return indices;
            }

            double step = (double)(count - 1) / (maxCount - 1);
            int previous = -1;
            for (int i = 0; i < maxCount; i++)
            {
                int idx = (int)Math.Round(i * step);
                if (idx <= previous)
                    idx = previous + 1;
                if (idx >= count)
                    idx = count - 1;

                indices.Add(idx);
                previous = idx;
            }

            return indices;
        }

        // 中文：将 OpenCV Mat 帧（BGR/BGRA）转换为符合 PNG 规范的 RGBA 图像并写入磁盘；
        //       使用纯托管代码手工构造 PNG 文件结构（IHDR/IDAT/IEND），
        //       图像数据经 zlib（Deflate + Adler-32）压缩后写入 IDAT 块
        private static void SaveMatAsPng(Mat mat, string path)
        {
            int width = mat.Cols;
            int height = mat.Rows;
            int channels = mat.Channels();
            int step = (int)mat.Step();

            int rowStride = 1 + width * 4;
            byte[] rawRows = new byte[height * rowStride];

            unsafe
            {
                byte* src = (byte*)mat.Data;
                for (int y = 0; y < height; y++)
                {
                    int dstRowStart = y * rowStride;
                    rawRows[dstRowStart] = 0; // PNG filter: None
                    byte* row = src + y * step;
                    for (int x = 0; x < width; x++)
                    {
                        int dstOff = dstRowStart + 1 + x * 4;
                        int srcOff = x * channels;
                        rawRows[dstOff]     = row[srcOff + 2]; // R (BGR→RGB)
                        rawRows[dstOff + 1] = row[srcOff + 1]; // G
                        rawRows[dstOff + 2] = row[srcOff];     // B
                        rawRows[dstOff + 3] = channels >= 4 ? row[srcOff + 3] : (byte)255;
                    }
                }
            }

            byte[] compressed;
            using (var ms = new MemoryStream())
            {
                ms.WriteByte(0x78); // zlib header
                ms.WriteByte(0x01);
                using (var deflate = new DeflateStream(ms, CompressionLevel.Fastest, true))
                    deflate.Write(rawRows, 0, rawRows.Length);

                uint a = 1, b = 0;
                for (int i = 0; i < rawRows.Length; i++)
                {
                    a = (a + rawRows[i]) % 65521;
                    b = (b + a) % 65521;
                }
                uint adler = (b << 16) | a;
                ms.WriteByte((byte)(adler >> 24));
                ms.WriteByte((byte)(adler >> 16));
                ms.WriteByte((byte)(adler >> 8));
                ms.WriteByte((byte)adler);
                compressed = ms.ToArray();
            }

            using var fs = File.Create(path);
            fs.Write(new byte[] { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A }, 0, 8);
            WritePngChunk(fs, "IHDR", BuildIHDR(width, height));
            WritePngChunk(fs, "IDAT", compressed);
            WritePngChunk(fs, "IEND", Array.Empty<byte>());
        }

        // 中文：构造 PNG IHDR 块的 13 字节负载，指定图像宽高、位深（8 位）及颜色类型（RGBA=6）
        private static byte[] BuildIHDR(int width, int height)
        {
            byte[] ihdr = new byte[13];
            ihdr[0] = (byte)(width >> 24); ihdr[1] = (byte)(width >> 16);
            ihdr[2] = (byte)(width >> 8);  ihdr[3] = (byte)width;
            ihdr[4] = (byte)(height >> 24); ihdr[5] = (byte)(height >> 16);
            ihdr[6] = (byte)(height >> 8);  ihdr[7] = (byte)height;
            ihdr[8] = 8;  // bit depth
            ihdr[9] = 6;  // color type RGBA
            return ihdr;
        }

        // 中文：向流写入一个完整的 PNG 数据块（长度 + 类型 + 数据 + CRC32 校验值）
        private static void WritePngChunk(Stream s, string type, byte[] data)
        {
            byte[] lenBuf = { (byte)(data.Length >> 24), (byte)(data.Length >> 16),
                              (byte)(data.Length >> 8),  (byte)data.Length };
            byte[] typeBuf = System.Text.Encoding.ASCII.GetBytes(type);
            s.Write(lenBuf, 0, 4);
            s.Write(typeBuf, 0, 4);
            if (data.Length > 0) s.Write(data, 0, data.Length);

            uint crc = Crc32Png(typeBuf, data);
            byte[] crcBuf = { (byte)(crc >> 24), (byte)(crc >> 16),
                              (byte)(crc >> 8),  (byte)crc };
            s.Write(crcBuf, 0, 4);
        }

        private static readonly uint[] Crc32Table = BuildCrc32Table();

        // 中文：预计算 CRC32 查表（IEEE 802.3 多项式 0xEDB88320 反转形式），用于加速 PNG 块校验
        private static uint[] BuildCrc32Table()
        {
            var t = new uint[256];
            for (uint n = 0; n < 256; n++)
            {
                uint c = n;
                for (int k = 0; k < 8; k++)
                    c = (c & 1) != 0 ? 0xEDB88320 ^ (c >> 1) : c >> 1;
                t[n] = c;
            }
            return t;
        }

        // 中文：使用查表法计算 PNG 数据块（类型字段 + 数据字段）的 CRC32 校验值
        private static uint Crc32Png(byte[] type, byte[] data)
        {
            uint crc = 0xFFFFFFFF;
            foreach (byte b in type)
                crc = Crc32Table[(crc ^ b) & 0xFF] ^ (crc >> 8);
            foreach (byte b in data)
                crc = Crc32Table[(crc ^ b) & 0xFF] ^ (crc >> 8);
            return crc ^ 0xFFFFFFFF;
        }
    }
}
