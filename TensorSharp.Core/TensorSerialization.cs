// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
﻿using System;
using System.IO;
using System.Runtime.InteropServices;

namespace TensorSharp
{
    public static class TensorSerialization
    {
        // 中文：将张量（先转为连续内存）的维度、类型、尺寸与原始字节写入流。
        public static void Serialize(Tensor tensor, Stream stream)
        {
            using (Tensor src = Ops.AsContiguous(tensor))
            {
                // Note: don't dispose writer - it does not own the stream's lifetime
                BinaryWriter writer = new System.IO.BinaryWriter(stream);

                // Can infer strides - src is contiguous
                writer.Write(tensor.DimensionCount); // int32
                writer.Write((int)tensor.ElementType);
                for (int i = 0; i < tensor.DimensionCount; ++i)
                {
                    writer.Write(tensor.Sizes[i]);
                }

                long byteCount = src.ElementType.Size() * tensor.ElementCount();
                writer.Write(byteCount);
                WriteBytes(writer, src.Storage, src.StorageOffset, byteCount);

                writer.Flush();
            }
        }

        // 中文：从流读取维度、类型、尺寸与字节数据，重建并返回张量。
        public static Tensor Deserialize(IAllocator allocator, Stream stream)
        {
            // Note: don't dispose reader - it does not own the stream's lifetime
            BinaryReader reader = new BinaryReader(stream);

            int dimCount = reader.ReadInt32();
            DType elementType = (DType)reader.ReadInt32();
            long[] sizes = new long[dimCount];
            for (int i = 0; i < dimCount; ++i)
            {
                sizes[i] = reader.ReadInt64();
            }

            long byteCount = reader.ReadInt64();
            Tensor result = new Tensor(allocator, elementType, sizes);

            ReadBytes(reader, result.Storage, result.StorageOffset, byteCount);

            return result;
        }

        // 中文：通过固定缓冲区分块从存储拷出字节并写入 writer。
        private static void WriteBytes(BinaryWriter writer, Storage storage, long startIndex, long byteCount)
        {
            byte[] buffer = new byte[4096];
            GCHandle bufferHandle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
            try
            {
                long curStart = startIndex;
                long afterLastByte = startIndex + byteCount;
                while (curStart < afterLastByte)
                {
                    int length = (int)Math.Min(buffer.Length, afterLastByte - curStart);
                    storage.CopyFromStorage(bufferHandle.AddrOfPinnedObject(), curStart, length);
                    writer.Write(buffer, 0, length);
                    curStart += length;
                }
            }
            finally
            {
                bufferHandle.Free();
            }
        }

        // 中文：通过固定缓冲区分块从 reader 读字节并拷入存储。
        private static void ReadBytes(BinaryReader reader, Storage storage, long startIndex, long byteCount)
        {
            byte[] buffer = new byte[4096];
            GCHandle bufferHandle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
            try
            {
                long curStart = startIndex;
                long afterLastByte = startIndex + byteCount;
                while (curStart < afterLastByte)
                {
                    int length = (int)Math.Min(buffer.Length, afterLastByte - curStart);
                    reader.Read(buffer, 0, length);
                    storage.CopyToStorage(curStart, bufferHandle.AddrOfPinnedObject(), length);
                    curStart += length;
                }
            }
            finally
            {
                bufferHandle.Free();
            }
        }
    }
}
