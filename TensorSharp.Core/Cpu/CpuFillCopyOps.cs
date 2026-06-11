// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
﻿// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using AdvUtils;
using System;
using System.Reflection;

namespace TensorSharp.Cpu
{
    [OpsClass]
    public class CpuFillCopyOps
    {
        // 中文：默认构造函数，无额外初始化。
        public CpuFillCopyOps()
        {
        }

        [RegisterOpStorageType("fill", typeof(CpuStorage))]
        // 中文：用标量 value 填充整个结果张量。
        public void Fill(Tensor result, float value)
        {
            TensorApplyCPU.Fill(result, value);
        }


        [RegisterOpStorageType("copy", typeof(CpuStorage))]
        // 中文：将 src 张量逐元素拷贝到结果张量，校验元素数量一致并记录异常日志。
        public void Copy(Tensor result, Tensor src)
        {
            try
            {
                var resEC = result.ElementCount();
                var srcEC = src.ElementCount();
                if (resEC != srcEC)
                {
                    throw new InvalidOperationException($"Tensors must have equal numbers of elements. result element count = '{resEC}', source element count = '{srcEC}', result tensor = '{result.ToString()}', source tensor = '{src.ToString()}'");
                }

                TensorApplyCPU.Copy(result, src);
            }
            catch (Exception err)
            {
                Logger.WriteLine(Logger.Level.err, $"Failed to run Copy operation on CPU. Message = '{err.Message}'.");
                Logger.WriteLine(Logger.Level.debug, $"Call stack = '{err.StackTrace}'");
                throw;
            }
        }

    }
}
