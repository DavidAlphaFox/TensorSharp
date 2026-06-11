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

namespace TensorSharp.Core
{
    public class DelegateDisposable : IDisposable
    {
        private readonly Action action;

        // 中文：构造函数，保存在释放时要执行的委托动作。
        public DelegateDisposable(Action action)
        {
            this.action = action;
        }

        // 中文：释放时调用所保存的委托，实现 using 退出时执行自定义清理逻辑。
        public virtual void Dispose()
        {
            action();
        }
    }
}
