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
using System.Threading;

// ───────────────────────────────────────────────────────────────────────────
// 【文件说明】线程安全的引用计数基类。
// 【主要类型】RefCounted：管理张量存储（Storage）等资源的生命周期，引用计数归零时
//             自动调用 Destroy() 释放底层内存，是显存 / 内存复用的基础。
// ───────────────────────────────────────────────────────────────────────────
namespace TensorSharp
{
    /// <summary>
    /// Provides a thread safe reference counting implementation. Inheritors need only implement the Destroy() method,
    /// which will be called when the reference count reaches zero. The reference count automatically starts at 1.
    /// </summary>

    [Serializable]
    public abstract class RefCounted
    {
        private int refCount = 1;

        /// <summary>
        /// Construct a new reference counted object. The reference count automatically starts at 1.
        /// </summary>
        // 中文：构造引用计数对象，初始引用计数为 1。
        public RefCounted()
        {
        }

        // 中文：终结器；若对象尚未释放则调用 Destroy() 释放资源。
        ~RefCounted()
        {
            if (refCount > 0)
            {
                Destroy();
                refCount = 0;
            }
        }

        /// <summary>
        /// This method is called when the reference count reaches zero. It will be called at most once to allow subclasses to release resources.
        /// </summary>
        // 中文：引用计数归零时调用，由子类实现以释放底层资源。
        protected abstract void Destroy();

        /// <summary>
        /// Returns true if the object has already been destroyed; false otherwise.
        /// </summary>
        /// <returns>true if the object is destroyed; false otherwise.</returns>
        // 中文：返回对象是否已被销毁（引用计数为 0）。
        protected bool IsDestroyed()
        {
            return refCount == 0;
        }

        /// <summary>
        /// Throws an exception if the object has been destroyed, otherwise does nothing.
        /// </summary>
        // 中文：若对象已被销毁则抛出异常，否则不做任何操作。
        protected void ThrowIfDestroyed()
        {
            if (IsDestroyed())
            {
                throw new InvalidOperationException("Reference counted object has been destroyed");
            }
        }

        // 中文：返回当前引用计数值。
        protected int GetCurrentRefCount()
        {
            return refCount;
        }

        /// <summary>
        /// Increments the reference count. If the object has previously been destroyed, an exception is thrown.
        /// </summary>
        // 中文：以无锁自旋方式原子递增引用计数；已销毁则抛异常。
        public void AddRef()
        {
            int curRefCount;
            int original;
            SpinWait spin = new SpinWait();
            while (true)
            {
                curRefCount = refCount;
                if (curRefCount == 0)
                {
                    throw new InvalidOperationException("Cannot AddRef - object has already been destroyed");
                }

                int desiredRefCount = curRefCount + 1;
                original = Interlocked.CompareExchange(ref refCount, desiredRefCount, curRefCount);
                if (original == curRefCount)
                {
                    break;
                }

                spin.SpinOnce();
            }
        }

        /// <summary>
        /// Decrements the reference count. If the reference count reaches zero, the object is destroyed.
        /// If the object has previously been destroyed, an exception is thrown.
        /// </summary>
        // 中文：以无锁自旋方式原子递减引用计数，归零时调用 Destroy() 销毁对象。
        public void Release()
        {
            int original;
            int curRefCount;
            SpinWait spin = new SpinWait();
            while (true)
            {
                curRefCount = refCount;
                if (curRefCount == 0)
                {
                    throw new InvalidOperationException("Cannot release object - object has already been destroyed");
                }

                int desiredRefCount = refCount - 1;
                original = Interlocked.CompareExchange(ref refCount, desiredRefCount, curRefCount);
                if (original == curRefCount)
                {
                    break;
                }

                spin.SpinOnce();
            }

            if (refCount <= 0)
            {
                Destroy();
            }
        }
    }
}
