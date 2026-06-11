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
using Microsoft.AspNetCore.Hosting;

namespace TensorSharp.Server.Hosting
{
    /// <summary>
    /// Resolves the wwwroot directory used to serve the Web UI's static assets.
    /// When the standard ASP.NET resolution fails (e.g. <c>dotnet run</c> from a
    /// non-standard working directory), we look for a sibling <c>wwwroot</c>
    /// folder relative to the executable. If neither exists we fall back to a
    /// fresh empty directory so the server can still start.
    /// </summary>
    internal static class WebRootSetup
    {
        // 中文：解析并确保 Web UI 静态资源的 wwwroot 目录存在，标准解析失败时回退到可执行文件旁的目录或新建空目录。
        public static void Resolve(IWebHostEnvironment environment, string baseDirectory)
        {
            if (environment == null) throw new ArgumentNullException(nameof(environment));
            if (string.IsNullOrEmpty(baseDirectory)) throw new ArgumentNullException(nameof(baseDirectory));

            string webRoot = environment.WebRootPath;
            if (!string.IsNullOrEmpty(webRoot) && Directory.Exists(webRoot))
                return;

            string srcWwwRoot = Path.Combine(baseDirectory, "..", "wwwroot");
            if (Directory.Exists(srcWwwRoot))
                webRoot = Path.GetFullPath(srcWwwRoot);
            else
                webRoot = Path.Combine(baseDirectory, "wwwroot");

            environment.WebRootPath = webRoot;
            Directory.CreateDirectory(webRoot);
        }
    }
}
