// ──────【文件说明】──────
// 文件：MlxMemorySnapshot.cs
// 用途：定义 MLX 后端内存快照结构，用于记录 MLX 设备在某一时刻的内存使用状态，
//       包括活跃内存、缓存内存和峰值内存，便于性能监控与内存分析。
// 主要类型：MlxMemorySnapshot（只读记录结构体）
// ────────────────────────

namespace TensorSharp.MLX
{
    // 中文：表示 MLX 后端在某一时刻的内存使用快照，包含活跃字节数、缓存字节数和峰值字节数
    public readonly record struct MlxMemorySnapshot(ulong ActiveBytes, ulong CacheBytes, ulong PeakBytes);
}
