// 与 csrc/ops/low_latency.cpp 中 pybind 入口一一对应的 host 侧封装。
// 只做参数/缓冲准备与 launch，不改 kernel、布局和 tiling 算法；
// 指针顺序与 Python 版本保持一致，改布局时必须两边同时改。
#pragma once
#include <cstdint>

#include "runtime/session.hpp"

struct LlDispatchTensors {
    void* input = nullptr;               // [b, h]        bf16 / fp16
    void* indices = nullptr;             // [b, k]        int32
    void* weights = nullptr;             // [b, k]        float32
    void* output = nullptr;              // [rows, h]     bf16 / fp16
    void* expert_recv_counts = nullptr;  // [e / world]   int64
    void* send_counts = nullptr;         // [e]           int32
    void* token_type = nullptr;          // [rows]        int32
    void* destination_index = nullptr;   // [rows, 3]     int32
    void* relay_read_index = nullptr;    // [rows, k, 2]  int32
    void* source_mask = nullptr;         // [b]           int32
    void* source_weights = nullptr;      // [b, k]        float32
    void* comm_args = nullptr;           // DispatchDedup::CommArgs
    void* tiling = nullptr;              // DispatchDedup::EpServerDedupTilingData
};

// 输出行容量，等价于 Python 的 low_latency_layout()["capacity_rows"]。
uint32_t ll_dispatch_capacity_rows(uint32_t world, int b, int h, int k, int e);

// 等价于 low_latency_comm_args()：填好 CommArgs 后 H2D 到 device_comm_args。
void ll_dispatch_make_comm_args(deepep::Session& session, void* device_comm_args);

// 等价于 low_latency_dispatch_tiling()：填好 tiling 后 H2D 到 device_tiling。
void ll_dispatch_make_tiling(uint32_t world, int b, int h, int k, int e, int capacity, bool fp16,
                             void* device_tiling);

// 等价于 dispatch_low_latency()：组装 EpServerDedupKernelArgs 并下发 kernel。
void ll_dispatch_run(deepep::Session& session, void* stream, const LlDispatchTensors& tensors, int b, int h, int k,
                     int e, int rows);
