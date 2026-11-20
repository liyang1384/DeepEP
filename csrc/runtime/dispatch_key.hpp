// Copyright (c) 2026, Lu Lu

// One DeepJIT specialization identity for LL dispatch. Every value becomes a
// template argument; runtime values (actual tokens, pointers) are launch
// parameters. The key is computed once in csrc/ops/low_latency.cpp.
#pragma once

#include <cstdint>
#include <string>

namespace deep_ep {

struct DispatchKey {
    uint32_t num_ranks = 0U;
    uint32_t num_max_tokens_per_rank = 0U;
    uint32_t num_hidden_bytes = 0U;
    uint32_t num_topk = 0U;
    uint32_t num_experts = 0U;
    uint32_t num_vec_cores = 0U;
    uint32_t input_class = 0U;
    uint32_t num_ranks_per_group = 0U;
    uint32_t input_mode = 0U;
    bool e4m3 = false;
    bool dedup = false;
    bool rdma = false;

    std::string ToArgs() const
    {
        return std::to_string(num_ranks) + ", " + std::to_string(num_max_tokens_per_rank) + ", " +
               std::to_string(num_hidden_bytes) + ", " + std::to_string(num_topk) + ", " +
               std::to_string(num_experts) + ", " + std::to_string(num_vec_cores) + ", " +
               std::to_string(input_class) + ", " + std::to_string(num_ranks_per_group) + ", " +
               std::to_string(input_mode) + ", " + (e4m3 ? "true" : "false") + ", " +
               (dedup ? "true" : "false") + ", " + (rdma ? "true" : "false");
    }
};

}  // namespace deep_ep
