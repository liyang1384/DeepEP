// Copyright (c) 2026, Lu Lu

// Host-side LL dispatch wrapper: layout admission and the DeepJIT compile/launch.
// The device implementation is one kernel in deep_ep/include/deep_ep/impls/ep/dispatch.hpp.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/common/exception.hpp>
#include <deep_ep/layout/ep/token.hpp>
#include <deep_ep/layout/ep/workspace.hpp>

#include "runtime/dispatch_key.hpp"
#include "runtime/jit.hpp"

namespace deep_ep {

struct DispatchLaunchArgs {
    uint8_t* x = nullptr;
    uint8_t* expert_ids = nullptr;
    uint8_t* expert_scales = nullptr;
    uint8_t* expand_x_out = nullptr;
    uint8_t* dynamic_scales_out = nullptr;
    uint8_t* expert_token_nums_out = nullptr;
    uint8_t* send_counts_out = nullptr;
    uint8_t* token_type_out = nullptr;
    uint8_t* destination_index_out = nullptr;
    uint8_t* relay_read_index_out = nullptr;
    uint8_t* source_mask_out = nullptr;
    uint8_t* source_weights_out = nullptr;
    uint8_t* input_scales = nullptr;
    uint8_t* peer_ptrs = nullptr;  // GM table of per-rank buffer bases (CommArgs::peerMems)
    uint8_t* rdma_remote_base = nullptr;
    uint8_t* rdma_stage_base = nullptr;
    uint8_t* stage_timer = nullptr;
    uint8_t* expert_timer = nullptr;
    int32_t rank_idx = 0;
    int32_t num_tokens = 0;
    int32_t physical_ranks_per_host = 16;
    int32_t backend = 0;
    int32_t stage_block_num = 0;
    int64_t peer_mem_bytes = 0;
    int64_t ffts_val = 0;
    int32_t timer_mode = 1;
};

// Host admission. Mirrors the device DispatchUBLayout constants; capacity
// (not the actual token count) determines the GM and UB footprint.
inline EpMemoryServerDedupLayout get_dispatch_buffer_size(uint32_t num_ranks, uint32_t num_max_tokens_per_rank,
                                                         uint32_t num_hidden_elems, uint32_t num_topk,
                                                         uint32_t num_experts, EpServerDedupInputMode mode,
                                                         uint32_t num_vec_cores)
{
    EP_HOST_ASSERT(num_max_tokens_per_rank > 0U && num_hidden_elems > 0U);
    EP_HOST_ASSERT(num_topk > 0U && num_topk <= kEpServerDedupMaxTopK);
    EP_HOST_ASSERT(num_experts > 0U && num_experts <= kEpServerDedupMaxExperts && num_experts % num_ranks == 0U);
    EP_HOST_ASSERT(LlCoreCountValid(num_vec_cores));
    EP_HOST_ASSERT(num_topk <= num_experts);

    EpMemoryServerDedupLayout layout{};
    layout.inputMode = mode;
    layout.globalBs = uint64_t(num_ranks) * num_max_tokens_per_rank;
    layout.localExpertNum = num_experts / num_ranks;
    layout.capacityRows = layout.globalBs *
        std::min<uint64_t>(num_topk, layout.localExpertNum);
    layout.data = DispatchServerDedupDataLayout(num_max_tokens_per_rank, num_hidden_elems, num_topk, num_experts,
                                                mode);
    const bool prequantized = mode == EpServerDedupInputMode::PrequantizedFp8Packs;
    const bool wire_fp8 = mode != EpServerDedupInputMode::Plain16;
    layout.xBytes = uint64_t(num_max_tokens_per_rank) * num_hidden_elems * (prequantized ? 1U : 2U);
    layout.inputScalesBytes = prequantized ? uint64_t(num_max_tokens_per_rank) * layout.data.scaleBytes : 0U;
    layout.expertIdsBytes = uint64_t(num_max_tokens_per_rank) * num_topk * sizeof(int32_t);
    layout.weightBytes = uint64_t(num_max_tokens_per_rank) * num_topk * sizeof(float);
    layout.expandXBytes = layout.capacityRows * (wire_fp8 ? num_hidden_elems : num_hidden_elems * 2U);
    layout.dynamicScalesBytes = layout.capacityRows * layout.data.scaleBytes;
    layout.expertTokenNumsBytes = layout.localExpertNum * sizeof(int64_t);
    layout.sendCountsBytes = uint64_t(num_experts) * sizeof(int32_t);
    layout.tokenTypeBytes = layout.capacityRows * sizeof(uint32_t);
    layout.destinationIndexBytes = layout.capacityRows * 3U * sizeof(uint32_t);
    layout.sourceMaskBytes = uint64_t(num_max_tokens_per_rank) * sizeof(uint32_t);
    layout.relayReadIndexRowStrideBytes = uint64_t(num_topk) * 8U;
    layout.relayReadIndexBytes = layout.capacityRows * layout.relayReadIndexRowStrideBytes;
    EP_HOST_ASSERT(layout.data.requiredBytes <= kEpServerDedupWindowUsableBytes);
    EP_HOST_ASSERT(layout.data.blockCount > 0U && layout.data.blockCount <= UINT8_MAX);
    return layout;
}

// True when the fixed sender/receiver arenas fit the physical UB.
inline bool get_dispatch_ub_bytes(const EpMemoryServerDedupLayout& layout, uint32_t num_max_tokens_per_rank,
                                  uint32_t num_topk, uint32_t num_experts, uint32_t num_ranks,
                                  uint32_t num_vec_cores, uint32_t num_ranks_per_group)
{
    if (!LlCoreCountValid(num_vec_cores)) return false;
    const auto align = [](uint64_t n, uint64_t a) { return (n + a - 1U) / a * a; };
    const uint32_t send_vec_cores = num_vec_cores / 2U;
    const uint32_t count_vec_cores =
        num_max_tokens_per_rank < kEpServerDedupAllCoreCountBsThreshold ? num_vec_cores : num_vec_cores / 2U;
    const bool packed_frames = num_ranks > kNumRanksPerHost;
    const uint32_t count_records = (num_experts + count_vec_cores - 1U) / count_vec_cores;
    const uint64_t reserved = num_max_tokens_per_rank < kEpServerDedupAllCoreCountBsThreshold
        ? (packed_frames ? CountWireBytes(count_records) : align(count_records, 8U) * 32U)
        : 0U;
    const uint32_t token_slice = (num_max_tokens_per_rank + send_vec_cores - 1U) / send_vec_cores;
    const bool need_quantize = layout.inputMode == EpServerDedupInputMode::Quantize16;
    const uint64_t input_bytes = need_quantize ? layout.data.quantCount * 2U : layout.data.packedPayloadBytes;
    const uint64_t q_bytes = align(uint64_t(num_max_tokens_per_rank) * num_topk * 4U, 32U);
    const uint64_t scale_count = layout.data.quantCount / 32U;
    const uint64_t quant_work = align(align(scale_count, 32U) * 4U + scale_count * 2U, 32U);
    const uint64_t scratch_bytes = need_quantize ? std::max(q_bytes, quant_work) : q_bytes;
    const uint64_t groups = num_ranks / num_ranks_per_group;
    const uint64_t sender =
        reserved + 2U * input_bytes + (need_quantize ? layout.data.packedPayloadBytes : 0U) +
        align(uint64_t(num_max_tokens_per_rank) * num_topk * 4U, 256U) + 2U * scratch_bytes + q_bytes +
        2U * align(uint64_t(token_slice) * num_topk * 4U, 32U) + align(uint64_t(token_slice) * 4U, 32U) +
        align(uint64_t(token_slice) * num_topk * 16U, 32U) + 64U +
        (groups > 1U ? align(groups * 4U, 32U) : 0U) + layout.data.tokenStride;
    const uint64_t receiver =
        align(uint64_t(num_experts) * 4U, 32U) +
        2U * align(uint64_t((layout.capacityRows + send_vec_cores - 1U) / send_vec_cores) * 4U, 32U) +
        16U * layout.data.blockCount * 32U + 256U +
        16U * align((layout.data.blockCount + 7U) / 8U, 32U) * 4U + 256U + 16U * 36U +
        layout.data.tokenStride + 64U;
    return sender <= kNumUbBytes && receiver <= kNumUbBytes;
}

inline std::vector<std::string> dispatch_jit_flags()
{
    std::vector<std::string> flags = {
        "-std=c++17",
        "-DCATLASS_ARCH=3510",
        "-DUSE_SIMT",
        "-DL2_CACHE_HINT",
        "-DDEEPEP_LL_RDMA=1",
        "-DDEEPEP_LL_JIT_TU=1",
        "-DACLSHMEMI_RDMA_K_BACKEND_HNS_1825=1",
        "-DACLSHMEMI_HNS_1825_PI_ON_CHIP=0",
        "--cce-auto-infer-kernel-type=false",
    };
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    flags.emplace_back("-DDEEPEP_LL_REFERENCE_PROFILE=1");
#endif
    return flags;
}

inline std::string dispatch_source(const DispatchKey& key)
{
    const std::string shape = key.ToArgs();
    std::ostringstream text;
    text << "#include <deep_ep/impls/ep/dispatch.hpp>\n"
         << "using namespace deep_ep;\n"
         << "extern \"C\" __global__ __aicore__ void dispatch_jit(\n"
            "    GM_ADDR x, GM_ADDR expert_ids, GM_ADDR expert_scales, GM_ADDR expand_x_out,\n"
            "    GM_ADDR dynamic_scales_out, GM_ADDR expert_token_nums_out, GM_ADDR send_counts_out,\n"
            "    GM_ADDR token_type_out, GM_ADDR destination_index_out, GM_ADDR relay_read_index_out,\n"
            "    GM_ADDR source_mask_out, GM_ADDR source_weights_out, GM_ADDR input_scales, GM_ADDR peer_ptrs,\n"
            "    GM_ADDR rdma_remote_base, GM_ADDR rdma_stage_base, GM_ADDR stage_timer, GM_ADDR expert_timer,\n"
            "    int32_t rank_idx, int32_t num_tokens, int32_t physical_ranks_per_host, int32_t backend,\n"
            "    int32_t stage_block_num, int64_t peer_mem_bytes, int64_t ffts_val, int32_t timer_mode)\n"
         << "{\n"
         << "    if constexpr ("
         << (key.input_class == 1U ? "true" : "false")
         << ")\n"
            "        dispatch_body<"
         << shape
         << ", bfloat16_t>(x, expert_ids, expert_scales, expand_x_out, dynamic_scales_out,\n"
            "        expert_token_nums_out, send_counts_out, token_type_out, destination_index_out,\n"
            "        relay_read_index_out, source_mask_out, source_weights_out, input_scales, peer_ptrs,\n"
            "        rdma_remote_base, rdma_stage_base, stage_timer, expert_timer, rank_idx, num_tokens,\n"
            "        physical_ranks_per_host, backend, stage_block_num, peer_mem_bytes, ffts_val, timer_mode,\n"
            "        OrdinaryCoreContext());\n"
            "    else\n"
            "        dispatch_body<"
         << shape
         << ", half>(x, expert_ids, expert_scales, expand_x_out, dynamic_scales_out,\n"
            "        expert_token_nums_out, send_counts_out, token_type_out, destination_index_out,\n"
            "        relay_read_index_out, source_mask_out, source_weights_out, input_scales, peer_ptrs,\n"
            "        rdma_remote_base, rdma_stage_base, stage_timer, expert_timer, rank_idx, num_tokens,\n"
            "        physical_ranks_per_host, backend, stage_block_num, peer_mem_bytes, ffts_val, timer_mode,\n"
            "        OrdinaryCoreContext());\n"
         << "}\n"
         << "static void __instantiate_kernel()\n{\n"
            "    auto ptr = reinterpret_cast<void*>(&dispatch_jit);\n    (void)ptr;\n}\n";
    for (uint32_t slot = 0; slot < 4U; ++slot) {
        text << "extern \"C\" __sk__ __vector__ void dispatch_jit_sk" << slot
             << "(const DispatchArgs* args, sk::SkSystemArgs* system_args)\n{\n"
                "    dispatch_impl_sk<"
             << shape << ", " << slot << ">(args, system_args);\n}\n";
    }
    text << "SK_BIND(dispatch_jit, 0, dispatch_jit_sk0, dispatch_jit_sk1, dispatch_jit_sk2, dispatch_jit_sk3);\n";
    return text.str();
}

inline void launch_dispatch(const DispatchKey& key, const DispatchLaunchArgs& args, void* stream)
{
    EP_HOST_ASSERT(args.x != nullptr && args.peer_ptrs != nullptr);
    const auto kernel = jit_compile("dispatch", dispatch_source(key), dispatch_jit_flags());
    jit_launch(kernel, stream, static_cast<int>(key.num_vec_cores), 0,
               args.x, args.expert_ids, args.expert_scales, args.expand_x_out, args.dynamic_scales_out,
               args.expert_token_nums_out, args.send_counts_out, args.token_type_out, args.destination_index_out,
               args.relay_read_index_out, args.source_mask_out, args.source_weights_out, args.input_scales,
               args.peer_ptrs, args.rdma_remote_base, args.rdma_stage_base, args.stage_timer, args.expert_timer,
               args.rank_idx, args.num_tokens, args.physical_ranks_per_host, args.backend, args.stage_block_num,
               args.peer_mem_bytes, args.ffts_val, args.timer_mode);
}

}  // namespace deep_ep
