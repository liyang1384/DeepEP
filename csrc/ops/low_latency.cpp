// Copyright (c) 2026, Lu Lu

#include "ops/low_latency.hpp"
#include <pybind11/stl.h>
#include <climits>
#include <stdexcept>
#include <string>

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/layout/ep/token.hpp>
#include <deep_ep/layout/ep/workspace.hpp>
#include <deep_ep/layout/ep/diagnostics.hpp>

#include <vector>
#include "runtime/session.hpp"
#include "runtime/ll_core_config.hpp"
#include "ubx/v2.hpp"
#include "host/mem/shmem_host_heap.h"
#include "host/data_plane/shmem_host_rma.h"
#include "host/data_plane/shmem_host_p2p_sync.h"
#include "../kernels/combine/low_latency/combine_layout.h"
#include "../kernels/combine/low_latency/ep_memory_server_dedup_combine_tiling.h"
#include "kernels/ep/dispatch.hpp"
#include "deep_ep/comm/comm_args.hpp"
#include "../kernels/combine/low_latency/launch.hpp"
#include "../kernels/runtime/clock_alignment.h"

namespace py = pybind11;
namespace deepep {
// Defined below; declared here because the timer helpers run before them.
extern "C" uint32_t deepep_ll_stage_timer_enabled();
extern "C" uint32_t deepep_ll_reference_profile_enabled();
namespace {
using Shape = std::vector<int64_t>;
constexpr uint64_t kRankBaseBiasBytes = 1024U;
uint32_t pin_session_cores(Session& session)
{
    session.workspace();
    const auto config = query_ll_core_config();
    if (config.device < 0) throw std::runtime_error("LL launch requires a current device for core verification");
    return session.pin_low_latency_cores(config.active);
}

uint8_t* peer_buffer_base(Session& session, int peer)
{
    if (peer < 0 || peer >= session.world()) throw std::invalid_argument("Invalid LL peer");
    if (session.ll_rdma() && !deep_ep::LlSameHost(session.rank(), peer)) return nullptr;
    auto* original = static_cast<uint8_t*>(peer == session.rank() ? session.workspace() : aclshmem_ptr(session.workspace(), peer));
    if (original == nullptr) throw std::invalid_argument("Invalid LL peer mapping");
    if (session.capacity() != (8ULL << 30)) return original;
    // Translate the original symmetric pointer before applying the destination-rank
    // bias. The shared peer table keeps local receive and remote send views equal.
    return original + uint64_t(peer) * kRankBaseBiasBytes;
}
uint8_t* tensor(py::handle value, const Shape& shape, const char* dtype)
{
    auto torch = py::module_::import("torch");
    if (!torch.attr("is_tensor")(value).cast<bool>()) {
        throw std::invalid_argument("LL requires tensors");
    }
    auto obj = py::reinterpret_borrow<py::object>(value);
    if (!obj.attr("is_contiguous")().cast<bool>() || obj.attr("shape").cast<Shape>() != shape ||
        !obj.attr("dtype").is(torch.attr(dtype))) {
        throw std::invalid_argument("LL tensor shape, dtype or contiguity mismatch");
    }
    return reinterpret_cast<uint8_t*>(obj.attr("data_ptr")().cast<uintptr_t>());
}
uint8_t* stage_timer(py::dict tensors, uint32_t cores, uint32_t& mode, bool allow_reference = false)
{
    mode = 1U;
    if (tensors.contains("ll_reference_profile") && !tensors["ll_reference_profile"].is_none()) {
        if (!allow_reference) throw std::invalid_argument("LL reference profiling supports dispatch only");
        if ((tensors.contains("ll_stage_timer") && !tensors["ll_stage_timer"].is_none()) ||
            (tensors.contains("ll_expert_timer") && !tensors["ll_expert_timer"].is_none()))
            throw std::invalid_argument("LL reference profiling cannot be combined with stage/expert tracing");
        if (!deepep_ll_reference_profile_enabled() || deepep_ll_stage_timer_enabled())
            throw std::invalid_argument("Rebuild with DEEPEP_LL_REFERENCE_PROFILE=ON and DEEPEP_LL_STAGE_TIMER=OFF");
        auto profile = py::reinterpret_borrow<py::object>(tensors["ll_reference_profile"]);
        auto device = profile.attr("device");
        if (device.attr("type").cast<std::string>() != "npu" ||
            !device.attr("__eq__")(tensors["input"].attr("device")).cast<bool>())
            throw std::invalid_argument("LL reference profile must be on the input NPU");
        auto* pointer = tensor(profile, {cores, profile::kSlots, profile::kFields}, "int64");
        if (reinterpret_cast<uintptr_t>(pointer) % 512U != 0U)
            throw std::invalid_argument("LL reference profile requires 512-byte alignment");
        mode = 3U;  // Private launch selector; the reference record's boundary mode remains 1.
        return pointer;
    }
    if (!tensors.contains("ll_stage_timer") || tensors["ll_stage_timer"].is_none()) return nullptr;
    if (!deepep_ll_stage_timer_enabled()) throw std::invalid_argument("Legacy LL stage tracing was removed; use LL reference profiling");
    if (tensors.contains("ll_stage_timer_mode")) mode = tensors["ll_stage_timer_mode"].cast<uint32_t>();
    if (mode != 1U && mode != 2U) throw std::invalid_argument("Invalid LL timer mode");
    auto timer = py::reinterpret_borrow<py::object>(tensors["ll_stage_timer"]);
    auto device = timer.attr("device");
    if (device.attr("type").cast<std::string>() != "npu" ||
        !device.attr("__eq__")(tensors["input"].attr("device")).cast<bool>())
        throw std::invalid_argument("LL timer must be on the input NPU");
    auto* pointer = tensor(timer, {cores, stage_timer::kSlots, stage_timer::kFields}, "int64");
    if (reinterpret_cast<uintptr_t>(pointer) % 512U != 0U)
        throw std::invalid_argument("LL timer requires 512-byte alignment for independent core writeback");
    return pointer;
}
uint8_t* expert_timer(py::dict tensors, uint32_t cores, uint32_t experts, uint8_t* stage)
{
    if (!tensors.contains("ll_expert_timer") || tensors["ll_expert_timer"].is_none()) return nullptr;
    if (!stage || !deepep_ll_stage_timer_enabled())
        throw std::invalid_argument("Expert diagnostics require stage capture and timer ON");
    auto value=py::reinterpret_borrow<py::object>(tensors["ll_expert_timer"]);
    auto device=value.attr("device");
    if (device.attr("type").cast<std::string>() != "npu" ||
        !device.attr("__eq__")(tensors["input"].attr("device")).cast<bool>())
        throw std::invalid_argument("Expert timer must be on the input NPU");
    auto* pointer=tensor(value, {cores, expert_timer::kHeaderRows + experts*expert_timer::StageCount,
                                expert_timer::kFields}, "int64");
    if (reinterpret_cast<uintptr_t>(pointer)%512U)
        throw std::invalid_argument("Expert timer requires 512-byte alignment");
    return pointer;
}
deep_ep::EpMemoryServerDedupLayout layout(int p, int b, int h, int k, int e, bool fp8)
{
    const uint32_t coreNum = query_ll_core_config().active;
    const auto mode = fp8 ? deep_ep::EpServerDedupInputMode::PrequantizedFp8Packs
                         : deep_ep::EpServerDedupInputMode::Plain16;
    if (b <= 0 || b > 65536) {
        throw std::invalid_argument("LL dispatch exceeds topology, shape, UB or workspace limits");
    }
    auto result = deep_ep::get_dispatch_buffer_size(uint32_t(p), uint32_t(b), uint32_t(h), uint32_t(k),
                                                    uint32_t(e), mode, coreNum);
    if (!deep_ep::get_dispatch_ub_bytes(result, uint32_t(b), uint32_t(k), uint32_t(e), uint32_t(p), coreNum,
                                        deep_ep::LlGroupForWorld(uint32_t(p))) ||
        result.capacityRows > INT32_MAX) {
        throw std::invalid_argument("LL dispatch exceeds topology, shape, UB or workspace limits");
    }
    return result;
}
template <class T>
py::bytes bytes(const T& data)
{
    return py::bytes(reinterpret_cast<const char*>(&data), sizeof(data));
}
uint8_t dtype_code(const std::string& dtype)
{
    if (dtype == "bfloat16") return deep_ep::DISPATCH_DEDUP_DATA_TYPE_BFP16;
    if (dtype == "float16") return deep_ep::DISPATCH_DEDUP_DATA_TYPE_FP16;
    if (dtype == "float8_e4m3fn") return deep_ep::DISPATCH_DEDUP_DATA_TYPE_FP8E4M3;
    throw std::invalid_argument("LL dispatch supports BF16/FP16 or packed FP8 E4M3FN");
}
static_assert(deep_ep::kLlDispatchFp16 == deep_ep::DISPATCH_DEDUP_DATA_TYPE_FP16 &&
              deep_ep::kLlDispatchBf16 == deep_ep::DISPATCH_DEDUP_DATA_TYPE_BFP16 &&
              deep_ep::kLlDispatchE4M3 == deep_ep::DISPATCH_DEDUP_DATA_TYPE_FP8E4M3 &&
              deep_ep::kLlDispatchE5M2 == deep_ep::DISPATCH_DEDUP_DATA_TYPE_FP8E5M2,
              "LL dispatch variant dtype values must match tiling");
static_assert(deep_ep::kLlDispatchPlain16 == uint8_t(deep_ep::EpServerDedupInputMode::Plain16) &&
              deep_ep::kLlDispatchQuantize16 == uint8_t(deep_ep::EpServerDedupInputMode::Quantize16) &&
              deep_ep::kLlDispatchPrequantized == uint8_t(deep_ep::EpServerDedupInputMode::PrequantizedFp8Packs),
              "LL dispatch variant input mode values must match tiling");

struct DispatchFormat {
    uint8_t dtype = 0;
    uint8_t quantMode = 0;
    uint8_t expandXOutDtype = 0;
    uint8_t inputMode = 0;
};
// Host-only format selection; the JIT key carries the device-side axes.
DispatchFormat dispatch_format(const std::string& dtype)
{
    const bool fp8 = dtype == "float8_e4m3fn";
    DispatchFormat format{};
    format.dtype = dtype_code(dtype);
    format.quantMode = fp8 ? 4 : 0;
    format.inputMode = uint8_t(fp8 ? deep_ep::EpServerDedupInputMode::PrequantizedFp8Packs
                                  : deep_ep::EpServerDedupInputMode::Plain16);
    format.expandXOutDtype = format.dtype;
    return format;
}
}  // namespace

// LL capability exports now live in the Host module: dispatch is compiled
// through DeepJIT, so no dispatch device translation unit exists any more.
// Combine/clock still load the device library for their own exports.
extern "C" uint32_t deepep_ll_core_protocol_version()
{
    return deep_ep::kLlCoreProtocolVersion;
}
extern "C" uint32_t deepep_ll_rdma_enabled()
{
#ifdef DEEPEP_LL_RDMA
    return 1U;
#else
    return 0U;
#endif
}
extern "C" uint32_t deepep_ll_ranks_per_group(uint32_t world)
{
    return deep_ep::LlGroupForWorld(world);
}
extern "C" uint32_t deepep_ll_stage_timer_enabled()
{
    return 0U;
}
extern "C" uint32_t deepep_ll_reference_profile_enabled()
{
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    return 1U;
#else
    return 0U;
#endif
}
extern "C" uint32_t deepep_ll_reference_profile_schema()
{
    return profile::kSchema;
}

void bind_low_latency(py::module_& module)
{
    if (deepep_ll_core_protocol_version() != deep_ep::kLlCoreProtocolVersion)
        throw std::runtime_error("LL Host/device core protocol mismatch; rebuild and reinstall the complete wheel");
    module.attr("low_latency_core_protocol") = deep_ep::kLlCoreProtocolVersion;
    module.attr("low_latency_roce_qps_per_peer") = 0; // actual active physical cores
    module.attr("low_latency_implementation") = "ndr-hns1825";
#ifdef DEEPEP_LL_RDMA
    if (!deepep_ll_rdma_enabled()) throw std::runtime_error("LL Host/device RDMA build mismatch; rebuild complete wheel");
    module.attr("low_latency_rdma_enabled") = true;
#else
    if (deepep_ll_rdma_enabled()) throw std::runtime_error("LL Host/device RDMA build mismatch; rebuild complete wheel");
    module.attr("low_latency_rdma_enabled") = false;
#endif
    module.attr("low_latency_stage_timer_enabled") = bool(deepep_ll_stage_timer_enabled());
    const bool reference_enabled = bool(deepep_ll_reference_profile_enabled());
    if (deepep_ll_reference_profile_schema() != profile::kSchema ||
        (reference_enabled && deepep_ll_stage_timer_enabled()))
        throw std::runtime_error("LL Host/device reference profile mismatch; rebuild and reinstall the complete wheel");
    module.attr("low_latency_reference_profile_enabled") = reference_enabled;
    module.def("low_latency_reference_profile_layout", []() {
        py::dict result;
        result["schema"] = profile::kSchema;
        result["slots"] = profile::kSlots;
        result["fields"] = profile::kFields;
        std::vector<std::string> names;
        for (const auto* name : profile::kNames) names.emplace_back(name);
        result["words"] = names;
        return result;
    });
    module.def("low_latency_expert_timer_layout", []() {
        py::dict result;
        result["schema"]=expert_timer::kSchema;
        result["header_rows"]=expert_timer::kHeaderRows;
        result["fields"]=expert_timer::kFields;
        std::vector<std::string> names;
        for (const auto* name : expert_timer::kNames) names.emplace_back(name);
        result["stages"]=names;
        return result;
    });
    module.def("low_latency_stage_timer_layout", []() {
        py::dict result;
        result["schema"] = stage_timer::kSchema;
        result["slots"] = stage_timer::kSlots;
        result["fields"] = stage_timer::kFields;
        std::vector<std::string> names;
        for (const auto* name : stage_timer::kNames) names.emplace_back(name);
        result["stages"] = names;
        return result;
    });
    module.def("low_latency_core_config", []() {
        const auto cfg = query_ll_core_config();
        py::dict result;
        result["platform_aiv"] = cfg.platform; result["available_aiv"] = cfg.available;
        result["active_aiv"] = cfg.active; result["send_cores"] = cfg.active / 2U;
        result["recv_cores"] = cfg.active / 2U; result["device"] = cfg.device;
        result["runtime_device_queried"] = cfg.device >= 0;
        result["control_capacity_aiv"] = deep_ep::kNumMaxVecCores;
        return result;
    });
    for (uint32_t world : {16U, 32U, 64U, 128U, 256U})
        if (deepep_ll_ranks_per_group(world) != deep_ep::LlGroupForWorld(world))
            throw std::runtime_error("LL Host/device tiling-key policy mismatch; rebuild the complete wheel");
    module.def("low_latency_ranks_per_group", [](uint32_t world) { return deep_ep::LlGroupForWorld(world); });
    module.attr("low_latency_clock_alignment_version") = clock_alignment::kVersion;
    module.def("low_latency_clock_sync",
               [](Session& session, py::object comm_args, py::object samples, uint32_t sample_count,
                  uint32_t warmup_sample_count, uint64_t magic, uint64_t timeout_cycles,
                  uintptr_t stream_address) {
                   if (!session.ll_rdma() || !stream_address || !magic || !timeout_cycles ||
                       sample_count == 0U || sample_count > clock_alignment::kMaxSamples ||
                       warmup_sample_count > clock_alignment::kMaxWarmupSamples) {
                       throw std::invalid_argument("Invalid LL clock synchronization configuration");
                   }
                   auto* comm = tensor(
                       comm_args, {sizeof(deep_ep::CommArgs)}, "uint8");
                   auto* output = tensor(
                       samples, {sample_count, clock_alignment::kSampleWords}, "int64");
                   session.workspace();
                   py::gil_scoped_release release;
                   deepep_clock_sync_kernel_do(
                       reinterpret_cast<void*>(stream_address), comm, output, sample_count,
                       warmup_sample_count, magic, timeout_cycles);
               });
    module.def("low_latency_clock_sleep_at",
               [](Session& session, uint64_t local_deadline, py::object release_cycle,
                  uintptr_t stream_address) {
                   if (!session.ll_rdma() || !stream_address) {
                       throw std::invalid_argument("Invalid LL clock sleep-at configuration");
                   }
                   uint8_t* output = nullptr;
                   if (!release_cycle.is_none()) {
                       output = tensor(release_cycle, {1}, "int64");
                   }
                   session.workspace();
                   py::gil_scoped_release release;
                   deepep_clock_sleep_at_kernel_do(
                       reinterpret_cast<void*>(stream_address), local_deadline, output);
               });
    module.def("low_latency_layout", [](int p, int b, int h, int k, int e, const std::string& dtype) {
        // A rejected shape is a Host-only capability result, not a launch failure.
        dtype_code(dtype);
        try {
            auto plan = layout(p, b, h, k, e, dtype == "float8_e4m3fn");
            py::dict result;
            result["capacity_rows"] = plan.capacityRows;
            result["scale_packs"] = (h + 127) / 128;
            return py::object(result);
        } catch (const std::invalid_argument&) {
            return py::object(py::none());
        }
    });
    module.def("low_latency_comm_args", [](Session& session) {
        if (session.combine_only()) throw std::invalid_argument("LL requires a dense-channel Session");
        if (session.capacity() != deep_ep::kEpServerDedupPeerBytes ||
            session.world() > deep_ep::DISPATCH_DEDUP_MAX_RANK_SIZE ||
            session.world() < 16 || session.world() % deep_ep::LlGroupForWorld(session.world()) != 0) {
            throw std::invalid_argument("LL requires exactly 8 GiB of symmetric workspace");
        }
        pin_session_cores(session);
        deep_ep::CommArgs args{};
        args.rank = session.rank();
        args.rankSize = session.world();
        args.localRank = session.rank() % deep_ep::LlGroupForWorld(session.world());
        args.localRankSize = deep_ep::LlGroupForWorld(session.world());
        args.reserved0 = deep_ep::kNumRanksPerHost;
        args.peerMemBytes = session.capacity();
        args.fftsVal = shmemx_get_ffts_config();
        if (session.ll_rdma()) {
            args.reservedAddresses[0] = static_cast<uint8_t*>(session.workspace());
            args.reservedAddresses[1] = static_cast<uint8_t*>(session.ll_rdma_stage());
            args.reservedConfig[0] = deep_ep::rdma::kBackend;
            args.reservedConfig[1] = session.ll_rdma_qps();
        }
        for (int peer = 0; peer < session.world(); ++peer) {
            args.peerMems[peer] = peer_buffer_base(session, peer);
            if (session.ll_rdma() && !deep_ep::LlSameHost(session.rank(), peer)) continue;
            if (args.peerMems[peer] == nullptr || reinterpret_cast<uintptr_t>(args.peerMems[peer]) % 32) {
                throw std::invalid_argument("LL requires aligned MTE mappings for every peer");
            }
        }
        return bytes(args);
    });
    module.def("low_latency_combine_tiling",
               [](int p, int b, int h, int k, int e, const std::string& dtype, int num_max_tokens_per_rank, bool rdma) {
                   if (rdma) {
                       return bytes(ubx::v2_tiling(p, b, h, k, e, dtype_code(dtype), num_max_tokens_per_rank,
                                                 query_ll_core_config().active, deep_ep::kNumRanksPerHost));
                   }
                   CombineDedup::Layout plan{};
                   if ((dtype != "bfloat16" && dtype != "float16") || !CombineDedup::BuildLayout(p, b, h, k, e, plan)) {
                       throw std::invalid_argument("LL combine exceeds dtype, topology, UB or workspace limits");
                   }
                   CombineDedup::EpServerDedupCombineTilingData data{};
                   data.bs = b;
                   data.h = h;
                   data.numMaxTokensPerRank = num_max_tokens_per_rank;
                   data.topK = k;
                   data.moeExpertNum = e;
                   data.dtype = dtype_code(dtype);
                   return bytes(data);
               },
               py::arg("world"), py::arg("tokens"), py::arg("hidden"), py::arg("topk"), py::arg("experts"),
               py::arg("dtype"), py::arg("capacity"), py::arg("rdma") = false);
    module.def("dispatch_low_latency", [](Session& session, py::dict tensors, int b, int h, int k, int e, int rows,
                                          int capacity, const std::string& dtype, uintptr_t stream_address) {
        if (session.combine_only()) throw std::invalid_argument("LL requires a dense-channel Session");
        dtype_code(dtype);
        const bool fp8 = dtype == "float8_e4m3fn";
        const auto format = dispatch_format(dtype);
        const uint32_t input_class = deep_ep::LlDispatchInputClass(
            format.dtype, format.quantMode, format.expandXOutDtype, format.inputMode);
        if (input_class == deep_ep::kLlDispatchInvalidVariant)
            throw std::invalid_argument("Unsupported LL dispatch format");
        const uint32_t cores = pin_session_cores(session);
        const auto mode = fp8 ? deep_ep::EpServerDedupInputMode::PrequantizedFp8Packs
                              : deep_ep::EpServerDedupInputMode::Plain16;
        const auto plan = deep_ep::get_dispatch_buffer_size(
            uint32_t(session.world()), uint32_t(capacity), uint32_t(h), uint32_t(k), uint32_t(e), mode, cores);
        const uint32_t group = deep_ep::LlGroupForWorld(uint32_t(session.world()));
        if (!deep_ep::get_dispatch_ub_bytes(plan, uint32_t(capacity), uint32_t(k), uint32_t(e),
                                            uint32_t(session.world()), cores, group)) {
            throw std::invalid_argument("LL dispatch exceeds UB capacity for this specialization");
        }
        const int64_t c = plan.capacityRows;
        if (!stream_address || rows < c || session.capacity() != deep_ep::kEpServerDedupPeerBytes) {
            throw std::invalid_argument("Invalid LL output capacity, workspace or stream");
        }
        auto get = [&](const char* name, Shape shape, const char* type) { return tensor(tensors[name], shape, type); };
        deep_ep::DispatchLaunchArgs args{};
        uint32_t timer_mode = 1U;
        // cores travel as the template axis and the launch block count.
        deep_ep::DispatchKey key{};
        key.num_ranks = uint32_t(session.world());
        key.num_max_tokens_per_rank = uint32_t(capacity);
        key.num_hidden_bytes = (fp8 ? uint32_t(h) : uint32_t(h) * 2U);
        key.num_topk = uint32_t(k);
        key.num_experts = uint32_t(e);
        key.num_vec_cores = cores;
        key.input_class = input_class;
        key.num_ranks_per_group = group;
        key.input_mode = format.inputMode;
        key.e4m3 = true;
        key.dedup = deep_ep::EpServerDedupEnabled(uint32_t(capacity), uint32_t(session.world()));
        key.rdma = session.ll_rdma();

        auto* comm_args = get("comm_args", {sizeof(deep_ep::comm::CommArgs)}, "uint8");
        args.peer_ptrs = comm_args + offsetof(deep_ep::comm::CommArgs, peerMems);
        args.x = get("input", {b, h}, dtype.c_str());
        args.expert_ids = get("indices", {b, k}, "int32");
        args.expert_scales = get("weights", {b, k}, "float32");
        args.expand_x_out = get("output", {rows, h}, dtype.c_str());
        // Both formats occupy four bytes per 128 hidden elements. The kernel
        // transports opaque bytes: FP32 per-128 or packed E8M0 per-32.
        const char* scale_dtype = "int32";
        if (fp8) {
            auto torch = py::module_::import("torch");
            if (tensors["scales"].attr("dtype").is(torch.attr("float32"))) {
                scale_dtype = "float32";
            }
        }
        args.dynamic_scales_out = fp8 ? get("output_scales", {rows, (h + 127) / 128}, scale_dtype) : nullptr;
        args.expert_token_nums_out = get("expert_recv_counts", {e / session.world()}, "int64");
        args.send_counts_out = get("send_counts", {e}, "int32");
        args.token_type_out = get("token_type", {c}, "int32");
        args.destination_index_out = get("destination_index", {c, 3}, "int32");
        args.relay_read_index_out = get("relay_read_index", {c, k, 2}, "int32");
        args.source_mask_out = get("source_mask", {b}, "int32");
        args.source_weights_out = get("source_weights", {b, k}, "float32");
        args.input_scales = fp8 ? get("scales", {b, (h + 127) / 128}, scale_dtype) : nullptr;
        args.stage_timer = stage_timer(tensors, cores, timer_mode, true);
        args.timer_mode = int32_t(timer_mode);
        args.expert_timer = expert_timer(tensors, cores, e, args.stage_timer);
        args.rank_idx = session.rank();
        args.num_tokens = b;
        args.physical_ranks_per_host = int32_t(deep_ep::kNumRanksPerHost);
        args.backend = session.ll_rdma() ? int32_t(deep_ep::rdma::kBackend) : 0;
        args.stage_block_num = session.ll_rdma() ? int32_t(session.ll_rdma_qps()) : 0;
        args.peer_mem_bytes = int64_t(session.capacity());
        args.ffts_val = shmemx_get_ffts_config();
        if (session.ll_rdma()) {
            args.rdma_remote_base = static_cast<uint8_t*>(session.workspace());
            args.rdma_stage_base = static_cast<uint8_t*>(session.ll_rdma_stage());
        }
        session.workspace();  // Validate the owning context before launch.
        py::gil_scoped_release release;
        session.invalidate_dispatch_controls();
        deep_ep::launch_dispatch(key, args, reinterpret_cast<void*>(stream_address));
    });
    module.def("combine_low_latency", [](Session& session, py::dict tensors, int b, int h, int k, int e,
                                         const std::string& dtype, uintptr_t stream_address) {
        if (session.combine_only()) throw std::invalid_argument("LL requires a dense-channel Session");
        CombineDedup::Layout plan{};
        if (!stream_address || session.capacity() != deep_ep::kEpServerDedupPeerBytes ||
            (dtype != "bfloat16" && dtype != "float16")) {
            throw std::invalid_argument("Invalid LL combine dtype, shape, workspace or stream");
        }
        uint32_t cores = 0;
        if (session.ll_rdma()) {
            cores = pin_session_cores(session);
            plan.capacityRows =
                ubx::v2_capacity(session.world(), b, h, k, e, cores, deep_ep::kNumRanksPerHost);
        } else if (!CombineDedup::BuildLayout(session.world(), b, h, k, e, plan)) {
            throw std::invalid_argument("Invalid LL combine shape");
        }
        const int64_t c = plan.capacityRows;
        int32_t device = -1;
        if (session.ll_rdma() && aclrtGetDevice(&device) != 0)
            throw std::runtime_error("LL Combine requires a current NPU");
        auto get = [&](const char* name, Shape shape, const char* type) {
            auto value = py::reinterpret_borrow<py::object>(tensors[name]);
            auto* pointer = tensor(value, shape, type);
            if (session.ll_rdma() && (value.attr("device").attr("type").cast<std::string>() != "npu" ||
                                     value.attr("device").attr("index").cast<int>() != device))
                throw std::invalid_argument("LL Combine tensors must be on the current NPU");
            return pointer;
        };
        deepep::EpServerDedupCombineKernelArgs args{};
        args.commArgs = get("comm_args", {sizeof(deep_ep::CommArgs)}, "uint8");
        // Python validates rows against the dispatch handle. The device selects
        // rows below the actual receive count; C_B sizes metadata, not expert storage.
        const auto input_shape = tensors["input"].attr("shape").cast<Shape>();
        if (input_shape.size() != 2 || input_shape[0] < 0 || input_shape[1] != h) {
            throw std::invalid_argument("LL combine input must have shape [rows, H]");
        }
        args.expertOut = get("input", input_shape, dtype.c_str());
        args.destinationIndex = get("destination_index", {c, 3}, "int32");
        args.sendCounts = get("send_counts", {e}, "int32");
        args.expertScales = get("source_weights", {b, k}, "float32");
        args.tokenType = get("token_type", {c}, "int32");
        args.relayReadIndex = get("relay_read_index", {c, k, 2}, "int32");
        args.sourceMask = get("source_mask", {b}, "int32");
        args.yOut = get("output", {b, h}, dtype.c_str());
        args.tilingData = get("tiling", {sizeof(CombineDedup::EpServerDedupCombineTilingData)}, "uint8");
        session.workspace();
        py::gil_scoped_release release;
        session.invalidate_dispatch_controls();
        if (session.ll_rdma()) {
            const ubx::CombineArgs rdma_args{args.commArgs,     args.expertOut, args.destinationIndex, args.sendCounts,
                                             args.expertScales, args.tokenType, args.relayReadIndex,   args.sourceMask,
                                             args.yOut,         args.tilingData, cores};
            ubx_combine_launch(reinterpret_cast<void*>(stream_address), &rdma_args);
        } else {
            combine_server_dedup_kernel_do(reinterpret_cast<void*>(stream_address), &args);
        }
    });
}
}  // namespace deepep
