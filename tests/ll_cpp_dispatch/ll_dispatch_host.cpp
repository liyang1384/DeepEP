#include "ll_dispatch_c_api.h"

#include <acl/acl.h>

#include <climits>
#include <stdexcept>
#include <string>

#include "host/data_plane/shmem_host_heap.h"
#include "host/data_plane/shmem_host_p2p_sync.h"
#include "host/data_plane/shmem_host_rma.h"
#include "kernels/dispatch/low_latency/common/comm_args.h"
#include "kernels/dispatch/low_latency/common/ep_memory_server_dedup_layout.h"
#include "kernels/dispatch/low_latency/common/ep_memory_server_dedup_tiling.h"
#include "kernels/dispatch/low_latency/common/ll_dispatch_variant.h"
#include "kernels/dispatch/low_latency/common/ll_group_policy.h"
#include "kernels/dispatch/low_latency/common/ll_rdma_layout.h"
#include "kernels/dispatch/low_latency/launch.hpp"
#include "runtime/ll_core_config.hpp"
#include "runtime/session.hpp"

namespace {

// 与 Host 侧 peer_buffer_base() 相同的寻址口径：8 GiB 对称空间按目标 rank 加偏置。
constexpr uint64_t kRankBaseBiasBytes = 1024U;

void check(aclError status, const char* operation)
{
    if (status != ACL_SUCCESS) throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(status));
}

uint32_t pin_cores(deepep::Session& session)
{
    session.workspace();
    const auto config = deepep::query_ll_core_config();
    if (config.device < 0) throw std::runtime_error("LL launch requires a current device");
    return session.pin_low_latency_cores(config.active);
}

DispatchDedup::EpMemoryServerDedupLayout build_layout(uint32_t world, int b, int h, int k, int e)
{
    const uint32_t cores = deepep::query_ll_core_config().active;
    DispatchDedup::EpMemoryServerDedupLayout plan{};
    const int status = DispatchDedup::BuildDispatchServerDedupLayout(
        world, b, h, k, e, &plan, DispatchDedup::EpServerDedupInputMode::Plain16, cores);
    if (status != DispatchDedup::DISPATCH_DEDUP_SUCCESS ||
        !DispatchDedup::DispatchDedupPacketUbFits(plan, b, k, e, world, cores) || plan.capacityRows > INT32_MAX) {
        throw std::invalid_argument("dispatch shape exceeds topology, UB or workspace limits");
    }
    return plan;
}

uint8_t* peer_base(deepep::Session& session, int peer)
{
    if (peer < 0 || peer >= session.world()) throw std::invalid_argument("invalid peer");
    auto* base = static_cast<uint8_t*>(peer == session.rank() ? session.workspace()
                                                             : aclshmem_ptr(session.workspace(), peer));
    if (base == nullptr) throw std::runtime_error("invalid peer mapping");
    if (session.capacity() != DispatchDedup::kEpServerDedupPeerBytes) return base;
    return base + uint64_t(peer) * kRankBaseBiasBytes;
}

template <class T>
void upload(void* device, const T& value)
{
    check(aclrtMemcpy(device, sizeof(T), &value, sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(H2D)");
}

}  // namespace

uint32_t ll_dispatch_capacity_rows(uint32_t world, int b, int h, int k, int e)
{
    return uint32_t(build_layout(world, b, h, k, e).capacityRows);
}

void ll_dispatch_make_comm_args(deepep::Session& session, void* device_comm_args)
{
    if (session.combine_only()) throw std::invalid_argument("LL requires a dense-channel Session");
    if (session.capacity() != DispatchDedup::kEpServerDedupPeerBytes)
        throw std::invalid_argument("LL requires exactly 8 GiB of symmetric workspace");
    // 头文件里的协议号必须与已安装的 kernel 库一致，否则计数布局会对不上。
    if (deepep_ll_core_protocol_version() != DispatchDedup::kLlCoreProtocolVersion)
        throw std::runtime_error("LL Host/device core protocol mismatch; rebuild the wheel");
    pin_cores(session);

    const uint32_t group = DispatchDedup::LlGroupForWorld(uint32_t(session.world()));
    if (session.world() < 16 || session.world() > DispatchDedup::DISPATCH_DEDUP_MAX_RANK_SIZE ||
        session.world() % int(group) != 0)
        throw std::invalid_argument("LL requires world >= 16 and a multiple of the LL group size");
    DispatchDedup::CommArgs args{};
    args.rank = session.rank();
    args.rankSize = session.world();
    args.localRank = session.rank() % int(group);
    args.localRankSize = int(group);
    args.reserved0 = DispatchDedup::kLlPhysicalRanksPerHost;
    args.peerMemBytes = session.capacity();
    args.fftsVal = shmemx_get_ffts_config();
    if (session.ll_rdma()) {
        args.reservedAddresses[0] = static_cast<uint8_t*>(session.workspace());
        args.reservedAddresses[1] = static_cast<uint8_t*>(session.ll_rdma_stage());
        args.reservedConfig[0] = LlRdma::kBackend;
        args.reservedConfig[1] = session.ll_rdma_qps();
    }
    for (int peer = 0; peer < session.world(); ++peer) {
        // 跨机 peer 在 RDMA 模式下走 QP，不映射对端 GM。
        if (session.ll_rdma() && !DispatchDedup::LlSameHost(uint32_t(session.rank()), uint32_t(peer))) continue;
        auto* base = peer_base(session, peer);
        if (reinterpret_cast<uintptr_t>(base) % 32U != 0U)
            throw std::invalid_argument("LL requires 32-byte aligned peer mappings");
        args.peerMems[peer] = base;
    }
    upload(device_comm_args, args);
}

void ll_dispatch_make_tiling(uint32_t world, int b, int h, int k, int e, int capacity, bool fp16,
                             void* device_tiling)
{
    build_layout(world, b, h, k, e);
    DispatchDedup::EpServerDedupTilingData data{};
    data.tilingKey = DispatchDedup::LlTilingKeyForWorld(world);
    data.dtype = fp16 ? DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP16 : DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_BFP16;
    data.expandXOutDtype = data.dtype;
    data.quantMode = 0;
    data.inputMode = uint8_t(DispatchDedup::EpServerDedupInputMode::Plain16);
    data.bs = uint32_t(b);
    data.h = uint32_t(h);
    data.numMaxTokensPerRank = uint32_t(capacity);
    data.topK = uint8_t(k);
    data.moeExpertNum = uint16_t(e);
    data.expertTokenNumsType = 1;  // 输出计数而不是前缀和
    upload(device_tiling, data);
}

void ll_dispatch_run(deepep::Session& session, void* stream, const LlDispatchTensors& t, int b, int h, int k, int e,
                     int rows)
{
    if (session.combine_only()) throw std::invalid_argument("LL requires a dense-channel Session");
    if (stream == nullptr) throw std::invalid_argument("LL requires a stream");
    const auto plan = build_layout(uint32_t(session.world()), b, h, k, e);
    if (int64_t(rows) < int64_t(plan.capacityRows) || session.capacity() != DispatchDedup::kEpServerDedupPeerBytes)
        throw std::invalid_argument("invalid output capacity or workspace");

    deepep::EpServerDedupKernelArgs args{};
    args.coreNum = pin_cores(session);
    args.timerMode = 1U;  // 与 Python 侧一致；未挂 timer 时 kernel 不记录
    args.commArgs = static_cast<uint8_t*>(t.comm_args);
    args.x = static_cast<uint8_t*>(t.input);
    args.expertIds = static_cast<uint8_t*>(t.indices);
    args.expertScales = static_cast<uint8_t*>(t.weights);
    args.expandXOut = static_cast<uint8_t*>(t.output);
    args.dynamicScalesOut = nullptr;
    args.expertTokenNumsOut = static_cast<uint8_t*>(t.expert_recv_counts);
    args.sendCountsOut = static_cast<uint8_t*>(t.send_counts);
    args.tokenTypeOut = static_cast<uint8_t*>(t.token_type);
    args.destinationIndexOut = static_cast<uint8_t*>(t.destination_index);
    args.relayReadIndexOut = static_cast<uint8_t*>(t.relay_read_index);
    args.sourceMaskOut = static_cast<uint8_t*>(t.source_mask);
    args.expertScalesOut = static_cast<uint8_t*>(t.source_weights);
    args.tilingData = static_cast<uint8_t*>(t.tiling);
    args.inputScales = nullptr;
    args.stageTimer = nullptr;
    args.expertTimer = nullptr;
    session.workspace();  // 校验当前 context 属于本 Session
    session.invalidate_dispatch_controls();
    dispatch_server_dedup_kernel_do(stream, &args);
}
