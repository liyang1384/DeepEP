// Copyright (c) 2026, Lu Lu
// Folded from the LL dispatch split headers; see csrc/kernels/ep/dispatch.hpp
// LL dispatch: DispatchUBLayout, one dispatch_impl kernel and its SuperKernel entries.
#pragma once

#include <cstdint>
#include <cstddef>

#include <kernel_operator.h>
#ifdef DEEPEP_LL_RDMA
#include <shmem.h>
#endif

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/common/exception.hpp>
#include <deep_ep/common/ub.hpp>
#include <deep_ep/comm/transport.hpp>
#include <deep_ep/layout/ep/token.hpp>
#include <deep_ep/layout/ep/workspace.hpp>
#include <deep_ep/layout/ep/diagnostics.hpp>
#include <kernels/common/ep_packet_receive_layout.h>
#include <kernels/common/core_context.h>
#include <kernels/common/ep_server_order.h>

namespace deep_ep {
// Shared packet layout still lives in the common header for combine.
namespace PacketReceiveLayout = DispatchDedup::PacketReceiveLayout;
}

// Copyright (c) 2026, Lu Lu
// Modified by candy_cloud_fly 2026


// Public SHMEM headers: MTE performs UB -> peer symmetric GM on PIPE_MTE3.
// Completion is tracked by the caller's existing MTE3 events before UB reuse.
// No SHMEM global barrier is inserted into per-rank, uneven packet loops.
#include "device/gm2gm/engine/shmem_device_mte.h"
#include "device/ub2gm/engine/shmem_device_mte.h"

// Copyright (c) 2026, Lu Lu

namespace deep_ep {
using comm::TransportView;
// Each core owns one local 512B magic slot. Retire its 32B scratch before Counts uses UB0.
__aicore__ inline uint32_t RotateDispatchServerDedupMagic(const TransportView& view, uint32_t core)
{
    auto* state = ub::Ub<uint32_t>(0U);
    auto* slot = view.local_ptr() + uint64_t(core) * kEpServerDedupMagicStride;
    ub::Load(state, slot, 32U);
    ub::Sync<PIPE_MTE2, PIPE_S>();
    const uint32_t magic = state[0] ? 0U : 1U;
    state[0] = magic;
    ub::Sync<PIPE_S, PIPE_MTE3>();
    ub::Store(slot, state, 32U);
    ub::Sync<PIPE_MTE3, PIPE_S>();
    return magic;
}
}  // namespace DispatchDedup

// Copyright (c) 2026, Lu Lu

namespace deep_ep {
__aicore__ inline uint32_t ServerDedupWireOffset(uint32_t logical)
{
    return logical + logical / 480U * 32U;
}
class ServerDedupPacketTail {
public:
    __aicore__ inline ServerDedupPacketTail(__ubuf__ uint8_t* wire, uint32_t header, uint32_t index)
        : header_(reinterpret_cast<__ubuf__ uint32_t*>(wire + header)),
          index_(reinterpret_cast<__ubuf__ uint32_t*>(wire + index)) {}
    __aicore__ inline __ubuf__ uint32_t* Header() { return header_; }
    __aicore__ inline __ubuf__ uint32_t* ReadIndex() { return index_; }
    __aicore__ inline __ubuf__ uint32_t* Entry(uint32_t index) { return index_ + (index + 1U) * 2U; }
private:
    __ubuf__ uint32_t *header_, *index_;
};
__aicore__ inline void ReadServerDedupFlags(GM_ADDR slot, uint32_t blocks, __ubuf__ float* flags,
                                           bool uncached = false)
{
    ub::Load(flags, slot + 480U, 32U, blocks, 480U, 0U, uncached);
}
__aicore__ inline void WriteServerDedupPayloadRange(GM_ADDR output, __ubuf__ uint8_t* wire,
                                                   uint32_t logical, uint32_t bytes)
{
    uint32_t written = 0U;
    const uint32_t within = logical % 480U;
    if (within && bytes) {
        const uint32_t available = 480U - within, head = bytes < available ? bytes : available;
        ub::Store(output, wire + ServerDedupWireOffset(logical), head);
        logical += head; bytes -= head; written += head;
    }
    const uint32_t full = bytes / 480U;
    if (full) {
        ub::Store(output + written, wire + ServerDedupWireOffset(logical), 480U, full, 0U, 32U);
        logical += full * 480U; bytes -= full * 480U; written += full * 480U;
    }
    if (bytes) ub::Store(output + written, wire + ServerDedupWireOffset(logical), bytes);
}
__aicore__ inline uint32_t BuildServerDedupMetadata(__ubuf__ uint32_t* type, __ubuf__ uint32_t* destination,
                                                  ServerDedupPacketTail tail, uint32_t slot)
{
    auto* header = tail.Header();
    const uint32_t role = header[3U];
    type[0] = role;
    if (role == uint32_t(EpServerDedupTokenType::Terminal)) {
        destination[0] = slot; destination[1] = UINT32_MAX; destination[2] = UINT32_MAX;
    } else {
        for (uint32_t i = 0; i < 3U; ++i) destination[i] = header[i];
    }
    return role;
}
__aicore__ inline void PrepareServerDedupForward(ServerDedupPacketTail tail)
{
    tail.Header()[3U] = uint32_t(EpServerDedupTokenType::Terminal);
}
}  // namespace DispatchDedup

// Copyright (c) 2026, Lu Lu
// Modified by candy_cloud_fly 2026



namespace deep_ep {

// R05-01/R05-02: stable physical slot positions and a compact pending list.
// The caller supplies the complete, ready prefix in localExpert/source order.
// Both tables remain in UB until this core has finished clearing its slots.
class ServerDedupReceiveTasks {
public:
    __aicore__ inline void Build(__ubuf__ int32_t* prefix, uint32_t rank, uint32_t ranks, uint32_t experts,
                                 uint32_t bs, uint32_t receiver, ub::UbLayout& layout, uint32_t recvCores)
    {
        const uint32_t total = uint32_t(prefix[experts - 1U]);
        // 交错分配：receiver 核拥有全局行号 {receiver, receiver+recvCores, ...}。
        // 这样每个核同时拿到早到源与晚到源的 token，relay 转发量也随 token 摊平，
        // 不会再整段继承单个源的到达时刻；输出行由首行和步长直接还原。
        begin_ = receiver;
        stride_ = recvCores;
        count_ = begin_ < total ? (total - 1U - begin_) / stride_ + 1U : 0U;
        pendingCount_ = count_;
        if (count_ == 0U) return;

        const uint32_t bytes = uint32_t(DispatchDedupAlign(uint64_t(count_) * sizeof(uint32_t), 32U));
        tokenPos_ = layout.Allocate<uint32_t>(bytes);
        pendingIdx_ = layout.Allocate<uint32_t>(bytes);

        const uint32_t localExperts = experts / ranks;
        uint32_t record = 0U;
        // 行号递增，二分只需在剩余 record 区间内进行；小批量下约 1µs。
        for (uint32_t j = 0; j < count_; ++j) {
            const uint32_t row = begin_ + j * stride_;
            uint32_t low = record, high = experts;
            while (low < high) {
                const uint32_t middle = low + (high - low) / 2U;
                if (uint32_t(prefix[middle]) <= row)
                    low = middle + 1U;
                else
                    high = middle;
            }
            record = low;
            const uint32_t previous = record == 0U ? 0U : uint32_t(prefix[record - 1U]);
            const uint32_t source = record % ranks;
            const uint32_t localExpert = record / ranks;
            const uint32_t region = ((source + rank) % ranks) * localExperts + localExpert;
            tokenPos_[j] = region * bs + row - previous;
            pendingIdx_[j] = j;
        }
    }

    __aicore__ inline uint32_t Begin() const
    {
        return begin_;
    }
    // 本核拥有的全局行号步长；输出行 = Begin() + j * Stride()。
    __aicore__ inline uint32_t Stride() const
    {
        return stride_;
    }
    __aicore__ inline uint32_t Count() const
    {
        return count_;
    }
    __aicore__ inline uint32_t PendingCount() const
    {
        return pendingCount_;
    }
    __aicore__ inline uint32_t Position(uint32_t j)
    {
        return tokenPos_[j];
    }
    __aicore__ inline uint32_t PendingIndex(uint32_t p)
    {
        return pendingIdx_[p];
    }

    // Call only after every required write for PendingIndex(p) has been issued.
    // Recheck p after removal; output row and cleanup ownership still use j.
    __aicore__ inline void RemovePending(uint32_t p)
    {
        --pendingCount_;
        pendingIdx_[p] = pendingIdx_[pendingCount_];
    }

private:
    uint32_t begin_ = 0U, stride_ = 1U, count_ = 0U, pendingCount_ = 0U;
    __ubuf__ uint32_t *tokenPos_ = nullptr, *pendingIdx_ = nullptr;
};
}  // namespace DispatchDedup



namespace deep_ep {

/**
 *  for (int i = 0; i < num_i32_elems; ++ i)
 *      if (offset <= data[i] and data[i] < offset + num_bins)
 *          acc[data[i] - offset + 1] ++;
 */
__simd_vf__ void histogram_accumulate_u16(__ubuf__ int32_t* data, __ubuf__ uint16_t* acc,
                                          const int num_i32_elems, const int offset, const int num_bins) {
    static constexpr int kNumIdsPerVector = sizeof(vector_int32_t) / sizeof(int32_t);
    vector_bool mask_b16 = asc_create_mask_b16(PAT_ALL);
    vector_int32_t value_vec;
    vector_uint16_t count_vec;
    asc_duplicate_scalar(count_vec, static_cast<uint16_t>(0), mask_b16);

    uint32_t remaining = static_cast<uint32_t>(num_i32_elems);
    for (int elem_offset = 0; elem_offset < num_i32_elems; elem_offset += kNumIdsPerVector) {
        uint32_t active_bytes = remaining * static_cast<uint32_t>(sizeof(int32_t));
        const vector_bool mask_b32 = asc_update_mask_b32(remaining);
        asc_loadalign(value_vec, data + elem_offset);
        asc_max_scalar(value_vec, value_vec, offset - 1, mask_b32);
        asc_min_scalar(value_vec, value_vec, offset + num_bins, mask_b32);
        asc_add_scalar(value_vec, value_vec, 1 - offset, mask_b32);

        const vector_bool mask_b8 = asc_update_mask_b8(active_bytes);
        asc_frequency_histogram_bin0(count_vec, (vector_uint8_t&)value_vec, mask_b8);
    }
    asc_storealign(acc, count_vec, mask_b16);
}

}  // namespace DispatchDedup


namespace deep_ep {
// At most ceil(1024 / 8)=128 logical records per receiver and world>=32.
// Direct counts can change layout at each 16-source host boundary. A
// 128-record partition starting mid-host needs at most nine contiguous runs.
constexpr uint32_t kLlCountReadMaxRuns = 9U;
struct LlCountReadRun {
    uint64_t recordOffset = 0, commitOffset = 0;
    uint32_t first = 0, records = 0; // first is relative to this receiver's UB
};
struct LlCountReadPlan {
    LlCountReadRun runs[kLlCountReadMaxRuns]{};
    uint32_t size = 0;
    uint32_t sourceStrideBlocks = 0; // gap in 32B units, excluding the copied block
    uint32_t expectedFlagSum = 0;
};
__aicore__ __forceinline__ bool BuildCountReadPlan(uint32_t experts, uint32_t world, uint32_t countCores,
    uint32_t destination, uint32_t first, uint32_t count, LlCountReadPlan& out,
    bool directRanks = false)
{
    out.size = 0;
    out.expectedFlagSum = 0;
    if (world < 32U || world > 256U || experts < world || experts > 1024U || experts % world ||
        !countCores || countCores > 72U || destination >= world || first > experts ||
        count > experts - first || count > 128U) return false;
    out.sourceStrideBlocks = (experts / world) * 16U - 1U;
    for (uint32_t consumed = 0; consumed < count;) {
        if (out.size == kLlCountReadMaxRuns) return false;
        const uint32_t logical = first + consumed;
        const uint32_t source = logical % world;
        const uint32_t available = directRanks
            ? kNumRanksPerHost - source % kNumRanksPerHost : world - source;
        const uint32_t length = count - consumed < available ? count - consumed : available;
        const auto frame = CountFrame(experts, world, countCores, destination, source, logical / world, directRanks);
        out.runs[out.size++] = {frame.recordOffset, frame.commitOffset, consumed, length};
        out.expectedFlagSum += length * frame.flagWords;
        consumed += length;
    }
    return true;
}
}


namespace deep_ep {
// Issue only; caller retains its existing MTE2->Vector->Scalar wait chain.
// Source-major frames are strided across source ranks. Each expert row becomes
// one DMA while the UB snapshot remains in the original expert-major order.
__aicore__ inline void ReadCountFrames(__ubuf__ uint32_t* snapshot,
                                      GM_ADDR state,
                                      const LlCountReadPlan& plan, bool body)
{
    for (uint32_t i = 0; i < plan.size; ++i) {
        const auto& run = plan.runs[i];
        const uint64_t offset = body ? run.recordOffset : run.commitOffset;
        count_ub::Load(snapshot + run.first * 8U, state + offset, 32U,
                       run.records, uint64_t(plan.sourceStrideBlocks) * 32U);
    }
}
}


namespace deep_ep {
struct LlCountAssignment {
    uint32_t first, end, remoteRank, alignedRecords, remoteRecords;
};
__aicore__ __forceinline__ uint32_t CountRemotePeer(uint32_t rank, uint32_t world, uint32_t task)
{
    // Compact remote-rank enumeration skips this source's physical host.
    const uint32_t hostFirst = rank / kNumRanksPerHost * kNumRanksPerHost;
    return task < world - kNumRanksPerHost
        ? (task < hostFirst ? task : task + kNumRanksPerHost) : world;
}
__aicore__ __forceinline__ LlCountAssignment CountAssignment(uint32_t rank, uint32_t world, uint32_t experts,
    uint32_t cores, uint32_t core, bool allCore, bool directRanks)
{
    const uint32_t countCores = allCore ? cores : cores / 2U;
    uint32_t first = 0U, end = 0U;
    if (allCore || core >= cores / 2U) {
        const uint32_t worker = allCore ? core : core - cores / 2U;
        const uint32_t base = experts / countCores, extra = experts % countCores;
        first = worker * base + (worker < extra ? worker : extra);
        end = first + base + (worker < extra ? 1U : 0U);
    }
    if (directRanks) {
        const uint32_t hostExperts = kNumRanksPerHost * (experts / world);
        const uint32_t hostFirst = (rank / kNumRanksPerHost) * hostExperts;
        const uint32_t hostEnd = hostFirst + hostExperts;
        first = first < hostFirst ? hostFirst : first;
        end = end > hostEnd ? hostEnd : end;
        if (end < first) end = first;
    }
    const uint32_t records = end > first ? (experts + countCores - 1U) / countCores : 0U;
    const uint32_t remote = directRanks ? CountRemotePeer(rank, world, core) : world;
    const uint32_t remoteRecords = remote < world ? (experts / world + 1U + 7U) / 8U * 8U : 0U;
    return {first, end, remote, (records + 7U) / 8U * 8U, remoteRecords};
}
}


namespace deep_ep {
// A non-dedup core can own flat routes without owning a token-weight slice.
// If total routes are fewer than sender cores, only the first routeCount cores
// own routes; otherwise every sender owns at least one.
__aicore__ __forceinline__ bool LlSenderHasWork(bool dedup, uint32_t tokenCount,
    uint32_t routeCount, uint32_t core)
{
    return tokenCount != 0U || (!dedup && core < routeCount);
}

template <class Words>
class LlSplitRouteWords {
    Words first_, second_;
    uint32_t half_;
public:
    __aicore__ __forceinline__ LlSplitRouteWords(Words first, Words second, uint32_t half)
        : first_(first), second_(second), half_(half) {}
    __aicore__ __forceinline__ uint32_t GetValue(uint32_t index) const {
        return index < half_ ? first_.GetValue(index) : second_.GetValue(index - half_);
    }
    __aicore__ __forceinline__ void SetValue(uint32_t index, uint32_t value) {
        if (index < half_) first_.SetValue(index, value);
        else second_.SetValue(index - half_, value);
    }
};

// Two old q-reduction scratch arrays provide at least 2*routeCount slots. Use
// a direct histogram when experts fit, otherwise a <=50%-loaded hash table.
// A packed entry holds expert+1 in its low 16 bits and count in its high 16:
// the existing admission bounds E<=1024 and B*K<2^14 make both fields exact.
// Q remains immutable and independent of packet order after this forward pass.
template <class Ids, class Words>
__aicore__ __forceinline__ void LlBuildRouteQ(Ids ids, uint32_t routeCount, uint32_t topK, uint32_t experts,
    Words firstScratch, Words secondScratch, Words q, uint32_t halfSlots)
{
    if (routeCount <= topK) {
        // Every original q call returns zero for the first token. No histogram
        // or hash state is needed when the complete input is that one token.
        for (uint32_t index = 0; index < routeCount; ++index) q.SetValue(index, 0U);
        return;
    }
    LlSplitRouteWords<Words> table(firstScratch, secondScratch, halfSlots);
    const uint32_t slots = halfSlots * 2U;
    const bool dense = experts <= slots;
    const uint32_t initialized = dense ? experts : slots;
    for (uint32_t slot = 0; slot < initialized; ++slot) table.SetValue(slot, 0U);
    for (uint32_t index = 0; index < routeCount; ++index) {
        const uint32_t expert = uint32_t(ids.GetValue(index));
        if (dense) {
            const uint32_t previous = table.GetValue(expert);
            q.SetValue(index, index < topK ? 0U : previous);
            table.SetValue(expert, previous + 1U);
            continue;
        }
        uint32_t hash = expert * 0x9e3779b9U;
        hash ^= hash >> 16U;
        uint32_t slot = hash % slots;
        const uint32_t key = expert + 1U;
        uint32_t entry = table.GetValue(slot);
        while (entry != 0U && (entry & 0xffffU) != key) {
            if (++slot == slots) slot = 0U;
            entry = table.GetValue(slot);
        }
        const uint32_t previous = entry >> 16U;
        // Preserve the original first-token special case even for repeated IDs.
        q.SetValue(index, index < topK ? 0U : previous);
        table.SetValue(slot, key | ((previous + 1U) << 16U));
    }
}
}


namespace deep_ep {
// Cache one token in the existing input queue (plain/prequantized) or quantBuf
// (quantizing). Packet tails are mutable, so a cache hit must retire the prior
// Vector pack read before Scalar changes that tail. A miss retains the original
// queue/quantization dependencies. No packet lane is used as cached storage.
class LlSenderInputReuse {
    uint32_t token_ = 0;
    bool valid_ = false;
public:
    template <class Operations>
    __aicore__ __forceinline__ void Prepare(uint32_t token, Operations& operations) {
        if (valid_ && token == token_) {
            operations.WaitPackedRead();
            return;
        }
        if (valid_) operations.ReleaseInput();
        operations.LoadInput(token);
        token_ = token;
        valid_ = true;
    }
    template <class Operations>
    __aicore__ __forceinline__ void Finish(Operations& operations) {
        if (valid_) operations.ReleaseInput();
        valid_ = false;
    }
};

// Prequantized padding plan moved to layout/ep/token.hpp.
}


namespace deep_ep {
// Rotate an existing permutation; never change route ownership or destination.
// For a scheduled entry ordinal<count. count==0 has no entries; handle it
// defensively as well. Only one modulo is needed because ordinal+offset<2*count.
__aicore__ __forceinline__ uint32_t LlSendStaggerOrdinal(uint32_t count, uint32_t phase, uint32_t ordinal)
{
    if (count <= 1U) return 0U;
    const uint32_t index = ordinal + phase % count;
    return index < count ? index : index - count;
}

// Only rotate the current token's first-hop tasks. Expert selection is owned
// by the sender's topology policy; this function never changes that selection.
__aicore__ __forceinline__ uint32_t LlFirstHopOrder(uint32_t count, uint32_t rankInGroup,
    uint32_t core, uint32_t token, uint32_t ordinal)
{
    return LlSendStaggerOrdinal(count, rankInGroup + core + token, ordinal);
}

// Each source card/core enters the existing target-group permutation at a
// different point. Preserve the large-domain half-interleaving policy.
__aicore__ __forceinline__ uint32_t LlDirectGroupOrder(uint32_t groups, uint32_t ownGroup,
    uint32_t rankInGroup, uint32_t core, uint32_t ordinal,
    bool remoteFirst = false)
{
    if (remoteFirst && groups > 1U) {
        // Post RDMA before local copies so a singleton remote DIRECT is not
        // delayed by a long local group. Rotate only the remote permutation.
        // Remove ownGroup from the original order; keep its half-interleaving.
        if (ordinal == groups - 1U) return ownGroup;
        const uint32_t split = groups / 2U;
        const uint32_t ownLength = ownGroup < split ? split : groups - split;
        const uint32_t ownOrdinal = groups < 16U ? 0U :
            (ownLength > split ? groups - 1U : 2U * (ownLength - 1U));
        uint32_t remoteOrdinal = LlSendStaggerOrdinal(groups - 1U, rankInGroup + core, ordinal);
        if (remoteOrdinal >= ownOrdinal) ++remoteOrdinal;
        return EpServerOrder::ServerDedupDispatchDirectOrder(groups, ownGroup, remoteOrdinal);
    }
    return EpServerOrder::ServerDedupDispatchDirectOrder(
        groups, ownGroup, LlSendStaggerOrdinal(groups, rankInGroup + core, ordinal));
}
}  // namespace DispatchDedup


namespace deep_ep {
// Each target physical host contains four consecutive
// full-mesh groups of four ranks. Count actual expert deliveries, not distinct
// cards: each relay carries one equal-sized token for its target expert.
// Two scalar words suffice for K<=32; no UB array, route rescan or cross-core work.
class LlFullMeshFirstHop {
    uint32_t counts_ = 0U;
    uint32_t first_ = 0U;
public:
    __aicore__ __forceinline__ void Add(uint32_t destinationRank, uint32_t k) {
        const uint32_t shift = ((destinationRank & 15U) >> 2U) * 8U;
        counts_ += 1U << shift;
        if (((first_ >> shift) & 255U) == 0U) first_ |= (k + 1U) << shift;
    }
    __aicore__ __forceinline__ uint32_t FirstK() const {
        uint32_t bestCount = 0U, bestK = 32U;
        for (uint32_t group = 0U; group < 4U; ++group) {
            const uint32_t shift = group * 8U;
            const uint32_t count = (counts_ >> shift) & 255U;
            if (!count) continue;
            const uint32_t k = ((first_ >> shift) & 255U) - 1U;
            if (count > bestCount || (count == bestCount && k < bestK)) {
                bestCount = count;
                bestK = k;
            }
        }
        return bestCount ? bestK : 0U;
    }
};

// Classify supplies a mask for exactly one target host. Select independently
// for each mask; a single token-wide firstK is invalid with multiple hosts.
// Masks are disjoint, so each grouped route is loaded at most once here.
__aicore__ __forceinline__ uint32_t LlRelayMemberMask(uint32_t mask, uint32_t firstK) {
    return mask & ~(1U << firstK);
}
}

// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

namespace mxfp8_quant {
// MXFP8 keeps its original 32-element groups, exponent rules and cast rounding.
// Deinterleave pairs reduce 16 lanes per data block to one 32-input exponent.
template <class T>
__simd_vf__ inline void ComputeMaxExp(__ubuf__ T* src, __ubuf__ uint16_t* output, uint32_t count)
{
    const auto all = asc_create_mask_b16(PAT_ALL);
    vector_uint16_t exponentMask, invalidHalf;
    asc_duplicate_scalar(exponentMask, uint16_t(0x7f80U), all);
    asc_duplicate_scalar(invalidHalf, uint16_t(0x7c00U), all);
    vector_store_unalign unalign;
    for (uint32_t offset = 0, remaining = count; offset < count; offset += 256U) {
        const auto mask0 = asc_update_mask_b16(remaining);
        const auto mask1 = asc_update_mask_b16(remaining);
        vector_uint16_t input0, input1, exponent0, exponent1, maximum;
        asc_loadalign_deintlv(input0, input1, reinterpret_cast<__ubuf__ uint16_t*>(src + offset));
        if constexpr (std::is_same<T, half>::value) {
            vector_uint16_t selected0, selected1;
            asc_and(selected0, input0, invalidHalf, mask0);
            asc_and(selected1, input1, invalidHalf, mask0);
            const auto valid0 = asc_ne(selected0, invalidHalf, mask0);
            const auto valid1 = asc_ne(selected1, invalidHalf, mask0);
            vector_bfloat16_t bf0, bf1;
            asc_half2bfloat16_rz(bf0, reinterpret_cast<vector_half&>(input0), mask0);
            asc_half2bfloat16_rz(bf1, reinterpret_cast<vector_half&>(input1), mask0);
            asc_and(exponent0, reinterpret_cast<vector_uint16_t&>(bf0), exponentMask, mask0);
            asc_and(exponent1, reinterpret_cast<vector_uint16_t&>(bf1), exponentMask, mask0);
            asc_select(exponent0, exponent0, exponentMask, valid0);
            asc_select(exponent1, exponent1, exponentMask, valid1);
        } else {
            asc_and(exponent0, input0, exponentMask, mask0);
            asc_and(exponent1, input1, exponentMask, mask0);
        }
        (void)mask1;  // The admitted compute count is padded to complete pairs.
        asc_max(maximum, exponent0, exponent1, mask0);
        asc_reduce_max_datablock(maximum, maximum, mask0);
        asc_storeunalign(output + offset / 32U, unalign, maximum, 8U);
    }
    asc_storeunalign_post(output + count / 32U, unalign);
}

template <bool E4M3>
__simd_vf__ inline void ComputeScale(__ubuf__ uint16_t* maxExp, __ubuf__ uint16_t* mxScale,
                                    __ubuf__ uint16_t* reciprocal, uint32_t count)
{
    const auto all = asc_create_mask_b16(PAT_ALL);
    vector_uint16_t exponentMask, fp8Exponent, bias, fp8Nan, zero, nan, special;
    asc_duplicate_scalar(exponentMask, uint16_t(0x7f80U), all);
    asc_duplicate_scalar(fp8Exponent, uint16_t(E4M3 ? 0x0400U : 0x0780U), all);
    asc_duplicate_scalar(bias, uint16_t(0x7f00U), all);
    asc_duplicate_scalar(fp8Nan, uint16_t(0x00ffU), all);
    asc_duplicate_scalar(zero, uint16_t(0U), all);
    asc_duplicate_scalar(nan, uint16_t(0x7f81U), all);
    asc_duplicate_scalar(special, uint16_t(0x0040U), all);
    for (uint32_t offset = 0, remaining = count; offset < count; offset += 128U) {
        const auto mask = asc_update_mask_b16(remaining);
        vector_uint16_t maximum, shared, scale, inv;
        asc_loadalign(maximum, maxExp + offset);
        const auto valid = asc_ne(maximum, exponentMask, mask);
        const auto nonzero = asc_ne(maximum, zero, mask);
        const auto bounded = asc_le(maximum, fp8Exponent, mask);
        asc_select(maximum, fp8Exponent, maximum, bounded);
        asc_sub(shared, maximum, fp8Exponent, mask);
        asc_shiftright_scalar(scale, shared, int16_t(7), mask);
        asc_select(scale, scale, fp8Nan, valid);
        asc_select(scale, scale, zero, nonzero);
        // Pack the low byte of each uint16 into the existing E8M0 scale packs.
        asc_storealign_pack(mxScale + offset / 2U, scale, mask);
        const auto isSpecial = asc_eq(shared, bias, mask);
        asc_sub(inv, bias, shared, mask);
        asc_select(inv, inv, nan, valid);
        asc_select(inv, inv, zero, nonzero);
        asc_select(inv, special, inv, isSpecial);
        asc_storealign(reciprocal + offset, inv, mask);
    }
}

template <bool E4M3>
__simd_callee__ inline void StoreFp8(__ubuf__ int8_t* dst, vector_float values, vector_bool mask)
{
    constexpr auto quarter = std::integral_constant<asc_position_quarter_mode,
        asc_position_quarter_mode::DISPERSE_FIRST_QUARTER>{};
    if constexpr (E4M3) {
        vector_fp8_e4m3fn_t fp8;
        asc_float2e4m3_rn_sat(fp8, values, mask, quarter);
        asc_storealign_pack_quarter(reinterpret_cast<__ubuf__ float*>(dst),
            reinterpret_cast<vector_float&>(fp8), mask);
    } else {
        vector_fp8_e5m2_t fp8;
        asc_float2e5m2_rn_sat(fp8, values, mask, quarter);
        asc_storealign_pack_quarter(reinterpret_cast<__ubuf__ float*>(dst),
            reinterpret_cast<vector_float&>(fp8), mask);
    }
}

template <class T, bool E4M3>
__simd_vf__ inline void ComputeFp8Data(__ubuf__ T* src, __ubuf__ uint16_t* reciprocal,
                                      __ubuf__ int8_t* dst, uint32_t count)
{
    const auto all = asc_create_mask_b16(PAT_ALL);
    constexpr auto even = std::integral_constant<asc_position_mode, asc_position_mode::EVEN>{};
    constexpr auto odd = std::integral_constant<asc_position_mode, asc_position_mode::ODD>{};
    uint32_t doubled = count * 2U, remaining = count;
    for (uint32_t offset = 0; offset < count; offset += 256U) {
        const auto input0Mask = asc_update_mask_b16(remaining);
        const auto input1Mask = asc_update_mask_b16(remaining);
        const auto float0Mask = asc_update_mask_b16(doubled);
        const auto float1Mask = asc_update_mask_b16(doubled);
        vector_uint16_t raw0, raw1, scales;
        vector_float f00, f01, f10, f11;
        asc_loadalign_deintlv(raw0, raw1, reinterpret_cast<__ubuf__ uint16_t*>(src + offset));
        asc_loadalign_brc_elem2datablock(scales, reciprocal + offset / 32U);
        if constexpr (std::is_same<T, half>::value) {
            vector_float fpScale;
            asc_half2float(f00, reinterpret_cast<vector_half&>(raw0), input0Mask, even);
            asc_half2float(f01, reinterpret_cast<vector_half&>(raw0), input0Mask, odd);
            asc_bfloat162float(fpScale, reinterpret_cast<vector_bfloat16_t&>(scales), all, even);
            asc_mul(f00, f00, fpScale, float0Mask);
            asc_mul(f01, f01, fpScale, float1Mask);
            asc_intlv(f00, f01, f00, f01);
            asc_half2float(f10, reinterpret_cast<vector_half&>(raw1), input0Mask, even);
            asc_half2float(f11, reinterpret_cast<vector_half&>(raw1), input0Mask, odd);
            asc_mul(f10, f10, fpScale, float0Mask);
            asc_mul(f11, f11, fpScale, float1Mask);
            asc_intlv(f10, f11, f10, f11);
            asc_intlv(f00, f10, f00, f10);
            asc_intlv(f01, f11, f01, f11);
            StoreFp8<E4M3>(dst + offset, f00, float0Mask);
            StoreFp8<E4M3>(dst + offset + 64U, f10, float0Mask);
            StoreFp8<E4M3>(dst + offset + 128U, f01, float1Mask);
            StoreFp8<E4M3>(dst + offset + 192U, f11, float1Mask);
        } else {
            vector_bfloat16_t input0, input1;
            asc_mul(input0, reinterpret_cast<vector_bfloat16_t&>(raw0),
                    reinterpret_cast<vector_bfloat16_t&>(scales), input0Mask);
            asc_mul(input1, reinterpret_cast<vector_bfloat16_t&>(raw1),
                    reinterpret_cast<vector_bfloat16_t&>(scales), input0Mask);
            asc_intlv(reinterpret_cast<vector_uint16_t&>(input0), reinterpret_cast<vector_uint16_t&>(input1),
                      reinterpret_cast<vector_uint16_t&>(input0), reinterpret_cast<vector_uint16_t&>(input1));
            asc_bfloat162float(f00, input0, input0Mask, even);
            asc_bfloat162float(f01, input0, input0Mask, odd);
            asc_intlv(f00, f01, f00, f01);
            asc_bfloat162float(f10, input1, input0Mask, even);
            asc_bfloat162float(f11, input1, input0Mask, odd);
            asc_intlv(f10, f11, f10, f11);
            StoreFp8<E4M3>(dst + offset, f00, float0Mask);
            StoreFp8<E4M3>(dst + offset + 64U, f01, float0Mask);
            StoreFp8<E4M3>(dst + offset + 128U, f10, float1Mask);
            StoreFp8<E4M3>(dst + offset + 192U, f11, float1Mask);
        }
        (void)input1Mask;
    }
}
}  // namespace mxfp8_quant

#ifdef DEEPEP_LL_REFERENCE_PROFILE
#endif
#if defined(DEEPEP_LL_STAGE_TIMER)
#error Legacy LL stage/expert instrumentation has been removed
#endif

namespace profile {
// Count-boundary micro profiling is independent of the schema-4 reference
// record. Mode 1 keeps the shared record limited to the parsed 29 stages; the
// count call sites still compile, so the reference-profile identity and the
// runtime wheel stay one variant.
constexpr uint32_t kCountComputeMode = 1U;
// Reference dispatch style: one clock and one scalar GM store per fixed point.
// Packet timing stays in local Scalar state. Receiver identities use spare UB
// and are copied only after business completion; no live lane is removed.
class DeviceProfile {
public:
    __aicore__ static inline uint64_t EntryClock(GM_ADDR address)
    {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        return address ? AscendC::GetSystemCycle() : 0U;
#else
        return 0U;
#endif
    }
    __aicore__ inline void Init(GM_ADDR address)
    {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        record_ = address ? reinterpret_cast<__gm__ uint64_t*>(address) +
                            AscendC::GetBlockIdx() * kWordsPerCore : nullptr;
        valid_ = 0U;
        for (uint32_t i = 0; i < 32U; ++i) send_[i] = 0U;
        sendRemote_ = sendGroup_ = false;
#endif
    }
    __aicore__ inline bool Enabled() const
    {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        return record_ != nullptr;
#else
        return false;
#endif
    }
    __aicore__ inline uint64_t Clock() const
    {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        return record_ ? AscendC::GetSystemCycle() : 0U;
#else
        return 0U;
#endif
    }
    template <uint32_t Word> __aicore__ inline void Value(uint64_t value)
    {
        static_assert(Word <= 21U || (Word >= 24U && Word <= 49U) || Word == 57U || Word == 62U,
                      "reference point/counter index");
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (!record_) return;
        record_[Word] = value;
        valid_ |= uint64_t(1) << Word;
#endif
    }
    template <uint32_t Word> __aicore__ inline void Mark()
    {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (record_) Value<Word>(AscendC::GetSystemCycle());
#endif
    }
    // Count-call boundaries. The validated schema-4 record has no count
    // sub-phase storage (its 29 stages come from Marks and the packet detail
    // words), so mode 1 keeps these as typed no-ops. Keeping the names on the
    // profile object is what lets the Count C API read the same source in both
    // wheel identities.
    __aicore__ inline uint64_t CountClock() const { return Clock(); }
    __aicore__ inline uint64_t PublishClock() const { return Clock(); }
    template <uint32_t Word, bool Publish = false> __aicore__ inline void CountBoundary() {}
    template <uint32_t Word> __aicore__ inline void CompletePublishAdd(uint64_t) {}
    template <uint32_t Word> __aicore__ inline void PublishAdd(uint64_t) {}
    __aicore__ inline void CompleteCountCall(bool, uint32_t) {}
    __aicore__ inline bool CountNeedsCalibration() const { return false; }
    __aicore__ inline void CountCalibration(uint64_t, uint64_t) {}
    __aicore__ inline void CountSections(bool, uint32_t, uint64_t, uint64_t, uint64_t,
                                         uint64_t, uint64_t) {}
    __aicore__ inline void CountWirePack(uint64_t) {}
    __aicore__ inline void CountPublish(bool, uint64_t) {}
    __aicore__ inline void CountPostSV(uint64_t) {}
    __aicore__ inline void CountRoute(uint64_t) {}
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    __aicore__ inline void SendBegin(bool remote, bool group, uint32_t token, uint32_t k,
                                    uint32_t expert, uint32_t peer) {
        if (!record_) return;
        sendRemote_ = remote; sendGroup_ = group;
        const uint64_t order = ++send_[24];
        if (remote) {
            const uint64_t n = ++send_[1];
            if (n == 1U) { send_[3] = Clock(); send_[20] = order; }
            send_[21] = order;
            if (n == 2U) {
                send_[7] = Clock(); send_[12] = token; send_[13] = k;
                send_[14] = expert; send_[15] = peer;
            }
        }
        if (group) {
            if (++send_[2] == 1U) { send_[5] = Clock(); send_[22] = order; }
            send_[23] = order;
        }
    }
    __aicore__ inline bool SecondRemote() const {
        return record_ && sendRemote_ && send_[1] == 2U;
    }
    __aicore__ inline void SecondEnqueue() { if (SecondRemote()) send_[8] = Clock(); }
    __aicore__ inline void SecondPutBegin(bool selected) { if (record_ && selected) send_[9] = Clock(); }
    __aicore__ inline void SecondPutBegin() { SecondPutBegin(SecondRemote()); }
    __aicore__ inline void PutReturned(bool selected) {
        if (!record_) return;
        const uint64_t now = Clock();
        send_[4] = now;
        if (selected) send_[10] = now;
    }
    __aicore__ inline void PutReturned() { PutReturned(SecondRemote()); }
    __aicore__ inline void StageQueued(bool selected, bool deferred) {
        if (!record_) return;
        if (deferred) ++send_[29];
        if (selected) { send_[26] = Clock(); send_[31] = deferred ? 1U : 0U; }
    }
    __aicore__ inline void StageWaitBegin(bool selected) { if (record_ && selected) send_[27] = Clock(); }
    __aicore__ inline void StageWaitEnd(bool selected) { if (record_ && selected) send_[28] = Clock(); }
    __aicore__ inline void PreviousSubmit(uint64_t ticks) { if (SecondRemote()) send_[30] += ticks; }
    __aicore__ inline void GroupSubmitted() { if (record_ && sendGroup_) send_[6] = Clock(); }
    __aicore__ inline void MemberQ(uint64_t ticks) { if (SecondRemote()) send_[11] += ticks; }
    __aicore__ inline void FlushDone(uint64_t begin, bool secondTriggered) {
        if (!record_) return;
        const uint64_t end = Clock();
        send_[16] += end - begin; ++send_[17]; send_[18] = end;
        if (secondTriggered) send_[19] = 1U;
    }
    __aicore__ inline void SendLanes(uint32_t lanes) { if (record_) send_[25] = lanes; }
    __aicore__ inline void Detail(uint32_t word, uint64_t value) { if (record_) record_[word] = value; }
    __aicore__ inline GM_ADDR PacketLogAddress() const {
        return record_ ? reinterpret_cast<GM_ADDR>(record_ + kPacketLogWord) : nullptr;
    }
#endif
    __aicore__ inline void Finish(uint32_t rank, uint32_t world, uint32_t bs)
    {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (!record_) return;
        Mark<57>();  // w57: before_finalize; includes post-business packet-log export
        send_[0] = 1U;
        for (uint32_t i = 0; i < 32U; ++i) record_[64U + i] = send_[i];
        record_[50] = rank;                         // w50: rank (metadata, not a clock)
        record_[51] = AscendC::GetBlockIdx();        // w51: AIV_core
        record_[52] = world;                        // w52: EP
        record_[53] = AscendC::GetBlockNum();        // w53: active_aiv
        record_[54] = bs;                           // w54: BS
        record_[55] = kSchema;                      // w55: schema
        record_[56] = 1U;                           // w56: operation, dispatch only
        record_[58] = valid_;                       // w58: valid_mask, absent != zero
        record_[59] = kMagic;                       // w59: magic
        record_[60] = 1U;                           // w60: boundary_mode, observe
        record_[61] = 1U;                           // w61: complete
        // Python supplies a fresh zeroed tensor; reserved/absent words remain
        // zero. Publish Scalar header/meta after business finish and log DMA.
        // This matches the reference dispatch cache publication, not PIPE_ALL.
        // Pinned SHMEM's helper includes the cacheline containing addr+length.
        // Use the last byte so an aligned record never flushes its neighbor.
        dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(record_), kPacketLogWord * sizeof(uint64_t) - 1U);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
#endif
    }
private:
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    __gm__ uint64_t* record_ = nullptr;
    uint64_t valid_ = 0U;
    uint64_t send_[32]{};
    bool sendRemote_ = false, sendGroup_ = false;
#endif
};
}  // namespace LlReferenceProfile

// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026



namespace deep_ep {
using namespace count_ub;

// 中文审计：这里包含跨rank计数交换，不属于four_metadata测得的本地路由阶段。
// 修改计数协议需联查dispatch_server_dedup_kernel.h中的阶段顺序及Receiver使用的前缀布局。
// Adapted from Memory's CalAndSendCntByRankAndExpert / WaitDispatch / cumsum
// chain. Counts cover the complete original routing tensor, including relay routes.
// 本类没有 Process()：上层依次调用 SendCnt，再由接收核执行
// RecvCnt → CalcCumsum → CompleteCumsum；发送核在 SendCnt 后进入 token 发送。
// 计数覆盖原始专家路由（含后续 relay 路由），不按去重后的跨 server 包数计数。
template <uint32_t kNumMaxTokensPerRank>
class ServerDedupCounts {
public:
    // 摘要：绑定路由与计数输出，选择本轮控制区；保存并调整 CTRL[60]，供整数位模式的 FP32 归约使用。
    __aicore__ inline void Init(const TransportView& transport, GM_ADDR expertIds, GM_ADDR expertTokenNumsOut,
                                GM_ADDR sendCountsOut, uint32_t num_tokens, uint32_t topK, uint32_t experts,
                                uint32_t expertTokenNumsType, int64_t magic, CoreContext core,
                                profile::DeviceProfile* profile = nullptr)
    {
        profile_ = profile;
        view_ = transport;
        epRankId_ = view_.rank;
        epWorldSize_ = view_.num_ranks;
        aivId_ = core.index;
        coreNum_ = core.count;
        aivUsedAllToAll_ = aivUsedCumSum_ = coreNum_ / 2U;
        num_tokens_ = num_tokens;
        expertIdsCnt_ = num_tokens * topK;
        ubRouteCount_ = kNumMaxTokensPerRank * topK;
        rscvStatusNum_ = experts;
        moeExpertNumPerRank_ = experts / epWorldSize_;
        expertTokenNumsType_ = expertTokenNumsType;
        useAllCoreCount_ = kNumMaxTokensPerRank < kEpServerDedupAllCoreCountBsThreshold;
        packedFrames_ = view_.is_rdma() && epWorldSize_ > 16U;
        directRanks_ = view_.is_rdma() && epWorldSize_ > kNumRanksPerHost;
        countWindow_ = DispatchServerDedupWindowOffset(magic);
        retirement_.Init(view_);
        layout_ = DispatchServerDedupCountLayout(experts, coreNum_);
        stateOffset_ = DispatchServerDedupWindowOffset(magic) + layout_.cntOffset;
        GM_ADDR localState = view_.peer_ptr(epRankId_) + stateOffset_;
        stateGM_ = localState;
        // 计数记录与 cumsum 槽位都由本机 L2 可达的写者更新（远端 RDMA 与同机 MTE3），
        // 读侧保持默认缓存视图才能立即看到新值；绕过 L2 会读到尚未回写的 HBM，
        // 表现为 ready 轮询多等十几微秒。
        recvCntWorkspaceGM_ = localState + layout_.prefixOffset;
        sendCountsOutGM_ = sendCountsOut;
        expertIdsGM_ = expertIds;
        expertTokenNumsGM_ = expertTokenNumsOut;
        // Memory's count reductions add integer bit patterns as FP32 subnormals.
        originalOverflowMode_ = uint64_t(asc_get_ctrl()) & (1ULL << 60U);
        asc_set_ctrl(uint64_t(asc_get_ctrl()) & ~(1ULL << 60U));
    }

    // 摘要：统计完整原始路由的专家计数并异步发往专家所属 rank，同时保留仍被 MTE3 使用的记录 UB。
    __aicore__ inline void SendCnt();
    // 摘要：将本 rank 的计数记录分给接收核，等待本片计数到齐并计算局部前缀与总数。
    __aicore__ inline void RecvCnt();
    // 摘要：汇合接收核片段总数，补齐跨核前缀偏移并发布完整前缀表的就绪标记。
    __aicore__ inline void CalcCumsum()
    {
        CalRecvAndSetFlag();
    }
    // 摘要：等待所有前缀片段就绪，仅首个接收核加载前缀并输出各专家计数。
    __aicore__ inline void CompleteCumsum();
    // 摘要：返回本核计数发送记录占用的 UB 字节数，供后续阶段计算剩余容量。
    __aicore__ inline uint32_t ReservedBytes() const
    {
        return statusCntAlign_ * kEpServerDedupCountRecordBytes;
    }
    __aicore__ inline comm::CountRetirement* Retirement() { return &retirement_; }
    __aicore__ inline void RetireOwnScratch() { retirement_.Finish(reinterpret_cast<__ubuf__ uint8_t*>(histogram_)); }
    // 摘要：等待本核全部流水收尾，并恢复初始化前的 CTRL[60] 设置。
    __aicore__ inline void Finish()
    {
        asc_sync_pipe(PIPE_ALL);
        asc_set_ctrl((uint64_t(asc_get_ctrl()) & ~(1ULL << 60U)) | originalOverflowMode_);
    }

private:
    static constexpr uint32_t UB_ALIGN = 32U;
    static constexpr uint32_t UB_ALIGN_DATA_COUNT = 8U;
    static constexpr uint32_t SIZE_ALIGN_256 = 256U;
    uint32_t aivUsedAllToAll_ = 0, coreNum_ = 0;
    uint32_t aivUsedCumSum_ = 0;

    __aicore__ inline uint32_t Align(uint32_t bytes, uint32_t alignment = 32U)
    { return (bytes + alignment - 1U) / alignment * alignment; }
    __aicore__ inline __ubuf__ uint32_t* Scratch(uint32_t bytes)
    {
        auto ptr = Ub<uint32_t>(scratchEnd_);
        scratchEnd_ += Align(bytes);
        return ptr;
    }
    // 摘要：将连续区间均分到各核，余数优先分配给前面的核，返回半开区间及长度。
    __aicore__ inline void SplitToCore(uint32_t total, uint32_t cores, uint32_t core, uint32_t& start, uint32_t& end,
                                       uint32_t& count)
    {
        const uint32_t base = total / cores;
        const uint32_t remainder = total % cores;
        count = base + (core < remainder ? 1U : 0U);
        start = core * base + (core < remainder ? core : remainder);
        end = start + count;
    }
    // 摘要：将专家计数按目标 rank 分组，写到远端“本地专家 × 来源 rank”的 32B 记录槽。
    __aicore__ inline void SendExpertCntRange(uint32_t start, uint32_t end);
    // 摘要：清零槽位并批量打 ready，再用硬件直方图回填 count；本机专家区间与跨机远端 rank 共用。
    __aicore__ inline void CountSlots(__ubuf__ uint32_t* records, uint32_t first, uint32_t bins,
                                     bool wire = false, bool remoteTail = false);
    __aicore__ inline void PackLocalCountWire(uint32_t records);
    // 摘要：清除本核已收取记录的 ready 标记，保留计数字段，为后续窗口复用做准备。
    __aicore__ inline void WaitDispatchClearStatus();
    // 摘要：提取计数字段，计算本片累计前缀，并向其他接收核发布本片总数。
    __aicore__ inline void GatherSumRecvCnt();
    // 摘要：将本核片段总数和就绪标记写入本 rank 的跨核汇总矩阵；空片段也必须发布。
    __aicore__ inline void PublishRecvCntSum(uint32_t sumOfRecvCnt);
    // 摘要：等待所有接收核总数就绪，求本核之前各片段之和，再清理本核消费的汇总标记。
    __aicore__ inline uint32_t GetCumSum(uint32_t newAivId);
    // 摘要：轮询本片所有来源计数的 ready 标记；到齐后提取计数、发布片段总数并清除 ready。
    __aicore__ inline void WaitDispatch();
    // 摘要：为本片局部前缀加上跨核偏移，写出 sendCountsOut 与共享前缀，再发布完成标记。
    __aicore__ inline void CalRecvAndSetFlag();
    // 摘要：等待所有接收核的前缀写出标记到齐，并清除本核对应的标记行。
    __aicore__ inline void WaitCumSumFlag();

    profile::DeviceProfile* profile_ = nullptr;
    TransportView view_{};
    EpServerDedupCountLayout layout_{};
    uint64_t stateOffset_ = 0;
    uint64_t originalOverflowMode_ = 0;
    uint32_t epRankId_ = 0, epWorldSize_ = 0, aivId_ = 0;
    uint32_t expertIdsCnt_ = 0, num_tokens_ = 0, ubRouteCount_ = 0;
    uint32_t rscvStatusNum_ = 0, moeExpertNumPerRank_ = 0, expertTokenNumsType_ = 0;
    uint32_t statusCntAlign_ = 0;
    uint32_t startStatusIndex_ = 0, endStatusIndex_ = 0, recStatusNumPerCore_ = 0;
    uint32_t activeCumSumCores_ = 0;
    bool useAllCoreCount_ = false;
    bool packedFrames_ = false;
    bool directRanks_ = false;
    uint64_t countWindow_ = 0U;
    comm::CountRetirement retirement_;
    GM_ADDR recvCntWorkspaceGM_ = nullptr;
    GM_ADDR sendCountsOutGM_ = nullptr;
    GM_ADDR expertIdsGM_ = nullptr;
    GM_ADDR expertTokenNumsGM_ = nullptr;
    GM_ADDR stateGM_ = nullptr;
    // Retained records occupy [0, ReservedBytes()); all scratch follows them.
    uint32_t scratchEnd_ = 0;
    __ubuf__ uint32_t* records_ = nullptr;
    __ubuf__ int32_t* routes_ = nullptr;
    __ubuf__ uint16_t* histogram_ = nullptr;
    __ubuf__ uint32_t* work_ = nullptr;
    __ubuf__ uint32_t* scalar_ = nullptr;
    __ubuf__ uint32_t* snapshot_ = nullptr;
    __ubuf__ uint32_t* dense_ = nullptr;
    __ubuf__ uint32_t* sumSlot_ = nullptr;
    __ubuf__ uint32_t* sumSnapshot_ = nullptr;
    __ubuf__ uint32_t* sumDense_ = nullptr;
    __ubuf__ uint32_t* ready_ = nullptr;
    __ubuf__ uint32_t* clean_ = nullptr;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    bool countRemote_ = false;
#endif
};

// 摘要：统计完整原始路由的专家计数并异步发往专家所属 rank，同时保留仍被 MTE3 使用的记录 UB。
template <uint32_t kNumMaxTokensPerRank>
__aicore__ inline void ServerDedupCounts<kNumMaxTokensPerRank>::SendCnt()
{
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t routeBegin = profile_ ? profile_->CountClock() : 0U;
#endif
    // Same-host work keeps the original expert split. Each physical core owns
    // remote rank tasks core, core+coreNum, ... independent of the BS threshold.
    const uint32_t cores = useAllCoreCount_ ? coreNum_ : aivUsedCumSum_;
    const auto assignment = CountAssignment(epRankId_, epWorldSize_, rscvStatusNum_,
                                            coreNum_, aivId_, useAllCoreCount_, directRanks_);
    const uint32_t start = assignment.first, end = assignment.end, count = end - start;
    const bool remoteOwner = assignment.remoteRank < epWorldSize_;
    // Only local records stay reserved into payload. MTE3_S retires the remote
    // UB frame copy; its isolated GM source survives independently until CQ retirement.
    statusCntAlign_ = packedFrames_ ? CountWireBytes(count) / UB_ALIGN : assignment.alignedRecords;
    records_ = Ub<uint32_t>(0U);
    scratchEnd_ = ReservedBytes();
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    if (profile_) profile_->CountBoundary<1>();
#endif
    if (count == 0U && !remoteOwner) return;

    // 3. 为完整 bs×topK 路由及直方图临时数据分配 UB，等待 GM 路由搬入。
    const uint32_t routeBytes = Align(ubRouteCount_ * sizeof(int32_t), SIZE_ALIGN_256);
    const uint32_t remoteBytes = assignment.remoteRecords * UB_ALIGN;
    const uint32_t workBytes = routeBytes < remoteBytes ? remoteBytes : routeBytes;
    routes_ = reinterpret_cast<__ubuf__ int32_t*>(Scratch(routeBytes));
    histogram_ = reinterpret_cast<__ubuf__ uint16_t*>(Scratch(routeBytes));
    work_ = Scratch(workBytes);
    Load(routes_, expertIdsGM_, expertIdsCnt_ * sizeof(int32_t));
    Sync<PIPE_MTE2, PIPE_V>();
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    if (profile_) profile_->CountBoundary<2>();
    if (profile_) profile_->CountRoute(profile_->CountClock() - routeBegin);
#endif
    if (remoteOwner) {
        auto remoteFrame = work_;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        uint32_t firstFrameIssued = 0U;
#endif
        for (uint32_t task = aivId_; task < epWorldSize_ - kNumRanksPerHost; task += coreNum_) {
            const uint32_t destination = CountRemotePeer(epRankId_, epWorldSize_, task);
            const uint32_t first = destination * moeExpertNumPerRank_;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            countRemote_ = true;
#endif
            CountSlots(remoteFrame, first, moeExpertNumPerRank_, false, true);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            if (profile_) profile_->CountBoundary<12>();
            const uint64_t publishBegin = profile_ ? profile_->CountClock() : 0U;
            const uint64_t p0 = profile_ ? profile_->PublishClock() : 0U;
#endif
            Sync<PIPE_S, PIPE_MTE3>();
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            const uint64_t p1 = profile_ ? profile_->PublishClock() : 0U;
            if (profile_) profile_->CountBoundary<2, true>();
#endif
            const auto frame = CountFrame(rscvStatusNum_, epWorldSize_, cores,
                                          destination, epRankId_, 0U, true);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            const uint64_t p2 = profile_ ? profile_->PublishClock() : 0U;
            if (profile_) profile_->CountBoundary<3, true>();
#endif
            comm::DirectCountFrame(view_, destination, stateOffset_ + frame.frameOffset,
                reinterpret_cast<__ubuf__ uint8_t*>(remoteFrame), moeExpertNumPerRank_,
                reinterpret_cast<__ubuf__ uint8_t*>(histogram_), countWindow_
#ifdef DEEPEP_LL_REFERENCE_PROFILE
                , profile_
#endif
                );
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            const uint64_t p3 = profile_ ? profile_->PublishClock() : 0U;
#endif
            retirement_.Posted(task);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            if (profile_) {
                profile_->CountBoundary<9, true>();
                profile_->CompletePublishAdd<1>(1U);
                const uint64_t p4 = profile_->PublishClock();
                profile_->PublishAdd<1>(1U); profile_->PublishAdd<2>(p1-p0);
                profile_->PublishAdd<3>(p2-p1); profile_->PublishAdd<9>(p4-p3);
                profile_->CountPublish(true, profile_->CountClock() - publishBegin);
                if (!firstFrameIssued) { profile_->Mark<44>(); firstFrameIssued = 1U; }  // w44: send_count_first_frame_issued
                // Only the last frame of this core's stride gets a clock so the
                // per-frame post cost is not inflated by repeated scalar stores.
                if (task + coreNum_ >= epWorldSize_ - kNumRanksPerHost) profile_->Mark<45>();  // w45: send_count_last_frame_issued
            }
#endif
            // Source GM slots are disjoint. Acquire the SDK's retired UB scratch
            // before the next Vector pass; no network completion wait here.
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            const uint64_t postSVBegin = profile_ ? profile_->CountClock() : 0U;
            if (profile_) profile_->CountBoundary<12>();
#endif
            Sync<PIPE_S, PIPE_V>();
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            if (profile_) {
                profile_->CountBoundary<11>();
                profile_->CountPostSV(profile_->CountClock() - postSVBegin);
            }
#endif
        }
    }

    // 4. 本机专家区间统一走硬件直方图；零计数专家也发布 ready=1 和 count=0，避免对端永久等待。
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    countRemote_ = false;
#endif
    if (count) {
        CountSlots(records_, start, count, packedFrames_);
        if (packedFrames_) PackLocalCountWire(count);
    }

    // 5. Scalar 填好的记录通过 MTE3 写到目标专家所属 rank。
    // 此处仅下发，不全量等待：records_ 保持不变，而筛选临时 UB 可以交给后续阶段。
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t localPublishBegin = profile_ ? profile_->CountClock() : 0U;
    const uint64_t l0 = count && profile_ ? profile_->PublishClock() : 0U;
    if (profile_) profile_->CountBoundary<13>();
#endif
    Sync<PIPE_S, PIPE_MTE3>();
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t l1 = count && profile_ ? profile_->PublishClock() : 0U;
    if (profile_) profile_->CountBoundary<12, true>();
#endif
    if (count) SendExpertCntRange(start, end);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    if (count && profile_) {
        profile_->CompletePublishAdd<10>(1U);
        profile_->PublishAdd<10>(1U); profile_->PublishAdd<12>(l1-l0);
        profile_->CountPublish(false, profile_->CountClock() - localPublishBegin);
    }
#endif
    // Only records_ is still read by MTE3. V_S retired the shared scratch.
    // Keep CTRL[60] cleared through the subsequent FP8 quantization.
}

// 本机专家区间与跨机远端 rank 共用：清零对齐槽、批量打 ready，再用一次硬件直方图回填 count。
// 跨机帧同时初始化尾 ready 记录；本机 wire 初始化所有数据及 Flag 块。
template <uint32_t kNumMaxTokensPerRank>
__aicore__ inline void ServerDedupCounts<kNumMaxTokensPerRank>::CountSlots(__ubuf__ uint32_t* records, uint32_t first, uint32_t bins,
                                                    bool wire, bool remoteTail)
{
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    if (profile_) {
        if (countRemote_) profile_->CountBoundary<12>();
        else profile_->CountBoundary<13>();
        profile_->CompleteCountCall(countRemote_, bins);
    }
    if (profile_ && profile_->CountNeedsCalibration()) {
        const uint64_t c0 = profile_->CountClock(), c1 = profile_->CountClock();
        uint64_t emptyWait = 0U;
        if constexpr (profile::kCountComputeMode == 2U) {
            // MTE2_V submits a Vector dependency. Retire it before measuring an
            // idle V_S so the calibration cannot include the initial route DMA.
            Sync<PIPE_V, PIPE_S>();
            const uint64_t e0 = profile_->Clock();
            Sync<PIPE_V, PIPE_S>();
            emptyWait = profile_->Clock() - e0;
        }
        profile_->CountCalibration(c1 - c0, emptyWait);
    }
    const uint64_t t0 = profile_ ? profile_->CountClock() : 0U;
#endif
    const uint32_t slots = (bins + (remoteTail ? 1U : 0U) + 7U) / 8U * 8U;
    Fill(records, 0U, wire ? CountWireBytes(bins) / sizeof(int32_t) : slots * UB_ALIGN_DATA_COUNT);
    asc_sync_pipe(PIPE_V);
    RecordReady(records, 0x3f800000U, slots);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    if constexpr (profile::kCountComputeMode == 2U) {
        if (profile_ && profile_->Enabled()) Sync<PIPE_V, PIPE_S>();
    }
    const uint64_t t1 = profile_ ? profile_->CountClock() : 0U;
    if (profile_) {
        if (countRemote_) profile_->CountBoundary<3>();
        else profile_->CountBoundary<7>();
    }
#endif
    histogram_accumulate_u16(routes_, histogram_,
                             static_cast<int>(expertIdsCnt_), static_cast<int>(first),
                             static_cast<int>(bins));
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t t2 = profile_ ? profile_->CountClock() : 0U;
    if (profile_) {
        if (countRemote_) profile_->CountBoundary<4>();
        else profile_->CountBoundary<8>();
    }
#endif
    Sync<PIPE_V, PIPE_S>();
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t t3 = profile_ ? profile_->CountClock() : 0U;
    if (profile_) {
        if (countRemote_) profile_->CountBoundary<5>();
        else profile_->CountBoundary<9>();
    }
#endif
    for (uint32_t i = 0; i < bins; ++i)
        records[i * UB_ALIGN_DATA_COUNT + 1U] = uint32_t(histogram_[i + 1U]);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    if (profile_) {
        if (countRemote_) profile_->CountBoundary<6>();
        else profile_->CountBoundary<10>();
        profile_->CountSections(countRemote_, bins, t0, t1, t2, t3, profile_->CountClock());
    }
#endif
}

template <uint32_t kNumMaxTokensPerRank>
__aicore__ inline void ServerDedupCounts<kNumMaxTokensPerRank>::PackLocalCountWire(uint32_t records)
{
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t wirePackBegin = profile_ ? profile_->PublishClock() : 0U;
    if (profile_) profile_->CountBoundary<13>();
#endif
    // Reverse order preserves dense sources while inserting one 32B flag per
    // fifteen records. Every destination reads this same image without edits.
    for (uint32_t i = records; i > kLlCountWireRecords;) {
        --i;
        const uint32_t count = records_[i * UB_ALIGN_DATA_COUNT + 1U];
        const uint32_t target = CountWireRecordOffset(i) / sizeof(int32_t);
        records_[target] = 0x3f800000U;
        records_[target + 1U] = count;
    }
    for (uint32_t block = 0; block < CountWireBytes(records); block += kLlCountWireBlockBytes)
        for (uint32_t word = 0; word < UB_ALIGN_DATA_COUNT; ++word)
            records_[(block + kLlCountWireFlagOffset) / sizeof(int32_t) + word] = 0x3f800000U;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    if (profile_) {
        profile_->CountBoundary<16, true>();
        profile_->CompletePublishAdd<17>(1U);
        profile_->CountWirePack(profile_->PublishClock() - wirePackBegin);
    }
#endif
}

// 摘要：将专家计数按目标 rank 分组，写到远端“本地专家 × 来源 rank”的 32B 记录槽。
template <uint32_t kNumMaxTokensPerRank>
__aicore__ inline void ServerDedupCounts<kNumMaxTokensPerRank>::SendExpertCntRange(uint32_t start, uint32_t end)
{
    for (uint32_t expert = start; expert < end;) {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        const uint64_t planBegin = profile_ ? profile_->PublishClock() : 0U;
        if (profile_) profile_->CountBoundary<13>();
#endif
        const uint32_t dst = expert / moeExpertNumPerRank_;
        const uint32_t localExpert = expert % moeExpertNumPerRank_;
        const uint32_t count =
            end - expert < moeExpertNumPerRank_ - localExpert ? end - expert : moeExpertNumPerRank_ - localExpert;
        const uint64_t offset = (uint64_t(localExpert) * epWorldSize_ + epRankId_) * kEpServerDedupCountGmStride;
        if (packedFrames_) {
            const auto frame = CountFrame(rscvStatusNum_, epWorldSize_,
                useAllCoreCount_ ? coreNum_ : aivUsedCumSum_, dst, epRankId_, localExpert, directRanks_);
            if (frame.firstExpert != localExpert || frame.records != count) { trap(); return; }
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            const uint64_t planEnd = profile_ ? profile_->PublishClock() : 0U;
            if (profile_) profile_->CountBoundary<13, true>();
#endif
            comm::CountFrame(view_, dst, stateOffset_ + frame.frameOffset,
                reinterpret_cast<__ubuf__ uint8_t*>(records_) + frame.sourceOffset, frame.wireBytes, count
#ifdef DEEPEP_LL_REFERENCE_PROFILE
                , profile_
#endif
                );
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            if (profile_) profile_->PublishAdd<13>(planEnd-planBegin);
#endif
            expert += count;
            continue;
        }
        comm::Records(view_, dst, stateOffset_ + offset,
            reinterpret_cast<__ubuf__ uint8_t*>(records_) + (expert - start) * UB_ALIGN,
            count, epWorldSize_ * kEpServerDedupCountGmStride, true,
            reinterpret_cast<__ubuf__ uint8_t*>(work_));
        expert += count;
    }
}

// 摘要：将本 rank 的计数记录分给接收核，等待本片计数到齐并计算局部前缀与总数。
template <uint32_t kNumMaxTokensPerRank>
__aicore__ inline void ServerDedupCounts<kNumMaxTokensPerRank>::RecvCnt()
{
    // 接收端控制区共有 localExperts×worldSize（等于 experts）条记录，
    // 按“本地专家 → 来源 rank”排列；每个接收核负责一段连续记录。
    // Each count owns a 512B GM slot, so contiguous ownership can split at any
    // record. Keep all nonempty receivers active and their loads within one record.
    activeCumSumCores_ = rscvStatusNum_ < aivUsedCumSum_ ? rscvStatusNum_ : aivUsedCumSum_;
    scratchEnd_ = ReservedBytes();
    scalar_ = Scratch(3U * UB_ALIGN);
    const uint32_t receiver = aivId_ - aivUsedAllToAll_;
    // Inactive receivers still join prefix readiness in CompleteCumsum.
    if (receiver >= activeCumSumCores_) return;
    SplitToCore(rscvStatusNum_, aivUsedCumSum_, receiver, startStatusIndex_, endStatusIndex_, recStatusNumPerCore_);
    snapshot_ = Scratch(Align(recStatusNumPerCore_ * UB_ALIGN + SIZE_ALIGN_256, SIZE_ALIGN_256));
    dense_ = Scratch(rscvStatusNum_ * sizeof(int32_t));
    // Vector publication/cleanup touches complete groups of eight 32B records.
    // Padding is UB-only; matrix strides and participation use actual core counts.
    const uint32_t sumBytes = Align(aivUsedCumSum_ * UB_ALIGN, SIZE_ALIGN_256);
    sumSlot_ = Scratch(sumBytes);
    sumSnapshot_ = Scratch(sumBytes);
    sumDense_ = Scratch(aivUsedCumSum_ * sizeof(float));
    // 等待本片 ready 到齐后提取 count，并计算局部累计值、发布本片总数。
    // 本阶段不会等待 token payload；即使分到空片段也要发布总数 0，参与后续跨核汇总。
    WaitDispatch();
}

// 摘要：等待所有前缀片段就绪，仅首个接收核加载前缀并输出各专家计数。
template <uint32_t kNumMaxTokensPerRank>
__aicore__ inline void ServerDedupCounts<kNumMaxTokensPerRank>::CompleteCumsum()
{
    // Convert the physical AIV index to its receive-group index.
    // All receivers join readiness; only receiver0 needs a count-stage prefix.
    const uint32_t receiver = aivId_ - aivUsedAllToAll_;
    ready_ = Scratch(Align(activeCumSumCores_ * UB_ALIGN + SIZE_ALIGN_256, SIZE_ALIGN_256));
    clean_ = Scratch(activeCumSumCores_ * UB_ALIGN);
    // 各核先写前缀再发完成标记；全部标记到齐后，读取才不会混入未更新片段。
    if (profile_) profile_->Mark<28>();  // w28: prefix_ready_wait_begin
    WaitCumSumFlag();
    if (profile_) profile_->Mark<29>();  // w29: prefix_ready_wait_clear_issue_end
    // 仅接收组的第 0 核写公开专家计数，避免多核重复写同一输出。
    // 每个专家最后一个来源 rank 的前缀即该专家结束位置；type=0 输出累计数，
    // 其他已验证类型输出与上一专家累计数之差，即该专家自己的接收条数。
    if (receiver == 0U) {
        // Receiver::Process resets the count UB and loads its own prefix.
        // Other cores have no consumer for an additional count-stage copy.
auto prefix = Scratch(rscvStatusNum_ * sizeof(int32_t));
        Load(prefix, recvCntWorkspaceGM_, rscvStatusNum_ * sizeof(int32_t));
        Sync<PIPE_MTE2, PIPE_S>();
        auto expertCounts = reinterpret_cast<__ubuf__ int64_t*>(Scratch(moeExpertNumPerRank_ * sizeof(int64_t)));
        int64_t previous = 0;
        for (uint32_t expert = 0; expert < moeExpertNumPerRank_; ++expert) {
            const int64_t cumulative = prefix[(expert + 1U) * epWorldSize_ - 1U];
            expertCounts[expert] = expertTokenNumsType_ == 0U ? cumulative : cumulative - previous;
            previous = cumulative;
        }
        Sync<PIPE_S, PIPE_MTE3>();
        Store(expertTokenNumsGM_, expertCounts, moeExpertNumPerRank_ * sizeof(int64_t));
    }
    // No listening table or payload receive/copy is constructed in this increment.
}

// 摘要：清除本核已收取记录的 ready 标记，保留计数字段，为后续窗口复用做准备。
template <uint32_t kNumMaxTokensPerRank>
__aicore__ inline void ServerDedupCounts<kNumMaxTokensPerRank>::WaitDispatchClearStatus()
{
    if (packedFrames_) {
        // Called only after GetCumSum has observed every receiver's snapshot.
        // Exactly the owner of a frame's first logical record clears its wire.
        auto zero = snapshot_;
        Fill(zero, 0U, kLlCountWireBlockBytes / sizeof(float));
        Sync<PIPE_V, PIPE_MTE3>();
        for (uint32_t logical = startStatusIndex_; logical < endStatusIndex_; ++logical) {
            const uint32_t expert = logical / epWorldSize_, source = logical % epWorldSize_;
            const auto frame = CountFrame(rscvStatusNum_, epWorldSize_,
                useAllCoreCount_ ? coreNum_ : aivUsedCumSum_, epRankId_, source, expert, directRanks_);
            // Remote frames retain dense-record cleanup. Local copies include
            // neighbouring source records, so reclaim their entire wire too:
            // a later shape may put a remote commit at any old record boundary.
            if (frame.flagWords == 1U)
                Store(stateGM_ + frame.recordOffset, zero, 32U);
            if (expert == frame.firstExpert) {
                if (frame.flagWords == 1U)
                    Store(stateGM_ + frame.commitOffset, zero, 32U);
                else
                    for (uint32_t block = 0; block < frame.wireBytes; block += kLlCountWireBlockBytes)
                        Store(stateGM_ + frame.frameOffset + block, zero, kLlCountWireBlockBytes);
            }
        }
        return;
    }
    Sync<PIPE_S, PIPE_V>();
    RecordReady(snapshot_, 0U, Align(recStatusNumPerCore_, 8U));
    Sync<PIPE_V, PIPE_MTE3>();
    Store(stateGM_ + startStatusIndex_ * kEpServerDedupCountGmStride, snapshot_,
          32U, recStatusNumPerCore_, kEpServerDedupCountGmStride - 32U);
    // The snapshot is no longer modified after this issue. Its DMA can overlap
    // cumsum; the caller retires all count-stage writes before receiver Reset.
}

// 摘要：提取计数字段，计算本片累计前缀，并向其他接收核发布本片总数。
template <uint32_t kNumMaxTokensPerRank>
__aicore__ inline void ServerDedupCounts<kNumMaxTokensPerRank>::GatherSumRecvCnt()
{
    Sync<PIPE_S, PIPE_V>();
    GatherWords(dense_, snapshot_, recStatusNumPerCore_, 1);
    Sync<PIPE_V, PIPE_S>();

    // Compute the relative prefix while outgoing cnt may still be in flight.
    // Gather preserves integer count bits; no floating-point count arithmetic.
    uint32_t cumulative = 0;
    for (uint32_t i = 0; i < recStatusNumPerCore_; ++i) {
        cumulative += dense_[i];
        dense_[i] = cumulative;
    }
    // The final prefix is already the subtotal; keep its original wire bits.
    PublishRecvCntSum(cumulative);
}

// 摘要：将本核片段总数和就绪标记写入本 rank 的紧凑记分板；空片段也必须发布。
template <uint32_t kNumMaxTokensPerRank>
__aicore__ inline void ServerDedupCounts<kNumMaxTokensPerRank>::PublishRecvCntSum(uint32_t sumOfRecvCnt)
{
    uint32_t newAivId = aivId_ - aivUsedAllToAll_;

    // Compact scoreboard: one writer-owned 32B slot per receiver core, so
    // consumers read a single contiguous burst instead of a transposed row.
    for (uint32_t i = 0; i < UB_ALIGN_DATA_COUNT; ++i) sumSlot_[i] = 0U;
    sumSlot_[0] = sumOfRecvCnt;
    sumSlot_[1] = 0x3f800000U;
    Sync<PIPE_S, PIPE_MTE3>();
    Store(stateGM_ + layout_.cumsumOffset + newAivId * UB_ALIGN, sumSlot_, UB_ALIGN);
    // sumSlot_ stays immutable until CalRecvAndSetFlag reuses it. Polling
    // incoming sums must not wait here for this core's outgoing cnt.
}

// 摘要：等待所有接收核总数就绪，求本核之前各片段之和；槽位回收在后续屏障后执行。
template <uint32_t kNumMaxTokensPerRank>
__aicore__ inline uint32_t ServerDedupCounts<kNumMaxTokensPerRank>::GetCumSum(uint32_t newAivId)
{
    uint32_t offset = 0;
    // Subtotals are contiguous 32B slots, so one burst covers every active core.
    Sync<PIPE_S, PIPE_V>();

    while (true) {
        Load(sumSnapshot_, stateGM_ + layout_.cumsumOffset, activeCumSumCores_ * UB_ALIGN);
        Sync<PIPE_MTE2, PIPE_V>();
        GatherWords(sumDense_, sumSnapshot_, activeCumSumCores_, 1);
        asc_sync_pipe(PIPE_V);
        Sum<true>(scalar_ + 8U, sumDense_, activeCumSumCores_);
        Sync<PIPE_V, PIPE_S>();
        if (*reinterpret_cast<__ubuf__ float*>(scalar_ + 8U) == float(activeCumSumCores_)) break;
    }
    if (newAivId != 0U) {
        Sync<PIPE_S, PIPE_V>();
        GatherWords(sumDense_, sumSnapshot_, newAivId, 0);
        asc_sync_pipe(PIPE_V);
        Sum<false>(scalar_ + 8U, sumDense_, newAivId);
        Sync<PIPE_V, PIPE_S>();
        offset = scalar_[8U];
    }

    // The subtotal slots are reclaimed after WaitCumSumFlag, when every receiver
    // has already consumed them; no per-consumer column clear remains here.
    return offset;
}

// 摘要：轮询本片所有来源计数的 ready 标记；到齐后提取计数、发布片段总数并清除 ready。
template <uint32_t kNumMaxTokensPerRank>
__aicore__ inline void ServerDedupCounts<kNumMaxTokensPerRank>::WaitDispatch()
{
    float compareTarget = static_cast<float>(1.0) * recStatusNumPerCore_;
    float sumOfFlag = static_cast<float>(-1.0);
    LlCountReadPlan readPlan;
    if (packedFrames_ && !BuildCountReadPlan(rscvStatusNum_, epWorldSize_,
            useAllCoreCount_ ? coreNum_ : aivUsedCumSum_, epRankId_,
            startStatusIndex_, recStatusNumPerCore_, readPlan, directRanks_)) { trap(); return; }
    if (packedFrames_) {
        compareTarget = static_cast<float>(readPlan.expectedFlagSum);
    }

    Sync<PIPE_S, PIPE_V>();
    if (profile_) profile_->Mark<24>();  // w24: count_ready_poll_begin
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    uint32_t firstReadySeen = 0U, firstIterDone = 0U;
#endif
    while (sumOfFlag != compareTarget) {
        if (packedFrames_) {
            ReadCountFrames(snapshot_, stateGM_, readPlan, false);
        } else Load(snapshot_, stateGM_ + startStatusIndex_ * kEpServerDedupCountGmStride,
                    UB_ALIGN, recStatusNumPerCore_, kEpServerDedupCountGmStride - UB_ALIGN);
        Sync<PIPE_MTE2, PIPE_V>();
        if (packedFrames_) {
            Sum<true>(scalar_ + 8U, snapshot_, recStatusNumPerCore_ * UB_ALIGN_DATA_COUNT);
        } else {
            GatherWords(dense_, snapshot_, recStatusNumPerCore_, 0);
            asc_sync_pipe(PIPE_V);
            Sum<true>(scalar_ + 8U, dense_, recStatusNumPerCore_);
        }
        Sync<PIPE_V, PIPE_S>();
        sumOfFlag = *reinterpret_cast<__ubuf__ float*>(scalar_ + 8U);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (profile_) {
            // Partial arrival plus the first full poll iteration separate "count
            // data is late on the fabric" from "one scan is expensive".
            if (!firstReadySeen && sumOfFlag != 0.0f) { profile_->Mark<46>(); firstReadySeen = 1U; }  // w46: recv_count_first_ready
            if (!firstIterDone) { profile_->Mark<47>(); firstIterDone = 1U; }  // w47: recv_count_poll_first_iter_end
        }
#endif
    }
    if (profile_) profile_->Mark<25>();  // w25: count_ready_poll_end_after_V_S

    if (packedFrames_) {
        // Commit acquired: load the actual dense records, preserving the old
        // logical expert/source ordering for prefix sums and Receiver metadata.
        ReadCountFrames(snapshot_, stateGM_, readPlan, true);
        Sync<PIPE_MTE2, PIPE_V>();
    }
    GatherSumRecvCnt();
    if (!packedFrames_) WaitDispatchClearStatus();
}

// 摘要：为本片局部前缀加上跨核偏移，写出 sendCountsOut 与共享前缀，再发布完成标记。
template <uint32_t kNumMaxTokensPerRank>
__aicore__ inline void ServerDedupCounts<kNumMaxTokensPerRank>::CalRecvAndSetFlag()
{
    if (recStatusNumPerCore_ == 0U) return;
    uint32_t newAivId = aivId_ - aivUsedAllToAll_;
    // 当前 UB 中是片段内前缀；先等待所有片段总数，再累加编号更小接收核的总数。
    // 这样得到全 rank 接收表中的基址，各核可独立修正自己负责的片段。
    if (profile_) profile_->Mark<26>();  // w26: cumsum_peer_join_begin
    const uint32_t offset = GetCumSum(newAivId);
    if (profile_) profile_->Mark<27>();  // w27: cumsum_peer_join_clear_issue_end
    if (packedFrames_) WaitDispatchClearStatus();

    if (recStatusNumPerCore_ != 0U) {
        for (uint32_t i = 0; i < recStatusNumPerCore_; ++i) {
            dense_[i] += offset;
        }
        Sync<PIPE_S, PIPE_MTE3>();
        // sendCountsOut 写的是累计前缀，而非单条记录的原始发送数量。
        Store(sendCountsOutGM_ + startStatusIndex_ * sizeof(int32_t), dense_,
              recStatusNumPerCore_ * sizeof(int32_t));

        // Producers own disjoint fragments of one read-only GM prefix. All
        // receivers acquire the existing ready row before reading this copy.
        Store(recvCntWorkspaceGM_ + startStatusIndex_ * sizeof(int32_t), dense_,
              recStatusNumPerCore_ * sizeof(int32_t));
    }

    // 发布前缀完成标记前，先等待旧片段总数发送不再读取 sumSlot_，
    // 再复用它构造标记；前缀写入与完成标记之间保留原 MTE3 顺序约束。
    // This is the first overwrite of the asynchronous local-sum publication.
    Sync<PIPE_MTE3, PIPE_V>();
    Fill(sumSlot_, 1U, Align(aivUsedCumSum_, 8U) * UB_ALIGN_DATA_COUNT);
    Sync<PIPE_V, PIPE_MTE3>();
    asc_sync_pipe(PIPE_MTE3);

    Store(stateGM_ + layout_.cumsumFlagOffset + newAivId * UB_ALIGN,
          sumSlot_, UB_ALIGN, aivUsedCumSum_, (aivUsedCumSum_ - 1U) * UB_ALIGN);
}

// 摘要：等待所有接收核的前缀写出标记到齐，并清除本核对应的标记行。
template <uint32_t kNumMaxTokensPerRank>
__aicore__ inline void ServerDedupCounts<kNumMaxTokensPerRank>::WaitCumSumFlag()
{
    const uint32_t targetFlag = activeCumSumCores_ * UB_ALIGN_DATA_COUNT;
    const uint32_t newAivId = aivId_ - aivUsedAllToAll_;
    uint32_t cumSumFlagOffset =
        layout_.cumsumFlagOffset + newAivId * aivUsedCumSum_ * UB_ALIGN;

    while (true) {
        Load(ready_, stateGM_ + cumSumFlagOffset, activeCumSumCores_ * UB_ALIGN);
        Sync<PIPE_MTE2, PIPE_V>();
        Sum<false>(scalar_, ready_, targetFlag);
        Sync<PIPE_V, PIPE_S>();
        if (scalar_[0] == targetFlag) break;
    }
    Fill(clean_, 0U, targetFlag);
    Sync<PIPE_S, PIPE_MTE3>();

    Sync<PIPE_V, PIPE_MTE3>();
    Store(stateGM_ + cumSumFlagOffset, clean_, activeCumSumCores_ * UB_ALIGN);

    // This barrier opens only after every receiver has completed GetCumSum, so
    // each publisher may reclaim its own subtotal slot. The next use of this
    // bank then starts from zeros without an extra epoch tag.
    if (newAivId < activeCumSumCores_) {
        Store(stateGM_ + layout_.cumsumOffset + newAivId * UB_ALIGN,
              clean_ + newAivId * UB_ALIGN_DATA_COUNT, UB_ALIGN);
    }
}

}  // namespace DispatchDedup

// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

namespace deep_ep {
template <uint32_t kNumMaxTokensPerRank, class XType, uint32_t Group = 16U>
class ServerDedupSender {
public:
    __aicore__ inline void Init(const TransportView& view, GM_ADDR x, GM_ADDR ids, GM_ADDR weights,
        GM_ADDR sourceMask, GM_ADDR sourceWeights, uint32_t num_tokens, uint32_t h, uint32_t k,
        uint32_t experts, EpServerDedupInputMode inputMode, bool e4m3, int64_t magic,
        GM_ADDR inputScales, bool enableDedup, CoreContext core)
    {
        view_ = view;
        core_ = core;
        prequantized_ = inputMode == EpServerDedupInputMode::PrequantizedFp8Packs;
        needQuantize_ = inputMode == EpServerDedupInputMode::Quantize16;
        e4m3_ = e4m3;
        inputRowBytes_ = prequantized_ ? h : h * sizeof(XType);
        sendCores_ = core.count / 2U;
        enableDedup_ = enableDedup;
        fullMeshFirstHop_ = Group == kNumRanksPerHost && view.is_rdma();
        num_tokens_ = num_tokens; h_ = h; k_ = k; experts_ = experts;
        localExperts_ = experts / view.num_ranks;
        const uint32_t index = core.index, base = kNumMaxTokensPerRank / sendCores_,
            remainder = kNumMaxTokensPerRank % sendCores_;
        count_ = base + (index < remainder ? 1U : 0U);
        start_ = index * base + (index < remainder ? index : remainder);
        work_count_ = num_tokens_ > start_ ?
            (num_tokens_ - start_ < count_ ? num_tokens_ - start_ : count_) : 0U;
        data_ = DispatchServerDedupDataLayout(kNumMaxTokensPerRank, h, k, experts, inputMode);
        data_.dispatchOffset += DispatchServerDedupWindowOffset(magic);
        x_ = x; inputScales_ = inputScales;
        ids_ = reinterpret_cast<__gm__ int32_t*>(ids);
        weights_ = reinterpret_cast<__gm__ float*>(weights);
        maskOut_ = reinterpret_cast<__gm__ uint32_t*>(sourceMask);
        weightOut_ = reinterpret_cast<__gm__ float*>(sourceWeights);
    }

    __aicore__ inline void Process(uint32_t reservedBytes,
        profile::DeviceProfile* profile = nullptr, comm::CountRetirement* retirement = nullptr)
    {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        packetProfile_ = profile;
        rdmaBatch_.SetProfile(profile);
#endif
        if (profile && profile->Enabled()) {
            const uint32_t routes = num_tokens_ * k_;
            profile->Value<43>(enableDedup_ ? count_ * k_ :  // w43: send_route_count
                routes / sendCores_ + (core_.index < routes % sendCores_ ? 1U : 0U));
        }
        if (!LlSenderHasWork(enableDedup_, work_count_, num_tokens_ * k_, core_.index)) return;
        // The count image at UB0 remains owned by MTE3 until count retirement.
        ub::UbLayout layout(reservedBytes);
        if (view_.is_rdma()) rdmaBatch_.SetCountRetirement(retirement);
        inputBytes_ = needQuantize_ ? uint32_t(data_.quantCount * sizeof(XType)) :
                                    uint32_t(data_.packedPayloadBytes);
        input_[0] = layout.Allocate(inputBytes_);
        input_[1] = layout.Allocate(inputBytes_);
        if (needQuantize_) {
            ub::FillBytes(input_[0], 0U, 2U * inputBytes_);
            ub::Sync<PIPE_V, PIPE_MTE2>();
        }
        const uint32_t routeBytes = uint32_t(DispatchDedupAlign(uint64_t(kNumMaxTokensPerRank) * k_ * 4U, 256U));
        const uint32_t qBytes = uint32_t(DispatchDedupAlign(uint64_t(kNumMaxTokensPerRank) * k_ * 4U, 32U));
        const uint32_t scaleCount = uint32_t(data_.quantCount / 32U);
        const uint32_t quantWorkBytes = uint32_t(DispatchDedupAlign(
            DispatchDedupAlign(scaleCount, 32U) * 4U + uint64_t(scaleCount) * 2U, 32U));
        const uint32_t scratchBytes = needQuantize_ && quantWorkBytes > qBytes ? quantWorkBytes : qBytes;
        routeIds_ = layout.Allocate<int32_t>(routeBytes);
        qDst_ = layout.Allocate<uint32_t>(scratchBytes);
        // Preserve the admitted UB footprint and hence lane count.
        layout.Allocate(scratchBytes);
        layout.Allocate(qBytes);
        if (needQuantize_) quant_ = layout.Allocate(uint32_t(data_.packedPayloadBytes));
        routeWeights_ = layout.Allocate<float>(count_ * k_ * 4U);
        sourceWeights_ = layout.Allocate<float>(count_ * k_ * 4U);
        sourceMasks_ = layout.Allocate<uint32_t>(count_ * 4U);
        directTasks_ = layout.Allocate<uint32_t>(count_ * k_ * 16U);
        const uint32_t serverBytes = ub::Align(view_.num_ranks / Group * 4U);
        // Shared phase-wise: Classify uses [server] as per-token scratch, then
        // the direct-task list reuses it as head links. The two phases never
        // overlap, so one array keeps the admitted UB footprint.
        serverState_ = layout.Allocate<uint32_t>(serverBytes < 64U ? 64U : serverBytes);
        const uint32_t remoteServers = view_.num_ranks / Group - 1U;
        const uint32_t maxGroups = remoteServers < k_ / 2U ? remoteServers : k_ / 2U;
        if (maxGroups) firstHopMasks_ = layout.Allocate<uint32_t>(maxGroups * 4U);
        if (view_.is_rdma()) {
            const uint32_t tokenStride = uint32_t(data_.tokenStride);
            if (!tokenStride || tokenStride % 512U || tokenStride > rdma::kCoreBytes) { trap(); return; }
            const uint32_t stageSlots = rdma::kCoreBytes / tokenStride;
            rdmaWorkspace_ = layout.Allocate<uint8_t>(
                (stageSlots + 1U) * comm::DispatchRdmaBatch::kWorkspaceBytes);
            rdmaBatch_.SetWorkspace(rdmaWorkspace_);
            rdmaBatch_.Init(view_, tokenStride);
        }
        const uint32_t available = (ub::kUbBytes - layout.UsedBytes()) / uint32_t(data_.tokenStride);
        lanes_ = available < 8U ? available : 8U;
        if (!lanes_) { trap(); return; }
        out_ = layout.Allocate(lanes_ * uint32_t(data_.tokenStride));
        hasScratch_ = true;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (packetProfile_) packetProfile_->SendLanes(lanes_);
#endif
        ub::Load(routeIds_, ids_, num_tokens_ * k_ * 4U);
        if (work_count_) ub::Load(routeWeights_, weights_ + uint64_t(start_) * k_, work_count_ * k_ * 4U);
        ub::Sync<PIPE_MTE2, PIPE_S>();
        for (uint32_t lane = 0; lane < lanes_; ++lane) {
            ub::FillPacketFlags(out_ + lane * data_.tokenStride, uint32_t(data_.blockCount));
            asc_sync_notify(PIPE_MTE3, PIPE_V, event_t(lane));
        }
        asc_sync_pipe(PIPE_V);
        if (profile) profile->Mark<38>();  // w38: send_buffers_routes_ready
        if (enableDedup_) SendWithDedup();
        else SendWithoutDedup();
        rdmaBatch_.SubmitPending();
        if (profile) profile->Mark<39>();  // w39: send_packets_issue_end
        SenderInputOperations operations{*this};
        inputReuse_.Finish(operations);
        rdmaBatch_.Flush();
        for (uint32_t lane = 0; lane < lanes_; ++lane) asc_sync_wait(PIPE_MTE3, PIPE_V, event_t(lane));
        ub::Sync<PIPE_S, PIPE_MTE3>();
        if (enableDedup_) ub::Store(maskOut_ + start_, sourceMasks_, work_count_ * 4U);
        if (work_count_) ub::Store(weightOut_ + uint64_t(start_) * k_,
            enableDedup_ ? sourceWeights_ : routeWeights_, work_count_ * k_ * 4U);
        ub::Sync<PIPE_MTE3, PIPE_S>();
        if (profile) profile->Mark<40>();  // w40: send_metadata_end_after_MTE3_S
    }
    __aicore__ inline bool HasFinalScratch() const { return hasScratch_; }
    __aicore__ inline __ubuf__ uint8_t* FinalScratch() { return out_; }

private:
    __aicore__ inline uint32_t FirstHop(uint32_t base, uint32_t mask)
    {
        LlFullMeshFirstHop mesh;
        for (uint32_t k = 0; k < k_; ++k) {
            if (!(mask & (1U << k))) continue;
            if (!fullMeshFirstHop_) return k;
            mesh.Add(uint32_t(routeIds_[base + k]) / localExperts_, k);
        }
        return mesh.FirstK();
    }
    __aicore__ inline void SendWithDedup()
    {
        const uint32_t rankInGroup = uint32_t(view_.rank) % Group, sendCore = core_.index;
        for (uint32_t token = 0; token < work_count_; ++token) {
            uint32_t directMask = 0U, firstHopCount = 0U;
            Classify(token, directMask, firstHopCount);
            uint32_t sourceMask = directMask;
            for (uint32_t group = 0; group < firstHopCount; ++group) {
                const uint32_t scheduled = LlFirstHopOrder(firstHopCount, rankInGroup, sendCore, start_ + token, group);
                const uint32_t mask = firstHopMasks_[scheduled];
                const uint32_t firstK = FirstHop((start_ + token) * k_, mask);
                const uint32_t index = (start_ + token) * k_ + firstK, expert = uint32_t(routeIds_[index]);
                SendPacket(start_ + token, firstK, expert, CalTokenSendExpertCnt(expert, index), mask);
                sourceMask |= 1U << firstK;
                sourceWeights_[token * k_ + firstK] = 1.0f;
                // Relay members travel inside this packet. Publish their real
                // weights so the destination's source_weights row has no stale
                // UB left in mask-active slots.
                const uint32_t members = LlRelayMemberMask(mask, firstK);
                for (uint32_t k = 0; members && k < k_; ++k)
                    if (members & (1U << k)) sourceWeights_[token * k_ + k] = routeWeights_[token * k_ + k];
            }
            for (uint32_t k = 0; k < k_; ++k) if (directMask & (1U << k)) {
                directTasks_[directCount_ * 4U] = token;
                directTasks_[directCount_ * 4U + 1U] = k;
                directTasks_[directCount_ * 4U + 2U] = uint32_t(routeIds_[(start_ + token) * k_ + k]);
                ++directCount_;
            }
            sourceMasks_[token] = sourceMask;
        }
        const uint32_t serverNum = view_.num_ranks / Group, expertsPerServer = localExperts_ * Group;
        const uint32_t ownServer = view_.rank / Group;
        for (uint32_t server = 0; server < serverNum; ++server) serverState_[server] = UINT32_MAX;
        for (uint32_t next = directCount_; next; --next) {
            const uint32_t task = next - 1U, server = directTasks_[task * 4U + 2U] / expertsPerServer;
            directTasks_[task * 4U + 3U] = serverState_[server];
            serverState_[server] = task;
        }
        for (uint32_t step = 0; step < serverNum; ++step) {
            const uint32_t server = LlDirectGroupOrder(
                serverNum, ownServer, rankInGroup, sendCore, step, fullMeshFirstHop_);
            for (uint32_t task = serverState_[server]; task != UINT32_MAX; task = directTasks_[task * 4U + 3U]) {
                const uint32_t expert = directTasks_[task * 4U + 2U], token = directTasks_[task * 4U];
                const uint32_t k = directTasks_[task * 4U + 1U];
                SendPacket(start_ + token, k, expert, CalTokenSendExpertCnt(expert, (start_ + token) * k_ + k), 0U);
                sourceWeights_[token * k_ + k] = routeWeights_[token * k_ + k];
            }
        }
    }
    __aicore__ inline void SendWithoutDedup()
    {
        const uint32_t total = num_tokens_ * k_, core = core_.index, base = total / sendCores_;
        const uint32_t remainder = total % sendCores_, begin = core * base + (core < remainder ? core : remainder);
        const uint32_t count = base + (core < remainder ? 1U : 0U), serverNum = view_.num_ranks / Group;
        if (serverNum < 16U) {
            for (uint32_t i = begin; i < begin + count; ++i) {
                const uint32_t expert = uint32_t(routeIds_[i]);
                SendPacket(i / k_, i % k_, expert, CalTokenSendExpertCnt(expert, i), 0U);
            }
            return;
        }
        const uint32_t expertsPerServer = localExperts_ * Group, ownServer = view_.rank / Group;
        for (uint32_t step = 0; step < serverNum; ++step) {
            const uint32_t server = EpServerOrder::ServerDedupDispatchDirectOrder(serverNum, ownServer, step);
            const uint32_t firstExpert = server * expertsPerServer;
            for (uint32_t i = begin; i < begin + count; ++i) {
                const uint32_t expert = uint32_t(routeIds_[i]);
                if (expert < firstExpert || expert >= firstExpert + expertsPerServer) continue;
                SendPacket(i / k_, i % k_, expert, CalTokenSendExpertCnt(expert, i), 0U);
            }
        }
    }
    __aicore__ inline void Classify(uint32_t token, uint32_t& directMask, uint32_t& groups)
    {
        uint32_t seen = 0U, grouped = 0U;
        for (uint32_t k = 0; k < k_; ++k) {
            const uint32_t expert = uint32_t(routeIds_[(start_ + token) * k_ + k]);
            const uint32_t server = expert / localExperts_ / Group, bit = 1U << k, serverBit = 1U << server;
            if (server == uint32_t(view_.rank) / Group) directMask |= bit;
            else if (!(seen & serverBit)) {
                seen |= serverBit; serverState_[server] = k; directMask |= bit;
            } else if (!(grouped & serverBit)) {
                const uint32_t firstBit = 1U << serverState_[server];
                directMask &= ~firstBit;
                firstHopMasks_[groups] = firstBit | bit;
                serverState_[server] = groups++; grouped |= serverBit;
            } else {
                const uint32_t index = serverState_[server];
                firstHopMasks_[index] |= bit;
            }
        }
    }
    __aicore__ inline uint32_t CalTokenSendExpertCnt(uint32_t expert, uint32_t index)
    {
        if (index < k_) return 0U;
        ub::RoutePrefixCount(qDst_, routeIds_, index, int32_t(expert));
        ub::Sync<PIPE_V, PIPE_S>();
        return qDst_[0];
    }
    template <bool E4M3>
    __aicore__ inline void QuantMxfp8(__ubuf__ uint8_t* output, __ubuf__ XType* input)
    {
        const uint32_t scales = uint32_t(data_.quantCount / 32U);
        auto* maxExp = reinterpret_cast<__ubuf__ uint16_t*>(qDst_);
        auto* halfScale = reinterpret_cast<__ubuf__ uint16_t*>(qDst_ + DispatchDedupAlign(scales, 32U));
        mxfp8_quant::ComputeMaxExp(input, maxExp, uint32_t(data_.quantCount));
        mxfp8_quant::ComputeScale<E4M3>(maxExp,
            reinterpret_cast<__ubuf__ uint16_t*>(output + data_.quantCount), halfScale, scales);
        mxfp8_quant::ComputeFp8Data<XType, E4M3>(
            input, halfScale, reinterpret_cast<__ubuf__ int8_t*>(output), uint32_t(data_.quantCount));
    }
    __aicore__ inline void FillTail(__ubuf__ uint8_t* payload, uint32_t token, uint32_t firstK, uint32_t mask)
    {
        auto* header = reinterpret_cast<__ubuf__ uint32_t*>(payload + data_.headerPayloadOffset);
        auto* index = reinterpret_cast<__ubuf__ uint32_t*>(payload + data_.readIndexPayloadOffset);
        header[0] = uint32_t(view_.rank); header[1] = token; header[2] = firstK;
        header[3] = uint32_t(mask ? EpServerDedupTokenType::FirstHop : EpServerDedupTokenType::Direct);
        if (!enableDedup_) { index[0] = 0U; return; }
        auto* weights = reinterpret_cast<__ubuf__ float*>(index);
        token -= start_;
        weights[1] = routeWeights_[token * k_ + firstK];
        uint32_t entries = 0U;
        const uint32_t members = LlRelayMemberMask(mask, firstK);
        for (uint32_t k = 0; members && k < k_; ++k) if (members & (1U << k)) {
            const uint32_t routeIndex = (start_ + token) * k_ + k, expert = uint32_t(routeIds_[routeIndex]);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            const bool traceQ = packetProfile_ && packetProfile_->SecondRemote();
            const uint64_t qBegin = traceQ ? packetProfile_->Clock() : 0U;
#endif
            const uint32_t q = CalTokenSendExpertCnt(expert, routeIndex);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            if (traceQ) packetProfile_->MemberQ(packetProfile_->Clock() - qBegin);
#endif
            ++entries;
            index[entries * 2U] = expert | (q << 16U);
            weights[entries * 2U + 1U] = routeWeights_[token * k_ + k];
        }
        if (prequantized_) {
            for (uint32_t stale = entries + 1U; stale <= cachedTailEntries_; ++stale) {
                index[stale * 2U] = 0U; index[stale * 2U + 1U] = 0U;
            }
            cachedTailEntries_ = entries;
        }
        index[0] = entries;
    }
    struct SenderInputOperations {
        ServerDedupSender& sender;
        __aicore__ inline void LoadInput(uint32_t token) { sender.LoadTokenInput(token); }
        // The cache is read by V packing. Retire it before either slot or quant_ is overwritten.
        __aicore__ inline void ReleaseInput() { ub::Sync<PIPE_V, PIPE_MTE2>(); }
        __aicore__ inline void WaitPackedRead() { ub::Sync<PIPE_V, PIPE_S>(); }
    };
    __aicore__ inline void LoadTokenInput(uint32_t token)
    {
        auto* input = input_[inputSequence_++ % 2U];
        if (prequantized_) {
            const auto padding = LlPlanPrequantPadding(h_, uint32_t(data_.quantCount),
                uint32_t(data_.scaleBytes), uint32_t(data_.packedPayloadBytes));
            ub::FillPrequantPadding(input, padding);
            ub::Sync<PIPE_V, PIPE_MTE2>();
            cachedTailEntries_ = 0U;
        }
        ub::Load(input, x_ + uint64_t(token) * inputRowBytes_, inputRowBytes_);
        if (prequantized_) ub::Load(input + data_.quantCount,
            inputScales_ + uint64_t(token) * data_.scaleBytes, uint32_t(data_.scaleBytes));
        if (needQuantize_) {
            ub::Sync<PIPE_MTE2, PIPE_V>();
            if (e4m3_) QuantMxfp8<true>(quant_, reinterpret_cast<__ubuf__ XType*>(input));
            else QuantMxfp8<false>(quant_, reinterpret_cast<__ubuf__ XType*>(input));
            ub::Sync<PIPE_V, PIPE_S>();
            cachedPayload_ = quant_;
        } else {
            ub::Sync<PIPE_MTE2, PIPE_S>();
            cachedPayload_ = input;
        }
    }
    __aicore__ inline void SendPacket(uint32_t token, uint32_t firstK, uint32_t expert, uint32_t q, uint32_t mask)
    {
        const uint32_t destination = expert / localExperts_;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (packetProfile_) packetProfile_->SendBegin(view_.remote_write(destination),
            destination / Group == uint32_t(view_.rank) / Group, token, firstK, expert, destination);
#endif
        SenderInputOperations operations{*this};
        inputReuse_.Prepare(token, operations);
        FillTail(cachedPayload_, token, firstK, mask);
        ub::Sync<PIPE_S, PIPE_V>();
        const uint32_t lane = packets_ % lanes_;
        asc_sync_wait(PIPE_MTE3, PIPE_V, event_t(lane));
        auto* out = out_ + lane * data_.tokenStride;
        ub::PackPayload(out, cachedPayload_, uint32_t(data_.blockCount));
        // Publish the layout marker through V too; avoid a simultaneous Scalar store.
        ub::SetPacketMarker(out, mask ? PacketReceiveLayout::kKeepFlags : PacketReceiveLayout::kCompact);
        ub::Sync<PIPE_V, PIPE_MTE3>(event_t(lane));
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        const bool tracePrevious = packetProfile_ && packetProfile_->SecondRemote() && rdmaBatch_.HasStaged();
        const uint64_t previousBegin = tracePrevious ? packetProfile_->Clock() : 0U;
#endif
        rdmaBatch_.SubmitPending();
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (tracePrevious) packetProfile_->PreviousSubmit(packetProfile_->Clock() - previousBegin);
#endif
        const uint64_t offset = DispatchServerDedupDispatchOffset(data_, view_.rank, destination,
            expert % localExperts_, q, view_.num_ranks, localExperts_, kNumMaxTokensPerRank);
#ifndef DEEPEP_LL_NO_COMM
        if (view_.remote_write(destination)) rdmaBatch_.Enqueue(destination, offset, out, lanes_ > 1U);
        else
#endif
            comm::Packet(view_, destination, offset, out, uint32_t(data_.tokenStride), 7U);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (packetProfile_) packetProfile_->GroupSubmitted();
#endif
        asc_sync_notify(PIPE_MTE3, PIPE_V, event_t(lane));
        ++packets_;
    }
    TransportView view_{};
    comm::DispatchRdmaBatch rdmaBatch_;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    profile::DeviceProfile* packetProfile_ = nullptr;
#endif
    LlSenderInputReuse inputReuse_;
    __ubuf__ uint8_t* input_[2]{};
    __ubuf__ uint8_t *cachedPayload_ = nullptr, *quant_ = nullptr, *out_ = nullptr;
    __ubuf__ uint8_t* rdmaWorkspace_ = nullptr;
    __ubuf__ int32_t* routeIds_ = nullptr;
    __ubuf__ uint32_t *qDst_ = nullptr, *sourceMasks_ = nullptr, *directTasks_ = nullptr;
    __ubuf__ uint32_t *serverState_ = nullptr, *firstHopMasks_ = nullptr;
    __ubuf__ float *routeWeights_ = nullptr, *sourceWeights_ = nullptr;
    __gm__ uint8_t *x_ = nullptr, *inputScales_ = nullptr;
    __gm__ int32_t* ids_ = nullptr;
    __gm__ float *weights_ = nullptr, *weightOut_ = nullptr;
    __gm__ uint32_t* maskOut_ = nullptr;
    uint32_t inputSequence_ = 0U, inputBytes_ = 0U, cachedTailEntries_ = 0U;
    bool hasScratch_ = false, enableDedup_ = false, fullMeshFirstHop_ = false;
    bool prequantized_ = false, needQuantize_ = false, e4m3_ = true;
    CoreContext core_{};
    uint32_t sendCores_ = 0U, num_tokens_ = 0U, work_count_ = 0U, h_ = 0U, k_ = 0U, experts_ = 0U,
        localExperts_ = 0U;
    uint32_t start_ = 0U, count_ = 0U, inputRowBytes_ = 0U, lanes_ = 0U, packets_ = 0U, directCount_ = 0U;
    EpServerDedupDataLayout data_{};
};
}  // namespace DispatchDedup

// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

namespace deep_ep {
template <uint32_t kNumMaxTokensPerRank>
class ServerDedupReceiver {
public:
    __aicore__ inline void Init(const TransportView& view, GM_ADDR expandX, GM_ADDR dynamicScales,
        GM_ADDR tokenType, GM_ADDR destinationIndex, GM_ADDR relayReadIndex,
        uint32_t h, uint32_t k, uint32_t experts, EpServerDedupInputMode inputMode,
        int64_t magic, bool enableDedup, CoreContext core)
    {
        view_ = view; wireFp8_ = inputMode != EpServerDedupInputMode::Plain16;
        coreNum_ = core.count; recvCoreNum_ = coreNum_ / 2U;
        enableDedup_ = enableDedup; k_ = k; experts_ = experts;
        localExperts_ = experts / uint32_t(view_.num_ranks);
        receiver_ = core.index - recvCoreNum_;
        hiddenBytes_ = wireFp8_ ? h : h * 2U;
        data_ = DispatchServerDedupDataLayout(kNumMaxTokensPerRank, h, k, experts, inputMode);
        windowOffset_ = DispatchServerDedupWindowOffset(magic);
        data_.dispatchOffset += windowOffset_;
        expandX_ = expandX; dynamicScales_ = dynamicScales;
        tokenType_ = reinterpret_cast<__gm__ uint32_t*>(tokenType);
        destinationIndex_ = reinterpret_cast<__gm__ uint32_t*>(destinationIndex);
        relayReadIndex_ = reinterpret_cast<__gm__ uint32_t*>(relayReadIndex);
    }
    __aicore__ inline void Process(profile::DeviceProfile* profile = nullptr)
    {
        // Caller has drained all count UB users. Reuse physical UB from zero.
        ub::UbLayout layout;
        const uint32_t prefixBytes = ub::Align(experts_ * 4U);
        prefix_ = layout.Allocate<int32_t>(prefixBytes);
        const auto counts = DispatchServerDedupCountLayout(experts_, coreNum_);
        ub::Load(prefix_, view_.local_ptr() + windowOffset_ +
            counts.cntOffset + counts.prefixOffset, prefixBytes);
        ub::Sync<PIPE_MTE2, PIPE_S>();
        if (profile) profile->Mark<30>();  // w30: receive_prefix_loaded_after_MTE2_S
        tasks_.Build(prefix_, uint32_t(view_.rank), uint32_t(view_.num_ranks), experts_, kNumMaxTokensPerRank, receiver_,
                     layout, recvCoreNum_);
        if (profile) {
            profile->Mark<31>();  // w31: receive_tasks_built
            profile->Value<37>(tasks_.Count());  // w37: receive_task_count
        }
        if (!tasks_.Count()) {
            if (profile) {
                profile->Value<33>(0U);  // w33: receive_packet_poll_accum_cycles
                profile->Value<34>(0U);  // w34: receive_load_consume_submit_accum_cycles
                profile->Value<41>(0U);  // w41: receive_probe_batch_count
                profile->Value<48>(0U);  // w48: receive_first_hop_tokens
                profile->Value<49>(0U);  // w49: receive_relay_members
            }
            return;
        }
        const uint32_t blocks = uint32_t(data_.blockCount);
        const uint32_t flagsBytes = kProbeBatch * blocks * 32U + 256U;
        const uint32_t workBytes = kProbeBatch * uint32_t(DispatchDedupAlign((blocks + 7U) / 8U, 32U)) * 4U + 256U;
        flags_ = layout.Allocate<float>(flagsBytes);
        sums_ = layout.Allocate<float>(kProbeBatch * 32U);
        // Preserve the old reduction workspace budget: changing APIs does not grow lane count.
        layout.Allocate(workBytes);
        ub::FillBytes(reinterpret_cast<__ubuf__ uint8_t*>(flags_), 0U, flagsBytes);
        ub::Sync<PIPE_V, PIPE_MTE2>();
        snapshot_ = layout.Allocate<uint32_t>(kProbeBatch * 4U);
        laneStride_ = uint32_t(data_.tokenStride) + 64U;
        const uint32_t fixed = layout.UsedBytes(), available = (ub::kUbBytes - fixed) / laneStride_;
        lanes_ = available < 8U ? available : 8U;
        if (lanes_ > tasks_.Count()) lanes_ = tasks_.Count();
        if (!lanes_) { trap(); return; }
        laneBytes_ = layout.Allocate(lanes_ * laneStride_);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (profile && profile->Enabled()) {
            traceEnabled_ = true;
            traceCapacity_ = (ub::kUbBytes - layout.UsedBytes()) / 32U;
            if (traceCapacity_ > profile::kPacketTraceCapacity)
                traceCapacity_ = profile::kPacketTraceCapacity;
            if (traceCapacity_ > tasks_.Count()) traceCapacity_ = tasks_.Count();
            if (traceCapacity_) trace_ = layout.Allocate<uint32_t>(traceCapacity_ * 32U);
        }
#endif
        for (uint32_t lane = 0; lane < lanes_; ++lane) asc_sync_notify(PIPE_MTE3, PIPE_MTE2, event_t(lane));
        uint32_t scan = 0U, issued = 0U, consumed = 0U;
        if (profile) profile->Mark<32>();  // w32: receive_buffers_ready
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        uint64_t pollCycles = 0U, consumeCycles = 0U, batches = 0U;
        const bool referenceEnabled = profile && profile->Enabled();
#endif
        while (tasks_.PendingCount()) {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            const uint64_t pollBegin = referenceEnabled ? profile->Clock() : 0U;
#endif
            ProbeResult probe;
            uint32_t count;
            do {
                if (scan >= tasks_.PendingCount()) scan = 0U;
                const uint32_t remaining = tasks_.PendingCount() - scan;
                count = remaining < kProbeBatch ? remaining : kProbeBatch;
                probe = Probe(scan, count);
                if (!probe.readyMask) scan += count;
            } while (!probe.readyMask);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            const uint64_t pollEnd = referenceEnabled ? profile->Clock() : 0U;
            if (referenceEnabled) {
                if (!batches) profile->Value<42>(pollEnd);  // w42: first_packet_ready
                pollCycles += pollEnd - pollBegin; ++batches; RecordReady(probe.readyMask, count);
            }
#endif
            uint32_t nextLoad = count;
            for (uint32_t i = count; i; --i) {
                const uint32_t index = i - 1U;
                if (!(probe.readyMask & (1U << index))) continue;
                while (issued - consumed < lanes_ && nextLoad) {
                    --nextLoad;
                    if (!(probe.readyMask & (1U << nextLoad))) continue;
                    IssueLoad(snapshot_[nextLoad], issued % lanes_, (probe.keepFlagsMask & (1U << nextLoad)) != 0U);
                    ++issued;
                }
                ConsumeLoaded(snapshot_[index], consumed % lanes_, (probe.keepFlagsMask & (1U << index)) != 0U);
                tasks_.RemovePending(scan + index);
                ++consumed;
            }
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            if (referenceEnabled) consumeCycles += profile->Clock() - pollEnd;
#endif
        }
        for (uint32_t lane = 0; lane < lanes_; ++lane) asc_sync_wait(PIPE_MTE3, PIPE_MTE2, event_t(lane));
        ClearFlags();
        if (profile) profile->Mark<35>();  // w35: receive_lane_wait_clear_issue_end
        asc_sync_pipe(PIPE_ALL);
        if (profile) profile->Mark<36>();  // w36: receive_final_PIPE_ALL_end
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if (referenceEnabled) {
            profile->Value<33>(pollCycles);  // w33: receive_packet_poll_accum_cycles
            profile->Value<34>(consumeCycles);  // w34: receive_load_consume_submit_accum_cycles
            profile->Value<41>(batches);  // w41: receive_probe_batch_count
            profile->Value<48>(firstHopTokens_);  // w48: receive_first_hop_tokens
            profile->Value<49>(relayMembers_);  // w49: receive_relay_members
            profile->Value<62>(relayDistinctRanks_);  // w62: receive_relay_distinct_ranks
        }
#endif
    }
    __aicore__ inline __ubuf__ uint8_t* FinalScratch() { return reinterpret_cast<__ubuf__ uint8_t*>(prefix_); }
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    __aicore__ inline void ExportPacketTrace(profile::DeviceProfile* profile)
    {
        if (!profile || !profile->Enabled()) return;
        profile->Detail(96U, traceCapacity_); profile->Detail(97U, traceCapacity_);
        profile->Detail(98U, tasks_.Count()); profile->Detail(99U, traceReady_);
        profile->Detail(100U, traceProcessed_); profile->Detail(101U, lanes_);
        profile->Detail(104U, tasks_.Begin()); profile->Detail(105U, kNumMaxTokensPerRank);
        profile->Detail(106U, localExperts_); profile->Detail(107U, tasks_.Count() > traceCapacity_ ? 1U : 0U);
        profile->Detail(102U, profile->Clock());
        if (traceCapacity_) {
            ub::Sync<PIPE_S, PIPE_MTE3>();
            ub::Store(profile->PacketLogAddress(), trace_, traceCapacity_ * 32U, 1U, 0U, 0U, true);
            ub::Sync<PIPE_MTE3, PIPE_S>();
        }
        profile->Detail(103U, profile->Clock());
    }
#endif
private:
    static constexpr uint32_t kProbeBatch = 16U;
    struct ProbeResult { uint32_t readyMask = 0U, keepFlagsMask = 0U; };
    __aicore__ inline GM_ADDR Slot(uint32_t j)
    {
        return view_.local_ptr() + data_.dispatchOffset + uint64_t(tasks_.Position(j)) * data_.tokenStride;
    }
    __aicore__ inline ProbeResult Probe(uint32_t start, uint32_t count)
    {
        const uint32_t blocks = uint32_t(data_.blockCount);
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t j = tasks_.PendingIndex(start + i);
            snapshot_[i] = j;
            ReadServerDedupFlags(Slot(j), blocks, flags_ + i * blocks * 8U, view_.is_rdma());
        }
        ub::Sync<PIPE_MTE2, PIPE_V>();
        ub::SumPacketFlags(sums_, flags_, count, blocks * 8U);
        ub::Sync<PIPE_V, PIPE_S>();
        ProbeResult result;
        for (uint32_t i = 0; i < count; ++i) {
            const auto state = PacketReceiveLayout::Decode(sums_[i * 8U], blocks);
            if (state != PacketReceiveLayout::State::Pending) result.readyMask |= 1U << i;
            if (state == PacketReceiveLayout::State::KeepFlags) result.keepFlagsMask |= 1U << i;
        }
        return result;
    }
    __aicore__ inline void IssueLoad(uint32_t j, uint32_t lane, bool keepFlags)
    {
        asc_sync_wait(PIPE_MTE3, PIPE_MTE2, event_t(lane));
        auto* wire = laneBytes_ + lane * laneStride_;
        if (keepFlags) ub::Load(wire, Slot(j), uint32_t(data_.tokenStride), 1U, 0U, 0U, view_.is_rdma());
        else ub::Load(wire, Slot(j), 480U, uint32_t(data_.blockCount), 32U, 0U, view_.is_rdma());
        asc_sync_notify(PIPE_MTE2, PIPE_MTE3, event_t(lane));
        asc_sync_notify(PIPE_MTE2, PIPE_S, event_t(lane));
    }
    __aicore__ inline void ConsumeLoaded(uint32_t j, uint32_t lane, bool keepFlags)
    {
        asc_sync_wait(PIPE_MTE2, PIPE_S, event_t(lane));
        auto* wire = laneBytes_ + lane * laneStride_;
        const uint32_t row = tasks_.Begin() + j * tasks_.Stride();
        const uint32_t headerOffset = uint32_t(keepFlags ? data_.headerWireOffset : data_.headerPayloadOffset);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        RecordProcessed(j, wire, headerOffset, keepFlags);
#endif
        if (!enableDedup_ && !keepFlags) {
            asc_sync_wait(PIPE_MTE2, PIPE_MTE3, event_t(lane));
            ub::Store(expandX_ + uint64_t(row) * hiddenBytes_, wire, hiddenBytes_);
            if (wireFp8_) ub::Store(dynamicScales_ + uint64_t(row) * data_.scaleBytes,
                wire + data_.quantCount, uint32_t(data_.scaleBytes));
            ub::Store(destinationIndex_ + uint64_t(row) * 3U, wire + headerOffset, 12U);
            asc_sync_notify(PIPE_MTE3, PIPE_MTE2, event_t(lane));
            return;
        }
        auto* type = reinterpret_cast<__ubuf__ uint32_t*>(wire + data_.tokenStride);
        auto* destination = reinterpret_cast<__ubuf__ uint32_t*>(wire + data_.tokenStride + 32U);
        ServerDedupPacketTail tail(wire, headerOffset,
            uint32_t(keepFlags ? data_.relayReadIndexWireOffset : data_.readIndexPayloadOffset));
        bool firstHop = false;
        if (enableDedup_) firstHop = BuildServerDedupMetadata(type, destination, tail, tasks_.Position(j)) ==
            uint32_t(EpServerDedupTokenType::FirstHop);
        else for (uint32_t i = 0; i < 3U; ++i) destination[i] = tail.Header()[i];
        if (firstHop) {
            PrepareServerDedupForward(tail);
            reinterpret_cast<__ubuf__ float*>(wire)[PacketReceiveLayout::kMarkerOffset / 4U] =
                PacketReceiveLayout::kCompact;
        }
        ub::Sync<PIPE_S, PIPE_MTE3>();
        asc_sync_wait(PIPE_MTE2, PIPE_MTE3, event_t(lane));
        if (keepFlags) {
            WriteServerDedupPayloadRange(expandX_ + uint64_t(row) * hiddenBytes_, wire, 0U, hiddenBytes_);
            if (wireFp8_) WriteServerDedupPayloadRange(dynamicScales_ + uint64_t(row) * data_.scaleBytes,
                wire, uint32_t(data_.quantCount), uint32_t(data_.scaleBytes));
        } else {
            ub::Store(expandX_ + uint64_t(row) * hiddenBytes_, wire, hiddenBytes_);
            if (wireFp8_) ub::Store(dynamicScales_ + uint64_t(row) * data_.scaleBytes,
                wire + data_.quantCount, uint32_t(data_.scaleBytes));
        }
        if (enableDedup_) ub::Store(tokenType_ + row, type, 4U);
        ub::Store(destinationIndex_ + uint64_t(row) * 3U, destination, 12U);
        if (firstHop) {
            ub::Store(relayReadIndex_ + uint64_t(row) * k_ * 2U, tail.ReadIndex(), k_ * 8U);
            const uint32_t source = tail.Header()[0U], ranks = uint32_t(view_.num_ranks), members = tail.ReadIndex()[0U];
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            ++firstHopTokens_; relayMembers_ += members;
            uint64_t seen[4]{};
            for (uint32_t i = 0; i < members; ++i) {
                const uint32_t peer = (tail.Entry(i)[0U] & 0xffffU) / localExperts_, slot = peer >> 6U;
                const uint64_t bit = uint64_t(1) << (peer & 63U);
                if (slot < 4U && !(seen[slot] & bit)) { seen[slot] |= bit; ++relayDistinctRanks_; }
            }
#endif
            for (uint32_t i = 0; i < members; ++i) {
                const uint32_t index = tail.Entry(i)[0U], expert = index & 0xffffU, q = index >> 16U;
                const uint32_t destinationRank = expert / localExperts_;
                comm::Packet(view_, destinationRank, DispatchServerDedupDispatchOffset(data_, source,
                    destinationRank, expert % localExperts_, q, ranks, localExperts_, kNumMaxTokensPerRank),
                    wire, uint32_t(data_.tokenStride), lane);
            }
        }
        asc_sync_notify(PIPE_MTE3, PIPE_MTE2, event_t(lane));
    }
    __aicore__ inline void ClearFlags()
    {
        ub::FillWords(reinterpret_cast<__ubuf__ uint32_t*>(flags_), 0U, uint32_t(data_.blockCount) * 8U);
        ub::Sync<PIPE_V, PIPE_MTE3>();
        for (uint32_t j = 0; j < tasks_.Count(); ++j)
            ub::Store(Slot(j) + 480U, flags_, 32U, uint32_t(data_.blockCount), 480U, 0U, view_.is_rdma());
    }
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    __aicore__ inline void RecordReady(uint32_t mask, uint32_t count)
    {
        ++traceBatches_;
        for (uint32_t i = 0; i < count; ++i) if (mask & (1U << i)) {
            const uint32_t j = snapshot_[i]; ++traceReady_;
            if (j < traceCapacity_) { trace_[j * 8U] = traceReady_; trace_[j * 8U + 1U] = traceBatches_; }
        }
    }
    __aicore__ inline void RecordProcessed(uint32_t j, __ubuf__ uint8_t* wire, uint32_t headerOffset, bool keepFlags)
    {
        if (traceEnabled_) ++traceProcessed_;
        if (!traceEnabled_ || j >= traceCapacity_) return;
        auto* record = trace_ + j * 8U;
        auto* header = reinterpret_cast<__ubuf__ uint32_t*>(wire + headerOffset);
        const uint32_t role = header[3U], members = role == uint32_t(EpServerDedupTokenType::FirstHop) ?
            reinterpret_cast<__ubuf__ uint32_t*>(wire +
                (keepFlags ? data_.relayReadIndexWireOffset : data_.readIndexPayloadOffset))[0U] : 0U;
        record[2U] = traceProcessed_; record[3U] = header[0U]; record[4U] = header[1U];
        record[5U] = header[2U]; record[6U] = role | (members << 8U); record[7U] = tasks_.Position(j);
    }
    __ubuf__ uint32_t* trace_ = nullptr;
    bool traceEnabled_ = false;
    uint32_t traceCapacity_ = 0U, traceReady_ = 0U, traceProcessed_ = 0U, traceBatches_ = 0U;
    uint32_t firstHopTokens_ = 0U, relayMembers_ = 0U, relayDistinctRanks_ = 0U;
#endif
    TransportView view_{};
    bool enableDedup_ = false, wireFp8_ = false;
    uint64_t windowOffset_ = 0U;
    uint32_t recvCoreNum_ = 0U, coreNum_ = 0U, k_ = 0U, experts_ = 0U;
    uint32_t localExperts_ = 0U, receiver_ = 0U, hiddenBytes_ = 0U, laneStride_ = 0U, lanes_ = 0U;
    EpServerDedupDataLayout data_{};
    ServerDedupReceiveTasks tasks_;
    __ubuf__ int32_t* prefix_ = nullptr;
    __ubuf__ float *flags_ = nullptr, *sums_ = nullptr;
    __ubuf__ uint32_t* snapshot_ = nullptr;
    __ubuf__ uint8_t* laneBytes_ = nullptr;
    __gm__ uint8_t *expandX_ = nullptr, *dynamicScales_ = nullptr;
    __gm__ uint32_t *tokenType_ = nullptr, *destinationIndex_ = nullptr, *relayReadIndex_ = nullptr;
};
}  // namespace DispatchDedup

// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

namespace deep_ep {

// Runtime argument block of the SuperKernel sub-entry. One field per explicit
// kernel parameter and in the same order, so the bind record and the ordinary
// launch share one ABI.
struct DispatchArgs {
    GM_ADDR x;
    GM_ADDR expert_ids;
    GM_ADDR expert_scales;
    GM_ADDR expand_x_out;
    GM_ADDR dynamic_scales_out;
    GM_ADDR expert_token_nums_out;
    GM_ADDR send_counts_out;
    GM_ADDR token_type_out;
    GM_ADDR destination_index_out;
    GM_ADDR relay_read_index_out;
    GM_ADDR source_mask_out;
    GM_ADDR source_weights_out;
    GM_ADDR input_scales;
    GM_ADDR peer_ptrs;
    GM_ADDR rdma_remote_base;
    GM_ADDR rdma_stage_base;
    GM_ADDR stage_timer;
    GM_ADDR expert_timer;
    int32_t rank_idx;
    int32_t num_tokens;
    int32_t physical_ranks_per_host;
    int32_t backend;
    int32_t stage_block_num;
    int64_t peer_mem_bytes;
    int64_t ffts_val;
    int32_t timer_mode;
};
EP_STATIC_ASSERT(sizeof(DispatchArgs) == 192U, "SuperKernel argument view must match the global ABI");
EP_STATIC_ASSERT(offsetof(DispatchArgs, rank_idx) == 144U, "rank slot offset");
EP_STATIC_ASSERT(offsetof(DispatchArgs, num_tokens) == 148U, "token count slot offset");
EP_STATIC_ASSERT(offsetof(DispatchArgs, physical_ranks_per_host) == 152U, "ranks-per-host slot offset");
EP_STATIC_ASSERT(offsetof(DispatchArgs, peer_mem_bytes) == 168U, "peer memory slot offset");
EP_STATIC_ASSERT(offsetof(DispatchArgs, ffts_val) == 176U, "ffts slot offset");

// Fixed UB map of one dispatch specialization. Every region is sized from the
// per-rank capacity, so the lane counts and the admitted footprint are
// compile-time facts for this specialization.
template <uint32_t kNumRanks, uint32_t kNumMaxTokensPerRank, uint32_t kNumHiddenElems, uint32_t kNumTopk,
          uint32_t kNumExperts, uint32_t kNumVecCores, uint32_t kNumRanksPerGroup, uint32_t kInputMode>
struct DispatchUBLayout {
    static constexpr uint32_t kNumMaxRoutes = kNumMaxTokensPerRank * kNumTopk;
    static constexpr uint32_t kNumSendVecCores = kNumVecCores / 2U;
    static constexpr uint32_t kNumLocalExperts = kNumExperts / kNumRanks;
    static constexpr uint32_t kNumCapacityRows = kNumRanks * kNumMaxTokensPerRank *
        (kNumTopk < kNumLocalExperts ? kNumTopk : kNumLocalExperts);
    static constexpr uint32_t kNumCountVecCores =
        kNumMaxTokensPerRank < kEpServerDedupAllCoreCountBsThreshold ? kNumVecCores : kNumVecCores / 2U;
    static constexpr uint32_t kNumCountRecords = (kNumExperts + kNumCountVecCores - 1U) / kNumCountVecCores;
    static constexpr uint32_t kNumCountReserved = ((kNumCountRecords + 7U) / 8U) * 32U;
    static constexpr EpServerDedupDataLayout kData = DispatchServerDedupDataLayout(
        kNumMaxTokensPerRank, kNumHiddenElems, kNumTopk, kNumExperts,
        static_cast<EpServerDedupInputMode>(kInputMode));
    static constexpr uint32_t kNumInputBytes = kInputMode == uint32_t(EpServerDedupInputMode::Quantize16)
        ? uint32_t(kData.quantCount * 2U) : uint32_t(kData.packedPayloadBytes);
    static constexpr uint32_t kNumRouteBytes = uint32_t(DispatchDedupAlign(uint64_t(kNumMaxRoutes) * 4U, 256U));
    static constexpr uint32_t kNumQBytes = uint32_t(DispatchDedupAlign(uint64_t(kNumMaxRoutes) * 4U, 32U));
    static constexpr uint32_t kNumScaleCount = uint32_t(kData.quantCount / 32U);
    static constexpr uint32_t kNumQuantWork = uint32_t(DispatchDedupAlign(
        DispatchDedupAlign(kNumScaleCount, 32U) * 4U + uint64_t(kNumScaleCount) * 2U, 32U));
    static constexpr uint32_t kNumScratchBytes = kInputMode == uint32_t(EpServerDedupInputMode::Quantize16) &&
            kNumQuantWork > kNumQBytes ? kNumQuantWork : kNumQBytes;
    static constexpr uint32_t kNumTokenSlice = (kNumMaxTokensPerRank + kNumSendVecCores - 1U) / kNumSendVecCores;
    static constexpr uint32_t kNumGroups = kNumRanks / kNumRanksPerGroup;
    static constexpr uint32_t kNumSenderEnd =
        kNumCountReserved + 2U * kNumInputBytes +
        (kInputMode == uint32_t(EpServerDedupInputMode::Quantize16) ? uint32_t(kData.packedPayloadBytes) : 0U) +
        kNumRouteBytes + 2U * kNumScratchBytes + kNumQBytes + 2U * kNumTokenSlice * kNumTopk * 4U +
        kNumTokenSlice * 4U + kNumTokenSlice * kNumTopk * 16U + 64U +
        (kNumGroups > 1U ? kNumGroups * 4U : 0U) + uint32_t(kData.tokenStride);
    static constexpr uint32_t kNumReceiverEnd =
        uint32_t(DispatchDedupAlign(uint64_t(kNumExperts) * 4U, 32U)) +
        2U * uint32_t(DispatchDedupAlign(((uint64_t(kNumCapacityRows) +
            kNumSendVecCores - 1U) / kNumSendVecCores) * 4U, 32U)) +
        16U * uint32_t(kData.blockCount) * 32U + 256U + 16U * uint32_t(DispatchDedupAlign((uint64_t(kData.blockCount) + 7U) / 8U, 32U)) * 4U +
        256U + 16U * 36U + uint32_t(kData.tokenStride) + 64U;
    EP_STATIC_ASSERT(kNumSenderEnd <= kNumUbBytes, "Dispatch sender UB exceeds capacity");
    EP_STATIC_ASSERT(kNumReceiverEnd <= kNumUbBytes, "Dispatch receiver UB exceeds capacity");
};

template <uint32_t kNumRanks, uint32_t kNumMaxTokensPerRank, uint32_t kNumHiddenBytes, uint32_t kNumTopk,
          uint32_t kNumExperts, uint32_t kNumVecCores, uint32_t kInputClass, uint32_t kNumRanksPerGroup,
          uint32_t kInputMode, bool kE4M3, bool kDedup, bool kRdma, class XType>
__aicore__ inline void dispatch_body(
    GM_ADDR x, GM_ADDR expert_ids, GM_ADDR expert_scales,
    GM_ADDR expand_x_out, GM_ADDR dynamic_scales_out, GM_ADDR expert_token_nums_out,
    GM_ADDR send_counts_out, GM_ADDR token_type_out, GM_ADDR destination_index_out,
    GM_ADDR relay_read_index_out, GM_ADDR source_mask_out, GM_ADDR source_weights_out,
    GM_ADDR input_scales, GM_ADDR peer_ptrs, GM_ADDR rdma_remote_base, GM_ADDR rdma_stage_base,
    GM_ADDR stage_timer, GM_ADDR expert_timer,
    const int32_t rank_idx, const int32_t num_tokens, const int32_t physical_ranks_per_host,
    const int32_t backend, const int32_t stage_block_num, const int64_t peer_mem_bytes,
    const int64_t ffts_val, const int32_t timer_mode, CoreContext core)
{
    (void)expert_timer;
    (void)timer_mode;
    (void)rdma_stage_base;
    TransportView view{};
    view.peer_words = reinterpret_cast<__gm__ const uint64_t*>(peer_ptrs);
    view.rdma_remote_base = rdma_remote_base;
    view.rdma_stage_base = rdma_stage_base;
    view.rank = uint32_t(rank_idx);
    view.num_ranks = kNumRanks;
    view.ranks_per_host = uint32_t(physical_ranks_per_host);
    view.stage_block_num = uint32_t(stage_block_num);
    view.backend = uint32_t(backend);
    view.peer_mem_bytes = uint64_t(peer_mem_bytes);

    EP_DEVICE_ASSERT(view.num_ranks == kNumRanks && view.rank < kNumRanks);
    EP_DEVICE_ASSERT(core.count == kNumVecCores && core.index < kNumVecCores);
    EP_DEVICE_ASSERT(num_tokens >= 0 && uint32_t(num_tokens) <= kNumMaxTokensPerRank);
    EP_DEVICE_ASSERT(view.is_rdma() == kRdma);

    util_set_ffts_config(uint64_t(ffts_val));

    // Hidden counts are element counts in the device layout; the template
    // carries the byte count, matching DeepEP-Ascend's kNumHiddenBytes.
    constexpr uint32_t kNumHiddenElems = kInputMode == uint32_t(EpServerDedupInputMode::PrequantizedFp8Packs)
        ? kNumHiddenBytes : kNumHiddenBytes / sizeof(XType);
    using ub_layout_t = DispatchUBLayout<kNumRanks, kNumMaxTokensPerRank, kNumHiddenElems, kNumTopk,
                                         kNumExperts, kNumVecCores, kNumRanksPerGroup, kInputMode>;
    (void)sizeof(ub_layout_t);

    profile::DeviceProfile profile_state;
    if constexpr (true) {
        profile_state.Init(stage_timer);
        profile_state.Value<0>(profile::DeviceProfile::EntryClock(stage_timer));
        profile_state.Mark<1>();
    }

    const uint32_t magic = RotateDispatchServerDedupMagic(view, core.index);
    profile_state.Mark<2>();
    profile_state.Mark<3>();

    constexpr auto inputMode = static_cast<EpServerDedupInputMode>(kInputMode);
    ServerDedupCounts<kNumMaxTokensPerRank> counts;
    counts.Init(view, expert_ids, expert_token_nums_out, send_counts_out, uint32_t(num_tokens), kNumTopk,
                kNumExperts, 0U, int64_t(magic), core, &profile_state);
    profile_state.Mark<4>();

    ServerDedupSender<kNumMaxTokensPerRank, XType, kNumRanksPerGroup> sender;
    ServerDedupReceiver<kNumMaxTokensPerRank> receiver;
    const uint32_t send_cores = core.count / 2U;
    if (core.index < send_cores)
        sender.Init(view, x, expert_ids, expert_scales, source_mask_out, source_weights_out, uint32_t(num_tokens),
                    kNumHiddenElems, kNumTopk, kNumExperts, inputMode, kE4M3, int64_t(magic), input_scales, kDedup,
                    core);
    else
        receiver.Init(view, expand_x_out, dynamic_scales_out, token_type_out, destination_index_out,
                      relay_read_index_out, kNumHiddenElems, kNumTopk, kNumExperts, inputMode, int64_t(magic),
                      kDedup, core);
    profile_state.Mark<5>();

    profile_state.Mark<6>();
    counts.SendCnt();
    profile_state.Mark<7>();

    if (core.index < send_cores) {
        profile_state.Mark<8>();
        sender.Process(counts.ReservedBytes(), &profile_state, counts.Retirement());
        profile_state.Mark<9>();
    } else {
        profile_state.Mark<10>();
        counts.RecvCnt();
        profile_state.Mark<11>();
        profile_state.Mark<12>();
        counts.CalcCumsum();
        profile_state.Mark<13>();
        profile_state.Mark<14>();
        counts.CompleteCumsum();
        profile_state.Mark<15>();
        profile_state.Mark<16>();
        asc_sync_pipe(PIPE_ALL);
        profile_state.Mark<17>();
        profile_state.Mark<18>();
        receiver.Process(&profile_state);
        profile_state.Mark<19>();
    }

    profile_state.Mark<20>();
    counts.Finish();
    if (counts.Retirement()->Pending()) {
        if (core.index >= send_cores) counts.Retirement()->Finish(receiver.FinalScratch());
        else if (sender.HasFinalScratch()) counts.Retirement()->Finish(sender.FinalScratch());
        else counts.RetireOwnScratch();
    }
    profile_state.Mark<21>();
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    if (core.index >= send_cores) receiver.ExportPacketTrace(&profile_state);
#endif
    profile_state.Finish(uint32_t(rank_idx), kNumRanks, uint32_t(num_tokens));
}

template <uint32_t kNumRanks, uint32_t kNumMaxTokensPerRank, uint32_t kNumHiddenBytes, uint32_t kNumTopk,
          uint32_t kNumExperts, uint32_t kNumVecCores, uint32_t kInputClass, uint32_t kNumRanksPerGroup,
          uint32_t kInputMode, bool kE4M3, bool kDedup, bool kRdma>
__global__ __aicore__ void dispatch_impl(
    GM_ADDR x, GM_ADDR expert_ids, GM_ADDR expert_scales,
    GM_ADDR expand_x_out, GM_ADDR dynamic_scales_out, GM_ADDR expert_token_nums_out,
    GM_ADDR send_counts_out, GM_ADDR token_type_out, GM_ADDR destination_index_out,
    GM_ADDR relay_read_index_out, GM_ADDR source_mask_out, GM_ADDR source_weights_out,
    GM_ADDR input_scales, GM_ADDR peer_ptrs, GM_ADDR rdma_remote_base, GM_ADDR rdma_stage_base,
    GM_ADDR stage_timer, GM_ADDR expert_timer,
    const int32_t rank_idx, const int32_t num_tokens, const int32_t physical_ranks_per_host,
    const int32_t backend, const int32_t stage_block_num, const int64_t peer_mem_bytes,
    const int64_t ffts_val, const int32_t timer_mode)
{
    EP_STATIC_ASSERT(kNumVecCores > 0 && kNumMaxTokensPerRank > 0 && kNumHiddenBytes > 0);
    EP_STATIC_ASSERT(kNumRanks <= kNumMaxRanksPerGroup * kNumMaxVecCores);
    if constexpr (kInputClass == 1)
        dispatch_body<kNumRanks, kNumMaxTokensPerRank, kNumHiddenBytes, kNumTopk, kNumExperts, kNumVecCores,
                      kInputClass, kNumRanksPerGroup, kInputMode, kE4M3, kDedup, kRdma, bfloat16_t>(
            x, expert_ids, expert_scales, expand_x_out, dynamic_scales_out, expert_token_nums_out, send_counts_out,
            token_type_out, destination_index_out, relay_read_index_out, source_mask_out, source_weights_out,
            input_scales, peer_ptrs, rdma_remote_base, rdma_stage_base, stage_timer, expert_timer, rank_idx,
            num_tokens, physical_ranks_per_host, backend, stage_block_num, peer_mem_bytes, ffts_val, timer_mode,
            OrdinaryCoreContext());
    else
        dispatch_body<kNumRanks, kNumMaxTokensPerRank, kNumHiddenBytes, kNumTopk, kNumExperts, kNumVecCores,
                      kInputClass, kNumRanksPerGroup, kInputMode, kE4M3, kDedup, kRdma, half>(
            x, expert_ids, expert_scales, expand_x_out, dynamic_scales_out, expert_token_nums_out, send_counts_out,
            token_type_out, destination_index_out, relay_read_index_out, source_mask_out, source_weights_out,
            input_scales, peer_ptrs, rdma_remote_base, rdma_stage_base, stage_timer, expert_timer, rank_idx,
            num_tokens, physical_ranks_per_host, backend, stage_block_num, peer_mem_bytes, ffts_val, timer_mode,
            OrdinaryCoreContext());
}

// One SuperKernel sub-entry template. `kSkSlot` only distinguishes the four
// symbols the bind table requires; the logic is identical.
template <uint32_t kNumRanks, uint32_t kNumMaxTokensPerRank, uint32_t kNumHiddenBytes, uint32_t kNumTopk,
          uint32_t kNumExperts, uint32_t kNumVecCores, uint32_t kInputClass, uint32_t kNumRanksPerGroup,
          uint32_t kInputMode, bool kE4M3, bool kDedup, bool kRdma, uint32_t kSkSlot>
__aicore__ inline void dispatch_impl_sk(const DispatchArgs* args, sk::SkSystemArgs* system_args)
{
    system_args->SkInit();
    const CoreContext core{uint32_t(system_args->SkGetBlockIdx()), uint32_t(system_args->SkGetNumBlocks())};
    if constexpr (kInputClass == 1)
        dispatch_body<kNumRanks, kNumMaxTokensPerRank, kNumHiddenBytes, kNumTopk, kNumExperts, kNumVecCores,
                      kInputClass, kNumRanksPerGroup, kInputMode, kE4M3, kDedup, kRdma, bfloat16_t>(
            args->x, args->expert_ids, args->expert_scales, args->expand_x_out, args->dynamic_scales_out,
            args->expert_token_nums_out, args->send_counts_out, args->token_type_out, args->destination_index_out,
            args->relay_read_index_out, args->source_mask_out, args->source_weights_out, args->input_scales,
            args->peer_ptrs, args->rdma_remote_base, args->rdma_stage_base, args->stage_timer, args->expert_timer,
            args->rank_idx, args->num_tokens, args->physical_ranks_per_host, args->backend, args->stage_block_num,
            args->peer_mem_bytes, args->ffts_val, args->timer_mode, core);
    else
        dispatch_body<kNumRanks, kNumMaxTokensPerRank, kNumHiddenBytes, kNumTopk, kNumExperts, kNumVecCores,
                      kInputClass, kNumRanksPerGroup, kInputMode, kE4M3, kDedup, kRdma, half>(
            args->x, args->expert_ids, args->expert_scales, args->expand_x_out, args->dynamic_scales_out,
            args->expert_token_nums_out, args->send_counts_out, args->token_type_out, args->destination_index_out,
            args->relay_read_index_out, args->source_mask_out, args->source_weights_out, args->input_scales,
            args->peer_ptrs, args->rdma_remote_base, args->rdma_stage_base, args->stage_timer, args->expert_timer,
            args->rank_idx, args->num_tokens, args->physical_ranks_per_host, args->backend, args->stage_block_num,
            args->peer_mem_bytes, args->ffts_val, args->timer_mode, core);
}

}  // namespace deep_ep
