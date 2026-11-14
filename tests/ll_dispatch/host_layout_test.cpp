// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026
//
// Host-only checks for the LL count ABI and DIRECT send order.
// These helpers are header-only and independent of CANN, so a plain
// C++17 compiler can validate the wire arithmetic without an NPU:
//
//   g++ -std=c++17 -O2 -I csrc tests/ll_dispatch/host_layout_test.cpp -o /tmp/ll_host_layout
//   /tmp/ll_host_layout
//
// They cover: 512B DataAsFlag block arithmetic, direct/local LlCountFrame
// addressing, the read-plan run reconstruction and expected flag sum, and the
// remote-first DIRECT group permutation (own group last, other groups a
// bijection).

#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <set>

#include "kernels/dispatch/low_latency/common/ll_count_frame_layout.h"
#include "kernels/dispatch/low_latency/common/ll_count_read_plan.h"
#include "kernels/dispatch/low_latency/common/ll_send_order.h"

namespace {

using DispatchDedup::CountFrame;
using DispatchDedup::CountWireBytes;
using DispatchDedup::CountWireRecordOffset;
using DispatchDedup::LlCountFrame;
using DispatchDedup::LlCountReadPlan;
using DispatchDedup::BuildCountReadPlan;

constexpr uint32_t kWorld = 32U;
constexpr uint32_t kExperts = 384U;  // 12 local experts per rank
constexpr uint32_t kLocal = kExperts / kWorld;

void CheckWireArithmetic()
{
    assert(CountWireBytes(0U) == 0U);
    assert(CountWireBytes(1U) == 512U);
    assert(CountWireBytes(15U) == 512U);
    assert(CountWireBytes(16U) == 1024U);
    assert(CountWireBytes(30U) == 1024U);
    // Fifteen 32B records then eight FP32 flags occupy one 512B block.
    assert(CountWireRecordOffset(0U) == 0U);
    assert(CountWireRecordOffset(14U) == 448U);
    assert(CountWireRecordOffset(15U) == 512U);
    assert(CountWireRecordOffset(29U) == 512U + 448U);
}

void CheckRemoteDirectFrame()
{
    // A cross-host DIRECT count frame is the dense local-expert body plus one
    // trailing 32B SO commit, all inside one source/core reservation.
    const LlCountFrame f = CountFrame(kExperts, kWorld, 16U, 16U, 0U, 0U, true);
    assert(f.records == kLocal);
    assert(f.flagWords == 1U);
    assert(f.wireBytes == (kLocal + 1U) * 32U);
    assert(f.commitOffset == uint64_t(kLocal) * 32U);
    assert(f.recordOffset == 0U);
    assert(f.sourceOffset == 0U);
}

void CheckLocalDataAsFlagFrame()
{
    // Same-host frames send complete 512B blocks. Destination rank 1 receives
    // this core's whole host-clamped range, so record zero and the block flag
    // image share one block: records 0..14 * 32B, flags at 480.
    const LlCountFrame f = CountFrame(kExperts, kWorld, 32U, 1U, 0U, 0U, true);
    assert(f.flagWords == 8U);
    assert(f.wireBytes % 512U == 0U);
    assert(f.commitOffset == 480U);
    assert(f.sourceOffset == 0U);
    assert(f.recordOffset == 0U);
    assert(f.wireBytes / 512U == 1U);  // 12 records still fit one block
}

void CheckReadPlan()
{
    LlCountReadPlan plan;
    // Receiver 0 owns a 64-expert window that starts at logical record 96.
    // Direct reads change source-host layout every 16 sources, so the plan is
    // four 16-record runs; flags are summed per logical record.
    const bool ok = BuildCountReadPlan(kExperts, kWorld, 16U, 0U, 96U, 64U, plan, true);
    assert(ok);
    assert(plan.size == 4U);
    uint32_t consumed = 0U, flags = 0U;
    for (uint32_t r = 0; r < plan.size; ++r) {
        assert(plan.runs[r].first == consumed);
        assert(plan.runs[r].records == 16U);
        const uint32_t logical = 96U + consumed;
        const uint32_t source = logical % kWorld;
        const LlCountFrame f = CountFrame(kExperts, kWorld, 16U, 0U, source, logical / kWorld, true);
        assert(f.flagWords == (source < 16U ? 8U : 1U));
        flags += plan.runs[r].records * f.flagWords;
        consumed += plan.runs[r].records;
    }
    assert(consumed == 64U);
    assert(plan.expectedFlagSum == flags);
    // Invalid shapes and oversized runs are rejected, never silently clamped.
    assert(!BuildCountReadPlan(16U, 16U, 8U, 0U, 0U, 16U, plan, true));
    assert(!BuildCountReadPlan(kExperts, kWorld, 16U, 0U, 0U, 129U, plan, true));
    assert(!BuildCountReadPlan(kExperts, kWorld, 0U, 0U, 0U, 8U, plan, true));
}

void CheckDirectOrder(bool remoteFirst)
{
    // EP32 => two Group16 groups; EP64/128 => four/eight. Verify both the
    // small-domain rotation and the larger half-interleaved permutation.
    for (uint32_t world : {32U, 64U, 128U}) {
        const uint32_t groups = world / 16U;
        const uint32_t own = 1U;
        std::array<bool, 64> seen{};
        for (uint32_t ordinal = 0; ordinal < groups; ++ordinal) {
            const uint32_t got = DispatchDedup::LlDirectGroupOrder(
                groups, own, 3U, 5U, ordinal, remoteFirst);
            assert(got < groups);
            assert(!seen[got]);
            seen[got] = true;
            if (remoteFirst) assert(ordinal == groups - 1U ? got == own : got != own);
        }
    }
}

void CheckSendStagger()
{
    // Rotation is a permutation for every phase and stays in range at the wrap.
    for (uint32_t count : {1U, 2U, 5U, 16U}) {
        for (uint32_t phase = 0; phase < 40U; ++phase) {
            std::set<uint32_t> seen;
            for (uint32_t ordinal = 0; ordinal < count; ++ordinal)
                seen.insert(DispatchDedup::LlSendStaggerOrdinal(count, phase, ordinal));
            assert(seen.size() == count);
            assert(*seen.begin() == 0U && *seen.rbegin() == count - 1U);
        }
    }
}

}  // namespace

int main()
{
    CheckWireArithmetic();
    CheckRemoteDirectFrame();
    CheckLocalDataAsFlagFrame();
    CheckReadPlan();
    CheckDirectOrder(false);
    CheckDirectOrder(true);
    CheckSendStagger();
    std::printf("ll host layout checks passed\n");
    return 0;
}
