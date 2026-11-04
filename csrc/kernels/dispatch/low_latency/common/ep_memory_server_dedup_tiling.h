// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#ifndef DISPATCH_DEDUP_EP_MEMORY_SERVER_DEDUP_TILING_H
#define DISPATCH_DEDUP_EP_MEMORY_SERVER_DEDUP_TILING_H

#include <cstddef>
#include <cstdint>

namespace DispatchDedup {
// Shared Host/AICore wire layout. Host validates before narrowing.
// Magic is persistent device state, not tiling or a Host launch value.
//
// expertTokenNumsType is a protocol constant, not a per-case choice: the Host
// tiling builder writes this value and neither side ever varies it. The JIT
// build bakes the same constant, so the field is only kept for the tiling
// recheck and the Ahead-of-Time path.
constexpr uint8_t kLlExpertTokenNumsType = 1;
struct EpServerDedupTilingData {
    uint32_t bs;
    uint32_t h;
    uint32_t numMaxTokensPerRank;
    uint16_t moeExpertNum;
    uint8_t topK;
    uint8_t reservedSendCoreNum;
    uint8_t expertTokenNumsType;
    uint8_t quantMode;
    uint8_t dtype;
    uint8_t expandXOutDtype;
    uint8_t inputMode;  // EpServerDedupInputMode; P is derived from H.
    uint8_t tilingKey;  // LlTilingKey: selects Group4/Group16 specialization.
    uint8_t reserved[10];
};
static_assert(sizeof(EpServerDedupTilingData) == 32U, "tiling occupies one 32B block");
// Keep the wire structure naturally aligned. Kernel reads use scalar GM
// accesses, not an implicit vector/MTE load of the entire block.
static_assert(alignof(EpServerDedupTilingData) == 4U, "tiling uses natural scalar alignment");
static_assert(offsetof(EpServerDedupTilingData, numMaxTokensPerRank) == 8U, "per-rank capacity offset");
static_assert(offsetof(EpServerDedupTilingData, moeExpertNum) == 12U, "expert count offset");
static_assert(offsetof(EpServerDedupTilingData, topK) == 14U, "topK offset");
static_assert(offsetof(EpServerDedupTilingData, reservedSendCoreNum) == 15U, "reserved offset");
static_assert(offsetof(EpServerDedupTilingData, inputMode) == 20U, "input mode offset");

}  // namespace DispatchDedup

#endif
