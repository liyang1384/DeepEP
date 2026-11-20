// Copyright (c) 2026, Lu Lu
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#include <kernel_operator.h>
#endif

#ifndef GM_ADDR
using GM_ADDR = uint8_t*;
#endif

// Host-side stand-ins for the CCE keywords, mirroring
// DeepEP-Ascend/deep_ep/common/compiled.hpp. Device builds keep the
// compiler-provided keywords, so every header spells
// `constexpr __aicore__ __forceinline__` directly instead of wrapping each
// function in a project-specific macro.
#if not defined(__CCE__)
#define __aicore__
#define __forceinline__ inline
#define __gm__
#define __ubuf__
#define __simd_vf__
#define __simd_callee__
#endif

#include <cstdint>
namespace deep_ep {
constexpr uint32_t kNumRanksPerHost = 16U;
constexpr uint32_t kNumMaxRanksPerGroup = 16U;
enum class LlTilingKey : uint8_t { Group4 = 4U, Group16 = 16U };
constexpr __aicore__ __forceinline__ uint32_t LlGroupForWorld(uint32_t world) { return world <= 16U ? 4U : 16U; }
constexpr __aicore__ __forceinline__ uint8_t LlTilingKeyForWorld(uint32_t world) { return uint8_t(LlGroupForWorld(world)); }
constexpr __aicore__ __forceinline__ bool LlTilingKeyValid(uint32_t world, uint8_t key) { return key == LlTilingKeyForWorld(world); }
constexpr __aicore__ __forceinline__ bool LlSameHost(uint32_t rank, uint32_t peer, uint32_t ranksPerHost = kNumRanksPerHost) {
    return ranksPerHost != 0U && rank / ranksPerHost == peer / ranksPerHost;
}
}

#include <cstdint>
namespace deep_ep {
// Count frames and the full Dispatch C API are one wire. The remote/local
// histogram split is gone, so this protocol is the only LL count identity.
constexpr uint32_t kLlCoreProtocolVersion = 36U; // multi-host direct counts, 512B DataAsFlag frames, raw C API
// Capacity is not participation. Pure AIV launches use the queried even count.
constexpr uint32_t kNumMaxVecCores = 72U;
constexpr uint32_t kNumMaxRecvVecCores = kNumMaxVecCores / 2U;
constexpr uint32_t kNumDefaultPlanVecCores = 64U; // CPU-only legacy helper default
constexpr uint32_t kNumNotifyVecCores = 8U;
// Physical UB capacity. Host admission and every device layout read this.
constexpr uint32_t kNumUbBytes = 190U * 1024U;
constexpr __aicore__ __forceinline__ bool LlCoreCountValid(uint32_t cores) {
    return cores >= 16U && cores <= kNumMaxVecCores && cores % 2U == 0U;
}
}

namespace deep_ep {
namespace rdma {
constexpr uint32_t kBackend = 1U;
constexpr uint32_t kCoreBytes = 128U * 1024U;
// ABI v3 tail: addresses[0]=unbiased symmetric workspace, [1]=per-core staging;
// config[0]=backend, [1]=QP count. Remote peerMems are NULL in RDMA mode.
}
}  // namespace deep_ep

#include <cstdint>

namespace deep_ep {
// Internal dispatch classification, independent of the shared-memory
// communication ABI. Numeric format values are checked against the public
// enums by the Host binding.
constexpr uint8_t kLlDispatchFp16 = 3;
constexpr uint8_t kLlDispatchBf16 = 11;
constexpr uint8_t kLlDispatchE4M3 = 15;
constexpr uint8_t kLlDispatchE5M2 = 16;
constexpr uint8_t kLlDispatchPlain16 = 0;
constexpr uint8_t kLlDispatchQuantize16 = 1;
constexpr uint8_t kLlDispatchPrequantized = 2;
constexpr uint32_t kLlDispatchInvalidVariant = 0xffffffffU;

// The entry kernel has two compile-time input element classes; the group size
// is the second axis. Plain16, Quantize16 with E4M3/E5M2 and
// PrequantizedFp8Packs are runtime tiling fields of that one kernel, so the
// format matrix never multiplies translation units. Prequantized rows are
// opaque FP8 byte packs whose element type is never read; they use the
// FP16-class kernel because both 16-bit families have the same width.
constexpr uint32_t kLlDispatchInputFp16 = 0U;
constexpr uint32_t kLlDispatchInputBf16 = 1U;

// Returns the compile-time kernel class for a tiling format, or
// kLlDispatchInvalidVariant when the format is not admitted. Host and device
// evaluate the same table before launching or running the kernel.
constexpr __aicore__ __forceinline__ uint32_t LlDispatchInputClass(uint8_t dtype, uint8_t quantMode,
    uint8_t expandXOutDtype, uint8_t inputMode)
{
    if (dtype == kLlDispatchE4M3 && quantMode == 4U &&
        expandXOutDtype == kLlDispatchE4M3 && inputMode == kLlDispatchPrequantized) {
        return kLlDispatchInputFp16;
    }
    if (dtype == kLlDispatchFp16 || dtype == kLlDispatchBf16) {
        const uint32_t inputClass = dtype == kLlDispatchFp16 ? kLlDispatchInputFp16 : kLlDispatchInputBf16;
        if (quantMode == 0U && inputMode == kLlDispatchPlain16 &&
            (expandXOutDtype == dtype || expandXOutDtype == 255U)) {
            return inputClass;
        }
        if (quantMode == 4U && inputMode == kLlDispatchQuantize16 &&
            (expandXOutDtype == kLlDispatchE4M3 || expandXOutDtype == kLlDispatchE5M2)) {
            return inputClass;
        }
    }
    return kLlDispatchInvalidVariant;
}
}  // namespace DispatchDedup

#include <cstdint>

namespace deep_ep {
// The dedup resource layout is fixed for the lifetime of the binary. The
// execution path is chosen at runtime from the per-rank capacity instead.
constexpr uint32_t kEpServerDedupDedupThreshold = 32U;
// Only the per-rank capacity and communicator size select the execution path;
// the actual launch B no longer participates.
constexpr __aicore__ __forceinline__ bool EpServerDedupEnabled(uint32_t numMaxTokensPerRank, uint32_t world = 16U)
{
    return world > 16U || numMaxTokensPerRank > kEpServerDedupDedupThreshold;
}
}  // namespace DispatchDedup

#include <cstddef>
#include <cstdint>

#ifdef DEEPEP_LL_RANKS_PER_GROUP
#error LL grouping is selected by tiling key; remove DEEPEP_LL_RANKS_PER_GROUP
#endif

namespace deep_ep {
// Routing group is selected by Host tiling key; this constant is capacity only.
constexpr uint32_t kLlRanksPerGroup = kNumMaxRanksPerGroup;
constexpr uint32_t kEpServerDedupAlignmentBytes = 32U;
constexpr uint32_t kEpServerDedupMaxTopK = 32U;

// The wire header and TokenType output use the same values.
enum class EpServerDedupTokenType : uint32_t { Direct = 1U, FirstHop = 2U, Terminal = 3U };

struct alignas(32) EpServerDedupPacketHeader {
    uint32_t srcRank;
    uint32_t srcTokenId;
    uint32_t topK;  // Original route index / returnK, not the shape K.
    uint32_t type;
    uint32_t padding[4];
};

// Block 0: index=N, weightBits=selfWeight. Member: index=expert|(q<<16).
// A RelayReadIndex row contains exactly K blocks; only blocks 0..N are valid.
struct EpServerDedupReadIndexBlock {
    uint32_t index;
    uint32_t weightBits;
};

struct EpServerDedupDirectTask {
    uint32_t tokenOffset;
    uint32_t originalK;
    uint32_t expertId;
    uint32_t q;
};
static_assert(sizeof(EpServerDedupTokenType) == 4U, "token type ABI");
static_assert(sizeof(EpServerDedupPacketHeader) == 32U, "packet header ABI");
static_assert(alignof(EpServerDedupPacketHeader) == 32U, "packet header alignment");
static_assert(offsetof(EpServerDedupPacketHeader, srcTokenId) == 4U, "source token offset");
static_assert(offsetof(EpServerDedupPacketHeader, topK) == 8U, "return k offset");
static_assert(offsetof(EpServerDedupPacketHeader, type) == 12U, "packet role offset");
static_assert(sizeof(EpServerDedupReadIndexBlock) == 8U, "read index block ABI");
static_assert(offsetof(EpServerDedupReadIndexBlock, weightBits) == 4U, "weight bits offset");
static_assert(sizeof(EpServerDedupDirectTask) == 16U, "source direct task ABI");
}  // namespace DispatchDedup

// Host/device wire constants shared with the 256-rank CommArgs ABI.

namespace deep_ep {

constexpr int DISPATCH_DEDUP_MAX_RANK_SIZE = 256;

constexpr int DFX_COUNT = 50;

// Flag tail reserved at the end of every symmetric window.

constexpr int DISPATCH_DEDUP_FLAG_BUFF_BYTES = 4 * 1024 * 1024;

// Tiling dtype codes. The values are the Host/device wire ABI.

enum DispatchDedupDataType {

    DISPATCH_DEDUP_DATA_TYPE_FP16 = 3,

    DISPATCH_DEDUP_DATA_TYPE_BFP16 = 11,

    DISPATCH_DEDUP_DATA_TYPE_FP8E4M3 = 15,

    DISPATCH_DEDUP_DATA_TYPE_FP8E5M2 = 16,

    DISPATCH_DEDUP_DATA_TYPE_RESERVED = 255

};

}  // namespace deep_ep

#include <cstdint>

namespace deep_ep {
// Wire FP8 does not imply internal quantization. Values also form the tiling ABI.
enum class EpServerDedupInputMode : uint8_t {
    Plain16 = 0,
    Quantize16 = 1,
    PrequantizedFp8Packs = 2,
};

// Host-only classification; no implicit conversion or dtype-based quantization.
inline bool DispatchDedupResolveInputMode(int64_t quantMode, int64_t dtype, bool hasInputScales, int64_t outputDtype,
                                          EpServerDedupInputMode* mode)
{
    if (mode == nullptr) return false;
    const bool input16 =
        dtype == DISPATCH_DEDUP_DATA_TYPE_FP16 || dtype == DISPATCH_DEDUP_DATA_TYPE_BFP16;
    if (quantMode == 0 && input16 && !hasInputScales &&
        (outputDtype == dtype || outputDtype == DISPATCH_DEDUP_DATA_TYPE_RESERVED)) {
        *mode = EpServerDedupInputMode::Plain16;
        return true;
    }
    if (quantMode != 4) return false;
    if (input16 && !hasInputScales &&
        (outputDtype == DISPATCH_DEDUP_DATA_TYPE_FP8E4M3 ||
         outputDtype == DISPATCH_DEDUP_DATA_TYPE_FP8E5M2)) {
        *mode = EpServerDedupInputMode::Quantize16;
        return true;
    }
    if (dtype == DISPATCH_DEDUP_DATA_TYPE_FP8E4M3 && hasInputScales && outputDtype == dtype) {
        *mode = EpServerDedupInputMode::PrequantizedFp8Packs;
        return true;
    }
    return false;
}
}  // namespace DispatchDedup

namespace deep_ep {
constexpr uint64_t kEpServerDedupPeerBytes = 8ULL * 1024U * 1024U * 1024U;
constexpr uint64_t kEpServerDedupWindowBytes = kEpServerDedupPeerBytes / 2U;
constexpr uint64_t kEpServerDedupControlBytes = 2ULL * 1024U * 1024U;
// Runtime owns the domain's final flag region. Leave the same tail unused in
// window 0 so both windows have identical capacity and shape-independent stride.
constexpr uint64_t kEpServerDedupWindowUsableBytes =
    kEpServerDedupWindowBytes - DISPATCH_DEDUP_FLAG_BUFF_BYTES;
}  // namespace deep_ep
