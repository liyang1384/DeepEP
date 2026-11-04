// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

// Optional runtime compilation of the LL dispatch kernel through DeepJIT.
// Only the Host module uses this; the Ahead-of-Time device library compiles the
// same entry header without the specialization macros, so the kernel body is
// shared and only the compile-time axes differ.
#pragma once
#include <cstdint>

#include "../kernels/dispatch/low_latency/launch.hpp"

namespace deepep {
// Launch identity baked into a JIT translation unit. The generated entry
// source defines the DEEPEP_LL_JIT_SPEC_* macros from these fields, so one
// compiled kernel is valid for exactly one shape/world/core combination. The
// Host adds the same values to the cache key; the kernel does not read tiling.
struct LlDispatchJitSpec {
    uint32_t bs;
    uint32_t h;
    uint32_t topK;
    uint32_t experts;
    uint32_t cores;       // active AIV count of this launch
    uint32_t world;       // EP world size (CommArgs.rankSize)
    uint32_t inputClass;  // DispatchDedup::kLlDispatchInputFp16/Bf16
    uint32_t group;       // LlTilingKey: 4 or 16
    uint32_t inputMode;   // EpServerDedupInputMode
    bool e4m3;
    bool dedup;
};

#ifdef DEEPEP_LL_JIT
// Compiles the entry for this exact case on first use (cached on disk) and
// launches it on the provided stream. Throws with a diagnostic when the
// environment is incomplete.
void ll_jit_launch(void* stream, const EpServerDedupKernelArgs& args, const LlDispatchJitSpec& spec);
#else
inline void ll_jit_launch(void*, const EpServerDedupKernelArgs&, const LlDispatchJitSpec&) {}
#endif
}  // namespace deepep
