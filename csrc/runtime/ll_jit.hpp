// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

// Optional runtime compilation of the LL dispatch kernel through DeepJIT.
// Only the Host module uses this; the Ahead-of-Time device library keeps the
// same entry header, so both paths build identical kernel code.
#pragma once
#include <cstdint>

#include "../kernels/dispatch/low_latency/launch.hpp"

namespace deepep {
#ifdef DEEPEP_LL_JIT
// Compiles the entry on first use (cached on disk) and launches it on the
// provided stream. Throws with a diagnostic when the environment is incomplete.
void ll_jit_launch(void* stream, const EpServerDedupKernelArgs& args);
#else
inline void ll_jit_launch(void*, const EpServerDedupKernelArgs&) {}
#endif
}  // namespace deepep
