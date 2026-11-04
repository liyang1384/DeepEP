// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

// Launch identity: world size and active core count.
//
// Ahead-of-Time builds read them at run time (CommArgs.rankSize and
// GetBlockNum()); the JIT build bakes them into the translation unit together
// with the shape, so per-core partitioning and world-derived offsets fold to
// constants. The Host adds them to the JIT cache key, so one compiled kernel is
// only ever launched with the world size and core count it was built for.
#pragma once

#ifdef DEEPEP_LL_JIT_SPECIALIZED
#define DEEPEP_LL_LAUNCH_WORLD(args) (DEEPEP_LL_JIT_SPEC_WORLD)
#define DEEPEP_LL_LAUNCH_CORES() (DEEPEP_LL_JIT_SPEC_CORES)
#else
#define DEEPEP_LL_LAUNCH_WORLD(args) (uint32_t((args)->rankSize))
#define DEEPEP_LL_LAUNCH_CORES() (GetBlockNum())
#endif
