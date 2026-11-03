// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

// Ahead-of-time translation unit for the LL dispatch kernel. The entry body is
// shared with any JIT backend through dispatch_entry.h; Host capability exports
// live in csrc/ops/ll_capabilities.cpp.
#if CATLASS_ARCH == 3510
#include "low_latency/dispatch_entry.h"
#endif
