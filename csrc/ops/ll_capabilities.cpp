// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

// Host-side LL capability exports. They never touch device state, so keeping
// them out of the dispatch translation unit lets that kernel be built either
// ahead of time or through a JIT backend without changing the Host contract.
#include <cstdint>

#include "../kernels/dispatch/low_latency/common/ll_core_config.h"
#include "../kernels/dispatch/low_latency/common/ll_group_policy.h"
#include "../kernels/dispatch/low_latency/common/ll_reference_profile_layout.h"
#include "../kernels/launch.hpp"

extern "C" DEEPEP_EXPORT uint32_t deepep_ll_stage_timer_enabled()
{
    return 0U;  // Legacy stage/expert device instrumentation has been removed.
}
extern "C" DEEPEP_EXPORT uint32_t deepep_ll_reference_profile_enabled()
{
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    return 1U;
#else
    return 0U;
#endif
}
extern "C" DEEPEP_EXPORT uint32_t deepep_ll_reference_profile_schema()
{
    return LlReferenceProfile::kSchema;
}
extern "C" DEEPEP_EXPORT uint32_t deepep_ll_ranks_per_group(uint32_t world)
{
    return world <= 16U ? uint32_t(DispatchDedup::LlTilingKey::Group4) : uint32_t(DispatchDedup::LlTilingKey::Group16);
}

extern "C" DEEPEP_EXPORT uint32_t deepep_ll_core_protocol_version()
{
    return DispatchDedup::kLlCoreProtocolVersion;
}
extern "C" DEEPEP_EXPORT uint32_t deepep_ll_rdma_enabled()
{
#ifdef DEEPEP_LL_RDMA
    return 1U;
#else
    return 0U;
#endif
}
