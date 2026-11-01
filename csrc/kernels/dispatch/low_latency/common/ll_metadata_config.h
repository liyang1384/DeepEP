#pragma once
#include <cstdint>
#ifndef DEEPEP_LL_METADATA_CORES
#define DEEPEP_LL_METADATA_CORES 4
#endif
#if DEEPEP_LL_METADATA_CORES < 2 || DEEPEP_LL_METADATA_CORES > 70 || (DEEPEP_LL_METADATA_CORES % 2)
#error DEEPEP_LL_METADATA_CORES must be an even integer in [2,70]
#endif
namespace DispatchDedup {
constexpr uint32_t kLlMetadataCores = DEEPEP_LL_METADATA_CORES;
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_META_INLINE __aicore__ inline
#else
#define LL_META_INLINE inline
#endif
LL_META_INLINE constexpr bool MetadataSplitValid(uint32_t cores) {
    return cores >= kLlMetadataCores + 2U && cores <= 72U && cores % 2U == 0U;
}
LL_META_INLINE constexpr uint32_t PayloadCores(uint32_t cores) { return (cores-kLlMetadataCores)/2U; }
LL_META_INLINE constexpr uint32_t ReceiveBegin(uint32_t cores) { return kLlMetadataCores+PayloadCores(cores); }
// Physical wire slots stay region*B+q. Rotate each region's q ownership so
// even B and mostly q=0 routes do not strand half of an even receiver group.
LL_META_INLINE constexpr uint32_t ForwardOwner(uint32_t region,uint32_t q,uint32_t cores) {
    return (region+q)%cores;
}
LL_META_INLINE constexpr uint32_t ForwardFirstQ(uint32_t region,uint32_t core,uint32_t cores) {
    return (core+cores-region%cores)%cores;
}
// Number of this core's slots in [0,regions). Full groups of R regions
// contribute B slots; count the intersection of the remaining cyclic arc.
LL_META_INLINE constexpr uint32_t ForwardSlots(uint32_t regions,uint32_t bs,uint32_t core,uint32_t cores) {
    const uint32_t rem=regions%cores,tail=bs%cores,high=core+1U;
    const uint32_t low=high>tail?high-tail:0U;
    const uint32_t upper=rem<high?rem:high,lower=rem<low?rem:low;
    const uint32_t wrap=tail>high?cores-(tail-high):cores;
    return regions*(bs/cores)+(regions/cores)*tail+upper-lower+(rem>wrap?rem-wrap:0U);
}
LL_META_INLINE constexpr uint32_t ForwardIndex(uint32_t region,uint32_t q,uint32_t bs,uint32_t core,uint32_t cores) {
    return ForwardSlots(region,bs,core,cores)+q/cores;
}
// Invert the compact bitmap index. Each full R-region block has exactly B
// entries, so only the at-most-R regions inside that block need searching.
LL_META_INLINE constexpr uint32_t ForwardPosition(uint32_t index,uint32_t bs,uint32_t core,uint32_t cores) {
    uint32_t low=(index/bs)*cores,high=low+cores;
    while(low<high) {
        const uint32_t middle=low+(high-low)/2U;
        if(ForwardSlots(middle+1U,bs,core,cores)<=index)low=middle+1U;
        else high=middle;
    }
    return low*bs+ForwardFirstQ(low,core,cores)+
        (index-ForwardSlots(low,bs,core,cores))*cores;
}
LL_META_INLINE constexpr uint32_t ForwardFirstBit(uint32_t word) {
    uint32_t bit=0;
    if((word&0xffffU)==0U){word>>=16U;bit+=16U;}
    if((word&0xffU)==0U){word>>=8U;bit+=8U;}
    if((word&0xfU)==0U){word>>=4U;bit+=4U;}
    if((word&0x3U)==0U){word>>=2U;bit+=2U;}
    return bit+((word&1U)==0U?1U:0U); // caller supplies a nonzero word
}
LL_META_INLINE constexpr uint32_t BalancedBegin(uint32_t total,uint32_t core,uint32_t cores) {
    return uint32_t(uint64_t(total)*core/cores);
}
#undef LL_META_INLINE
}
