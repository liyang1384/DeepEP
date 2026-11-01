#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <cstdio>
#include <mutex>
#include <chrono>
#include <vector>
#include <atomic>
#define __aicore__
#define __gm__
#define __ubuf__
constexpr int PIPE_ALL=0;
namespace AscendC {
enum class HardEvent { S_MTE3, MTE3_S, MTE2_S, V_S, S_MTE2, MTE2_V, MTE3_V, V_MTE3, MTE3_MTE2, S_V, V_MTE2, MTE2_MTE3 };
inline thread_local bool events[12][8]{};
inline std::mutex memoryMutex;
#ifdef DEEPEP_TEST_METADATA_COPY
inline thread_local void (*observeStore)(void*,const void*,uint32_t)=nullptr;
inline thread_local void (*observeMte3Completion)()=nullptr;
inline thread_local void (*observeLoad)(const void*,uint32_t)=nullptr;
#endif
#ifdef DEEPEP_TEST_ROUTE_THREADS
struct DeferredMte2Copy { void* dst;const void* src;uint32_t bytes; };
inline thread_local std::vector<DeferredMte2Copy> deferredMte2;
#endif
#ifdef DEEPEP_TEST_DEFERRED_MTE3
struct DeferredMte3Copy { void* dst; const void* src; uint32_t bytes; uint64_t serial; };
inline thread_local std::vector<DeferredMte3Copy> deferredMte3;
inline thread_local uint64_t issuedMte3=0, mte3Fence[12][8]{};
#endif
template<HardEvent E> void SetFlag(uint32_t id) {
    if(id>=8 || events[int(E)][id]) throw std::runtime_error("event reused while live");
    events[int(E)][id]=true;
#ifdef DEEPEP_TEST_DEFERRED_MTE3
    if constexpr(E==HardEvent::MTE3_S || E==HardEvent::MTE3_V || E==HardEvent::MTE3_MTE2) mte3Fence[int(E)][id]=issuedMte3;
#endif
}
template<HardEvent E> void WaitFlag(uint32_t id) {
    if(id>=8 || !events[int(E)][id]) throw std::runtime_error("wait without event");
    events[int(E)][id]=false;
#ifdef DEEPEP_TEST_METADATA_COPY
    if constexpr(E==HardEvent::MTE3_S || E==HardEvent::MTE3_V || E==HardEvent::MTE3_MTE2)
        if(observeMte3Completion)observeMte3Completion();
#endif
#ifdef DEEPEP_TEST_DEFERRED_MTE3
    if constexpr(E==HardEvent::MTE3_S || E==HardEvent::MTE3_V || E==HardEvent::MTE3_MTE2) {
        std::lock_guard<std::mutex> guard(memoryMutex);
        while(!deferredMte3.empty() && deferredMte3.front().serial<=mte3Fence[int(E)][id]) {
            auto copy=deferredMte3.front();std::memcpy(copy.dst,copy.src,copy.bytes);
            deferredMte3.erase(deferredMte3.begin());
        }
    }
#endif
#ifdef DEEPEP_TEST_ROUTE_THREADS
    if constexpr(E==HardEvent::MTE2_S || E==HardEvent::MTE2_V) {
        std::lock_guard<std::mutex> guard(memoryMutex);
        for(const auto& copy:deferredMte2) std::memcpy(copy.dst,copy.src,copy.bytes);
        deferredMte2.clear();
    }
#endif
}
using std::printf;
inline uint64_t cycles=0;
inline uint64_t GetSystemCycle() {
#ifdef DEEPEP_TEST_ROUTE_THREADS
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
#else
    return cycles;
#endif
}
enum class CacheMode { CACHE_MODE_DISABLE };
inline thread_local uint32_t core=0, cores=16;
inline uint32_t GetBlockIdx() { return core; }
inline uint32_t GetBlockNum() { return cores; }
inline void Trap() { throw std::runtime_error("device trap"); }
template<int P> inline void PipeBarrier() {
#ifdef DEEPEP_TEST_METADATA_COPY
    if constexpr(P==PIPE_ALL)if(observeMte3Completion)observeMte3Completion();
#endif
}
template<class T> struct LocalTensor {
    T* p;
    T GetValue(uint32_t index) const { return p[index]; }
    T operator()(uint32_t index) const { return p[index]; }
    void SetValue(uint32_t index,T value) { p[index]=value; }
    LocalTensor<T> operator[](uint32_t index) { return {p + index}; }
    T* GetPhyAddr() { return p; }
    template<class U> LocalTensor<U> ReinterpretCast() { return {reinterpret_cast<U*>(p)}; }
};
template<class T> struct GlobalTensor {
    T* p=nullptr;
    GlobalTensor<T> operator[](uint64_t index) { return {p + index}; }
    T& operator()(uint32_t index) { return p[index]; }
    void SetGlobalBuffer(T* q) { p=q; }
    void SetL2CacheHint(CacheMode) {}
    T GetValue(uint32_t i) { std::lock_guard<std::mutex> guard(memoryMutex); return p[i]; }
    void SetValue(uint32_t i, T value) { std::lock_guard<std::mutex> guard(memoryMutex); p[i]=value; }
};
enum class CacheLine { SINGLE_CACHE_LINE };
enum class DcciDst { CACHELINE_OUT };
enum class MemDsbT { DDR };
template<class T,CacheLine,DcciDst> void DataCacheCleanAndInvalid(GlobalTensor<T>) {}
template<MemDsbT> void DataSyncBarrier() {}
struct DataCopyExtParams { uint16_t count; uint32_t bytes, srcStride, dstStride, reserved; };
struct DataCopyParams { uint16_t blockCount, blockLen, srcStride, dstStride; };
inline std::atomic<uint32_t> stridedCopies{0};
template<class T> void DataCopy(LocalTensor<T> dst, GlobalTensor<T> src, DataCopyParams c) {
    std::lock_guard<std::mutex> guard(memoryMutex);
#ifdef DEEPEP_TEST_METADATA_COPY
    if(observeLoad)observeLoad(src.p,c.blockCount*c.blockLen*32U);
#endif
    ++stridedCopies;
    for (uint32_t i=0; i<c.blockCount; ++i)
        std::memcpy(reinterpret_cast<uint8_t*>(dst.p) + i*(c.blockLen+c.dstStride)*32U,
                    reinterpret_cast<const uint8_t*>(src.p) + i*(c.blockLen+c.srcStride)*32U,
                    c.blockLen*32U);
}
template<class T> struct DataCopyPadExtParams { bool pad; uint8_t left, right; T value; };
template<class T> void DataCopy(GlobalTensor<T> d, LocalTensor<T> s, uint32_t n) {
#ifdef DEEPEP_TEST_DEFERRED_MTE3
    deferredMte3.push_back({d.p,s.p,uint32_t(n*sizeof(T)),++issuedMte3});
#else
    std::lock_guard<std::mutex> guard(memoryMutex);
#ifdef DEEPEP_TEST_METADATA_COPY
    if(observeStore)observeStore(d.p,s.p,n*sizeof(T));
#endif
    std::memcpy(d.p,s.p,n*sizeof(T));
#endif
}
template<class T> void DataCopy(LocalTensor<T> d, GlobalTensor<T> s, uint32_t n) {
#ifdef DEEPEP_TEST_METADATA_COPY
    if(observeLoad)observeLoad(s.p,n*sizeof(T));
#endif
#ifdef DEEPEP_TEST_ROUTE_THREADS
    deferredMte2.push_back({d.p,s.p,uint32_t(n*sizeof(T))});
#else
    std::lock_guard<std::mutex> guard(memoryMutex);std::memcpy(d.p,s.p,n*sizeof(T));
#endif
}
template<class T> void DataCopyPad(GlobalTensor<T> d, LocalTensor<T> s, DataCopyExtParams c) {
    std::lock_guard<std::mutex> guard(memoryMutex);
    for (uint32_t i=0;i<c.count;++i) {
        auto dst=reinterpret_cast<uint8_t*>(d.p)+i*(c.bytes+c.dstStride);
        auto src=reinterpret_cast<uint8_t*>(s.p)+i*((c.bytes+31U)/32U*32U+c.srcStride*32U);
#ifdef DEEPEP_TEST_METADATA_COPY
        if(observeStore)observeStore(dst,src,c.bytes);
#endif
        std::memcpy(dst,src,c.bytes);
    }
}
template<class T> void DataCopyPad(LocalTensor<T> d, GlobalTensor<T> s, DataCopyExtParams c, DataCopyPadExtParams<T>) {
    std::lock_guard<std::mutex> guard(memoryMutex);
    for (uint32_t i=0;i<c.count;++i) std::memcpy(reinterpret_cast<uint8_t*>(d.p)+i*((c.bytes+31U)/32U*32U+c.dstStride*32U),
        reinterpret_cast<uint8_t*>(s.p)+i*(c.bytes+c.srcStride),c.bytes);
}
}
