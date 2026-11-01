#include <cassert>
#include <thread>
#include <atomic>
#include <iostream>
#include <chrono>
#include <algorithm>
#define DEEPEP_TEST_METADATA_COPY 1
#include "kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h"
#include "kernels/combine/low_latency/combine_stage_wire.h"
using namespace AscendC;
using namespace DispatchDedup;
using namespace std::chrono_literals;
#ifndef DEEPEP_TEST_AIV
#define DEEPEP_TEST_AIV 16
#endif
constexpr uint32_t P=32,E=32,H=256,K=2,C=DEEPEP_TEST_AIV;
constexpr uint32_t R=(C-kLlMetadataCores)/2;
static uint8_t* observedPeer;
static uint32_t observedBytes, relayCopies;
static void observeRelay(void* dst,const void*,uint32_t bytes) {
    if(static_cast<uint8_t*>(dst)>=observedPeer && static_cast<uint8_t*>(dst)<observedPeer+(4U<<20) && bytes==observedBytes)++relayCopies;
}

static void stage_test() {
    for(uint32_t bytes: {2U,32U,478U,480U,512U,960U,1024U,7168U,16384U}) {
        const uint32_t size=((bytes+479)/480)*512;
        std::vector<uint16_t> input((bytes+31)/32*16,0x3f80), output(input.size());
        std::vector<uint8_t> gm(size+32,0x77);
        std::vector<float> zeros((bytes+479)/480*8,0);
        CommArgs args{};args.rank=0;args.rankSize=P;args.peerMemBytes=size;args.peerMems[0]=gm.data();
        CombineDedup::StoreStageWire(GlobalTensor<uint16_t>{reinterpret_cast<uint16_t*>(gm.data())},
            LocalTensor<uint16_t>{input.data()},LocalTensor<float>{zeros.data()},bytes);
        CombineDedup::LoadStageWire(&args,0,0,LocalTensor<uint16_t>{output.data()},bytes);
        assert(std::memcmp(input.data(),output.data(),bytes)==0);
        for(uint32_t block=0;block<size;block+=512)
            for(uint32_t i=480;i<512;++i)assert(gm[block+i]==0);
        for(uint32_t i=size;i<size+32;++i)assert(gm[i]==0x77);
    }
}
struct Fixture {
    std::vector<uint8_t> gm=std::vector<uint8_t>(4U<<20), peer=gm;
    CommArgs args{};
    Fixture(){args.rank=0;args.rankSize=P;args.peerMemBytes=gm.size();args.peerMems[0]=gm.data();args.peerMems[1]=peer.data();}
    void combineStage(uint32_t bs,uint32_t hidden){
        auto d=DispatchServerDedupDataLayout(bs,hidden,K,E,false);
        std::vector<uint16_t> input((hidden*2+479)/480*240,0);
        for(uint32_t i=1;i<input.size();i+=2)input[i]=0x3f80; // float-ready bit patterns
        std::vector<float> zero(d.returnStride/512*8,0);
        for(uint32_t slot=0;slot<E*bs;++slot)
            CombineDedup::StoreStageWire(GlobalTensor<uint16_t>{reinterpret_cast<uint16_t*>(gm.data()+d.dispatchOffset+slot*d.returnStride)},
                LocalTensor<uint16_t>{input.data()},LocalTensor<float>{zero.data()},hidden*2);
    }
    template<bool Fp8=false,bool Dedup=true>
    void run(uint32_t bs,uint32_t sources,bool delayedPrefix,bool outputFirst) {
        constexpr uint32_t hiddenBytes=Fp8?H:H*2;
        auto d=DispatchServerDedupDataLayout(bs,H,K,E,Fp8?EpServerDedupInputMode::PrequantizedFp8Packs:EpServerDedupInputMode::Plain16);
        auto l=DispatchServerDedupCountLayout(E,C);
        uint32_t n=sources*bs;
        observedPeer=peer.data();observedBytes=d.tokenStride;relayCopies=0;observeStore=observeRelay;
        std::vector<int32_t> prefix(E),slots;
        for(uint32_t s=0;s<P;++s){uint32_t count=s>=16 && s<16+sources?bs:0;prefix[s]=(s?prefix[s-1]:0)+count;for(uint32_t q=0;q<count;++q)slots.push_back(s*bs+q);}
        std::vector<uint8_t> out(n*hiddenBytes,0),scale(n*d.scaleBytes+32);
        std::vector<uint32_t> type(n),dest(n*3),relay(n*K*2);
        std::vector<uint64_t> trace(C*LlReferenceProfile::kWordsPerCore,0);
        auto packet=[&](uint32_t index){
            const uint32_t slot=slots[index],source=slot/bs;
            std::vector<uint8_t> wire(d.tokenStride,0),compact(d.packedPayloadBytes,0);
            std::fill_n(compact.begin(),hiddenBytes,uint8_t(index+1));
            if(Fp8)std::fill_n(compact.begin()+d.quantCount,d.scaleBytes,0x73);
            uint32_t header[]={source,slot%bs,0,uint32_t(Dedup?EpServerDedupTokenType::FirstHop:EpServerDedupTokenType::Direct),0,0,0,0};
            std::memcpy(compact.data()+d.headerPayloadOffset,header,32);
            uint32_t read[]={1,0,1U|((slot%bs)<<16),0};
            std::memcpy(compact.data()+d.readIndexPayloadOffset,read,16);
            for(uint32_t b=0;b<d.blockCount;++b){std::memcpy(wire.data()+b*512,compact.data()+b*480,480);for(uint32_t f=0;f<8;++f){float v=b==0 && f==0?1.5f:1.0f;std::memcpy(wire.data()+b*512+480+f*4,&v,4);}}
            std::lock_guard<std::mutex> lock(memoryMutex);
            std::memcpy(gm.data()+d.dispatchOffset+slot*d.tokenStride,wire.data(),wire.size());
        };
        auto publish=[&]{
            std::lock_guard<std::mutex> lock(memoryMutex);
            std::memcpy(gm.data()+l.cntOffset+l.prefixOffset,prefix.data(),E*4);
            if(n)std::memcpy(gm.data()+d.relayOffset,slots.data(),n*4);
            for(uint32_t c=0;c<R;++c){uint32_t info[8]={BalancedBegin(n,c,R),BalancedBegin(n,c+1,R)};std::memcpy(gm.data()+l.cntOffset+l.outputInfoOffset+c*32,info,32);float ready[8]={1,1,1,1,1,1,1,1};std::memcpy(gm.data()+l.cntOffset+l.outputReadyOffset+c*32,ready,32);}
        };
        std::atomic<bool> release{!outputFirst};
        const uint32_t owner=n?ForwardOwner(slots[0]/bs,slots[0]%bs,R):0U, outputOwner=R-1; // single-row case
        if(outputFirst)assert(n==1 && owner!=outputOwner);
        if(n)packet(0);
        if(!delayedPrefix)publish();
        std::vector<std::thread> workers;
        for(uint32_t c=0;c<R;++c)workers.emplace_back([&,c]{
            cores=C;core=ReceiveBegin(C)+c;
            observeStore=observeRelay;
            if(c==owner)while(!release.load())std::this_thread::yield();
            TPipe pipe;ServerDedupReceiver<uint16_t,Fp8,false,Fp8> receiver;
            receiver.Init(&args,out.data(),scale.data(),reinterpret_cast<uint8_t*>(type.data()),reinterpret_cast<uint8_t*>(dest.data()),reinterpret_cast<uint8_t*>(relay.data()),bs,H,K,E,&pipe,0,Dedup);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
            LlReferenceProfile::DeviceProfile profile;profile.Init(reinterpret_cast<uint8_t*>(trace.data()));
            receiver.Process(&profile);profile.Mark<21>();receiver.ExportPacketTrace(&profile);profile.Finish(0,P,bs);
            auto record=trace.data()+core*LlReferenceProfile::kWordsPerCore;
            assert(record[37]==BalancedBegin(n,c+1,R)-BalancedBegin(n,c,R));
            assert(record[99]==record[37] && record[100]==record[37] && record[62]==kLlMetadataCores && record[63]==1);
            for(uint32_t j=0;j<record[96];++j){uint32_t fields[8];std::memcpy(fields,record+128+j*4,32);
                assert(fields[0]>0 && fields[2]>0 && fields[7]==uint32_t(slots[record[104]+j]));}
#else
            receiver.Process();
#endif
            for(auto &event:events)for(bool live:event)assert(!live);
        });
        if(outputFirst){
            bool moved=false;
            for(uint32_t retry=0;retry<2000 && !moved;++retry){
                {std::lock_guard<std::mutex> lock(memoryMutex);moved=out[0]==1;}
                if(!moved)std::this_thread::sleep_for(1ms);
            }
            assert(moved);
            {std::lock_guard<std::mutex> lock(memoryMutex);float flag;std::memcpy(&flag,gm.data()+d.dispatchOffset+slots[0]*d.tokenStride+480,4);assert(flag==1.5f);}
            release=true;
        }
        if(delayedPrefix){std::this_thread::sleep_for(2ms);publish();}
        // Half of the payloads arrive after the prefix. Repeated scans may not relay twice.
        for(uint32_t i=1;i<n;++i){if(i%7==0)std::this_thread::sleep_for(1ms);packet(i);}
        for(auto &w:workers)w.join();
        assert(relayCopies==(Dedup?n:0));observeStore=nullptr;
        for(uint32_t row=0;row<n;++row){
            for(uint32_t j=0;j<hiddenBytes;++j)assert(out[row*hiddenBytes+j]==uint8_t(row+1));
            if(Fp8)for(uint32_t j=0;j<d.scaleBytes;++j)assert(scale[row*d.scaleBytes+j]==0x73);
            assert(type[row]==(Dedup?uint32_t(EpServerDedupTokenType::FirstHop):0U));
            assert(dest[row*3]==slots[row]/bs && dest[row*3+1]==slots[row]%bs);
            uint64_t forwarded=DispatchServerDedupDispatchOffset(d,slots[row]/bs,1,0,slots[row]%bs,P,1,bs);
            if(Dedup){assert(peer[forwarded]==uint8_t(row+1));
                uint32_t forwardedType;std::memcpy(&forwardedType,peer.data()+forwarded+d.headerWireOffset+12,4);assert(forwardedType==uint32_t(EpServerDedupTokenType::Terminal));}
        }
        for(uint32_t pos=0;pos<E*bs;++pos)for(uint32_t b=0;b<d.blockCount;++b)
            for(uint32_t f=0;f<32;++f)assert(gm[d.dispatchOffset+pos*d.tokenStride+b*512+480+f]==0);
        for(uint32_t i=0;i<R*32;++i)assert(gm[l.cntOffset+l.outputReadyOffset+i]==0);
    }
};
int main(){stage_test();Fixture a,b;for(uint32_t cycle=0;cycle<6;++cycle){a.run(1,1,false,true);b.run(2,16,true,false);a.combineStage(1,384);a.run(3,7,false,false);b.run(1,1,true,false);b.run(2,0,false,false);a.run<true>(2,8,true,false);a.run<false,false>(1,5,false,false);}std::cout<<"PASS stage wire and 42 concurrent receiver reuse cases\n";}
