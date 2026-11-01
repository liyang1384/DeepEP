#pragma once
#include "kernel_operator.h"
#include "ll_transport.h"
#include "dispatch_server_dedup_receive_packet.h"
#include "../common/ll_reference_profile_device.h"
#include "../../../common/ep_packet_receive_layout.h"
#include "adv_api/reduce/sum.h"
namespace DispatchDedup {
using namespace AscendC;

// Two independent owners: (region+q) modulo R for inspection/relay,
// balanced output-row intervals for output. Only output owner clears flags,
// after its own load and the inspector's 1.25 publication. Never spin on a
// single handoff: both work streams keep making progress in each outer turn.
template<class XType,bool WireFp8,bool E4M3,bool Prequantized=false>
class ServerDedupReceiver {
    static constexpr uint32_t kBatch=16U, kMoved=0x80000000U;
public:
    __aicore__ inline void Init(const __gm__ CommArgs* args, GM_ADDR expandX, GM_ADDR scales,
            GM_ADDR types, GM_ADDR destinations, GM_ADDR relay, uint32_t bs,uint32_t h,uint32_t k,
            uint32_t experts,TPipe* pipe,int64_t magic=0, bool dedup=true) {
        args_=args;pipe_=pipe;bs_=bs;k_=k;experts_=experts;dedup_=dedup;
        receivers_=PayloadCores(GetBlockNum());receiver_=GetBlockIdx()-ReceiveBegin(GetBlockNum());
        localExperts_=experts/uint32_t(args->rankSize);
        hiddenBytes_=WireFp8?h:h*sizeof(XType);
        constexpr auto mode=Prequantized?EpServerDedupInputMode::PrequantizedFp8Packs:
            (WireFp8?EpServerDedupInputMode::Quantize16:EpServerDedupInputMode::Plain16);
        data_=DispatchServerDedupDataLayout(bs,h,k,experts,mode);
        window_=DispatchServerDedupWindowOffset(magic);
        slots_=experts*bs;owned_=ForwardSlots(experts,bs,receiver_,receivers_);forwardLeft_=owned_;
        forwardQ_=ForwardFirstQ(0U,receiver_,receivers_);
        taskCapacity_=uint32_t((uint64_t(args->rankSize)*bs*(k<localExperts_?k:localExperts_)+receivers_-1)/receivers_);
        x_.SetGlobalBuffer(expandX);scales_.SetGlobalBuffer(scales);
        types_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(types));
        destinations_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(destinations));
        relay_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(relay));
    }
    __aicore__ inline void Process(LlReferenceProfile::DeviceProfile* profile=nullptr) {
        profile_=profile;
        const uint32_t prefixBytes=Align(experts_*4), bitmapBytes=Align(((owned_+31U)/32U)*4U);
        const uint32_t taskBytes=Align(taskCapacity_*4U), blocks=uint32_t(data_.blockCount);
        const uint32_t flagBytes=kBatch*blocks*32U+256U;
        const uint32_t workBytes=kBatch*Align((blocks+7U)/8U)*4U+256U;
        pipe_->InitBuffer(prefixBuf_,prefixBytes);pipe_->InitBuffer(seenBuf_,bitmapBytes?bitmapBytes:32U);
        pipe_->InitBuffer(positionBuf_,taskBytes);pipe_->InitBuffer(pendingBuf_,taskBytes);
        pipe_->InitBuffer(noticeBuf_,64U);pipe_->InitBuffer(flagsBuf_,flagBytes);
        pipe_->InitBuffer(sumBuf_,kBatch*32U);pipe_->InitBuffer(workBuf_,workBytes);
        pipe_->InitBuffer(indexBuf_,kBatch*4U);pipe_->InitBuffer(zeroBuf_,blocks*32U);
        const uint32_t fixed=prefixBytes+(bitmapBytes?bitmapBytes:32U)+2U*taskBytes+64U+
            flagBytes+kBatch*32U+workBytes+kBatch*4U+blocks*32U;
        stride_=uint32_t(data_.tokenStride)+64U;
        lanes_=(kEpServerDedupCountUbBytes-fixed)/stride_;if(lanes_>8U)lanes_=8U;
        pipe_->InitBuffer(lanesBuf_,lanes_*stride_);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if(profile_ && profile_->Enabled()) {
            traceEnabled_=true;
            const uint32_t spare=kEpServerDedupCountUbBytes-fixed-lanes_*stride_;
            traceCapacity_=spare/32U;
            if(traceCapacity_>LlReferenceProfile::kPacketTraceCapacity)traceCapacity_=LlReferenceProfile::kPacketTraceCapacity;
            if(traceCapacity_>taskCapacity_)traceCapacity_=taskCapacity_;
            if(traceCapacity_)pipe_->InitBuffer(packetTraceBuf_,traceCapacity_*32U);
        }
#endif
        Duplicate<uint32_t>(seenBuf_.Get<uint32_t>(),0U,(bitmapBytes?bitmapBytes:32U)/4U);
        Duplicate<float>(flagsBuf_.Get<float>(),0.0f,flagBytes/4U);
        Duplicate<float>(zeroBuf_.Get<float>(),0.0f,blocks*8U);
        ServerDedupReceiveSync<HardEvent::V_S>();
        for(uint32_t lane=0;lane<lanes_;++lane)SetFlag<HardEvent::MTE3_MTE2>(lane);
        if(profile_)profile_->Mark<32>();  // w32: receive_buffers_ready
        while(!prefixReady_ || forwardLeft_ || pending_) {
            ForwardBatch();
            if(!prefixReady_) AcquirePrefix();
            if(prefixReady_ && pending_) OutputBatch();
        }
        for(uint32_t lane=0;lane<lanes_;++lane)WaitFlag<HardEvent::MTE3_MTE2>(lane);
        if(profile_)profile_->Mark<35>();  // w35: receive_lane_wait_clear_issue_end
        PipeBarrier<PIPE_ALL>();
        if(profile_) {
            profile_->Mark<36>();  // w36: receive_final_PIPE_ALL_end
            profile_->Value<37>(outputCount_);  // w37: receive_task_count
            profile_->Value<33>(pollTicks_);  // w33: receive_packet_poll_accum_cycles
            profile_->Value<34>(consumeTicks_);  // w34: receive_load_consume_submit_accum_cycles
            profile_->Value<41>(batches_);  // w41: receive_probe_batch_count
        }
    }
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    // Output-owner observations only; forwarding ownership is independent.
    // Export after w21, with no live business lane or flag reader left.
    __aicore__ inline void ExportPacketTrace(LlReferenceProfile::DeviceProfile* profile) {
        if(!profile || !profile->Enabled())return;
        profile->Detail(96U,traceCapacity_);profile->Detail(97U,traceCapacity_);
        profile->Detail(98U,outputCount_);profile->Detail(99U,traceReady_);profile->Detail(100U,traceProcessed_);
        profile->Detail(101U,lanes_);profile->Detail(104U,begin_);profile->Detail(105U,bs_);
        profile->Detail(106U,localExperts_);profile->Detail(107U,outputCount_>traceCapacity_?1U:0U);
        profile->Detail(108U,traceBatches_);profile->Detail(102U,profile->Clock());
        if(traceCapacity_) {
            ServerDedupReceiveSync<HardEvent::S_MTE3>();
            DataCopy(LlTransport::Tensor(profile->PacketLogAddress(),true),packetTraceBuf_.Get<uint8_t>(),traceCapacity_*32U);
            ServerDedupReceiveSync<HardEvent::MTE3_S>();
        }
        profile->Detail(103U,profile->Clock());
    }
#endif
private:
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    __aicore__ inline void RecordReady(LocalTensor<uint32_t> snapshot,uint32_t readyMask,uint32_t count) {
        if(!traceEnabled_ || !readyMask)return;
        ++traceBatches_;
        for(uint32_t i=0;i<count;++i) {
            if(!(readyMask&(1U<<i)))continue;
            const uint32_t j=snapshot.GetValue(i);++traceReady_;
            if(j<traceCapacity_) {
                auto row=packetTraceBuf_.Get<uint32_t>()[j*8U];
                row.SetValue(0U,traceReady_);row.SetValue(1U,traceBatches_);
            }
        }
    }
    __aicore__ inline void RecordProcessed(uint32_t j,LocalTensor<uint8_t> wire,uint32_t headerOffset,bool keepFlags) {
        if(traceEnabled_)++traceProcessed_;
        if(traceEnabled_ && j<traceCapacity_) {
            auto row=packetTraceBuf_.Get<uint32_t>()[j*8U];
            auto header=wire[headerOffset].ReinterpretCast<uint32_t>();
            const uint32_t role=header.GetValue(3U);
            const uint32_t members=role==uint32_t(EpServerDedupTokenType::FirstHop)?
                wire[uint32_t(keepFlags?data_.relayReadIndexWireOffset:data_.readIndexPayloadOffset)].ReinterpretCast<uint32_t>().GetValue(0U):0U;
            row.SetValue(2U,traceProcessed_);row.SetValue(3U,header.GetValue(0U));
            row.SetValue(4U,header.GetValue(1U));row.SetValue(5U,header.GetValue(2U));
            row.SetValue(6U,role|(members<<8U));row.SetValue(7U,positionBuf_.Get<uint32_t>().GetValue(j));
        }
    }
    TBuf<> packetTraceBuf_;
    bool traceEnabled_=false;
    uint32_t traceCapacity_=0,traceReady_=0,traceProcessed_=0,traceBatches_=0;
#endif
    __aicore__ inline uint32_t Align(uint32_t n) {return (n+31U)/32U*32U;}
    __aicore__ inline GM_ADDR Slot(uint32_t pos) {return args_->peerMems[args_->rank]+window_+data_.dispatchOffset+uint64_t(pos)*data_.tokenStride;}
    __aicore__ inline bool Seen(uint32_t j) {return (seenBuf_.Get<uint32_t>().GetValue(j/32)&(1U<<(j%32)))!=0;}
    __aicore__ inline void See(uint32_t j) {
        auto bits=seenBuf_.Get<uint32_t>();bits.SetValue(j/32,bits.GetValue(j/32)|(1U<<(j%32)));--forwardLeft_;
    }
    __aicore__ inline bool Valid(uint32_t pos) {
        const uint32_t region=pos/bs_,q=pos%bs_;
        const uint32_t source=(region/localExperts_+uint32_t(args_->rankSize)-uint32_t(args_->rank))%uint32_t(args_->rankSize);
        const uint32_t record=(region%localExperts_)*uint32_t(args_->rankSize)+source;
        auto prefix=prefixBuf_.Get<int32_t>();
        const uint32_t previous=record?uint32_t(prefix.GetValue(record-1)):0;
        return q<uint32_t(prefix.GetValue(record))-previous;
    }
    __aicore__ inline void AcquirePrefix() {
        const auto layout=DispatchServerDedupCountLayout(experts_,GetBlockNum());
        auto base=args_->peerMems[args_->rank]+window_+layout.cntOffset;
        auto notice=noticeBuf_.Get<uint8_t>();
        DataCopy(notice,LlTransport::Tensor(base+layout.outputReadyOffset+receiver_*32U,true),32U);
        ServerDedupReceiveSync<HardEvent::MTE2_S>();
        if(notice.ReinterpretCast<float>().GetValue(0)!=1.0f)return;
        // This source stays zero and unused until final drain. Descriptor DMA
        // uses the other 32B half, so it cannot race the flag clear DMA.
        Duplicate<uint32_t>(notice.ReinterpretCast<uint32_t>(),0U,8U);
        ServerDedupReceiveSync<HardEvent::V_MTE3>();
        DataCopy(LlTransport::Tensor(base+layout.outputReadyOffset+receiver_*32U,true),notice,32U);
        ServerDedupReceiveSync<HardEvent::S_MTE2>();
        DataCopy(prefixBuf_.Get<uint8_t>(),LlTransport::Tensor(base+layout.prefixOffset,true),experts_*4U);
        DataCopy(notice[32U],LlTransport::Tensor(base+layout.outputInfoOffset+receiver_*32U,true),32U);
        ServerDedupReceiveSync<HardEvent::MTE2_S>();
        auto info=notice[32U].ReinterpretCast<uint32_t>();begin_=info.GetValue(0);outputCount_=info.GetValue(1)-begin_;pending_=outputCount_;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if(traceCapacity_>outputCount_)traceCapacity_=outputCount_;
#endif
        if(profile_)profile_->Mark<30>();  // w30: receive_prefix_loaded_after_MTE2_S
        if(pending_) {
            DataCopyPad(positionBuf_.Get<uint8_t>(),LlTransport::Tensor(args_->peerMems[args_->rank]+window_+data_.relayOffset+uint64_t(begin_)*4U,true),
                {1U,pending_*4U,0U,0U,0U},{false,0U,0U,uint8_t(0)});
            CreateVecIndex(pendingBuf_.Get<int32_t>(),int32_t(0),pending_);
            ServerDedupReceiveSync<HardEvent::MTE2_S>();ServerDedupReceiveSync<HardEvent::V_S>();
        }
        prefixReady_=true;if(profile_)profile_->Mark<31>();  // w31: receive_tasks_built
    }
    __aicore__ inline void Probe(uint32_t count) {
        ServerDedupReceiveSync<HardEvent::MTE2_V>();
        Sum<float>(sumBuf_.Get<float>(),flagsBuf_.Get<float>(),workBuf_.Get<uint8_t>(),
                   SumParams{count,uint32_t(data_.blockCount)*8U,uint32_t(data_.blockCount)*8U});
        ServerDedupReceiveSync<HardEvent::V_S>();
    }
    __aicore__ inline uint32_t Lane() {
        uint32_t lane=sequence_++%lanes_;WaitFlag<HardEvent::MTE3_MTE2>(lane);return lane;
    }
    __aicore__ inline void Release(uint32_t lane) {SetFlag<HardEvent::MTE3_MTE2>(lane);}
    __aicore__ inline void Inspect(uint32_t pos) {
        const uint32_t lane=Lane();auto wire=lanesBuf_.Get<uint8_t>()[lane*stride_];
        // Ordinary/terminal packets need only a header inspection, not another full payload read.
        DataCopy(wire[uint32_t(data_.headerWireOffset)],LlTransport::Tensor(Slot(pos)+data_.headerWireOffset,true),32U);
        ServerDedupReceiveSync<HardEvent::MTE2_S>();
        auto header=wire[uint32_t(data_.headerWireOffset)].ReinterpretCast<uint32_t>();
        if(dedup_ && header.GetValue(3U)==uint32_t(EpServerDedupTokenType::FirstHop)) {
            DataCopy(wire,LlTransport::Tensor(Slot(pos),true),uint32_t(data_.tokenStride));
            ServerDedupReceiveSync<HardEvent::MTE2_S>();
            ServerDedupPacketTail tail(wire,uint32_t(data_.headerWireOffset),uint32_t(data_.relayReadIndexWireOffset));
            PrepareServerDedupForward(tail);
            wire.ReinterpretCast<float>().SetValue(PacketReceiveLayout::kMarkerOffset/4U,PacketReceiveLayout::kCompact);
            ServerDedupReceiveSync<HardEvent::S_MTE3>();
            const uint32_t source=tail.Header().GetValue(0),members=tail.ReadIndex().GetValue(0);
            for(uint32_t i=0;i<members;++i) {
                uint32_t packed=tail.Entry(i).GetValue(0),expert=packed&0xffffU,q=packed>>16U,dst=expert/localExperts_;
                LlTransport::Packet(args_,dst,window_+DispatchServerDedupDispatchOffset(data_,source,dst,expert%localExperts_,q,
                    uint32_t(args_->rankSize),localExperts_,bs_),wire,uint32_t(data_.tokenStride),lane);
            }
        }
        // Separate from wire: relay DMA may still read its original marker.
        auto marker=wire[uint32_t(data_.tokenStride)+32U].ReinterpretCast<float>();
        marker.SetValue(0,PacketReceiveLayout::kTaken);ServerDedupReceiveSync<HardEvent::S_MTE3>();
        DataCopyPad(LlTransport::Tensor(Slot(pos)+PacketReceiveLayout::kMarkerOffset,true),marker.ReinterpretCast<uint8_t>(),{1U,4U,0U,0U,0U});
        Release(lane);
    }
    __aicore__ inline void ForwardBatch() {
        if(!forwardLeft_)return;
        StartPoll();
        uint32_t count=0;auto ids=indexBuf_.Get<uint32_t>();
        for(uint32_t visit=0;visit<kBatch && visit<owned_;++visit) {
            if(forwardCursor_==owned_) {
                forwardCursor_=0;forwardRegion_=0;forwardQ_=ForwardFirstQ(0U,receiver_,receivers_);
            }
            while(forwardQ_>=bs_) {
                ++forwardRegion_;forwardQ_=ForwardFirstQ(forwardRegion_,receiver_,receivers_);
            }
            const uint32_t j=forwardCursor_++,pos=forwardRegion_*bs_+forwardQ_;
            forwardQ_+=receivers_;
            if(Seen(j))continue;
            if(prefixReady_ && !Valid(pos)){See(j);continue;}
            ids.SetValue(count,pos);ReadServerDedupFlags(Slot(pos),uint32_t(data_.blockCount),flagsBuf_.Get<float>()[count*data_.blockCount*8U],true);++count;
        }
        if(!count)return;
        Probe(count);
        uint32_t readyMask=0U;
        for(uint32_t i=0;i<count;++i)
            if(PacketReceiveLayout::Decode(sumBuf_.Get<float>().GetValue(i),uint32_t(data_.blockCount))!=PacketReceiveLayout::State::Pending)
                readyMask|=1U<<i;
        if(!readyMask)return;
        const uint64_t ready=ReadyPoll();
        for(uint32_t i=0;i<count;++i) {
            if(!(readyMask&(1U<<i)))continue;
            auto state=PacketReceiveLayout::Decode(sumBuf_.Get<float>().GetValue(i),uint32_t(data_.blockCount));
            const uint32_t pos=ids.GetValue(i);
            See(ForwardIndex(pos/bs_,pos%bs_,bs_,receiver_,receivers_));
            if(state!=PacketReceiveLayout::State::Taken)Inspect(pos);
        }
        FinishBatch(ready);
    }
    __aicore__ inline void Move(uint32_t j,uint32_t pos) {
        uint32_t lane=Lane();auto wire=lanesBuf_.Get<uint8_t>()[lane*stride_];
        DataCopyPad(wire,LlTransport::Tensor(Slot(pos),true),{uint16_t(data_.blockCount),480U,32U,0U,0U},{false,0U,0U,uint8_t(0)});
        ServerDedupReceiveSync<HardEvent::MTE2_S>();
        const uint32_t row=begin_+j;
        auto type=wire[uint32_t(data_.tokenStride)].ReinterpretCast<uint32_t>();
        auto dest=wire[uint32_t(data_.tokenStride)+32U].ReinterpretCast<uint32_t>();
        ServerDedupPacketTail tail(wire,uint32_t(data_.headerPayloadOffset),uint32_t(data_.readIndexPayloadOffset));
        const uint32_t role=BuildServerDedupMetadata(type,dest,tail,pos);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        RecordProcessed(j,wire,uint32_t(data_.headerPayloadOffset),false);
#endif
        ServerDedupReceiveSync<HardEvent::S_MTE3>();
        DataCopyPad(x_[uint64_t(row)*hiddenBytes_],wire,{1U,hiddenBytes_,0U,0U,0U});
        if constexpr(WireFp8)DataCopyPad(scales_[uint64_t(row)*data_.scaleBytes],wire[uint32_t(data_.quantCount)],{1U,uint32_t(data_.scaleBytes),0U,0U,0U});
        if(dedup_)DataCopyPad(types_[row],type,{1U,4U,0U,0U,0U});
        DataCopyPad(destinations_[uint64_t(row)*3U],dest,{1U,12U,0U,0U,0U});
        if(dedup_ && role==uint32_t(EpServerDedupTokenType::FirstHop))
            DataCopyPad(relay_[uint64_t(row)*k_*2U],tail.ReadIndex(),{1U,k_*8U,0U,0U,0U});
        Release(lane);
    }
    __aicore__ inline void OutputBatch() {
        if(outputCursor_>=pending_)outputCursor_=0;
        const uint32_t count=pending_-outputCursor_<kBatch?pending_-outputCursor_:kBatch;
        auto pending=pendingBuf_.Get<uint32_t>();auto pos=positionBuf_.Get<uint32_t>();
        StartPoll();
        for(uint32_t i=0;i<count;++i) {
            uint32_t j=pending.GetValue(outputCursor_+i)&~kMoved;
            ReadServerDedupFlags(Slot(pos.GetValue(j)),uint32_t(data_.blockCount),flagsBuf_.Get<float>()[i*data_.blockCount*8U],true);
        }
        Probe(count);
        uint32_t readyMask=0U;
        for(uint32_t i=0;i<count;++i) {
            const auto state=PacketReceiveLayout::Decode(sumBuf_.Get<float>().GetValue(i),uint32_t(data_.blockCount));
            if(state!=PacketReceiveLayout::State::Pending &&
                (!(pending.GetValue(outputCursor_+i)&kMoved) || state==PacketReceiveLayout::State::Taken))readyMask|=1U<<i;
        }
        if(!readyMask){outputCursor_+=count;return;}
        const uint64_t ready=ReadyPoll();
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        uint32_t newlyReady=0U;
        for(uint32_t i=0;i<count;++i) {
            if((readyMask&(1U<<i)) && !(pending.GetValue(outputCursor_+i)&kMoved))
                newlyReady|=1U<<i;
        }
        RecordReady(pending[outputCursor_],newlyReady,count);
#endif
        for(uint32_t i=count;i>0;--i) {
            if(!(readyMask&(1U<<(i-1))))continue;
            const uint32_t at=outputCursor_+i-1,tag=pending.GetValue(at),j=tag&~kMoved;
            auto state=PacketReceiveLayout::Decode(sumBuf_.Get<float>().GetValue(i-1),uint32_t(data_.blockCount));
            if(!(tag&kMoved)){Move(j,pos.GetValue(j));pending.SetValue(at,j|kMoved);}
            if(state==PacketReceiveLayout::State::Taken) {
                DataCopyPad(LlTransport::Tensor(Slot(pos.GetValue(j))+480U,true),zeroBuf_.Get<uint8_t>(),
                    {uint16_t(data_.blockCount),32U,0U,480U,0U});
                --pending_;pending.SetValue(at,pending.GetValue(pending_));
            }
        }
        outputCursor_+=count;
        FinishBatch(ready);
    }
    // One start per waiting interval, then two clocks only when a batch can
    // inspect, move or clear. Empty retries and unacquired moved packets do not
    // repeatedly read clocks; the wait spans alternating scans/prefix work.
    __aicore__ inline void StartPoll() {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if(profile_ && !pollStarted_){pollStart_=profile_->Clock();pollStarted_=true;}
#endif
    }
    __aicore__ inline uint64_t ReadyPoll() {
        ++batches_;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if(profile_) {
            const uint64_t now=profile_->Clock();pollTicks_+=now-pollStart_;pollStarted_=false;
            if(!firstReady_){firstReady_=true;profile_->Value<42>(now);}  // w42: first_packet_ready
            return now;
        }
#endif
        return 0U;
    }
    __aicore__ inline void FinishBatch(uint64_t ready) {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
        if(profile_)consumeTicks_+=profile_->Clock()-ready;
#endif
    }
    const __gm__ CommArgs* args_=nullptr;TPipe* pipe_=nullptr;LlReferenceProfile::DeviceProfile* profile_=nullptr;
    EpServerDedupDataLayout data_{};uint64_t window_=0;
    uint32_t receivers_=0,receiver_=0,bs_=0,k_=0,experts_=0,localExperts_=0,hiddenBytes_=0;
    uint32_t slots_=0,owned_=0,forwardLeft_=0,forwardCursor_=0,outputCursor_=0,sequence_=0;
    uint32_t forwardRegion_=0,forwardQ_=0;
    uint32_t taskCapacity_=0,begin_=0,outputCount_=0,pending_=0,lanes_=0,stride_=0,batches_=0;
    bool prefixReady_=false,dedup_=false,firstReady_=false;
    uint64_t pollTicks_=0,consumeTicks_=0;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    uint64_t pollStart_=0;
    bool pollStarted_=false;
#endif
    TBuf<> prefixBuf_,seenBuf_,positionBuf_,pendingBuf_,noticeBuf_,flagsBuf_,sumBuf_,workBuf_,indexBuf_,zeroBuf_,lanesBuf_;
    GlobalTensor<uint8_t> x_,scales_;GlobalTensor<uint32_t> types_,destinations_,relay_;
};
}
