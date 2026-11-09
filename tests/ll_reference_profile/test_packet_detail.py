"""Packet detail decoding contracts; not hardware timing/performance evidence."""
from pathlib import Path
import pytest
from tests.ll_reference_profile.test_parse import sample, timer, setword
from tests.ll_rdma.test_combine_overhead import block
ROOT=Path(__file__).resolve().parents[2]


def sender_raw():
    saved=sample(offset=10**17)
    w=[v for row in saved['raw'][0] for v in row];base=w[0]
    w[65]=3;w[66]=2;w[67]=base+102;w[68]=base+155
    w[69]=base+110;w[70]=base+150
    w[71]=base+120;w[72]=base+130;w[73]=base+135;w[74]=base+140;w[75]=3
    w[76:80]=[0,1,12,1]
    w[80:90]=[6,1,base+159,1,1,5,2,4,5,8]
    w[90:96]=[base+131,base+132,base+134,3,2,1]
    saved['raw'][0]=[w[i:i+8] for i in range(0,timer.WORD_COUNT,8)]
    return saved


def receiver_raw():
    saved=sample();c=15;w=[v for row in saved['raw'][c] for v in row]
    w[96:102]=[2,2,2,2,2,2];w[107]=0
    # Both found in batch1, scan finds j0 then j1, consumption runs reverse.
    # Position1 decodes source1 with B1/localExperts1/rank0.
    for j,fields in enumerate(([1,1,2,0,0,0,1,0],[2,1,1,1,0,0,2|(3<<8),1])):
        w[128+j*4:132+j*4]=[fields[i]|(fields[i+1]<<32) for i in range(0,8,2)]
    saved['raw'][c]=[w[i:i+8] for i in range(0,timer.WORD_COUNT,8)]
    return saved


def test_second_packet_excludes_member_q_and_batch_quiet():
    d=timer.parse_records(sender_raw()['raw'])['cores'][0]['packet_detail']
    assert d['ticks']==dict(second_schedule_without_q=5,second_member_q=3,second_stage=5,
        second_previous_submit=2,second_defer_gap=1,second_stage_wait=2,
        second_put_call=5,second_enqueue_to_put_return=10,remote_submit_span=53,
        remote_retire_span=57,group_submit_span=40,batch_quiet_total=6)
    assert d['intervals_interleave'] and d['second']['triggered_flush']
    assert d['second']['peer']==1
    assert timer.parse_records(sample()['raw'])['cores'][0]['packet_detail']['second'] is None


def test_stable_identity_and_distinct_discovery_consumption_order():
    saved=receiver_raw();d=timer.parse_records(saved['raw'])['cores'][15]['packet_detail']
    assert [e['ready_order'] for e in d['entries']]==[1,2]
    assert [e['process_order'] for e in d['entries']]==[2,1]
    assert d['entries'][1]['packet_role']=='FIRST_HOP' and d['entries'][1]['forward_members']==3
    assert not d['truncated']
    # Truncation preserves original order numbers instead of inventing complete coverage.
    setword(saved['raw'],15,96,1);setword(saved['raw'],15,97,1);setword(saved['raw'],15,107,1)
    for i in range(132,136):setword(saved['raw'],15,i,0)
    d=timer.parse_records(saved['raw'])['cores'][15]['packet_detail']
    assert d['truncated'] and d['entries'][0]['process_order']==2


@pytest.mark.parametrize('core,index,value',[(0,64,0),(0,75,99),(0,82,1),(0,90,1),(15,107,1),(15,99,1),(15,102,0)])
def test_bad_packet_details_fail_closed(core,index,value):
    saved=sender_raw() if core==0 else receiver_raw()
    setword(saved['raw'],core,index,value)
    with pytest.raises(ValueError):timer.parse_records(saved['raw'])


def test_trace_uses_only_spare_ub_and_exports_after_business():
    recv=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h').read_text(encoding='utf-8')
    kernel=(ROOT/'csrc/kernels/dispatch/low_latency/dispatch_server_dedup_kernel.h').read_text(encoding='utf-8')
    assert 'kEpServerDedupCountUbBytes - fixed - lanes_ * laneStride_' in recv
    allocation=recv[recv.index('traceCapacity_ = spare'):recv.index('auto snapshot =')]
    assert 'lanes_ =' not in allocation
    assert kernel.index('profile_->Mark<21>')<kernel.index('receiver_.ExportPacketTrace')
    export=block(recv,'__aicore__ inline void ExportPacketTrace(')
    assert export.count('DataCopy(')==1


def test_receiver_batch_table_keeps_tuple_alignment_and_unknown_truncated_count():
    from tests.ll_reference_profile.export_xlsx import receiver_batch_rows
    saved=receiver_raw()
    setword(saved['raw'],15,41,1)
    rows=receiver_batch_rows([saved],timer,1e9)
    first=rows[1]
    assert first[:6]==[0,15,1,2,2,True]
    assert first[6:9]==['(DIRECT, FIRST_HOP)','(0, 1)','(0, 0)']
    assert first[9]=='((DIRECT, 0, 0), (FIRST_HOP, 1, 0))'
    assert first[11]=='(2, 1)'
    # A truncated trace can omit an entire successful batch.
    setword(saved['raw'],15,41,2)
    setword(saved['raw'],15,96,1);setword(saved['raw'],15,97,1);setword(saved['raw'],15,107,1)
    for i in range(132,136):setword(saved['raw'],15,i,0)
    rows=receiver_batch_rows([saved],timer,1e9)
    assert rows[1][3] is None and rows[1][4:6]==[1,False]
    assert rows[2][3] is None and rows[2][4]==0
