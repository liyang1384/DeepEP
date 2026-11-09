"""Contracts for sender batch ordering; no CPU C++ model."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def test_sender_drains_before_metadata_and_preserves_no_comm():
    source = (ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h').read_text(encoding='utf-8')
    assert source.index('rdmaBatch_.Flush') < source.index('profile->Mark<40>')
    assert '#ifndef DEEPEP_LL_NO_COMM\n        if (LlTransport::RemoteWrite' in source
    assert 'rdmaBatch_.Enqueue' in source

def test_sender_defers_only_with_distinct_lanes_and_posts_before_next_issue():
    code=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h').read_text(encoding='utf-8')
    packet=code[code.index('__aicore__ inline void SendPacket('):]
    assert packet.index('WaitFlag<HardEvent::V_MTE3>') < packet.index('rdmaBatch_.SubmitPending()') < packet.index('rdmaBatch_.Enqueue(')
    assert 'out, lanes_ > 1U' in packet
    assert code.index('rdmaBatch_.SubmitPending()') < code.index('profile->Mark<39>')
