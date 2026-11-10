"""Contracts for count ownership and frame transport wiring."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
COUNTS = ROOT / 'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h'


def block(source, marker):
    start = source.index('{', source.index(marker))
    depth = 1
    for end in range(start + 1, len(source)):
        depth += (source[end] == '{') - (source[end] == '}')
        if not depth:
            return source[start + 1:end]
    raise AssertionError(marker)


def test_remote_protocol_is_wired_consistently():
    source = COUNTS.read_text(encoding='utf-8')
    send = block(source, 'void ServerDedupCounts::SendCnt()')
    assert 'CountAssignment(' in send and 'CountRemotePeer(' in send
    assert 'task += coreNum_' in send
    assert 'CountSlots(remoteFrame, first, moeExpertNumPerRank_);' in send
    assert 'LlTransport::DirectCountFrame' in send
    assert 'auto remoteFrame = workBuf_.Get<int32_t>();' in send
    assert 'moeExpertNumPerRank_, dstExpBuf_.Get<uint8_t>()' in send
    assert 'routeExchange_' not in source and 'publishingSource_' not in source
    assert 'readPlan, directRanks_' in source
    assert 'epRankId_, source, expert, directRanks_' in source
    transport = (COUNTS.parent / 'll_count_frame_transport.h').read_text(encoding='utf-8')
    remote = block(transport, 'void DirectCountFrame(')
    assert remote.count('Put(') == 1 and 'Quiet(' not in remote
    assert remote.index('Sync<HardEvent::MTE3_S>') < remote.index('Put(')
