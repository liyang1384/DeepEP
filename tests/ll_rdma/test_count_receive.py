from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def test_read_plan_outside_poll_and_body_after_commit():
    source=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h').read_text(encoding='utf-8')
    start=source.index('void ServerDedupCounts::WaitDispatch()')
    body=source[start:source.index('void ServerDedupCounts::CalRecvAndSetFlag()',start)]
    assert body.index('BuildCountReadPlan') < body.index('while (sumOfFlag')
    poll=body[body.index('while (sumOfFlag'):body.index('// Commit acquired')]
    assert 'CountFrame(' not in poll
    assert 'readPlan, false' in poll and 'readPlan, true' not in poll
    assert 'readPlan, true' in body
    assert 'if (!packedFrames_) WaitDispatchClearStatus();' in body
