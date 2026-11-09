from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def test_commit_clear_follows_all_receiver_snapshots():
    s=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h').read_text(encoding='utf-8')
    assert 'if (!packedFrames_) WaitDispatchClearStatus();' in s
    tail = s[s.index('const uint32_t offset = GetCumSum(newAivId);'):]
    assert tail.index('profile_->Mark<27>()') < tail.index('if (packedFrames_) WaitDispatchClearStatus();')
    assert 'if (expert == frame.firstExpert)' in s
