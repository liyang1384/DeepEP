from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def test_retirement_after_role_completion_before_final_timestamp():
    code=(ROOT/'csrc/kernels/dispatch/low_latency/dispatch_server_dedup_kernel.h').read_text(encoding='utf-8')
    assert code.index('counts_.Finish();') < code.index('counts_.Retirement()->Pending()') < code.index('profile_->Mark<21>()')
    assert 'receiver_.FinalScratch()' in code and 'counts_.RetireOwnScratch()' in code
    receive=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h').read_text(encoding='utf-8')
    assert 'return prefixBuf_.Get<uint8_t>();' in receive
