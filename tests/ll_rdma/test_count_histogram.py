"""Contract checks for the count-histogram build switch; not CANN qualification."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_switch_reaches_host_device_cache_and_identity():
    import build_support
    assert build_support.build_cache_key({'ll_count_histogram': False}) != build_support.build_cache_key({'ll_count_histogram': True})
    for path in ('CMakeLists.txt', 'cmake/Device.cmake', 'cmake/device/CMakeLists.txt', 'build_support.py'):
        assert 'DEEPEP_LL_COUNT_HISTOGRAM' in (ROOT / path).read_text(encoding='utf-8')
    counts = (ROOT / 'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h').read_text(encoding='utf-8')
    body = counts.split('void ServerDedupCounts::CountRemoteRank(', 1)[1]
    alternative = body.split('#ifdef DEEPEP_LL_COUNT_HISTOGRAM', 1)[1].split('#else', 1)[0]
    assert alternative.count('SyncFunc<HardEvent::V_S>') == 1
    assert alternative.index('SyncFunc<HardEvent::V_S>') < alternative.index('for (uint32_t i')
