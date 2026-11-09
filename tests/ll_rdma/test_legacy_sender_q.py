from pathlib import Path
from tests.ll_rdma.test_combine_overhead import block
ROOT=Path(__file__).resolve().parents[2]

def test_no_eight_qp_planning_remains():
    code=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h').read_text(encoding='utf-8')
    for token in ['qPlanBuf_', 'SendRemotePackets', 'LlBuildRemotePacketOwners', 'networkRoundRobin_', 'RouteWeight(', 'LlBuildRouteQ(']:
        assert token not in code
    assert 'LlSenderHasWork(enableDedup_, count_' in code
    assert 'LlStageTimer' not in code
