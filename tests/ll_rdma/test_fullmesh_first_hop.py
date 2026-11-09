from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_sender_scopes_policy_and_keeps_earlier_members():
    code = (ROOT / "csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h").read_text(encoding="utf-8")
    assert "Group == kLlPhysicalRanksPerHost && LlTransport::Rdma(args)" in code
    assert "LlFirstHopForMask(routeIds_" in code
    assert "meshFirstK" not in code
    assert "LlRelayMemberMask(mask, firstK)" in code
    assert "firstK + 1U" not in code
    assert "for (uint32_t k = 0U; members != 0U && k < k_; ++k)" in code
