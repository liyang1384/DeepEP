"""Native SDK capacity admission contracts; CPU only, no CANN or device qualification."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_limit_precedes_initialization():
    session = (ROOT / "csrc/runtime/session.cpp").read_text()
    guard = session.index("validate_ll_rdma_qps(ll_rdma_qps_, size > int(DispatchDedup::kLlPhysicalRanksPerHost))")
    assert guard < session.index("occupied = true")
    assert guard < session.index("aclshmemx_set_qp_num(")
    assert guard < session.index("aclshmemx_init_attr(")
    assert "ll_rdma_qps_ = query_ll_core_config().active;" in session
    assert "low_latency_cores_ = ll_rdma_qps_;" in session
    assert "uint64_t(ll_rdma_qps_) * LlRdma::kCoreBytes" in session
    smoke = (ROOT / "tests/ll_rdma/smoke_host.cpp").read_text()
    assert smoke.index("validate_ll_rdma_qps(cores, ranksPerHost==1)") < smoke.index("aclshmemx_set_qp_num(")
