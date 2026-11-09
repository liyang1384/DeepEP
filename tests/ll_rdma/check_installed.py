"""Installed Host/device capability and rejected-selector checks; no device launch."""
import argparse
import os
from pathlib import Path
import runpy
from environment import environment

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--rdma", type=int, choices=(1,), default=1, help="Compatibility argument; NDR always includes RDMA")
parser.add_argument("--timer", type=int, choices=(0, 1), default=0)
args = parser.parse_args()
os.environ.update(environment())
helpers = runpy.run_path(str(ROOT / "tests/elastic/test_dispatch.py"))
package = helpers["_import_installed_deep_ep"]()
from deep_ep._native import require_native
native = require_native("RDMA build capability check")
assert native.low_latency_core_protocol == 35
assert native.low_latency_roce_qps_per_peer == 0
for p in (16, 32, 64, 128, 256):
    assert native.low_latency_ranks_per_group(p) == (4 if p <= 16 else 16)
    d = native.low_latency_dispatch_tiling(p, 4, 256, 4, 256, "bfloat16", 64)
    c = native.low_latency_combine_tiling(p, 4, 256, 4, 256, "bfloat16", 64)
    assert d[21] == (4 if p <= 16 else 16) and c[15] == d[21]
assert bool(native.low_latency_rdma_enabled) == bool(args.rdma)
assert bool(native.low_latency_stage_timer_enabled) == bool(args.timer)
os.environ["DEEPEP_LL_BACKEND"] = "invalid-selector"
try:
    native.Runtime(0, 2, 2 << 20, "tcp://127.0.0.1:1", 1)
except ValueError as exc:
    assert "must be mte or rdma" in str(exc)
else:
    raise AssertionError("Invalid selector was accepted")
print(f"PASS installed {package.__file__}: ABI=5 RDMA=1 timer={args.timer}; Host negative checks only")
