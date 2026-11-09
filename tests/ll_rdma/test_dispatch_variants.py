"""Dispatch format and single-entry contracts; never starts an NPU."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_single_dispatch_entry_replaces_variants():
    """The branch keeps main's single entry and one device translation unit."""
    entry = (ROOT / "csrc/kernels/dispatch/low_latency.cpp").read_text(encoding="utf-8")
    assert entry.count("__global__ __aicore__ void dispatch_server_dedup_kernel(") == 1
    assert "extern \"C\" DEEPEP_EXPORT void dispatch_server_dedup_kernel_do(" in entry
    assert "_do_v2" not in entry
    assert "DISPATCH_SERVER_DEDUP_RUN(half, 4U)" in entry
    assert "DISPATCH_SERVER_DEDUP_RUN(half, 16U)" in entry
    assert "DISPATCH_SERVER_DEDUP_RUN(bfloat16_t, 4U)" in entry
    assert "DISPATCH_SERVER_DEDUP_RUN(bfloat16_t, 16U)" in entry
    assert not (ROOT / "csrc/kernels/dispatch/low_latency/variants").exists()
    assert not (ROOT / "csrc/kernels/dispatch/low_latency/dispatch_variant_kernel.h").exists()


def test_host_launch_uses_same_format_builder_as_tiling():
    host = (ROOT / "csrc/ops/low_latency.cpp").read_text(encoding="utf-8")
    assert "auto data = dispatch_format_tiling(p, dtype);" in host
    assert "dispatch_format_tiling(session.world(), dtype)" in host
    assert "LlDispatchInputClass(" in host
    assert "dispatch_server_dedup_kernel_do(reinterpret_cast<void*>(stream_address), &args)" in host
    assert "dispatch_server_dedup_kernel_do_v2" not in host
    launch = (ROOT / "csrc/kernels/dispatch/low_latency/launch.hpp").read_text(encoding="utf-8")
    assert "dispatch_server_dedup_kernel_do(" in launch
    assert "_do_v2" not in launch
