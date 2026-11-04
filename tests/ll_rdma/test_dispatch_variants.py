"""CPU-only dispatch format and single-entry checks; never starts an NPU."""
from pathlib import Path
import os
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_dispatch_input_class_mapping_and_invalid_inputs(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if compiler is None and not zig:
        pytest.skip("C++17 compiler required for the CPU dispatch selector fixture")
    command = [zig, "c++"] if zig else [compiler]
    source = tmp_path / "dispatch_class.cpp"
    source.write_text(r'''
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include "csrc/kernels/dispatch/low_latency/common/ll_dispatch_variant.h"
#include "csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_input.h"
#include "csrc/kernels/dispatch/low_latency/launch.hpp"
using namespace DispatchDedup;
// Independent table: dtype, quant mode, output dtype, input mode.
// Modes 0..3 run in the FP16-class kernel, 4..6 in the BF16-class kernel.
constexpr uint8_t formats[7][4] = {
    {15,4,15,2}, {3,0,3,0}, {3,4,15,1}, {3,4,16,1},
    {11,0,11,0}, {11,4,15,1}, {11,4,16,1}
};
uint32_t expected(const uint8_t* fields) {
    for (unsigned mode=0; mode<7; ++mode) {
        bool matches = true;
        for (unsigned field=0; field<4; ++field) {
            // Plain16's reserved output dtype means the input dtype.
            if (field == 2 && fields[field] == 255 && (mode == 1 || mode == 4)) continue;
            if (fields[field] != formats[mode][field]) matches = false;
        }
        if (matches) return mode >= 4 ? kLlDispatchInputBf16 : kLlDispatchInputFp16;
    }
    return kLlDispatchInvalidVariant;
}
void check(const uint8_t* fields) {
    assert(LlDispatchInputClass(fields[0],fields[1],fields[2],fields[3]) == expected(fields));
}
int main() {
    static_assert(sizeof(deepep::EpServerDedupKernelArgs)==152);
    static_assert(kLlDispatchFp16==DISPATCH_DEDUP_DATA_TYPE_FP16);
    static_assert(kLlDispatchBf16==DISPATCH_DEDUP_DATA_TYPE_BFP16);
    static_assert(kLlDispatchE4M3==DISPATCH_DEDUP_DATA_TYPE_FP8E4M3);
    static_assert(kLlDispatchE5M2==DISPATCH_DEDUP_DATA_TYPE_FP8E5M2);
    static_assert(kLlDispatchPlain16==uint8_t(EpServerDedupInputMode::Plain16));
    static_assert(kLlDispatchQuantize16==uint8_t(EpServerDedupInputMode::Quantize16));
    static_assert(kLlDispatchPrequantized==uint8_t(EpServerDedupInputMode::PrequantizedFp8Packs));
    static_assert(LlDispatchInputClass(15,4,15,2)==kLlDispatchInputFp16);
    static_assert(LlDispatchInputClass(3,0,3,0)==kLlDispatchInputFp16);
    static_assert(LlDispatchInputClass(11,4,16,1)==kLlDispatchInputBf16);
    for (unsigned mode=0; mode<7; ++mode) {
        check(formats[mode]);
        for (unsigned field=0; field<4; ++field) for (unsigned value=0; value<256; ++value) {
            uint8_t modified[4];
            for (unsigned i=0;i<4;++i) modified[i]=formats[mode][i];
            modified[field]=uint8_t(value);
            check(modified);
        }
    }
    const uint8_t dtypes[]={0,3,11,15,16,255};
    const uint8_t quants[]={0,1,4,255};
    const uint8_t modes[]={0,1,2,3,255};
    for (uint8_t dtype:dtypes) for (uint8_t output:dtypes)
        for (uint8_t quant:quants) for (uint8_t input:modes) {
            uint8_t fields[]={dtype,quant,output,input};
            check(fields);
        }
}
''', encoding="utf-8")
    executable = tmp_path / ("dispatch_class.exe" if os.name == "nt" else "dispatch_class")
    subprocess.run([*command, "-std=c++17", "-UNDEBUG", "-iquote", str(ROOT), str(source), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)


def test_single_dispatch_entry_replaces_variants():
    """The branch keeps main's single entry and one device translation unit."""
    entry = (ROOT / "csrc/kernels/dispatch/low_latency/dispatch_entry.h").read_text(encoding="utf-8")
    assert entry.count("__global__ __aicore__ void dispatch_server_dedup_kernel(") == 1
    assert "extern \"C\" DEEPEP_EXPORT void dispatch_server_dedup_kernel_do(" in entry
    assert "_do_v2" not in entry
    assert "DISPATCH_SERVER_DEDUP_RUN(half, 4U)" in entry
    assert "DISPATCH_SERVER_DEDUP_RUN(half, 16U)" in entry
    assert "DISPATCH_SERVER_DEDUP_RUN(bfloat16_t, 4U)" in entry
    assert "DISPATCH_SERVER_DEDUP_RUN(bfloat16_t, 16U)" in entry
    unit = (ROOT / "csrc/kernels/dispatch/low_latency.cpp").read_text(encoding="utf-8")
    assert '#include "low_latency/dispatch_entry.h"' in unit
    assert "__global__" not in unit
    capabilities = (ROOT / "csrc/ops/ll_capabilities.cpp").read_text(encoding="utf-8")
    assert "deepep_ll_core_protocol_version" in capabilities
    assert "deepep_ll_core_protocol_version" not in unit
    assert not (ROOT / "csrc/kernels/dispatch/low_latency/variants").exists()
    assert not (ROOT / "csrc/kernels/dispatch/low_latency/dispatch_variant_kernel.h").exists()


def test_jit_build_reuses_the_single_dispatch_entry():
    """The JIT path differs only in launch; the entry source stays shared."""
    entry = (ROOT / "csrc/kernels/dispatch/low_latency/dispatch_entry.h").read_text(encoding="utf-8")
    kernel, launch = entry.split("#ifndef DEEPEP_LL_JIT", 1)
    assert "__global__ __aicore__ void dispatch_server_dedup_kernel(" in kernel
    assert "DISPATCH_SERVER_DEDUP_RUN(bfloat16_t, 16U)" in kernel
    assert "dispatch_server_dedup_kernel_do" in launch
    host = (ROOT / "csrc/ops/low_latency.cpp").read_text(encoding="utf-8")
    assert "#ifdef DEEPEP_LL_JIT" in host
    assert "deepep::ll_jit_launch(reinterpret_cast<void*>(stream_address), args, spec)" in host
    device = (ROOT / "cmake/device/CMakeLists.txt").read_text(encoding="utf-8")
    assert device.index("if(NOT DEEPEP_LL_JIT)") < device.index(
        '"${DEEPEP_SOURCE_DIR}/csrc/kernels/dispatch/low_latency.cpp"')
    assert "csrc/kernels/combine/low_latency.cpp" in device
    jit = (ROOT / "csrc/runtime/ll_jit.cpp").read_text(encoding="utf-8")
    assert "kernels/dispatch/low_latency/dispatch_entry.h" in jit
    # DeepJIT's parser rejects quoted includes, so only the entry line uses an
    # angle include and the header trees are hashed into the cache signature.
    assert '<< "#include <" << kEntryHeader' in jit
    assert "header_fingerprint()" in jit
    # Every distinct case is baked into its own translation unit and cache key.
    assert "#define DEEPEP_LL_JIT_SPECIALIZED 1" in jit
    assert "jit.compile(spec_tag(spec), entry_source(spec), overrides)" in jit
    for macro in ("DEEPEP_LL_JIT_SPEC_BS", "DEEPEP_LL_JIT_SPEC_H", "DEEPEP_LL_JIT_SPEC_TOPK",
                  "DEEPEP_LL_JIT_SPEC_EXPERTS", "DEEPEP_LL_JIT_SPEC_CORES", "DEEPEP_LL_JIT_SPEC_WORLD",
                  "DEEPEP_LL_JIT_SPEC_INPUT_MODE", "DEEPEP_LL_JIT_SPEC_E4M3", "DEEPEP_LL_JIT_SPEC_DEDUP"):
        assert f'#define {macro} "' in jit
    assert "DEEPEP_LL_JIT_SPEC_BS" in entry and "AscendC::Trap();" in entry
    # The JIT unit carries the whole launch identity: tiling is not read, and
    # Python skips building/uploading it for this build.
    assert "(void)tilingData;" in entry and "tiling->" not in entry.split("#else", 1)[0]
    assert "spec.cores = args.coreNum;" in host and "low_latency_jit" in host
    assert "args.tilingData = nullptr;" in host
    python = (ROOT / "deep_ep/buffers/elastic.py").read_text(encoding="utf-8")
    assert 'getattr(native, "low_latency_jit", False)' in python
    spec = (ROOT / "csrc/kernels/dispatch/low_latency/common/ll_jit_spec.h").read_text(encoding="utf-8")
    assert "DEEPEP_LL_LAUNCH_WORLD" in spec and "DEEPEP_LL_LAUNCH_CORES" in spec


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
