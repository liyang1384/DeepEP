"""Exercise native build orchestration without claiming CANN compilation coverage."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
from setuptools import Distribution, Extension


ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "build_support.py").is_file():
    pytest.skip("Build-source tests run before packaging", allow_module_level=True)


@pytest.fixture
def native(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("incremental_build_support", ROOT / "build_support.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    project = tmp_path / "relocated project with spaces"
    project.mkdir()
    monkeypatch.setattr(module, "ROOT", project)
    monkeypatch.setenv("CMAKE_BUILD_PARALLEL_LEVEL", "4")
    pybind = ModuleType("pybind11")
    pybind.get_cmake_dir = lambda: str(project / "pybind cmake")
    monkeypatch.setitem(sys.modules, "pybind11", pybind)
    configuration = {
        "source_sha256": "first", "source_version": "1", "ll_stage_timer": False,
        "cmake_path": "fake cmake", "compiler_path": "fake compiler",
        "ASCEND_HOME_PATH": str(project / "cann sdk"),
        "SHMEM_ROOT": str(project / "shmem sdk"), "build_type": "Release",
        "compiler_sha256": "host", "device_compiler_sha256": "device",
        "python_executable": sys.executable, "build_environment": {},
    }
    monkeypatch.setattr(module, "build_configuration", lambda: dict(configuration))
    ext = Extension("deep_ep._C", sources=[])
    command = module.CMakeBuild(Distribution({"ext_modules": [ext]}))
    command.ensure_finalized()
    command.build_temp = str(project / "native temp")
    command.build_lib = str(project / "wheel lib")
    destination = Path(command.get_ext_fullpath(ext.name)).parent
    destination.mkdir(parents=True)
    state = SimpleNamespace(module=module, configuration=configuration, command=command,
                            ext=ext, calls=[], fail=None, mutate_cache_on_build=False)

    def run(argv):
        args = list(map(str, argv))
        state.calls.append(args)
        phase = "configure" if "-B" in args else "build" if "--build" in args else "install"
        if state.fail == phase:
            raise subprocess.CalledProcessError(1, args)
        if phase == "configure":
            directory = Path(args[args.index("-B") + 1])
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "CMakeCache.txt").write_text("fake configured cache\n")
            (directory / "Makefile").write_text("fake build graph\n")
            (directory / "device").mkdir(exist_ok=True)
            (directory / "device/CMakeCache.txt").write_text("fake child cache\n")
            (directory / "device/Makefile").write_text("fake child build graph\n")
            (directory / "compiler-info.json").write_text('{"id":"fake","version":"1"}')
            (directory / "capabilities.json").write_text('{"capabilities":[]}')
        elif phase == "build" and state.mutate_cache_on_build:
            directory = Path(args[args.index("--build") + 1])
            (directory / "CMakeCache.txt").write_text("cache updated by automatic regeneration\n")

    monkeypatch.setattr(module, "checked_run", run)
    state.run = lambda: command.build_extension(ext)
    state.directory = lambda: Path(command.build_temp) / module.build_cache_key(configuration)
    state.phases = lambda: ["configure" if "-B" in args else "build" if "--build" in args else "install"
                            for args in state.calls]
    return state


def test_repeated_wheel_keeps_build_and_install_but_skips_configure(native):
    native.run()
    native.calls.clear()
    native.run()
    assert native.phases() == ["build", "install"]
    report = json.loads((native.directory() / "build-timing.json").read_text())
    assert report["configure_skipped"] is True
    assert report["cache_key"] == native.module.build_cache_key(native.configuration)
    assert report["ll_stage_timer"] is False
    assert set(report["seconds"]) >= {"preflight", "configure", "build", "install", "total"}
    assert all(value >= 0 for value in report["seconds"].values())


def test_source_identity_changes_preserve_native_cache_and_update_metadata(native):
    native.run()
    before = native.directory()
    native.configuration.update(source_sha256="edited Python or kernel", source_version="2")
    native.calls.clear()
    native.run()
    assert native.directory() == before
    assert native.phases() == ["build", "install"]
    destination = Path(native.command.get_ext_fullpath(native.ext.name)).parent
    assert json.loads((destination / "_build_info.json").read_text())["source_sha256"] == "edited Python or kernel"


def test_timer_switch_reconfigures_shared_cache(native):
    native.run()
    directory = native.directory()
    for timer in (True, False):
        native.configuration["ll_stage_timer"] = timer
        native.calls.clear()
        native.run()
        assert native.directory() == directory
        assert native.phases() == ["configure", "build", "install"]
        assert f"-DDEEPEP_LL_STAGE_TIMER={'ON' if timer else 'OFF'}" in native.calls[0]


@pytest.mark.parametrize("damage", ["cache", "graph", "signature", "missing_cache",
                                    "missing_child_cache", "missing_child_graph", "child_cache"])
def test_external_cache_or_graph_damage_requires_configure(native, damage):
    native.run()
    directory = native.directory()
    if damage == "cache":
        (directory / "CMakeCache.txt").write_text("external edit\n")
    elif damage == "graph":
        (directory / "Makefile").unlink()
    elif damage == "missing_cache":
        (directory / "CMakeCache.txt").unlink()
    elif damage == "missing_child_cache":
        (directory / "device/CMakeCache.txt").unlink()
    elif damage == "missing_child_graph":
        (directory / "device/Makefile").unlink()
    elif damage == "child_cache":
        (directory / "device/CMakeCache.txt").write_text("external child edit\n")
    else:
        (directory / ".deepep-configure.json").write_text("invalid JSON")
    native.calls.clear()
    native.run()
    assert native.phases() == ["configure", "build", "install"]


def test_jobs_change_reconfigures_without_new_cache(native, monkeypatch):
    native.run()
    directory = native.directory()
    monkeypatch.setenv("CMAKE_BUILD_PARALLEL_LEVEL", "2")
    native.calls.clear()
    native.run()
    assert native.directory() == directory
    assert native.phases() == ["configure", "build", "install"]
    assert "-DDEEPEP_DEVICE_JOBS=2" in native.calls[0]


@pytest.mark.parametrize("phase", ["configure", "build"])
def test_failed_build_does_not_commit_new_configure_signature(native, phase):
    native.run()
    signature = native.directory() / ".deepep-configure.json"
    native.configuration["ll_stage_timer"] = True
    native.fail = phase
    with pytest.raises(subprocess.CalledProcessError):
        native.run()
    assert not signature.exists()
    native.fail = None
    native.calls.clear()
    native.run()
    assert native.phases() == ["configure", "build", "install"]


def test_success_records_cache_after_automatic_regeneration(native):
    native.mutate_cache_on_build = True
    native.run()
    native.calls.clear()
    native.run()
    assert native.phases() == ["build", "install"]


def test_toolchain_changes_keep_cache_isolation(native):
    original = native.module.build_cache_key(native.configuration)
    for key, value in (("compiler_sha256", "new host"), ("device_compiler_sha256", "new device"),
                       ("SHMEM_ROOT", "new sdk"), ("build_environment", {"CXXFLAGS": "-g"})):
        assert native.module.build_cache_key({**native.configuration, key: value}) != original


def test_ll_jit_ships_device_headers_with_the_extension(native):
    entry = native.module.ROOT / "csrc/kernels/dispatch/low_latency/dispatch_entry.h"
    entry.parent.mkdir(parents=True)
    entry.write_text("// device header\n")
    native.configuration.update(
        ll_jit=True,
        TORCH_NPU_PATH=str(native.module.ROOT / "torch_npu"),
        Torch_DIR=str(native.module.ROOT / "Torch"),
    )
    native.run()
    assert "-DDEEPEP_LL_JIT=ON" in native.calls[0]
    assert f"-DTORCH_NPU_PATH={native.configuration['TORCH_NPU_PATH']}" in native.calls[0]
    destination = Path(native.command.get_ext_fullpath(native.ext.name)).parent
    assert (destination / "_kernel_src/csrc/kernels/dispatch/low_latency/dispatch_entry.h").is_file()
    info = json.loads((destination / "_build_info.json").read_text())
    assert info["ll_jit"] is True
    assert "TORCH_NPU_PATH" not in info
