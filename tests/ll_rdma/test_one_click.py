"""CPU-only launcher checks; no dependency installation or devices."""
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest
from deep_ep.utils.ll_reference_profile_parse import WORDS

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("ndr_one_click", Path(__file__).with_name("one_click.py"))
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


@pytest.mark.parametrize("mode", ["perf", "reference"])
def test_defaults_use_real_st_parser_and_keep_modes_separate(mode):
    a = launch.parse_args(["1", "192.0.2.1", "run-001", "--mode", mode])
    arguments = launch.test_arguments(a)
    assert arguments[:4] == ["--backend", "rdma", "--devices", ",".join(map(str, range(16)))]
    st = runpy.run_path(str(ROOT / "tests/elastic/test_ep_low_latency.py"))
    # The shared wrapper consumes --devices before delegating to this parser.
    options = st["parse_args"](arguments[:2] + arguments[4:])
    assert st["validate_args"](options, 32) == 256
    cases = st["matrix_cases"](options)
    assert [(case.num_tokens, case.dtype) for case in cases] == [(8, "bf16"), (8, "fp8")]
    assert options.warmup == 10 and options.num_iters == 100 and options.perf_tail == 50
    assert options.correctness_repeats == options.repeat_combine == 3
    assert not options.skip_check and options.dispatch_only == (mode == "reference")
    assert options.group_backend == "hccl" and options.do_cpu_sync == 0
    assert options.ll_stage_timer == (mode == "timer")
    assert options.skip_perf_test == (mode != "perf")
    assert options.ll_reference_profile == (mode == "reference")
    if mode == "reference": assert options.ll_timer_iters == 1
    assert launch.run_id(a) == "run-001-" + mode


def test_plan_clears_stale_rank_and_sdk_environment(monkeypatch):
    for key in (*launch.RANK_ENV, "SHMEM_ROOT", "CATLASS_ROOT", "DEEPEP_LL_SEND_CORES"):
        monkeypatch.setenv(key, "stale")
    a = launch.parse_args(["0", "192.0.2.1", "plan"])
    env = launch.child_environment(a)
    assert not set(launch.RANK_ENV).intersection(env)
    assert "SHMEM_ROOT" not in env and "DEEPEP_LL_SEND_CORES" not in env
    assert all(env[key] == a.devices for key in launch.VISIBILITY)
    assert env["DEEPEP_LL_STAGE_TIMER"] == "OFF"
    assert env["DEEPEP_LL_REFERENCE_PROFILE"] == "ON"


@pytest.mark.parametrize("extra", [["--devices", "0,1"], ["--devices", ",".join(["0"] * 16)],
    ["--master-port", "29501"], ["--ready-port", "70000"], ["--ready-timeout", "0"],
    ["--skip-build", "--rebuild-shmem"]])
def test_invalid_invocation_is_rejected(extra):
    with pytest.raises(SystemExit):
        launch.parse_args(["0", "192.0.2.1", "valid", *extra])


@pytest.mark.parametrize("name", ["../escape", "/absolute", "two words", "-option", ""])
def test_run_name_cannot_escape_project(name):
    with pytest.raises(SystemExit):
        launch.parse_args(["0", "192.0.2.1", name])


def test_rebuild_is_explicit_and_skip_build_is_read_only(monkeypatch):
    monkeypatch.delenv("DEEPEP_LL_JIT", raising=False)
    a = launch.parse_args(["0", "192.0.2.1", "x"])
    commands = launch.command_plan(a, "/project with spaces/.venv/bin/python")
    assert commands[1][1:] == ["scripts/build_shmem.py"]
    assert commands[-2][1:] == ["scripts/record_shmem_sdk.py", "--check", "--sdk", "third-party/shmem/install/shmem"]
    a.rebuild_shmem = True
    assert "--rebuild" in launch.command_plan(a, "python")[1]
    a.skip_build = True
    commands = launch.command_plan(a, "python")
    assert len(commands) == 2 and all("pip" not in command and "git" not in command for command in commands)


def test_deep_jit_wheel_initializes_its_own_submodule(monkeypatch):
    a = launch.parse_args(["0", "192.0.2.1", "jit"])
    monkeypatch.delenv("DEEPEP_LL_JIT", raising=False)
    commands = launch.command_plan(a, "python")
    assert commands[1][1:] == ["scripts/build_shmem.py"]
    assert not any("third-party/deep_jit" in command for command in commands)
    monkeypatch.setenv("DEEPEP_LL_JIT", "ON")
    commands = launch.command_plan(a, "python")
    assert commands[1][1:] == ["submodule", "update", "--init", "--recursive", "--", "third-party/deep_jit"]
    assert commands[2][1:] == ["scripts/build_shmem.py"]
    a.skip_build = True
    assert not any("deep_jit" in command for command in launch.command_plan(a, "python"))


def test_paths_resolve_from_project_not_working_directory(tmp_path, monkeypatch):
    root = tmp_path / "copied project with spaces"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(launch, "ROOT", root)
    monkeypatch.chdir(elsewhere)
    assert launch.project_path("results/run") == root / "results/run"
    with pytest.raises(ValueError):
        launch.project_path("../outside")


def test_plan_does_not_prepare_or_create_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(launch, "ROOT", tmp_path)
    monkeypatch.setattr(launch, "validate_test_arguments", lambda a: None)
    called = []
    monkeypatch.setattr(launch.subprocess, "call", lambda command, **kwargs: called.append((command, kwargs)) or 0)
    monkeypatch.setattr(launch, "run_logged", lambda *a: pytest.fail("plan attempted a build"))
    assert launch.main(["0", "192.0.2.1", "dry", "--plan"]) == 0
    assert len(called) == 1 and called[0][0][-1] == "--plan"
    assert not list(tmp_path.iterdir())
    assert json.loads(capsys.readouterr().out)["mode"] == "reference"


def test_git_commands_do_not_inherit_vendor_loader_paths():
    env = {"LD_LIBRARY_PATH": "vendor", "LD_PRELOAD": "shim", "PATH": "tools"}
    assert launch.git_environment(env) == {"PATH": "tools"}
    assert "LD_LIBRARY_PATH" in env


@pytest.mark.parametrize("failure", ["build", "identity", "ready", "test"])
def test_failed_workflow_never_launches_following_stages(tmp_path, monkeypatch, failure):
    root = tmp_path / "project with spaces"
    root.mkdir()
    python = root / ".venv-ll-rdma/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(launch, "ROOT", root)
    monkeypatch.setattr(launch, "validate_test_arguments", lambda a: None)
    monkeypatch.setattr(launch.sys, "platform", "linux")
    monkeypatch.setattr(launch.os, "environ", dict(os.environ))
    monkeypatch.setenv("ASCEND_HOME_PATH", "/toolchain")
    monkeypatch.setattr(launch.subprocess, "check_output", lambda *a, **k: launch.BRANCH + "\n")
    monkeypatch.setattr(launch.runpy, "run_path", lambda *a, **k: {"environment": lambda: dict(os.environ)})
    commands = []

    def execute(command, env, log):
        command = list(map(str, command))
        commands.append(command)
        if failure == "build" and command[0] == "git":
            raise RuntimeError("build failed")
        if "--identity-output" in command:
            if failure == "identity":
                raise RuntimeError("wrong wheel")
            Path(command[-1]).write_text('{"schema":1,"sdk_build":{"hns_1825_pi_on_chip":0}}')
        if failure == "test" and "tests/ll_stage_timer/run.py" in command and "--plan" not in command:
            raise RuntimeError("test failed")

    def ready(*args):
        commands.append(["READY"])
        if failure == "ready":
            raise RuntimeError("peer mismatch")
        return {"status": "ready"}

    monkeypatch.setattr(launch, "run_logged", execute)
    monkeypatch.setitem(sys.modules, "target_ready", SimpleNamespace(synchronize=ready))
    with pytest.raises(RuntimeError):
        launch.main(["0", "192.0.2.1", "failure"])
    assert not any("tests/ll_stage_timer/analyze.py" in command for command in commands)
    launched = [c for c in commands if "tests/ll_stage_timer/run.py" in c and "--plan" not in c]
    assert bool(launched) == (failure == "test")


@pytest.mark.parametrize("tokens", ["0", "257"])
def test_bad_shape_rejected_before_any_preparation(tokens, monkeypatch):
    monkeypatch.setattr(launch, "run_logged", lambda *a: pytest.fail("invalid shape started preparation"))
    with pytest.raises(ValueError, match="actual B"):
        launch.main(["0", "192.0.2.1", "bad-shape", "--tokens", tokens])


@pytest.mark.parametrize("mode", ["perf", "reference"])
@pytest.mark.parametrize("nodes", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("histogram", [False, True])
@pytest.mark.parametrize("fault", [None, "timer", "source", "sdk", "firmware", "library", "branch", "protocol", "qps", "reference", "reference_metadata", "group", "histogram", "jit"])
def test_installed_check_rejects_stale_or_mixed_builds(tmp_path, monkeypatch, mode, nodes, histogram, fault):
    monkeypatch.setenv('DEEPEP_LL_COUNT_HISTOGRAM', 'ON' if histogram else 'OFF')
    monkeypatch.delenv("DEEPEP_LL_JIT", raising=False)
    root = tmp_path / "project"
    root.mkdir()
    (root / "dependencies.lock.json").write_text('{"shmem":{"revision":"sdk"}}')
    package = root / ".venv-ll-rdma/lib/site-packages/deep_ep"
    package.mkdir(parents=True)
    sdk = {"build": {"hns_1825_pi_on_chip": 0}}
    info = {"source_sha256": "source", "shmem": {"revision": "sdk", "build": dict(sdk["build"])},
            "ll_stage_timer": mode == "timer", "ll_reference_profile": (mode == "reference") != (fault == "reference_metadata")}
    if fault == "source": info["source_sha256"] = "old"
    info['ll_count_histogram'] = histogram != (fault == 'histogram')
    info["ll_jit"] = fault == "jit"
    if fault == "sdk": info["shmem"]["revision"] = "old"
    if fault == "firmware": info["shmem"]["build"]["hns_1825_pi_on_chip"] = 1
    (package / "_build_info.json").write_text(json.dumps(info))
    native = SimpleNamespace(
        low_latency_implementation="wrong" if fault == "branch" else "ndr-hns1825",
        low_latency_core_protocol=11 if fault == "protocol" else (35 if histogram else 34), low_latency_rdma_enabled=True,
        low_latency_core_config=lambda: {"active_aiv": 56},
        low_latency_roce_qps_per_peer=4 if fault == "qps" else 0,
        low_latency_stage_timer_enabled=(mode == "timer") != (fault == "timer"),
        low_latency_reference_profile_enabled=(mode == "reference") != (fault == "reference"),
        low_latency_reference_profile_layout=lambda: dict(schema=4, slots=80, fields=8,
                words=list(WORDS)),
        low_latency_ranks_per_group=lambda world: (99 if fault == "group" else (4 if world <= 16 else 16)),
        low_latency_stage_timer_layout=lambda: {"schema": 3},
        shmem_library_path=lambda: str(root / ("wrong.so" if fault == "library" else "third-party/shmem/install/shmem/lib/libshmem.so")))
    monkeypatch.setattr(launch, "ROOT", root)
    monkeypatch.setitem(sys.modules, "build_support", SimpleNamespace(
        validate_shmem_sdk=lambda *args: sdk, source_fingerprint=lambda: "source"))
    monkeypatch.setitem(sys.modules, "deep_ep._native", SimpleNamespace(require_native=lambda _: native))
    monkeypatch.setattr(launch.runpy, "run_path", lambda _: {"_import_installed_deep_ep":
        lambda: SimpleNamespace(__file__=str(package / "__init__.py"))})
    monkeypatch.setattr(launch.subprocess, "check_output", lambda *a, **kw: "revision\n")
    a = launch.parse_args([str(nodes - 1), "192.0.2.1", "check", "--mode", mode, "--num-nodes", str(nodes)])
    if fault:
        with pytest.raises(RuntimeError): launch.installed_identity(a)
    else:
        identity = launch.installed_identity(a)
        assert identity["project_root"] == str(root)
        assert identity["sdk_build"] == sdk["build"]
        assert identity["ll_jit"] is False
        assert identity["arguments"][identity["arguments"].index("--node-rank") + 1] == "<node>"


def test_peer_comparison_allows_different_pi_modes_but_keeps_other_checks():
    first = {"source_sha256": "same", "sdk_build": {"rdma_backend": "HNS_1825", "hns_1825_pi_on_chip": 0}}
    second = {"source_sha256": "same", "sdk_build": {"rdma_backend": "HNS_1825", "hns_1825_pi_on_chip": 1}}
    assert launch.peer_identity(first) == launch.peer_identity(second)
    assert first["sdk_build"]["hns_1825_pi_on_chip"] == 0
    assert second["sdk_build"]["hns_1825_pi_on_chip"] == 1
    second["source_sha256"] = "different"
    assert launch.peer_identity(first) != launch.peer_identity(second)
    second["source_sha256"] = "same"
    second["sdk_build"]["rdma_backend"] = "different"
    assert launch.peer_identity(first) != launch.peer_identity(second)


@pytest.mark.parametrize("mode", [None, True, "0", 2])
def test_peer_comparison_requires_a_validated_local_doorbell_mode(mode):
    with pytest.raises(RuntimeError):
        launch.peer_identity({"sdk_build": {"hns_1825_pi_on_chip": mode}})


def test_launcher_branch_matches_checked_out_experiment():
    actual = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if not actual:
        pytest.skip("Detached source snapshot has no branch name")
    assert launch.BRANCH == actual, "Launcher must not retain the original branch after an experiment fork"
