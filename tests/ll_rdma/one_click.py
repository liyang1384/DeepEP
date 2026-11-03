"""Prepare and run NDR on the requested number of 16-rank target hosts."""

import argparse
import ipaddress
import json
import os
from pathlib import Path
import re
import runpy
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BRANCH = "kernel/ll-ndr-ab-per-core-qp"
VISIBILITY = ("CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "PROBE_VISIBLE_DEVICES")
RANK_ENV = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK",
            "MASTER_ADDR", "MASTER_PORT", "DEEPEP_SHMEM_ENDPOINT")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
        epilog='Offline export: bash tests/ll_rdma/one_click.sh --analyze-only RUN_DIR [--output-dir NEW_DIR] [--iteration last|N]. No CANN required.')
    p.add_argument("node_rank", type=int)
    p.add_argument("--num-nodes", type=int, default=2, help="Host count; existing LL EP domains are 16/32/64/128/256")
    p.add_argument("master_addr", help="Reachable numeric IPv4 address of node 0; identical on all hosts")
    p.add_argument("run_name", help="New shared name; the script appends -reference or -perf")
    p.add_argument("--mode", choices=("perf", "reference"), default="reference")
    p.add_argument("--devices", default=",".join(map(str, range(16))))
    p.add_argument("--tokens", nargs="+", type=int, default=[8])
    p.add_argument("--dtype", nargs="+", choices=("bf16", "fp8"), default=["bf16", "fp8"])
    p.add_argument("--capacity", type=int, default=256)
    p.add_argument("--hidden", type=int, default=3584)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--experts", type=int, default=384)
    p.add_argument("--route", choices=("normal", "hotspot"), default="normal")
    p.add_argument("--seed", type=int, default=0, help="Routing seed offset; normal uses base 20261014")
    p.add_argument("--reference-iters", type=int, default=1)
    p.add_argument("--master-port", type=int, default=29500)
    p.add_argument("--shmem-port", type=int, default=29501)
    p.add_argument("--ready-port", type=int, default=29502)
    p.add_argument("--ready-timeout", type=int, default=1800, help="Seconds to wait for all other prepared hosts")
    p.add_argument("--rebuild-shmem", action="store_true", help="Explicitly rebuild an old SDK after stopping its users")
    p.add_argument("--skip-build", action="store_true", help="Use the existing project venv/wheel after strict checks")
    p.add_argument("--prepare-only", action="store_true", help="Build and check without joining or launching")
    p.add_argument("--plan", action="store_true", help="CPU plan only; no download/build/venv/device launch")
    p.add_argument("--identity-output", type=Path, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if os.environ.get("DEEPEP_LL_COUNT_HISTOGRAM", "OFF") not in ("ON", "OFF"):
        p.error("DEEPEP_LL_COUNT_HISTOGRAM must be ON or OFF (compile-time option)")
    if a.num_nodes < 1 or not 0 <= a.node_rank < a.num_nodes:
        p.error("node_rank must be in [0, num_nodes), with positive num_nodes")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", a.run_name):
        p.error("run_name must be 1..80 letters/digits/underscore/dot/hyphen and start with a letter or digit")
    try:
        ipaddress.IPv4Address(a.master_addr)
    except ipaddress.AddressValueError:
        p.error("master_addr must be node 0's numeric IPv4 address")
    ports = (a.master_port, a.shmem_port, a.ready_port)
    if len(set(ports)) != 3 or any(not 1 <= n <= 65535 for n in ports):
        p.error("HCCL/SHMEM/preparation ports must be distinct values in 1..65535")
    devices = a.devices.split(",")
    if len(devices) != 16 or any(not s.isdecimal() for s in devices) or len(set(map(int, devices))) != 16:
        p.error("Supply exactly 16 distinct physical devices on each host")
    a.devices = ",".join(str(int(s)) for s in devices)
    if a.skip_build and a.rebuild_shmem:
        p.error("--skip-build cannot be combined with --rebuild-shmem")
    if not 1 <= a.ready_timeout <= 86400:
        p.error("ready-timeout must be in 1..86400 seconds")
    if not 1 <= a.reference_iters <= 1000:
        p.error("reference-iters must be in 1..1000")
    return a


def project_path(path):
    path = Path(path)
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Path must remain inside this project: {path}")
    return resolved


def run_id(a):
    return f"{a.run_name}-{a.mode}"


def ll_jit_enabled(environment=None):
    """Same OFF/ON/1/0 spelling as the build switches in build_support."""
    env = os.environ if environment is None else environment
    return env.get("DEEPEP_LL_JIT", "OFF").upper() in {"ON", "1", "TRUE"}


def test_arguments(a):
    result = ["--backend", "rdma", "--devices", a.devices,
              "--num-processes", "16", "--num-nodes", str(a.num_nodes), "--node-rank", str(a.node_rank),
              "--master-addr", a.master_addr, "--master-port", str(a.master_port),
              "--shmem-port", str(a.shmem_port), "--num-tokens", *map(str, a.tokens),
              "--num-max-tokens-per-rank", str(a.capacity), "--hidden", str(a.hidden),
              "--num-topk", str(a.topk), "--num-experts", str(a.experts), "--dtype", *a.dtype,
              "--fp8-scale-dtype", "int32", "--do-cpu-sync", "0", "--group-backend", "hccl",
              "--use-1g-huge-pages", "0", "--route", a.route, "--seed", str(a.seed),
              "--warmup", "10", "--num-iters", "100", "--perf-tail", "50",
              "--correctness-repeats", "3", "--repeat-combine", "3", "--num-cpu-timeout-secs", "300",
              "--output-dir", "results/ll-ndr", "--run-name", run_id(a)]
    if a.mode == "reference":
        result += ["--dispatch-only", "--ll-reference-profile", "--ll-timer-iters", str(a.reference_iters),
                   "--ll-timer-frequency-hz", "1000000000", "--skip-perf-test"]
    return result


def command_plan(a, python):
    commands = []
    if not a.skip_build:
        commands += [["git", "submodule", "update", "--init", "--recursive", "--depth", "1", "--", "third-party/shmem"]]
        if ll_jit_enabled():
            # DeepJIT compiles the dispatch entry on the first call, so its
            # headers must be present. The pinned commit is not a branch tip,
            # so do not shallow-fetch this submodule.
            commands += [["git", "submodule", "update", "--init", "--recursive", "--", "third-party/deep_jit"]]
        commands += [[str(python), "scripts/build_shmem.py", *(["--rebuild"] if a.rebuild_shmem else [])],
                     [str(python), "-m", "pip", "install", "-r", "requirements-build.txt"],
                     [str(python), "-m", "pip", "install", "-v", "--force-reinstall", "--no-build-isolation", "--no-deps", "."]]
    commands += [[str(python), "scripts/record_shmem_sdk.py", "--check", "--sdk", "third-party/shmem/install/shmem"],
                 [str(python), "tests/ll_stage_timer/run.py", *test_arguments(a), "--plan"]]
    return commands


def validate_test_arguments(a):
    st = runpy.run_path(str(ROOT / "tests/elastic/test_ep_low_latency.py"))
    arguments = test_arguments(a)
    # --devices belongs to the shared wrapper, not the ST parser.
    st["validate_args"](st["parse_args"](arguments[:2] + arguments[4:]), 16 * a.num_nodes)


def child_environment(a):
    env = dict(os.environ)
    for key in (*RANK_ENV, "SHMEM_ROOT", "CATLASS_ROOT", "DEEPEP_LL_RDMA",
                "DEEPEP_LL_RANKS_PER_GROUP", "DEEPEP_LL_SEND_CORES", "DEEPEP_LL_RECV_CORES"):
        env.pop(key, None)
    env.update({key: a.devices for key in VISIBILITY})
    env.update(DEEPEP_NPU_ARCH="Ascend950", DEEPEP_LL_BACKEND="rdma",
               DEEPEP_LL_STAGE_TIMER="OFF",
               DEEPEP_LL_REFERENCE_PROFILE="ON" if a.mode == "reference" else "OFF",
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", GIT_TERMINAL_PROMPT="0")
    return env


def git_environment(env):
    result = dict(env)
    result.pop("LD_LIBRARY_PATH", None)
    result.pop("LD_PRELOAD", None)
    return result


def run_logged(command, env, log):
    command = [str(v) for v in command]
    print("+ " + shlex.join(command), flush=True)
    with log.open("x", encoding="utf-8") as output:
        output.write("+ " + shlex.join(command) + "\n")
        output.flush()
        command_env = git_environment(env) if command[0] == "git" else env
        with subprocess.Popen(command, cwd=ROOT, env=command_env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                output.write(line)
                output.flush()
            code = process.wait()
        if code:
            raise RuntimeError(f"Command failed (exit {code}); see {log}")


def installed_identity(a):
    # This branch is run by a fresh process using the selected venv, after pip.
    sys.path.insert(0, str(ROOT))
    import build_support
    lock = json.loads((ROOT / "dependencies.lock.json").read_text())["shmem"]
    sdk_root = project_path("third-party/shmem/install/shmem")
    sdk = build_support.validate_shmem_sdk(sdk_root, lock)
    fingerprint = build_support.source_fingerprint()
    helpers = runpy.run_path(str(ROOT / "tests/elastic/test_dispatch.py"))
    package = helpers["_import_installed_deep_ep"]()
    from deep_ep._native import require_native
    native = require_native("NDR one-click installation check")
    info = json.loads((Path(package.__file__).parent / "_build_info.json").read_text())
    if info.get("source_sha256") != fingerprint:
        raise RuntimeError("Installed wheel does not match current sources; rerun without --skip-build")
    if (info.get("shmem", {}).get("revision") != lock["revision"] or
            info.get("shmem", {}).get("build") != sdk["build"]):
        raise RuntimeError("Installed wheel and current SHMEM SDK identity differ; rebuild the wheel")
    histogram = os.environ.get("DEEPEP_LL_COUNT_HISTOGRAM", "OFF") == "ON"
    expected_protocol = 35 if histogram else 34
    if (getattr(native, "low_latency_implementation", None) != "ndr-hns1825"
            or native.low_latency_core_protocol != expected_protocol
            or info.get("ll_count_histogram", False) != histogram):
        raise RuntimeError(f"Installed wheel does not match count histogram={histogram}, protocol={expected_protocol}; rebuild the complete wheel")
    if not native.low_latency_rdma_enabled or bool(native.low_latency_stage_timer_enabled):
        raise RuntimeError("Installed RDMA/timer capability does not match the requested mode")
    if getattr(native, 'low_latency_roce_qps_per_peer', None) != 0:
        raise RuntimeError('Installed wheel must use dynamic per-physical-core RoCE QPs')
    reference = getattr(native, 'low_latency_reference_profile_enabled', None)
    if type(reference) is not bool or reference != (a.mode == 'reference'):
        raise RuntimeError('Installed reference-profile capability does not match the requested mode')
    if type(info.get('ll_reference_profile')) is not bool or info['ll_reference_profile'] != reference:
        raise RuntimeError('Installed reference-profile metadata and native capability differ')
    if reference:
        from deep_ep.utils.ll_reference_profile_parse import SCHEMA, SLOTS, FIELDS, WORDS
        layout = native.low_latency_reference_profile_layout()
        if (layout != dict(schema=SCHEMA, slots=SLOTS, fields=FIELDS, words=list(WORDS)) or
                any(type(layout.get(key)) is not int for key in ('schema', 'slots', 'fields'))):
            raise RuntimeError('Installed reference-profile schema differs from this NDR branch')
        if info.get('ll_stage_timer') is not False:
            raise RuntimeError('Reference-profile wheel must compile legacy stage tracing OFF')
    # The JIT and Ahead-of-Time wheels carry the same sources, so only this
    # marker distinguishes them; never mix the two on one domain.
    jit = ll_jit_enabled()
    if bool(info.get('ll_jit', False)) != jit:
        raise RuntimeError(
            f"Installed wheel was built with DEEPEP_LL_JIT={'ON' if info.get('ll_jit') else 'OFF'}; "
            f"rebuild it with DEEPEP_LL_JIT={'ON' if jit else 'OFF'}")
    world = 16 * a.num_nodes
    if (native.low_latency_ranks_per_group(world) != (4 if world <= 16 else 16)
            or native.low_latency_stage_timer_layout()["schema"] != 3):
        raise RuntimeError("Installed group/timer schema differs from this NDR branch")
    if Path(native.shmem_library_path()).resolve() != (sdk_root / "lib/libshmem.so").resolve():
        raise RuntimeError("Loaded SHMEM library is outside the selected SDK")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       env=git_environment(os.environ), text=True).strip()
    # Local library hashes may differ between independent builds. Each SDK has
    # been validated locally. Keep the local firmware mode in the saved identity.
    shared_args = test_arguments(a)
    shared_args[shared_args.index("--node-rank") + 1] = "<node>"
    return {"schema": 1, "project_root": str(ROOT.resolve()), "git_revision": revision,
            "source_sha256": fingerprint, "sdk_revision": lock["revision"], "sdk_build": sdk["build"],
            "core_protocol": expected_protocol, "count_histogram": histogram,
            "roce_qps_per_peer": native.low_latency_core_config()["active_aiv"], "timer_schema": 3, "reference_profile": reference,
            "reference_profile_schema": 4 if reference else None, "ll_jit": jit, "mode": a.mode, "arguments": shared_args}


def peer_identity(identity):
    """Compare shared run facts while retaining each node's local PI mode."""
    shared = json.loads(json.dumps(identity))
    build = shared["sdk_build"]
    mode = build.pop("hns_1825_pi_on_chip", None)
    if type(mode) is not int or mode not in (0, 1):
        raise RuntimeError("Missing validated local HNS doorbell mode")
    # PI mode configures the local NIC doorbell. installed_identity still
    # requires this node's SDK and wheel to match, and identity.json retains
    # its actual mode. Only the comparison copy omits this single field.
    return shared


def main(argv=None):
    a = parse_args(argv)
    validate_test_arguments(a)
    env = child_environment(a)
    test_command = ["tests/ll_stage_timer/run.py", *test_arguments(a)]
    if a.identity_output:
        target = project_path(a.identity_output)
        with target.open("x", encoding="utf-8") as output:
            output.write(json.dumps(installed_identity(a), indent=2) + "\n")
        return 0
    if a.plan:
        print(json.dumps({"project_root": str(ROOT), "mode": a.mode, "run_id": run_id(a),
                          "build": command_plan(a, ROOT / ".venv-ll-rdma/bin/python"),
                          "ready_endpoint": f"{a.master_addr}:{a.ready_port}",
                          "launch": ["<venv-python>", *test_command]}, indent=2), flush=True)
        return subprocess.call([sys.executable, *test_command, "--plan"], cwd=ROOT, env=env)
    if sys.platform != "linux" or not env.get("ASCEND_HOME_PATH"):
        raise RuntimeError("Run one_click.sh on Linux after loading CANN or setting CANN_ENV")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT,
                                     env=git_environment(env), text=True).strip()
    if branch != BRANCH:
        raise RuntimeError(f"Expected branch {BRANCH}, found {branch!r}; no branch is changed automatically")
    result = project_path(Path("results/ll-ndr") / run_id(a))
    if result.exists():
        raise FileExistsError(f"Run already exists on this host: {result}; use a new run_name")
    phase = "prepare" if a.prepare_only else "execute"
    logs = project_path(Path(".logs/ndr-one-click") / run_id(a) / f"node-{a.node_rank}" / phase)
    logs.mkdir(parents=True, exist_ok=False)
    # Load only the existing path policy; the shell entry has already sourced CANN.
    os.environ.clear()
    os.environ.update(env)
    env = runpy.run_path(str(ROOT / "tests/ll_rdma/environment.py"))["environment"]()
    venv = project_path(".venv-ll-rdma")
    python = venv / "bin/python"
    env["PATH"] = str(venv / "bin") + os.pathsep + env.get("PATH", "")
    env["VIRTUAL_ENV"] = str(venv)
    env.pop("PYTHONHOME", None)
    if not python.is_file():
        if a.skip_build:
            raise FileNotFoundError("Project venv is absent; omit --skip-build")
        run_logged([sys.executable, "-m", "venv", "--system-site-packages", venv], env, logs / "00-venv.log")
    commands = command_plan(a, python)
    for index, command in enumerate(commands):
        run_logged(command, env, logs / f"10-step-{index:02d}.log")
    original = list(sys.argv[1:] if argv is None else argv)
    identity_path = logs / "identity.json"
    run_logged([python, Path(__file__).resolve(), *original, "--identity-output", identity_path], env, logs / "20-identity.log")
    run_logged([python, "tests/ll_rdma/preflight.py", "--output", logs / "preflight.json"], env, logs / "21-preflight.log")
    print(f"Preparation passed: node={a.node_rank}, logs={logs}", flush=True)
    if a.prepare_only:
        print("Prepared only; launch all hosts with the same name and --skip-build.")
        return 0
    from target_ready import synchronize
    print(f"Waiting for all {a.num_nodes} prepared hosts at {a.master_addr}:{a.ready_port} (up to {a.ready_timeout}s)", flush=True)
    agreed = synchronize(a.node_rank, a.master_addr, a.ready_port, run_id(a),
                         peer_identity(json.loads(identity_path.read_text())), a.ready_timeout, a.num_nodes)
    (logs / "agreement.json").write_text(json.dumps(agreed, indent=2) + "\n", encoding="utf-8")
    run_logged([python, *test_command], env, logs / "30-test.log")
    if a.mode == "reference":
        run_logged([python, "tests/ll_rdma/analyze.py", result, "--iteration", "last"],
                   env, logs / "40-analysis.log")
    print(f"PASS node {a.node_rank}: {result}\nLogs: {logs}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"NDR one-click failed: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
