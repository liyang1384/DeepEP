"""Standalone Notify precision, profiling and metadata regression checks."""

import json
from pathlib import Path
import re
import runpy
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import pytest
import argparse
import csv
import math
import statistics
import os
import importlib
from importlib.machinery import PathFinder
import sysconfig
import time
import traceback
import random


_ROUTES = ("normal", "random", "duplicates", "invalid", "empty", "hotspot")
_cases = runpy.run_path(str(Path(__file__).with_name("combine_cases.py")))


def _progress_logger(operation, rank):
    started = previous = time.perf_counter()

    def log(message):
        nonlocal previous
        now = time.perf_counter()
        print(
            f"[{operation}][rank={rank}] {time.strftime('%H:%M:%S')} "
            f"elapsed={now - started:.3f}s delta={now - previous:.3f}s {message}",
            flush=True,
        )
        previous = now

    return log


def _log_failure(log):
    # Cleanup may wait for device work or peers; expose the original error first.
    log("FAILED: traceback before buffer.destroy")
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()


def group_peer(rank, world, group, index):
    if world == 128:
        toggle, subround = divmod(group, 4)
        row, side = rank % 64 // 32, rank % 8 // 4
        peer_side = side ^ (subround % 2)
        peer_row = (row + subround // 2) % 2
        frame = rank // 64 ^ side ^ toggle
        return frame * 64 + (peer_row * 4 + index // 4) * 8 + peer_side * 4 + index % 4
    width, half = min(world, 16), world // 2
    half_width, groups = width // 2, world // width
    own = rank % half // half_width
    target = (own + group) % groups
    return index // half_width * half + target * half_width + index % half_width


def group_steps(rank, world):
    steps = [0] * world
    for group in range(world // min(world, 16)):
        for index in range(min(world, 16)):
            steps[group_peer(rank, world, group, index)] = (group * min(world, 16) + index) // min(world, 64)
    return steps


def build_notify_golden(routes, experts, chunk_tokens, *, rank_dedup=False):
    world = len(routes)
    if world not in (2, 4, 8, 16, 32, 64, 128, 256, 512) or experts <= 0 or experts % world:
        raise ValueError("unsupported Notify shape")
    tokens = len(routes[0])
    topk = len(routes[0][0]) if tokens else 0
    if (
        chunk_tokens <= 0
        or not 1 <= topk <= 16
        or any(len(rows) != tokens or any(len(row) != topk for row in rows) for rows in routes)
    ):
        raise ValueError("routes must have a common [tokens, topk] shape")
    local_experts = experts // world
    splits = min(4, tokens, chunk_tokens)
    first = min(tokens, chunk_tokens)
    chunks = (tokens + chunk_tokens - 1) // chunk_tokens + splits - 1

    def token_chunk(token):
        return token * splits // first if token < first else token // chunk_tokens + splits - 1

    def chunk_end(chunk):
        return (
            (chunk + 1) * first // splits + bool((chunk + 1) * first % splits)
            if chunk < splits
            else min((chunk + 2 - splits) * chunk_tokens, tokens)
        )

    counts = [0] * experts
    for rows in routes:
        for row in rows:
            for expert in row:
                if 0 <= expert < experts:
                    counts[expert] += 1
    starts = [0] * experts
    rank_rows = [0] * world
    for expert, count in enumerate(counts):
        peer = expert // local_experts
        starts[expert] = rank_rows[peer]
        rank_rows[peer] += count
    stride = max(1, max(rank_rows, default=0))
    invalid = -(1 << 63)
    plans = []
    for source in range(world):
        plans.append(
            {
                "server_mask": [0] * tokens,
                "dst": [invalid] * (tokens * topk),
                "weight_return_meta": [-1] * rank_rows[source],
                "forward_list": [[] for _ in range(world)],
                "backward_list": [[] for _ in range(min(world, 8))],
                "chunk_ranges": [[(0, 0) for _ in range(chunks)] for _ in range(world)],
                "chunk_masks": [0] * (world // min(world, 64) * chunks),
            }
        )

    expert_cursors = [0] * experts
    gathers = [0] * world
    for source, rows in enumerate(routes):
        for token, route in enumerate(rows):
            by_server = {}
            for slot, expert in enumerate(route):
                if not 0 <= expert < experts:
                    continue
                peer = expert // local_experts
                output_row = starts[expert] + expert_cursors[expert]
                expert_cursors[expert] += 1
                plans[peer]["weight_return_meta"][output_row] = (source * tokens + token) * topk + slot
                server = peer if rank_dedup else peer // 8
                mask_slot = list(by_server).index(server) if server in by_server else len(by_server)
                mask_slot = mask_slot if rank_dedup else server
                plans[source]["server_mask"][token] |= (1 << mask_slot) if mask_slot < 63 else -(1 << 63)
                by_server.setdefault(server, []).append((slot, peer, output_row))
            for records in by_server.values():
                primary_slot, gateway, primary_row = next((r for r in records if r[1] == source), records[0])
                for slot, peer, output_row in records:
                    encoded = peer * stride + output_row
                    value = encoded if slot == primary_slot else ~encoded
                    plans[source]["dst"][token * topk + slot] = value
                forward = plans[gateway]["forward_list"][source]
                if len(records) == 1:
                    forward.append((source, token, primary_row, -1, -1, -1))
                    continue
                for slot, peer, output_row in records:
                    if slot == primary_slot:
                        continue
                    reduce_row = -1
                    if peer != gateway:
                        reduce_row = gathers[gateway]
                        gathers[gateway] += 1
                        plans[peer]["backward_list"][gateway % 8].append((source, output_row, reduce_row, token))
                    forward.append((source, token, primary_row, peer, output_row, reduce_row))

    for gateway, plan in enumerate(plans):
        for source, block in enumerate(plan["forward_list"]):
            begin = 0
            while begin < len(block):
                end = begin + 1
                while end < len(block) and block[end][1] == block[begin][1]:
                    end += 1
                for row in range(begin, end):
                    block[row] = (end, *block[row][1:])
                begin = end
            cursor = 0
            for chunk in range(chunks):
                begin = cursor
                token_end = chunk_end(chunk)
                while cursor < len(block) and block[cursor][1] < token_end:
                    peer = block[cursor][3]
                    if peer >= 0 and peer != gateway:
                        plan["chunk_masks"][group_steps(gateway, world)[source] * chunks + chunk] |= 1 << (peer % 8)
                    cursor += 1
                plan["chunk_ranges"][source][chunk] = (begin, cursor)
            if rank_dedup:
                slots = max((world + 7) // 8, topk)
                for index, row in enumerate(block):
                    token = row[1]
                    peers = list(dict.fromkeys(e // local_experts for e in routes[source][token] if 0 <= e < experts))
                    block[index] = (row[0], -(token * slots + peers.index(gateway)) - 1, *row[2:])
        for lane, block in enumerate(plan["backward_list"]):
            gateway_peer = gateway // 8 * 8 + lane
            gateway_steps = group_steps(gateway_peer, world)
            ordered = list(enumerate(block))
            ordered.sort(
                key=lambda item: (
                    token_chunk(item[1][3]),
                    gateway_steps[item[1][0]],
                    item[1][0],
                    item[0],
                )
            )
            block[:] = [item[1] for item in ordered]
            for index, (source, row, reduce_row, token) in enumerate(block):
                key = gateway_steps[source] * chunks + token_chunk(token) + 1
                next_key = 0
                if index + 1 == len(block):
                    next_key = key
                else:
                    next_source, _, _, next_token = block[index + 1]
                    candidate = gateway_steps[next_source] * chunks + token_chunk(next_token) + 1
                    if candidate != key:
                        next_key = key
                block[index] = (source, row, reduce_row | (next_key << 32))
        plan["forward_counts"] = [len(block) for block in plan["forward_list"]]
        plan["backward_counts"] = [len(block) for block in plan["backward_list"]]
    for plan in plans:
        plan["rank_rows"] = rank_rows
        plan["gather_rows"] = gathers
    return {"plans": plans, "rank_rows": rank_rows, "gather_rows": plans[0]["gather_rows"], "address_stride": stride}


@pytest.mark.parametrize("world", [8, 16])
@pytest.mark.parametrize("mode", ["random", "duplicates", "invalid", "empty"])
def test_rank_notify_golden_matches_rank_return_plan(world, mode):
    combine = runpy.run_path(str(Path(__file__).with_name("test_combine.py")))
    routes = make_routes(world, 9, 6, 384, 0, mode, seed=2026)
    golden = build_notify_golden(routes, 384, 256, rank_dedup=True)
    for rank, plan in enumerate(golden["plans"]):
        reference = combine["build_reference"](routes, 384, rank, 256, server_dedup=False)
        for name, expected in reference.items():
            actual = plan[name]
            if name == "chunk_ranges":
                actual = [v for peer in actual for pair in peer for v in pair]
            assert actual == expected, (rank, mode, name)
        assert not any(plan["backward_counts"]) and not any(plan["gather_rows"])


@pytest.mark.parametrize("world", [8, 16, 32, 64, 128, 256, 512])
def test_notify_prefers_source_gateway_without_reordering_slots(world):
    combine = runpy.run_path(str(Path(__file__).with_name("test_combine.py")))
    dispatch = runpy.run_path(str(Path(__file__).with_name("test_dispatch.py")))
    experts, topk = world * 2, 6
    routes = []
    for source in range(world):
        other, remote = source // 8 * 8 + (source + 1) % 8, (source + 8) % world
        routes.append(
            [
                [other * 2, remote * 2, source * 2 + 1, source * 2, -1, other * 2],
                [other * 2, remote * 2, -1, -1, experts, other * 2],
                [-1] * topk,
            ]
        )
    original = [[row[:] for row in rows] for rows in routes]
    golden = build_notify_golden(routes, experts, 256)
    if world <= 256:
        assert golden == dispatch["build_notify_golden"](routes, experts, 256, rank_dedup=False)
    assert routes == original
    stride = golden["address_stride"]
    for source, rows in enumerate(routes):
        plan = golden["plans"][source]
        for token, row in enumerate(rows):
            groups = {}
            for slot, expert in enumerate(row):
                if 0 <= expert < experts:
                    groups.setdefault(expert // 16, []).append(slot)
            for slots in groups.values():
                selected = next((k for k in slots if row[k] // 2 == source), slots[0])
                for slot in slots:
                    encoded = plan["dst"][token * topk + slot]
                    assert (encoded >= 0) == (slot == selected)
                    peer, output_row = divmod(encoded if encoded >= 0 else ~encoded, stride)
                    assert peer == row[slot] // 2
                    assert golden["plans"][peer]["weight_return_meta"][output_row] == (
                        (source * len(rows) + token) * topk + slot
                    )
    for rank in (0, world // 2, world - 1):
        for name, expected in combine["build_reference"](routes, experts, rank).items():
            actual = golden["plans"][rank][name]
            if name == "chunk_ranges":
                actual = [v for peer in actual for pair in peer for v in pair]
            assert actual == expected, (rank, name)


def make_routes(world, tokens, topk, experts, generation, mode, *, seed=1, without_replacement=False):
    """Generate deterministic routes, including duplicate expert slots.

    Seed/source/token determine routing; generation only changes normal fixtures.
    Keep random routes fixed across payload generations for cached dispatch.
    """
    if mode == "random" and without_replacement and not 1 <= topk <= experts:
        raise ValueError("Random sampling without replacement requires 1 <= topk <= experts")
    result = []
    for source in range(world):
        rows = []
        for token in range(tokens):
            row = [(source * 17 + token * 7 + slot * 131 + generation) % experts for slot in range(topk)]
            if mode == "random":
                state = (seed ^ (source * 0x9E3779B9) ^ (token * 0x85EBCA6B)) & 0xFFFFFFFF
                if without_replacement:
                    row = random.Random(state).sample(range(experts), topk)
                else:
                    row = []
                    for _ in range(topk):
                        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
                        row.append((state >> 8) % experts)
            if mode == "duplicates" and topk > 1:
                row[1] = row[0]
            if mode == "invalid":
                row = [
                    (experts + 17 if (source + token + slot) % 2 else -1) if (source + token + slot) % 3 == 0 else e
                    for slot, e in enumerate(row)
                ]
            if mode == "empty":
                row = [-1] * topk
            if mode == "hotspot":
                row = [experts - 1] * topk
            if mode == "sq_pressure":
                row = [0 if source in (8, 20, 48, 52, 80, 84, 112, 116) else -1] * topk
            if mode in ("local_duplicates", "local_experts"):
                local = experts // world
                peer = (source + token) % world
                row = [peer * local + (slot % local if mode == "local_experts" else 0) for slot in range(topk)]
            rows.append(row)
        result.append(rows)
    return result


def add_random_route_args(parser):
    parser.add_argument("--seed", type=int, default=1, help="Random routing seed; identical on every rank")
    parser.add_argument(
        "--without-replacement",
        action="store_true",
        help="Sample distinct expert IDs per token in random routes (default: with replacement)",
    )


def validate_random_route_args(parser, args):
    if args.without_replacement:
        if args.route not in ("random", "all"):
            parser.error("--without-replacement requires --route random or all")
        if not 1 <= args.topk <= args.experts:
            parser.error("--without-replacement requires 1 <= topk <= experts")


@pytest.mark.parametrize("script", ["test_notify.py", "test_dispatch.py", "test_combine.py"])
def test_without_replacement_cli(script, monkeypatch, tmp_path):
    namespace = runpy.run_path(str(Path(__file__).with_name(script)))
    launch_scope = (
        namespace["_notify"]["_launch"].__globals__ if script == "test_combine.py" else namespace["main"].__globals__
    )
    calls = []
    if script == "test_combine.py":
        monkeypatch.setitem(namespace["_notify"], "_launch", lambda args, worker: calls.append(args))
    else:
        monkeypatch.setitem(launch_scope, "_launch", lambda args, worker: calls.append(args))
    command = [script, "--ep-size", "8", "--experts", "8", "--output", str(tmp_path)]
    for route in ("random", "all"):
        for enabled in (False, True):
            monkeypatch.setattr(
                sys, "argv", command + ["--route", route] + (["--without-replacement"] if enabled else [])
            )
            namespace["main"]()
            assert calls[-1].without_replacement == enabled
    count = len(calls)
    for extra in (["--route", "invalid"], ["--route", "random", "--topk", "9"]):
        monkeypatch.setattr(sys, "argv", command + ["--without-replacement", *extra])
        with pytest.raises(SystemExit) as exc:
            namespace["main"]()
        assert exc.value.code == 2
    assert len(calls) == count


@pytest.mark.parametrize("script", ["test_notify.py", "test_dispatch.py", "test_combine.py"])
def test_dedup_defaults_and_server_override(script, monkeypatch, tmp_path):
    namespace = runpy.run_path(str(Path(__file__).with_name(script)))
    calls = []
    scope = namespace["_notify"] if script == "test_combine.py" else namespace["main"].__globals__
    monkeypatch.setitem(scope, "_launch", lambda args, worker: calls.append(args))
    dest = "no_server_dedup" if script == "test_combine.py" else "rank_dedup"
    for world in (8, 16, 32, 64, 128, 256):
        for option in ([], ["--server-dedup"], *([["--rank-dedup"]] if world <= 16 else [])):
            command = [script, "--ep-size", str(world), "--output", str(tmp_path), *option]
            if script == "test_combine.py":
                command.append("--with-dispatch")
            monkeypatch.setattr(sys, "argv", command)
            namespace["main"]()
            assert getattr(calls[-1], dest) == (world in (8, 16) and option != ["--server-dedup"])


@pytest.mark.parametrize("script", ["test_notify.py", "test_dispatch.py"])
def test_without_replacement_sampling(script):
    generate = runpy.run_path(str(Path(__file__).with_name(script)))["make_routes"]
    state = random.getstate()
    for seed in (0, 1, 2026, -1, 1 << 40):
        for experts, topk in ((1, 1), (6, 6), (384, 6), (512, 16)):
            routes = generate(8, 17, topk, experts, 0, "random", seed=seed, without_replacement=True)
            assert all(len(row) == len(set(row)) == topk for rows in routes for row in rows)
            assert all(0 <= expert < experts for rows in routes for row in rows for expert in row)
            assert routes == generate(8, 17, topk, experts, 9, "random", seed=seed, without_replacement=True)
            assert routes[0] == generate(8, 18, topk, experts, 0, "random", seed=seed, without_replacement=True)[0][:17]
    assert random.getstate() == state
    with pytest.raises(ValueError, match="topk <= experts"):
        generate(8, 1, 9, 8, 0, "random", without_replacement=True)
    assert generate(8, 17, 6, 384, 0, "random", seed=1, without_replacement=True) != generate(
        8, 17, 6, 384, 0, "random", seed=2026, without_replacement=True
    )
    for mode in (*_ROUTES, "local_duplicates", "local_experts", "sq_pressure"):
        legacy = generate(8, 3, 6, 384, 0, mode)
        assert legacy == generate(8, 3, 6, 384, 0, mode, without_replacement=False)
        if mode != "random":
            assert legacy == generate(8, 3, 6, 384, 0, mode, without_replacement=True)
    invalid = generate(8, 3, 6, 384, 0, "invalid")
    assert any(-1 in row for rows in invalid for row in rows)
    assert any(expert >= 384 for rows in invalid for row in rows for expert in row)
    assert all(row == [-1] * 6 for rows in generate(8, 3, 6, 384, 0, "empty") for row in rows)


def check_notify_outputs(outputs, golden, rank, routes, experts):
    """Compare native metadata with the independent CPU golden, without Dispatch."""
    import torch

    plan = golden["plans"][rank]
    for name in (
        "server_mask",
        "dst",
        "forward_counts",
        "backward_counts",
        "chunk_ranges",
        "chunk_masks",
        "rank_rows",
        "gather_rows",
    ):
        actual = outputs[name]
        if actual.device.type != "npu":
            raise AssertionError(f"{name} is not an NPU tensor")
        expected = torch.tensor(plan[name], dtype=actual.dtype).reshape(actual.shape)
        if not torch.equal(actual.cpu(), expected):
            raise AssertionError(f"rank {rank}: {name} mismatch")
    for name, counts in (("forward_list", "forward_counts"), ("backward_list", "backward_counts")):
        actual = outputs[name].cpu()
        for row, count in enumerate(plan[counts]):
            expected = torch.tensor(plan[name][row], dtype=actual.dtype).reshape(count, actual.shape[-1])
            if not torch.equal(actual[row, :count], expected):
                raise AssertionError(f"rank {rank}: {name}[{row}] mismatch")
    weight_meta = outputs["weight_return_meta"][: golden["rank_rows"][rank]]
    weight_meta_cpu = weight_meta.cpu()
    expected_weight_meta = torch.tensor(
        [(s * len(routes[0]) + t) * len(routes[0][0]) + k for s, t, k in expert_rows(routes, experts, rank)],
        dtype=torch.int32,
    )
    if (
        weight_meta.dtype != torch.int32
        or weight_meta.device.type != "npu"
        or not weight_meta.is_contiguous()
        or not torch.equal(weight_meta_cpu, expected_weight_meta)
    ):
        raise AssertionError(f"rank {rank}: weight_return_meta mismatch")
    totals = [0] * experts
    for source in routes:
        for row in source:
            for expert in row:
                if 0 <= expert < experts:
                    totals[expert] += 1
    if not torch.equal(outputs["expert_totals"].cpu(), torch.tensor(totals, dtype=torch.int32)):
        raise AssertionError(f"rank {rank}: expert_totals mismatch")
    return totals


def _import_installed_deep_ep():
    # Launching from the checkout or inheriting PYTHONPATH must not shadow the wheel.
    locations = list(dict.fromkeys(sysconfig.get_path(name) for name in ("purelib", "platlib")))
    spec = PathFinder.find_spec("deep_ep", locations)
    if spec is None or spec.origin is None:
        raise RuntimeError(
            f"deep_ep is not installed in {sys.executable}; "
            "run python -m pip install --no-build-isolation --no-deps . first"
        )
    expected = Path(spec.origin).resolve()
    sys.path.insert(0, str(Path(spec.origin).parent.parent))
    package = importlib.import_module("deep_ep")
    actual = Path(package.__file__).resolve()
    if actual != expected:
        raise RuntimeError(f"deep_ep was already loaded from {actual}; expected installed package {expected}")
    return package


def _add_launch_args(parser):
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=8899)
    parser.add_argument("--shmem-ip-port", help="Defaults to DEEPEP_SHMEM_ENDPOINT or master-port + 1")


def _launch(args, worker):
    if getattr(args, "bootstrap", "default") == "mpi":
        try:
            rank, world = (int(os.environ[name]) for name in ("PMI_RANK", "PMI_SIZE"))
        except (KeyError, ValueError) as exc:
            raise ValueError("--bootstrap mpi requires an MPICH mpiexec launcher") from exc
        if (
            world != args.ep_size
            or args.nproc_per_node < 1
            or world % args.nproc_per_node
            or not 0 <= rank < world
            or rank // args.nproc_per_node != args.node_rank
        ):
            raise ValueError("MPI ranks must match EP/node-rank and use contiguous ranks per node")
        local_rank = rank % args.nproc_per_node
        os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(local_rank))
        # mpiexec already created one process per NPU. Never fork/spawn another
        # set of processes, which would lose the MPI_COMM_WORLD membership.
        worker(local_rank, args)
        return
    # torchrun may still supply process identities, but no process group is created.
    launched = [name in os.environ for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK")]
    if any(launched) and not all(launched):
        raise ValueError("RANK, WORLD_SIZE and LOCAL_RANK must be supplied together")
    if all(launched):
        args.ep_size = int(os.environ["WORLD_SIZE"])
    elif (
        args.nproc_per_node < 1
        or args.ep_size % args.nproc_per_node
        or not 0 <= args.node_rank < args.ep_size // args.nproc_per_node
    ):
        raise ValueError("EP must be divisible by nproc-per-node; node-rank must be within the job")
    if args.ep_size not in (2, 4, 8, 16, 32, 64, 128, 256, 512):
        raise ValueError("Dispatch/combine EP must be a power of two from 2 to 512")
    endpoint = args.shmem_ip_port or os.environ.get("DEEPEP_SHMEM_ENDPOINT")
    if not endpoint:
        master = os.environ.get("MASTER_ADDR", args.master_addr) if all(launched) else args.master_addr
        port = int(os.environ.get("MASTER_PORT", args.master_port)) if all(launched) else args.master_port
        endpoint = f"tcp://{master}:{port + 1}"
    os.environ["DEEPEP_SHMEM_ENDPOINT"] = endpoint
    if all(launched):
        worker(int(os.environ["LOCAL_RANK"]), args)
    else:
        import torch.multiprocessing as mp

        mp.spawn(worker, args=(args,), nprocs=args.nproc_per_node, join=True)


def _worker_identity(local_rank, args):
    rank = int(os.environ.get("RANK", args.node_rank * args.nproc_per_node + local_rank))
    if not 0 <= rank < args.ep_size or local_rank < 0:
        raise ValueError("Invalid rank or local rank")
    return rank, args.ep_size


def expert_rows(routes, experts, rank):
    """Independent destination order: local expert, source rank, token, top-k slot."""
    local = experts // len(routes)
    begin = rank * local
    buckets = [[] for _ in range(local)]
    for source, tokens in enumerate(routes):
        for token, row in enumerate(tokens):
            for slot, selected in enumerate(row):
                if begin <= selected < begin + local:
                    buckets[selected - begin].append((source, token, slot))
    return [entry for bucket in buckets for entry in bucket]


def assert_bytes(actual, expected, name, progress=None):
    import torch

    if progress:
        progress(f"{name}: readback.begin (waiting for device result)")
    actual = actual.cpu().contiguous()
    if progress:
        progress(f"{name}: readback.done; precision.begin")
    expected = expected.cpu().contiguous()
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"{name}: shape/dtype {actual.shape}/{actual.dtype} != {expected.shape}/{expected.dtype}")
    a, b = actual.view(torch.uint8).flatten(), expected.view(torch.uint8).flatten()
    bad = torch.nonzero(a != b).flatten()
    if bad.numel():
        first = bad[:8]
        raise AssertionError(
            f"{name}: {bad.numel()} mismatched bytes; offsets={first.tolist()}, "
            f"actual={a[first].tolist()}, expected={b[first].tolist()}"
        )
    if progress:
        progress(f"{name}: precision.passed")


_PROFILE_KERNELS = {"notify": "notify_full_kernel", "dispatch": "dispatch_kernel", "combine": "combine_kernel"}
_PROFILE_L2_BYTES = 8_000_000_000


def add_profile_args(parser, operations):
    parser.add_argument("--profile-op", choices=operations, help="Profile device kernels after precision validation")
    parser.add_argument("--profile-warmup", type=int, default=10)
    parser.add_argument("--profile-iters", type=int, default=50)
    parser.add_argument(
        "--profile-flush-l2",
        action="store_true",
        help="Clear an 8GB NPU buffer before each profiling call; exclude clearing from kernel timing",
    )


def validate_profile_args(parser, args):
    if args.profile_warmup < 0 or args.profile_iters < 1:
        parser.error("--profile-warmup must be >=0 and --profile-iters must be >=1")
    if args.profile_flush_l2 and not args.profile_op:
        parser.error("--profile-flush-l2 requires --profile-op")


def with_l2_flush(args, invoke, stream):
    if not getattr(args, "profile_flush_l2", False):
        return invoke
    if stream is None:
        raise ValueError("L2 flushing requires the operator's communication stream")
    import torch

    with torch.npu.stream(stream):
        scratch = torch.empty(_PROFILE_L2_BYTES // 4, dtype=torch.int32, device="npu")

    def call(*positional, **keywords):
        # Order clearing after the prior call and before the next kernel on this stream.
        with torch.npu.stream(stream):
            scratch.zero_()
        return invoke(*positional, **keywords)

    return call


def _statistics(kernel, durations):
    return dict(
        kernel=kernel,
        unit="us",
        samples=len(durations),
        durations_us=durations,
        avg_us=statistics.mean(durations),
        median_us=statistics.median(durations),
        min_us=min(durations),
        max_us=max(durations),
    )


def read_kernel_summary(directory, operator, expected_samples, *, ep8_rank_mte=False, combine_reuse=False):
    """Select one CSV source only; retain all samples and reject incomplete captures."""
    kernel = _PROFILE_KERNELS[operator]
    if combine_reuse:
        if operator != "combine" or ep8_rank_mte:
            raise ValueError("Reuse profiling requires server Combine")
        kernel = "combine_reuse_kernel"
    if ep8_rank_mte:
        if operator != "combine":
            raise ValueError("EP8 rank MTE profiling requires Combine")
        kernel = "combine_ep8_mte_kernel"
    wanted = {kernel}
    if operator == "combine" and not combine_reuse:
        wanted.add("combine_prepare_kernel")
    for pattern in ("kernel_details*.csv", "op_summary*.csv", "task_time*.csv"):
        tables = []
        for path in sorted(Path(directory).rglob(pattern)):
            values = {name: [] for name in wanted}
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                headers = {key.strip(): key for key in reader.fieldnames or ()}
                name_key = next((headers[k] for k in ("Name", "Op Name", "Kernel Name") if k in headers), None)
                units = [
                    (f"{prefix}({unit})", scale)
                    for unit, scale in (("us", 1.0), ("ns", 0.001), ("ms", 1000.0))
                    for prefix in ("Duration", "Task Duration")
                ]
                duration = next(((headers[k], scale) for k, scale in units if k in headers), None)
                if name_key is None or duration is None:
                    continue
                duration_key, scale = duration
                for row in reader:
                    name = (row.get(name_key) or "").strip()
                    if name not in wanted:
                        continue
                    try:
                        value = float(row[duration_key]) * scale
                    except (ValueError, TypeError) as exc:
                        raise ValueError(f"Invalid duration for {name} in {path}") from exc
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError(f"Invalid duration for {name} in {path}: {value}")
                    values[name].append(value)
            if values[kernel]:
                tables.append((path, values))
        if not tables:
            continue
        if len(tables) != 1:
            raise ValueError(f"Ambiguous {kernel} capture: {len(tables)} {pattern} tables under {directory}")
        path, values = tables[0]
        for name in wanted:
            if len(values[name]) != expected_samples:
                raise ValueError(f"{name}: expected {expected_samples} samples, found {len(values[name])} in {path}")
        summary = _statistics(kernel, values[kernel])
        summary.update(
            expected_samples=expected_samples, source_csv=str(path.resolve()), metric="device_kernel_duration"
        )
        if operator == "combine" and not combine_reuse:
            summary["prepare"] = _statistics("combine_prepare_kernel", values["combine_prepare_kernel"])
        return summary
    raise ValueError(f"No {kernel} duration records found in {directory}")


def capture_calls(profiler, invoke, synchronize, warmup, iterations, log):
    """One warmup step and one active step, with exactly the requested call counts."""
    last = None
    if warmup:
        log(f"profile warmup.begin calls={warmup}")
        for _ in range(warmup):
            last = invoke()
        synchronize()
        profiler.step()
    log(f"profile capture.begin calls={iterations}")
    for _ in range(iterations):
        last = invoke()
    synchronize()
    profiler.step()
    return last


@pytest.mark.parametrize(
    "script,operator", [("test_notify.py", "notify"), ("test_dispatch.py", "dispatch"), ("test_combine.py", "combine")]
)
def test_profile_l2_cli_defaults_and_requires_capture(script, operator):
    namespace = runpy.run_path(str(Path(__file__).with_name(script)))
    options = namespace["_notify"] if operator == "combine" else namespace
    parser = argparse.ArgumentParser()
    options["add_profile_args"](parser, (operator,))
    defaults = parser.parse_args([])
    assert (defaults.profile_warmup, defaults.profile_iters, defaults.profile_flush_l2) == (10, 50, False)
    args = parser.parse_args(["--profile-op", operator, "--profile-flush-l2", "--profile-iters", "7"])
    options["validate_profile_args"](parser, args)
    assert args.profile_flush_l2 and args.profile_iters == 7
    with pytest.raises(SystemExit):
        options["validate_profile_args"](parser, parser.parse_args(["--profile-flush-l2"]))


@pytest.mark.parametrize("warmup", [0, 2])
def test_profile_l2_orders_every_call_and_reuses_storage(monkeypatch, warmup):
    from contextlib import contextmanager

    events, allocations = [], []
    stream, active = object(), None

    @contextmanager
    def use_stream(selected):
        nonlocal active
        assert selected is stream and active is None
        active = selected
        try:
            yield
        finally:
            active = None

    def clear():
        assert active is stream
        events.append("clear")

    def allocate(count, **kwargs):
        assert active is stream and count * 4 == 8_000_000_000
        assert kwargs == dict(dtype="int32", device="npu")
        allocations.append(count)
        return SimpleNamespace(zero_=clear)

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(int32="int32", empty=allocate, npu=SimpleNamespace(stream=use_stream))
    )

    def invoke(*, profile=None):
        assert events[-1] == "clear" and active is None
        events.append("trace" if profile else "kernel")
        return len(events)

    args = SimpleNamespace(profile_flush_l2=True)
    call = with_l2_flush(args, invoke, stream)
    profiler = SimpleNamespace(step=lambda: events.append("step"))
    capture_calls(profiler, call, lambda: events.append("sync"), warmup, 50, lambda _: None)
    expected = (["clear", "kernel"] * warmup + ["sync", "step"]) if warmup else []
    assert events == expected + ["clear", "kernel"] * 50 + ["sync", "step"]
    call(profile=True)
    assert events[-2:] == ["clear", "trace"] and len(allocations) == 1
    assert with_l2_flush(SimpleNamespace(profile_flush_l2=False), invoke, None) is invoke
    with pytest.raises(ValueError, match="communication stream"):
        with_l2_flush(args, invoke, None)


@pytest.mark.parametrize("operator", ["notify", "dispatch", "combine"])
def test_profile_l2_clear_is_excluded_from_kernel_samples(tmp_path, operator):
    rows = ["Name,Duration(us)"]
    for i in range(50):
        rows.append("InplaceZero,9999")
        if operator == "combine":
            rows.append("combine_prepare_kernel,15")
        rows.append(f"{_PROFILE_KERNELS[operator]},{100 + i}")
    (tmp_path / "kernel_details.csv").write_text("\n".join(rows), encoding="utf-8")
    summary = read_kernel_summary(tmp_path, operator, 50)
    assert summary["durations_us"] == list(range(100, 150))
    assert summary["samples"] == 50 and summary["avg_us"] == 124.5


def profile_operator(args, rank, scenario, invoke, validate, build_info, log):
    """Capture rank-local kernel durations after warmup."""
    import torch
    import torch_npu

    directory = args.output / "profile" / args.profile_op / scenario / f"rank_{rank}" / str(time.time_ns())
    directory.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        operator=args.profile_op,
        rank=rank,
        world_size=args.ep_size,
        scenario=scenario,
        seed=args.seed,
        without_replacement=getattr(args, "without_replacement", False),
        tokens=args.tokens,
        topk=args.topk,
        experts=args.experts,
        hidden=getattr(args, "hidden", 7168),
        chunk_tokens=getattr(args, "chunk_tokens", 256),
        notify_num_cores=getattr(args, "notify_num_cores", 64),
        warmup=args.profile_warmup,
        iterations=args.profile_iters,
        flush_l2=getattr(args, "profile_flush_l2", False),
        flush_l2_bytes=_PROFILE_L2_BYTES if getattr(args, "profile_flush_l2", False) else 0,
        torch_version=str(torch.__version__),
        torch_npu_version=str(torch_npu.__version__),
        ascend_home_path=os.environ.get("ASCEND_HOME_PATH"),
        build_info=build_info,
        deduplication="rank"
        if getattr(args, "rank_dedup", False) or getattr(args, "no_server_dedup", False)
        else "server",
    )
    if args.profile_op == "combine":
        metadata.update(
            local_expert_first=getattr(args, "local_expert_first", False),
            combine_metadata_source="golden" if getattr(args, "golden_notify", False) else "notify",
            traffic=getattr(args, "combine_traffic", {}),
            input_storage=getattr(args, "input_storage", "gm"),
        )
    toolkit = Path(os.environ.get("ASCEND_HOME_PATH", "/nonexistent"))
    metadata["cann_version_files"] = {
        str(path): path.read_text(encoding="utf-8", errors="replace")
        for path in (toolkit / "version.cfg", toolkit / "version.info")
        if path.is_file()
    }
    (directory / "capture.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    log(f"profile op={args.profile_op} scenario={scenario} flush_l2={int(metadata['flush_l2'])} output={directory}")
    torch.npu.synchronize()
    profiler = torch_npu.profiler
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        schedule=profiler.schedule(wait=0, warmup=int(args.profile_warmup > 0), active=1, repeat=1),
        on_trace_ready=profiler.tensorboard_trace_handler(str(directory)),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        with_flops=False,
        experimental_config=profiler._ExperimentalConfig(
            export_type=[profiler.ExportType.Text],
            profiler_level=profiler.ProfilerLevel.Level0,
            aic_metrics=profiler.AiCMetrics.AiCoreNone,
            data_simplification=False,
        ),
    ) as capture:
        last = capture_calls(capture, invoke, torch.npu.synchronize, args.profile_warmup, args.profile_iters, log)
    torch.npu.synchronize()
    validate(last)
    if getattr(args, "combine_reuse", False):
        summary = read_kernel_summary(directory, args.profile_op, args.profile_iters, combine_reuse=True)
    elif getattr(args, "ep8_rank_mte", False):
        summary = read_kernel_summary(directory, args.profile_op, args.profile_iters, ep8_rank_mte=True)
    else:
        summary = read_kernel_summary(directory, args.profile_op, args.profile_iters)
    summary.update(metadata, precision_checked=True)
    if args.profile_op == "combine":
        payload_bytes = args.tokens * args.topk * getattr(args, "hidden", 7168) * 2
        summary.update(
            payload_bytes=payload_bytes,
            bandwidth_gbps=payload_bytes / (summary["avg_us"] * 1000),
            bandwidth_metric="logical_bf16_payload_per_rank_over_mean_kernel_duration",
        )
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log(
        f"profile kernel={summary['kernel']} samples={summary['samples']} avg_us={summary['avg_us']:.3f} "
        f"median_us={summary['median_us']:.3f} min_us={summary['min_us']:.3f} max_us={summary['max_us']:.3f}"
    )
    if "prepare" in summary:
        log(f"profile combine_prepare_kernel avg_us={summary['prepare']['avg_us']:.3f} excluded_from_combine=1")
    return summary


def profile_notify(args, rank, scenario, invoke, validate, info, log, *, stream=None):
    """Profile the clean kernel, then collect exactly one separate diagnostic call."""
    import torch
    from deep_ep._native import require_native

    view = runpy.run_path(str(Path(__file__).with_name("notify_profile.py")))
    native = require_native("Notify diagnostics")
    if not hasattr(native, "notify_profile_layout"):
        raise RuntimeError("Rebuild/install the wheel with Notify diagnostics v2")
    selected_cores = getattr(args, "notify_num_cores", 64)
    version, cores, words = native.notify_profile_layout(selected_cores)
    if (version, words) != (view["VERSION"], view["WORDS"]) or cores != selected_cores or cores not in (32, 64):
        raise RuntimeError("Notify profile ABI differs; update the test and installed wheel together")
    invoke = with_l2_flush(args, invoke, stream)
    clean = profile_operator(args, rank, scenario, invoke, validate, info, log)
    if clean["kernel"] != "notify_full_kernel":
        raise RuntimeError("Notify performance must use the uninstrumented kernel")
    profile = torch.zeros((cores, words), dtype=torch.int64, device="npu")
    torch.npu.synchronize()  # Allocation/zeroing on default stream precedes comm-stream export.
    log("diagnostic.begin kernel=notify_profile_kernel (outside profiler)")
    result = invoke(profile=profile, sample=1)
    torch.npu.synchronize()
    validate(result)
    data = view["decode"](
        profile.cpu().tolist(),
        rank=rank,
        world=args.ep_size,
        tokens=args.tokens,
        topk=args.topk,
        experts=args.experts,
        chunk_tokens=args.chunk_tokens,
        sample=1,
    )
    data.update(scenario=scenario, seed=args.seed, clean_profile=clean, precision_checked=True, build_info=info)
    paths = view["write_timeline"](data, args.output / "diagnostics" / scenario / f"rank-{rank}.notify.json")
    log(f"diagnostic.done span_us={data['span_us']:.3f} svg={paths['svg']}")
    return dict(profile=clean, diagnostics=paths)


def notify_only_worker(local_rank, args):
    """Run all Notify cases with one process group and symmetric heap."""
    rank, world = _worker_identity(local_rank, args)
    log = _progress_logger("notify", rank)
    deep_ep = _import_installed_deep_ep()
    import torch
    import torch_npu  # noqa: F401
    from deep_ep._native import require_native
    from deep_ep._runtime import runtime_of

    info = json.loads(Path(deep_ep.__file__).with_name("_build_info.json").read_text())
    native = require_native("Notify-only validation")
    cases = _cases["from_args"](args)
    specifications = [
        native.notify_layout(
            rank,
            world,
            case["tokens"],
            args.topk,
            args.experts,
            case["chunk_tokens"],
            combine_only=True,
            num_cores=case.get("notify_num_cores", getattr(args, "notify_num_cores", 64)),
        )
        for case in cases
    ]
    if any("_peer_plan" in spec["tensors"] for spec in specifications):
        raise RuntimeError("Notify-only must not build Dispatch peer plans")
    torch.npu.set_device(local_rank)
    if local_rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    workspace = max(spec["workspace_bytes"] for spec in specifications)
    capacity = 2 * ((workspace + (2 << 20) - 1) // (2 << 20) * (2 << 20))
    log(f"buffer.init.begin bytes={capacity} cases={len(cases)} dispatch_plan=0")
    started = time.monotonic()
    buffer = deep_ep.ElasticBuffer(
        rank=rank,
        world_size=world,
        num_bytes=capacity,
        num_max_tokens_per_rank=max(case["tokens"] for case in cases),
        hidden=7168,
        num_topk=args.topk,
        explicitly_destroy=True,
        combine_only=True,
        bootstrap=getattr(args, "bootstrap", "default"),
    )
    init_seconds = time.monotonic() - started
    log("buffer.init.done")
    try:
        runtime = runtime_of(buffer)
        native.notify_check_mapping(runtime.native)
        for case, specification in zip(cases, specifications, strict=True):
            current = _cases["case_args"](args, case, len(cases) > 1)
            current.output.mkdir(parents=True, exist_ok=True)
            torch.npu.synchronize()
            log(f"case.begin id={case['case_id']}")
            run_notify_case(runtime, native, current, specification, rank, world, info, log)
            torch.npu.synchronize()
            log(f"case.done id={case['case_id']}")
    except BaseException:
        _log_failure(log)
        raise
    finally:
        log("buffer.destroy.begin")
        buffer.destroy()
        log("buffer.destroy.done")
    if len(cases) > 1:
        (args.output / f"rank-{rank}.json").write_text(
            json.dumps(
                dict(
                    rank=rank,
                    world_size=world,
                    operator="notify",
                    notify_only=True,
                    initialization_count=1,
                    buffer_init_seconds=init_seconds,
                    session_bytes=capacity,
                    cases=cases,
                    passed=True,
                    source_sha256=info["source_sha256"],
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def run_notify_case(runtime, native, args, specification, rank, world, info, log):
    import torch

    notify_cores = getattr(args, "notify_num_cores", 64)
    results = []
    # Reuse all output buffers, including across routes, to expose stale-count bugs.
    with torch.npu.stream(runtime.stream):
        outputs = {
            name: torch.empty(shape, dtype=getattr(torch, dtype), device="npu")
            for name, (shape, dtype) in specification["tensors"].items()
        }
    for mode in _ROUTES if args.route == "all" else (args.route,):
        log(f"route={mode} golden.begin (CPU)")
        routes = make_routes(
            world,
            args.tokens,
            args.topk,
            args.experts,
            0,
            mode,
            seed=args.seed,
            without_replacement=getattr(args, "without_replacement", False),
        )
        rank_dedup = getattr(args, "rank_dedup", False) or getattr(args, "no_server_dedup", False)
        golden = build_notify_golden(routes, args.experts, args.chunk_tokens, rank_dedup=rank_dedup)
        with torch.npu.stream(runtime.stream):
            indices = torch.tensor(routes[rank], dtype=torch.int64, device="npu")
        torch.npu.synchronize()

        def invoke(*, profile=None, sample=1):
            call_outputs = outputs if profile is None else dict(outputs, profile=profile, profile_sample=sample)
            with torch.npu.stream(runtime.stream):
                native.notify(
                    runtime.native,
                    indices,
                    call_outputs,
                    args.tokens,
                    args.topk,
                    args.experts,
                    runtime.stream.npu_stream,
                    args.chunk_tokens,
                    num_cores=notify_cores,
                    rank_dedup=rank_dedup,
                )
            return outputs

        def validate(result):
            check_notify_outputs(result, golden, rank, routes, args.experts)

        for iteration in range(args.repeat):
            log(f"route={mode} notify.begin iteration={iteration + 1}")
            result = invoke()
            torch.npu.synchronize()
            validate(result)
            log(f"route={mode} notify.precision.passed iteration={iteration + 1}")
        entry = dict(route=mode, checks=args.repeat, passed=True)
        if args.profile_op:
            entry.update(profile_notify(args, rank, mode, invoke, validate, info, log, stream=runtime.stream))
        results.append(entry)
    (args.output / f"rank-{rank}.json").write_text(
        json.dumps(
            dict(
                rank=rank,
                world_size=world,
                operator="notify",
                notify_only=True,
                rank_dedup=rank_dedup,
                notify_num_cores=notify_cores,
                dispatch_plan=False,
                tokens=args.tokens,
                topk=args.topk,
                experts=args.experts,
                chunk_tokens=args.chunk_tokens,
                seed=args.seed,
                without_replacement=getattr(args, "without_replacement", False),
                cases=results,
                passed=True,
                source_sha256=info["source_sha256"],
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_native_notify_combine_only_layout():
    """Host-only layout check when a compiled wheel is available; no NPU allocation."""
    import pytest
    from deep_ep._native import require_native

    try:
        native = require_native("Notify layout check")
    except NotImplementedError:
        pytest.skip("Notify layout check requires a compiled wheel")
    for world in (2, 128, 256, 512):
        for tokens in (4096, 16384):
            for chunk in (256, 512, 1024):
                normal = native.notify_layout(0, world, tokens, 6, world, chunk)
                lean = native.notify_layout(0, world, tokens, 6, world, chunk, combine_only=True)
                assert "_peer_plan" in normal["tensors"]
                assert "_peer_plan" not in lean["tensors"]
                assert normal["workspace_bytes"] == lean["workspace_bytes"]
                assert {k: v for k, v in normal["tensors"].items() if k != "_peer_plan"} == lean["tensors"]
                for cores in (32, 64):
                    assert native.notify_layout(0, world, tokens, 6, world, chunk, num_cores=cores) == normal
                    assert native.notify_profile_layout(cores) == (2, cores, 64)
    for cores in (0, 16, 63, 128):
        with pytest.raises(ValueError, match="32 or 64"):
            native.notify_layout(0, 128, 4096, 6, 512, num_cores=cores)
        with pytest.raises(ValueError, match="32 or 64"):
            native.notify_profile_layout(cores)


def test_hand_computed_routes():
    golden = build_notify_golden([[[0, 2]], [[1, 2]]], 4, 256)
    assert golden["rank_rows"] == [2, 2]
    assert golden["plans"][0]["dst"] == [0, ~2]
    assert golden["plans"][1]["dst"] == [~1, 3]
    duplicate = build_notify_golden([[[0, 0]], [[-1, -1]]], 4, 256)
    assert duplicate["rank_rows"] == [2, 0]
    assert duplicate["plans"][0]["dst"] == [0, ~1]


def test_group_chunk_order_and_token_ends():
    # Source 1 prefers itself; other sources use gateway 0. Each token has
    # two contributions on each of ranks 0 and 1. Two tokens per raw chunk.
    routes = [[[0, 2, 0, 2] for _ in range(6)] for _ in range(32)]
    golden = build_notify_golden(routes, 64, 2)
    for source in range(32):
        gateway = 1 if source == 1 else 0
        forward = golden["plans"][gateway]["forward_list"][source]
        assert [row[0] for row in forward] == [end for end in (3, 6, 9, 12, 15, 18) for _ in range(3)]
    # One 32-rank group. First raw chunk [0,2) splits into [0,1), [1,2).
    expected = []
    for token_ids in ((0,), (1,), (2, 3), (4, 5)):
        for source in range(32):
            if source == 1:
                continue
            expected.extend((source, token, slot) for token in token_ids for slot in (1, 3))
    backward = golden["plans"][1]["backward_list"][0]
    entries = expert_rows(routes, 64, 1)
    assert [entries[row] for _, row, _ in backward] == expected
    tails = {61: 1, 123: 2, 247: 3, 371: 4}
    assert [packed >> 32 for _, _, packed in backward] == [tails.get(i, 0) for i in range(len(backward))]


def add_notify_args(parser):
    cores = parser.add_mutually_exclusive_group()
    cores.add_argument("--notify-num-cores", type=int, choices=(32, 64), default=64, help="Notify AIV count")
    cores.add_argument(
        "--notify-num-cores-list",
        type=int,
        nargs="+",
        choices=(32, 64),
        help="Notify AIV counts tested in one SHMEM session",
    )


def add_dedup_args(parser, dest="rank_dedup"):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--rank-dedup",
        "--no-server-dedup",
        dest=dest,
        action="store_true",
        default=None,
        help="Use rank deduplication (default for EP8/16)",
    )
    group.add_argument(
        "--server-dedup",
        dest=dest,
        action="store_false",
        help="Use server deduplication (default for other EP sizes)",
    )


def resolve_dedup_args(args, dest="rank_dedup"):
    if getattr(args, dest) is None:
        setattr(args, dest, args.ep_size in (8, 16))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    _add_launch_args(parser)
    add_notify_args(parser)
    _cases["add_arguments"](parser, 4096)
    parser.add_argument("--bootstrap", choices=("default", "mpi"), default="default")
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--experts", type=int, default=512)
    parser.add_argument("--route", choices=(*_ROUTES, "all"), default="random")
    add_dedup_args(parser)
    add_random_route_args(parser)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    add_profile_args(parser, ("notify",))
    args = parser.parse_args()
    resolve_dedup_args(args)
    validate_random_route_args(parser, args)
    validate_profile_args(parser, args)
    try:
        cases = _cases["from_args"](args)
    except ValueError as exc:
        parser.error(str(exc))
    args.tokens, args.chunk_tokens = cases[0]["tokens"], cases[0]["chunk_tokens"]
    if not __debug__ or os.environ.get("PYTHONOPTIMIZE") or args.repeat < 1:
        parser.error("Enable assertions and use --repeat >=1")
    if not 1 <= args.tokens <= 16384 or not 1 <= args.topk <= 16:
        parser.error("tokens1..16384 and topk1..16 required")
    if args.experts < args.ep_size or args.experts > 1024 or args.experts % args.ep_size:
        parser.error("experts must be divisible by EP and <=1024")
    _launch(args, notify_only_worker)


_trace_ROOT = Path(__file__).resolve().parents[2]

_trace_VIEW = runpy.run_path(str(_trace_ROOT / "tests/elastic/notify_profile.py"))

_trace_DISPATCH = globals()

_trace_CONFIG = dict(rank=1, world=128, tokens=4096, topk=6, experts=512, chunk_tokens=256, sample=3)


def _trace_records(cores=32):
    result = []
    for core in range(cores):
        header = [
            _trace_VIEW["MAGIC"],
            _trace_VIEW["VERSION"],
            1,
            core,
            128,
            4096,
            6,
            512,
            256,
            3,
            cores,
            len(_trace_VIEW["PHASES"]),
            0,
            0,
            0,
            0,
        ]
        stamps = [2**54 + core * 100 + i * 2000 for i in range(len(_trace_VIEW["PHASES"]) + 1)]
        header[13:16] = [stamps[0] - 5000, stamps[0], 1]
        row = header + stamps
        result.append(row + [0] * (_trace_VIEW["WORDS"] - len(row)))
    return result


@pytest.mark.parametrize("cores", [32, 64])
def test_trace_notify_trace_abi_and_views(tmp_path, cores):
    text = (_trace_ROOT / "csrc/kernels/notify/profile.hpp").read_text(encoding="utf-8")
    phases = re.findall("^\\s+(\\w+) = (\\d+)U,", text, re.M)
    assert [name for name, _ in phases] == [*_trace_VIEW["PHASES"], "End"]
    assert [int(value) for _, value in phases] == list(range(len(phases)))
    data = _trace_VIEW["decode"](_trace_records(cores), **_trace_CONFIG)
    skew_us = (cores - 1) * 0.1
    assert data["events"][0]["duration_us"] == 2.0
    assert data["span_us"] == pytest.approx(len(_trace_VIEW["PHASES"]) * 2 + skew_us)
    assert all((e["mean_us"] == 2 for e in data["phases"]))
    assert data["alignment"]["wait_us_by_core"] == [5.0] * cores
    assert data["alignment"]["exit_skew_us"] == pytest.approx(skew_us)
    assert data["alignment"]["calibrated_clocks"] is False
    paths = _trace_VIEW["write_timeline"](data, tmp_path / "rank-1.notify.json")
    ET.parse(paths["svg"])
    raw = json.loads(Path(paths["json"]).read_text())
    assert raw["raw"] == _trace_records(cores)
    trace = json.loads(Path(paths["perfetto"]).read_text())
    slices = [e for e in trace["traceEvents"] if e["ph"] == "X"]
    assert len(slices) == cores * len(_trace_VIEW["PHASES"])
    assert slices[0]["dur"] == 2 and slices[0]["ts"] == 0
    assert {e["tid"] for e in slices} == set(range(cores))
    assert len(Path(paths["csv"]).read_text().splitlines()) == len(slices) + 1
    assert len([e for e in slices if e["cat"] == "global_wait"]) == 3 * cores


@pytest.mark.parametrize(
    "field,value",
    [(0, 0), (1, 9), (2, 0), (3, 31), (9, 2), (11, 0), (12, 64), (13, 0), (14, 1), (15, 0), (16, 0), (17, 1), (63, 1)],
)
def test_trace_notify_rejects_incomplete_stale_or_invalid_trace(field, value):
    rows = _trace_records()
    rows[0][field] = value
    with pytest.raises(ValueError):
        _trace_VIEW["decode"](rows, **_trace_CONFIG)


@pytest.mark.parametrize("flags", [0, 16, 31, 63])
def test_trace_notify_cache_flags_accept_old_and_count_cached_traces(flags):
    rows = _trace_records()
    for row in rows:
        row[12] = flags
    data = _trace_VIEW["decode"](rows, **_trace_CONFIG)
    assert data["cache_flags"]["forward_counts"] == bool(flags & 32)
    assert data["cache_flags"]["parse_hist"] == bool(flags & 1)


@pytest.mark.parametrize("cores", [32, 64])
@pytest.mark.parametrize("flush_l2", [False, True])
def test_trace_notify_profile_is_outside_clean_capture(monkeypatch, tmp_path, cores, flush_l2):
    from contextlib import nullcontext

    scope = _trace_DISPATCH["profile_notify"].__globals__
    calls = []
    rows = _trace_records(cores)
    for row in rows:
        row[9] = 1
    tensor = SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: rows))

    def allocate(shape, **kwargs):
        assert shape == (cores, _trace_VIEW["WORDS"])
        return tensor

    fake_torch = SimpleNamespace(
        int64="int64",
        int32="int32",
        zeros=allocate,
        empty=lambda *a, **kw: SimpleNamespace(zero_=lambda: calls.append("clear")),
        npu=SimpleNamespace(synchronize=lambda: calls.append("sync"), stream=lambda _: nullcontext()),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "deep_ep._native",
        SimpleNamespace(
            require_native=lambda _: SimpleNamespace(
                notify_profile_layout=lambda selected: (_trace_VIEW["VERSION"], selected, 64)
            )
        ),
    )

    def invoke(**kwargs):
        calls.append("trace" if kwargs.get("profile") is tensor else "clean")
        return "outputs"

    def capture(*args):
        calls.append("capture.begin")
        args[3]()
        calls.append("capture.end")
        return dict(kernel="notify_full_kernel")

    monkeypatch.setitem(scope, "profile_operator", capture)
    args = SimpleNamespace(
        output=tmp_path,
        ep_size=128,
        tokens=4096,
        topk=6,
        experts=512,
        chunk_tokens=256,
        seed=1,
        notify_num_cores=cores,
        profile_flush_l2=flush_l2,
    )
    result = _trace_DISPATCH["profile_notify"](
        args, 1, "random", invoke, lambda _: calls.append("validate"), {}, lambda _: None, stream=object()
    )
    expected = ["capture.begin", "clean", "capture.end", "sync", "trace", "sync", "validate"]
    if flush_l2:
        expected.insert(1, "clear")
        expected.insert(5, "clear")
    assert calls == expected
    assert Path(result["diagnostics"]["svg"]).is_file()


def test_trace_notify_metadata_checker_detects_wrong_outputs():
    torch = pytest.importorskip("torch")

    class NpuView:
        device = SimpleNamespace(type="npu")

        def __init__(self, tensor):
            self.tensor, self.dtype, self.shape = (tensor, tensor.dtype, tensor.shape)

        def cpu(self):
            return self.tensor

        def __getitem__(self, key):
            return NpuView(self.tensor[key])

        def is_contiguous(self):
            return self.tensor.is_contiguous()

    routes = [[[0, 2], [0, 0]], [[1, 2], [-1, -1]]]
    golden = _trace_DISPATCH["build_notify_golden"](routes, 4, 256)
    plan = golden["plans"][0]
    outputs = {}
    for name in (
        "server_mask",
        "dst",
        "forward_counts",
        "backward_counts",
        "chunk_ranges",
        "chunk_masks",
        "rank_rows",
        "gather_rows",
    ):
        outputs[name] = NpuView(
            torch.tensor(plan[name], dtype=torch.int64 if name in ("server_mask", "dst") else torch.int32)
        )
    for name, counts in (("forward_list", "forward_counts"), ("backward_list", "backward_counts")):
        width = 6 if name == "forward_list" else 3
        tensor = torch.full((2, 8, width), -987, dtype=torch.int32 if width == 6 else torch.int64)
        for i, n in enumerate(plan[counts]):
            if n:
                tensor[i, :n] = torch.tensor(plan[name][i], dtype=tensor.dtype)
        outputs[name] = NpuView(tensor)
    outputs["weight_return_meta"] = NpuView(torch.tensor([*plan["weight_return_meta"], -987, -987], dtype=torch.int32))
    outputs["expert_totals"] = NpuView(torch.tensor([3, 1, 2, 0], dtype=torch.int32))
    check = _trace_DISPATCH["check_notify_outputs"]
    check(outputs, golden, 0, routes, 4)
    for name, index in (("dst", 0), ("forward_counts", 0), ("weight_return_meta", 0), ("expert_totals", 0)):
        outputs[name].tensor.flatten()[index] += 1
        with pytest.raises(AssertionError, match=name):
            check(outputs, golden, 0, routes, 4)
        outputs[name].tensor.flatten()[index] -= 1


@pytest.mark.parametrize("fail_second", [False, True])
def test_notify_batch_reuses_one_buffer_and_cleans_up(tmp_path, monkeypatch, fail_second):
    from contextlib import nullcontext

    events, instances, visited = [], [], []
    test = sys.modules[__name__]
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            npu=SimpleNamespace(
                set_device=lambda rank: None, synchronize=lambda: events.append("sync"), stream=lambda _: nullcontext()
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    runtime = SimpleNamespace(native=object())
    native = SimpleNamespace(
        notify_layout=lambda rank, world, tokens, *a, **kw: dict(workspace_bytes=tokens * 1024, tensors={}),
        notify_check_mapping=lambda _: events.append("mapping"),
    )
    monkeypatch.setitem(sys.modules, "deep_ep._native", SimpleNamespace(require_native=lambda _: native))
    monkeypatch.setitem(sys.modules, "deep_ep._runtime", SimpleNamespace(runtime_of=lambda _: runtime))
    (tmp_path / "_build_info.json").write_text('{"source_sha256":"test"}')

    class Buffer:
        def __init__(self, **options):
            self.options = options
            instances.append(self)

        def destroy(self):
            events.append("destroy")

    monkeypatch.setattr(
        test,
        "_import_installed_deep_ep",
        lambda: SimpleNamespace(__file__=str(tmp_path / "__init__.py"), ElasticBuffer=Buffer),
    )
    monkeypatch.setattr(test, "_worker_identity", lambda *_: (0, 32))
    monkeypatch.setattr(test, "_progress_logger", lambda *_: lambda _: None)
    monkeypatch.setattr(test, "_log_failure", lambda _: events.append("failed"))
    args = SimpleNamespace(
        tokens=4096,
        tokens_list=[4096, 16384, 4096],
        chunk_tokens=256,
        chunk_tokens_list=[256, 512],
        notify_num_cores=64,
        notify_num_cores_list=[32, 64, 32],
        topk=6,
        experts=512,
        bootstrap="mpi",
        output=tmp_path / "output",
    )
    cases = _cases["from_args"](args)

    def run_case(actual_runtime, actual_native, current, spec, rank, world, info, log):
        assert actual_runtime is runtime and actual_native is native
        assert events[-1] == "sync"
        visited.append((current.tokens, current.chunk_tokens, current.notify_num_cores, current.output))
        if fail_second and len(visited) == 2:
            raise RuntimeError("case failure")
        (current.output / "rank-0.json").write_text('{"passed":true}')

    monkeypatch.setattr(test, "run_notify_case", run_case)
    if fail_second:
        with pytest.raises(RuntimeError, match="case failure"):
            notify_only_worker(0, args)
        assert len(visited) == 2
        assert not (args.output / "rank-0.json").exists()
    else:
        notify_only_worker(0, args)
        assert [(t, c, n) for t, c, n, _ in visited] == [
            (case["tokens"], case["chunk_tokens"], case["notify_num_cores"]) for case in cases
        ]
        assert len({output for *_, output in visited}) == len(cases)
        report = json.loads((args.output / "rank-0.json").read_text())
        assert report["initialization_count"] == 1 and report["cases"] == cases
    assert len(instances) == 1 and instances[0].options["num_max_tokens_per_rank"] == 16384
    assert instances[0].options["num_bytes"] == 32 << 20
    assert instances[0].options["bootstrap"] == "mpi"
    assert events.count("destroy") == 1 and events[-1] == "destroy"


def test_notify_cli_runs_only_metadata(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(sys.modules[__name__], "_launch", lambda args, worker: calls.append((args, worker)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_notify.py",
            "--tokens",
            "16384",
            "--chunk-tokens",
            "512",
            "--profile-op",
            "notify",
            "--output",
            str(tmp_path),
        ],
    )
    main()
    args, entry = calls.pop()
    assert entry is notify_only_worker and args.repeat == 1 and args.tokens == 16384
    assert args.notify_num_cores == 64
    for cores in (32, 64):
        monkeypatch.setattr(
            sys, "argv", ["test_notify.py", "--output", str(tmp_path), "--notify-num-cores", str(cores)]
        )
        main()
        assert calls.pop()[0].notify_num_cores == cores
    for flags in (("--repeat", "0"), ("--tokens", "32768"), ("--profile-op", "dispatch"), ("--notify-num-cores", "63")):
        monkeypatch.setattr(sys, "argv", ["test_notify.py", "--output", str(tmp_path), *flags])
        with pytest.raises(SystemExit):
            main()
    assert not calls
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_notify.py",
            "--output",
            str(tmp_path),
            "--tokens-list",
            "4096",
            "16384",
            "--chunk-tokens-list",
            "256",
            "512",
            "--notify-num-cores-list",
            "32",
            "64",
        ],
    )
    main()
    assert len(_cases["from_args"](calls.pop()[0])) == 8


def test_shmem_requires_explicit_dense_channels(tmp_path):
    import shutil
    import subprocess

    compiler = shutil.which("g++")
    if sys.platform != "linux" or compiler is None:
        pytest.skip("SHMEM host doubles require Linux and a C++17 compiler")
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "session_masks.cpp"
    source.write_text(
        r"""

#define main lifecycle_main
#include "session_test.cpp"
#undef main
int main()
{
    for (int world : {128, 256, 512}) {
        expected_world = world;
        for (bool dense : {false, true})
            for (bool combine : {true, false}) {
                setenv("DEEPEP_SHMEM_DENSE_QPS", dense ? "1" : "0", 1);
                if (!dense) {
                    const int previous_allocations = allocations;
                    try {
                        deepep::Session s(0, world, expected_bytes, "tcp://127.0.0.1:19091", 30, true, false, false,
                                          combine);
                        assert(false);
                    } catch (const std::runtime_error& error) {
                        assert(std::string(error.what()).find("DEEPEP_SHMEM_DENSE_QPS=1") != std::string::npos);
                    }
                    assert(!deepep::Session::busy() && initialized == 0 && allocations == previous_allocations);
                    continue;
                }
                deepep::Session s(0, world, expected_bytes, "tcp://127.0.0.1:19091", 30, true, false, false, combine);
                assert(s.combine_only() == combine);
                assert(!s.sparse_qp_enabled());
                s.close();
                assert(!deepep::Session::busy());
            }
    }
    setenv("DEEPEP_SHMEM_DENSE_QPS", "invalid", 1);
    fails([]() { deepep::Session s(0, 2, expected_bytes, "tcp://127.0.0.1:19091", 30, true); });
    assert(!deepep::Session::busy());
}
""",
        encoding="utf-8",
    )
    binary = tmp_path / "session_masks"
    command = [
        compiler,
        "-std=c++17",
        "-O2",
        "-I" + str(root / "tests/runtime/stubs"),
        "-I" + str(root / "tests/runtime"),
        "-I" + str(root / "csrc"),
        str(root / "csrc/runtime/session.cpp"),
        str(source),
        "-o",
        str(binary),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=30)


if __name__ == "__main__":
    main()
