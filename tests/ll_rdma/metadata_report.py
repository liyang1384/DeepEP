"""Export matched, completed M=4 OFF runs to one <=60000-byte report; no Torch/NPU."""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SHMEM_PAIR = {"4c671684e1698aec199619850a3120bfdf85c95e", "8e3fc610f91ae9e5e8cf13748cfb74826b272c2a"}
COMMON = ("world_size", "case", "route", "seed", "routing", "route_sha256", "warmup",
          "num_iters", "perf_tail", "active_aiv", "sampling_protocol", "exp_prefix_repeats",
          "per_iteration_exp", "per_iteration_all_reduce", "group_backend", "do_handle_copy",
          "correctness_repeats", "repeat_combine", "fresh_buffer_per_case", "trace_ops",
          "num_bytes", "use_1g_huge_pages", "logical_group", "transport")


def check_shmem_pair(project, revisions):
    if revisions.keys() != SHMEM_PAIR:
        raise ValueError("SHMEM A/B requires completed runs from both pinned SDKs")
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=project, text=True).strip()
    locks = []
    for sdk, source in revisions.items():
        if not re.fullmatch(r"[0-9a-f]{40}", source):
            raise ValueError("Invalid source commit in SHMEM A/B identity")
        lock = json.loads(git("show", f"{source}:dependencies.lock.json"))
        if (lock["shmem"].pop("revision") != sdk
                or git("rev-parse", f"{source}:third-party/shmem") != sdk):
            raise ValueError("Run SDK disagrees with its source lock or gitlink")
        locks.append(lock)
    changed = set(git("diff", "--name-only", *revisions.values(), "--").splitlines())
    if changed != {"dependencies.lock.json", "third-party/shmem"} or locks[0] != locks[1]:
        raise ValueError("SHMEM A/B requires project trees differing only in the SHMEM pin")


def render(project, runs, shmem_ab=False):
    project = Path(project).resolve()
    lines = ["# M=4 metadata OFF comparison", "",
             "Times are us. mean/p99 pool the per-rank tail samples; max-rank-mean is a separate statistic.",
             "FP8 refers to dispatch; combine uses BF16. One run is not a repeated A/B performance conclusion.",
             "Full raw samples, per-rank statistics and logs remain on the target. No samples are filtered here.", ""]
    common = sdk = None
    revisions = {}
    ab_config = None
    for run in runs:
        if Path(run).name != run or run in (".", ".."):
            raise ValueError("Supply run directory names, not paths")
        directory = project / "results/ll-ndr" / run
        report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        identity = json.loads((project / ".logs/ndr-one-click" / run / "node-0/execute/identity.json").read_text(encoding="utf-8"))
        if (report["status"] != "completed" or report["result_count"] != len(report["results"])
                or not report["results"] or identity["metadata_cores"] != 4
                or identity["reference_profile"] or identity["mode"] != "perf"):
            raise ValueError(f"{run}: incomplete run or wrong M/timer mode")
        this_sdk = (identity["sdk_revision"], identity["sdk_build"])
        if shmem_ab:
            revision, build = this_sdk
            if revision not in SHMEM_PAIR or build.get("source_patches") != []:
                raise ValueError(f"{run}: unexpected SHMEM revision or local SDK patches")
            if revisions.setdefault(revision, identity["git_revision"]) != identity["git_revision"]:
                raise ValueError(f"{run}: multiple project versions for one SDK")
            config = (build, identity["count_histogram"], identity["core_protocol"])
            if ab_config is not None and config != ab_config:
                raise ValueError(f"{run}: SDK build settings or LL configuration differ")
            ab_config = config
        elif sdk is not None and this_sdk != sdk:
            raise ValueError(f"{run}: SDK identity differs")
        sdk = this_sdk
        signature = []
        block = [f"## {run}", f"Source: {identity['git_revision']}; histogram={identity['count_histogram']}; protocol={identity['core_protocol']}",
                 f"Source fingerprint: {identity['source_sha256']}", f"Data: {directory}", "",
                 "| case | operation | mean | p99 | max-rank-mean | slowest rank | samples |",
                 "|---|---|---:|---:|---:|---:|---:|"]
        if shmem_ab:
            block.insert(2, f"SHMEM revision: {identity['sdk_revision']}")
        for result in report["results"]:
            meta = result["comparison"]
            if shmem_ab and meta.get("source_dirty") != "False":
                raise ValueError(f"{run}: SHMEM A/B requires a clean source checkout at sampling")
            if (meta["skip_check"] or meta["dispatch_only"] or not meta["perf_enabled"]
                    or meta["timer_mode"] is not None or meta["source_revision"] != identity["git_revision"]
                    or meta["sampling_protocol"] != "initial-reduce-exp100-batch-v3"
                    or not meta["route_sha256"]):
                raise ValueError(f"{run}: invalid OFF sampling/source identity")
            signature.append({key: meta[key] for key in COMMON})
            for op in ("dispatch", "combine"):
                value = result[op]
                if (value["summary_statistics_scope"] != "pooled_rank_tail_kernel_samples"
                        or len(value["ranks"]) != report["world_size"]):
                    raise ValueError(f"{run}: missing ranks or different statistics")
                block.append(f"| {result['case_id']} | {op} | {value['mean_kernel_us']:.3f} | {value['p99_kernel_us']:.3f} | "
                             f"{value['max_rank_mean_kernel_us']:.3f} | {value['slowest_rank']} | {value['kernel_samples']} |")
        if common is not None and signature != common:
            raise ValueError(f"{run}: shapes/routes/seeds/sampling/core configuration differ")
        if common is None:
            sdk_line = ("Matched SDK build settings: " + json.dumps(sdk[1], ensure_ascii=False)
                        if shmem_ab else "SDK: " + json.dumps(sdk, ensure_ascii=False))
            lines += [sdk_line, "Matched cases: " + json.dumps(signature, ensure_ascii=False), ""]
        common = signature
        lines += block + [""]
    if shmem_ab:
        check_shmem_pair(project, revisions)
    text = "\n".join(lines) + "\n"
    if len(text.encode("utf-8")) > 60000:
        raise ValueError("Report exceeds 60000 UTF-8 bytes; select fewer cases/runs. No output written.")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shmem-ab", action="store_true", help="Compare the fixed 4c671684/8e3fc610 SDK pair")
    args = parser.parse_args()
    text = render(args.project, args.runs, shmem_ab=args.shmem_ab)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    print(f"{args.output}: {len(text.encode('utf-8'))} UTF-8 bytes")


if __name__ == "__main__":
    main()
