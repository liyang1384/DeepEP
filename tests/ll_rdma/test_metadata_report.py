import importlib.util
import json
import subprocess
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("metadata_report", Path(__file__).with_name("metadata_report.py"))
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


def save(project, run, sha, mean=20):
    summary = project / "results/ll-ndr" / run / "summary.json"
    identity = project / ".logs/ndr-one-click" / run / "node-0/execute/identity.json"
    summary.parent.mkdir(parents=True); identity.parent.mkdir(parents=True)
    meta = {key: None for key in reporter.COMMON}
    meta.update(skip_check=False, dispatch_only=False, perf_enabled=True, timer_mode=None,
                source_revision=sha, sampling_protocol="initial-reduce-exp100-batch-v3", route_sha256="route1",
                case={"tokens": 8}, seed=0, active_aiv=56)
    perf = dict(summary_statistics_scope="pooled_rank_tail_kernel_samples", ranks=[{}, {}],
                mean_kernel_us=mean, p99_kernel_us=30, max_rank_mean_kernel_us=25, slowest_rank=1, kernel_samples=100)
    data = dict(status="completed", world_size=2, result_count=1,
                results=[dict(case_id=0, comparison=meta, dispatch=perf, combine=perf)])
    summary.write_text(json.dumps(data), encoding="utf-8")
    identity.write_text(json.dumps(dict(metadata_cores=4, reference_profile=False, mode="perf",
        sdk_revision="sdk", sdk_build={"version": 1}, git_revision=sha, count_histogram=True,
        core_protocol=41009, source_sha256=sha)), encoding="utf-8")
    return summary, identity


def test_compact_off_report_and_matched_route_guard(tmp_path):
    save(tmp_path, "a-perf", "source-a")
    path, _ = save(tmp_path, "b-perf", "source-b", 18)
    text = reporter.render(tmp_path, ["a-perf", "b-perf"])
    assert "20.000" in text and "18.000" in text and "source-b" in text
    assert "combine" in text and len(text.encode("utf-8")) <= 60000
    data = json.loads(path.read_text()); data["results"][0]["comparison"]["route_sha256"] = "route2"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="differ"):
        reporter.render(tmp_path, ["a-perf", "b-perf"])


@pytest.mark.parametrize("field,value", [("metadata_cores", 8), ("reference_profile", True), ("mode", "reference")])
def test_wrong_build_rejected(tmp_path, field, value):
    _, path = save(tmp_path, "a-perf", "a")
    data = json.loads(path.read_text()); data[field] = value; path.write_text(json.dumps(data))
    with pytest.raises(ValueError): reporter.render(tmp_path, ["a-perf"])


def test_size_limit_and_incomplete_run(tmp_path):
    path, _ = save(tmp_path, "a-perf", "a")
    data = json.loads(path.read_text()); data["status"] = "failed"; path.write_text(json.dumps(data))
    with pytest.raises(ValueError): reporter.render(tmp_path, ["a-perf"])
    data["status"] = "completed"; data["results"][0]["comparison"]["routing"] = "中" * 20000
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="60000"): reporter.render(tmp_path, ["a-perf"])


def test_explicit_shmem_ab_checks_both_pins_code_build_and_sampling(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-c", "user.name=fixture", "-c", "user.email=test@example.invalid",
                                        *args], cwd=tmp_path, text=True).strip()
    git("init", "-q")
    commits = {}
    for name, revision in zip(("a", "b"), sorted(reporter.SHMEM_PAIR)):
        (tmp_path/"dependencies.lock.json").write_text(json.dumps({"shmem": {"revision": revision}}))
        git("add", "dependencies.lock.json")
        git("update-index", "--add", "--cacheinfo", f"160000,{revision},third-party/shmem")
        git("commit", "-qm", name)
        commits[revision] = sha = git("rev-parse", "HEAD")
        path, ids = save(tmp_path, name, sha)
        data = json.loads(path.read_text())
        data["results"][0]["comparison"]["source_dirty"] = "False"
        path.write_text(json.dumps(data))
        identity = json.loads(ids.read_text())
        identity.update(sdk_revision=revision, sdk_build={"source_patches": [], "hns_1825_pi_on_chip": 1})
        ids.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="SDK identity differs"):
        reporter.render(tmp_path, ["a", "b"])
    report = reporter.render(tmp_path, ["a", "b"], shmem_ab=True)
    assert all(rev in report for rev in commits)
    with pytest.raises(ValueError, match="both pinned"):
        reporter.render(tmp_path, ["a"], shmem_ab=True)
    identity["sdk_build"]["hns_1825_pi_on_chip"] = 0
    ids.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="build settings"):
        reporter.render(tmp_path, ["a", "b"], shmem_ab=True)
    (tmp_path/"kernel.cpp").write_text("changed")
    git("add", "kernel.cpp")
    git("commit", "-qm", "also change kernel")
    commits[revision] = git("rev-parse", "HEAD")
    with pytest.raises(ValueError, match="trees differing only"):
        reporter.check_shmem_pair(tmp_path, commits)
