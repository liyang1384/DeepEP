"""Project-local writable paths; external CANN/provider paths are inherited."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def environment():
    env = dict(os.environ)
    paths = {
        "TMPDIR": ".tmp", "TMP": ".tmp", "TEMP": ".tmp", "XDG_CACHE_HOME": ".cache",
        "PIP_CACHE_DIR": ".cache/pip", "CUDA_CACHE_PATH": ".cache/cuda", "CCACHE_DIR": ".cache/ccache",
        "TORCH_EXTENSIONS_DIR": ".cache/torch_extensions", "TRITON_CACHE_DIR": ".cache/triton",
        "ASCEND_CACHE_PATH": ".cache/ascend", "ASCEND_WORK_PATH": ".logs/ascend",
        "ASCEND_PROCESS_LOG_PATH": ".logs/ascend", "SHMEM_LOG_PATH": ".logs/shmem",
    }
    for key, relative in paths.items():
        path = ROOT / relative
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    # JIT kernels and scratch stay in the project like every other cache, but an
    # explicit shared root (multi-rank/multi-node reuse) is never overridden.
    if "DEEPEP_JIT_CACHE_DIR" not in env:
        path = ROOT / ".cache/jit"
        path.mkdir(parents=True, exist_ok=True)
        env["DEEPEP_JIT_CACHE_DIR"] = str(path)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env
