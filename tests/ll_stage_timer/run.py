# Copyright (c) 2026, Lu Lu
# Modified by ryan_li 2026

"""Run the installed-wheel LL low-latency ST harness from this entry point.

one_click.py launches this path so the stage-timer/reference-profile case
matrix has one canonical location independent of the legacy elastic tests.
The implementation stays in tests/elastic/test_ep_low_latency.py; this module
only forwards the process arguments so both entries observe the same parser.
"""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "tests" / "elastic" / "test_ep_low_latency.py"

if __name__ == "__main__":
    sys.argv[0] = str(TEST)
    runpy.run_path(str(TEST), run_name="__main__")
