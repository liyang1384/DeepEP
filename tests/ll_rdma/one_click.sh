#!/usr/bin/env bash
# Copyright (c) 2026, Lu Lu
# Modified by ryan_li 2026
#
# Launch the NDR one-click entry with the project venv.
# Usage: bash tests/ll_rdma/one_click.sh <node-rank> <master-ip> <run-name> [options]
# Offline: bash tests/ll_rdma/one_click.sh --analyze-only RUN_DIR [--output-dir DIR]
set -euo pipefail
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$dir/../.." && pwd)"
python="${PYTHON:-$root/.venv-ll-rdma/bin/python}"
if [[ ! -x "$python" ]]; then
    echo "Missing project interpreter: $python (create .venv-ll-rdma first)" >&2
    exit 1
fi
exec "$python" "$dir/one_click.py" "$@"
