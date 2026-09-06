#!/usr/bin/env bash
set -eu
TASK_ROOT=/home/xangma/pycbc-taylorf2-triton-20260906
TASK_PY=/home/xangma/pycbc-torch-split-20260905/venv/bin/python
cd "$TASK_ROOT/source"
exec taskset -c 8-11 "$TASK_PY" -B "$TASK_ROOT/harness/run.py" \
    --root "$TASK_ROOT/source" \
    --expected-sha 6829dc9bcba07ca7d3da44de7589cc4e9fb84da5 \
    --python "$TASK_PY" --out "$TASK_ROOT/full-v3-clean"
