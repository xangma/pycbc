#!/usr/bin/env bash
set -eu
TASK_ROOT=/home/xangma/pycbc-taylorf2-triton-20260906
TASK_PY=/home/xangma/pycbc-torch-split-20260905/venv/bin/python
cd "$TASK_ROOT/source"
exec taskset -c 8-11 "$TASK_PY" -B "$TASK_ROOT/harness/run.py" \
    --root "$TASK_ROOT/source" \
    --expected-sha 68c87c0e99cd2e6c3797e72103559d5818074f01 \
    --python "$TASK_PY" --out "$TASK_ROOT/smoke-v2" \
    --batches 8 --delta-f .25 --replicates 1
