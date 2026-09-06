#!/usr/bin/env bash
set -u
TASK_ROOT=/home/xangma/pycbc-taylorf2-triton-20260906
TASK_PY=/home/xangma/pycbc-torch-split-20260905/venv/bin/python
cd "$TASK_ROOT/source" || exit 1
export PYTHONPATH="$TASK_ROOT/source"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TRITON_CACHE_DIR="$TASK_ROOT/test-triton-cache"
export PYCBC_TAYLORF2_TRITON=1
date -u
git rev-parse HEAD
taskset -c 8-11 "$TASK_PY" -m pytest -q test/waveform/test_taylorf2_batch.py test/waveform/test_taylorf2_torch.py --junitxml="$TASK_ROOT/logs/taylorf2-junit.xml"
TASK_EXIT=$?
echo "$TASK_EXIT" > "$TASK_ROOT/logs/test-exit-code.txt"
date -u
exit "$TASK_EXIT"
