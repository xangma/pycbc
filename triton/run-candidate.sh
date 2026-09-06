#!/usr/bin/env bash
set -u
TASK_ROOT=/home/xangma/pycbc-taylorf2-triton-20260906
TASK_PY=/home/xangma/pycbc-torch-split-20260905/venv/bin/python
TASK_QLTY=/home/xangma/pycbc-torch-finish-20260904/qlty/qlty-x86_64-unknown-linux-gnu/qlty
cd "$TASK_ROOT/source" || exit 1
export PYTHONPATH="$TASK_ROOT/source"
export PATH="/home/xangma/pycbc-torch-split-20260905/venv/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TRITON_CACHE_DIR="$TASK_ROOT/test-triton-cache-v3"
export PYCBC_TAYLORF2_TRITON=1
date -u
git rev-parse HEAD
taskset -c 8-11 "$TASK_PY" -m pytest -q test/waveform/test_taylorf2_batch.py test/waveform/test_taylorf2_torch.py --junitxml="$TASK_ROOT/logs/taylorf2-junit-v3.xml" > "$TASK_ROOT/logs/tests-v3.log" 2>&1
TASK_TEST_EXIT=$?
echo "tests=$TASK_TEST_EXIT"
echo "$TASK_TEST_EXIT" > "$TASK_ROOT/logs/tests-v3.exit"
test "$TASK_TEST_EXIT" -eq 0 || exit "$TASK_TEST_EXIT"
"$TASK_QLTY" check --no-upgrade-check --no-fix --sarif --no-progress --upstream 607bce53ead14f12af32552a5b2441d3bc667267 > "$TASK_ROOT/logs/candidate-qlty.sarif" 2> "$TASK_ROOT/logs/candidate-qlty.log"
TASK_QUALITY_EXIT=$?
echo "qlty=$TASK_QUALITY_EXIT"
echo "$TASK_QUALITY_EXIT" > "$TASK_ROOT/logs/candidate-qlty.exit"
test "$TASK_QUALITY_EXIT" -eq 0 || exit "$TASK_QUALITY_EXIT"
unset PYTHONPATH PYCBC_TAYLORF2_TRITON TRITON_CACHE_DIR
taskset -c 8-11 "$TASK_PY" -B "$TASK_ROOT/harness/run.py" \
    --root "$TASK_ROOT/source" \
    --expected-sha 6829dc9bcba07ca7d3da44de7589cc4e9fb84da5 \
    --python "$TASK_PY" --out "$TASK_ROOT/full-v3"
TASK_BENCH_EXIT=$?
echo "$TASK_BENCH_EXIT" > "$TASK_ROOT/logs/full-v3.exit"
date -u
exit "$TASK_BENCH_EXIT"
