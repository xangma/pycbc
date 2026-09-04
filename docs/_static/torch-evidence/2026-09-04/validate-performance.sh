#!/usr/bin/env bash
set -euo pipefail
root=/home/xangma/pycbc-torch-finish-20260904
SRC="$root/tree"
PY=/home/xangma/pycbc-torch-fixes-20260904-epKdaA/venv/bin/python
OUT="$root/performance"
CPUSET=8-11
mkdir -p "$OUT"
cd "$SRC"
export PYTHONPATH="$SRC"
test "$(git rev-parse HEAD)" = "$(cat "$root/new-pr8.sha")"
{
    date -u
    hostname
    git rev-parse HEAD
    git status --short
    uname -a
    lscpu
    nvidia-smi
    sha256sum tools/bench_production_live_batch.py
    find pycbc -name '*.so' -type f -exec sha256sum {} +
} > "$OUT/environment-before.txt"
timeout 20m "$PY" "$SRC/tools/bench_production_live_batch.py" orchestrate \
    --root "$SRC" --python "$PY" \
    --output "$OUT/live-batch-cpu-cuda-t1.json" \
    --routes branch_standard torch_cpu torch_cpu_native torch_cuda torch_cuda_native \
    --batches 1 8 32 --size 131072 --num-blocks 3 \
    --threads 1 --replicates 3 --samples 5 --warmups 2 \
    --cuda-device 0 --affinity "$CPUSET" --seed 7101 --call-surface public \
    > "$OUT/live-batch-cpu-cuda-t1.log" 2>&1
timeout 10m "$PY" "$SRC/tools/bench_production_live_batch.py" orchestrate \
    --root "$SRC" --python "$PY" \
    --output "$OUT/live-batch-cpu-t4.json" \
    --routes branch_standard torch_cpu torch_cpu_native \
    --batches 1 32 --size 131072 --num-blocks 3 \
    --threads 4 --replicates 3 --samples 5 --warmups 2 \
    --affinity "$CPUSET" --seed 7101 --call-surface public \
    > "$OUT/live-batch-cpu-t4.log" 2>&1
for setting in torch_cpu_native:1 torch_cpu_native:4 torch_cuda_native:1; do
    route=${setting%:*}
    threads=${setting#*:}
    for batch in 8 32; do
        env OMP_DYNAMIC=FALSE OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads" \
            OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
            timeout 2m taskset -c "$CPUSET" "$PY" "$root/probe-native-route.py" \
            "$SRC" "$route" "$threads" "$batch" \
            > "$OUT/probe-$route-t$threads-b$batch.log" 2>&1
    done
done
{
    date -u
    nvidia-smi
    sha256sum "$OUT"/*.json
} > "$OUT/environment-after.txt"
echo 'Performance refresh PASS'
