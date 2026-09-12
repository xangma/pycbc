#!/usr/bin/env bash
set -u
ROOT=/home/xangma/pycbc-remediation-20260912
cd "$ROOT/qualified" || exit 1
export PYTHONPATH="$ROOT/qualified:$ROOT/torchwave-qualified/src"
PY=/home/xangma/pycbc-torch-fixes-20260904-epKdaA/venv/bin/python
exec 9>"$ROOT/gpu.lock"
flock 9
printf 'PID=%s HOST=%s CWD=%s\n' "$$" "$(hostname)" "$PWD"
for mode in cpu1 cpu4 cuda; do
  for size in 2048 131072; do
    name="final-generation-$mode-n$size"
    "$PY" artifacts/torch-remediation-20260912/compare_native_taylorf2_generation.py --mode "$mode" --batch-size 16 --size "$size" --output "$ROOT/logs/$name.json" > "$ROOT/logs/$name.log" 2>&1
    printf '%s\n' "$?" > "$ROOT/logs/$name.exit"
    printf 'FINISHED %s\n' "$name"
  done
done
