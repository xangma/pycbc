#!/usr/bin/env bash
set -u
ROOT=/home/xangma/pycbc-remediation-20260912
printf 'PID=%s HOST=%s CWD=%s\n' "$$" "$(hostname)" "$PWD"
bash "$ROOT/run-final-tests.sh" > "$ROOT/logs/final-tests.log" 2>&1 || exit
bash "$ROOT/run-final-experiments.sh" > "$ROOT/logs/final-experiments.log" 2>&1 || exit
bash "$ROOT/run-final-generation.sh" > "$ROOT/logs/final-generation.log" 2>&1 || exit
cd "$ROOT/qualified" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$ROOT/qualified:$ROOT/torchwave-qualified/src"
flock "$ROOT/gpu.lock" /home/xangma/pycbc-torch-fixes-20260904-epKdaA/venv/bin/python "$ROOT/final-power-chisq-before-after/benchmark.py" --repo . --output "$ROOT/logs/final-power-chisq-before-after.json" > "$ROOT/logs/final-power-chisq-before-after.log" 2>&1
result=$?
printf '%s\n' "$result" > "$ROOT/logs/final-power-chisq-before-after.exit"
exit "$result"
