#!/usr/bin/env bash
set -u
ROOT=/home/xangma/pycbc-remediation-20260912
cd "$ROOT/qualified" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$ROOT/qualified:$ROOT/torchwave-qualified/src"
export OMPI_MCA_btl=self,tcp OMPI_MCA_btl_tcp_if_include=lo PRTE_MCA_oob_tcp_if_include=lo
PY=/home/xangma/pycbc-torch-fixes-20260904-epKdaA/venv/bin/python
exec 9>"$ROOT/gpu.lock"
flock 9
printf 'PID=%s HOST=%s CWD=%s\n' "$$" "$(hostname)" "$PWD"
git rev-parse HEAD
failures=0
run_step() {
  name=$1
  shift
  printf 'START %s %s\n' "$name" "$(date -u +%FT%TZ)"
  "$@" > "$ROOT/logs/$name.log" 2>&1
  result=$?
  if [ "$result" -ne 0 ]; then failures=$((failures + 1)); fi
  printf '%s\n' "$result" > "$ROOT/logs/$name.exit"
  printf 'END %s exit=%s %s\n' "$name" "$result" "$(date -u +%FT%TZ)"
}
for route in native-cuda reference-cuda reference-cpu; do
  run_step "final-cli-$route" "$PY" tools/run_inspiral_campaign.py --manifest "$ROOT/cli-fixture/$route.json" --output "$ROOT/final-cli-$route" --cache-mib 32
done
run_step final-cli-contract-compare "$PY" "$ROOT/compare_offline_campaigns.py" --native-cuda "$ROOT/final-cli-native-cuda/receipt.json" --reference-cuda "$ROOT/final-cli-reference-cuda/receipt.json" --reference-cpu "$ROOT/final-cli-reference-cpu/receipt.json" --pycbc-contract --output "$ROOT/logs/final-cli-contract-comparison.json"
run_step final-live "$PY" "$ROOT/live_cli_smoke.py" --repo . --output "$ROOT/final-live" --device cuda --mpiexec /home/xangma/miniconda3/envs/pycbc3g/bin/mpiexec
run_step final-provider-cuda "$PY" tools/bench_torchwave_pipeline.py --device cuda:0 --batch-size 16 --flen 1025 --delta-f .5 --cold-runs 5 --warm-samples 20 --output "$ROOT/logs/final-provider-cuda.json"
run_step final-provider-cpu1 "$PY" tools/bench_torchwave_pipeline.py --device cpu --batch-size 16 --flen 1025 --delta-f .5 --cold-runs 5 --warm-samples 20 --output "$ROOT/logs/final-provider-cpu1.json"
run_step final-engine "$PY" tools/benchmarking/benchmark_gpu_search.py --size 131072 --num-templates 3 --tile-size 2 --iterations 20 --num-blocks 500 --sample-rate 1024 --output "$ROOT/logs/final-engine.json"
run_step final-profile "$PY" "$ROOT/profile_candidate_pipeline.py" --device cuda --output-dir "$ROOT/logs/final-profile"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
run_step final-provider-cpu4 "$PY" tools/bench_torchwave_pipeline.py --device cpu --batch-size 16 --flen 1025 --delta-f .5 --cold-runs 5 --warm-samples 20 --output "$ROOT/logs/final-provider-cpu4.json"
printf 'ALL STEPS FINISHED failures=%s\n' "$failures"
[ "$failures" -eq 0 ]
