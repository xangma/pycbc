#!/usr/bin/env bash
set -u
ROOT=/home/xangma/pycbc-remediation-20260912
cd "$ROOT/qualified" || exit 1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$ROOT/qualified:$ROOT/torchwave-qualified/src"
exec 9>"$ROOT/gpu.lock"
flock 9
PY=/home/xangma/pycbc-torch-fixes-20260904-epKdaA/venv/bin/python
printf 'PID=%s HOST=%s CWD=%s\n' "$$" "$(hostname)" "$PWD"
git rev-parse HEAD
$PY -c 'import pycbc, torchwave, torch; print(pycbc.__file__, torchwave.__file__, torch.__version__, torch.cuda.get_device_name())' || exit 1
$PY -m pytest -q test/test_gpu_search*.py test/test_torchwave*.py test/test_torch_evidence_tools.py test/test_inspiral_campaign.py test/test_live_batch_torch_peaks.py test/test_live_batch_torch_fft_integration.py test/test_torch_cuda_native_batch.py test/test_torch_cpu_native_batch.py test/test_frame_buffer_torch.py test/test_torch_strain_buffer_whitening.py --junitxml="$ROOT/logs/final-tests.xml"
rc=$?
$PY -m flake8 --select F401 pycbc/ test/ > "$ROOT/logs/flake8-f401-final.log" 2>&1
lc=$?
printf '%s %s\n' "$rc" "$lc" > "$ROOT/logs/final-tests.exit"
exit "$rc"
