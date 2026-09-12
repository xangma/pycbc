#!/usr/bin/env bash
set -eu
ROOT=/home/xangma/pycbc-remediation-20260912/identity-relocation
PY=/home/xangma/pycbc-torch-fixes-20260904-epKdaA/venv/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$ROOT/pycbc:$ROOT/torchwave/src"
exec 9>/home/xangma/pycbc-remediation-20260912/gpu.lock
flock 9
cd "$ROOT/pycbc"
printf 'PID=%s HOST=%s CWD=%s\n' "$$" "$(hostname)" "$PWD"
$PY -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())'
$PY -m pytest -q test/test_torchwave*.py --junitxml="$ROOT/logs/pycbc-tests.xml"
cd "$ROOT/torchwave"
$PY -m pytest -q tests/test_provenance.py --junitxml="$ROOT/logs/torchwave-tests.xml"
$PY -m flake8 --select F401 "$ROOT/pycbc/pycbc/waveform/torchwave.py" "$ROOT/pycbc/test/test_torchwave_provider_identity.py" src/torchwave/provenance.py tests/test_provenance.py
printf '0\n' > "$ROOT/logs/tests.exit"
