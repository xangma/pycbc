#!/usr/bin/env python3
"""Read-only reference environment and actual MKL dispatch qualification."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess

import numpy as np
import pycbc
from pycbc import fft, scheme
from pycbc.types import complex64, zeros

source = Path(pycbc.__file__).resolve().parents[1]
frame = Path('/home/xangma/pycbc_bench_repo/docs/_include/H-H1_LOSC_CLN_4_V1-1187007040-2048.gwf')
with scheme.CPUScheme(1):
    fft.backend_support.set_backend(['mkl'])
    a = zeros(32768, dtype=complex64)
    b = zeros(32768, dtype=complex64)
    a[0] = 1
    plan = fft.IFFT(a, b)
    plan.execute()
    assert np.allclose(b.numpy(), 1)
    result = dict(backend=fft.backend_support.get_backend().__name__,
                  ifft_class=type(plan).__module__ + '.' + type(plan).__name__,
                  scheme_threads=scheme.mgr.state.num_threads)
result.update(hostname=platform.node(), platform=platform.platform(),
              pycbc_file=str(pycbc.__file__), affinity=list(os.sched_getaffinity(0)),
              source_commit=subprocess.check_output(
                  ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
              versions={name: importlib.metadata.version(name) for name in
                        ['numpy', 'scipy', 'h5py', 'torch', 'lalsuite']},
              frame=dict(path=str(frame), bytes=frame.stat().st_size,
                         sha256=hashlib.sha256(frame.read_bytes()).hexdigest()),
              numerical_library_maps=sorted({line.split()[-1] for line in
                  Path('/proc/self/maps').read_text().splitlines()
                  if any(name in line for name in ['libmkl_', 'libgomp', 'libfftw', 'libopenblas'])}),
              cpu=subprocess.check_output(['lscpu'], text=True),
              gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
                                           '--format=csv,noheader'], text=True),
              perf_version=subprocess.check_output(['perf', '--version'], text=True),
              perf_event_paranoid=Path('/proc/sys/kernel/perf_event_paranoid').read_text().strip())
print(json.dumps(result, indent=2))
