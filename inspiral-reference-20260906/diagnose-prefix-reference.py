#!/usr/bin/env python3
"""Measure serial and parallel float64 scans against a wider reference."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

root = Path(__file__).resolve().parent
source = root / 'source-v3'
sys.path.insert(0, str(source))
from pycbc import scheme
from pycbc.filter.matchedfilter import sigmasq_series
from pycbc.types import FrequencySeries

head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
assert head == '6c82155044d58f3344b281869d87745f71ba2285'
assert np.finfo(np.longdouble).eps < np.finfo(np.float64).eps
paths = [Path(__file__), source / 'test/test_sigmasq_series_precision.py',
         source / 'pycbc/filter/matchedfilter.py']
before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
frequency = np.arange(1048577) / 512
band_start, band_end = 30 * 512, 1999 * 512
amplitude = (np.maximum(frequency, 30) / 30) ** (-7 / 6)
waveform = (amplitude * np.exp(0.2j * frequency)).astype(np.complex128)
psd_values = 1 + (frequency / 300) ** 2
rows = []
for use_psd in (False, True):
    band = waveform[band_start:band_end]
    power = band.real**2 + band.imag**2
    if use_psd:
        power /= psd_values[band_start:band_end]
    truth = np.cumsum(power, dtype=np.longdouble) / 128
    serial = np.cumsum(power, dtype=np.float64) / 128
    row = dict(use_psd=use_psd, input_power_sha256=hashlib.sha256(power.tobytes()).hexdigest(),
               serial_reference_max_relative_error=float(np.max(abs(serial - truth) / truth)))
    for name, context in [('cpu', scheme.CPUScheme(1)),
                          ('torch_cpu', scheme.TorchScheme('cpu', num_threads=1)),
                          ('torch_cuda', scheme.TorchScheme('cuda', num_threads=1))]:
        with context:
            template = FrequencySeries(waveform, delta_f=1 / 512)
            psd = FrequencySeries(psd_values, delta_f=1 / 512) if use_psd else None
            actual = sigmasq_series(template, psd, 30, 1999).numpy()[band_start:band_end]
        row[name] = dict(max_relative_error_vs_wide=float(np.max(abs(actual - truth) / truth)),
                         max_relative_error_vs_serial=float(np.max(abs(actual - serial) / serial)))
    rows.append(row)
after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
assert before == after
record = dict(source_commit=head, float64_eps=float(np.finfo(np.float64).eps),
              reference_eps=float(np.finfo(np.longdouble).eps), samples=len(power),
              input_sha256=before, input_sha256_after=after, rows=rows)
with (root / 'prefix-reference-diagnostic.json').open('x') as output:
    json.dump(record, output, indent=2)
print(json.dumps(record))
