"""Deterministic fixed strain for actual offline/Live CLI validation on len."""
import json
from pathlib import Path
import sys
import h5py
import numpy as np
from pycbc.types import TimeSeries
from pycbc.waveform import get_fd_waveform
from pycbc.frame import write_frame

root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=True)
sample_rate, duration, gps = 1024, 256, 1000000000
size = sample_rate * duration
rows = dict(mass1=[10., 12., 18., 24.], mass2=[8., 8., 10., 12.],
            spin1z=[0.] * 4, spin2z=[0.] * 4, f_lower=[30.] * 4,
            f_final=[500.] * 4)
with h5py.File(root / 'bank.hdf', 'w') as out:
    for key, values in rows.items():
        out[key] = values
    out['approximant'] = np.array(['TaylorF2'] * 4, dtype=h5py.string_dtype())
    out.attrs['parameters'] = list(rows) + ['approximant']
rng = np.random.default_rng(20260912)
strain = TimeSeries((rng.standard_normal(size) * 1.e-22).astype(np.float32),
                    delta_t=1. / sample_rate, epoch=gps)
hp, _ = get_fd_waveform(approximant='TaylorF2', mass1=10., mass2=8.,
                        delta_f=1. / duration, f_lower=30., f_final=500.,
                        distance=100., phase_order=-1)
hp.resize(size // 2 + 1)
injection = hp.cyclic_time_shift(128.).to_timeseries()
strain.data[:] += injection.numpy().astype(np.float32)
frame = root / f'H-FAKE-{gps}-{duration}.gwf'
write_frame(str(frame), 'H1:FAKE-STRAIN', strain)
common = ['--verbose', '--bank-file', str(root / 'bank.hdf'),
          '--frame-files', str(frame), '--channel-name', 'H1:FAKE-STRAIN',
          '--gps-start-time', str(gps + 8), '--gps-end-time', str(gps + duration - 8),
          '--sample-rate', str(sample_rate), '--low-frequency-cutoff', '30',
          '--strain-high-pass', '25', '--pad-data', '8',
          '--segment-length', '64', '--segment-start-pad', '16', '--segment-end-pad', '4',
          '--psd-estimation', 'median', '--psd-segment-length', '4',
          '--psd-segment-stride', '2', '--psd-inverse-length', '4',
          '--approximant', 'TaylorF2', '--order', '-1', '--snr-threshold', '5.5',
          '--newsnr-threshold', '3', '--chisq-bins', '16', '--cluster-window', '1',
          '--cluster-function', 'symmetric', '--fft-backends', 'mkl', '--batch-size', '4']
for label, device, flag in [('native-cuda', 'torch:cuda:0', '--enable-torchwave'),
                             ('reference-cuda', 'torch:cuda:0', '--disable-torchwave'),
                             ('reference-cpu', 'torch:cpu:1', '--disable-torchwave')]:
    argv = common + ['--processing-scheme', device, flag]
    (root / f'{label}.json').write_text(json.dumps({'shards': [{'argv': argv}] * 2}, indent=2))
(root / 'fixture.json').write_text(json.dumps({'seed': 20260912,
    'sample_rate': sample_rate, 'duration': duration, 'gps': gps,
    'noise_std': 1.e-22, 'injection': {'provider': 'PyCBC reference get_fd_waveform',
    'approximant': 'TaylorF2', 'mass1': 10., 'mass2': 8., 'distance_mpc': 100.,
    'coalescence_offset': 128.}, 'rows': rows}, indent=2))
print(root)
