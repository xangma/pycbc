"""Warm public TaylorF2 batch timings, then separate diagnostic profiles."""
import argparse
import cProfile
import io
import json
import os
from pathlib import Path
import pstats
import statistics
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--device', required=True)
p.add_argument('--batch', type=int, required=True)
p.add_argument('--threads', type=int, default=1)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(a.root))
os.environ['PYCBC_TAYLORF2_TRITON'] = '0'
import torch
from pycbc import scheme
from pycbc.waveform import get_fd_waveform_batch

params = dict(approximant='TaylorF2',
              mass1=[1.4 + 0.05 * (i % 8) for i in range(a.batch)],
              mass2=[1.3 - 0.02 * (i % 8) for i in range(a.batch)],
              spin1z=0.02, spin2z=-0.01, distance=100.0, inclination=0.4,
              coa_phase=0.2, delta_f=0.25, f_lower=20.0, f_final=1024.0, f_ref=30.0)
torch.set_num_threads(a.threads)


def sync():
    if a.device == 'cuda':
        torch.cuda.synchronize()


def call():
    return get_fd_waveform_batch(**params)


with torch.no_grad(), scheme.TorchScheme(a.device):
    for _ in range(3):
        call()
    samples = []
    for _ in range(7):
        sync()
        start = time.perf_counter()
        result = call()
        sync()
        samples.append(time.perf_counter() - start)
    (a.out / 'timing.json').write_text(json.dumps(dict(device=a.device,
        batch=a.batch, threads=a.threads, samples_seconds=samples,
        waveforms_per_second=a.batch / statistics.median(samples)),indent=2))
    prof = cProfile.Profile()
    with prof:
        call()
        sync()
    prof.dump_stats(str(a.out / 'python.pstats'))
    output = io.StringIO()
    pstats.Stats(prof, stream=output).strip_dirs().sort_stats('cumulative').print_stats(35)
    (a.out / 'python.txt').write_text(output.getvalue())
    activities = [torch.profiler.ProfilerActivity.CPU]
    if a.device == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities,record_shapes=True) as prof:
        call()
        sync()
    (a.out / 'operators.txt').write_text(prof.key_averages().table(
        sort_by='self_cpu_time_total',row_limit=35))
    prof.export_chrome_trace(str(a.out / 'trace.json'))
print(a.out,flush=True)
