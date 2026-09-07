#!/usr/bin/env python3
"""Replay frozen correlations to isolate FFT arithmetic; diagnostic only."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
CAMPAIGN = Path('/home/xangma/pycbc-torch-current-batch-sweep-20260907-r3')
CAPTURES = Path('/home/xangma/pycbc-torch-fft-precision-20260907')
SOURCE = CAMPAIGN / 'source'
sys.path.insert(0, str(SOURCE))
for key in list(os.environ):
    if key.startswith('PYCBC_TORCH_'):
        del os.environ[key]
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.update(OMP_DYNAMIC='FALSE', MKL_DYNAMIC='FALSE')
import numpy as np
import torch
from pycbc import scheme
from pycbc.types import Array, zeros
from pycbc.fft import mkl, torchfft
from pycbc.filter.matchedfilter import BatchCorrelator

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)
spec = importlib.util.spec_from_file_location('frozen_worker', CAMPAIGN / 'batch-worker.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
w.np = np
reference = json.loads((CAMPAIGN / 'runs/qual-b1-branch_standard/result.json').read_text())
args = argparse.Namespace(**reference['arguments'])
bank, psd, sigma, strain, geometry, injections, identity = w.workload(args)
for key in ('template_bank_sha256', 'psd_sha256', 'sigmasq_sha256', 'overwhitened_block_sha256'):
    assert identity[key] == reference['inputs'][key], key
n = args.size
report = {'purpose': __doc__, 'pid': os.getpid(), 'pgid': os.getpgrp(),
          'started': time.time(), 'source': w.source_identity(SOURCE),
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'numpy': np.__version__, 'torch': torch.__version__,
          'cuda': torch.version.cuda, 'input_hashes_verified': True,
          'affinity': sorted(os.sched_getaffinity(0)), 'rows': [],
          'qualification_authorized': False, 'state': 'running'}


def save():
    w.atomic_json(ROOT / 'replay.json', report)


def metrics(a, b):
    return w.row_metrics(a, b)


def cvalue(z):
    return [float(z.real), float(z.imag)]


with scheme.CPUScheme(num_threads=1):
    cpu_in = zeros(n, dtype=np.complex64)
    cpu_out = zeros(n, dtype=np.complex64)
    cpu_fft = mkl.IFFT(cpu_in, cpu_out)


def mkl_replay(corr):
    with scheme.CPUScheme(num_threads=1):
        cpu_in[:] = corr
        cpu_fft.execute()
        return cpu_out.numpy().copy()


plans = {}
for name, device, batch in [('b1-torch_cpu', 'cpu', 1), ('b8-torch_cpu', 'cpu', 8),
                            ('b1-torch_cuda', 'cuda:0', 1), ('b8-torch_cuda', 'cuda:0', 8)]:
    with scheme.TorchScheme(device):
        source = zeros(n * batch, dtype=np.complex64)
        target = zeros(n * batch, dtype=np.complex64)
        fft = torchfft.IFFT(source, target, nbatch=batch, size=n)
        plans[name] = (device, batch, source, target, fft)
report['plans'] = {name: {
    'class': type(p[4]).__name__, 'mkl': type(p[4]._mkl_plan).__name__,
    'fftw': type(p[4]._fftw_plan).__name__,
    'fftw_batch': type(p[4]._fftw_batch_plan).__name__,
    'promoted_batch': type(p[4]._promoted_batch_plan).__name__,
    'promoted_dtype': str(p[4]._promoted_batch_plan.source.dtype) if p[4]._promoted_batch_plan else None,
    'direct_batch': torchfft._can_use_direct_batch_ifft(p[4])} for name, p in plans.items()}
save()
standard = {}
for directory in sorted(CAPTURES.glob('b*')):
    if not directory.is_dir():
        continue
    diagnostic = json.loads((directory / 'diagnostic.json').read_text())
    frozen = json.loads((CAMPAIGN / f'runs/qual-b{diagnostic["batch"]}-{diagnostic["route"]}/result.json').read_text())
    for obs in diagnostic['observations']:
        block, tid = obs['block'], obs['template_id']
        row = tid - 1000
        path = directory / f'block{block}-template{tid}.npz'
        with np.load(path, allow_pickle=False) as data:
            actual, ref, corr, exact_corr = (data[k] for k in ('actual', 'reference', 'correlation', 'exact_correlation'))
        assert w.digest_bytes(actual) == frozen['pointwise_blocks'][block]['rows'][row]['sha256']
        assert w.digest_bytes(ref) == reference['pointwise_blocks'][block]['rows'][row]['sha256']
        expected_corr = bank[row].astype(np.complex128).conj() * strain[block].astype(np.complex128)
        assert np.array_equal(exact_corr[:len(expected_corr)], expected_corr)
        assert np.count_nonzero(exact_corr[len(expected_corr):]) == 0
        if (block, tid) not in standard:
            with scheme.CPUScheme(num_threads=1):
                x, y, z = Array(bank[row]), Array(strain[block]), zeros(n, dtype=np.complex64)
                BatchCorrelator([x], [z], len(x)).execute(y)
                std_corr = z.numpy().copy()
            std_ref = mkl_replay(std_corr)
            assert np.array_equal(std_ref, ref), f'Standard replay mismatch {block}/{tid}'
            standard[block, tid] = (std_corr, np.fft.ifft(std_corr.astype(np.complex128)) * n)
        std_corr, std_double = standard[block, tid]
        same_double = np.fft.ifft(corr.astype(np.complex128)) * n
        exact = np.fft.ifft(exact_corr) * n
        device, batch, source, target, fft = plans[directory.name]
        with scheme.TorchScheme(device):
            source._data.tensor.view(batch, n).copy_(torch.as_tensor(corr, device=device))
            fft.execute()
            replay = target._data.tensor.view(batch, n)[0].cpu().numpy().copy()
            promoted = torch.fft.ifft(torch.as_tensor(corr, device=device, dtype=torch.complex128), norm='forward').cpu().numpy()
        assert np.array_equal(replay, actual), f'Candidate replay mismatch {directory.name}/{block}/{tid}'
        same_mkl = mkl_replay(corr)
        pair_limit = w.POLICY['pointwise_complex']['atol'] + w.POLICY['pointwise_complex']['rtol'] * np.abs(ref.astype(np.complex128))
        failed = np.flatnonzero(np.abs(actual.astype(np.complex128) - ref) > pair_limit)
        samples = []
        for i in failed:
            fft_candidate = actual[i] - same_double[i]
            correlation_difference = same_double[i] - std_double[i]
            fft_reference = std_double[i] - ref[i]
            samples.append({'index': int(i), 'raw_reference': cvalue(ref[i]),
                            'raw_actual': cvalue(actual[i]), 'exact_double': cvalue(exact[i]),
                            'limit': float(pair_limit[i]),
                            'pair_error': float(abs(actual[i] - ref[i])),
                            'candidate_fft_error': float(abs(fft_candidate)),
                            'correlation_difference_effect': float(abs(correlation_difference)),
                            'reference_fft_error': float(abs(fft_reference)),
                            'signed_candidate_fft_error': cvalue(fft_candidate),
                            'signed_correlation_difference_effect': cvalue(correlation_difference),
                            'signed_reference_fft_error': cvalue(fft_reference),
                            'decomposition_residual': float(abs(fft_candidate + correlation_difference + fft_reference - (actual[i].astype(np.complex128) - ref[i]))),
                            'cancellation_condition': float(np.sum(np.abs(exact_corr)) / abs(exact[i])),
                            'common_snr_scale': float(4 * geometry['delta_f'] / np.sqrt(sigma[row]))})
        report['rows'].append({'cell': directory.name, 'block': block, 'template_id': tid,
            'capture_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'actual_hash_verified': True, 'reference_hash_verified': True,
            'actual_replay_bitwise_equal': True, 'standard_replay_bitwise_equal': True,
            'correlation_bitwise_equal_to_standard': bool(np.array_equal(corr, std_corr)),
            'correlation_different_samples': int(np.count_nonzero(corr != std_corr)),
            'correlation_max_abs_difference': float(np.max(np.abs(corr.astype(np.complex128)-std_corr))),
            'actual_vs_reference': metrics(actual, ref),
            'same_input_actual_vs_mkl': metrics(actual, same_mkl),
            'candidate_fft_vs_double': metrics(actual, same_double),
            'reference_fft_vs_double': metrics(ref, std_double),
            'correlation_difference_in_output': metrics(same_double, std_double),
            'candidate_vs_exact': metrics(actual, exact),
            'reference_vs_exact': metrics(ref, exact),
            'promoted_vs_exact': metrics(promoted, exact),
            'promoted_public_c64_vs_exact': metrics(promoted.astype(np.complex64), exact),
            'promoted_vs_same_input_double': metrics(promoted, same_double),
            'failed_samples': samples})
    save()
    print(json.dumps({'cell': directory.name, 'rows_complete': len(report['rows'])}), flush=True)
report.update(state='complete', finished=time.time(), unique_standard_rows=len(standard))
save()
print(json.dumps({'state': report['state'], 'rows': len(report['rows']), 'unique_standard_rows': len(standard)}), flush=True)
