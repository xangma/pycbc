#!/usr/bin/env python3
"""Replay fixed input arrays with a single host argument precision intervention."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
source = root / 'source-v4'
head = 'f2c0abe61e787a26f41208f489c62c877bbd5667'
output = root / 'chisq-coefficient-v4.json'
assert not output.exists()
assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == head
assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
config = json.loads((root / 'config.json').read_text())
os.sched_setaffinity(0, {config['core']})
os.environ.update({key: '1' for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']})
sys.path.insert(0, str(source))
import numpy as np
import torch
import triton
import triton.language as tl
from triton.runtime.jit import mangle_type
from pycbc import scheme
from pycbc.types import FrequencySeries
from pycbc.vetoes import chisq_torch
from pycbc.vetoes.chisq import power_chisq_at_points_from_precomputed

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

helper = root / 'diagnose-chisq-inputs.py'
spec = importlib.util.spec_from_file_location('captured_input_helper', helper)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
inputs = {str(p): digest(p) for p in [Path(__file__).resolve(), helper, root / 'source-v4.json',
    source / 'pycbc/vetoes/chisq.py', source / 'pycbc/vetoes/chisq_torch.py']}
record = dict(state='running', source_commit=head, pid=os.getpid(), triton_version=triton.__version__,
              specialization_plain=mangle_type(2*np.pi/2**21),
              specialization_constexpr=mangle_type(tl.constexpr(2*np.pi/2**21)),
              input_sha256=inputs, rows=[])
started = time.perf_counter()
original_kernel = chisq_torch._triton_pointwise_chisq_bin_kernel
launches = []
class ObservedKernel:
    def __init__(self, constexpr):
        self.constexpr = constexpr
    def __getitem__(self, grid):
        launch = original_kernel[grid]
        def call(*args, **kwargs):
            args = list(args)
            assert len(args) == 10
            coefficient = args[7]
            assert isinstance(coefficient, float)
            launches.append(dict(grid=list(grid), original_coefficient=coefficient,
                                 coefficient_fp32=float(np.float32(coefficient)), constexpr=self.constexpr))
            if self.constexpr:
                args[7] = tl.constexpr(coefficient)
            return launch(*args, **kwargs)
        return call
try:
    for origin in ['cpu', 'torch-cuda']:
        capture_root = root / f'chisq-capture-v4b-{origin}'
        status_path = capture_root / 'status.json'
        inputs[str(status_path)] = digest(status_path)
        status = json.loads(status_path.read_text())
        assert status['state'] == 'complete'
        assert status['source_info']['commit'] == head
        assert status['input_sha256'] == status['input_sha256_after']
        assert status['source_status_after'] == ''
        for row in status['captures']:
            data = {}
            for key, meta in row['arrays'].items():
                path = capture_root / Path(meta['path']).name
                assert digest(path) == meta['sha256']
                inputs[str(path)] = meta['sha256']
                data[key] = np.load(path, allow_pickle=False)
            corr, points, bins = data['corr'], data['indices'], data['bins']
            snr, norm = data['snrv'], row['snr_norm']
            active = np.ones(len(points), dtype=bool)
            if row['snr_threshold']:
                active = abs(snr * norm) > row['snr_threshold']
            points, snr = points[active], snr[active]
            if not len(points):
                record['rows'].append(dict(origin=origin, segment=row['segment'], active_points=0))
                continue
            sums = module.direct_bin_sums(corr, points.astype(np.float64), bins)
            expected = ((len(bins)-1)*np.sum(abs(sums)**2, axis=1)-abs(snr.astype(np.complex128))**2)*norm**2
            variants = {}
            counts = {}
            with scheme.TorchScheme('cuda:0'):
                correlation = FrequencySeries(corr, delta_f=1/512)
                for variant in ['original', 'torch_fallback', 'constexpr_coefficient']:
                    chisq_torch._HAS_TRITON = variant != 'torch_fallback'
                    chisq_torch._triton_pointwise_chisq_bin_kernel = ObservedKernel(variant == 'constexpr_coefficient')
                    before = len(launches)
                    result = power_chisq_at_points_from_precomputed(correlation, snr, norm, bins, points)
                    variants[variant] = result.numpy().copy()
                    counts[variant] = len(launches)-before
            for variant in variants:
                assert np.all(np.isfinite(variants[variant]))
                assert counts[variant] == (int(len(points)>1) if variant != 'torch_fallback' else 0)
            budget = 1e-5 + 1e-4*np.maximum(abs(expected), abs(variants['constexpr_coefficient']))
            record['rows'].append(dict(origin=origin, segment=row['segment'], points=points.tolist(),
                expected=expected.tolist(), recorded_chisq=data['chisq'][active].tolist(),
                variants={k:v.tolist() for k,v in variants.items()}, launches=counts,
                max_abs_error={k:float(np.max(abs(v-expected))) for k,v in variants.items()},
                violations={k:int(np.count_nonzero(abs(v-expected)>budget)) for k,v in variants.items()}))
    record.update(state='complete', launches=launches)
except BaseException as exc:
    record.update(state='failed', error=repr(exc))
    raise
finally:
    chisq_torch._HAS_TRITON = True
    chisq_torch._triton_pointwise_chisq_bin_kernel = original_kernel
    record.update(elapsed_wall_seconds=time.perf_counter()-started,
                  input_sha256_after={p:digest(p) for p in inputs},
                  source_status_after=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True))
    assert record['input_sha256_after'] == inputs
    assert record['source_status_after'] == ''
    output.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'state':record['state'],'rows':[{k:r[k] for k in ['origin','segment','max_abs_error','violations'] if k in r} for r in record['rows']]}))
