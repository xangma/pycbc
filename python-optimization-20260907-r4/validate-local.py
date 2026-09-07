"""Revalidate the isolated R4 candidate and baseline expression in fresh workers."""
import ast
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/private/tmp/pycbc-torch-python-optimization-r4-20260907')
FOCUSED = [
    'test/test_torch_squared_norm.py',
    'test/test_array_torch_reductions.py',
    'test/test_chisq_precision.py',
    'test/test_sigmasq_series_precision.py',
    'test/test_live_batch_torch_peaks.py',
    'test/test_live_batch_veto_reuse.py',
    'test/test_live_batch_torch_fft_integration.py',
]
INTEGRATION = [
    'test/test_array.py',
    'test/test_torch_filter_pipeline.py',
    'test/test_torch_psd_pipeline.py',
    'test/test_torch_search_kernels.py',
]


def worker(role):
    import pycbc
    import pytest
    import torch
    from pycbc.types import array_torch

    assert Path(pycbc.__file__).resolve() == SOURCE / 'pycbc/__init__.py'
    assert Path(array_torch.__file__).resolve() == SOURCE / 'pycbc/types/array_torch.py'
    if role == 'baseline':
        data = (ROOT / 'baseline-array-torch.py').read_bytes()
        assert hashlib.sha256(data).hexdigest() == json.loads(
            (ROOT / 'lineage.json').read_text())['baseline_array_sha256']
        functions = [node for node in ast.parse(data).body
                     if isinstance(node, ast.FunctionDef) and node.name == 'squared_norm']
        assert len(functions) == 1
        exec(compile(ast.Module(body=functions, type_ignores=[]),
                     str(ROOT / 'baseline-array-torch.py'), 'exec'), array_torch.__dict__)
    print(json.dumps(dict(role=role, pycbc=pycbc.__file__, torch=torch.__version__,
                          python=sys.version, pid=os.getpid())), flush=True)
    return pytest.main(['-q', '-ra', '--tb=short',
                        *(INTEGRATION if role == 'integration' else FOCUSED)])


def main():
    if len(sys.argv) == 2:
        return worker(sys.argv[1])
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONPATH=str(SOURCE))
    results = dict(host=socket.gethostname(), cwd=str(SOURCE), pid=os.getpid(),
                   pgid=os.getpgrp(), python=sys.executable, environment={
                       key: env[key] for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                                'MKL_NUM_THREADS', 'PYTHONPATH')}, tests=[])
    for role in ('candidate', 'baseline', 'integration'):
        role_env = dict(env)
        if role == 'integration':
            role_env['PYCBC_TEST_SCHEME'] = 'torch:cpu'
        command = [sys.executable, __file__, role]
        started = time.time()
        with (ROOT / (role + '-tests.log')).open('x') as stream:
            completed = subprocess.run(command, cwd=SOURCE, env=role_env,
                                       stdout=stream, stderr=subprocess.STDOUT, timeout=300)
        results['tests'].append(dict(role=role, command=command,
                                     started=started, finished=time.time(),
                                     returncode=completed.returncode))
        (ROOT / 'local-validation.json').write_text(json.dumps(results, indent=2) + '\n')
        print(role, completed.returncode, flush=True)
        if completed.returncode:
            return completed.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
