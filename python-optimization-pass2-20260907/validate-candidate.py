"""Bounded fresh-worker contracts and integration validation for pass two."""
import ast
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/private/tmp/pycbc-torch-python-optimization-pass2-20260907')
BASELINE = 'd2647addb884ead3249914ebc980f3c132076d93'
FOCUSED = [
    'test/test_torch_squared_norm.py', 'test/test_array_torch_reductions.py',
    'test/test_chisq_precision.py', 'test/test_sigmasq_series_precision.py',
    'test/test_live_batch_torch_peaks.py', 'test/test_live_batch_veto_reuse.py',
    'test/test_live_batch_torch_fft_integration.py',
]
BROAD = [
    'test/test_array.py', 'test/test_torch_ops.py',
    'test/test_torch_backend_protocol.py', 'test/test_torch_backend_coercion.py',
    'test/test_torch_context_conversion.py', 'test/test_torch_filter_pipeline.py',
    'test/test_torch_psd_pipeline.py', 'test/test_torch_search_kernels.py',
]


def git(*args):
    return subprocess.check_output(['git', '-C', str(SOURCE), *args])


def source_record():
    assert git('rev-parse', 'HEAD').decode().strip() == BASELINE
    return dict(revision=BASELINE, status=git('status', '--porcelain').decode(),
                diff_sha256=hashlib.sha256(git('diff', '--binary', BASELINE)).hexdigest(),
                sha256={name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                        for name in ('pycbc/types/array_torch.py', *FOCUSED, *BROAD)})


def worker(role):
    import pycbc
    import pytest
    import torch
    from pycbc.types import array_torch

    assert Path(pycbc.__file__).resolve() == SOURCE / 'pycbc/__init__.py'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if role == 'baseline-focused':
        source = git('show', BASELINE + ':pycbc/types/array_torch.py')
        node = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'squared_norm')
        exec(compile(ast.Module(body=[node], type_ignores=[]),
                     '<exact-baseline-squared-norm>', 'exec'), array_torch.__dict__)
    print(json.dumps(dict(role=role, pycbc=pycbc.__file__, torch=torch.__version__,
                          python=sys.version, pid=os.getpid())), flush=True)
    return pytest.main(['-q', '-ra', '--tb=short',
                        *(BROAD if role == 'candidate-broad' else FOCUSED)])


def main():
    if len(sys.argv) == 2:
        return worker(sys.argv[1])
    state = dict(host=socket.gethostname(), cwd=str(SOURCE), pid=os.getpid(),
                 pgid=os.getpgrp(), started=time.time(), state='running',
                 timeout_per_worker_seconds=300, source_before=source_record(), runs=[])

    def save():
        temporary = ROOT / 'candidate-validation.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(ROOT / 'candidate-validation.json')

    def interrupted(signum, frame):
        raise InterruptedError(f'Received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    child = None
    save()
    try:
        for role in ('candidate-focused', 'baseline-focused', 'candidate-broad'):
            env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', PYTHONPATH=str(SOURCE))
            if role == 'candidate-broad':
                env['PYCBC_TEST_SCHEME'] = 'torch:cpu'
            run = dict(role=role, started=time.time(),
                       command=[sys.executable, str(Path(__file__).resolve()), role],
                       environment={key: env[key] for key in (
                           'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                           'MKL_NUM_THREADS', 'PYTHONPATH')})
            run['environment']['PYCBC_TEST_SCHEME'] = env.get('PYCBC_TEST_SCHEME')
            state['runs'].append(run)
            with (ROOT / (role + '.log')).open('x') as output:
                child = subprocess.Popen(run['command'], cwd=SOURCE, env=env,
                                         stdout=output, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                run.update(pid=child.pid, pgid=child.pid)
                save()
                run.update(returncode=child.wait(timeout=300), finished=time.time())
            print(role, run['returncode'], flush=True)
            assert run['returncode'] == 0
            child = None
            save()
        state['source_after'] = source_record()
        assert state['source_before'] == state['source_after']
        state['state'] = 'complete'
    except BaseException as exc:
        state.update(state='failed', error=repr(exc))
        raise
    finally:
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        state['finished'] = time.time()
        save()


if __name__ == '__main__':
    raise SystemExit(main())
