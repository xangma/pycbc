"""Bounded real MKL/CUDA contracts for the frozen production integration."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source'
LOCK = Path('/home/xangma/pycbc-torch-performance-coordination-20260907/len-benchmark.lock')
PYTHON = '/home/xangma/pycbc-torch-split-20260905/venv/bin/python'


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def main():
    record = dict(state='running', host=os.uname().nodename, cwd=str(ROOT),
                  pid=os.getpid(), pgid=os.getpgrp(), started=time.time(), runs=[])
    lock = LOCK.open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    record['lock'] = dict(path=str(LOCK), fd=lock.fileno(), inode=os.fstat(lock.fileno()).st_ino)
    status = ROOT / 'native-status.json'
    save(status, record)
    try:
        pins = json.loads((ROOT / 'source-pins.json').read_text())
        for name, expected in pins['tracked_files'].items():
            assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == expected, name
        assert subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True) == ''
        env = {k: v for k, v in os.environ.items() if not k.startswith(('PYCBC_', 'TORCH_', 'MKL_', 'OMP_'))}
        env.update(PYTHONPATH=str(SOURCE), PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MKL_DYNAMIC='FALSE',
                   MKL_THREADING_LAYER='GNU', OMP_DYNAMIC='FALSE',
                   OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                   CUDA_VISIBLE_DEVICES='0', PYTHONHASHSEED='0')
        preflight = [PYTHON, '-c', 'import pycbc,torch,triton; import pycbc.fft.mkl as m; '
                     'assert torch.cuda.is_available(); '
                     'print(pycbc.__file__,torch.__version__,torch.version.cuda,triton.__version__,m.lib._name)']
        record['preflight'] = subprocess.check_output(preflight, cwd=SOURCE, env=env, text=True)
        suites = [('contracts', ['test/test_mkl_function_cache.py', 'test/test_torch_offline_cuda_graph.py']),
                  ('regression', ['test/test_matched_filter_symm.py', 'test/test_torch_fft_writes.py',
                   'test/test_torch_fft_cpu_native.py', 'test/test_torch_fft_cuda_workspace.py',
                   'test/test_torch_batched_fft.py', 'test/test_torch_filter_pipeline.py',
                   'test/test_torch_peak_contracts.py', 'test/test_torch_search_kernels.py'])]
        for name, tests in suites:
            log, xml = ROOT / (name + '.log'), ROOT / (name + '.xml')
            command = ['taskset', '-c', '8', PYTHON, '-m', 'pytest', '-q', *tests,
                       '--junitxml=' + str(xml)]
            row = dict(name=name, command=command, log=str(log), started=time.time())
            record['runs'].append(row)
            save(status, record)
            with log.open('x') as output:
                child = subprocess.Popen(command, cwd=SOURCE, env=env, stdout=output,
                                         stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
                row['pid'] = child.pid
                save(status, record)
                try:
                    row['returncode'] = child.wait(timeout=900)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                    raise
            row['finished'] = time.time()
            tree = ET.parse(xml)
            row['counts'] = {key: sum(int(s.attrib.get(key, 0)) for s in tree.findall('.//testsuite'))
                             for key in ('tests', 'errors', 'failures', 'skipped')}
            save(status, record)
            assert row['returncode'] == 0, name
            assert row['counts']['tests'] > 0 and row['counts']['errors'] == row['counts']['failures'] == 0
            if name == 'contracts':
                assert row['counts']['skipped'] == 0, 'Native contract skipped'
        for name, expected in pins['tracked_files'].items():
            assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == expected, name
        record['source_unchanged'] = True
        record['state'] = 'complete'
    except BaseException as error:
        record.update(state='failed', error=repr(error))
        raise
    finally:
        record['finished'] = time.time()
        save(status, record)
        lock.close()


if __name__ == '__main__':
    main()
