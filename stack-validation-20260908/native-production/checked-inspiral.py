#!/usr/bin/env python3
"""Verify the runtime of one fresh, single-core inspiral worker."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import socket
import sys
import time


LOCK = Path('/home/xangma/pycbc-torch-performance-coordination-20260907/len-benchmark.lock')

PREFIXES = ('PYCBC_', 'OMP_', 'MKL_', 'OPENBLAS_', 'NUMEXPR_',
            'BLIS_', 'VECLIB_', 'TORCH_')
EXACT = ('PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'PYTHONHASHSEED',
         'CUDA_VISIBLE_DEVICES')
EXTRA_ENV = {'OMP_DYNAMIC': 'FALSE', 'CUDA_VISIBLE_DEVICES': '0'}


def relevant_environment(environment):
    return {k: v for k, v in sorted(environment.items())
            if k.startswith(PREFIXES) or k in EXACT}


def fixed_environment(config, source):
    return dict(config['environment'], **EXTRA_ENV, PYTHONPATH=str(source),
                PYTHONDONTWRITEBYTECODE='1')


def clean_environment(environment, config, source):
    result = {k: v for k, v in environment.items()
              if not (k.startswith(PREFIXES) or k in EXACT)}
    result.update(fixed_environment(config, source))
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_threadpoolctl():
    path = Path(__file__).resolve().parent / 'threadpoolctl.py'
    return dict(path=str(path), version='3.6.0', sha256=digest(path))


def snapshot(torch):
    # Introspection must not enable threadpoolctl's duplicate-OpenMP escape hatch.
    duplicate_policy = os.environ.get('KMP_DUPLICATE_LIB_OK')
    try:
        import threadpoolctl
    finally:
        if duplicate_policy is None:
            os.environ.pop('KMP_DUPLICATE_LIB_OK', None)
        else:
            os.environ['KMP_DUPLICATE_LIB_OK'] = duplicate_policy
    from pycbc import scheme
    affinity = sorted(os.sched_getaffinity(0))
    topology = {str(cpu): (Path('/sys/devices/system/cpu') /
                f'cpu{cpu}/topology/thread_siblings_list').read_text().strip()
                for cpu in affinity}
    state = scheme.mgr.state
    return dict(intra_op=torch.get_num_threads() if torch is not None else None,
                inter_op=torch.get_num_interop_threads() if torch is not None else None,
                torch_imported='torch' in sys.modules,
                affinity=affinity, smt_siblings=topology,
                scheme=type(state).__name__, device=str(getattr(state, 'device', '')),
                threadpools=threadpoolctl.threadpool_info(),
                threadpoolctl=dict(path=str(Path(threadpoolctl.__file__).resolve()),
                                   version=threadpoolctl.__version__,
                                   sha256=digest(threadpoolctl.__file__)),
                environment=relevant_environment(os.environ), observed_at=time.time())


def check(value, environment, bank=False, scheme='torch:cpu:1'):
    if value['threadpoolctl'] != expected_threadpoolctl():
        raise ValueError('Native threadpool observation imported another helper')
    expected_threads = 1 if scheme.startswith('torch:') else None
    if (value['intra_op'] != expected_threads or value['inter_op'] != expected_threads or
            value['affinity'] != [8] or value['smt_siblings'] != {'8': '8,72'}):
        raise ValueError('Observed thread counts or affinity differ from one-core setup')
    if value['environment'] != environment:
        raise ValueError('Unexpected inherited runtime environment')
    if not value['threadpools'] or any(p['num_threads'] != 1
                                     for p in value['threadpools']):
        raise ValueError('A loaded native thread pool is not single-threaded')
    expected_scheme, expected_device = {
        'cpu:1': ('CPUScheme', ''),
        'torch:cpu:1': ('TorchScheme', 'cpu'),
        'torch:cuda:0': ('TorchScheme', 'cuda:0'),
    }[scheme]
    if bank and (value['scheme'], value['device']) != (expected_scheme, expected_device):
        raise ValueError('Bank was consumed under an unexpected processing scheme')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--lock-fd', type=int, required=True)
    parser.add_argument('--scheme', choices=('cpu:1', 'torch:cpu:1', 'torch:cuda:0'),
                        required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Missing executable command')
    source = args.source.resolve()
    environment = fixed_environment(json.loads(args.config.read_text()), source)
    record = dict(state='running', pid=os.getpid(), hostname=socket.gethostname(),
                  command=command, source=str(source), started_at=time.time(),
                  helper_sha256=digest(__file__), environment=environment)
    previous_argv, bank, original = sys.argv, None, None

    def save():
        temporary = args.receipt.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(args.receipt)

    try:
        actual, expected = os.fstat(args.lock_fd), LOCK.stat()
        if args.lock_fd < 3 or (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError('Worker did not inherit benchmark lock')
        fcntl.flock(args.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record['inherited_lock'] = dict(fd=args.lock_fd, device=actual.st_dev, inode=actual.st_ino)
        if relevant_environment(os.environ) != environment:
            raise ValueError('Runner did not sanitize inherited environment')
        torch = None
        if args.scheme.startswith('torch:'):
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        import pycbc
        from pycbc.waveform.bank import FilterBank
        record.update(torch_version=str(torch.__version__) if torch is not None else None,
                      torch_cuda_version=torch.version.cuda if torch is not None else None,
                      imported_source=str(Path(pycbc.__file__).resolve().parent.parent),
                      processing_scheme=args.scheme,
                      source_modules={})
        if record['imported_source'] != str(source):
            raise ValueError('Worker imported another source checkout')
        if torch is not None:
            from pycbc.types import array_torch
            imported_module = str(Path(array_torch.__file__).resolve())
            if imported_module != str(source / 'pycbc/types/array_torch.py'):
                raise ValueError('Worker imported another Torch implementation')
            record['source_modules'][imported_module] = digest(imported_module)
        record['before_executable'] = snapshot(torch)
        check(record['before_executable'], environment, scheme=args.scheme)
        bank, original = FilterBank, FilterBank.__getitem__

        def first_bank(self, index):
            if 'at_first_bank' not in record:
                record['at_first_bank'] = snapshot(torch)
                check(record['at_first_bank'], environment, bank=True, scheme=args.scheme)
                if bank.__getitem__ is first_bank:
                    bank.__getitem__ = original
                save()
            return original(self, index)

        bank.__getitem__ = first_bank
        save()
        sys.argv = command
        try:
            runpy.run_path(command[0], run_name='__main__')
        except SystemExit as error:
            if error.code not in (None, 0):
                raise
        record['after_executable'] = snapshot(torch)
        check(record['after_executable'], environment, scheme=args.scheme)
        if 'at_first_bank' not in record:
            raise ValueError('Workload did not consume a bank')
        for name, module in list(sys.modules.items()):
            path = getattr(module, '__file__', None)
            if name.startswith('pycbc') and path and str(path).endswith('.so'):
                resolved = Path(path).resolve()
                if not resolved.is_relative_to(source):
                    raise ValueError('Worker imported a native extension outside pinned source')
                record['source_modules'][str(resolved)] = digest(resolved)
        if os.fstat(args.lock_fd).st_ino != expected.st_ino:
            raise ValueError('Worker lost benchmark lock')
        record['state'] = 'complete'
    except BaseException as error:
        record.update(state='failed', error=repr(error))
        raise
    finally:
        if bank is not None and original is not None:
            bank.__getitem__ = original
        sys.argv = previous_argv
        record['finished_at'] = time.time()
        save()


if __name__ == '__main__':
    main()
