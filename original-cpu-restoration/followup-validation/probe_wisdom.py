"""Exercise real FFTW cold/warm cache integration in separate processes."""
import gc
import json
import pathlib
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import torch
from pycbc import scheme
from pycbc.fft import fftw, parser_support, torchfft, wisdom_cache
from pycbc.types import aligned

mode, cache_dir = sys.argv[1:]
size = 131072
calls = []
lock = torchfft._FFTW_PLANNING_LOCK
for name in ('import_single_wisdom_from_filename', 'export_single_wisdom_to_filename'):
    original = getattr(fftw, name)
    def operation(*args, _name=name, _original=original, **kwargs):
        assert lock._is_owned(), f'{_name} ran outside Torch planner lock'
        calls.append(_name)
        if mode == 'retry' and _name.startswith('export') and len(calls) == 1:
            raise OSError('injected initial persistence failure')
        return _original(*args, **kwargs)
    setattr(fftw, name, operation)
parser_support.import_wisdom_from_cli(SimpleNamespace(
    fftw_wisdom_cache=mode != 'disabled', fftw_wisdom_cache_dir=cache_dir))
rng = np.random.default_rng(7439)
values = (rng.normal(size=size) + 1j*rng.normal(size=size)).astype(np.complex64)
source = torch.from_numpy(aligned.zeros(size, np.complex64))
target = torch.from_numpy(aligned.zeros(size, np.complex64))
source.copy_(torch.from_numpy(values))
original_input = source.clone()
version_before = target._version
if mode == 'fallback':
    def fail_direct(*args, **kwargs):
        assert lock._is_owned()
        raise RuntimeError('injected direct-plan setup failure')
    torchfft._FFTWCPUDirectPlan = fail_direct
with scheme.TorchScheme('cpu', num_threads=1):
    start = time.monotonic()
    plan = torchfft._create_fftw_cpu_plan(size, False, direct=True, aligned=True,
                                        nthreads=1, source=source, target=target)
    elapsed = time.monotonic() - start
    assert plan is not None
    assert torch.equal(source, original_input), 'planning changed the live input'
    plan.execute(source, target)
    assert torch.equal(source, original_input), 'execution changed the live input'
    assert target._version > version_before
    reference = (np.fft.ifft(values.astype(np.complex128)) * size).astype(np.complex64)
    np.testing.assert_allclose(target.numpy(), reference, rtol=3e-5, atol=6e-4)
    entries = list(wisdom_cache._config.entries.values())
    assert all(entry.lock_file is None for entry in entries)
    if mode == 'retry':
        assert wisdom_cache.has_pending_export()
        parser_support.export_wisdom_from_cli(SimpleNamespace())
        assert calls == ['export_single_wisdom_to_filename'] * 2
        assert len(list(pathlib.Path(cache_dir).glob('*.wisdom'))) == 1
    assert not wisdom_cache.has_pending_export()
    if mode == 'cold':
        assert plan._measure_level == 1
        assert calls == ['export_single_wisdom_to_filename']
        assert len(list(pathlib.Path(cache_dir).glob('*.wisdom'))) == 1
    elif mode == 'warm':
        assert plan._measure_level == 0
        assert calls == ['import_single_wisdom_from_filename']
    elif mode == 'retry':
        assert plan._measure_level == 1
    elif mode == 'fallback':
        assert isinstance(plan, torchfft._FFTWCPUWorkPlan)
        assert not list(pathlib.Path(cache_dir).glob('*.wisdom'))
    else:
        assert plan._measure_level == 0
        assert not entries and not calls
    started = threading.Event()
    destroyed = threading.Event()
    def finalize():
        started.set()
        plan._finalizer()
        destroyed.set()
    with lock:
        thread = threading.Thread(target=finalize)
        thread.start()
        assert started.wait(2)
        assert not destroyed.wait(.05), 'destruction bypassed the Torch lock'
    thread.join(2)
    assert destroyed.is_set()
    result = dict(mode=mode, plan=type(plan).__name__, measure_level=getattr(plan, '_measure_level', None),
                  calls=calls, planning_seconds=elapsed, destructor_serialized=True,
                  max_abs_error=float(np.max(np.abs(target.numpy()-reference))),
                  source=fftw.__file__, library=fftw.float_lib._name)
    del plan
    gc.collect()
print(json.dumps(result))
