"""Descriptor ownership tests, plus an optional real-MKL numerical check."""
import ctypes
import gc
import importlib.util
from pathlib import Path
import threading
import types
import weakref

import numpy as np
import pytest

from pycbc import libutils, scheme
from pycbc.fft import core  # noqa: F401 -- load backends before installing fake MKL


class Call:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args):
        return self.function(*args)


class FakeMKL:
    def __init__(self):
        self.created, self.freed, self.computes = [], [], []
        self.failure = None
        self.live = {}
        for name, fn in (
            ('DftiCreateDescriptor_s_1d', self.create),
            ('DftiCreateDescriptor_d_1d', self.create),
            ('DftiSetValue', self.configure),
            ('DftiCommitDescriptor', self.commit),
            ('DftiComputeForward', self.compute),
            ('DftiComputeBackward', self.compute),
            ('DftiFreeDescriptor', self.free),
            ('DftiErrorMessage', lambda status: b'injected MKL error'),
        ):
            setattr(self, name, Call(fn))

    def create(self, pointer, domain, size):
        if self.failure == 'create':
            return 1
        value = len(self.created) + 10
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = value
        self.created.append(value)
        self.live[value] = {'size': size, 'settings': []}
        return 0

    def configure(self, descriptor, key, value):
        self.live[descriptor.value]['settings'].append((key, value))
        return int(self.failure == key or (key == 10 and value not in (39, 40)))

    def commit(self, descriptor):
        return int(self.failure == 'commit')

    def compute(self, descriptor, source, target):
        assert descriptor.value in self.live
        self.computes.append((descriptor.value, source, target))
        return int(self.failure == 'compute')

    def free(self, pointer):
        pointer = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))
        value = pointer[0]
        assert value in self.live
        self.freed.append(value)
        del self.live[value]
        pointer[0] = None
        return int(self.failure == 'free')


def vectors(size=16384):
    def vector(length, dtype):
        data = np.empty(length, dtype)
        return types.SimpleNamespace(_data=data, dtype=data.dtype,
                                     ptr=data.ctypes.data)
    # Special-method lookup uses the class, not instance attributes.
    class Vector(types.SimpleNamespace):
        def __len__(self):
            return self._data.size
    return tuple(Vector(**vars(v)) for v in (
        vector(size, np.float32), vector(size // 2 + 1, np.complex64)))


@pytest.fixture
def fake_mkl(monkeypatch):
    native = FakeMKL()
    monkeypatch.setattr(libutils, 'get_ctypes_library', lambda *a: native)
    path = Path(__file__).parents[1] / 'pycbc/fft/mkl.py'
    spec = importlib.util.spec_from_file_location('pycbc.fft._cache_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with scheme.CPUScheme(num_threads=1):
        yield module, native
        module.clear_function_cache()


def forward(module, pair):
    module.fft(*pair, 'single', 'real', 'complex')


def test_reuses_settings_with_fresh_pointers_without_retaining_arrays(fake_mkl):
    module, native = fake_mkl
    first, second = vectors(), vectors()
    refs = [weakref.ref(v._data) for v in first]
    forward(module, first)
    forward(module, second)
    assert len(native.created) == 1
    assert native.computes[0][1:] != native.computes[1][1:]
    settings = native.live[native.created[0]]['settings']
    assert settings == [(11, 44), (27, 1), (10, 39)]
    del first
    gc.collect()
    assert all(ref() is None for ref in refs)
    module.clear_function_cache()
    module.clear_function_cache()
    assert native.freed == native.created


@pytest.mark.parametrize('failure', ['create', 11, 27, 10, 'commit', 'compute'])
def test_native_error_evicts_and_frees_once(fake_mkl, failure):
    module, native = fake_mkl
    native.failure = failure
    with pytest.raises(RuntimeError, match='injected MKL error'):
        forward(module, vectors())
    assert not module._function_cache
    assert native.freed == native.created
    native.failure = None
    forward(module, vectors())
    assert len(module._function_cache) == 1


def test_free_error_never_retries_or_leaves_cache_entry(fake_mkl):
    module, native = fake_mkl
    forward(module, vectors())
    native.failure = 'free'
    with pytest.raises(RuntimeError):
        module.clear_function_cache()
    module.clear_function_cache()
    assert not module._function_cache
    assert native.freed == native.created


@pytest.mark.parametrize('case', ['length', 'threads', 'fork', 'overlap', 'stride'])
def test_unqualified_calls_use_independent_descriptors(fake_mkl, monkeypatch, case):
    module, native = fake_mkl
    pair = vectors(32 if case == 'length' else 16384)
    if case == 'threads':
        monkeypatch.setattr(scheme.mgr.state, 'num_threads', 2)
    elif case == 'fork':
        monkeypatch.setattr(module, '_function_cache_pid', -1)
    elif case == 'overlap':
        pair[1].ptr = pair[0].ptr + 4
    elif case == 'stride':
        pair[0]._data = np.empty(32768, np.float32)[::2]
        pair[0].ptr = pair[0]._data.ctypes.data
    forward(module, pair)
    forward(module, pair)
    assert len(native.created) == 2
    assert native.freed == native.created
    assert not module._function_cache


def test_worker_thread_does_not_share_main_thread_plan(fake_mkl):
    module, native = fake_mkl
    forward(module, vectors())
    errors = []
    def run():
        try:
            forward(module, vectors())
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    assert not errors
    assert len(native.created) == 2
    assert native.freed == [native.created[1]]


def test_busy_entry_uses_an_independent_plan_and_cannot_be_cleared(fake_mkl):
    module, native = fake_mkl
    forward(module, vectors())
    plan = next(iter(module._function_cache.values()))
    plan.busy = True
    try:
        with pytest.raises(RuntimeError, match='active'):
            module.clear_function_cache()
        forward(module, vectors())
    finally:
        plan.busy = False
    assert native.freed == [native.created[1]]
    assert len(module._function_cache) == 1


def test_capacity_does_not_evict_or_reuse_unrelated_plan(fake_mkl, monkeypatch):
    module, native = fake_mkl
    monkeypatch.setattr(module, '_FUNCTION_CACHE_LIMIT', 0)
    forward(module, vectors())
    assert native.freed == native.created
    assert not module._function_cache


def test_finalizer_skips_inherited_descriptor(fake_mkl, monkeypatch):
    module, native = fake_mkl
    forward(module, vectors())
    plan = next(iter(module._function_cache.values()))
    with monkeypatch.context() as patch:
        patch.setattr(module.os, 'getpid', lambda: -1)
        module.clear_function_cache()
        assert len(module._function_cache) == 1
        # Exercise the callback directly without detaching the parent's owner.
        module._free_function_descriptor(native.DftiFreeDescriptor,
                                         plan.descriptor.value, 123)
    assert not native.freed


def legacy_transform(mkl, direction, source, target):
    """Independent pre-cache function API, including its ignored enum error."""
    descriptor = ctypes.c_void_p()
    create = mkl.mkl_descriptor[source.precision]
    create.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_long]
    mkl.check_status(create(ctypes.byref(descriptor),
                           mkl.mkl_domain[source.kind][target.kind],
                           max(len(source), len(target))))
    try:
        mkl.check_status(mkl.lib.DftiSetValue(descriptor, 11, 44))
        mkl.check_status(mkl.lib.DftiSetValue(descriptor, 27, 1))
        # The original implementation silently left the default storage.
        mkl.lib.DftiSetValue(descriptor, 10, 54)
        mkl.check_status(mkl.lib.DftiCommitDescriptor(descriptor))
        compute = (mkl.lib.DftiComputeForward if direction == 'fft'
                   else mkl.lib.DftiComputeBackward)
        compute.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        mkl.check_status(compute(descriptor, source.ptr, target.ptr))
    finally:
        mkl.check_status(mkl.lib.DftiFreeDescriptor(ctypes.byref(descriptor)))


@pytest.mark.parametrize('direction,size,idtype,odtype,cached', [
    ('fft', 16384, np.float32, np.complex64, True),
    ('fft', 16777216, np.float32, np.complex64, True),
    ('ifft', 16777216, np.complex64, np.float32, True),
    ('fft', 16384, np.float64, np.complex128, False),
    ('ifft', 16384, np.complex128, np.float64, False),
    ('fft', 16384, np.complex64, np.complex64, False),
    ('ifft', 16384, np.complex64, np.complex64, False),
])
def test_native_mkl_reuse_matches_legacy_bytes(direction, size, idtype, odtype, cached):
    mkl = pytest.importorskip('pycbc.fft.mkl', exc_type=ImportError)
    from pycbc.types import Array, zeros
    rng = np.random.default_rng(70908)
    with scheme.CPUScheme(num_threads=1):
        mkl.clear_function_cache()
        try:
            for _ in range(3):
                real_input = np.issubdtype(idtype, np.floating)
                real_output = np.issubdtype(odtype, np.floating)
                ilen = size // 2 + 1 if real_output else size
                olen = size // 2 + 1 if real_input else size
                values = rng.standard_normal(ilen).astype(idtype)
                if not real_input:
                    values.imag = rng.standard_normal(ilen)
                    if real_output:
                        values[0] = values[0].real
                        values[-1] = values[-1].real
                source = Array(values)
                expected = zeros(olen, dtype=odtype)
                actual = zeros(olen, dtype=odtype)
                before = source.numpy().tobytes()
                legacy_transform(mkl, direction, source, expected)
                assert source.numpy().tobytes() == before
                getattr(mkl, direction)(source, actual, source.precision,
                                        source.kind, actual.kind)
                assert actual.numpy().tobytes() == expected.numpy().tobytes()
                assert source.numpy().tobytes() == before
                assert len(mkl._function_cache) == int(cached)
        finally:
            mkl.clear_function_cache()
