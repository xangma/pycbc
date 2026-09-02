# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Batch correctness tests for NumPy-, CuPy-, and cuFFT-based FFTs."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from pycbc.types import Array, zeros


def _load_backend(alias, filename):
    path = Path(__file__).parents[1] / "pycbc" / "fft" / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["numpy", "cupy"])
def array_fft_backend(request, monkeypatch):
    if request.param == "numpy":
        from pycbc.fft import npfft

        yield npfft
        return

    fake_cupy = types.ModuleType("cupy")
    fake_cupy.__path__ = []
    fake_fft = types.ModuleType("cupy.fft")
    fake_fft.fft = np.fft.fft
    fake_fft.ifft = np.fft.ifft
    fake_fft.rfft = np.fft.rfft
    fake_fft.irfft = np.fft.irfft
    fake_cupy.fft = fake_fft
    fake_cupy.asarray = np.asarray
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setitem(sys.modules, "cupy.fft", fake_fft)

    alias = "pycbc.fft._test_cupyfft"
    try:
        yield _load_backend(alias, "cupyfft.py")
    finally:
        sys.modules.pop(alias, None)


@pytest.fixture
def fake_cufft(monkeypatch):
    class Plan:
        created = []

        def __init__(self, shape, itype, otype, batch=1):
            self.shape = shape
            self.itype = itype
            self.otype = otype
            self.batch = batch
            self.created.append(self)

    fake_fft = types.ModuleType("skcuda.fft")
    fake_fft.Plan = Plan
    fake_fft.fft = lambda invec, outvec, plan: None
    fake_fft.ifft = lambda invec, outvec, plan: None
    fake_skcuda = types.ModuleType("skcuda")
    fake_skcuda.__path__ = []
    fake_skcuda.fft = fake_fft
    monkeypatch.setitem(sys.modules, "skcuda", fake_skcuda)
    monkeypatch.setitem(sys.modules, "skcuda.fft", fake_fft)

    import pycbc.scheme

    monkeypatch.setattr(pycbc.scheme, "register_clean_cuda", lambda _: None)
    alias = "pycbc.fft._test_cufft"
    try:
        yield _load_backend(alias, "cufft.py")
    finally:
        sys.modules.pop(alias, None)


def _complex_values(rows, size, dtype, seed):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(rows, size))
        + 1j * rng.normal(size=(rows, size))
    ).astype(dtype)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("inverse", [False, True])
def test_array_backends_execute_independent_complex_batches(
    array_fft_backend, dtype, inverse
):
    rows, size = 3, 13
    values = _complex_values(rows, size, dtype, seed=2201)
    source = Array(values.ravel())
    target = zeros(rows * size, dtype=dtype)
    source_before = source.numpy().copy()

    if inverse:
        engine = array_fft_backend.IFFT(
            source, target, nbatch=rows, size=size
        )
        expected = np.fft.ifft(values, axis=-1) * size
    else:
        engine = array_fft_backend.FFT(
            source, target, nbatch=rows, size=size
        )
        expected = np.fft.fft(values, axis=-1)
    engine.execute()

    np.testing.assert_array_equal(source.numpy(), source_before)
    np.testing.assert_allclose(
        target.numpy().reshape(rows, size), expected,
        rtol=2e-6 if dtype == np.complex64 else 2e-13,
        atol=2e-6 if dtype == np.complex64 else 2e-13,
    )

    replacement = _complex_values(rows, size, dtype, seed=2202)
    source[:] = Array(replacement.ravel())
    engine.execute()
    if inverse:
        expected = np.fft.ifft(replacement, axis=-1) * size
    else:
        expected = np.fft.fft(replacement, axis=-1)
    np.testing.assert_allclose(
        target.numpy().reshape(rows, size), expected,
        rtol=2e-6 if dtype == np.complex64 else 2e-13,
        atol=2e-6 if dtype == np.complex64 else 2e-13,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("size", [12, 13])
def test_array_backends_execute_independent_real_batches(
    array_fft_backend, dtype, size
):
    rows = 3
    complex_dtype = np.complex64 if dtype == np.float32 else np.complex128
    rng = np.random.default_rng(2301 + size)
    real_values = rng.normal(size=(rows, size)).astype(dtype)
    frequency_values = _complex_values(
        rows, size // 2 + 1, complex_dtype, seed=2401 + size
    )

    real_input = Array(real_values.ravel())
    frequency_output = zeros(
        rows * (size // 2 + 1), dtype=complex_dtype
    )
    array_fft_backend.FFT(
        real_input, frequency_output, nbatch=rows, size=size
    ).execute()

    frequency_input = Array(frequency_values.ravel())
    real_output = zeros(rows * size, dtype=dtype)
    array_fft_backend.IFFT(
        frequency_input, real_output, nbatch=rows, size=size
    ).execute()

    tolerance = 2e-6 if dtype == np.float32 else 2e-13
    np.testing.assert_allclose(
        frequency_output.numpy().reshape(rows, -1),
        np.fft.rfft(real_values, axis=-1),
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        real_output.numpy().reshape(rows, size),
        np.fft.irfft(frequency_values, n=size, axis=-1) * size,
        rtol=tolerance,
        atol=tolerance,
    )


def test_array_backends_preserve_inplace_rejection(array_fft_backend):
    values = Array(np.zeros(24, dtype=np.complex64))
    engine = array_fft_backend.FFT(values, values, nbatch=3, size=8)

    with pytest.raises(NotImplementedError, match="in-place transforms"):
        engine.execute()


@pytest.mark.parametrize(
    ("input_dtype", "output_dtype", "inverse"),
    [
        (np.complex64, np.complex64, False),
        (np.float32, np.complex64, False),
        (np.complex64, np.complex64, True),
        (np.complex64, np.float32, True),
    ],
)
def test_cufft_batch_plans_use_transform_size(
    fake_cufft, input_dtype, output_dtype, inverse
):
    rows, size = 3, 14
    if input_dtype == np.complex64 and output_dtype == np.float32:
        input_size, output_size = size // 2 + 1, size
    elif input_dtype == np.float32 and output_dtype == np.complex64:
        input_size, output_size = size, size // 2 + 1
    else:
        input_size = output_size = size
    source = zeros(rows * input_size, dtype=input_dtype)
    target = zeros(rows * output_size, dtype=output_dtype)

    engine_type = fake_cufft.IFFT if inverse else fake_cufft.FFT
    engine = engine_type(source, target, nbatch=rows, size=size)

    assert engine.plan.shape == (size,)
    assert engine.plan.batch == rows


@pytest.mark.parametrize("getter_name", ["_get_fwd_plan", "_get_inv_plan"])
def test_cufft_plan_cache_keys_include_batch(fake_cufft, getter_name):
    getter = getattr(fake_cufft, getter_name)
    dtype = np.dtype(np.complex64)

    first = getter(dtype, dtype, 16, batch=2)
    assert getter(dtype, dtype, 16, batch=2) is first
    assert getter(dtype, dtype, 16, batch=3) is not first
