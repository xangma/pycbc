# Copyright (C) 2026 PyCBC contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Opt-in original conditioning preserves exact values and series metadata."""

import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from pycbc import scheme  # noqa: E402
from pycbc.filter import resample, resample_jax  # noqa: E402
from pycbc.types import TimeSeries  # noqa: E402


@pytest.fixture
def deterministic_original_fft(monkeypatch):
    # FFTW aligned/unaligned plans can round differently after independent
    # CPU allocations. Fix the original provider for this byte comparison.
    from pycbc.fft import backend_cpu

    monkeypatch.setattr(backend_cpu, "cpu_backend", "numpy")


@pytest.fixture
def device_spec():
    specification = os.environ.get("PYCBC_TEST_SCHEME", "jax:cpu")
    if specification == "jax":
        return "cpu"
    if not specification.startswith("jax:"):
        pytest.skip("Conditioning reference tests require a JAX scheme")
    return specification.split(":", 1)[1]


def _apply(operation, source, coefficients):
    if operation in ("highpass", "lowpass"):
        return getattr(resample, operation)(source, 100, filter_order=3,
                                           attenuation=.2)
    if operation in ("lfilter", "fir_zero_filter"):
        return getattr(resample, operation)(coefficients, source)
    return resample.resample_to_delta_t(
        source, 1 / 1024, method=operation.split("_", 1)[1])


def _snapshot(result):
    return (result.numpy().copy(), result.delta_t, result.start_time,
            getattr(result, "corrupted_samples", None))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("operation", [
    "highpass", "lowpass", "lfilter", "fir_zero_filter",
    "resample_ldas", "resample_butterworth",
])
def test_conditioning_reference_matches_original(
        device_spec, dtype, operation, monkeypatch, deterministic_original_fft):
    samples = np.random.default_rng(441).normal(size=512).astype(dtype)
    coefficients = np.array([.125, -.3, .7], dtype=dtype)
    with scheme.CPUScheme():
        original = _snapshot(_apply(
            operation, TimeSeries(samples, delta_t=1 / 2048, epoch=123.25),
            coefficients))

    selector = "resample" if operation.startswith("resample_") else operation
    with scheme.JAXScheme(device_spec, reference_operations=(selector,)) as ctx:
        source = TimeSeries(samples, delta_t=1 / 2048, epoch=123.25)
        with monkeypatch.context() as guard:
            def reject_default(*args, **kwargs):
                raise AssertionError("selected original route used a JAX kernel")

            for name in ("_circular_fir", "_butterworth_core",
                         "_highpass_lal_coefficients_core", "resample_ldas"):
                guard.setattr(resample_jax, name, reject_default)
            result = _apply(operation, source, coefficients)

        values, spacing, epoch, corrupted = original
        assert result.numpy().dtype == values.dtype
        assert result.numpy().tobytes() == values.tobytes()
        assert result.delta_t == spacing
        assert result.start_time == epoch
        assert getattr(result, "corrupted_samples", None) == corrupted
        assert result.data.array.devices() == {ctx.jax_device}
        assert scheme.mgr.state is ctx
        assert result.data is not source.data
        np.testing.assert_array_equal(source.numpy(), samples)


def test_conditioning_reference_keeps_none_epoch(device_spec):
    samples = np.arange(64, dtype=np.float32)
    coefficients = np.array([.2, .3, .5])
    with scheme.CPUScheme():
        original = resample.lfilter(
            coefficients, TimeSeries(samples, delta_t=.01, epoch=None))
        expected = original.numpy().copy()
    with scheme.JAXScheme(device_spec,
                          reference_operations=("lfilter",)) as ctx:
        result = resample.lfilter(
            coefficients, TimeSeries(samples, delta_t=.01, epoch=None))
        assert result.start_time is None
        assert result.dtype == expected.dtype
        assert result.numpy().tobytes() == expected.tobytes()
        assert scheme.mgr.state is ctx


def test_conditioning_reference_preserves_cached_cpu_setting(
        device_spec, monkeypatch):
    monkeypatch.setattr(resample, "USE_CACHING_FOR_LFILTER", True)
    samples = np.random.default_rng(442).normal(size=256)
    coefficients = np.array([.2, .3, .5])
    with scheme.CPUScheme():
        expected = resample.lfilter(
            coefficients, TimeSeries(samples, delta_t=.01)).numpy().copy()
    with scheme.JAXScheme(device_spec, reference_operations=("lfilter",)):
        result = resample.lfilter(
            coefficients, TimeSeries(samples, delta_t=.01))
        assert result.numpy().tobytes() == expected.tobytes()
        assert resample.USE_CACHING_FOR_LFILTER is True


@pytest.mark.parametrize("pass_zero", [False, True])
def test_firwin_reference_uses_original_scipy(
        device_spec, pass_zero, monkeypatch):
    from scipy.signal import firwin

    expected = firwin(63, .13, window=("kaiser", 7.5), pass_zero=pass_zero)
    with scheme.JAXScheme(device_spec,
                          reference_operations=("firwin",)) as ctx:
        def reject_default(*args, **kwargs):
            raise AssertionError("selected original design used a JAX kernel")

        monkeypatch.setattr(resample_jax, "_firwin_core", reject_default)
        result = resample_jax.firwin(63, .13, window=("kaiser", 7.5),
                                    pass_zero=pass_zero)
        assert np.asarray(result).tobytes() == expected.tobytes()
        assert result.devices() == {ctx.jax_device}
        assert scheme.mgr.state is ctx


@pytest.mark.parametrize("selectors", [("fft",), ("ifft",), ("fft", "ifft")])
def test_fir_fft_reference_routes_independently(
        device_spec, selectors, monkeypatch):
    samples = np.random.default_rng(443).normal(size=257)
    coefficients = np.array([.2, -.3, .5])
    with scheme.CPUScheme():
        expected = resample.lfilter(
            coefficients, TimeSeries(samples, delta_t=.01)).numpy().copy()
    calls = []
    original_fft = resample_jax._native_fft

    def record_native(values, operation, length):
        calls.append(operation)
        return original_fft(values, operation, length)

    def reject_selected(*args, **kwargs):
        raise AssertionError("selected original FFT used its JAX primitive")

    with scheme.JAXScheme(device_spec, reference_operations=selectors) as ctx:
        monkeypatch.setattr(resample_jax, "_native_fft", record_native)
        if "fft" in selectors:
            monkeypatch.setattr(jax.numpy.fft, "rfft", reject_selected)
        if "ifft" in selectors:
            monkeypatch.setattr(jax.numpy.fft, "irfft", reject_selected)
        result = resample.lfilter(
            coefficients, TimeSeries(samples, delta_t=.01))
        np.testing.assert_allclose(result.numpy(), expected,
                                   rtol=3e-12, atol=3e-12)
        assert calls.count("fft") == (2 if "fft" in selectors else 0)
        assert calls.count("ifft") == (1 if "ifft" in selectors else 0)
        assert result.data.array.devices() == {ctx.jax_device}
        assert scheme.mgr.state is ctx


def test_default_conditioning_never_starts_cpu_worker(device_spec, monkeypatch):
    from pycbc import reference_jax

    def reject_cpu(*args, **kwargs):
        raise AssertionError("default conditioning started a CPU worker")

    monkeypatch.setattr(reference_jax, "cpu_reference", reject_cpu)
    with scheme.JAXScheme(device_spec):
        for operation in ("highpass", "lowpass", "lfilter",
                          "fir_zero_filter", "resample_ldas"):
            result = _apply(operation, TimeSeries(np.ones(512),
                            delta_t=1 / 2048), np.array([.2, .3, .5]))
            assert isinstance(result.data.array, jax.Array)
