# Copyright (C) 2026 PyCBC contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Original CPU PSD routes retain exact values, metadata and JAX state."""

import os

import numpy as np
import pytest

pytest.importorskip("jax")

from pycbc import scheme  # noqa: E402
from pycbc.psd import estimate, analytical  # noqa: E402
from pycbc.types import FrequencySeries, TimeSeries  # noqa: E402


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
        pytest.skip("PSD reference tests require a JAX processing scheme")
    return specification.split(":", 1)[1]


def _assert_native(actual, expected, context):
    values, delta_f, epoch = expected
    result = actual.numpy()
    assert result.dtype == values.dtype
    assert result.shape == values.shape
    assert result.tobytes() == values.tobytes()
    assert actual.delta_f == delta_f
    assert actual.epoch == epoch
    assert scheme.mgr.state is context
    assert actual._data.device == context.jax_device


def _snapshot(series):
    return series.numpy().copy(), series.delta_f, series.epoch


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("average", ["mean", "median", "median-mean"])
def test_welch_reference_matches_original_cpu(
        device_spec, dtype, average, deterministic_original_fft):
    values = np.random.default_rng(381).normal(size=132).astype(dtype)
    options = dict(seg_len=32, seg_stride=16, avg_method=average)
    with scheme.CPUScheme():
        expected = _snapshot(estimate.welch(
            TimeSeries(values, delta_t=.03, epoch=1234567890.125), **options))
    with scheme.JAXScheme(device_spec, reference_operations=("welch",)) as ctx:
        series = TimeSeries(values, delta_t=.03, epoch=1234567890.125)
        _assert_native(estimate.welch(series, **options), expected, ctx)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("spectrum", ["invasd", "invpsd"])
def test_truncation_reference_matches_original_cpu(
        device_spec, dtype, spectrum, deterministic_original_fft):
    values = np.linspace(.5, 4, 33).astype(dtype)
    options = dict(max_filter_len=16, which_spectrum=spectrum,
                   low_frequency_cutoff=1.25, low_frequency_fill_value="fmin",
                   trunc_method="hann")
    with scheme.CPUScheme():
        expected = _snapshot(estimate.inverse_spectrum_truncation(
            FrequencySeries(values, delta_f=.5, epoch=123.25), **options))
    with scheme.JAXScheme(device_spec,
                          reference_operations=("inverse_spectrum_truncation",)) as ctx:
        series = FrequencySeries(values, delta_f=.5, epoch=123.25)
        _assert_native(estimate.inverse_spectrum_truncation(series, **options),
                       expected, ctx)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_interpolation_reference_matches_original_cpu(device_spec, dtype):
    values = np.array([.25, .8, 1.7, 5.1, 6.0], dtype=dtype)
    with scheme.CPUScheme():
        expected = _snapshot(estimate.interpolate(
            FrequencySeries(values, delta_f=.5), .3, length=9))
    with scheme.JAXScheme(device_spec, reference_operations=("interpolate",)) as ctx:
        series = FrequencySeries(values, delta_f=.5)
        _assert_native(estimate.interpolate(series, .3, length=9), expected, ctx)


@pytest.mark.parametrize("model", ["flat_unity", "aLIGOZeroDetHighPower"])
def test_analytical_reference_matches_original_cpu(device_spec, model):
    with scheme.CPUScheme():
        expected = _snapshot(analytical.from_string(model, 257, .5, 10.25))
    with scheme.JAXScheme(device_spec, reference_operations=("analytical_psd",)) as ctx:
        _assert_native(analytical.from_string(model, 257, .5, 10.25), expected, ctx)


def test_native_psd_errors_propagate_without_changing_jax_context(device_spec):
    with scheme.JAXScheme(device_spec, reference_operations=("welch",)) as ctx:
        values = TimeSeries(np.ones(64), delta_t=.25)
        with pytest.raises(ValueError, match="Invalid averaging method"):
            estimate.welch(values, seg_len=16, seg_stride=8, avg_method="invalid")
        assert scheme.mgr.state is ctx


def test_default_psd_calculations_do_not_call_cpu_worker(monkeypatch, device_spec):
    from pycbc.psd import reference_jax

    def reject_worker(*args, **kwargs):
        raise AssertionError("default JAX PSD selected the CPU worker")

    monkeypatch.setattr(reference_jax, "cpu_psd_reference", reject_worker)
    with scheme.JAXScheme(device_spec):
        data = TimeSeries(np.arange(64, dtype=np.float64), delta_t=.03)
        psd = estimate.welch(data, seg_len=16, seg_stride=8)
        psd = estimate.interpolate(psd, psd.delta_f, length=len(psd))
        estimate.inverse_spectrum_truncation(psd, 8)
        analytical.from_string("aLIGOZeroDetHighPower", 257, .5, 10.25)
