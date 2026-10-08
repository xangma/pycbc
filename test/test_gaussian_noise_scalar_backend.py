"""Regression tests for scalar likelihood backend separation."""

import types

import numpy
import pytest

from pycbc.inference.models import gaussian_noise
from pycbc.inference.models import marginalized_gaussian_noise
from pycbc.types import FrequencySeries


def _model(waveform, data, weight):
    model = types.SimpleNamespace()
    model.get_waveforms = lambda: {"H1": waveform}
    model._whitened_data = {"H1": data}
    model._weight = {"H1": weight}
    model._kmin = {"H1": 1}
    model._kmax = {"H1": len(waveform)}
    model._current_stats = types.SimpleNamespace()
    model.lognl = 0.0
    return model


@pytest.mark.parametrize("dtype", [numpy.complex64, numpy.complex128])
def test_scalar_cpu_likelihood_keeps_native_inner_path(monkeypatch, dtype):
    values = (numpy.arange(8) + 1j * numpy.arange(8)).astype(dtype)
    data_values = 2 * values - 1j
    weights = numpy.linspace(0.5, 1.25, 8).astype(values.real.dtype)
    waveform = FrequencySeries(values.copy(), delta_f=0.25)
    data = FrequencySeries(data_values, delta_f=0.25)
    weight = FrequencySeries(weights, delta_f=0.25)
    expected_waveform = waveform.copy()
    expected_waveform[1:] *= weight[1:]
    expected_hd = expected_waveform[1:].inner(data[1:])
    expected_hh = expected_waveform[1:].inner(expected_waveform[1:]).real

    def unexpected_fused(*args, **kwargs):
        raise AssertionError("scalar CPU path used the fused backend helper")

    monkeypatch.setattr("pycbc.inference.models.tools_jax.fused_inner_hd_hh", unexpected_fused)
    model = _model(waveform, data, weight)
    result = gaussian_noise.GaussianNoise._loglr(model)

    assert result == expected_hd.real - 0.5 * expected_hh
    assert model._current_stats.H1_cplx_loglr == expected_hd - 0.5 * expected_hh
    assert model._current_stats.H1_optimal_snrsq == expected_hh
    numpy.testing.assert_array_equal(waveform.numpy(), expected_waveform.numpy())


@pytest.mark.parametrize("dtype", [numpy.complex64, numpy.complex128])
def test_marginalized_scalar_cpu_keeps_native_inner_path(monkeypatch, dtype):
    values = (numpy.arange(8) + 1j * numpy.arange(8)).astype(dtype)
    data = FrequencySeries((2 * values - 1j).astype(dtype), delta_f=0.25)
    weight = FrequencySeries(numpy.linspace(0.5, 1.25, 8).astype(
        values.real.dtype), delta_f=0.25)
    waveform = FrequencySeries(values.copy(), delta_f=0.25)
    expected = waveform.copy()
    expected[1:] *= weight[1:]

    def unexpected_fused(*args, **kwargs):
        raise AssertionError("CPU marginalized path used fused helper")

    monkeypatch.setattr("pycbc.inference.models.tools_jax.fused_inner_hd_hh", unexpected_fused)
    model = types.SimpleNamespace(
        current_params={},
        all_ifodata_same_rate_length=True,
        waveform_generator=types.SimpleNamespace(
            generate=lambda **_: {"H1": waveform}),
        data={"H1": data},
        _kmin={"H1": 1},
        _kmax={"H1": len(waveform)},
        _weight={"H1": weight},
        _whitened_data={"H1": data},
        _current_stats=types.SimpleNamespace(),
    )
    expected_hd = expected[1:].inner(data[1:])
    expected_hh = expected[1:].inner(expected[1:]).real
    result = marginalized_gaussian_noise.MarginalizedPhaseGaussianNoise._loglr(model)

    assert numpy.array_equal(waveform.numpy(), expected.numpy())
    assert model._current_stats.H1_optimal_snrsq == expected_hh
    assert model._current_stats.maxl_phase == numpy.angle(expected_hd)
    assert result == marginalized_gaussian_noise.marginalize_likelihood(
        expected_hd, expected_hh, phase=True)
