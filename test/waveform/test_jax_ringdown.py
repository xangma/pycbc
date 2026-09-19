# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Parity and device-residency tests for JAX ringdown waveforms."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import pycbc  # noqa: E402
from pycbc import scheme  # noqa: E402
from pycbc.types import FrequencySeries, TimeSeries  # noqa: E402
from pycbc.types.array_jax import JAXArrayData  # noqa: E402
from pycbc.waveform import ringdown  # noqa: E402
from pycbc.waveform import ringdown_jax  # noqa: E402

if not getattr(pycbc, "HAVE_JAX", False):
    pytest.skip("PyCBC built without JAX support", allow_module_level=True)


@pytest.fixture
def jax_ctx():
    ctx = scheme.JAXScheme()
    try:
        yield ctx
    finally:
        del ctx
        scheme.Scheme._single = None


def _parameters():
    return {
        "lmns": ["221", "331", "201"],
        "amp220": 1.2,
        "phi220": 0.4,
        "f_220": 250.0,
        "tau_220": 0.02,
        "amp330": 0.35,
        "phi330": -0.3,
        "f_330": 410.0,
        "tau_330": 0.012,
        "amp200": 0.15,
        "phi200": 0.8,
        "f_200": 180.0,
        "tau_200": 0.018,
        "inclination": 0.7,
        "azimuthal": 0.2,
    }


def _reject_host_transfer(_self):
    raise AssertionError("ringdown copied JAX data to the host unexpectedly")


def _reject_numpy_exp(*_args, **_kwargs):
    raise AssertionError("ringdown evaluated exponentials with NumPy")


def _reject_lal_harmonic(*_args, **_kwargs):
    raise AssertionError("ringdown evaluated harmonics with LAL")


def test_public_primitive_dispatches_to_jax_backend(jax_ctx, monkeypatch):
    expected = object(), object()

    def fake_damped_sinusoid(*args, **kwargs):
        assert args == (100.0, 0.1, 1.0, 0.0, [0.0])
        assert kwargs["m"] == 2
        return expected

    monkeypatch.setattr(
        ringdown_jax, "td_damped_sinusoid", fake_damped_sinusoid
    )
    with jax_ctx:
        actual = ringdown.td_damped_sinusoid(
            100.0, 0.1, 1.0, 0.0, [0.0]
        )
    assert actual == expected


def test_td_ringdown_jax_matches_cpu_without_host_transfer(jax_ctx, monkeypatch):
    parameters = _parameters() | {
        "delta_t": 1 / 4096,
        "t_final": 0.05,
        "taper": True,
        "dbeta": 0.1,
        "dphi": -0.2,
    }
    reference_plus, reference_cross = ringdown.get_td_from_freqtau(
        **parameters
    )
    expected_plus = reference_plus.numpy().copy()
    expected_cross = reference_cross.numpy().copy()

    with jax_ctx:
        monkeypatch.setattr(ringdown.numpy, "exp", _reject_numpy_exp)
        monkeypatch.setattr(
            ringdown.lal,
            "SpinWeightedSphericalHarmonic",
            _reject_lal_harmonic,
        )
        plus, cross = ringdown.get_td_from_freqtau(**parameters)

    for result in (plus, cross):
        assert isinstance(result, TimeSeries)
        assert isinstance(result._data, JAXArrayData)
    assert plus.delta_t == reference_plus.delta_t
    assert float(plus.start_time) == float(reference_plus.start_time)
    np.testing.assert_allclose(plus.numpy(), expected_plus, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(cross.numpy(), expected_cross, rtol=1e-12, atol=1e-14)


def test_fd_ringdown_jax_matches_cpu_without_host_transfer(jax_ctx, monkeypatch):
    parameters = _parameters() | {
        "f_lower": 20.0,
        "f_final": 1024.0,
        "t_0": 0.003,
    }
    reference_plus, reference_cross = ringdown.get_fd_from_freqtau(
        **parameters
    )
    expected_plus = reference_plus.numpy().copy()
    expected_cross = reference_cross.numpy().copy()

    with jax_ctx:
        monkeypatch.setattr(ringdown.numpy, "exp", _reject_numpy_exp)
        monkeypatch.setattr(
            ringdown.lal,
            "SpinWeightedSphericalHarmonic",
            _reject_lal_harmonic,
        )
        plus, cross = ringdown.get_fd_from_freqtau(**parameters)

    for result in (plus, cross):
        assert isinstance(result, FrequencySeries)
        assert isinstance(result._data, JAXArrayData)
    assert plus.delta_f == reference_plus.delta_f
    kmin = int(parameters["f_lower"] / plus.delta_f)
    assert np.count_nonzero(plus.numpy()[:kmin]) == 0
    np.testing.assert_allclose(plus.numpy(), expected_plus, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(cross.numpy(), expected_cross, rtol=1e-12, atol=1e-14)
