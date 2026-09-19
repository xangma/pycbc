# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""JAX integration and autodiff tests for waveform utility helpers."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from pycbc import scheme  # noqa: E402
from pycbc.types import FrequencySeries, TimeSeries  # noqa: E402
from pycbc.types.backend import backend_array, wrap_backend_array  # noqa: E402
from pycbc.waveform import utils  # noqa: E402


@pytest.fixture
def jax_ctx():
    ctx = scheme.JAXScheme()
    try:
        yield ctx
    finally:
        del ctx
        scheme.Scheme._single = None


@pytest.mark.parametrize("series_type", (TimeSeries, FrequencySeries))
def test_scheme_cast_preserves_attributes(jax_ctx, series_type):
    delta_name = "delta_t" if series_type is TimeSeries else "delta_f"
    values = np.linspace(1.0, 5.0, 16)
    source = series_type(values, epoch=123, **{delta_name: 0.25})

    with jax_ctx:
        actual = utils.scheme_cast_series(source)

    assert backend_array(actual, "jax") is not None
    if series_type is TimeSeries:
        assert actual.start_time == source.start_time
    else:
        assert actual.epoch == source.epoch
    assert getattr(actual, delta_name) == getattr(source, delta_name)
    np.testing.assert_allclose(actual.numpy(), source.numpy(), rtol=1e-12)


def test_apply_fseries_time_shift_matches_cpu(jax_ctx):
    samples = np.sin(np.arange(64) / 7.0)
    source = TimeSeries(samples, delta_t=1 / 1024, epoch=10)
    reference = utils.apply_fd_time_shift(
        source.to_frequencyseries(), 10.125
    )

    with jax_ctx:
        jax_source = TimeSeries(samples, delta_t=1 / 1024, epoch=10)
        actual = utils.apply_fd_time_shift(
            jax_source.to_frequencyseries(), 10.125
        )

    assert backend_array(actual, "jax") is not None
    np.testing.assert_allclose(
        actual.numpy(),
        reference.numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("side", ("left", "right"))
def test_td_taper_matches_cpu(jax_ctx, side):
    samples = np.linspace(1.0, 2.0, 64)
    source = TimeSeries(samples, delta_t=0.25, epoch=-4)
    reference = utils.td_taper(source, -2, 2, side=side)

    with jax_ctx:
        jax_source = TimeSeries(samples, delta_t=0.25, epoch=-4)
        actual = utils.td_taper(jax_source, -2, 2, side=side)

    np.testing.assert_allclose(actual.numpy(), reference.numpy(), rtol=1e-14)


@pytest.mark.parametrize("side", ("left", "right"))
def test_fd_taper_matches_cpu(jax_ctx, side):
    samples = np.linspace(1.0, 2.0, 64).astype(np.complex128)
    source = FrequencySeries(samples, delta_f=0.25)
    reference = utils.fd_taper(source, 2, 6, side=side)

    with jax_ctx:
        jax_source = FrequencySeries(samples, delta_f=0.25)
        actual = utils.fd_taper(jax_source, 2, 6, side=side)

    np.testing.assert_allclose(actual.numpy(), reference.numpy(), rtol=1e-14)


@pytest.mark.parametrize("copy", (False, True))
@pytest.mark.parametrize("kmin", (0, 2))
def test_time_shift_autodiff(jax_ctx, copy, kmin):
    samples = np.array([1 + 2j, 3 - 1j, 2 + 1j, -1 + 3j, 4 - 2j], dtype=np.complex128)
    storage = jnp.asarray(samples)

    def loss(dt):
        source = FrequencySeries(
            wrap_backend_array(storage), delta_f=0.25, epoch=123, copy=copy
        )
        shifted = utils.apply_fseries_time_shift(source, dt, kmin=kmin, copy=copy)
        arr = backend_array(shifted, "jax")
        return jnp.sum(jnp.real(arr))

    with jax_ctx:
        grad_fn = jax.grad(loss)
        grad_val = grad_fn(0.125)

    assert np.isfinite(float(grad_val))
    # Analytic derivative: d/dt [ Re( h_k * exp(-2pi i dt f_k) ) ]
    # = Re( -2pi i f_k h_k exp(-2pi i dt f_k) )
    indices = np.arange(len(samples))
    freqs = indices * 0.25
    phase = np.exp(-2j * np.pi * 0.125 * freqs)
    d_dt = -2j * np.pi * freqs * samples * phase
    expected_grad = np.sum(np.real(d_dt[kmin:]))
    np.testing.assert_allclose(float(grad_val), expected_grad, rtol=1e-12, atol=1e-12)
