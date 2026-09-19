"""Focused tests for JAX-aware marginalized Gaussian likelihoods."""

import types

import numpy
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from pycbc.inference.models import gaussian_noise  # noqa: E402
from pycbc.inference.models.marginalized_gaussian_noise import (  # noqa: E402
    MarginalizedHMPolPhase,
    MarginalizedPhaseGaussianNoise,
)
from pycbc.inference.models.tools import marginalize_likelihood  # noqa: E402


def _phase_model():
    model = types.SimpleNamespace(
        pol=numpy.asarray([0.0, 0.25]),
        phase=numpy.asarray([0.0, 0.5]),
        _phase_fac={},
        _jax_phase_fac={},
        _jax_marginalization_grids={},
    )
    model._marginalization_grids = types.MethodType(
        MarginalizedHMPolPhase._marginalization_grids, model
    )
    model.phase_fac = types.MethodType(
        MarginalizedHMPolPhase.phase_fac, model
    )
    return model


def test_fixed_marginalization_grids_and_phase_factors_are_cached():
    model = _phase_model()
    like = jnp.ones(2, dtype=jnp.complex128)

    first_grids = model._marginalization_grids(like)
    second_grids = model._marginalization_grids(like)
    first_factor = model.phase_fac(2, first_grids[1])
    second_factor = model.phase_fac(2, second_grids[1])

    assert first_grids[0] is second_grids[0]
    assert first_grids[1] is second_grids[1]
    assert first_factor is second_factor


def test_explicit_phase_values_are_distinct_and_differentiable():
    model = _phase_model()
    first_phase = jnp.array([0.1, 0.2], dtype=jnp.float64)
    second_phase = jnp.array([0.3, 0.4], dtype=jnp.float64)

    first = model.phase_fac(2, first_phase)
    second = model.phase_fac(2, second_phase)

    numpy.testing.assert_allclose(numpy.asarray(first), numpy.exp(2j * numpy.asarray(first_phase)))
    numpy.testing.assert_allclose(numpy.asarray(second), numpy.exp(2j * numpy.asarray(second_phase)))
    assert not numpy.array_equal(numpy.asarray(first), numpy.asarray(second))

    def loss(phase_in):
        fac = model.phase_fac(3, phase_in)
        return jnp.sum(jnp.real(fac))

    grad = jax.grad(loss)(jnp.array([0.2, 0.6], dtype=jnp.float64))
    assert grad is not None
    assert numpy.all(numpy.isfinite(numpy.asarray(grad)))

    numpy_phase = numpy.asarray([0.15, 0.35])
    numpy.testing.assert_allclose(
        model.phase_fac(2, numpy_phase), numpy.exp(2j * numpy_phase)
    )


def test_batched_phase_marginalization_stays_on_jax(monkeypatch):
    hd = jnp.array(
        [1.0 + 0.5j, 0.2 - 0.3j],
        dtype=jnp.complex128,
    )
    hh = jnp.array([0.8, 1.1], dtype=jnp.float64)
    model = types.SimpleNamespace(
        _parse_batched_params=lambda *args, **params: params or args[0]
    )

    def inner_products(actual_model, params, *, zero_phase):
        assert actual_model is model
        assert params == {"ra": 0.1}
        assert zero_phase
        return hd, hh, {}, {}

    monkeypatch.setattr(
        gaussian_noise, "_batched_waveform_inner_products", inner_products
    )

    actual = MarginalizedPhaseGaussianNoise._batched_loglr(model, ra=0.1)
    expected = marginalize_likelihood(
        hd, hh, phase=True, skip_vector=True
    )

    assert isinstance(actual, (jax.Array, jnp.ndarray))
    assert actual.shape == (2,)
    numpy.testing.assert_allclose(numpy.asarray(actual), numpy.asarray(expected))

    def loss(hd_in):
        def mock_inner(*args, **kwargs):
            return hd_in, hh, {}, {}

        monkeypatch.setattr(
            gaussian_noise, "_batched_waveform_inner_products", mock_inner
        )
        res = MarginalizedPhaseGaussianNoise._batched_loglr(model, ra=0.1)
        return jnp.sum(res)

    grad = jax.grad(loss)(hd)
    assert grad is not None
    assert numpy.all(numpy.isfinite(numpy.asarray(grad)))
