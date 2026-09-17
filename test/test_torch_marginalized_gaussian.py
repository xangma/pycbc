"""Focused tests for Torch-aware marginalized Gaussian likelihoods."""

import types

import numpy
import pytest


torch = pytest.importorskip("torch")

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
        _torch_phase_fac={},
        _torch_marginalization_grids={},
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
    like = torch.ones(2, dtype=torch.complex128)

    first_grids = model._marginalization_grids(like)
    second_grids = model._marginalization_grids(like)
    first_factor = model.phase_fac(2, first_grids[1])
    second_factor = model.phase_fac(2, second_grids[1])

    assert first_grids[0] is second_grids[0]
    assert first_grids[1] is second_grids[1]
    assert first_factor is second_factor
    assert first_factor.device == like.device


def test_explicit_phase_values_are_distinct_and_differentiable():
    model = _phase_model()
    first_phase = torch.tensor([0.1, 0.2], dtype=torch.float64)
    second_phase = torch.tensor([0.3, 0.4], dtype=torch.float64)

    first = model.phase_fac(2, first_phase)
    second = model.phase_fac(2, second_phase)

    torch.testing.assert_close(first, torch.exp(2j * first_phase))
    torch.testing.assert_close(second, torch.exp(2j * second_phase))
    assert not torch.equal(first, second)

    differentiable = torch.tensor(
        [0.2, 0.6], dtype=torch.float64, requires_grad=True
    )
    model.phase_fac(3, differentiable).real.sum().backward()
    assert differentiable.grad is not None
    assert torch.all(torch.isfinite(differentiable.grad))

    numpy_phase = numpy.asarray([0.15, 0.35])
    numpy.testing.assert_allclose(
        model.phase_fac(2, numpy_phase), numpy.exp(2j * numpy_phase)
    )


def test_batched_phase_marginalization_stays_on_torch(monkeypatch):
    hd = torch.tensor(
        [1.0 + 0.5j, 0.2 - 0.3j],
        dtype=torch.complex128,
        requires_grad=True,
    )
    hh = torch.tensor([0.8, 1.1], dtype=torch.float64)
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

    assert isinstance(actual, torch.Tensor)
    assert actual.shape == (2,)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert hd.grad is not None
    assert torch.all(torch.isfinite(hd.grad))
