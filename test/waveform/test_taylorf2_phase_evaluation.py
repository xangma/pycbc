"""Equivalence of TaylorF2's sparse logarithms and the full PN evaluator."""

import itertools

import pytest

torch = pytest.importorskip("torch")

from pycbc.constants import MTSUN_SI  # noqa: E402
from pycbc.waveform.taylorf2_torch import (  # noqa: E402
    _evaluate_phase_polynomial,
    taylorf2_aligned_phasing,
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return request.param


def _legacy_phase(v, coeff, coeff_log, coeff_log_sq):
    """The original evaluator, retaining every dense logarithmic term."""
    log_v = torch.log(v)
    log_v_sq = log_v * log_v
    result = coeff[15] + coeff_log[15] * log_v + coeff_log_sq[15] * log_v_sq
    for index in range(14, -1, -1):
        term = (coeff[index] + coeff_log[index] * log_v
                + coeff_log_sq[index] * log_v_sq)
        result = torch.addcmul(term, result, v)
    v2 = v * v
    return result / (v2 * v2 * v)


def _parameters(device, dtype):
    values = {
        "mass1": [1.4, 2.1, 8.0],
        "mass2": [1.3, 1.7, 3.0],
        "chi1": [0.02, -0.15, 0.3],
        "chi2": [-0.01, 0.1, -0.2],
        "qm_def1": [0.1, 0.3, -0.05],
        "qm_def2": [0.2, -0.1, 0.4],
        "lambda1": [500.0, 1000.0, 10.0],
        "lambda2": [200.0, 800.0, 20.0],
    }
    for index, name in enumerate(
        [f"dchi{i}" for i in range(8)] + ["dchi5l", "dchi6l"]
    ):
        values[name] = [0.003 * (index + 1), -0.002, 0.004]
    return {name: torch.tensor(value, device=device, dtype=dtype)
            for name, value in values.items()}


def _coefficients(parameters, orders):
    phase_order, spin_order, tidal_order = orders
    phasing = taylorf2_aligned_phasing(
        **{key: value for key, value in parameters.items()
           if not key.startswith("dchi")},
        dchi={key: value for key, value in parameters.items()
              if key.startswith("dchi")},
        spin_order=spin_order,
        tidal_order=tidal_order,
    )
    coeffs = (phasing.v, phasing.vlogv, phasing.vlogvsq)
    if phase_order != -1:
        mask = torch.ones_like(phasing.v)
        mask[phase_order + 1:8] = 0.0
        coeffs = tuple(coeff * mask for coeff in coeffs)
    return tuple(coeff.unsqueeze(-1) for coeff in coeffs)


def _velocity(parameters):
    mass = parameters["mass1"] + parameters["mass2"]
    frequencies = mass.new_tensor([20.3, 30.0, 48.7, 127.0, 511.5, 1024.0])
    return torch.pow(torch.pi * mass[:, None] * MTSUN_SI * frequencies,
                     1.0 / 3.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_all_pn_orders_match_original_evaluation(device, dtype):
    parameters = _parameters(device, dtype)
    velocity = _velocity(parameters)
    orders = itertools.product(
        [-1, 0, 1, 2, 3, 4, 5, 6, 7],
        [-1, 0, 1, 2, 3, 4, 5, 6, 7],
        [-1, 0, 10, 12, 13, 14, 15],
    )
    for order in orders:
        coeffs = _coefficients(parameters, order)
        # This structural invariant justifies enabling the sparse evaluator.
        assert torch.count_nonzero(coeffs[2]) == 0
        other_logs = torch.cat((coeffs[1][:5], coeffs[1][7:]))
        assert torch.count_nonzero(other_logs) == 0
        expected = _legacy_phase(velocity, *coeffs)
        actual = _evaluate_phase_polynomial(
            velocity, *coeffs, taylorf2_log_terms=True
        )
        tolerance = 5e-14 if dtype == torch.float64 else 0.0
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=0.0,
                                   msg=f"PN orders: {order}")
        if dtype == torch.float64:
            # Small relative phase errors must also preserve the complex
            # waveform, where a large absolute phase error would be visible.
            error = (torch.exp(-1j * actual) - torch.exp(-1j * expected)).abs()
            assert error.max() < 2e-10
            assert (torch.linalg.vector_norm(error, dim=-1)
                    / error.shape[-1]**0.5).max() < 1e-11


def test_scalar_and_reference_velocity_shapes(device):
    parameters = _parameters(device, torch.float64)
    coeffs = _coefficients(parameters, (-1, -1, 15))
    velocity = _velocity(parameters)
    for values, coefficients in (
        (velocity[0], tuple(coeff[:, 0, 0] for coeff in coeffs)),
        (velocity[:, :1], coeffs),
        (velocity[0, 0], tuple(coeff[:, 0, 0] for coeff in coeffs)),
    ):
        torch.testing.assert_close(
            _evaluate_phase_polynomial(
                values, *coefficients, taylorf2_log_terms=True
            ),
            _legacy_phase(values, *coefficients), rtol=5e-14, atol=0.0,
        )


def test_generic_evaluator_preserves_arbitrary_log_coefficients(device):
    values = torch.linspace(0.07, 0.5, 9, device=device, dtype=torch.float64)
    coeff = torch.arange(1, 17, device=device, dtype=values.dtype)
    coeff_log = coeff / 7.0
    coeff_log_sq = -coeff / 13.0
    torch.testing.assert_close(
        _evaluate_phase_polynomial(values, coeff, coeff_log, coeff_log_sq),
        _legacy_phase(values, coeff, coeff_log, coeff_log_sq),
        rtol=0.0, atol=0.0,
    )


@pytest.mark.parametrize("orders", [(-1, -1, 15), (4, 5, 12)])
def test_reverse_autograd_matches_original(device, orders):
    parameters = {name: value.requires_grad_() for name, value in
                  _parameters(device, torch.float64).items()}
    velocity = _velocity(parameters)
    coeffs = _coefficients(parameters, orders)
    expected = _legacy_phase(velocity, *coeffs)
    actual = _evaluate_phase_polynomial(
        velocity, *coeffs, taylorf2_log_terms=True
    )
    weights = torch.linspace(0.2, 1.0, actual.numel(),
                             device=device, dtype=actual.dtype).reshape_as(actual)
    expected_gradients = torch.autograd.grad(
        (expected * weights).sum(), tuple(parameters.values()), retain_graph=True
    )
    actual_gradients = torch.autograd.grad(
        (actual * weights).sum(), tuple(parameters.values())
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient,
                                   rtol=5e-14, atol=0.0)


@pytest.mark.parametrize("orders", [(-1, -1, 15), (4, 5, 12)])
def test_forward_autograd_matches_original(device, orders):
    with torch.autograd.forward_ad.dual_level():
        parameters = {
            name: torch.autograd.forward_ad.make_dual(
                value, torch.full_like(value, 0.2 + index * 0.01)
            )
            for index, (name, value) in enumerate(
                _parameters(device, torch.float64).items()
            )
        }
        velocity = _velocity(parameters)
        coeffs = _coefficients(parameters, orders)
        expected = torch.autograd.forward_ad.unpack_dual(
            _legacy_phase(velocity, *coeffs)
        )
        actual = torch.autograd.forward_ad.unpack_dual(
            _evaluate_phase_polynomial(
                velocity, *coeffs, taylorf2_log_terms=True
            )
        )
        torch.testing.assert_close(actual.primal, expected.primal,
                                   rtol=5e-14, atol=0.0)
        torch.testing.assert_close(actual.tangent, expected.tangent,
                                   rtol=5e-14, atol=0.0)
