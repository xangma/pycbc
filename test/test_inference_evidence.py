import sys
import types

import lal
import numpy as np
import pytest
from scipy.special import logsumexp

try:
    import lal.utils  # noqa: F401
except ModuleNotFoundError:
    lal_utils = types.ModuleType("lal.utils")
    sys.modules["lal.utils"] = lal_utils
    lal.utils = lal_utils

from pycbc.inference import evidence


def test_mean_evidence_estimators_match_logsumexp():
    log_likelihood = np.array([-20.0, -3.0, -1.0, 2.0])

    arithmetic = evidence.arithmetic_mean_estimator(log_likelihood)
    harmonic = evidence.harmonic_mean_estimator(log_likelihood)

    np.testing.assert_allclose(
        arithmetic,
        logsumexp(log_likelihood) - np.log(log_likelihood.size),
    )
    np.testing.assert_allclose(
        harmonic,
        np.log(log_likelihood.size) - logsumexp(-log_likelihood),
    )


def test_torch_mean_evidence_estimators_handle_integral_and_empty_inputs():
    torch = pytest.importorskip("torch")
    log_likelihood = torch.tensor([-3, -1, 2])

    arithmetic = evidence.arithmetic_mean_estimator(log_likelihood)
    harmonic = evidence.harmonic_mean_estimator(log_likelihood)

    assert arithmetic.is_floating_point()
    assert harmonic.is_floating_point()
    with pytest.raises(ValueError, match="zero-size array"):
        evidence.arithmetic_mean_estimator(torch.tensor([]))
    with pytest.raises(ValueError, match="zero-size array"):
        evidence.harmonic_mean_estimator(torch.tensor([]))


def test_stepping_stone_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    betas = np.array([0.0, 0.25, 0.6, 1.0])
    log_likelihood = np.array([
        [[-5.0, -4.0], [-3.0, -2.0]],
        [[-4.0, -3.0], [-2.0, -1.0]],
        [[-3.0, -2.0], [-1.0, 0.0]],
        [[-2.0, -1.0], [0.0, 1.0]],
    ])

    expected = evidence.stepping_stone_algorithm(log_likelihood, betas)
    actual = evidence.stepping_stone_algorithm(
        torch.tensor(log_likelihood, dtype=torch.float64), betas
    )

    np.testing.assert_allclose(actual[0].numpy(), expected[0])
    np.testing.assert_allclose(actual[1].numpy(), expected[1])

    with pytest.raises(ValueError, match="zero-size array"):
        evidence.stepping_stone_algorithm(
            torch.empty((len(betas), 0, 0), dtype=torch.float64), betas
        )


@pytest.mark.parametrize(
    "method", ["trapezoid", "trapezoid_corrected", "simpsons"]
)
def test_thermodynamic_integration_torch_matches_numpy(method):
    torch = pytest.importorskip("torch")
    betas = np.array([1.0, 0.0, 0.6, 0.25])
    log_likelihood = np.array([
        [[-2.0, -1.0], [0.0, 1.0]],
        [[-5.0, -4.0], [-3.0, -2.0]],
        [[-3.0, -2.0], [-1.0, 0.0]],
        [[-4.0, -3.0], [-2.0, -1.0]],
    ])

    expected = evidence.thermodynamic_integration(
        log_likelihood, betas, method=method
    )
    actual = evidence.thermodynamic_integration(
        torch.tensor(log_likelihood, dtype=torch.float64),
        betas,
        method=method,
    )

    np.testing.assert_allclose(actual[0].numpy(), expected[0])
    np.testing.assert_allclose(actual[1].numpy(), expected[1])


def test_thermodynamic_integration_simpsons_preserves_even_last_rule():
    betas = np.array([1.0, 0.0, 0.6, 0.25])
    log_likelihood = np.array([
        [[-2.0, -1.0], [0.0, 1.0]],
        [[-5.0, -4.0], [-3.0, -2.0]],
        [[-3.0, -2.0], [-1.0, 0.0]],
        [[-4.0, -3.0], [-2.0, -1.0]],
    ])

    log_evidence, _ = evidence.thermodynamic_integration(
        log_likelihood, betas, method="simpsons"
    )

    np.testing.assert_allclose(log_evidence, -1.841517857142857)
