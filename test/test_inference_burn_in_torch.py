import sys
import types

import lal
import numpy as np
import pytest
from scipy.stats import ks_2samp

try:
    import lal.utils  # noqa: F401
except ModuleNotFoundError:
    lal_utils = types.ModuleType("lal.utils")
    sys.modules["lal.utils"] = lal_utils
    lal.utils = lal_utils

from pycbc.inference import burn_in


@pytest.mark.parametrize(
    "values1, values2",
    [
        ([0.0, 0.5, 1.0], [0.0, 0.5, 1.0]),
        ([0.0, 0.0, 1.0], [0.0, 1.0, 1.0]),
        (np.arange(7.0), np.linspace(-1.0, 8.0, 11)),
        ([False, False, True], [False, True, True]),
        ([1.0, np.inf], [1.0, 2.0]),
        ([16777216, 16777217], np.array([16777216.0], dtype=np.float32)),
    ],
)
def test_ks_test_torch_matches_scipy(values1, values2):
    torch = pytest.importorskip("torch")
    expected = ks_2samp(values1, values2).pvalue
    actual = burn_in._torch_ks_2samp_pvalue(
        torch.as_tensor(values1), torch.as_tensor(values2)
    )
    assert actual == pytest.approx(expected, rel=1e-14, abs=1e-15)

    samples1 = {"x": torch.as_tensor(values1)}
    samples2 = {"x": torch.as_tensor(values2)}
    assert burn_in.ks_test(
        samples1, samples2, threshold=expected - 1e-12
    ) == {"x": True}
    assert burn_in.ks_test(
        samples1, samples2, threshold=expected + 1e-12
    ) == {"x": False}


def test_ks_test_torch_asymptotic_and_mixed_input():
    torch = pytest.importorskip("torch")
    values1 = np.arange(10001, dtype=float)
    values2 = np.arange(0.25, 10001.25, dtype=float)
    expected = ks_2samp(values1, values2).pvalue
    actual = burn_in._torch_ks_2samp_pvalue(
        torch.from_numpy(values1), values2
    )
    assert actual == pytest.approx(expected, rel=1e-13, abs=1e-15)


def test_ks_test_torch_nan_empty_and_type_errors():
    torch = pytest.importorskip("torch")
    assert burn_in.ks_test(
        {"x": torch.tensor([1.0, float("nan")])},
        {"x": torch.tensor([1.0, 2.0])},
    ) == {"x": False}
    assert burn_in.ks_test(
        {"x": torch.tensor([])}, {"x": torch.tensor([1.0])}
    ) == {"x": False}
    with pytest.raises(ValueError, match="one-dimensional"):
        burn_in.ks_test(
            {"x": torch.ones((2, 2))}, {"x": torch.ones(4)}
        )
    with pytest.raises(TypeError, match="must be real"):
        burn_in.ks_test(
            {"x": torch.ones(4, dtype=torch.cdouble)},
            {"x": torch.ones(4)},
        )


def test_max_posterior_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    values = np.array([
        [-8.0, -7.0, -6.0, -5.0],
        [-12.0, -11.0, -10.0, -9.0],
        [-7.5, -6.5, -4.0, -3.0],
    ])
    expected = burn_in.max_posterior(values, 4)
    actual = burn_in.max_posterior(
        torch.tensor(values, dtype=torch.float64), 4
    )
    np.testing.assert_array_equal(actual[0].numpy(), expected[0])
    np.testing.assert_array_equal(actual[1].numpy(), expected[1])
    assert actual[0].dtype == torch.int64
    assert actual[1].dtype == torch.bool


def test_max_posterior_torch_integral_input():
    torch = pytest.importorskip("torch")
    values = np.array([[-5, -4, -3], [-10, -9, -8]])
    expected = burn_in.max_posterior(values, 2)
    actual = burn_in.max_posterior(torch.tensor(values), 2)
    np.testing.assert_array_equal(actual[0].numpy(), expected[0])
    np.testing.assert_array_equal(actual[1].numpy(), expected[1])


@pytest.mark.parametrize(
    "values, dim",
    [
        (np.array([-4.0, -3.0, 0.0, 0.5, 4.0]), 6),
        (np.array([-4.0, -3.0, -2.0]), 6),
    ],
)
def test_posterior_step_torch_matches_numpy(values, dim):
    torch = pytest.importorskip("torch")
    expected = burn_in.posterior_step(values, dim)
    actual = burn_in.posterior_step(torch.tensor(values), dim)
    assert actual.device.type == "cpu"
    assert actual.dtype == torch.int64
    assert actual.item() == expected


def test_torch_burn_in_shape_and_type_errors():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="nwalkers x niterations"):
        burn_in.max_posterior(torch.ones(4), 2)
    with pytest.raises(ValueError, match="1D array"):
        burn_in.posterior_step(torch.ones((2, 3)), 2)
    with pytest.raises(TypeError, match="must be real"):
        burn_in.max_posterior(torch.ones((2, 3), dtype=torch.cdouble), 2)
