import sys
import types

import lal
import numpy as np
import pytest

try:
    import lal.utils  # noqa: F401
except ModuleNotFoundError:
    lal_utils = types.ModuleType("lal.utils")
    sys.modules["lal.utils"] = lal_utils
    lal.utils = lal_utils

from pycbc.inference import gelman_rubin


@pytest.fixture
def chains():
    rng = np.random.default_rng(1234)
    offsets = np.arange(4)[:, None, None] * 0.08
    return rng.normal(size=(4, 3, 40)) + offsets


@pytest.mark.parametrize("auto_burn_in", [False, True])
def test_gelman_rubin_torch_matches_numpy(chains, auto_burn_in):
    torch = pytest.importorskip("torch")
    expected = gelman_rubin.gelman_rubin(chains, auto_burn_in)
    actual = gelman_rubin.gelman_rubin(
        torch.tensor(chains, dtype=torch.float64), auto_burn_in
    )
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12)


def test_gelman_rubin_single_parameter_and_mixed_list(chains):
    torch = pytest.importorskip("torch")
    expected = gelman_rubin.gelman_rubin(chains[:, :1, :], False)
    mixed = [
        torch.tensor(chains[0, :1, :], dtype=torch.float64),
        *chains[1:, :1, :],
    ]
    actual = gelman_rubin.gelman_rubin(mixed, False)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12)


def test_gelman_rubin_walk_torch_matches_numpy(chains):
    torch = pytest.importorskip("torch")
    expected = gelman_rubin.walk(chains, 10, 35, 7)
    actual = gelman_rubin.walk(
        torch.tensor(chains, dtype=torch.float64), 10, 35, 7
    )
    np.testing.assert_array_equal(actual[0].numpy(), expected[0])
    np.testing.assert_array_equal(actual[1].numpy(), expected[1])
    np.testing.assert_allclose(actual[2].numpy(), expected[2], rtol=1e-12)


def test_gelman_rubin_rejects_complex_torch_chains():
    torch = pytest.importorskip("torch")
    with pytest.raises(TypeError, match="must be real"):
        gelman_rubin.gelman_rubin(torch.ones((3, 1, 8), dtype=torch.cdouble))
