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

from pycbc.inference import geweke


def test_geweke_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    values = np.array([
        -1.2, 0.4, 1.1, -0.7, 0.2, 1.6, -0.3, 0.8, 1.4, -0.5,
    ])
    args = (3, 2, 6, 6)

    expected = geweke.geweke(values, *args)
    actual = geweke.geweke(
        torch.tensor(values, dtype=torch.float64), *args
    )

    np.testing.assert_array_equal(actual[0].numpy(), expected[0])
    np.testing.assert_array_equal(actual[1].numpy(), expected[1])
    np.testing.assert_allclose(actual[2].numpy(), expected[2])


def test_geweke_torch_integral_and_empty_segments():
    torch = pytest.importorskip("torch")

    starts, ends, stats = geweke.geweke(
        torch.arange(10), 3, 2, 0, 6
    )

    assert starts.dtype == torch.int64
    assert ends.dtype == torch.int64
    assert stats.is_floating_point()
    assert starts.numel() == ends.numel() == stats.numel() == 0
