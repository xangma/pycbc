"""Check point chi-squared phases across long correlation vectors."""

import numpy as np
import pytest

from pycbc import scheme
from pycbc.types import Array, FrequencySeries
from pycbc.vetoes.chisq import power_chisq_at_points_from_precomputed


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("array_indices", [False, True])
def test_long_point_chisq_matches_direct_phases(dtype, array_indices):
    size = 2**21
    rng = np.random.default_rng(782)
    values = (rng.normal(size=size) + 1j * rng.normal(size=size)).astype(dtype)
    points = np.array([137, 1884341, size - 3], dtype=np.int64)
    bins = np.array([15360, 65536, 262144, 1048576], dtype=np.uint32)
    # Evaluate direct complex128 phases independently of the recurrence. The
    # late sample indices and long final bin expose accumulated phase error.
    bin_sums = np.zeros((len(points), len(bins) - 1), dtype=np.complex128)
    for column, (start, end) in enumerate(zip(bins[:-1], bins[1:])):
        for first in range(int(start), int(end), 32768):
            last = min(first + 32768, int(end))
            frequencies = np.arange(first, last, dtype=np.float64)
            phases = np.exp(
                2j * np.pi * points[:, None] * frequencies[None, :] / size
            )
            bin_sums[:, column] += np.sum(
                values[None, first:last] * phases, axis=1
            )

    snr = bin_sums.sum(axis=1).astype(dtype)
    norm = float(1 / np.sqrt(size))
    expected = (
        (len(bins) - 1) * np.sum(abs(bin_sums) ** 2, axis=1)
        - abs(snr.astype(np.complex128)) ** 2
    ) * norm**2

    with scheme.CPUScheme(1):
        correlation = FrequencySeries(values, delta_f=1 / 512)
        indices = Array(points) if array_indices else points
        actual = power_chisq_at_points_from_precomputed(
            correlation, snr, norm, bins, indices
        )

    expected_dtype = np.float32 if dtype == np.complex64 else np.float64
    assert actual.dtype == expected_dtype
    tolerance = 3e-6 if dtype == np.complex64 else 3e-8
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=1e-9)
