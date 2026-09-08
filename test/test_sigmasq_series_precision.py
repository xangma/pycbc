"""Cumulative template power must retain small terms in long spectra."""

import numpy as np
import pytest

from pycbc import scheme
from pycbc.filter.matchedfilter import sigmasq_series
from pycbc.types import FrequencySeries
from pycbc.vetoes.chisq import power_chisq_bins_from_sigmasq_series


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("use_psd", [False, True])
def test_long_sigmasq_series_and_equal_power_bins(dtype, use_psd):
    context = scheme.CPUScheme(1)
    delta_f = 1 / 512
    frequency = np.arange(1048577) * delta_f
    low, high = 30.0, 1999.0
    kmin, kmax = int(low / delta_f), int(high / delta_f)
    # An inspiral-like amplitude gives a long, low-power tail. A sequential
    # float32 prefix loses tail contributions after its running sum grows.
    amplitude = (np.maximum(frequency, low) / low) ** (-7 / 6)
    waveform = (amplitude * np.exp(0.2j * frequency)).astype(dtype)
    real_dtype = np.float32 if dtype == np.complex64 else np.float64
    psd_values = (1 + (frequency / 300) ** 2).astype(real_dtype)

    band = waveform[kmin:kmax].astype(np.complex128)
    reference_power = band.real**2 + band.imag**2
    if use_psd:
        reference_power /= psd_values[kmin:kmax].astype(np.float64)
    reference_prefix = np.cumsum(reference_power, dtype=np.longdouble)

    with context:
        template = FrequencySeries(waveform, delta_f=delta_f)
        psd = FrequencySeries(psd_values, delta_f=delta_f) if use_psd else None
        result = sigmasq_series(template, psd, low, high)
        bins = power_chisq_bins_from_sigmasq_series(result, 16, kmin, kmax)
        actual = result.numpy()

    assert result.dtype == real_dtype
    assert result.delta_f == delta_f
    assert np.count_nonzero(actual[:kmin]) == 0
    assert np.count_nonzero(actual[kmax:]) == 0
    # Bound a million sequential float64 additions by accumulated rounding
    # error. Longdouble gives a wider reference on platforms that support it.
    roundoff = len(reference_prefix) * np.finfo(np.float64).eps
    tolerance = (
        3e-7 if dtype == np.complex64 else 2 * roundoff / (1 - roundoff)
    )
    np.testing.assert_allclose(
        actual[kmin:kmax], reference_prefix * (4 * delta_f), rtol=tolerance
    )
    # Assess the bins with independent double-precision power, allowing only
    # discrete frequency-cell placement and the public output's rounding.
    power_at_edges = np.append([0.0], reference_prefix)[bins - kmin]
    bin_power = np.diff(power_at_edges)
    equal_power = reference_prefix[-1] / 16
    error_budget = 2 * reference_power.max() + 5e-6 * equal_power
    assert np.all(abs(bin_power - equal_power) <= error_budget)


def test_equal_power_thresholds_between_adjacent_float32_values():
    # Float32 thirds lie above the true thresholds. Keep the thresholds in
    # double precision so searchsorted does not move either internal edge.
    prefix = np.array([0, 0, 0.1, 1 / 3, 0.5, 2 / 3, 0.8, 1, 0],
                      dtype=np.float32)
    with scheme.CPUScheme(1):
        series = FrequencySeries(prefix, delta_f=0.25)
        bins = power_chisq_bins_from_sigmasq_series(series, 3, 2, 8)
    np.testing.assert_array_equal(bins, [2, 3, 5, 8])
