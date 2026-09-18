"""Cumulative template power must preserve CPU search bin placement."""

import numpy as np
import pytest

from pycbc import scheme
from pycbc.filter.matchedfilter import sigmasq_series
from pycbc.types import FrequencySeries
from pycbc.vetoes.chisq import power_chisq_bins_from_sigmasq_series


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("use_psd", [False, True])
@pytest.mark.parametrize("backend", ["torch-cpu", "torch-cuda"])
def test_long_sigmasq_series_and_equal_power_bins(dtype, use_psd, backend):
    torch = pytest.importorskip("torch")
    device = backend.removeprefix("torch-")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    context = scheme.TorchScheme(device=device, num_threads=1)
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
    with scheme.CPUScheme(1):
        cpu_series = sigmasq_series(
            FrequencySeries(waveform, delta_f=delta_f),
            FrequencySeries(psd_values, delta_f=delta_f) if use_psd else None,
            low, high,
        )
        cpu_bins = power_chisq_bins_from_sigmasq_series(cpu_series, 16, kmin, kmax)
        cpu_values = cpu_series.numpy().copy()

    with context:
        template = FrequencySeries(waveform, delta_f=delta_f)
        psd = FrequencySeries(psd_values, delta_f=delta_f) if use_psd else None
        result = sigmasq_series(template, psd, low, high)
        bins = power_chisq_bins_from_sigmasq_series(result, 16, kmin, kmax)
        actual = result.numpy()
        if backend == "torch-cuda":
            # A parallel float32 scan can move bin edges between identical
            # calls. Repeated full calculations must retain the same bins.
            for _ in range(20):
                repeated = sigmasq_series(template, psd, low, high)
                repeated_bins = power_chisq_bins_from_sigmasq_series(
                    repeated, 16, kmin, kmax
                )
                np.testing.assert_array_equal(repeated_bins, bins)

    assert result.dtype == real_dtype
    assert result.delta_f == delta_f
    assert np.count_nonzero(actual[:kmin]) == 0
    assert np.count_nonzero(actual[kmax:]) == 0
    if dtype == np.complex64:
        # Search compatibility includes the original float32 rounding, even
        # where a wider or parallel scan is closer to the mathematical sum.
        np.testing.assert_array_equal(actual, cpu_values)
        np.testing.assert_array_equal(bins, cpu_bins)
        return
    # A million sequential float64 additions can be less accurate than a
    # parallel scan. Bound both scans by their accumulated rounding error;
    # longdouble gives a wider reference on platforms that support it.
    roundoff = len(reference_prefix) * np.finfo(np.float64).eps
    tolerance = 2 * roundoff / (1 - roundoff)
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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_equal_power_thresholds_between_adjacent_float32_values(device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    # The float32 values for thirds lie above the true thresholds. Rounding
    # those thresholds to float32 would incorrectly include the equal value
    # in searchsorted(side="right") and move both internal edges one cell.
    prefix = np.array([0, 0, 0.1, 1 / 3, 0.5, 2 / 3, 0.8, 1, 0], dtype=np.float32)
    with scheme.TorchScheme(device, num_threads=1):
        series = FrequencySeries(prefix, delta_f=0.25)
        bins = power_chisq_bins_from_sigmasq_series(series, 3, 2, 8)
    np.testing.assert_array_equal(bins, [2, 3, 5, 8])
