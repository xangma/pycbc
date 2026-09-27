"""Independent Fourier and native-recurrence checks for point chi-square."""

import numpy as np
import pytest

from pycbc import scheme
from pycbc.types import FrequencySeries
from pycbc.vetoes.chisq import power_chisq_at_points_from_precomputed
from pycbc.vetoes.chisq_numpy import shift_sum


def _fft_oracle(corr, points, bins, ntime, base_k=0):
    power = np.zeros(len(points), dtype=np.float64)
    for start, end in zip(bins[:-1], bins[1:]):
        band = np.zeros(ntime, dtype=np.complex128)
        band[start:end] = corr[start-base_k:end-base_k]
        transformed = np.fft.ifft(band)*ntime
        power += abs(transformed[points % ntime])**2
    return power


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_cropped_long_bins_match_fft(dtype):
    rng = np.random.default_rng(138)
    ntime, kmin, nfreq = 262144, 317, 170003
    corr = (rng.normal(size=nfreq)+1j*rng.normal(size=nfreq)).astype(dtype)
    # The large bins cross internal chunk boundaries; include a zero-width bin.
    bins = np.array([kmin, kmin+110007, kmin+110007, kmin+nfreq])
    points = np.array([0, 150001, ntime+150001, -1], dtype=np.int64)
    expected = _fft_oracle(corr, points, bins, ntime, kmin)
    actual = shift_sum(corr, points, bins, ntime, kmin)
    assert actual.dtype == corr.real.dtype
    np.testing.assert_allclose(actual, expected,
                               rtol=1e-6 if dtype == np.complex64 else 1e-12)


@pytest.mark.parametrize("mode", ["cpu-compatible", "direct-phase"])
def test_jax_precomputed_long_bins_match_mode_reference(mode):
    pytest.importorskip("jax")
    context = scheme.JAXScheme("cpu", chisq_mode=mode)
    rng = np.random.default_rng(810)
    ntime = 2097152
    corr = np.zeros(ntime, dtype=np.complex64)
    corr[15360:146432] = (
        rng.normal(size=131072)+1j*rng.normal(size=131072)
    ).astype(np.complex64)
    points = np.array([1276244, 1500001], dtype=np.int64)
    bins = np.array([15360, 115363, 146432], dtype=np.uint32)
    snrv = np.array([2+3j, 1-2j], dtype=np.complex64)
    if mode == "cpu-compatible":
        # The default follows the compiled CPU recurrence, whose accumulated
        # phase rounding differs from independently evaluated Fourier phases.
        with scheme.DefaultScheme():
            expected = power_chisq_at_points_from_precomputed(
                FrequencySeries(corr, delta_f=0.25), snrv, 0.5, bins, points
            )
    else:
        expected = (2*_fft_oracle(corr, points, bins, ntime)-abs(snrv)**2)*0.25
    with context:
        actual = power_chisq_at_points_from_precomputed(
            FrequencySeries(corr, delta_f=0.25), snrv, 0.5, bins, points
        )
    np.testing.assert_allclose(actual, expected, rtol=2e-6)


def test_empty_points():
    actual = shift_sum(np.ones(8, dtype=np.complex64), [], [0, 8])
    assert actual.shape == (0,)
    assert actual.dtype == np.float32


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_bins_outside_crop_match_zero_padded_fft(dtype):
    rng = np.random.default_rng(563)
    ntime, kmin, nfreq = 1024, 113, 271
    corr = (rng.normal(size=nfreq)+1j*rng.normal(size=nfreq)).astype(dtype)
    full = np.zeros(ntime, dtype=dtype)
    full[kmin:kmin+nfreq] = corr
    # Include entirely empty bins as well as bins crossing both crop edges.
    bins = np.array([0, 51, 179, 305, 431, 512])
    points = np.array([0, 359, 1023])
    expected = _fft_oracle(full, points, bins, ntime)
    actual = shift_sum(corr, points, bins, ntime, kmin)
    np.testing.assert_allclose(actual, expected,
                               rtol=1e-6 if dtype == np.complex64 else 1e-12)
