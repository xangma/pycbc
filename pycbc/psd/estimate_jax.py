# Copyright (C) 2026
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""JAX PSD estimation kernels, inverse spectrum truncation,
and interpolation.
"""

import numpy as np
import jax.numpy as jnp

from pycbc.types import FrequencySeries, TimeSeries
from pycbc.types.array_jax import (
    JAXArrayData,
    _ensure_x64,
    to_jax,
)


def _wrap_frequency_series(values, delta_f, epoch=None):
    """Wrap values in FrequencySeries matching the active scheme."""
    from pycbc import scheme as _scheme

    state = _scheme.mgr.state
    if isinstance(state, _scheme.JAXScheme):
        data = JAXArrayData(values)
    else:
        data = np.asarray(values)
    return FrequencySeries(data, delta_f=delta_f, epoch=epoch, copy=False)


def median_bias(n):
    """Calculate the bias of the median average PSD computed from `n` segments.

    See arXiv:gr-qc/0509116 appendix B for details.
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if n >= 1000:
        return float(np.log(2))
    ans = 1.0
    for i in range(1, (n - 1) // 2 + 1):
        ans += 1.0 / (2 * i + 1) - 1.0 / (2 * i)
    return ans


def _unfold_segments(samples, seg_len, seg_stride, num_segments):
    """Unfold 1D array into 2D segments of shape (num_segments, seg_len)."""
    starts = jnp.arange(num_segments) * seg_stride
    indices = starts[:, None] + jnp.arange(seg_len)[None, :]
    return samples[indices]


def welch_jax(
    timeseries,
    seg_len=4096,
    seg_stride=2048,
    window="hann",
    avg_method="median",
    num_segments=None,
    require_exact_data_fit=False,
    device=None,
):
    """PSD estimator based on Welch's method implemented in pure JAX.

    Evaluates segment FFTs in double precision on JAX devices (CPU/GPU/TPU)
    using batched transforms.

    Parameters
    ----------
    timeseries : TimeSeries or jax.Array or numpy.ndarray
        Time series for which the PSD is to be estimated.
    seg_len : int
        Segment length in samples.
    seg_stride : int
        Separation between consecutive segments, in samples.
    window : {'hann', numpy.ndarray, jax.Array}
        Function used to window segments before Fourier transforming.
    avg_method : {'median', 'mean', 'median-mean'}
        Method used for averaging individual segment PSDs.
    num_segments : int, optional
        Number of segments to use. If None, calculated from data length.
    require_exact_data_fit : bool, default False
        If True, require data length to match exactly.
    device : jax.Device, optional
        Target JAX device.

    Returns
    -------
    psd : FrequencySeries or jax.Array
        Estimated power spectral density.
    """
    _ensure_x64()
    if avg_method not in ("mean", "median", "median-mean"):
        raise ValueError(f"Invalid averaging method {avg_method!r}")
    if (
        not isinstance(seg_len, int)
        or not isinstance(seg_stride, int)
        or seg_len <= 0
        or seg_stride <= 0
    ):
        raise ValueError("Segment length and stride must be positive integers")

    is_series = isinstance(timeseries, TimeSeries)
    delta_t = float(getattr(timeseries, "delta_t", 1.0))
    epoch = getattr(
        timeseries, "start_time", getattr(timeseries, "epoch", None)
    )

    samples = to_jax(timeseries, device=device)
    num_samples = len(samples)

    if num_segments is None:
        num_segments = int(num_samples // seg_stride)
        if (num_segments - 1) * seg_stride + seg_len > num_samples:
            num_segments -= 1

    if not require_exact_data_fit:
        data_len = (num_segments - 1) * seg_stride + seg_len
        if data_len < num_samples:
            diff = num_samples - data_len
            start = diff // 2
            if diff % 2:
                start = start + 1
            end = num_samples - diff // 2
            samples = samples[start:end]
            num_samples = len(samples)
        elif data_len > num_samples:
            raise ValueError(
                f"I was asked to estimate a PSD on {data_len} data samples. "
                f"However data provided contains only {num_samples} samples."
            )

    if num_samples != (num_segments - 1) * seg_stride + seg_len:
        raise ValueError("Incorrect choice of segmentation parameters")

    delta_f = 1.0 / (delta_t * seg_len)

    # Window preparation
    if isinstance(window, str):
        if window == "hann":
            window_arr = jnp.asarray(np.hanning(seg_len), dtype=samples.dtype)
        else:
            raise ValueError(f"Unknown window string {window!r}")
    else:
        window_arr = jnp.asarray(window, dtype=samples.dtype)
        if len(window_arr) != seg_len:
            raise ValueError("Invalid window: incorrect window length")

    # Unfold segments: shape (num_segments, seg_len)
    segments = _unfold_segments(samples, seg_len, seg_stride, num_segments)
    windowed = segments * window_arr

    # Batched FFT
    spectra = jnp.fft.rfft(windowed, axis=-1) * delta_t
    seg_psds = jnp.real(spectra * jnp.conj(spectra))
    seg_psds = seg_psds.at[:, 0].multiply(0.5)
    seg_psds = seg_psds.at[:, -1].multiply(0.5)

    if avg_method == "mean":
        psd = jnp.mean(seg_psds, axis=0)
    elif avg_method == "median":
        psd = jnp.median(seg_psds, axis=0) / median_bias(num_segments)
    elif avg_method == "median-mean":
        odd_psds = seg_psds[::2]
        even_psds = seg_psds[1::2]
        odd_median = jnp.median(odd_psds, axis=0) / median_bias(len(odd_psds))
        even_median = (
            jnp.median(even_psds, axis=0) / median_bias(len(even_psds))
        )
        psd = (odd_median + even_median) / 2.0

    # Window energy normalization
    norm = 2.0 * delta_f * seg_len / jnp.sum(jnp.square(window_arr))
    psd = psd * norm

    if is_series or isinstance(timeseries, (TimeSeries, FrequencySeries)):
        return _wrap_frequency_series(psd, delta_f=delta_f, epoch=epoch)
    return psd


def inverse_spectrum_truncation_jax(
    psd,
    max_filter_len,
    which_spectrum="invasd",
    low_frequency_cutoff=None,
    low_frequency_fill_value=0.0,
    trunc_method="hann",
    device=None,
):
    """Modify a PSD using pure JAX operations for impulse response truncation.

    Parameters
    ----------
    psd : FrequencySeries or jax.Array
        PSD whose inverse spectrum is to be truncated.
    max_filter_len : int
        Maximum length of the time-domain filter in samples.
    which_spectrum : {'invasd', 'invpsd'}
        Spectrum to truncate.
    low_frequency_cutoff : float, optional
        Frequencies below cutoff are set to ``low_frequency_fill_value``.
    low_frequency_fill_value : float or 'fmin', default 0.0
        Fill value below cutoff.
    trunc_method : {None, 'hann'}, default 'hann'
        Window function for truncating the time-domain filter.
    device : jax.Device, optional
        Target JAX device.

    Returns
    -------
    psd_out : FrequencySeries
        Truncated PSD.
    """
    _ensure_x64()
    if not isinstance(max_filter_len, int) or max_filter_len <= 0:
        raise ValueError("max_filter_len must be a positive integer")

    delta_f = float(getattr(psd, "delta_f", 1.0))
    epoch = getattr(psd, "epoch", None)

    if low_frequency_cutoff is not None:
        max_freq = (len(psd) - 1) * delta_f
        if low_frequency_cutoff < 0.0 or low_frequency_cutoff > max_freq:
            raise ValueError(
                "low_frequency_cutoff must be within the bandwidth of the PSD"
            )

    psd_arr = to_jax(psd, device=device)
    n_freq = len(psd_arr)
    n_time = (n_freq - 1) * 2

    inv_spectrum = jnp.zeros(n_freq, dtype=jnp.complex128)
    kmin = 1
    if low_frequency_cutoff:
        kmin = int(low_frequency_cutoff / delta_f)

    if low_frequency_fill_value != 0.0:
        if low_frequency_fill_value == "fmin":
            fill_val = 1.0 / psd_arr[kmin]
        else:
            fill_val = float(low_frequency_fill_value)
        inv_spectrum = inv_spectrum.at[:kmin].set(fill_val)

    half_n = n_time // 2
    inv_spectrum = inv_spectrum.at[kmin:half_n].set(
        1.0 / psd_arr[kmin:half_n]
    )

    if which_spectrum == "invasd":
        inv_spectrum = inv_spectrum.at[:half_n].set(
            inv_spectrum[:half_n] ** 0.5
        )
    elif which_spectrum != "invpsd":
        raise ValueError(
            f"Invalid which_spectrum input {which_spectrum}; "
            "must be 'invpsd' or 'invasd'"
        )

    # IFFT to time domain: irfft preserves 1.0 roundtrip normalization
    q = jnp.fft.irfft(inv_spectrum, n=n_time)

    trunc_start = max_filter_len // 2
    trunc_end = n_time - max_filter_len // 2
    if trunc_end < trunc_start:
        raise ValueError("Invalid value in inverse_spectrum_truncation")

    if trunc_method == "hann":
        tw = jnp.asarray(np.hanning(max_filter_len), dtype=q.dtype)
        q = q.at[0:trunc_start].multiply(tw[-trunc_start:])
        q = q.at[trunc_end:n_time].multiply(tw[0:max_filter_len // 2])

    if trunc_start < trunc_end:
        q = q.at[trunc_start:trunc_end].set(0.0)

    # FFT back to frequency domain
    psd_trunc = jnp.fft.rfft(q)
    if which_spectrum == "invasd":
        psd_trunc = psd_trunc * jnp.conj(psd_trunc)

    psd_out = 1.0 / jnp.abs(psd_trunc)
    return _wrap_frequency_series(psd_out, delta_f=delta_f, epoch=epoch)


def interpolate_jax(series, delta_f, length=None, device=None):
    """Interpolate a frequency series to desired delta_f using JAX.

    Parameters
    ----------
    series : FrequencySeries or jax.Array
        Input frequency series.
    delta_f : float
        Desired frequency resolution of output.
    length : int, optional
        Number of output frequency samples.
    device : jax.Device, optional
        Target JAX device.

    Returns
    -------
    interpolated : FrequencySeries
        Interpolated frequency series.
    """
    _ensure_x64()
    old_df = float(getattr(series, "delta_f", 1.0))
    epoch = getattr(series, "epoch", None)
    old_vals = to_jax(series, device=device)

    if len(old_vals) == 0:
        raise ValueError("array of sample points is empty")

    if length is None:
        new_n = int(round((len(old_vals) - 1) * old_df / delta_f + 1))
    else:
        new_n = int(length)

    old_freqs = jnp.arange(len(old_vals), dtype=jnp.float64) * old_df
    new_freqs = jnp.arange(new_n, dtype=jnp.float64) * delta_f

    interpolated = jnp.interp(
        new_freqs,
        old_freqs,
        old_vals,
        left=old_vals[0],
        right=old_vals[-1],
    )

    return _wrap_frequency_series(interpolated, delta_f=delta_f, epoch=epoch)
