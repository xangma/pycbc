# Copyright (C) 2012 Tito Dal Canton
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
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
"""Utilites to estimate PSDs from data.
"""

import numpy
from pycbc import scheme as _scheme
from pycbc.types import Array, FrequencySeries, TimeSeries, zeros
from pycbc.types import real_same_precision_as, complex_same_precision_as
from pycbc.types.backend import backend_array, is_backend, wrap_backend_array
from pycbc.fft import fft, ifft
import pycbc

try:
    import torch
    _HAVE_TORCH = pycbc.HAVE_TORCH
except Exception:  # pragma: no cover - torch optional
    torch = None
    _HAVE_TORCH = False

# Change to True in front-end if you want this function to use caching
# This is a mostly-hidden optimization option that most users will not want
# to use. It is used in PyCBC Live
USE_CACHING_FOR_WELCH_FFTS = False
USE_CACHING_FOR_INV_SPEC_TRUNC = False
# If using caching we want output to be unique if called at different places
# (and if called from different modules/functions), these unique IDs acheive
# that. The numbers are not significant, only that they are unique.
WELCH_UNIQUE_ID = 438716587
INVSPECTRUNC_UNIQUE_ID = 100257896

# Bound the explicit windowed-input and FFT-output temporaries used by the
# batched Torch Welch path.  The full PSD stack is required by the median
# estimator regardless of batching and is therefore not part of this budget.
_TORCH_WELCH_TEMPORARY_BYTES = 128 * 1024 * 1024


def median_bias(n):
    """Calculate the bias of the median average PSD computed from `n` segments.

    Parameters
    ----------
    n : int
        Number of segments used in PSD estimation.

    Returns
    -------
    ans : float
        Calculated bias.

    Raises
    ------
    ValueError
        For non-integer or non-positive `n`.

    Notes
    -----
    See arXiv:gr-qc/0509116 appendix B for details.
    """
    if type(n) is not int or n <= 0:
        raise ValueError('n must be a positive integer')
    if n >= 1000:
        return numpy.log(2)
    ans = 1
    for i in range(1, (n - 1) // 2 + 1):
        ans += 1.0 / (2*i + 1) - 1.0 / (2*i)
    return ans


def _is_torch_series(obj):
    return _HAVE_TORCH and is_backend(obj, "torch")


def _inverse_spectrum_max_frequency(psd):
    """Return the last PSD grid coordinate without synchronizing Torch data.

    Torch regular grids use float32 coordinates on MPS and float64 coordinates
    elsewhere.  Reproduce that arithmetic using host metadata so cutoff
    validation retains the legacy rounded boundary without allocating a grid
    on the device.  Non-Torch schemes keep their existing grid semantics.
    """
    state = _scheme.mgr.state
    if not (_HAVE_TORCH and isinstance(state, _scheme.TorchScheme)):
        return psd.sample_frequencies[-1]

    coordinate_dtype = (
        numpy.float32
        if state.torch_device.type == 'mps'
        else numpy.float64
    )
    with numpy.errstate(invalid='ignore', over='ignore'):
        max_frequency = (
            coordinate_dtype(len(psd) - 1)
            * coordinate_dtype(psd.delta_f)
        )
    return float(max_frequency)


def _torch_median(values, dim=0):
    """Return a NumPy-compatible median along ``dim``.

    ``torch.median`` selects the lower of the two middle values for an even
    number of samples, while ``numpy.median`` averages them.  Welch PSDs use
    the NumPy definition in every other processing scheme, so keep that
    behavior for Torch as well.
    """
    ordered = torch.sort(values, dim=dim).values
    count = ordered.shape[dim]
    lower = ordered.select(dim, (count - 1) // 2)
    upper = ordered.select(dim, count // 2)
    return (lower + upper) / 2


def _torch_welch_batch_size(seg_len, dtype, num_segments,
                            temporary_bytes=None):
    """Return a bounded number of Welch segments to transform together."""
    if temporary_bytes is None:
        temporary_bytes = _TORCH_WELCH_TEMPORARY_BYTES
    real_bytes = torch.empty((), dtype=dtype).element_size()
    # One real windowed segment plus its one-sided complex FFT output.
    bytes_per_segment = (
        seg_len * real_bytes
        + (seg_len // 2 + 1) * 2 * real_bytes
    )
    return min(num_segments, max(1, temporary_bytes // bytes_per_segment))


def _torch_welch_segment_psds(timeseries, window, seg_len, seg_stride,
                              num_segments):
    """Calculate one-sided segment PSDs with bounded batched Torch FFTs."""
    samples = backend_array(timeseries, "torch")
    segments = samples.unfold(0, seg_len, seg_stride)
    psds = torch.empty(
        (num_segments, seg_len // 2 + 1),
        dtype=samples.real.dtype,
        device=samples.device,
    )
    batch_size = _torch_welch_batch_size(
        seg_len, samples.real.dtype, num_segments
    )
    for start in range(0, num_segments, batch_size):
        stop = min(start + batch_size, num_segments)
        spectra = torch.fft.rfft(
            segments[start:stop] * window,
            n=seg_len,
            dim=-1,
        )
        # pycbc.fft.fft applies the TimeSeries sample spacing after every
        # transform.  Preserve that normalization before forming power.
        spectra.mul_(timeseries.delta_t)
        batch_psds = psds[start:stop]
        torch.square(spectra.real, out=batch_psds)
        batch_psds.addcmul_(spectra.imag, spectra.imag)
        batch_psds[:, 0].div_(2)
        batch_psds[:, -1].div_(2)
    return psds


def welch(timeseries, seg_len=4096, seg_stride=2048, window='hann',
          avg_method='median', num_segments=None, require_exact_data_fit=False):
    """PSD estimator based on Welch's method.

    Parameters
    ----------
    timeseries : TimeSeries
        Time series for which the PSD is to be estimated.
    seg_len : int
        Segment length in samples.
    seg_stride : int
        Separation between consecutive segments, in samples.
    window : {'hann', numpy.ndarray}
        Function used to window segments before Fourier transforming, or
        a `numpy.ndarray` that specifies the window.
    avg_method : {'median', 'mean', 'median-mean'}
        Method used for averaging individual segment PSDs.

    Returns
    -------
    psd : FrequencySeries
        Frequency series containing the estimated PSD.

    Raises
    ------
    ValueError
        For invalid choices of `seg_len`, `seg_stride` `window` and
        `avg_method` and for inconsistent combinations of len(`timeseries`),
        `seg_len` and `seg_stride`.

    Notes
    -----
    See arXiv:gr-qc/0509116 for details.
    """
    window_map = {
        'hann': numpy.hanning
    }

    # sanity checks
    if isinstance(window, numpy.ndarray) and window.size != seg_len:
        raise ValueError('Invalid window: incorrect window length')
    if not isinstance(window, numpy.ndarray) and window not in window_map:
        raise ValueError('Invalid window: unknown window {!r}'.format(window))
    if avg_method not in ('mean', 'median', 'median-mean'):
        raise ValueError('Invalid averaging method')
    if type(seg_len) is not int or type(seg_stride) is not int \
        or seg_len <= 0 or seg_stride <= 0:
        raise ValueError('Segment length and stride must be positive integers')

    if timeseries.precision == 'single':
        fs_dtype = numpy.complex64
    elif timeseries.precision == 'double':
        fs_dtype = numpy.complex128

    num_samples = len(timeseries)
    if num_segments is None:
        num_segments = int(num_samples // seg_stride)
        # NOTE: Is this not always true?
        if (num_segments - 1) * seg_stride + seg_len > num_samples:
            num_segments -= 1

    if not require_exact_data_fit:
        data_len = (num_segments - 1) * seg_stride + seg_len

        # Get the correct amount of data
        if data_len < num_samples:
            diff = num_samples - data_len
            start = diff // 2
            end = num_samples - diff // 2
            # Want this to be integers so if diff is odd, catch it here.
            if diff % 2:
                start = start + 1

            timeseries = timeseries[start:end]
            num_samples = len(timeseries)
        if data_len > num_samples:
            err_msg = "I was asked to estimate a PSD on %d " %(data_len)
            err_msg += "data samples. However the data provided only contains "
            err_msg += "%d data samples." %(num_samples)

    if num_samples != (num_segments - 1) * seg_stride + seg_len:
        raise ValueError('Incorrect choice of segmentation parameters')

    use_torch = _is_torch_series(timeseries)
    if not isinstance(window, numpy.ndarray) and use_torch:
        tensor = backend_array(timeseries, "torch")
        window_tensor = torch.hann_window(
            seg_len,
            periodic=False,
            dtype=tensor.real.dtype,
            device=tensor.device,
        )
        if tensor.is_complex():
            window_tensor = window_tensor.to(dtype=tensor.dtype)
        w = Array(wrap_backend_array(window_tensor), copy=False)
    else:
        if not isinstance(window, numpy.ndarray):
            window = window_map[window](seg_len)
        w = Array(window.astype(timeseries.dtype))

    # calculate psd of each segment
    delta_f = 1. / timeseries.delta_t / seg_len
    batched_torch = (
        use_torch
        and not backend_array(timeseries, "torch").is_complex()
        and not USE_CACHING_FOR_WELCH_FFTS
    )
    if not USE_CACHING_FOR_WELCH_FFTS and not batched_torch:
        segment_tilde = FrequencySeries(
            zeros(int(seg_len / 2 + 1), dtype=fs_dtype),
            delta_f=delta_f,
            copy=False,
        )

    if batched_torch:
        segment_psds = _torch_welch_segment_psds(
            timeseries,
            backend_array(w, "torch"),
            seg_len,
            seg_stride,
            num_segments,
        )
    else:
        segment_psds = []
        for i in range(num_segments):
            segment_start = i * seg_stride
            segment_end = segment_start + seg_len
            segment = timeseries[segment_start:segment_end]
            assert len(segment) == seg_len
            if not USE_CACHING_FOR_WELCH_FFTS:
                fft(segment * w, segment_tilde)
            else:
                from pycbc.strain.strain import execute_cached_fft
                segment_tilde = execute_cached_fft(segment * w,
                                                   uid=WELCH_UNIQUE_ID)
            if use_torch:
                tensor = backend_array(segment_tilde, "torch")
                if tensor.is_complex():
                    seg_psd = torch.view_as_real(tensor).square().sum(dim=-1)
                else:
                    seg_psd = torch.square(tensor)
                # halve DC and Nyquist
                seg_psd[0] /= 2
                seg_psd[-1] /= 2
                segment_psds.append(seg_psd)
            else:
                seg_psd = segment_tilde * segment_tilde.conj()
                seg_psd = abs(seg_psd).numpy()
                seg_psd[0] /= 2
                seg_psd[-1] /= 2
                segment_psds.append(seg_psd)

    if use_torch:
        stack = (
            segment_psds
            if batched_torch
            else torch.stack(segment_psds, dim=0)
        )
        if avg_method == 'mean':
            psd = torch.mean(stack, dim=0)
        elif avg_method == 'median':
            psd = _torch_median(stack, dim=0) / median_bias(num_segments)
        elif avg_method == 'median-mean':
            odd_psds = stack[::2]
            even_psds = stack[1::2]
            odd_median = _torch_median(odd_psds, dim=0) / \
                median_bias(len(odd_psds))
            even_median = _torch_median(even_psds, dim=0) / \
                median_bias(len(even_psds))
            psd = (odd_median + even_median) / 2
        window_tensor = backend_array(w, "torch")
        psd = psd * (2 * delta_f * seg_len) / (
            window_tensor * window_tensor
        ).sum()
        psd = psd.to(dtype=backend_array(timeseries, "torch").real.dtype)
        return FrequencySeries(wrap_backend_array(psd), delta_f=delta_f,
                               epoch=timeseries.start_time, copy=False)

    segment_psds = numpy.array(segment_psds)

    if avg_method == 'mean':
        psd = numpy.mean(segment_psds, axis=0)
    elif avg_method == 'median':
        psd = numpy.median(segment_psds, axis=0) / median_bias(num_segments)
    elif avg_method == 'median-mean':
        odd_psds = segment_psds[::2]
        even_psds = segment_psds[1::2]
        odd_median = numpy.median(odd_psds, axis=0) / \
            median_bias(len(odd_psds))
        even_median = numpy.median(even_psds, axis=0) / \
            median_bias(len(even_psds))
        psd = (odd_median + even_median) / 2

    w = w.numpy()
    psd *= 2 * delta_f * seg_len / (w*w).sum()

    return FrequencySeries(psd, delta_f=delta_f, dtype=timeseries.dtype,
                           epoch=timeseries.start_time)

def inverse_spectrum_truncation(psd, max_filter_len, which_spectrum='invasd',
                                low_frequency_cutoff=None, 
                                low_frequency_fill_value=0., trunc_method=None):
    """Modify a PSD such that the impulse response associated with its inverse
    square root is no longer than `max_filter_len` time samples. In practice
    this corresponds to a coarse graining or smoothing of the PSD.

    Parameters
    ----------
    psd : FrequencySeries
        PSD whose inverse spectrum is to be truncated.
    max_filter_len : int
        Maximum length of the time-domain filter in samples.
    which_spectrum : {'invasd', 'invpsd'}
        Which spectrum to truncate. If 'invasd' (default), apply truncation to
        the inverse ASD. If 'invpsd', apply to the inverse PSD.
    low_frequency_cutoff : {None, int}
        Frequencies below `low_frequency_cutoff` are set to value specified by 
        `low_frequency_fill_value`.
    low_frequency_fill_value : {float, 'fmin'}
        Value to set PSD to at frequencies below `low_frequency_cutoff`.
        Default 0. If 'fmin', set to the value of the PSD at the low frequency
        cutoff index.
    trunc_method : {None, 'hann'}
        Function used for truncating the time-domain filter.
        None produces a hard truncation at `max_filter_len`.
    

    Returns
    -------
    psd : FrequencySeries
        PSD whose inverse spectrum has been truncated.

    Raises
    ------
    ValueError
        For invalid types or values of `max_filter_len` and `low_frequency_cutoff`.

    Notes
    -----
    See arXiv:gr-qc/0509116 for details.
    """
    # sanity checks
    if type(max_filter_len) is not int or max_filter_len <= 0:
        raise ValueError('max_filter_len must be a positive integer')
    if low_frequency_cutoff is not None:
        max_frequency = _inverse_spectrum_max_frequency(psd)
        if low_frequency_cutoff < 0. or \
                low_frequency_cutoff > max_frequency:
            raise ValueError(
                'low_frequency_cutoff must be within the bandwidth of the PSD'
            )

    N = (len(psd)-1)*2

    inv_spectrum = FrequencySeries(
        zeros(len(psd), dtype=complex_same_precision_as(psd)),
        delta_f=psd.delta_f,
        dtype=complex_same_precision_as(psd),
    )

    kmin = 1
    if low_frequency_cutoff:
        kmin = int(low_frequency_cutoff / psd.delta_f)
    
    # set values below low frequency cutoff
    if low_frequency_fill_value != 0.:
        if low_frequency_fill_value == 'fmin':
            low_frequency_fill_value = 1./psd[kmin]
        inv_spectrum[:kmin] = float(low_frequency_fill_value)

    inv_spectrum[kmin:N//2] = (1.0 / psd[kmin:N//2])

    # if truncating asd, take sqrt
    if which_spectrum == 'invasd':
        inv_spectrum[:N//2] = inv_spectrum[:N//2]**0.5
    elif which_spectrum != 'invpsd':
        raise ValueError(f'Invalid which_spectrum input {which_spectrum}; '
                         f'input must be either "invpsd" or "invasd"')

    use_torch = _is_torch_series(psd)
    if not USE_CACHING_FOR_INV_SPEC_TRUNC:
        q = TimeSeries(
            zeros(N, dtype=real_same_precision_as(psd)),
            delta_t=(N / psd.delta_f)
        )
        ifft(inv_spectrum, q)
    else:
        from pycbc.strain.strain import execute_cached_ifft
        q = execute_cached_ifft(inv_spectrum, copy_output=False,
                                uid=INVSPECTRUNC_UNIQUE_ID)

    trunc_start = max_filter_len // 2
    trunc_end = N - max_filter_len // 2
    if trunc_end < trunc_start:
        raise ValueError('Invalid value in inverse_spectrum_truncation')

    if trunc_method == 'hann':
        if use_torch:
            q_tensor = backend_array(q, "torch")
            tw = torch.hann_window(max_filter_len, device=q_tensor.device,
                                   dtype=q_tensor.dtype, periodic=False)
            q_tensor[0:trunc_start] *= tw[-trunc_start:]
            q_tensor[trunc_end:N] *= tw[0:max_filter_len//2]
        else:
            trunc_window = Array(numpy.hanning(max_filter_len), dtype=q.dtype)
            q[0:trunc_start] *= trunc_window[-trunc_start:]
            q[trunc_end:N] *= trunc_window[0:max_filter_len//2]

    if trunc_start < trunc_end:
        q[trunc_start:trunc_end] = 0
    if not USE_CACHING_FOR_INV_SPEC_TRUNC:
        psd_trunc = FrequencySeries(
            zeros(len(psd), dtype=complex_same_precision_as(psd)),
            delta_f=psd.delta_f
        )
        fft(q, psd_trunc)
    else:
        from pycbc.strain.strain import execute_cached_fft
        psd_trunc = execute_cached_fft(q, copy_output=False,
                                       uid=INVSPECTRUNC_UNIQUE_ID)
    if which_spectrum == 'invasd':
        psd_trunc *= psd_trunc.conj()

    if use_torch:
        psd_out = 1. / torch.abs(backend_array(psd_trunc, "torch"))
        return FrequencySeries(wrap_backend_array(psd_out), delta_f=psd.delta_f,
                               epoch=psd.epoch, copy=False)
    psd_out = 1. / abs(psd_trunc)

    return psd_out

def interpolate(series, delta_f, length=None):
    """Return a new PSD that has been interpolated to the desired delta_f.

    Parameters
    ----------
    series : FrequencySeries
        Frequency series to be interpolated.
    delta_f : float
        The desired delta_f of the output
    length : None or int
        The desired number of frequency samples. The default is None,
        so it will be calculated from the given `series` and `delta_f`.
        But this will cause an inconsistency issue of length sometimes,
        so if `length` is given, then just use it.

    Returns
    -------
    interpolated series : FrequencySeries
        A new FrequencySeries that has been interpolated.
    """
    if length is None:
        new_n = (len(series)-1) * series.delta_f / delta_f + 1
    else:
        new_n = length

    use_torch = _is_torch_series(series)
    if use_torch:
        # torch.rint was removed in newer torch; use python round for length
        nsamp = int(round(float(new_n)))
        old_vals = backend_array(series, "torch")
        if old_vals.numel() == 0:
            raise ValueError("array of sample points is empty")

        # FrequencySeries samples are uniformly spaced, so interpolate from
        # fractional source-bin positions without constructing a second
        # frequency grid. Clamping both neighbors to the final bin preserves
        # numpy.interp's constant right-edge behavior for an explicit length
        # that extends beyond the input band (and also handles one-bin PSDs).
        dtype = old_vals.real.dtype
        positions = torch.arange(
            nsamp, device=old_vals.device, dtype=dtype
        ) * (delta_f / series.delta_f)
        lower_unclamped = torch.floor(positions)
        last = old_vals.numel() - 1
        idx_lo = lower_unclamped.to(torch.long).clamp(0, last)
        idx_hi = (idx_lo + 1).clamp(max=last)
        weight = positions - lower_unclamped
        interpolated_series = old_vals[idx_lo] + (
            old_vals[idx_hi] - old_vals[idx_lo]
        ) * weight
        return FrequencySeries(wrap_backend_array(interpolated_series),
                               epoch=series.epoch,
                               delta_f=delta_f, copy=False)

    samples = numpy.arange(0, numpy.rint(new_n)) * delta_f
    interpolated_series = numpy.interp(samples, series.sample_frequencies.numpy(), series.numpy())
    return FrequencySeries(interpolated_series, epoch=series.epoch,
                           delta_f=delta_f, dtype=series.dtype)

def bandlimited_interpolate(series, delta_f):
    """Return a new PSD that has been interpolated to the desired delta_f.

    Parameters
    ----------
    series : FrequencySeries
        Frequency series to be interpolated.
    delta_f : float
        The desired delta_f of the output

    Returns
    -------
    interpolated series : FrequencySeries
        A new FrequencySeries that has been interpolated.
    """
    series = FrequencySeries(series, dtype=complex_same_precision_as(series), delta_f=series.delta_f)

    N = (len(series) - 1) * 2
    delta_t = 1.0 / series.delta_f / N

    new_N = int(1.0 / (delta_t * delta_f))
    new_n = new_N // 2 + 1

    series_in_time = TimeSeries(zeros(N), dtype=real_same_precision_as(series), delta_t=delta_t)
    ifft(series, series_in_time)

    padded_series_in_time = TimeSeries(zeros(new_N), dtype=series_in_time.dtype, delta_t=delta_t)
    padded_series_in_time[0:N//2] = series_in_time[0:N//2]
    padded_series_in_time[new_N-N//2:new_N] = series_in_time[N//2:N]

    interpolated_series = FrequencySeries(zeros(new_n), dtype=series.dtype, delta_f=delta_f)
    fft(padded_series_in_time, interpolated_series)

    return interpolated_series
