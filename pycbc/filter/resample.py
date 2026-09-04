# Copyright (C) 2012  Alex Nitz
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


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
from pycbc.types.backend import (
    backend_array, is_backend, wrap_backend_array,
)
import functools
import operator

import numpy
import scipy.signal

import lal
from pycbc.types import (
    Array,
    FrequencySeries,
    TimeSeries,
    complex_same_precision_as,
    real_same_precision_as,
    zeros,
)
from pycbc.fft import ifft, fft
import pycbc

try:
    import torch
    from .zpk import _torch_sosfilt
    _HAVE_TORCH = pycbc.HAVE_TORCH
except Exception:  # pragma: no cover - torch optional
    torch = None
    _torch_sosfilt = None
    _HAVE_TORCH = False

_resample_func = {
    numpy.dtype('float32'): lal.ResampleREAL4TimeSeries,
    numpy.dtype('float64'): lal.ResampleREAL8TimeSeries,
}

@functools.lru_cache(maxsize=20)
def cached_firwin(*args, **kwargs):
    """Cache the FIR filter coefficients.
    This is mostly done for PyCBC Live, which rapidly and repeatedly resamples data.
    """
    return scipy.signal.firwin(*args, **kwargs)


@functools.lru_cache(maxsize=20)
def _butterworth_resample_sos(resample_factor):
    """Construct the second-order sections used by LAL's resampler."""

    filter_order = 20
    nyquist_amplitude = 0.1
    cutoff = numpy.tan(numpy.pi * 0.5 / resample_factor)
    cutoff *= (1 / numpy.sqrt(nyquist_amplitude) - 1) ** (
        -0.5 / filter_order
    )
    return _butterworth_sos(cutoff, filter_order, highpass=False)


@functools.lru_cache(maxsize=100)
def _butterworth_sos(cutoff, filter_order, highpass):
    """Construct LAL's transformed-frequency Butterworth sections."""

    sections = []
    for index in range(filter_order // 2):
        theta = numpy.pi * (index + 0.5) / filter_order
        pole_real = cutoff * numpy.cos(theta)
        pole_imag = cutoff * numpy.sin(theta)
        pole = ((1 - pole_imag) + 1j * pole_real) / (
            (1 + pole_imag) - 1j * pole_real
        )
        gain = (1 if highpass else cutoff**2) / (
            pole_real**2 + (1 + pole_imag) ** 2
        )
        sections.append(
            [
                gain,
                (-2 if highpass else 2) * gain,
                gain,
                1,
                -2 * pole.real,
                abs(pole) ** 2,
            ]
        )

    if filter_order % 2:
        pole = (1 - cutoff) / (1 + cutoff)
        gain = (1 if highpass else cutoff) / (1 + cutoff)
        sections.append(
            [
                gain,
                (-1 if highpass else 1) * gain,
                0,
                1,
                -pole,
                0,
            ]
        )
    return numpy.asarray(sections)


def _torch_zero_phase_sos(data, sections):
    """Apply LAL's section-by-section forward/reverse filtering."""

    for section in sections:
        section = section[numpy.newaxis, :]
        data = _torch_sosfilt(section, data)
        data = _torch_sosfilt(section, data.flip(0)).flip(0)
    return data


def _torch_butterworth_resample(timeseries, delta_t):
    """Reproduce LAL's Butterworth resampler on the active Torch device."""

    ratio = delta_t / timeseries.delta_t
    factor = int(numpy.floor(ratio + 0.5))
    if (
        factor < 1
        or abs(delta_t - factor * timeseries.delta_t)
        > 1e-3 * timeseries.delta_t
        or factor & (factor - 1)
    ):
        raise RuntimeError("Invalid argument")

    data = backend_array(timeseries, "torch")
    if factor == 1:
        return TimeSeries(
            wrap_backend_array(data.clone()),
            delta_t=delta_t,
            epoch=timeseries._epoch,
            copy=False,
        )

    # LAL constructs each pole pair in the transformed-frequency plane,
    # maps it bilinearly to one digital section, then filters forward and
    # backward before advancing to the next pair. Calling the device scan one
    # section at a time preserves both that boundary behavior and REAL4's
    # per-pass rounding.
    data = _torch_zero_phase_sos(
        data, _butterworth_resample_sos(factor)
    )

    # LAL truncates the input length before decimation. Cloning releases the
    # full filtered allocation instead of retaining it through a strided view.
    output_length = data.numel() // factor
    data = data[: output_length * factor : factor].clone()
    return TimeSeries(
        wrap_backend_array(data),
        delta_t=delta_t,
        epoch=timeseries._epoch,
        copy=False,
    )


def _torch_butterworth_filter(
        timeseries, frequency, filter_order, attenuation, highpass):
    """Apply LAL's high- or low-pass Butterworth filter on device."""

    try:
        filter_order = operator.index(filter_order)
    except TypeError as exc:
        raise TypeError("filter_order must be an integer") from exc

    normalized_frequency = float(frequency) * timeseries.delta_t
    if not 0 < normalized_frequency < 0.5 or filter_order <= 0:
        raise RuntimeError("Internal function call failed: Invalid argument")

    cutoff = numpy.tan(numpy.pi * normalized_frequency)
    amplitude = float(1 - attenuation)
    if 0 < amplitude < 1:
        exponent = 0.5 / filter_order
        if not highpass:
            exponent *= -1
        cutoff *= (1 / numpy.sqrt(amplitude) - 1) ** exponent

    data = _torch_zero_phase_sos(
        backend_array(timeseries, "torch"),
        _butterworth_sos(cutoff, filter_order, highpass),
    )
    return TimeSeries(
        wrap_backend_array(data),
        delta_t=timeseries.delta_t,
        epoch=timeseries._epoch,
        copy=False,
    )


# Change to True in front-end if you want this function to use caching
# This is a mostly-hidden optimization option that most users will not want
# to use. It is used in PyCBC Live
USE_CACHING_FOR_LFILTER = False
# If using caching we want output to be unique if called at different places
# (and if called from different modules/functions), these unique IDs acheive
# that. The numbers are not significant, only that they are unique.
LFILTER_UNIQUE_ID_1 = 651273657
LFILTER_UNIQUE_ID_2 = 154687641
LFILTER_UNIQUE_ID_3 = 548946442

# Bound temporary Torch FFT allocations while keeping the number of device
# launches modest for long time series.
_TORCH_LFILTER_TARGET_BLOCK_SIZE = 2**18


def _torch_lfilter_work_dtype(signal):
    """Return the dtype used for Torch FFT convolution.

    Single-precision CUDA FFT roundoff is large enough to survive the
    conditioning pipeline and exceed PyCBC's pointwise parity gate.  Use
    double-precision CUDA intermediates while preserving the public dtype.
    CPU and MPS keep their existing precision and performance.
    """
    if signal.device.type != "cuda":
        return signal.dtype
    return {
        torch.float32: torch.float64,
        torch.complex64: torch.complex128,
    }.get(signal.dtype, signal.dtype)


def _torch_lfilter(coefficients, timeseries):
    """Apply a causal FIR filter to a Torch-backed time series on device."""
    signal = backend_array(timeseries, "torch")
    output_dtype = signal.dtype
    work_dtype = _torch_lfilter_work_dtype(signal)
    work_signal = signal.to(dtype=work_dtype)
    coefficient_data = backend_array(coefficients)
    if is_backend(coefficient_data, "torch"):
        coefficient_data = backend_array(coefficient_data, "torch")
    elif isinstance(coefficient_data, numpy.ndarray):
        coefficient_data = numpy.ascontiguousarray(coefficient_data)

    coefficient_tensor = torch.as_tensor(
        coefficient_data, device=signal.device, dtype=work_dtype
    )
    if coefficient_tensor.ndim != 1:
        raise ValueError("Filter coefficients must be one-dimensional")
    if coefficient_tensor.numel() == 0:
        raise ValueError("Filter coefficients cannot be empty")

    signal_length = work_signal.numel()
    coefficient_length = coefficient_tensor.numel()
    target_block_length = min(
        signal_length, _TORCH_LFILTER_TARGET_BLOCK_SIZE
    )
    convolution_length = target_block_length + coefficient_length - 1
    fft_length = 1 << (convolution_length - 1).bit_length()
    block_length = fft_length - coefficient_length + 1

    output = torch.zeros_like(work_signal)
    if work_signal.is_complex():
        response = torch.fft.fft(coefficient_tensor, n=fft_length)
        transform = torch.fft.fft
        inverse_transform = torch.fft.ifft
    else:
        response = torch.fft.rfft(coefficient_tensor, n=fft_length)
        transform = torch.fft.rfft
        inverse_transform = torch.fft.irfft

    for start in range(0, signal_length, block_length):
        block = work_signal[start:start + block_length]
        filtered = inverse_transform(
            transform(block, n=fft_length) * response, n=fft_length
        )
        output_length = min(fft_length, signal_length - start)
        output[start:start + output_length] += filtered[:output_length]

    output = output.to(dtype=output_dtype)

    return TimeSeries(
        wrap_backend_array(output), delta_t=timeseries.delta_t,
        epoch=timeseries.start_time, copy=False
    )


def lfilter(coefficients, timeseries):
    """ Apply filter coefficients to a time series

    Parameters
    ----------
    coefficients : numpy.ndarray
        Filter coefficients to apply
    timeseries : pycbc.types.TimeSeries
        Time series to be filtered.

    Returns
    -------
    tseries : pycbc.types.TimeSeries
        filtered array
    """
    torch_input = _HAVE_TORCH and is_backend(timeseries, "torch")
    if torch_input:
        return _torch_lfilter(coefficients, timeseries)

    fillen = len(coefficients)
    from pycbc.filter import correlate

    # If there aren't many points just use the default scipy method
    if len(timeseries) < 2**7:
        series = scipy.signal.lfilter(coefficients, 1.0, timeseries)
        return TimeSeries(series,
                          epoch=timeseries.start_time,
                          delta_t=timeseries.delta_t)
    elif (len(timeseries) < fillen * 10) or (len(timeseries) < 2**18):
        from pycbc.strain.strain import create_memory_and_engine_for_class_based_fft
        from pycbc.strain.strain import execute_cached_fft

        cseries = (Array(coefficients[::-1] * 1)).astype(timeseries.dtype)
        cseries.resize(len(timeseries))
        cseries.roll(len(timeseries) - fillen + 1)

        flen = len(cseries) // 2 + 1
        ftype = complex_same_precision_as(timeseries)

        if not USE_CACHING_FOR_LFILTER:
            cfreq = zeros(flen, dtype=ftype)
            tfreq = zeros(flen, dtype=ftype)
            fft(Array(cseries), cfreq)
            fft(Array(timeseries), tfreq)
            cout = zeros(flen, ftype)
            correlate(cfreq, tfreq, cout)
            out = zeros(len(timeseries), dtype=timeseries)
            ifft(cout, out)

        else:
            npoints = len(cseries)
            # NOTE: This function is cached!
            ifftouts = create_memory_and_engine_for_class_based_fft(
                npoints,
                timeseries.dtype,
                ifft=True,
                uid=LFILTER_UNIQUE_ID_1
            )

            # FFT contents of cseries into cfreq
            cfreq = execute_cached_fft(cseries, uid=LFILTER_UNIQUE_ID_2,
                                       copy_output=False,
                                       normalize_by_rate=False)

            # FFT contents of timeseries into tfreq
            tfreq = execute_cached_fft(timeseries, uid=LFILTER_UNIQUE_ID_3,
                                       copy_output=False,
                                       normalize_by_rate=False)

            cout, out, fft_class = ifftouts

            # Correlate cfreq and tfreq
            correlate(cfreq, tfreq, cout)
            # IFFT correlation output into out
            fft_class.execute()

        return TimeSeries(out.numpy()  / len(out), epoch=timeseries.start_time,
                          delta_t=timeseries.delta_t)
    else:
        # recursively perform which saves a bit on memory usage
        # but must keep within recursion limit
        chunksize = max(fillen * 5, len(timeseries) // 2)
        part1 = lfilter(coefficients, timeseries[0:chunksize])
        part2 = lfilter(coefficients, timeseries[chunksize - fillen:])
        out = timeseries.copy()
        out[:len(part1)] = part1
        out[len(part1):] = part2[fillen:]
        return out

def fir_zero_filter(coeff, timeseries):
    """Filter the timeseries with a set of FIR coefficients

    Parameters
    ----------
    coeff: numpy.ndarray
        FIR coefficients. Should be and odd length and symmetric.
    timeseries: pycbc.types.TimeSeries
        Time series to be filtered.

    Returns
    -------
    filtered_series: pycbc.types.TimeSeries
        Return the filtered timeseries, which has been properly shifted to account
    for the FIR filter delay and the corrupted regions zeroed out.
    """
    # ``lfilter`` dispatches to the blocked Torch FIR implementation for
    # device-backed inputs. Reusing it here preserves linear-convolution
    # semantics and supports both real and complex time series.
    series = lfilter(coeff, timeseries)

    # reverse the time shift caused by the filter,
    # corruption regions contain zeros
    # If the number of filter coefficients is odd, the central point *should*
    # be included in the output so we only zero out a region of len(coeff) - 1
    series[:(len(coeff) // 2) * 2] = 0
    series.roll(-len(coeff)//2)
    return series

def resample_to_delta_t(timeseries, delta_t, method='butterworth'):
    """Resample the time series to ``delta_t``.

    Resamples the TimeSeries instance time_series to the given time step,
    delta_t. Only powers of two and real valued time series are supported
    at this time. Additional restrictions may apply to particular filter
    methods.

    Parameters
    ----------
    time_series: TimeSeries
        The time series to be resampled
    delta_t: float
        The desired time step
    method: {"butterworth", "ldas"}
        Low-pass filter to apply before decimation.

    Returns
    -------
    Time Series: TimeSeries
        A TimeSeries that has been resampled to delta_t.

    Raises
    ------
    TypeError:
        time_series is not an instance of TimeSeries.
    TypeError:
        time_series is not real valued

    Examples
    --------

    >>> h_plus_sampled = resample_to_delta_t(h_plus, 1.0/2048)
    """
    if not isinstance(timeseries, TimeSeries):
        raise TypeError("Can only resample time series")

    if timeseries.kind != 'real':
        raise TypeError("Time series must be real")

    torch_input = _HAVE_TORCH and is_backend(timeseries, "torch")
    if timeseries.sample_rate_close(1.0 / delta_t):
        return timeseries * 1

    if method == 'butterworth':
        if torch_input:
            ts = _torch_butterworth_resample(timeseries, delta_t)
        else:
            lal_data = timeseries.lal()
            _resample_func[timeseries.dtype](lal_data, delta_t)
            data = lal_data.data.data
            ts = TimeSeries(
                data,
                delta_t=delta_t,
                epoch=timeseries.start_time,
                copy=True,
            )

        # Preserve the metadata contract of the historical shared return path.
        ts.corrupted_samples = 10
        return ts

    elif method == 'ldas':
        factor = int(round(delta_t / timeseries.delta_t))
        numtaps = factor * 20 + 1

        # The kaiser window has been testing using the LDAS implementation
        # and is in the same configuration as used in the original lalinspiral
        filter_coefficients = cached_firwin(numtaps, 1.0 / factor,
                                            window=('kaiser', 5))

        # apply the filter and decimate
        data = fir_zero_filter(filter_coefficients, timeseries)[::factor]

    else:
        raise ValueError('Invalid resampling method: %s' % method)

    if torch_input:
        # ``data`` is already a Torch-backed, decimated TimeSeries. Retain its
        # device storage instead of copying it through NumPy and back.
        ts = TimeSeries(data, delta_t=delta_t, epoch=timeseries._epoch,
                        copy=False)
    else:
        ts = TimeSeries(data, delta_t=delta_t, dtype=timeseries.dtype,
                        epoch=timeseries._epoch)

    # From the construction of the LDAS FIR filter there will be 10 corrupted samples
    # explanation here https://lscsoft.docs.ligo.org/lalsuite/lal/group___resample_time_series__c.html
    ts.corrupted_samples = 10
    return ts


_highpass_func = {
    numpy.dtype('float32'): lal.HighPassREAL4TimeSeries,
    numpy.dtype('float64'): lal.HighPassREAL8TimeSeries,
}
_lowpass_func = {
    numpy.dtype('float32'): lal.LowPassREAL4TimeSeries,
    numpy.dtype('float64'): lal.LowPassREAL8TimeSeries,
}


def notch_fir(timeseries, f1, f2, order, beta=5.0):
    """ notch filter the time series using an FIR filtered generated from
    the ideal response passed through a time-domain kaiser window (beta = 5.0)

    The suppression of the notch filter is related to the bandwidth and
    the number of samples in the filter length. For a few Hz bandwidth,
    a length corresponding to a few seconds is typically
    required to create significant suppression in the notched band.
    To achieve frequency resolution df at sampling frequency fs,
    order should be at least fs/df.

    Parameters
    ----------
    Time Series: TimeSeries
        The time series to be notched.
    f1: float
        The start of the frequency suppression.
    f2: float
        The end of the frequency suppression.
    order: int
        Number of corrupted samples on each side of the time series
        (Extent of the filter on either side of zero)
    beta: float
        Beta parameter of the kaiser window that sets the side lobe attenuation.
    """
    k1 = f1 / float((int(1.0 / timeseries.delta_t) / 2))
    k2 = f2 / float((int(1.0 / timeseries.delta_t) / 2))
    coeff = cached_firwin(order * 2 + 1, [k1, k2], window=('kaiser', beta))
    return fir_zero_filter(coeff, timeseries)

def lowpass_fir(timeseries, frequency, order, beta=5.0):
    """ Lowpass filter the time series using an FIR filtered generated from
    the ideal response passed through a kaiser window (beta = 5.0)

    Parameters
    ----------
    Time Series: TimeSeries
        The time series to be low-passed.
    frequency: float
        The frequency below which is suppressed.
    order: int
        Number of corrupted samples on each side of the time series
    beta: float
        Beta parameter of the kaiser window that sets the side lobe attenuation.
    """
    k = frequency / float((int(1.0 / timeseries.delta_t) / 2))
    coeff = cached_firwin(order * 2 + 1, k, window=('kaiser', beta))
    return fir_zero_filter(coeff, timeseries)

def highpass_fir(timeseries, frequency, order, beta=5.0):
    """ Highpass filter the time series using an FIR filtered generated from
    the ideal response passed through a kaiser window (beta = 5.0)

    Parameters
    ----------
    Time Series: TimeSeries
        The time series to be high-passed.
    frequency: float
        The frequency below which is suppressed.
    order: int
        Number of corrupted samples on each side of the time series
    beta: float
        Beta parameter of the kaiser window that sets the side lobe attenuation.
    """
    k = frequency / float((int(1.0 / timeseries.delta_t) / 2))
    coeff = cached_firwin(order * 2 + 1, k, window=('kaiser', beta), pass_zero=False)
    return fir_zero_filter(coeff, timeseries)

def highpass(timeseries, frequency, filter_order=8, attenuation=0.1):
    """Return a new timeseries that is highpassed.

    Return a new time series that is highpassed above the `frequency`.

    Parameters
    ----------
    Time Series: TimeSeries
        The time series to be high-passed.
    frequency: float
        The frequency below which is suppressed.
    filter_order: {8, int}, optional
        The order of the filter to use when high-passing the time series.
    attenuation: {0.1, float}, optional
        The attenuation of the filter.

    Returns
    -------
    Time Series: TimeSeries
        A  new TimeSeries that has been high-passed.

    Raises
    ------
    TypeError:
        time_series is not an instance of TimeSeries.
    TypeError:
        time_series is not real valued

    """

    if not isinstance(timeseries, TimeSeries):
        raise TypeError("Can only resample time series")

    if timeseries.kind != 'real':
        raise TypeError("Time series must be real")

    if _HAVE_TORCH and is_backend(timeseries, "torch"):
        return _torch_butterworth_filter(
            timeseries,
            frequency,
            filter_order,
            attenuation,
            highpass=True,
        )

    lal_data = timeseries.lal()
    _highpass_func[timeseries.dtype](lal_data, frequency,
                                     1-attenuation, filter_order)

    return TimeSeries(lal_data.data.data, delta_t = lal_data.deltaT,
                      dtype=timeseries.dtype, epoch=timeseries._epoch)

def lowpass(timeseries, frequency, filter_order=8, attenuation=0.1):
    """Return a new timeseries that is lowpassed.

    Return a new time series that is lowpassed below the `frequency`.

    Parameters
    ----------
    Time Series: TimeSeries
        The time series to be low-passed.
    frequency: float
        The frequency above which is suppressed.
    filter_order: {8, int}, optional
        The order of the filter to use when low-passing the time series.
    attenuation: {0.1, float}, optional
        The attenuation of the filter.

    Returns
    -------
    Time Series: TimeSeries
        A  new TimeSeries that has been low-passed.

    Raises
    ------
    TypeError:
        time_series is not an instance of TimeSeries.
    TypeError:
        time_series is not real valued
    """

    if not isinstance(timeseries, TimeSeries):
        raise TypeError("Can only resample time series")

    if timeseries.kind != 'real':
        raise TypeError("Time series must be real")

    if _HAVE_TORCH and is_backend(timeseries, "torch"):
        return _torch_butterworth_filter(
            timeseries,
            frequency,
            filter_order,
            attenuation,
            highpass=False,
        )

    lal_data = timeseries.lal()
    _lowpass_func[timeseries.dtype](lal_data, frequency,
                                    1-attenuation, filter_order)

    return TimeSeries(lal_data.data.data, delta_t = lal_data.deltaT,
                      dtype=timeseries.dtype, epoch=timeseries._epoch)


def interpolate_complex_frequency(series, delta_f, zeros_offset=0, side='right'):
    """Interpolate complex frequency series to desired delta_f.

    Return a new complex frequency series that has been interpolated to the
    desired delta_f.

    Parameters
    ----------
    series : FrequencySeries
        Frequency series to be interpolated.
    delta_f : float
        The desired delta_f of the output
    zeros_offset : optional, {0, int}
        Number of sample to delay the start of the zero padding
    side : optional, {'right', str}
        The side of the vector to zero pad

    Returns
    -------
    interpolated series : FrequencySeries
        A new FrequencySeries that has been interpolated.
    """
    new_n = int( (len(series)-1) * series.delta_f / delta_f + 1)
    old_N = int( (len(series)-1) * 2 )
    new_N = int( (new_n - 1) * 2 )
    time_series = TimeSeries(zeros(old_N), delta_t =1.0/(series.delta_f*old_N),
                             dtype=real_same_precision_as(series))

    ifft(series, time_series)

    time_series.roll(-zeros_offset)
    time_series.resize(new_N)

    if side == 'left':
        time_series.roll(zeros_offset + new_N - old_N)
    elif side == 'right':
        time_series.roll(zeros_offset)

    out_series = FrequencySeries(zeros(new_n), epoch=series.epoch,
                           delta_f=delta_f, dtype=series.dtype)
    fft(time_series, out_series)

    return out_series

__all__ = ['resample_to_delta_t', 'highpass', 'lowpass',
           'interpolate_complex_frequency', 'highpass_fir',
           'lowpass_fir', 'notch_fir', 'fir_zero_filter']
