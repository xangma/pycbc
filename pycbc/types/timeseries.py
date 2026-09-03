# Copyright (C) 2014  Tito Dal Canton, Josh Willis, Alex Nitz
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

"""
Provides a class representing a time series.
"""
import os as _os
import h5py
import numpy as _numpy
try:
    from igwn_segments import segmentlist, segment
except ImportError:
    segmentlist = segment = None

from pycbc.types.array import (
    Array,
    _convert,
    _regular_grid,
    complex_same_precision_as,
    zeros,
)
from pycbc.types.utils import determine_epoch
from pycbc.types.array import _nocomplex
from pycbc.types.frequencyseries import FrequencySeries
from pycbc.types import float32, float64
from pycbc import lal_compat as _lal
from scipy.io.wavfile import write as write_wav

_TORCH_AT_TIME_FALLBACK = object()


def _torch_has_autograd_state(tensor, torch):
    """Whether native scalar arithmetic would discard Torch AD state."""
    if tensor.requires_grad:
        return True
    try:
        return (
            torch.autograd.forward_ad.unpack_dual(tensor).tangent is not None
        )
    except (AttributeError, RuntimeError):
        # Keep an uninspectable tensor on Torch's autograd-aware path.
        return True


def _torch_at_time_host_scalar_numpy(tensor, index, offset, interpolate):
    """Interpolate an ordinary CPU scalar through a zero-copy NumPy view.

    Several scalar Torch indexing and arithmetic operations each dispatch a
    separate kernel, even on CPU.  A contiguous tensor that cannot carry a
    gradient can instead expose its existing storage to NumPy and perform the
    same scalar operations natively.  Return a sentinel for every storage or
    indexing case whose established Torch behavior must be retained.
    """
    import torch

    if (
        interpolate not in ('linear', 'quadratic')
        or tensor.device.type != 'cpu'
        or tensor.layout != torch.strided
        or tensor.ndim != 1
        or not tensor.is_contiguous()
        or tensor.is_conj()
        or tensor.is_neg()
    ):
        return _TORCH_AT_TIME_FALLBACK
    if _torch_has_autograd_state(tensor, torch):
        return _TORCH_AT_TIME_FALLBACK

    size = tensor.numel()
    required = (index, index + 1)
    if interpolate == 'quadratic':
        required += (index - 1,)
    if not all(-size <= sample_index < size for sample_index in required):
        return _TORCH_AT_TIME_FALLBACK

    values = tensor.numpy()
    if interpolate == 'linear':
        a = values[index]
        b = values[index + 1]
        result = a + (b - a) * offset
    else:
        c = values[index]
        xr = values[index + 1] - c
        xl = values[index - 1] - c
        a = 0.5 * (xr + xl)
        b = 0.5 * (xr - xl)
        offset_squared = type(offset)(offset * offset)
        result = a * offset_squared + b * offset + c

    # Keep the public result as an ordinary owning zero-dimensional tensor.
    # In particular, do not return a ``from_numpy`` view whose storage cannot
    # be resized like the result from the established Torch interpolation.
    return tensor.new_tensor(result)


def _torch_at_time_host_scalar(
        series, tensor, time, nearest_sample, interpolate, extrapolate):
    """Evaluate one host-time query without device coordinate tensors.

    Ordinary scalar times cannot carry Torch gradients.  Computing their
    index and fractional offset as host scalars avoids several one-element
    kernels while all sampled data and differentiable interpolation remain on
    ``tensor.device``.  A sentinel retains the generic path for non-finite
    coordinates that are not handled by extrapolation.
    """
    import torch

    try:
        relative_time = float(time) - float(series.start_time)
    except (TypeError, ValueError, OverflowError):
        return _TORCH_AT_TIME_FALLBACK

    if nearest_sample:
        relative_time += series.delta_t / 2.0

    if extrapolate is not None:
        if not (_numpy.isscalar(extrapolate)
                and _numpy.isreal(extrapolate)):
            raise ValueError(f"Unsupported extrapolate: {extrapolate}")
        facl = facr = 0
        if interpolate == 'quadratic':
            facl = facr = 1.1
        elif interpolate == 'linear':
            facl, facr = 0.1, 1.1
        keep = (
            relative_time >= series.delta_t * facl
            and relative_time < series.duration - series.delta_t * facr
        )
        if not keep:
            fill_value = extrapolate
            if not tensor.is_complex() and _numpy.iscomplexobj(fill_value):
                fill_value = fill_value.real
            result = torch.full(
                (), fill_value, device=tensor.device, dtype=tensor.dtype
            )
            if tensor.requires_grad:
                # The generic vector path retains a zero-valued dependency
                # through its empty selected slice. Preserve that established
                # backward contract for scalar extrapolation as well.
                result = result + tensor[:0].sum()
            return result

    fi = relative_time * series.sample_rate
    if not _numpy.isfinite(fi):
        return _TORCH_AT_TIME_FALLBACK
    index = int(_numpy.floor(fi))
    scalar_dtype = (
        _numpy.float32 if tensor.real.dtype == torch.float32
        else _numpy.float64
    )
    offset = scalar_dtype(fi - index)

    if extrapolate is None:
        native_result = _torch_at_time_host_scalar_numpy(
            tensor, index, offset, interpolate
        )
        if native_result is not _TORCH_AT_TIME_FALLBACK:
            return native_result

    if interpolate == 'linear':
        a = tensor[index]
        b = tensor[index + 1]
        return a + (b - a) * offset
    if interpolate == 'quadratic':
        c = tensor[index]
        xr = tensor[index + 1] - c
        xl = tensor[index - 1] - c
        a = 0.5 * (xr + xl)
        b = 0.5 * (xr - xl)
        offset_squared = scalar_dtype(offset * offset)
        return a * offset_squared + b * offset + c
    return tensor[index]


def _torch_whiten_work_dtype(data):
    """Return an optional dtype for the final Torch whitening transform."""
    import torch

    if (data.device.type == "cuda"
            and data.dtype == torch.float32
            and data.numel() % 2 == 0):
        return torch.float64
    return None


def _torch_taper_peak(amplitude, start, end, step):
    """Return the second eligible LAL taper peak in one direction.

    The returned index remains a scalar tensor on ``amplitude.device``.  In
    particular, this deliberately avoids ``nonzero`` and boolean-index
    compaction: both need the data-dependent result size on the host and can
    therefore synchronize a CUDA stream.
    """
    import torch

    count = amplitude.numel()
    indices = torch.arange(count, device=amplitude.device)

    # The boundary samples are excluded below, so their rolled neighbours do
    # not affect the result.  Roll keeps every intermediate fixed-size.
    previous = torch.roll(amplitude, 1)
    following = torch.roll(amplitude, -1)
    peaks = (amplitude >= previous) & (amplitude >= following)
    inside = (indices - start) * step > 0
    inside &= (end - indices) * step > 0
    candidates = peaks & inside

    # Work in the direction in which LAL scans the waveform.  A flat peak
    # advances LAL's loop by an extra sample.  Equivalently, within each run
    # of adjacent candidates only candidates 0, 2, 4, ... are visited.
    scan_candidates = candidates if step > 0 else candidates.flip(0)
    scan_order = indices
    prior_candidate = torch.cat((
        torch.zeros(1, dtype=torch.bool, device=amplitude.device),
        scan_candidates[:-1],
    ))
    new_run = scan_candidates & ~prior_candidate
    run_starts = torch.where(
        new_run, scan_order, torch.zeros_like(scan_order)
    )
    run_starts = torch.cummax(run_starts, dim=0).values
    visited = scan_candidates & (
        (scan_order - run_starts).remainder(2) == 0
    )

    neighbour = following if step > 0 else previous
    flat = amplitude == neighbour
    effective = indices + flat.to(indices.dtype) * step
    scan_effective = effective if step > 0 else effective.flip(0)
    eligible = visited & ((scan_effective - start) * step > 19)

    # Select the second eligible peak without compacting the candidates.  If
    # it does not exist, ``end`` is the midpoint fallback used by LAL.
    second = eligible & (eligible.to(torch.int64).cumsum(dim=0) == 2)
    selected = torch.where(second, scan_effective, end)
    if step > 0:
        return selected.amin()
    return selected.amax()


def _torch_apply_taper_side(result, indices, boundary, length, step, valid):
    """Apply one LAL taper side using only fixed-size device operations."""
    import torch

    distance = (indices - boundary) * step
    interior = valid & (distance > 0) & (distance < length - 1)

    distance = distance.to(dtype=result.dtype)
    extent = (length - 1).to(dtype=result.dtype)
    safe_distance = torch.where(
        interior, distance, torch.ones_like(distance)
    )
    safe_remainder = torch.where(
        interior, distance - extent, -torch.ones_like(distance)
    )
    exponent = extent / safe_distance + extent / safe_remainder
    taper = 1 / (torch.exp(exponent) + 1)
    result = result * torch.where(
        interior, taper, torch.ones_like(taper)
    )

    # Multiplication by zero would retain NaNs.  LAL assigns the endpoint,
    # so use where to reproduce that behavior for non-finite inputs too.
    endpoint = valid & (indices == boundary)
    return torch.where(endpoint, torch.zeros_like(result), result)


def _torch_lal_taper(tensor, location):
    """Apply LAL's inspiral waveform taper algorithm on a Torch device."""
    import torch

    result = tensor.clone()
    if location == 'none':
        return result

    count = result.numel()
    if count == 0:
        return result

    indices = torch.arange(count, device=result.device)
    nonzero = result != 0
    start = torch.where(
        nonzero, indices, torch.full_like(indices, count)
    ).amin()
    end = torch.where(
        nonzero, indices, torch.full_like(indices, -1)
    ).amax()
    valid = nonzero.any() & (end - start > 1)
    midpoint = (start + end) // 2
    amplitude = result.abs()

    if location != 'end':
        peak = _torch_taper_peak(amplitude, start, midpoint, 1)
        result = _torch_apply_taper_side(
            result, indices, start, peak - start, 1, valid
        )

    if location in ('end', 'startend'):
        peak = _torch_taper_peak(amplitude, end, midpoint, -1)
        result = _torch_apply_taper_side(
            result, indices, end, end - peak, -1, valid
        )

    return result


def _torch_constant_taper_parameters(count, delta_t, epoch, taper_window):
    """Return exact integer-rate parameters for the Torch constant taper.

    The legacy implementation locates each gate through ``LIGOTimeGPS``
    arithmetic. That rounds the sample time and taper width separately to
    nanoseconds before converting the result back to a sample offset. The
    integer construction below reproduces that behavior exactly for positive
    power-of-two sample rates, without extracting a data-dependent index from
    the device. Less common rates retain the established implementation.
    """
    if _lal is None:
        return None
    # Keep this deliberately narrow. In particular, unusual scalar classes
    # must continue through the legacy arithmetic and preserve its errors.
    if type(taper_window) not in (bool, int, float):
        return None

    window = float(taper_window)
    if not _numpy.isfinite(window) or window <= 0:
        return None

    sample_rate = 1.0 / delta_t
    if not _numpy.isfinite(sample_rate) or sample_rate <= 0:
        return None
    integer_rate = int(sample_rate)
    if sample_rate != integer_rate or integer_rate & (integer_rate - 1):
        return None

    nanoseconds = 1_000_000_000
    int64_max = _numpy.iinfo(_numpy.int64).max
    if integer_rate > int64_max // 2:
        return None

    # Preserve the LIGOTimeGPS conversion, including ties-to-even rounding.
    # Avoid taking this path near its signed-seconds boundary so the legacy
    # implementation remains responsible for the corresponding exceptions.
    if window >= 2**31 - 2:
        return None
    window_gps = _lal.LIGOTimeGPS(window)
    window_ns = (
        int(window_gps.gpsSeconds) * nanoseconds
        + int(window_gps.gpsNanoSeconds)
    )

    last_index = max(count - 1, 0)
    if last_index * nanoseconds > int64_max:
        return None
    max_center_ns = (
        last_index * nanoseconds + integer_rate - 1
    ) // integer_rate
    if (max_center_ns + abs(window_ns)) * integer_rate > int64_max:
        return None

    epoch_gps = epoch if hasattr(epoch, "gpsSeconds") else _lal.LIGOTimeGPS(epoch)
    epoch_seconds = int(epoch_gps.gpsSeconds)
    duration_seconds = (last_index + integer_rate - 1) // integer_rate
    if (
        epoch_seconds - int(_numpy.ceil(window)) - 2 <= -(2**31)
        or epoch_seconds + duration_seconds + 2 >= 2**31 - 1
    ):
        return None

    window_length = int(2 * sample_rate * window)
    padding_length = int(sample_rate * window)
    return integer_rate, window_ns, window_length, padding_length


def _torch_round_sample_time_ns(index, sample_rate):
    """Round ``index / sample_rate`` to GPS nanoseconds, ties to even."""
    import torch

    numerator = index * 1_000_000_000
    quotient = torch.div(numerator, sample_rate, rounding_mode='floor')
    remainder = numerator - quotient * sample_rate
    twice_remainder = 2 * remainder
    round_up = (twice_remainder > sample_rate) | (
        (twice_remainder == sample_rate) & (quotient.remainder(2) != 0)
    )
    return quotient + round_up.to(quotient.dtype)


def _torch_constant_taper(
        tensor, location, sample_rate, window_ns, window_length,
        padding_length):
    """Apply the legacy constant taper without a device-to-host sync."""
    import torch

    count = tensor.numel()
    if count == 0 or window_length == 0:
        return tensor

    indices = torch.arange(
        count, device=tensor.device, dtype=torch.int64
    )
    nonzero = tensor != 0
    valid = nonzero.any()
    first = torch.where(
        nonzero, indices, torch.full_like(indices, count)
    ).amin()
    last = torch.where(
        nonzero, indices, torch.full_like(indices, -1)
    ).amax()

    boundaries = []
    if location in ('start', 'startend'):
        boundaries.append(first)
    if location in ('end', 'startend'):
        boundaries.append(last)

    for boundary in boundaries:
        # Invalid all-zero sentinels are replaced before integer arithmetic;
        # ``valid`` then makes the corresponding multiplier exactly one.
        boundary = torch.where(valid, boundary, torch.zeros_like(boundary))
        center_ns = _torch_round_sample_time_ns(boundary, sample_rate)
        relative_ns = center_ns - window_ns
        product_ns = relative_ns * sample_rate
        offset = torch.div(
            product_ns, 1_000_000_000, rounding_mode='trunc'
        )

        window_index = indices - offset
        inside = valid & (window_index >= 0) & (
            window_index < window_length
        )
        if padding_length:
            padding_region = (window_index < padding_length) | (
                window_index >= window_length - padding_length
            )
            padding_index = torch.where(
                window_index < padding_length,
                window_index,
                window_length - 1 - window_index,
            ).clamp(0, padding_length - 1)
            padding_value = 0.5 * (
                1.0 + torch.cos(
                    torch.pi * padding_index.to(tensor.dtype)
                    / padding_length
                )
            )
            selected = torch.where(
                padding_region, padding_value,
                torch.zeros_like(tensor),
            )
        else:
            selected = torch.zeros_like(tensor)
        multiplier = torch.where(
            inside, selected, torch.ones_like(tensor)
        )
        tensor.mul_(multiplier)

    return tensor


class TimeSeries(Array):
    """Models a time series consisting of uniformly sampled scalar values.

    Parameters
    ----------
    initial_array : array-like
        Array containing sampled data.
    delta_t : float
        Time between consecutive samples in seconds.
    epoch : {None, lal.LIGOTimeGPS}, optional
        Time of the first sample in seconds.
    dtype : {None, data-type}, optional
        Sample data type.
    copy : boolean, optional
        If True, samples are copied to a new array.
    """

    def __init__(self, initial_array, delta_t=None,
                 epoch="", dtype=None, copy=True):
        if len(initial_array) < 1:
            raise ValueError('initial_array must contain at least one sample.')
        if delta_t is None:
            try:
                delta_t = initial_array.delta_t
            except AttributeError:
                raise TypeError('must provide either an initial_array with a delta_t attribute, or a value for delta_t')
        if not delta_t > 0:
            raise ValueError('delta_t must be a positive number')

        self._epoch = determine_epoch(epoch, initial_array)
        Array.__init__(self, initial_array, dtype=dtype, copy=copy)
        self._delta_t = delta_t

    def to_astropy(self, name='pycbc'):
        """ Return an astropy.timeseries.TimeSeries instance
        """
        from astropy.timeseries import TimeSeries as ATimeSeries
        from astropy.time import Time
        from astropy.units import s

        start = Time(float(self.start_time), format='gps', scale='utc')
        delta = self.delta_t * s
        return ATimeSeries({name: self.numpy()},
                           time_start=start,
                           time_delta=delta,
                           n_samples=len(self))

    def epoch_close(self, other):
        """ Check if the epoch is close enough to allow operations """
        if self._epoch is None or other._epoch is None:
            return False
        dt = abs(float(self.start_time - other.start_time))
        return dt <= 1e-7

    def sample_rate_close(self, other):
        """ Check if the sample rate is close enough to allow operations """

        # compare our delta_t either to a another time series' or
        # to a given sample rate (float)
        if isinstance(other, TimeSeries):
            odelta_t = other.delta_t
        else:
            odelta_t = 1.0/other

        if (odelta_t - self.delta_t) / self.delta_t > 1e-4:
            return False

        if abs(1 - odelta_t / self.delta_t) * len(self) > 0.5:
            return False

        return True

    def _return(self, ary):
        return TimeSeries(ary, self._delta_t, epoch=self._epoch, copy=False)

    def _typecheck(self, other):
        if isinstance(other, TimeSeries):
            if not self.sample_rate_close(other):
                raise ValueError('different delta_t, {} vs {}'.format(
                    self.delta_t, other.delta_t))
            if not self.epoch_close(other):
                raise ValueError('different epoch, {} vs {}'.format(
                    self.start_time, other.start_time))

    def _getslice(self, index):
        # Set the new epoch - index.start or self._epoch may be None
        if index.start is None or self._epoch is None:
            new_epoch = self._epoch
        else:
            if index.start < 0:
                raise ValueError(('Negative start index ({})'
                                  ' not supported').format(index.start))
            new_epoch = self._epoch + index.start * self._delta_t

        if index.step is not None:
            new_delta_t = self._delta_t * index.step
        else:
            new_delta_t = self._delta_t

        return TimeSeries(Array._getslice(self, index), new_delta_t,
                          new_epoch, copy=False)


    def prepend_zeros(self, num):
        """Prepend num zeros onto the beginning of this TimeSeries. Update also
        epoch to include this prepending.
        """
        self.resize(len(self) + num)
        self.roll(num)
        self._epoch = self._epoch - num * self._delta_t

    def append_zeros(self, num):
        """Append num zeros onto the end of this TimeSeries.
        """
        self.resize(len(self) + num)

    def get_delta_t(self):
        """Return time between consecutive samples in seconds.
        """
        return self._delta_t
    delta_t = property(get_delta_t,
                       doc="Time between consecutive samples in seconds.")

    def get_duration(self):
        """Return duration of time series in seconds.
        """
        return len(self) * self._delta_t
    duration = property(get_duration,
                        doc="Duration of time series in seconds.")

    def get_sample_rate(self):
        """Return the sample rate of the time series.
        """
        return 1.0/self.delta_t
    sample_rate = property(get_sample_rate,
                           doc="The sample rate of the time series.")

    def time_slice(self, start, end, mode='floor'):
        """Return the slice of the time series that contains the time range
        in GPS seconds.
        """
        if start < self.start_time:
            raise ValueError('Time series does not contain a time as early as %s' % start)

        if end > self.end_time:
            raise ValueError('Time series does not contain a time as late as %s' % end)

        start_idx = float(start - self.start_time) * self.sample_rate
        end_idx = float(end - self.start_time) * self.sample_rate

        if _numpy.isclose(start_idx, round(start_idx), rtol=0, atol=1E-3):
            start_idx = round(start_idx)

        if _numpy.isclose(end_idx, round(end_idx), rtol=0, atol=1E-3):
            end_idx = round(end_idx)

        if mode == 'floor':
            start_idx = int(start_idx)
            end_idx = int(end_idx)
        elif mode == 'nearest':
            start_idx = int(round(start_idx))
            end_idx = int(round(end_idx))
        else:
            raise ValueError("Invalid mode: {}".format(mode))

        return self[start_idx:end_idx]

    @property
    def delta_f(self):
        """Return the delta_f this ts would have in the frequency domain
        """
        return 1.0 / self.duration

    @property
    def start_time(self):
        """Return time series start time.
        """
        return self._epoch

    @start_time.setter
    def start_time(self, time):
        """ Set the start time
        """
        self._epoch = float64(time)

    def get_end_time(self):
        """Return time series end time.
        """
        return self._epoch + self.get_duration()
    end_time = property(get_end_time,
                        doc="Time series end time.")

    def get_sample_times(self):
        """Return an Array containing the sample times.
        """
        epoch = None if self._epoch is None else float(self._epoch)
        return _regular_grid(len(self), self._delta_t, offset=epoch)
    sample_times = property(get_sample_times,
                            doc="Array containing the sample times.")

    def at_time(self, time, nearest_sample=False,
                interpolate=None, extrapolate=None):
        """Return the value of the TimeSeries at the specified GPS time.

        Parameters
        ----------
        time: scalar or array-like
            GPS time at which the value is wanted. Note that LIGOTimeGPS
            objects count as scalar.
        nearest_sample: bool
            Return the sample at the time nearest to the chosen time rather
            than rounded down.
        interpolate: str, None
            Return the interpolated value of the time series. Choices
            are simple linear or quadratic interpolation.
        extrapolate: str or float, None
            Value to return if time is outside the range of the vector or
            method of extrapolating the value.
        """
        tensor = getattr(self._data, 'tensor', None)
        if tensor is not None:
            import torch

            query_tensor = getattr(
                getattr(time, '_data', None), 'tensor', None
            )
            if query_tensor is None and isinstance(time, torch.Tensor):
                query_tensor = time

            if (
                query_tensor is None
                and tensor.device.type in ('cpu', 'cuda')
                and tensor.real.dtype in (torch.float32, torch.float64)
                and _numpy.ndim(time) == 0
            ):
                scalar_result = _torch_at_time_host_scalar(
                    self, tensor, time, nearest_sample,
                    interpolate, extrapolate
                )
                if scalar_result is not _TORCH_AT_TIME_FALLBACK:
                    return scalar_result

            if query_tensor is not None:
                if query_tensor.is_complex():
                    raise TypeError("Time values must be real")
                scalar_input = query_tensor.ndim == 0
                coordinate_dtype = (
                    tensor.real.dtype
                    if tensor.device.type == 'mps'
                    else torch.float64
                )
                relative_times = query_tensor.to(
                    device=tensor.device, dtype=coordinate_dtype
                ).reshape(-1)
                relative_times = relative_times - float(self.start_time)
            else:
                scalar_input = _numpy.ndim(time) == 0
                values = [float(time)] if scalar_input else time
                if tensor.device.type == 'mps':
                    # MPS has no float64 support. Center host-provided query
                    # coordinates before their one-way upload so GPS epochs
                    # do not lose their fractional samples in float32.
                    values = (
                        _numpy.asarray(values, dtype=_numpy.float64)
                        - float(self.start_time)
                    )
                    relative_times = torch.as_tensor(
                        values, device=tensor.device,
                        dtype=tensor.real.dtype
                    ).reshape(-1)
                else:
                    relative_times = torch.as_tensor(
                        values, device=tensor.device, dtype=torch.float64
                    ).reshape(-1)
                    relative_times = (
                        relative_times - float(self.start_time)
                    )

            if nearest_sample:
                relative_times = relative_times + self.delta_t / 2.0

            fill_value = None
            keep = None
            size = relative_times.numel()
            if extrapolate is not None:
                if (_numpy.isscalar(extrapolate)
                        and _numpy.isreal(extrapolate)):
                    fill_value = extrapolate
                    facl = facr = 0
                    if interpolate == 'quadratic':
                        facl = facr = 1.1
                    elif interpolate == 'linear':
                        facl, facr = 0.1, 1.1

                    keep = (
                        (relative_times >= self.delta_t * facl)
                        & (relative_times
                           < self.duration - self.delta_t * facr)
                    )
                    relative_times = relative_times[keep]
                else:
                    raise ValueError(
                        f"Unsupported extrapolate: {extrapolate}"
                    )

            fi = relative_times * self.sample_rate
            indices = torch.floor(fi).to(dtype=torch.int64)
            offsets = (fi - indices).to(dtype=tensor.real.dtype)
            if interpolate == 'linear':
                a = tensor[indices]
                b = tensor[indices + 1]
                ans = a + (b - a) * offsets
            elif interpolate == 'quadratic':
                c = tensor[indices]
                xr = tensor[indices + 1] - c
                xl = tensor[indices - 1] - c
                a = 0.5 * (xr + xl)
                b = 0.5 * (xr - xl)
                ans = a * offsets.square() + b * offsets + c
            else:
                ans = tensor[indices]

            if fill_value is not None:
                if not tensor.is_complex() and _numpy.iscomplexobj(fill_value):
                    fill_value = fill_value.real
                old = ans
                ans = torch.full(
                    (size,), fill_value,
                    device=tensor.device, dtype=tensor.dtype
                )
                ans[keep] = old

            if scalar_input:
                return ans[0]
            return ans

        if nearest_sample:
            time = time + self.delta_t / 2.0
        vtime = _numpy.array(time, ndmin=1)

        fill_value = None
        keep_idx = None
        size = len(vtime)
        if extrapolate is not None:
            if _numpy.isscalar(extrapolate) and _numpy.isreal(extrapolate):
                fill_value = extrapolate
                facl = facr = 0
                if interpolate == 'quadratic':
                    facl = facr = 1.1
                elif interpolate == 'linear':
                    facl, facr = 0.1, 1.1

                left = (vtime >= self.start_time + self.delta_t * facl)
                right = (vtime < self.end_time - self.delta_t * facr)
                keep_idx = _numpy.where(left & right)[0]
                vtime = vtime[keep_idx]
            else:
                raise ValueError(f"Unsupported extrapolate: {extrapolate}")

        fi = (vtime - float(self.start_time)) * self.sample_rate
        i = _numpy.asarray(_numpy.floor(fi)).astype(int)
        di = fi - i

        if interpolate == 'linear':
            a = self[i]
            b = self[i+1]
            ans = a + (b - a) * di
        elif interpolate == 'quadratic':
            c = self.data[i]
            xr = self.data[i + 1] - c
            xl = self.data[i - 1] - c
            a = 0.5 * (xr + xl)
            b = 0.5 * (xr - xl)
            ans = a * di**2.0 + b * di + c
        else:
            ans = self[i]

        ans = _numpy.array(ans, ndmin=1)
        if fill_value is not None:
            old = ans
            ans = _numpy.zeros(size) + fill_value
            ans[keep_idx] = old
            ans = _numpy.array(ans, ndmin=1)

        if _numpy.ndim(time) == 0:
            return ans[0]
        return ans

    at_times = at_time

    def __eq__(self,other):
        """
        This is the Python special method invoked whenever the '=='
        comparison is used.  It will return true if the data of two
        time series are identical, and all of the numeric meta-data
        are identical, irrespective of whether or not the two
        instances live in the same memory (for that comparison, the
        Python statement 'a is b' should be used instead).

        Thus, this method returns 'True' if the types of both 'self'
        and 'other' are identical, as well as their lengths, dtypes,
        epochs, delta_ts and the data in the arrays, element by element.
        Same-device Torch arrays are reduced on their device,
        synchronizing only the final boolean. Mixed backends retain the
        CPU comparison path. Neither object is relocated nor has its
        scheme changed.

        Note in particular that this function returns a single boolean,
        and not an array of booleans as Numpy does.  If the numpy
        behavior is instead desired it can be obtained using the numpy()
        method of the PyCBC type to get a numpy instance from each
        object, and invoking '==' on those two instances.

        Parameters
        ----------
        other: another Python object, that should be tested for equality
            with 'self'.

        Returns
        -------
        boolean: 'True' if the types, dtypes, lengths, epochs, delta_ts
            and data of the two objects are each identical.
        """
        if super(TimeSeries,self).__eq__(other):
            return (self._epoch == other._epoch and self._delta_t == other._delta_t)
        else:
            return False

    def almost_equal_elem(self,other,tol,relative=True,dtol=0.0):
        """
        Compare whether two time series are almost equal, element
        by element.

        If the 'relative' parameter is 'True' (the default) then the
        'tol' parameter (which must be positive) is interpreted as a
        relative tolerance, and the comparison returns 'True' only if
        abs(self[i]-other[i]) <= tol*abs(self[i])
        for all elements of the series.

        If 'relative' is 'False', then 'tol' is an absolute tolerance,
        and the comparison is true only if
        abs(self[i]-other[i]) <= tol
        for all elements of the series.

        The method also checks that self.delta_t is within 'dtol' of
        other.delta_t; if 'dtol' has its default value of 0 then exact
        equality between the two is required.

        Other meta-data (type, dtype, length, and epoch) must be exactly
        equal. Same-device Torch arrays are reduced on their device,
        synchronizing only the final boolean. Mixed backends retain the
        CPU comparison path. Neither object is relocated nor has its
        scheme changed.

        Parameters
        ----------
        other: another Python object, that should be tested for
            almost-equality with 'self', element-by-element.
        tol: a non-negative number, the tolerance, which is interpreted
            as either a relative tolerance (the default) or an absolute
            tolerance.
        relative: A boolean, indicating whether 'tol' should be interpreted
            as a relative tolerance (if True, the default if this argument
            is omitted) or as an absolute tolerance (if tol is False).
        dtol: a non-negative number, the tolerance for delta_t. Like 'tol',
            it is interpreted as relative or absolute based on the value of
            'relative'.  This parameter defaults to zero, enforcing exact
            equality between the delta_t values of the two TimeSeries.

        Returns
        -------
        boolean: 'True' if the data and delta_ts agree within the tolerance,
            as interpreted by the 'relative' keyword, and if the types,
            lengths, dtypes, and epochs are exactly the same.
        """
        # Check that the delta_t tolerance is non-negative; raise an exception
        # if needed.
        if (dtol < 0.0):
            raise ValueError("Tolerance in delta_t cannot be negative")
        if super(TimeSeries,self).almost_equal_elem(other,tol=tol,relative=relative):
            if relative:
                return (self._epoch == other._epoch and
                        abs(self._delta_t-other._delta_t) <= dtol*self._delta_t)
            else:
                return (self._epoch == other._epoch and
                        abs(self._delta_t-other._delta_t) <= dtol)
        else:
            return False

    def almost_equal_norm(self,other,tol,relative=True,dtol=0.0):
        """
        Compare whether two time series are almost equal, normwise.

        If the 'relative' parameter is 'True' (the default) then the
        'tol' parameter (which must be positive) is interpreted as a
        relative tolerance, and the comparison returns 'True' only if
        abs(norm(self-other)) <= tol*abs(norm(self)).

        If 'relative' is 'False', then 'tol' is an absolute tolerance,
        and the comparison is true only if
        abs(norm(self-other)) <= tol

        The method also checks that self.delta_t is within 'dtol' of
        other.delta_t; if 'dtol' has its default value of 0 then exact
        equality between the two is required.

        Other meta-data (type, dtype, length, and epoch) must be exactly
        equal. Same-device Torch arrays are reduced on their device,
        synchronizing only the final boolean. Mixed backends retain the
        CPU comparison path. Neither object is relocated nor has its
        scheme changed.

        Parameters
        ----------
        other: another Python object, that should be tested for
            almost-equality with 'self', based on their norms.
        tol: a non-negative number, the tolerance, which is interpreted
            as either a relative tolerance (the default) or an absolute
            tolerance.
        relative: A boolean, indicating whether 'tol' should be interpreted
            as a relative tolerance (if True, the default if this argument
            is omitted) or as an absolute tolerance (if tol is False).
        dtol: a non-negative number, the tolerance for delta_t. Like 'tol',
            it is interpreted as relative or absolute based on the value of
            'relative'.  This parameter defaults to zero, enforcing exact
            equality between the delta_t values of the two TimeSeries.

        Returns
        -------
        boolean: 'True' if the data and delta_ts agree within the tolerance,
            as interpreted by the 'relative' keyword, and if the types,
            lengths, dtypes, and epochs are exactly the same.
        """
        # Check that the delta_t tolerance is non-negative; raise an exception
        # if needed.
        if (dtol < 0.0):
            raise ValueError("Tolerance in delta_t cannot be negative")
        if super(TimeSeries,self).almost_equal_norm(other,tol=tol,relative=relative):
            if relative:
                return (self._epoch == other._epoch and
                        abs(self._delta_t-other._delta_t) <= dtol*self._delta_t)
            else:
                return (self._epoch == other._epoch and
                        abs(self._delta_t-other._delta_t) <= dtol)
        else:
            return False

    @_convert
    def lal(self):
        """Produces a LAL time series object equivalent to self.

        Returns
        -------
        lal_data : {lal.*TimeSeries}
            LAL time series object containing the same data as self.
            The actual type depends on the sample's dtype.  If the epoch of
            self is 'None', the epoch of the returned LAL object will be
            the same as that of self.

        Raises
        ------
        TypeError
            If time series is stored in GPU memory.
        """
        lal = _lal.require_lal("TimeSeries.lal() conversion")
        lal_data = None
        ep = _lal.LIGOTimeGPS(self._epoch)

        if self._data.dtype == _numpy.float32:
            lal_data = lal.CreateREAL4TimeSeries("",ep,0,self.delta_t,lal.SecondUnit,len(self))
        elif self._data.dtype == _numpy.float64:
            lal_data = lal.CreateREAL8TimeSeries("",ep,0,self.delta_t,lal.SecondUnit,len(self))
        elif self._data.dtype == _numpy.complex64:
            lal_data = lal.CreateCOMPLEX8TimeSeries("",ep,0,self.delta_t,lal.SecondUnit,len(self))
        elif self._data.dtype == _numpy.complex128:
            lal_data = lal.CreateCOMPLEX16TimeSeries("",ep,0,self.delta_t,lal.SecondUnit,len(self))

        lal_data.data.data[:] = self.numpy()

        return lal_data

    def crop(self, left, right):
        """ Remove given seconds from either end of time series

        Parameters
        ----------
        left : float
            Number of seconds of data to remove from the left of the time series.
        right : float
            Number of seconds of data to remove from the right of the time series.

        Returns
        -------
        cropped : pycbc.types.TimeSeries
            The reduced time series
        """
        if left + right > self.duration:
            raise ValueError('Cannot crop more data than we have')

        s = int(left * self.sample_rate)
        e = len(self) - int(right * self.sample_rate)
        return self[s:e]

    def save_to_wav(self, file_name):
        """ Save this time series to a wav format audio file.

        Parameters
        ----------
        file_name : string
             The output file name
        """
        scaled = _numpy.int16(self.numpy()/max(abs(self)) * 32767)
        write_wav(file_name, int(self.sample_rate), scaled)

    def psd(self, segment_duration, **kwds):
        """ Calculate the power spectral density of this time series.

        Use the `pycbc.psd.welch` method to estimate the psd of this time segment.
        For more complete options, please see that function.

        Parameters
        ----------
        segment_duration: float
            Duration in seconds to use for each sample of the spectrum.
        kwds : keywords
            Additional keyword arguments are passed on to the `pycbc.psd.welch` method.

        Returns
        -------
        psd : FrequencySeries
            Frequency series containing the estimated PSD.
        """
        from pycbc.psd import welch
        seg_len = int(round(segment_duration * self.sample_rate))
        seg_stride = int(seg_len / 2)
        return welch(self, seg_len=seg_len,
                           seg_stride=seg_stride,
                           **kwds)
    
    # map between tapering string in sim_inspiral table or inspiral
    # code option and lalsimulation constants

    def taper_timeseries(self, location=None, tapermethod='lal',
                         return_lal=False, taper_window=None):
        """
        Taper either or both ends of a time series using wrapped
        LALSimulation functions or a constant window taper.

        Parameters
        ----------
        tsdata : TimeSeries
            Series to be tapered, dtype must be either float32 or float64
        location : string
            Should be one of ('TAPER_NONE', 'TAPER_START', 'TAPER_END',
            'TAPER_STARTEND', 'start', 'end', 'startend') - NB 'TAPER_NONE' will
            not change the series!
        tapermethod : string
            Should be one of ('lal', 'constant'). 'lal' uses the LAL tapering
            functions, 'constant' uses a constant window tapering. Default is 'lal'.
        taper_window : float
            If tapermethod is 'constant', this is the length in seconds of
            the tapering window.
        return_lal : Boolean
            If True, return a wrapped LAL time series object, else return a
            PyCBC time series.
        """
        if hasattr(location, 'decode'):
            location = location.decode()

        if hasattr(tapermethod, 'decode'):
            tapermethod = tapermethod.decode()

        taper_locations = {
            'TAPER_NONE': 'none',
            'TAPER_START': 'start',
            'start': 'start',
            'TAPER_END': 'end',
            'end': 'end',
            'TAPER_STARTEND': 'startend',
            'startend': 'startend',
        }

        tsdata = self

        if location is None:
            raise ValueError("Must specify a tapering method (function was called"
                            "with location=None)")
        if location not in taper_locations:
            raise ValueError("Unknown location %s, valid locations are %s" % \
                            (location, ", ".join(taper_locations)))
        if tsdata.dtype not in (float32, float64):
            raise TypeError("Strain dtype must be float32 or float64, not "
                        + str(tsdata.dtype))
        if tapermethod == 'lal':
            tensor = getattr(tsdata._data, 'tensor', None)
            if tensor is not None and not return_lal:
                from pycbc.types.array_torch import TorchArrayData
                tapered = _torch_lal_taper(
                    tensor, taper_locations[location]
                )
                return tsdata._return(TorchArrayData(tapered))

            import lalsimulation as sim
            taper_map = {
                'none': None,
                'start': sim.SIM_INSPIRAL_TAPER_START,
                'end': sim.SIM_INSPIRAL_TAPER_END,
                'startend': sim.SIM_INSPIRAL_TAPER_STARTEND,
            }
            taper_func_map = {
                _numpy.dtype(float32): sim.SimInspiralREAL4WaveTaper,
                _numpy.dtype(float64): sim.SimInspiralREAL8WaveTaper,
            }
            taper_func = taper_func_map[tsdata.dtype]
            # make a LAL TimeSeries to pass to the LALSim function
            ts_lal = tsdata.astype(tsdata.dtype).lal()
            taper_location = taper_map[taper_locations[location]]
            if taper_location is not None:
                taper_func(ts_lal.data, taper_location)
            if return_lal:
                return ts_lal
            else:
                return TimeSeries(ts_lal.data.data[:], delta_t=ts_lal.deltaT,
                                epoch=ts_lal.epoch)
        elif tapermethod == 'constant':
            # constant window tapering
            if taper_window is None:
                raise ValueError("If taper_method is 'constant', taper_window must be set")

            gate_params = []
            taper_location = taper_locations[location]
            if taper_location != 'none':
                tensor = getattr(tsdata._data, 'tensor', None)
                if tensor is not None:
                    parameters = _torch_constant_taper_parameters(
                        tensor.numel(), tsdata.delta_t,
                        tsdata.start_time, taper_window,
                    )
                    # The legacy all-zero autograd case returns before any
                    # in-place multiply. Even multiplying a nonleaf by ones
                    # would bump its version and invalidate saved graphs, so
                    # keep every requires-grad tensor on the exact path.
                    if parameters is not None and not tensor.requires_grad:
                        _torch_constant_taper(
                            tensor, taper_location, *parameters
                        )
                        return tsdata
                    import torch
                    nonzero = torch.nonzero(
                        tensor != 0, as_tuple=False
                    ).flatten()
                    if nonzero.numel() == 0:
                        return tsdata
                    first_nonzero = int(nonzero[0].item())
                    last_nonzero = int(nonzero[-1].item())
                else:
                    nonzero = _numpy.flatnonzero(tsdata.numpy())
                    if len(nonzero) == 0:
                        return tsdata
                    first_nonzero = int(nonzero[0])
                    last_nonzero = int(nonzero[-1])

            if taper_location in ('start', 'startend'):
                nonzero_starttime = (
                    tsdata.start_time + first_nonzero * tsdata.delta_t
                )
                gate_params.append((nonzero_starttime, 0, taper_window))
            if taper_location in ('end', 'startend'):
                nonzero_endtime = (
                    tsdata.start_time + last_nonzero * tsdata.delta_t
                )
                gate_params.append((nonzero_endtime, 0, taper_window))
            from pycbc.strain import gate_data
            return gate_data(tsdata, gate_params)
        else:
            raise ValueError("Unknown tapering method %s, valid methods are lal and constant" % \
                            (tapermethod))

    
    def get_gate_indices(self, time, window):
        """Calculates the indices at which a gate should be applied.

        Parameters
        ----------
        time: float
            Central time of the gate in seconds
        window: float
            Half-length in seconds to remove data around gate time.

        Returns
        -------
        lindex: int
            The left index of the gate
        rindex: int
            The right index of the gate
        """
        st = float(self.start_time)
        dt = float(self.delta_t)
        lindex = int((time - window - st) / dt)
        rindex = int((time + window - st) / dt)
        lindex = lindex if lindex >= 0 else 0
        rindex = rindex if rindex <= len(self) else len(self)
        return lindex, rindex

    def gate(self, time, window=0.25, method='taper', copy=True,
             taper_width=0.25, invpsd=None, paint_method='toeplitz',
             paint_invmat=None):
        """ Gate out portion of time series

        Parameters
        ----------
        time: float
            Central time of the gate in seconds
        window: float
            Half-length in seconds to remove data around gate time.
        method: str
            Method to apply gate, options are 'hard', 'taper', and 'paint'.
        copy: bool
            If False, do operations inplace to this time series, else return
            new time series.
        taper_width: float
            Length of tapering region on either side of excized data. Only
            applies to the taper gating method.
        invpsd: pycbc.types.FrequencySeries
            The inverse PSD to use for painting method. If not given,
            a PSD is generated using default settings.
        paint_method: str
            Which method to use for inpainting the gated region if
            method='paint'. If 'toeplitz', use a Toeplitz solver. If 'matmul',
            use explicit matrix inversion and multiplication.
        paint_invmat: array
            The uninverted covariance matrix to use to calculate inpainting if
            paint_method='matmul'. If None (default), calculate from given
            invpsd.

        Returns
        -------
        data: pycbc.types.TimeSeries
            Gated time series
        """
        data = self.copy() if copy else self
        if method == 'taper':
            from pycbc.strain import gate_data
            return gate_data(data, [(time, window, taper_width)])
        elif method == 'paint':
            # Uses the hole-filling method of
            # https://arxiv.org/pdf/1908.05644.pdf
            from pycbc.strain.gate import (gate_and_paint, 
                                           gate_and_paint_matmul)
            from pycbc.waveform.utils import apply_fd_time_shift
            if invpsd is None:
                # These are some bare minimum settings, normally you
                # should probably provide a psd
                invpsd = 1. / self.filter_psd(self.duration/32, self.delta_f, 0)
            lindex, rindex = self.get_gate_indices(time, window)
            rindex_time = float(self.start_time + rindex * self.delta_t)
            offset = rindex_time - (time + window)
            if offset == 0:
                if paint_method == 'toeplitz':
                    return gate_and_paint(data, lindex, rindex, invpsd, copy=False)
                elif paint_method == 'matmul':
                    return gate_and_paint_matmul(data, lindex, rindex, invpsd,
                                                 invmat=paint_invmat, copy=False)
                else:
                    raise ValueError(f'Unrecognized paint_method input {paint_method}')
            else:
                # time shift such that gate end time lands on a specific data sample
                fdata = data.to_frequencyseries()
                fdata = apply_fd_time_shift(fdata, offset + fdata.epoch, copy=False)
                # gate and paint in time domain
                data = fdata.to_timeseries()
                if paint_method == 'toeplitz':
                    data = gate_and_paint(data, lindex, rindex, invpsd, copy=False)
                elif paint_method == 'matmul':
                    data = gate_and_paint_matmul(data, lindex, rindex, invpsd,
                                                 invmat=paint_invmat, copy=False)
                else:
                    raise ValueError(f'Unrecognized paint_method input {paint_method}')
                # shift back to the original time
                fdata = data.to_frequencyseries()
                fdata = apply_fd_time_shift(fdata, -offset + fdata.epoch, copy=False)
                tdata = fdata.to_timeseries()
                return tdata
        elif method == 'hard':
            tslice = data.time_slice(time - window, time + window)
            tslice[:] = 0
            return data
        else:
            raise ValueError('Invalid method name: {}'.format(method))

    def filter_psd(self, segment_duration, delta_f, flow):
        """ Calculate the power spectral density of this time series.

        Use the `pycbc.psd.welch` method to estimate the psd of this time segment.
        The psd is then truncated in the time domain to the segment duration
        and interpolated to the requested sample frequency.

        Parameters
        ----------
        segment_duration: float
            Duration in seconds to use for each sample of the spectrum.
        delta_f : float
            Frequency spacing to return psd at.
        flow : float
            The low frequency cutoff to apply when truncating the inverse
            spectrum.

        Returns
        -------
        psd : FrequencySeries
            Frequency series containing the estimated PSD.
        """
        from pycbc.psd import interpolate, inverse_spectrum_truncation
        p = self.psd(segment_duration)
        samples = int(round(p.sample_rate * segment_duration))
        p = interpolate(p, delta_f)
        return inverse_spectrum_truncation(p, samples,
                                           low_frequency_cutoff=flow,
                                           trunc_method='hann')

    def whiten(self, segment_duration, max_filter_duration, trunc_method='hann',
                     remove_corrupted=True, low_frequency_cutoff=None,
                     return_psd=False, **kwds):
        """ Return a whitened time series

        Parameters
        ----------
        segment_duration: float
            Duration in seconds to use for each sample of the spectrum.
        max_filter_duration : int
            Maximum length of the time-domain filter in seconds.
        trunc_method : {None, 'hann'}
            Function used for truncating the time-domain filter.
            None produces a hard truncation at `max_filter_len`.
        remove_corrupted : {True, boolean}
            If True, the region of the time series corrupted by the whitening
            is excised before returning. If false, the corrupted regions
            are not excised and the full time series is returned.
        low_frequency_cutoff : {None, float}
            Low frequency cutoff to pass to the inverse spectrum truncation.
            This should be matched to a known low frequency cutoff of the
            data if there is one.
        return_psd : {False, Boolean}
            Return the estimated and conditioned PSD that was used to whiten
            the data.
        kwds : keywords
            Additional keyword arguments are passed on to the `pycbc.psd.welch` method.

        Returns
        -------
        whitened_data : TimeSeries
            The whitened time series
        """
        from pycbc.psd import inverse_spectrum_truncation, interpolate
        # Estimate the noise spectrum
        psd = self.psd(segment_duration, **kwds)
        psd = interpolate(psd, self.delta_f)
        max_filter_len = int(round(max_filter_duration * self.sample_rate))

        # Interpolate and smooth to the desired corruption length
        psd = inverse_spectrum_truncation(psd,
                   max_filter_len=max_filter_len,
                   low_frequency_cutoff=low_frequency_cutoff,
                   trunc_method=trunc_method)

        # Whiten the data by the asd
        tensor = getattr(self._data, "tensor", None)
        work_dtype = (
            _torch_whiten_work_dtype(tensor) if tensor is not None else None
        )
        if work_dtype is not None:
            # CUDA's single-precision FFT residual can exceed the conditioning
            # parity gate.  Promote only the final transform and division; the
            # estimated PSD and returned public data remain single precision.
            # The usual forward delta_t and inverse delta_f * length scaling
            # cancel, so applying the raw Torch FFT pair is equivalent while
            # avoiding two PyCBC output allocations and their copies.
            import torch
            from pycbc.types.array_torch import TorchArrayData

            spectrum = torch.fft.rfft(tensor.to(dtype=work_dtype))
            psd_tensor = getattr(getattr(psd, "_data", None), "tensor", None)
            if psd_tensor is None:
                psd_tensor = torch.as_tensor(psd.numpy(), device=tensor.device)
            spectrum.div_(
                torch.sqrt(psd_tensor.to(dtype=work_dtype))
            )
            white_tensor = torch.fft.irfft(
                spectrum, n=tensor.numel()
            ).to(dtype=tensor.dtype)
            white = TimeSeries(
                TorchArrayData(white_tensor),
                delta_t=self.delta_t,
                epoch=self.start_time,
                copy=False,
            )
        else:
            freq = self.to_frequencyseries()
            white = (freq / psd**0.5).to_timeseries()

        if remove_corrupted:
            white = white[int(max_filter_len/2):int(len(self)-max_filter_len/2)]

        if return_psd:
            return white, psd

        return white

    def qtransform(self, delta_t=None, delta_f=None, logfsteps=None,
                  frange=None, qrange=(4,64), mismatch=0.2, return_complex=False):
        """ Return the interpolated 2d qtransform of this data

        Parameters
        ----------
        delta_t : {self.delta_t, float}
            The time resolution to interpolate to
        delta_f : float, Optional
            The frequency resolution to interpolate to
        logfsteps : int
            Do a log interpolation (incompatible with delta_f option) and set
            the number of steps to take.
        frange : {(30, nyquist*0.8), tuple of ints}
            frequency range
        qrange : {(4, 64), tuple}
            q range
        mismatch : float
            Mismatch between frequency tiles
        return_complex: {False, bool}
            return the raw complex series instead of the normalized power.

        Returns
        -------
        times : numpy.ndarray
            The time that the qtransform is sampled.
        freqs : numpy.ndarray
            The frequencies that the qtransform is sampled.
        qplane : numpy.ndarray (2d)
            The two dimensional interpolated qtransform of this time series.
        """
        from pycbc.filter.qtransform import qtiling, qplane

        if logfsteps and delta_f:
            raise ValueError("Provide only one (or none) of delta_f and logfsteps")

        if frange is None:
            frange = (30, int(self.sample_rate / 2 * 8))

        use_torch = hasattr(self._data, "tensor")
        # Torch path stays on-device and avoids SciPy interpolation.
        if use_torch:
            import torch
            from pycbc.filter import qtransform_torch as _qtorch

            q_base = qtiling(self, qrange, frange, mismatch)
            fft_input = self
            if (self._data.tensor.device.type != "mps"
                    and self.dtype == _numpy.dtype(_numpy.float32)):
                # The legacy implementation performs the downstream
                # window/IFFT in double precision.  Promote before the
                # forward FFT as well so every double-capable Torch device
                # uses at least that precision and CPU/CUDA results do not
                # diverge when low-energy tiles are normalized by their
                # median.  This remains an on-device conversion.
                fft_input = self.astype(_numpy.float64)
            _, times, freqs, q_plane = _qtorch.qplane(
                q_base,
                fft_input.to_frequencyseries(),
                return_complex=return_complex,
            )

            target_times = times
            target_freqs = freqs
            device = self._data.tensor.device
            grid_dtype = (
                torch.float32 if device.type == "mps" else torch.float64
            )

            if delta_t:
                target_times = torch.arange(float(self.start_time),
                                            float(self.end_time), delta_t,
                                            device=device, dtype=grid_dtype)
            if delta_f:
                target_freqs = torch.arange(int(frange[0]), int(frange[1]),
                                            delta_f, device=device,
                                            dtype=grid_dtype)
            if logfsteps:
                if device.type == "mps":
                    # torch.logspace has no MPS kernel. Build the equivalent
                    # logarithmic grid from supported on-device operations.
                    target_freqs = torch.linspace(
                        _numpy.log(float(frange[0])),
                        _numpy.log(float(frange[1])),
                        steps=logfsteps,
                        device=device,
                        dtype=grid_dtype,
                    ).exp()
                else:
                    target_freqs = torch.logspace(
                        torch.log10(torch.tensor(
                            float(frange[0]), device=device, dtype=grid_dtype
                        )),
                        torch.log10(torch.tensor(
                            float(frange[1]), device=device, dtype=grid_dtype
                        )),
                        steps=logfsteps,
                        device=device,
                        dtype=grid_dtype,
                    )

            if delta_f or delta_t or logfsteps:
                q_plane = _qtorch.interpolate_qplane(
                    q_plane, times, freqs, target_times, target_freqs,
                    return_complex=return_complex
                )
                times, freqs = target_times, target_freqs

            return times, freqs, q_plane

        # CPU path (unchanged)
        from scipy.interpolate import RectBivariateSpline as interp2d

        q_base = qtiling(self, qrange, frange, mismatch)
        _, times, freqs, q_plane = qplane(
            q_base, self.to_frequencyseries(), return_complex=return_complex
        )

        if delta_f or delta_t or logfsteps:
            if return_complex:
                interp_amp = interp2d(freqs, times, abs(q_plane), kx=1, ky=1)
                interp_phase = interp2d(freqs, times, _numpy.angle(q_plane),
                                        kx=1, ky=1)
            else:
                interp = interp2d(freqs, times, q_plane, kx=1, ky=1)

        if delta_t:
            times = _numpy.arange(float(self.start_time),
                                  float(self.end_time), delta_t)
        if delta_f:
            freqs = _numpy.arange(int(frange[0]), int(frange[1]), delta_f)
        if logfsteps:
            freqs = _numpy.logspace(_numpy.log10(frange[0]),
                                    _numpy.log10(frange[1]),
                                    logfsteps)

        if delta_f or delta_t or logfsteps:
            if return_complex:
                q_plane = _numpy.exp(1.0j * interp_phase(freqs, times))
                q_plane *= interp_amp(freqs, times)
            else:
                q_plane = interp(freqs, times)

        return times, freqs, q_plane

    def notch_fir(self, f1, f2, order, beta=5.0, remove_corrupted=True):
        """ notch filter the time series using an FIR filtered generated from
        the ideal response passed through a time-domain kaiser
        window (beta = 5.0)

        The suppression of the notch filter is related to the bandwidth and
        the number of samples in the filter length. For a few Hz bandwidth,
        a length corresponding to a few seconds is typically
        required to create significant suppression in the notched band.

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
        beta: float
            Beta parameter of the kaiser window that sets the side lobe attenuation.
        """
        from pycbc.filter import notch_fir
        ts = notch_fir(self, f1, f2, order, beta=beta)
        if remove_corrupted:
            ts = ts[order:len(ts)-order]
        return ts

    def lowpass_fir(self, frequency, order, beta=5.0, remove_corrupted=True):
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
        remove_corrupted : {True, boolean}
            If True, the region of the time series corrupted by the filtering
            is excised before returning. If false, the corrupted regions
            are not excised and the full time series is returned.
        """
        from pycbc.filter import lowpass_fir
        ts = lowpass_fir(self, frequency, order, beta=beta)
        if remove_corrupted:
            ts = ts[order:len(ts)-order]
        return ts

    def highpass_fir(self, frequency, order, beta=5.0, remove_corrupted=True):
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
        remove_corrupted : {True, boolean}
            If True, the region of the time series corrupted by the filtering
            is excised before returning. If false, the corrupted regions
            are not excised and the full time series is returned.
        """
        from pycbc.filter import highpass_fir
        ts = highpass_fir(self, frequency, order, beta=beta)
        if remove_corrupted:
            ts = ts[order:len(ts)-order]
        return ts

    def fir_zero_filter(self, coeff):
        """Filter the timeseries with a set of FIR coefficients

        Parameters
        ----------
        coeff: numpy.ndarray
            FIR coefficients. Should be and odd length and symmetric.

        Returns
        -------
        filtered_series: pycbc.types.TimeSeries
            Return the filtered timeseries, which has been properly shifted to account
        for the FIR filter delay and the corrupted regions zeroed out.
        """
        from pycbc.filter import fir_zero_filter
        return self._return(fir_zero_filter(coeff, self))

    def resample(self, delta_t):
        """ Resample this time series to the new delta_t

        Parameters
        -----------
        delta_t: float
            The time step to resample the times series to.

        Returns
        -------
        resampled_ts: pycbc.types.TimeSeries
            The resample timeseries at the new time interval delta_t.
        """
        from pycbc.filter import resample_to_delta_t
        return resample_to_delta_t(self, delta_t)

    def save(self, path, group = None):
        """
        Save time series to a Numpy .npy, hdf, or text file. The first column
        contains the sample times, the second contains the values.
        In the case of a complex time series saved as text, the imaginary
        part is written as a third column. When using hdf format, the data is stored
        as a single vector, along with relevant attributes.

        Parameters
        ----------
        path: string
            Destination file path. Must end with either .hdf, .npy or .txt.

        group: string
            Additional name for internal storage use. Ex. hdf storage uses
            this as the key value.

        Raises
        ------
        ValueError
            If path does not end in .npy or .txt.
        """

        ext = _os.path.splitext(path)[1]
        if ext == '.npy':
            output = _numpy.vstack((self.sample_times.numpy(), self.numpy())).T
            _numpy.save(path, output)
        elif ext == '.txt':
            if self.kind == 'real':
                output = _numpy.vstack((self.sample_times.numpy(),
                                        self.numpy())).T
            elif self.kind == 'complex':
                output = _numpy.vstack((self.sample_times.numpy(),
                                        self.numpy().real,
                                        self.numpy().imag)).T
            _numpy.savetxt(path, output)
        elif ext =='.hdf':
            key = 'data' if group is None else group
            with h5py.File(path, 'a') as f:
                ds = f.create_dataset(key, data=self.numpy(),
                                      compression='gzip',
                                      compression_opts=9, shuffle=True)
                ds.attrs['start_time'] = float(self.start_time)
                ds.attrs['delta_t'] = float(self.delta_t)
        else:
            raise ValueError('Path must end with .npy, .txt or .hdf')

    def to_timeseries(self):
        """ Return time series"""
        return self

    @_nocomplex
    def to_frequencyseries(self, delta_f=None):
        """ Return the Fourier transform of this time series

        Parameters
        ----------
        delta_f : {None, float}, optional
            The frequency resolution of the returned frequency series. By
        default the resolution is determined by the duration of the timeseries.

        Returns
        -------
        FrequencySeries:
            The fourier transform of this time series.
        """
        from pycbc.fft import fft
        if not delta_f:
            delta_f = 1.0 / self.duration

        # add 0.5 to round integer
        tlen  = int(1.0 / delta_f / self.delta_t + 0.5)
        flen = int(tlen / 2 + 1)

        if tlen < len(self):
            raise ValueError("The value of delta_f (%s) would be "
                             "undersampled. Maximum delta_f "
                             "is %s." % (delta_f, 1.0 / self.duration))
        if not delta_f:
            tmp = self
        else:
            tmp = TimeSeries(zeros(tlen, dtype=self.dtype),
                             delta_t=self.delta_t, epoch=self.start_time)
            tmp[:len(self)] = self[:]

        f = FrequencySeries(zeros(flen,
                           dtype=complex_same_precision_as(self)),
                           delta_f=delta_f)
        fft(tmp, f)
        f._delta_f = delta_f
        return f

    def inject(self, other, copy=True):
        """Return copy of self with other injected into it.

        The other vector will be resized and time shifted with sub-sample
        precision before adding. This assumes that one can assume zeros
        outside of the original vector range.
        """
        # only handle equal sample rate for now.
        if not self.sample_rate_close(other):
            raise ValueError('Sample rate must be the same')
        # determine if we want to inject in place or not
        if copy:
            ts = self.copy()
        else:
            ts = self
        # Other is disjoint
        if ((other.start_time >= ts.end_time) or
           (ts.start_time > other.end_time)):
            return ts

        other = other.copy()
        dt = float((other.start_time - ts.start_time) * ts.sample_rate)

        # This coaligns other to the time stepping of self
        if not dt.is_integer():
            diff = (dt - _numpy.floor(dt)) * ts.delta_t

            # insert zeros at end
            other.resize(len(other) + (len(other) + 1) % 2 + 1)

            # fd shift to the right
            other = other.cyclic_time_shift(diff)

        # get indices of other with respect to self
        # this is already an integer to floating point precission
        left = float(other.start_time - ts.start_time) * ts.sample_rate
        left = int(round(left))
        right = left + len(other)

        oleft = 0
        oright = len(other)

        # other overhangs on left so truncate
        if left < 0:
            oleft = -left
            left = 0

        # other overhangs on right so truncate
        if right > len(ts):
            oright = len(other) - (right - len(ts))
            right = len(ts)

        ts[left:right] += other[oleft:oright]
        return ts

    add_into = inject  # maintain backwards compatibility for now

    @_nocomplex
    def cyclic_time_shift(self, dt):
        """Shift the data and timestamps by a given number of seconds

        Shift the data and timestamps in the time domain a given number of
        seconds. To just change the time stamps, do ts.start_time += dt.
        The time shift may be smaller than the intrinsic sample rate of the data.
        Note that data will be cyclically rotated, so if you shift by 2
        seconds, the final 2 seconds of your data will now be at the
        beginning of the data set.

        Parameters
        ----------
        dt : float
            Amount of time to shift the vector.

        Returns
        -------
        data : pycbc.types.TimeSeries
            The time shifted time series.
        """
        # We do this in the frequency domain to allow us to do sub-sample
        # time shifts. This also results in the shift being circular. It
        # is left to a future update to do a faster impelementation in the case
        # where the time shift can be done with an exact number of samples.
        return self.to_frequencyseries().cyclic_time_shift(dt).to_timeseries()

    def match(self, other, psd=None,
              low_frequency_cutoff=None, high_frequency_cutoff=None):
        """ Return the match between the two TimeSeries or FrequencySeries.

        Return the match between two waveforms. This is equivalent to the overlap
        maximized over time and phase. By default, the other vector will be
        resized to match self. This may remove high frequency content or the
        end of the vector.

        Parameters
        ----------
        other : TimeSeries or FrequencySeries
            The input vector containing a waveform.
        psd : Frequency Series
            A power spectral density to weight the overlap.
        low_frequency_cutoff : {None, float}, optional
            The frequency to begin the match.
        high_frequency_cutoff : {None, float}, optional
            The frequency to stop the match.

        Returns
        -------
        match: float
        index: int
            The number of samples to shift to get the match.
        """
        return self.to_frequencyseries().match(other, psd=psd,
                     low_frequency_cutoff=low_frequency_cutoff,
                     high_frequency_cutoff=high_frequency_cutoff)

    def detrend(self, type='linear'):
        """ Remove linear trend from the data

        Remove a linear trend from the data to improve the approximation that
        the data is circularly convolved, this helps reduce the size of filter
        transients from a circular convolution / filter.

        Parameters
        ----------
        type: str
            The choice of detrending. The default ('linear') removes a linear
            least squares fit. 'constant' removes only the mean of the data.
        """
        if hasattr(self._data, 'tensor'):
            import torch
            from pycbc.types.array_torch import TorchArrayData

            tensor = self._data.tensor
            if type in ('constant', 'c'):
                result = tensor - tensor.mean()
            elif type in ('linear', 'l'):
                if len(tensor) == 1:
                    result = tensor - tensor.mean()
                else:
                    n = len(tensor)
                    positions = torch.arange(
                        n, dtype=tensor.real.dtype,
                        device=tensor.device
                    )
                    positions -= (n - 1) / 2
                    norm_sq = (n * (n * n - 1)) / 12.0
                    slope = torch.sum(positions * tensor) / norm_sq
                    result = tensor - (
                        tensor.mean() + slope * positions
                    )
            else:
                raise ValueError(
                    "Trend type must be 'linear' or 'constant'."
                )

            return self._return(TorchArrayData(result))

        from scipy.signal import detrend
        return self._return(detrend(self.numpy(), type=type))

    def plot(self, **kwds):
        """ Basic plot of this time series
        """
        from matplotlib import pyplot

        if self.kind == 'real':
            plot = pyplot.plot(self.sample_times, self, **kwds)
            return plot
        elif self.kind == 'complex':
            plot1 = pyplot.plot(self.sample_times, self.real(), **kwds)
            plot2 = pyplot.plot(self.sample_times, self.imag(), **kwds)
            return plot1, plot2

    def bool_to_segmentlist(self):
        """
        Convert a boolean pycbc TimeSeries (this must be bool or integer) to
        an igwn_segments.segmentlist of (start, end) in GPS seconds.
        """

        # Is the data truthlike?
        # bools or numbers are OK, but we require finite values
        arr = self.numpy()
        if arr.dtype.kind not in ['b', 'i']:
            raise TypeError(
                'To use bool_to_segmentlist, we require that the timeseries '
                'is boolean or integer'
            )

        segs = segmentlist([])

        if arr.size == 0:
            return segs.coalesce()

        # Convert to bool
        b = arr.astype(bool)

        # Pad with a leading/trailing False to detect edges at boundaries.
        b = _numpy.concatenate(([False], b, [False]))

        # Work out the transitions between true and false.
        # starts = False to True transitions
        starts = _numpy.flatnonzero((~b[:-1]) & b[1:])
        # ends = True to False Transitions
        ends = _numpy.flatnonzero(b[:-1] & (~b[1:])) 

        # Convert indices to GPS times
        starts_time = self.start_time + starts * self.delta_t
        ends_time = self.start_time + ends * self.delta_t

        # Convert to segments
        for s, e in zip(starts_time, ends_time):
            segs.append(segment(s, e))

        return segs.coalesce()

def load_timeseries(path, group=None):
    """Load a TimeSeries from an HDF5, ASCII or Numpy file. The file type is
    inferred from the file extension, which must be `.hdf`, `.txt` or `.npy`.

    For ASCII and Numpy files, the first column of the array is assumed to
    contain the sample times. If the array has two columns, a real-valued time
    series is returned. If the array has three columns, the second and third
    ones are assumed to contain the real and imaginary parts of a complex time
    series.

    For HDF files, the dataset is assumed to contain the attributes `delta_t`
    and `start_time`, which should contain respectively the sampling period in
    seconds and the start GPS time of the data.

    The default data types will be double precision floating point.

    Parameters
    ----------
    path: string
        Input file path. Must end with either `.npy`, `.txt` or `.hdf`.

    group: string
        Additional name for internal storage use. When reading HDF files, this
        is the path to the HDF dataset to read.

    Raises
    ------
    ValueError
        If path does not end in a supported extension.
        For Numpy and ASCII input files, this is also raised if the array
        does not have 2 or 3 dimensions.
    """
    ext = _os.path.splitext(path)[1]
    if ext == '.npy':
        data = _numpy.load(path)
    elif ext == '.txt':
        data = _numpy.loadtxt(path)
    elif ext == '.hdf':
        key = 'data' if group is None else group
        with h5py.File(path, 'r') as f:
            data = f[key][:]
            series = TimeSeries(data, delta_t=f[key].attrs['delta_t'],
                                epoch=f[key].attrs['start_time'])
        return series
    else:
        raise ValueError('Path must end with .npy, .hdf, or .txt')

    delta_t = (data[-1][0] - data[0][0]) / (len(data) - 1)
    epoch = _lal.LIGOTimeGPS(data[0][0])
    if data.ndim == 2:
        return TimeSeries(data[:,1], delta_t=delta_t, epoch=epoch)
    elif data.ndim == 3:
        return TimeSeries(data[:,1] + 1j*data[:,2],
                          delta_t=delta_t, epoch=epoch)

    raise ValueError('File has %s dimensions, cannot convert to TimeSeries, \
                      must be 2 (real) or 3 (complex)' % data.ndim)
