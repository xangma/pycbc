# Copyright (C) 2014  Christopher M. Biwer
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

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#

import numpy as np

from scipy.signal import zpk2sos, sosfilt
from pycbc.types import TimeSeries


try:
    import torch
    from pycbc.types.array_torch import TorchArrayData
except Exception:  # pragma: no cover - torch is optional
    torch = None
    TorchArrayData = None


# Bound the temporary state used by the parallel Torch recurrence.
_TORCH_SOS_TARGET_BLOCK_SIZE = 2**18


def _torch_sosfilt_section(signal, section, section_numpy, block_size):
    """Apply one second-order section with a parallel affine scan."""

    b0, b1, b2, _, a1, a2 = section
    transition = torch.stack(
        (
            torch.stack((-a1, torch.ones_like(a1))),
            torch.stack((-a2, torch.zeros_like(a2))),
        )
    )
    drive = torch.stack((b1 - a1 * b0, b2 - a2 * b0))

    # Computing the tiny transition powers in double precision avoids
    # magnifying round-off for poles close to the unit circle on devices that
    # only support single precision. The sample-sized scan remains on device.
    transition_numpy = np.array(
        [[-section_numpy[4], 1.0], [-section_numpy[5], 0.0]],
        dtype=np.result_type(section_numpy.dtype, np.float64),
    )
    powers = []
    offset = 1
    while offset < block_size:
        powers.append(
            (
                offset,
                torch.as_tensor(
                    np.ascontiguousarray(transition_numpy),
                    device=signal.device,
                    dtype=signal.dtype,
                ),
            )
        )
        transition_numpy = transition_numpy @ transition_numpy
        offset *= 2

    state = signal.new_zeros(2)
    output = []
    for start in range(0, signal.numel(), block_size):
        block = signal[start : start + block_size]
        updates = block.unsqueeze(1) * drive.unsqueeze(0)
        states = torch.cat(
            (
                updates[:1] + (transition @ state).unsqueeze(0),
                updates[1:],
            )
        )
        for offset, power in powers:
            if offset >= block.numel():
                break
            states = torch.cat(
                (
                    states[:offset],
                    states[offset:] + states[:-offset] @ power.T,
                )
            )

        previous_state = torch.cat((state[:1], states[:-1, 0]))
        output.append(b0 * block + previous_state)
        state = states[-1]

    return torch.cat(output)


def _torch_sosfilt(sos, signal):
    """Filter a one-dimensional tensor without transferring samples to CPU."""

    if signal.numel() == 0:
        return signal.clone()

    if signal.device.type != "mps":
        work_dtype = {
            torch.float32: torch.float64,
            torch.complex64: torch.complex128,
        }.get(signal.dtype, signal.dtype)
    else:
        work_dtype = signal.dtype

    filtered = signal.to(work_dtype)
    sos_numpy = np.ascontiguousarray(sos)
    sos_tensor = torch.as_tensor(
        sos_numpy, device=signal.device, dtype=work_dtype
    )
    block_size = min(filtered.numel(), _TORCH_SOS_TARGET_BLOCK_SIZE)
    for section, section_numpy in zip(sos_tensor, sos_numpy):
        filtered = _torch_sosfilt_section(
            filtered, section, section_numpy, block_size
        )
    return filtered.to(signal.dtype)


def filter_zpk(timeseries, z, p, k):
    """Return a new timeseries that was filtered with a zero-pole-gain filter.
    The transfer function in the s-domain looks like:
    .. math::
    \\frac{H(s) = (s - s_1) * (s - s_3) * ... * (s - s_n)}{(s - s_2) * (s - s_4) * ... * (s - s_m)}, m >= n

    The zeroes, and poles entered in Hz are converted to angular frequency,
    along the imaginary axis in the s-domain s=i*omega.  Then the zeroes, and
    poles are bilinearly transformed via:
    .. math::
    z(s) = \\frac{(1 + s*T/2)}{(1 - s*T/2)}

    Where z is the z-domain value, s is the s-domain value, and T is the
    sampling period.  After the poles and zeroes have been bilinearly
    transformed, the second-order sections are found and used to filter the
    data. Torch-backed inputs are filtered on their active device.

    Parameters
    ----------
    timeseries: TimeSeries
        The TimeSeries instance to be filtered.
    z: array
        Array of zeros to include in zero-pole-gain filter design.
        In units of Hz.
    p: array
        Array of poles to include in zero-pole-gain filter design.
        In units of Hz.
    k: float
        Gain to include in zero-pole-gain filter design. This gain is a
        constant multiplied to the transfer function.

    Returns
    -------
    Time Series: TimeSeries
        A  new TimeSeries that has been filtered.

    Examples
    --------
    To apply a 5 zeroes at 100Hz, 5 poles at 1Hz, and a gain of 1e-10 filter
    to a TimeSeries instance, do:
    >>> filtered_data = filter_zpk(timeseries, [100]*5, [1]*5, 1e-10)
    """

    # sanity check type
    if not isinstance(timeseries, TimeSeries):
        raise TypeError("Can only filter TimeSeries instances.")

    # sanity check causal filter
    degree = len(p) - len(z)
    if degree < 0:
        raise TypeError("May not have more zeroes than poles. \
                         Filter is not causal.")

    # cast zeroes and poles as arrays and gain as a float
    z = np.asarray(z) * (-2 * np.pi)
    p = np.asarray(p) * (-2 * np.pi)
    k = float(k)

    # get denominator of bilinear transform
    fs = 2.0 * timeseries.sample_rate

    # zeroes in the z-domain
    z_zd = (1 + z/fs) / (1 - z/fs)

    # any zeros that were at infinity are moved to the Nyquist frequency
    z_zd = z_zd[np.isfinite(z_zd)]
    z_zd = np.append(z_zd, -np.ones(degree))

    # poles in the z-domain
    p_zd = (1 + p/fs) / (1 - p/fs)

    # gain change in z-domain
    k_zd = k * np.prod(fs - z) / np.prod(fs - p)

    # get second-order sections
    sos = zpk2sos(z_zd, p_zd, k_zd)

    torch_input = TorchArrayData is not None and isinstance(
        getattr(timeseries, "_data", None), TorchArrayData
    )
    if torch_input:
        filtered_data = _torch_sosfilt(sos, timeseries._data.tensor)
        return TimeSeries(
            TorchArrayData(filtered_data),
            delta_t=timeseries.delta_t,
            epoch=timeseries._epoch,
            copy=False,
        )

    filtered_data = sosfilt(sos, timeseries.numpy())
    return TimeSeries(
        filtered_data,
        delta_t=timeseries.delta_t,
        dtype=timeseries.dtype,
        epoch=timeseries._epoch,
    )
