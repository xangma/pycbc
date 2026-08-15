# Copyright (C) 2016 Collin Capano
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
""" Functions for applying gates to data.
"""

from scipy import linalg
from . import strain
import pycbc
try:
    import torch
    _HAVE_TORCH = pycbc.HAVE_TORCH
except Exception:  # pragma: no cover
    torch = None
    _HAVE_TORCH = False


def _torch_solve_toeplitz(c, b):
    """Solve a Hermitian Toeplitz system without leaving a Torch device.

    This is a vectorized Torch adaptation of the public-domain Levinson
    recursion used by :func:`scipy.linalg.solve_toeplitz`. It retains the
    O(n**2) time and O(n) memory behavior needed for practical paint gates.
    """
    if c.ndim != 1 or b.ndim != 1 or c.shape[0] != b.shape[0]:
        raise ValueError("Incompatible dimensions.")
    if c.device != b.device:
        raise ValueError("Toeplitz inputs must be on the same device.")
    if not (c.is_floating_point() or c.is_complex()):
        raise TypeError("Toeplitz coefficients must be floating point.")
    if not (b.is_floating_point() or b.is_complex()):
        raise TypeError("Toeplitz right-hand side must be floating point.")
    if not torch.is_nonzero(torch.isfinite(c).all()):
        raise ValueError("Toeplitz coefficients contain non-finite values.")
    if not torch.is_nonzero(torch.isfinite(b).all()):
        raise ValueError("Toeplitz right-hand side contains non-finite values.")

    n = b.shape[0]
    if n == 0:
        return torch.empty_like(b)

    # SciPy performs the recursion in double precision. MPS only supports
    # float32 here, although paint gating itself currently needs complex FFTs
    # and is therefore unavailable on that backend.
    is_complex = c.is_complex() or b.is_complex()
    if c.device.type == "mps":
        dtype = torch.complex64 if is_complex else torch.float32
    else:
        dtype = torch.complex128 if is_complex else torch.float64
    c = c.to(dtype=dtype)
    b = b.to(dtype=dtype)

    # scipy.linalg.solve_toeplitz(c, b) assumes the first row is conj(c).
    a = torch.cat((c[1:].conj().flip(0), c))
    x = torch.zeros_like(b)
    g = torch.zeros_like(b)
    h = torch.zeros_like(b)

    singular = a[n - 1] == 0
    diagonal = torch.where(
        singular, torch.ones_like(a[n - 1]), a[n - 1]
    )
    x[0] = b[0] / diagonal
    if n == 1:
        if torch.is_nonzero(singular):
            raise linalg.LinAlgError("Singular principal minor")
        return x

    g[0] = a[n - 2] / diagonal
    h[0] = a[n] / diagonal
    for m in range(1, n):
        upper = a[n:n + m]
        x_num = -b[m] + torch.sum(upper.flip(0) * x[:m])
        x_den = -a[n - 1] + torch.sum(upper * g[:m])
        x_den_zero = x_den == 0
        singular = singular | x_den_zero
        safe_x_den = torch.where(
            x_den_zero, torch.ones_like(x_den), x_den
        )
        x[m] = x_num / safe_x_den
        x[:m] = x[:m] - x[m] * g[:m].flip(0)
        if m == n - 1:
            break

        lower = a[n - m - 1:n - 1]
        g_num = -a[n - m - 2] + torch.sum(lower * g[:m])
        h_num = -a[n + m] + torch.sum(upper.flip(0) * h[:m])
        g_den = -a[n - 1] + torch.sum(lower * h[:m].flip(0))
        g_den_zero = g_den == 0
        singular = singular | g_den_zero
        safe_g_den = torch.where(
            g_den_zero, torch.ones_like(g_den), g_den
        )
        g[m] = g_num / safe_g_den
        h[m] = h_num / safe_x_den

        # Both reflected updates must use the workspace values from before
        # either assignment.
        old_g = g[:m].clone()
        old_h = h[:m].clone()
        g[:m] = old_g - g[m] * old_h.flip(0)
        h[:m] = old_h - h[m] * old_g.flip(0)

    if torch.is_nonzero(singular):
        raise linalg.LinAlgError("Singular principal minor")
    return x


def _gates_from_cli(opts, gate_opt):
    """Parses the given `gate_opt` into something understandable by
    `strain.gate_data`.
    """
    gates = {}
    if getattr(opts, gate_opt) is None:
        return gates
    for gate in getattr(opts, gate_opt):
        try:
            ifo, central_time, half_dur, taper_dur = gate.split(':')
            central_time = float(central_time)
            half_dur = float(half_dur)
            taper_dur = float(taper_dur)
        except ValueError:
            raise ValueError("--gate {} not formatted correctly; ".format(
                gate) + "see help")
        try:
            gates[ifo].append((central_time, half_dur, taper_dur))
        except KeyError:
            gates[ifo] = [(central_time, half_dur, taper_dur)]
    return gates


def gates_from_cli(opts):
    """Parses the --gate option into something understandable by
    `strain.gate_data`.
    """
    return _gates_from_cli(opts, 'gate')


def psd_gates_from_cli(opts):
    """Parses the --psd-gate option into something understandable by
    `strain.gate_data`.
    """
    return _gates_from_cli(opts, 'psd_gate')


def apply_gates_to_td(strain_dict, gates):
    """Applies the given dictionary of gates to the given dictionary of
    strain.

    Parameters
    ----------
    strain_dict : dict
        Dictionary of time-domain strain, keyed by the ifos.
    gates : dict
        Dictionary of gates. Keys should be the ifo to apply the data to,
        values are a tuple giving the central time of the gate, the half
        duration, and the taper duration.

    Returns
    -------
    dict
        Dictionary of time-domain strain with the gates applied.
    """
    # copy data to new dictionary
    outdict = dict(strain_dict.items())
    for ifo in gates:
        outdict[ifo] = strain.gate_data(outdict[ifo], gates[ifo])
    return outdict


def apply_gates_to_fd(stilde_dict, gates):
    """Applies the given dictionary of gates to the given dictionary of
    strain in the frequency domain.

    Gates are applied by IFFT-ing the strain data to the time domain, applying
    the gate, then FFT-ing back to the frequency domain.

    Parameters
    ----------
    stilde_dict : dict
        Dictionary of frequency-domain strain, keyed by the ifos.
    gates : dict
        Dictionary of gates. Keys should be the ifo to apply the data to,
        values are a tuple giving the central time of the gate, the half
        duration, and the taper duration.

    Returns
    -------
    dict
        Dictionary of frequency-domain strain with the gates applied.
    """
    # copy data to new dictionary
    outdict = dict(stilde_dict.items())
    # create a time-domin strain dictionary to apply the gates to
    strain_dict = dict([[ifo, outdict[ifo].to_timeseries()] for ifo in gates])
    # apply gates and fft back to the frequency domain
    for ifo,d in apply_gates_to_td(strain_dict, gates).items():
        outdict[ifo] = d.to_frequencyseries()
    return outdict


def add_gate_option_group(parser):
    """Adds the options needed to apply gates to data.

    Parameters
    ----------
    parser : object
        ArgumentParser instance.
    """
    gate_group = parser.add_argument_group("Options for gating data")

    gate_group.add_argument("--gate", nargs="+", type=str,
                            metavar="IFO:CENTRALTIME:HALFDUR:TAPERDUR",
                            help="Apply one or more gates to the data before "
                                 "filtering.")
    gate_group.add_argument("--gate-overwhitened", action="store_true",
                            help="Overwhiten data first, then apply the "
                                 "gates specified in --gate. Overwhitening "
                                 "allows for sharper tapers to be used, "
                                 "since lines are not blurred.")
    gate_group.add_argument("--psd-gate", nargs="+", type=str,
                            metavar="IFO:CENTRALTIME:HALFDUR:TAPERDUR",
                            help="Apply one or more gates to the data used "
                                 "for computing the PSD. Gates are applied "
                                 "prior to FFT-ing the data for PSD "
                                 "estimation.")
    return gate_group


def gate_and_paint(data, lindex, rindex, invpsd, copy=True):
    """Gates and in-paints data.

    Parameters
    ----------
    data : TimeSeries
        The data to gate.
    lindex : int
        The start index of the gate.
    rindex : int
        The end index of the gate.
    invpsd : FrequencySeries
        The inverse of the PSD.
    copy : bool, optional
        Copy the data before applying the gate. Otherwise, the gate will
        be applied in-place. Default is True.

    Returns
    -------
    TimeSeries :
        The gated and in-painted time series.
    """
    # Uses the hole-filling method of
    # https://arxiv.org/pdf/1908.05644.pdf
    # Copy the data and zero inside the hole
    if copy:
        data = data.copy()
    data[lindex:rindex] = 0
    # get the over-whitened gated data
    # If torch-backed, stay on device for intermediate steps
    use_torch = _HAVE_TORCH and hasattr(invpsd, "_data") and hasattr(invpsd._data, "tensor")
    tdfilter = invpsd.astype('complex').to_timeseries() * invpsd.delta_t
    owhgated_data = (data.to_frequencyseries() * invpsd).to_timeseries()

    # remove the projection into the null space
    if use_torch:
        hole_length = rindex - lindex
        proj = _torch_solve_toeplitz(
            tdfilter._data.tensor[:hole_length],
            owhgated_data._data.tensor[lindex:rindex],
        ).to(dtype=data._data.tensor.dtype)
        data._data.tensor[lindex:rindex].sub_(proj)
    else:
        proj = linalg.solve_toeplitz(tdfilter[:(rindex - lindex)],
                                     owhgated_data[lindex:rindex])
        data[lindex:rindex] -= proj
    return data
