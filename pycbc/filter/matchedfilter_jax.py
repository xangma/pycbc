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

"""JAX backend for matched filtering primitives, correlation,
sigmasq, and match calculations.
"""

import numpy as np
import jax
import jax.numpy as jnp

from pycbc.filter.matchedfilter import _BaseCorrelator
from pycbc.types.array_jax import (
    JAXArrayData,
    _ensure_x64,
    to_jax,
)


def _set_output_array(z, val):
    """Store result val into output container z."""
    if isinstance(z, JAXArrayData):
        z.set_array(val)
    elif hasattr(z, "_data") and isinstance(z._data, JAXArrayData):
        z._data.set_array(val)
    elif hasattr(z, "data") and isinstance(z.data, JAXArrayData):
        z.data.set_array(val)
    elif hasattr(z, "_data"):
        try:
            z._data[:] = np.asarray(val)
        except Exception:
            z._data = val
    elif hasattr(z, "data"):
        try:
            z.data[:] = np.asarray(val)
        except Exception:
            z.data = val
    else:
        try:
            z[:] = np.asarray(val)
        except Exception:
            pass


def correlate(x, y, z):
    """Elementwise z = conj(x) * y in JAX scheme."""
    _ensure_x64()
    x_arr = to_jax(x)
    y_arr = to_jax(y)
    prod = jnp.conj(x_arr) * y_arr
    _set_output_array(z, prod)


class JAXCorrelator(_BaseCorrelator):
    """JAX correlator engine for persistent/repeated correlation."""

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def correlate(self):
        """Execute elementwise correlation."""
        correlate(self.x, self.y, self.z)


def _correlate_factory(x, y, z):
    """Factory returning the JAX correlator class."""
    return JAXCorrelator


def batch_correlate_execute(self, y):
    """Vectorized batch correlation for BatchCorrelator in JAX scheme."""
    _ensure_x64()
    y_arr = to_jax(y)
    for x, z in zip(self.xs, self.zs):
        correlate(x, y_arr, z)


# ----------------------------------------------------------------------
# Pure Functional JAX API (JIT-compilable & Autodiff-compatible)
# ----------------------------------------------------------------------


@jax.jit
def correlate_jax(x, y):
    """Pure JAX elementwise conjugate multiplication: conj(x) * y."""
    return jnp.conj(x) * y


def sigmasq_jax(
    htilde,
    psd=None,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
    delta_f=1.0,
):
    """Pure JAX calculation of template loudness (sigma)^2."""
    _ensure_x64()
    h_arr = to_jax(htilde)
    n_pts = (len(h_arr) - 1) * 2
    flow = low_frequency_cutoff
    fhigh = high_frequency_cutoff

    kmin = int(flow / delta_f) if flow else 1
    if kmin < 0:
        raise ValueError("flow cannot be negative")

    if fhigh:
        kmax = min(int(fhigh / delta_f), int((n_pts + 1) / 2.0))
    else:
        kmax = int((n_pts + 1) / 2.0)

    if kmax <= kmin:
        raise ValueError(f"kmax ({kmax}) must be greater than kmin ({kmin})")

    ht = h_arr[kmin:kmax]
    mag_sq = jnp.abs(ht) ** 2

    if psd is not None:
        psd_arr = to_jax(psd)
        mag_sq = mag_sq / psd_arr[kmin:kmax]

    norm = 4.0 * delta_f
    return jnp.sum(mag_sq) * norm


def sigmasq_series_jax(
    htilde,
    psd=None,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
    delta_f=1.0,
):
    """Pure JAX calculation of cumulative power frequency series."""
    _ensure_x64()
    h_arr = to_jax(htilde)
    n_pts = (len(h_arr) - 1) * 2
    flow = low_frequency_cutoff
    fhigh = high_frequency_cutoff

    kmin = int(flow / delta_f) if flow else 1
    if fhigh:
        kmax = min(int(fhigh / delta_f), int((n_pts + 1) / 2.0))
    else:
        kmax = int((n_pts + 1) / 2.0)

    mag_sq = jnp.abs(h_arr) ** 2
    if psd is not None:
        psd_arr = to_jax(psd)
        mag_sq = mag_sq / psd_arr

    sub = jnp.cumsum(mag_sq[kmin:kmax])
    norm = 4.0 * delta_f

    series = jnp.zeros(len(h_arr), dtype=h_arr.real.dtype)
    series = series.at[kmin:kmax].set(sub * norm)
    return series


def matched_filter_core_jax(
    template,
    data,
    psd=None,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
    h_norm=None,
    delta_f=None,
    delta_t=None,
):
    """Pure JAX matched filter core returning (snr, correlation, norm)."""
    _ensure_x64()
    from pycbc.types import TimeSeries
    from pycbc.filter import make_frequency_series

    if isinstance(template, TimeSeries):
        template = make_frequency_series(template)
    if isinstance(data, TimeSeries):
        data = make_frequency_series(data)

    htilde = to_jax(template)
    stilde = to_jax(data)

    if len(htilde) != len(stilde):
        raise ValueError("Length of template and data must match")

    if delta_f is None:
        delta_f = getattr(data, "delta_f", 1.0)
    if delta_t is None:
        delta_t = getattr(
            data, "delta_t", 1.0 / (2.0 * (len(stilde) - 1) * delta_f)
        )

    n_time = (len(stilde) - 1) * 2
    flow = low_frequency_cutoff
    fhigh = high_frequency_cutoff

    kmin = int(flow / delta_f) if flow else 1
    if fhigh:
        kmax = min(int(fhigh / delta_f), int((n_time + 1) / 2.0))
    else:
        kmax = int((n_time + 1) / 2.0)

    qtilde = jnp.zeros(n_time, dtype=stilde.dtype)
    corr_slice = jnp.conj(htilde[kmin:kmax]) * stilde[kmin:kmax]

    if psd is not None:
        psd_arr = to_jax(psd)
        corr_slice = corr_slice / psd_arr[kmin:kmax]

    qtilde = qtilde.at[kmin:kmax].set(corr_slice)

    # Complex-to-complex IFFT of length n_time
    snr_time = jnp.fft.ifft(qtilde) * n_time

    if h_norm is None:
        h_norm = sigmasq_jax(
            htilde,
            psd=psd,
            low_frequency_cutoff=low_frequency_cutoff,
            high_frequency_cutoff=high_frequency_cutoff,
            delta_f=delta_f,
        )

    norm = (4.0 * delta_f) / jnp.sqrt(h_norm)
    return snr_time, qtilde, norm


def matched_filter_jax(
    template,
    data,
    psd=None,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
    sigmasq=None,
    delta_f=None,
    delta_t=None,
):
    """Pure JAX matched filter returning normalized complex/real SNR."""
    snr_time, _, norm = matched_filter_core_jax(
        template,
        data,
        psd=psd,
        low_frequency_cutoff=low_frequency_cutoff,
        high_frequency_cutoff=high_frequency_cutoff,
        h_norm=sigmasq,
        delta_f=delta_f,
        delta_t=delta_t,
    )
    return snr_time * norm


def overlap_jax(
    vec1,
    vec2,
    psd=None,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
    delta_f=1.0,
    normalized=True,
):
    """Calculate overlap between two frequency series in JAX."""
    from pycbc.types import TimeSeries
    from pycbc.filter import make_frequency_series

    if isinstance(vec1, TimeSeries):
        vec1 = make_frequency_series(vec1)
    if isinstance(vec2, TimeSeries):
        vec2 = make_frequency_series(vec2)

    v1 = to_jax(vec1)
    v2 = to_jax(vec2)
    n_pts = (len(v1) - 1) * 2
    flow = low_frequency_cutoff
    fhigh = high_frequency_cutoff

    kmin = int(flow / delta_f) if flow else 1
    if fhigh:
        kmax = min(int(fhigh / delta_f), int((n_pts + 1) / 2.0))
    else:
        kmax = int((n_pts + 1) / 2.0)

    v1_sub = v1[kmin:kmax]
    v2_sub = v2[kmin:kmax]
    prod = jnp.conj(v1_sub) * v2_sub

    if psd is not None:
        psd_arr = to_jax(psd)
        prod = prod / psd_arr[kmin:kmax]

    inn = jnp.sum(prod).real * (4.0 * delta_f)
    if not normalized:
        return inn

    norm1 = sigmasq_jax(
        v1,
        psd=psd,
        low_frequency_cutoff=flow,
        high_frequency_cutoff=fhigh,
        delta_f=delta_f,
    )
    norm2 = sigmasq_jax(
        v2,
        psd=psd,
        low_frequency_cutoff=flow,
        high_frequency_cutoff=fhigh,
        delta_f=delta_f,
    )
    return inn / jnp.sqrt(norm1 * norm2)


def match_jax(
    vec1,
    vec2,
    psd=None,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
    delta_f=1.0,
    v1_norm=None,
    v2_norm=None,
):
    """Calculate match (overlap maximized over time and phase) in JAX."""
    from pycbc.types import TimeSeries
    from pycbc.filter import make_frequency_series

    if isinstance(vec1, TimeSeries):
        vec1 = make_frequency_series(vec1)
    if isinstance(vec2, TimeSeries):
        vec2 = make_frequency_series(vec2)

    v1 = to_jax(vec1)
    v2 = to_jax(vec2)
    n_pts = (len(v1) - 1) * 2
    flow = low_frequency_cutoff
    fhigh = high_frequency_cutoff

    kmin = int(flow / delta_f) if flow else 1
    if fhigh:
        kmax = min(int(fhigh / delta_f), int((n_pts + 1) / 2.0))
    else:
        kmax = int((n_pts + 1) / 2.0)

    corr = jnp.zeros(n_pts, dtype=v1.dtype)
    prod = jnp.conj(v1[kmin:kmax]) * v2[kmin:kmax]
    if psd is not None:
        psd_arr = to_jax(psd)
        prod = prod / psd_arr[kmin:kmax]

    corr = corr.at[kmin:kmax].set(prod)
    time_series = jnp.fft.ifft(corr) * n_pts

    mag = jnp.abs(time_series)
    max_idx = int(jnp.argmax(mag))
    max_val = mag[max_idx] * (4.0 * delta_f)

    if v1_norm is None:
        v1_norm = sigmasq_jax(
            v1,
            psd=psd,
            low_frequency_cutoff=flow,
            high_frequency_cutoff=fhigh,
            delta_f=delta_f,
        )
    if v2_norm is None:
        v2_norm = sigmasq_jax(
            v2,
            psd=psd,
            low_frequency_cutoff=flow,
            high_frequency_cutoff=fhigh,
            delta_f=delta_f,
        )

    norm = jnp.sqrt(v1_norm * v2_norm)
    return max_val / norm, max_idx
