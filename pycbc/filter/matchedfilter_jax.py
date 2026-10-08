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

Functional helpers support JIT with the default JAX kernels. Optional CPU
validation routes perform host work and run outside JAX transformations.
"""

import functools
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp

from pycbc.filter.matchedfilter import _BaseCorrelator, get_cutoff_indices
from pycbc.types.array_jax import (
    JAXArrayData,
    _as_jax_array,
    _ensure_x64,
    _cpu_reference,
    _divide,
    _fast_inner,
    _fast_inner_self,
    _fast_weighted_inner,
    _fast_weighted_inner_self,
    _reference_enabled,
    to_jax,
)


_EMPTY_F32 = np.zeros(0, dtype=np.float32)
_EMPTY_U32 = np.zeros(0, dtype=np.uint32)


def _functional_device(*inputs):
    """Use the active scheme, an existing input device, or JAX's default."""
    from pycbc import scheme

    if isinstance(scheme.mgr.state, scheme.JAXScheme):
        return scheme.mgr.state.jax_device
    arrays = [_as_jax_array(value) for value in inputs]
    # Traced computations inherit their execution device from JAX rather than
    # committing intermediate arrays to the device available during tracing.
    if any(isinstance(value, jax.core.Tracer) for value in arrays):
        return None
    for value in arrays:
        if value is not None:
            return value.device
    return jax.config.jax_default_device or jax.devices()[0]


def _functional_array(value, device):
    """Place eager inputs together while leaving JAX tracers untouched."""
    array = _as_jax_array(value)
    return (array if isinstance(array, jax.core.Tracer)
            else to_jax(value, device=device))


def _set_output_array(z, val):
    """Store results, preserving output wrappers and exposing write errors."""
    data = z if isinstance(z, JAXArrayData) else getattr(z, "_data", None)
    if isinstance(data, JAXArrayData):
        data.set_array(val)
    elif data is not None:
        data[:] = np.asarray(val)
    else:
        z[:] = np.asarray(val)


def _cpu_correlate(x, y):
    """Execute original compiled correlation on explicit host arrays."""
    from pycbc.filter import matchedfilter_cpu

    # Cython's original writable memoryviews cannot accept read-only JAX views.
    x = np.asarray(x).copy()
    y = np.asarray(y).copy()
    z = np.empty_like(x)
    matchedfilter_cpu.correlate(
        SimpleNamespace(data=x), SimpleNamespace(data=y),
        SimpleNamespace(data=z))
    return to_jax(z)


def _inner_product(x, y, weight=None):
    """Share array kernels and their independent original-CPU selectors."""
    operation = "inner" if weight is None else "weighted_inner"
    if _reference_enabled(operation):
        args = (y,) if weight is None else (y, weight)
        return to_jax(_cpu_reference(x, operation, *args))
    if weight is None:
        return _fast_inner_self(x) if x is y else _fast_inner(x, y)
    return (_fast_weighted_inner_self(x, weight) if x is y
            else _fast_weighted_inner(x, y, weight))


def _inverse_transform(correlation):
    """Use the original CPU IFFT only when its validation route is selected."""
    if not _reference_enabled("ifft"):
        return jnp.fft.ifft(correlation) * len(correlation)
    from pycbc.fft import ifft
    from pycbc.types import Array

    source = Array(JAXArrayData(correlation), copy=False)
    target = Array(JAXArrayData(jnp.zeros_like(correlation)), copy=False)
    ifft(source, target)
    return to_jax(target)


@jax.jit
def correlate_jax(x, y):
    """Pure JAX elementwise conjugate multiplication: conj(x) * y."""
    return jnp.conj(x) * y


def correlate(x, y, z):
    """Elementwise z = conj(x) * y in JAX scheme."""
    _ensure_x64()
    x_arr = to_jax(x)
    y_arr = to_jax(y)
    prod = (_cpu_correlate(x_arr, y_arr) if _reference_enabled("correlate")
            else correlate_jax(x_arr, y_arr))
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


@functools.partial(jax.jit, static_argnums=(4,))
def _batch_correlate_update(templates, y, parent, base, row_stride):
    """Correlate and publish one contiguous sibling-view block."""
    products = correlate_jax(templates, y)
    block_len = templates.shape[0] * row_stride
    block = jax.lax.dynamic_slice(parent, (base,), (block_len,))
    block = block.reshape(templates.shape[0], row_stride)
    block = block.at[:, :templates.shape[1]].set(products)
    return jax.lax.dynamic_update_slice(parent, block.reshape(-1), (base,))


def batch_correlate_execute(self, y):
    """Vectorized batch correlation for BatchCorrelator in JAX scheme."""
    _ensure_x64()
    size = self.size
    y_arr = to_jax(y)[:size]
    if _reference_enabled("correlate"):
        for x, z in zip(self.xs, self.zs):
            products = _cpu_correlate(to_jax(x)[:size], y_arr)
            _set_output_array(z[:size], products)
        return

    # Lazy bank templates retain the complete 2-D device tensor.  When this
    # batch is an aligned contiguous run of rows, use it directly.  Reading
    # each LazyFrequencySeries first would materialize a separate static row
    # slice before stack, which is particularly costly during CUDA startup.
    templates = getattr(self, "_jax_template_matrix", None)
    if templates is not None:
        templates = templates[:, :size]

    if templates is None and self.xs:
        batch_tensor = getattr(self.xs[0], "_batch_tensor", None)
        positions = []
        if batch_tensor is not None:
            for x in self.xs:
                if getattr(x, "_batch_tensor", None) is not batch_tensor:
                    batch_tensor = None
                    break
                pos = getattr(x, "_batch_pos", None)
                if pos is None:
                    batch_tensor = None
                    break
                positions.append(int(pos))
        if batch_tensor is not None and positions:
            start = positions[0]
            aligned = positions == list(range(start, start + len(positions)))
            shape = getattr(batch_tensor, "shape", ())
            if aligned and len(shape) == 2 and start >= 0:
                stop = start + len(positions)
                if stop <= shape[0] and size <= shape[1]:
                    templates = to_jax(batch_tensor)[start:stop, :size]

    if templates is None:
        templates = jnp.stack([to_jax(x)[:size] for x in self.xs], axis=0)
    # LiveBatchMatchedFilter passes sibling views into one contiguous output
    # allocation.  Updating each view separately makes JAX rebuild the whole
    # immutable parent for every template.  Fold the updates into one JAX
    # expression and publish the parent once, preserving any untouched tails.
    zdata = []
    for z in self.zs:
        data = z if isinstance(z, JAXArrayData) else getattr(z, "_data", None)
        if not isinstance(data, JAXArrayData):
            zdata = []
            break
        info = data.slice_info
        parent = data.parent
        if (parent is None or not isinstance(info, slice)
                or info.step not in (None, 1)):
            zdata = []
            break
        start = 0 if info.start is None else info.start
        stop = parent.shape[0] if info.stop is None else info.stop
        zdata.append((data, parent, start, stop))

    contiguous = False
    if zdata and all(item[1] is zdata[0][1] for item in zdata):
        parent = zdata[0][1]
        row_stride = zdata[0][3] - zdata[0][2]
        base = zdata[0][2]
        contiguous = (
            row_stride >= size and
            all(stop - start == row_stride for _, _, start, stop in zdata) and
            all(start == base + row * row_stride
                for row, (_, _, start, _) in enumerate(zdata)) and
            base + len(zdata) * row_stride <= parent.shape[0]
        )
        if contiguous:
            updated = _batch_correlate_update(
                templates, y_arr, parent.array, base, row_stride,
            )
            parent.set_array(updated)
            return

    # General BatchCorrelator users may provide unrelated output arrays.
    # Retain the existing per-output behavior for those cases.
    products = correlate_jax(templates, y_arr)
    for row, z in enumerate(self.zs):
        _set_output_array(z[:size], products[row])


# ----------------------------------------------------------------------
# Pure Functional JAX API (JIT-compilable & Autodiff-compatible)
# ----------------------------------------------------------------------


def sigmasq_jax(
    htilde,
    psd=None,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
    delta_f=1.0,
):
    """Pure JAX calculation of template loudness (sigma)^2."""
    _ensure_x64()
    device = _functional_device(htilde, psd)
    h_arr = _functional_array(htilde, device)
    n_pts = (len(h_arr) - 1) * 2
    flow = low_frequency_cutoff
    fhigh = high_frequency_cutoff

    kmin, kmax = get_cutoff_indices(flow, fhigh, delta_f, n_pts)

    ht = h_arr[kmin:kmax]
    weight = (None if psd is None
              else _functional_array(psd, device)[kmin:kmax])
    return _inner_product(ht, ht, weight).real * (4.0 * delta_f)


def sigmasq_series_jax(
    htilde,
    psd=None,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
    delta_f=1.0,
):
    """Pure JAX calculation of cumulative power frequency series."""
    _ensure_x64()
    device = _functional_device(htilde, psd)
    h_arr = _functional_array(htilde, device)
    n_pts = (len(h_arr) - 1) * 2
    flow = low_frequency_cutoff
    fhigh = high_frequency_cutoff

    kmin, kmax = get_cutoff_indices(flow, fhigh, delta_f, n_pts)

    mag_sq = (to_jax(_cpu_reference(h_arr, "squared_norm"))
              if _reference_enabled("squared_norm")
              else h_arr.real ** 2 + h_arr.imag ** 2)
    if psd is not None:
        psd_arr = _functional_array(psd, device)
        mag_sq = _divide(mag_sq, psd_arr, inplace=True)

    sub = mag_sq[kmin:kmax]
    sub = (to_jax(_cpu_reference(sub, "cumsum"))
           if _reference_enabled("cumsum") else jnp.cumsum(sub))
    norm = 4.0 * delta_f

    series = jnp.zeros(len(h_arr), dtype=h_arr.real.dtype, device=device)
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

    device = _functional_device(template, data, psd, h_norm)
    if isinstance(template, TimeSeries):
        template = make_frequency_series(template)
    if isinstance(data, TimeSeries):
        data = make_frequency_series(data)

    htilde = _functional_array(template, device)
    stilde = _functional_array(data, device)
    psd_arr = None if psd is None else _functional_array(psd, device)

    if len(htilde) != len(stilde):
        raise ValueError("Length of template and data must match")

    if delta_f is None:
        delta_f = getattr(data, "delta_f", 1.0)
    n_time = (len(stilde) - 1) * 2
    flow = low_frequency_cutoff
    fhigh = high_frequency_cutoff

    kmin, kmax = get_cutoff_indices(flow, fhigh, delta_f, n_time)

    qtilde = jnp.zeros(n_time, dtype=stilde.dtype, device=device)
    corr_slice = (_cpu_correlate(htilde[kmin:kmax], stilde[kmin:kmax])
                  if _reference_enabled("correlate")
                  else correlate_jax(htilde[kmin:kmax], stilde[kmin:kmax]))

    if psd_arr is not None:
        corr_slice = _divide(corr_slice, psd_arr[kmin:kmax], inplace=True)

    qtilde = qtilde.at[kmin:kmax].set(corr_slice)

    # Complex-to-complex IFFT of length n_time
    snr_time = _inverse_transform(qtilde)

    if h_norm is None:
        h_norm = sigmasq_jax(
            htilde,
            psd=psd_arr,
            low_frequency_cutoff=low_frequency_cutoff,
            high_frequency_cutoff=high_frequency_cutoff,
            delta_f=delta_f,
        )
    else:
        h_norm = _functional_array(h_norm, device)

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

    device = _functional_device(vec1, vec2, psd)
    if isinstance(vec1, TimeSeries):
        vec1 = make_frequency_series(vec1)
    if isinstance(vec2, TimeSeries):
        vec2 = make_frequency_series(vec2)

    v1 = _functional_array(vec1, device)
    v2 = _functional_array(vec2, device)
    psd_arr = None if psd is None else _functional_array(psd, device)
    n_pts = (len(v1) - 1) * 2
    flow = low_frequency_cutoff
    fhigh = high_frequency_cutoff

    kmin, kmax = get_cutoff_indices(flow, fhigh, delta_f, n_pts)

    v1_sub = v1[kmin:kmax]
    v2_sub = v2[kmin:kmax]
    weight = None if psd_arr is None else psd_arr[kmin:kmax]
    inn = _inner_product(v1_sub, v2_sub, weight).real * (4.0 * delta_f)
    if not normalized:
        return inn

    norm1 = sigmasq_jax(
        v1,
        psd=psd_arr,
        low_frequency_cutoff=flow,
        high_frequency_cutoff=fhigh,
        delta_f=delta_f,
    )
    norm2 = sigmasq_jax(
        v2,
        psd=psd_arr,
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

    device = _functional_device(vec1, vec2, psd, v1_norm, v2_norm)
    if isinstance(vec1, TimeSeries):
        vec1 = make_frequency_series(vec1)
    if isinstance(vec2, TimeSeries):
        vec2 = make_frequency_series(vec2)

    v1 = _functional_array(vec1, device)
    v2 = _functional_array(vec2, device)
    psd_arr = None if psd is None else _functional_array(psd, device)
    n_pts = (len(v1) - 1) * 2
    flow = low_frequency_cutoff
    fhigh = high_frequency_cutoff

    kmin, kmax = get_cutoff_indices(flow, fhigh, delta_f, n_pts)

    corr = jnp.zeros(n_pts, dtype=v1.dtype, device=device)
    prod = (_cpu_correlate(v1[kmin:kmax], v2[kmin:kmax])
            if _reference_enabled("correlate")
            else correlate_jax(v1[kmin:kmax], v2[kmin:kmax]))
    if psd_arr is not None:
        prod = _divide(prod, psd_arr[kmin:kmax], inplace=True)

    corr = corr.at[kmin:kmax].set(prod)
    time_series = _inverse_transform(corr)

    if _reference_enabled("abs_max_loc"):
        max_value, max_idx = _cpu_reference(time_series, "abs_max_loc")
    else:
        magnitude = time_series.real ** 2 + time_series.imag ** 2
        max_idx = int(jnp.argmax(magnitude))
        max_value = jnp.sqrt(magnitude[max_idx])

    if v1_norm is None:
        v1_norm = sigmasq_jax(
            v1,
            psd=psd_arr,
            low_frequency_cutoff=flow,
            high_frequency_cutoff=fhigh,
            delta_f=delta_f,
        )
    else:
        v1_norm = _functional_array(v1_norm, device)
    if v2_norm is None:
        v2_norm = sigmasq_jax(
            v2,
            psd=psd_arr,
            low_frequency_cutoff=flow,
            high_frequency_cutoff=fhigh,
            delta_f=delta_f,
        )
    else:
        v2_norm = _functional_array(v2_norm, device)

    # NumPy scalar arithmetic in the original match retains peak precision.
    peak = _functional_array(max_value, device)
    snr_norm = ((4.0 * delta_f) / jnp.sqrt(v1_norm)).astype(peak.dtype)
    second_norm = jnp.sqrt(v2_norm).astype(peak.dtype)
    return _divide(peak * snr_norm, second_norm), max_idx
