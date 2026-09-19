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

import functools
import numpy as np
import jax
import jax.numpy as jnp

from pycbc.filter.matchedfilter import _BaseCorrelator
from pycbc.types.array_jax import (
    JAXArrayData,
    _ensure_x64,
    is_jax_array,
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


@jax.jit
def _fast_conj_mul(x, y):
    return jnp.conj(x) * y


def correlate(x, y, z):
    """Elementwise z = conj(x) * y in JAX scheme."""
    _ensure_x64()
    x_arr = to_jax(x)
    y_arr = to_jax(y)
    prod = _fast_conj_mul(x_arr, y_arr)
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


def batch_peak_values(output, template_count, template_size, segment):
    """Materialize one peak index and value per template from contiguous output.

    Avoids per-template host synchronization by reducing the full 2D batch
    allocation in a single vectorized JAX kernel.
    """
    _ensure_x64()
    import jax.numpy as jnp

    tensor = to_jax(output)
    template_count = int(template_count)
    template_size = int(template_size)
    if (
        template_count < 1
        or template_size < 1
        or tensor.size != template_count * template_size
    ):
        return None

    if segment.step not in (None, 1):
        return None

    values = tensor.reshape(template_count, template_size)[:, segment]
    if values.shape[1] == 0:
        return None

    if jnp.iscomplexobj(values):
        sq_mag = values.real ** 2 + values.imag ** 2
    else:
        sq_mag = values ** 2

    indices = jnp.argmax(sq_mag, axis=-1)
    peaks = values[jnp.arange(values.shape[0]), indices]
    return (
        np.asarray(indices),
        np.asarray(peaks),
    )


def batch_peak_magnitudes(peak_values):
    """Materialize batch peak magnitudes in JAX."""
    _ensure_x64()
    import jax.numpy as jnp

    jarr = to_jax(peak_values)
    return np.asarray(jnp.abs(jarr))


def batch_matched_filter_bank(
    templates,
    strain,
    psd=None,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
    delta_f=None,
):
    """High-throughput batched template bank matched filtering in pure JAX.

    Performs 2D overwhitening, batched correlation, and batched IFFT across
    the entire template bank in an XLA-fused execution graph.

    Parameters
    ----------
    templates : jax.Array or FrequencySeries
        2D array of templates with shape (num_templates, n_freq) or list of
        FrequencySeries.
    strain : jax.Array or FrequencySeries
        1D strain frequency series of length n_freq.
    psd : jax.Array or FrequencySeries, optional
        1D PSD frequency series of length n_freq.
    low_frequency_cutoff : float, optional
        Low frequency cutoff for integration in Hz.
    high_frequency_cutoff : float, optional
        High frequency cutoff for integration in Hz.
    delta_f : float, optional
        Frequency spacing in Hz.

    Returns
    -------
    norm_snr : jax.Array
        2D array of normalized complex SNR time series of shape
        (num_templates, n_time).
    sigmasq : jax.Array
        1D array of template variances (sigmasq) of length num_templates.
    """
    _ensure_x64()
    import jax.numpy as jnp

    # Unwrap templates to 2D jax array
    if (
        hasattr(templates, "__len__")
        and not is_jax_array(templates)
        and not isinstance(templates, np.ndarray)
    ):
        t_arrs = [to_jax(t) for t in templates]
        templates_j = jnp.stack(t_arrs, axis=0)
        df = (
            templates[0].delta_f
            if delta_f is None and hasattr(templates[0], "delta_f")
            else delta_f
        )
    templates_j = to_jax(templates)
    if templates_j.ndim == 1:
        templates_j = templates_j[None, :]
    df = delta_f if delta_f is not None else 1.0

    num_templates = templates_j.shape[0]
    strain_j = to_jax(strain)
    n_freq = strain_j.shape[-1]
    n_time = (n_freq - 1) * 2

    # Bandpass mask
    kmin, kmax = 0, n_freq
    if low_frequency_cutoff is not None and df is not None:
        kmin = int(round(low_frequency_cutoff / df))
    if high_frequency_cutoff is not None and df is not None:
        kmax = min(int(round(high_frequency_cutoff / df)), n_freq)

    # 2D overwhitening
    if psd is not None:
        psd_j = to_jax(psd)
        inv_psd = jnp.where(psd_j > 0, 1.0 / psd_j, 0.0)
    else:
        inv_psd = jnp.ones(n_freq, dtype=strain_j.dtype)

    # Apply frequency cutoffs to inv_psd
    freq_mask = (jnp.arange(n_freq) >= kmin) & (jnp.arange(n_freq) < kmax)
    inv_psd_masked = jnp.where(freq_mask, inv_psd, 0.0)

    # Template normalizations: sigmasq = 4 * df * sum(|h|^2 * inv_psd)
    t_mag_sq = templates_j.real ** 2 + templates_j.imag ** 2
    sigmasq = 4.0 * df * jnp.sum(t_mag_sq * inv_psd_masked[None, :], axis=-1)

    # Batched correlation:
    # qtilde has length n_time = (n_freq - 1) * 2
    # In PyCBC matched_filter_core:
    # correlate(htilde[kmin:kmax], stilde[kmin:kmax], qtilde[kmin:kmax])
    # which is qtilde[kmin:kmax] = conj(htilde[kmin:kmax]) * stilde[kmin:kmax]
    # then qtilde[kmin:kmax] /= psd[kmin:kmax]
    qtilde = jnp.zeros((num_templates, n_time), dtype=templates_j.dtype)
    corr = (
        jnp.conj(templates_j[:, kmin:kmax])
        * (strain_j[kmin:kmax] * inv_psd[kmin:kmax])[None, :]
    )
    qtilde = qtilde.at[:, kmin:kmax].set(corr)

    # Batched IFFT: n_time length
    snr_series = jnp.fft.ifft(qtilde, axis=-1) * n_time

    # Normalize by (4 * df) / sqrt(sigmasq)
    norm = (4.0 * df) / jnp.sqrt(jnp.maximum(sigmasq, 1e-30))[:, None]
    norm_snr = snr_series * norm

    return norm_snr, sigmasq


@functools.partial(
    jax.jit,
    static_argnames=(
        "kmin", "kmax", "tlen", "valid_start", "valid_stop", "full_templates"
    ),
)
def _batched_filter_and_screen(
    templates_2d,
    seg_slice,
    kmin,
    kmax,
    tlen,
    valid_start,
    valid_stop,
    full_templates=False,
):
    """JIT-compiled batched correlation, IFFT, and peak screening."""
    if full_templates:
        templates_2d = templates_2d[:, kmin:kmax]
    corr_slice = jnp.conj(templates_2d) * seg_slice[None, :]
    pad_left = kmin
    pad_right = tlen - kmax
    qtilde = jnp.pad(corr_slice, ((0, 0), (pad_left, pad_right)))
    snr_series = jnp.fft.ifft(qtilde, axis=-1) * tlen
    valid_snr = snr_series[:, valid_start:valid_stop]
    valid_mag_sq = valid_snr.real ** 2 + valid_snr.imag ** 2
    row_max_sq = jnp.max(valid_mag_sq, axis=-1)
    return snr_series, valid_snr, corr_slice, row_max_sq


@functools.partial(
    jax.jit,
    static_argnames=(
        "kmin", "kmax", "tlen", "valid_start", "valid_stop", "full_templates"
    ),
)
def _batched_filter_and_screen_lean(
    templates_2d,
    seg_slice,
    kmin,
    kmax,
    tlen,
    valid_start,
    valid_stop,
    full_templates=False,
):
    """JIT-compiled batched correlation, IFFT, and peak screening without retaining full time series."""
    if full_templates:
        templates_2d = templates_2d[:, kmin:kmax]
    corr_slice = jnp.conj(templates_2d) * seg_slice[None, :]
    pad_left = kmin
    pad_right = tlen - kmax
    qtilde = jnp.pad(corr_slice, ((0, 0), (pad_left, pad_right)))
    snr_series = jnp.fft.ifft(qtilde, axis=-1) * tlen
    valid_snr = snr_series[:, valid_start:valid_stop]
    valid_mag_sq = valid_snr.real ** 2 + valid_snr.imag ** 2
    row_max_sq = jnp.max(valid_mag_sq, axis=-1)
    return valid_snr, row_max_sq, corr_slice


def batched_matched_filter_and_cluster_jax(
    mf_control,
    segnum,
    templates,
    sigmasqs,
    window,
    epoch=None,
):
    """Batched matched filtering, thresholding, and clustering on JAX device.

    Parameters
    ----------
    mf_control : MatchedFilterControl
        The matched filter control object holding analysis configuration.
    segnum : int
        Index of the segment to filter against.
    templates : list of FrequencySeries
        Templates in the current batch.
    sigmasqs : sequence of float
        Normalization factors for each template in the batch.
    window : int
        Clustering window size in samples.
    epoch : optional
        GPS epoch for the returned TimeSeries.

    Returns
    -------
    list of tuples
        For each template, returns (snr, norm, corr, idx, snrv) matching
        MatchedFilterControl's contract.
    """
    _ensure_x64()
    import jax.numpy as jnp
    from pycbc.types import Array, TimeSeries
    from pycbc.types.array_jax import JAXArrayData, to_jax
    from pycbc.events.threshold_jax import _batched_cluster_core

    b = len(sigmasqs)
    if b == 0:
        return []

    seg = mf_control.segments[segnum]
    kmin, kmax = mf_control.kmin, mf_control.kmax
    tlen = mf_control.tlen
    delta_f = mf_control.delta_f
    delta_t = mf_control.delta_t
    valid_start = seg.analyze.start
    valid_stop = seg.analyze.stop
    threshold = float(mf_control.snr_threshold)

    from pycbc import scheme
    state = getattr(scheme.mgr, "state", None)
    target_dev = getattr(state, "jax_device", None)

    # Pre-stack all segments into a persistent 2D tensor in VRAM during setup / first call
    cached_seg_tensor = getattr(mf_control, "_jax_segments_tensor", None)
    if (
        cached_seg_tensor is None
        or (target_dev is not None and getattr(cached_seg_tensor, "device", lambda: None)() != target_dev)
    ):
        import jax
        slices = []
        for s in mf_control.segments:
            s_jax = to_jax(s, device=target_dev)
            slices.append(s_jax[kmin:kmax])
        mf_control._jax_segments_tensor = jnp.stack(slices, axis=0)
        if target_dev is not None:
            mf_control._jax_segments_tensor = jax.device_put(
                mf_control._jax_segments_tensor, target_dev
            )

    seg_slice = mf_control._jax_segments_tensor[segnum]

    # Cache 2D templates on mf_control across segments to avoid re-stacking 5x per batch
    batch_tensor = getattr(templates, "_batch_tensor", None)
    if batch_tensor is not None:
        cache_key = (id(batch_tensor), kmin, kmax, True)
        full_templates = True
    else:
        cache_key = (tuple(id(t) for t in templates), kmin, kmax, False)
        full_templates = False

    if getattr(mf_control, "_cached_templates_key", None) == cache_key:
        templates_2d = mf_control._cached_templates_2d
    else:
        import jax
        if batch_tensor is not None:
            # Keep the full batch tensor cached. Slicing inside the JIT avoids
            # retaining a second device allocation for the cropped templates.
            templates_2d = to_jax(batch_tensor, device=target_dev)
        elif hasattr(templates, "ndim") and templates.ndim == 2:
            templates_2d = to_jax(templates, device=target_dev)[:, kmin:kmax]
        else:
            templates_2d = jnp.stack(
                [to_jax(t, device=target_dev)[kmin:kmax] for t in templates], axis=0
            )
        if target_dev is not None:
            templates_2d = jax.device_put(templates_2d, target_dev)
        mf_control._cached_templates_key = cache_key
        mf_control._cached_templates_2d = templates_2d

    sigmasqs_arr = jnp.asarray(sigmasqs, dtype=jnp.float32)
    norms = (4.0 * delta_f) / jnp.sqrt(jnp.maximum(sigmasqs_arr, 1e-30))
    unnorm_thresh = threshold / norms

    # Execute JIT-compiled batched correlation, IFFT, and peak screening
    need_snr = getattr(mf_control, "need_snr_series", False)
    if need_snr:
        snr_series, valid_snr, corr_slice, row_max_sq = _batched_filter_and_screen(
            templates_2d,
            seg_slice,
            kmin,
            kmax,
            tlen,
            valid_start,
            valid_stop,
            full_templates=full_templates,
        )
    else:
        valid_snr, row_max_sq, corr_slice = _batched_filter_and_screen_lean(
            templates_2d,
            seg_slice,
            kmin,
            kmax,
            tlen,
            valid_start,
            valid_stop,
            full_templates=full_templates,
        )
        snr_series = None

    thresh_sq = unnorm_thresh ** 2
    has_trigs = np.asarray(row_max_sq > thresh_sq)

    empty_idx = np.empty(0, dtype=np.uint32)
    empty_snrv = np.empty(0, dtype=np.complex64)

    results = []
    if not np.any(has_trigs):
        # Transfer normalization values once instead of synchronizing per row.
        norms_np = np.asarray(norms)
        for i in range(b):
            results.append(([], float(norms_np[i]), [], empty_idx, empty_snrv))
        return results

    # Run batched reduction and clustering across all templates simultaneously on GPU
    batched_max_idx, batched_survivor_mask, batched_max_snr = _batched_cluster_core(
        valid_snr, thresh_sq, window=window
    )
    survivor_mask_np = np.asarray(batched_survivor_mask)
    global_max_idx_np = np.asarray(batched_max_idx)
    block_max_snr_np = np.asarray(batched_max_snr)
    norms_np = np.asarray(norms)

    from pycbc.waveform.bank import LazyFrequencySeries

    for i in range(b):
        norm_i = float(norms_np[i])
        if not has_trigs[i]:
            results.append(([], norm_i, [], empty_idx, empty_snrv))
            continue

        mask_np = survivor_mask_np[i]
        if not np.any(mask_np):
            results.append(([], norm_i, [], empty_idx, empty_snrv))
            continue

        survivor_indices = global_max_idx_np[i][mask_np].astype(np.uint32)
        survivor_values = block_max_snr_np[i][mask_np]

        corr = LazyFrequencySeries(corr_slice, i, delta_f)
        corr._kmin = kmin
        corr._tlen = tlen

        if need_snr and snr_series is not None:
            snr = TimeSeries(
                Array(JAXArrayData(snr_series[i]), copy=False),
                epoch=epoch,
                delta_t=delta_t,
                copy=False,
            )
        else:
            snr = None

        results.append((snr, norm_i, corr, survivor_indices, survivor_values))

    del valid_snr, row_max_sq, batched_max_idx, batched_survivor_mask, batched_max_snr
    if snr_series is not None:
        del snr_series
    if corr_slice is not None:
        del corr_slice

    return results


@functools.partial(jax.jit, static_argnames=("delta_f",))
def _batched_sigmasq_core(tmpls_stack, psd_j, delta_f):
    """JIT-compiled GPU reduction for template batch sigmasq values."""
    mag_sq = tmpls_stack.real ** 2 + tmpls_stack.imag ** 2
    return jnp.sum(mag_sq / psd_j[None, :], axis=-1) * (4.0 * delta_f)


def batch_sigmasq_jax(templates, psd):
    """Compute sigmasq for a batch of FrequencySeries templates against psd on GPU."""
    from pycbc.types.array_jax import to_jax, _ensure_x64
    from pycbc import scheme

    _ensure_x64()
    b = len(templates)
    if b == 0:
        return []
    df = float(templates[0].delta_f)
    flow = float(templates[0].f_lower) if hasattr(templates[0], "f_lower") else 30.0
    kmin = int(flow / df)

    state = getattr(scheme.mgr, "state", None)
    target_dev = getattr(state, "jax_device", None)

    batch_tensor = getattr(templates, "_batch_tensor", None)
    if batch_tensor is not None:
        tmpls_stack = batch_tensor[:, kmin:]
    else:
        tmpls_stack = jnp.stack([to_jax(t, device=target_dev)[kmin:] for t in templates], axis=0)
    psd_j = to_jax(psd, device=target_dev)[kmin:]
    ssq_gpu = _batched_sigmasq_core(tmpls_stack, psd_j, df)
    return [float(x) for x in np.asarray(ssq_gpu)]
