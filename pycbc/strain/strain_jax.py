# Copyright (C) 2026 The PyCBC Collaboration
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

"""JAX backend for strain conditioning and autogating routines in PyCBC."""

import functools
import numpy as np
import jax
import jax.numpy as jnp

import pycbc.events
import pycbc.psd
import pycbc.types
from pycbc.filter.resample import resample_to_delta_t
from pycbc.psd.estimate_jax import (
    _welch_core,
    _interp_core,
    _inv_trunc_core,
    median_bias,
)
from pycbc.strain.strain import next_power_of_2
from pycbc.types.array_jax import _ensure_x64, to_jax


def _zero_pad_on_device(strain, strain_pad_length, pad_start, pad_end):
    """Copy a strain series into a zero-padded JAX array.

    The padding is deliberately performed after conversion to JAX so that a
    GPU path does not first materialize the padded buffer in host memory.
    """
    strain_jax = to_jax(strain)
    padded = jnp.zeros(strain_pad_length, dtype=strain_jax.dtype)
    return padded.at[pad_start:pad_end].set(strain_jax)


@functools.partial(
    jax.jit,
    static_argnames=(
        "strain_pad_length",
        "pad_start",
        "pad_end",
        "corrupt_length",
        "kmin",
        "kmax",
    ),
)
def _whiten_pad_core(
    s_arr,
    psd_arr,
    norm,
    strain_pad_length,
    pad_start,
    pad_end,
    corrupt_length,
    kmin,
    kmax,
):
    safe_psd = jnp.where(psd_arr > 0, psd_arr, 1.0)
    freq_indices = jnp.arange(len(psd_arr))
    inv_asd = jnp.where(
        (freq_indices < kmin) | (freq_indices >= kmax),
        0.0,
        (safe_psd * norm) ** (-0.5),
    )
    s_tilde = jnp.fft.rfft(s_arr)
    white_tilde = s_tilde * inv_asd
    white_pad = jnp.fft.irfft(white_tilde, n=strain_pad_length)
    mag = jnp.abs(white_pad[pad_start:pad_end])
    mag = mag.at[:corrupt_length].set(0.0)
    mag = mag.at[-corrupt_length:].set(0.0)
    return mag


@functools.partial(
    jax.jit,
    static_argnames=(
        "seg_len",
        "seg_stride",
        "num_segments",
        "avg_method",
        "n_freq",
        "n_time",
        "kmin",
        "trunc_start",
        "trunc_end",
        "which_spectrum",
        "use_hann",
        "strain_pad_length",
        "pad_start",
        "pad_end",
        "corrupt_length",
        "kmax",
    ),
)
def _fused_autogate_pipeline_core(
    s_trim,
    s_arr,
    window_arr,
    tw_trunc,
    delta_t,
    old_df,
    new_df,
    norm,
    fill_val,
    seg_len,
    seg_stride,
    num_segments,
    avg_method,
    n_freq,
    n_time,
    kmin,
    trunc_start,
    trunc_end,
    which_spectrum,
    use_hann,
    strain_pad_length,
    pad_start,
    pad_end,
    corrupt_length,
    kmax,
):
    raw_psd = _welch_core(
        s_trim,
        window_arr,
        delta_t,
        seg_len,
        seg_stride,
        num_segments,
        avg_method,
    )
    if avg_method == "median":
        raw_psd = raw_psd / median_bias(num_segments)
    norm_w = 2.0 * old_df * seg_len / jnp.sum(jnp.square(window_arr))
    psd = raw_psd * norm_w

    psd_interp = _interp_core(psd, old_df, new_df, n_freq)
    psd_trunc = _inv_trunc_core(
        psd_interp,
        tw_trunc,
        fill_val,
        n_freq,
        n_time,
        kmin,
        trunc_start,
        trunc_end,
        which_spectrum,
        use_hann,
    )
    mag = _whiten_pad_core(
        s_arr,
        psd_trunc,
        norm,
        strain_pad_length,
        pad_start,
        pad_end,
        corrupt_length,
        kmin,
        kmax,
    )
    return mag


def detect_loud_glitches_jax(
    strain,
    psd_duration=4.0,
    psd_stride=2.0,
    psd_avg_method="median",
    low_freq_cutoff=30.0,
    threshold=50.0,
    cluster_window=5.0,
    corrupt_time=4.0,
    high_freq_cutoff=None,
    output_intermediates=False,
):
    """Automatic identification of loud transients for gating purposes in JAX.

    Performs conditioning, zero-padding, Welch PSD estimation, whitening,
    and FindChirp peak clustering on the active JAXScheme accelerator in
    double precision (float64/complex128). Operates on an isolated data copy
    to ensure bit-exact scientific parity with reference implementations.
    """
    _ensure_x64()
    if high_freq_cutoff:
        s = resample_to_delta_t(strain, 0.5 / high_freq_cutoff, method="ldas")
    else:
        s = strain.copy()
        if s.dtype != np.float64:
            s = s.astype(np.float64)

    # Taper strain ends
    corrupt_length = int(corrupt_time * s.sample_rate)
    w = np.arange(corrupt_length) / float(corrupt_length)
    s[0:corrupt_length] *= pycbc.types.Array(w, dtype=s.dtype)
    s[(len(s) - corrupt_length):] *= pycbc.types.Array(w[::-1], dtype=s.dtype)

    if output_intermediates:
        s.save_to_wav("strain_conditioned.wav")

    # Zero-pad strain to a power-of-2 length
    strain_pad_length = next_power_of_2(len(s))
    pad_start = int(strain_pad_length / 2 - len(s) / 2)
    pad_end = pad_start + len(s)
    s_arr = _zero_pad_on_device(s, strain_pad_length, pad_start, pad_end)

    # Prepare segmentation parameters for Welch PSD
    s_trim = s[corrupt_length:(len(s) - corrupt_length)]
    seg_len = int(psd_duration * s.sample_rate)
    seg_stride = int(psd_stride * s.sample_rate)
    num_samples = len(s_trim)
    num_segments = int(num_samples // seg_stride)
    if (num_segments - 1) * seg_stride + seg_len > num_samples:
        num_segments -= 1
    data_len = (num_segments - 1) * seg_stride + seg_len
    if data_len < num_samples:
        diff = num_samples - data_len
        start = diff // 2
        if diff % 2:
            start = start + 1
        end = num_samples - diff // 2
        s_trim = s_trim[start:end]

    window_arr = jnp.asarray(np.hanning(seg_len), dtype=jnp.float64)
    max_filter_len = int(psd_duration * s.sample_rate)
    tw_trunc = jnp.asarray(np.hanning(max_filter_len), dtype=jnp.float64)
    delta_t = float(s.delta_t)
    old_df = 1.0 / (delta_t * seg_len)
    new_df = 1.0 / (strain_pad_length * delta_t)
    n_freq = strain_pad_length // 2 + 1
    n_time = (n_freq - 1) * 2
    kmin = int(low_freq_cutoff / new_df)
    if high_freq_cutoff:
        kmax = int(high_freq_cutoff / new_df)
        norm = high_freq_cutoff - low_freq_cutoff
    else:
        kmax = n_freq
        norm = s.sample_rate / 2.0 - low_freq_cutoff

    trunc_start = max_filter_len // 2
    trunc_end = n_time - max_filter_len // 2

    mag_jax = _fused_autogate_pipeline_core(
        to_jax(s_trim),
        s_arr,
        window_arr,
        tw_trunc,
        delta_t,
        old_df,
        new_df,
        norm,
        0.0,
        seg_len,
        seg_stride,
        num_segments,
        psd_avg_method,
        n_freq,
        n_time,
        kmin,
        trunc_start,
        trunc_end,
        "invasd",
        True,
        strain_pad_length,
        pad_start,
        pad_end,
        corrupt_length,
        kmax,
    )
    mag = np.asarray(mag_jax)

    indices = np.where(mag > threshold)[0]
    if len(indices) == 0:
        return [], []

    cluster_window_samples = int(cluster_window * s.sample_rate)
    cluster_idx = pycbc.events.findchirp_cluster_over_window(
        indices.astype(np.int32),
        np.ascontiguousarray(mag[indices], dtype=np.float64),
        cluster_window_samples,
    )
    peak_indices = indices[cluster_idx]
    times = [
        float(s.start_time + idx * s.delta_t)
        for idx in peak_indices
    ]
    snrs = [float(mag[idx]) for idx in peak_indices]
    return times, snrs
