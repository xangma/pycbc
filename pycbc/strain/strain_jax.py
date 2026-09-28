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
import jax
import jax.numpy as jnp

import pycbc.events
import pycbc.types
from pycbc.filter.resample import resample_to_delta_t
from pycbc.filter.resample_jax import (
    _circular_fir_zero,
    firwin,
)
from pycbc.psd.estimate_jax import (
    _welch_core,
    _interp_core,
    _inv_trunc_core,
    _median_bias_numpy_compat,
)
from pycbc.strain.strain import next_power_of_2
from pycbc.types.array_jax import JAXArrayData, _ensure_x64, to_jax


@functools.partial(
    jax.jit,
    static_argnames=("factor", "corruption", "sample_step"),
)
def _condition_and_stitch_core(
    raw_buffer,
    strain_buffer,
    highpass_coefficients,
    resample_coefficients,
    dynamic_range_factor,
    factor,
    corruption,
    sample_step,
):
    """Condition one live block and stitch it into the rolling buffer."""
    conditioned_size = sample_step + 2 * corruption
    raw_size = conditioned_size * factor
    block = raw_buffer[-raw_size:]

    block = _circular_fir_zero(block, highpass_coefficients)
    scale = jnp.asarray(dynamic_range_factor, dtype=block.dtype)
    block = (block * scale).astype(jnp.float32)
    block = _circular_fir_zero(block, resample_coefficients)[::factor]
    block = block[corruption:]

    output = jnp.roll(strain_buffer, -sample_step)
    write_start = len(strain_buffer) - conditioned_size + corruption
    return output.at[write_start:].set(block)


def can_fuse_strain_buffer_jax(buffer, blocksize):
    """Return whether both live FIR stages use their circular-filter path."""
    sample_step = int(blocksize * buffer.sample_rate)
    raw_size = (sample_step + 2 * buffer.corruption) * buffer.factor
    coefficient_sizes = (
        buffer.highpass_samples * 2 + 1,
        buffer.factor * 20 + 1,
    )
    return raw_size >= 128 and all(
        raw_size < size * 10 or raw_size < 2**18
        for size in coefficient_sizes
    )


def condition_strain_buffer_jax(buffer, blocksize):
    """Condition and stitch one ``StrainBuffer`` block on the JAX device."""
    if not can_fuse_strain_buffer_jax(buffer, blocksize):
        raise ValueError("live conditioning shape does not use circular FIRs")
    _ensure_x64()
    factor = int(buffer.factor)
    corruption = int(buffer.corruption)
    sample_step = int(blocksize * buffer.sample_rate)

    raw = to_jax(buffer.raw_buffer)
    strain = to_jax(buffer.strain)
    raw_device = raw.device() if callable(raw.device) else raw.device
    cache_key = (
        str(raw.dtype),
        float(buffer.raw_buffer.delta_t),
        float(buffer.sample_rate),
        float(buffer.highpass_frequency),
        int(buffer.highpass_samples),
        float(buffer.beta),
        factor,
        str(raw_device),
    )
    cache = getattr(buffer, "_jax_conditioning_cache", None)
    if cache is None or cache[0] != cache_key:
        raw_nyquist = int(1.0 / buffer.raw_buffer.delta_t) / 2.0
        highpass_coefficients = firwin(
            buffer.highpass_samples * 2 + 1,
            buffer.highpass_frequency / raw_nyquist,
            window=("kaiser", buffer.beta),
            pass_zero=False,
        )
        resample_coefficients = firwin(
            factor * 20 + 1,
            1.0 / factor,
            window=("kaiser", 5),
        )
        cache = (
            cache_key,
            highpass_coefficients,
            resample_coefficients,
        )
        buffer._jax_conditioning_cache = cache

    _, highpass_coefficients, resample_coefficients = cache
    values = _condition_and_stitch_core(
        raw,
        strain,
        highpass_coefficients,
        resample_coefficients,
        buffer.dyn_range_fac,
        factor,
        corruption,
        sample_step,
    )
    buffer.strain._data = JAXArrayData(values)
    buffer.strain._saved.clear()
    buffer.strain.start_time += blocksize


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
    freq_indices = jnp.arange(len(psd_arr))
    inv_asd = jnp.where(
        (freq_indices < kmin) | (freq_indices >= kmax),
        0.0,
        (psd_arr * norm) ** (-0.5),
    )
    s_tilde = jnp.fft.rfft(s_arr)
    white_tilde = s_tilde * inv_asd
    white_pad = jnp.fft.irfft(white_tilde, n=strain_pad_length)
    mag = jnp.abs(white_pad[pad_start:pad_end])
    if corrupt_length:
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
        raw_psd = _median_bias_numpy_compat(raw_psd, num_segments)
    norm_w = 2.0 * old_df * seg_len / jnp.sum(jnp.square(window_arr))
    psd = raw_psd * norm_w

    psd_interp = _interp_core(psd, old_df, new_df, n_freq).astype(psd.dtype)
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
    the input precision. Operates on an isolated data copy so that gating
    does not modify its input.
    """
    _ensure_x64()
    if high_freq_cutoff:
        s = resample_to_delta_t(strain, 0.5 / high_freq_cutoff, method="ldas")
    else:
        s = strain.copy()

    # Taper strain ends
    corrupt_length = int(corrupt_time * s.sample_rate)
    w = jnp.arange(corrupt_length) / float(corrupt_length)
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

    window_arr = jnp.hanning(seg_len).astype(s.dtype)
    max_filter_len = int(psd_duration * s.sample_rate)
    tw_trunc = jnp.hanning(max_filter_len).astype(s.dtype)
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
    indices = jnp.where(mag_jax > threshold)[0]
    if len(indices) == 0:
        return jnp.empty(0, dtype=jnp.float64), jnp.empty(0, dtype=jnp.float64)

    cluster_window_samples = int(cluster_window * s.sample_rate)
    cluster_idx = pycbc.events.findchirp_cluster_over_window(
        indices.astype(jnp.int32),
        mag_jax[indices],
        cluster_window_samples,
    )
    peak_indices = indices[cluster_idx]
    times = float(s.start_time) + peak_indices * s.delta_t
    snrs = mag_jax[peak_indices]
    return times, snrs


def execute_fft_jax(invec_data, normalize_by_rate=True, ifft=False):
    """Transform on the selected device without sharing mutable native plans.

    XLA caches transform executables; immutable outputs need no buffer cache.
    """
    values = to_jax(invec_data)
    epoch = getattr(invec_data, '_epoch', None)
    if ifft:
        ntime = (len(invec_data) - 1) * 2
        delta_f = getattr(invec_data, 'delta_f', 1.0 / ntime)
        data = jnp.fft.irfft(values, n=ntime) * ntime
        if normalize_by_rate:
            data *= invec_data.delta_f
        return pycbc.types.TimeSeries(JAXArrayData(data), copy=False,
                                     delta_t=1.0 / (ntime * delta_f),
                                     epoch=epoch)
    delta_t = getattr(invec_data, 'delta_t', 1.0)
    data = jnp.fft.rfft(values)
    if normalize_by_rate:
        data *= invec_data.delta_t
    return pycbc.types.FrequencySeries(
        JAXArrayData(data), copy=False,
        delta_f=1.0 / (len(values) * delta_t), epoch=epoch)


def gate_data_jax(data, gate_params):
    """Apply the standard inverted Tukey gates to a device-resident series."""
    values = to_jax(data)
    sample_rate = 1.0 / data.delta_t
    for glitch_time, glitch_width, pad_width in gate_params:
        t_start = glitch_time - glitch_width - pad_width - data.start_time
        t_end = glitch_time + glitch_width + pad_width - data.start_time
        if t_start > data.duration or t_end < 0.0:
            continue
        win_samples = int(2 * sample_rate * (glitch_width + pad_width))
        pad_samples = int(sample_rate * pad_width)
        midlen = win_samples - 2 * pad_samples
        if midlen < 0:
            raise ValueError("No zeros left after applying padding.")
        pad = 0.5 * (1.0 + jnp.cos(jnp.pi *
                                 jnp.arange(pad_samples) / pad_samples))
        window = jnp.concatenate((pad, jnp.zeros(midlen), pad[::-1]))
        offset = int(t_start * sample_rate)
        idx1 = max(0, -offset)
        idx2 = min(len(window), len(data) - offset)
        # Match the original in-place multiply, including its dtype cast.
        values = values.at[idx1 + offset:idx2 + offset].set(
            (values[idx1 + offset:idx2 + offset] * window[idx1:idx2])
            .astype(values.dtype))
    data._data = JAXArrayData(values)
    return data


def fourier_segments_jax(segments):
    """Transform complete strain segments as one batch on the active device."""
    values = to_jax(segments.strain)
    stacked = jnp.stack([values[s] for s in segments.segment_slices])
    spectra = jnp.fft.rfft(stacked, axis=-1) * segments.strain.delta_t
    result = []
    for i, (seg, analyze) in enumerate(zip(segments.segment_slices,
                                         segments.analyze_slices)):
        series = pycbc.types.FrequencySeries(
            JAXArrayData(spectra[i]), delta_f=segments.delta_f,
            epoch=segments.strain[seg]._epoch, copy=False)
        series.analyze = analyze
        series.cumulative_index = seg.start + analyze.start
        series.seg_slice = seg
        result.append(series)
    return result
