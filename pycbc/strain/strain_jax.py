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
import math
from dataclasses import dataclass
import jax
import jax.numpy as jnp

from pycbc import scheme
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
from pycbc.types.array_jax import (
    JAXArrayData, _cpu_reference, _divide, _ensure_x64, _reference_enabled, to_jax,
)


_PSD_REFERENCES = frozenset((
    "welch", "interpolate", "inverse_spectrum_truncation", "fft", "ifft",
))
_OVERWHITEN_REFERENCES = _PSD_REFERENCES | {"divide", "gate_data"}


def _selected_references():
    return getattr(scheme.mgr.state, "jax_reference_operations", frozenset())


def _check_reference_cache(buffer):
    """Do not reuse PSDs or spectra made with different validation routes."""
    key = _selected_references() & _OVERWHITEN_REFERENCES
    previous = getattr(buffer, "_jax_reference_key", key)
    if previous != key:
        buffer.psds = {}
        buffer.segments = {}
    buffer._jax_reference_key = key


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
    if factor != 1:
        block = _circular_fir_zero(block, resample_coefficients)[::factor]
    block = block[corruption:]

    output = jnp.roll(strain_buffer, -sample_step)
    write_start = len(strain_buffer) - conditioned_size + corruption
    return output.at[write_start:].set(block)


def can_fuse_strain_buffer_jax(buffer, blocksize):
    """Return whether both live FIR stages use their circular-filter path."""
    if _selected_references() & {
        "fir_zero_filter", "lfilter", "resample", "fft", "ifft",
    }:
        return False
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
        _reference_enabled("firwin"),
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
        resample_coefficients = (
            firwin(factor * 20 + 1, 1.0 / factor, window=("kaiser", 5))
            if factor != 1 else jnp.ones(1, dtype=jnp.float64)
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


def _autogate_magnitude_jax(
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
    w = (jnp.arange(corrupt_length) / float(corrupt_length)).astype(s.dtype)
    s[0:corrupt_length] *= pycbc.types.Array(JAXArrayData(w), copy=False)
    s[(len(s) - corrupt_length):] *= pycbc.types.Array(
        JAXArrayData(w[::-1]), copy=False)

    if output_intermediates:
        s.save_to_wav("strain_conditioned.wav")

    # Zero-pad strain to a power-of-2 length
    strain_pad_length = next_power_of_2(len(s))
    pad_start = int(strain_pad_length / 2 - len(s) / 2)
    pad_end = pad_start + len(s)
    s_arr = _zero_pad_on_device(s, strain_pad_length, pad_start, pad_end)

    if (_selected_references() & _PSD_REFERENCES) or output_intermediates:
        # Public stages retain each independent validation switch. The normal
        # resident path below still compiles the complete computation together.
        from pycbc import psd as psd_module

        psd = psd_module.welch(
            s[corrupt_length:len(s) - corrupt_length],
            seg_len=int(psd_duration * s.sample_rate),
            seg_stride=int(psd_stride * s.sample_rate),
            avg_method=psd_avg_method, require_exact_data_fit=False)
        psd = psd_module.interpolate(
            psd, 1.0 / (strain_pad_length * s.delta_t))
        psd = psd_module.inverse_spectrum_truncation(
            psd, int(psd_duration * s.sample_rate),
            low_frequency_cutoff=low_freq_cutoff, trunc_method="hann")
        psd[:int(low_freq_cutoff / psd.delta_f)] = math.inf
        if high_freq_cutoff:
            psd[int(high_freq_cutoff / psd.delta_f):] = math.inf
        padded = pycbc.types.TimeSeries(
            JAXArrayData(s_arr), delta_t=s.delta_t,
            epoch=s.start_time - pad_start * s.delta_t, copy=False)
        spectrum = padded.to_frequencyseries()
        norm = (high_freq_cutoff or s.sample_rate / 2.0) - low_freq_cutoff
        spectrum *= (psd * norm) ** (-0.5)
        whitened = spectrum.to_timeseries()[pad_start:pad_end]
        magnitude = abs(whitened)
        if output_intermediates:
            whitened.save_to_wav("strain_whitened.wav")
            magnitude.save("strain_whitened_mag.npy")
        mag = to_jax(magnitude)
        if corrupt_length:
            mag = mag.at[:corrupt_length].set(0)
            mag = mag.at[-corrupt_length:].set(0)
        return s, mag

    # Prepare segmentation parameters for Welch PSD
    s_trim = s[corrupt_length:(len(s) - corrupt_length)]
    seg_len = int(psd_duration * s.sample_rate)
    seg_stride = int(psd_stride * s.sample_rate)
    if psd_avg_method not in ("mean", "median", "median-mean"):
        raise ValueError("Invalid averaging method")
    if seg_len <= 0 or seg_stride <= 0:
        raise ValueError("Segment length and stride must be positive integers")
    num_samples = len(s_trim)
    num_segments = int(num_samples // seg_stride)
    if (num_segments - 1) * seg_stride + seg_len > num_samples:
        num_segments -= 1
    data_len = (num_segments - 1) * seg_stride + seg_len
    if num_segments < 1 or data_len > num_samples:
        raise ValueError("Incorrect choice of segmentation parameters")
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
    return s, mag_jax


def detect_loud_glitches_jax(
    strain, psd_duration=4.0, psd_stride=2.0, psd_avg_method="median",
    low_freq_cutoff=30.0, threshold=50.0, cluster_window=5.0,
    corrupt_time=4.0, high_freq_cutoff=None, output_intermediates=False,
):
    """Retain the public compact glitch-list API outside the resident path."""
    if _reference_enabled("detect_loud_glitches"):
        import numpy as np
        from pycbc.reference_jax import cpu_reference

        return cpu_reference(
            "detect_loud_glitches", np.asarray(to_jax(strain)),
            spacing=strain.delta_t, epoch=strain._epoch,
            psd_duration=psd_duration, psd_stride=psd_stride,
            psd_avg_method=psd_avg_method, low_freq_cutoff=low_freq_cutoff,
            threshold=threshold, cluster_window=cluster_window,
            corrupt_time=corrupt_time, high_freq_cutoff=high_freq_cutoff,
            output_intermediates=output_intermediates)
    s, mag_jax = _autogate_magnitude_jax(
        strain, psd_duration, psd_stride, psd_avg_method, low_freq_cutoff,
        threshold, cluster_window, corrupt_time, high_freq_cutoff,
        output_intermediates)
    indices = jnp.where(mag_jax > threshold)[0]
    if len(indices) == 0:
        return jnp.empty(0, dtype=jnp.float64)

    cluster_window_samples = int(cluster_window * s.sample_rate)
    cluster_idx = pycbc.events.findchirp_cluster_over_window(
        indices.astype(jnp.int32),
        mag_jax[indices],
        cluster_window_samples,
    )
    peak_indices = indices[cluster_idx]
    times = float(s.start_time) + peak_indices * s.delta_t
    return times


@functools.partial(
    jax.jit, static_argnames=("cluster_samples", "window_samples", "pad_samples"))
def _resident_autogate_core(
    values, magnitude, threshold, detection_epoch, delta_t, strain_epoch,
    gate_width, gate_taper, *, cluster_samples, window_samples, pad_samples,
):
    """Select FindChirp peaks and apply ordered Tukey gates without readback."""
    size = magnitude.size
    capacity = (size + cluster_samples) // (cluster_samples + 1)
    if size == 0:
        return values, jnp.zeros(0, jnp.int32), jnp.int32(0)
    above = magnitude > threshold
    indices = jnp.nonzero(above, size=size, fill_value=size)[0].astype(jnp.int32)
    count = jnp.sum(above, dtype=jnp.int32)
    selected = jnp.zeros(capacity, dtype=jnp.int32)

    def select_step(carry):
        index, current, selected_count, output = carry
        candidate = indices[index]
        new_group = ((selected_count == 0)
                     | ((candidate - current) > cluster_samples))
        replace = (~new_group) & (magnitude[candidate] > magnitude[current])
        slot = jnp.where(new_group, selected_count, selected_count - 1)
        output = jax.lax.cond(
            new_group | replace,
            lambda old: old.at[slot].set(candidate), lambda old: old, output)
        current = jnp.where(new_group | replace, candidate, current)
        return index + 1, current, selected_count + new_group, output

    _, _, selected_count, selected = jax.lax.while_loop(
        lambda carry: carry[0] < count, select_step,
        (jnp.int32(0), jnp.int32(0), jnp.int32(0), selected))
    pad = 0.5 * (1.0 + jnp.cos(jnp.pi * jnp.arange(pad_samples) / pad_samples))
    middle = jnp.zeros(window_samples - 2 * pad_samples)
    window = jnp.concatenate((pad, middle, pad[::-1]))
    window_indices = jnp.arange(window_samples, dtype=jnp.int64)

    def apply_gate(index, output):
        # Match the former separately evaluated absolute-time operations; GPS
        # epoch cancellation or FMA must not move a gate by one sample.
        barrier = jax.lax.optimization_barrier
        relative = barrier(selected[index] * delta_t)
        glitch_time = barrier(detection_epoch + relative)
        start = barrier(barrier(glitch_time - gate_width) - gate_taper)
        start = barrier(start - strain_epoch)
        end = barrier(barrier(glitch_time + gate_width) + gate_taper)
        end = barrier(end - strain_epoch)
        offset = (start * (1.0 / delta_t)).astype(jnp.int64)
        targets = offset + window_indices
        valid = (targets >= 0) & (targets < values.size)
        destinations = jnp.where(valid, targets, values.size)
        current = output[jnp.clip(targets, 0, values.size - 1)]
        updated = (current * window).astype(values.dtype)
        return jax.lax.cond(
            (start > values.size * delta_t) | (end < 0.0),
            lambda old: old,
            lambda old: old.at[destinations].set(updated, mode="drop"), output)

    gated = jax.lax.fori_loop(0, selected_count, apply_gate, values)
    return gated, selected, selected_count


@dataclass(frozen=True)
class _ResidentGateParameters:
    """Frozen device peak metadata collected only by terminal output code."""
    indices: object
    count: object
    epoch: float
    delta_t: float
    width: float
    taper: float

    def snapshot(self):
        return self

    def materialize(self):
        import numpy as np
        indices, count = jax.device_get((self.indices, self.count))
        times = self.epoch + np.asarray(indices)[:int(count)] * self.delta_t
        return [(time, self.width, self.taper) for time in times]

    def __iter__(self):
        return iter(self.materialize())

    def __len__(self):
        return len(self.materialize())

    def __array__(self, dtype=None, copy=None):
        import numpy as np
        return np.asarray(self.materialize(), dtype=dtype)


def _is_cuda_autogate_array(values):
    return (getattr(getattr(values, "device", None), "platform", None)
            in ("gpu", "cuda"))


def autogate_strain_buffer_jax(buffer):
    """Handle CUDA live autogating through the terminal output boundary."""
    if _selected_references() & {
        "detect_loud_glitches", "gate_data", "findchirp_cluster",
    }:
        return False
    values = to_jax(buffer.strain)
    if not _is_cuda_autogate_array(values):
        return False
    if buffer.autogating_threshold is None:
        return True
    _ensure_x64()
    rate = buffer.sample_rate
    cluster = int(buffer.autogating_cluster * rate)
    win = int(2 * rate * (buffer.autogating_width + buffer.autogating_taper))
    pad = int(rate * buffer.autogating_taper)
    if cluster <= 0 or win < 0 or pad < 0 or win < 2 * pad:
        return False
    start = int(len(buffer.strain) - buffer.autogating_duration * rate)
    detection = buffer.strain[start:-buffer.corruption]
    s, magnitude = _autogate_magnitude_jax(
        detection,
        psd_duration=buffer.autogating_psd_segment_length,
        psd_stride=buffer.autogating_psd_stride,
        threshold=buffer.autogating_threshold,
        cluster_window=buffer.autogating_cluster,
        low_freq_cutoff=buffer.highpass_frequency,
        corrupt_time=buffer.autogating_pad)
    epoch = float(s.start_time)
    gated, indices, count = _resident_autogate_core(
        values, magnitude, buffer.autogating_threshold, epoch, s.delta_t,
        float(buffer.strain.start_time), buffer.autogating_width,
        buffer.autogating_taper, cluster_samples=cluster,
        window_samples=win, pad_samples=pad)
    buffer.strain.data.set_array(gated)
    buffer.gate_params = _ResidentGateParameters(
        indices, count, epoch, s.delta_t,
        buffer.autogating_width, buffer.autogating_taper)
    return True


def execute_fft_jax(invec_data, normalize_by_rate=True, ifft=False):
    """Transform on the selected device without sharing mutable native plans.

    XLA caches transform executables; immutable outputs need no buffer cache.
    """
    values = to_jax(invec_data)
    epoch = getattr(invec_data, '_epoch', None)
    operation = "ifft" if ifft else "fft"
    native = _reference_enabled(operation)
    if ifft:
        ntime = (len(invec_data) - 1) * 2
        delta_f = getattr(invec_data, 'delta_f', 1.0 / ntime)
        data = (_unscaled_reference_fft(values, operation, ntime) if native
                else jnp.fft.irfft(values, n=ntime) * ntime)
        if normalize_by_rate:
            # Original cached plans derive their frequency spacing from the
            # input's delta_t, including its floating-point rounding.
            data *= (1.0 / (ntime * invec_data.delta_t) if native
                     else invec_data.delta_f)
        return pycbc.types.TimeSeries(JAXArrayData(data), copy=False,
                                     delta_t=1.0 / (ntime * delta_f),
                                     epoch=epoch)
    delta_t = getattr(invec_data, 'delta_t', 1.0)
    data = (_unscaled_reference_fft(values, operation, len(values)) if native
            else jnp.fft.rfft(values))
    if normalize_by_rate:
        data *= invec_data.delta_t
    return pycbc.types.FrequencySeries(
        JAXArrayData(data), copy=False,
        delta_f=1.0 / (len(values) * delta_t), epoch=epoch)


def _unscaled_reference_fft(values, operation, ntime):
    """Execute the original native class plan with explicit device buffers."""
    from pycbc.fft import FFT, IFFT

    length = ntime if operation == "ifft" else ntime // 2 + 1
    if operation == "ifft":
        dtype = jnp.float32 if values.dtype == jnp.complex64 else jnp.float64
    else:
        dtype = jnp.complex64 if values.dtype == jnp.float32 else jnp.complex128
    source = pycbc.types.Array(JAXArrayData(values), copy=False)
    target = pycbc.types.Array(
        JAXArrayData(jnp.empty(length, dtype=dtype)), copy=False)
    (IFFT if operation == "ifft" else FFT)(source, target).execute()
    return to_jax(target)


def gate_data_jax(data, gate_params):
    """Apply the standard inverted Tukey gates to a device-resident series."""
    values = to_jax(data)
    if _reference_enabled("gate_data"):
        import numpy as np
        from pycbc.reference_jax import cpu_reference

        result, _, _ = cpu_reference(
            "gate_data", np.asarray(values), spacing=data.delta_t,
            epoch=data._epoch, gate_params=list(gate_params))
        data.data.set_array(to_jax(result, device=values.device))
        return data
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
    data.data.set_array(values)
    return data


def fourier_segments_jax(segments):
    """Transform complete strain segments as one batch on the active device."""
    values = to_jax(segments.strain)
    stacked = jnp.stack([values[s] for s in segments.segment_slices])
    if _reference_enabled("fft"):
        spectra = jnp.stack([
            _unscaled_reference_fft(row, "fft", row.size)
            for row in stacked]) * segments.strain.delta_t
    else:
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


@functools.partial(
    jax.jit,
    static_argnames=("reduced_pad", "taper_samples", "output_length"),
)
def _overwhiten_fused_core(
    strain_slice,
    psdt_array,
    delta_t,
    reduced_pad,
    taper_samples,
    output_length,
):
    """Fused real FFT -> PSD division -> inverse FFT -> taper -> real FFT.

    Executes on the active JAX device in a single fused XLA computation without
    allocating intermediate TimeSeries or FrequencySeries objects.
    """
    if reduced_pad == 0:
        return (jnp.fft.rfft(strain_slice) * delta_t) / psdt_array

    # 1. Forward FFT & PSD division & Inverse FFT
    w1 = jnp.fft.irfft(
        jnp.fft.rfft(strain_slice) / psdt_array,
        n=strain_slice.shape[0],
    )
    # 2. Trim padding
    w2 = w1[reduced_pad:reduced_pad + output_length]
    # 3. Taper ends
    if taper_samples > 0:
        pad_samples = taper_samples // 2
        pad = 0.5 * (1.0 + jnp.cos(
            jnp.pi * jnp.arange(pad_samples, dtype=jnp.float64) / pad_samples
        ))
        # Match gate_data's full window and int() offsets, including
        # truncation toward zero when an oversized right gate starts before
        # the segment. Applying each clipped gate separately preserves the
        # in-place dtype casts where the gates overlap.
        middle = jnp.zeros(taper_samples - 2 * pad_samples, dtype=pad.dtype)
        window = jnp.concatenate((pad, middle, pad[::-1]))
        offsets = (-pad_samples, int(output_length - taper_samples / 2))
        for offset in offsets:
            idx1 = max(0, -offset)
            idx2 = min(taper_samples, output_length - offset)
            start, end = idx1 + offset, idx2 + offset
            w2 = w2.at[start:end].set(
                (w2[start:end] * window[idx1:idx2]).astype(w2.dtype))
    # 4. Final Forward FFT
    return jnp.fft.rfft(w2) * delta_t


@functools.partial(jax.jit, static_argnames=("kend",))
def _psd_horizon_distance_core(cumulative_norm, amplitude_squared, *, kend):
    """Finish the BNS horizon on device with the legacy rounding boundaries."""
    barrier = jax.lax.optimization_barrier
    power = barrier(cumulative_norm[kend] * amplitude_squared)
    distance = barrier(jnp.sqrt(power))
    distance = barrier(distance / 8.0)
    distance = barrier(distance * pycbc.DYN_RANGE_FAC)
    return distance, power < 0.0


def psd_horizon_payload_jax(psd, lower_frequency_cutoff):
    """Enqueue the Live BNS horizon and domain status without collecting."""
    from pycbc.waveform.spa_tmplt import spa_amplitude_factor, spa_tmplt_end
    from pycbc.vetoes.chisq_jax import (
        _ordered_cumsum_rows, _use_gpu_ordered_scan, _weighted_power_divide,
    )

    _ensure_x64()
    kend = int(spa_tmplt_end(mass1=1.4, mass2=1.4) / psd.delta_f)
    if kend >= len(psd):
        kend = len(psd) - 2
    values = to_jax(psd)
    if _reference_enabled("psd_horizon"):
        from pycbc.reference_jax import cpu_reference

        distance = cpu_reference(
            "psd_horizon", psd, spacing=psd.delta_f,
            epoch=getattr(psd, "_epoch", None),
            lower_frequency_cutoff=lower_frequency_cutoff)
        return to_jax(distance, device=values.device), to_jax(
            False, device=values.device)
    if _reference_enabled("psd_horizon_amplitude"):
        from pycbc.reference_jax import cpu_reference

        amp = to_jax(cpu_reference(
            "psd_horizon_amplitude", None, spacing=psd.delta_f,
            length=len(psd)), device=values.device)
    else:
        # The original preconditioner starts at delta_f, then rounds to
        # float32 before squaring even for a double-precision PSD.
        frequencies = jnp.arange(
            1, len(psd) + 1, dtype=jnp.float64, device=values.device) * psd.delta_f
        amp = (frequencies ** (-7. / 6.)).astype(jnp.float32)
    kmin = int(lower_frequency_cutoff / psd.delta_f)
    power = amp[kmin:] ** 2
    spectrum = values[kmin:]
    weighted = (_divide(power, spectrum) if _reference_enabled("divide")
                else (_weighted_power_divide(power, spectrum)
                      if spectrum.dtype == jnp.float32 else power / spectrum))
    cumulative = (to_jax(_cpu_reference(weighted, "cumsum"), device=values.device)
                  if _reference_enabled("cumsum") else _ordered_cumsum_rows(
                      weighted[None, :], _use_gpu_ordered_scan(weighted))[0])
    scaled = cumulative * 4.0
    scaled = scaled * psd.delta_f
    norm = jnp.zeros(len(psd), dtype=jnp.float64, device=values.device)
    norm = norm.at[kmin:].set(scaled.astype(jnp.float64))
    amplitude_squared = spa_amplitude_factor(mass1=1.4, mass2=1.4) ** 2.0
    return _psd_horizon_distance_core(norm, amplitude_squared, kend=kend)


def psd_horizon_distance_jax(psd, lower_frequency_cutoff):
    """Collect the completed horizon at the PSD scientific control boundary."""
    distance, negative_power = jax.device_get(
        psd_horizon_payload_jax(psd, lower_frequency_cutoff))
    if negative_power:
        raise ValueError("math domain error")
    return float(distance)


def _ensure_psd_for_delta_f(buffer, delta_f):
    """Ensure truncated device PSDs are prepared for a given delta_f."""
    if delta_f in buffer.psds:
        psd = buffer.psds[delta_f]
        if getattr(psd, "_jax_psdt", None) is None:
            psd._jax_psdt = to_jax(psd.psdt)
        if getattr(psd, "_jax_psd", None) is None:
            psd._jax_psd = to_jax(psd)
        return psd

    import pycbc.psd

    buffer_length = int(1.0 / delta_f)
    e = len(buffer.strain)
    reduced_pad = int(buffer.reduced_pad)
    s = int(e - buffer_length * buffer.sample_rate - reduced_pad * 2)
    fseries_len = e - s
    fseries_delta_f = 1.0 / (fseries_len * buffer.strain.delta_t)

    psdt = pycbc.psd.interpolate(buffer.psd, fseries_delta_f)
    psdt = pycbc.psd.inverse_spectrum_truncation(
        psdt,
        int(buffer.sample_rate * buffer.psd_inverse_length),
        low_frequency_cutoff=buffer.low_frequency_cutoff,
    )
    psdt._delta_f = fseries_delta_f

    psd = pycbc.psd.interpolate(buffer.psd, delta_f)
    psd = pycbc.psd.inverse_spectrum_truncation(
        psd,
        int(buffer.sample_rate * buffer.psd_inverse_length),
        low_frequency_cutoff=buffer.low_frequency_cutoff,
    )
    psd.psdt = psdt
    psd._jax_psdt = to_jax(psdt)
    psd._jax_psd = to_jax(psd)
    buffer.psds[delta_f] = psd
    return psd


@functools.partial(jax.jit, static_argnames=("static_configs",))
def _multi_psd_prepare_core(psd, old_df, grid_dfs, static_configs):
    """Prepare the existing padded/unpadded PSD grids in one dispatch."""
    results = []
    for df, (n_freq, kmin, trunc_start, trunc_end) in zip(
        grid_dfs, static_configs
    ):
        # Retain the public interpolation's rounding before hard truncation.
        interpolated = _interp_core(psd, old_df, df, n_freq).astype(psd.dtype)
        results.append(_inv_trunc_core(
            interpolated, None, 0.0, n_freq, (n_freq - 1) * 2,
            kmin, trunc_start, trunc_end, "invasd", False,
        ))
    return tuple(results)


def _prepare_missing_psds_cuda(buffer, delta_fs):
    """Batch valid CUDA cache misses; leave other inputs to the usual path.

    Only compiled kernels are reused. Numerical results remain in ``psds``,
    which StrainBuffer invalidates when its current PSD changes.
    """
    if _selected_references() & _PSD_REFERENCES:
        return False
    # NumPy scalars have different promotion rules in cutoff division and
    # interpolation. Generic collections retain their sequential API behavior.
    if type(delta_fs) is not tuple or any(
        type(df) not in (int, float) for df in delta_fs
    ):
        return False
    missing = tuple(df for df in delta_fs if df not in buffer.psds)
    if not missing or len(set(delta_fs)) != len(delta_fs):
        return False
    data = getattr(buffer.psd, "_data", None)
    if not isinstance(data, JAXArrayData) or data.parent is not None:
        return False
    values = data.array
    if not isinstance(values, jax.Array) or values.dtype not in (
        jnp.float32, jnp.float64
    ):
        return False
    devices = values.devices()
    if len(devices) != 1:
        return False
    device = next(iter(devices))
    if (device.platform != "gpu"
            or device != getattr(scheme.mgr.state, "jax_device", None)):
        return False

    # Validate before dispatch/publishing. Unsupported geometry uses the
    # sequential API, retaining its errors and partial-cache behavior.
    grid_dfs = []
    configs = []
    try:
        old_df = float(buffer.psd.delta_f)
        max_filter_len = int(buffer.sample_rate * buffer.psd_inverse_length)
        cutoff = buffer.low_frequency_cutoff
        if not len(values) or not math.isfinite(old_df) or old_df <= 0:
            return False
        if max_filter_len <= 0:
            return False
        for df in missing:
            if not math.isfinite(df) or df <= 0:
                return False
            duration = int(1.0 / df)
            e = len(buffer.strain)
            s = int(e - duration * buffer.sample_rate
                    - int(buffer.reduced_pad) * 2)
            if s < 0 or s >= e:
                return False
            padded_df = 1.0 / ((e - s) * buffer.strain.delta_t)
            for grid_df in (padded_df, df):
                n_freq = int(round((len(values) - 1) * old_df / grid_df + 1))
                trunc_df = float(grid_df)
                n_time = (n_freq - 1) * 2
                start = max_filter_len // 2
                end = n_time - start
                if n_freq <= 1 or end < start:
                    return False
                if cutoff is not None and (
                    not math.isfinite(cutoff)
                    or cutoff < 0 or cutoff > (n_freq - 1) * trunc_df
                ):
                    return False
                kmin = int(cutoff / trunc_df) if cutoff else 1
                grid_dfs.append(grid_df)
                configs.append((n_freq, kmin, start, end))
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False

    prepared = _multi_psd_prepare_core(
        values, old_df, tuple(grid_dfs), tuple(configs),
    )
    # The original inverse truncation creates a new zero-epoch spectrum.
    epoch = 0
    for i, df in enumerate(missing):
        psdt = pycbc.types.FrequencySeries(
            JAXArrayData(prepared[2 * i]), delta_f=grid_dfs[2 * i],
            epoch=epoch, copy=False,
        )
        psd = pycbc.types.FrequencySeries(
            JAXArrayData(prepared[2 * i + 1]), delta_f=float(df),
            epoch=epoch, copy=False,
        )
        psd.psdt = psdt
        psd._jax_psdt = prepared[2 * i]
        psd._jax_psd = prepared[2 * i + 1]
        buffer.psds[df] = psd
    return True


@functools.partial(
    jax.jit,
    static_argnames=("static_configs", "delta_t"),
)
def _multi_overwhiten_fused_core(
    strain_array,
    psdts_tuple,
    static_configs,
    delta_t,
):
    """Batch-process multiple duration overwhitening passes on device."""
    results = []
    for (s, e, reduced_pad, taper_samples, output_length), psdt_arr in zip(
        static_configs, psdts_tuple
    ):
        sl = strain_array[s:e]
        val = _overwhiten_fused_core(
            sl,
            psdt_arr,
            delta_t,
            reduced_pad,
            taper_samples,
            output_length,
        )
        results.append(val)
    return tuple(results)


def _overwhiten_single_delta_f(buffer, delta_f):
    """Compute and cache a single duration overwhitened segment."""
    _ensure_psd_for_delta_f(buffer, delta_f)
    psd = buffer.psds[delta_f]
    if _selected_references() & _OVERWHITEN_REFERENCES:
        return _overwhiten_staged(buffer, delta_f, psd)
    psdt_array = getattr(psd, "_jax_psdt", None)
    if psdt_array is None:
        psdt_array = to_jax(psd.psdt)
        psd._jax_psdt = psdt_array

    buffer_length = int(1.0 / delta_f)
    e = len(buffer.strain)
    reduced_pad = int(buffer.reduced_pad)
    s = int(e - buffer_length * buffer.sample_rate - reduced_pad * 2)
    output_length = int(buffer_length * buffer.sample_rate)
    strain_slice = to_jax(buffer.strain)[s:e]
    delta_t = float(buffer.strain.delta_t)

    if reduced_pad != 0:
        taper_window = buffer.trim_padding / 2.0 / buffer.sample_rate
        taper_samples = int(2 * buffer.sample_rate * taper_window)
    else:
        taper_samples = 0

    values = _overwhiten_fused_core(
        strain_slice,
        psdt_array,
        delta_t,
        reduced_pad,
        taper_samples,
        output_length,
    )

    epoch = buffer.strain._epoch + (s + reduced_pad) * buffer.strain.delta_t
    result = pycbc.types.FrequencySeries(
        JAXArrayData(values),
        delta_f=delta_f,
        epoch=epoch,
        copy=False,
    )
    result.psd = psd
    buffer.segments[delta_f] = result
    return result


def _overwhiten_staged(buffer, delta_f, psd):
    """Retain the original stage order while validating individual kernels."""
    e = len(buffer.strain)
    s = int(e - int(1.0 / delta_f) * buffer.sample_rate
            - buffer.reduced_pad * 2)
    spectrum = execute_fft_jax(buffer.strain[s:e])
    spectrum /= psd.psdt
    if buffer.reduced_pad:
        overwhite = execute_fft_jax(spectrum, ifft=True)
        trimmed = overwhite[
            buffer.reduced_pad:len(overwhite) - buffer.reduced_pad]
        taper = buffer.trim_padding / 2.0 / overwhite.sample_rate
        gate_data_jax(trimmed, [(trimmed.start_time, 0.0, taper),
                               (trimmed.end_time, 0.0, taper)])
        result = execute_fft_jax(trimmed)
        result.start_time = (spectrum.start_time
                             + buffer.reduced_pad * buffer.strain.delta_t)
    else:
        result = spectrum
    result.psd = psd
    buffer.segments[delta_f] = result
    return result


def preload_overwhitened_data_jax(buffer, delta_fs=None):
    """Precompute and cache device-resident overwhitened segments."""
    _ensure_x64()
    _check_reference_cache(buffer)
    if delta_fs is None:
        delta_fs = getattr(buffer, "required_delta_fs", None)
    if delta_fs is None:
        delta_fs = tuple(buffer.psds.keys())
    if not delta_fs:
        return {}

    missing_dfs = tuple(df for df in delta_fs if df not in buffer.segments)
    if not missing_dfs:
        return {
            df: buffer.segments[df] for df in delta_fs if df in buffer.segments
        }

    _prepare_missing_psds_cuda(buffer, missing_dfs)
    for df in missing_dfs:
        _ensure_psd_for_delta_f(buffer, df)

    if (len(missing_dfs) == 1
            or _selected_references() & _OVERWHITEN_REFERENCES):
        for df in missing_dfs:
            _overwhiten_single_delta_f(buffer, df)
        return {
            df: buffer.segments[df] for df in delta_fs if df in buffer.segments
        }

    e = len(buffer.strain)
    reduced_pad = int(buffer.reduced_pad)
    delta_t = float(buffer.strain.delta_t)
    if reduced_pad != 0:
        taper_window = buffer.trim_padding / 2.0 / buffer.sample_rate
        taper_samples = int(2 * buffer.sample_rate * taper_window)
    else:
        taper_samples = 0

    configs = []
    psdts = []
    epochs = []
    for df in missing_dfs:
        buf_len = int(1.0 / df)
        s = int(e - buf_len * buffer.sample_rate - reduced_pad * 2)
        out_len = int(buf_len * buffer.sample_rate)
        configs.append((s, e, reduced_pad, taper_samples, out_len))
        psdts.append(buffer.psds[df]._jax_psdt)
        epochs.append(buffer.strain._epoch + (s + reduced_pad) * delta_t)

    strain_array = to_jax(buffer.strain)
    values_tuple = _multi_overwhiten_fused_core(
        strain_array,
        tuple(psdts),
        tuple(configs),
        delta_t,
    )

    for df, values, ep in zip(missing_dfs, values_tuple, epochs):
        res = pycbc.types.FrequencySeries(
            JAXArrayData(values),
            delta_f=df,
            epoch=ep,
            copy=False,
        )
        res.psd = buffer.psds[df]
        buffer.segments[df] = res

    return {
        df: buffer.segments[df] for df in delta_fs if df in buffer.segments
    }


def overwhitened_data_jax(buffer, delta_f):
    """Compute and cache overwhitened strain data on the JAX device."""
    _check_reference_cache(buffer)
    if delta_f in buffer.segments:
        return buffer.segments[delta_f]

    _ensure_x64()
    required_dfs = getattr(buffer, "required_delta_fs", None)
    if (type(required_dfs) is tuple and required_dfs
            and type(delta_f) in (int, float)
            and all(type(df) in (int, float) for df in required_dfs)):
        if delta_f not in required_dfs:
            required_dfs = required_dfs + (delta_f,)
        _prepare_missing_psds_cuda(buffer, required_dfs)
    _ensure_psd_for_delta_f(buffer, delta_f)

    target_dfs = getattr(buffer, "required_delta_fs", None)
    if target_dfs is None:
        target_dfs = tuple(buffer.psds.keys())
    if delta_f not in target_dfs:
        target_dfs = target_dfs + (delta_f,)

    if len(target_dfs) > 1:
        preload_overwhitened_data_jax(buffer, delta_fs=target_dfs)
        if delta_f in buffer.segments:
            return buffer.segments[delta_f]

    return _overwhiten_single_delta_f(buffer, delta_f)
