"""Torch kernels for the relative-binning likelihood."""

import torch


# Match the constant used by the established Cython kernels exactly.
_RELBIN_PI = 3.141592653
_SNR_PREDICTOR_TARGET_ELEMENTS = 2 ** 20


def _torch_tensor(value):
    """Return the raw tensor stored by a Torch or PyCBC value."""
    if isinstance(value, torch.Tensor):
        return value
    tensor = getattr(value, "tensor", None)
    if tensor is None:
        tensor = getattr(getattr(value, "_data", None), "tensor", None)
    return tensor


def _as_tensor(value, like, dtype):
    """Move ``value`` beside ``like`` without copying matching tensors."""
    tensor = _torch_tensor(value)
    if tensor is None:
        tensor = value
    return torch.as_tensor(tensor, device=like.device, dtype=dtype)


def prepare_likelihood_data(like, freqs, h00, a0, a1, b0, b1):
    """Prepare static relative-binning data on a waveform's device."""
    like = _torch_tensor(like)
    if like is None:
        raise TypeError("a Torch-backed waveform is required")

    real_dtype = like.real.dtype
    complex_dtype = like.dtype
    return (
        _as_tensor(freqs, like, real_dtype),
        _as_tensor(h00, like, complex_dtype),
        _as_tensor(a0, like, complex_dtype),
        _as_tensor(a1, like, complex_dtype),
        _as_tensor(b0, like, real_dtype),
        _as_tensor(b1, like, real_dtype),
    )


def _linearized_filter(ratio, a0, a1):
    """Calculate the linearized data-waveform inner product."""
    ratio_lo = ratio[..., :-1]
    ratio_delta = ratio[..., 1:] - ratio_lo
    return (a0 * ratio_lo + a1 * ratio_delta).sum(dim=-1).conj()


def _linearized_norm(ratio, b0, b1):
    """Calculate the linearized waveform norm."""
    power = ratio.real.square() + ratio.imag.square()
    power_lo = power[..., :-1]
    power_delta = power[..., 1:] - power_lo
    return (b0 * power_lo + b1 * power_delta).sum(dim=-1).real


def _summaries(ratio, a0, a1, b0, b1):
    """Calculate the linearized data and waveform inner products."""
    return (
        _linearized_filter(ratio, a0, a1),
        _linearized_norm(ratio, b0, b1),
    )


def _sample_axis(value):
    """Add a trailing frequency axis to a vector of samples."""
    return value.unsqueeze(-1) if value.ndim else value


def likelihood_parts(freqs, fp, fc, dtc, hp, hc, h00,
                     a0, a1, b0, b1):
    """Calculate scalar or frequency-varying relative likelihood parts."""
    hp = _torch_tensor(hp)
    if hp is None:
        raise TypeError("a Torch-backed waveform is required")

    real_dtype = hp.real.dtype
    hc = _as_tensor(hc, hp, hp.dtype)
    freqs = _as_tensor(freqs, hp, real_dtype)
    h00 = _as_tensor(h00, hp, hp.dtype)
    fp = _as_tensor(fp, hp, real_dtype)
    fc = _as_tensor(fc, hp, real_dtype)
    dtc = _as_tensor(dtc, hp, real_dtype)
    a0 = _as_tensor(a0, hp, hp.dtype)
    a1 = _as_tensor(a1, hp, hp.dtype)
    b0 = _as_tensor(b0, hp, real_dtype)
    b1 = _as_tensor(b1, hp, real_dtype)

    phase = -2.0 * _RELBIN_PI * dtc * freqs
    shift = torch.polar(torch.ones_like(phase), phase)
    ratio = shift * (fp * hp + fc * hc) / h00
    return _summaries(ratio, a0, a1, b0, b1)


def likelihood_parts_vector(freqs, fp, fc, dtc, hp, hc, h00,
                            a0, a1, b0, b1):
    """Calculate likelihood parts for paired sky, time, or pol samples."""
    hp = _torch_tensor(hp)
    if hp is None:
        raise TypeError("a Torch-backed waveform is required")

    real_dtype = hp.real.dtype
    hc = _as_tensor(hc, hp, hp.dtype)
    freqs = _as_tensor(freqs, hp, real_dtype)
    h00 = _as_tensor(h00, hp, hp.dtype)
    fp = _sample_axis(_as_tensor(fp, hp, real_dtype))
    fc = _sample_axis(_as_tensor(fc, hp, real_dtype))
    dtc = _sample_axis(_as_tensor(dtc, hp, real_dtype))
    a0 = _as_tensor(a0, hp, hp.dtype)
    a1 = _as_tensor(a1, hp, hp.dtype)
    b0 = _as_tensor(b0, hp, real_dtype)
    b1 = _as_tensor(b1, hp, real_dtype)

    phase = -2.0 * _RELBIN_PI * dtc * freqs
    shift = torch.polar(torch.ones_like(phase), phase)
    ratio = shift * (fp * hp + fc * hc) / h00
    return _summaries(ratio, a0, a1, b0, b1)


def _likelihood_parts_v_vector(freqs, fp, fc, times, dtc, pol_phase,
                               hp, hc, h00, a0, a1, b0, b1):
    """Calculate frequency-varying responses for paired samples."""
    hp = _torch_tensor(hp)
    if hp is None:
        raise TypeError("a Torch-backed waveform is required")

    real_dtype = hp.real.dtype
    hc = _as_tensor(hc, hp, hp.dtype)
    freqs = _as_tensor(freqs, hp, real_dtype)
    h00 = _as_tensor(h00, hp, hp.dtype)
    fp = _as_tensor(fp, hp, real_dtype)
    fc = _as_tensor(fc, hp, real_dtype)
    times = _as_tensor(times, hp, real_dtype)
    dtc = _sample_axis(_as_tensor(dtc, hp, real_dtype))
    a0 = _as_tensor(a0, hp, hp.dtype)
    a1 = _as_tensor(a1, hp, hp.dtype)
    b0 = _as_tensor(b0, hp, real_dtype)
    b1 = _as_tensor(b1, hp, real_dtype)

    response = fp + 1.0j * fc
    if pol_phase is not None:
        pol_phase = _sample_axis(_as_tensor(pol_phase, hp, hp.dtype))
        response = response * pol_phase

    phase = -2.0 * _RELBIN_PI * (times + dtc) * freqs
    shift = torch.polar(torch.ones_like(phase), phase)
    ratio = shift * (response.real * hp + response.imag * hc) / h00
    return _summaries(ratio, a0, a1, b0, b1)


def likelihood_parts_v_pol(freqs, fp, fc, dtc, pol_phase,
                           hp, hc, h00, a0, a1, b0, b1):
    """Calculate an Earth-rotation likelihood over polarization samples."""
    return _likelihood_parts_v_vector(
        freqs, fp, fc, dtc, 0.0, pol_phase,
        hp, hc, h00, a0, a1, b0, b1)


def likelihood_parts_v_time(freqs, fp, fc, times, dtc,
                            hp, hc, h00, a0, a1, b0, b1):
    """Calculate an Earth-rotation likelihood over time samples."""
    return _likelihood_parts_v_vector(
        freqs, fp, fc, times, dtc, None,
        hp, hc, h00, a0, a1, b0, b1)


def likelihood_parts_v_pol_time(freqs, fp, fc, times, dtc, pol_phase,
                                hp, hc, h00, a0, a1, b0, b1):
    """Calculate an Earth-rotation likelihood over time and polarization."""
    return _likelihood_parts_v_vector(
        freqs, fp, fc, times, dtc, pol_phase,
        hp, hc, h00, a0, a1, b0, b1)


def likelihood_parts_det(freqs, dtc, channel, h00, a0, a1, b0, b1):
    """Calculate likelihood parts for a detector-frame waveform."""
    channel = _torch_tensor(channel)
    if channel is None:
        raise TypeError("a Torch-backed waveform is required")

    real_dtype = channel.real.dtype
    freqs = _as_tensor(freqs, channel, real_dtype)
    h00 = _as_tensor(h00, channel, channel.dtype)
    dtc = _as_tensor(dtc, channel, real_dtype)
    a0 = _as_tensor(a0, channel, channel.dtype)
    a1 = _as_tensor(a1, channel, channel.dtype)
    b0 = _as_tensor(b0, channel, real_dtype)
    b1 = _as_tensor(b1, channel, real_dtype)

    phase = -2.0 * _RELBIN_PI * dtc * freqs
    shift = torch.polar(torch.ones_like(phase), phase)
    ratio = shift * channel / h00
    return _summaries(ratio, a0, a1, b0, b1)


def _time_shifted_filters(freqs, tstart, delta_t, num_samples,
                          ratio, a0, a1):
    """Evaluate relative-bin filters over a blocked uniform time grid."""
    num_samples = int(num_samples)
    if num_samples == 0:
        return ratio.new_empty((0,))

    block_size = max(
        1, _SNR_PREDICTOR_TARGET_ELEMENTS // max(1, freqs.numel()))
    filters = []
    for start in range(0, num_samples, block_size):
        stop = min(start + block_size, num_samples)
        sample_indices = torch.arange(
            start, stop, device=ratio.device, dtype=freqs.dtype)
        times = tstart + delta_t * sample_indices
        phase = -2.0 * _RELBIN_PI * times.unsqueeze(-1) * freqs
        shift = torch.polar(torch.ones_like(phase), phase)
        filters.append(_linearized_filter(shift * ratio, a0, a1))
    return torch.cat(filters)


def snr_predictor(freqs, tstart, delta_t, num_samples,
                  hp, hc, h00, a0, a1, b0, b1):
    """Return the polarization-averaged SNR on a uniform time grid."""
    hp = _torch_tensor(hp)
    if hp is None:
        raise TypeError("a Torch-backed waveform is required")

    real_dtype = hp.real.dtype
    hc = _as_tensor(hc, hp, hp.dtype)
    freqs = _as_tensor(freqs, hp, real_dtype)
    h00 = _as_tensor(h00, hp, hp.dtype)
    a0 = _as_tensor(a0, hp, hp.dtype)
    a1 = _as_tensor(a1, hp, hp.dtype)
    b0 = _as_tensor(b0, hp, real_dtype)
    b1 = _as_tensor(b1, hp, real_dtype)
    tstart = _as_tensor(tstart, hp, real_dtype)
    delta_t = _as_tensor(delta_t, hp, real_dtype)

    hp_ratio = hp / h00
    hc_ratio = hc / h00
    hh = _linearized_norm(hp_ratio, b0, b1)
    chh = _linearized_norm(hc_ratio, b0, b1)
    sh = _time_shifted_filters(
        freqs, tstart, delta_t, num_samples, hp_ratio, a0, a1)
    csh = _time_shifted_filters(
        freqs, tstart, delta_t, num_samples, hc_ratio, a0, a1)
    snr2 = (
        (sh.real.square() + sh.imag.square()) / (2.0 * hh)
        + (csh.real.square() + csh.imag.square()) / (2.0 * chh)
    )
    return torch.sqrt(snr2)


def snr_predictor_dom(freqs, tstart, delta_t, num_samples,
                      hp, h00, a0, a1, b0, b1):
    """Return dominant-mode data products on a uniform time grid."""
    hp = _torch_tensor(hp)
    if hp is None:
        raise TypeError("a Torch-backed waveform is required")

    real_dtype = hp.real.dtype
    freqs = _as_tensor(freqs, hp, real_dtype)
    h00 = _as_tensor(h00, hp, hp.dtype)
    a0 = _as_tensor(a0, hp, hp.dtype)
    a1 = _as_tensor(a1, hp, hp.dtype)
    b0 = _as_tensor(b0, hp, real_dtype)
    b1 = _as_tensor(b1, hp, real_dtype)
    tstart = _as_tensor(tstart, hp, real_dtype)
    delta_t = _as_tensor(delta_t, hp, real_dtype)

    ratio = hp / h00
    hh = _linearized_norm(ratio, b0, b1)
    sh = _time_shifted_filters(
        freqs, tstart, delta_t, num_samples, ratio, a0, a1)
    return sh, hh
