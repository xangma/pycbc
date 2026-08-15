"""Torch kernels for the relative-binning likelihood."""

import torch


# Match the constant used by the established Cython kernels exactly.
_RELBIN_PI = 3.141592653


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


def _summaries(ratio, a0, a1, b0, b1):
    """Calculate the linearized data and waveform inner products."""
    ratio_lo = ratio[..., :-1]
    ratio_delta = ratio[..., 1:] - ratio_lo
    hd = (a0 * ratio_lo + a1 * ratio_delta).sum(dim=-1)

    power = ratio.real.square() + ratio.imag.square()
    power_lo = power[..., :-1]
    power_delta = power[..., 1:] - power_lo
    hh = (b0 * power_lo + b1 * power_delta).sum(dim=-1)
    return hd.conj(), hh.real


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
