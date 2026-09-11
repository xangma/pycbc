"""Torch kernels for the relative-binning likelihood."""

import torch

from pycbc.types.backend import backend_array

# Match the constant used by the established Cython kernels exactly.
_RELBIN_PI = 3.141592653
_SNR_PREDICTOR_TARGET_ELEMENTS = 2**20


def _torch_tensor(value):
    """Return Torch storage through the public backend protocol."""
    return backend_array(value, "torch")


def _as_tensor(value, like, dtype):
    """Move ``value`` beside ``like`` without copying matching tensors."""
    tensor = _torch_tensor(value)
    if tensor is None:
        tensor = value
    return torch.as_tensor(tensor, device=like.device, dtype=dtype)


def detector_response(detector, right_ascension, declination, times, like):
    """Evaluate zero-polarization detector factors beside a waveform.

    Converting the sky coordinates before calling the detector selects its
    Torch antenna and timing kernels.  GPS times remain in their original
    representation so float32-only devices do not lose precision from the
    large absolute epoch before the detector forms sidereal angles.
    """
    like = _torch_tensor(like)
    if like is None:
        raise TypeError("a Torch-backed waveform is required")

    dtype = like.real.dtype
    right_ascension = _as_tensor(right_ascension, like, dtype)
    declination = _as_tensor(declination, like, dtype)
    polarization = torch.zeros((), device=like.device, dtype=dtype)
    time_tensor = _torch_tensor(times)
    if time_tensor is not None:
        times = time_tensor
    combined = getattr(detector, "antenna_pattern_and_time_delay", None)
    if combined is not None:
        response = combined(right_ascension, declination, polarization, times)
        return tuple(_as_tensor(value, like, dtype) for value in response)

    fp, fc = detector.antenna_pattern(right_ascension, declination, polarization, times)
    delay = detector.time_delay_from_earth_center(right_ascension, declination, times)
    return tuple(_as_tensor(value, like, dtype) for value in (fp, fc, delay))


def detector_response_at_arrival(
    detector,
    reference_time,
    right_ascension,
    declination,
    polarization,
    reference_frame,
    like,
):
    """Evaluate a detector response at its signal arrival time.

    Sky coordinates and polarization are anchored beside ``like`` before
    calling the detector, selecting its Torch timing and antenna kernels.
    The detector's reference-frame handling remains authoritative.
    """
    like = _torch_tensor(like)
    if like is None:
        raise TypeError("a Torch-backed waveform is required")

    dtype = like.real.dtype
    right_ascension = _as_tensor(right_ascension, like, dtype)
    declination = _as_tensor(declination, like, dtype)
    polarization = _as_tensor(polarization, like, dtype)
    arrival_time = detector.arrival_time(
        reference_time, right_ascension, declination, reference_frame
    )
    fp, fc = detector.antenna_pattern(
        right_ascension, declination, polarization, arrival_time
    )
    return tuple(_as_tensor(value, like, dtype) for value in (fp, fc, arrival_time))


def polarization_phase(polarization, like):
    """Build the spin-2 polarization phase beside a Torch waveform."""
    like = _torch_tensor(like)
    if like is None:
        raise TypeError("a Torch-backed waveform is required")

    angle = _as_tensor(polarization, like, like.real.dtype)
    return torch.polar(torch.ones_like(angle), -2.0 * angle)


def polarized_antenna_response(fp, fc, pol_phase, like):
    """Rotate zero-polarization antenna factors on a Torch device."""
    like = _torch_tensor(like)
    if like is None:
        raise TypeError("a Torch-backed waveform is required")

    real_dtype = like.real.dtype
    fp = _as_tensor(fp, like, real_dtype)
    fc = _as_tensor(fc, like, real_dtype)
    pol_phase = _as_tensor(pol_phase, like, like.dtype)
    response = torch.complex(fp, fc) * pol_phase
    return response.real, response.imag


def dominant_mode_projection(fp, fc, polarization, inclination, like):
    """Project a dominant-mode response beside a Torch likelihood sample."""
    like = _torch_tensor(like)
    if like is None:
        raise TypeError("a Torch-backed likelihood sample is required")

    pol_phase = polarization_phase(polarization, like)
    fp, fc = polarized_antenna_response(fp, fc, pol_phase, like)
    angle = _as_tensor(inclination, like, like.real.dtype)
    cosi = torch.cos(angle)
    plus = 0.5 * (1.0 + cosi.square())
    return torch.complex(fp * plus, fc * cosi)


def _complex_cumsum(value, dim=-1):
    """Cumulatively sum complex values on devices without complex cumsum."""
    if torch.is_complex(value):
        try:
            return torch.cumsum(value, dim=dim)
        except (RuntimeError, NotImplementedError):
            pass
    return torch.complex(
        torch.cumsum(value.real, dim=dim),
        torch.cumsum(value.imag, dim=dim),
    )


def _wrap_like(value, tensor):
    """Wrap ``tensor`` in the same PyCBC container family as ``value``."""
    from pycbc.types import Array
    from pycbc.types.array_torch import TorchArrayData

    data = TorchArrayData(tensor)
    return value._return(data) if hasattr(value, "_return") else Array(data, copy=False)


def prepare_reference_data(waveform, data, size, offset, delta_f, time_shift):
    """Pad, place, and time-shift relative-bin inputs on a Torch device."""
    like = next(
        (
            tensor
            for value in (waveform, data)
            if (tensor := _torch_tensor(value)) is not None
        ),
        None,
    )
    if like is None:
        raise TypeError("a Torch-backed waveform or data series is required")

    real_dtype = like.real.dtype
    complex_dtype = (
        like.dtype
        if like.is_complex()
        else torch.complex128
        if real_dtype == torch.float64
        else torch.complex64
    )
    waveform_tensor = _as_tensor(waveform, like, complex_dtype).reshape(-1)
    data_tensor = _as_tensor(data, like, complex_dtype).reshape(-1)

    reference = torch.zeros(size, dtype=complex_dtype, device=like.device)
    copied = min(size, waveform_tensor.numel())
    reference[:copied] = waveform_tensor[:copied]
    reference = torch.roll(reference, shifts=int(offset))

    frequencies = torch.arange(size, dtype=real_dtype, device=like.device) * delta_f
    phase = -2.0 * torch.pi * frequencies * time_shift
    shift = torch.complex(torch.cos(phase), torch.sin(phase))
    shifted_data = data_tensor * shift.conj()
    return (
        _wrap_like(waveform, reference),
        _wrap_like(data, shifted_data),
    )


def active_edge_bins(h1, h2, freqs, edges):
    """Filter a shared edge grid and build its bins on a Torch device."""
    like = next(
        (
            tensor
            for value in (h1, h2, freqs)
            if (tensor := _torch_tensor(value)) is not None
        ),
        None,
    )
    if like is None:
        raise TypeError("a Torch-backed waveform or frequency grid is required")

    real_dtype = like.real.dtype
    complex_dtype = (
        like.dtype
        if like.is_complex()
        else torch.complex128
        if real_dtype == torch.float64
        else torch.complex64
    )
    h1 = _as_tensor(h1, like, complex_dtype)
    h2 = _as_tensor(h2, like, complex_dtype)
    freqs = _as_tensor(freqs, like, real_dtype)
    edges = _as_tensor(edges, like, torch.int64).reshape(-1)

    active = (h1[edges] != 0) | (h2[edges] != 0)
    edges = edges[active]
    bins = torch.stack((edges[:-1], edges[1:]), dim=1)
    return bins, freqs[edges]


def summary_product(h1, h2, psd, freqs, bins, delta_f):
    """Calculate relative-binning coefficients on a Torch device.

    Supports 1D frequency series, 2D batched waveforms (N, F),
    or 3D tensors (B, N, F).
    """
    like = next(
        (
            tensor
            for value in (h1, h2, psd)
            if (tensor := _torch_tensor(value)) is not None
        ),
        None,
    )
    if like is None:
        raise TypeError("a Torch-backed waveform or PSD is required")

    real_dtype = like.real.dtype
    complex_dtype = (
        like.dtype
        if like.is_complex()
        else torch.complex128
        if real_dtype == torch.float64
        else torch.complex64
    )
    h1 = _as_tensor(h1, like, complex_dtype)
    h2 = _as_tensor(h2, like, complex_dtype)
    psd = _as_tensor(psd, like, real_dtype)
    freqs = _as_tensor(freqs, like, real_dtype)
    bins = _as_tensor(bins, like, torch.int64).reshape(-1, 2)
    delta_f = _as_tensor(delta_f, like, real_dtype)

    h12 = h1.conj() * h2 / psd
    zero = torch.zeros(h12.shape[:-1] + (1,), dtype=h12.dtype, device=h12.device)
    prefix = torch.cat((zero, _complex_cumsum(h12, dim=-1)), dim=-1)
    weighted_prefix = torch.cat((zero, _complex_cumsum(h12 * freqs, dim=-1)), dim=-1)

    low, high = bins.unbind(dim=1)
    totals = prefix[..., high] - prefix[..., low]
    weighted_totals = weighted_prefix[..., high] - weighted_prefix[..., low]
    widths = (high - low).to(real_dtype)
    scale = 4.0 * delta_f
    a0 = scale * totals
    a1 = 4.0 * (weighted_totals - freqs[low] * totals) / widths
    return a0, a1


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


def prepare_multi_likelihood_data(like, freqs, h00, h002, a0, a1):
    """Prepare static multi-signal summary data beside a waveform."""
    like = _torch_tensor(like)
    if like is None:
        raise TypeError("a Torch-backed waveform is required")

    real_dtype = like.real.dtype
    complex_dtype = like.dtype
    return (
        _as_tensor(freqs, like, real_dtype),
        _as_tensor(h00, like, complex_dtype),
        _as_tensor(h002, like, complex_dtype),
        _as_tensor(a0, like, complex_dtype),
        _as_tensor(a1, like, complex_dtype),
    )


def _linearized_filter(ratio, a0, a1):
    """Calculate the linearized data-waveform inner product."""
    ratio_lo = ratio[..., :-1]
    ratio_hi = ratio[..., 1:]
    return ((a0 - a1) * ratio_lo + a1 * ratio_hi).sum(dim=-1).conj()


def _linearized_norm(ratio, b0, b1):
    """Calculate the linearized waveform norm."""
    power = ratio.real.square() + ratio.imag.square()
    power_lo = power[..., :-1]
    power_hi = power[..., 1:]
    return ((b0 - b1) * power_lo + b1 * power_hi).sum(dim=-1).real


def _linearized_cross(ratio, ratio2, a0, a1):
    """Calculate a linearized cross term between two waveform ratios."""
    cross = ratio * ratio2.conj()
    cross_lo = cross[..., :-1]
    cross_hi = cross[..., 1:]
    return ((a0 - a1) * cross_lo + a1 * cross_hi).sum(dim=-1)


def _summaries(ratio, a0, a1, b0, b1):
    """Calculate the linearized data and waveform inner products."""
    return (
        _linearized_filter(ratio, a0, a1),
        _linearized_norm(ratio, b0, b1),
    )


def _sample_axis(value):
    """Map a scalar or sample shape ``(...)`` to ``(...)`` or ``(..., 1)``."""
    # Callers normalize values with ``_as_tensor`` first.  Avoid ``hasattr``
    # here because TorchDynamo treats attribute probing on tensors as a graph
    # break.
    if value.ndim == 0:
        return value
    return value.unsqueeze(-1)


def _frequency_axis(value, freqs_len, name="parameter"):
    """Validate a scalar or a value whose final axis is frequency.

    Scalar values broadcast over frequency.  Non-scalars must have shape
    ``(..., F)``; any leading dimensions are batch dimensions.
    """
    if value.ndim and value.shape[-1] != freqs_len:
        raise ValueError(f"{name} must be scalar or have shape (..., {freqs_len})")
    return value


def likelihood_parts(freqs, fp, fc, dtc, hp, hc, h00, a0, a1, b0, b1):
    """Calculate scalar or sample-batched relative likelihood parts.

    Non-scalar ``fp``, ``fc``, and ``dtc`` values are sample batches and gain
    a trailing frequency-broadcast axis.  Use :func:`likelihood_parts_v` when
    those values vary over frequency.
    """
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
    shift = torch.complex(torch.cos(phase), torch.sin(phase))
    ratio = shift * (fp * hp + fc * hc) / h00
    return _summaries(ratio, a0, a1, b0, b1)


def likelihood_parts_v(freqs, fp, fc, dtc, hp, hc, h00, a0, a1, b0, b1):
    """Calculate likelihood parts with an Earth-rotation response.

    ``fp``, ``fc``, and ``dtc`` must be scalars or have shape ``(..., F)``,
    where ``F`` is the number of frequencies.  A single ``(F,)`` response
    therefore produces scalar likelihood parts, while leading dimensions are
    preserved as batch dimensions.
    """
    hp = _torch_tensor(hp)
    if hp is None:
        raise TypeError("a Torch-backed waveform is required")

    real_dtype = hp.real.dtype
    hc = _as_tensor(hc, hp, hp.dtype)
    freqs = _as_tensor(freqs, hp, real_dtype)
    flen = freqs.shape[-1]
    h00 = _as_tensor(h00, hp, hp.dtype)
    fp = _frequency_axis(_as_tensor(fp, hp, real_dtype), flen, "fp")
    fc = _frequency_axis(_as_tensor(fc, hp, real_dtype), flen, "fc")
    dtc = _frequency_axis(_as_tensor(dtc, hp, real_dtype), flen, "dtc")
    a0 = _as_tensor(a0, hp, hp.dtype)
    a1 = _as_tensor(a1, hp, hp.dtype)
    b0 = _as_tensor(b0, hp, real_dtype)
    b1 = _as_tensor(b1, hp, real_dtype)

    phase = -2.0 * _RELBIN_PI * dtc * freqs
    shift = torch.complex(torch.cos(phase), torch.sin(phase))
    ratio = shift * (fp * hp + fc * hc) / h00
    return _summaries(ratio, a0, a1, b0, b1)


def likelihood_parts_vector(freqs, fp, fc, dtc, hp, hc, h00, a0, a1, b0, b1):
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
    shift = torch.complex(torch.cos(phase), torch.sin(phase))
    ratio = shift * (fp * hp + fc * hc) / h00
    return _summaries(ratio, a0, a1, b0, b1)


def _likelihood_parts_v_vector(
    freqs, fp, fc, times, dtc, pol_phase, hp, hc, h00, a0, a1, b0, b1
):
    """Calculate frequency-varying responses for paired samples.

    ``fp``, ``fc``, and ``times`` end in the frequency axis.  ``dtc`` and
    ``pol_phase`` instead use their leading dimensions as sample batches.
    """
    hp = _torch_tensor(hp)
    if hp is None:
        raise TypeError("a Torch-backed waveform is required")

    real_dtype = hp.real.dtype
    hc = _as_tensor(hc, hp, hp.dtype)
    freqs = _as_tensor(freqs, hp, real_dtype)
    flen = freqs.shape[-1]
    h00 = _as_tensor(h00, hp, hp.dtype)
    fp = _frequency_axis(_as_tensor(fp, hp, real_dtype), flen, "fp")
    fc = _frequency_axis(_as_tensor(fc, hp, real_dtype), flen, "fc")
    times = _frequency_axis(_as_tensor(times, hp, real_dtype), flen, "times")
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
    shift = torch.complex(torch.cos(phase), torch.sin(phase))
    ratio = shift * (response.real * hp + response.imag * hc) / h00
    return _summaries(ratio, a0, a1, b0, b1)


def likelihood_parts_v_pol(freqs, fp, fc, dtc, pol_phase, hp, hc, h00, a0, a1, b0, b1):
    """Calculate an Earth-rotation likelihood over polarization samples."""
    return _likelihood_parts_v_vector(
        freqs, fp, fc, dtc, 0.0, pol_phase, hp, hc, h00, a0, a1, b0, b1
    )


def likelihood_parts_v_time(freqs, fp, fc, times, dtc, hp, hc, h00, a0, a1, b0, b1):
    """Calculate an Earth-rotation likelihood over time samples."""
    return _likelihood_parts_v_vector(
        freqs, fp, fc, times, dtc, None, hp, hc, h00, a0, a1, b0, b1
    )


def likelihood_parts_v_pol_time(
    freqs, fp, fc, times, dtc, pol_phase, hp, hc, h00, a0, a1, b0, b1
):
    """Calculate an Earth-rotation likelihood over time and polarization."""
    return _likelihood_parts_v_vector(
        freqs, fp, fc, times, dtc, pol_phase, hp, hc, h00, a0, a1, b0, b1
    )


def likelihood_parts_det(freqs, dtc, channel, h00, a0, a1, b0, b1):
    """Calculate likelihood parts for a detector-frame waveform."""
    channel = _torch_tensor(channel)
    if channel is None:
        raise TypeError("a Torch-backed waveform is required")

    real_dtype = channel.real.dtype
    freqs = _as_tensor(freqs, channel, real_dtype)
    h00 = _as_tensor(h00, channel, channel.dtype)
    dtc = _sample_axis(_as_tensor(dtc, channel, real_dtype))
    a0 = _as_tensor(a0, channel, channel.dtype)
    a1 = _as_tensor(a1, channel, channel.dtype)
    b0 = _as_tensor(b0, channel, real_dtype)
    b1 = _as_tensor(b1, channel, real_dtype)

    phase = -2.0 * _RELBIN_PI * dtc * freqs
    shift = torch.complex(torch.cos(phase), torch.sin(phase))
    ratio = shift * channel / h00
    return _summaries(ratio, a0, a1, b0, b1)


def likelihood_parts_multi(
    freqs,
    fp,
    fc,
    dtc,
    hp,
    hc,
    h00,
    fp2,
    fc2,
    dtc2,
    hp2,
    hc2,
    h002,
    a0,
    a1,
    *,
    _frequency_varying=False,
):
    """Calculate a cross term between two polarization waveforms."""
    hp = _torch_tensor(hp)
    if hp is None:
        raise TypeError("a Torch-backed waveform is required")

    real_dtype = hp.real.dtype
    hc = _as_tensor(hc, hp, hp.dtype)
    hp2 = _as_tensor(hp2, hp, hp.dtype)
    hc2 = _as_tensor(hc2, hp, hp.dtype)
    freqs = _as_tensor(freqs, hp, real_dtype)
    flen = freqs.shape[-1]
    h00 = _as_tensor(h00, hp, hp.dtype)
    h002 = _as_tensor(h002, hp, hp.dtype)
    axis = _frequency_axis if _frequency_varying else _sample_axis
    if _frequency_varying:
        fp = axis(_as_tensor(fp, hp, real_dtype), flen)
        fc = axis(_as_tensor(fc, hp, real_dtype), flen)
        dtc = axis(_as_tensor(dtc, hp, real_dtype), flen)
        fp2 = axis(_as_tensor(fp2, hp, real_dtype), flen)
        fc2 = axis(_as_tensor(fc2, hp, real_dtype), flen)
        dtc2 = axis(_as_tensor(dtc2, hp, real_dtype), flen)
    else:
        fp = axis(_as_tensor(fp, hp, real_dtype))
        fc = axis(_as_tensor(fc, hp, real_dtype))
        dtc = axis(_as_tensor(dtc, hp, real_dtype))
        fp2 = axis(_as_tensor(fp2, hp, real_dtype))
        fc2 = axis(_as_tensor(fc2, hp, real_dtype))
        dtc2 = axis(_as_tensor(dtc2, hp, real_dtype))
    a0 = _as_tensor(a0, hp, hp.dtype)
    a1 = _as_tensor(a1, hp, hp.dtype)

    phase = -2.0 * _RELBIN_PI * dtc * freqs
    phase2 = -2.0 * _RELBIN_PI * dtc2 * freqs
    shift = torch.complex(torch.cos(phase), torch.sin(phase))
    shift2 = torch.complex(torch.cos(phase2), torch.sin(phase2))
    ratio = shift * (fp * hp + fc * hc) / h00
    ratio2 = shift2 * (fp2 * hp2 + fc2 * hc2) / h002
    return _linearized_cross(ratio, ratio2, a0, a1)


def likelihood_parts_multi_v(
    freqs, fp, fc, dtc, hp, hc, h00, fp2, fc2, dtc2, hp2, hc2, h002, a0, a1
):
    """Calculate a cross term with frequency-varying responses."""
    return likelihood_parts_multi(
        freqs,
        fp,
        fc,
        dtc,
        hp,
        hc,
        h00,
        fp2,
        fc2,
        dtc2,
        hp2,
        hc2,
        h002,
        a0,
        a1,
        _frequency_varying=True,
    )


def likelihood_parts_det_multi(freqs, dtc, channel, h00, dtc2, channel2, h002, a0, a1):
    """Calculate a detector-frame cross term between two waveforms."""
    channel = _torch_tensor(channel)
    if channel is None:
        raise TypeError("a Torch-backed waveform is required")

    real_dtype = channel.real.dtype
    channel2 = _as_tensor(channel2, channel, channel.dtype)
    freqs = _as_tensor(freqs, channel, real_dtype)
    h00 = _as_tensor(h00, channel, channel.dtype)
    h002 = _as_tensor(h002, channel, channel.dtype)
    dtc = _sample_axis(_as_tensor(dtc, channel, real_dtype))
    dtc2 = _sample_axis(_as_tensor(dtc2, channel, real_dtype))
    a0 = _as_tensor(a0, channel, channel.dtype)
    a1 = _as_tensor(a1, channel, channel.dtype)

    phase = -2.0 * _RELBIN_PI * dtc * freqs
    phase2 = -2.0 * _RELBIN_PI * dtc2 * freqs
    shift = torch.complex(torch.cos(phase), torch.sin(phase))
    shift2 = torch.complex(torch.cos(phase2), torch.sin(phase2))
    ratio = shift * channel / h00
    ratio2 = shift2 * channel2 / h002
    # Preserve the established detector-frame kernel's argument order.
    return _linearized_cross(ratio2, ratio, a0, a1)


def _time_shifted_filters(freqs, tstart, delta_t, num_samples, ratio, a0, a1):
    """Evaluate relative-bin filters over a blocked uniform time grid."""
    num_samples = int(num_samples)
    if num_samples == 0:
        return ratio.new_empty((0,))

    block_size = max(1, _SNR_PREDICTOR_TARGET_ELEMENTS // max(1, freqs.numel()))
    filters = []
    for start in range(0, num_samples, block_size):
        stop = min(start + block_size, num_samples)
        sample_indices = torch.arange(
            start, stop, device=ratio.device, dtype=freqs.dtype
        )
        times = tstart + delta_t * sample_indices
        phase = -2.0 * _RELBIN_PI * times.unsqueeze(-1) * freqs
        shift = torch.complex(torch.cos(phase), torch.sin(phase))
        filters.append(_linearized_filter(shift * ratio, a0, a1))
    return torch.cat(filters)


def snr_predictor(freqs, tstart, delta_t, num_samples, hp, hc, h00, a0, a1, b0, b1):
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
    sh = _time_shifted_filters(freqs, tstart, delta_t, num_samples, hp_ratio, a0, a1)
    csh = _time_shifted_filters(freqs, tstart, delta_t, num_samples, hc_ratio, a0, a1)
    snr2 = (sh.real.square() + sh.imag.square()) / (2.0 * hh) + (
        csh.real.square() + csh.imag.square()
    ) / (2.0 * chh)
    return torch.sqrt(snr2)


def snr_predictor_dom(freqs, tstart, delta_t, num_samples, hp, h00, a0, a1, b0, b1):
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
    sh = _time_shifted_filters(freqs, tstart, delta_t, num_samples, ratio, a0, a1)
    return sh, hh
