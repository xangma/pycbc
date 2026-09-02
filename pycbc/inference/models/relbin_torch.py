"""Torch kernels for the relative-binning likelihood."""

import numpy
import torch


# Match the constant used by the established Cython kernels exactly.
_RELBIN_PI = 3.141592653
_SNR_PREDICTOR_TARGET_ELEMENTS = 2 ** 20
_LIGHT_SPEED_SI = 299792458.0
_EXPLICIT_RESPONSE_MIN_THREADS = 64
_EXPLICIT_RESPONSE_MAX_VECTORS = 4096


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


def _detector_response_constants(detector, like):
    """Cache immutable detector tensors beside a likelihood batch."""
    key = (like.device.type, like.device.index, like.real.dtype)
    cache = getattr(detector, "_torch_response_constants", None)
    if cache is None:
        cache = {}
        detector._torch_response_constants = cache
    if key not in cache:
        response = torch.as_tensor(
            detector.response,
            device=like.device,
            dtype=like.real.dtype,
        )
        delay_vector = torch.as_tensor(
            detector.location,
            device=like.device,
            dtype=like.real.dtype,
        ).neg().div(_LIGHT_SPEED_SI)
        cache[key] = response, delay_vector
    return cache[key]


def _contract_detector_response(response, basis):
    """Apply a detector's 3-by-3 response without tiny CPU BLAS work."""
    # On highly threaded CPU jobs, einsum lowers this small contraction to a
    # batched matrix multiply whose thread-team launch can cost far more than
    # its nine multiplies.  Keep large batches and accelerator devices on the
    # established einsum kernel.
    if (
        basis.device.type == "cpu"
        and torch.get_num_threads() >= _EXPLICIT_RESPONSE_MIN_THREADS
        and basis[0].numel() <= _EXPLICIT_RESPONSE_MAX_VECTORS
    ):
        return torch.stack(tuple(
            response[row, 0] * basis[0]
            + response[row, 1] * basis[1]
            + response[row, 2] * basis[2]
            for row in range(3)
        ))
    return torch.einsum("ij,j...->i...", response, basis)


def _explicit_antenna_response(response, x0, x1, y0, y1, y2):
    """Contract the two polarization bases without materializing 3-vectors.

    The retained-grid inference path commonly evaluates only a few thousand
    sky positions.  At high CPU thread counts, the six stacks/reductions in
    the generic tensor formulation cost more than this fixed 3-by-3
    contraction.  Spell out the same row-wise arithmetic while retaining the
    full (not necessarily symmetric) response matrix so detector-like test
    objects and autograd keep their established semantics.
    """
    zero = torch.zeros_like(x0)
    dx0 = (
        response[0, 0] * x0
        + response[0, 1] * x1
        + response[0, 2] * zero
    )
    dx1 = (
        response[1, 0] * x0
        + response[1, 1] * x1
        + response[1, 2] * zero
    )
    dx2 = (
        response[2, 0] * x0
        + response[2, 1] * x1
        + response[2, 2] * zero
    )
    dy0 = (
        response[0, 0] * y0
        + response[0, 1] * y1
        + response[0, 2] * y2
    )
    dy1 = (
        response[1, 0] * y0
        + response[1, 1] * y1
        + response[1, 2] * y2
    )
    dy2 = (
        response[2, 0] * y0
        + response[2, 1] * y1
        + response[2, 2] * y2
    )
    # Preserve the established row-wise reduction order exactly.  Keeping
    # only these two final stacks still avoids materializing both bases and
    # both contracted three-vectors.
    fplus = torch.sum(torch.stack((
        x0 * dx0 - y0 * dy0,
        x1 * dx1 - y1 * dy1,
        zero * dx2 - y2 * dy2,
    )), dim=0)
    fcross = torch.sum(torch.stack((
        x0 * dy0 + y0 * dx0,
        x1 * dy1 + y1 * dx1,
        zero * dy2 + y2 * dx2,
    )), dim=0)
    return fplus, fcross


def _zero_polarization_response_and_delay(
        detector, right_ascension, declination, times, like):
    """Fuse the zero-polarization antenna and geocentric-delay kernels."""
    dtype = like.real.dtype
    right_ascension = _as_tensor(right_ascension, like, dtype)
    declination = _as_tensor(declination, like, dtype)

    time_tensor = _torch_tensor(times)
    time_is_array = time_tensor is None and numpy.ndim(times) > 0
    if time_tensor is not None or time_is_array:
        if detector.reference_time is None:
            raise NotImplementedError(
                "Torch GPS-time grids require a detector GMST reference time"
            )
        if detector.gmst_reference is None:
            detector.set_gmst_reference()
        if time_tensor is not None:
            relative_time = _as_tensor(time_tensor, like, dtype)
            relative_time = relative_time - float(detector.reference_time)
        else:
            # Center on the host before uploading to preserve fractional GPS
            # seconds on float32-only devices.
            relative_time = _as_tensor(
                numpy.asarray(times, dtype=numpy.float64)
                - float(detector.reference_time),
                like,
                dtype,
            )
        # Static sky positions are the common batched-likelihood case.  Keep
        # them scalar so their trigonometry is not needlessly evaluated once
        # per time sample.  Tensor broadcasting below still returns the full
        # time-grid shape, with gradients accumulating back to the scalars.
        if (
            like.device.type != "cpu"
            or right_ascension.ndim
            or declination.ndim
        ):
            right_ascension, declination, relative_time = (
                torch.broadcast_tensors(
                    right_ascension, declination, relative_time
                )
            )
        phase_offsets = (
            relative_time / float(detector.sday) * (2.0 * torch.pi)
        )
        gmst_start = detector.gmst_reference
    else:
        right_ascension, declination = torch.broadcast_tensors(
            right_ascension, declination
        )
        phase_offsets = torch.zeros_like(right_ascension)
        gmst_start = detector.gmst_estimate(times)

    # Use angle addition rather than adding the small sidereal offset to the
    # absolute angle. This retains sub-second changes in float32.
    gha_start = torch.as_tensor(
        gmst_start, device=like.device, dtype=dtype
    ) - right_ascension
    cos_start = torch.cos(gha_start)
    sin_start = torch.sin(gha_start)
    cos_offset = torch.cos(phase_offsets)
    sin_offset = torch.sin(phase_offsets)
    cosgha = cos_start * cos_offset - sin_start * sin_offset
    singha = sin_start * cos_offset + cos_start * sin_offset
    cosdec = torch.cos(declination)
    sindec = torch.sin(declination)

    # These are the polarization-basis vectors for psi=0. Reusing their sky
    # trigonometry for the delay avoids evaluating the geometry twice.
    response, delay_vector = _detector_response_constants(detector, like)
    x0 = -singha
    x1 = -cosgha
    y0 = -cosgha * sindec
    y1 = singha * sindec
    y2 = cosdec.expand_as(cosgha)
    if (
        like.device.type == "cpu"
        and torch.get_num_threads() >= _EXPLICIT_RESPONSE_MIN_THREADS
        and cosgha.numel() <= _EXPLICIT_RESPONSE_MAX_VECTORS
    ):
        fplus, fcross = _explicit_antenna_response(
            response, x0, x1, y0, y1, y2
        )
    else:
        x = torch.stack((x0, x1, torch.zeros_like(x0)))
        y = torch.stack((y0, y1, y2))
        dx = _contract_detector_response(response, x)
        dy = _contract_detector_response(response, y)
        fplus = torch.sum(x * dx - y * dy, dim=0)
        fcross = torch.sum(x * dy + y * dx, dim=0)
    delay = (
        delay_vector[0] * cosdec * cosgha
        - delay_vector[1] * cosdec * singha
        + delay_vector[2] * sindec
    )
    return fplus, fcross, delay


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

    # Real Detector instances expose the immutable geometry needed by a
    # fused kernel. Lightweight detector-like objects used by downstream
    # callers retain the established public-method fallback below.
    if all(hasattr(detector, name) for name in (
            "response", "location", "sday", "gmst_estimate")):
        return _zero_polarization_response_and_delay(
            detector, right_ascension, declination, times, like
        )

    dtype = like.real.dtype
    right_ascension = _as_tensor(right_ascension, like, dtype)
    declination = _as_tensor(declination, like, dtype)
    polarization = torch.zeros((), device=like.device, dtype=dtype)
    fp, fc = detector.antenna_pattern(
        right_ascension, declination, polarization, times
    )
    delay = detector.time_delay_from_earth_center(
        right_ascension, declination, times
    )
    return tuple(
        _as_tensor(value, like, dtype) for value in (fp, fc, delay)
    )


def detector_response_at_arrival(
        detector, reference_time, right_ascension, declination,
        polarization, reference_frame, like):
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
    return tuple(
        _as_tensor(value, like, dtype)
        for value in (fp, fc, arrival_time)
    )


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


def dominant_mode_template_factor(
        fp, fc, polarization, inclination, coa_phase, distance, like):
    """Build a full dominant-mode extrinsic factor beside a Torch sample."""
    like = _torch_tensor(like)
    if like is None:
        raise TypeError("a Torch-backed likelihood sample is required")

    projection = dominant_mode_projection(
        fp, fc, polarization, inclination, like)
    angle = _as_tensor(coa_phase, like, like.real.dtype)
    phase = torch.polar(torch.ones_like(angle), -2.0 * angle)
    distance = _as_tensor(distance, like, like.real.dtype)
    return projection * phase / distance


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
    return value._return(data) if hasattr(value, "_return") else Array(
        data, copy=False)


def prepare_reference_data(
        waveform, data, size, offset, delta_f, time_shift):
    """Pad, place, and time-shift relative-bin inputs on a Torch device."""
    like = next(
        (tensor for value in (waveform, data)
         if (tensor := _torch_tensor(value)) is not None),
        None,
    )
    if like is None:
        raise TypeError("a Torch-backed waveform or data series is required")

    real_dtype = like.real.dtype
    complex_dtype = (
        like.dtype if like.is_complex()
        else torch.complex128 if real_dtype == torch.float64
        else torch.complex64
    )
    waveform_tensor = _as_tensor(waveform, like, complex_dtype).reshape(-1)
    data_tensor = _as_tensor(data, like, complex_dtype).reshape(-1)

    reference = torch.zeros(size, dtype=complex_dtype, device=like.device)
    copied = min(size, waveform_tensor.numel())
    reference[:copied] = waveform_tensor[:copied]
    reference = torch.roll(reference, shifts=int(offset))

    frequencies = (
        torch.arange(size, dtype=real_dtype, device=like.device) * delta_f
    )
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
        (tensor for value in (h1, h2, freqs)
         if (tensor := _torch_tensor(value)) is not None),
        None,
    )
    if like is None:
        raise TypeError(
            "a Torch-backed waveform or frequency grid is required"
        )

    real_dtype = like.real.dtype
    complex_dtype = (
        like.dtype if like.is_complex()
        else torch.complex128 if real_dtype == torch.float64
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
        (tensor for value in (h1, h2, psd)
         if (tensor := _torch_tensor(value)) is not None),
        None,
    )
    if like is None:

        raise TypeError("a Torch-backed waveform or PSD is required")

    real_dtype = like.real.dtype
    complex_dtype = (
        like.dtype if like.is_complex()
        else torch.complex128 if real_dtype == torch.float64
        else torch.complex64
    )
    h1 = _as_tensor(h1, like, complex_dtype)
    h2 = _as_tensor(h2, like, complex_dtype)
    psd = _as_tensor(psd, like, real_dtype)
    freqs = _as_tensor(freqs, like, real_dtype)
    bins = _as_tensor(bins, like, torch.int64).reshape(-1, 2)
    delta_f = _as_tensor(delta_f, like, real_dtype)

    h12 = h1.conj() * h2 / psd
    zero = torch.zeros(
        h12.shape[:-1] + (1,), dtype=h12.dtype, device=h12.device
    )
    prefix = torch.cat((zero, _complex_cumsum(h12, dim=-1)), dim=-1)
    weighted_prefix = torch.cat(
        (zero, _complex_cumsum(h12 * freqs, dim=-1)), dim=-1
    )

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


def _sample_axis(value, freqs_len=None):
    """Add a trailing frequency axis to sample parameters if needed."""
    if not hasattr(value, "ndim") or value.ndim == 0:
        return value
    if freqs_len is not None and value.shape[-1] == freqs_len:
        return value
    return value.unsqueeze(-1)


def likelihood_parts(freqs, fp, fc, dtc, hp, hc, h00,
                     a0, a1, b0, b1):
    """Calculate scalar or batched relative likelihood parts."""
    hp = _torch_tensor(hp)
    if hp is None:
        raise TypeError("a Torch-backed waveform is required")

    real_dtype = hp.real.dtype
    hc = _as_tensor(hc, hp, hp.dtype)
    freqs = _as_tensor(freqs, hp, real_dtype)
    flen = freqs.shape[-1]
    h00 = _as_tensor(h00, hp, hp.dtype)
    fp = _sample_axis(_as_tensor(fp, hp, real_dtype), flen)
    fc = _sample_axis(_as_tensor(fc, hp, real_dtype), flen)
    dtc = _sample_axis(_as_tensor(dtc, hp, real_dtype), flen)
    a0 = _as_tensor(a0, hp, hp.dtype)
    a1 = _as_tensor(a1, hp, hp.dtype)
    b0 = _as_tensor(b0, hp, real_dtype)
    b1 = _as_tensor(b1, hp, real_dtype)

    phase = -2.0 * _RELBIN_PI * dtc * freqs
    shift = torch.complex(torch.cos(phase), torch.sin(phase))
    ratio = shift * (fp * hp + fc * hc) / h00
    return _summaries(ratio, a0, a1, b0, b1)


def batched_likelihood_parts(freqs, fp, fc, dtc, hp, hc, h00,
                             a0, a1, b0, b1, *, use_vmap=False):
    """Evaluate relative-binning likelihood parts across N sample points.

    Supports both broadcasted 3D tensor evaluation and torch.vmap
    vectorization.

    Parameters
    ----------
    freqs : torch.Tensor
        Frequency grid (F,).
    fp, fc, dtc : torch.Tensor
        Sample parameter tensors of shape (N,) or (B, N).
    hp, hc, h00 : torch.Tensor
        Waveform frequency series of shape (F,) or (B, N, F).
    a0, a1, b0, b1 : torch.Tensor
        Summary bin coefficients.
    use_vmap : bool, optional
        If True, evaluates the batch via torch.vmap vectorization.
        If False (default), evaluates via broadcasted tensor operations.


    Returns
    -------
    filt : torch.Tensor
        Linearized data inner product of shape (N,) or (B, N).
    norm : torch.Tensor
        Linearized waveform norm of shape (N,) or (B, N).
    """
    if use_vmap:
        v_fn = torch.vmap(
            lambda p, c, t: likelihood_parts(
                freqs, p, c, t, hp, hc, h00, a0, a1, b0, b1
            )
        )
        return v_fn(fp, fc, dtc)
    return likelihood_parts(
        freqs, fp, fc, dtc, hp, hc, h00, a0, a1, b0, b1
    )


def likelihood_parts_vector(freqs, fp, fc, dtc, hp, hc, h00,
                            a0, a1, b0, b1):
    """Calculate likelihood parts for paired sky, time, or pol samples."""
    hp = _torch_tensor(hp)
    if hp is None:
        raise TypeError("a Torch-backed waveform is required")

    real_dtype = hp.real.dtype
    hc = _as_tensor(hc, hp, hp.dtype)
    freqs = _as_tensor(freqs, hp, real_dtype)
    flen = freqs.shape[-1]
    h00 = _as_tensor(h00, hp, hp.dtype)
    fp = _sample_axis(_as_tensor(fp, hp, real_dtype), flen)
    fc = _sample_axis(_as_tensor(fc, hp, real_dtype), flen)
    dtc = _sample_axis(_as_tensor(dtc, hp, real_dtype), flen)
    a0 = _as_tensor(a0, hp, hp.dtype)
    a1 = _as_tensor(a1, hp, hp.dtype)
    b0 = _as_tensor(b0, hp, real_dtype)
    b1 = _as_tensor(b1, hp, real_dtype)

    phase = -2.0 * _RELBIN_PI * dtc * freqs
    shift = torch.complex(torch.cos(phase), torch.sin(phase))
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
    flen = freqs.shape[-1]
    h00 = _as_tensor(h00, hp, hp.dtype)
    fp = _sample_axis(_as_tensor(fp, hp, real_dtype), flen)
    fc = _sample_axis(_as_tensor(fc, hp, real_dtype), flen)
    times = _sample_axis(_as_tensor(times, hp, real_dtype), flen)
    dtc = _sample_axis(_as_tensor(dtc, hp, real_dtype), flen)
    a0 = _as_tensor(a0, hp, hp.dtype)
    a1 = _as_tensor(a1, hp, hp.dtype)
    b0 = _as_tensor(b0, hp, real_dtype)
    b1 = _as_tensor(b1, hp, real_dtype)

    response = fp + 1.0j * fc
    if pol_phase is not None:
        pol_phase = _sample_axis(_as_tensor(pol_phase, hp, hp.dtype), flen)
        response = response * pol_phase

    phase = -2.0 * _RELBIN_PI * (times + dtc) * freqs
    shift = torch.complex(torch.cos(phase), torch.sin(phase))
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
    flen = freqs.shape[-1]
    h00 = _as_tensor(h00, channel, channel.dtype)
    dtc = _sample_axis(_as_tensor(dtc, channel, real_dtype), flen)
    a0 = _as_tensor(a0, channel, channel.dtype)
    a1 = _as_tensor(a1, channel, channel.dtype)
    b0 = _as_tensor(b0, channel, real_dtype)
    b1 = _as_tensor(b1, channel, real_dtype)

    phase = -2.0 * _RELBIN_PI * dtc * freqs
    shift = torch.complex(torch.cos(phase), torch.sin(phase))
    ratio = shift * channel / h00
    return _summaries(ratio, a0, a1, b0, b1)


def likelihood_parts_multi(freqs, fp, fc, dtc, hp, hc, h00,
                           fp2, fc2, dtc2, hp2, hc2, h002, a0, a1):
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
    fp = _sample_axis(_as_tensor(fp, hp, real_dtype), flen)
    fc = _sample_axis(_as_tensor(fc, hp, real_dtype), flen)
    dtc = _sample_axis(_as_tensor(dtc, hp, real_dtype), flen)
    fp2 = _sample_axis(_as_tensor(fp2, hp, real_dtype), flen)
    fc2 = _sample_axis(_as_tensor(fc2, hp, real_dtype), flen)
    dtc2 = _sample_axis(_as_tensor(dtc2, hp, real_dtype), flen)
    a0 = _as_tensor(a0, hp, hp.dtype)
    a1 = _as_tensor(a1, hp, hp.dtype)

    phase = -2.0 * _RELBIN_PI * dtc * freqs
    phase2 = -2.0 * _RELBIN_PI * dtc2 * freqs
    shift = torch.complex(torch.cos(phase), torch.sin(phase))
    shift2 = torch.complex(torch.cos(phase2), torch.sin(phase2))
    ratio = shift * (fp * hp + fc * hc) / h00
    ratio2 = shift2 * (fp2 * hp2 + fc2 * hc2) / h002
    return _linearized_cross(ratio, ratio2, a0, a1)


def likelihood_parts_multi_v(freqs, fp, fc, dtc, hp, hc, h00,
                             fp2, fc2, dtc2, hp2, hc2, h002, a0, a1):
    """Calculate a cross term with frequency-varying responses."""
    return likelihood_parts_multi(
        freqs, fp, fc, dtc, hp, hc, h00,
        fp2, fc2, dtc2, hp2, hc2, h002, a0, a1)


def likelihood_parts_det_multi(freqs, dtc, channel, h00,
                               dtc2, channel2, h002, a0, a1):
    """Calculate a detector-frame cross term between two waveforms."""
    channel = _torch_tensor(channel)
    if channel is None:
        raise TypeError("a Torch-backed waveform is required")

    real_dtype = channel.real.dtype
    channel2 = _as_tensor(channel2, channel, channel.dtype)
    freqs = _as_tensor(freqs, channel, real_dtype)
    flen = freqs.shape[-1]
    h00 = _as_tensor(h00, channel, channel.dtype)
    h002 = _as_tensor(h002, channel, channel.dtype)
    dtc = _sample_axis(_as_tensor(dtc, channel, real_dtype), flen)
    dtc2 = _sample_axis(_as_tensor(dtc2, channel, real_dtype), flen)
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
        shift = torch.complex(torch.cos(phase), torch.sin(phase))
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
