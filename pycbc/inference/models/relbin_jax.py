"""JAX kernels for the relative-binning likelihood."""

import jax.numpy as jnp

from pycbc.types.backend import backend_array

# Match the constant used by the established Cython kernels exactly.
_RELBIN_PI = 3.141592653
_SNR_PREDICTOR_TARGET_ELEMENTS = 2**20


def _jax_array(value):
    """Return JAX storage through the public backend protocol."""
    if isinstance(value, (int, float, bool)):
        return None
    return backend_array(value, "jax")


def _as_array(value, like, dtype):
    """Convert ``value`` to JAX array with ``dtype``."""
    if isinstance(value, (int, float, bool)):
        return jnp.asarray(value, dtype=dtype)
    arr = _jax_array(value)
    if arr is None:
        arr = value
    return jnp.asarray(arr, dtype=dtype)


def detector_response(detector, right_ascension, declination, times, like):
    """Evaluate zero-polarization detector factors beside a waveform.

    Converting the sky coordinates before calling the detector selects its
    JAX antenna and timing kernels. GPS times remain in their original
    representation so float32-only devices do not lose precision from the
    large absolute epoch before the detector forms sidereal angles.
    """
    like = _jax_array(like)
    if like is None:
        raise TypeError("a JAX-backed waveform is required")

    dtype = like.real.dtype
    right_ascension = _as_array(right_ascension, like, dtype)
    declination = _as_array(declination, like, dtype)
    polarization = jnp.zeros((), dtype=dtype)
    time_arr = _jax_array(times)
    if time_arr is not None:
        times = time_arr
    combined = getattr(detector, "antenna_pattern_and_time_delay", None)
    if combined is not None:
        response = combined(right_ascension, declination, polarization, times)
        return tuple(_as_array(value, like, dtype) for value in response)

    fp, fc = detector.antenna_pattern(right_ascension, declination, polarization, times)
    delay = detector.time_delay_from_earth_center(right_ascension, declination, times)
    return tuple(_as_array(value, like, dtype) for value in (fp, fc, delay))


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
    calling the detector, selecting its JAX timing and antenna kernels.
    The detector's reference-frame handling remains authoritative.
    """
    like = _jax_array(like)
    if like is None:
        raise TypeError("a JAX-backed waveform is required")

    dtype = like.real.dtype
    right_ascension = _as_array(right_ascension, like, dtype)
    declination = _as_array(declination, like, dtype)
    polarization = _as_array(polarization, like, dtype)
    arrival_time = detector.arrival_time(
        reference_time, right_ascension, declination, reference_frame
    )
    fp, fc = detector.antenna_pattern(
        right_ascension, declination, polarization, arrival_time
    )
    return tuple(_as_array(value, like, dtype) for value in (fp, fc, arrival_time))


def polarization_phase(polarization, like):
    """Build the spin-2 polarization phase beside a JAX waveform."""
    like = _jax_array(like)
    if like is None:
        raise TypeError("a JAX-backed waveform is required")

    angle = _as_array(polarization, like, like.real.dtype)
    two_psi = 2.0 * angle
    return jnp.cos(two_psi) - 1j * jnp.sin(two_psi)


def polarized_antenna_response(fp, fc, pol_phase, like):
    """Rotate zero-polarization antenna factors on a JAX device."""
    like = _jax_array(like)
    if like is None:
        raise TypeError("a JAX-backed waveform is required")

    real_dtype = like.real.dtype
    fp = _as_array(fp, like, real_dtype)
    fc = _as_array(fc, like, real_dtype)
    if jnp.issubdtype(jnp.asarray(pol_phase).dtype, jnp.complexfloating):
        pol_phase = _as_array(pol_phase, like, like.dtype)
        pr = pol_phase.real
        pi = pol_phase.imag
    else:
        pol_phase = _as_array(pol_phase, like, real_dtype)
        two_psi = 2.0 * pol_phase
        pr = jnp.cos(two_psi)
        pi = -jnp.sin(two_psi)
    return fp * pr - fc * pi, fp * pi + fc * pr


def dominant_mode_projection(fp, fc, polarization, inclination, like):
    """Project a dominant-mode response beside a JAX likelihood sample."""
    like = _jax_array(like)
    if like is None:
        raise TypeError("a JAX-backed likelihood sample is required")

    fp_rot, fc_rot = polarized_antenna_response(fp, fc, polarization, like)
    angle = _as_array(inclination, like, like.real.dtype)
    cosi = jnp.cos(angle)
    plus = 0.5 * (1.0 + jnp.square(cosi))
    return (fp_rot * plus) + 1j * (fc_rot * cosi)


def _wrap_like(value, arr):
    """Wrap ``arr`` in the same PyCBC container family as ``value``."""
    from pycbc.types import Array
    from pycbc.types.array_jax import JaxArrayData

    data = JaxArrayData(arr)
    return value._return(data) if hasattr(value, "_return") else Array(data, copy=False)


def prepare_reference_data(waveform, data, size, offset, delta_f, time_shift):
    """Pad, place, and time-shift relative-bin inputs on a JAX device."""
    like = next(
        (
            arr
            for value in (waveform, data)
            if (arr := _jax_array(value)) is not None
        ),
        None,
    )
    if like is None:
        raise TypeError("a JAX-backed waveform or data series is required")

    real_dtype = like.real.dtype
    complex_dtype = (
        like.dtype
        if jnp.issubdtype(like.dtype, jnp.complexfloating)
        else (jnp.complex128 if real_dtype == jnp.float64 else jnp.complex64)
    )
    waveform_arr = _as_array(waveform, like, complex_dtype).reshape(-1)
    data_arr = _as_array(data, like, complex_dtype).reshape(-1)

    reference = jnp.zeros(size, dtype=complex_dtype)
    copied = min(size, waveform_arr.size)
    reference = reference.at[:copied].set(waveform_arr[:copied])
    reference = jnp.roll(reference, shift=int(offset))

    frequencies = jnp.arange(size, dtype=real_dtype) * delta_f
    phase = -2.0 * jnp.pi * frequencies * time_shift
    cos_p = jnp.cos(phase)
    sin_p = jnp.sin(phase)
    d_r = (
        data_arr.real
        if jnp.issubdtype(data_arr.dtype, jnp.complexfloating)
        else data_arr
    )
    d_i = (
        data_arr.imag
        if jnp.issubdtype(data_arr.dtype, jnp.complexfloating)
        else jnp.zeros_like(d_r)
    )
    shifted_data = (d_r * cos_p + d_i * sin_p) + 1j * (d_i * cos_p - d_r * sin_p)
    return (
        _wrap_like(waveform, reference),
        _wrap_like(data, shifted_data),
    )


def active_edge_bins(h1, h2, freqs, edges):
    """Filter a shared edge grid and build its bins on a JAX device."""
    like = next(
        (
            arr
            for value in (h1, h2, freqs)
            if (arr := _jax_array(value)) is not None
        ),
        None,
    )
    if like is None:
        raise TypeError("a JAX-backed waveform or frequency grid is required")

    real_dtype = like.real.dtype
    complex_dtype = (
        like.dtype
        if jnp.issubdtype(like.dtype, jnp.complexfloating)
        else (jnp.complex128 if real_dtype == jnp.float64 else jnp.complex64)
    )
    h1 = _as_array(h1, like, complex_dtype)
    h2 = _as_array(h2, like, complex_dtype)
    freqs = _as_array(freqs, like, real_dtype)
    edges = _as_array(edges, like, jnp.int64).reshape(-1)

    active = (h1[edges] != 0) | (h2[edges] != 0)
    edges = edges[active]
    bins = jnp.stack((edges[:-1], edges[1:]), axis=1)
    return bins, freqs[edges]


def summary_product(h1, h2, psd, freqs, bins, delta_f):
    """Calculate relative-binning coefficients on a JAX device.

    Supports 1D frequency series, 2D batched waveforms (N, F),
    or 3D tensors (B, N, F).
    """
    like = next(
        (
            arr
            for value in (h1, h2, psd)
            if (arr := _jax_array(value)) is not None
        ),
        None,
    )
    if like is None:
        raise TypeError("a JAX-backed waveform or PSD is required")

    real_dtype = like.real.dtype
    complex_dtype = (
        like.dtype
        if jnp.issubdtype(like.dtype, jnp.complexfloating)
        else (jnp.complex128 if real_dtype == jnp.float64 else jnp.complex64)
    )
    h1 = _as_array(h1, like, complex_dtype)
    h2 = _as_array(h2, like, complex_dtype)
    psd = _as_array(psd, like, real_dtype)
    freqs = _as_array(freqs, like, real_dtype)
    bins = _as_array(bins, like, jnp.int64).reshape(-1, 2)
    delta_f = _as_array(delta_f, like, real_dtype)

    h1_r = h1.real if jnp.issubdtype(h1.dtype, jnp.complexfloating) else h1
    h1_i = h1.imag if jnp.issubdtype(h1.dtype, jnp.complexfloating) else 0.0
    h2_r = h2.real if jnp.issubdtype(h2.dtype, jnp.complexfloating) else h2
    h2_i = h2.imag if jnp.issubdtype(h2.dtype, jnp.complexfloating) else 0.0
    inv_psd = 1.0 / psd
    h12_r = (h1_r * h2_r + h1_i * h2_i) * inv_psd
    h12_i = (h1_r * h2_i - h1_i * h2_r) * inv_psd

    zero = jnp.zeros(h12_r.shape[:-1] + (1,), dtype=real_dtype)
    prefix_r = jnp.concatenate((zero, jnp.cumsum(h12_r, axis=-1)), axis=-1)
    prefix_i = jnp.concatenate((zero, jnp.cumsum(h12_i, axis=-1)), axis=-1)
    weighted_prefix_r = jnp.concatenate((zero, jnp.cumsum(h12_r * freqs, axis=-1)), axis=-1)
    weighted_prefix_i = jnp.concatenate((zero, jnp.cumsum(h12_i * freqs, axis=-1)), axis=-1)

    low = bins[:, 0]
    high = bins[:, 1]
    totals_r = prefix_r[..., high] - prefix_r[..., low]
    totals_i = prefix_i[..., high] - prefix_i[..., low]
    weighted_totals_r = weighted_prefix_r[..., high] - weighted_prefix_r[..., low]
    weighted_totals_i = weighted_prefix_i[..., high] - weighted_prefix_i[..., low]
    widths = (high - low).astype(real_dtype)
    scale = 4.0 * delta_f
    a0_r = scale * totals_r
    a0_i = scale * totals_i
    f_low = freqs[low]
    a1_r = 4.0 * (weighted_totals_r - f_low * totals_r) / widths
    a1_i = 4.0 * (weighted_totals_i - f_low * totals_i) / widths
    a0 = a0_r + 1j * a0_i
    a1 = a1_r + 1j * a1_i
    return a0, a1


def prepare_likelihood_data(like, freqs, h00, a0, a1, b0, b1):
    """Prepare static relative-binning data on a waveform's device."""
    like = _jax_array(like)
    if like is None:
        raise TypeError("a JAX-backed waveform is required")

    real_dtype = like.real.dtype
    complex_dtype = like.dtype
    return (
        _as_array(freqs, like, real_dtype),
        _as_array(h00, like, complex_dtype),
        _as_array(a0, like, complex_dtype),
        _as_array(a1, like, complex_dtype),
        _as_array(b0, like, real_dtype),
        _as_array(b1, like, real_dtype),
    )


def prepare_multi_likelihood_data(like, freqs, h00, h002, a0, a1):
    """Prepare static multi-signal summary data beside a waveform."""
    like = _jax_array(like)
    if like is None:
        raise TypeError("a JAX-backed waveform is required")

    real_dtype = like.real.dtype
    complex_dtype = like.dtype
    return (
        _as_array(freqs, like, real_dtype),
        _as_array(h00, like, complex_dtype),
        _as_array(h002, like, complex_dtype),
        _as_array(a0, like, complex_dtype),
        _as_array(a1, like, complex_dtype),
    )


def _cmul(ar, ai, br, bi):
    """Multiply complex pairs in Cartesian coordinates."""
    return ar * br - ai * bi, ar * bi + ai * br


def _cdiv(ar, ai, br, bi):
    """Divide complex pairs in Cartesian coordinates."""
    denom = br * br + bi * bi
    return (ar * br + ai * bi) / denom, (ai * br - ar * bi) / denom


def _linearized_filter_cartesian(ratio_r, ratio_i, a0_r, a0_i, a1_r, a1_i):
    """Calculate the linearized data-waveform inner product in Cartesian form."""
    da_r = a0_r - a1_r
    da_i = a0_i - a1_i
    r_lo_r, r_lo_i = ratio_r[..., :-1], ratio_i[..., :-1]
    r_hi_r, r_hi_i = ratio_r[..., 1:], ratio_i[..., 1:]
    t1_r, t1_i = _cmul(da_r, da_i, r_lo_r, r_lo_i)
    t2_r, t2_i = _cmul(a1_r, a1_i, r_hi_r, r_hi_i)
    filt_r = (t1_r + t2_r).sum(axis=-1)
    filt_i = -(t1_i + t2_i).sum(axis=-1)
    return filt_r + 1j * filt_i


def _linearized_norm_cartesian(ratio_r, ratio_i, b0, b1):
    """Calculate the linearized waveform norm in Cartesian form."""
    power = jnp.square(ratio_r) + jnp.square(ratio_i)
    power_lo = power[..., :-1]
    power_hi = power[..., 1:]
    return ((b0 - b1) * power_lo + b1 * power_hi).sum(axis=-1)


def _linearized_cross_cartesian(
    ratio_r, ratio_i, ratio2_r, ratio2_i, a0_r, a0_i, a1_r, a1_i
):
    """Calculate a linearized cross term in Cartesian form."""
    cross_r = ratio_r * ratio2_r + ratio_i * ratio2_i
    cross_i = ratio_i * ratio2_r - ratio_r * ratio2_i
    da_r = a0_r - a1_r
    da_i = a0_i - a1_i
    c_lo_r, c_lo_i = cross_r[..., :-1], cross_i[..., :-1]
    c_hi_r, c_hi_i = cross_r[..., 1:], cross_i[..., 1:]
    t1_r, t1_i = _cmul(da_r, da_i, c_lo_r, c_lo_i)
    t2_r, t2_i = _cmul(a1_r, a1_i, c_hi_r, c_hi_i)
    return (t1_r + t2_r).sum(axis=-1) + 1j * (t1_i + t2_i).sum(axis=-1)


def _summaries_cartesian(ratio_r, ratio_i, a0, a1, b0, b1):
    """Calculate linearized data and waveform inner products in Cartesian coordinates."""
    return (
        _linearized_filter_cartesian(
            ratio_r, ratio_i, a0.real, a0.imag, a1.real, a1.imag
        ),
        _linearized_norm_cartesian(ratio_r, ratio_i, b0, b1),
    )


def _linearized_filter(ratio, a0, a1):
    """Calculate the linearized data-waveform inner product."""
    r_r = ratio.real if jnp.issubdtype(ratio.dtype, jnp.complexfloating) else ratio
    r_i = ratio.imag if jnp.issubdtype(ratio.dtype, jnp.complexfloating) else 0.0
    a0_r = a0.real if jnp.issubdtype(a0.dtype, jnp.complexfloating) else a0
    a0_i = a0.imag if jnp.issubdtype(a0.dtype, jnp.complexfloating) else 0.0
    a1_r = a1.real if jnp.issubdtype(a1.dtype, jnp.complexfloating) else a1
    a1_i = a1.imag if jnp.issubdtype(a1.dtype, jnp.complexfloating) else 0.0
    return _linearized_filter_cartesian(r_r, r_i, a0_r, a0_i, a1_r, a1_i)


def _linearized_norm(ratio, b0, b1):
    """Calculate the linearized waveform norm."""
    r_r = ratio.real if jnp.issubdtype(ratio.dtype, jnp.complexfloating) else ratio
    r_i = ratio.imag if jnp.issubdtype(ratio.dtype, jnp.complexfloating) else 0.0
    return _linearized_norm_cartesian(r_r, r_i, b0, b1)


def _linearized_cross(ratio, ratio2, a0, a1):
    """Calculate a linearized cross term between two waveform ratios."""
    r1_r = ratio.real if jnp.issubdtype(ratio.dtype, jnp.complexfloating) else ratio
    r1_i = ratio.imag if jnp.issubdtype(ratio.dtype, jnp.complexfloating) else 0.0
    r2_r = ratio2.real if jnp.issubdtype(ratio2.dtype, jnp.complexfloating) else ratio2
    r2_i = ratio2.imag if jnp.issubdtype(ratio2.dtype, jnp.complexfloating) else 0.0
    a0_r = a0.real if jnp.issubdtype(a0.dtype, jnp.complexfloating) else a0
    a0_i = a0.imag if jnp.issubdtype(a0.dtype, jnp.complexfloating) else 0.0
    a1_r = a1.real if jnp.issubdtype(a1.dtype, jnp.complexfloating) else a1
    a1_i = a1.imag if jnp.issubdtype(a1.dtype, jnp.complexfloating) else 0.0
    return _linearized_cross_cartesian(
        r1_r, r1_i, r2_r, r2_i, a0_r, a0_i, a1_r, a1_i
    )


def _summaries(ratio, a0, a1, b0, b1):
    """Calculate the linearized data and waveform inner products."""
    r_r = ratio.real if jnp.issubdtype(ratio.dtype, jnp.complexfloating) else ratio
    r_i = ratio.imag if jnp.issubdtype(ratio.dtype, jnp.complexfloating) else 0.0
    return _summaries_cartesian(r_r, r_i, a0, a1, b0, b1)


def _sample_axis(value):
    """Map a scalar or sample shape ``(...)`` to ``(...)`` or ``(..., 1)``."""
    if value.ndim == 0:
        return value
    return jnp.expand_dims(value, axis=-1)


def _frequency_axis(value, freqs_len, name="parameter"):
    """Validate a scalar or a value whose final axis is frequency."""
    if value.ndim and value.shape[-1] != freqs_len:
        raise ValueError(f"{name} must be scalar or have shape (..., {freqs_len})")
    return value


def likelihood_parts(freqs, fp, fc, dtc, hp, hc, h00, a0, a1, b0, b1):
    """Calculate scalar or sample-batched relative likelihood parts."""
    hp = _jax_array(hp)
    if hp is None:
        raise TypeError("a JAX-backed waveform is required")

    real_dtype = hp.real.dtype
    hc = _as_array(hc, hp, hp.dtype)
    freqs = _as_array(freqs, hp, real_dtype)
    h00 = _as_array(h00, hp, hp.dtype)
    fp = _sample_axis(_as_array(fp, hp, real_dtype))
    fc = _sample_axis(_as_array(fc, hp, real_dtype))
    dtc = _sample_axis(_as_array(dtc, hp, real_dtype))
    a0 = _as_array(a0, hp, hp.dtype)
    a1 = _as_array(a1, hp, hp.dtype)
    b0 = _as_array(b0, hp, real_dtype)
    b1 = _as_array(b1, hp, real_dtype)

    phase = -2.0 * _RELBIN_PI * dtc * freqs
    shift_r = jnp.cos(phase)
    shift_i = jnp.sin(phase)
    h_r = fp * hp.real + fc * hc.real
    h_i = fp * hp.imag + fc * hc.imag
    prod_r, prod_i = _cmul(shift_r, shift_i, h_r, h_i)
    ratio_r, ratio_i = _cdiv(prod_r, prod_i, h00.real, h00.imag)
    return _summaries_cartesian(ratio_r, ratio_i, a0, a1, b0, b1)


def likelihood_parts_v(freqs, fp, fc, dtc, hp, hc, h00, a0, a1, b0, b1):
    """Calculate likelihood parts with an Earth-rotation response."""
    hp = _jax_array(hp)
    if hp is None:
        raise TypeError("a JAX-backed waveform is required")

    real_dtype = hp.real.dtype
    hc = _as_array(hc, hp, hp.dtype)
    freqs = _as_array(freqs, hp, real_dtype)
    flen = freqs.shape[-1]
    h00 = _as_array(h00, hp, hp.dtype)
    fp = _frequency_axis(_as_array(fp, hp, real_dtype), flen, "fp")
    fc = _frequency_axis(_as_array(fc, hp, real_dtype), flen, "fc")
    dtc = _frequency_axis(_as_array(dtc, hp, real_dtype), flen, "dtc")
    a0 = _as_array(a0, hp, hp.dtype)
    a1 = _as_array(a1, hp, hp.dtype)
    b0 = _as_array(b0, hp, real_dtype)
    b1 = _as_array(b1, hp, real_dtype)

    phase = -2.0 * _RELBIN_PI * dtc * freqs
    shift_r = jnp.cos(phase)
    shift_i = jnp.sin(phase)
    h_r = fp * hp.real + fc * hc.real
    h_i = fp * hp.imag + fc * hc.imag
    prod_r, prod_i = _cmul(shift_r, shift_i, h_r, h_i)
    ratio_r, ratio_i = _cdiv(prod_r, prod_i, h00.real, h00.imag)
    return _summaries_cartesian(ratio_r, ratio_i, a0, a1, b0, b1)


def likelihood_parts_vector(freqs, fp, fc, dtc, hp, hc, h00, a0, a1, b0, b1):
    """Calculate likelihood parts for paired sky, time, or pol samples."""
    hp = _jax_array(hp)
    if hp is None:
        raise TypeError("a JAX-backed waveform is required")

    real_dtype = hp.real.dtype
    hc = _as_array(hc, hp, hp.dtype)
    freqs = _as_array(freqs, hp, real_dtype)
    h00 = _as_array(h00, hp, hp.dtype)
    fp = _sample_axis(_as_array(fp, hp, real_dtype))
    fc = _sample_axis(_as_array(fc, hp, real_dtype))
    dtc = _sample_axis(_as_array(dtc, hp, real_dtype))
    a0 = _as_array(a0, hp, hp.dtype)
    a1 = _as_array(a1, hp, hp.dtype)
    b0 = _as_array(b0, hp, real_dtype)
    b1 = _as_array(b1, hp, real_dtype)

    phase = -2.0 * _RELBIN_PI * dtc * freqs
    shift_r = jnp.cos(phase)
    shift_i = jnp.sin(phase)
    h_r = fp * hp.real + fc * hc.real
    h_i = fp * hp.imag + fc * hc.imag
    prod_r, prod_i = _cmul(shift_r, shift_i, h_r, h_i)
    ratio_r, ratio_i = _cdiv(prod_r, prod_i, h00.real, h00.imag)
    return _summaries_cartesian(ratio_r, ratio_i, a0, a1, b0, b1)


def _likelihood_parts_v_vector(
    freqs, fp, fc, times, dtc, pol_phase, hp, hc, h00, a0, a1, b0, b1
):
    """Calculate frequency-varying responses for paired samples."""
    hp = _jax_array(hp)
    if hp is None:
        raise TypeError("a JAX-backed waveform is required")

    real_dtype = hp.real.dtype
    hc = _as_array(hc, hp, hp.dtype)
    freqs = _as_array(freqs, hp, real_dtype)
    flen = freqs.shape[-1]
    h00 = _as_array(h00, hp, hp.dtype)
    fp = _frequency_axis(_as_array(fp, hp, real_dtype), flen, "fp")
    fc = _frequency_axis(_as_array(fc, hp, real_dtype), flen, "fc")
    times = _frequency_axis(_as_array(times, hp, real_dtype), flen, "times")
    dtc = _sample_axis(_as_array(dtc, hp, real_dtype))
    a0 = _as_array(a0, hp, hp.dtype)
    a1 = _as_array(a1, hp, hp.dtype)
    b0 = _as_array(b0, hp, real_dtype)
    b1 = _as_array(b1, hp, real_dtype)

    if pol_phase is not None:
        pol_phase = _sample_axis(_as_array(pol_phase, hp, hp.dtype))
        resp_r, resp_i = _cmul(fp, fc, pol_phase.real, pol_phase.imag)
    else:
        resp_r, resp_i = fp, fc

    phase = -2.0 * _RELBIN_PI * (times + dtc) * freqs
    shift_r = jnp.cos(phase)
    shift_i = jnp.sin(phase)
    h_r = resp_r * hp.real + resp_i * hc.real
    h_i = resp_r * hp.imag + resp_i * hc.imag
    prod_r, prod_i = _cmul(shift_r, shift_i, h_r, h_i)
    ratio_r, ratio_i = _cdiv(prod_r, prod_i, h00.real, h00.imag)
    return _summaries_cartesian(ratio_r, ratio_i, a0, a1, b0, b1)


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
    channel = _jax_array(channel)
    if channel is None:
        raise TypeError("a JAX-backed waveform is required")

    real_dtype = channel.real.dtype
    freqs = _as_array(freqs, channel, real_dtype)
    h00 = _as_array(h00, channel, channel.dtype)
    dtc = _sample_axis(_as_array(dtc, channel, real_dtype))
    a0 = _as_array(a0, channel, channel.dtype)
    a1 = _as_array(a1, channel, channel.dtype)
    b0 = _as_array(b0, channel, real_dtype)
    b1 = _as_array(b1, channel, real_dtype)

    phase = -2.0 * _RELBIN_PI * dtc * freqs
    shift_r = jnp.cos(phase)
    shift_i = jnp.sin(phase)
    ch_r = channel.real if jnp.issubdtype(channel.dtype, jnp.complexfloating) else channel
    ch_i = channel.imag if jnp.issubdtype(channel.dtype, jnp.complexfloating) else 0.0
    prod_r, prod_i = _cmul(shift_r, shift_i, ch_r, ch_i)
    ratio_r, ratio_i = _cdiv(prod_r, prod_i, h00.real, h00.imag)
    return _summaries_cartesian(ratio_r, ratio_i, a0, a1, b0, b1)


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
    hp = _jax_array(hp)
    if hp is None:
        raise TypeError("a JAX-backed waveform is required")

    real_dtype = hp.real.dtype
    hc = _as_array(hc, hp, hp.dtype)
    hp2 = _as_array(hp2, hp, hp.dtype)
    hc2 = _as_array(hc2, hp, hp.dtype)
    freqs = _as_array(freqs, hp, real_dtype)
    flen = freqs.shape[-1]
    h00 = _as_array(h00, hp, hp.dtype)
    h002 = _as_array(h002, hp, hp.dtype)
    axis = _frequency_axis if _frequency_varying else _sample_axis
    if _frequency_varying:
        fp = axis(_as_array(fp, hp, real_dtype), flen)
        fc = axis(_as_array(fc, hp, real_dtype), flen)
        dtc = axis(_as_array(dtc, hp, real_dtype), flen)
        fp2 = axis(_as_array(fp2, hp, real_dtype), flen)
        fc2 = axis(_as_array(fc2, hp, real_dtype), flen)
        dtc2 = axis(_as_array(dtc2, hp, real_dtype), flen)
    else:
        fp = axis(_as_array(fp, hp, real_dtype))
        fc = axis(_as_array(fc, hp, real_dtype))
        dtc = axis(_as_array(dtc, hp, real_dtype))
        fp2 = axis(_as_array(fp2, hp, real_dtype))
        fc2 = axis(_as_array(fc2, hp, real_dtype))
        dtc2 = axis(_as_array(dtc2, hp, real_dtype))
    a0 = _as_array(a0, hp, hp.dtype)
    a1 = _as_array(a1, hp, hp.dtype)

    phase = -2.0 * _RELBIN_PI * dtc * freqs
    phase2 = -2.0 * _RELBIN_PI * dtc2 * freqs
    shift_r = jnp.cos(phase)
    shift_i = jnp.sin(phase)
    shift2_r = jnp.cos(phase2)
    shift2_i = jnp.sin(phase2)

    h_r = fp * hp.real + fc * hc.real
    h_i = fp * hp.imag + fc * hc.imag
    prod_r, prod_i = _cmul(shift_r, shift_i, h_r, h_i)
    ratio_r, ratio_i = _cdiv(prod_r, prod_i, h00.real, h00.imag)

    h2_r = fp2 * hp2.real + fc2 * hc2.real
    h2_i = fp2 * hp2.imag + fc2 * hc2.imag
    prod2_r, prod2_i = _cmul(shift2_r, shift2_i, h2_r, h2_i)
    ratio2_r, ratio2_i = _cdiv(prod2_r, prod2_i, h002.real, h002.imag)

    return _linearized_cross_cartesian(
        ratio_r, ratio_i, ratio2_r, ratio2_i, a0.real, a0.imag, a1.real, a1.imag
    )


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
    channel = _jax_array(channel)
    if channel is None:
        raise TypeError("a JAX-backed waveform is required")

    real_dtype = channel.real.dtype
    channel2 = _as_array(channel2, channel, channel.dtype)
    freqs = _as_array(freqs, channel, real_dtype)
    h00 = _as_array(h00, channel, channel.dtype)
    h002 = _as_array(h002, channel, channel.dtype)
    dtc = _sample_axis(_as_array(dtc, channel, real_dtype))
    dtc2 = _sample_axis(_as_array(dtc2, channel, real_dtype))
    a0 = _as_array(a0, channel, channel.dtype)
    a1 = _as_array(a1, channel, channel.dtype)

    phase = -2.0 * _RELBIN_PI * dtc * freqs
    phase2 = -2.0 * _RELBIN_PI * dtc2 * freqs
    shift_r = jnp.cos(phase)
    shift_i = jnp.sin(phase)
    shift2_r = jnp.cos(phase2)
    shift2_i = jnp.sin(phase2)

    ch_r = channel.real if jnp.issubdtype(channel.dtype, jnp.complexfloating) else channel
    ch_i = channel.imag if jnp.issubdtype(channel.dtype, jnp.complexfloating) else 0.0
    prod_r, prod_i = _cmul(shift_r, shift_i, ch_r, ch_i)
    ratio_r, ratio_i = _cdiv(prod_r, prod_i, h00.real, h00.imag)

    ch2_r = channel2.real if jnp.issubdtype(channel2.dtype, jnp.complexfloating) else channel2
    ch2_i = channel2.imag if jnp.issubdtype(channel2.dtype, jnp.complexfloating) else 0.0
    prod2_r, prod2_i = _cmul(shift2_r, shift2_i, ch2_r, ch2_i)
    ratio2_r, ratio2_i = _cdiv(prod2_r, prod2_i, h002.real, h002.imag)

    # Preserve the established detector-frame kernel's argument order.
    return _linearized_cross_cartesian(
        ratio2_r, ratio2_i, ratio_r, ratio_i, a0.real, a0.imag, a1.real, a1.imag
    )


def _time_shifted_filters_cartesian(
    freqs, tstart, delta_t, num_samples, ratio_r, ratio_i, a0_r, a0_i, a1_r, a1_i
):
    """Evaluate relative-bin filters over a blocked uniform time grid in Cartesian form."""
    num_samples = int(num_samples)
    if num_samples == 0:
        return jnp.empty((0,), dtype=ratio_r.dtype), jnp.empty((0,), dtype=ratio_i.dtype)

    block_size = max(1, _SNR_PREDICTOR_TARGET_ELEMENTS // max(1, freqs.size))
    filters_r = []
    filters_i = []
    da_r = a0_r - a1_r
    da_i = a0_i - a1_i

    for start in range(0, num_samples, block_size):
        stop = min(start + block_size, num_samples)
        sample_indices = jnp.arange(start, stop, dtype=freqs.dtype)
        times = tstart + delta_t * sample_indices
        phase = -2.0 * _RELBIN_PI * jnp.expand_dims(times, axis=-1) * freqs
        shift_r = jnp.cos(phase)
        shift_i = jnp.sin(phase)

        sr_r, sr_i = _cmul(shift_r, shift_i, ratio_r, ratio_i)
        sr_lo_r, sr_lo_i = sr_r[..., :-1], sr_i[..., :-1]
        sr_hi_r, sr_hi_i = sr_r[..., 1:], sr_i[..., 1:]

        t1_r, t1_i = _cmul(da_r, da_i, sr_lo_r, sr_lo_i)
        t2_r, t2_i = _cmul(a1_r, a1_i, sr_hi_r, sr_hi_i)
        filters_r.append((t1_r + t2_r).sum(axis=-1))
        filters_i.append(-(t1_i + t2_i).sum(axis=-1))

    return jnp.concatenate(filters_r), jnp.concatenate(filters_i)


def _time_shifted_filters(freqs, tstart, delta_t, num_samples, ratio, a0, a1):
    """Evaluate relative-bin filters over a blocked uniform time grid."""
    ratio_r = ratio.real if jnp.issubdtype(ratio.dtype, jnp.complexfloating) else ratio
    ratio_i = ratio.imag if jnp.issubdtype(ratio.dtype, jnp.complexfloating) else 0.0
    a0_r = a0.real if jnp.issubdtype(a0.dtype, jnp.complexfloating) else a0
    a0_i = a0.imag if jnp.issubdtype(a0.dtype, jnp.complexfloating) else 0.0
    a1_r = a1.real if jnp.issubdtype(a1.dtype, jnp.complexfloating) else a1
    a1_i = a1.imag if jnp.issubdtype(a1.dtype, jnp.complexfloating) else 0.0
    filt_r, filt_i = _time_shifted_filters_cartesian(
        freqs, tstart, delta_t, num_samples, ratio_r, ratio_i, a0_r, a0_i, a1_r, a1_i
    )
    return filt_r + 1j * filt_i


def snr_predictor(freqs, tstart, delta_t, num_samples, hp, hc, h00, a0, a1, b0, b1):
    """Return the polarization-averaged SNR on a uniform time grid."""
    hp = _jax_array(hp)
    if hp is None:
        raise TypeError("a JAX-backed waveform is required")

    real_dtype = hp.real.dtype
    hc = _as_array(hc, hp, hp.dtype)
    freqs = _as_array(freqs, hp, real_dtype)
    h00 = _as_array(h00, hp, hp.dtype)
    a0 = _as_array(a0, hp, hp.dtype)
    a1 = _as_array(a1, hp, hp.dtype)
    b0 = _as_array(b0, hp, real_dtype)
    b1 = _as_array(b1, hp, real_dtype)
    tstart = _as_array(tstart, hp, real_dtype)
    delta_t = _as_array(delta_t, hp, real_dtype)

    hp_r = hp.real if jnp.issubdtype(hp.dtype, jnp.complexfloating) else hp
    hp_i = hp.imag if jnp.issubdtype(hp.dtype, jnp.complexfloating) else 0.0
    hc_r = hc.real if jnp.issubdtype(hc.dtype, jnp.complexfloating) else hc
    hc_i = hc.imag if jnp.issubdtype(hc.dtype, jnp.complexfloating) else 0.0
    h00_r = h00.real if jnp.issubdtype(h00.dtype, jnp.complexfloating) else h00
    h00_i = h00.imag if jnp.issubdtype(h00.dtype, jnp.complexfloating) else 0.0
    hp_ratio_r, hp_ratio_i = _cdiv(hp_r, hp_i, h00_r, h00_i)
    hc_ratio_r, hc_ratio_i = _cdiv(hc_r, hc_i, h00_r, h00_i)

    hh = _linearized_norm_cartesian(hp_ratio_r, hp_ratio_i, b0, b1)
    chh = _linearized_norm_cartesian(hc_ratio_r, hc_ratio_i, b0, b1)

    a0_r = a0.real if jnp.issubdtype(a0.dtype, jnp.complexfloating) else a0
    a0_i = a0.imag if jnp.issubdtype(a0.dtype, jnp.complexfloating) else 0.0
    a1_r = a1.real if jnp.issubdtype(a1.dtype, jnp.complexfloating) else a1
    a1_i = a1.imag if jnp.issubdtype(a1.dtype, jnp.complexfloating) else 0.0

    sh_r, sh_i = _time_shifted_filters_cartesian(
        freqs,
        tstart,
        delta_t,
        num_samples,
        hp_ratio_r,
        hp_ratio_i,
        a0_r,
        a0_i,
        a1_r,
        a1_i,
    )
    csh_r, csh_i = _time_shifted_filters_cartesian(
        freqs,
        tstart,
        delta_t,
        num_samples,
        hc_ratio_r,
        hc_ratio_i,
        a0_r,
        a0_i,
        a1_r,
        a1_i,
    )

    snr2 = (jnp.square(sh_r) + jnp.square(sh_i)) / (2.0 * hh) + (
        jnp.square(csh_r) + jnp.square(csh_i)
    ) / (2.0 * chh)
    return jnp.sqrt(snr2)


def snr_predictor_dom(freqs, tstart, delta_t, num_samples, hp, h00, a0, a1, b0, b1):
    """Return dominant-mode data products on a uniform time grid."""
    hp = _jax_array(hp)
    if hp is None:
        raise TypeError("a JAX-backed waveform is required")

    real_dtype = hp.real.dtype
    freqs = _as_array(freqs, hp, real_dtype)
    h00 = _as_array(h00, hp, hp.dtype)
    a0 = _as_array(a0, hp, hp.dtype)
    a1 = _as_array(a1, hp, hp.dtype)
    b0 = _as_array(b0, hp, real_dtype)
    b1 = _as_array(b1, hp, real_dtype)
    tstart = _as_array(tstart, hp, real_dtype)
    delta_t = _as_array(delta_t, hp, real_dtype)

    hp_r = hp.real if jnp.issubdtype(hp.dtype, jnp.complexfloating) else hp
    hp_i = hp.imag if jnp.issubdtype(hp.dtype, jnp.complexfloating) else 0.0
    h00_r = h00.real if jnp.issubdtype(h00.dtype, jnp.complexfloating) else h00
    h00_i = h00.imag if jnp.issubdtype(h00.dtype, jnp.complexfloating) else 0.0
    ratio_r, ratio_i = _cdiv(hp_r, hp_i, h00_r, h00_i)
    hh = _linearized_norm_cartesian(ratio_r, ratio_i, b0, b1)

    a0_r = a0.real if jnp.issubdtype(a0.dtype, jnp.complexfloating) else a0
    a0_i = a0.imag if jnp.issubdtype(a0.dtype, jnp.complexfloating) else 0.0
    a1_r = a1.real if jnp.issubdtype(a1.dtype, jnp.complexfloating) else a1
    a1_i = a1.imag if jnp.issubdtype(a1.dtype, jnp.complexfloating) else 0.0

    sh_r, sh_i = _time_shifted_filters_cartesian(
        freqs, tstart, delta_t, num_samples, ratio_r, ratio_i, a0_r, a0_i, a1_r, a1_i
    )
    return sh_r + 1j * sh_i, hh
