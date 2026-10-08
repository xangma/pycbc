"""JAX kernels for the relative-binning likelihood."""


import jax


import numpy


import jax.numpy as jnp


from pycbc.types.array_jax import _reference_enabled, to_jax


from pycbc.types.backend import backend_array


def _jax_array(value):
    """Return JAX storage through the public backend protocol."""
    if isinstance(value, (int, float, bool)):
        return None
    return backend_array(value, "jax")


def _as_array(value, like, dtype):
    """Convert beside the waveform, preserving traced device execution."""
    arr = _jax_array(value)
    if arr is None:
        arr = value
    device = None if isinstance(like, jax.core.Tracer) else like.device
    return to_jax(arr, dtype=dtype, device=device)


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

    if _reference_enabled("inference_projection"):
        if isinstance(like, jax.core.Tracer) or isinstance(
            polarization, jax.core.Tracer
        ):
            raise RuntimeError("Native projection cannot run inside jax.jit")
        return to_jax(
            numpy.exp(-2.0j * numpy.asarray(polarization)), device=like.device
        )
    angle = _as_array(polarization, like, like.real.dtype)
    two_psi = 2.0 * angle
    return jnp.cos(two_psi) - 1j * jnp.sin(two_psi)


def polarized_antenna_response(fp, fc, pol_phase, like):
    """Rotate zero-polarization antenna factors on a JAX device."""
    like = _jax_array(like)
    if like is None:
        raise TypeError("a JAX-backed waveform is required")

    if _reference_enabled("inference_projection"):
        if any(
            isinstance(value, jax.core.Tracer) for value in (like, fp, fc, pol_phase)
        ):
            raise RuntimeError("Native projection cannot run inside jax.jit")
        phase = numpy.asarray(pol_phase)
        if not numpy.iscomplexobj(phase):
            phase = numpy.exp(-2.0j * phase)
        rotated = (numpy.asarray(fp) + 1.0j * numpy.asarray(fc)) * phase
        return tuple(
            to_jax(value, device=like.device) for value in (rotated.real, rotated.imag)
        )
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
