# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""JAX implementations of ground-detector geometry operations."""

import numpy as np
from astropy import constants


def _jax_antenna_pattern(
    response, right_ascension, declination, polarization, gmst_start, phase_offsets
):
    """Evaluate a tensor-polarization response in JAX."""
    import jax.numpy as jnp

    dtype = phase_offsets.dtype
    right_ascension = jnp.asarray(right_ascension, dtype=dtype)
    declination = jnp.asarray(declination, dtype=dtype)
    polarization = jnp.asarray(polarization, dtype=dtype)

    gha_start = jnp.asarray(gmst_start, dtype=dtype) - right_ascension
    cos_start = jnp.cos(gha_start)
    sin_start = jnp.sin(gha_start)
    cos_offset = jnp.cos(phase_offsets)
    sin_offset = jnp.sin(phase_offsets)
    cosgha = cos_start * cos_offset - sin_start * sin_offset
    singha = sin_start * cos_offset + cos_start * sin_offset
    cosdec = jnp.cos(declination)
    sindec = jnp.sin(declination)
    cospsi = jnp.cos(polarization)
    sinpsi = jnp.sin(polarization)

    x = jnp.stack(
        (
            -cospsi * singha - sinpsi * cosgha * sindec,
            -cospsi * cosgha + sinpsi * singha * sindec,
            sinpsi * cosdec + jnp.zeros_like(phase_offsets),
        )
    )
    y = jnp.stack(
        (
            sinpsi * singha - cospsi * cosgha * sindec,
            sinpsi * cosgha + cospsi * singha * sindec,
            cospsi * cosdec + jnp.zeros_like(phase_offsets),
        )
    )

    response_is_complex = (
        jnp.iscomplexobj(response)
        if isinstance(response, (jnp.ndarray, np.ndarray))
        else np.iscomplexobj(response)
    )
    if response_is_complex:
        response_dtype = (
            jnp.complex128 if dtype == jnp.float64 else jnp.complex64
        )
    else:
        response_dtype = dtype
    response = jnp.asarray(response, dtype=response_dtype)

    response_grid_dims = response.ndim - 2
    vector_grid_dims = x.ndim - 1
    if response_grid_dims < vector_grid_dims:
        response = response.reshape(
            response.shape + (1,) * (vector_grid_dims - response_grid_dims)
        )
    x = x.astype(response_dtype)
    y = y.astype(response_dtype)
    dx = jnp.einsum("ij...,j...->i...", response, x)
    dy = jnp.einsum("ij...,j...->i...", response, y)
    fplus = jnp.sum(x * dx - y * dy, axis=0)
    fcross = jnp.sum(x * dy + y * dx, axis=0)
    return fplus, fcross


def _jax_single_arm_frequency_response(frequency, direction, arm_length):
    """Evaluate the finite-arm transfer function with JAX operations."""
    import jax.numpy as jnp

    array_inputs = tuple(
        value
        for value in (frequency, direction, arm_length)
        if isinstance(value, (jnp.ndarray, np.ndarray))
    )
    if any(jnp.iscomplexobj(value) for value in array_inputs):
        raise TypeError("JAX finite-arm response inputs must be real")

    dtype = None
    for value in (frequency, direction, arm_length):
        if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating):
            value_dtype = value.dtype
        elif isinstance(value, float):
            value_dtype = jnp.float64
        else:
            value_dtype = jnp.float64
        dtype = (
            value_dtype if dtype is None else jnp.promote_types(dtype, value_dtype)
        )
    if dtype in (jnp.float16, jnp.bfloat16):
        dtype = jnp.float32

    frequency = jnp.asarray(frequency, dtype=dtype)
    direction = jnp.clip(jnp.asarray(direction, dtype=dtype), -0.999, 0.999)
    arm_length = jnp.asarray(arm_length, dtype=dtype)

    phase = 2.0 * jnp.pi * frequency * arm_length / float(constants.c.value)
    minus = 1.0 - direction
    plus = 1.0 + direction
    theta1 = 0.5 * phase * minus
    sinc1 = jnp.sinc(phase * minus / (2.0 * jnp.pi))
    cos1 = jnp.cos(theta1)
    sin1 = jnp.sin(theta1)

    theta2 = 0.5 * phase * (3.0 - direction)
    sinc2 = jnp.sinc(phase * plus / (2.0 * jnp.pi))
    cos2 = jnp.cos(theta2)
    sin2 = jnp.sin(theta2)

    re = 0.5 * (cos1 * sinc1 + cos2 * sinc2)
    im = 0.5 * (-sin1 * sinc1 - sin2 * sinc2)
    return re + 1j * im


def _jax_time_delay(
    detector_location,
    other_location,
    right_ascension,
    declination,
    gmst_start,
    phase_offsets,
):
    """Evaluate a detector time delay in JAX."""
    import jax.numpy as jnp

    dtype = phase_offsets.dtype
    right_ascension = jnp.asarray(right_ascension, dtype=dtype)
    declination = jnp.asarray(declination, dtype=dtype)

    gha_start = jnp.asarray(gmst_start, dtype=dtype) - right_ascension
    cos_start = jnp.cos(gha_start)
    sin_start = jnp.sin(gha_start)
    cos_offset = jnp.cos(phase_offsets)
    sin_offset = jnp.sin(phase_offsets)
    cosgha = cos_start * cos_offset - sin_start * sin_offset
    singha = sin_start * cos_offset + cos_start * sin_offset
    cosdec = jnp.cos(declination)

    ehat = jnp.stack(
        (
            cosdec * cosgha,
            -cosdec * singha,
            jnp.sin(declination) + jnp.zeros_like(phase_offsets),
        )
    )
    displacement = jnp.asarray(
        np.asarray(other_location) - np.asarray(detector_location),
        dtype=dtype,
    )
    displacement = displacement.reshape((3,) + (1,) * (ehat.ndim - 1))
    return jnp.sum(displacement * ehat, axis=0) / float(constants.c.value)


def _jax_antenna_pattern_and_time_delay(
    detector_location,
    response,
    right_ascension,
    declination,
    polarization,
    gmst_start,
    phase_offsets,
):
    """Evaluate a tensor response and geocentric delay in JAX."""
    import jax.numpy as jnp

    dtype = phase_offsets.dtype
    right_ascension = jnp.asarray(right_ascension, dtype=dtype)
    declination = jnp.asarray(declination, dtype=dtype)
    polarization = jnp.asarray(polarization, dtype=dtype)

    gha_start = jnp.asarray(gmst_start, dtype=dtype) - right_ascension
    cos_start = jnp.cos(gha_start)
    sin_start = jnp.sin(gha_start)
    cos_offset = jnp.cos(phase_offsets)
    sin_offset = jnp.sin(phase_offsets)
    cosgha = cos_start * cos_offset - sin_start * sin_offset
    singha = sin_start * cos_offset + cos_start * sin_offset
    cosdec = jnp.cos(declination)
    sindec = jnp.sin(declination)
    cospsi = jnp.cos(polarization)
    sinpsi = jnp.sin(polarization)

    x = jnp.stack(
        (
            -cospsi * singha - sinpsi * cosgha * sindec,
            -cospsi * cosgha + sinpsi * singha * sindec,
            sinpsi * cosdec + jnp.zeros_like(phase_offsets),
        )
    )
    y = jnp.stack(
        (
            sinpsi * singha - cospsi * cosgha * sindec,
            sinpsi * cosgha + cospsi * singha * sindec,
            cospsi * cosdec + jnp.zeros_like(phase_offsets),
        )
    )

    response_is_complex = (
        jnp.iscomplexobj(response)
        if isinstance(response, (jnp.ndarray, np.ndarray))
        else np.iscomplexobj(response)
    )
    if response_is_complex:
        response_dtype = (
            jnp.complex128 if dtype == jnp.float64 else jnp.complex64
        )
    else:
        response_dtype = dtype
    response = jnp.asarray(response, dtype=response_dtype)

    response_grid_dims = response.ndim - 2
    vector_grid_dims = x.ndim - 1
    if response_grid_dims < vector_grid_dims:
        response = response.reshape(
            response.shape + (1,) * (vector_grid_dims - response_grid_dims)
        )
    x = x.astype(response_dtype)
    y = y.astype(response_dtype)
    dx = jnp.einsum("ij...,j...->i...", response, x)
    dy = jnp.einsum("ij...,j...->i...", response, y)
    fplus = jnp.sum(x * dx - y * dy, axis=0)
    fcross = jnp.sum(x * dy + y * dx, axis=0)

    ehat = jnp.stack(
        (
            cosdec * cosgha,
            -cosdec * singha,
            sindec + jnp.zeros_like(phase_offsets),
        )
    )
    location = jnp.asarray(
        -np.asarray(detector_location),
        dtype=dtype,
    )
    location = location.reshape((3,) + (1,) * (ehat.ndim - 1))
    delay = jnp.sum(location * ehat, axis=0) / float(constants.c.value)
    return fplus, fcross, delay


def _jax_network_antenna_pattern_and_time_delay(
    detector_locations,
    responses,
    right_ascension,
    declination,
    polarization,
    gmst_start,
    phase_offsets,
):
    """Evaluate tensor responses and delays for a detector network in JAX."""
    import jax.numpy as jnp

    dtype = phase_offsets.dtype
    right_ascension = jnp.asarray(right_ascension, dtype=dtype)
    declination = jnp.asarray(declination, dtype=dtype)
    polarization = jnp.asarray(polarization, dtype=dtype)

    gha_start = jnp.asarray(gmst_start, dtype=dtype) - right_ascension
    cos_start = jnp.cos(gha_start)
    sin_start = jnp.sin(gha_start)
    cos_offset = jnp.cos(phase_offsets)
    sin_offset = jnp.sin(phase_offsets)
    cosgha = cos_start * cos_offset - sin_start * sin_offset
    singha = sin_start * cos_offset + cos_start * sin_offset
    cosdec = jnp.cos(declination)
    sindec = jnp.sin(declination)
    cospsi = jnp.cos(polarization)
    sinpsi = jnp.sin(polarization)

    x = jnp.stack(
        (
            -cospsi * singha - sinpsi * cosgha * sindec,
            -cospsi * cosgha + sinpsi * singha * sindec,
            sinpsi * cosdec + jnp.zeros_like(phase_offsets),
        )
    )
    y = jnp.stack(
        (
            sinpsi * singha - cospsi * cosgha * sindec,
            sinpsi * cosgha + cospsi * singha * sindec,
            cospsi * cosdec + jnp.zeros_like(phase_offsets),
        )
    )

    responses_tensor = jnp.asarray(responses, dtype=dtype)
    dx = jnp.einsum("dij,j...->di...", responses_tensor, x)
    dy = jnp.einsum("dij,j...->di...", responses_tensor, y)
    fplus = jnp.sum(x * dx - y * dy, axis=1)
    fcross = jnp.sum(x * dy + y * dx, axis=1)

    ehat = jnp.stack(
        (
            cosdec * cosgha,
            -cosdec * singha,
            sindec + jnp.zeros_like(phase_offsets),
        )
    )
    locations_tensor = jnp.asarray(detector_locations, dtype=dtype)
    delay = -jnp.einsum("dj,j...->d...", locations_tensor, ehat) / float(
        constants.c.value
    )
    return fplus, fcross, delay


def _input_spec(values, angular_values=()):
    """Validate JAX inputs and return their common dtype."""
    import jax
    import jax.numpy as jnp

    angular_arrays = tuple(
        value for value in angular_values if isinstance(value, (jax.Array, jnp.ndarray))
    )
    if any(not jnp.issubdtype(value.dtype, jnp.floating) for value in angular_arrays):
        raise TypeError("JAX detector angles must be floating")

    arrays = tuple(
        value for value in values if isinstance(value, (jax.Array, jnp.ndarray))
    )
    if any(jnp.iscomplexobj(value) for value in arrays):
        raise TypeError("JAX detector inputs must be real")

    dtype = None
    for value in values:
        if isinstance(value, (jax.Array, jnp.ndarray)):
            val_dtype = (
                value.dtype
                if jnp.issubdtype(value.dtype, jnp.floating)
                else jnp.float64
            )
        elif isinstance(value, float):
            val_dtype = jnp.float64
        elif isinstance(value, int):
            val_dtype = jnp.float64
        elif isinstance(value, np.ndarray):
            val_dtype = (
                value.dtype
                if np.issubdtype(value.dtype, np.floating)
                else jnp.float64
            )
        else:
            val_dtype = jnp.float64
        dtype = val_dtype if dtype is None else jnp.promote_types(dtype, val_dtype)

    if dtype in (jnp.float16, jnp.bfloat16):
        dtype = jnp.float32
    return dtype


def _sky_grid(owner, angular_values, t_gps, dtype, extras=()):
    """Broadcast sky, time, and optional extra grids in JAX."""
    import jax
    import jax.numpy as jnp

    time_is_array = (
        isinstance(t_gps, (jax.Array, jnp.ndarray, np.ndarray))
        or np.ndim(t_gps) > 0
    )
    values = list(angular_values)
    if time_is_array:
        if owner.reference_time is None:
            raise NotImplementedError(
                "JAX GPS-time grids require a detector GMST reference time"
            )
        if owner.gmst_reference is None:
            owner.set_gmst_reference()
        relative_time = jnp.asarray(t_gps, dtype=dtype) - float(
            owner.reference_time
        )
        values.append(relative_time)
    values.extend(extras)
    converted = tuple(jnp.asarray(value, dtype=dtype) for value in values)
    broadcast = jnp.broadcast_arrays(*converted)
    angles = broadcast[: len(angular_values)]
    next_value = len(angular_values)
    if time_is_array:
        relative_time = broadcast[next_value]
        next_value += 1
        phase_offsets = relative_time / float(owner.sday) * (2.0 * np.pi)
        gmst_start = owner.gmst_reference
    else:
        phase_offsets = jnp.zeros_like(angles[0])
        gmst_start = owner.gmst_estimate(t_gps)
    return angles, broadcast[next_value:], gmst_start, phase_offsets


def single_arm_frequency_response(frequency, direction, arm_length):
    """Evaluate the finite-arm transfer function with JAX operations."""
    return _jax_single_arm_frequency_response(frequency, direction, arm_length)


def antenna_pattern(
    detector,
    right_ascension,
    declination,
    polarization,
    t_gps,
    frequency=0,
    polarization_type="tensor",
):
    """Return a detector antenna pattern for JAX-backed inputs."""
    import jax
    import jax.numpy as jnp

    if polarization_type != "tensor":
        raise NotImplementedError(
            "JAX antenna patterns currently support only the tensor response"
        )
    angular_inputs = (right_ascension, declination, polarization)
    inputs = angular_inputs + (frequency, t_gps)
    dtype = _input_spec(inputs, angular_inputs)
    finite_arm = (
        isinstance(frequency, (jax.Array, jnp.ndarray))
        or np.ndim(frequency) > 0
        or frequency != 0
    )
    extras = (frequency,) if finite_arm else ()
    angles, extra_grid, gmst_start, phase_offsets = _sky_grid(
        detector, angular_inputs, t_gps, dtype, extras
    )

    response = detector.response
    if finite_arm:
        frequency_tensor = extra_grid[0]
        gmst = jnp.asarray(gmst_start, dtype=dtype) + phase_offsets
        gha = gmst - angles[0]
        cosdec = jnp.cos(angles[1])
        direction = jnp.stack(
            (
                cosdec * jnp.cos(gha),
                -cosdec * jnp.sin(gha),
                jnp.sin(angles[1]),
            )
        )
        grid_dims = direction.ndim - 1
        xvec = jnp.asarray(detector.info["xvec"], dtype=dtype).reshape(
            (3,) + (1,) * grid_dims
        )
        yvec = jnp.asarray(detector.info["yvec"], dtype=dtype).reshape(
            (3,) + (1,) * grid_dims
        )
        nx = jnp.sum(direction * xvec, axis=0)
        ny = jnp.sum(direction * yvec, axis=0)
        rx = single_arm_frequency_response(
            frequency_tensor, nx, detector.info["xlength"]
        )
        ry = single_arm_frequency_response(
            frequency_tensor, ny, detector.info["ylength"]
        )
        matrix_shape = (3, 3) + (1,) * rx.ndim
        xresp = jnp.asarray(detector.info["xresp"], dtype=dtype).reshape(matrix_shape)
        yresp = jnp.asarray(detector.info["yresp"], dtype=dtype).reshape(matrix_shape)
        response = ry * yresp - rx * xresp

    return _jax_antenna_pattern(
        response,
        angles[0],
        angles[1],
        angles[2],
        gmst_start,
        phase_offsets,
    )


def antenna_pattern_and_time_delay(
    detector, right_ascension, declination, polarization, t_gps
):
    """Return antenna patterns and geocentric delay for JAX inputs."""
    angular_inputs = (right_ascension, declination, polarization)
    dtype = _input_spec(
        angular_inputs + (t_gps,), angular_inputs
    )
    angles, _, gmst_start, phase_offsets = _sky_grid(
        detector, angular_inputs, t_gps, dtype
    )
    return _jax_antenna_pattern_and_time_delay(
        detector.location,
        detector.response,
        angles[0],
        angles[1],
        angles[2],
        gmst_start,
        phase_offsets,
    )


def time_delay_from_location(
    detector, other_location, right_ascension, declination, t_gps
):
    """Return a detector time delay for JAX-backed sky coordinates."""
    angular_inputs = (right_ascension, declination)
    dtype = _input_spec(
        angular_inputs + (t_gps,), angular_inputs
    )
    angles, _, gmst_start, phase_offsets = _sky_grid(
        detector, angular_inputs, t_gps, dtype
    )
    return _jax_time_delay(
        detector.location,
        other_location,
        angles[0],
        angles[1],
        gmst_start,
        phase_offsets,
    )


def network_antenna_pattern_and_time_delay(
    network, right_ascension, declination, polarization, t_gps
):
    """Return vectorized network geometry for JAX-backed inputs."""
    angular_inputs = (right_ascension, declination, polarization)
    dtype = _input_spec(
        angular_inputs + (t_gps,), angular_inputs
    )
    angles, _, gmst_start, phase_offsets = _sky_grid(
        network, angular_inputs, t_gps, dtype
    )
    return _jax_network_antenna_pattern_and_time_delay(
        network.locations,
        network.responses,
        angles[0],
        angles[1],
        angles[2],
        gmst_start,
        phase_offsets,
    )


def effective_distance(detector, distance, ra, dec, pol, time, inclination):
    """Return effective distance while preserving detector overrides."""
    import jax
    import jax.numpy as jnp

    angular_inputs = (ra, dec, pol)
    values = (distance,) + angular_inputs + (time, inclination)
    dtype = _input_spec(values, angular_inputs)
    broadcast_values = (distance, ra, dec, pol, inclination)
    time_is_array = isinstance(time, (jax.Array, jnp.ndarray, np.ndarray)) or np.ndim(time) > 0
    if time_is_array:
        broadcast_values += (time,)
    broadcast = jnp.broadcast_arrays(
        *(jnp.asarray(value, dtype=dtype) for value in broadcast_values)
    )
    distance, ra, dec, pol, inclination = broadcast[:5]
    if time_is_array:
        time = broadcast[5]
    fplus, fcross = detector.antenna_pattern(ra, dec, pol, time)
    cos_inclination = jnp.cos(inclination)
    plus_inclination = 0.5 * (1.0 + cos_inclination**2)
    scale = jnp.sqrt(
        (fplus * plus_inclination)**2 + (fcross * cos_inclination)**2
    )
    return distance / scale


__all__ = [
    "antenna_pattern",
    "antenna_pattern_and_time_delay",
    "effective_distance",
    "network_antenna_pattern_and_time_delay",
    "single_arm_frequency_response",
    "time_delay_from_location",
]
