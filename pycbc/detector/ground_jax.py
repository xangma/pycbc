# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""JAX implementations of ground-detector geometry operations."""

import numpy as np
import jax
import jax.numpy as jnp

from pycbc.constants import C_SI
from pycbc.domain_jax import native_result, REFERENCE_UNSELECTED
from pycbc.types.backend import coerce_jax_values, jax_module_for


def _selected(name):
    from pycbc import scheme

    selected = getattr(scheme.mgr.state, "jax_reference_operations", ())
    return "detector" in selected or "detector." + name in selected


def _place(value, reference, dtype=None):
    value = jnp.asarray(value, dtype=dtype)
    sharding = getattr(reference, "sharding", None)
    return jax.device_put(value, sharding) if sharding is not None else value


def _convert(values, reference, dtype):
    reference = _place(reference, reference, dtype)
    _, converted = coerce_jax_values(reference, *values)
    return converted[1:]


def _reference_input(values):
    return next(value for value in values if jax_module_for(value) is not None)


def _legacy_static_grid(shape):
    """Whether the original static response dot/broadcast can evaluate a grid."""
    vector = (3,) + tuple(shape)
    if len(vector) == 1:
        return True
    if vector[-2] != 3:
        return False
    contracted = (3,) + vector[:-2] + vector[-1:]
    return all(
        a == b or a == 1 or b == 1
        for a, b in zip(reversed(vector), reversed(contracted))
    )


def antenna_reference(
    function, detector, ra, dec, pol, time, frequency=0, polarization_type="tensor"
):
    """Replay scalar native calls only for new grid/frequency-grid inputs."""
    values = np.broadcast_arrays(ra, dec, pol, time, frequency)
    if _legacy_static_grid(values[0].shape) and np.ndim(frequency) == 0:
        return function(detector, ra, dec, pol, time, frequency, polarization_type)
    rows = [
        function(detector, *(value[index] for value in values), polarization_type)
        for index in np.ndindex(values[0].shape)
    ]
    return tuple(
        np.asarray([row[i] for row in rows]).reshape(values[0].shape) for i in range(2)
    )


def distance_reference(function, detector, distance, ra, dec, pol, time, inclination):
    """Replay the original scalar API for the new multidimensional grid."""
    values = np.broadcast_arrays(distance, ra, dec, pol, time, inclination)
    if _legacy_static_grid(values[0].shape):
        return function(detector, distance, ra, dec, pol, time, inclination)
    rows = [
        function(detector, *(value[index] for value in values))
        for index in np.ndindex(values[0].shape)
    ]
    return np.asarray(rows).reshape(values[0].shape)


def _original_distance_scale(detector, distance, fplus, fcross, inclination):
    """Execute the unchanged distance body with already computed responses."""
    from types import SimpleNamespace
    from pycbc.detector.ground import Detector

    proxy = SimpleNamespace(antenna_pattern=lambda *args: (fplus, fcross))
    return Detector.effective_distance.__wrapped__(
        proxy, distance, 0.0, 0.0, 0.0, 0.0, inclination
    )


def _polarization_basis(
    right_ascension, declination, polarization, gmst_start, phase_offsets
):
    """Compute wave-frame basis vectors x, y and propagation direction ehat."""
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
    ehat = jnp.stack(
        (
            cosdec * cosgha,
            -cosdec * singha,
            sindec + jnp.zeros_like(phase_offsets),
        )
    )
    return x, y, ehat


def _response_contraction(response, x, y, reference):
    dtype = reference.dtype
    response_dtype = (
        (jnp.complex128 if dtype == jnp.float64 else jnp.complex64)
        if jnp.iscomplexobj(response)
        else dtype
    )
    response = _place(response, reference, response_dtype)
    legacy = response.ndim == 2 and _legacy_static_grid(x.shape[1:])
    if not legacy and response.ndim - 2 < x.ndim - 1:
        response = response.reshape(
            response.shape + (1,) * (x.ndim + 1 - response.ndim)
        )
    x, y = x.astype(response_dtype), y.astype(response_dtype)
    if legacy:
        # Retain the original dot axes for native-supported array shapes.
        dx, dy = jnp.dot(response, x), jnp.dot(response, y)
    else:
        dx = jnp.einsum("ij...,j...->i...", response, x)
        dy = jnp.einsum("ij...,j...->i...", response, y)
    return jnp.sum(x * dx - y * dy, axis=0), jnp.sum(x * dy + y * dx, axis=0)


def _jax_antenna_pattern(
    response, right_ascension, declination, polarization, gmst_start, phase_offsets
):
    x, y, _ = _polarization_basis(
        right_ascension, declination, polarization, gmst_start, phase_offsets
    )
    return _response_contraction(response, x, y, phase_offsets)


def _jax_single_arm_frequency_response(frequency, direction, arm_length):
    """Evaluate the finite-arm transfer function with JAX operations."""
    if any(jnp.iscomplexobj(value) for value in (frequency, direction, arm_length)):
        raise TypeError("JAX finite-arm response inputs must be real")

    dtype = None
    for value in (frequency, direction, arm_length):
        if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating):
            value_dtype = value.dtype
        elif isinstance(value, float):
            value_dtype = jnp.float64
        else:
            value_dtype = jnp.float64
        dtype = value_dtype if dtype is None else jnp.promote_types(dtype, value_dtype)
    if dtype in (jnp.float16, jnp.bfloat16):
        dtype = jnp.float32

    reference = _reference_input((frequency, direction, arm_length))
    frequency, direction, arm_length = _convert(
        (frequency, direction, arm_length), reference, dtype
    )
    direction = jnp.clip(direction, -0.999, 0.999)

    phase = 2.0 * jnp.pi * frequency * arm_length / C_SI
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
    result = re + 1j * im
    # The original transfer function is undefined at zero frequency.
    return jnp.where(frequency == 0, jnp.nan + 1j * jnp.nan, result)


def _jax_time_delay(
    detector_location,
    other_location,
    right_ascension,
    declination,
    gmst_start,
    phase_offsets,
):
    """Evaluate a detector time delay in JAX."""
    _, _, ehat = _polarization_basis(
        right_ascension, declination, 0.0, gmst_start, phase_offsets
    )
    dtype = phase_offsets.dtype
    displacement = _place(other_location, phase_offsets, dtype) - _place(
        detector_location, phase_offsets, dtype
    )
    displacement = displacement.reshape((3,) + (1,) * (ehat.ndim - 1))
    return jnp.sum(displacement * ehat, axis=0) / C_SI


def _jax_antenna_pattern_and_time_delay(
    detector_location,
    response,
    right_ascension,
    declination,
    polarization,
    gmst_start,
    phase_offsets,
):
    x, y, ehat = _polarization_basis(
        right_ascension, declination, polarization, gmst_start, phase_offsets
    )
    fplus, fcross = _response_contraction(response, x, y, phase_offsets)
    location = -_place(detector_location, phase_offsets, phase_offsets.dtype)
    location = location.reshape((3,) + (1,) * (ehat.ndim - 1))
    return fplus, fcross, jnp.sum(location * ehat, axis=0) / C_SI


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
    x, y, ehat = _polarization_basis(
        right_ascension, declination, polarization, gmst_start, phase_offsets
    )
    dtype = phase_offsets.dtype
    responses_tensor = _place(responses, phase_offsets, dtype)
    fplus, fcross = jax.vmap(
        lambda response: _response_contraction(response, x, y, phase_offsets)
    )(responses_tensor)

    locations_tensor = _place(detector_locations, phase_offsets, dtype)
    delay = -jnp.einsum("dj,j...->d...", locations_tensor, ehat) / C_SI
    return fplus, fcross, delay


def _input_spec(values, angular_values=()):
    """Validate JAX inputs and return their common dtype."""

    angular_arrays = tuple(
        value for value in angular_values if jax_module_for(value) is not None
    )
    if any(not jnp.issubdtype(value.dtype, jnp.floating) for value in angular_arrays):
        raise TypeError("JAX detector angles must be floating")

    if any(jnp.iscomplexobj(value) for value in values):
        raise TypeError("JAX detector inputs must be real")

    dtype = None
    for value in values:
        if jax_module_for(value) is not None:
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
                value.dtype if np.issubdtype(value.dtype, np.floating) else jnp.float64
            )
        else:
            val_dtype = jnp.float64
        dtype = val_dtype if dtype is None else jnp.promote_types(dtype, val_dtype)

    if dtype in (jnp.float16, jnp.bfloat16):
        dtype = jnp.float32
    return dtype


def _sky_grid(owner, angular_values, t_gps, dtype, extras=(), reference=None):
    """Broadcast sky, time, and optional extra grids in JAX."""
    time_is_array = (
        jax_module_for(t_gps) is not None
        or isinstance(t_gps, np.ndarray)
        or np.ndim(t_gps) > 0
    )
    from pycbc.detector.ground import Detector

    custom_clock = (
        getattr(owner.gmst_estimate, "__func__", None) is not Detector.gmst_estimate
    )
    native_clock = _selected("Detector.gmst_estimate")
    values = list(angular_values)
    if time_is_array:
        if owner.reference_time is None and not custom_clock and not native_clock:
            raise NotImplementedError(
                "JAX GPS-time grids require a detector GMST reference time"
            )
        if owner.reference_time is not None and owner.gmst_reference is None:
            owner.set_gmst_reference()
        values.append(t_gps)
    values.extend(extras)
    converted = _convert(values, reference, dtype)
    broadcast = jnp.broadcast_arrays(*converted)
    angles = broadcast[: len(angular_values)]
    next_value = len(angular_values)
    if time_is_array:
        gps_time = broadcast[next_value]
        next_value += 1
        if custom_clock or native_clock:
            gmst_start = owner.gmst_estimate(gps_time)
            phase_offsets = jnp.zeros_like(gps_time)
        else:
            relative_time = gps_time - float(owner.reference_time)
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
    if polarization_type != "tensor":
        raise NotImplementedError(
            "JAX antenna patterns currently support only the tensor response"
        )
    angular_inputs = (right_ascension, declination, polarization)
    inputs = angular_inputs + (t_gps, frequency)
    dtype = _input_spec(inputs, angular_inputs)
    finite_arm = (
        isinstance(frequency, jax.core.Tracer)
        or np.ndim(frequency) > 0
        or bool(frequency != 0)
    )
    extras = (frequency,) if finite_arm else ()
    angles, extra_grid, gmst_start, phase_offsets = _sky_grid(
        detector, angular_inputs, t_gps, dtype, extras, _reference_input(inputs)
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
        xvec = _place(detector.info["xvec"], phase_offsets, dtype).reshape(
            (3,) + (1,) * grid_dims
        )
        yvec = _place(detector.info["yvec"], phase_offsets, dtype).reshape(
            (3,) + (1,) * grid_dims
        )
        nx = jnp.sum(direction * xvec, axis=0)
        ny = jnp.sum(direction * yvec, axis=0)
        from pycbc.detector.ground import single_arm_frequency_response as original_arm

        rx = original_arm(frequency_tensor, nx, detector.info["xlength"])
        ry = original_arm(frequency_tensor, ny, detector.info["ylength"])
        matrix_shape = (3, 3) + (1,) * rx.ndim
        xresp = _place(detector.info["xresp"], phase_offsets, dtype).reshape(
            matrix_shape
        )
        yresp = _place(detector.info["yresp"], phase_offsets, dtype).reshape(
            matrix_shape
        )
        response = ry * yresp - rx * xresp
        response = jnp.where(
            frequency_tensor == 0,
            _place(detector.response, phase_offsets, dtype).reshape(matrix_shape),
            response,
        )

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
    if any(
        _selected(name)
        for name in (
            "Detector.antenna_pattern",
            "Detector.time_delay_from_location",
            "Detector.gmst_estimate",
        )
    ):
        return (
            *detector.antenna_pattern(
                right_ascension, declination, polarization, t_gps
            ),
            detector.time_delay_from_earth_center(right_ascension, declination, t_gps),
        )
    angular_inputs = (right_ascension, declination, polarization)
    dtype = _input_spec(angular_inputs + (t_gps,), angular_inputs)
    angles, _, gmst_start, phase_offsets = _sky_grid(
        detector,
        angular_inputs,
        t_gps,
        dtype,
        reference=_reference_input(angular_inputs + (t_gps,)),
    )
    return _jax_antenna_pattern_and_time_delay(
        detector.location, detector.response, *angles, gmst_start, phase_offsets
    )


def time_delay_from_location(
    detector, other_location, right_ascension, declination, t_gps
):
    """Return a detector time delay for JAX-backed sky coordinates."""
    angular_inputs = (right_ascension, declination)
    dtype = _input_spec(angular_inputs + (t_gps,), angular_inputs)
    angles, _, gmst_start, phase_offsets = _sky_grid(
        detector,
        angular_inputs,
        t_gps,
        dtype,
        reference=_reference_input((other_location,) + angular_inputs + (t_gps,)),
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
    """Fuse only detectors with the same built-in methods and clock policy."""
    from pycbc.detector.ground import Detector

    methods = (
        "antenna_pattern",
        "time_delay_from_earth_center",
        "time_delay_from_location",
        "gmst_estimate",
        "antenna_pattern_and_time_delay",
    )
    first = network.detectors[0]
    for detector in network.detectors:
        if detector.reference_time is not None and detector.gmst_reference is None:
            detector.set_gmst_reference()
    qualified = all(
        detector.reference_time == first.reference_time
        and detector.gmst_reference == first.gmst_reference
        and detector.sday == first.sday
        and all(
            getattr(getattr(detector, name), "__func__", None)
            is getattr(Detector, name)
            for name in methods
        )
        for detector in network.detectors
    )
    if not qualified or any(
        _selected(name)
        for name in (
            "Detector.antenna_pattern",
            "Detector.time_delay_from_location",
            "Detector.gmst_estimate",
        )
    ):
        rows = [
            detector.antenna_pattern_and_time_delay(
                right_ascension, declination, polarization, t_gps
            )
            for detector in network.detectors
        ]
        reference = _reference_input(
            (right_ascension, declination, polarization, t_gps)
        )
        return tuple(
            jnp.stack([_place(row[i], reference) for row in rows])
            for i in range(3)
        )
    angular_inputs = (right_ascension, declination, polarization)
    dtype = _input_spec(angular_inputs + (t_gps,), angular_inputs)
    angles, _, gmst_start, phase_offsets = _sky_grid(
        first,
        angular_inputs,
        t_gps,
        dtype,
        reference=_reference_input(angular_inputs + (t_gps,)),
    )
    return _jax_network_antenna_pattern_and_time_delay(
        network.locations, network.responses, *angles, gmst_start, phase_offsets
    )


def effective_distance(detector, distance, ra, dec, pol, time, inclination):
    """Keep original argument shapes at independently validated boundaries."""
    values = (distance, ra, dec, pol, time, inclination)
    dtype = _input_spec(values, (ra, dec, pol))
    reference = _reference_input(values)
    # Keep each angle's native precision/shape for the antenna validation
    # boundary. Only host integer angles need floating-point conversion.
    angles = tuple(
        _place(
            value,
            reference,
            (
                dtype
                if not jnp.issubdtype(jnp.asarray(value).dtype, jnp.floating)
                else None
            ),
        )
        for value in (ra, dec, pol)
    )
    response_time = (
        _place(time, reference)
        if jax_module_for(time) is not None or np.ndim(time) > 0
        else time
    )
    fplus, fcross = detector.antenna_pattern(*angles, response_time)
    fplus, fcross = _place(fplus, reference), _place(fcross, reference)
    original = native_result(
        "detector",
        "Detector.effective_distance_scale",
        _original_distance_scale,
        detector,
        distance,
        fplus,
        fcross,
        inclination,
    )
    if original is not REFERENCE_UNSELECTED:
        return original
    distance, inclination = _convert((distance, inclination), reference, dtype)
    cos_inclination = jnp.cos(inclination)
    plus_inclination = 0.5 * (1.0 + cos_inclination**2)
    scale = jnp.sqrt((fplus * plus_inclination) ** 2 + (fcross * cos_inclination) ** 2)
    from pycbc.types.array_jax import _divide

    return _divide(distance, scale)


__all__ = [
    "antenna_pattern",
    "antenna_pattern_and_time_delay",
    "effective_distance",
    "network_antenna_pattern_and_time_delay",
    "single_arm_frequency_response",
    "time_delay_from_location",
]
