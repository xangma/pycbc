# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch implementations of ground-detector geometry operations."""

import numpy as np
import torch
from astropy import constants


def _torch_antenna_pattern(
    response, right_ascension, declination, polarization, gmst_start, phase_offsets
):
    """Evaluate a tensor-polarization response on a Torch device."""
    device = phase_offsets.device
    dtype = phase_offsets.dtype
    right_ascension = torch.as_tensor(right_ascension, device=device, dtype=dtype)
    declination = torch.as_tensor(declination, device=device, dtype=dtype)
    polarization = torch.as_tensor(polarization, device=device, dtype=dtype)

    gha_start = (
        torch.as_tensor(gmst_start, device=device, dtype=dtype) - right_ascension
    )
    # Keep sub-second sidereal changes visible on float32-only devices.
    cos_start = torch.cos(gha_start)
    sin_start = torch.sin(gha_start)
    cos_offset = torch.cos(phase_offsets)
    sin_offset = torch.sin(phase_offsets)
    cosgha = cos_start * cos_offset - sin_start * sin_offset
    singha = sin_start * cos_offset + cos_start * sin_offset
    cosdec = torch.cos(declination)
    sindec = torch.sin(declination)
    cospsi = torch.cos(polarization)
    sinpsi = torch.sin(polarization)

    x = torch.stack(
        (
            -cospsi * singha - sinpsi * cosgha * sindec,
            -cospsi * cosgha + sinpsi * singha * sindec,
            sinpsi * cosdec + torch.zeros_like(phase_offsets),
        )
    )
    y = torch.stack(
        (
            sinpsi * singha - cospsi * cosgha * sindec,
            sinpsi * cosgha + cospsi * singha * sindec,
            cospsi * cosdec + torch.zeros_like(phase_offsets),
        )
    )

    response_is_complex = (
        response.is_complex()
        if isinstance(response, torch.Tensor)
        else np.iscomplexobj(response)
    )
    if response_is_complex:
        response_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    else:
        response_dtype = dtype
    response = torch.as_tensor(response, device=device, dtype=response_dtype)

    # A frequency-dependent response carries broadcast grid dimensions after
    # its two matrix dimensions. Match static responses to that grid.
    response_grid_dims = response.ndim - 2
    vector_grid_dims = x.ndim - 1
    if response_grid_dims < vector_grid_dims:
        response = response.reshape(
            response.shape + (1,) * (vector_grid_dims - response_grid_dims)
        )
    x = x.to(dtype=response_dtype)
    y = y.to(dtype=response_dtype)
    dx = torch.einsum("ij...,j...->i...", response, x)
    dy = torch.einsum("ij...,j...->i...", response, y)
    fplus = torch.sum(x * dx - y * dy, dim=0)
    fcross = torch.sum(x * dy + y * dx, dim=0)
    return fplus, fcross


def _torch_single_arm_frequency_response(frequency, direction, arm_length):
    """Evaluate the finite-arm transfer function with Torch operations."""
    tensor_inputs = tuple(
        value
        for value in (frequency, direction, arm_length)
        if isinstance(value, torch.Tensor)
    )
    anchor = tensor_inputs[0]
    if any(value.is_complex() for value in tensor_inputs):
        raise TypeError("Torch finite-arm response inputs must be real")

    dtype = None
    for value in tensor_inputs:
        value_dtype = (
            value.dtype if torch.is_floating_point(value) else torch.get_default_dtype()
        )
        dtype = (
            value_dtype if dtype is None else torch.promote_types(dtype, value_dtype)
        )
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32

    frequency = torch.as_tensor(frequency, device=anchor.device, dtype=dtype)
    direction = torch.as_tensor(direction, device=anchor.device, dtype=dtype).clamp(
        -0.999, 0.999
    )
    arm_length = torch.as_tensor(arm_length, device=anchor.device, dtype=dtype)

    # This sinc form preserves the limit of one at zero frequency and avoids
    # cancellation around that limit.
    phase = 2.0 * torch.pi * frequency * arm_length / float(constants.c.value)
    minus = 1.0 - direction
    plus = 1.0 + direction
    return 0.5 * (
        torch.exp(-0.5j * phase * minus) * torch.sinc(phase * minus / (2.0 * torch.pi))
        + torch.exp(-0.5j * phase * (3.0 - direction))
        * torch.sinc(phase * plus / (2.0 * torch.pi))
    )


def _torch_time_delay(
    detector_location,
    other_location,
    right_ascension,
    declination,
    gmst_start,
    phase_offsets,
):
    """Evaluate a detector time delay without leaving a Torch device."""
    device = phase_offsets.device
    dtype = phase_offsets.dtype
    right_ascension = torch.as_tensor(right_ascension, device=device, dtype=dtype)
    declination = torch.as_tensor(declination, device=device, dtype=dtype)

    gha_start = (
        torch.as_tensor(gmst_start, device=device, dtype=dtype) - right_ascension
    )
    # Keep the large absolute sidereal angle separate from the usually small
    # time offset so float32 devices retain sub-second changes.
    cos_start = torch.cos(gha_start)
    sin_start = torch.sin(gha_start)
    cos_offset = torch.cos(phase_offsets)
    sin_offset = torch.sin(phase_offsets)
    cosgha = cos_start * cos_offset - sin_start * sin_offset
    singha = sin_start * cos_offset + cos_start * sin_offset
    cosdec = torch.cos(declination)

    ehat = torch.stack(
        (
            cosdec * cosgha,
            -cosdec * singha,
            torch.sin(declination),
        )
    )
    displacement = torch.as_tensor(
        np.asarray(other_location) - np.asarray(detector_location),
        device=device,
        dtype=dtype,
    )
    displacement = displacement.reshape((3,) + (1,) * (ehat.ndim - 1))
    return torch.sum(displacement * ehat, dim=0) / float(constants.c.value)


def _torch_antenna_pattern_and_time_delay(
    detector_location,
    response,
    right_ascension,
    declination,
    polarization,
    gmst_start,
    phase_offsets,
):
    """Evaluate a tensor response and geocentric delay on a Torch device."""
    device = phase_offsets.device
    dtype = phase_offsets.dtype
    right_ascension = torch.as_tensor(right_ascension, device=device, dtype=dtype)
    declination = torch.as_tensor(declination, device=device, dtype=dtype)
    polarization = torch.as_tensor(polarization, device=device, dtype=dtype)

    gha_start = (
        torch.as_tensor(gmst_start, device=device, dtype=dtype) - right_ascension
    )
    cos_start = torch.cos(gha_start)
    sin_start = torch.sin(gha_start)
    cos_offset = torch.cos(phase_offsets)
    sin_offset = torch.sin(phase_offsets)
    cosgha = cos_start * cos_offset - sin_start * sin_offset
    singha = sin_start * cos_offset + cos_start * sin_offset
    cosdec = torch.cos(declination)
    sindec = torch.sin(declination)
    cospsi = torch.cos(polarization)
    sinpsi = torch.sin(polarization)

    x = torch.stack(
        (
            -cospsi * singha - sinpsi * cosgha * sindec,
            -cospsi * cosgha + sinpsi * singha * sindec,
            sinpsi * cosdec + torch.zeros_like(phase_offsets),
        )
    )
    y = torch.stack(
        (
            sinpsi * singha - cospsi * cosgha * sindec,
            sinpsi * cosgha + cospsi * singha * sindec,
            cospsi * cosdec + torch.zeros_like(phase_offsets),
        )
    )

    response_is_complex = (
        response.is_complex()
        if isinstance(response, torch.Tensor)
        else np.iscomplexobj(response)
    )
    if response_is_complex:
        response_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    else:
        response_dtype = dtype
    response = torch.as_tensor(response, device=device, dtype=response_dtype)

    response_grid_dims = response.ndim - 2
    vector_grid_dims = x.ndim - 1
    if response_grid_dims < vector_grid_dims:
        response = response.reshape(
            response.shape + (1,) * (vector_grid_dims - response_grid_dims)
        )
    x = x.to(dtype=response_dtype)
    y = y.to(dtype=response_dtype)
    dx = torch.einsum("ij...,j...->i...", response, x)
    dy = torch.einsum("ij...,j...->i...", response, y)
    fplus = torch.sum(x * dx - y * dy, dim=0)
    fcross = torch.sum(x * dy + y * dx, dim=0)

    ehat = torch.stack(
        (
            cosdec * cosgha,
            -cosdec * singha,
            sindec + torch.zeros_like(phase_offsets),
        )
    )
    location = torch.as_tensor(
        -np.asarray(detector_location),
        device=device,
        dtype=dtype,
    )
    location = location.reshape((3,) + (1,) * (ehat.ndim - 1))
    delay = torch.sum(location * ehat, dim=0) / float(constants.c.value)
    return fplus, fcross, delay


def _torch_network_antenna_pattern_and_time_delay(
    detector_locations,
    responses,
    right_ascension,
    declination,
    polarization,
    gmst_start,
    phase_offsets,
):
    """Evaluate tensor responses and delays for a detector network."""
    device = phase_offsets.device
    dtype = phase_offsets.dtype
    right_ascension = torch.as_tensor(right_ascension, device=device, dtype=dtype)
    declination = torch.as_tensor(declination, device=device, dtype=dtype)
    polarization = torch.as_tensor(polarization, device=device, dtype=dtype)

    gha_start = (
        torch.as_tensor(gmst_start, device=device, dtype=dtype) - right_ascension
    )
    cos_start = torch.cos(gha_start)
    sin_start = torch.sin(gha_start)
    cos_offset = torch.cos(phase_offsets)
    sin_offset = torch.sin(phase_offsets)
    cosgha = cos_start * cos_offset - sin_start * sin_offset
    singha = sin_start * cos_offset + cos_start * sin_offset
    cosdec = torch.cos(declination)
    sindec = torch.sin(declination)
    cospsi = torch.cos(polarization)
    sinpsi = torch.sin(polarization)

    x = torch.stack(
        (
            -cospsi * singha - sinpsi * cosgha * sindec,
            -cospsi * cosgha + sinpsi * singha * sindec,
            sinpsi * cosdec + torch.zeros_like(phase_offsets),
        )
    )
    y = torch.stack(
        (
            sinpsi * singha - cospsi * cosgha * sindec,
            sinpsi * cosgha + cospsi * singha * sindec,
            cospsi * cosdec + torch.zeros_like(phase_offsets),
        )
    )

    responses_tensor = torch.as_tensor(responses, device=device, dtype=dtype)
    dx = torch.einsum("dij,j...->di...", responses_tensor, x)
    dy = torch.einsum("dij,j...->di...", responses_tensor, y)
    fplus = torch.sum(x * dx - y * dy, dim=1)
    fcross = torch.sum(x * dy + y * dx, dim=1)

    ehat = torch.stack(
        (
            cosdec * cosgha,
            -cosdec * singha,
            sindec + torch.zeros_like(phase_offsets),
        )
    )
    locations_tensor = torch.as_tensor(detector_locations, device=device, dtype=dtype)
    delay = -torch.einsum("dj,j...->d...", locations_tensor, ehat) / float(
        constants.c.value
    )
    return fplus, fcross, delay


def _input_spec(values, angular_values=(), *, mps_safe=False):
    """Validate Torch inputs and return their common device and dtype."""
    angular_tensors = tuple(
        value for value in angular_values if isinstance(value, torch.Tensor)
    )
    if any(not torch.is_floating_point(value) for value in angular_tensors):
        raise TypeError("Torch detector angles must be floating")

    tensors = tuple(value for value in values if isinstance(value, torch.Tensor))
    if any(value.is_complex() for value in tensors):
        raise TypeError("Torch detector inputs must be real")

    anchor = tensors[0]
    dtype = None
    for value in tensors:
        value_dtype = (
            value.dtype if torch.is_floating_point(value) else torch.get_default_dtype()
        )
        dtype = (
            value_dtype if dtype is None else torch.promote_types(dtype, value_dtype)
        )
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    if mps_safe and anchor.device.type == "mps" and dtype == torch.float64:
        dtype = torch.float32
    return anchor.device, dtype


def _sky_grid(owner, angular_values, t_gps, device, dtype, extras=()):
    """Broadcast sky, time, and optional extra grids on one Torch device."""
    time_is_tensor = isinstance(t_gps, torch.Tensor)
    time_is_array = not time_is_tensor and np.ndim(t_gps) > 0
    time_grid = time_is_tensor or time_is_array
    values = list(angular_values)
    if time_grid:
        if owner.reference_time is None:
            raise NotImplementedError(
                "Torch GPS-time grids require a detector GMST reference time"
            )
        if owner.gmst_reference is None:
            owner.set_gmst_reference()
        if time_is_tensor:
            relative_time = t_gps.to(device=device, dtype=dtype) - float(
                owner.reference_time
            )
        else:
            relative_time = torch.as_tensor(
                np.asarray(t_gps, dtype=np.float64) - float(owner.reference_time),
                device=device,
                dtype=dtype,
            )
        values.append(relative_time)
    values.extend(extras)
    broadcast = torch.broadcast_tensors(
        *(torch.as_tensor(value, device=device, dtype=dtype) for value in values)
    )
    angles = broadcast[: len(angular_values)]
    next_value = len(angular_values)
    if time_grid:
        relative_time = broadcast[next_value]
        next_value += 1
        phase_offsets = relative_time / float(owner.sday) * (2.0 * np.pi)
        gmst_start = owner.gmst_reference
    else:
        phase_offsets = torch.zeros_like(angles[0])
        gmst_start = owner.gmst_estimate(t_gps)
    return angles, broadcast[next_value:], gmst_start, phase_offsets


def single_arm_frequency_response(frequency, direction, arm_length):
    """Evaluate the finite-arm transfer function with Torch operations."""
    return _torch_single_arm_frequency_response(frequency, direction, arm_length)


def antenna_pattern(
    detector,
    right_ascension,
    declination,
    polarization,
    t_gps,
    frequency=0,
    polarization_type="tensor",
):
    """Return a detector antenna pattern for Torch-backed inputs."""
    if polarization_type != "tensor":
        raise NotImplementedError(
            "Torch antenna patterns currently support only the tensor response"
        )
    angular_inputs = (right_ascension, declination, polarization)
    inputs = angular_inputs + (frequency, t_gps)
    device, dtype = _input_spec(inputs, angular_inputs, mps_safe=True)
    finite_arm = (
        isinstance(frequency, torch.Tensor) or np.ndim(frequency) > 0 or frequency != 0
    )
    extras = (frequency,) if finite_arm else ()
    angles, extra_grid, gmst_start, phase_offsets = _sky_grid(
        detector, angular_inputs, t_gps, device, dtype, extras
    )

    response = detector.response
    if finite_arm:
        frequency_tensor = extra_grid[0]
        gmst = torch.as_tensor(gmst_start, device=device, dtype=dtype) + phase_offsets
        gha = gmst - angles[0]
        cosdec = torch.cos(angles[1])
        direction = torch.stack(
            (
                cosdec * torch.cos(gha),
                -cosdec * torch.sin(gha),
                torch.sin(angles[1]),
            )
        )
        grid_dims = direction.ndim - 1
        xvec = torch.as_tensor(
            detector.info["xvec"], device=device, dtype=dtype
        ).reshape((3,) + (1,) * grid_dims)
        yvec = torch.as_tensor(
            detector.info["yvec"], device=device, dtype=dtype
        ).reshape((3,) + (1,) * grid_dims)
        nx = torch.sum(direction * xvec, dim=0)
        ny = torch.sum(direction * yvec, dim=0)
        rx = single_arm_frequency_response(
            frequency_tensor, nx, detector.info["xlength"]
        )
        ry = single_arm_frequency_response(
            frequency_tensor, ny, detector.info["ylength"]
        )
        matrix_shape = (3, 3) + (1,) * rx.ndim
        xresp = torch.as_tensor(
            detector.info["xresp"], device=device, dtype=dtype
        ).reshape(matrix_shape)
        yresp = torch.as_tensor(
            detector.info["yresp"], device=device, dtype=dtype
        ).reshape(matrix_shape)
        response = ry * yresp - rx * xresp

    return _torch_antenna_pattern(
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
    """Return antenna patterns and geocentric delay for Torch inputs."""
    angular_inputs = (right_ascension, declination, polarization)
    device, dtype = _input_spec(
        angular_inputs + (t_gps,), angular_inputs, mps_safe=True
    )
    angles, _, gmst_start, phase_offsets = _sky_grid(
        detector, angular_inputs, t_gps, device, dtype
    )
    return _torch_antenna_pattern_and_time_delay(
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
    """Return a detector time delay for Torch-backed sky coordinates."""
    angular_inputs = (right_ascension, declination)
    device, dtype = _input_spec(
        angular_inputs + (t_gps,), angular_inputs, mps_safe=True
    )
    angles, _, gmst_start, phase_offsets = _sky_grid(
        detector, angular_inputs, t_gps, device, dtype
    )
    return _torch_time_delay(
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
    """Return vectorized network geometry for Torch-backed inputs."""
    angular_inputs = (right_ascension, declination, polarization)
    device, dtype = _input_spec(
        angular_inputs + (t_gps,), angular_inputs, mps_safe=True
    )
    angles, _, gmst_start, phase_offsets = _sky_grid(
        network, angular_inputs, t_gps, device, dtype
    )
    return _torch_network_antenna_pattern_and_time_delay(
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
    angular_inputs = (ra, dec, pol)
    values = (distance,) + angular_inputs + (time, inclination)
    device, dtype = _input_spec(values, angular_inputs)
    broadcast_values = (distance, ra, dec, pol, inclination)
    time_is_tensor = isinstance(time, torch.Tensor)
    if time_is_tensor:
        broadcast_values += (time,)
    broadcast = torch.broadcast_tensors(
        *(
            torch.as_tensor(value, device=device, dtype=dtype)
            for value in broadcast_values
        )
    )
    distance, ra, dec, pol, inclination = broadcast[:5]
    if time_is_tensor:
        time = broadcast[5]
    fplus, fcross = detector.antenna_pattern(ra, dec, pol, time)
    cos_inclination = torch.cos(inclination)
    plus_inclination = 0.5 * (1.0 + cos_inclination.square())
    scale = torch.sqrt(
        (fplus * plus_inclination).square() + (fcross * cos_inclination).square()
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
