# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Compiled scalar recurrence for the Torch-native TaylorT4 port."""

# cython: language_level=3
# distutils: language=c++

cimport cython
cimport numpy as cnp
import numpy as np

from libc.math cimport isfinite, log
from libcpp.vector cimport vector


cdef struct TaylorT4Coefficients:
    double total_mass_seconds
    double leading
    double pn2
    double pn3
    double pn4
    double pn5
    double pn6
    double pn6_log
    double pn7
    double tidal10
    double tidal12
    int phase_order


@cython.cdivision(True)
cdef inline void rhs(
    double velocity,
    TaylorT4Coefficients* coefficients,
    double* velocity_derivative,
    double* phase_derivative,
) noexcept nogil:
    cdef double velocity2 = velocity * velocity
    cdef double velocity3 = velocity2 * velocity
    cdef double velocity4 = velocity3 * velocity
    cdef double velocity5 = velocity4 * velocity
    cdef double velocity6 = velocity5 * velocity
    cdef double velocity7 = velocity6 * velocity
    cdef double velocity9 = velocity7 * velocity2
    cdef double velocity10 = velocity9 * velocity
    cdef double velocity12 = velocity10 * velocity2
    cdef double series = 1.0
    cdef int order = coefficients.phase_order

    if order == -1 or order >= 2:
        series += coefficients.pn2 * velocity2
    if order == -1 or order >= 3:
        series += coefficients.pn3 * velocity3
    if order == -1 or order >= 4:
        series += coefficients.pn4 * velocity4
    if order == -1 or order >= 5:
        series += coefficients.pn5 * velocity5
    if order == -1 or order >= 6:
        series += (
            coefficients.pn6 + coefficients.pn6_log * log(velocity)
        ) * velocity6
    if order == -1 or order >= 7:
        series += coefficients.pn7 * velocity7
    if order != 0:
        series += (
            coefficients.tidal10 * velocity10
            + coefficients.tidal12 * velocity12
        )

    velocity_derivative[0] = coefficients.leading * series * velocity9
    phase_derivative[0] = velocity3 / coefficients.total_mass_seconds


@cython.boundscheck(False)
@cython.wraparound(False)
def evolve_taylor_t4(
    double initial_velocity,
    double delta_t,
    double reference_velocity,
    double coa_phase,
    double total_mass_seconds,
    double leading,
    double pn2,
    double pn3,
    double pn4,
    double pn5,
    double pn6,
    double pn6_log,
    double pn7,
    double tidal10,
    double tidal12,
    int phase_order,
):
    """Evolve and phase-anchor the two-scalar TaylorT4 RK4 recurrence."""

    cdef TaylorT4Coefficients coefficients
    coefficients.total_mass_seconds = total_mass_seconds
    coefficients.leading = leading
    coefficients.pn2 = pn2
    coefficients.pn3 = pn3
    coefficients.pn4 = pn4
    coefficients.pn5 = pn5
    coefficients.pn6 = pn6
    coefficients.pn6_log = pn6_log
    coefficients.pn7 = pn7
    coefficients.tidal10 = tidal10
    coefficients.tidal12 = tidal12
    coefficients.phase_order = phase_order

    cdef vector[double] velocities
    cdef vector[double] phases
    cdef double velocity = initial_velocity
    cdef double phase = 0.0
    cdef double k1_velocity, k1_phase
    cdef double k2_velocity, k2_phase
    cdef double k3_velocity, k3_phase
    cdef double k4_velocity, k4_phase
    cdef double next_velocity = initial_velocity
    cdef double next_phase = 0.0
    cdef double phase_offset
    cdef double scale = delta_t / 6.0
    cdef Py_ssize_t index, size
    cdef Py_ssize_t low, high, middle, phase_index
    cdef bint nonfinite = False
    cdef cnp.ndarray[cnp.float64_t, ndim=2] output
    cdef double[:, ::1] output_view

    velocities.reserve(65536)
    phases.reserve(65536)
    velocities.push_back(velocity)
    phases.push_back(phase)

    with nogil:
        while True:
            rhs(velocity, &coefficients, &k1_velocity, &k1_phase)
            rhs(
                velocity + 0.5 * delta_t * k1_velocity,
                &coefficients,
                &k2_velocity,
                &k2_phase,
            )
            rhs(
                velocity + 0.5 * delta_t * k2_velocity,
                &coefficients,
                &k3_velocity,
                &k3_phase,
            )
            rhs(
                velocity + delta_t * k3_velocity,
                &coefficients,
                &k4_velocity,
                &k4_phase,
            )
            next_velocity = velocity + scale * (
                k1_velocity
                + 2.0 * k2_velocity
                + 2.0 * k3_velocity
                + k4_velocity
            )
            next_phase = phase + scale * (
                k1_phase + 2.0 * k2_phase + 2.0 * k3_phase + k4_phase
            )
            if not isfinite(next_velocity) or not isfinite(next_phase):
                nonfinite = True
                break
            if next_velocity > 0.4082482904638631:
                break
            velocities.push_back(next_velocity)
            phases.push_back(next_phase)
            velocity = next_velocity
            phase = next_phase

    if nonfinite:
        raise RuntimeError("TaylorT4 evolution produced a non-finite state")

    size = velocities.size()
    if reference_velocity == 0.0:
        phase_index = size - 1
    elif reference_velocity == initial_velocity:
        phase_index = 0
    else:
        low = 0
        high = size
        while low < high:
            middle = low + (high - low) // 2
            if velocities[middle] <= reference_velocity:
                low = middle + 1
            else:
                high = middle
        if low == size:
            raise ValueError("f_ref must lie between f_lower and ISCO")
        phase_index = low - 1

    phase_offset = coa_phase - phases[phase_index]
    output = np.empty((2, size), dtype=np.float64)
    output_view = output
    with nogil:
        for index in range(size):
            output_view[0, index] = velocities[index]
            output_view[1, index] = phases[index] + phase_offset
    return output
