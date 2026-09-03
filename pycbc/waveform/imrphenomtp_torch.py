# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch-backed orbit evolution shared by IMRPhenomTP and IMRPhenomTPHM.

The default numerical precession prescription in the IMRPhenomTP family
evolves the orbit-averaged SpinTaylor vector equations while taking the
orbital velocity from the IMRPhenomT carrier.  The adaptive solver evolves its
small, sequential 13-scalar state with fused host arithmetic, then transfers
the completed orbit to the carrier device once.  This avoids launching and
synchronizing thousands of tiny device kernels while keeping the bulk carrier,
angle, mode, and waveform operations on the active Torch device.  The solver
intentionally preserves the reference-index and endpoint conventions of
LALSuite 7.26.1, including the duplicated reference state immediately before a
non-grid-aligned reference time.

This is the common dynamical layer only.  Source-frame construction, Euler
angles, mode twisting, and public waveform dispatch are assembled separately.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass

from pycbc import lal_compat as lal
import torch

from ._cubic_spline_torch import (
    _natural_cubic_coeff,
    _spline_derivative,
    _spline_eval,
)
from .imrphenomt_fits_torch import qnm_fring_22
from .imrphenomt_torch import _IMRPhenomTCore, _IMRPhenomTCoreSetup
from .imrphenomx_spintaylor_torch import (
    _rotate_y,
    _rotate_z,
    SpinTaylorJFrame,
    spintaylor_j_frame,
    spintaylor_unwrap_angle,
)
from .imrphenomx_utils_torch import (
    final_spin_2017,
    precessing_final_spin_2017,
    qnm_fring_21,
)


@dataclass(frozen=True)
class IMRPhenomTPOrbit:
    """PhenomT-driven precessing orbit on the carrier time grid.

    Spins use the usual component-normalized ``chi`` convention.  The vector
    fields all have shape ``(samples, 3)`` and remain on the carrier device.
    ``reference_index`` is the index at which the exact input state is placed.
    """

    time_m: torch.Tensor
    velocity: torch.Tensor
    lnhat: torch.Tensor
    spin1: torch.Tensor
    spin2: torch.Tensor
    e1: torch.Tensor
    reference_index: int
    epoch: float
    delta_t: float


@dataclass(frozen=True)
class IMRPhenomTPEulerAngles:
    """Numerical TP/TPHM Euler angles on the full carrier grid."""

    time_m: torch.Tensor
    alpha: torch.Tensor
    cosbeta: torch.Tensor
    gamma: torch.Tensor
    frame: SpinTaylorJFrame
    evolved_final_spin: torch.Tensor
    reference_index: int
    epoch: float
    delta_t: float


def _coerce_ordered_spins(carrier, spin1, spin2):
    """Coerce full spins consistently with prepared or complete T state."""

    if not isinstance(carrier, (_IMRPhenomTCoreSetup, _IMRPhenomTCore)):
        raise TypeError("IMRPhenomTP requires prepared IMRPhenomT state")
    dtype = carrier.binary.dtype
    device = carrier.binary.device
    spin1 = torch.as_tensor(spin1, dtype=dtype, device=device)
    spin2 = torch.as_tensor(spin2, dtype=dtype, device=device)
    if spin1.shape != (3,) or spin2.shape != (3,):
        raise ValueError("IMRPhenomTP spins must have length three")
    if not bool(torch.all(torch.isfinite(torch.cat((spin1, spin2)))).detach().cpu()):
        raise ValueError("IMRPhenomTP spins must be finite")
    if not torch.allclose(
        torch.stack((spin1[2], spin2[2])),
        carrier.binary[2:],
        rtol=0.0,
        atol=8.0 * torch.finfo(dtype).eps,
    ):
        raise ValueError("IMRPhenomTP spin z components must match the carrier")
    return spin1, spin2


def imrphenomtp_initial_final_spin(carrier, spin1, spin2):
    """Return TP's initial PhenomX precessing-remnant spin prescription.

    The result supplies the precessing ringdown quantities in the initial
    carrier.  The aligned remnant spin still controls the default merger
    reconstruction.  Both are distinct from the evolved peak spin used for
    the Euler-angle ringdown slope and final co-precessing reconstruction.
    """

    spin1, spin2 = _coerce_ordered_spins(carrier, spin1, spin2)
    mass1, mass2 = carrier.binary[:2].unbind()
    total_mass = mass1 + mass2
    fraction1 = mass1 / total_mass
    fraction2 = mass2 / total_mass
    eta = fraction1 * fraction2
    total_perpendicular_spin = torch.linalg.vector_norm(
        fraction1**2 * spin1[:2] + fraction2**2 * spin2[:2]
    )
    chi_total_perpendicular = total_perpendicular_spin / fraction1**2
    final_spin = precessing_final_spin_2017(
        eta,
        spin1[2],
        spin2[2],
        chi_total_perpendicular,
    )
    return torch.clamp(final_spin, -1.0, 1.0)


def _cross(first, second):
    """Return the cross product of two three-scalar host vectors."""

    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _uniform_natural_cubic_coeff(values, step):
    """Build natural-cubic coefficients for uniformly spaced host values."""

    count = len(values)
    slopes = [(values[index + 1] - values[index]) / step for index in range(count - 1)]
    alpha = [3.0 * (slopes[index + 1] - slopes[index]) for index in range(count - 2)]
    diagonal = [1.0] * count
    upper = [0.0] * count
    rhs = [0.0] * count
    for index in range(1, count - 1):
        diagonal[index] = 4.0 * step - step * upper[index - 1]
        upper[index] = step / diagonal[index]
        rhs[index] = (alpha[index - 1] - step * rhs[index - 1]) / diagonal[index]

    quadratic = [0.0] * count
    linear = [0.0] * (count - 1)
    cubic = [0.0] * (count - 1)
    for index in range(count - 2, -1, -1):
        quadratic[index] = rhs[index] - upper[index] * quadratic[index + 1]
        linear[index] = (
            slopes[index] - step * (quadratic[index + 1] + 2.0 * quadratic[index]) / 3.0
        )
        cubic[index] = (quadratic[index + 1] - quadratic[index]) / (3.0 * step)
    return linear, quadratic, cubic


def _uniform_spline_eval(point, step, values, linear, quadratic, cubic):
    """Evaluate a uniformly spaced host natural-cubic spline."""

    index = min(max(math.ceil(point / step) - 1, 0), len(values) - 2)
    offset = point - index * step
    return (
        values[index]
        + linear[index] * offset
        + quadratic[index] * offset**2
        + cubic[index] * offset**3
    )


def _dormand_prince_step(state, step, rhs):
    """Take one embedded Dormand--Prince 5(4) step on-device."""

    k1 = rhs(state)
    k2 = rhs(state + step * (1.0 / 5.0) * k1)
    k3 = rhs(state + step * (3.0 / 40.0 * k1 + 9.0 / 40.0 * k2))
    k4 = rhs(state + step * (44.0 / 45.0 * k1 - 56.0 / 15.0 * k2 + 32.0 / 9.0 * k3))
    k5 = rhs(
        state
        + step
        * (
            19372.0 / 6561.0 * k1
            - 25360.0 / 2187.0 * k2
            + 64448.0 / 6561.0 * k3
            - 212.0 / 729.0 * k4
        )
    )
    k6 = rhs(
        state
        + step
        * (
            9017.0 / 3168.0 * k1
            - 355.0 / 33.0 * k2
            + 46732.0 / 5247.0 * k3
            + 49.0 / 176.0 * k4
            - 5103.0 / 18656.0 * k5
        )
    )
    fifth_order = state + step * (
        35.0 / 384.0 * k1
        + 500.0 / 1113.0 * k3
        + 125.0 / 192.0 * k4
        - 2187.0 / 6784.0 * k5
        + 11.0 / 84.0 * k6
    )
    k7 = rhs(fifth_order)
    fourth_order = state + step * (
        5179.0 / 57600.0 * k1
        + 7571.0 / 16695.0 * k3
        + 393.0 / 640.0 * k4
        - 92097.0 / 339200.0 * k5
        + 187.0 / 2100.0 * k6
        + 1.0 / 40.0 * k7
    )
    error = fifth_order - fourth_order
    return fifth_order, error, k1, k7


def _dormand_prince_host_step(state, step, rhs):
    """Take one fused host Dormand--Prince 5(4) step."""

    k1 = rhs(state)
    k2 = rhs([value + step * (1.0 / 5.0) * slope for value, slope in zip(state, k1)])
    k3 = rhs(
        [
            value + step * (3.0 / 40.0 * slope1 + 9.0 / 40.0 * slope2)
            for value, slope1, slope2 in zip(state, k1, k2)
        ]
    )
    k4 = rhs(
        [
            value
            + step * (44.0 / 45.0 * slope1 - 56.0 / 15.0 * slope2 + 32.0 / 9.0 * slope3)
            for value, slope1, slope2, slope3 in zip(state, k1, k2, k3)
        ]
    )
    k5 = rhs(
        [
            value
            + step
            * (
                19372.0 / 6561.0 * slope1
                - 25360.0 / 2187.0 * slope2
                + 64448.0 / 6561.0 * slope3
                - 212.0 / 729.0 * slope4
            )
            for value, slope1, slope2, slope3, slope4 in zip(state, k1, k2, k3, k4)
        ]
    )
    k6 = rhs(
        [
            value
            + step
            * (
                9017.0 / 3168.0 * slope1
                - 355.0 / 33.0 * slope2
                + 46732.0 / 5247.0 * slope3
                + 49.0 / 176.0 * slope4
                - 5103.0 / 18656.0 * slope5
            )
            for value, slope1, slope2, slope3, slope4, slope5 in zip(
                state, k1, k2, k3, k4, k5
            )
        ]
    )
    fifth_order = [
        value
        + step
        * (
            35.0 / 384.0 * slope1
            + 500.0 / 1113.0 * slope3
            + 125.0 / 192.0 * slope4
            - 2187.0 / 6784.0 * slope5
            + 11.0 / 84.0 * slope6
        )
        for value, slope1, slope3, slope4, slope5, slope6 in zip(
            state, k1, k3, k4, k5, k6
        )
    ]
    k7 = rhs(fifth_order)
    fourth_order = [
        value
        + step
        * (
            5179.0 / 57600.0 * slope1
            + 7571.0 / 16695.0 * slope3
            + 393.0 / 640.0 * slope4
            - 92097.0 / 339200.0 * slope5
            + 187.0 / 2100.0 * slope6
            + 1.0 / 40.0 * slope7
        )
        for value, slope1, slope3, slope4, slope5, slope6, slope7 in zip(
            state, k1, k3, k4, k5, k6, k7
        )
    ]
    error = [fifth - fourth for fifth, fourth in zip(fifth_order, fourth_order)]
    return fifth_order, error, k1, k7


def _cubic_hermite_state(start, end, start_slope, end_slope, step, fraction):
    """Interpolate an accepted step at one uniform output sample on-device."""

    fraction2 = fraction * fraction
    fraction3 = fraction2 * fraction
    start_weight = 2.0 * fraction3 - 3.0 * fraction2 + 1.0
    start_slope_weight = fraction3 - 2.0 * fraction2 + fraction
    end_weight = -2.0 * fraction3 + 3.0 * fraction2
    end_slope_weight = fraction3 - fraction2
    return (
        start_weight * start
        + (start_slope_weight * step) * start_slope
        + end_weight * end
        + (end_slope_weight * step) * end_slope
    )


def _cubic_hermite_host_state(start, end, start_slope, end_slope, step, fraction):
    """Interpolate an accepted host step at one uniform output sample."""

    fraction2 = fraction * fraction
    fraction3 = fraction2 * fraction
    start_weight = 2.0 * fraction3 - 3.0 * fraction2 + 1.0
    start_slope_weight = fraction3 - 2.0 * fraction2 + fraction
    end_weight = -2.0 * fraction3 + 3.0 * fraction2
    end_slope_weight = fraction3 - fraction2
    return [
        start_weight * start_value
        + start_slope_weight * step * start_derivative
        + end_weight * end_value
        + end_slope_weight * step * end_derivative
        for start_value, end_value, start_derivative, end_derivative in zip(
            start, end, start_slope, end_slope
        )
    ]


def _integrate_uniform_branch(
    initial_state,
    output_count,
    output_step,
    integration_span,
    rhs,
    *,
    rtol,
    atol,
    epsilon,
    max_steps,
):
    """Adaptively evolve a branch and retain uniform output samples on-device."""

    if output_count < 1:
        return torch.empty(
            (0, 13), dtype=initial_state.dtype, device=initial_state.device
        )

    outputs = torch.empty(
        (output_count, 13), dtype=initial_state.dtype, device=initial_state.device
    )
    outputs[0] = initial_state
    if output_count == 1:
        return outputs

    current_state = initial_state
    current_time = 0.0
    next_output_time = output_step
    step_size = output_step
    attempts = 0
    out_idx = 1

    while out_idx < output_count:
        if attempts >= max_steps:
            raise RuntimeError(
                "IMRPhenomTP orbit integration exceeded the maximum step count"
            )
        attempts += 1
        remaining = integration_span - current_time
        if remaining <= 0.0:
            raise RuntimeError(
                "IMRPhenomTP orbit integration ended before its output grid"
            )
        proposed_step = min(step_size, remaining)
        candidate, error, start_slope, end_slope = _dormand_prince_step(
            current_state,
            proposed_step,
            rhs,
        )
        scale = atol + rtol * torch.maximum(
            torch.abs(current_state),
            torch.abs(candidate),
        )
        error_ratio = float(torch.max(torch.abs(error) / scale).item())

        if math.isfinite(error_ratio) and error_ratio <= 1.0:
            previous_time = current_time
            current_time += proposed_step
            tolerance = (
                16.0
                * epsilon
                * max(
                    1.0,
                    abs(current_time),
                    abs(next_output_time),
                )
            )
            while (
                current_time - next_output_time >= -tolerance
                and out_idx < output_count
            ):
                fraction = (next_output_time - previous_time) / proposed_step
                outputs[out_idx] = _cubic_hermite_state(
                    current_state,
                    candidate,
                    start_slope,
                    end_slope,
                    proposed_step,
                    fraction,
                )
                out_idx += 1
                next_output_time += output_step

            current_state = candidate
            factor = (
                5.0
                if error_ratio == 0.0
                else min(5.0, max(0.2, 0.9 * error_ratio ** (-0.2)))
            )
            step_size = proposed_step * factor
            continue

        factor = (
            0.2
            if not math.isfinite(error_ratio)
            else min(0.5, max(0.1, 0.9 * error_ratio ** (-0.25)))
        )
        step_size = proposed_step * factor
        minimum_step = 16.0 * epsilon * max(1.0, abs(current_time))
        if step_size < minimum_step:
            raise RuntimeError(
                "IMRPhenomTP adaptive step size fell below machine precision"
            )

    return outputs


def _imrphenomtp_rhs(
    state,
    sign,
    mass1_fraction,
    mass2_fraction,
    velocity_step,
    velocity,
    linear,
    quadratic,
    cubic,
    spin_coeffs_1,
    spin_coeffs_2,
    eta,
):
    """Evaluate TP's fused historical SpinTaylor vector field on-device."""

    point = state[0]
    idx = min(
        max(math.ceil(float(point.item()) / velocity_step) - 1, 0),
        velocity.numel() - 2,
    )
    offset = point - idx * velocity_step
    orbital_velocity = (
        velocity[idx]
        + linear[idx] * offset
        + quadratic[idx] * offset**2
        + cubic[idx] * offset**3
    )
    lnhat = state[1:4]
    spin1 = state[4:7]
    spin2 = state[7:10]
    e1 = state[10:13]

    lnhat_cross_spin1 = torch.linalg.cross(lnhat, spin1)
    lnhat_cross_spin2 = torch.linalg.cross(lnhat, spin2)
    spin1_cross_spin2 = torch.linalg.cross(spin1, spin2)

    lnhat_dot_spin1 = torch.dot(lnhat, spin1)
    lnhat_dot_spin2 = torch.dot(lnhat, spin2)

    velocity2 = orbital_velocity * orbital_velocity
    velocity5 = velocity2 * velocity2 * orbital_velocity
    velocity6 = velocity5 * orbital_velocity
    velocity7 = velocity6 * orbital_velocity
    velocity9 = velocity7 * velocity2

    spin1_rate = (
        spin_coeffs_1[0] * velocity5
        + spin_coeffs_1[1] * velocity7
        + spin_coeffs_1[2] * velocity9
        - 1.5 * lnhat_dot_spin2 * velocity6
    )
    spin2_rate = (
        spin_coeffs_2[0] * velocity5
        + spin_coeffs_2[1] * velocity7
        + spin_coeffs_2[2] * velocity9
        - 1.5 * lnhat_dot_spin1 * velocity6
    )

    dspin1 = spin1_rate * lnhat_cross_spin1 - (0.5 * velocity6) * spin1_cross_spin2
    dspin2 = spin2_rate * lnhat_cross_spin2 + (0.5 * velocity6) * spin1_cross_spin2

    newtonian_lmag = eta / orbital_velocity
    lmag = newtonian_lmag * (
        1.0
        + velocity2 * (1.5 + eta / 6.0)
        + velocity2**2 * (27.0 / 8.0 - 19.0 / 8.0 * eta + eta**2 / 24.0)
    )
    raw_dlnhat = -(dspin1 + dspin2) / lmag
    precession = torch.linalg.cross(lnhat, raw_dlnhat)
    dlnhat = torch.linalg.cross(precession, lnhat)
    de1 = torch.linalg.cross(precession, e1)

    dstate = torch.cat(
        (
            state.new_tensor([1.0]),
            dlnhat,
            dspin1,
            dspin2,
            de1,
        )
    )
    if sign != 1.0:
        dstate = sign * dstate
    return dstate


def _imrphenomtp_host_rhs(
    state,
    sign,
    mass1_fraction,
    mass2_fraction,
    velocity_step,
    velocity,
    linear,
    quadratic,
    cubic,
):
    """Evaluate TP's fused historical SpinTaylor vector field on the host."""

    orbital_velocity = _uniform_spline_eval(
        state[0],
        velocity_step,
        velocity,
        linear,
        quadratic,
        cubic,
    )
    lnhat = state[1:4]
    spin1 = state[4:7]
    spin2 = state[7:10]
    e1 = state[10:13]
    lnhat_cross_spin1 = _cross(lnhat, spin1)
    lnhat_cross_spin2 = _cross(lnhat, spin2)
    spin1_cross_spin2 = _cross(spin1, spin2)
    lnhat_dot_spin1 = sum(first * second for first, second in zip(lnhat, spin1))
    lnhat_dot_spin2 = sum(first * second for first, second in zip(lnhat, spin2))

    velocity2 = orbital_velocity * orbital_velocity
    velocity5 = velocity2 * velocity2 * orbital_velocity
    velocity6 = velocity5 * orbital_velocity
    velocity7 = velocity6 * orbital_velocity
    velocity9 = velocity7 * velocity2

    def spin_rate(mass_fraction, partner_projection):
        spin3 = 1.5 - mass_fraction - 0.5 * mass_fraction**2
        spin5 = (
            9.0 / 8.0
            - mass_fraction / 2.0
            + 7.0 * mass_fraction**2 / 12.0
            - 7.0 * mass_fraction**3 / 6.0
            - mass_fraction**4 / 24.0
        )
        spin7 = (
            mass_fraction**6 / 48.0
            - 3.0 / 8.0 * mass_fraction**5
            - 39.0 / 16.0 * mass_fraction**4
            - 23.0 / 6.0 * mass_fraction**3
            + 181.0 / 16.0 * mass_fraction**2
            - 51.0 / 8.0 * mass_fraction
            + 27.0 / 16.0
        )
        return (
            spin3 * velocity5
            + spin5 * velocity7
            + spin7 * velocity9
            - 1.5 * partner_projection * velocity6
        )

    spin1_rate = spin_rate(mass1_fraction, lnhat_dot_spin2)
    spin2_rate = spin_rate(mass2_fraction, lnhat_dot_spin1)
    dspin1 = tuple(
        spin1_rate * cross_l - 0.5 * velocity6 * cross_s
        for cross_l, cross_s in zip(lnhat_cross_spin1, spin1_cross_spin2)
    )
    dspin2 = tuple(
        spin2_rate * cross_l + 0.5 * velocity6 * cross_s
        for cross_l, cross_s in zip(lnhat_cross_spin2, spin1_cross_spin2)
    )

    eta = mass1_fraction * mass2_fraction
    newtonian_lmag = eta / orbital_velocity
    lmag = newtonian_lmag * (
        1.0
        + velocity2 * (1.5 + eta / 6.0)
        + velocity2**2 * (27.0 / 8.0 - 19.0 / 8.0 * eta + eta**2 / 24.0)
    )
    raw_dlnhat = tuple(
        -(first + second) / lmag for first, second in zip(dspin1, dspin2)
    )
    precession = _cross(lnhat, raw_dlnhat)
    dlnhat = _cross(precession, lnhat)
    de1 = _cross(precession, e1)
    return [sign * value for value in (1.0, *dlnhat, *dspin1, *dspin2, *de1)]


def evolve_imrphenomtp_orbit(
    core: _IMRPhenomTCore,
    spin1,
    spin2,
    *,
    rtol=None,
    atol=None,
    max_steps=100000,
):
    """Evolve the numerical IMRPhenomTP orbit for an ordered binary.

    ``core`` supplies the aligned-spin IMRPhenomT carrier.  ``spin1`` and
    ``spin2`` are full dimensionless spins at the carrier reference frequency,
    ordered consistently with ``core.binary``.  The z components must match
    those used to construct the carrier.
    """

    spin1, spin2 = _coerce_ordered_spins(core, spin1, spin2)
    dtype = core.binary.dtype
    device = core.binary.device

    epsilon = float(torch.finfo(dtype).eps)
    rtol = max(1.0e-12, 32.0 * epsilon) if rtol is None else float(rtol)
    atol = max(1.0e-12, 32.0 * epsilon) if atol is None else float(atol)
    try:
        max_steps = operator.index(max_steps)
    except TypeError as exc:
        raise ValueError("IMRPhenomTP max_steps must be a positive integer") from exc
    if (
        not math.isfinite(rtol)
        or not math.isfinite(atol)
        or rtol <= 0.0
        or atol <= 0.0
        or max_steps < 1
    ):
        raise ValueError("IMRPhenomTP solver tolerances and max_steps must be positive")

    mass1, mass2 = core.binary[:2].unbind()
    total_mass = mass1 + mass2
    mass1_fraction = mass1 / total_mass
    mass2_fraction = mass2 / total_mass
    velocity = torch.sqrt(core.x_orbital)

    m1_f = float(mass1_fraction.item())
    m2_f = float(mass2_fraction.item())
    eta = m1_f * m2_f

    def calc_spin_coeffs(m_f):
        spin3 = 1.5 - m_f - 0.5 * m_f**2
        spin5 = (
            9.0 / 8.0
            - m_f / 2.0
            + 7.0 * m_f**2 / 12.0
            - 7.0 * m_f**3 / 6.0
            - m_f**4 / 24.0
        )
        spin7 = (
            m_f**6 / 48.0
            - 3.0 / 8.0 * m_f**5
            - 39.0 / 16.0 * m_f**4
            - 23.0 / 6.0 * m_f**3
            + 181.0 / 16.0 * m_f**2
            - 51.0 / 8.0 * m_f
            + 27.0 / 16.0
        )
        return spin3, spin5, spin7

    spin_coeffs_1 = calc_spin_coeffs(m1_f)
    spin_coeffs_2 = calc_spin_coeffs(m2_f)

    initial_vectors = torch.cat(
        (
            core.binary.new_tensor((0.0, 0.0, 1.0)),
            mass1_fraction**2 * spin1,
            mass2_fraction**2 * spin2,
            core.binary.new_tensor((1.0, 0.0, 0.0)),
        )
    )

    minimum_time_m = float(core.time_m[0].item())
    second_time_m = float(core.time_m[1].item())
    reference_time_m = float(core.reference_time_m.item())

    delta_time_m = second_time_m - minimum_time_m
    reference_offset_m = abs(-minimum_time_m + reference_time_m)
    reference_index = math.floor(reference_offset_m / delta_time_m)
    forward_span = -minimum_time_m - delta_time_m - reference_offset_m
    forward_length = math.ceil(forward_span / delta_time_m)
    backward_span = reference_offset_m - delta_time_m
    backward_length = (
        math.ceil(backward_span / delta_time_m) if backward_span > 0.0 else 0
    )
    total_length = reference_index + forward_length
    if forward_length < 1 or total_length > core.time_m.numel():
        raise ValueError("IMRPhenomTP carrier has an invalid precession support grid")

    linear, quadratic, cubic = _natural_cubic_coeff(core.time_m, velocity)

    def branch_rhs(sign):
        def rhs(state):
            return _imrphenomtp_rhs(
                state,
                sign,
                m1_f,
                m2_f,
                delta_time_m,
                velocity,
                linear,
                quadratic,
                cubic,
                spin_coeffs_1,
                spin_coeffs_2,
                eta,
            )

        return rhs

    forward_initial = torch.cat(
        (
            core.binary.new_tensor((reference_offset_m,)),
            initial_vectors,
        )
    )
    forward = _integrate_uniform_branch(
        forward_initial,
        forward_length,
        delta_time_m,
        forward_span,
        branch_rhs(1.0),
        rtol=rtol,
        atol=atol,
        epsilon=epsilon,
        max_steps=max_steps,
    )

    states = torch.zeros((total_length, 12), dtype=dtype, device=device)
    states[reference_index:] = forward[:, 1:]
    if backward_length:
        backward_initial = torch.cat(
            (
                core.binary.new_tensor((reference_offset_m - delta_time_m,)),
                initial_vectors,
            )
        )
        backward = _integrate_uniform_branch(
            backward_initial,
            backward_length,
            delta_time_m,
            backward_span,
            branch_rhs(-1.0),
            rtol=rtol,
            atol=atol,
            epsilon=epsilon,
            max_steps=max_steps,
        )
        states[:backward_length] = torch.flip(backward[:, 1:], dims=[0])

    return IMRPhenomTPOrbit(
        time_m=core.time_m[:total_length],
        velocity=velocity[:total_length],
        lnhat=states[:, :3],
        spin1=states[:, 3:6] / mass1_fraction**2,
        spin2=states[:, 6:9] / mass2_fraction**2,
        e1=states[:, 9:12],
        reference_index=reference_index,
        epoch=core.epoch,
        delta_t=core.inputs.delta_t,
    )


def imrphenomtp_evolved_final_spin(core, orbit):
    """Estimate the remnant spin from TP's evolved peak configuration."""

    if not isinstance(core, _IMRPhenomTCore):
        raise TypeError("IMRPhenomTP final spin requires an IMRPhenomT core")
    if not isinstance(orbit, IMRPhenomTPOrbit):
        raise TypeError("IMRPhenomTP final spin requires an IMRPhenomTP orbit")
    if orbit.lnhat.ndim != 2 or orbit.lnhat.shape[-1] != 3:
        raise ValueError("IMRPhenomTP orbit vectors must have shape (samples, 3)")
    if orbit.lnhat.shape[0] == 0:
        raise ValueError("IMRPhenomTP final spin requires a nonempty orbit")

    mass1, mass2 = core.binary[:2].unbind()
    total_mass = mass1 + mass2
    fraction1 = mass1 / total_mass
    fraction2 = mass2 / total_mass
    norm1 = fraction1**2
    norm2 = fraction2**2
    lnhat = orbit.lnhat[-1]
    spin1 = norm1 * orbit.spin1[-1]
    spin2 = norm2 * orbit.spin2[-1]
    spin1_l = torch.dot(spin1, lnhat)
    spin2_l = torch.dot(spin2, lnhat)
    perpendicular_spin = spin1 - spin1_l * lnhat + spin2 - spin2_l * lnhat
    eta = fraction1 * fraction2
    aligned_final_spin = final_spin_2017(
        eta,
        spin1_l / norm1,
        spin2_l / norm2,
    )
    final_spin = torch.copysign(
        torch.sqrt(
            aligned_final_spin**2 + torch.linalg.vector_norm(perpendicular_spin) ** 2
        ),
        aligned_final_spin,
    )
    return torch.clamp(final_spin, -1.0, 1.0)


def _validate_angle_orbit(core, orbit):
    if not isinstance(core, _IMRPhenomTCore):
        raise TypeError("IMRPhenomTP Euler angles require an IMRPhenomT core")
    if not isinstance(orbit, IMRPhenomTPOrbit):
        raise TypeError("IMRPhenomTP Euler angles require an IMRPhenomTP orbit")
    orbit_length = orbit.time_m.numel()
    if orbit_length < 2 or orbit_length > core.time_m.numel():
        raise ValueError("IMRPhenomTP Euler angles require a valid orbit support")
    if not 0 <= orbit.reference_index < orbit_length:
        raise ValueError("IMRPhenomTP orbit reference index is out of range")
    if any(
        value.device != core.binary.device or value.dtype != core.binary.dtype
        for value in (
            orbit.time_m,
            orbit.lnhat,
            orbit.spin1,
            orbit.spin2,
        )
    ):
        raise ValueError("IMRPhenomTP orbit and carrier must share dtype and device")
    if not torch.allclose(
        orbit.time_m,
        core.time_m[:orbit_length],
        rtol=0.0,
        atol=16.0 * torch.finfo(core.binary.dtype).eps,
    ):
        raise ValueError("IMRPhenomTP orbit must use the carrier time grid")
    return orbit_length


def imrphenomtp_euler_angles(core, orbit, *, convention=1):
    """Construct numerical TP/TPHM Euler angles on the active Torch device.

    The implementation follows the default PhenomTP prescription: it rotates
    the evolved orbital direction into the J frame, applies the reference
    offset, attaches the QNM ringdown continuation, and integrates the
    minimal-rotation angle with Boole's rule over natural cubic splines.
    """

    orbit_length = _validate_angle_orbit(core, orbit)
    mass1, mass2 = core.binary[:2].unbind()
    total_mass = mass1 + mass2
    fraction1 = mass1 / total_mass
    fraction2 = mass2 / total_mass
    spin1_ref = orbit.spin1[orbit.reference_index]
    spin2_ref = orbit.spin2[orbit.reference_index]
    reference_mf = total_mass * lal.MTSUN_SI * core.inputs.f_ref
    frame = spintaylor_j_frame(
        reference_mf,
        mass1,
        mass2,
        spin1_ref,
        spin2_ref,
        inclination=core.inputs.inclination,
        phi_ref=core.inputs.coa_phase,
        convention=convention,
    )

    rotated_lnhat = _rotate_z(
        -frame.kappa,
        _rotate_y(
            -frame.theta_j_source,
            _rotate_z(-frame.phi_j_source, orbit.lnhat),
        ),
    )
    raw_alpha = torch.atan2(rotated_lnhat[:, 1], rotated_lnhat[:, 0])
    weighted_spin = fraction1**2 * spin1_ref + fraction2**2 * spin2_ref
    alpha_offset = torch.atan2(weighted_spin[1], weighted_spin[0]) - math.pi
    shifted_alpha = raw_alpha + alpha_offset - raw_alpha[orbit.reference_index]
    inspiral_alpha = spintaylor_unwrap_angle(shifted_alpha)
    inspiral_cosbeta = rotated_lnhat[:, 2]

    evolved_final_spin = imrphenomtp_evolved_final_spin(core, orbit)
    ringdown_slope = (
        2.0
        * math.pi
        * (qnm_fring_22(evolved_final_spin) - qnm_fring_21(evolved_final_spin))
        / core.final_mass
    )
    ringdown_slope = torch.where(
        evolved_final_spin < 0.0,
        -ringdown_slope,
        ringdown_slope,
    )
    ringdown_time = core.time_m[orbit_length - 1 :]
    ringdown_alpha = inspiral_alpha[orbit_length - 2] + ringdown_slope * ringdown_time
    alpha = torch.cat((inspiral_alpha[: orbit_length - 1], ringdown_alpha))
    cosbeta = torch.cat(
        (
            inspiral_cosbeta[: orbit_length - 1],
            inspiral_cosbeta[orbit_length - 2].expand(
                core.time_m.numel() - orbit_length + 1
            ),
        )
    )

    alpha_linear, alpha_quadratic, alpha_cubic = _natural_cubic_coeff(
        core.time_m, alpha
    )
    cos_linear, cos_quadratic, cos_cubic = _natural_cubic_coeff(
        core.time_m, cosbeta
    )
    left = core.time_m[:-1]
    width = core.time_m[1:] - left
    fractions = core.binary.new_tensor((0.0, 0.25, 0.5, 0.75, 1.0))
    points = left[:, None] + width[:, None] * fractions
    alpha_derivative = _spline_derivative(
        points,
        core.time_m,
        alpha_linear,
        alpha_quadratic,
        alpha_cubic,
    )
    interpolated_cosbeta = _spline_eval(
        points,
        core.time_m,
        cosbeta,
        cos_linear,
        cos_quadratic,
        cos_cubic,
    )
    weights = core.binary.new_tensor((7.0, 32.0, 12.0, 32.0, 7.0))
    gamma_increment = (
        width
        / 90.0
        * torch.sum(
            weights * (-alpha_derivative * interpolated_cosbeta),
            dim=1,
        )
    )
    gamma0 = -alpha[0]
    gamma = torch.cat(
        (
            gamma0.reshape(1),
            gamma0 + torch.cumsum(gamma_increment, dim=0),
        )
    )
    gamma_offset = -gamma[orbit.reference_index] - alpha[orbit.reference_index]
    gamma = torch.cat((gamma[:1], gamma[1:] + gamma_offset))

    return IMRPhenomTPEulerAngles(
        time_m=core.time_m,
        alpha=alpha,
        cosbeta=cosbeta,
        gamma=gamma,
        frame=frame,
        evolved_final_spin=evolved_final_spin,
        reference_index=orbit.reference_index,
        epoch=core.epoch,
        delta_t=core.inputs.delta_t,
    )


__all__ = [
    "IMRPhenomTPEulerAngles",
    "IMRPhenomTPOrbit",
    "evolve_imrphenomtp_orbit",
    "imrphenomtp_euler_angles",
    "imrphenomtp_evolved_final_spin",
    "imrphenomtp_initial_final_spin",
]
