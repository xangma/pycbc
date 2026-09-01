# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch-native amplitude model for IMRPhenomT and IMRPhenomTHM.

This module ports the coefficient construction and analytic amplitude ansatz
functions for the (2, 2), (2, 1), (3, 3), (4, 4), and (5, 5) modes from
``LALSimIMRPhenomTHM_internals.c`` in LALSuite 7.26.1. All operations remain
on the input Torch device and support batched leading dimensions.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

from . import imrphenomt_fits_torch as fits
from .imrphenomt_phase_torch import (
    IMRPhenomTPhase22Coefficients,
    omega22,
    omega22_derivative,
)

_EULER_GAMMA = 0.5772156649015329
_PI = math.pi
_INSPIRAL_TIMES = (-2000.0, -250.0, -150.0)
_AMPLITUDE_CUT_TIME = -150.0
_MERGER_COLLOCATION_TIME = -25.0
_SUPPORTED_MODES = ((2, 2), (2, 1), (3, 3), (4, 4), (5, 5))


class IMRPhenomTHMAmplitudeCoefficients(NamedTuple):
    """Amplitude coefficients for one IMRPhenomT/THM mode.

    ``pn_real`` and ``pn_imag`` contain the coefficients of powers
    ``x**(n/2)`` for ``n = 0, ..., 7``. ``inspiral`` contains the added
    ``x**4``, ``x**(9/2)``, and ``x**5`` coefficients; ``merger`` contains
    the four coefficients of the hyperbolic-secant ansatz.
    """

    fac0: torch.Tensor
    pn_real: torch.Tensor
    pn_imag: torch.Tensor
    amp_log: torch.Tensor
    inspiral: torch.Tensor
    merger: torch.Tensor
    t_shift: torch.Tensor
    alpha1_rd: torch.Tensor
    alpha2_rd: torch.Tensor
    alpha21_rd: torch.Tensor
    alpha1_rd_prec: torch.Tensor
    alpha2_rd_prec: torch.Tensor
    alpha21_rd_prec: torch.Tensor
    c1: torch.Tensor
    c2: torch.Tensor
    c3: torch.Tensor
    c4: torch.Tensor
    c1_prec: torch.Tensor
    c2_prec: torch.Tensor
    c4_prec: torch.Tensor
    omega_cut_pn_amp: torch.Tensor
    phi_cut_pn_amp: torch.Tensor


# Preserve the original dominant-mode API for callers introduced with the
# IMRPhenomT port.
IMRPhenomTAmplitude22Coefficients = IMRPhenomTHMAmplitudeCoefficients


class _ModeAmplitudeData(NamedTuple):
    pn_real: torch.Tensor
    pn_imag: torch.Tensor
    amp_log: torch.Tensor
    inspiral_points: torch.Tensor
    merger_point: torch.Tensor
    peak_amplitude: torch.Tensor
    ringdown_c3: torch.Tensor
    time_shift: torch.Tensor
    fdamp: torch.Tensor
    fdamp_n2: torch.Tensor
    fdamp_prec: torch.Tensor
    fdamp_n2_prec: torch.Tensor


def _coerce_like_phase(coefficients, *values):
    converted = []
    for value in values:
        if isinstance(value, torch.Tensor):
            if value.device != coefficients.eta.device:
                raise ValueError(
                    "IMRPhenomT amplitude and phase parameters must be on "
                    "one Torch device"
                )
            if value.is_complex():
                raise TypeError("IMRPhenomT amplitude parameters must be real")
        elif isinstance(value, complex):
            raise TypeError("IMRPhenomT amplitude parameters must be real")
        converted.append(
            torch.as_tensor(
                value,
                device=coefficients.eta.device,
                dtype=coefficients.eta.dtype,
            )
        )
    broadcast = torch.broadcast_tensors(coefficients.eta, *converted)
    if broadcast[0].shape != coefficients.eta.shape:
        raise ValueError(
            "amplitude parameters may not add batch dimensions absent from "
            "the phase coefficients"
        )
    return broadcast[1:]


def _mode_amplitude_data(
    mode,
    eta,
    chi1z,
    chi2z,
    shat,
    dchi,
    delta,
    final_spin,
    final_spin_prec,
):
    """Return the PN, calibration, and QNM data specific to ``mode``."""
    zero = torch.zeros_like(eta)
    one = torch.ones_like(eta)

    if mode == (2, 2):
        amp_1pn_real = -2.5476190476190474 + 55.0 * eta / 42.0
        amp_1half_pn_real = (
            -2.0 * chi1z / 3.0
            - 2.0 * chi2z / 3.0
            - 2.0 * chi1z * delta / 3.0
            + 2.0 * chi2z * delta / 3.0
            + 2.0 * chi1z * eta / 3.0
            + 2.0 * chi2z * eta / 3.0
            + 2.0 * _PI
        )
        amp_2pn_real = (
            -1.437169312169312
            + chi1z**2 / 2.0
            + chi2z**2 / 2.0
            + chi1z**2 * delta / 2.0
            - chi2z**2 * delta / 2.0
            - 1069.0 * eta / 216.0
            - chi1z**2 * eta
            + 2.0 * chi1z * chi2z * eta
            - chi2z**2 * eta
            + 2047.0 * eta**2 / 1512.0
        )
        amp_2half_pn_real = -107.0 * _PI / 21.0 + 34.0 * eta * _PI / 21.0
        amp_3pn_real = (
            41.78634662956092
            - 278185.0 * eta / 33264.0
            - 20261.0 * eta**2 / 2772.0
            + 114635.0 * eta**3 / 99792.0
            - 856.0 * _EULER_GAMMA / 105.0
            + 2.0 * _PI**2 / 3.0
            + 41.0 * eta * _PI**2 / 96.0
        )
        amp_3half_pn_real = (
            -2173.0 * _PI / 756.0
            - 2495.0 * eta * _PI / 378.0
            + 40.0 * eta**2 * _PI / 27.0
        )
        pn_real = torch.stack(
            (
                one,
                zero,
                amp_1pn_real,
                amp_1half_pn_real,
                amp_2pn_real,
                amp_2half_pn_real,
                amp_3pn_real,
                amp_3half_pn_real,
            ),
            dim=-1,
        )
        pn_imag = torch.stack(
            (
                zero,
                zero,
                zero,
                zero,
                zero,
                -24.0 * eta,
                eta.new_full(eta.shape, 428.0 * _PI / 105.0),
                14333.0 * eta / 162.0 - 4066.0 * eta**2 / 945.0,
            ),
            dim=-1,
        )
        amp_log = eta.new_full(eta.shape, -428.0 / 105.0)
        inspiral_points = torch.stack(
            (
                fits.inspiral_amp_cp1_22(eta, shat, dchi, delta),
                fits.inspiral_amp_cp2_22(eta, shat, dchi, delta),
                fits.inspiral_amp_cp3_22(eta, shat, dchi, delta),
            ),
            dim=-1,
        )
        merger_point = fits.merger_amp_cp1_22(eta, shat, dchi, delta)
        peak_amplitude = fits.peak_amplitude_22(eta, shat, dchi, delta)
        ringdown_c3 = fits.ringdown_amp_c3_22(eta, shat)
        time_shift = zero
        fdamp = fits.qnm_fdamp_22(final_spin)
        fdamp_n2 = fits.qnm_fdamp_22_n2(final_spin)
        fdamp_prec = fits.qnm_fdamp_22(final_spin_prec)
        fdamp_n2_prec = fits.qnm_fdamp_22_n2(final_spin_prec)
    elif mode == (2, 1):
        pn_real = torch.stack(
            (
                zero,
                delta / 3.0,
                -chi1z / 4.0 + chi2z / 4.0 - chi1z * delta / 4.0 - chi2z * delta / 4.0,
                -17.0 * delta / 84.0 + 5.0 * delta * eta / 21.0,
                79.0 * chi1z / 84.0
                - 79.0 * chi2z / 84.0
                + 79.0 * chi1z * delta / 84.0
                + 79.0 * chi2z * delta / 84.0
                - 43.0 * chi1z / 42.0
                + 43.0 * chi2z / 42.0
                - 43.0 * chi1z * delta / 42.0
                - 43.0 * chi2z * delta / 42.0
                - 139.0 * chi1z * eta / 84.0
                + 139.0 * chi2z * eta / 84.0
                - 139.0 * chi1z * delta * eta / 84.0
                - 139.0 * chi2z * delta * eta / 84.0
                + 86.0 * chi1z * eta / 21.0
                - 86.0 * chi2z * eta / 21.0
                + 43.0 * chi1z * delta * eta / 21.0
                + 43.0 * chi2z * delta * eta / 21.0
                + delta * _PI / 3.0,
                -43.0 * delta / 378.0
                - 509.0 * delta * eta / 378.0
                + 79.0 * delta * eta**2 / 504.0,
                -17.0 * delta * _PI / 84.0 + delta * eta * _PI / 14.0,
                zero,
            ),
            dim=-1,
        )
        pn_imag = torch.stack(
            (
                zero,
                zero,
                zero,
                zero,
                -delta / 6.0 - 2.0 * delta * math.log(2.0) / 3.0,
                zero,
                -(
                    -17.0 * delta / 168.0
                    + 353.0 * delta * eta / 84.0
                    - 17.0 * delta * math.log(2.0) / 42.0
                    + delta * eta * math.log(2.0) / 7.0
                ),
                zero,
            ),
            dim=-1,
        )
        amp_log = zero
        inspiral_points = torch.stack(
            (
                fits.inspiral_amp_cp1_21(eta, shat, dchi, delta),
                fits.inspiral_amp_cp2_21(eta, shat, dchi, delta),
                fits.inspiral_amp_cp3_21(eta, shat, dchi, delta),
            ),
            dim=-1,
        )
        merger_point = fits.merger_amp_cp1_21(eta, shat, dchi, delta)
        peak_amplitude = fits.peak_amplitude_21(eta, shat, dchi, delta)
        ringdown_c3 = fits.ringdown_amp_c3_21(eta, shat, dchi)
        time_shift = fits.time_shift_21(eta, shat, dchi)
        fdamp = fits.qnm_fdamp_21(final_spin)
        fdamp_n2 = fits.qnm_fdamp_21_n2(final_spin)
        fdamp_prec = fits.qnm_fdamp_21(final_spin_prec)
        fdamp_n2_prec = fits.qnm_fdamp_21_n2(final_spin_prec)
    elif mode == (3, 3):
        pn_real = torch.stack(
            (
                zero,
                0.7763237542601484 * delta,
                zero,
                -3.1052950170405937 * delta + 1.5526475085202969 * delta * eta,
                -(
                    -0.5822428156951114 * chi1z
                    + 0.5822428156951114 * chi2z
                    - 7.316679009572791 * delta
                    - 0.5822428156951114 * chi1z * delta
                    - 0.5822428156951114 * chi2z * delta
                    + 1.3585665699552598 * chi1z
                    - 1.3585665699552598 * chi2z
                    + 1.3585665699552598 * chi1z * delta
                    + 1.3585665699552598 * chi2z * delta
                    + 1.7467284470853341 * chi1z * eta
                    - 1.7467284470853341 * chi2z * eta
                    + 1.7467284470853341 * chi1z * delta * eta
                    + 1.7467284470853341 * chi2z * delta * eta
                    - 5.434266279821039 * chi1z * eta
                    + 5.434266279821039 * chi2z * eta
                    - 2.7171331399105196 * chi1z * delta * eta
                    - 2.7171331399105196 * chi2z * delta * eta
                ),
                -(
                    -0.08680711070363478 * delta
                    + 8.647776123213047 * delta * eta
                    - 2.0866641516022777 * delta * eta**2
                ),
                zero,
                zero,
            ),
            dim=-1,
        )
        pn_imag = torch.stack(
            (
                zero,
                zero,
                zero,
                zero,
                -1.371926598204461 * delta,
                zero,
                zero,
                zero,
            ),
            dim=-1,
        )
        amp_log = zero
        inspiral_points = torch.stack(
            (
                fits.inspiral_amp_cp1_33(eta, shat, dchi, delta),
                fits.inspiral_amp_cp2_33(eta, shat, dchi, delta),
                fits.inspiral_amp_cp3_33(eta, shat, dchi, delta),
            ),
            dim=-1,
        )
        merger_point = fits.merger_amp_cp1_33(eta, shat, dchi, delta)
        peak_amplitude = fits.peak_amplitude_33(eta, shat, dchi, delta)
        ringdown_c3 = fits.ringdown_amp_c3_33(eta, shat)
        time_shift = fits.time_shift_33(eta, shat)
        fdamp = fits.qnm_fdamp_33(final_spin)
        fdamp_n2 = fits.qnm_fdamp_33_n2(final_spin)
        fdamp_prec = fits.qnm_fdamp_33(final_spin_prec)
        fdamp_n2_prec = fits.qnm_fdamp_33_n2(final_spin_prec)
    elif mode == (4, 4):
        pn_real = torch.stack(
            (
                zero,
                zero,
                0.751248226425348 * (1.0 - 3.0 * eta),
                zero,
                -4.049910893365739
                + 14.489984730901032 * eta
                - 5.9758381647470875 * eta**2,
                0.751248226425348 * (4.0 * _PI - 12.0 * eta * _PI),
                8.0
                * math.sqrt(0.7142857142857143)
                * (
                    5.338016983016983
                    - 1088119.0 * eta / 28600.0
                    + 146879.0 * eta**2 / 2340.0
                    - 226097.0 * eta**3 / 17160.0
                )
                / 9.0,
                zero,
            ),
            dim=-1,
        )
        pn_imag = torch.stack(
            (
                zero,
                zero,
                zero,
                zero,
                zero,
                0.751248226425348 * (-2.854822555520438 + 13.189467666561313 * eta),
                zero,
                zero,
            ),
            dim=-1,
        )
        amp_log = zero
        inspiral_points = torch.stack(
            (
                fits.inspiral_amp_cp1_44(eta, shat, dchi, delta),
                fits.inspiral_amp_cp2_44(eta, shat, dchi, delta),
                fits.inspiral_amp_cp3_44(eta, shat, dchi, delta),
            ),
            dim=-1,
        )
        merger_point = fits.merger_amp_cp1_44(eta, shat, dchi, delta)
        peak_amplitude = fits.peak_amplitude_44(eta, shat, dchi, delta)
        ringdown_c3 = fits.ringdown_amp_c3_44(eta, shat)
        time_shift = fits.time_shift_44(eta, shat)
        fdamp = fits.qnm_fdamp_44(final_spin)
        fdamp_n2 = fits.qnm_fdamp_44_n2(final_spin)
        fdamp_prec = fits.qnm_fdamp_44(final_spin_prec)
        fdamp_n2_prec = fits.qnm_fdamp_44_n2(final_spin_prec)
    else:
        pn_real = torch.stack(
            (
                zero,
                zero,
                zero,
                0.8013768943966973 * delta * (1.0 - 2.0 * eta),
                zero,
                0.8013768943966973
                * delta
                * (-6.743589743589744 + 688.0 * eta / 39.0 - 256.0 * eta**2 / 39.0),
                12.58799882096634 * delta - 25.175997641932675 * delta * eta,
                zero,
            ),
            dim=-1,
        )
        pn_imag = torch.stack(
            (
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
                -3.0177162096765713 * delta + 12.454250695829877 * delta * eta,
                zero,
            ),
            dim=-1,
        )
        amp_log = zero
        inspiral_points = torch.stack(
            (
                fits.inspiral_amp_cp1_55(eta, shat, dchi, delta),
                fits.inspiral_amp_cp2_55(eta, shat, dchi, delta),
                fits.inspiral_amp_cp3_55(eta, shat, dchi, delta),
            ),
            dim=-1,
        )
        merger_point = fits.merger_amp_cp1_55(eta, shat, dchi, delta)
        peak_amplitude = fits.peak_amplitude_55(eta, shat, dchi, delta)
        ringdown_c3 = fits.ringdown_amp_c3_55(eta, shat, dchi)
        time_shift = fits.time_shift_55(eta, shat)
        fdamp = fits.qnm_fdamp_55(final_spin)
        fdamp_n2 = fits.qnm_fdamp_55_n2(final_spin)
        fdamp_prec = fits.qnm_fdamp_55(final_spin_prec)
        fdamp_n2_prec = fits.qnm_fdamp_55_n2(final_spin_prec)

    return _ModeAmplitudeData(
        pn_real=pn_real,
        pn_imag=pn_imag,
        amp_log=amp_log,
        inspiral_points=inspiral_points,
        merger_point=merger_point,
        peak_amplitude=peak_amplitude,
        ringdown_c3=ringdown_c3,
        time_shift=time_shift,
        fdamp=fdamp,
        fdamp_n2=fdamp_n2,
        fdamp_prec=fdamp_prec,
        fdamp_n2_prec=fdamp_n2_prec,
    )


def _series_parts(x, pn_real, pn_imag, amp_log, inspiral):
    powers = torch.stack(tuple(x ** (0.5 * order) for order in range(8)), -1)
    real = (pn_real * powers).sum(dim=-1)
    imag = (pn_imag * powers).sum(dim=-1)
    real = real + amp_log * torch.log(16.0 * x) * x**3
    c1, c2, c3 = inspiral.unbind(dim=-1)
    real = real + c1 * x**4 + c2 * x**4.5 + c3 * x**5
    return real, imag


def _series_derivative(x, pn_real, pn_imag, amp_log, inspiral):
    derivative_powers = torch.stack(
        tuple(0.5 * order * x ** (0.5 * order - 1.0) for order in range(1, 8)),
        dim=-1,
    )
    real = (pn_real[..., 1:] * derivative_powers).sum(dim=-1)
    imag = (pn_imag[..., 1:] * derivative_powers).sum(dim=-1)
    real = real + amp_log * x**2 * (3.0 * torch.log(16.0 * x) + 1.0)
    c1, c2, c3 = inspiral.unbind(dim=-1)
    real = real + 4.0 * c1 * x**3 + 4.5 * c2 * x**3.5 + 5.0 * c3 * x**4
    return real, imag


def _inspiral_amplitude(x, fac0, pn_real, pn_imag, amp_log, inspiral):
    real, imag = _series_parts(x, pn_real, pn_imag, amp_log, inspiral)
    return torch.complex(fac0 * x * real, fac0 * x * imag)


def _inspiral_amplitude_derivative(x, fac0, pn_real, pn_imag, amp_log, inspiral):
    real, imag = _series_parts(x, pn_real, pn_imag, amp_log, inspiral)
    derivative_real, derivative_imag = _series_derivative(
        x, pn_real, pn_imag, amp_log, inspiral
    )
    return torch.complex(
        fac0 * (real + x * derivative_real),
        fac0 * (imag + x * derivative_imag),
    )


def _stable_quadratic_coefficients(points, values):
    """Fit ``c0 + c1*y + c2*y**2`` with a well-scaled solve."""
    center = points.mean(dim=-1, keepdim=True)
    scale = (points - center).abs().amax(dim=-1, keepdim=True)
    normalized = (points - center) / scale
    matrix = torch.stack(
        (torch.ones_like(normalized), normalized, normalized**2), dim=-1
    )
    transformed = torch.linalg.solve(matrix, values.unsqueeze(-1)).squeeze(-1)
    a0, a1, a2 = transformed.unbind(dim=-1)
    center = center.squeeze(-1)
    scale = scale.squeeze(-1)
    c2 = a2 / scale**2
    c1 = a1 / scale - 2.0 * center * c2
    c0 = a0 - center * a1 / scale + center**2 * c2
    return torch.stack((c0, c1, c2), dim=-1)


def _scaled_linear_solve(matrix, rhs):
    """Solve a small system after independent row and column scaling."""
    row_scale = matrix.abs().amax(dim=-1).clamp_min(torch.finfo(matrix.dtype).tiny)
    matrix = matrix / row_scale.unsqueeze(-1)
    rhs = rhs / row_scale
    column_scale = matrix.abs().amax(dim=-2).clamp_min(torch.finfo(matrix.dtype).tiny)
    solution = torch.linalg.solve(
        matrix / column_scale.unsqueeze(-2), rhs.unsqueeze(-1)
    ).squeeze(-1)
    return solution / column_scale


def build_hm_amplitude_coefficients(
    mode,
    chi1z,
    chi2z,
    final_mass,
    final_spin,
    phase_coefficients: IMRPhenomTPhase22Coefficients,
    *,
    final_spin_prec=None,
):
    """Build coefficients for one supported IMRPhenomT/THM amplitude mode.

    The phase coefficients must describe the same binary parameters. ``chi1z``
    belongs to the larger body, matching LALSuite's IMRPhenomT convention.
    ``final_mass`` is the remnant mass divided by total mass.
    """
    try:
        mode = tuple(mode)
    except TypeError as exc:
        raise ValueError(f"unsupported IMRPhenomTHM mode {mode!r}") from exc
    if mode not in _SUPPORTED_MODES:
        raise ValueError(
            f"unsupported IMRPhenomTHM mode {mode!r}; "
            f"available modes are {_SUPPORTED_MODES}"
        )
    if final_spin_prec is None:
        final_spin_prec = final_spin
    chi1z, chi2z, final_mass, final_spin, final_spin_prec = _coerce_like_phase(
        phase_coefficients,
        chi1z,
        chi2z,
        final_mass,
        final_spin,
        final_spin_prec,
    )
    eta = phase_coefficients.eta
    zero = torch.zeros_like(eta)
    one = torch.ones_like(eta)
    delta = torch.sqrt(torch.clamp(1.0 - 4.0 * eta, min=0.0))
    mass1 = 0.5 * (1.0 + delta)
    mass2 = 0.5 * (1.0 - delta)
    shat = (mass1**2 * chi1z + mass2**2 * chi2z) / (mass1**2 + mass2**2)
    dchi = chi1z - chi2z

    fac0 = 2.0 * eta * math.sqrt(16.0 * _PI / 5.0)
    mode_data = _mode_amplitude_data(
        mode,
        eta,
        chi1z,
        chi2z,
        shat,
        dchi,
        delta,
        final_spin,
        final_spin_prec,
    )
    pn_real = mode_data.pn_real
    pn_imag = mode_data.pn_imag
    amp_log = mode_data.amp_log
    t_shift = mode_data.time_shift

    x_points = []
    pn_offsets = []
    empty_inspiral = eta.new_zeros(eta.shape + (3,))
    for time in _INSPIRAL_TIMES:
        t = eta.new_full(eta.shape, time)
        x = (0.5 * omega22(t, phase_coefficients)) ** (2.0 / 3.0)
        x_points.append(x)
        pn_offsets.append(
            _inspiral_amplitude(x, fac0, pn_real, pn_imag, amp_log, empty_inspiral).real
        )
    x_points = torch.stack(tuple(x_points), dim=-1)
    pn_offsets = torch.stack(tuple(pn_offsets), dim=-1)
    values = (mode_data.inspiral_points - pn_offsets) / (fac0.unsqueeze(-1) * x_points)
    inspiral = _stable_quadratic_coefficients(
        torch.sqrt(x_points), values / x_points**4
    )

    amp_peak = mode_data.peak_amplitude
    c3 = mode_data.ringdown_c3
    alpha1_rd = 2.0 * _PI * mode_data.fdamp / final_mass
    alpha2_rd = 2.0 * _PI * mode_data.fdamp_n2 / final_mass
    alpha21_rd = 0.5 * (alpha2_rd - alpha1_rd)
    alpha1_rd_prec = 2.0 * _PI * mode_data.fdamp_prec / final_mass
    alpha2_rd_prec = 2.0 * _PI * mode_data.fdamp_n2_prec / final_mass
    alpha21_rd_prec = 0.5 * (alpha2_rd_prec - alpha1_rd_prec)
    tanh_c3 = torch.tanh(c3)
    cosh_c3 = torch.cosh(c3)
    c2 = alpha21_rd
    c2_prec = alpha21_rd_prec
    c2 = torch.where(
        c2.abs() > (0.5 * alpha1_rd / tanh_c3).abs(),
        -0.5 * alpha1_rd / tanh_c3,
        c2,
    )
    c2_prec = torch.where(
        c2_prec.abs() > (0.5 * alpha1_rd_prec / tanh_c3).abs(),
        -0.5 * alpha1_rd_prec / tanh_c3,
        c2_prec,
    )
    c1 = amp_peak * alpha1_rd * cosh_c3**2 / c2
    c1_prec = amp_peak * alpha1_rd_prec * cosh_c3**2 / c2_prec
    c4 = amp_peak - c1 * tanh_c3
    c4_prec = amp_peak - c1_prec * tanh_c3

    t_cut = eta.new_full(eta.shape, _AMPLITUDE_CUT_TIME)
    omega_cut = omega22(t_cut, phase_coefficients)
    domega_cut = omega22_derivative(t_cut, phase_coefficients)
    x_cut = (0.5 * omega_cut) ** (2.0 / 3.0)
    dx_dt = domega_cut * (0.5 * omega_cut) ** (-1.0 / 3.0) / 3.0
    amplitude_cut = _inspiral_amplitude(
        x_cut, fac0, pn_real, pn_imag, amp_log, inspiral
    )
    amplitude_cut_derivative = _inspiral_amplitude_derivative(
        x_cut, fac0, pn_real, pn_imag, amp_log, inspiral
    )
    amplitude_cut_abs = amplitude_cut.abs()
    nonzero_amplitude = amplitude_cut_abs > 0.0
    safe_amplitude_abs = torch.where(nonzero_amplitude, amplitude_cut_abs, one)
    magnitude_derivative = (
        amplitude_cut.real * amplitude_cut_derivative.real
        + amplitude_cut.imag * amplitude_cut_derivative.imag
    ) / safe_amplitude_abs
    amplitude_sign = torch.where(amplitude_cut.real < 0.0, -one, one)
    signed_amplitude_cut = amplitude_sign * amplitude_cut_abs
    signed_amplitude_derivative = amplitude_sign * magnitude_derivative * dx_dt

    merger_time = eta.new_full(eta.shape, _MERGER_COLLOCATION_TIME)

    def merger_basis(time):
        tau = time - t_shift
        sech1 = torch.reciprocal(torch.cosh(alpha1_rd * tau))
        sech2 = torch.reciprocal(torch.cosh(2.0 * alpha1_rd * tau))
        return torch.stack((one, sech1, sech2 ** (1.0 / 7.0), tau**2), -1)

    tau_cut = t_cut - t_shift
    sech1_cut = torch.reciprocal(torch.cosh(alpha1_rd * tau_cut))
    sech2_cut = torch.reciprocal(torch.cosh(2.0 * alpha1_rd * tau_cut))
    derivative_basis = torch.stack(
        (
            zero,
            -alpha1_rd * sech1_cut * torch.tanh(alpha1_rd * tau_cut),
            -2.0
            * alpha1_rd
            / 7.0
            * torch.sinh(2.0 * alpha1_rd * tau_cut)
            * sech2_cut ** (8.0 / 7.0),
            2.0 * tau_cut,
        ),
        dim=-1,
    )
    merger_matrix = torch.stack(
        (
            merger_basis(t_cut),
            merger_basis(merger_time),
            torch.stack((one, one, one, zero), dim=-1),
            derivative_basis,
        ),
        dim=-2,
    )
    merger_rhs = torch.stack(
        (
            signed_amplitude_cut,
            mode_data.merger_point,
            amp_peak,
            signed_amplitude_derivative,
        ),
        dim=-1,
    )
    merger = _scaled_linear_solve(merger_matrix, merger_rhs)

    orientation_derivative = (
        amplitude_cut.real * amplitude_cut_derivative.imag
        - amplitude_cut.imag * amplitude_cut_derivative.real
    ) / safe_amplitude_abs**2
    omega_cut_pn_amp = -orientation_derivative * dx_dt
    phi_cut_pn_amp = torch.atan2(amplitude_cut.imag, amplitude_cut.real)
    phi_cut_pn_amp = torch.where(
        amplitude_cut.real < 0.0, phi_cut_pn_amp + _PI, phi_cut_pn_amp
    )

    return IMRPhenomTHMAmplitudeCoefficients(
        fac0=fac0,
        pn_real=pn_real,
        pn_imag=pn_imag,
        amp_log=amp_log,
        inspiral=inspiral,
        merger=merger,
        t_shift=t_shift,
        alpha1_rd=alpha1_rd,
        alpha2_rd=alpha2_rd,
        alpha21_rd=alpha21_rd,
        alpha1_rd_prec=alpha1_rd_prec,
        alpha2_rd_prec=alpha2_rd_prec,
        alpha21_rd_prec=alpha21_rd_prec,
        c1=c1,
        c2=c2,
        c3=c3,
        c4=c4,
        c1_prec=c1_prec,
        c2_prec=c2_prec,
        c4_prec=c4_prec,
        omega_cut_pn_amp=omega_cut_pn_amp,
        phi_cut_pn_amp=phi_cut_pn_amp,
    )


def build_amplitude22_coefficients(
    chi1z,
    chi2z,
    final_mass,
    final_spin,
    phase_coefficients: IMRPhenomTPhase22Coefficients,
    *,
    final_spin_prec=None,
):
    """Build dominant-mode amplitude coefficients on the Torch device."""
    return build_hm_amplitude_coefficients(
        (2, 2),
        chi1z,
        chi2z,
        final_mass,
        final_spin,
        phase_coefficients,
        final_spin_prec=final_spin_prec,
    )


def inspiral_amplitude(x, coefficients):
    """Evaluate the complex inspiral amplitude at PN parameter ``x``."""
    x = torch.as_tensor(
        x, device=coefficients.fac0.device, dtype=coefficients.fac0.dtype
    )
    return _inspiral_amplitude(
        x,
        coefficients.fac0,
        coefficients.pn_real,
        coefficients.pn_imag,
        coefficients.amp_log,
        coefficients.inspiral,
    )


def complex_amplitude_orientation(x, coefficients):
    """Evaluate the phase contribution from the complex inspiral amplitude."""
    x = torch.as_tensor(
        x, device=coefficients.fac0.device, dtype=coefficients.fac0.dtype
    )
    real, imag = _series_parts(
        x,
        coefficients.pn_real,
        coefficients.pn_imag,
        coefficients.amp_log,
        coefficients.inspiral,
    )
    return torch.atan2(imag, real)


def merger_amplitude(t, coefficients):
    """Evaluate the real merger amplitude at dimensionless time ``t``."""
    t = torch.as_tensor(
        t, device=coefficients.fac0.device, dtype=coefficients.fac0.dtype
    )
    tau = t - coefficients.t_shift
    c1, c2, c3, c4 = coefficients.merger.unbind(dim=-1)
    sech1 = torch.reciprocal(torch.cosh(coefficients.alpha1_rd * tau))
    sech2 = torch.reciprocal(torch.cosh(2.0 * coefficients.alpha1_rd * tau))
    return c1 + c2 * sech1 + c3 * sech2 ** (1.0 / 7.0) + c4 * tau**2


def ringdown_amplitude(t, coefficients):
    """Evaluate the real ringdown amplitude at dimensionless time ``t``."""
    t = torch.as_tensor(
        t, device=coefficients.fac0.device, dtype=coefficients.fac0.dtype
    )
    tau = t - coefficients.t_shift
    return torch.exp(-coefficients.alpha1_rd_prec * tau) * (
        coefficients.c1_prec * torch.tanh(coefficients.c2_prec * tau + coefficients.c3)
        + coefficients.c4_prec
    )


def amplitude_lm(t, x, coefficients):
    """Evaluate one piecewise complex IMRPhenomT/THM mode amplitude."""
    t = torch.as_tensor(
        t, device=coefficients.fac0.device, dtype=coefficients.fac0.dtype
    )
    inspiral = inspiral_amplitude(x, coefficients)
    merger = merger_amplitude(t, coefficients)
    safe_ringdown_t = torch.where(t > coefficients.t_shift, t, coefficients.t_shift)
    ringdown = ringdown_amplitude(safe_ringdown_t, coefficients)
    merger = torch.complex(merger, torch.zeros_like(merger))
    ringdown = torch.complex(ringdown, torch.zeros_like(ringdown))
    return torch.where(
        t < _AMPLITUDE_CUT_TIME,
        inspiral,
        torch.where(t > coefficients.t_shift, ringdown, merger),
    )


def amplitude22(t, x, coefficients):
    """Evaluate the piecewise complex (2, 2) amplitude."""
    return amplitude_lm(t, x, coefficients)
