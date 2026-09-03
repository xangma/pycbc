# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch-native higher-mode phase model for IMRPhenomTHM.

This module ports the higher-mode merger and ringdown coefficient
construction from ``LALSimIMRPhenomTHM_internals.c`` in LALSuite 7.26.1.
The inspiral carrier phase is obtained by rescaling the native Torch (2, 2)
phase, exactly as in LALSuite. All numerical work stays on the input Torch
device and supports batched leading dimensions.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

from . import imrphenomt_fits_torch as fits
from .imrphenomt_amplitude_torch import (
    IMRPhenomTHMAmplitudeCoefficients,
    _coerce_like_phase,
    _scaled_linear_solve,
)
from .imrphenomt_phase_torch import (
    IMRPhenomTPhase22Coefficients,
    _merger_phase,
    _ringdown_omega_derivative,
    merger_omega as _shared_merger_omega,
    merger_phase as _shared_merger_phase,
    omega22,
    omega22_derivative,
    phase22,
    ringdown_omega as _shared_ringdown_omega,
    ringdown_phase as _shared_ringdown_phase,
)

_PI = math.pi
_PHASE_CUT_TIME = -150.0
_MERGER_COLLOCATION_TIME = -25.0
_SUPPORTED_MODES = ((2, 1), (3, 3), (4, 4), (5, 5))


class IMRPhenomTHMPhaseCoefficients(NamedTuple):
    """Merger and ringdown phase coefficients for one higher mode."""

    eta: torch.Tensor
    m: torch.Tensor
    omega_peak: torch.Tensor
    c1: torch.Tensor
    c1_prec: torch.Tensor
    c2: torch.Tensor
    c3: torch.Tensor
    c4: torch.Tensor
    omega_merger: torch.Tensor
    alpha1_rd: torch.Tensor
    alpha1_rd_prec: torch.Tensor
    alpha2_rd: torch.Tensor
    alpha21_rd: torch.Tensor
    domega_peak: torch.Tensor
    omega_ring: torch.Tensor
    omega_ring_prec: torch.Tensor
    phase_offset_merger: torch.Tensor
    phase_offset_ringdown: torch.Tensor
    t_cut: torch.Tensor


class _ModePhaseData(NamedTuple):
    omega_ring: torch.Tensor
    omega_ring_prec: torch.Tensor
    alpha1_rd: torch.Tensor
    alpha1_rd_prec: torch.Tensor
    alpha2_rd: torch.Tensor
    omega_merger_cp: torch.Tensor
    omega_peak: torch.Tensor
    c2: torch.Tensor
    c3: torch.Tensor


def _mode_phase_data(
    mode,
    eta,
    shat,
    dchi,
    delta,
    final_mass,
    final_spin,
    final_spin_prec,
):
    if mode == (2, 1):
        fring = fits.qnm_fring_21(final_spin)
        fring_prec = fits.qnm_fring_21(final_spin_prec)
        fdamp = fits.qnm_fdamp_21(final_spin)
        fdamp_prec = fits.qnm_fdamp_21(final_spin_prec)
        fdamp_n2 = fits.qnm_fdamp_21_n2(final_spin)
        merger_frequency = fits.merger_freq_cp1_21(eta, shat, dchi, delta)
        omega_peak = fits.peak_frequency_21(eta, shat, dchi, delta)
        c2 = fits.ringdown_freq_d2_21(eta, shat, dchi, delta)
        c3 = fits.ringdown_freq_d3_21(eta, shat, dchi, delta)
    elif mode == (3, 3):
        fring = fits.qnm_fring_33(final_spin)
        fring_prec = fits.qnm_fring_33(final_spin_prec)
        fdamp = fits.qnm_fdamp_33(final_spin)
        fdamp_prec = fits.qnm_fdamp_33(final_spin_prec)
        fdamp_n2 = fits.qnm_fdamp_33_n2(final_spin)
        merger_frequency = fits.merger_freq_cp1_33(eta, shat, dchi, delta)
        omega_peak = fits.peak_frequency_33(eta, shat, dchi)
        c2 = fits.ringdown_freq_d2_33(eta, shat, dchi, delta)
        c3 = fits.ringdown_freq_d3_33(eta, shat, dchi, delta)
    elif mode == (4, 4):
        fring = fits.qnm_fring_44(final_spin)
        fring_prec = fits.qnm_fring_44(final_spin_prec)
        fdamp = fits.qnm_fdamp_44(final_spin)
        fdamp_prec = fits.qnm_fdamp_44(final_spin_prec)
        fdamp_n2 = fits.qnm_fdamp_44_n2(final_spin)
        merger_frequency = fits.merger_freq_cp1_44(eta, shat, dchi, delta)
        omega_peak = fits.peak_frequency_44(eta, shat, dchi, delta)
        c2 = fits.ringdown_freq_d2_44(eta, shat, dchi, delta)
        c3 = fits.ringdown_freq_d3_44(eta, shat, dchi, delta)
    else:
        fring = fits.qnm_fring_55(final_spin)
        fring_prec = fits.qnm_fring_55(final_spin_prec)
        fdamp = fits.qnm_fdamp_55(final_spin)
        fdamp_prec = fits.qnm_fdamp_55(final_spin_prec)
        fdamp_n2 = fits.qnm_fdamp_55_n2(final_spin)
        merger_frequency = fits.merger_freq_cp1_55(eta, shat, dchi, delta)
        omega_peak = fits.peak_frequency_55(eta, shat, dchi, delta)
        c2 = fits.ringdown_freq_d2_55(eta, shat, dchi, delta)
        c3 = fits.ringdown_freq_d3_55(eta, shat, dchi, delta)

    scale = 2.0 * _PI / final_mass
    omega_ring = scale * fring
    return _ModePhaseData(
        omega_ring=omega_ring,
        omega_ring_prec=scale * fring_prec,
        alpha1_rd=scale * fdamp,
        alpha1_rd_prec=scale * fdamp_prec,
        alpha2_rd=scale * fdamp_n2,
        omega_merger_cp=1.0 - merger_frequency / omega_ring,
        omega_peak=omega_peak,
        c2=c2,
        c3=c3,
    )


def build_hm_phase_coefficients(
    mode,
    chi1z,
    chi2z,
    final_mass,
    final_spin,
    phase22_coefficients: IMRPhenomTPhase22Coefficients,
    amplitude_coefficients: IMRPhenomTHMAmplitudeCoefficients,
    *,
    final_spin_prec=None,
):
    """Build phase coefficients for one supported IMRPhenomTHM higher mode.

    The amplitude and dominant-phase coefficients must describe the same
    binary and mode. ``chi1z`` belongs to the larger body, and ``final_mass``
    is the remnant mass divided by total mass.
    """
    try:
        mode = tuple(mode)
    except TypeError as exc:
        raise ValueError(f"unsupported IMRPhenomTHM mode {mode!r}") from exc
    if mode not in _SUPPORTED_MODES:
        raise ValueError(
            f"unsupported IMRPhenomTHM higher mode {mode!r}; "
            f"available modes are {_SUPPORTED_MODES}"
        )
    if not isinstance(amplitude_coefficients, IMRPhenomTHMAmplitudeCoefficients):
        raise TypeError(
            "amplitude_coefficients must be IMRPhenomTHMAmplitudeCoefficients"
        )
    if final_spin_prec is None:
        final_spin_prec = final_spin
    chi1z, chi2z, final_mass, final_spin, final_spin_prec = _coerce_like_phase(
        phase22_coefficients,
        chi1z,
        chi2z,
        final_mass,
        final_spin,
        final_spin_prec,
    )
    eta = phase22_coefficients.eta
    for name in ("omega_cut_pn_amp", "phi_cut_pn_amp"):
        value = getattr(amplitude_coefficients, name)
        if (
            value.device != eta.device
            or value.dtype != eta.dtype
            or value.shape != eta.shape
        ):
            raise ValueError(
                "IMRPhenomTHM amplitude and phase coefficients must have "
                "the same device, dtype, and batch shape"
            )

    delta = torch.sqrt(torch.clamp(1.0 - 4.0 * eta, min=0.0))
    mass1 = 0.5 * (1.0 + delta)
    mass2 = 0.5 * (1.0 - delta)
    shat = (mass1**2 * chi1z + mass2**2 * chi2z) / (mass1**2 + mass2**2)
    dchi = chi1z - chi2z
    mode_data = _mode_phase_data(
        mode,
        eta,
        shat,
        dchi,
        delta,
        final_mass,
        final_spin,
        final_spin_prec,
    )

    zero = torch.zeros_like(eta)
    m = eta.new_full(eta.shape, float(mode[1]))
    rescale = 0.5 * m
    c4 = zero
    c1 = (
        (1.0 + mode_data.c3 + c4)
        * (mode_data.omega_ring - mode_data.omega_peak)
        / (mode_data.c2 * (mode_data.c3 + 2.0 * c4))
    )
    c1_prec = (
        (1.0 + mode_data.c3 + c4)
        * (mode_data.omega_ring_prec - mode_data.omega_peak)
        / (mode_data.c2 * (mode_data.c3 + 2.0 * c4))
    )
    alpha21_rd = 0.5 * (mode_data.alpha2_rd - mode_data.alpha1_rd)

    t_cut = eta.new_full(eta.shape, _PHASE_CUT_TIME)
    omega_cut = rescale * omega22(t_cut, phase22_coefficients)
    omega_cut_bar = (
        1.0
        - (omega_cut + amplitude_coefficients.omega_cut_pn_amp) / mode_data.omega_ring
    )
    domega_cut = (
        -rescale
        * omega22_derivative(t_cut, phase22_coefficients)
        / mode_data.omega_ring
    )
    domega_peak = (
        -_ringdown_omega_derivative(zero, c1, mode_data.c2, mode_data.c3, c4)
        / mode_data.omega_ring
    )

    merger_time = eta.new_full(eta.shape, _MERGER_COLLOCATION_TIME)
    as_cut = torch.asinh(mode_data.alpha1_rd * t_cut)
    as_merger = torch.asinh(mode_data.alpha1_rd * merger_time)
    denominator_cut = torch.sqrt(1.0 + (mode_data.alpha1_rd * t_cut) ** 2)
    merger_matrix = torch.stack(
        (
            torch.stack((as_cut**2, as_cut**3, as_cut**4), dim=-1),
            torch.stack((as_merger**2, as_merger**3, as_merger**4), dim=-1),
            torch.stack(
                (
                    2.0 * mode_data.alpha1_rd * as_cut / denominator_cut,
                    3.0 * mode_data.alpha1_rd * as_cut**2 / denominator_cut,
                    4.0 * mode_data.alpha1_rd * as_cut**3 / denominator_cut,
                ),
                dim=-1,
            ),
        ),
        dim=-2,
    )
    merger_rhs = torch.stack(
        (
            omega_cut_bar
            - (1.0 - mode_data.omega_peak / mode_data.omega_ring)
            - (domega_peak / mode_data.alpha1_rd) * as_cut,
            mode_data.omega_merger_cp
            - (1.0 - mode_data.omega_peak / mode_data.omega_ring)
            - (domega_peak / mode_data.alpha1_rd) * as_merger,
            domega_cut - domega_peak / denominator_cut,
        ),
        dim=-1,
    )
    omega_merger = _scaled_linear_solve(merger_matrix, merger_rhs)

    dominant_phase_cut = phase22(t_cut, phase22_coefficients)
    phase_offset_merger = rescale * dominant_phase_cut - _merger_phase(
        t_cut,
        mode_data.omega_peak,
        domega_peak,
        mode_data.omega_ring,
        mode_data.alpha1_rd,
        omega_merger,
        zero,
    )
    phase_offset_ringdown = _merger_phase(
        zero,
        mode_data.omega_peak,
        domega_peak,
        mode_data.omega_ring,
        mode_data.alpha1_rd,
        omega_merger,
        phase_offset_merger,
    )

    return IMRPhenomTHMPhaseCoefficients(
        eta=eta,
        m=m,
        omega_peak=mode_data.omega_peak,
        c1=c1,
        c1_prec=c1_prec,
        c2=mode_data.c2,
        c3=mode_data.c3,
        c4=c4,
        omega_merger=omega_merger,
        alpha1_rd=mode_data.alpha1_rd,
        alpha1_rd_prec=mode_data.alpha1_rd_prec,
        alpha2_rd=mode_data.alpha2_rd,
        alpha21_rd=alpha21_rd,
        domega_peak=domega_peak,
        omega_ring=mode_data.omega_ring,
        omega_ring_prec=mode_data.omega_ring_prec,
        phase_offset_merger=phase_offset_merger,
        phase_offset_ringdown=phase_offset_ringdown,
        t_cut=t_cut,
    )


def merger_omega(t, coefficients):
    """Evaluate the rescaled higher-mode merger-frequency ansatz."""
    return _shared_merger_omega(t, coefficients)


def ringdown_omega(t, coefficients):
    """Evaluate the physical higher-mode ringdown-frequency ansatz."""
    return _shared_ringdown_omega(t, coefficients)


def merger_phase(t, coefficients):
    """Evaluate the integrated higher-mode merger-phase ansatz."""
    return _shared_merger_phase(t, coefficients)


def ringdown_phase(t, coefficients):
    """Evaluate the integrated higher-mode ringdown-phase ansatz."""
    return _shared_ringdown_phase(t, coefficients)


def phase_lm(
    t,
    coefficients,
    phase22_coefficients,
    amplitude_coefficients,
):
    """Evaluate one piecewise IMRPhenomTHM higher-mode carrier phase.

    The complex inspiral amplitude supplies an additional phase contribution
    before ``t=-150M``. Its boundary value is subtracted from the real-amplitude
    merger and ringdown carrier phase, matching LALSuite's mode assembly.
    """
    t = torch.as_tensor(t, device=coefficients.eta.device, dtype=coefficients.eta.dtype)
    inspiral = 0.5 * coefficients.m * phase22(t, phase22_coefficients)
    merger = merger_phase(t, coefficients) - amplitude_coefficients.phi_cut_pn_amp
    safe_ringdown_t = torch.where(t > 0.0, t, torch.zeros_like(t))
    ringdown = (
        ringdown_phase(safe_ringdown_t, coefficients)
        - amplitude_coefficients.phi_cut_pn_amp
    )
    return torch.where(
        t < coefficients.t_cut,
        inspiral,
        torch.where(t > 0.0, ringdown, merger),
    )
