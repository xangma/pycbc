# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch-native IMRPhenomXHM :math:`(2, -1)` mode.

This is a direct port of the default 2022-release path in LALSuite 7.26.  The
parameter-space fits and scalar matching systems are evaluated once per
waveform; frequency-dependent amplitude, phase, and assembly remain on the
active Torch device.  The mode has no spheroidal-mode mixing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import lal
import torch

from . import imrphenomx_utils_torch as _xutils
from ._torch_jax import torch_context
from .imrphenomxas_torch import Phase, PhaseDerivative, get_inspiral_phase


_MTSUN = lal.MTSUN_SI
_MPC = 1.0e6 * lal.PC_SI
_C = lal.C_SI
_PI = lal.PI
_FALSE_ZERO = 1.0e-15


@dataclass(frozen=True)
class _Mode21State:
    mass1: float
    mass2: float
    chi1: float
    chi2: float
    total_mass_seconds: float
    eta: float
    delta: float
    dchi: float
    dchi_half: float
    chi_s: float
    chi_a: float
    chi_pn_hat: float
    s_tot_r: float
    final_spin: float
    radiated_energy: float
    f_ring_22: float
    f_damp_22: float
    f_meco_22: float
    f_ring_21: float
    f_damp_21: float
    amp0: float

    @property
    def q(self):
        return self.mass1 / self.mass2


def _as_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _qnm_fring_21(final_spin):
    numerator = (
        0.059471695665734674
        - 0.07585416297991414 * final_spin
        + 0.021967909664591865 * final_spin**2
        - 0.0018964744613388146 * final_spin**3
        + 0.001164879406179587 * final_spin**4
        - 0.0003387374454044957 * final_spin**5
    )
    denominator = (
        1.0 - 1.4437415542456158 * final_spin + 0.49246920313191234 * final_spin**2
    )
    return numerator / denominator


def _qnm_fdamp_21(final_spin):
    numerator = (
        2.0696914454467294
        - 3.1358071947583093 * final_spin
        + 0.14456081596393977 * final_spin**2
        + 1.2194717985037946 * final_spin**3
        - 0.2947372598589144 * final_spin**4
        + 0.002943057145913646 * final_spin**5
    )
    denominator = (
        146.1779212636481
        - 219.81790388304876 * final_spin
        + 17.7141194900164 * final_spin**2
        + 75.90115083917898 * final_spin**3
        - 18.975287709794745 * final_spin**4
    )
    return numerator / denominator


def _mode21_state(params):
    mass1 = float(params["mass1"])
    mass2 = float(params["mass2"])
    chi1 = float(params.get("spin1z", 0.0))
    chi2 = float(params.get("spin2z", 0.0))
    if mass2 > mass1:
        mass1, mass2 = mass2, mass1
        chi1, chi2 = chi2, chi1

    total_mass = mass1 + mass2
    x1 = mass1 / total_mass
    x2 = mass2 / total_mass
    eta = x1 * x2
    delta = math.sqrt(max(1.0 - 4.0 * eta, 0.0))
    dchi = chi1 - chi2
    chi_eff = x1 * chi1 + x2 * chi2
    chi_pn_hat = (chi_eff - (38.0 / 113.0) * eta * (chi1 + chi2)) / (
        1.0 - 76.0 * eta / 113.0
    )
    remnant = _xutils.get_remnant_fMs(mass1, mass2, chi1, chi2)
    final_spin = _as_float(remnant.final_spin)
    radiated_energy = _as_float(remnant.radiated_energy)
    final_mass = 1.0 - radiated_energy
    total_mass_seconds = total_mass * _MTSUN
    distance = float(params["distance"])
    return _Mode21State(
        mass1=mass1,
        mass2=mass2,
        chi1=chi1,
        chi2=chi2,
        total_mass_seconds=total_mass_seconds,
        eta=eta,
        delta=delta,
        dchi=dchi,
        dchi_half=0.5 * dchi,
        chi_s=0.5 * (chi1 + chi2),
        chi_a=0.5 * dchi,
        chi_pn_hat=chi_pn_hat,
        s_tot_r=(x1 * x1 * chi1 + x2 * x2 * chi2) / (x1 * x1 + x2 * x2),
        final_spin=final_spin,
        radiated_energy=radiated_energy,
        f_ring_22=_as_float(remnant.ringdown_frequency),
        f_damp_22=_as_float(remnant.damping_frequency),
        f_meco_22=_as_float(remnant.meco_frequency),
        f_ring_21=_qnm_fring_21(final_spin) / final_mass,
        f_damp_21=_qnm_fdamp_21(final_spin) / final_mass,
        amp0=total_mass_seconds**2 / ((distance * _MPC) / _C),
    )


def _tensor(value, like, *, complex_value=False):
    dtype = (
        torch.complex64
        if complex_value and like.dtype == torch.float32
        else torch.complex128
        if complex_value
        else like.dtype
    )
    return torch.as_tensor(value, device=like.device, dtype=dtype)


def _solve(rows, values, like):
    matrix = _tensor(rows, like)
    rhs = torch.stack(
        [
            value if isinstance(value, torch.Tensor) else _tensor(value, like)
            for value in values
        ]
    )
    return torch.linalg.solve(matrix, rhs)


def _lambda_pn(state):
    if state.eta > 0.01:
        return 2.0 * _PI * (0.5 + 2.0 * math.log(2.0))

    eta = state.eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta
    s = state.s_tot_r
    s2 = s * s
    no_spin = (
        13.664473636545068
        - 170.08866400251395 * eta
        + 3535.657736681598 * eta2
        - 26847.690494515424 * eta3
        + 96463.68163125668 * eta4
        - 133820.89317471132 * eta5
    )
    eq_spin = (
        s
        * (
            18.52571430563905
            - 41.55066592130464 * s
            + eta3
            * (83493.24265292779 + 16501.749243703132 * s - 149700.4915210766 * s2)
            + eta
            * (3642.5891077598003 + 1198.4163078715173 * s - 6961.484805326852 * s2)
            + 33.8697137964237 * s2
            + eta2
            * (-35031.361998480075 - 7233.191207000735 * s + 62149.00902591944 * s2)
        )
        / (6.880288191574696 + s)
    )
    unequal_spin = -134.27742343186577 * state.dchi * state.delta * eta2
    return no_spin + eq_spin + unequal_spin


def _ringdown_phase_fits(state):
    eta = state.eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    s = state.s_tot_r
    s2 = s * s
    alpha2_22 = (
        0.2088669311744758
        - 0.37138987533788487 * eta
        + 6.510807976353186 * eta2
        - 31.330215053905395 * eta3
        + 55.45508989446867 * eta4
        + (
            0.2393965714370633
            + 1.6966740823756759 * eta
            - 16.874355161681766 * eta2
            + 38.61300158832203 * eta3
        )
        * s
        / (1.0 - 0.633218538432246 * s)
        + state.dchi
        * (0.9088578269496244 * eta**2.5 + 15.619592332008951 * state.dchi * eta**3.5)
        * state.delta
    )
    alpha_l_22 = (
        eta
        * (
            -1.1926122248825484
            + 2.5400257699690143 * eta
            - 16.504334734464244 * eta2
            + 27.623649807617376 * eta3
        )
        + eta3
        * s
        * (35.803988443700824 + 9.700178927988006 * s - 77.2346297158916 * s2)
        + eta
        * s
        * (0.1034526554654983 - 0.21477847929548569 * s - 0.06417449517826644 * s2)
        + eta2
        * s
        * (-4.7282481007397825 + 0.8743576195364632 * s + 8.170616575493503 * s2)
        + eta4
        * s
        * (-72.50310678862684 - 39.83460092417137 * s + 180.8345521274853 * s2)
        + (
            -0.7428134042821221 * state.chi1 * eta**3.5
            + 0.7428134042821221 * state.chi2 * eta**3.5
            + 17.588573345324154 * state.chi1**2 * eta**4.5
            - 35.17714669064831 * state.chi1 * state.chi2 * eta**4.5
            + 17.588573345324154 * state.chi2**2 * eta**4.5
        )
        * state.delta
    )
    alpha2 = alpha2_22 / (3.0 * state.f_ring_21**2)
    return alpha2, alpha_l_22 / eta


def _intermediate_phase_fit_values(state):
    eta = state.eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta
    eta6 = eta5 * eta
    s = state.s_tot_r
    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2
    dchi_delta = state.dchi * state.delta

    p1 = (
        4045.84
        + 7.63226 / eta
        - 1956.93 * eta
        - 23428.1 * eta2
        + 369153.0 * eta3
        - 2.28832e6 * eta4
        + 6.82533e6 * eta5
        - 7.86254e6 * eta6
        - 347.273 * s
        + 83.5428 * s2
        - 355.67 * s3
        + (4.44457 * s + 16.5548 * s2 + 13.6971 * s3) / eta
        + eta * (-79.761 * s - 355.299 * s2 + 1114.51 * s3 - 1077.75 * s4)
        + 92.6654 * s4
        + eta2 * (-619.837 * s - 722.787 * s2 + 2392.73 * s3 + 2689.18 * s4)
        + dchi_delta * (918.976 * eta + 91.7679 * eta2)
    )
    p2 = (
        3509.09
        + 0.91868 / eta
        + 194.72 * eta
        - 27556.2 * eta2
        + 369153.0 * eta3
        - 2.28832e6 * eta4
        + 6.82533e6 * eta5
        - 7.86254e6 * eta6
        + (
            (0.7084 - 60.1611 * eta + 131.815 * eta2 - 619.837 * eta3) * s
            + (6.10472 - 59.2068 * eta + 278.588 * eta2 - 722.787 * eta3) * s2
            + (5.7791 + 117.913 * eta - 1180.4 * eta2 + 2392.73 * eta3) * s3
            + eta * (92.6654 - 1077.75 * eta + 2689.18 * eta2) * s4
        )
        / eta
        + 91.7679 * dchi_delta * eta * (1.6012352903357276 + eta)
    )
    p3 = (
        3241.68
        + 890.016 * eta
        - 28651.9 * eta2
        + 369153.0 * eta3
        - 2.28832e6 * eta4
        + 6.82533e6 * eta5
        - 7.86254e6 * eta6
        + (-2.2484 + 187.641 * eta - 619.837 * eta2) * s
        + (3.22603 + 166.323 * eta - 722.787 * eta2) * s2
        + (117.913 - 1094.59 * eta + 2392.73 * eta2) * s3
        + (92.6654 - 1077.75 * eta + 2689.18 * eta2) * s4
        + 91.7679 * dchi_delta * eta2
    )
    p4 = (
        3160.88
        + 974.355 * eta
        - 28932.5 * eta2
        + 369780.0 * eta3
        - 2.28832e6 * eta4
        + 6.82533e6 * eta5
        - 7.86254e6 * eta6
        + (26.3355 - 196.851 * eta + 438.401 * eta2) * s
        + (45.9957 - 256.248 * eta + 117.563 * eta2) * s2
        + (-20.0261 + 467.057 * eta - 1613.0 * eta2) * s3
        + (-61.7446 + 577.057 * eta - 1096.81 * eta2) * s4
        + 65.3326 * dchi_delta * eta2
    )
    p5 = (
        3102.36
        + 315.911 * eta
        - 1688.26 * eta2
        + 3635.76 * eta3
        + (-23.0959 + 320.93 * eta - 1029.76 * eta2) * s
        + (-49.5435 + 826.816 * eta - 3079.39 * eta2) * s2
        + (40.7054 - 365.842 * eta + 1094.11 * eta2) * s3
        + (81.8379 - 1243.26 * eta + 4689.22 * eta2) * s4
        + 119.014 * dchi_delta * eta2
    )
    p6 = (
        3089.18
        + 4.89194 * eta
        + 190.008 * eta2
        - 255.245 * eta3
        + (2.96997 + 57.1612 * eta - 432.223 * eta2) * s
        + (-18.8929 + 630.516 * eta - 2804.66 * eta2) * s2
        + (-24.6193 + 549.085 * eta2) * s3
        + (-12.8798 - 722.674 * eta + 3967.43 * eta2) * s4
        + 74.0984 * dchi_delta * eta2
    )
    return p1, p2, p3, p4, p5, p6


def _value_and_derivative(function, point, like):
    with torch.enable_grad():
        frequency = _tensor(point, like).detach().requires_grad_(True)
        value = function(frequency)
        derivative = torch.autograd.grad(value, frequency)[0]
    return value.detach(), derivative.detach()


def _phase_21(mf, state, intrinsic, phase_coeffs, reference_frequency, coa_phase):
    fcut = (1.0 + 0.001 * (0.25 / state.eta - 1.0)) * 0.5 * state.f_meco_22
    fmatch_in = 0.5 * state.f_meco_22
    fmatch_rd = state.f_ring_21 - state.f_damp_21
    f_ring = state.f_ring_21
    f_damp = state.f_damp_21
    points = (
        fcut,
        (math.sqrt(3.0) * (fcut - f_ring) + 2.0 * (fcut + f_ring)) / 4.0,
        (3.0 * fcut + f_ring) / 4.0,
        (fcut + f_ring) / 2.0,
        (fcut + 3.0 * f_ring) / 4.0,
        (fcut + 7.0 * f_ring) / 8.0,
    )
    if state.eta < 0.05:
        selected = (0, 1, 3, 4, 5)
    elif state.s_tot_r >= 0.8:
        selected = (0, 1, 2, 4, 5)
    else:
        selected = (0, 1, 2, 3, 5)

    lina, linb_fit, psi4_to_strain = _xutils.calc_phaseatpeak(
        state.eta, state.s_tot_r, state.dchi, state.delta
    )
    lina = _as_float(lina)
    linb_fit = _as_float(linb_fit)
    psi4_to_strain = _as_float(psi4_to_strain)
    delta_t = -2.0 * _PI * (500.0 + psi4_to_strain)
    values = [value + delta_t for value in _intermediate_phase_fit_values(state)]
    if state.s_tot_r >= 0.8:
        derivatives = [
            PhaseDerivative(
                _tensor(2.0 * points[index] / state.total_mass_seconds, mf),
                intrinsic,
                phase_coeffs,
            )
            / state.total_mass_seconds
            + delta_t
            for index in range(3)
        ]
        values[1] = values[2] + derivatives[1] - derivatives[2]
        values[0] = values[1] + derivatives[0] - derivatives[1]

    rows = [
        [
            1.0,
            f_damp / (f_damp * f_damp + (points[index] - f_ring) ** 2),
            1.0 / points[index],
            1.0 / points[index] ** 2,
            1.0 / points[index] ** 4,
        ]
        for index in selected
    ]
    inter = _solve(rows, [values[index] for index in selected], mf)
    c0, c_l, c1, c2, c4 = inter.unbind()

    lambda_pn = _lambda_pn(state)

    def inspiral_raw(frequency):
        return (
            0.5
            / state.eta
            * get_inspiral_phase(2.0 * frequency, intrinsic, phase_coeffs)
            + lambda_pn * frequency
        )

    def intermediate_raw(frequency):
        return (
            c0 * frequency
            + c1 * torch.log(frequency)
            - c2 / frequency
            - c4 / (3.0 * frequency**3)
            + c_l * torch.atan((frequency - f_ring) / f_damp)
        )

    def intermediate_derivative(frequency):
        return (
            c0
            + c_l * f_damp / (f_damp**2 + (frequency - f_ring) ** 2)
            + c1 / frequency
            + c2 / frequency**2
            + c4 / frequency**4
        )

    alpha2, alpha_l = _ringdown_phase_fits(state)

    def ringdown_raw(frequency):
        return -(f_ring**2) * alpha2 / frequency + alpha_l * torch.atan(
            (frequency - f_ring) / f_damp
        )

    def ringdown_derivative(frequency):
        return f_ring**2 * alpha2 / frequency**2 + alpha_l * f_damp / (
            f_damp**2 + (frequency - f_ring) ** 2
        )

    insp_at_in, dinsp_at_in = _value_and_derivative(inspiral_raw, fmatch_in, mf)
    int_at_in = intermediate_raw(_tensor(fmatch_in, mf))
    dint_at_in = intermediate_derivative(_tensor(fmatch_in, mf))
    c1_insp = dint_at_in - dinsp_at_in
    c_insp = -c1_insp * fmatch_in + int_at_in - insp_at_in

    int_at_rd = intermediate_raw(_tensor(fmatch_rd, mf))
    dint_at_rd = intermediate_derivative(_tensor(fmatch_rd, mf))
    rd_at_rd = ringdown_raw(_tensor(fmatch_rd, mf))
    drd_at_rd = ringdown_derivative(_tensor(fmatch_rd, mf))
    c1_rd = dint_at_rd - drd_at_rd
    c_rd = -c1_rd * fmatch_rd + int_at_rd - rd_at_rd

    dphi22_ref = (
        PhaseDerivative(
            _tensor(
                (state.f_ring_22 - state.f_damp_22) / state.total_mass_seconds,
                mf,
            ),
            intrinsic,
            phase_coeffs,
        )
        / state.total_mass_seconds
    )
    timeshift = linb_fit - dphi22_ref + delta_t
    mf_ref = reference_frequency * state.total_mass_seconds
    phase_ref_22 = Phase(_tensor(reference_frequency, mf), intrinsic, phase_coeffs)
    phiref22 = -phase_ref_22 - timeshift * mf_ref - lina + 2.0 * coa_phase + _PI / 4.0
    f_align = 0.5 * state.f_meco_22
    if state.eta > 0.05:
        f_align *= 0.6
    mode_insp_align = inspiral_raw(_tensor(f_align, mf)) + c1_insp * f_align + c_insp
    aligned_22 = (
        0.5
        * (
            Phase(
                _tensor(2.0 * f_align / state.total_mass_seconds, mf),
                intrinsic,
                phase_coeffs,
            )
            + lina
            + phiref22
        )
        + timeshift * f_align
    )
    delta_phi = torch.fmod(
        aligned_22 - 3.0 * _PI / 8.0 - mode_insp_align,
        2.0 * _PI,
    )
    if _pn21_amplitude_sign(state) > 0:
        delta_phi = delta_phi + _PI

    inspiral = inspiral_raw(mf) + c1_insp * mf + c_insp + delta_phi
    intermediate = intermediate_raw(mf) + delta_phi
    ringdown = ringdown_raw(mf) + c1_rd * mf + c_rd + delta_phi
    return torch.where(
        mf <= fmatch_in,
        inspiral,
        torch.where(mf <= fmatch_rd, intermediate, ringdown),
    )


def _pn21_amplitude_sign(state):
    frequency = 0.008
    output = (
        -16.0 * state.delta * state.eta * frequency * _PI**1.5 / (3.0 * math.sqrt(5.0))
        + 4.0
        * 2.0 ** (1.0 / 3.0)
        * (state.chi1 - state.chi2 + state.delta * (state.chi1 + state.chi2))
        * state.eta
        * frequency ** (4.0 / 3.0)
        * _PI ** (11.0 / 6.0)
        / math.sqrt(5.0)
        + 2.0
        * 2.0 ** (2.0 / 3.0)
        * state.eta
        * (306.0 * state.delta - 360.0 * state.delta * state.eta)
        * frequency ** (5.0 / 3.0)
        * _PI ** (13.0 / 6.0)
        / (189.0 * math.sqrt(5.0))
    )
    return 1 if output >= 0.0 else -1


def _pn_amplitude_coefficients(state):
    chi_a = state.chi_a
    chi_s = state.chi_s
    eta = state.eta
    delta = state.delta
    pi = _PI
    return (
        0.0j,
        delta * pi ** (1.0 / 3.0) * 2.0 ** (1.0 / 3.0),
        -1.5 * (chi_a + chi_s * delta) * pi ** (2.0 / 3.0) * 2.0 ** (2.0 / 3.0),
        (335.0 * delta + 1404.0 * delta * eta) / 672.0 * pi * 2.0,
        (
            3427.0 * chi_a
            - 1j * 672.0 * delta
            + 3427.0 * chi_s * delta
            - 8404.0 * chi_a * eta
            - 3860.0 * chi_s * delta * eta
            - 1344.0 * delta * pi
            - 1j * 672.0 * delta * math.log(16.0)
        )
        / 1344.0
        * pi ** (4.0 / 3.0)
        * 2.0 ** (4.0 / 3.0),
        (
            -155965824.0 * chi_a * chi_s
            - 964357.0 * delta
            + 432843264.0 * chi_a * chi_s * eta
            - 23670792.0 * delta * eta
            + 24385536.0 * chi_a * pi
            + 24385536.0 * chi_s * delta * pi
            - 77982912.0 * delta * chi_a**2
            + 81285120.0 * delta * eta * chi_a**2
            - 77982912.0 * delta * chi_s**2
            + 39626496.0 * delta * eta * chi_s**2
            + 21535920.0 * delta * eta**2
        )
        / 8128512.0
        * pi ** (5.0 / 3.0)
        * 2.0 ** (5.0 / 3.0),
        (
            143063173.0 * chi_a
            - 1j * 1350720.0 * delta
            + 143063173.0 * chi_s * delta
            - 546199608.0 * chi_a * eta
            - 1j * 72043776.0 * delta * eta
            - 169191096.0 * chi_s * delta * eta
            - 9898560.0 * delta * pi
            + 20176128.0 * delta * eta * pi
            - 1j * 5402880.0 * delta * math.log(2.0)
            - 1j * 17224704.0 * delta * eta * math.log(2.0)
            + 61725888.0 * chi_s * delta * chi_a**2
            - 81285120.0 * chi_s * delta * eta * chi_a**2
            + 20575296.0 * chi_a**3
            - 81285120.0 * eta * chi_a**3
            + 61725888.0 * chi_a * chi_s**2
            - 165618432.0 * chi_a * eta * chi_s**2
            + 20575296.0 * delta * chi_s**3
            - 1016064.0 * delta * eta * chi_s**3
            + 128873808.0 * chi_a * eta**2
            - 3859632.0 * chi_s * delta * eta**2
        )
        / 5419008.0
        * pi**2
        * 4.0,
    )


def _pn_amplitude(mf, coefficients, amp_norm):
    frequency_power = mf ** (1.0 / 3.0)
    series = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        series = series * frequency_power + coefficient
    global_factor = 2.0 ** (-7.0 / 6.0) * math.sqrt(2.0) / 3.0
    return torch.abs(series) * global_factor * mf ** (-7.0 / 6.0) * amp_norm


def _inspiral_cutoff(state):
    comparable_mass = 0.5 * state.f_meco_22
    if state.q < 20.0:
        return comparable_mass
    extreme_mass_ratio = 1.25 * (
        (
            0.011671068725758493
            - 0.0000858396080377194 * state.chi1
            + 0.000316707064291237 * state.chi1**2
        )
        * (0.8447212540381764 + 6.2873167352395125 * state.eta)
        / (1.2857082764038923 - 0.9977728883419751 * state.chi1)
    )
    blend = 0.5 + 0.5 * math.tanh((state.eta - 0.0192234) / 0.004)
    return blend * comparable_mass + (1.0 - blend) * extreme_mass_ratio


def _inspiral_amplitude_fit_values(state):
    eta = state.eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta
    sqrt_eta = math.sqrt(eta)
    delta = state.delta
    dchi = state.dchi_half
    s = state.chi_pn_hat
    s2 = s * s
    value1 = abs(
        dchi
        * eta5
        * (-3962.5020052272976 + 987.635855365408 * s - 134.98527058315528 * s2)
        + delta
        * (
            19.30531354642419
            + 16.6640319856064 * eta
            - 120.58166037019478 * eta2
            + 220.77233521626252 * eta3
        )
        * sqrt_eta
        + dchi
        * delta
        * (
            31.364509907424765 * eta
            - 843.6414532232126 * eta2
            + 2638.3077554662905 * eta3
        )
        * sqrt_eta
        + dchi
        * delta
        * (
            32.374226994179054 * eta
            - 202.86279451816662 * eta2
            + 347.1621871204769 * eta3
        )
        * s
        * sqrt_eta
        + delta
        * s
        * (
            -16.75726972301224
            * (
                1.1787350890261943
                - 7.812073811917883 * eta
                + 99.47071002831267 * eta2
                - 500.4821414428368 * eta3
                + 876.4704270866478 * eta4
            )
            + 2.3439955698372663
            * (
                0.9373952326655807
                + 7.176140122833879 * eta
                - 279.6409723479635 * eta2
                + 2178.375177755584 * eta3
                - 4768.212511142035 * eta4
            )
            * s
        )
        * sqrt_eta
    )
    value2 = abs(
        dchi
        * eta5
        * (-2898.9172078672705 + 580.9465034962822 * s + 22.251142639924076 * s2)
        + delta
        * (
            dchi**2
            * (
                -18.541685007214625 * eta
                + 166.7427445020744 * eta2
                - 417.5186332459383 * eta3
            )
            + dchi
            * (
                41.61457952037761 * eta
                - 779.9151607638761 * eta2
                + 2308.6520892707795 * eta3
            )
        )
        * sqrt_eta
        + delta
        * (
            11.414934585404561
            + 30.883118528233638 * eta
            - 260.9979123967537 * eta2
            + 1046.3187137392433 * eta3
            - 1556.9475493549746 * eta4
        )
        * sqrt_eta
        + delta
        * s
        * (
            -10.809007068469844
            * (
                1.1408749895922659
                - 18.140470190766937 * eta
                + 368.25127088896744 * eta2
                - 3064.7291458207815 * eta3
                + 11501.848278358668 * eta4
                - 16075.676528787526 * eta5
            )
            + 1.0088254664333147
            * (
                1.2322739396680107
                - 192.2461213084741 * eta
                + 4257.760834055382 * eta2
                - 35561.24587952242 * eta3
                + 130764.22485304279 * eta4
                - 177907.92440833704 * eta5
            )
            * s
        )
        * sqrt_eta
        + delta
        * (
            dchi
            * (
                36.88578491943111 * eta
                - 321.2569602623214 * eta2
                + 748.6659668096737 * eta3
            )
            * s
            + dchi
            * (
                -95.42418611585117 * eta
                + 1217.338674959742 * eta2
                - 3656.192371615541 * eta3
            )
            * s2
        )
        * sqrt_eta
    )
    value3 = abs(
        dchi
        * eta5
        * (-2282.9983216879655 + 157.94791186394787 * s + 16.379731479465033 * s2)
        + dchi
        * delta
        * (
            21.935833431534224 * eta
            - 460.7130131927895 * eta2
            + 1350.476411541137 * eta3
        )
        * sqrt_eta
        + delta
        * (
            5.390240326328237
            + 69.01761987509603 * eta
            - 568.0027716789259 * eta2
            + 2435.4098320959706 * eta3
            - 3914.3390484239667 * eta4
        )
        * sqrt_eta
        + dchi
        * delta
        * (
            29.731007410186827 * eta
            - 372.09609843131386 * eta2
            + 1034.4897198648962 * eta3
        )
        * s
        * sqrt_eta
        + delta
        * s
        * (
            -7.1976397556450715
            * (
                0.7603360145475428
                - 6.587249958654174 * eta
                + 120.87934060776237 * eta2
                - 635.1835857158857 * eta3
                + 1109.0598539312573 * eta4
            )
            - 0.0811847192323969
            * (
                7.951454648295709
                + 517.4039644814231 * eta
                - 9548.970156895082 * eta2
                + 52586.63520999897 * eta3
                - 93272.17990295641 * eta4
            )
            * s
            - 0.28384547935698246
            * (
                -0.8870770459576875
                + 180.0378964169756 * eta
                - 2707.9572896559484 * eta2
                + 14158.178124971111 * eta3
                - 24507.800226675925 * eta4
            )
            * s2
        )
        * sqrt_eta
    )
    return value1, value2, value3


def _ringdown_amplitude_fit_values(state):
    eta = state.eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta
    eta6 = eta5 * eta
    delta = state.delta
    dchi = state.dchi_half
    s = state.chi_pn_hat
    s2 = s * s
    value1 = abs(
        delta
        * eta
        * (
            12.880905080761432
            - 23.5291063016996 * eta
            + 92.6090002736012 * eta2
            - 175.16681482428694 * eta3
        )
        + dchi
        * delta
        * eta
        * (
            26.89427230731867 * eta
            - 710.8871223808559 * eta2
            + 2255.040486907459 * eta3
        )
        + dchi
        * delta
        * eta
        * (
            21.402708785047853 * eta
            - 232.07306353130417 * eta2
            + 591.1097623278739 * eta3
        )
        * s
        + delta
        * eta
        * s
        * (
            -10.090867481062709
            * (
                0.9580746052260011
                + 5.388149112485179 * eta
                - 107.22993216128548 * eta2
                + 801.3948756800821 * eta3
                - 2688.211889175019 * eta4
                + 3950.7894052628735 * eta5
                - 1992.9074348833092 * eta6
            )
            - 0.42972412296628143
            * (
                1.9193131231064235
                + 139.73149069609775 * eta
                - 1616.9974609915555 * eta2
                - 3176.4950303461164 * eta3
                + 107980.65459735804 * eta4
                - 479649.75188253267 * eta5
                + 658866.0983367155 * eta6
            )
            * s
        )
        + dchi
        * eta5
        * (-1512.439342647443 + 175.59081294852444 * s + 10.13490934572329 * s2)
    )
    value2 = abs(
        delta * (9.112452928978168 - 7.5304766811877455 * eta) * eta
        + dchi
        * delta
        * eta
        * (
            16.236533863306132 * eta
            - 500.11964987628926 * eta2
            + 1618.0818430353293 * eta3
        )
        + dchi
        * delta
        * eta
        * (
            2.7866868976718226 * eta
            - 0.4210629980868266 * eta2
            - 20.274691328125606 * eta3
        )
        * s
        + dchi
        * eta5
        * (-1116.4039232324135 + 245.73200219767514 * s + 21.159179960295855 * s2)
        + delta
        * eta
        * s
        * (
            -8.236485576091717
            * (
                0.8917610178208336
                + 5.1501231412520285 * eta
                - 87.05136337926156 * eta2
                + 519.0146702141192 * eta3
                - 997.6961311502365 * eta4
            )
            + 0.2836840678615208
            * (
                -0.19281297100324718
                - 57.65586769647737 * eta
                + 586.7942442434971 * eta2
                - 1882.2040277496196 * eta3
                + 2330.3534917059906 * eta4
            )
            * s
            + 0.40226131643223145
            * (
                -3.834742668014861
                + 190.42214703482531 * eta
                - 2885.5110686004946 * eta2
                + 16087.433824017446 * eta3
                - 29331.524552164105 * eta4
            )
            * s2
        )
    )
    value3 = abs(
        delta * (2.920930733198033 - 3.038523690239521 * eta) * eta
        + dchi
        * delta
        * eta
        * (
            6.3472251472354975 * eta
            - 171.23657247338042 * eta2
            + 544.1978232314333 * eta3
        )
        + dchi
        * delta
        * eta
        * (
            1.9701247529688362 * eta
            - 2.8616711550845575 * eta2
            - 0.7347258030219584 * eta3
        )
        * s
        + dchi
        * eta5
        * (-334.0969956136684 + 92.91301644484749 * s - 5.353399481074393 * s2)
        + delta
        * eta
        * s
        * (
            -2.7294297839371824
            * (
                1.148166706456899
                - 4.384077347340523 * eta
                + 36.120093043420326 * eta2
                - 87.26454353763077 * eta3
            )
            + 0.23949142867803436
            * (
                -0.6931516433988293
                + 33.33372867559165 * eta
                - 307.3404155231787 * eta2
                + 862.3123076782916 * eta3
            )
            * s
            + 0.1930861073906724
            * (
                3.7735099269174106
                - 19.11543562444476 * eta
                - 78.07256429516346 * eta2
                + 485.67801863289293 * eta3
            )
            * s2
        )
    )
    return value1, value2, value3


def _intermediate_amplitude_fit_values(state):
    eta = state.eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta
    delta = state.delta
    dchi = state.dchi_half
    s = state.s_tot_r
    s2 = s * s
    value1 = abs(
        delta
        * eta
        * (
            dchi**2
            * (
                5.159755997682368 * eta
                - 30.293198248154948 * eta2
                + 63.70715919820867 * eta3
            )
            + dchi
            * (
                8.262642080222694 * eta
                - 415.88826990259116 * eta2
                + 1427.5951158851076 * eta3
            )
        )
        + delta
        * eta
        * (
            18.55363583212328
            - 66.46950491124205 * eta
            + 447.2214642597892 * eta2
            - 1614.178472020212 * eta3
            + 2199.614895727586 * eta4
        )
        + dchi
        * eta5
        * (-1698.841763891122 - 195.27885562092342 * s - 1.3098861736238572 * s2)
        + delta
        * eta
        * (
            dchi
            * (
                34.17829404207186 * eta
                - 386.34587928670015 * eta2
                + 1022.8553774274128 * eta3
            )
            * s
            + dchi
            * (
                56.76554600963724 * eta
                - 491.4593694689354 * eta2
                + 1016.6019654342113 * eta3
            )
            * s2
        )
        + delta
        * eta
        * s
        * (
            -8.276366844994188
            * (
                1.0677538075697492
                - 24.12941323757896 * eta
                + 516.7886322104276 * eta2
                - 4389.799658723288 * eta3
                + 16770.447637953577 * eta4
                - 23896.392706809565 * eta5
            )
            - 1.6908277400304084
            * (
                3.4799140066657928
                - 29.00026389706585 * eta
                + 114.8330693231833 * eta2
                - 184.13091281984674 * eta3
                + 592.300353344717 * eta4
                - 2085.0821513466053 * eta5
            )
            * s
            - 0.46006975902558517
            * (
                -2.1663474937625975
                + 826.026625945615 * eta
                - 17333.549622759732 * eta2
                + 142904.08962903373 * eta3
                - 528521.6231015554 * eta4
                + 731179.456702448 * eta5
            )
            * s2
        )
    )
    value3 = abs(
        delta
        * eta
        * (
            13.318990196097973
            - 21.755549987331054 * eta
            + 76.14884211156267 * eta2
            - 127.62161159798488 * eta3
        )
        + dchi
        * delta
        * eta
        * (
            17.704321326939414 * eta
            - 434.4390350012534 * eta2
            + 1366.2408490833282 * eta3
        )
        + dchi
        * delta
        * eta
        * (
            11.877985158418596 * eta
            - 131.04937626836355 * eta2
            + 343.79587860999874 * eta3
        )
        * s
        + dchi
        * eta5
        * (-1522.8543551416456 - 16.639896279650678 * s + 3.0053086651515843 * s2)
        + delta
        * eta
        * s
        * (
            -8.665646058245033
            * (
                0.7862132291286934
                + 8.293609541933655 * eta
                - 111.70764910503321 * eta2
                + 576.7172598056907 * eta3
                - 1001.2370065269745 * eta4
            )
            - 0.9459820574514348
            * (
                1.309016452198605
                + 48.94077040282239 * eta
                - 817.7854010574645 * eta2
                + 4331.56002883546 * eta3
                - 7518.309520232795 * eta4
            )
            * s
            - 0.4308267743835775
            * (
                9.970654092010587
                - 302.9708323417439 * eta
                + 3662.099161055873 * eta2
                - 17712.883990278668 * eta3
                + 29480.158198408903 * eta4
            )
            * s2
        )
    )
    return value1, value3


def _ringdown_amplitude_parameters(state):
    value1, value2, value3 = _ringdown_amplitude_fit_values(state)
    if value3 >= value2 * value2 / value1:
        value3 = 0.5 * value2 * value2 / value1
    if value3 > value2:
        value3 = 0.5 * value2
    if value1 < value2 and value3 > value1:
        value3 = value1
    denominator = math.sqrt(value1 / value3) - value1 / value2
    denominator = max(denominator, 1.0e-16)
    a0 = value1 * state.f_damp_21 / denominator
    sigma = math.sqrt(a0 / (value2 * state.f_damp_21))
    decay = 0.5 * sigma * math.log(value1 / value3)
    return a0, sigma, decay


def _ringdown_amplitude_core(frequency, state, parameters):
    a0, sigma, decay = parameters
    offset = frequency - state.f_ring_21
    width = state.f_damp_21 * sigma
    return (
        a0
        * state.f_damp_21
        / (torch.exp(decay * offset / width) * (offset * offset + width * width))
    )


def _ringdown_amplitude_derivative(frequency, state, parameters):
    a0, sigma, decay = parameters
    offset = frequency - state.f_ring_21
    fdamp = state.f_damp_21
    numerator = a0 * (
        offset * offset * decay
        + 2.0 * fdamp * offset * sigma
        + fdamp * fdamp * decay * sigma * sigma
    )
    denominator = (
        sigma
        * (offset * offset + fdamp * fdamp * sigma * sigma) ** 2
        * torch.exp(offset * decay / (fdamp * sigma))
    )
    return -numerator / denominator


def _amplitude_21(mf, state):
    amp_norm = math.sqrt(2.0 * state.eta / 3.0) * _PI ** (-1.0 / 6.0)
    pn_dominant = amp_norm * 2.0 ** (-7.0 / 6.0)
    pn_coefficients = _tensor(_pn_amplitude_coefficients(state), mf, complex_value=True)

    fcut_inspiral = _inspiral_cutoff(state)
    collocation_frequencies = (
        0.5 * fcut_inspiral,
        0.75 * fcut_inspiral,
        fcut_inspiral,
    )
    fit_values = _inspiral_amplitude_fit_values(state)
    collocation_tensors = [_tensor(value, mf) for value in collocation_frequencies]
    pn_values = [
        _pn_amplitude(value, pn_coefficients, amp_norm) for value in collocation_tensors
    ]
    pseudo_targets = [
        (fit - pn_value) / (pn_dominant * frequency ** (-7.0 / 6.0))
        for fit, pn_value, frequency in zip(fit_values, pn_values, collocation_tensors)
    ]
    pseudo_rows = [
        [
            (frequency / fcut_inspiral) ** (7.0 / 3.0),
            (frequency / fcut_inspiral) ** (8.0 / 3.0),
            (frequency / fcut_inspiral) ** 3.0,
        ]
        for frequency in collocation_frequencies
    ]
    pseudo = _solve(pseudo_rows, pseudo_targets, mf)

    def inspiral(frequency):
        ratio = frequency / fcut_inspiral
        pseudo_terms = (
            pseudo[0] * ratio ** (7.0 / 3.0)
            + pseudo[1] * ratio ** (8.0 / 3.0)
            + pseudo[2] * ratio**3
        )
        return (
            _pn_amplitude(frequency, pn_coefficients, amp_norm)
            + pn_dominant * frequency ** (-7.0 / 6.0) * pseudo_terms
        )

    ringdown_parameters = _ringdown_amplitude_parameters(state)
    fmatch_ringdown = state.f_ring_21 - state.f_damp_21
    ffalloff = state.f_ring_21 + 2.0 * state.f_damp_21
    falloff_tensor = _tensor(ffalloff, mf)
    tail_amplitude = _ringdown_amplitude_core(
        falloff_tensor, state, ringdown_parameters
    )
    tail_decay = (
        -_ringdown_amplitude_derivative(falloff_tensor, state, ringdown_parameters)
        / tail_amplitude
    )

    def ringdown(frequency):
        core = _ringdown_amplitude_core(frequency, state, ringdown_parameters)
        tail = tail_amplitude * torch.exp(-tail_decay * (frequency - ffalloff))
        return torch.where(frequency < ffalloff, core, tail)

    spacing = (fmatch_ringdown - fcut_inspiral) / 5.0
    intermediate_frequencies = tuple(
        fcut_inspiral + index * spacing for index in range(6)
    )
    fit1, fit3 = _intermediate_amplitude_fit_values(state)
    right = _tensor(fmatch_ringdown, mf)
    values = (
        inspiral(_tensor(fcut_inspiral, mf)),
        _tensor(fit1, mf),
        _tensor(fit3, mf),
        ringdown(right),
        _ringdown_amplitude_derivative(right, state, ringdown_parameters),
    )
    point_indices = (0, 1, 3, 5)
    rows = [
        [frequency ** (power - 7.0 / 6.0) for power in range(5)]
        for frequency in (intermediate_frequencies[index] for index in point_indices)
    ]
    rows.append(
        [
            (power - 7.0 / 6.0) * fmatch_ringdown ** (power - 1.0 - 7.0 / 6.0)
            for power in range(5)
        ]
    )
    intermediate_coefficients = _solve(rows, values, mf)

    def intermediate(frequency):
        polynomial = intermediate_coefficients[-1]
        for coefficient in reversed(intermediate_coefficients[:-1]):
            polynomial = polynomial * frequency + coefficient
        return frequency ** (-7.0 / 6.0) * polynomial

    amplitude = torch.where(
        mf <= fcut_inspiral,
        inspiral(mf),
        torch.where(mf <= fmatch_ringdown, intermediate(mf), ringdown(mf)),
    )
    return torch.where(amplitude < 0.0, _FALSE_ZERO, amplitude)


def imrphenomxhm_h2m1_samples(
    core,
    params,
    *,
    frequencies=None,
    reference_frequency=None,
):
    r"""Return active positive-frequency samples of LAL's :math:`h_{2,-1}`."""

    state = _mode21_state(params)
    if state.mass1 == state.mass2 and state.chi1 == state.chi2:
        return torch.zeros_like(core.polarization)

    if frequencies is None:
        frequencies = (
            torch.arange(
                core.first_bin,
                core.stop_bin,
                device=core.polarization.device,
                dtype=core.polarization.real.dtype,
            )
            * core.delta_f
        )
    mf = frequencies * state.total_mass_seconds
    intrinsic = torch.tensor(
        [state.mass1, state.mass2, state.chi1, state.chi2],
        device=frequencies.device,
        dtype=frequencies.dtype,
    )
    phase_coeffs = _xutils.PhenomX_phase_coeff_table.to(
        device=frequencies.device,
        dtype=frequencies.dtype,
    )
    if reference_frequency is None:
        reference_frequency = float(params.get("f_ref", 0.0))
        if reference_frequency <= 0.0:
            reference_frequency = float(params["f_lower"])
    coa_phase = float(params.get("coa_phase", 0.0))
    with torch_context(frequencies):
        phase = _phase_21(
            mf,
            state,
            intrinsic,
            phase_coeffs,
            reference_frequency,
            coa_phase,
        )
        amplitude = _amplitude_21(mf, state)
        samples = state.amp0 * amplitude * torch.exp(1j * phase)
    return samples.to(core.polarization.dtype)
