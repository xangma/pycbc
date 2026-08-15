# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch-native IMRPhenomXHM :math:`(3, -3)` mode.

This ports the default 2022-release, no-mode-mixing path in LALSuite 7.26.
Scalar parameter-space fits are evaluated once per waveform; all matching,
frequency-dependent evaluation, and waveform assembly stay on the active
Torch device.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import lal
import torch

from . import imrphenomx_utils_torch as _xutils
from ._torch_jax import torch_context
from .imrphenomxas_torch import Phase, PhaseDerivative, get_inspiral_phase
from .imrphenomxhm_mode21_torch import (
    _as_float,
    _mode21_state,
    _ringdown_phase_fits as _ringdown_phase_fits_21,
    _solve,
    _tensor,
    _value_and_derivative,
)


_PI = lal.PI
_FALSE_ZERO = 1.0e-15


@dataclass(frozen=True)
class _Mode33State:
    """Mode-specific QNM data layered over the common XHM state."""

    base: object
    f_ring_33: float
    f_damp_33: float

    def __getattr__(self, name):
        return getattr(self.base, name)


def _qnm_fring_33(final_spin):
    numerator = (
        0.09540436245212061
        - 0.22799517865876945 * final_spin
        + 0.13402916709362475 * final_spin**2
        + 0.03343753057911253 * final_spin**3
        - 0.030848060170259615 * final_spin**4
        - 0.006756504382964637 * final_spin**5
        + 0.0027301732074159835 * final_spin**6
    )
    denominator = (
        1.0
        - 2.7265947806178334 * final_spin
        + 2.144070539525238 * final_spin**2
        - 0.4706873667569393 * final_spin**4
        + 0.05321818246993958 * final_spin**6
    )
    return numerator / denominator


def _qnm_fdamp_33(final_spin):
    numerator = (
        0.014754148319335946
        - 0.03124423610028678 * final_spin
        + 0.017192623913708124 * final_spin**2
        + 0.001034954865629645 * final_spin**3
        - 0.0015925124814622795 * final_spin**4
        - 0.0001414350555699256 * final_spin**5
    )
    denominator = (
        1.0
        - 2.0963684630756894 * final_spin
        + 1.196809702382645 * final_spin**2
        - 0.09874113387889819 * final_spin**4
    )
    return numerator / denominator


def _mode33_state(params):
    base = _mode21_state(params)
    final_mass = 1.0 - base.radiated_energy
    return _Mode33State(
        base=base,
        f_ring_33=_qnm_fring_33(base.final_spin) / final_mass,
        f_damp_33=_qnm_fdamp_33(base.final_spin) / final_mass,
    )


def _poly(value, coefficients):
    result = 0.0
    for coefficient in reversed(coefficients):
        result = result * value + coefficient
    return result


def _lambda_pn(state):
    if state.eta > 0.01:
        return -(2.0 * _PI / 3.0) * (-21.0 / 5.0 + 6.0 * math.log(1.5))

    eta = state.eta
    eta2 = eta * eta
    s = state.s_tot_r
    no_spin = _poly(
        eta,
        (4.1138398568400705, 9.772510519809892, -103.92956504520747, 242.3428625556764),
    )
    eq_spin = (
        (-0.13253553909611435 + 26.644159828590055 * eta - 105.09339163109497 * eta2)
        * s
        / (1.0 + 0.11322426762297967 * s)
    )
    unequal_spin = -19.705359163581168 * state.dchi * eta2 * state.delta
    return no_spin + eq_spin + unequal_spin


def _intermediate_phase_fit_values(state):
    eta = state.eta
    eta2 = eta * eta
    s = state.s_tot_r
    s2 = s * s
    s3 = s2 * s
    dchi_delta = state.dchi * state.delta

    p1 = (
        4360.19
        + 4.27128 / eta
        + _poly(
            eta,
            (
                0.0,
                -8727.4,
                18485.9,
                371303.00000000006,
                -3.22792e6,
                1.01799e7,
                -1.15659e7,
            ),
        )
        + (
            _poly(eta, (11.6635, -251.579, -3255.6400000000003, 19614.6, -34860.2)) * s
            + _poly(eta, (14.8017, 204.025, -5421.92, 36587.3, -74299.5)) * s2
        )
        / eta
        + 223.65100000000004 * dchi_delta * eta * (3.9201300240106223 + eta)
    )
    p2 = (
        3797.06
        + 0.786684 / eta
        + _poly(
            eta,
            (
                0.0,
                -2397.09,
                -25514.0,
                518314.99999999994,
                -3.41708e6,
                1.01799e7,
                -1.15659e7,
            ),
        )
        + (
            _poly(eta, (6.7812399999999995, 39.4668, -3520.37, 19614.6, -34860.2)) * s
            + _poly(eta, (4.80384, 293.215, -5914.61, 36587.3, -74299.5)) * s2
        )
        / eta
        + 223.65100000000004 * dchi_delta * eta * (1.3095134830606614 + eta)
    )
    p3 = (
        _poly(
            eta,
            (3321.83, 1796.03, -52406.1, 605028.0, -3.52532e6, 1.01799e7, -1.15659e7),
        )
        + _poly(eta, (223.601, -3714.77, 19614.6, -34860.2)) * s
        + _poly(eta, (314.317, -5906.46, 36587.3, -74299.5)) * s2
        + 223.651 * dchi_delta * eta2
    )
    p4 = (
        _poly(
            eta, (3239.44, -661.15, 5139.79, 3456.2, -248477.0, 1.17255e6, -1.70363e6)
        )
        + _poly(eta, (225.859, -4150.09, 24364.0, -46537.3)) * s
        + _poly(eta, (35.2439, -994.971, 8953.98, -23603.5)) * s2
        + _poly(eta, (-310.489, 5946.15, -35337.1, 67102.4)) * s3
        + 30.484 * dchi_delta * eta2
    )
    p5 = (
        _poly(
            eta,
            (3114.3, 2143.06, -49428.3, 563997.0, -3.35991e6, 9.99745e6, -1.17123e7),
        )
        + _poly(eta, (190.051, -3705.08, 23046.2, -46537.3)) * s
        + _poly(eta, (63.6615, -1414.2, 10166.1, -23603.5)) * s2
        + _poly(eta, (-257.524, 5179.97, -33001.4, 67102.4)) * s3
        + 54.9833 * dchi_delta * eta2
    )
    p6 = (
        _poly(
            eta,
            (3111.46, 384.121, -13003.6, 179537.0, -1.19313e6, 3.79886e6, -4.64858e6),
        )
        + _poly(eta, (182.864, -3834.22, 24532.9, -50165.9)) * s
        + _poly(eta, (21.0158, -746.957, 6701.33, -17842.3)) * s2
        + _poly(eta, (-292.855, 5886.62, -37382.4, 75501.8)) * s3
        + 75.5162 * dchi_delta * eta2
    )
    return p1, p2, p3, p4, p5, p6


def _phase_33(mf, state, intrinsic, phase_coeffs, reference_frequency, coa_phase):
    m_over_2 = 1.5
    two_over_m = 2.0 / 3.0
    fcut = (1.0 + 0.001 * (0.25 / state.eta - 1.0)) * m_over_2 * state.f_meco_22
    fmatch_in = m_over_2 * state.f_meco_22
    fmatch_rd = state.f_ring_33 - state.f_damp_33
    f_ring = state.f_ring_33
    f_damp = state.f_damp_33
    points = (
        fcut,
        (math.sqrt(3.0) * (fcut - f_ring) + 2.0 * (fcut + f_ring)) / 4.0,
        (3.0 * fcut + f_ring) / 4.0,
        (fcut + f_ring) / 2.0,
        (fcut + 3.0 * f_ring) / 4.0,
        (fcut + 7.0 * f_ring) / 8.0,
    )
    if state.eta < 0.05 or state.s_tot_r >= 0.8 or state.s_tot_r < 0.0:
        selected = (0, 1, 3, 4, 5)
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
    c0, c_l, c1, c2, c4 = _solve(
        rows, [values[index] for index in selected], mf
    ).unbind()

    lambda_pn = _lambda_pn(state)

    def inspiral_raw(frequency):
        return (
            m_over_2
            / state.eta
            * get_inspiral_phase(two_over_m * frequency, intrinsic, phase_coeffs)
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

    alpha2_21, alpha_l = _ringdown_phase_fits_21(state.base)
    alpha2 = 6.0 * alpha2_21 * state.f_ring_21**2 / f_ring**2

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
            _tensor((state.f_ring_22 - state.f_damp_22) / state.total_mass_seconds, mf),
            intrinsic,
            phase_coeffs,
        )
        / state.total_mass_seconds
    )
    timeshift = linb_fit - dphi22_ref + delta_t
    mf_ref = reference_frequency * state.total_mass_seconds
    phase_ref_22 = Phase(_tensor(reference_frequency, mf), intrinsic, phase_coeffs)
    phiref22 = -phase_ref_22 - timeshift * mf_ref - lina + 2.0 * coa_phase + _PI / 4.0

    f_align = m_over_2 * state.f_meco_22
    if state.eta > 0.05:
        f_align *= 0.6
    mode_insp_align = inspiral_raw(_tensor(f_align, mf)) + c1_insp * f_align + c_insp
    aligned_22 = (
        m_over_2
        * (
            Phase(
                _tensor(two_over_m * f_align / state.total_mass_seconds, mf),
                intrinsic,
                phase_coeffs,
            )
            + lina
            + phiref22
        )
        + timeshift * f_align
    )
    delta_phi = torch.fmod(
        aligned_22 + 3.0 * _PI / 8.0 - mode_insp_align,
        2.0 * _PI,
    )

    inspiral = inspiral_raw(mf) + c1_insp * mf + c_insp + delta_phi
    intermediate = intermediate_raw(mf) + delta_phi
    ringdown = ringdown_raw(mf) + c1_rd * mf + c_rd + delta_phi
    return torch.where(
        mf <= fmatch_in,
        inspiral,
        torch.where(mf <= fmatch_rd, intermediate, ringdown),
    )


def _pn_amplitude_coefficients(state):
    chi_a = state.chi_a
    chi_s = state.chi_s
    eta = state.eta
    delta = state.delta
    pi = _PI
    mode_scale = 2.0 / 3.0
    return (
        0.0j,
        delta * pi ** (1.0 / 3.0) * mode_scale ** (1.0 / 3.0),
        0.0j,
        (-1945.0 * delta + 2268.0 * delta * eta) / 672.0 * pi * mode_scale,
        (
            325.0 * chi_a
            - 1j * 504.0 * delta
            + 325.0 * chi_s * delta
            - 1120.0 * chi_a * eta
            - 80.0 * chi_s * delta * eta
            + 120.0 * delta * pi
            + 1j * 720.0 * delta * math.log(1.5)
        )
        / 120.0
        * pi ** (4.0 / 3.0)
        * mode_scale ** (4.0 / 3.0),
        (
            -2263282560.0 * chi_a * chi_s
            - 1077664867.0 * delta
            + 9053130240.0 * chi_a * chi_s * eta
            - 5926068792.0 * delta * eta
            - 1131641280.0 * delta * chi_a**2
            + 4470681600.0 * delta * eta * chi_a**2
            - 1131641280.0 * delta * chi_s**2
            + 55883520.0 * delta * eta * chi_s**2
            + 2966264784.0 * delta * eta**2
        )
        / 447068160.0
        * pi ** (5.0 / 3.0)
        * mode_scale ** (5.0 / 3.0),
        (
            22007835.0 * chi_a
            + 1j * 26467560.0 * delta
            + 22007835.0 * chi_s * delta
            - 80190540.0 * chi_a * eta
            - 1j * 98774368.0 * delta * eta
            - 31722300.0 * chi_s * delta * eta
            - 9193500.0 * delta * pi
            + 17826480.0 * delta * eta * pi
            - 1j * 37810800.0 * delta * math.log(1.5)
            + 1j * 37558080.0 * delta * eta * math.log(1.5)
            - 12428640.0 * chi_a * eta**2
            - 6078240.0 * chi_s * delta * eta**2
        )
        / 2177280.0
        * pi**2
        * mode_scale**2,
    )


def _pn_amplitude(mf, coefficients, amp_norm):
    frequency_power = mf ** (1.0 / 3.0)
    series = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        series = series * frequency_power + coefficient
    global_factor = (2.0 / 3.0) ** (-7.0 / 6.0) * 0.75 * math.sqrt(5.0 / 7.0)
    return torch.abs(series) * global_factor * mf ** (-7.0 / 6.0) * amp_norm


def _inspiral_cutoff(state):
    comparable_mass = 1.5 * state.f_meco_22
    if state.q < 20.0:
        return comparable_mass
    extreme_mass_ratio = (
        1.25
        * 3.0
        * (
            (
                0.011671068725758493
                - 0.0000858396080377194 * state.chi1
                + 0.000316707064291237 * state.chi1**2
            )
            * (0.8447212540381764 + 6.2873167352395125 * state.eta)
            / (1.2857082764038923 - 0.9977728883419751 * state.chi1)
        )
    )
    blend = 0.5 + 0.5 * math.tanh((state.eta - 0.0192234) / 0.004)
    return blend * comparable_mass + (1.0 - blend) * extreme_mass_ratio


def _inspiral_amplitude_fit_values(state):
    e = state.eta
    e5 = e**5
    e7 = e**7
    sqrt_e = math.sqrt(e)
    delta = state.delta
    d = state.dchi_half
    d2 = d * d
    s = state.chi_pn_hat
    s2 = s * s

    value1 = abs(
        d * e5 * (155.1434307076563 + 26.852777193715088 * s + 1.4157230717300835 * s2)
        + d
        * delta
        * e
        * _poly(e, (6.296698171560171, 15.81328761563562, -141.85538063933927))
        * sqrt_e
        + delta
        * _poly(
            e,
            (
                20.94372147101354,
                68.14577638017842,
                -898.470298591732,
                4598.64854748635,
                -8113.199260593833,
            ),
        )
        * sqrt_e
        + d
        * delta
        * e
        * _poly(e, (29.221863857271703, -348.1658322276406, 965.4670353331536))
        * s
        * sqrt_e
        + delta
        * s
        * (
            -9.753610761811967
            * _poly(
                e,
                (
                    1.7819678168496158,
                    -44.07982999150369,
                    750.8933447725581,
                    -5652.44754829634,
                    19794.855873435758,
                    -26407.40988450443,
                ),
            )
            + 0.014210376114848208
            * _poly(
                e,
                (
                    -196.97328616330392,
                    7264.159472864562,
                    -125763.47850622259,
                    1.1458022059130718e6,
                    -4.948175330328345e6,
                    7.911048294733888e6,
                ),
            )
            * s
            - 0.26859293613553986
            * _poly(
                e,
                (
                    -8.029069605349488,
                    888.7768796633982,
                    -16664.276483466252,
                    128973.72291098491,
                    -462437.2690007375,
                    639989.1197424605,
                ),
            )
            * s2
        )
        * sqrt_e
    )
    value2 = abs(
        d
        * e5
        * (161.62678370819597 + 37.141092711336846 * s - 0.16889712161410445 * s2)
        + d
        * delta
        * e
        * _poly(e, (3.4895829486899825, 51.07954458810889, -249.71072528701757))
        * sqrt_e
        + delta
        * _poly(
            e,
            (
                12.501397517602173,
                35.75290806646574,
                -357.6437296928763,
                1773.8883882162215,
                -3100.2396041211605,
            ),
        )
        * sqrt_e
        + d
        * delta
        * e
        * _poly(e, (13.854211287141906, -135.54916401086845, 327.2467193417936))
        * s
        * sqrt_e
        + delta
        * s
        * (
            -5.2580116732827085
            * _poly(
                e,
                (
                    1.7794900975289085,
                    -48.20753331991333,
                    861.1650630146937,
                    -6879.681319382729,
                    25678.53964955809,
                    -36383.824902258915,
                ),
            )
            + 0.028627002336747746
            * _poly(
                e,
                (
                    -50.57295946557892,
                    734.7581857539398,
                    -2287.0465658878725,
                    15062.821881048358,
                    -168311.2370167227,
                    454655.37836367317,
                ),
            )
            * s
            - 0.15528289788512326
            * _poly(
                e,
                (
                    -12.738184090548508,
                    1129.44485109116,
                    -25091.14888164863,
                    231384.03447562453,
                    -953010.5908118751,
                    1.4516597366230418e6,
                ),
            )
            * s2
        )
        * sqrt_e
    )
    value3 = abs(
        d
        * delta
        * e
        * _poly(e, (-0.5869777957488564, 32.65536124256588, -110.10276573567405))
        + d
        * delta
        * e
        * _poly(e, (3.524800489907584, -40.26479860265549, 113.77466499598913))
        * s
        + delta
        * s
        * (
            -1.2846335585108297
            * _poly(
                e,
                (
                    0.09991079016763821,
                    1.37856806162599,
                    23.26434219690476,
                    -34.842921754693386,
                    -70.83896459998664,
                ),
            )
            - 0.03496714763391888
            * _poly(
                e,
                (
                    -0.230558571912664,
                    188.38585449575902,
                    -3736.1574640444287,
                    22714.70643022915,
                    -43221.0453556626,
                ),
            )
            * s
        )
        + d
        * e7
        * (
            2667.3441342894776
            + 47.94869769580204 * d2
            + 793.5988192446642 * s
            + 293.89657731755483 * s2
        )
        + delta
        * _poly(
            e,
            (
                5.148353856800232,
                148.98231189649468,
                -2774.5868652930294,
                29052.156454239772,
                -162498.31493332976,
                460912.76402476896,
                -521279.50781871413,
            ),
        )
        * sqrt_e
    )
    return value1, value2, value3


def _intermediate_amplitude_fit_values(state):
    e = state.eta
    e5 = e**5
    delta = state.delta
    d = state.dchi_half
    d2 = d * d
    s = state.s_tot_r
    s2 = s * s

    value1 = abs(
        d
        * delta
        * e**2
        * _poly(e, (-0.3516244197696068, 40.425151307421416, -148.3162618111991))
        + delta
        * e
        * _poly(
            e,
            (
                26.998512565991778,
                -146.29035440932105,
                914.5350366065115,
                -3047.513201789169,
                3996.417635728702,
            ),
        )
        + d
        * delta
        * e**2
        * _poly(e, (5.575274516197629, -44.592719238427094, 99.91399033058927))
        * s
        + delta
        * e
        * s
        * (
            -0.5383304368673182
            * _poly(
                e,
                (
                    -7.456619067234563,
                    129.36947401891433,
                    -843.7897535238325,
                    3507.3655567272644,
                    -9675.194644814854,
                    11959.83533107835,
                ),
            )
            - 0.28042799223829407
            * _poly(
                e,
                (
                    -6.212827413930676,
                    266.69059813274475,
                    -4241.537539226717,
                    32634.43965039936,
                    -119209.70783201039,
                    166056.27237509796,
                ),
            )
            * s
        )
        + d * e5 * (199.6863414922219 + 53.36849263931051 * s + 7.650565415855383 * s2)
    )
    value2 = abs(
        delta
        * e
        * _poly(e, (17.42562079069636, -28.970875603981295, 50.726220750178435))
        + d
        * delta
        * e**2
        * _poly(e, (-7.861956897615623, 93.45476935080045, -273.1170921735085))
        + d
        * delta
        * e**2
        * _poly(e, (-0.3265505633310564, -9.861644053348053, 60.38649425562178))
        * s
        + d * e5 * (234.13476431269862 + 51.2153901931183 * s - 10.05114600643587 * s2)
        + delta
        * e
        * s
        * (
            0.3104472390387834
            * _poly(
                e,
                (
                    6.073591341439855,
                    169.85423386969634,
                    -4964.199967099143,
                    42566.59565666228,
                    -154255.3408672655,
                    205525.13910847943,
                ),
            )
            + 0.2295327944679772
            * _poly(
                e,
                (
                    19.236275867648594,
                    -354.7914372697625,
                    1876.408148917458,
                    2404.4151687877525,
                    -41567.07396803811,
                    79210.33893514868,
                ),
            )
            * s
            + 0.30983324991828787
            * _poly(
                e,
                (
                    11.302200127272357,
                    -719.9854052004307,
                    13278.047199998868,
                    -104863.50453518033,
                    376409.2335857397,
                    -504089.07690692553,
                ),
            )
            * s2
        )
    )
    value3 = abs(
        delta
        * e
        * _poly(e, (14.555522136327964, -12.799844096694798, 16.79500349318081))
        + d
        * delta
        * e**2
        * _poly(e, (-16.292654447108134, 190.3516012682791, -562.0936797781519))
        + d
        * delta
        * e**2
        * _poly(e, (-7.048898856045782, 49.941617405768135, -73.62033985436068))
        * s
        + d
        * e5
        * (263.5151703818307 + 44.408527093031566 * s + 10.457035444964653 * s2)
        + delta
        * e
        * s
        * (
            0.4590550434774332
            * _poly(
                e,
                (
                    3.0594364612798635,
                    207.74562213604057,
                    -5545.0086137386525,
                    50003.94075934942,
                    -195187.55422847517,
                    282064.174913521,
                ),
            )
            + 0.657748992123043
            * _poly(
                e,
                (
                    5.57939137343977,
                    -124.06189543062042,
                    1276.6209573025596,
                    -6999.7659193505915,
                    19714.675715229736,
                    -20879.999628681435,
                ),
            )
            * s
            + 0.3695850566805098
            * _poly(
                e,
                (
                    6.077183107132255,
                    -498.95526910874986,
                    10426.348944657859,
                    -91096.64982858274,
                    360950.6686625352,
                    -534437.8832860565,
                ),
            )
            * s2
        )
    )
    value4 = abs(
        delta
        * e
        * _poly(e, (13.312095699772305, -7.449975618083432, 17.098576301150125))
        + delta
        * e**2
        * (
            d * _poly(e, (-31.171150896110156, 371.1389274783572, -1103.1917047361735))
            + d2
            * _poly(e, (32.78644599730888, -395.15713118955387, 1164.9282236341376))
        )
        + d
        * delta
        * e**2
        * _poly(e, (-46.85669289852532, 522.3965959942979, -1485.5134187612182))
        * s
        + d
        * e5
        * (
            287.90444670305715
            - 21.102665129433042 * d2
            + 7.635582066682054 * s
            - 29.471275170013012 * s2
        )
        + delta
        * e
        * s
        * (
            0.6893003654021495
            * _poly(
                e,
                (
                    3.1014226377197027,
                    -44.83989278653052,
                    565.3767256471909,
                    -4797.429130246123,
                    19514.812242035154,
                    -27679.226582207506,
                ),
            )
            + 0.7068016563068026
            * _poly(
                e,
                (
                    4.071212304920691,
                    -118.51094098279343,
                    1788.1730303291356,
                    -13485.270489656365,
                    48603.96661003743,
                    -65658.74746265226,
                ),
            )
            * s
            + 0.2181399561677432
            * _poly(
                e,
                (
                    -1.6754158383043574,
                    303.9394443302189,
                    -6857.936471898544,
                    59288.71069769708,
                    -216137.90827404748,
                    277256.38289831823,
                ),
            )
            * s2
        )
    )
    return value1, value2, value3, value4


def _ringdown_amplitude_fit_values(state):
    e = state.eta
    e5 = e**5
    delta = state.delta
    d = state.dchi_half
    d2 = d * d
    s = state.s_tot_r
    s2 = s * s

    value1 = abs(
        delta
        * e
        * _poly(e, (12.439702602599235, -4.436329538596615, 22.780673360839497))
        + delta
        * e**2
        * (
            d * _poly(e, (-41.04442169938298, 502.9246970179746, -1524.2981907688634))
            + d2 * _poly(e, (32.23960072974939, -365.1526474476759, 1020.6734178547847))
        )
        + d
        * delta
        * e**2
        * _poly(e, (-52.85961155799673, 577.6347407795782, -1653.496174539196))
        * s
        + d
        * e5
        * (
            257.33227387984863
            - 34.5074027042393 * d2
            - 21.836905132600755 * s
            - 15.81624534976308 * s2
        )
        + 13.5
        * delta
        * e
        * s
        * (
            -0.13654149379906394
            * _poly(
                e,
                (
                    2.719687834084113,
                    29.023992126142304,
                    -742.1357702210267,
                    4142.974510926698,
                    -6167.08766058184,
                    -3591.1757995710486,
                ),
            )
            - 0.06248535354306988
            * _poly(
                e,
                (
                    6.697567446351289,
                    -78.23231700361792,
                    444.79350113344543,
                    -1907.008984765889,
                    6601.918552659412,
                    -10056.98422430965,
                ),
            )
            * s
        )
        / (-3.9329308614837704 + s)
    )
    value2 = abs(
        delta * e * (8.425057692276933 + 4.543696144846763 * e)
        + d
        * delta
        * e**2
        * _poly(e, (-32.18860840414171, 412.07321398189293, -1293.422289802462))
        + d
        * delta
        * e**2
        * _poly(e, (-17.18006888428382, 190.73514518113845, -636.4802385540647))
        * s
        + delta
        * e
        * s
        * (
            0.1206817303851239
            * _poly(
                e,
                (
                    8.667503604073314,
                    -144.08062755162752,
                    3188.189172446398,
                    -35378.156133055556,
                    163644.2192178668,
                    -265581.70142471837,
                ),
            )
            + 0.08028332044013944
            * _poly(
                e,
                (
                    12.632478544060636,
                    -322.95832000179297,
                    4777.45310151897,
                    -35625.58409457366,
                    121293.97832549023,
                    -148782.33687815256,
                ),
            )
            * s
        )
        + d
        * e5
        * (
            159.72371180117415
            - 29.10412708633528 * d2
            - 1.873799747678187 * s
            + 41.321480132899524 * s2
        )
    )
    value3 = abs(
        delta * e * (2.485784720088995 + 2.321696430921996 * e)
        + delta
        * e**2
        * (
            d * _poly(e, (-10.454376404653859, 147.10344302665484, -496.1564538739011))
            + d2
            * _poly(e, (-5.9236399792925996, 65.86115501723127, -197.51205149250532))
        )
        + d
        * delta
        * e**2
        * _poly(e, (-10.27418232676514, 136.5150165348149, -473.30988537734174))
        * s
        + d
        * e5
        * (
            32.07819766300362
            - 3.071422453072518 * d2
            + 35.09131921815571 * s
            + 67.23189816732847 * s2
        )
        + 13.5
        * delta
        * e
        * s
        * (
            0.0011484326782460882
            * _poly(
                e,
                (
                    4.1815722950796035,
                    -172.58816646768219,
                    5709.239330076732,
                    -67368.27397765424,
                    316864.0589150127,
                    -517034.11171277676,
                ),
            )
            - 0.009496797093329243
            * _poly(
                e,
                (
                    0.9233282181397624,
                    -118.35865186626413,
                    2628.6024206791726,
                    -23464.64953722729,
                    94309.57566199072,
                    -140089.40725211444,
                ),
            )
            * s
        )
        / (0.09549360183532198 - 0.41099904730526465 * s + s2)
    )
    return value1, value2, value3


def _ringdown_amplitude_parameters(state):
    value1, value2, value3 = _ringdown_amplitude_fit_values(state)
    if value3 >= value2 * value2 / value1:
        value3 = 0.5 * value2 * value2 / value1
    if value3 > value2:
        value3 = 0.5 * value2
    if value1 < value2 and value3 > value1:
        value3 = value1
    denominator = math.sqrt(value1 / value3) - value1 / value2
    if denominator <= 0.0:
        denominator = 1.0e-16
    a0 = value1 * state.f_damp_33 / denominator
    sigma = math.sqrt(a0 / (value2 * state.f_damp_33))
    decay = 0.5 * sigma * math.log(value1 / value3)
    return a0, sigma, decay


def _ringdown_amplitude_core(frequency, state, parameters):
    a0, sigma, decay = parameters
    offset = frequency - state.f_ring_33
    width = state.f_damp_33 * sigma
    return (
        a0
        * state.f_damp_33
        / (torch.exp(decay * offset / width) * (offset * offset + width * width))
    )


def _ringdown_amplitude_derivative(frequency, state, parameters):
    a0, sigma, decay = parameters
    offset = frequency - state.f_ring_33
    fdamp = state.f_damp_33
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


def _inspiral_boundary(function, point, like):
    frequency = _tensor(point, like)
    value = function(frequency)
    if like.dtype != torch.float64:
        return _value_and_derivative(function, point, like)

    # Match LAL's fourth-order finite difference exactly in double precision.
    step = 1.0e-9
    derivative = (
        -function(_tensor(point + 2.0 * step, like))
        + 8.0 * function(_tensor(point + step, like))
        - 8.0 * function(_tensor(point - step, like))
        + function(_tensor(point - 2.0 * step, like))
    ) / (12.0 * step)
    return value, derivative


def _amplitude_33(mf, state):
    amp_norm = math.sqrt(2.0 * state.eta / 3.0) * _PI ** (-1.0 / 6.0)
    pn_dominant = amp_norm * (2.0 / 3.0) ** (-7.0 / 6.0)
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
    fmatch_ringdown = state.f_ring_33 - state.f_damp_33
    ffalloff = state.f_ring_33 + 2.0 * state.f_damp_33
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
    left_value, left_derivative = _inspiral_boundary(inspiral, fcut_inspiral, mf)
    right = _tensor(fmatch_ringdown, mf)
    right_value = ringdown(right)
    right_derivative = _ringdown_amplitude_derivative(right, state, ringdown_parameters)
    values = (
        left_value,
        left_derivative,
        *(_tensor(value, mf) for value in _intermediate_amplitude_fit_values(state)),
        right_value,
        right_derivative,
    )
    rows = []
    for index, frequency in enumerate(intermediate_frequencies):
        inverse_leading_power = frequency ** (-7.0 / 6.0)
        frequency_power = 1.0
        value_row = []
        for _ in range(8):
            value_row.append(frequency_power * inverse_leading_power)
            frequency_power *= frequency
        rows.append(value_row)
        if index in (0, 5):
            frequency_power = 1.0 / frequency
            derivative_row = []
            for power in range(8):
                derivative_row.append(
                    (power - 7.0 / 6.0) * frequency_power * inverse_leading_power
                )
                frequency_power *= frequency
            rows.append(derivative_row)
    intermediate_coefficients = _solve(rows, values, mf)

    def intermediate(frequency):
        polynomial = torch.zeros_like(frequency)
        frequency_power = torch.ones_like(frequency)
        for coefficient in intermediate_coefficients:
            polynomial = polynomial + coefficient * frequency_power
            frequency_power = frequency_power * frequency
        return frequency ** (-7.0 / 6.0) * polynomial

    amplitude = torch.where(
        mf <= fcut_inspiral,
        inspiral(mf),
        torch.where(mf <= fmatch_ringdown, intermediate(mf), ringdown(mf)),
    )
    return torch.where(amplitude < 0.0, _FALSE_ZERO, amplitude)


def imrphenomxhm_h3m3_samples(core, params):
    r"""Return active positive-frequency samples of LAL's :math:`h_{3,-3}`."""

    state = _mode33_state(params)
    if state.mass1 == state.mass2 and state.chi1 == state.chi2:
        return torch.zeros_like(core.polarization)

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
    reference_frequency = float(params.get("f_ref", 0.0))
    if reference_frequency <= 0.0:
        reference_frequency = float(params["f_lower"])
    coa_phase = float(params.get("coa_phase", 0.0))
    with torch_context(frequencies):
        phase = _phase_33(
            mf,
            state,
            intrinsic,
            phase_coeffs,
            reference_frequency,
            coa_phase,
        )
        amplitude = _amplitude_33(mf, state)
        # LAL's positive-frequency convention constructs h_{l,-m} with an
        # additional (-1)^l amplitude factor.  This is negative for l=3.
        samples = -state.amp0 * amplitude * torch.exp(1j * phase)
    return samples.to(core.polarization.dtype)
