# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch-native IMRPhenomXHM :math:`(4, -4)` mode.

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
from .imrphenomxhm_mode33_torch import _inspiral_boundary, _poly


_PI = lal.PI
_FALSE_ZERO = 1.0e-15


@dataclass(frozen=True)
class _Mode44State:
    """Mode-specific QNM data layered over the common XHM state."""

    base: object
    f_ring_44: float
    f_damp_44: float

    def __getattr__(self, name):
        return getattr(self.base, name)


def _qnm_fring_44(final_spin):
    numerator = (
        0.1287821193485683
        - 0.21224284094693793 * final_spin
        + 0.0710926778043916 * final_spin**2
        + 0.015487322972031054 * final_spin**3
        - 0.002795401084713644 * final_spin**4
        + 0.000045483523029172406 * final_spin**5
        + 0.00034775290179000503 * final_spin**6
    )
    denominator = (
        1.0
        - 1.9931645124693607 * final_spin
        + 1.0593147376898773 * final_spin**2
        - 0.06378640753152783 * final_spin**4
    )
    return numerator / denominator


def _qnm_fdamp_44(final_spin):
    numerator = (
        0.014986847152355699
        - 0.01722587715950451 * final_spin
        - 0.0016734788189065538 * final_spin**2
        + 0.0002837322846047305 * final_spin**3
        + 0.002510528746148588 * final_spin**4
        + 0.00031983835498725354 * final_spin**5
        + 0.000812185411753066 * final_spin**6
    )
    denominator = (
        1.0
        - 1.1350205970682399 * final_spin
        - 0.0500827971270845 * final_spin**2
        + 0.13983808071522857 * final_spin**4
        + 0.051876225199833995 * final_spin**6
    )
    return numerator / denominator


def _mode44_state(params):
    base = _mode21_state(params)
    final_mass = 1.0 - base.radiated_energy
    return _Mode44State(
        base=base,
        f_ring_44=_qnm_fring_44(base.final_spin) / final_mass,
        f_damp_44=_qnm_fdamp_44(base.final_spin) / final_mass,
    )


def _lambda_pn(state):
    eta = state.eta
    if eta > 0.01:
        output = (
            45045.0
            * _PI
            * (336.0 - 1193.0 * eta + 320.0 * (-1.0 + 3.0 * eta) * math.log(2.0))
            / (2.0 * (-1801800.0 + 5405400.0 * eta))
        )
        return -output

    eta2 = eta * eta
    s = state.s_tot_r
    no_spin = _poly(
        eta,
        (
            5.254484747463392,
            -21.277760168559862,
            160.43721442910618,
            -1162.954360723399,
            1685.5912722190276,
            -1538.6661348106031,
        ),
    )
    eq_spin = (
        _poly(
            eta,
            (
                0.007067861615983771,
                -10.945895160727437,
                246.8787141453734,
                -810.7773268493444,
            ),
        )
        * s
        + _poly(
            eta,
            (
                0.17447830920234977,
                4.530539154777984,
                -176.4987316167203,
                621.6920322846844,
            ),
        )
        * s**2
    )
    unequal_spin = -8.384066369867833 * state.dchi * state.delta * eta2
    return no_spin + eq_spin + unequal_spin


def _intermediate_phase_fit_values(state):
    eta = state.eta
    eta2 = eta * eta
    s = state.s_tot_r
    s2 = s * s
    s3 = s2 * s
    dchi_delta = state.dchi * state.delta

    p1 = (
        4349.66
        + 4.34125 / eta
        + _poly(
            eta, (0.0, -8202.33, 5534.1, 536500.0, -4.33197e6, 1.37792e7, -1.60802e7)
        )
        + (
            _poly(eta, (12.0704, -528.098, 1822.9100000000003, -9349.73, 17900.9)) * s
            + _poly(eta, (10.4092, 253.334, -5452.04, 35416.6, -71523.0)) * s2
            + eta * _poly(eta, (492.60300000000007, -9508.5, 57303.4, -109418.0)) * s3
        )
        / eta
        + 262.143 * dchi_delta * eta * (3.0782778864970646 + eta)
    )
    p2 = (
        3804.19
        + 0.66144 / eta
        + _poly(
            eta, (0.0, -2421.77, -33475.8, 665951.0, -4.50145e6, 1.37792e7, -1.60802e7)
        )
        + (
            _poly(eta, (5.83038, -172.047, 926.576, -7676.87, 17900.9)) * s
            + _poly(eta, (6.17601, 253.334, -5672.02, 35722.1, -71523.0)) * s2
            + eta * _poly(eta, (492.60300000000007, -9508.5, 57303.4, -109418.0)) * s3
        )
        / eta
        + 262.143 * dchi_delta * eta * (1.0543062374352932 + eta)
    )
    p3 = (
        _poly(
            eta,
            (3308.97, 2353.58, -66340.1, 777272.0, -4.64438e6, 1.37792e7, -1.60802e7),
        )
        + _poly(eta, (-21.5697, 926.576, -7989.26, 17900.9)) * s
        + _poly(eta, (353.539, -6403.24, 37599.5, -71523.0)) * s2
        + _poly(eta, (492.603, -9508.5, 57303.4, -109418.0)) * s3
        + 262.143 * dchi_delta * eta2
    )
    p4 = (
        _poly(
            eta, (3245.63, -928.56, 8463.89, -17422.6, -165169.0, 908279.0, -1.31138e6)
        )
        + _poly(eta, (32.506, -590.293, 3536.61, -6758.52)) * s
        + _poly(eta, (-25.7716, 738.141, -4867.87, 9129.45)) * s2
        + _poly(eta, (-15.7439, 620.695, -4679.24, 9582.58)) * s3
        + 87.0832 * dchi_delta * eta2
    )
    p5 = (
        _poly(
            eta,
            (
                3108.38,
                3722.46,
                -119588.0,
                1.92148e6,
                -1.69796e7,
                8.39194e7,
                -2.17143e8,
                2.2829700000000003e8,
            ),
        )
        + eta * _poly(eta, (118.319, -529.854)) * s
        + _poly(eta, (21.0314, -240.648, 516.333)) * s2
        + _poly(eta, (20.3384, -356.241, 999.417)) * s3
        + 97.1364 * dchi_delta * eta2
    )
    p6 = (
        _poly(
            eta,
            (3096.03, 986.752, -20371.1, 220332.0, -1.31523e6, 4.29193e6, -6.01179e6),
        )
        + _poly(eta, (-9.96292, -118.526, 2255.76, -6758.52)) * s
        + _poly(eta, (-14.4869, 370.039, -3605.8, 9129.45)) * s2
        + _poly(eta, (17.0209, 70.1931, -3070.08, 9582.58)) * s3
        + 23.0759 * dchi_delta * eta2
    )
    return p1, p2, p3, p4, p5, p6


def _phase_44(mf, state, intrinsic, phase_coeffs, reference_frequency, coa_phase):
    m_over_2 = 2.0
    two_over_m = 0.5
    fcut = (1.0 + 0.001 * (0.25 / state.eta - 1.0)) * m_over_2 * state.f_meco_22
    fmatch_in = m_over_2 * state.f_meco_22
    fmatch_rd = state.f_ring_44 - state.f_damp_44
    f_ring = state.f_ring_44
    f_damp = state.f_damp_44
    points = (
        fcut,
        (math.sqrt(3.0) * (fcut - f_ring) + 2.0 * (fcut + f_ring)) / 4.0,
        (3.0 * fcut + f_ring) / 4.0,
        (fcut + f_ring) / 2.0,
        (fcut + 3.0 * f_ring) / 4.0,
        (fcut + 7.0 * f_ring) / 8.0,
    )
    if state.eta < 0.05 or state.s_tot_r >= 0.8:
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
    delta_phi = torch.fmod(aligned_22 + 3.0 * _PI / 4.0 - mode_insp_align, 2.0 * _PI)

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
    mode_scale = 0.5
    return (
        0.0j,
        0.0j,
        (1.0 - 3.0 * eta) * pi ** (2.0 / 3.0) * mode_scale ** (2.0 / 3.0),
        0.0j,
        (-158383.0 + 641105.0 * eta - 446460.0 * eta**2)
        / 36960.0
        * pi ** (4.0 / 3.0)
        * mode_scale ** (4.0 / 3.0),
        (
            -1j * 1008.0
            + 565.0 * chi_s
            + 565.0 * chi_a * delta
            + 1j * 3579.0 * eta
            - 2075.0 * chi_s * eta
            - 1695.0 * chi_a * delta * eta
            + 240.0 * pi
            - 720.0 * eta * pi
            + 1j * 960.0 * math.log(2.0)
            - 1j * 2880.0 * eta * math.log(2.0)
            + 1140.0 * chi_s * eta**2
        )
        / 120.0
        * pi ** (5.0 / 3.0)
        * mode_scale ** (5.0 / 3.0),
        (
            7888301437.0
            - 147113366400.0 * chi_a * chi_s * delta
            - 745140957231.0 * eta
            + 441340099200.0 * chi_a * chi_s * delta * eta
            - 73556683200.0 * chi_a**2
            + 511264353600.0 * eta * chi_a**2
            - 73556683200.0 * chi_s**2
            + 224302478400.0 * eta * chi_s**2
            + 2271682065240.0 * eta**2
            - 871782912000.0 * chi_a**2 * eta**2
            - 10897286400.0 * chi_s**2 * eta**2
            - 805075876080.0 * eta**3
        )
        / 29059430400.0
        * pi**2
        * mode_scale**2,
    )


def _pn_amplitude(mf, coefficients, amp_norm):
    frequency_power = mf ** (1.0 / 3.0)
    series = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        series = series * frequency_power + coefficient
    global_factor = 0.5 ** (-7.0 / 6.0) * (4.0 / 9.0) * math.sqrt(10.0 / 7.0)
    return torch.abs(series) * global_factor * mf ** (-7.0 / 6.0) * amp_norm


def _inspiral_cutoff(state):
    comparable_mass = 2.0 * state.f_meco_22
    if state.q < 20.0:
        return comparable_mass
    extreme_mass_ratio = (
        1.25
        * 4.0
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
    sqrt_e = math.sqrt(e)
    delta = state.delta
    d = state.dchi_half
    d2 = d * d
    s = state.chi_pn_hat
    s2 = s * s
    s3 = s2 * s

    # Return the calibrated values in the ascending frequency order used by
    # the Torch solve: 0.5, 0.75, and 1.0 times f_cut.
    iv1 = abs(
        (
            d
            * delta
            * _poly(e, (0.0, 0.5697308729057493, 8.895576813118867, -34.98399465240273))
            + d2
            * _poly(
                e, (0.0, 1.6370346538130884, -14.597095790380884, 33.182723737396294)
            )
        )
        * sqrt_e
        + _poly(
            e,
            (
                5.2601381002242595,
                -3.557926105832778,
                -138.9749850448088,
                603.7453704122706,
                -923.5495700703648,
            ),
        )
        * sqrt_e
        + s
        * (
            -0.41839636169678796
            * _poly(
                e,
                (
                    5.143510231379954,
                    104.62892421207803,
                    -4232.508174045782,
                    50694.024801783446,
                    -283097.33358214336,
                    758333.2655404843,
                    -788783.0559069642,
                ),
            )
            - 0.05653522061311774
            * _poly(
                e,
                (
                    5.605483124564013,
                    694.00652410087,
                    -17551.398321516353,
                    165236.6480734229,
                    -761661.9645651339,
                    1.7440315410044065e6,
                    -1.6010489769238676e6,
                ),
            )
            * s
            - 0.023693246676754775
            * _poly(
                e,
                (
                    16.437107575918503,
                    -2911.2154288136217,
                    89338.32554683842,
                    -1.0803340811860575e6,
                    6.255666490084672e6,
                    -1.7434160932177313e7,
                    1.883460394974573e7,
                ),
            )
            * s2
        )
        * sqrt_e
    )
    iv2 = abs(
        (
            d2
            * _poly(
                e, (0.0, -0.8318312659717388, 7.6541168007977864, -16.648660653220123)
            )
            + d
            * delta
            * _poly(e, (0.0, 2.214478316304753, -7.028104574328955, 5.56587823143958))
        )
        * sqrt_e
        + _poly(
            e,
            (
                3.173191054680422,
                6.707695566702527,
                -155.22519772642607,
                604.0067075996933,
                -876.5048298377644,
            ),
        )
        * sqrt_e
        + d
        * delta
        * _poly(e, (0.0, 4.749663394334708, -42.62996105525792, 97.01712147349483))
        * s
        * sqrt_e
        + s
        * (
            -0.2627203100303006
            * _poly(
                e,
                (
                    6.460396349297595,
                    -52.82425783851536,
                    -552.1725902144143,
                    12546.255587592654,
                    -81525.50289542897,
                    227254.37897941095,
                    -234487.3875219032,
                ),
            )
            - 0.008424003742397579
            * _poly(
                e,
                (
                    -109.26773035716548,
                    15514.571912666677,
                    -408022.6805482195,
                    4.620165968920881e6,
                    -2.6446950627957724e7,
                    7.539643948937692e7,
                    -8.510662871580401e7,
                ),
            )
            * s
            - 0.008830881730801855
            * _poly(
                e,
                (
                    -37.49992494976597,
                    1359.7883958101172,
                    -23328.560285901796,
                    260027.4121353132,
                    -1.723865744472182e6,
                    5.858455766230802e6,
                    -7.756341721552802e6,
                ),
            )
            * s2
            - 0.027167813927224657
            * _poly(
                e,
                (
                    34.281932237450256,
                    -3312.7658728016568,
                    84126.14531363266,
                    -956052.0170024392,
                    5.570748509263883e6,
                    -1.6270212243584689e7,
                    1.8855858173287075e7,
                ),
            )
            * s3
        )
        * sqrt_e
    )
    iv3 = abs(
        (
            d
            * delta
            * _poly(
                e, (0.0, 1.4739380748149558, 0.06541707987699942, -9.473290540936633)
            )
            + d2
            * _poly(
                e, (0.0, -0.3640838331639651, 3.7369795937033756, -8.709159662885131)
            )
        )
        * sqrt_e
        + _poly(
            e,
            (
                1.7335503724888923,
                12.656614578053683,
                -139.6610487470118,
                456.78649322753824,
                -599.2709938848282,
            ),
        )
        * sqrt_e
        + d
        * delta
        * _poly(e, (0.0, 2.3532739003216254, -21.37216554136868, 53.35003268489743))
        * s
        * sqrt_e
        + s
        * (
            -0.15782329022461472
            * _poly(
                e,
                (
                    6.0309399412954345,
                    -229.16361598098678,
                    3777.477006415653,
                    -31109.307191210424,
                    139319.8239886073,
                    -324891.4001578353,
                    307714.3954026392,
                ),
            )
            - 0.03050157254864058
            * _poly(
                e,
                (
                    4.232861441291087,
                    1609.4251694451375,
                    -51213.27604422822,
                    612317.1751155312,
                    -3.5589766538499263e6,
                    1.0147654212772278e7,
                    -1.138861230369246e7,
                ),
            )
            * s
            - 0.026407497690308382
            * _poly(
                e,
                (
                    -17.184685557542196,
                    744.4743953122965,
                    -10494.512487701073,
                    66150.52694069289,
                    -184787.79377504133,
                    148102.4257785174,
                    128167.89151782403,
                ),
            )
            * s2
        )
        * sqrt_e
    )
    return iv1, iv2, iv3


def _intermediate_amplitude_fit_values(state):
    e = state.eta
    delta = state.delta
    d = state.dchi_half
    d2 = d * d
    s = state.chi_pn_hat
    s2 = s * s

    value1 = abs(
        e
        * (
            d
            * delta
            * _poly(
                e, (0.0, 1.5378890240544967, -3.4499418893734903, 16.879953490422782)
            )
            + d2
            * _poly(e, (0.0, 1.720226708214248, -11.87925165364241, 23.259283336239545))
        )
        + e
        * _poly(
            e,
            (
                8.790173464969538,
                -64.95499142822892,
                324.1998823562892,
                -1111.9864921907126,
                1575.602443847111,
            ),
        )
        + e
        * s
        * (
            -0.062333275821238224
            * _poly(
                e,
                (
                    -21.630297087123807,
                    137.4395894877131,
                    64.92115530780129,
                    -1013.1110639471394,
                ),
            )
            - 0.11014697070998722
            * _poly(
                e,
                (
                    4.149721483857751,
                    -108.6912882442823,
                    831.6073263887092,
                    -1828.2527520190122,
                ),
            )
            * s
            - 0.07704777584463054
            * _poly(
                e,
                (
                    4.581767671445529,
                    -50.35070009227704,
                    344.9177692251726,
                    -858.9168637051405,
                ),
            )
            * s2
        )
    )
    value2 = abs(
        e
        * (
            d
            * delta
            * _poly(
                e, (0.0, 2.3123974306694057, -12.237594841284904, 44.78225529547671)
            )
            + d2
            * _poly(
                e, (0.0, 2.9282931698944292, -25.624210264341933, 61.05270871360041)
            )
        )
        + e
        * _poly(
            e,
            (
                6.98072197826729,
                -46.81443520117986,
                236.76146303619544,
                -920.358408667518,
                1478.050456337336,
            ),
        )
        + e
        * s
        * (
            -0.07801583359561987
            * _poly(
                e,
                (
                    -28.29972282146242,
                    752.1603553640072,
                    -10671.072606753183,
                    83447.0461509547,
                    -350025.2112501252,
                    760889.6919776166,
                    -702172.2934567826,
                ),
            )
            + 0.013159545629626014
            * _poly(
                e,
                (
                    91.1469833190294,
                    -3557.5003799977294,
                    52391.684517955284,
                    -344254.9973814295,
                    1.0141877915334814e6,
                    -1.1505186449682908e6,
                    268756.85659532435,
                ),
            )
            * s
        )
    )
    value3 = abs(
        e
        * (
            d
            * delta
            * _poly(
                e, (0.0, -0.8765502142143329, 22.806632458441996, -43.675503209991184)
            )
            + d2
            * _poly(
                e, (0.0, 0.48698617426180074, -4.302527065360426, 16.18571810759235)
            )
        )
        + e
        * _poly(
            e,
            (
                6.379772583015967,
                -44.10631039734796,
                269.44092930942793,
                -1285.7635006711453,
                2379.538739132234,
            ),
        )
        + e
        * s
        * (
            -0.23316184683282615
            * _poly(
                e,
                (
                    -1.7279023138971559,
                    -23.606399143993716,
                    409.3387618483284,
                    -1115.4147472977265,
                ),
            )
            - 0.09653777612560172
            * _poly(
                e,
                (
                    -5.310643306559746,
                    -2.1852511802701264,
                    541.1248219096527,
                    -1815.7529908827103,
                ),
            )
            * s
            - 0.060477799540741804
            * _poly(
                e,
                (
                    -14.578189130145661,
                    175.6116682068523,
                    -569.4799973930861,
                    426.0861915646515,
                ),
            )
            * s2
        )
    )
    value4 = abs(
        e
        * (
            d
            * delta
            * _poly(e, (0.0, -2.461738962276138, 45.3240543970684, -112.2714974622516))
            + d2
            * _poly(e, (0.0, 0.9158352037567031, -8.724582331021695, 28.44633544874233))
        )
        + e
        * _poly(
            e,
            (
                6.098676337298138,
                -45.42463610529546,
                350.97192927929433,
                -2002.2013283876834,
                4067.1685640401033,
            ),
        )
        + e
        * s
        * (
            -0.36068516166901304
            * _poly(
                e,
                (
                    -2.120354236840677,
                    -47.56175350408845,
                    1618.4222330016048,
                    -14925.514654896673,
                    60287.45399959349,
                    -91269.3745059139,
                ),
            )
            - 0.09635801207669747
            * _poly(
                e,
                (
                    -11.824692837267394,
                    371.7551657959369,
                    -4176.398139238679,
                    16655.87939259747,
                    -4102.218189945819,
                    -67024.98285179552,
                ),
            )
            * s
            - 0.06565232123453196
            * _poly(
                e,
                (
                    -26.15227471380236,
                    1869.0168486099005,
                    -33951.35186039629,
                    253694.6032002248,
                    -845341.6001856657,
                    1.0442282862506858e6,
                ),
            )
            * s2
        )
    )
    return value1, value2, value3, value4


def _ringdown_amplitude_fit_values(state):
    e = state.eta
    delta = state.delta
    d = state.dchi_half
    d2 = d * d
    s = state.s_tot_r
    s2 = s * s

    value1 = abs(
        e
        * (
            d
            * delta
            * _poly(e, (0.0, -8.51952446214978, 117.76530248141987, -297.2592736781142))
            + d2
            * _poly(
                e, (0.0, -0.2750098647982238, 4.456900599347149, -8.017569928870929)
            )
        )
        + e
        * _poly(
            e,
            (
                5.635069974807398,
                -33.67252878543393,
                287.9418482197136,
                -3514.3385364216438,
                25108.811524802128,
                -98374.18361532023,
                158292.58792484726,
            ),
        )
        + e
        * s
        * (
            -0.4360849737360132
            * _poly(
                e,
                (
                    -0.9543114627170375,
                    -58.70494649755802,
                    1729.1839588870455,
                    -16718.425586396803,
                    71236.86532610047,
                    -111910.71267453219,
                ),
            )
            - 0.024861802943501172
            * _poly(
                e,
                (
                    -52.25045490410733,
                    1585.462602954658,
                    -15866.093368857853,
                    35332.328181283,
                    168937.32229060197,
                    -581776.5303770923,
                ),
            )
            * s
            + 0.005856387555754387
            * _poly(
                e,
                (
                    186.39698091707513,
                    -9560.410655118145,
                    156431.3764198244,
                    -1.0461268207440731e6,
                    3.054333578686424e6,
                    -3.2369858387064277e6,
                ),
            )
            * s2
        )
    )
    value2 = abs(
        e
        * (
            d
            * delta
            * _poly(
                e, (0.0, -2.861653255976984, 50.50227103211222, -123.94152825700999)
            )
            + d2
            * _poly(e, (0.0, 2.9415751419018865, -28.79779545444817, 72.40230240887851))
        )
        + e
        * _poly(
            e,
            (
                3.2461722686239307,
                25.15310593958783,
                -792.0167314124681,
                7168.843978909433,
                -30595.4993786313,
                49148.57065911245,
            ),
        )
        + e
        * s
        * (
            -0.23311779185707152
            * _poly(
                e,
                (
                    -1.0795711755430002,
                    -20.12558747513885,
                    1163.9107546486134,
                    -14672.23221502075,
                    73397.72190288734,
                    -127148.27131388368,
                ),
            )
            + 0.025805905356653
            * _poly(
                e,
                (
                    11.929946153728276,
                    350.93274421955806,
                    -14580.02701600596,
                    174164.91607515427,
                    -819148.9390278616,
                    1.3238624538095295e6,
                ),
            )
            * s
            + 0.019740635678180102
            * _poly(
                e,
                (
                    -7.046295936301379,
                    1535.781942095697,
                    -27212.67022616794,
                    201981.0743810629,
                    -696891.1349708183,
                    910729.0219043035,
                ),
            )
            * s2
        )
    )
    value3 = abs(
        e
        * (
            d
            * delta
            * _poly(
                e, (0.0, 2.4286414692113816, -23.213332913737403, 66.58241012629095)
            )
            + d2
            * _poly(e, (0.0, 3.085167288859442, -31.60440418701438, 78.49621016381445))
        )
        + e
        * _poly(
            e,
            (
                0.861883217178703,
                13.695204704208976,
                -337.70598252897696,
                2932.3415281149432,
                -12028.786386004691,
                18536.937955014455,
            ),
        )
        + e
        * s
        * (
            -0.048465588779596405
            * _poly(
                e,
                (
                    -0.34041762314288154,
                    -81.33156665674845,
                    1744.329802302927,
                    -16522.343895064576,
                    76620.18243090731,
                    -133340.93723954144,
                ),
            )
            + 0.024804027856323612
            * _poly(
                e,
                (
                    -8.666095805675418,
                    711.8727878341302,
                    -13644.988225595187,
                    112832.04975245205,
                    -422282.0368440555,
                    584744.0406581408,
                ),
            )
            * s
        )
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
    a0 = value1 * state.f_damp_44 / denominator
    sigma = math.sqrt(a0 / (value2 * state.f_damp_44))
    decay = 0.5 * sigma * math.log(value1 / value3)
    return a0, sigma, decay


def _ringdown_amplitude_core(frequency, state, parameters):
    a0, sigma, decay = parameters
    offset = frequency - state.f_ring_44
    width = state.f_damp_44 * sigma
    return (
        a0
        * state.f_damp_44
        / (torch.exp(decay * offset / width) * (offset * offset + width * width))
    )


def _ringdown_amplitude_derivative(frequency, state, parameters):
    a0, sigma, decay = parameters
    offset = frequency - state.f_ring_44
    fdamp = state.f_damp_44
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


def _amplitude_44(mf, state):
    amp_norm = math.sqrt(2.0 * state.eta / 3.0) * _PI ** (-1.0 / 6.0)
    pn_dominant = amp_norm * 0.5 ** (-7.0 / 6.0)
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
    fmatch_ringdown = state.f_ring_44 - state.f_damp_44
    ffalloff = state.f_ring_44 + 2.0 * state.f_damp_44
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


def imrphenomxhm_h4m4_samples(
    core,
    params,
    *,
    frequencies=None,
    reference_frequency=None,
):
    r"""Return active positive-frequency samples of LAL's :math:`h_{4,-4}`."""

    state = _mode44_state(params)
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
        phase = _phase_44(
            mf,
            state,
            intrinsic,
            phase_coeffs,
            reference_frequency,
            coa_phase,
        )
        amplitude = _amplitude_44(mf, state)
        # LAL's positive-frequency convention multiplies h_{l,-m} by
        # (-1)^l.  The factor is positive for l=4.
        samples = state.amp0 * amplitude * torch.exp(1j * phase)
    return samples.to(core.polarization.dtype)
