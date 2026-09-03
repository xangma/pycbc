# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""IMRPhenomXHM 2019-release amplitude for the :math:`(4, 4)` mode.

The phase model is shared by the 2019 and 2022 releases, but their amplitude
fits and matching ansatzes differ. This module contains only the legacy
amplitude used by IMRPhenomXO4a.
"""

from __future__ import annotations

import math

import torch

from .imrphenomxhm_mode21_2019_torch import (
    _finite_difference,
    _safe_intermediate_coefficients,
    _solve,
    _tensor,
)


_PI = math.pi
_EMR_TWO_REGION_ETA = 0.013886133703630232


def _inspiral_cutoff(state):
    f_meco = 2.0 * state.f_meco_22
    if state.eta < 0.04535147392290249:
        return 5.0 * (
            (
                0.011671068725758493
                - 0.0000858396080377194 * state.chi1
                + 0.000316707064291237 * state.chi1**2
            )
            * (0.8447212540381764 + 6.2873167352395125 * state.eta)
            / (1.2857082764038923 - 0.9977728883419751 * state.chi1)
        )

    total_mass = state.mass1 + state.mass2
    chi_eff = (state.mass1 * state.chi1 + state.mass2 * state.chi2) / total_mass
    f_isco = 2.0 * state.f_isco_22
    return f_meco + (0.75 - 0.235 * chi_eff) * abs(f_isco - f_meco)


def _inspiral_fit_values(state):
    eta = state.eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta
    delta = state.delta
    spin = state.chi_pn_hat
    spin2 = spin * spin
    spin3 = spin2 * spin
    dchi = state.dchi
    common = math.sqrt(max(1.0 - 3.0 * eta, 0.0))

    value1 = (
        common
        * (
            0.06190013067931406
            + 0.1928897813606222 * eta
            + 1.9024723168424225 * eta2
            - 15.988716302668415 * eta3
            + 35.21461767354364 * eta4
        )
        + common
        * spin
        * (
            0.011454874900772544
            + 0.044702230915643903 * spin
            + eta
            * (
                0.6600413908621988
                + 0.12149520289658673 * spin
                - 0.4482406547006759 * spin2
            )
            + 0.07327810908370004 * spin2
            + eta2
            * (
                -2.1705970511116486
                - 0.6512813450832168 * spin
                + 1.1237234702682313 * spin2
            )
        )
        + 0.4766851579723911 * dchi * (1.0 - 15.950025762198988 * eta2) * eta2
        + 0.127900699645338 * dchi**2 * (1.0 - 15.79329306044842 * eta2) * eta2
    )
    value2 = (
        0.08406011695496626
        - 0.1469952725049322 * eta
        + 0.2997223283799925 * eta2
        - 1.2910560244510723 * eta3
        + (0.023924074703897662 + 0.26110236039648027 * eta - 1.1536009170220438 * eta2)
        * spin
        + (0.04479727299752669 - 0.1439868858871802 * eta + 0.05736387085230215 * eta2)
        * spin2
        + (0.06028104440131858 - 0.4759412992529712 * eta + 1.1090751649419717 * eta2)
        * spin3
        + 0.10346324686812074 * dchi**2 * (1.0 - 16.135903382018213 * eta2) * eta2
        + 0.2648241309154185 * dchi * eta2 * delta
    )
    value3 = (
        0.08212436946985402
        - 0.025332770704783136 * eta
        - 3.2466088293309885 * eta2
        + 28.404235115663706 * eta3
        - 111.36325359782991 * eta4
        + 157.05954559045156 * eta5
        + spin
        * (
            0.03488890057062679
            + 0.039491331923244756 * spin
            + eta
            * (
                -0.08968833480313292
                - 0.12754920943544915 * spin
                - 0.11199012099701576 * spin2
            )
            + 0.034468577523793176 * spin2
        )
        + 0.2062291124580944 * dchi * eta2 * delta
    )
    return abs(value1), abs(value2), abs(value3)


def _intermediate_fit_values(state):
    eta = state.eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    delta = state.delta
    spin = state.s_tot_r
    dchi = state.dchi
    common = math.sqrt(max(eta - 3.0 * eta2, 0.0))

    value1 = (
        common
        * (
            10.804555518381166
            - 72.3834734399584 * eta
            + 540.0541240482852 * eta2
            - 2612.999845214264 * eta3
            + 4779.096001663427 * eta4
        )
        + common
        * spin
        * (
            4.26336253142121
            + eta * (-47.94914754514519 - 39.31284390368824 * spin)
            + 3.0973959822174297 * spin
            + eta2 * (119.70401520575753 + 106.91295627237112 * spin)
        )
        + 0.7262636326998003 * dchi**2 * (1.0 - 4.0 * eta) * eta
        + 3.001401833124412 * dchi * eta2 * delta
    )
    value2 = (
        common
        * (
            9.020721305469884
            - 53.221883492311235 * eta
            + 508.07176447172264 * eta2
            - 3194.0620894511508 * eta3
            + 6769.9274392345915 * eta4
        )
        + common
        * spin
        * (
            3.256591670091969
            + eta * (-38.38922554651356 - 25.286684856422735 * spin)
            + 2.374434219852751 * spin
            + eta2 * (96.41777041220982 + 64.74544118094362 * spin)
        )
        + 3.2337593375595417 * dchi * eta2 * delta
    )
    return abs(value1), abs(value2)


def _ringdown_parameters(state):
    eta = state.eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta
    eta6 = eta5 * eta
    eta7 = eta6 * eta
    delta = state.delta
    spin = state.s_tot_r
    spin2 = spin * spin
    dchi = state.dchi
    common = math.sqrt(max(eta - 3.0 * eta2, 0.0))

    alambda = abs(
        common
        * (
            0.007904587819112173
            + 0.09558474985614368 * eta
            - 2.663803397359775 * eta2
            + 28.298192768381554 * eta3
            - 136.10446022757958 * eta4
            + 233.23167528016833 * eta5
        )
        + common
        * spin
        * (
            0.0049703757209330025
            + 0.004122811292229324 * spin
            + eta * (-0.06166686913913691 + 0.014107365722576927 * spin) * spin
            + eta2
            * (
                -0.2945455034809188
                + 0.4139026619690879 * spin
                - 0.1389170612199015 * spin2
            )
            + eta3
            * (
                0.9225758392294605
                - 0.9656098473922222 * spin
                + 0.19708289555425246 * spin2
            )
            + 0.000657528128497184 * spin2
        )
        + 0.00659873279539475 * dchi * eta2 * delta
    )
    decay = (
        0.7702864948772887
        + 32.81532373698395 * eta
        - 1625.1795901450212 * eta2
        + 31305.458876573215 * eta3
        - 297375.5347399236 * eta4
        + 1.4726521941846698e6 * eta5
        - 3.616582470072637e6 * eta6
        + 3.4585865843680725e6 * eta7
        + (
            -0.03011582009308575 * spin
            + 0.09034746027925727 * eta * spin
            + 1.8738784391649446 * eta2 * spin
            - 5.621635317494836 * eta3 * spin
        )
        / (-1.1340218677260014 + spin)
        + 0.959943270591552 * dchi**2 * eta2
        + 0.853573071529436 * dchi * eta2 * delta
    )
    return alambda, decay, 1.33


def _ringdown_rescaled(frequency, state, parameters):
    alambda, decay, sigma = parameters
    offset = frequency - state.f_ring_44
    width = state.f_damp_44 * sigma
    return (
        state.f_damp_44
        * alambda
        * sigma
        * torch.exp(-offset * decay / width)
        / (offset * offset + width * width)
        * frequency ** (-1.0 / 12.0)
    )


def _pn_rescaled(frequency, state):
    from .imrphenomxhm_mode44_torch import _pn_amplitude_coefficients

    coefficients = _tensor(
        _pn_amplitude_coefficients(state), frequency, complex_value=True
    )
    frequency_power = frequency ** (1.0 / 3.0)
    series = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        series = series * frequency_power + coefficient
    global_factor = 0.5 ** (-7.0 / 6.0) * (4.0 / 9.0) * math.sqrt(10.0 / 7.0)
    return torch.abs(series) * global_factor


def _inspiral_model(mf, state, original_cutoff):
    frequencies = [original_cutoff, 0.75 * original_cutoff, 0.5 * original_cutoff]
    fit_values = list(_inspiral_fit_values(state))
    targets = [
        fit - _pn_rescaled(_tensor(frequency, mf), state)
        for fit, frequency in zip(fit_values, frequencies)
    ]
    cutoff = frequencies[0]
    rows = [
        [
            (frequency / cutoff) ** ((7.0 + power) / 3.0)
            for power in range(3)
        ]
        for frequency in frequencies
    ]
    pseudo = _solve(rows, targets, mf)

    def rescaled_amplitude(frequency):
        amplitude = _pn_rescaled(frequency, state)
        ratio = frequency / cutoff
        for power, coefficient in enumerate(pseudo):
            amplitude = amplitude + coefficient * ratio ** ((7.0 + power) / 3.0)
        return amplitude

    return cutoff, rescaled_amplitude


def amplitude_44_2019(mf, state):
    """Return the full dimensionless 2019-release mode amplitude."""

    if state.eta < _EMR_TWO_REGION_ETA and state.chi1 <= 0.9:
        raise ValueError(
            "the 2019 (4, 4) amplitude's q > 70 two-region branch is not "
            "Torch-native yet"
        )

    amp_norm = math.sqrt(2.0 * state.eta / 3.0) * _PI ** (-1.0 / 6.0)
    original_cutoff = _inspiral_cutoff(state)
    cutoff, inspiral_rescaled = _inspiral_model(mf, state, original_cutoff)
    ringdown_match = 0.9 * state.f_ring_44
    ringdown_parameters = _ringdown_parameters(state)
    use_inspiral_ringdown = state.q > 7.0 and state.chi1 > 0.95

    if use_inspiral_ringdown:
        _, decay, sigma = ringdown_parameters
        ringdown_without_scale = _ringdown_rescaled(
            _tensor(ringdown_match, mf),
            state,
            (1.0, decay, sigma),
        )
        alambda = 0.9 * torch.abs(
            inspiral_rescaled(_tensor(cutoff, mf)) / ringdown_without_scale
        )
        ringdown_parameters = (alambda, decay, sigma)

    def ringdown_rescaled(frequency):
        return _ringdown_rescaled(frequency, state, ringdown_parameters)

    original_width = ringdown_match - original_cutoff
    f2 = original_cutoff + original_width / 3.0
    f3 = original_cutoff + 2.0 * original_width / 3.0
    f1 = cutoff
    f4 = ringdown_match
    insp_f1 = inspiral_rescaled(_tensor(f1, mf))
    rd_f4 = ringdown_rescaled(_tensor(f4, mf))
    dinsp_f1 = _finite_difference(inspiral_rescaled, f1, mf)
    with torch.enable_grad():
        rd_point = _tensor(f4, mf).detach().requires_grad_(True)
        rd_value = ringdown_rescaled(rd_point)
        drd_f4 = torch.autograd.grad(rd_value, rd_point)[0].detach()

    d1 = (7.0 / 6.0) * f1 ** (1.0 / 6.0) / insp_f1 - f1 ** (
        7.0 / 6.0
    ) * dinsp_f1 / insp_f1**2
    d4 = (7.0 / 6.0) * f4 ** (1.0 / 6.0) / rd_f4 - f4 ** (
        7.0 / 6.0
    ) * drd_f4 / rd_f4**2
    fit2, fit3 = _intermediate_fit_values(state)
    v1 = 1.0 / (f1 ** (-7.0 / 6.0) * insp_f1)
    v2 = 1.0 / fit2
    v3 = 1.0 / fit3
    v4 = 1.0 / (f4 ** (-7.0 / 6.0) * rd_f4)
    version = 105

    if use_inspiral_ringdown:
        v2 = v3 = 1.0
        version = 101
    elif state.q > 40.0 and state.chi1 > 0.9:
        v2 = v3 = 1.0
        version = 1032

    if v3 == 1.0:
        v3, f3 = v2, f2
        v2 = 1.0

    values = (v1, v2, v3, v4)
    frequencies = (f1, f2, f3, f4)
    coefficients = _safe_intermediate_coefficients(
        version, values, frequencies, (d1, d4), mf
    )

    def intermediate(frequency):
        polynomial = coefficients[-1]
        for coefficient in reversed(coefficients[:-1]):
            polynomial = polynomial * frequency + coefficient
        return amp_norm / polynomial

    inspiral = amp_norm * mf ** (-7.0 / 6.0) * inspiral_rescaled(mf)
    ringdown = amp_norm * mf ** (-7.0 / 6.0) * ringdown_rescaled(mf)
    return torch.where(
        mf <= cutoff,
        inspiral,
        torch.where(mf <= ringdown_match, intermediate(mf), ringdown),
    )
