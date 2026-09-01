# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""IMRPhenomXHM 2019-release amplitude for the :math:`(3, 3)` mode.

The phase model is shared by the 2019 and 2022 releases, but their amplitude
fits and matching ansatzes differ.  This module contains only the legacy
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
    f_meco = 1.5 * state.f_meco_22
    if state.eta < 0.04535147392290249:
        return 3.75 * (
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
    f_isco = 1.5 * state.f_isco_22
    return f_meco + (0.75 - (0.235 + 5.0 / 6.0) * chi_eff) * abs(
        f_isco - f_meco
    )


def _inspiral_fit_values(state):
    eta = state.eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta
    eta6 = eta5 * eta
    delta = state.delta
    spin = state.chi_pn_hat
    spin2 = spin * spin
    dchi = state.dchi

    value1 = (
        delta
        * (-0.056586690934283326 - 0.14374841547279146 * eta + 0.5584776628959615 * eta2)
        / (-0.3996185676368123 + eta)
        + delta
        * spin
        * (
            (0.056042044149691175 + 0.12482426029674777 * spin) * spin
            + eta * (2.1108074577110343 - 1.7827773156978863 * spin2)
            + eta2
            * (-7.657635515668849 - 0.07646730296478217 * spin + 5.343277927456605 * spin2)
        )
        + 0.45866449225302536 * dchi * (1.0 - 9.603750707244906 * eta2) * eta2
    )
    value2 = (
        delta
        * (
            0.2137734510411439
            - 0.7692194209223682 * eta
            + 26.10570221351058 * eta2
            - 316.0643979123107 * eta3
            + 2090.9063511488234 * eta4
            - 6897.3285171507105 * eta5
            + 8968.893362362503 * eta6
        )
        + delta
        * spin
        * (
            0.018546836505210842
            + 0.05924304311104228 * spin
            + eta
            * (
                1.6484440612224325
                - 0.4683932646001618 * spin
                - 2.110311135456494 * spin2
            )
            + 0.10701786057882816 * spin2
            + eta2
            * (-6.51575737684721 + 1.6692205620001157 * spin + 8.351789152096782 * spin2)
        )
        + 0.3929315188124088 * dchi * (1.0 - 11.289452844364227 * eta2) * eta2
    )
    value3 = (
        delta
        * (
            0.2363760327127446
            + 0.2855410252403732 * eta
            - 10.159877125359897 * eta2
            + 162.65372389693505 * eta3
            - 1154.7315106095564 * eta4
            + 3952.61320206691 * eta5
            - 5207.67472857814 * eta6
        )
        + delta
        * spin
        * (
            0.04573095188775319
            + 0.048249943132325494 * spin
            + eta
            * (
                0.15922377052827502
                - 0.1837289613228469 * spin
                - 0.2834348500565196 * spin2
            )
            + 0.052963737236081304 * spin2
        )
        + 0.25187274502769835 * dchi * (1.0 - 12.172961866410864 * eta2) * eta2
    )
    return abs(value1), abs(value2), abs(value3)


def _intermediate_fit_values(state):
    eta = state.eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta6 = eta4 * eta2
    spin = state.s_tot_r
    common = math.sqrt(max(eta - 4.0 * eta2, 0.0))

    value1 = (
        common
        * (
            27.927652424857733
            - 133.56611389260297 * eta
            + 974.8550901501316 * eta2
            - 3744.785831952632 * eta3
            + 5621.897260910284 * eta4
        )
        + common
        * spin
        * (
            7.348313807306079
            + eta * (-60.248696675045565 - 37.07212326362276 * spin)
            + 5.059236579431119 * spin
            + eta2 * (159.68630712802727 + 83.33807316873204 * spin)
        )
        + 1412.367880056888 * state.dchi * eta6
    )
    value2 = (
        common * (20.162169689041903 - 18.666422946967764 * eta + 53.04107631052987 * eta2)
        + common
        * spin
        * (
            3.896260108714186
            + eta * (-33.707998325000965 - 61.1244771077077 * spin)
            + 4.878506403725656 * spin
            + eta2 * (91.31681057861915 + 196.40535070402336 * spin)
        )
        + 1637.4256048973248 * state.dchi * eta6
    )
    return abs(value1), abs(value2)


def _ringdown_parameters(state):
    eta = state.eta
    eta2 = eta * eta
    eta4 = eta2 * eta2
    delta = state.delta
    spin = state.s_tot_r
    spin2 = spin * spin
    common = math.sqrt(max(eta - 4.0 * eta2, 0.0))
    alambda = abs(
        common * (0.013700854227665184 + 0.01202732427321774 * eta + 0.0898095508889557 * eta2)
        + common
        * spin
        * (
            0.0075858980586079065
            + eta * (-0.013132320758494439 - 0.018186317026076343 * spin)
            + 0.0035617441651710473 * spin
        )
        + eta4
        * (
            state.chi2 * (-0.09802218411554885 - 0.05745949361626237 * spin)
            + state.chi1 * (0.09802218411554885 + 0.05745949361626237 * spin)
            + eta2
            * (
                state.chi1 * (-4.2679864481479886 - 11.877399902871485 * spin)
                + state.chi2 * (4.2679864481479886 + 11.877399902871485 * spin)
            )
            * delta
        )
    )
    decay = (
        0.7435306475478924
        - 0.06688558533374556 * eta
        + 1.471989765837694 * eta2
        + spin
        * (
            0.19457194111990656
            + 0.07564220573555203 * spin
            + eta
            * (
                -0.4809350398289311
                + 0.17261430318577403 * spin
                - 0.1988991467974821 * spin2
            )
        )
        + 1.8881959341735146 * state.dchi * eta2 * delta
    )
    return alambda, decay, 1.3


def _ringdown_rescaled(frequency, state, parameters):
    alambda, decay, sigma = parameters
    offset = frequency - state.f_ring_33
    width = state.f_damp_33 * sigma
    return (
        state.f_damp_33
        * alambda
        * sigma
        * torch.exp(-offset * decay / width)
        / (offset * offset + width * width)
        * frequency ** (-1.0 / 12.0)
    )


def _pn_rescaled(frequency, state):
    from .imrphenomxhm_mode33_torch import _pn_amplitude_coefficients

    coefficients = _tensor(
        _pn_amplitude_coefficients(state), frequency, complex_value=True
    )
    frequency_power = frequency ** (1.0 / 3.0)
    series = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        series = series * frequency_power + coefficient
    global_factor = (2.0 / 3.0) ** (-7.0 / 6.0) * 0.75 * math.sqrt(5.0 / 7.0)
    return torch.abs(series) * global_factor


def _inspiral_model(mf, state, original_cutoff):
    frequencies = [original_cutoff, 0.75 * original_cutoff, 0.5 * original_cutoff]
    fit_values = list(_inspiral_fit_values(state))
    targets = [
        fit - _pn_rescaled(_tensor(frequency, mf), state)
        for fit, frequency in zip(fit_values, frequencies)
    ]
    selected = [0, 1, 2]
    if (
        1.0 < state.q < 1.2
        and state.chi1 < -0.1
        and state.chi2 > 0.0
    ):
        selected.pop(0)

    cutoff = frequencies[selected[0]]
    rows = [
        [
            (frequencies[index] / cutoff) ** ((7.0 + power) / 3.0)
            for power in range(len(selected))
        ]
        for index in selected
    ]
    pseudo = _solve(rows, [targets[index] for index in selected], mf)

    def rescaled_amplitude(frequency):
        amplitude = _pn_rescaled(frequency, state)
        ratio = frequency / cutoff
        for power, coefficient in enumerate(pseudo):
            amplitude = amplitude + coefficient * ratio ** ((7.0 + power) / 3.0)
        return amplitude

    return cutoff, rescaled_amplitude


def amplitude_33_2019(mf, state):
    """Return the full dimensionless 2019-release mode amplitude."""

    if state.eta < _EMR_TWO_REGION_ETA and state.chi1 <= 0.9:
        raise ValueError(
            "the 2019 (3, 3) amplitude's q > 70 two-region branch is not "
            "Torch-native yet"
        )

    amp_norm = math.sqrt(2.0 * state.eta / 3.0) * _PI ** (-1.0 / 6.0)
    original_cutoff = _inspiral_cutoff(state)
    cutoff, inspiral_rescaled = _inspiral_model(mf, state, original_cutoff)
    ringdown_match = 0.95 * state.f_ring_33
    ringdown_parameters = _ringdown_parameters(state)
    use_inspiral_ringdown = state.q > 7.0 and state.chi1 > 0.95

    if use_inspiral_ringdown:
        alambda, decay, sigma = ringdown_parameters
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
