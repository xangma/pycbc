# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""IMRPhenomXHM 2019-release amplitude helpers for the mixed (3, 2) mode.

The legacy model uses the 2018 amplitude fits, an inverse-polynomial
intermediate ansatz, and different inspiral/ringdown matching frequencies.
Frequency-dependent assembly remains in :mod:`imrphenomxhm_mode32_torch` so
the spherical/spheroidal mixing is shared with the current release.
"""

from __future__ import annotations

import math

import torch

from .imrphenomxhm_mode21_2019_torch import _solve, _tensor


_EMR_TWO_REGION_ETA = 0.013886133703630232


def inspiral_cutoff_32_2019(state):
    """Return the pre-veto inspiral/intermediate matching frequency."""

    if state.eta < 0.04535147392290249:
        return 2.5 * (
            (
                0.011671068725758493
                - 0.0000858396080377194 * state.chi1
                + 0.000316707064291237 * state.chi1**2
            )
            * (0.8447212540381764 + 6.2873167352395125 * state.eta)
            / (1.2857082764038923 - 0.9977728883419751 * state.chi1)
        )
    else:
        total_mass = state.mass1 + state.mass2
        chi_eff = (state.mass1 * state.chi1 + state.mass2 * state.chi2) / total_mass
        cutoff = state.f_meco_32 + (0.75 - 0.235 * abs(chi_eff)) * abs(
            state.f_isco_22 - state.f_meco_32
        )
    return cutoff * state.f_ring_32 / state.f_ring_22


def ringdown_match_32_2019(state):
    """Return the intermediate/ringdown matching frequency."""

    if state.eta < 0.0453515:
        exp_spin = math.exp(5.0 * state.chi1)
        match = (state.f_ring_32 * math.exp(2.5) + state.f_ring_22 * exp_spin) / (
            math.exp(2.5) + exp_spin
        ) - state.f_damp_32
    else:
        match = state.f_ring_22
    if 0.02126654064272212 < state.eta < 0.12244897959183673 and state.chi1 > 0.95:
        match = state.f_ring_32 - 2.0 * state.f_damp_32
    return match


def _inspiral_fit_values(state):
    eta = state.eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta
    eta6 = eta5 * eta
    eta8 = eta4 * eta4
    spin = state.chi_pn_hat
    spin2 = spin * spin
    spin3 = spin2 * spin
    delta = state.delta
    dchi = state.dchi
    common = math.sqrt(1.0 - 3.0 * eta)

    value1 = (
        common
        * (
            0.019069933430190773
            - 0.19396651989685837 * eta
            + 11.95224600241255 * eta2
            - 158.90113442757382 * eta3
            + 1046.65239329071 * eta4
            - 3476.940285294999 * eta5
            + 4707.249209858949 * eta6
        )
        + common
        * spin
        * (
            0.0046910348789512895
            + 0.40231360805609434 * eta
            - 0.0038263656140933152 * spin
            + 0.018963579407636953 * spin2
            + eta2
            * (
                -1.955352354930108
                + 2.3753413452420133 * spin
                - 0.9085620866763245 * spin3
            )
            + 0.02738043801805805 * spin3
            + eta3
            * (
                7.977057990568723
                - 7.9259853291789515 * spin
                + 0.49784942656123987 * spin2
                + 5.2255665027119145 * spin3
            )
        )
        + 0.058560321425018165 * dchi**2 * (1.0 - 19.936477485971217 * eta2) * eta2
        + 1635.4240644598524 * dchi * eta8 * delta
        + 0.2735219358839411 * dchi * eta2 * spin * delta
    )
    value2 = (
        common
        * (
            0.024621376891809633
            - 0.09692699636236377 * eta
            + 2.7200998230836158 * eta2
            - 16.160563094841066 * eta3
            + 32.930430889650836 * eta4
        )
        + common
        * spin
        * (
            0.008522695567479373
            - 1.1104639098529456 * eta2
            - 0.00362963820787208 * spin
            + 0.016978054142418417 * spin2
            + eta
            * (
                0.24280554040831698
                + 0.15878436411950506 * spin
                - 0.1470288177047577 * spin3
            )
            + 0.029465887557447824 * spin3
            + eta3
            * (
                4.649438233164449
                - 0.7550771176087877 * spin
                + 0.3381436950547799 * spin2
                + 2.5663386135613093 * spin3
            )
        )
        - 0.007061187955941243 * dchi**2 * (1.0 - 2.024701925508361 * eta2) * eta2
        + 215.06940561269835 * dchi * eta8 * delta
        + 0.1465612311350642 * dchi * eta2 * spin * delta
    )
    value3 = (
        common
        * (
            -0.006150151041614737
            + 0.017454430190035 * eta
            + 0.02620962593739105 * eta2
            - 0.019043090896351363 * eta3
        )
        / (-0.2655505633361449 + eta)
        + common
        * spin
        * (
            0.011073381681404716
            + 0.00347699923233349 * spin
            + eta * spin * (0.05592992411391443 - 0.15666140197050316 * spin2)
            + 0.012079324401547036 * spin2
            + eta2
            * (
                0.5440307361144313
                - 0.008730335213434078 * spin
                + 0.04615964369925028 * spin2
                + 0.6703688097531089 * spin3
            )
            + 0.016323101357296865 * spin3
        )
        - 0.020140175824954427 * dchi**2 * (1.0 - 12.675522774051249 * eta2) * eta2
        - 417.3604094454253 * dchi * eta8 * delta
        + 0.10464021067936538 * dchi * eta2 * spin * delta
    )
    return abs(value1), abs(value2), abs(value3)


def inspiral_configuration_32_2019(state):
    """Return original cutoff, post-veto cutoff, and live fit indices."""

    original_cutoff = inspiral_cutoff_32_2019(state)
    frequencies = (
        original_cutoff,
        0.75 * original_cutoff,
        0.5 * original_cutoff,
    )
    fit_values = _inspiral_fit_values(state)
    selected = [0, 1, 2]
    if state.q > 2.5 and state.chi1 < -0.9 and state.chi2 < -0.9:
        selected = []
        cutoff = state.f_meco_32
    elif (
        state.q > 2.5
        and state.chi1 < -0.6
        and state.chi2 > 0.0
        and fit_values[0] != 0.0
    ):
        selected = [1, 2]
        cutoff = frequencies[1]
    else:
        scaled = tuple(
            value * frequency ** (-7.0 / 6.0)
            for value, frequency in zip(fit_values, frequencies)
        )
        if (scaled[0] > scaled[1] < scaled[2]) or (scaled[0] < scaled[1] > scaled[2]):
            selected = [0, 2]
        cutoff = original_cutoff
    return original_cutoff, cutoff, frequencies, fit_values, tuple(selected)


def _pn_rescaled(frequency, state):
    from .imrphenomxhm_mode32_torch import _pn_amplitude_coefficients

    coefficients = _tensor(
        _pn_amplitude_coefficients(state), frequency, complex_value=True
    )
    frequency_power = frequency ** (1.0 / 3.0)
    series = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        series = series * frequency_power + coefficient
    return torch.abs(series) * math.sqrt(5.0 / 7.0) / 3.0


def inspiral_model_32_2019(mf, state):
    """Return cutoffs and the legacy rescaled inspiral ansatz."""

    (
        original_cutoff,
        cutoff,
        frequencies,
        fit_values,
        selected,
    ) = inspiral_configuration_32_2019(state)
    targets = [
        fit_values[index] - _pn_rescaled(_tensor(frequencies[index], mf), state)
        for index in selected
    ]
    rows = [
        [
            (frequencies[index] / cutoff) ** ((7.0 + power) / 3.0)
            for power in range(len(selected))
        ]
        for index in selected
    ]
    pseudo = _solve(rows, targets, mf) if selected else _tensor((), mf)

    def rescaled_amplitude(frequency):
        result = _pn_rescaled(frequency, state)
        ratio = frequency / cutoff
        for power, coefficient in enumerate(pseudo):
            result = result + coefficient * ratio ** ((7.0 + power) / 3.0)
        return result

    return original_cutoff, cutoff, pseudo, rescaled_amplitude


def intermediate_fit_values_32_2019(state):
    eta = state.eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta
    eta6 = eta5 * eta
    spin = state.s_tot_r
    spin2 = spin * spin
    delta = state.delta
    common = math.sqrt(eta - 3.0 * eta2)

    value1 = (
        common
        * (
            6.523612598187996
            - 56.93956111746338 * eta
            + 1021.6414686597869 * eta2
            - 12107.114370361525 * eta3
            + 76320.90587515048 * eta4
            - 244144.92645448362 * eta5
            + 321790.55131499085 * eta6
        )
        + common
        * spin
        * (
            2.9649243713119895
            + eta3 * (1790.8363334078751 - 5438.911035114849 * spin)
            + eta * (-37.87005271181108 - 126.1263286618178 * spin)
            + 4.063724538613828 * spin
            + eta2 * (48.39743086535961 + 1341.2619677741804 * spin)
            + eta4 * (-5200.659417644607 + 7369.386205324284 * spin)
        )
        + eta2
        * (
            -0.4386152975075188
            * (state.chi1**2 - 2.0 * state.chi1 * state.chi2 + state.chi2**2)
            + (
                state.chi2 * (3.6527252109313233 - 7.324266404418883 * spin)
                + state.chi1 * (-3.6527252109313233 + 7.324266404418883 * spin)
            )
            * delta
        )
    )
    value2 = (
        common
        * (
            5.941845842405418
            - 31.905244419036794 * eta
            + 271.105632998832 * eta2
            - 2113.9652334868965 * eta3
            + 6214.038393898584 * eta4
        )
        + common
        * spin
        * (
            -2.726472456645038
            + 2.9454485454761827 * spin
            + eta3
            * (
                10581.664858726683
                - 8474.190197512324 * spin
                - 11680.937129551317 * spin2
            )
            + eta
            * (
                98.08119212251981
                - 119.88112323140916 * spin
                - 145.5079981415436 * spin2
            )
            + 3.5684571473795095 * spin2
            + eta2
            * (
                -1595.8027347570667
                + 1686.7137359336039 * spin
                + 2139.8290160628144 * spin2
            )
            + eta4
            * (
                -21488.25117198268
                + 13866.428366595079 * spin
                + 20863.270079587106 * spin2
            )
        )
        + 0.0038732029045487884 * state.dchi * eta2 * delta
    )
    return abs(value1), abs(value2)


def ringdown_parameters_32_2019(state):
    eta = state.eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta
    eta6 = eta5 * eta
    spin = state.s_tot_r
    delta = state.delta
    dchi = state.dchi
    common = math.sqrt(1.0 - 3.0 * eta)

    alambda = abs(
        0.00012587900257140724
        + 0.03927886286971654 * eta
        - 0.8109309606583066 * eta2
        + 8.820604907164254 * eta3
        - 51.43344812454074 * eta4
        + 141.81940900657446 * eta5
        - 140.0426973304466 * eta6
        + spin
        * (
            -0.00006001471234796344
            + eta4 * (-0.7849112300598181 - 2.09188976953315 * spin)
            + eta2 * (0.08311497969032984 - 0.15569578955822236 * spin)
            + eta * (-0.01083175709906557 + 0.00568899459837252 * spin)
            - 0.00009363591928190229 * spin
            + 1.0670798489407887 * eta3 * spin
        )
        - 0.04537308968659669
        * dchi**2
        * eta2
        * (1.0 - 8.711096029480697 * eta + 18.362371966229926 * eta2)
        + dchi
        * (
            -297.36978685672733
            + 3103.2516759087644 * eta
            - 10001.774055779177 * eta2
            + 9386.734883473799 * eta3
        )
        * eta6
    )
    decay = (
        common
        * (0.0341611244787871 - 0.3197209728114808 * eta + 0.7689553234961991 * eta2)
        / (0.048429644168112324 - 0.43758296068790314 * eta + eta2)
        + common
        * spin
        * (
            0.11057199932233873
            + eta2 * (25.536336676250748 - 71.18182757443142 * spin)
            + 9.790509295728649 * eta * spin
            + eta3 * (-56.96407763839491 + 175.47259563543165 * spin)
        )
        - 5.002106168893265 * dchi**2 * eta2 * delta
    )
    return alambda, decay, 1.33


def reject_two_region_32_2019(state):
    if state.eta < _EMR_TWO_REGION_ETA and state.chi1 <= 0.9:
        raise ValueError(
            "the 2019 (3, 2) amplitude's q > 70 two-region branch is not "
            "Torch-native yet"
        )
