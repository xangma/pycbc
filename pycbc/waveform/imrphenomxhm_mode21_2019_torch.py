# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""IMRPhenomXHM 2019-release amplitude for the :math:`(2, 1)` mode.

The 2019 model uses different fits and matching ansatzes from the current
IMRPhenomXHM release.  In particular, the intermediate amplitude is the
inverse of a polynomial and the comparable-mass inspiral uses the original
unreexpanded time-domain-PN plus stationary-phase expression.

Only scalar coefficient setup and veto decisions leave the frequency axis;
all frequency-dependent evaluation remains on the active Torch device.
"""

from __future__ import annotations

import math

import torch


_PI = math.pi
_EMR_TWO_REGION_ETA = 0.013886133703630232
_UNEXPANDED_PN_ETA = 0.0237954


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


def _inspiral_cutoff(state):
    f_meco = 0.5 * state.f_meco_22
    if state.eta < 0.023795359904818562:
        return 1.25 * (
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
    f_isco = 0.5 * state.f_isco_22
    return f_meco + (0.75 - 0.235 * chi_eff - (5.0 / 6.0) * chi_eff**2) * abs(
        f_isco - f_meco
    )


def _inspiral_fit_values(state):
    eta = state.eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta
    delta = state.delta
    dchi = state.dchi
    spin = state.chi_pn_hat
    spin2 = spin * spin

    value1 = (
        delta
        * (
            0.037868557189995156
            + 0.10740090317702103 * eta
            + 1.963812986867654 * eta2
            - 16.706455229589558 * eta3
            + 69.75910808095745 * eta4
            - 98.3062466823662 * eta5
        )
        + delta
        * spin
        * (
            -0.007963757232702219
            + 0.10627108779259965 * eta
            - 0.008044970210401218 * spin
            + eta2
            * (
                -0.4735861262934258
                - 0.5985436493302649 * spin
                - 0.08217216660522082 * spin2
            )
        )
        - 0.257787704938017 * dchi * eta2 * (1.0 + 8.75928187268504 * eta2)
        - 0.2597503605427412 * dchi * eta2 * spin
    )
    value2 = (
        delta
        * (
            0.05511628628738656
            - 0.12579599745414977 * eta
            + 2.831411618302815 * eta2
            - 14.27268643447161 * eta3
            + 28.3307320191161 * eta4
        )
        + delta
        * spin
        * (
            -0.008692738851491525
            + eta * (0.09512553997347649 + 0.116470975986383 * spin)
            - 0.009520793625590234 * spin
            + eta2
            * (
                -0.3409769288480959
                - 0.8321002363767336 * spin
                - 0.13099477081654226 * spin2
            )
            - 0.006383232900211555 * spin2
        )
        - 0.2962753588645467 * dchi * eta2 * (1.0 + 1.3993978458830476 * eta2)
        - 0.17100612756133535
        * dchi
        * eta2
        * spin
        * (1.0 + 18.974303741922743 * eta2 * delta)
    )
    value3 = (
        delta
        * (
            0.059110044024271766
            - 0.0024538774422098405 * eta
            + 0.2428578654261086 * eta2
        )
        + delta
        * spin
        * (
            -0.007044339356171243
            - 0.006952154764487417 * spin
            + eta2
            * (
                -0.016643018304732624
                - 0.12702579620537421 * spin
                + 0.004623467175906347 * spin2
            )
            - 0.007685497720848461 * spin2
        )
        - 0.3172310538516028 * dchi * (1.0 - 2.9155919835488024 * eta2) * eta2
        - 0.11975485688200693
        * dchi
        * eta2
        * spin
        * (1.0 + 17.27626751837825 * eta2 * delta)
    )
    return abs(value1), abs(value2), abs(value3)


def _intermediate_fit_values(state):
    eta = state.eta
    eta2 = eta * eta
    dchi = state.dchi
    spin = state.s_tot_r
    spin2 = spin * spin
    common = math.sqrt(max(eta - 4.0 * eta2, 0.0))
    value1 = (
        common
        * (21.256776327599113 - 25.594352690383847 * eta + 30.14761650482866 * eta2)
        + common
        * spin
        * (
            -11.262044985632757
            - 1.8167045597937677 * spin
            + eta
            * (
                -1.1798437990445079
                + 6.344825546437461 * spin
                - 4.881427482271166 * spin2
            )
        )
        - 3.6366100759176696 * dchi**2 * (1.0 - 4.0 * eta) * eta
        - 31.60048733143782 * dchi * eta2 * (1.0 + 2.1502870640831855 * eta2)
    )
    value2 = (
        common
        * (19.15445065708005 - 21.13596229438309 * eta + 29.742565944285772 * eta2)
        + common
        * spin
        * (
            -12.766814596085734
            - 2.123816950673979 * spin
            + eta * (-2.913184982025043 + 6.006571549661901 * spin)
        )
        - 25.856046423804255 * dchi * eta2 * (1.0 + 5.7871199275552 * eta2)
    )
    return abs(value1), abs(value2)


def _ringdown_parameters(state):
    eta = state.eta
    eta2 = eta * eta
    delta = state.delta
    dchi = state.dchi
    spin = state.s_tot_r
    spin2 = spin * spin
    common = math.sqrt(max(eta - 4.0 * eta2, 0.0))
    alambda = abs(
        common
        * (
            0.00734983387668636
            - 0.0012619735607202085 * eta
            + 0.01042318959002753 * eta2
        )
        + common
        * spin
        * (
            -0.004839645742570202
            - 0.0013927779195756036 * spin
            + eta2
            * (
                -0.054621206928483663
                + 0.025956604949552205 * spin
                + 0.020360826886107204 * spin2
            )
        )
        - 0.018115657394753674 * dchi * eta2 * (1.0 - 10.539795474715346 * eta2 * delta)
    )
    decay = (
        0.5566284518926176
        + 0.12651770333481904 * eta
        + 1.8084545267208734 * eta2
        + (
            0.29074922226651545
            + eta2 * (-2.101111399437034 - 3.4969956644617946 * spin)
            + eta * (0.059317243606471406 - 0.31924748117518226 * spin)
            + 0.27420263462336675 * spin
        )
        * spin
        + 1.0122975748481835 * dchi * eta2 * delta
    )
    sigma = (
        1.2922261617161441
        + 0.0019318405961363861 * eta
        + (0.04927982551108649 - 0.6703778360948937 * eta + 2.6625014134659772 * eta2)
        * spin
        + 1.2001101665670462
        * (
            state.chi1
            + state.chi1**2
            - 2.0 * state.chi1 * state.chi2
            + (-1.0 + state.chi2) * state.chi2
        )
        * eta2
        * delta
    )
    return alambda, decay, sigma


def _ringdown_rescaled(frequency, state, parameters):
    alambda, decay, sigma = parameters
    offset = frequency - state.f_ring_21
    width = state.f_damp_21 * sigma
    return (
        state.f_damp_21
        * alambda
        * sigma
        * torch.exp(-offset * decay / width)
        / (offset * offset + width * width)
        * frequency ** (-1.0 / 12.0)
    )


def _reexpanded_pn_rescaled(frequency, state):
    from .imrphenomxhm_mode21_torch import _pn_amplitude_coefficients

    coefficients = _tensor(
        _pn_amplitude_coefficients(state), frequency, complex_value=True
    )
    frequency_power = frequency ** (1.0 / 3.0)
    series = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        series = series * frequency_power + coefficient
    return torch.abs(series) * (2.0 ** (-7.0 / 6.0) * math.sqrt(2.0) / 3.0)


def _pn_rescaled(frequency, state):
    # The unreexpanded comparable-mass expression is filled in below.  Keeping
    # the dispatch separate also documents the exact q=40 release boundary.
    if state.eta >= _UNEXPANDED_PN_ETA:
        return _unexpanded_pn_rescaled(frequency, state)
    return _reexpanded_pn_rescaled(frequency, state)


def _inspiral_model(mf, state, amp_norm, original_cutoff):
    frequencies = [original_cutoff, 0.75 * original_cutoff, 0.5 * original_cutoff]
    fit_values = list(_inspiral_fit_values(state))
    frequency_tensors = [_tensor(value, mf) for value in frequencies]
    pn_values = [_pn_rescaled(value, state) for value in frequency_tensors]
    targets = [fit - pn for fit, pn in zip(fit_values, pn_values)]
    active = [True, True, True]

    if state.q < 8.0:
        threshold = 0.2 / amp_norm
        for index, (fit, frequency) in enumerate(zip(fit_values, frequencies)):
            if fit < threshold * frequency ** (7.0 / 6.0):
                active[index] = False

    if all(active):
        rescaled = [
            fit * frequency ** (-7.0 / 6.0)
            for fit, frequency in zip(fit_values, frequencies)
        ]
        if (rescaled[0] > rescaled[1] < rescaled[2]) or (
            rescaled[0] < rescaled[1] > rescaled[2]
        ):
            active[1] = False

    selected = [index for index, keep in enumerate(active) if keep]
    cutoff = frequencies[selected[0]] if selected else 0.5 * state.f_meco_22
    rows = [
        [
            (frequencies[index] / cutoff) ** ((7.0 + power) / 3.0)
            for power in range(len(selected))
        ]
        for index in selected
    ]
    pseudo = (
        _solve(rows, [targets[index] for index in selected], mf) if selected else None
    )

    def rescaled_amplitude(frequency):
        amplitude = _pn_rescaled(frequency, state)
        if pseudo is not None:
            ratio = frequency / cutoff
            for power, coefficient in enumerate(pseudo):
                amplitude = amplitude + coefficient * ratio ** ((7.0 + power) / 3.0)
        return amplitude

    return cutoff, rescaled_amplitude


def _finite_difference(function, point, like):
    step = 1.0e-9
    return (
        -function(_tensor(point + 2.0 * step, like))
        + 8.0 * function(_tensor(point + step, like))
        - 8.0 * function(_tensor(point - step, like))
        + function(_tensor(point - 2.0 * step, like))
    ) / (12.0 * step)


def _value_row(frequency, degree):
    return [frequency**power for power in range(degree + 1)]


def _derivative_row(frequency, degree):
    return [0.0] + [power * frequency ** (power - 1) for power in range(1, degree + 1)]


def _intermediate_coefficients(version, values, frequencies, derivatives, like):
    v1, v2, v3, v4 = values
    f1, f2, f3, f4 = frequencies
    d1, d4 = derivatives
    if version == 105:
        degree = 5
        rows = [_value_row(point, degree) for point in (f1, f2, f3, f4)]
        rows.extend((_derivative_row(f1, degree), _derivative_row(f4, degree)))
        rhs = (v1, v2, v3, v4, d1, d4)
    elif version in (104, 1042):
        degree = 4
        rows = (
            _value_row(f1, degree),
            _derivative_row(f1, degree),
            _value_row(f3, degree),
            _value_row(f4, degree),
            _derivative_row(f4, degree),
        )
        rhs = (v1, d1, v3, v4, d4)
    elif version == 1032:
        degree = 3
        rows = (
            _value_row(f1, degree),
            _value_row(f4, degree),
            _derivative_row(f1, degree),
            _derivative_row(f4, degree),
        )
        rhs = (v1, v4, d1, d4)
    elif version == 102:
        degree = 2
        rows = (
            _value_row(f1, degree),
            _value_row(f4, degree),
            _derivative_row(f4, degree),
        )
        rhs = (v1, v4, d4)
    else:
        degree = 1
        rows = (_value_row(f1, degree), _value_row(f4, degree))
        rhs = (v1, v4)
    return _solve(rows, rhs, like)


def _crosses_zero(coefficients, start, stop):
    degree = len(coefficients) - 1
    if degree <= 1:
        return False
    coefficients = coefficients.detach().cpu().double()
    companion = torch.zeros((degree, degree), dtype=torch.float64)
    if degree > 1:
        companion[1:, :-1] = torch.eye(degree - 1, dtype=torch.float64)
    companion[:, -1] = -coefficients[:-1] / coefficients[-1]
    roots = torch.linalg.eigvals(companion)

    if degree == 5:
        # Preserve the 2019 model's published LAL behavior. CrossZeroP5 uses
        # this GSL companion orientation and indexes its packed roots
        # asymmetrically. In particular, two checks use one root's imaginary
        # part and the following root's real part. XO4a inherits the resulting
        # occasional quintic-to-quartic fallback, so mirror those indices here
        # rather than replacing them with a mathematically corrected test.
        packed = torch.stack((roots.real, roots.imag), dim=1).reshape(-1)
        threshold = 1.0e-15
        checks = (
            (packed[1], packed[0]),
            (packed[3], packed[2]),
            (packed[4], packed[4]),
            (packed[5], packed[6]),
            (packed[7], packed[8]),
        )
        return any(
            abs(float(imaginary)) < threshold
            and start <= float(real) <= stop
            for imaginary, real in checks
        )

    real = roots.real[
        (
            torch.abs(roots.imag)
            <= 1.0e-9
            * torch.maximum(torch.ones_like(roots.real), torch.abs(roots.real))
        )
    ]
    return bool(torch.any((real >= start) & (real <= stop)).item())


def _safe_intermediate_coefficients(version, values, frequencies, derivatives, like):
    while True:
        coefficients = _intermediate_coefficients(
            version, values, frequencies, derivatives, like
        )
        if not _crosses_zero(coefficients, frequencies[0], frequencies[3]):
            return coefficients
        if version == 105:
            version = 104
        elif version in (104, 1042):
            version = 1032
        elif version == 1032:
            version = 102
        elif version == 102:
            version = 101
        else:
            return coefficients


def amplitude_21_2019(mf, state):
    """Return the full dimensionless 2019-release mode amplitude."""

    if state.eta < _EMR_TWO_REGION_ETA and state.chi1 <= 0.9:
        raise ValueError(
            "the 2019 (2, 1) amplitude's q > 70 two-region branch is not "
            "Torch-native yet"
        )

    amp_norm = math.sqrt(2.0 * state.eta / 3.0) * _PI ** (-1.0 / 6.0)
    original_cutoff = _inspiral_cutoff(state)
    cutoff, inspiral_rescaled = _inspiral_model(mf, state, amp_norm, original_cutoff)
    ringdown_match = 0.75 * state.f_ring_21
    ringdown_parameters = _ringdown_parameters(state)

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
    d4 = (7.0 / 6.0) * f4 ** (1.0 / 6.0) / rd_f4 - f4 ** (7.0 / 6.0) * drd_f4 / rd_f4**2
    fit2, fit3 = _intermediate_fit_values(state)
    v1 = 1.0 / (f1 ** (-7.0 / 6.0) * insp_f1)
    v2 = 1.0 / fit2
    v3 = 1.0 / fit3
    v4 = 1.0 / (f4 ** (-7.0 / 6.0) * rd_f4)
    version = 105

    if state.q < 8.0:
        threshold = 0.2 / amp_norm
        if 1.0 / v2 < threshold:
            v2 = 1.0
            version = 1042
            if 1.0 / v3 < threshold:
                v3 = 1.0
                version = 1032
        elif 1.0 / v3 < threshold:
            v3 = 1.0
            version = 1042

    if 1.0 / v4 < 0.01 / amp_norm:
        v2 = v3 = 1.0
        version = 1032

    if version == 105 and ((v2 > v3 < v4) or (v2 < v3 > v4)):
        v3, f3 = v2, f2
        v2 = 1.0
        version = 1042

    if (state.q > 40.0 and state.chi1 > 0.9 and v2 != 1.0 and v3 != 1.0) or (
        state.eta < 0.23 and state.chi1 > 0.7 and state.chi2 < -0.5
    ):
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


def _unexpanded_pn_rescaled(frequency, state):
    """Return the legacy, unreexpanded TD-PN plus SPA amplitude."""

    total_mass = state.mass1 + state.mass2
    m1 = state.mass1 / total_mass
    m2 = state.mass2 / total_mass
    m12 = m1 * m1
    m22 = m2 * m2
    m13 = m12 * m1
    m23 = m22 * m2
    m14 = m13 * m1
    m24 = m23 * m2
    m15 = m14 * m1
    m25 = m24 * m2
    m16 = m15 * m1
    chi1 = state.chi1
    chi2 = state.chi2
    chi_s = 0.5 * (chi1 + chi2)
    chi_a = 0.5 * (chi1 - chi2)
    delta = state.delta
    eta = state.eta
    chi12 = chi1 * chi1
    chi22 = chi2 * chi2
    sc = m12 * chi1 + m22 * chi2
    sigma_c = m2 * chi2 - m1 * chi1
    tau = 2.0 * _PI
    log_two = math.log(2.0)
    log_four = 1.3862943611198906
    euler_gamma = 0.5772156649015329

    pn_td_factor = 8.0 * eta * tau ** (2.0 / 3.0) * math.sqrt(_PI / 5.0)
    x05 = 1j * delta / 3.0 * tau ** (1.0 / 3.0)
    x1 = -0.5j * (chi_a + chi_s * delta) * tau ** (2.0 / 3.0)
    x15 = 1j * delta * (-17.0 / 28.0 + 5.0 * eta / 7.0) / 3.0 * tau
    x2 = (
        1j * (-43.0 / 21.0 * delta * sc + (-79.0 + 139.0 * eta) / 42.0 * sigma_c)
        + 1j / 3.0 * delta * (_PI + 1j * (-0.5 - 2.0 * log_two))
    ) * tau ** (4.0 / 3.0)
    x25 = (
        1j
        * delta
        * ((-43.0 - 509.0 * eta) / 126.0 + 79.0 * eta * eta / 168.0)
        / 3.0
        * tau ** (5.0 / 3.0)
    )
    x3 = (
        1j
        * delta
        * (
            (-17.0 + 6.0 * eta) / 28.0 * _PI
            + 1j
            * (
                17.0 / 56.0
                + eta * (-353.0 / 28.0 - 3.0 * log_two / 7.0)
                + 17.0 * log_two / 14.0
            )
        )
        / 3.0
        * tau**2
    )

    denominator = 3.274425e7
    xdot5 = (
        -m1
        * m2
        * (-838252800.0 * m1 * m2 - 419126400.0 * m12 - 419126400.0 * m22)
        / denominator
        * tau ** (10.0 / 3.0)
    )
    xdot6 = (
        -m1
        * m2
        * (
            1152597600.0 * m2 * m13
            + 926818200.0 * m22
            + 2494800.0 * m1 * m2 * (743.0 + 462.0 * m22)
            + 1247400.0 * m12 * (743.0 + 1848.0 * m22)
        )
        / denominator
        * tau**4
    )
    xdot65 = (
        -m1
        * m2
        * (
            -34927200.0
            * m1
            * m2
            * (-(m2 * (75.0 * chi1 + 376.0 * chi2 * m2)) + 96.0 * _PI)
            - 34927200.0
            * (-(m2 * (75.0 * chi2 + 188.0 * (chi1 + chi2) * m2)) + 48.0 * _PI)
            * m12
            - 2619540000.0 * chi1 * m13
            + 13132627200.0 * chi1 * m2 * m13
            + 6566313600.0 * chi1 * m14
            - 34927200.0 * (chi2 * (75.0 - 188.0 * m2) * m2 + 48.0 * _PI) * m22
        )
        / denominator
        * tau ** (13.0 / 3.0)
    )
    xdot7 = (
        -m1
        * m2
        * (
            207900.0
            * m2
            * (-13661.0 - 19908.0 * chi1 * chi2 + 10206.0 * chi12 + 10206.0 * chi22)
            * m13
            - 23100.0 * (34103.0 + 91854.0 * chi22) * m22
            - 1373803200.0 * m14 * m22
            + 23100.0
            * m1
            * m2
            * (
                -2.0 * (34103.0 + 45927.0 * chi12 + 45927.0 * chi22)
                + 9.0
                * (-13661.0 - 19908.0 * chi1 * chi2 + 10206.0 * chi12 + 10206.0 * chi22)
                * m22
            )
            - 2747606400.0 * m13 * m23
            - 23100.0
            * m12
            * (
                34103.0
                + 91854.0 * chi12
                - 18.0
                * (-13661.0 - 19908.0 * chi1 * chi2 + 10206.0 * chi12 + 10206.0 * chi22)
                * m22
                + 59472.0 * m24
            )
        )
        / denominator
        * tau ** (14.0 / 3.0)
    )
    xdot75 = (
        -m1
        * m2
        * (
            -4036586400.0 * chi1 * m13
            + 5821200.0 * m2 * (5861.0 * chi1 + 1701.0 * _PI) * m13
            + 17059026600.0 * chi1 * m14
            + 14721814800.0 * chi1 * m2 * m14
            - 34962127200.0 * chi1 * m2 * m15
            + 207900.0
            * (2.0 * chi2 * m2 * (-9708.0 + 41027.0 * m2) + 12477.0 * _PI)
            * m22
            - 14721814800.0 * chi2 * m13 * m22
            - 69924254400.0 * chi1 * m14 * m22
            - 34962127200.0 * (chi1 + chi2) * m13 * m23
            + 207900.0
            * m12
            * (
                3.0 * _PI * (4159.0 + 31752.0 * m22)
                + 2.0
                * m2
                * (
                    9708.0 * chi2
                    + 41027.0 * (chi1 + chi2) * m2
                    - 35406.0 * chi1 * m22
                    - 168168.0 * chi2 * m23
                )
            )
            + 415800.0
            * m1
            * m2
            * (
                9708.0 * chi1 * m2
                + 82054.0 * chi2 * m22
                + 3.0 * _PI * (4159.0 + 7938.0 * m22)
                + 35406.0 * chi2 * m23
                - 84084.0 * chi2 * m24
            )
        )
        / denominator
        * tau**5
    )
    xdot8 = (
        -m1
        * m2
        * (
            -10548014400.0 * chi1 * _PI * m13
            - 63392868000.0 * chi1 * chi2 * m2 * m14
            + 34927200.0 * chi1 * (-375.0 * chi1 + 752.0 * _PI) * m14
            + 63392868000.0 * chi12 * m15
            - 153213984000.0 * m2 * chi12 * m15
            - 76606992000.0 * chi12 * m16
            - 63392868000.0 * chi1 * (chi1 - chi2) * m13 * m22
            - 51975.0
            * (4869.0 + 2711352.0 * chi1 * chi2 + 1702428.0 * chi12 + 228508.0 * chi22)
            * m14
            * m22
            - 103950.0
            * (4869.0 + 2711352.0 * chi1 * chi2 + 228508.0 * chi12 + 228508.0 * chi22)
            * m13
            * m23
            + 906328500.0 * m15 * m23
            + 1812657000.0 * m14 * m24
            + 906328500.0 * m13 * m25
            + 1925.0
            * m2
            * m13
            * (
                56198689.0
                + 13635864.0 * chi1 * chi2
                + 27288576.0 * chi1 * _PI
                + 30746952.0 * chi12
                + 3617892.0 * chi22
                - 2045736.0 * _PI**2
            )
            - 3.0
            * m22
            * (
                16447322263.0
                - 2277918720.0 * euler_gamma
                - 23284800.0 * chi2 * m2 * (-151.0 + 376.0 * m2) * _PI
                - 2277918720.0 * log_four
                + 2321480700.0 * chi22
                + 4365900000.0 * chi22 * m22
                - 21130956000.0 * chi22 * m23
                + 25535664000.0 * chi22 * m24
                + 745113600.0 * _PI**2
            )
            + m12
            * (
                6833756160.0 * euler_gamma
                + 10548014400.0 * chi2 * m2 * _PI
                + 63392868000.0 * (chi1 - chi2) * chi2 * m23
                - 51975.0
                * (
                    4869.0
                    + 2711352.0 * chi1 * chi2
                    + 228508.0 * chi12
                    + 1702428.0 * chi22
                )
                * m24
                - 3850.0
                * m22
                * (
                    -56198689.0
                    + 13580136.0 * chi1 * chi2
                    - 6822144.0 * (chi1 + chi2) * _PI
                    - 6976422.0 * chi12
                    - 6976422.0 * chi22
                    + 2045736.0 * _PI**2
                )
                - 3.0
                * (
                    16447322263.0
                    - 2277918720.0 * log_four
                    + 2321480700.0 * chi12
                    + 745113600.0 * _PI**2
                )
            )
            + m1
            * m2
            * (
                13667512320.0 * euler_gamma
                + 10548014400.0 * chi1 * m2 * _PI
                - 63392868000.0 * chi1 * chi2 * m23
                - 153213984000.0 * chi22 * m24
                - 1925.0
                * m22
                * (
                    -56198689.0
                    - 13635864.0 * chi1 * chi2
                    - 27288576.0 * chi2 * _PI
                    - 3617892.0 * chi12
                    - 30746952.0 * chi22
                    + 2045736.0 * _PI**2
                )
                - 6.0
                * (
                    16447322263.0
                    - 2277918720.0 * log_four
                    + 1160740350.0 * chi12
                    + 1160740350.0 * chi22
                    + 745113600.0 * _PI**2
                )
            )
        )
        / denominator
        * tau ** (16.0 / 3.0)
    )
    xdot8_log = -m1 * m2 * 3416878080.0 / denominator * tau ** (16.0 / 3.0)
    xdot85 = (
        -m1
        * m2
        * (
            -14891068500.0 * chi1 * m13
            + 1925.0
            * m2
            * (97151928.0 * chi1 + 6613488.0 * chi2 - 12912300.0 * _PI)
            * m13
            + 87143248500.0 * chi1 * m14
            + 33313480200.0 * chi1 * m2 * m14
            - 198816225300.0 * chi1 * m2 * m15
            + 57750.0
            * (2.0 * chi2 * m2 * (-128927.0 + 754487.0 * m2) + 7947.0 * _PI)
            * m22
            - 33313480200.0 * chi2 * m13 * m22
            - 138600.0
            * (3399633.0 * chi1 + 530712.0 * chi2 + 182990.0 * _PI)
            * m14
            * m22
            - 35665037100.0 * chi1 * m15 * m22
            + 84184254000.0 * chi1 * m16 * m22
            - 23100.0
            * m1
            * m2
            * (
                15.0 * _PI * (-2649.0 + 71735.0 * m22)
                + m2
                * (
                    -chi1 * (644635.0 + 551124.0 * m2)
                    + chi2 * m2 * (-8095994.0 - 1442142.0 * m2 + 8606763.0 * m22)
                )
            )
            - 69300.0 * (4991769.0 * (chi1 + chi2) + 731960.0 * _PI) * m13 * m23
            + 35665037100.0 * chi2 * m14 * m23
            + 170726094000.0 * chi1 * m15 * m23
            + 2357586000.0 * chi2 * m15 * m23
            + 35665037100.0 * chi1 * m13 * m24
            + 88899426000.0 * (chi1 + chi2) * m14 * m24
            + 9702000.0 * (243.0 * chi1 + 17597.0 * chi2) * m13 * m25
            - 11550.0
            * m12
            * (
                15.0 * _PI * (-2649.0 + 286940.0 * m22 + 146392.0 * m24)
                + 2.0
                * m2
                * (
                    -644635.0 * chi2
                    - 4874683.0 * (chi1 + chi2) * m2
                    + 1442142.0 * chi1 * m22
                    + 54.0 * (58968.0 * chi1 + 377737.0 * chi2) * m23
                    + 1543941.0 * chi2 * m24
                    - 3644340.0 * chi2 * m25
                )
            )
        )
        / denominator
        * tau ** (17.0 / 3.0)
    )

    f13 = frequency ** (1.0 / 3.0)
    f23 = f13 * f13
    complex_dtype = (
        torch.complex64 if frequency.dtype == torch.float32 else torch.complex128
    )
    td_amplitude = (
        f13 * torch.as_tensor(x05, device=frequency.device, dtype=complex_dtype)
        + f23 * torch.as_tensor(x1, device=frequency.device, dtype=complex_dtype)
        + frequency * torch.as_tensor(x15, device=frequency.device, dtype=complex_dtype)
        + frequency
        * f13
        * torch.as_tensor(x2, device=frequency.device, dtype=complex_dtype)
        + frequency
        * f23
        * torch.as_tensor(x25, device=frequency.device, dtype=complex_dtype)
        + frequency**2
        * torch.as_tensor(x3, device=frequency.device, dtype=complex_dtype)
    )
    td_amplitude = td_amplitude * f23 * pn_td_factor
    f16_thirds = frequency ** (16.0 / 3.0)
    xdot = (
        frequency ** (10.0 / 3.0) * xdot5
        + frequency**4 * xdot6
        + frequency ** (13.0 / 3.0) * xdot65
        + frequency ** (14.0 / 3.0) * xdot7
        + frequency**5 * xdot75
        + f16_thirds * xdot8
        + ((2.0 / 3.0) * torch.log(frequency) + (2.0 / 3.0) * math.log(tau))
        * f16_thirds
        * xdot8_log
        + frequency ** (17.0 / 3.0) * xdot85
    )
    spa_amplitude = (
        2.0
        * math.sqrt(_PI / 3.0)
        * torch.abs(td_amplitude)
        * (2.0 ** (-1.0 / 6.0) * _PI ** (-1.0 / 6.0) * frequency ** (-1.0 / 6.0))
        / torch.sqrt(xdot)
    )
    amp_norm = math.sqrt(2.0 * eta / 3.0) * _PI ** (-1.0 / 6.0)
    return spa_amplitude / (amp_norm * frequency ** (-7.0 / 6.0))
