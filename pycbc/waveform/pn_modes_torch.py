# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch-native nonprecessing post-Newtonian waveform modes.

This is the vectorized Torch counterpart of LALSuite's
``XLALCreateSimInspiralPNModeCOMPLEX16TimeSeriesLALConvention``.  It provides
all modes with ``2 <= l <= 6`` through the PN amplitude orders implemented by
LAL and is shared by the nonspinning TaylorT waveform families.

Modes are represented as ``(real, imaginary)`` tensor pairs.  Keeping the
components separate avoids requiring complex-tensor support on every Torch
device used by PyCBC.
"""

from __future__ import annotations

import math
import operator

import torch

from pycbc.waveform.constants import _EULER_GAMMA, _MRSUN_SI, _PC_SI


_MPC_SI = 1.0e6 * _PC_SI
_SUPPORTED_AMPLITUDE_ORDERS = frozenset((-1, 0, 1, 2, 3, 4, 5, 6))


def _amplitude_order(value):
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError("amplitude_order must be an integer") from exc
    if value not in _SUPPORTED_AMPLITUDE_ORDERS:
        choices = ", ".join(str(item) for item in sorted(_SUPPORTED_AMPLITUDE_ORDERS))
        raise ValueError(
            f"unsupported amplitude_order {value}; expected one of {choices}"
        )
    return value


def _validate_inputs(velocity, phase, mass1, mass2, distance, v0):
    if not isinstance(velocity, torch.Tensor) or not isinstance(phase, torch.Tensor):
        raise TypeError("velocity and phase must be Torch tensors")
    if velocity.shape != phase.shape:
        raise ValueError("velocity and phase must have matching shapes")
    if velocity.ndim != 1:
        raise ValueError("velocity and phase must be one-dimensional")
    if velocity.device != phase.device or velocity.dtype != phase.dtype:
        raise ValueError("velocity and phase must share a device and dtype")
    if velocity.dtype not in (torch.float32, torch.float64):
        raise ValueError("PN modes require float32 or float64")

    mass1 = float(mass1)
    mass2 = float(mass2)
    distance = float(distance)
    v0 = float(v0)
    if not all(math.isfinite(value) for value in (mass1, mass2, distance, v0)):
        raise ValueError("PN mode parameters must be finite")
    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("component masses must be positive")
    if distance <= 0.0:
        raise ValueError("distance must be positive")
    if v0 <= 0.0:
        raise ValueError("v0 must be positive")
    if bool((velocity <= 0.0).any().item()) or not bool(
        torch.isfinite(velocity).all().item()
    ):
        raise ValueError("velocity must be finite and positive")
    if not bool(torch.isfinite(phase).all().item()):
        raise ValueError("phase must be finite")
    return mass1, mass2, distance, v0


def _enabled(order, minimum):
    return order == -1 or order >= minimum


def _rotate(re, im, phase, m):
    """Multiply ``re + i im`` by ``exp(-i m phase)`` as real tensors."""

    angle = m * phase
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    return cosine * re + sine * im, cosine * im - sine * re


def _assemble(re, im, phase, m, scale, *, multiply_i=False):
    real, imaginary = _rotate(re, im, phase, m)
    if multiply_i:
        real, imaginary = -imaginary, real
    return scale * real, scale * imaginary


def _positive_mode(velocity, phase, mass1, mass2, distance, order, ell, emm, v0):
    """Build the positive-``m`` Kidder mode before LAL sign conversion."""

    total_mass = mass1 + mass2
    eta = mass1 * mass2 / total_mass**2
    eta2 = eta * eta
    eta3 = eta2 * eta
    distance_scale = _MRSUN_SI / (distance * _MPC_SI)
    total_scale = total_mass * distance_scale
    difference_scale = (mass1 - mass2) * distance_scale

    v = velocity
    v2 = v * v
    v3 = v * v2
    v4 = v2 * v2
    v5 = v * v4
    v6 = v3 * v3
    v7 = v * v6
    zeros = torch.zeros_like(v)
    ones = torch.ones_like(v)
    logv = torch.log(v / v0)

    if (ell, emm) == (2, 2):
        re = ones.clone()
        im = zeros.clone()
        if _enabled(order, 2):
            re = re + v2 * (-(107.0 / 42.0) + 55.0 / 42.0 * eta)
        if _enabled(order, 3):
            re = re + v3 * (2.0 * math.pi)
            im = im + v3 * (12.0 * logv)
        if _enabled(order, 4):
            re = re + v4 * (
                -(2173.0 / 1512.0) - 1069.0 / 216.0 * eta + 2047.0 / 1512.0 * eta2
            )
        if _enabled(order, 5):
            re = re + v5 * (-(107.0 / 21.0 - 34.0 / 21.0 * eta) * math.pi)
            im = im + v5 * (-24.0 * eta - (214.0 / 7.0 - 68.0 / 7.0 * eta) * logv)
        if _enabled(order, 6):
            re6 = (
                27027409.0 / 646800.0
                - 856.0 / 105.0 * _EULER_GAMMA
                + 2.0 / 3.0 * math.pi**2
                - 1712.0 / 105.0 * math.log(2.0)
                - (278185.0 / 33264.0 - 41.0 / 96.0 * math.pi**2) * eta
                - 20261.0 / 2772.0 * eta2
                + 114635.0 / 99792.0 * eta3
            )
            re = re + v6 * (re6 - 856.0 / 105.0 * torch.log(v) - 72.0 * logv * logv)
            im = im + v6 * (428.0 / 105.0 * math.pi + 24.0 * math.pi * logv)
        scale = -8.0 * math.sqrt(math.pi / 5.0) * eta * total_scale * v2
        return _assemble(re, im, phase, emm, scale)

    if (ell, emm) == (2, 1):
        re = zeros.clone()
        im = zeros.clone()
        if _enabled(order, 1):
            re = re + 1.0
        if _enabled(order, 3):
            re = re + v2 * (-(17.0 / 28.0) + 5.0 / 7.0 * eta)
        if _enabled(order, 4):
            re = re + v3 * math.pi
            im = im + v3 * (-0.5 - 2.0 * math.log(2.0) + 6.0 * logv)
        if _enabled(order, 5):
            re = re + v4 * (-43.0 / 126.0 - 509.0 / 126.0 * eta + 79.0 / 168.0 * eta2)
        scale = -(8.0 / 3.0) * math.sqrt(math.pi / 5.0) * eta * difference_scale * v3
        return _assemble(re, im, phase, emm, scale, multiply_i=True)

    if (ell, emm) == (2, 0):
        scale = 2.0 / 7.0 * math.sqrt(10.0 * math.pi / 3.0)
        return scale * eta * total_scale * v2, zeros

    if (ell, emm) == (3, 3):
        re = zeros.clone()
        im = zeros.clone()
        if _enabled(order, 1):
            re = re + 1.0
        if _enabled(order, 3):
            re = re + v2 * (-4.0 + 2.0 * eta)
        if _enabled(order, 4):
            re = re + v3 * (3.0 * math.pi)
            im = im + v3 * (-21.0 / 5.0 + 6.0 * math.log(3.0 / 2.0) + 18.0 * logv)
        if _enabled(order, 5):
            re = re + v4 * (123.0 / 110.0 - 1838.0 / 165.0 * eta - 887.0 / 330.0 * eta2)
        scale = 3.0 * math.sqrt(6.0 * math.pi / 7.0) * eta * difference_scale * v3
        return _assemble(re, im, phase, emm, scale, multiply_i=True)

    if (ell, emm) == (3, 2):
        eta_term = 1.0 - 3.0 * eta
        re = zeros.clone()
        im = zeros.clone()
        if _enabled(order, 2):
            re = re + eta_term
        if _enabled(order, 4):
            re = re + v2 * (-193.0 / 90.0 + 145.0 / 18.0 * eta - 73.0 / 18.0 * eta2)
        if _enabled(order, 5):
            re = re + v3 * (2.0 * math.pi * eta_term)
            im = im + v3 * (-3.0 + 66.0 / 5.0 * eta + 12.0 * eta_term * logv)
        scale = -(8.0 / 3.0) * math.sqrt(math.pi / 7.0) * eta * total_scale * v4
        return _assemble(re, im, phase, emm, scale)

    if (ell, emm) == (3, 1):
        re = zeros.clone()
        im = zeros.clone()
        if _enabled(order, 1):
            re = re + 1.0
        if _enabled(order, 3):
            re = re + v2 * (-(8.0 / 3.0) - 2.0 / 3.0 * eta)
        if _enabled(order, 4):
            re = re + v3 * math.pi
            im = im + v3 * (-7.0 / 5.0 - 2.0 * math.log(2.0) + 6.0 * logv)
        if _enabled(order, 5):
            re = re + v4 * (607.0 / 198.0 - 136.0 / 99.0 * eta - 247.0 / 198.0 * eta2)
        scale = -(1.0 / 3.0) * math.sqrt(2.0 * math.pi / 35.0)
        scale = scale * eta * difference_scale * v3
        return _assemble(re, im, phase, emm, scale, multiply_i=True)

    if (ell, emm) == (3, 0):
        re = ones if _enabled(order, 5) else zeros
        scale = 16.0 / 5.0 * math.sqrt(6.0 * math.pi / 35.0)
        scale = scale * eta2 * total_scale * v7
        return _assemble(re, zeros, phase, emm, scale, multiply_i=True)

    if (ell, emm) in ((4, 4), (4, 2)):
        eta_term = 1.0 - 3.0 * eta
        re = zeros.clone()
        im = zeros.clone()
        if _enabled(order, 2):
            re = re + eta_term
        if emm == 4:
            re4 = 593.0 / 110.0 - 1273.0 / 66.0 * eta + 175.0 / 22.0 * eta2
            re6 = (
                1068671.0 / 200200.0
                - 1088119.0 / 28600.0 * eta
                + 146879.0 / 2340.0 * eta2
                - 226097.0 / 17160.0 * eta3
            )
            im5 = -42.0 / 5.0 + 1193.0 / 40.0 * eta
            factor = 64.0 / 9.0 * math.sqrt(math.pi / 7.0)
        else:
            re4 = 437.0 / 110.0 - 805.0 / 66.0 * eta + 19.0 / 22.0 * eta2
            re6 = (
                1038039.0 / 200200.0
                - 606751.0 / 28600.0 * eta
                + 400453.0 / 25740.0 * eta2
                + 25783.0 / 17160.0 * eta3
            )
            im5 = -21.0 / 5.0 + 84.0 / 5.0 * eta
            factor = -(8.0 / 63.0) * math.sqrt(math.pi)
        if _enabled(order, 4):
            re = re + v2 * re4
        if _enabled(order, 5):
            re = re + v3 * (emm * math.pi * eta_term)
            im = im + v3 * (
                im5 + 8.0 * eta_term * math.log(2.0) + 6.0 * emm * eta_term * logv
            )
        if _enabled(order, 6):
            re = re + v4 * re6
        scale = factor * eta * total_scale * v4
        return _assemble(re, im, phase, emm, scale)

    if (ell, emm) in ((4, 3), (4, 1)):
        re = zeros.clone()
        if _enabled(order, 3):
            re = re + (1.0 - 2.0 * eta)
        if _enabled(order, 5):
            if emm == 3:
                re5 = 39.0 / 11.0 - 1267.0 / 132.0 * eta + 131.0 / 33.0 * eta2
                factor = 9.0 / 5.0 * math.sqrt(2.0 * math.pi / 7.0)
            else:
                re5 = -(101.0 / 33.0) + 337.0 / 44.0 * eta - 83.0 / 33.0 * eta2
                factor = -(1.0 / 105.0) * math.sqrt(2.0 * math.pi)
            re = re + v2 * re5
        elif emm == 3:
            factor = 9.0 / 5.0 * math.sqrt(2.0 * math.pi / 7.0)
        else:
            factor = -(1.0 / 105.0) * math.sqrt(2.0 * math.pi)
        scale = factor * eta * difference_scale * v5
        return _assemble(
            re,
            zeros,
            phase,
            emm,
            scale,
            multiply_i=emm == 1,
        )

    if (ell, emm) == (4, 0):
        scale = 1.0 / 63.0 * math.sqrt(math.pi / 10.0)
        return scale * eta * total_scale * v2, zeros

    if (ell, emm) in ((5, 5), (5, 3), (5, 1)):
        re = zeros.clone()
        if _enabled(order, 3):
            re = re + (1.0 - 2.0 * eta)
        if emm == 5:
            re5 = -263.0 / 39.0 + 688.0 / 39.0 * eta - 256.0 / 39.0 * eta2
            factor = -(125.0 / 12.0) * math.sqrt(5.0 * math.pi / 66.0)
        elif emm == 3:
            re5 = -69.0 / 13.0 + 464.0 / 39.0 * eta - 88.0 / 39.0 * eta2
            factor = -(9.0 / 20.0) * math.sqrt(2.0 * math.pi / 22.0)
        else:
            re5 = -179.0 / 39.0 + 352.0 / 39.0 * eta - 4.0 / 39.0 * eta2
            factor = -(1.0 / 180.0) * math.sqrt(math.pi / 77.0)
        if _enabled(order, 5):
            re = re + v2 * re5
        scale = factor * eta * difference_scale * v5
        return _assemble(re, zeros, phase, emm, scale, multiply_i=True)

    if (ell, emm) in ((5, 4), (5, 2)):
        re = zeros.clone()
        if _enabled(order, 4):
            re = re + (1.0 - 5.0 * eta + 5.0 * eta2)
        if emm == 4:
            re6 = (
                -4451.0 / 910.0
                + 3619.0 / 130.0 * eta
                - 521.0 / 13.0 * eta2
                + 339.0 / 26.0 * eta3
            )
            factor = 256.0 / 45.0 * math.sqrt(math.pi / 33.0)
        else:
            re6 = (
                -3911.0 / 910.0
                + 3079.0 / 130.0 * eta
                - 413.0 / 13.0 * eta2
                + 231.0 / 26.0 * eta3
            )
            factor = -(16.0 / 135.0) * math.sqrt(math.pi / 11.0)
        if _enabled(order, 6):
            re = re + v2 * re6
        scale = factor * eta * total_scale * v6
        return _assemble(re, zeros, phase, emm, scale)

    if (ell, emm) == (5, 0):
        return zeros, zeros

    if (ell, emm) in ((6, 6), (6, 4), (6, 2)):
        re = zeros.clone()
        if _enabled(order, 4):
            re = re + (1.0 - 5.0 * eta + 5.0 * eta2)
        if emm in (6, 4):
            re6 = -113.0 / 14.0 + 91.0 / 2.0 * eta - 64.0 * eta2 + 39.0 / 2.0 * eta3
        else:
            re6 = -81.0 / 14.0 + 59.0 / 2.0 * eta - 32.0 * eta2 + 7.0 / 2.0 * eta3
        if _enabled(order, 6):
            re = re + v2 * re6
        factors = {
            6: -(432.0 / 5.0) * math.sqrt(math.pi / 715.0),
            4: 1024.0 / 495.0 * math.sqrt(2.0 * math.pi / 195.0),
            2: -(16.0 / 1485.0) * math.sqrt(math.pi / 13.0),
        }
        scale = factors[emm] * eta * total_scale * v6
        return _assemble(re, zeros, phase, emm, scale)

    if (ell, emm) in ((6, 5), (6, 3), (6, 1)):
        re = zeros.clone()
        if _enabled(order, 5):
            re = re + (1.0 - 4.0 * eta + 3.0 * eta2)
        factors = {
            5: -(625.0 / 63.0) * math.sqrt(5.0 * math.pi / 429.0),
            3: 81.0 / 385.0 * math.sqrt(math.pi / 13.0),
            1: -(1.0 / 2079.0) * math.sqrt(2.0 * math.pi / 65.0),
        }
        scale = factors[emm] * eta * difference_scale * v7
        return _assemble(re, zeros, phase, emm, scale, multiply_i=True)

    if (ell, emm) == (6, 0):
        # This preserves the historical LAL implementation, including its
        # unit-amplitude (6, 0) placeholder.
        return ones, zeros

    raise ValueError(f"unsupported PN mode ({ell}, {emm})")


def _apply_lal_convention(real, imaginary, ell, emm):
    if emm < 0:
        sign = -1.0 if ell % 2 == 0 else 1.0
        return sign * real, -sign * imaginary
    return -real, -imaginary


def _validated_mode(
    velocity,
    phase,
    mass1,
    mass2,
    distance,
    amplitude_order,
    ell,
    emm,
    v0,
):
    real, imaginary = _positive_mode(
        velocity,
        phase,
        mass1,
        mass2,
        distance,
        amplitude_order,
        ell,
        abs(emm),
        v0,
    )
    return _apply_lal_convention(real, imaginary, ell, emm)


def pn_mode_lal_convention(
    velocity,
    phase,
    mass1,
    mass2,
    distance,
    ell,
    emm,
    *,
    amplitude_order=-1,
    v0=1.0,
):
    """Return one PN mode using LAL's mode sign convention."""

    mass1, mass2, distance, v0 = _validate_inputs(
        velocity, phase, mass1, mass2, distance, v0
    )
    amplitude_order = _amplitude_order(amplitude_order)
    try:
        ell = operator.index(ell)
        emm = operator.index(emm)
    except TypeError as exc:
        raise ValueError("ell and emm must be integers") from exc
    if ell < 2 or ell > 6 or abs(emm) > ell:
        raise ValueError(f"unsupported PN mode ({ell}, {emm})")

    return _validated_mode(
        velocity,
        phase,
        mass1,
        mass2,
        distance,
        amplitude_order,
        ell,
        emm,
        v0,
    )


def pn_modes_lal_convention(
    velocity,
    phase,
    mass1,
    mass2,
    distance,
    *,
    ell_max=5,
    amplitude_order=-1,
    v0=1.0,
):
    """Return every LAL-convention PN mode through ``ell_max``."""

    try:
        ell_max = operator.index(ell_max)
    except TypeError as exc:
        raise ValueError("ell_max must be an integer") from exc
    if ell_max < 2 or ell_max > 6:
        raise ValueError("ell_max must be between 2 and 6")

    mass1, mass2, distance, v0 = _validate_inputs(
        velocity, phase, mass1, mass2, distance, v0
    )
    amplitude_order = _amplitude_order(amplitude_order)

    return {
        (ell, emm): _validated_mode(
            velocity,
            phase,
            mass1,
            mass2,
            distance,
            amplitude_order,
            ell,
            emm,
            v0,
        )
        for ell in range(2, ell_max + 1)
        for emm in range(-ell, ell + 1)
    }


__all__ = ["pn_mode_lal_convention", "pn_modes_lal_convention"]
