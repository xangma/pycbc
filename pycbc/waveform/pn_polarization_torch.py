# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch-native nonprecessing post-Newtonian polarizations.

This is the vectorized Torch counterpart of LALSuite's
``XLALSimInspiralPNPolarizationWaveforms``.  It implements the non-memory
polarization amplitudes through 3PN and is shared by the nonspinning TaylorT
waveform families.
"""

from __future__ import annotations

import math
import operator

import torch

from pycbc.waveform.constants import _EULER_GAMMA, _MRSUN_SI, _PC_SI


_SUPPORTED_AMPLITUDE_ORDERS = frozenset((-1, 0, 1, 2, 3, 4, 5, 6))
_MPC_SI = 1.0e6 * _PC_SI


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


def pn_polarizations(
    velocity,
    phase,
    mass1,
    mass2,
    distance,
    inclination,
    *,
    amplitude_order=-1,
    v0=1.0,
):
    """Construct nonprecessing PN plus and cross polarizations with Torch.

    Parameters are expressed in PyCBC units: component masses are in solar
    masses and distance is in Mpc.  ``velocity`` and ``phase`` must be
    same-shaped real Torch tensors.  The returned tensors remain on their
    input device and retain their dtype.
    """

    if not isinstance(velocity, torch.Tensor) or not isinstance(phase, torch.Tensor):
        raise TypeError("velocity and phase must be Torch tensors")
    if velocity.shape != phase.shape:
        raise ValueError("velocity and phase must have matching shapes")
    if velocity.ndim != 1:
        raise ValueError("velocity and phase must be one-dimensional")
    if velocity.device != phase.device or velocity.dtype != phase.dtype:
        raise ValueError("velocity and phase must share a device and dtype")
    if velocity.dtype not in (torch.float32, torch.float64):
        raise ValueError("PN polarizations require float32 or float64")

    amplitude_order = _amplitude_order(amplitude_order)
    mass1 = float(mass1)
    mass2 = float(mass2)
    distance = float(distance)
    inclination = float(inclination)
    v0 = float(v0)
    scalars = (mass1, mass2, distance, inclination, v0)
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("PN polarization parameters must be finite")
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

    total_mass = mass1 + mass2
    eta = mass1 * mass2 / total_mass**2
    eta2 = eta * eta
    eta3 = eta2 * eta
    dm = (mass1 - mass2) / total_mass
    amplitude_factor = 2.0 * total_mass * _MRSUN_SI * eta / (distance * _MPC_SI)

    ci = math.cos(inclination)
    si = math.sin(inclination)
    ci2 = ci * ci
    ci4 = ci2 * ci2
    ci6 = ci2 * ci4
    ci8 = ci6 * ci2
    si2 = si * si
    si3 = si2 * si
    si4 = si2 * si2
    si5 = si * si4
    si6 = si4 * si2

    v = velocity
    v2 = v * v
    v3 = v * v2
    if amplitude_order == -1 or amplitude_order >= 5:
        phase_shift = 3.0 * v3 * (1.0 - v2 * eta / 2.0) * torch.log(v2 / (v0 * v0))
    elif amplitude_order >= 3:
        phase_shift = 3.0 * v3 * torch.log(v2 / (v0 * v0))
    else:
        phase_shift = torch.zeros_like(v)
    phi = phase - phase_shift

    cos1 = torch.cos(phi)
    cos2 = torch.cos(2.0 * phi)
    cos3 = torch.cos(3.0 * phi)
    cos4 = torch.cos(4.0 * phi)
    cos5 = torch.cos(5.0 * phi)
    cos6 = torch.cos(6.0 * phi)
    cos7 = torch.cos(7.0 * phi)
    cos8 = torch.cos(8.0 * phi)
    sin1 = torch.sin(phi)
    sin2 = torch.sin(2.0 * phi)
    sin3 = torch.sin(3.0 * phi)
    sin4 = torch.sin(4.0 * phi)
    sin5 = torch.sin(5.0 * phi)
    sin6 = torch.sin(6.0 * phi)
    sin7 = torch.sin(7.0 * phi)
    sin8 = torch.sin(8.0 * phi)

    zero = torch.zeros_like(v)
    hp05 = hp1 = hp15 = hp2 = hp25 = hp3 = zero
    hc05 = hc1 = hc15 = hc2 = hc25 = hc3 = zero

    if amplitude_order == -1 or amplitude_order >= 6:
        hp3 = (
            math.pi
            * dm
            * si
            * cos1
            * (
                19.0 / 64.0
                + ci2 * 5.0 / 16.0
                - ci4 / 192.0
                + eta * (-19.0 / 96.0 + ci2 * 3.0 / 16.0 + ci4 / 96.0)
            )
            + cos2
            * (
                -465497.0 / 11025.0
                + (
                    _EULER_GAMMA * 856.0 / 105.0
                    - 2.0 * math.pi**2 / 3.0
                    + torch.log(16.0 * v2) * 428.0 / 105.0
                )
                * (1.0 + ci2)
                - ci2 * 3561541.0 / 88200.0
                - ci4 * 943.0 / 720.0
                + ci6 * 169.0 / 720.0
                - ci8 / 360.0
                + eta
                * (
                    2209.0 / 360.0
                    - math.pi**2 * 41.0 / 96.0 * (1.0 + ci2)
                    + ci2 * 2039.0 / 180.0
                    + ci4 * 3311.0 / 720.0
                    - ci6 * 853.0 / 720.0
                    + ci8 * 7.0 / 360.0
                )
                + eta2
                * (
                    12871.0 / 540.0
                    - ci2 * 1583.0 / 60.0
                    - ci4 * 145.0 / 108.0
                    + ci6 * 56.0 / 45.0
                    - ci8 * 7.0 / 180.0
                )
                + eta3
                * (
                    -3277.0 / 810.0
                    + ci2 * 19661.0 / 3240.0
                    - ci4 * 281.0 / 144.0
                    - ci6 * 73.0 / 720.0
                    + ci8 * 7.0 / 360.0
                )
            )
            + math.pi
            * dm
            * si
            * cos3
            * (
                -1971.0 / 128.0
                - ci2 * 135.0 / 16.0
                + ci4 * 243.0 / 128.0
                + eta * (567.0 / 64.0 - ci2 * 81.0 / 16.0 - ci4 * 243.0 / 64.0)
            )
            + si2
            * cos4
            * (
                -2189.0 / 210.0
                + ci2 * 1123.0 / 210.0
                + ci4 * 56.0 / 9.0
                - ci6 * 16.0 / 45.0
                + eta
                * (
                    6271.0 / 90.0
                    - ci2 * 1969.0 / 90.0
                    - ci4 * 1432.0 / 45.0
                    + ci6 * 112.0 / 45.0
                )
                + eta2
                * (
                    -3007.0 / 27.0
                    + ci2 * 3493.0 / 135.0
                    + ci4 * 1568.0 / 45.0
                    - ci6 * 224.0 / 45.0
                )
                + eta3
                * (
                    161.0 / 6.0
                    - ci2 * 1921.0 / 90.0
                    - ci4 * 184.0 / 45.0
                    + ci6 * 112.0 / 45.0
                )
            )
            + dm
            * cos5
            * (math.pi * 3125.0 / 384.0 * si3 * (1.0 + ci2) * (1.0 - 2.0 * eta))
            + si4
            * cos6
            * (
                1377.0 / 80.0
                + ci2 * 891.0 / 80.0
                - ci4 * 729.0 / 280.0
                + eta * (-7857.0 / 80.0 - ci2 * 891.0 / 16.0 + ci4 * 729.0 / 40.0)
                + eta2 * (567.0 / 4.0 + ci2 * 567.0 / 10.0 - ci4 * 729.0 / 20.0)
                + eta3 * (-729.0 / 16.0 - ci2 * 243.0 / 80.0 + ci4 * 729.0 / 40.0)
            )
            + cos8
            * (
                -1024.0
                / 315.0
                * si6
                * (1.0 + ci2)
                * (1.0 - 7.0 * eta + 14.0 * eta2 - 7.0 * eta3)
            )
            + dm
            * si
            * sin1
            * (
                -2159.0 / 40320.0
                - math.log(2.0) * 19.0 / 32.0
                + (-95.0 / 224.0 - math.log(2.0) * 5.0 / 8.0) * ci2
                + (181.0 / 13440.0 + math.log(2.0) / 96.0) * ci4
                + eta
                * (
                    1369.0 / 160.0
                    + math.log(2.0) * 19.0 / 48.0
                    + (-41.0 / 48.0 - math.log(2.0) * 3.0 / 8.0) * ci2
                    + (-313.0 / 480.0 - math.log(2.0) / 48.0) * ci4
                )
            )
            + sin2 * (-428.0 * math.pi / 105.0 * (1.0 + ci2))
            + dm
            * si
            * sin3
            * (
                205119.0 / 8960.0
                - math.log(3.0 / 2.0) * 1971.0 / 64.0
                + (1917.0 / 224.0 - math.log(3.0 / 2.0) * 135.0 / 8.0) * ci2
                + (-43983.0 / 8960.0 + math.log(3.0 / 2.0) * 243.0 / 64.0) * ci4
                + eta
                * (
                    -54869.0 / 960.0
                    + math.log(3.0 / 2.0) * 567.0 / 32.0
                    + (-923.0 / 80.0 - math.log(3.0 / 2.0) * 81.0 / 8.0) * ci2
                    + (41851.0 / 2880.0 - math.log(3.0 / 2.0) * 243.0 / 32.0) * ci4
                )
            )
            + dm
            * si3
            * (1.0 + ci2)
            * sin5
            * (
                -113125.0 / 5376.0
                + math.log(5.0 / 2.0) * 3125.0 / 192.0
                + eta * (17639.0 / 320.0 - math.log(5.0 / 2.0) * 3125.0 / 96.0)
            )
        )
        hc3 = (
            dm
            * si
            * ci
            * cos1
            * (
                11617.0 / 20160.0
                + math.log(2.0) * 21.0 / 16.0
                + (-251.0 / 2240.0 - math.log(2.0) * 5.0 / 48.0) * ci2
                + eta
                * (
                    -2419.0 / 240.0
                    - math.log(2.0) * 5.0 / 24.0
                    + (727.0 / 240.0 + math.log(2.0) * 5.0 / 24.0) * ci2
                )
            )
            + ci * cos2 * (math.pi * 856.0 / 105.0)
            + dm
            * si
            * ci
            * cos3
            * (
                -36801.0 / 896.0
                + math.log(3.0 / 2.0) * 1809.0 / 32.0
                + (65097.0 / 4480.0 - math.log(3.0 / 2.0) * 405.0 / 32.0) * ci2
                + eta
                * (
                    28445.0 / 288.0
                    - math.log(3.0 / 2.0) * 405.0 / 16.0
                    + (-7137.0 / 160.0 + math.log(3.0 / 2.0) * 405.0 / 16.0) * ci2
                )
            )
            + dm
            * si3
            * ci
            * cos5
            * (
                113125.0 / 2688.0
                - math.log(5.0 / 2.0) * 3125.0 / 96.0
                + eta * (-17639.0 / 160.0 + math.log(5.0 / 2.0) * 3125.0 / 48.0)
            )
            + math.pi
            * dm
            * si
            * ci
            * sin1
            * (21.0 / 32.0 - ci2 * 5.0 / 96.0 + eta * (-5.0 / 48.0 + ci2 * 5.0 / 48.0))
            + ci
            * sin2
            * (
                -3620761.0 / 44100.0
                + _EULER_GAMMA * 1712.0 / 105.0
                - 4.0 * math.pi**2 / 3.0
                + torch.log(16.0 * v2) * 856.0 / 105.0
                - ci2 * 3413.0 / 1260.0
                + ci4 * 2909.0 / 2520.0
                - ci6 / 45.0
                + eta
                * (
                    743.0 / 90.0
                    - 41.0 * math.pi**2 / 48.0
                    + ci2 * 3391.0 / 180.0
                    - ci4 * 2287.0 / 360.0
                    + ci6 * 7.0 / 45.0
                )
                + eta2
                * (
                    7919.0 / 270.0
                    - ci2 * 5426.0 / 135.0
                    + ci4 * 382.0 / 45.0
                    - ci6 * 14.0 / 45.0
                )
                + eta3
                * (
                    -6457.0 / 1620.0
                    + ci2 * 1109.0 / 180.0
                    - ci4 * 281.0 / 120.0
                    + ci6 * 7.0 / 45.0
                )
            )
            + math.pi
            * dm
            * si
            * ci
            * sin3
            * (
                -1809.0 / 64.0
                + ci2 * 405.0 / 64.0
                + eta * (405.0 / 32.0 - ci2 * 405.0 / 32.0)
            )
            + si2
            * ci
            * sin4
            * (
                -1781.0 / 105.0
                + ci2 * 1208.0 / 63.0
                - ci4 * 64.0 / 45.0
                + eta * (5207.0 / 45.0 - ci2 * 536.0 / 5.0 + ci4 * 448.0 / 45.0)
                + eta2 * (-24838.0 / 135.0 + ci2 * 2224.0 / 15.0 - ci4 * 896.0 / 45.0)
                + eta3 * (1703.0 / 45.0 - ci2 * 1976.0 / 45.0 + ci4 * 448.0 / 45.0)
            )
            + dm * sin5 * (3125.0 * math.pi / 192.0 * si3 * ci * (1.0 - 2.0 * eta))
            + si4
            * ci
            * sin6
            * (
                9153.0 / 280.0
                - ci2 * 243.0 / 35.0
                + eta * (-7371.0 / 40.0 + ci2 * 243.0 / 5.0)
                + eta2 * (1296.0 / 5.0 - ci2 * 486.0 / 5.0)
                + eta3 * (-3159.0 / 40.0 + ci2 * 243.0 / 5.0)
            )
            + sin8
            * (
                -2048.0
                / 315.0
                * si6
                * ci
                * (1.0 - 7.0 * eta + 14.0 * eta2 - 7.0 * eta3)
            )
        )

    if amplitude_order == -1 or amplitude_order >= 5:
        hp25 = (
            cos1
            * si
            * dm
            * (
                1771.0 / 5120.0
                - ci2 * 1667.0 / 5120.0
                + ci4 * 217.0 / 9216.0
                - ci6 / 9126.0
                + eta
                * (
                    681.0 / 256.0
                    + ci2 * 13.0 / 768.0
                    - ci4 * 35.0 / 768.0
                    + ci6 / 2304.0
                )
                + eta2
                * (
                    -3451.0 / 9216.0
                    + ci2 * 673.0 / 3072.0
                    - ci4 * 5.0 / 9216.0
                    - ci6 / 3072.0
                )
            )
            + cos2
            * math.pi
            * (
                19.0 / 3.0
                + 3.0 * ci2
                - ci4 * 2.0 / 3.0
                + eta * (-16.0 / 3.0 + ci2 * 14.0 / 3.0 + 2.0 * ci4)
            )
            + cos3
            * si
            * dm
            * (
                3537.0 / 1024.0
                - ci2 * 22977.0 / 5120.0
                - ci4 * 15309.0 / 5120.0
                + ci6 * 729.0 / 5120.0
                + eta
                * (
                    -23829.0 / 1280.0
                    + ci2 * 5529.0 / 1280.0
                    + ci4 * 7749.0 / 1280.0
                    - ci6 * 729.0 / 1280.0
                )
                + eta2
                * (
                    29127.0 / 5120.0
                    - ci2 * 27267.0 / 5120.0
                    - ci4 * 1647.0 / 5120.0
                    + ci6 * 2187.0 / 5120.0
                )
            )
            + cos4 * (-16.0 * math.pi / 3.0 * (1.0 + ci2) * si2 * (1.0 - 3.0 * eta))
            + cos5
            * si
            * dm
            * (
                -108125.0 / 9216.0
                + ci2 * 40625.0 / 9216.0
                + ci4 * 83125.0 / 9216.0
                - ci6 * 15625.0 / 9216.0
                + eta
                * (
                    8125.0 / 256.0
                    - ci2 * 40625.0 / 2304.0
                    - ci4 * 48125.0 / 2304.0
                    + ci6 * 15625.0 / 2304.0
                )
                + eta2
                * (
                    -119375.0 / 9216.0
                    + ci2 * 40625.0 / 3072.0
                    + ci4 * 44375.0 / 9216.0
                    - ci6 * 15625.0 / 3072.0
                )
            )
            + cos7
            * dm
            * (117649.0 / 46080.0 * si5 * (1.0 + ci2) * (1.0 - 4.0 * eta + 3.0 * eta2))
            + sin2
            * (
                -9.0 / 5.0
                + ci2 * 14.0 / 5.0
                + ci4 * 7.0 / 5.0
                + eta * (32.0 + ci2 * 56.0 / 5.0 - ci4 * 28.0 / 5.0)
            )
            + sin4
            * si2
            * (1.0 + ci2)
            * (
                56.0 / 5.0
                - 32.0 * math.log(2.0) / 3.0
                + eta * (-1193.0 / 30.0 + 32.0 * math.log(2.0))
            )
        )
        hc25 = (
            cos2
            * ci
            * (2.0 - ci2 * 22.0 / 5.0 + eta * (-282.0 / 5.0 + ci2 * 94.0 / 5.0))
            + cos4
            * ci
            * si2
            * (
                -112.0 / 5.0
                + 64.0 * math.log(2.0) / 3.0
                + eta * (1193.0 / 15.0 - 64.0 * math.log(2.0))
            )
            + sin1
            * si
            * ci
            * dm
            * (
                -913.0 / 7680.0
                + ci2 * 1891.0 / 11520.0
                - ci4 * 7.0 / 4608.0
                + eta * (1165.0 / 384.0 - ci2 * 235.0 / 576.0 + ci4 * 7.0 / 1152.0)
                + eta2 * (-1301.0 / 4608.0 + ci2 * 301.0 / 2304.0 - ci4 * 7.0 / 1536.0)
            )
            + sin2
            * math.pi
            * ci
            * (34.0 / 3.0 - ci2 * 8.0 / 3.0 + eta * (-20.0 / 3.0 + 8.0 * ci2))
            + sin3
            * si
            * ci
            * dm
            * (
                12501.0 / 2560.0
                - ci2 * 12069.0 / 1280.0
                + ci4 * 1701.0 / 2560.0
                + eta * (-19581.0 / 640.0 + ci2 * 7821.0 / 320.0 - ci4 * 1701.0 / 640.0)
                + eta2
                * (18903.0 / 2560.0 - ci2 * 11403.0 / 1280.0 + ci4 * 5103.0 / 2560.0)
            )
            + sin4 * si2 * ci * (-32.0 * math.pi / 3.0 * (1.0 - 3.0 * eta))
            + sin5
            * si
            * ci
            * dm
            * (
                -101875.0 / 4608.0
                + ci2 * 6875.0 / 256.0
                - ci4 * 21875.0 / 4608.0
                + eta
                * (66875.0 / 1152.0 - ci2 * 44375.0 / 576.0 + ci4 * 21875.0 / 1152.0)
                + eta2
                * (-100625.0 / 4608.0 + ci2 * 83125.0 / 2304.0 - ci4 * 21875.0 / 1536.0)
            )
            + sin7
            * si5
            * ci
            * dm
            * (117649.0 / 23040.0 * (1.0 - 4.0 * eta + 3.0 * eta2))
        )

    if amplitude_order == -1 or amplitude_order >= 4:
        hp2 = (
            cos1 * math.pi * si * dm * (-5.0 / 8.0 - ci2 / 8.0)
            + cos2
            * (
                11.0 / 60.0
                + ci2 * 33.0 / 10.0
                + ci4 * 29.0 / 24.0
                - ci6 / 24.0
                + eta
                * (353.0 / 36.0 - 3.0 * ci2 - ci4 * 251.0 / 72.0 + ci6 * 5.0 / 24.0)
                + eta2
                * (-49.0 / 12.0 + ci2 * 9.0 / 2.0 - ci4 * 7.0 / 24.0 - ci6 * 5.0 / 24.0)
            )
            + cos3 * math.pi * si * dm * (27.0 / 8.0 * (1.0 + ci2))
            + cos4
            * si2
            * 2.0
            / 15.0
            * (
                59.0
                + ci2 * 35.0
                - ci4 * 8.0
                - eta * 5.0 / 3.0 * (131.0 + 59.0 * ci2 + 24.0 * ci4)
                + eta2 * 5.0 * (21.0 - 3.0 * ci2 - 8.0 * ci4)
            )
            + cos6 * (-81.0 / 40.0 * si4 * (1.0 + ci2) * (1.0 - 5.0 * eta + 5.0 * eta2))
            + sin1
            * si
            * dm
            * (
                11.0 / 40.0
                + 5.0 * math.log(2.0) / 4.0
                + ci2 * (7.0 / 40.0 + math.log(2.0) / 4.0)
            )
            + sin3
            * si
            * dm
            * ((-189.0 / 40.0 + 27.0 / 4.0 * math.log(3.0 / 2.0)) * (1.0 + ci2))
        )
        hc2 = (
            cos1 * si * ci * dm * (-9.0 / 20.0 - 3.0 / 2.0 * math.log(2.0))
            + cos3 * si * ci * dm * (189.0 / 20.0 - 27.0 / 2.0 * math.log(3.0 / 2.0))
            - sin1 * si * ci * dm * 3.0 * math.pi / 4.0
            + sin2
            * ci
            * (
                17.0 / 15.0
                + ci2 * 113.0 / 30.0
                - ci4 / 4.0
                + eta * (143.0 / 9.0 - ci2 * 245.0 / 18.0 + ci4 * 5.0 / 4.0)
                + eta2 * (-14.0 / 3.0 + ci2 * 35.0 / 6.0 - ci4 * 5.0 / 4.0)
            )
            + sin3 * si * ci * dm * 27.0 * math.pi / 4.0
            + sin4
            * ci
            * si2
            * 4.0
            / 15.0
            * (
                55.0
                - 12.0 * ci2
                - eta * 5.0 / 3.0 * (119.0 - 36.0 * ci2)
                + eta2 * 5.0 * (17.0 - 12.0 * ci2)
            )
            + sin6 * ci * (-81.0 / 20.0 * si4 * (1.0 - 5.0 * eta + 5.0 * eta2))
        )

    if amplitude_order == -1 or amplitude_order >= 3:
        hp15 = (
            cos1
            * si
            * dm
            * (
                19.0 / 64.0
                + ci2 * 5.0 / 16.0
                - ci4 / 192.0
                + eta * (-49.0 / 96.0 + ci2 / 8.0 + ci4 / 96.0)
            )
            + cos2 * (-2.0 * math.pi * (1.0 + ci2))
            + cos3
            * si
            * dm
            * (
                -657.0 / 128.0
                - ci2 * 45.0 / 16.0
                + ci4 * 81.0 / 128.0
                + eta * (225.0 / 64.0 - ci2 * 9.0 / 8.0 - ci4 * 81.0 / 64.0)
            )
            + cos5 * si * dm * (625.0 / 384.0 * si2 * (1.0 + ci2) * (1.0 - 2.0 * eta))
        )
        hc15 = (
            sin1
            * si
            * ci
            * dm
            * (21.0 / 32.0 - ci2 * 5.0 / 96.0 + eta * (-23.0 / 48.0 + ci2 * 5.0 / 48.0))
            - 4.0 * math.pi * ci * sin2
            + sin3
            * si
            * ci
            * dm
            * (
                -603.0 / 64.0
                + ci2 * 135.0 / 64.0
                + eta * (171.0 / 32.0 - ci2 * 135.0 / 32.0)
            )
            + sin5 * si * ci * dm * (625.0 / 192.0 * si2 * (1.0 - 2.0 * eta))
        )

    if amplitude_order == -1 or amplitude_order >= 2:
        hp1 = cos2 * (
            19.0 / 6.0
            + 3.0 / 2.0 * ci2
            - ci4 / 3.0
            + eta * (-19.0 / 6.0 + ci2 * 11.0 / 6.0 + ci4)
        ) - cos4 * (4.0 / 3.0 * si2 * (1.0 + ci2) * (1.0 - 3.0 * eta))
        hc1 = sin2 * ci * (
            17.0 / 3.0 - ci2 * 4.0 / 3.0 + eta * (-13.0 / 3.0 + 4.0 * ci2)
        ) + sin4 * ci * si2 * (-8.0 / 3.0 * (1.0 - 3.0 * eta))

    if amplitude_order == -1 or amplitude_order >= 1:
        hp05 = (
            -si
            * dm
            * (cos1 * (5.0 / 8.0 + ci2 / 8.0) - cos3 * (9.0 / 8.0 + 9.0 * ci2 / 8.0))
        )
        hc05 = si * ci * dm * (-sin1 * 3.0 / 4.0 + sin3 * 9.0 / 4.0)

    hp0 = -(1.0 + ci2) * cos2
    hc0 = -2.0 * ci * sin2
    plus = (
        amplitude_factor
        * v2
        * (hp0 + v * (hp05 + v * (hp1 + v * (hp15 + v * (hp2 + v * (hp25 + v * hp3))))))
    )
    cross = (
        amplitude_factor
        * v2
        * (hc0 + v * (hc05 + v * (hc1 + v * (hc15 + v * (hc2 + v * (hc25 + v * hc3))))))
    )
    return plus, cross


__all__ = ["pn_polarizations"]
