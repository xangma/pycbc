# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Torch-native post-Newtonian modes for precessing SpinTaylor orbits.

The source-frame amplitudes reproduce LALSuite's SpinPN modes through 1.5PN.
They are rotated into the radiation frame with the orbit's instantaneous
Euler angles and returned as separate real and imaginary tensors.
"""

from __future__ import annotations

import math
import operator

import torch

from pycbc.waveform._mode_rotation_torch import rotate_modes
from pycbc.waveform.constants import _MRSUN_SI, _PC_SI
from pycbc.waveform.pn_modes_torch import pn_modes_lal_convention


_MPC_SI = 1.0e6 * _PC_SI
_SUPPORTED_AMPLITUDE_ORDERS = frozenset((-1, 0, 1, 2, 3))


def _validate_inputs(
    velocity,
    phase,
    spin1,
    spin2,
    lnhat,
    e1,
    mass1,
    mass2,
    distance,
    amplitude_order,
    ells,
):
    tensors = (velocity, phase, spin1, spin2, lnhat, e1)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("SpinTaylor mode orbit inputs must be Torch tensors")
    if velocity.ndim != 1 or phase.shape != velocity.shape:
        raise ValueError(
            "velocity and phase must be same-shaped one-dimensional tensors"
        )
    vector_shape = velocity.shape + (3,)
    if any(value.shape != vector_shape for value in tensors[2:]):
        raise ValueError("SpinTaylor mode vectors must have shape (samples, 3)")
    if any(
        value.device != velocity.device or value.dtype != velocity.dtype
        for value in tensors[1:]
    ):
        raise ValueError("SpinTaylor mode orbit inputs must share a device and dtype")
    if velocity.dtype not in (torch.float32, torch.float64):
        raise ValueError("SpinTaylor modes require float32 or float64")
    combined = torch.cat(
        (
            velocity,
            phase,
            spin1.reshape(-1),
            spin2.reshape(-1),
            lnhat.reshape(-1),
            e1.reshape(-1),
        )
    )
    if not bool(torch.isfinite(combined).all().item()):
        raise ValueError("SpinTaylor mode orbit inputs must be finite")
    if bool((velocity <= 0.0).any().item()):
        raise ValueError("velocity must be positive")

    mass1 = float(mass1)
    mass2 = float(mass2)
    distance = float(distance)
    if not all(math.isfinite(value) for value in (mass1, mass2, distance)):
        raise ValueError("SpinTaylor mode parameters must be finite")
    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("component masses must be positive")
    if distance <= 0.0:
        raise ValueError("distance must be positive")

    try:
        amplitude_order = operator.index(amplitude_order)
    except TypeError as exc:
        raise ValueError("amplitude_order must be an integer") from exc
    if amplitude_order not in _SUPPORTED_AMPLITUDE_ORDERS:
        raise ValueError(
            "unsupported amplitude_order "
            f"{amplitude_order}; expected one of -1, 0, 1, 2, 3"
        )

    try:
        ells = tuple(operator.index(ell) for ell in ells)
    except (TypeError, ValueError) as exc:
        raise ValueError("SpinTaylor mode ells must be integers") from exc
    if (
        not ells
        or len(set(ells)) != len(ells)
        or any(ell < 2 or ell > 4 for ell in ells)
    ):
        raise ValueError("SpinTaylor mode ells must be unique values from 2 through 4")
    return mass1, mass2, distance, amplitude_order, tuple(sorted(ells))


def _complex(real, imaginary):
    return torch.complex(real, imaginary)


def spintaylor_modes_from_orbit(
    velocity,
    phase,
    spin1,
    spin2,
    lnhat,
    e1,
    mass1,
    mass2,
    distance,
    *,
    amplitude_order=-1,
    ells=(2, 3, 4),
):
    """Construct radiation-frame SpinTaylor modes from an evolved orbit.

    Component masses are in solar masses, distance is in Mpc, and component
    spins use the usual dimensionless convention. Only the mode degrees
    implemented by LAL's SpinTaylor mode routine (2 through 4) are accepted.
    """

    mass1, mass2, distance, amplitude_order, ells = _validate_inputs(
        velocity,
        phase,
        spin1,
        spin2,
        lnhat,
        e1,
        mass1,
        mass2,
        distance,
        amplitude_order,
        ells,
    )
    # Unlike the shared nonspinning helper, LAL's SpinPN routine uses -1 to
    # mean its own highest available order: 1.5PN (order 3).
    order = 3 if amplitude_order == -1 else amplitude_order
    ell_max = max(ells)
    base_modes = pn_modes_lal_convention(
        velocity,
        phase,
        mass1,
        mass2,
        distance,
        ell_max=ell_max,
        amplitude_order=order,
    )
    modes = {
        mode: _complex(real, imaginary)
        for mode, (real, imaginary) in base_modes.items()
    }
    complex_zero = _complex(torch.zeros_like(velocity), torch.zeros_like(velocity))
    for ell in range(2, ell_max + 1):
        modes[(ell, 0)] = complex_zero

    total_mass = mass1 + mass2
    eta = mass1 * mass2 / total_mass**2
    dm = (mass1 - mass2) / total_mass
    scale = _MRSUN_SI / (distance * _MPC_SI)
    amplitude22 = -8.0 * eta * total_mass * scale * math.sqrt(math.pi / 5.0)
    amplitude33 = eta * total_mass * scale * math.sqrt(2.0 * math.pi / 21.0)
    amplitude44 = 2.0 * eta * total_mass * scale * math.sqrt(math.pi / 7.0)

    velocity2 = velocity * velocity
    velocity3 = velocity2 * velocity
    source_phase = phase - math.pi / 2.0
    cosine = torch.cos(source_phase)
    sine = torch.sin(source_phase)
    cosine2 = torch.cos(2.0 * source_phase)
    sine2 = torch.sin(2.0 * source_phase)
    cosine3 = torch.cos(3.0 * source_phase)
    sine3 = torch.sin(3.0 * source_phase)

    # Resolve the spins into the co-precessing source triad (E1, E2, LNhat).
    e2 = torch.linalg.cross(lnhat, e1, dim=-1)
    source_spin1 = torch.stack(
        (
            torch.sum(spin1 * e1, dim=-1),
            torch.sum(spin1 * e2, dim=-1),
            torch.sum(spin1 * lnhat, dim=-1),
        ),
        dim=-1,
    )
    source_spin2 = torch.stack(
        (
            torch.sum(spin2 * e1, dim=-1),
            torch.sum(spin2 * e2, dim=-1),
            torch.sum(spin2 * lnhat, dim=-1),
        ),
        dim=-1,
    )
    s1x, s1y, s1z = source_spin1.unbind(dim=-1)
    s2x, s2y, s2z = source_spin2.unbind(dim=-1)
    sax = 0.5 * (s1x - s2x)
    say = 0.5 * (s1y - s2y)
    saz = 0.5 * (s1z - s2z)
    ssx = 0.5 * (s1x + s2x)
    ssy = 0.5 * (s1y + s2y)
    ssz = 0.5 * (s1z + s2z)

    # SpinPN's (2, +/-2) tail omits the logarithmic imaginary term used by
    # the generic PN mode implementation, so replace these modes in full.
    common22 = torch.ones_like(velocity)
    if order >= 2:
        common22 = common22 + velocity2 * (-107.0 / 42.0 + 55.0 / 42.0 * eta)
    if order >= 3:
        common22 = common22 + velocity3 * (2.0 * math.pi)
    modes[(2, 2)] = amplitude22 * velocity2 * _complex(cosine2, -sine2) * common22
    modes[(2, -2)] = amplitude22 * velocity2 * _complex(cosine2, sine2) * common22

    if order >= 2:
        spin_x = sax + dm * ssx
        spin_y = say + dm * ssy
        spin_z = saz + dm * ssz
        real = 0.5 * velocity2 * (-cosine * spin_x + sine * spin_y)
        imaginary = 0.5 * velocity2 * (sine * spin_x + cosine * spin_y)
        modes[(2, 2)] = modes[(2, 2)] + amplitude22 * velocity2 * _complex(
            real, imaginary
        )
        modes[(2, -2)] = modes[(2, -2)] + amplitude22 * velocity2 * _complex(
            -real, imaginary
        )

        real = 0.5 * velocity2 * cosine * spin_z
        imaginary = -0.5 * velocity2 * sine * spin_z
        modes[(2, 1)] = modes[(2, 1)] + amplitude22 * velocity2 * _complex(
            real, imaginary
        )
        modes[(2, -1)] = modes[(2, -1)] + amplitude22 * velocity2 * _complex(
            real, -imaginary
        )

        imaginary = velocity2 / 3.0 * (-sine * spin_x + cosine * spin_y)
        modes[(2, 0)] = modes[(2, 0)] + amplitude22 * velocity2 * math.sqrt(
            1.5
        ) * _complex(torch.zeros_like(imaginary), imaginary)

    if order >= 3:
        spin_z = ssz + dm * saz
        real = 4.0 / 3.0 * velocity3 * cosine2 * (-spin_z + eta * ssz)
        imaginary = 4.0 / 3.0 * velocity3 * sine2 * (spin_z - eta * ssz)
        modes[(2, 2)] = modes[(2, 2)] + amplitude22 * velocity2 * _complex(
            real, imaginary
        )
        modes[(2, -2)] = modes[(2, -2)] + amplitude22 * velocity2 * _complex(
            real, -imaginary
        )

        spin_x = dm * sax + ssx
        spin_y = dm * say + ssy
        real = velocity3 * (
            sine2 * (spin_y + 5.0 / 6.0 * eta * ssy)
            + (cosine2 - 1.0) * spin_x
            + eta * (5.0 / 6.0 * cosine2 + 0.5) * ssx
        )
        imaginary = velocity3 * (
            (cosine2 + 1.0) * spin_y
            + eta * (5.0 / 6.0 * cosine2 - 0.5) * ssy
            - sine2 * spin_x
            - eta * 5.0 / 6.0 * sine2 * ssx
        )
        modes[(2, 1)] = modes[(2, 1)] + amplitude22 * velocity2 * _complex(
            real, imaginary
        )
        modes[(2, -1)] = modes[(2, -1)] + amplitude22 * velocity2 * _complex(
            -real, imaginary
        )

        if ell_max >= 3:
            spin_x = s1x + s2x
            spin_y = s1y + s2y
            spin_z = s1z + s2z
            real = 16.0 * eta * velocity3 * (cosine2 * spin_x - sine2 * spin_y)
            imaginary = 16.0 * eta * velocity3 * (-sine2 * spin_x - cosine2 * spin_y)
            modes[(3, 3)] = modes[(3, 3)] + amplitude33 * velocity2 * _complex(
                real, imaginary
            )
            modes[(3, -3)] = modes[(3, -3)] + amplitude33 * velocity2 * _complex(
                real, -imaginary
            )

            real = -16.0 / 3.0 * eta * velocity3 * cosine2 * spin_z
            imaginary = 16.0 / 3.0 * eta * velocity3 * sine2 * spin_z
            factor = amplitude33 * math.sqrt(6.0) * velocity2
            modes[(3, 2)] = modes[(3, 2)] + factor * _complex(real, imaginary)
            modes[(3, -2)] = modes[(3, -2)] + factor * _complex(-real, imaginary)

            real = 16.0 / 3.0 * eta * velocity3 * (-cosine2 * spin_x - sine2 * spin_y)
            imaginary = (
                16.0 / 3.0 * eta * velocity3 * (sine2 * spin_x - cosine2 * spin_y)
            )
            factor = amplitude33 * math.sqrt(0.6) * velocity2
            modes[(3, 1)] = modes[(3, 1)] + factor * _complex(real, imaginary)
            modes[(3, -1)] = modes[(3, -1)] + factor * _complex(real, -imaginary)

        if ell_max >= 4:
            real = 0.9 * (1.0 - 2.0 * eta) * dm * velocity3 * cosine3
            imaginary = -0.9 * (1.0 - 2.0 * eta) * dm * velocity3 * sine3
            factor = amplitude44 * math.sqrt(2.0) * velocity2
            modes[(4, 3)] = factor * _complex(real, imaginary)
            modes[(4, -3)] = factor * _complex(real, -imaginary)

    source_modes = {mode: value for mode, value in modes.items() if mode[0] in ells}
    alpha = torch.atan2(lnhat[:, 1], lnhat[:, 0])
    beta = torch.acos(torch.clamp(lnhat[:, 2], -1.0, 1.0))
    cosine_alpha = torch.cos(alpha)
    sine_alpha = torch.sin(alpha)
    cosine_beta = torch.cos(beta)
    sine_beta = torch.sin(beta)
    e_alpha = torch.stack((-sine_alpha, cosine_alpha, torch.zeros_like(alpha)), dim=-1)
    e_beta = torch.stack(
        (
            cosine_alpha * cosine_beta,
            sine_alpha * cosine_beta,
            -sine_beta,
        ),
        dim=-1,
    )
    gamma = torch.atan2(
        torch.sum(e1 * e_alpha, dim=-1),
        torch.sum(e1 * e_beta, dim=-1),
    )
    rotated = rotate_modes(source_modes, alpha, beta, gamma)
    return {mode: (value.real, value.imag) for mode, value in rotated.items()}


__all__ = ["spintaylor_modes_from_orbit"]
