# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Torch evaluation of spin-weighted spherical harmonics."""

import math
from numbers import Integral

import torch


def spin_weighted_spherical_harmonic(
    theta,
    phi,
    spin_weight,
    ell,
    emm,
    *,
    dtype,
    device,
):
    r"""Evaluate :math:`{}_{s}Y_{\ell m}(\theta, \phi)` with Torch.

    The finite Wigner-:math:`d` sum follows the convention used by
    ``lal.SpinWeightedSphericalHarmonic``. ``theta`` and ``phi`` may be
    scalars or broadcastable tensors; the result remains on ``device`` and
    is differentiable with respect to tensor-valued angles.
    """

    indices = (spin_weight, ell, emm)
    if any(not isinstance(index, Integral) for index in indices):
        raise TypeError("spin weight, ell, and m must be integers")
    spin_weight, ell, emm = map(int, indices)
    if ell < 0 or abs(spin_weight) > ell or abs(emm) > ell:
        raise ValueError("spin weight and m must have magnitude at most ell")
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("spherical-harmonic angles require a real Torch dtype")

    theta = torch.as_tensor(theta, dtype=dtype, device=device)
    phi = torch.as_tensor(phi, dtype=dtype, device=device)
    theta, phi = torch.broadcast_tensors(theta, phi)

    # {}_sY_lm = (-1)^s sqrt((2l+1)/(4pi)) d^l_{m,-s} exp(i m phi).
    wigner_m = -spin_weight
    prefactor = (-1) ** spin_weight * math.sqrt(
        (2 * ell + 1) / (4 * math.pi)
        * math.factorial(ell + wigner_m)
        * math.factorial(ell - wigner_m)
        * math.factorial(ell + emm)
        * math.factorial(ell - emm)
    )
    cos_half = torch.cos(0.5 * theta)
    sin_half = torch.sin(0.5 * theta)
    amplitude = torch.zeros_like(theta)

    for index in range(2 * ell + 1):
        denominator_indices = (
            ell + wigner_m - index,
            index,
            emm - wigner_m + index,
            ell - emm - index,
        )
        if min(denominator_indices) < 0:
            continue
        denominator = math.prod(
            math.factorial(value) for value in denominator_indices
        )
        coefficient = (
            (-1) ** (emm - wigner_m + index)
            * prefactor
            / denominator
        )
        cos_power = 2 * ell + wigner_m - emm - 2 * index
        sin_power = emm - wigner_m + 2 * index
        amplitude = amplitude + coefficient * (
            cos_half**cos_power
        ) * (sin_half**sin_power)

    phase = emm * phi
    return torch.complex(
        amplitude * torch.cos(phase), amplitude * torch.sin(phase)
    )


__all__ = ["spin_weighted_spherical_harmonic"]
