# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Low-order Wigner rotations for Torch waveform modes."""

from __future__ import annotations

import math

import torch


def _wigner_d_from_sincos(ell, mprime, emm, cosine, sine):
    """Evaluate d^ell_(m,mprime)(beta) from precomputed half-angle cosine and sine."""
    lower = max(0, emm - mprime)
    upper = min(ell + emm, ell - mprime)
    prefactor = math.sqrt(
        math.factorial(ell + emm)
        * math.factorial(ell - emm)
        * math.factorial(ell + mprime)
        * math.factorial(ell - mprime)
    )
    result = torch.zeros_like(cosine)
    for index in range(lower, upper + 1):
        denominator = (
            math.factorial(ell + emm - index)
            * math.factorial(index)
            * math.factorial(mprime - emm + index)
            * math.factorial(ell - mprime - index)
        )
        sign = -1.0 if (index - mprime + emm) % 2 else 1.0
        p_cos = 2 * ell + emm - mprime - 2 * index
        p_sin = mprime - emm + 2 * index
        term = sign * prefactor / denominator
        if p_cos != 0:
            term = term * (cosine ** p_cos)
        if p_sin != 0:
            term = term * (sine ** p_sin)
        result = result + term
    if (emm - mprime) % 2:
        result = -result
    return result


def wigner_d_element(ell, mprime, emm, beta):
    """Return LAL's ``d^ell_(m,mprime)(beta)`` convention."""
    beta = torch.as_tensor(beta)
    if not beta.is_floating_point():
        beta = beta.to(torch.float64)
    cosine = torch.cos(0.5 * beta)
    sine = torch.sin(0.5 * beta)
    return _wigner_d_from_sincos(ell, mprime, emm, cosine, sine)


def wigner_d_columns(ell, mprime, beta):
    """Return the ``mprime`` and ``-mprime`` Wigner-d columns.

    Both tensors are ordered by increasing target index ``m=-ell,...,ell``.
    Scalar angles produce one-dimensional columns; batched angles append their
    broadcast dimensions after the column dimension.
    """
    if ell < 0 or abs(mprime) > ell:
        raise ValueError("Wigner indices must satisfy ell >= 0 and |mprime| <= ell")
    beta = torch.as_tensor(beta)
    if not beta.is_floating_point():
        beta = beta.to(torch.float64)
    cosine = torch.cos(0.5 * beta)
    sine = torch.sin(0.5 * beta)
    positive = torch.stack(
        [_wigner_d_from_sincos(ell, mprime, emm, cosine, sine) for emm in range(-ell, ell + 1)]
    )
    negative = torch.stack(
        [_wigner_d_from_sincos(ell, -mprime, emm, cosine, sine) for emm in range(-ell, ell + 1)]
    )
    return positive, negative


def wigner_d_from_cosbeta(ell, mprime, cos_beta):
    """Return the ``mprime`` and ``-mprime`` Wigner-d columns from cos(beta).

    Both tensors are ordered by increasing target index ``m=-ell,...,ell``.
    Scalar inputs produce one-dimensional columns; batched inputs append their
    broadcast dimensions after the column dimension.
    """
    if ell < 0 or abs(mprime) > ell:
        raise ValueError("Wigner indices must satisfy ell >= 0 and |mprime| <= ell")
    cos_beta = torch.as_tensor(cos_beta)
    if not cos_beta.is_floating_point():
        cos_beta = cos_beta.to(torch.float64)
    cos_half = torch.sqrt(torch.clamp(0.5 * (1.0 + cos_beta), min=0.0))
    sin_half = torch.sqrt(torch.clamp(0.5 * (1.0 - cos_beta), min=0.0))
    positive = torch.stack(
        [_wigner_d_from_sincos(ell, mprime, emm, cos_half, sin_half) for emm in range(-ell, ell + 1)]
    )
    negative = torch.stack(
        [_wigner_d_from_sincos(ell, -mprime, emm, cos_half, sin_half) for emm in range(-ell, ell + 1)]
    )
    return positive, negative


def rotate_modes(modes, alpha, beta, gamma):
    """Rotate waveform modes by scalar or sample-dependent Euler angles.

    The returned dictionary contains every target ``m`` for each ``ell`` in
    the input. Missing source modes are treated as zero. The convention is
    ``exp(-i*m*alpha) d^ell_(m,mprime)(beta)
    exp(-i*mprime*gamma)``.
    """
    if not modes:
        return {}

    template = next(iter(modes.values()))
    real_dtype = template.real.dtype
    device = template.device
    alpha, beta, gamma = torch.broadcast_tensors(
        *(
            torch.as_tensor(angle, dtype=real_dtype, device=device)
            for angle in (alpha, beta, gamma)
        )
    )
    if alpha.ndim > 1:
        raise ValueError("waveform Euler angles must be scalar or one-dimensional")

    cosine = torch.cos(0.5 * beta)
    sine = torch.sin(0.5 * beta)

    all_ells = sorted({mode[0] for mode in modes})
    max_ell = max(all_ells)

    # Complex exponential recurrence:
    # Compute z1 = cos(alpha) + 1j * sin(alpha) once, then z_m = z_1 ** m, z_{-m} = conj(z_m)
    # The convention exp(-i * emm * alpha) corresponds to z_powers[-emm]
    z1 = (torch.cos(alpha) + 1j * torch.sin(alpha)).to(template.dtype)
    z_powers = {0: torch.ones_like(z1)}
    for emm in range(1, max_ell + 1):
        z_powers[emm] = z_powers[emm - 1] * z1
        z_powers[-emm] = torch.conj(z_powers[emm])

    exp_alpha = {
        emm: z_powers[-emm]
        for emm in range(-max_ell, max_ell + 1)
    }

    # Complex exponential recurrence for gamma:
    max_mprime = max(abs(mode[1]) for mode in modes) if modes else 0
    w1 = (torch.cos(gamma) + 1j * torch.sin(gamma)).to(template.dtype)
    w_powers = {0: torch.ones_like(w1)}
    for mprime in range(1, max_mprime + 1):
        w_powers[mprime] = w_powers[mprime - 1] * w1
        w_powers[-mprime] = torch.conj(w_powers[mprime])

    exp_gamma = {
        mprime: w_powers[-mprime]
        for mprime in {mode[1] for mode in modes}
    }

    rotated = {}
    for ell in all_ells:
        source_modes = {
            mprime: value
            for (source_ell, mprime), value in modes.items()
            if source_ell == ell
        }
        ell_template = next(iter(source_modes.values()))
        if alpha.ndim == 1 and alpha.numel() != ell_template.numel():
            raise ValueError("waveform Euler angles must use the mode time grid")

        source_gamma = {
            mprime: source * exp_gamma[mprime]
            for mprime, source in source_modes.items()
        }

        for emm in range(-ell, ell + 1):
            target = torch.zeros_like(ell_template)
            for mprime, s_gamma in source_gamma.items():
                d_elem = _wigner_d_from_sincos(ell, mprime, emm, cosine, sine)
                target = target + (d_elem.to(template.dtype) * s_gamma)
            rotated[(ell, emm)] = exp_alpha[emm] * target
    return rotated


__all__ = [
    "rotate_modes",
    "wigner_d_columns",
    "wigner_d_element",
    "wigner_d_from_cosbeta",
]
