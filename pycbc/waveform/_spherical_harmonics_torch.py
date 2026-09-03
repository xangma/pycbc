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


_SPIN_MINUS_TWO_MODES = tuple(
    (ell, emm) for ell in range(2, 5) for emm in range(-ell, ell + 1)
)


def _spin_minus_two_terms():
    """Precompute scalar-equivalent Wigner-sum terms for ell 2 through 4."""

    terms = []
    spin_weight = -2
    wigner_m = -spin_weight
    for ell, emm in _SPIN_MINUS_TWO_MODES:
        prefactor = (-1) ** spin_weight * math.sqrt(
            (2 * ell + 1)
            / (4 * math.pi)
            * math.factorial(ell + wigner_m)
            * math.factorial(ell - wigner_m)
            * math.factorial(ell + emm)
            * math.factorial(ell - emm)
        )
        mode_terms = []
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
            mode_terms.append(
                (
                    coefficient,
                    2 * ell + wigner_m - emm - 2 * index,
                    emm - wigner_m + 2 * index,
                )
            )
        terms.append((ell, emm, tuple(mode_terms)))
    return tuple(terms)


_SPIN_MINUS_TWO_TERMS = {
    (ell, emm): terms for ell, emm, terms in _spin_minus_two_terms()
}


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
    max_power = 2 * ell + abs(wigner_m) + abs(emm) + 1
    cos_powers = [cos_half**power for power in range(max_power + 1)]
    sin_powers = [sin_half**power for power in range(max_power + 1)]
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
        amplitude = (
            amplitude
            + coefficient
            * cos_powers[cos_power]
            * sin_powers[sin_power]
        )

    phase = emm * phi
    return torch.complex(
        amplitude * torch.cos(phase), amplitude * torch.sin(phase)
    )


def _selected_spin_minus_two_values(theta, phi, modes):
    """Return selected scalar-equivalent values in caller mode order."""

    cos_half = torch.cos(0.5 * theta)
    sin_half = torch.sin(0.5 * theta)
    cos_powers = tuple(cos_half**power for power in range(9))
    sin_powers = tuple(sin_half**power for power in range(9))
    zero = torch.zeros_like(theta)
    harmonics = []

    for ell, emm in modes:
        amplitude = zero
        for coefficient, cos_power, sin_power in (
            _SPIN_MINUS_TWO_TERMS[ell, emm]
        ):
            amplitude = (
                amplitude
                + coefficient
                * cos_powers[cos_power]
                * sin_powers[sin_power]
            )
        phase = emm * phi
        harmonics.append(
            torch.complex(
                amplitude * torch.cos(phase), amplitude * torch.sin(phase)
            )
        )
    return tuple(harmonics)


def _selected_mode_lane_runtime_supported():
    """Reject transforms for which sharing the angle graph is observable."""

    try:
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return False
        if torch._C._get_tracing_state() is not None:
            return False
        if torch.autograd.forward_ad._current_level != -1:
            return False
        functorch = torch._C._functorch
        if functorch.get_dynamic_layer_stack_depth() != 0:
            return False
        if (
            torch._C._len_torch_dispatch_stack() != 0
            or torch._C._len_torch_function_stack() != 0
        ):
            return False
        compiler = getattr(
            getattr(torch, "compiler", None), "is_compiling", None
        )
        if compiler is not None and compiler():
            return False
        dynamo = getattr(
            getattr(torch, "_dynamo", None), "is_compiling", None
        )
        if dynamo is not None and dynamo():
            return False
        try:
            if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled(
                "cuda"
            ):
                return False
        except (RuntimeError, TypeError):
            if torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled():
                return False
    except Exception:
        return False
    return True


def _selected_mode_lane_angle_supported(value):
    """Accept plain scalar angles without an observable AD graph."""

    if type(value) in (int, float):
        return True
    return (
        type(value) is torch.Tensor
        and value.layout is torch.strided
        and not value.is_conj()
        and not value.is_neg()
        and not value.requires_grad
        and value.grad_fn is None
    )


def selected_spin_minus_two_spherical_harmonics(
    theta,
    phi,
    modes,
    *,
    dtype,
    device,
):
    r"""Evaluate selected :math:`{}_{-2}Y_{\ell m}(\theta, \phi)` values.

    Half-angle trigonometric values and their integer powers are shared across
    the selected modes. Inputs with observable autograd or transform semantics
    use the independent scalar evaluator.
    """

    if dtype not in (torch.float32, torch.float64):
        raise TypeError("spherical-harmonic angles require a real Torch dtype")
    try:
        normalized_modes = tuple(
            (int(ell), int(emm))
            for ell, emm in modes
            if isinstance(ell, Integral) and isinstance(emm, Integral)
        )
    except (TypeError, ValueError):
        normalized_modes = ()
    try:
        mode_count = len(modes)
    except TypeError:
        mode_count = -1
    if len(normalized_modes) != mode_count or any(
        mode not in _SPIN_MINUS_TWO_TERMS for mode in normalized_modes
    ):
        raise ValueError(
            "selected harmonics require valid ell 2 through 4 modes"
        )

    supported = (
        _selected_mode_lane_runtime_supported()
        and _selected_mode_lane_angle_supported(theta)
        and _selected_mode_lane_angle_supported(phi)
    )
    if not supported:
        return {
            (ell, emm): spin_weighted_spherical_harmonic(
                theta,
                phi,
                -2,
                ell,
                emm,
                dtype=dtype,
                device=device,
            )
            for ell, emm in normalized_modes
        }

    theta = torch.as_tensor(theta, dtype=dtype, device=device)
    phi = torch.as_tensor(phi, dtype=dtype, device=device)
    theta, phi = torch.broadcast_tensors(theta, phi)
    if theta.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        return {
            (ell, emm): spin_weighted_spherical_harmonic(
                theta,
                phi,
                -2,
                ell,
                emm,
                dtype=dtype,
                device=device,
            )
            for ell, emm in normalized_modes
        }
    return dict(
        zip(
            normalized_modes,
            _selected_spin_minus_two_values(theta, phi, normalized_modes),
        )
    )


__all__ = (
    "selected_spin_minus_two_spherical_harmonics",
    "spin_weighted_spherical_harmonic",
)
