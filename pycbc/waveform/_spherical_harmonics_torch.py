# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Torch evaluation of spin-weighted spherical harmonics."""

import math
import os
import threading
from numbers import Integral

import torch


_SPIN_MINUS_TWO_PHI_ZERO_MODES = tuple(
    (ell, emm) for ell in range(2, 5) for emm in range(-ell, ell + 1)
)


def _spin_minus_two_phi_zero_terms():
    """Precompute scalar-equivalent Wigner-sum terms for ell 2 through 4."""

    terms = []
    spin_weight = -2
    wigner_m = -spin_weight
    for ell, emm in _SPIN_MINUS_TWO_PHI_ZERO_MODES:
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


_SPIN_MINUS_TWO_PHI_ZERO_TERMS = _spin_minus_two_phi_zero_terms()
_SPIN_MINUS_TWO_TERMS_BY_MODE = {
    (ell, emm): terms
    for ell, emm, terms in _SPIN_MINUS_TWO_PHI_ZERO_TERMS
}


def _spin_minus_two_phi_zero_group_specs():
    """Group modes that have the same finite-sum width."""

    grouped = {}
    for mode_index, (_, emm, terms) in enumerate(
        _SPIN_MINUS_TWO_PHI_ZERO_TERMS
    ):
        grouped.setdefault(len(terms), []).append(
            (mode_index, emm, terms)
        )
    return tuple(
        (term_count, tuple(grouped[term_count]))
        for term_count in sorted(grouped)
    )


_SPIN_MINUS_TWO_PHI_ZERO_GROUP_SPECS = (
    _spin_minus_two_phi_zero_group_specs()
)
_VECTORIZED_SPIN_MINUS_TWO_PHI_ZERO_TABLES = {}
_VECTORIZED_SPIN_MINUS_TWO_PHI_ZERO_TABLES_LOCK = threading.Lock()
_VECTORIZED_SPIN_MINUS_TWO_PHI_ZERO_TABLES_MAX_ENTRIES = 16
_SCRIPTED_SPIN_MINUS_TWO_PHI_ZERO = {}
_SCRIPTED_SPIN_MINUS_TWO_PHI_ZERO_LOCK = threading.Lock()
_CUDAGRAPH_SPIN_MINUS_TWO_PHI_ZERO = {}
_CUDAGRAPH_SPIN_MINUS_TWO_PHI_ZERO_FAILURES = set()
_CUDAGRAPH_SPIN_MINUS_TWO_PHI_ZERO_LOCK = threading.Lock()
_CUDAGRAPH_SPIN_MINUS_TWO_PHI_ZERO_MAX_ENTRIES = 16


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
    cos_powers = [cos_half**p for p in range(max_power + 1)]
    sin_powers = [sin_half**p for p in range(max_power + 1)]
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
            cos_powers[cos_power]
        ) * (sin_powers[sin_power])

    phase = emm * phi
    return torch.complex(
        amplitude * torch.cos(phase), amplitude * torch.sin(phase)
    )


def _spin_minus_two_phi_zero_values(theta, phi):
    """Return scalar-equivalent values in canonical mode order."""

    cos_half = torch.cos(0.5 * theta)
    sin_half = torch.sin(0.5 * theta)
    cos_powers = tuple(cos_half**power for power in range(9))
    sin_powers = tuple(sin_half**power for power in range(9))
    zero = torch.zeros_like(theta)
    harmonics = []

    for ell, emm, terms in _SPIN_MINUS_TWO_PHI_ZERO_TERMS:
        amplitude = zero
        for coefficient, cos_power, sin_power in terms:
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


def _selected_spin_minus_two_values(theta, phi, modes):
    """Return selected scalar-equivalent values in caller mode order."""

    if isinstance(theta, (int, float)) and isinstance(phi, (int, float)):
        theta_val = float(theta)
        phi_val = float(phi)
        cos_half = math.cos(0.5 * theta_val)
        sin_half = math.sin(0.5 * theta_val)
        cos_powers = [cos_half**power for power in range(9)]
        sin_powers = [sin_half**power for power in range(9)]
        complex_dtype = torch.complex128
        device = torch.device("cpu")
        harmonics = []
        for ell, emm in modes:
            amplitude = sum(
                coefficient * cos_powers[cos_power] * sin_powers[sin_power]
                for coefficient, cos_power, sin_power in _SPIN_MINUS_TWO_TERMS_BY_MODE[ell, emm]
            )
            phase = emm * phi_val
            harmonics.append(
                torch.tensor(
                    complex(amplitude * math.cos(phase), amplitude * math.sin(phase)),
                    dtype=complex_dtype,
                    device=device,
                )
            )
        return tuple(harmonics)

    cos_half = torch.cos(0.5 * theta)
    sin_half = torch.sin(0.5 * theta)
    cos_powers = tuple(cos_half**power for power in range(9))
    sin_powers = tuple(sin_half**power for power in range(9))
    zero = torch.zeros_like(theta)
    harmonics = []

    for ell, emm in modes:
        amplitude = zero
        for coefficient, cos_power, sin_power in (
            _SPIN_MINUS_TWO_TERMS_BY_MODE[ell, emm]
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
    the selected modes.  Every individual mode retains the scalar evaluator's
    finite-sum and phase arithmetic order.  Inputs with observable autograd or
    transform semantics use the independent scalar evaluator.
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
        mode not in _SPIN_MINUS_TWO_TERMS_BY_MODE
        for mode in normalized_modes
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


def _vectorized_spin_minus_two_phi_zero_tables(dtype, device):
    """Return cached mode-lane tables for one dtype and device."""

    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = (dtype, device)
    cached = _VECTORIZED_SPIN_MINUS_TWO_PHI_ZERO_TABLES.get(key)
    if cached is not None:
        return cached

    with _VECTORIZED_SPIN_MINUS_TWO_PHI_ZERO_TABLES_LOCK:
        cached = _VECTORIZED_SPIN_MINUS_TWO_PHI_ZERO_TABLES.get(key)
        if cached is None:
            packed = []
            for _, modes in _SPIN_MINUS_TWO_PHI_ZERO_GROUP_SPECS:
                packed.append(
                    (
                        tuple(mode_index for mode_index, _, _ in modes),
                        torch.tensor(
                            [
                                [term[0] for term in terms]
                                for _, _, terms in modes
                            ],
                            dtype=dtype,
                            device=device,
                        ),
                        torch.tensor(
                            [
                                [term[1] for term in terms]
                                for _, _, terms in modes
                            ],
                            dtype=torch.long,
                            device=device,
                        ),
                        torch.tensor(
                            [
                                [term[2] for term in terms]
                                for _, _, terms in modes
                            ],
                            dtype=torch.long,
                            device=device,
                        ),
                        torch.tensor(
                            [emm for _, emm, _ in modes],
                            dtype=dtype,
                            device=device,
                        ),
                    )
                )
            cached = tuple(packed)
            if (
                len(_VECTORIZED_SPIN_MINUS_TWO_PHI_ZERO_TABLES)
                >= _VECTORIZED_SPIN_MINUS_TWO_PHI_ZERO_TABLES_MAX_ENTRIES
            ):
                oldest = next(
                    iter(_VECTORIZED_SPIN_MINUS_TWO_PHI_ZERO_TABLES)
                )
                del _VECTORIZED_SPIN_MINUS_TWO_PHI_ZERO_TABLES[oldest]
            _VECTORIZED_SPIN_MINUS_TWO_PHI_ZERO_TABLES[key] = cached
    return cached


def _vectorized_spin_minus_two_phi_zero_values(theta, phi):
    """Evaluate independent mode lanes with three fixed-width kernels."""

    cos_half = torch.cos(0.5 * theta)
    sin_half = torch.sin(0.5 * theta)
    cos_powers = torch.stack(
        tuple(cos_half**power for power in range(9))
    )
    sin_powers = torch.stack(
        tuple(sin_half**power for power in range(9))
    )
    harmonics = [None] * len(_SPIN_MINUS_TWO_PHI_ZERO_MODES)

    for mode_indices, coefficients, cos_indices, sin_indices, emms in (
        _vectorized_spin_minus_two_phi_zero_tables(theta.dtype, theta.device)
    ):
        terms = (
            coefficients
            * cos_powers[cos_indices]
            * sin_powers[sin_indices]
        )
        amplitude = torch.zeros_like(terms[:, 0])
        for term_index in range(terms.shape[1]):
            amplitude = amplitude + terms[:, term_index]
        phase = emms * phi
        values = torch.complex(
            amplitude * torch.cos(phase),
            amplitude * torch.sin(phase),
        )
        for lane, mode_index in enumerate(mode_indices):
            harmonics[mode_index] = values[lane]
    return tuple(harmonics)


def vectorized_spin_minus_two_spherical_harmonics_phi_zero(
    theta,
    *,
    dtype,
    device,
):
    r"""Evaluate 21 scalar harmonics as three exact internal mode packs.

    This preserves each mode's scalar arithmetic order while replacing the
    per-mode Python loop with fixed-width tensor lanes. Inputs that could
    change differentiation, layout, or dispatch semantics use the ordinary
    evaluator.
    """

    if dtype not in (torch.float32, torch.float64):
        raise TypeError("spherical-harmonic angles require a real Torch dtype")
    theta = torch.as_tensor(theta, dtype=dtype, device=device)
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and requested_device.index is None:
        requested_device = torch.device("cuda", torch.cuda.current_device())
    supported = (
        type(theta) is torch.Tensor
        and theta.layout is torch.strided
        and theta.ndim == 0
        and theta.dtype == dtype
        and theta.device == requested_device
        and not theta.is_conj()
        and not theta.is_neg()
        and not theta.requires_grad
        and torch.autograd.forward_ad.unpack_dual(theta).tangent is None
        and not (
            theta.device.type == "cuda"
            and torch.cuda.is_current_stream_capturing()
        )
    )
    if not supported:
        return spin_minus_two_spherical_harmonics_phi_zero(
            theta,
            dtype=dtype,
            device=device,
        )
    phi = torch.zeros_like(theta)
    return dict(
        zip(
            _SPIN_MINUS_TWO_PHI_ZERO_MODES,
            _vectorized_spin_minus_two_phi_zero_values(theta, phi),
        )
    )


def spin_minus_two_spherical_harmonics_phi_zero(theta, *, dtype, device):
    r"""Evaluate all :math:`{}_{-2}Y_{\ell m}(\theta, 0)`, ell 2 through 4.

    Trigonometric values and integer powers are shared across the 21 modes.
    Each mode retains the scalar evaluator's term order and phase operations,
    including the signed-zero phase produced for negative ``m``.
    """

    if dtype not in (torch.float32, torch.float64):
        raise TypeError("spherical-harmonic angles require a real Torch dtype")

    theta = torch.as_tensor(theta, dtype=dtype, device=device)
    phi = torch.as_tensor(0.0, dtype=dtype, device=device)
    theta, phi = torch.broadcast_tensors(theta, phi)
    return dict(
        zip(
            _SPIN_MINUS_TWO_PHI_ZERO_MODES,
            _spin_minus_two_phi_zero_values(theta, phi),
        )
    )


def _stacked_spin_minus_two_phi_zero(theta):
    """Tensor-only trace target for the exact bulk harmonic evaluation."""

    return torch.stack(
        _spin_minus_two_phi_zero_values(theta, torch.zeros_like(theta))
    )


def _scripted_spin_minus_two_phi_zero(theta):
    """Run the dtype/device-cached TorchScript trace for scalar ``theta``."""

    key = (theta.dtype, theta.device)
    traced = _SCRIPTED_SPIN_MINUS_TWO_PHI_ZERO.get(key)
    if traced is None:
        with _SCRIPTED_SPIN_MINUS_TWO_PHI_ZERO_LOCK:
            traced = _SCRIPTED_SPIN_MINUS_TWO_PHI_ZERO.get(key)
            if traced is None:
                example = torch.zeros((), dtype=theta.dtype, device=theta.device)
                traced = torch.jit.trace(
                    _stacked_spin_minus_two_phi_zero,
                    example,
                    check_trace=False,
                )
                _SCRIPTED_SPIN_MINUS_TWO_PHI_ZERO[key] = traced
    return traced(theta)


def scripted_spin_minus_two_spherical_harmonics_phi_zero(
    theta,
    *,
    dtype,
    device,
):
    r"""Evaluate the 21 scalar harmonics with a cached CPU TorchScript trace.

    CUDA falls back to eager execution because TorchScript changed low-order
    bits for otherwise valid XPHM requests during qualification.
    """

    if dtype not in (torch.float32, torch.float64):
        raise TypeError("spherical-harmonic angles require a real Torch dtype")
    theta = torch.as_tensor(theta, dtype=dtype, device=device)
    if theta.ndim != 0:
        raise ValueError("scripted bulk spherical harmonics require scalar theta")
    if theta.device.type != "cpu":
        return spin_minus_two_spherical_harmonics_phi_zero(
            theta,
            dtype=dtype,
            device=device,
        )
    stacked = _scripted_spin_minus_two_phi_zero(theta)
    return {
        mode: stacked[index]
        for index, mode in enumerate(_SPIN_MINUS_TWO_PHI_ZERO_MODES)
    }


def _cudagraph_spin_minus_two_phi_zero_key(dtype, device):
    """Return a process/thread/stream-local CUDA Graph cache key."""

    stream = torch.cuda.current_stream(device)
    return (
        os.getpid(),
        threading.get_ident(),
        dtype,
        device,
        stream.cuda_stream,
    )


def _build_cudagraph_spin_minus_two_phi_zero(dtype, device):
    """Capture the unchanged eager scalar kernels on a private stream."""

    static_theta = torch.zeros((), dtype=dtype, device=device)
    current_stream = torch.cuda.current_stream(device)
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(current_stream)
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            _stacked_spin_minus_two_phi_zero(static_theta)
    current_stream.wait_stream(capture_stream)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        packed = _stacked_spin_minus_two_phi_zero(static_theta)
    return static_theta, graph, packed


def cudagraphed_spin_minus_two_spherical_harmonics_phi_zero(
    theta,
    *,
    dtype,
    device,
):
    r"""Replay the exact eager CUDA kernels for the 21 scalar harmonics.

    The captured graph is local to a process, Python thread, and CUDA stream.
    Its static output is cloned before return, so values from one request
    cannot be overwritten by a subsequent replay. Unsupported inputs and
    safe capture failures use the ordinary eager evaluator.
    """

    if dtype not in (torch.float32, torch.float64):
        raise TypeError("spherical-harmonic angles require a real Torch dtype")
    requested_device = torch.device(device)
    if requested_device.type != "cuda" or not torch.cuda.is_available():
        return spin_minus_two_spherical_harmonics_phi_zero(
            theta,
            dtype=dtype,
            device=device,
        )
    if requested_device.index is None:
        requested_device = torch.device("cuda", torch.cuda.current_device())

    if type(theta) is torch.Tensor:
        graph_input = (
            theta
            if theta.layout is torch.strided
            and theta.ndim == 0
            and theta.dtype == dtype
            and theta.device == requested_device
            and not theta.is_conj()
            and not theta.is_neg()
            and not theta.requires_grad
            and torch.autograd.forward_ad.unpack_dual(theta).tangent is None
            else None
        )
    elif type(theta) is float and math.isfinite(theta):
        graph_input = theta
    else:
        graph_input = None
    if graph_input is None or torch.cuda.is_current_stream_capturing():
        return spin_minus_two_spherical_harmonics_phi_zero(
            theta,
            dtype=dtype,
            device=device,
        )

    key = _cudagraph_spin_minus_two_phi_zero_key(dtype, requested_device)
    with _CUDAGRAPH_SPIN_MINUS_TWO_PHI_ZERO_LOCK:
        cached = _CUDAGRAPH_SPIN_MINUS_TWO_PHI_ZERO.get(key)
        if cached is None and key not in _CUDAGRAPH_SPIN_MINUS_TWO_PHI_ZERO_FAILURES:
            try:
                cached = _build_cudagraph_spin_minus_two_phi_zero(
                    dtype,
                    requested_device,
                )
            except Exception:
                if (
                    len(_CUDAGRAPH_SPIN_MINUS_TWO_PHI_ZERO_FAILURES)
                    >= _CUDAGRAPH_SPIN_MINUS_TWO_PHI_ZERO_MAX_ENTRIES
                ):
                    _CUDAGRAPH_SPIN_MINUS_TWO_PHI_ZERO_FAILURES.pop()
                _CUDAGRAPH_SPIN_MINUS_TWO_PHI_ZERO_FAILURES.add(key)
            else:
                if (
                    len(_CUDAGRAPH_SPIN_MINUS_TWO_PHI_ZERO)
                    >= _CUDAGRAPH_SPIN_MINUS_TWO_PHI_ZERO_MAX_ENTRIES
                ):
                    oldest = next(
                        iter(_CUDAGRAPH_SPIN_MINUS_TWO_PHI_ZERO)
                    )
                    del _CUDAGRAPH_SPIN_MINUS_TWO_PHI_ZERO[oldest]
                _CUDAGRAPH_SPIN_MINUS_TWO_PHI_ZERO[key] = cached
    if cached is None:
        return spin_minus_two_spherical_harmonics_phi_zero(
            theta,
            dtype=dtype,
            device=device,
        )

    static_theta, graph, packed = cached
    if type(graph_input) is torch.Tensor:
        static_theta.copy_(graph_input)
    else:
        static_theta.fill_(graph_input)
    graph.replay()
    owned = packed.clone()
    return {
        mode: owned[index]
        for index, mode in enumerate(_SPIN_MINUS_TWO_PHI_ZERO_MODES)
    }


def scalar_spin_weighted_spherical_harmonic(
    theta, phi, spin_weight, ell, emm
):
    """Return the scalar harmonic as a Python complex value.

    This is the no-LAL compatibility path for legacy mode-summing helpers,
    whose public scalar contract predates the device-native implementation.
    """
    value = spin_weighted_spherical_harmonic(
        theta,
        phi,
        spin_weight,
        ell,
        emm,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    return complex(value.item())


__all__ = [
    "cudagraphed_spin_minus_two_spherical_harmonics_phi_zero",
    "scalar_spin_weighted_spherical_harmonic",
    "selected_spin_minus_two_spherical_harmonics",
    "scripted_spin_minus_two_spherical_harmonics_phi_zero",
    "spin_minus_two_spherical_harmonics_phi_zero",
    "spin_weighted_spherical_harmonic",
    "vectorized_spin_minus_two_spherical_harmonics_phi_zero",
]
