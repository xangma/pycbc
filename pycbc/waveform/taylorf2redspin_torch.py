# Copyright (C) 2026
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch-native TaylorF2 reduced-spin frequency-domain waveforms.

This ports ``XLALSimInspiralTaylorF2ReducedSpin`` and its tidal variant.
Scalar PN coefficients are assembled in Python; frequency-dependent phase,
amplitude, masking, and polarization work runs on the active Torch device.

LAL does not expose these models through its arbitrary-frequency API.  The
sequence generator below is therefore a native extension of the same analytic
waveforms, truncated at Schwarzschild ISCO.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import lal
import torch

import pycbc.scheme as _scheme
from pycbc.types import Array as PyCBCArray
from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData

_DEFAULT_ONLY_ORDER_KEYS = (
    "spin_order",
    "tidal_order",
    "eccentricity_order",
)
_TRANSVERSE_SPIN_KEYS = ("spin1x", "spin1y", "spin2x", "spin2y")
_TIDAL_EXTENSION_KEYS = (
    "quadparam1",
    "quadparam2",
    "dquadparam1",
    "dquadparam2",
)
_NON_GR_KEYS = (
    "dchi0", "dchi1", "dchi2", "dchi3", "dchi4", "dchi5", "dchi5l", "dchi6", "dchi6l", "dchi7",
    "dbeta2", "dbeta3", "dalpha2", "dalpha3", "dalpha4", "dalpha5",
)


def _is_default_order(value) -> bool:
    if value is None:
        return True
    return _as_order(value) == -1


def _is_nonzero(value) -> bool:
    if value is None:
        return False
    return bool(float(value) != 0.0)


_APPROXIMANTS = {"TaylorF2RedSpin", "TaylorF2RedSpinTidal"}
_PN_ORDERS = frozenset((-1, 0, 1, 2, 3, 4, 5, 6, 7))
_PI = lal.PI


def _as_order(value):
    """Return an exact integer order, or ``None`` for invalid values."""
    try:
        integer = int(value)
        return integer if float(value) == integer else None
    except (TypeError, ValueError, OverflowError):
        return None


def _finite_optional(value) -> bool:
    if value is None:
        return True
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def taylorf2redspin_native_supported(params) -> bool:
    """Return whether ``params`` can use the native reduced-spin port."""
    approximant = params.get("approximant")
    if approximant not in _APPROXIMANTS:
        return False
    if _as_order(params.get("phase_order", -1)) not in _PN_ORDERS:
        return False
    if _as_order(params.get("amplitude_order", -1)) not in _PN_ORDERS:
        return False
    if any(
        not _is_default_order(params.get(key, -1))
        for key in _DEFAULT_ONLY_ORDER_KEYS
    ):
        return False
    if any(
        _is_nonzero(params.get(key, 0.0))
        for key in (
            _TRANSVERSE_SPIN_KEYS
            + _TIDAL_EXTENSION_KEYS
            + _NON_GR_KEYS
            + (
                "eccentricity",
                "mean_per_ano",
                "frame_axis",
                "modes_choice",
                "side_bands",
            )
        )
    ):
        return False
    if params.get("mode_array") is not None or params.get("numrel_data", ""):
        return False

    lambdas = (params.get("lambda1"), params.get("lambda2"))
    if approximant == "TaylorF2RedSpin":
        if any(_is_nonzero(value) for value in lambdas):
            return False
    elif not all(_finite_optional(value) for value in lambdas):
        return False
    return True


def taylorf2redspin_sequence_native_supported(params) -> bool:
    """Return whether arbitrary-frequency reduced-spin generation is native."""
    return taylorf2redspin_native_supported(params)


@dataclass(frozen=True)
class _Inputs:
    """Validated scalar inputs shared by regular and sequence generation."""

    approximant: str
    mass1: float
    mass2: float
    total_mass: float
    eta: float
    chi: float
    lambda1: float
    lambda2: float
    distance_m: float
    inclination: float
    coa_phase: float
    long_asc_nodes: float
    phase_order: int
    amplitude_order: int
    device: torch.device
    real_dtype: torch.dtype
    complex_dtype: torch.dtype


@dataclass(frozen=True)
class _Coefficients:
    """Frequency-independent PN phase and amplitude coefficients."""

    psi_newtonian: float
    psi2: float
    psi3: float
    psi4: float
    psi5: float
    psi6: float
    psi6_log: float
    psi7: float
    psi10_tidal: float
    psi12_tidal: float
    alpha2: float
    alpha3: float
    alpha4: float
    alpha5: float
    alpha6: float
    alpha6_log: float
    alpha7: float


def _validated_inputs(params, *, sequence=False) -> _Inputs:
    if not taylorf2redspin_native_supported(params):
        raise ValueError(
            "TaylorF2RedSpin parameters are not supported by the native "
            "Torch path"
        )
    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise RuntimeError("native Torch TaylorF2RedSpin requires TorchScheme")

    approximant = params["approximant"]
    mass1 = float(params["mass1"])
    mass2 = float(params["mass2"])
    spin1z = float(params.get("spin1z", 0.0))
    spin2z = float(params.get("spin2z", 0.0))
    distance = float(params.get("distance", 1.0))
    inclination = float(params.get("inclination", 0.0))
    coa_phase = float(params.get("coa_phase", 0.0))
    f_ref = float(params.get("f_ref", 0.0))
    long_asc_nodes = (
        0.0 if sequence else float(params.get("long_asc_nodes", 0.0))
    )
    lambda1 = float(params.get("lambda1") or 0.0)
    lambda2 = float(params.get("lambda2") or 0.0)
    if not all(
        math.isfinite(value)
        for value in (
            mass1,
            mass2,
            spin1z,
            spin2z,
            distance,
            inclination,
            coa_phase,
            f_ref,
            long_asc_nodes,
            lambda1,
            lambda2,
        )
    ):
        raise ValueError("TaylorF2RedSpin parameters must be finite")
    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("TaylorF2RedSpin component masses must be positive")
    if distance <= 0.0:
        raise ValueError("TaylorF2RedSpin distance must be positive")
    if f_ref < 0.0:
        raise ValueError("TaylorF2RedSpin f_ref must be non-negative")

    total_mass = mass1 + mass2
    eta = mass1 * mass2 / total_mass**2
    if approximant == "TaylorF2RedSpin":
        if abs(spin1z) > 1.0 or abs(spin2z) > 1.0:
            raise ValueError(
                "TaylorF2RedSpin component spins must be between -1 and 1"
            )
        delta = (mass1 - mass2) / total_mass
        chi_s = 0.5 * (spin1z + spin2z)
        chi_a = 0.5 * (spin1z - spin2z)
        chi = chi_s * (1.0 - 76.0 * eta / 113.0) + delta * chi_a
    else:
        chi = (mass1 * spin1z + mass2 * spin2z) / total_mass
        if abs(chi) > 1.0:
            raise ValueError(
                "TaylorF2RedSpinTidal effective spin must be between -1 and 1"
            )

    device = state.torch_device
    real_dtype = torch.float32 if device.type == "mps" else torch.float64
    complex_dtype = (
        torch.complex64 if real_dtype == torch.float32 else torch.complex128
    )
    return _Inputs(
        approximant=approximant,
        mass1=mass1,
        mass2=mass2,
        total_mass=total_mass,
        eta=eta,
        chi=chi,
        lambda1=lambda1,
        lambda2=lambda2,
        distance_m=distance * 1.0e6 * lal.PC_SI,
        inclination=inclination,
        coa_phase=coa_phase,
        long_asc_nodes=long_asc_nodes,
        phase_order=_as_order(params.get("phase_order", -1)),
        amplitude_order=_as_order(params.get("amplitude_order", -1)),
        device=device,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
    )


def _coefficients(inputs: _Inputs) -> _Coefficients:
    eta = inputs.eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    chi = inputs.chi
    eta_factor = -113.0 + 76.0 * eta
    pi2 = _PI * _PI

    psi3_spin = 113.0 * chi / 3.0
    psi4_spin = (
        63845.0 * (-81.0 + 4.0 * eta) * chi * chi
        / (8.0 * eta_factor * eta_factor)
    )
    psi5_spin = (
        -565.0
        * (-146597.0 + 135856.0 * eta + 17136.0 * eta2)
        * chi
        / (2268.0 * eta_factor)
    )
    alpha3_spin = 113.0 * chi / 24.0
    alpha4_spin = (
        12769.0 * chi * chi * (-81.0 + 4.0 * eta)
        / (32.0 * eta_factor * eta_factor)
    )
    alpha5_spin = (
        -113.0
        * chi
        * (502429.0 - 591368.0 * eta + 1680.0 * eta2)
        / (16128.0 * eta_factor)
    )

    psi2 = 3715.0 / 756.0 + 55.0 * eta / 9.0
    psi3 = psi3_spin - 16.0 * _PI
    psi4 = (
        15293365.0 / 508032.0
        + 27145.0 * eta / 504.0
        + 3085.0 * eta2 / 72.0
        + psi4_spin
    )
    psi5 = 38645.0 * _PI / 756.0 - 65.0 * _PI * eta / 9.0 + psi5_spin
    psi6 = (
        11583231236531.0 / 4694215680.0
        - 640.0 * pi2 / 3.0
        - 6848.0 * lal.GAMMA / 21.0
        + (-5162.983708047263 + 2255.0 * pi2 / 12.0) * eta
        + 76055.0 * eta2 / 1728.0
        - 127825.0 * eta3 / 1296.0
    )
    psi6_log = -6848.0 / 21.0
    psi7 = (
        77096675.0 * _PI / 254016.0
        + 378515.0 * _PI * eta / 1512.0
        - 74045.0 * _PI * eta2 / 756.0
    )

    alpha2 = 1.1056547619047619 + 11.0 * eta / 8.0
    alpha3 = -2.0 * _PI + alpha3_spin
    alpha4 = (
        0.8939214212884228
        + 18913.0 * eta / 16128.0
        + 1379.0 * eta2 / 1152.0
        + alpha4_spin
    )
    alpha5 = -4757.0 * _PI / 1344.0 + 57.0 * eta * _PI / 16.0 + alpha5_spin
    alpha6 = (
        -58.601030974347324
        + 3526813753.0 * eta / 2.7869184e7
        - 1041557.0 * eta2 / 258048.0
        + 67999.0 * eta3 / 82944.0
        + 10.0 * pi2 / 3.0
        - 451.0 * eta * pi2 / 96.0
    )
    alpha6_log = 856.0 / 105.0
    alpha7 = (
        -5111593.0 * _PI / 2.709504e6
        - 72221.0 * eta * _PI / 24192.0
        - 1349.0 * eta2 * _PI / 24192.0
    )

    phase_order = inputs.phase_order
    if phase_order != -1:
        if phase_order < 2:
            psi2 = 0.0
        if phase_order < 3:
            psi3 = 0.0
        if phase_order < 4:
            psi4 = 0.0
        if phase_order < 5:
            psi5 = 0.0
        if phase_order < 6:
            psi6 = 0.0
            psi6_log = 0.0
        if phase_order < 7:
            psi7 = 0.0

    amplitude_order = inputs.amplitude_order
    if amplitude_order != -1:
        if amplitude_order < 2:
            alpha2 = 0.0
        if amplitude_order < 3:
            alpha3 = 0.0
        if amplitude_order < 4:
            alpha4 = 0.0
        if amplitude_order < 5:
            alpha5 = 0.0
        if amplitude_order < 6:
            alpha6 = 0.0
            alpha6_log = 0.0
        if amplitude_order < 7:
            alpha7 = 0.0

    psi10_tidal = 0.0
    psi12_tidal = 0.0
    if inputs.approximant == "TaylorF2RedSpinTidal":
        xi1 = inputs.mass1 / inputs.total_mass
        xi2 = inputs.mass2 / inputs.total_mass
        psi10_tidal = (
            -24.0 / xi1 * (1.0 + 11.0 * xi2) * inputs.lambda1 * xi1**5
            - 24.0 / xi2 * (1.0 + 11.0 * xi1) * inputs.lambda2 * xi2**5
        )
        psi12_tidal = (
            -5.0
            / 28.0
            / xi1
            * (3179.0 - 919.0 * xi1 - 2286.0 * xi1**2 + 260.0 * xi1**3)
            * inputs.lambda1
            * xi1**5
            - 5.0
            / 28.0
            / xi2
            * (3179.0 - 919.0 * xi2 - 2286.0 * xi2**2 + 260.0 * xi2**3)
            * inputs.lambda2
            * xi2**5
        )

    return _Coefficients(
        psi_newtonian=3.0 / (128.0 * eta),
        psi2=psi2,
        psi3=psi3,
        psi4=psi4,
        psi5=psi5,
        psi6=psi6,
        psi6_log=psi6_log,
        psi7=psi7,
        psi10_tidal=psi10_tidal,
        psi12_tidal=psi12_tidal,
        alpha2=alpha2,
        alpha3=alpha3,
        alpha4=alpha4,
        alpha5=alpha5,
        alpha6=alpha6,
        alpha6_log=alpha6_log,
        alpha7=alpha7,
    )


def _isco_frequency(inputs: _Inputs) -> float:
    pi_mass = _PI * inputs.total_mass * lal.MTSUN_SI
    return 1.0 / (6.0**1.5 * pi_mass)


def _model_samples(
    inputs: _Inputs,
    coefficients: _Coefficients,
    frequencies: torch.Tensor,
    *,
    time_shift=0.0,
) -> torch.Tensor:
    """Evaluate the inclination-independent waveform on-device."""
    pi_mass = _PI * inputs.total_mass * lal.MTSUN_SI
    v3 = pi_mass * frequencies
    velocity = torch.pow(v3, 1.0 / 3.0)
    v2 = velocity * velocity
    v4 = v3 * velocity
    v5 = v4 * velocity
    v6 = v3 * v3
    v7 = v6 * velocity
    log_velocity = torch.log(velocity)

    phase = coefficients.psi_newtonian / v5 * (
        1.0
        + coefficients.psi2 * v2
        + coefficients.psi3 * v3
        + coefficients.psi4 * v4
        + coefficients.psi5 * v5 * (1.0 + 3.0 * log_velocity)
        + (
            coefficients.psi6
            + coefficients.psi6_log * (math.log(4.0) + log_velocity)
        )
        * v6
        + coefficients.psi7 * v7
    )
    if inputs.approximant == "TaylorF2RedSpinTidal":
        phase = (
            phase
            + coefficients.psi10_tidal * v5 * v5
            + coefficients.psi12_tidal * v6 * v6
        )

    amplitude0 = (
        -pow(inputs.total_mass * lal.MTSUN_SI, 5.0 / 6.0)
        * math.sqrt(5.0 * inputs.eta / 24.0)
        / (math.cbrt(_PI * _PI) * inputs.distance_m / lal.C_SI)
    )
    amplitude = amplitude0 * torch.pow(frequencies, -7.0 / 6.0) * (
        1.0
        + coefficients.alpha2 * v2
        + coefficients.alpha3 * v3
        + coefficients.alpha4 * v4
        + coefficients.alpha5 * v5
        + (
            coefficients.alpha6
            + coefficients.alpha6_log
            * (lal.GAMMA + math.log(4.0) + log_velocity)
        )
        * v6
        + coefficients.alpha7 * v7
    )
    angle = (
        phase
        + 2.0 * _PI * time_shift * frequencies
        - 2.0 * inputs.coa_phase
        - _PI / 4.0
    )
    return torch.complex(
        amplitude * torch.cos(angle),
        -amplitude * torch.sin(angle),
    ).to(inputs.complex_dtype)


def _polarizations(inputs: _Inputs, samples: torch.Tensor):
    cosi = math.cos(inputs.inclination)
    plus0 = 0.5 * (1.0 + cosi * cosi) * samples
    cross0 = samples * complex(0.0, -cosi)
    cos_nodes = math.cos(2.0 * inputs.long_asc_nodes)
    sin_nodes = math.sin(2.0 * inputs.long_asc_nodes)
    return (
        (cos_nodes * plus0 + sin_nodes * cross0).to(inputs.complex_dtype),
        (cos_nodes * cross0 - sin_nodes * plus0).to(inputs.complex_dtype),
    )


def taylorf2redspin_fd_torch(**params):
    """Generate a regular-grid Torch-native reduced-spin waveform."""
    inputs = _validated_inputs(params)
    coefficients = _coefficients(inputs)
    delta_f = float(params["delta_f"])
    f_lower = float(params["f_lower"])
    f_final = float(params.get("f_final", 0.0))
    if not all(math.isfinite(value) for value in (delta_f, f_lower, f_final)):
        raise ValueError("TaylorF2RedSpin frequencies must be finite")
    if delta_f <= 0.0 or f_lower <= 0.0:
        raise ValueError(
            "TaylorF2RedSpin delta_f and f_lower must be positive"
        )
    if f_final < 0.0:
        raise ValueError("TaylorF2RedSpin f_final must be non-negative")

    f_max = f_final if f_final > 0.0 else _isco_frequency(inputs)
    length = int(f_max / delta_f + 1.0)
    first_bin = int(math.ceil(f_lower / delta_f))
    raw = torch.zeros(
        length,
        dtype=inputs.complex_dtype,
        device=inputs.device,
    )
    epoch = float(lal.LIGOTimeGPS(-1.0 / delta_f))
    if first_bin < length:
        frequencies = (
            torch.arange(
                first_bin,
                length,
                dtype=inputs.real_dtype,
                device=inputs.device,
            )
            * delta_f
        )
        raw[first_bin:] = _model_samples(
            inputs,
            coefficients,
            frequencies,
            time_shift=epoch,
        )
    plus, cross = _polarizations(inputs, raw)
    return (
        FrequencySeries(
            TorchArrayData(plus), delta_f=delta_f, epoch=epoch, copy=False
        ),
        FrequencySeries(
            TorchArrayData(cross), delta_f=delta_f, epoch=epoch, copy=False
        ),
    )


def _sequence_frequencies(sample_points, inputs: _Inputs) -> torch.Tensor:
    values = getattr(sample_points, "_data", sample_points)
    if isinstance(values, TorchArrayData):
        values = values.tensor
    frequencies = torch.as_tensor(
        values,
        dtype=inputs.real_dtype,
        device=inputs.device,
    )
    if frequencies.ndim != 1 or frequencies.numel() == 0:
        raise ValueError(
            "TaylorF2RedSpin sample_points must be a non-empty vector"
        )
    if not bool(torch.all(torch.isfinite(frequencies))):
        raise ValueError("TaylorF2RedSpin sample_points must be finite")
    if bool(torch.any(frequencies <= 0.0)):
        raise ValueError("TaylorF2RedSpin sample_points must be positive")
    return frequencies


def taylorf2redspin_fd_sequence_torch(**params):
    """Evaluate a reduced-spin waveform at arbitrary device frequencies."""
    inputs = _validated_inputs(params, sequence=True)
    coefficients = _coefficients(inputs)
    frequencies = _sequence_frequencies(params["sample_points"], inputs)
    plus = torch.zeros(
        frequencies.shape,
        dtype=inputs.complex_dtype,
        device=inputs.device,
    )
    cross = torch.zeros_like(plus)
    active = frequencies <= _isco_frequency(inputs)
    samples = _model_samples(inputs, coefficients, frequencies[active])
    plus_active, cross_active = _polarizations(inputs, samples)
    plus[active] = plus_active
    cross[active] = cross_active
    return (
        PyCBCArray(TorchArrayData(plus), copy=False),
        PyCBCArray(TorchArrayData(cross), copy=False),
    )


__all__ = [
    "taylorf2redspin_fd_sequence_torch",
    "taylorf2redspin_fd_torch",
    "taylorf2redspin_native_supported",
    "taylorf2redspin_sequence_native_supported",
]
