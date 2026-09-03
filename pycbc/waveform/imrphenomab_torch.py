# Copyright (C) 2026
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch-native IMRPhenomA and IMRPhenomB frequency-domain waveforms.

The implementation follows ``LALSimIMRPhenom.c``.  Scalar fit coefficients
are assembled in Python and every frequency-dependent operation runs on the
active Torch device.  The legacy models deliberately have different grid
boundary conventions; the regular generators below preserve both exactly.

Activation
----------
- Supported calls are native by default on CPU and CUDA.
- Per-model flags: ``PYCBC_IMRPHENOMA_NATIVE=1`` and
  ``PYCBC_IMRPHENOMB_NATIVE=1``
- Global flag: ``PYCBC_TORCH_NATIVE_PORTS``

Apple MPS remains opt-in because its float32 phase evaluation loses accuracy
for low-frequency and unequal-mass systems. Explicit flags override the
device-aware default.

LAL does not implement IMRPhenomA/B through its arbitrary-frequency API.  The
native sequence functions provide that missing operation by evaluating the
same analytic models below their fitted cutoff.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pycbc import lal_compat as lal
import torch

import pycbc.scheme as _scheme
from pycbc.types import Array as PyCBCArray
from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData
from pycbc.waveform.imrphenomd_torch import (
    _DEFAULT_ONLY_ORDER_KEYS,
    _NON_GR_KEYS,
    _TIDAL_EXTENSION_KEYS,
    _TRANSVERSE_SPIN_KEYS,
    _is_default_order,
    _is_nonzero,
)

_PI = lal.PI
_MTSUN_SI = lal.MTSUN_SI
_PC_SI = lal.PC_SI
_C_SI = lal.C_SI
_APPROXIMANTS = {"IMRPhenomA", "IMRPhenomB"}
_ALL_SPIN_KEYS = _TRANSVERSE_SPIN_KEYS + ("spin1z", "spin2z")


def imrphenomab_native_supported(params) -> bool:
    """Return whether the native implementation can honor ``params``."""
    approximant = params.get("approximant")
    if approximant not in _APPROXIMANTS:
        return False
    if any(
        not _is_default_order(params.get(key, -1))
        for key in (*_DEFAULT_ONLY_ORDER_KEYS, "phase_order", "amplitude_order")
    ):
        return False
    if any(
        _is_nonzero(params.get(key, 0.0))
        for key in (
            _TRANSVERSE_SPIN_KEYS
            + _TIDAL_EXTENSION_KEYS
            + _NON_GR_KEYS
            + (
                "lambda1",
                "lambda2",
                "eccentricity",
                "mean_per_ano",
                "frame_axis",
                "modes_choice",
                "side_bands",
            )
        )
    ):
        return False
    if approximant == "IMRPhenomA" and any(
        _is_nonzero(params.get(key, 0.0)) for key in _ALL_SPIN_KEYS
    ):
        return False
    if params.get("mode_array") is not None or params.get("numrel_data", ""):
        return False
    return True


def imrphenomab_sequence_native_supported(params) -> bool:
    """Return whether arbitrary-frequency IMRPhenomA/B is Torch-native."""
    return imrphenomab_native_supported(params)


def imrphenomab_default_native_supported(_params) -> bool:
    """Return whether unflagged native use is accurate on this device.

    The legacy fits contain large phase terms that cancel. Apple MPS evaluates
    them in float32, which can materially reduce accuracy at low frequencies
    and unequal masses, so its native path remains an explicit opt-in.
    """
    state = _scheme.mgr.state
    return not (
        isinstance(state, _scheme.TorchScheme)
        and state.torch_device.type == "mps"
    )


@dataclass(frozen=True)
class _Inputs:
    """Validated scalar inputs shared by regular and sequence generation."""

    approximant: str
    total_mass: float
    eta: float
    chi: float
    distance_m: float
    inclination: float
    coa_phase: float
    long_asc_nodes: float
    device: torch.device
    real_dtype: torch.dtype
    complex_dtype: torch.dtype


@dataclass(frozen=True)
class _Coefficients:
    """Frequency-independent phenomenological fit coefficients."""

    f_cut: float
    f_merger: float
    f_ring: float
    sigma: float
    psi0: float = 0.0
    psi1: float = 0.0
    psi2: float = 0.0
    psi3: float = 0.0
    psi4: float = 0.0
    psi5: float = 0.0
    psi6: float = 0.0
    psi7: float = 0.0
    psi8: float = 0.0


def _validated_inputs(params, *, sequence=False) -> _Inputs:
    if not imrphenomab_native_supported(params):
        raise ValueError(
            "IMRPhenomA/B parameters are not supported by the native Torch path"
        )
    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise RuntimeError("native Torch IMRPhenomA/B requires TorchScheme")

    approximant = params["approximant"]
    mass1 = float(params["mass1"])
    mass2 = float(params["mass2"])
    spin1z = float(params.get("spin1z", 0.0))
    spin2z = float(params.get("spin2z", 0.0))
    distance = float(params["distance"])
    inclination = float(params.get("inclination", 0.0))
    coa_phase = float(params.get("coa_phase", 0.0))
    f_ref = float(params.get("f_ref", 0.0))
    # SimInspiralChooseFDWaveformSequence has no ascending-node argument.
    long_asc_nodes = (
        0.0 if sequence else float(params.get("long_asc_nodes", 0.0))
    )
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
        )
    ):
        raise ValueError("IMRPhenomA/B parameters must be finite")
    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("IMRPhenomA/B component masses must be positive")
    if distance <= 0.0:
        raise ValueError("IMRPhenomA/B distance must be positive")
    if f_ref < 0.0:
        raise ValueError("IMRPhenomA/B f_ref must be non-negative")

    total_mass = mass1 + mass2
    eta = mass1 * mass2 / (total_mass * total_mass)
    chi = (mass1 * spin1z + mass2 * spin2z) / total_mass
    if approximant == "IMRPhenomB" and abs(chi) > 1.0:
        raise ValueError("IMRPhenomB effective spin must be between -1 and 1")

    device = state.torch_device
    real_dtype = torch.float32 if device.type == "mps" else torch.float64
    complex_dtype = (
        torch.complex64 if real_dtype == torch.float32 else torch.complex128
    )
    return _Inputs(
        approximant=approximant,
        total_mass=total_mass,
        eta=eta,
        chi=chi,
        distance_m=distance * 1.0e6 * _PC_SI,
        inclination=inclination,
        coa_phase=coa_phase,
        long_asc_nodes=long_asc_nodes,
        device=device,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
    )


def _phenoma_coefficients(inputs: _Inputs) -> _Coefficients:
    eta = inputs.eta
    eta2 = eta * eta
    pi_m = inputs.total_mass * _PI * _MTSUN_SI

    def fit(a, b, c):
        return (a * eta2 + b * eta + c) / pi_m

    return _Coefficients(
        f_cut=fit(1.7086, -0.26592, 0.28236),
        f_merger=fit(0.66389, -0.10321, 0.10979),
        f_ring=fit(1.3278, -0.20642, 0.21957),
        sigma=fit(1.1383, -0.17700, 0.046834),
        psi0=(-0.15829 * eta2 + 0.087016 * eta - 0.033382)
        / (eta * pow(pi_m, 5.0 / 3.0)),
        psi2=(32.967 * eta2 - 19.0 * eta + 2.1345) / (eta * pi_m),
        psi3=(-308.49 * eta2 + 182.11 * eta - 21.727)
        / (eta * pow(pi_m, 2.0 / 3.0)),
        psi4=(1152.5 * eta2 - 714.77 * eta + 99.692)
        / (eta * pow(pi_m, 1.0 / 3.0)),
        psi6=(1205.7 * eta2 - 842.33 * eta + 180.46)
        / (eta / pow(pi_m, 1.0 / 3.0)),
    )


def _phenomb_coefficients(inputs: _Inputs) -> _Coefficients:
    eta = inputs.eta
    chi = inputs.chi
    eta2 = eta * eta
    eta3 = eta2 * eta
    chi2 = chi * chi
    etachi = eta * chi
    etachi2 = eta * chi2
    eta2chi = eta2 * chi
    pi_m = inputs.total_mass * _PI * _MTSUN_SI
    one_minus_chi = 1.0 - chi

    psi2 = (
        3715.0 / 756.0
        - 920.91 * eta
        + 492.13 * etachi
        + 135.03 * etachi2
        + 6741.9 * eta2
        - 1053.4 * eta2chi
        - 13397.0 * eta3
    )
    psi3 = (
        -16.0 * _PI
        + 113.0 * chi / 3.0
        + 17022.0 * eta
        - 9565.9 * etachi
        - 2182.1 * etachi2
        - 121370.0 * eta2
        + 20752.0 * eta2chi
        + 238590.0 * eta3
    )
    psi4 = (
        15293365.0 / 508032.0
        - 405.0 * chi2 / 8.0
        - 125440.0 * eta
        + 75066.0 * etachi
        + 13382.0 * etachi2
        + 873540.0 * eta2
        - 165730.0 * eta2chi
        - 1693600.0 * eta3
    )
    psi6 = (
        -889770.0 * eta
        + 631020.0 * etachi
        + 50676.0 * etachi2
        + 5980800.0 * eta2
        - 1414800.0 * eta2chi
        - 11280000.0 * eta3
    )
    psi7 = (
        869600.0 * eta
        - 670980.0 * etachi
        - 30082.0 * etachi2
        - 5837900.0 * eta2
        + 1514500.0 * eta2chi
        + 10891000.0 * eta3
    )
    psi8 = (
        -366000.0 * eta
        + 306700.0 * etachi
        + 631.76 * etachi2
        + 2426500.0 * eta2
        - 721800.0 * eta2chi
        - 4552400.0 * eta3
    )
    f_merger = (
        1.0
        - 4.4547 * pow(one_minus_chi, 0.217)
        + 3.521 * pow(one_minus_chi, 0.26)
        + 0.64365 * eta
        + 0.82696 * etachi
        - 0.27063 * etachi2
        - 0.058218 * eta2
        - 3.9346 * eta2chi
        - 7.0916 * eta3
    )
    f_ring = (
        (1.0 - 0.63 * pow(one_minus_chi, 0.3)) / 2.0
        + 0.14690 * eta
        - 0.12281 * etachi
        - 0.026091 * etachi2
        - 0.024900 * eta2
        + 0.17013 * eta2chi
        + 2.3252 * eta3
    )
    sigma = (
        (1.0 - 0.63 * pow(one_minus_chi, 0.3))
        * pow(one_minus_chi, 0.45)
        / 4.0
        - 0.40979 * eta
        - 0.035226 * etachi
        + 0.10082 * etachi2
        + 1.8286 * eta2
        - 0.020169 * eta2chi
        - 2.8698 * eta3
    )
    f_cut = (
        0.32361
        + 0.048935 * chi
        + 0.013463 * chi2
        - 0.13313 * eta
        - 0.081719 * etachi
        + 0.14512 * etachi2
        - 0.27140 * eta2
        + 0.12788 * eta2chi
        + 4.9220 * eta3
    )
    return _Coefficients(
        f_cut=f_cut / pi_m,
        f_merger=f_merger / pi_m,
        f_ring=f_ring / pi_m,
        sigma=sigma / pi_m,
        psi0=3.0 / (128.0 * eta),
        psi2=psi2,
        psi3=psi3,
        psi4=psi4,
        psi6=psi6,
        psi7=psi7,
        psi8=psi8,
    )


def _coefficients(inputs: _Inputs) -> _Coefficients:
    if inputs.approximant == "IMRPhenomA":
        return _phenoma_coefficients(inputs)
    return _phenomb_coefficients(inputs)


def _amplitude_prefactor(inputs: _Inputs, coefficients: _Coefficients) -> float:
    return (
        -pow(_MTSUN_SI * inputs.total_mass, 5.0 / 6.0)
        * pow(coefficients.f_merger, -7.0 / 6.0)
        / pow(_PI, 2.0 / 3.0)
        * math.sqrt(5.0 * inputs.eta / 24.0)
        / (inputs.distance_m / _C_SI)
    )


def _lorentzian(frequencies, coefficients):
    sigma = coefficients.sigma
    return sigma / (
        2.0
        * _PI
        * (
            (frequencies - coefficients.f_ring) ** 2
            + sigma * sigma / 4.0
        )
    )


def _phenoma_samples(inputs, coefficients, frequencies):
    f_merger = coefficients.f_merger
    f_ring = coefficients.f_ring
    amp0 = _amplitude_prefactor(inputs, coefficients)
    normalized = frequencies / f_merger
    inspiral = amp0 * torch.pow(normalized, -7.0 / 6.0)
    merger = amp0 * torch.pow(normalized, -2.0 / 3.0)
    ringdown = (
        amp0
        * (_PI / 2.0)
        * pow(f_ring / f_merger, -2.0 / 3.0)
        * coefficients.sigma
        * _lorentzian(frequencies, coefficients)
    )
    amplitude = torch.where(
        frequencies <= f_merger,
        inspiral,
        torch.where(frequencies <= f_ring, merger, ringdown),
    )
    cbrt_f = torch.pow(frequencies, 1.0 / 3.0)
    phase = (
        -2.0 * inputs.coa_phase
        + coefficients.psi0 / (frequencies * frequencies) * cbrt_f
        + coefficients.psi1 / (frequencies * cbrt_f)
        + coefficients.psi2 / frequencies
        + coefficients.psi3 / frequencies * cbrt_f
        + coefficients.psi4 / cbrt_f
        + coefficients.psi5
        + coefficients.psi6 * cbrt_f
        + coefficients.psi7 * frequencies / cbrt_f
    )
    return torch.complex(amplitude * torch.cos(phase), -amplitude * torch.sin(phase))


def _phenomb_samples(inputs, coefficients, frequencies):
    eta = inputs.eta
    chi = inputs.chi
    pi_m = _PI * inputs.total_mass * _MTSUN_SI
    f_merger = coefficients.f_merger
    f_ring = coefficients.f_ring
    alpha2 = -323.0 / 224.0 + 451.0 * eta / 168.0
    alpha3 = (27.0 / 8.0 - 11.0 * eta / 6.0) * chi
    epsilon1 = 1.4547 * chi - 1.8897
    epsilon2 = -1.8153 * chi + 1.6557
    v_merger = pow(pi_m * f_merger, 1.0 / 3.0)
    v_ring = pow(pi_m * f_ring, 1.0 / 3.0)
    w1 = (
        1.0 + alpha2 * v_merger * v_merger + alpha3 * pi_m * f_merger
    ) / (1.0 + epsilon1 * v_merger + epsilon2 * v_merger * v_merger)
    w2 = (
        w1
        * (_PI * coefficients.sigma / 2.0)
        * pow(f_ring / f_merger, -2.0 / 3.0)
        * (1.0 + epsilon1 * v_ring + epsilon2 * v_ring * v_ring)
    )

    v = torch.pow(pi_m * frequencies, 1.0 / 3.0)
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v4 * v
    v6 = v3 * v3
    v7 = v6 * v
    v8 = v7 * v
    inspiral = torch.pow(frequencies / f_merger, -7.0 / 6.0) * (
        1.0 + alpha2 * v2 + alpha3 * v3
    )
    merger = w1 * torch.pow(frequencies / f_merger, -2.0 / 3.0) * (
        1.0 + epsilon1 * v + epsilon2 * v2
    )
    ringdown = w2 * _lorentzian(frequencies, coefficients)
    amplitude = torch.where(
        frequencies <= f_merger,
        inspiral,
        torch.where(frequencies <= f_ring, merger, ringdown),
    )
    phase = -2.0 * inputs.coa_phase + coefficients.psi0 / v5 * (
        1.0
        + coefficients.psi2 * v2
        + coefficients.psi3 * v3
        + coefficients.psi4 * v4
        + coefficients.psi5 * v5
        + coefficients.psi6 * v6
        + coefficients.psi7 * v7
        + coefficients.psi8 * v8
    )
    amplitude = _amplitude_prefactor(inputs, coefficients) * amplitude
    return torch.complex(amplitude * torch.cos(phase), -amplitude * torch.sin(phase))


def _model_samples(inputs, coefficients, frequencies):
    if inputs.approximant == "IMRPhenomA":
        return _phenoma_samples(inputs, coefficients, frequencies)
    return _phenomb_samples(inputs, coefficients, frequencies)


def _polarizations(inputs, samples):
    cosi = math.cos(inputs.inclination)
    plus0 = 0.5 * (1.0 + cosi * cosi) * samples
    cross0 = samples * complex(0.0, -cosi)
    cos_nodes = math.cos(2.0 * inputs.long_asc_nodes)
    sin_nodes = math.sin(2.0 * inputs.long_asc_nodes)
    return (
        (cos_nodes * plus0 + sin_nodes * cross0).to(inputs.complex_dtype),
        (cos_nodes * cross0 - sin_nodes * plus0).to(inputs.complex_dtype),
    )


def _next_power_of_two_layout(f_max, delta_f):
    # NextPow2 receives size_t in LAL, so f_max / delta_f is truncated first.
    layout_bins = int(f_max / delta_f)
    if layout_bins < 1:
        raise ValueError("IMRPhenomA/B f_final must span at least one bin")
    fft_length = (
        1 if layout_bins == 1 else 1 << (layout_bins - 1).bit_length()
    )
    return fft_length + 1


def imrphenomab_fd_torch(**params):
    """Generate a regular-grid Torch-native IMRPhenomA/B waveform."""
    inputs = _validated_inputs(params)
    coefficients = _coefficients(inputs)
    delta_f = float(params["delta_f"])
    f_lower = float(params["f_lower"])
    f_final = float(params.get("f_final", 0.0))
    if not all(math.isfinite(value) for value in (delta_f, f_lower, f_final)):
        raise ValueError("IMRPhenomA/B frequencies must be finite")
    if delta_f <= 0.0 or f_lower <= 0.0:
        raise ValueError("IMRPhenomA/B delta_f and f_lower must be positive")
    if f_final < 0.0:
        raise ValueError("IMRPhenomA/B f_final must be non-negative")
    if coefficients.f_cut <= f_lower:
        raise ValueError("IMRPhenomA/B f_cut is <= f_lower")
    f_max = f_final if f_final > 0.0 else coefficients.f_cut
    if f_max <= f_lower:
        raise ValueError("IMRPhenomA/B f_final is <= f_lower")

    nfreq = _next_power_of_two_layout(f_max, delta_f)
    bins = torch.arange(nfreq, device=inputs.device)
    frequencies = bins.to(inputs.real_dtype) * delta_f
    if inputs.approximant == "IMRPhenomA":
        # PhenomA checks physical frequency bounds and leaves DC/Nyquist zero.
        active = (
            (bins > 0)
            & (bins < nfreq - 1)
            & (frequencies >= f_lower)
            & (frequencies <= f_max)
        )
    else:
        # PhenomB converts both bounds to size_t and uses a half-open loop.
        first_bin = int(f_lower / delta_f)
        stop_bin = int(f_max / delta_f)
        if first_bin < 1:
            raise ValueError("IMRPhenomB f_lower must be at least delta_f")
        active = (bins >= first_bin) & (bins < stop_bin)
    if not bool(torch.any(active)):
        raise ValueError("IMRPhenomA/B frequency interval has no active bins")

    plus = torch.zeros(nfreq, dtype=inputs.complex_dtype, device=inputs.device)
    cross = torch.zeros_like(plus)
    active_frequencies = frequencies[active]
    samples = _model_samples(inputs, coefficients, active_frequencies)
    plus_segment, cross_segment = _polarizations(inputs, samples)
    plus[active] = plus_segment
    cross[active] = cross_segment
    return (
        FrequencySeries(
            TorchArrayData(plus), delta_f=delta_f, epoch=0.0, copy=False
        ),
        FrequencySeries(
            TorchArrayData(cross), delta_f=delta_f, epoch=0.0, copy=False
        ),
    )


def _sequence_frequencies(sample_points, inputs):
    values = getattr(sample_points, "_data", sample_points)
    if isinstance(values, TorchArrayData):
        values = values.tensor
    frequencies = torch.as_tensor(
        values,
        dtype=inputs.real_dtype,
        device=inputs.device,
    )
    if frequencies.ndim != 1 or frequencies.numel() == 0:
        raise ValueError("IMRPhenomA/B sample_points must be a non-empty vector")
    if not bool(torch.all(torch.isfinite(frequencies))):
        raise ValueError("IMRPhenomA/B sample_points must be finite")
    if bool(torch.any(frequencies <= 0.0)):
        raise ValueError("IMRPhenomA/B sample_points must be positive")
    return frequencies


def imrphenomab_fd_sequence_torch(**params):
    """Evaluate IMRPhenomA/B at arbitrary frequencies on the Torch device."""
    inputs = _validated_inputs(params, sequence=True)
    coefficients = _coefficients(inputs)
    frequencies = _sequence_frequencies(params["sample_points"], inputs)
    plus = torch.zeros(
        frequencies.shape, dtype=inputs.complex_dtype, device=inputs.device
    )
    cross = torch.zeros_like(plus)
    active = frequencies < coefficients.f_cut
    if bool(torch.any(active)):
        samples = _model_samples(inputs, coefficients, frequencies[active])
        plus_segment, cross_segment = _polarizations(inputs, samples)
        plus[active] = plus_segment
        cross[active] = cross_segment
    return (
        PyCBCArray(TorchArrayData(plus), copy=False),
        PyCBCArray(TorchArrayData(cross), copy=False),
    )


__all__ = [
    "imrphenomab_default_native_supported",
    "imrphenomab_fd_sequence_torch",
    "imrphenomab_fd_torch",
    "imrphenomab_native_supported",
    "imrphenomab_sequence_native_supported",
]
