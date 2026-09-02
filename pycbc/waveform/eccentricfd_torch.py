# Copyright (C) 2026
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

"""Torch-native implementation of LALSuite's ``EccentricFD`` model.

Scalar analytic coefficients are assembled in Python.  Frequency-grid
construction, 3.5PN phasing, ten-harmonic amplitude assembly, cutoff masking,
and both polarizations are evaluated on the active Torch device.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pycbc import lal_compat as lal
import torch

import pycbc.scheme as _scheme
from pycbc.types import Array, FrequencySeries
from pycbc.types.array_torch import TorchArrayData
from pycbc.waveform.eccentricfd_coefficients import (
    EccentricFDCoefficients,
    eccentricfd_coefficients,
)
from pycbc.waveform.imrphenomd_torch import (
    _NON_GR_KEYS,
    _TIDAL_EXTENSION_KEYS,
    _is_default_order,
    _is_nonzero,
)

_TRANSVERSE_SPIN_KEYS = ("spin1x", "spin1y", "spin2x", "spin2y")
_IGNORED_ORDER_KEYS = (
    "amplitude_order",
    "spin_order",
    "tidal_order",
    "eccentricity_order",
)
_PUBLIC_SCALARS = (
    "mass1",
    "mass2",
    "spin1x",
    "spin1y",
    "spin1z",
    "spin2x",
    "spin2y",
    "spin2z",
    "distance",
    "inclination",
    "coa_phase",
    "long_asc_nodes",
    "eccentricity",
    "mean_per_ano",
    "delta_f",
    "f_lower",
    "f_final",
    "f_ref",
)


def _phase_order(value):
    """Mirror the integer conversion performed by PyCBC's LAL wrapper."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _eccentricfd_native_supported(params, *, sequence: bool) -> bool:
    """Return whether the requested native interface can honor ``params``.

    ``EccentricFD`` itself ignores aligned spins, ``f_ref``, and mean anomaly;
    the native path preserves that behavior.  Options rejected by the LAL
    wrapper or unrelated extensions stay on the reference fallback path.
    """
    if params.get("approximant") != "EccentricFD":
        return False
    if _phase_order(params.get("phase_order", -1)) not in (-1, 7):
        return False
    if any(not _is_default_order(params.get(key, -1)) for key in _IGNORED_ORDER_KEYS):
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
                "frame_axis",
                "modes_choice",
                "side_bands",
            )
        )
    ):
        return False
    if params.get("mode_array") is not None or params.get("numrel_data", ""):
        return False

    try:
        values = {
            key: float(params.get(key, 1.0 if key == "distance" else 0.0))
            for key in _PUBLIC_SCALARS
        }
    except (TypeError, ValueError, OverflowError):
        return False
    if not all(math.isfinite(value) for value in values.values()):
        return False
    if not (
        values["mass1"] > 0.0
        and values["mass2"] > 0.0
        and values["distance"] > 0.0
    ):
        return False
    if sequence:
        return True
    return (
        values["delta_f"] > 0.0
        and values["f_lower"] > 0.0
        and values["f_final"] >= 0.0
    )


def eccentricfd_native_supported(params) -> bool:
    """Return whether regular-grid ``EccentricFD`` is native."""
    return _eccentricfd_native_supported(params, sequence=False)


def eccentricfd_sequence_native_supported(params) -> bool:
    """Return whether arbitrary-frequency ``EccentricFD`` is native."""
    return _eccentricfd_native_supported(params, sequence=True)


def eccentricfd_default_native_supported(_params) -> bool:
    """Return whether unflagged native use is accurate on this device.

    Apple MPS evaluates the model in float32. The large, cancellation-prone
    eccentric phase can then lose waveform-level accuracy, so MPS retains the
    LAL fallback unless native execution is requested explicitly.
    """
    state = _scheme.mgr.state
    return not (
        isinstance(state, _scheme.TorchScheme)
        and state.torch_device.type == "mps"
    )


@dataclass(frozen=True)
class _Inputs:
    """Validated scalar inputs and active Torch configuration."""

    mass1: float
    mass2: float
    distance_m: float
    inclination: float
    coa_phase: float
    long_asc_nodes: float
    eccentricity: float
    delta_f: float
    f_lower: float
    f_final: float
    device: torch.device
    real_dtype: torch.dtype
    complex_dtype: torch.dtype


def _validated_inputs(params, *, sequence=False) -> _Inputs:
    supported = (
        eccentricfd_sequence_native_supported(params)
        if sequence
        else eccentricfd_native_supported(params)
    )
    if not supported:
        raise ValueError(
            "EccentricFD parameters are not supported by the native Torch path"
        )
    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise RuntimeError("native Torch EccentricFD requires TorchScheme")
    device = state.torch_device
    real_dtype = torch.float32 if device.type == "mps" else torch.float64
    complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
    return _Inputs(
        mass1=float(params["mass1"]),
        mass2=float(params["mass2"]),
        distance_m=float(params.get("distance", 1.0)) * 1.0e6 * lal.PC_SI,
        inclination=float(params.get("inclination", 0.0)),
        coa_phase=float(params.get("coa_phase", 0.0)),
        long_asc_nodes=float(params.get("long_asc_nodes", 0.0)),
        eccentricity=float(params.get("eccentricity", 0.0)),
        delta_f=float(params.get("delta_f", -1.0)),
        f_lower=float(params.get("f_lower", -1.0)),
        f_final=float(params.get("f_final", 0.0)),
        device=device,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
    )


def _frequency_powers(frequencies):
    """Return the powers shared by LAL's phase and harmonic fits."""
    k4 = frequencies * frequencies * frequencies * frequencies
    k5 = k4 * frequencies
    k19 = k5 * k5 * k5 * k4
    k19_over_3 = torch.pow(k19, 1.0 / 3.0)
    k19_over_9 = torch.pow(k19_over_3, 1.0 / 3.0)
    f8 = torch.rsqrt(k19_over_9)
    f1 = f8 * f8
    f2 = f1 * f1 * f1
    f3 = f1 * f1
    f4 = f3 * f3
    f5 = f2 * f8
    f6 = f3 * f8
    f7 = f1 * f8
    return f1, f2, f3, f4, f5, f6, f7, f8


def _phase_accumulation(frequencies, coefficients, powers):
    """Evaluate and sum the seven PN phase contributions."""
    f1, f2, f3, f4, _, _, _, _ = powers
    a = coefficients
    lo1 = a[1] * frequencies
    lo4 = torch.pow(lo1, 1.0 / 3.0)
    lo3 = lo4 * lo4
    lo2 = lo3 * lo3 * lo4
    log_factor = 6.0 * math.sqrt(6.0)

    phase = (a[0] / lo2) * (1.0 + a[2] * f1 + a[3] * f2 + a[4] * f3 + a[5] * f4)
    phase = phase + (a[0] / lo1) * (
        a[6] + a[7] * f1 + a[8] * f2 + a[9] * f3 + a[10] * f4
    )
    phase = phase + (a[0] / lo3) * (
        a[11] + a[12] * f1 + a[13] * f2 + a[14] * f3 + a[15] * f4
    )
    phase = phase + (a[0] / lo4) * (
        a[16] + a[17] * f1 + a[18] * f2 + a[19] * f3 + a[20] * f4
    )
    phase = phase + a[0] * (
        a[21]
        + a[21] * torch.log(log_factor * lo1)
        + a[22] * f1
        + a[23] * f2
        + a[24] * f3
        + a[25] * f4
    )
    phase = phase + a[0] * lo4 * (
        a[26]
        + torch.log(4.0 * lo4)
        * (a[27] + a[28] * f1 + a[29] * f2 + a[30] * f3 + a[31] * f4)
        + a[32] * f1
        + a[33] * f2
        + a[34] * f3
        + a[35] * f4
    )
    phase = phase + a[0] * lo3 * (
        a[36] + a[37] * f1 + a[38] * f2 + a[39] * f3 + a[40] * f4
    )
    return -phase


def _zeta_component(harmonic, coefficients, powers):
    """Evaluate one real or imaginary polarization coefficient series."""
    f1, f2, f3, f4, f5, f6, f7, f8 = powers
    c = coefficients
    if harmonic == 1:
        return c[0] * f5 + c[1] * f6 + c[2] * f7 + c[3] * f8
    if harmonic == 2:
        return c[4] + c[5] * f1 + c[6] * f2 + c[7] * f3 + c[8] * f4
    if harmonic == 3:
        return c[9] * f5 + c[10] * f6 + c[11] * f7 + c[12] * f8
    if harmonic == 4:
        return c[13] * f1 + c[14] * f2 + c[15] * f3 + c[16] * f4
    if harmonic == 5:
        return c[17] * f5 + c[18] * f6 + c[19] * f7
    if harmonic == 6:
        return c[20] * f2 + c[21] * f3 + c[22] * f4
    if harmonic == 7:
        return c[23] * f5 + c[24] * f6
    if harmonic == 8:
        return c[25] * f2 + c[26] * f4
    if harmonic == 9:
        return c[27] * f5
    return c[28] * f4


def _coefficient_tensors(coefficients, inputs):
    """Copy the small scalar coefficient tables to the active device."""
    kwargs = {"dtype": inputs.real_dtype, "device": inputs.device}
    return (
        torch.as_tensor(coefficients.phase, **kwargs),
        torch.as_tensor(coefficients.zeta_real_plus, **kwargs),
        torch.as_tensor(coefficients.zeta_real_cross, **kwargs),
        torch.as_tensor(coefficients.zeta_imag_plus, **kwargs),
        torch.as_tensor(coefficients.zeta_imag_cross, **kwargs),
    )


def _eccentricfd_samples(
    inputs: _Inputs,
    frequencies,
    f_isco: float,
    epoch: float,
    coefficients: EccentricFDCoefficients,
):
    """Evaluate both polarizations at positive device frequencies."""
    phase_coeff, real_plus, real_cross, imag_plus, imag_cross = _coefficient_tensors(
        coefficients, inputs
    )
    powers = _frequency_powers(frequencies)
    phase_order = _phase_accumulation(frequencies, phase_coeff, powers)
    plus = torch.zeros_like(frequencies, dtype=inputs.complex_dtype)
    cross = torch.zeros_like(frequencies, dtype=inputs.complex_dtype)

    for harmonic in range(1, 11):
        harmonic_float = float(harmonic)
        phase = (
            math.pi / 4.0
            + (harmonic_float / 2.0) ** (8.0 / 3.0) * phase_order
            - 2.0 * math.pi * epoch * frequencies
            + harmonic_float * inputs.coa_phase
        )
        rotation = torch.complex(torch.cos(phase), torch.sin(phase)).to(
            inputs.complex_dtype
        )
        plus_zeta = torch.complex(
            _zeta_component(harmonic, real_plus, powers),
            _zeta_component(harmonic, imag_plus, powers),
        ).to(inputs.complex_dtype)
        cross_zeta = torch.complex(
            _zeta_component(harmonic, real_cross, powers),
            _zeta_component(harmonic, imag_cross, powers),
        ).to(inputs.complex_dtype)
        scale = (harmonic_float / 2.0) ** (2.0 / 3.0)
        active = (frequencies <= harmonic_float * f_isco).to(inputs.real_dtype)
        plus = plus + scale * active * plus_zeta * rotation
        cross = cross + scale * active * cross_zeta * rotation

    total_mass_seconds = (inputs.mass1 + inputs.mass2) * lal.MTSUN_SI
    eta = inputs.mass1 * inputs.mass2 / (inputs.mass1 + inputs.mass2) ** 2
    chirp_mass_seconds = eta ** (3.0 / 5.0) * total_mass_seconds
    amplitude0 = (
        -math.sqrt(5.0 / 384.0)
        * math.pi ** (-2.0 / 3.0)
        * chirp_mass_seconds ** (5.0 / 6.0)
        / inputs.distance_m
        * lal.MRSUN_SI
        / lal.MTSUN_SI
    )
    amplitude = amplitude0 * torch.pow(frequencies, -7.0 / 6.0)
    return amplitude * plus, amplitude * cross


def _sequence_frequencies(sample_points, inputs):
    """Validate arbitrary frequencies and place them on the active device."""
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
            "EccentricFD sample_points must be a non-empty vector"
        )
    if not bool(torch.all(torch.isfinite(frequencies))):
        raise ValueError("EccentricFD sample_points must be finite")
    if bool(torch.any(frequencies <= 0.0)):
        raise ValueError("EccentricFD sample_points must be positive")
    return frequencies


def _rotate_polarizations(plus, cross, long_asc_nodes):
    """Apply the legacy generator's final polarization-basis rotation."""
    if not long_asc_nodes:
        return plus, cross
    cosine = math.cos(2.0 * long_asc_nodes)
    sine = math.sin(2.0 * long_asc_nodes)
    return (
        cosine * plus + sine * cross,
        cosine * cross - sine * plus,
    )


def eccentricfd_fd_sequence_torch(**params):
    """Evaluate ``EccentricFD`` at arbitrary frequencies with Torch.

    LAL does not expose this model through its sequence API.  The minimum
    requested frequency supplies the regular model's ``f_lower`` coefficient
    convention, independent of sample ordering.  Frequency-dependent work
    remains on-device; one scalar is read back for the Python coefficient
    setup.
    """
    inputs = _validated_inputs(params, sequence=True)
    frequencies = _sequence_frequencies(params["sample_points"], inputs)
    total_mass_seconds = (inputs.mass1 + inputs.mass2) * lal.MTSUN_SI
    eta = inputs.mass1 * inputs.mass2 / (inputs.mass1 + inputs.mass2) ** 2
    f_isco = 1.0 / (6.0**1.5 * math.pi * total_mass_seconds)
    f_lower = float(torch.min(frequencies).item())
    coefficients = eccentricfd_coefficients(
        total_mass_seconds,
        eta,
        inputs.eccentricity,
        f_lower,
        inputs.inclination,
        inputs.long_asc_nodes,
    )
    plus, cross = _eccentricfd_samples(
        inputs,
        frequencies,
        f_isco,
        0.0,
        coefficients,
    )
    plus, cross = _rotate_polarizations(
        plus,
        cross,
        inputs.long_asc_nodes,
    )
    return (
        Array(TorchArrayData(plus), copy=False),
        Array(TorchArrayData(cross), copy=False),
    )


def eccentricfd_fd_torch(**params):
    """Generate regular-grid ``EccentricFD`` polarizations on Torch."""
    inputs = _validated_inputs(params)
    total_mass_seconds = (inputs.mass1 + inputs.mass2) * lal.MTSUN_SI
    eta = inputs.mass1 * inputs.mass2 / (inputs.mass1 + inputs.mass2) ** 2
    f_isco = 1.0 / (6.0**1.5 * math.pi * total_mass_seconds)
    f_max = f_isco if inputs.f_final == 0.0 else inputs.f_final
    length = int(f_max / inputs.delta_f + 1.0)
    first_bin = int(math.ceil(inputs.f_lower / inputs.delta_f))
    epoch = float(lal.LIGOTimeGPS(-1.0 / inputs.delta_f))
    plus = torch.zeros(length, dtype=inputs.complex_dtype, device=inputs.device)
    cross = torch.zeros_like(plus)

    if first_bin < length:
        frequencies = (
            torch.arange(
                first_bin,
                length,
                dtype=inputs.real_dtype,
                device=inputs.device,
            )
            * inputs.delta_f
        )
        coefficients = eccentricfd_coefficients(
            total_mass_seconds,
            eta,
            inputs.eccentricity,
            inputs.f_lower,
            inputs.inclination,
            inputs.long_asc_nodes,
        )
        plus[first_bin:], cross[first_bin:] = _eccentricfd_samples(
            inputs,
            frequencies,
            f_isco,
            epoch,
            coefficients,
        )

    # The legacy FD generator applies this polarization-basis rotation after
    # EccentricFD has already used long_asc_nodes as its inclination azimuth.
    plus, cross = _rotate_polarizations(
        plus,
        cross,
        inputs.long_asc_nodes,
    )

    return (
        FrequencySeries(
            TorchArrayData(plus),
            delta_f=inputs.delta_f,
            epoch=epoch,
            copy=False,
        ),
        FrequencySeries(
            TorchArrayData(cross),
            delta_f=inputs.delta_f,
            epoch=epoch,
            copy=False,
        ),
    )
