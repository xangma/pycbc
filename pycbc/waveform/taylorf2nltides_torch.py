# Copyright (C) 2026
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.

"""Native Torch implementation of the LAL ``TaylorF2NLTides`` model."""

import math

import lal

from .taylorf2_torch import (
    _as_order,
    _taylorf2_inputs,
    _taylorf2_polarizations,
    _taylorf2_samples,
    taylorf2_native_supported,
)


_NL_TIDE_KEYS = (
    "nl_tides_a1",
    "nl_tides_n1",
    "nl_tides_f1",
    "nl_tides_a2",
    "nl_tides_n2",
    "nl_tides_f2",
)
_TIDAL_ORDERS = frozenset((-1, 0, 10, 12))


def _base_params(params):
    """Return parameters with LAL's NLTides default tidal order expanded."""
    base = dict(params)
    if _as_order(base.get("tidal_order", -1)) == -1:
        # Unlike TaylorF2, TaylorF2NLTides' ALL setting stops at 6PN.
        base["tidal_order"] = 12
    return base


def taylorf2nltides_native_supported(params):
    """Return whether ``params`` are covered by the native Torch generator."""
    if _as_order(params.get("tidal_order", -1)) not in _TIDAL_ORDERS:
        return False
    if not taylorf2_native_supported(_base_params(params)):
        return False

    values = []
    for key in _NL_TIDE_KEYS:
        value = params.get(key)
        if value is None:
            return False
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(value):
            return False
        values.append(value)
    return values[2] > 0.0 and values[5] > 0.0


def _body_phase(frequencies, amplitude, spectral_index, turn_on, mass_fraction):
    """Evaluate one body's nonlinear-tide phase contribution."""
    import torch

    exponent = spectral_index - 3.0
    chirp_factor = mass_fraction ** (2.0 / 3.0) * amplitude
    if spectral_index == 3.0:
        phase = chirp_factor * (torch.log(frequencies) - math.log(turn_on))
    else:
        coefficient = chirp_factor * 100.0 ** (-exponent) / exponent
        phase = coefficient * (
            torch.pow(frequencies, exponent) - turn_on**exponent
        )
    return torch.where(frequencies >= turn_on, phase, 0.0)


def _nonlinear_tide_phase(frequencies, inputs, params):
    """Mirror ``XLALSimInspiralTaylorF2NLPhase`` on a Torch device."""
    total_mass = inputs.mass1 + inputs.mass2
    chirp_mass = (
        (inputs.mass1 * inputs.mass2) ** 0.6 / total_mass**0.2
    )
    coefficient = 50.0 * 2.0 ** (2.0 / 3.0) / 3072.0
    coefficient *= (
        1.0 / (chirp_mass * lal.MTSUN_SI * 100.0 * math.pi)
    ) ** (10.0 / 3.0)

    phase1 = _body_phase(
        frequencies,
        float(params["nl_tides_a1"]),
        float(params["nl_tides_n1"]),
        float(params["nl_tides_f1"]),
        inputs.mass1 / total_mass,
    )
    phase2 = _body_phase(
        frequencies,
        float(params["nl_tides_a2"]),
        float(params["nl_tides_n2"]),
        float(params["nl_tides_f2"]),
        inputs.mass2 / total_mass,
    )
    return coefficient * (phase1 + phase2)


def taylorf2nltides_fd_torch(**params):
    """Generate regular-grid ``TaylorF2NLTides`` with native Torch math."""
    import torch

    from pycbc.types import FrequencySeries
    from pycbc.types.array_torch import TorchArrayData

    if not taylorf2nltides_native_supported(params):
        raise ValueError(
            "TaylorF2NLTides parameters are not supported by the native "
            "Torch path"
        )

    inputs = _taylorf2_inputs(
        _base_params(params),
        infer_tidal_quadrupoles=False,
    )
    delta_f = float(params["delta_f"])
    f_lower = float(params["f_lower"])
    f_final = float(params.get("f_final", 0.0))
    if not all(math.isfinite(value) for value in (delta_f, f_lower, f_final)):
        raise ValueError("TaylorF2NLTides frequencies must be finite")
    if delta_f <= 0.0 or f_lower <= 0.0:
        raise ValueError(
            "TaylorF2NLTides delta_f and f_lower must be positive"
        )

    pi_mass = math.pi * (inputs.mass1 + inputs.mass2) * lal.MTSUN_SI
    f_max = f_final or 1.0 / (6.0**1.5 * pi_mass)
    if f_max <= f_lower:
        raise ValueError(
            "TaylorF2NLTides ending frequency must exceed f_lower"
        )

    length = int(f_max / delta_f + 1.0)
    first_bin = int(math.ceil(f_lower / delta_f))
    if first_bin >= length:
        raise ValueError(
            "TaylorF2NLTides frequency range contains no sampled bins"
        )

    raw = torch.zeros(
        length,
        dtype=inputs.complex_dtype,
        device=inputs.device,
    )
    frequencies = (
        torch.arange(
            first_bin,
            length,
            dtype=inputs.real_dtype,
            device=inputs.device,
        )
        * delta_f
    )
    epoch = -1.0 / delta_f
    samples = _taylorf2_samples(inputs, frequencies, time_shift=epoch)
    nonlinear_phase = _nonlinear_tide_phase(frequencies, inputs, params)
    correction = torch.polar(
        torch.ones_like(nonlinear_phase),
        -nonlinear_phase,
    ).to(inputs.complex_dtype)
    raw[first_bin:] = samples * correction

    plus, cross = _taylorf2_polarizations(raw, inputs)
    return (
        FrequencySeries(
            TorchArrayData(plus),
            delta_f=delta_f,
            epoch=epoch,
            copy=False,
        ),
        FrequencySeries(
            TorchArrayData(cross),
            delta_f=delta_f,
            epoch=epoch,
            copy=False,
        ),
    )


__all__ = (
    "taylorf2nltides_fd_torch",
    "taylorf2nltides_native_supported",
)
