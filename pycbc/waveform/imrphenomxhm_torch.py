# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch-native IMRPhenomXHM modes and polarizations.

The quadrupole shares the IMRPhenomXAS implementation.  The ``(2, +/-1)``,
``(3, +/-3)``, and ``(4, +/-4)`` modes use native XHM no-mixing kernels, while
``(3, +/-2)`` includes the native spheroidal-to-spherical ringdown mixing.  The
default and explicit mode sets can be returned directly or assembled into plus
and cross polarizations on the active Torch device.
"""

import math
from numbers import Integral
from typing import NamedTuple

import torch

from pycbc import scheme as _scheme
from pycbc.types import Array as PyCBCArray
from pycbc.types.array_torch import TorchArrayData

from ._spherical_harmonics_torch import spin_weighted_spherical_harmonic
from .imrphenomxas_torch import (
    _XAS_MODE_POLARIZATION_FACTOR,
    _imrphenomxas_core_torch,
    _imrphenomxas_sequence_samples,
    _series_from_active_samples,
    imrphenomxas_native_supported,
)
from .imrphenomxhm_mode21_torch import imrphenomxhm_h2m1_samples
from .imrphenomxhm_mode32_torch import imrphenomxhm_h3m2_samples
from .imrphenomxhm_mode33_torch import imrphenomxhm_h3m3_samples
from .imrphenomxhm_mode44_torch import imrphenomxhm_h4m4_samples


_NATIVE_MODES = frozenset(
    {
        (2, -2),
        (2, -1),
        (2, 1),
        (2, 2),
        (3, -3),
        (3, -2),
        (3, 2),
        (3, 3),
        (4, -4),
        (4, 4),
    }
)

# Keep this order in sync with waveform_modes.default_modes.  Defining it here
# avoids importing waveform_modes from its native dispatch target.
_DEFAULT_MODES = (
    (2, 2),
    (2, 1),
    (3, 3),
    (3, 2),
    (4, 4),
    (2, -2),
    (2, -1),
    (3, -3),
    (3, -2),
    (4, -4),
)


class _SequenceCore(NamedTuple):
    """Inclination-independent samples shared by the XHM mode kernels."""

    polarization: torch.Tensor


def _requested_modes(params):
    mode_array = params.get("mode_array")
    if mode_array is None:
        return list(_DEFAULT_MODES)

    modes = []
    try:
        for mode in mode_array:
            ell, emm = mode
            if not isinstance(ell, Integral) or not isinstance(emm, Integral):
                return None
            modes.append((int(ell), int(emm)))
    except (TypeError, ValueError):
        return None
    return modes


def _xas_params(params):
    xas = dict(params)
    xas["approximant"] = "IMRPhenomXAS"
    xas["mode_array"] = None
    # These angles belong to polarization assembly and are intentionally
    # ignored by the mode-by-mode interface.
    xas["inclination"] = 0.0
    xas["long_asc_nodes"] = 0.0
    return xas


def imrphenomxhm_modes_native_supported(params):
    """Return whether the requested XHM modes have a native implementation."""

    if params.get("approximant") != "IMRPhenomXHM":
        return False
    modes = _requested_modes(params)
    if modes is None or any(mode not in _NATIVE_MODES for mode in modes):
        return False
    return imrphenomxas_native_supported(_xas_params(params))


def imrphenomxhm_fd_native_supported(params):
    """Return whether native polarization generation covers the request."""

    modes = _requested_modes(params)
    return bool(modes) and imrphenomxhm_modes_native_supported(params)


def imrphenomxhm_sequence_native_supported(params):
    """Return whether arbitrary-frequency XHM generation is native."""

    return imrphenomxhm_modes_native_supported(params)


def _active_mode_samples(
    core,
    params,
    modes,
    *,
    frequencies=None,
    reference_frequency=None,
):
    """Generate each requested absolute-m mode family once."""

    active_modes = {}
    mode_families = {(ell, abs(emm)) for ell, emm in modes}
    if (2, 2) in mode_families:
        active_modes[2, 2] = core.polarization / _XAS_MODE_POLARIZATION_FACTOR
    if (2, 1) in mode_families:
        active_modes[2, 1] = imrphenomxhm_h2m1_samples(
            core,
            params,
            frequencies=frequencies,
            reference_frequency=reference_frequency,
        )
    if (3, 3) in mode_families:
        active_modes[3, 3] = imrphenomxhm_h3m3_samples(
            core,
            params,
            frequencies=frequencies,
            reference_frequency=reference_frequency,
        )
    if (3, 2) in mode_families:
        active_modes[3, 2] = imrphenomxhm_h3m2_samples(
            core,
            params,
            frequencies=frequencies,
            reference_frequency=reference_frequency,
        )
    if (4, 4) in mode_families:
        active_modes[4, 4] = imrphenomxhm_h4m4_samples(
            core,
            params,
            frequencies=frequencies,
            reference_frequency=reference_frequency,
        )
    return active_modes


def _polarizations_from_active_modes(
    core,
    params,
    modes,
    active_modes,
    *,
    sequence=False,
):
    """Assemble requested mode samples into plus and cross polarizations."""

    plus = core.polarization.new_zeros(core.polarization.shape)
    cross = core.polarization.new_zeros(core.polarization.shape)

    # XHM's aligned-spin convention evaluates the spin-weighted spherical
    # harmonics at phi=pi/2.  The generated positive-frequency waveform is
    # h_l,-m; an explicitly selected +m contribution is reconstructed from
    # equatorial symmetry with the same samples.
    selected = set(modes)
    inclination = float(params.get("inclination", 0.0))
    real_dtype = plus.real.dtype
    device = plus.device
    for (ell, emm), samples in active_modes.items():
        parity = (-1) ** ell
        factor_plus = plus.new_zeros(())
        factor_cross = plus.new_zeros(())
        if (ell, -emm) in selected:
            y_negative = spin_weighted_spherical_harmonic(
                inclination,
                math.pi / 2.0,
                -2,
                ell,
                -emm,
                dtype=real_dtype,
                device=device,
            )
            factor_plus += 0.5 * y_negative
            factor_cross += 0.5j * y_negative
        if (ell, emm) in selected:
            y_positive_conjugate = spin_weighted_spherical_harmonic(
                inclination,
                math.pi / 2.0,
                -2,
                ell,
                emm,
                dtype=real_dtype,
                device=device,
            ).conj()
            factor_plus += 0.5 * parity * y_positive_conjugate
            factor_cross -= 0.5j * parity * y_positive_conjugate
        plus += factor_plus * samples
        cross += factor_cross * samples

    # SimInspiralChooseFDWaveformSequence has no ascending-node argument.
    long_asc_nodes = 0.0 if sequence else float(
        params.get("long_asc_nodes", 0.0)
    )
    cos_nodes = math.cos(2.0 * long_asc_nodes)
    sin_nodes = math.sin(2.0 * long_asc_nodes)
    return (
        cos_nodes * plus + sin_nodes * cross,
        cos_nodes * cross - sin_nodes * plus,
    )


def imrphenomxhm_modes_torch(**params):
    """Generate the requested native XHM modes with Torch."""

    if not imrphenomxhm_modes_native_supported(params):
        raise ValueError(
            "only the default IMRPhenomXHM mode set or explicit (2, +/-1), "
            "(2, +/-2), (3, +/-2), (3, +/-3), and (4, +/-4) requests are "
            "supported by the native Torch path"
        )
    if not isinstance(_scheme.mgr.state, _scheme.TorchScheme):
        raise RuntimeError("native Torch IMRPhenomXHM modes require TorchScheme")

    modes = _requested_modes(params)
    if not modes:
        return {}

    core = _imrphenomxas_core_torch(_xas_params(params))
    active_modes = _active_mode_samples(core, params, modes)

    result = {}
    for ell, emm in modes:
        # LAL exposes the negative-m mode at positive frequencies. Positive-m
        # modes follow h_lm = (-1)^ell conjugate(h_l,-m).
        samples = active_modes[ell, abs(emm)]
        if emm > 0:
            samples = samples.conj()
            if ell % 2:
                samples = -samples
        hlm = _series_from_active_samples(core, samples)
        hplus = 0.5 * hlm
        hcross = (0.5j if emm < 0 else -0.5j) * hlm
        result[ell, emm] = (hplus, hcross)
    return result


def imrphenomxhm_fd_torch(**params):
    """Generate native IMRPhenomXHM plus and cross polarizations with Torch."""

    if not imrphenomxhm_fd_native_supported(params):
        raise ValueError("unsupported parameters for native Torch IMRPhenomXHM")
    if not isinstance(_scheme.mgr.state, _scheme.TorchScheme):
        raise RuntimeError("native Torch IMRPhenomXHM requires TorchScheme")

    modes = _requested_modes(params)
    core = _imrphenomxas_core_torch(_xas_params(params))
    active_modes = _active_mode_samples(core, params, modes)
    plus, cross = _polarizations_from_active_modes(
        core,
        params,
        modes,
        active_modes,
    )
    return (
        _series_from_active_samples(core, plus),
        _series_from_active_samples(core, cross),
    )


def imrphenomxhm_fd_sequence_torch(**params):
    """Evaluate IMRPhenomXHM polarizations at arbitrary frequencies."""

    if not imrphenomxhm_sequence_native_supported(params):
        raise ValueError(
            "IMRPhenomXHM sequence parameters are not supported by the "
            "native Torch path"
        )
    if not isinstance(_scheme.mgr.state, _scheme.TorchScheme):
        raise RuntimeError("native Torch IMRPhenomXHM requires TorchScheme")

    modes = _requested_modes(params)
    sequence = _imrphenomxas_sequence_samples(_xas_params(params))
    core = _SequenceCore(sequence.polarization)
    active_modes = _active_mode_samples(
        core,
        params,
        modes,
        frequencies=sequence.frequencies,
        reference_frequency=sequence.reference_frequency,
    )
    plus, cross = _polarizations_from_active_modes(
        core,
        params,
        modes,
        active_modes,
        sequence=True,
    )
    return (
        PyCBCArray(TorchArrayData(plus), copy=False),
        PyCBCArray(TorchArrayData(cross), copy=False),
    )
