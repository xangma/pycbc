# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch-native pieces of the IMRPhenomXHM mode-by-mode interface.

The quadrupole shares the IMRPhenomXAS implementation.  The ``(2, +/-1)`` and
``(3, +/-3)`` modes use native XHM no-mixing kernels.  Other higher modes
remain on the LAL path until their amplitude, phase, and mode-mixing models
are ported.
"""

from numbers import Integral

from pycbc import scheme as _scheme

from .imrphenomxas_torch import (
    _XAS_MODE_POLARIZATION_FACTOR,
    _imrphenomxas_core_torch,
    _series_from_active_samples,
    imrphenomxas_native_supported,
)
from .imrphenomxhm_mode21_torch import imrphenomxhm_h2m1_samples
from .imrphenomxhm_mode33_torch import imrphenomxhm_h3m3_samples


_NATIVE_MODES = frozenset({(2, -2), (2, -1), (2, 1), (2, 2), (3, -3), (3, 3)})


def _requested_modes(params):
    mode_array = params.get("mode_array")
    if mode_array is None:
        return None

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


def imrphenomxhm_modes_torch(**params):
    """Generate explicitly requested native XHM modes with Torch."""

    if not imrphenomxhm_modes_native_supported(params):
        raise ValueError(
            "only explicit IMRPhenomXHM (2, +/-1), (2, +/-2), and (3, +/-3) "
            "requests are supported by the native Torch path"
        )
    if not isinstance(_scheme.mgr.state, _scheme.TorchScheme):
        raise RuntimeError("native Torch IMRPhenomXHM modes require TorchScheme")

    modes = _requested_modes(params)
    if not modes:
        return {}

    core = _imrphenomxas_core_torch(_xas_params(params))
    active_modes = {}
    mode_families = {(ell, abs(emm)) for ell, emm in modes}
    if (2, 2) in mode_families:
        active_modes[2, 2] = core.polarization / _XAS_MODE_POLARIZATION_FACTOR
    if (2, 1) in mode_families:
        active_modes[2, 1] = imrphenomxhm_h2m1_samples(core, params)
    if (3, 3) in mode_families:
        active_modes[3, 3] = imrphenomxhm_h3m3_samples(core, params)

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
