# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch-native pieces of the IMRPhenomXHM mode-by-mode interface.

The IMRPhenomXHM quadrupole is the IMRPhenomXAS mode. This module exposes that
shared implementation through :func:`get_fd_waveform_modes` for explicitly
requested ``(2, +/-2)`` modes. Higher-mode fits remain on the LAL path until
their XHM amplitude, phase, and mode-mixing models are ported.
"""

from numbers import Integral

from pycbc import scheme as _scheme

from .imrphenomxas_torch import (
    imrphenomxas_h2m2_torch,
    imrphenomxas_native_supported,
)


_NATIVE_MODES = frozenset({(2, -2), (2, 2)})


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
    """Generate explicitly requested XHM quadrupole modes with Torch."""

    if not imrphenomxhm_modes_native_supported(params):
        raise ValueError(
            "only explicit IMRPhenomXHM (2, +/-2) mode requests are "
            "supported by the native Torch path"
        )
    if not isinstance(_scheme.mgr.state, _scheme.TorchScheme):
        raise RuntimeError("native Torch IMRPhenomXHM modes require TorchScheme")

    modes = _requested_modes(params)
    if not modes:
        return {}

    h2m2 = imrphenomxas_h2m2_torch(**_xas_params(params))
    result = {}
    for ell, emm in modes:
        # LAL exposes h_(2,-2) at positive frequencies. The positive-m mode
        # follows from equatorial symmetry and is conjugated for even ell.
        hlm = h2m2 if emm < 0 else h2m2.conj()
        hplus = 0.5 * hlm
        hcross = (0.5j if emm < 0 else -0.5j) * hlm
        result[ell, emm] = (hplus, hcross)
    return result
