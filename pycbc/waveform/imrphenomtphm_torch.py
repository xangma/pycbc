# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch-native waveform assembly for time-domain IMRPhenomTPHM."""

from __future__ import annotations

import operator

from pycbc.types import TimeSeries
from pycbc.types.array_torch import TorchArrayData

from .imrphenomthm_torch import (
    _normalize_modes,
    _project_modes,
    imrphenomthm_td_torch,
)
from .imrphenomtp_waveform_torch import (
    _aligned_limit,
    _build_imrphenomtp_modes,
    _imrphenomtp_family_native_supported,
    _wrap_polarizations,
)


def imrphenomtphm_native_supported(parameters):
    """Return whether the native port covers these IMRPhenomTPHM options."""
    if not _imrphenomtp_family_native_supported(
        parameters,
        "IMRPhenomTPHM",
    ):
        return False
    try:
        _normalize_modes(
            parameters.get("mode_array"),
            approximant="IMRPhenomTPHM",
        )
    except ValueError:
        return False
    return True


def imrphenomtphm_td_torch(**parameters):
    """Generate native time-domain IMRPhenomTPHM polarizations with Torch."""
    coprecessing_modes = _normalize_modes(
        parameters.get("mode_array"),
        approximant="IMRPhenomTPHM",
    )
    if _aligned_limit(parameters):
        return imrphenomthm_td_torch(**parameters)

    state, _, _, modes = _build_imrphenomtp_modes(
        parameters,
        coprecessing_modes,
    )
    plus, cross = _project_modes(state.core, tuple(modes), modes)
    return _wrap_polarizations(state.core, (plus, cross))


def imrphenomtphm_modes_native_supported(parameters):
    """Return whether the native mode interface covers ``parameters``."""
    if not imrphenomtphm_native_supported(parameters):
        return False
    if "ell_max" in parameters:
        try:
            operator.index(parameters["ell_max"])
        except TypeError:
            return False
    return True


def imrphenomtphm_modes_torch(**parameters):
    """Generate native inertial-L0 IMRPhenomTPHM modes with Torch.

    LAL's ChooseTDModes wrapper defines these modes independently of reference
    phase and inclination. Keep that convention separate from polarization
    generation, where both quantities are physical inputs.
    """
    coprecessing_modes = _normalize_modes(
        parameters.get("mode_array"),
        approximant="IMRPhenomTPHM",
    )
    mode_parameters = dict(parameters)
    mode_parameters.update(
        coa_phase=0.0,
        inclination=0.0,
        long_asc_nodes=0.0,
    )
    state, _, _, modes = _build_imrphenomtp_modes(
        mode_parameters,
        coprecessing_modes,
    )
    return {
        mode: (
            TimeSeries(
                TorchArrayData(samples.real),
                delta_t=state.core.inputs.delta_t,
                epoch=state.core.epoch,
                copy=False,
            ),
            TimeSeries(
                TorchArrayData(samples.imag),
                delta_t=state.core.inputs.delta_t,
                epoch=state.core.epoch,
                copy=False,
            ),
        )
        for mode, samples in modes.items()
    }


__all__ = [
    "imrphenomtphm_modes_native_supported",
    "imrphenomtphm_modes_torch",
    "imrphenomtphm_native_supported",
    "imrphenomtphm_td_torch",
]
