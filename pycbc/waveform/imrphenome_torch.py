# Copyright (C) 2025
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Torch-facing wrappers for IMRPhenomE / IMRPhenomHM FD waveforms.

These call lalsimulation to generate the waveform and return PyCBC
FrequencySeries, enabling use within TorchScheme without manual transfers.
"""

from __future__ import annotations

import lalsimulation
from pycbc import pnutils
from pycbc.types import FrequencySeries

from .waveform import _check_lal_pars

def _call_lalsim_fd(apx: str, **p):
    hp1, hc1 = lalsimulation.SimInspiralChooseFDWaveform(
        float(pnutils.solar_mass_to_kg(p["mass1"])),
        float(pnutils.solar_mass_to_kg(p["mass2"])),
        float(p.get("spin1x", 0.0)),
        float(p.get("spin1y", 0.0)),
        float(p.get("spin1z", 0.0)),
        float(p.get("spin2x", 0.0)),
        float(p.get("spin2y", 0.0)),
        float(p.get("spin2z", 0.0)),
        pnutils.megaparsecs_to_meters(float(p["distance"])),
        float(p.get("inclination", 0.0)),
        float(p.get("coa_phase", 0.0)),
        float(p.get("long_asc_nodes", 0.0)),
        float(p.get("eccentricity", 0.0)),
        float(p.get("mean_per_ano", 0.0)),
        p["delta_f"],
        float(p["f_lower"]),
        float(p.get("f_final", 0.0)),
        float(p.get("f_ref", p["f_lower"])),
        _check_lal_pars(p),
        apx,
    )
    hp = FrequencySeries(hp1.data.data[:], delta_f=hp1.deltaF, epoch=hp1.epoch)
    hc = FrequencySeries(hc1.data.data[:], delta_f=hc1.deltaF, epoch=hc1.epoch)
    return hp, hc


def imrphenome_fd_torch(**p):
    return _call_lalsim_fd(lalsimulation.IMRPhenomE, **p)


def imrphenomhm_fd_torch(**p):
    return _call_lalsim_fd(lalsimulation.IMRPhenomHM, **p)
