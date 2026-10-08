# Copyright (C) 2026 The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Unit tests for JAX-native live template generation and bank packing."""

import numpy as np
import pytest
from types import SimpleNamespace

jax = pytest.importorskip("jax")
from pycbc import scheme
from pycbc.types.array_jax import to_jax
from pycbc.waveform.live_bank_jax import generate_live_waveforms_jax
from pycbc.types import FrequencySeries


def _devices():
    devices = ["cpu"]
    try:
        if jax.devices("gpu"):
            devices.append("cuda:0")
    except RuntimeError:
        pass
    return devices


class DummyBankTable:
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        return self._rows[idx]


class DummyLiveFilterBank:
    """Mock LiveFilterBank for unit testing live waveform generation."""
    def __init__(self, templates, sample_rate=2048):
        self.sample_rate = sample_rate
        self.min_f_lower = 30.0
        self.table = DummyBankTable(templates)
        self.extra_args = {}

    def __len__(self):
        return len(self.table)

    def approximant(self, idx):
        return "TaylorF2"

    def freq_resolution_for_template(self, idx):
        return self.table[idx].delta_f

    def end_frequency(self, idx):
        return self.table[idx].f_final

    def _waveform_parameters(self, idx, delta_f=None):
        t = self.table[idx]
        df = delta_f if delta_f is not None else t.delta_f
        flen = round(self.sample_rate / (2 * df) + 1)
        params = {
            "approximant": "TaylorF2",
            "mass1": t.mass1,
            "mass2": t.mass2,
            "spin1z": t.spin1z,
            "spin2z": t.spin2z,
            "f_lower": t.f_lower,
            "f_final": t.f_final,
            "delta_f": df,
            "distance": 1.0,
            "inclination": 0., "coa_phase": 0., "delta_t": 1. / self.sample_rate,
        }
        return params, flen

    def id_from_param(self, param_tuple):
        return int(param_tuple[0] * 1000 + param_tuple[1])


def _make_dummy_bank():
    t1 = SimpleNamespace(
        mass1=1.4, mass2=1.4, spin1z=0.0, spin2z=0.0,
        f_lower=30.0, f_final=800.0, delta_f=0.5,
    )
    t2 = SimpleNamespace(
        mass1=2.0, mass2=1.5, spin1z=0.1, spin2z=-0.1,
        f_lower=30.0, f_final=750.0, delta_f=0.5,
    )
    t3 = SimpleNamespace(
        mass1=1.8, mass2=1.2, spin1z=0.05, spin2z=-0.05,
        f_lower=30.0, f_final=850.0, delta_f=0.25,
    )
    return DummyLiveFilterBank([t1, t2, t3])


@pytest.mark.parametrize("device", _devices())
def test_generate_live_waveforms_jax_attributes_and_residency(device):
    bank = _make_dummy_bank()
    with scheme.JAXScheme(device) as ctx:
        wfs = generate_live_waveforms_jax(bank, rank=1, size=1)
        assert len(wfs) == 3

        for i, wf in enumerate(wfs):
            assert isinstance(wf, FrequencySeries)
            assert wf.delta_f == bank.table[i].delta_f
            assert wf.f_lower == bank.table[i].f_lower
            assert wf.end_frequency == bank.table[i].f_final
            assert hasattr(wf, "sigmasq")
            assert hasattr(wf, "id")
            assert wf.approximant == "TaylorF2"

            arr = to_jax(wf)
            assert arr.devices() == {ctx.jax_device}
            assert arr.shape == wf.shape
            assert np.linalg.norm(np.asarray(arr)) > 0.0


@pytest.mark.parametrize("device", _devices())
def test_generate_live_waveforms_jax_rank_slicing(device):
    bank = _make_dummy_bank()
    with scheme.JAXScheme(device):
        wfs_r1 = generate_live_waveforms_jax(bank, rank=1, size=2)
        assert len(wfs_r1) == 3

        wfs_r2 = generate_live_waveforms_jax(bank, rank=2, size=3)
        assert len(wfs_r2) == 1
        assert wfs_r2[0].params.mass1 == 2.0
