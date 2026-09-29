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

"""Unit tests for batched live veto evaluation on JAX devices."""

from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from pycbc import scheme
from pycbc.filter.matchedfilter_jax import _batched_live_vetoes_gpu
from pycbc.types import FrequencySeries
from pycbc.types.array_jax import _ensure_x64, to_jax


def _devices():
    devices = ["cpu"]
    try:
        if jax.devices("gpu"):
            devices.append("cuda:0")
    except RuntimeError:
        pass
    return devices


@pytest.mark.parametrize("dev_name", _devices())
def test_batched_live_veto_gpu_equivalence(dev_name):
    """Batched GPU veto path must match scalar evaluation exactly."""
    _ensure_x64()
    with scheme.JAXScheme(dev_name):
        flen = 4096
        delta_f = 0.25
        kmin = 120
        bins1 = np.array([120, 200, 350, 600, 900], dtype=np.int32)
        bins2 = np.array([120, 220, 380, 650, 1000], dtype=np.int32)

        rng = np.random.default_rng(2026)
        data1 = (rng.normal(size=flen) +
                 1j * rng.normal(size=flen)).astype(np.complex64)
        data2 = (rng.normal(size=flen) +
                 1j * rng.normal(size=flen)).astype(np.complex64)
        s_data = (rng.normal(size=flen) +
                  1j * rng.normal(size=flen)).astype(np.complex64)

        psd = FrequencySeries(
            np.ones(flen, dtype=np.float32), delta_f=delta_f
        )
        stilde = FrequencySeries(s_data, delta_f=delta_f)
        stilde.psd = psd

        tmpl1 = FrequencySeries(data1, delta_f=delta_f)
        tmpl1.f_lower = kmin * delta_f
        tmpl1.cout = np.zeros(flen, dtype=np.complex64)

        tmpl2 = FrequencySeries(data2, delta_f=delta_f)
        tmpl2.f_lower = kmin * delta_f
        tmpl2.cout = np.zeros(flen, dtype=np.complex64)

        bin_map = {id(tmpl1): bins1, id(tmpl2): bins2}

        class MockPowerChisq:
            do = True
            snr_threshold = None

            def cached_chisq_bins(self, tmpl, psd):
                return bin_map.get(id(tmpl))

        control = SimpleNamespace(
            power_chisq=MockPowerChisq(),
            sg_chisq=SimpleNamespace(do=False),
            newsnr_threshold=None,
        )

        matrix = to_jax(np.stack([data1, data2]))
        veto_info = [
            (np.array([5.5 + 2.1j]), 0.12, 1024, tmpl1, stilde, matrix, 0),
            (np.array([4.2 - 3.7j]), 0.15, 2048, tmpl2, stilde, matrix, 1),
        ]

        results = {
            "snr": jnp.array([abs(5.5 + 2.1j) * 0.12, abs(4.2 - 3.7j) * 0.15])
        }

        batched_res = _batched_live_vetoes_gpu(
            control, dict(results), veto_info, control.power_chisq
        )
        assert batched_res is not None

        # Scalar reference computation
        from pycbc.vetoes.chisq_jax import _shift_sum_gpu_core
        ref_chisq = []
        for i, info in enumerate(veto_info):
            snrv, norm, l, tmpl, _ = info[:5]
            corr = np.conj(tmpl) * s_data
            bins = bin_map[id(tmpl)]
            bin_rel = bins - kmin
            shift = _shift_sum_gpu_core(
                to_jax(corr)[kmin:bins[-1]],
                float(kmin),
                np.array([l]),
                bin_rel,
                float(flen),
            )
            nb = len(bins) - 1
            dof = nb * 2 - 2
            raw = (float(shift[0]) * nb - abs(snrv[0]) ** 2) * (norm ** 2)
            ref_chisq.append(raw / dof)

        np.testing.assert_allclose(
            np.asarray(batched_res["chisq"]), ref_chisq, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_array_equal(
            np.asarray(batched_res["chisq_dof"]), [6, 6]
        )
        np.testing.assert_array_equal(
            np.asarray(batched_res["sg_chisq"]), [0.0, 0.0]
        )
