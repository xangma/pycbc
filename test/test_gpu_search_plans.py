# Copyright (C) 2026
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

"""
Unit tests for GPU search engine plans, versioning, and workspace budgets.
"""

import numpy as np
import pytest

try:
    import torch
except ImportError:
    torch = None

DEVICES = ["cpu"] + (
    ["cuda"] if (torch is not None and torch.cuda.is_available()) else []
)

from pycbc.types import FrequencySeries
from pycbc.filter.gpu_search.plans import (
    BankGeometry,
    WorkspaceBudget,
    prepare_bank,
    bind_psd,
)


def _make_dummy_templates(num_templates=10, filter_length=513, delta_f=1.0 / 16.0):
    templates = []
    np.random.seed(42)
    for i in range(num_templates):
        data = (
            np.random.randn(filter_length) + 1j * np.random.randn(filter_length)
        ).astype(np.complex64)
        fs = FrequencySeries(data, delta_f=delta_f)
        fs.id = i
        templates.append(fs)
    return templates


def test_bank_geometry():
    geom = BankGeometry.from_delta_f_and_length(delta_f=0.25, filter_length=1025)
    assert geom.delta_f == 0.25
    assert geom.filter_length == 1025
    assert geom.transform_length == 2048
    assert geom.sample_rate == 512.0
    assert geom.delta_t == 1.0 / 512.0


@pytest.mark.parametrize("device", DEVICES)
def test_prepare_bank(device):
    templates = _make_dummy_templates(num_templates=10, filter_length=513)
    bank_plan = prepare_bank(templates, tile_size=4, device=device)

    assert bank_plan.num_templates == 10
    assert len(bank_plan.tiles) == 3
    assert bank_plan.tiles[0].batch_size == 4
    assert bank_plan.tiles[1].batch_size == 4
    assert bank_plan.tiles[2].batch_size == 2

    assert bank_plan.version_hash != ""
    # Plan hash is deterministic
    bank_plan2 = prepare_bank(templates, tile_size=4, device=device)
    assert bank_plan.version_hash == bank_plan2.version_hash


@pytest.mark.parametrize("device", DEVICES)
def test_bind_psd_sigmasq_agreement(device):
    from pycbc.filter.matchedfilter import sigmasq

    templates = _make_dummy_templates(
        num_templates=4, filter_length=513, delta_f=0.5
    )
    bank_plan = prepare_bank(templates, tile_size=2, device=device)

    # Create dummy flat PSD
    psd_data = np.ones(513, dtype=np.float32) * 2.0
    psd = FrequencySeries(psd_data, delta_f=0.5)

    psd_plan = bind_psd(bank_plan, psd, device=device)
    assert psd_plan.version_hash != ""

    # Check sigmasq for each template matches standard PyCBC sigmasq
    for tile in bank_plan.tiles:
        computed_sigmasqs = psd_plan.tile_sigmasqs[tile.tile_id]
        if hasattr(computed_sigmasqs, "cpu"):
            computed_sigmasqs = computed_sigmasqs.cpu().numpy()
        elif hasattr(computed_sigmasqs, "numpy"):
            computed_sigmasqs = computed_sigmasqs.numpy()

        for idx, t_id in enumerate(tile.template_ids):
            ref_sigmasq = sigmasq(templates[t_id], psd=psd)
            np.testing.assert_allclose(
                computed_sigmasqs[idx], ref_sigmasq, rtol=1e-5
            )


def test_workspace_budget():
    bytes_req = WorkspaceBudget.calculate_workspace_bytes(
        batch_size=64,
        transform_length=131072,
        filter_length=65537,
    )
    # (65537 + 131072) * 8 * 64 + 65537 * 8 ≈ 100 MB
    assert bytes_req > 0
    assert bytes_req < 200 * 1024 * 1024

    recommended = WorkspaceBudget.recommend_batch_size(
        available_vram_bytes=4 * 1024 * 1024 * 1024,  # 4 GiB
        transform_length=131072,
        filter_length=65537,
        headroom_ratio=0.5,
    )
    assert recommended >= 1000


def test_workspace_budget_accurate_accounting():
    # A 16,400-byte budget must reject 20,488 bytes of core buffers
    batch_size = 1
    tlen = 1024
    flen = 513
    bytes_req = WorkspaceBudget.calculate_workspace_bytes(
        batch_size=batch_size,
        transform_length=tlen,
        filter_length=flen,
        candidate_capacity=0,
    )
    assert bytes_req == 20488
    assert bytes_req > 16400


def test_bank_plan_hash_includes_geometry():
    """Verify bank plan version hash incorporates geometry parameters."""
    templates = _make_dummy_templates(num_templates=2, filter_length=513)
    bank_plan1 = prepare_bank(
        templates, tile_size=2, f_lower=20.0, f_upper=200.0, device="cpu"
    )
    bank_plan2 = prepare_bank(
        templates, tile_size=2, f_lower=30.0, f_upper=150.0, device="cpu"
    )

    assert bank_plan1.version_hash != bank_plan2.version_hash
    assert bank_plan1.tiles[0].version_hash != bank_plan2.tiles[0].version_hash


def test_bind_psd_physical_scale_and_dyn_range_factor():
    """Verify physical PSD (e.g. 1e-47) avoids float32 underflow in bind_psd."""
    from pycbc import DYN_RANGE_FAC

    templates = _make_dummy_templates(num_templates=2, filter_length=513, delta_f=1.0)
    bank_plan = prepare_bank(
        templates, tile_size=2, f_lower=20.0, f_upper=200.0, device="cpu"
    )

    # Native physical PSD with in-band value ~ 1.5e-47
    psd_vals = np.zeros(513, dtype=np.float64)
    psd_vals[20:200] = 1.52e-47
    psd = FrequencySeries(psd_vals, delta_f=1.0)

    # Without dynamic range scaling, float32 conversion zeroes this out
    assert psd_vals[100].astype(np.float32) == 0.0

    # bind_psd auto-detects tiny physical PSD and scales by DYN_RANGE_FAC**2
    psd_plan = bind_psd(bank_plan, psd, device="cpu")
    psd_out = (
        psd_plan.psd_data.numpy()
        if hasattr(psd_plan.psd_data, "numpy")
        else np.asarray(psd_plan.psd_data)
    )

    assert psd_out[100] > 0.0
    assert psd_plan.dyn_range_factor == float(DYN_RANGE_FAC)
    assert psd_plan.tile_norms[0][0] > 0.0

