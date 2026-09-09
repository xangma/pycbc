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
