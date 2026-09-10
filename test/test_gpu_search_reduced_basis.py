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
Unit and parity tests for reduced-basis matched filtering in the
persistent GPU search engine.
"""

import numpy as np
import pytest

try:
    import torch
except ImportError:
    torch = None

from pycbc.types import FrequencySeries
from pycbc.filter.gpu_search import (
    compute_reduced_basis,
    bind_reduced_basis_psd,
    ReducedBasisSearchEngine,
    prepare_bank,
    bind_psd,
    SelectionPolicy,
    SearchEngine,
    prepare_power_chisq_plan,
    VetoManager,
)

CUDA_AVAILABLE = torch is not None and torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA_AVAILABLE else [])


def _make_reduced_basis_fixtures(num_templates=8, rank=3, size=1024, device="cpu"):
    flen = size // 2 + 1
    delta_f = 0.5
    rng = np.random.default_rng(12345)

    # Generate rank independent base modes
    base_modes = []
    for _ in range(rank):
        vec = (rng.normal(size=flen) + 1j * rng.normal(size=flen)).astype(np.complex64)
        vec[0] = 0.0
        vec /= np.linalg.norm(vec)
        base_modes.append(vec)
    base_modes = np.stack(base_modes, axis=0)

    # Construct templates as linear combinations of base modes + small noise
    templates = []
    mixing = rng.normal(size=(num_templates, rank)).astype(np.float32)
    for i in range(num_templates):
        h_vals = np.sum(mixing[i, :, None] * base_modes, axis=0)
        h_vals += 1e-4 * (rng.normal(size=flen) + 1j * rng.normal(size=flen)).astype(np.complex64)
        h_vals[0] = 0.0
        norm = np.linalg.norm(h_vals)
        if norm > 0:
            h_vals /= norm
        t = FrequencySeries(h_vals, delta_f=delta_f)
        t.id = 500 + i
        t.f_lower = 20.0
        t.f_upper = 200.0
        templates.append(t)

    basis_plan = compute_reduced_basis(
        templates,
        max_rank=rank + 1,
        tolerance=1e-3,
        f_lower=20.0,
        f_upper=200.0,
        device=device,
    )

    psd = FrequencySeries(np.ones(flen, dtype=np.float32) * 2.0, delta_f=delta_f)
    rb_psd = bind_reduced_basis_psd(basis_plan, psd, psd_version="psd-v1", device=device)

    # Standard dense bank plan for parity comparison
    dense_bank = prepare_bank(templates, tile_size=num_templates, f_lower=20.0, f_upper=200.0, device=device)
    dense_psd = bind_psd(dense_bank, psd, psd_version="psd-v1", device=device)

    # Inject template 2 at sample 180 with target SNR 16.0
    inj_idx = 2
    t0 = 180
    target_snr = 16.0
    sigmasq = float(dense_psd.tile_sigmasqs[0][inj_idx])
    A = target_snr / np.sqrt(sigmasq)
    phase_shift = np.exp(-2j * np.pi * np.arange(flen) * delta_f * (t0 / (size * delta_f)))
    data_vals = templates[inj_idx].numpy() * phase_shift * A
    data = FrequencySeries(data_vals, delta_f=delta_f)
    data.psd = psd

    return basis_plan, rb_psd, dense_bank, dense_psd, data, templates, inj_idx, t0


@pytest.mark.parametrize("device", DEVICES)
def test_reduced_basis_svd_rank_truncation(device):
    """Verify SVD rank truncation and template reconstruction accuracy."""
    basis_plan, _, _, _, _, templates, _, _ = _make_reduced_basis_fixtures(
        num_templates=8, rank=3, size=1024, device=device
    )
    assert basis_plan.rank <= 4
    assert basis_plan.rank < basis_plan.num_templates
    assert basis_plan.num_templates == 8

    if hasattr(basis_plan.coefficients, "cpu"):
        coeffs = basis_plan.coefficients.cpu().numpy()
        vh = basis_plan.basis_data.cpu().numpy()
    else:
        coeffs = np.asarray(basis_plan.coefficients)
        vh = np.asarray(basis_plan.basis_data)

    h_recon = np.matmul(coeffs, vh)
    for i, t in enumerate(templates):
        orig = t.numpy()
        recon = h_recon[i]
        rel_err = np.linalg.norm(orig - recon) / np.linalg.norm(orig)
        assert rel_err < 1e-2


@pytest.mark.parametrize("device", DEVICES)
def test_reduced_basis_search_parity(device):
    """Verify ReducedBasisSearchEngine produces parity with dense SearchEngine."""
    basis_plan, rb_psd, dense_bank, dense_psd, data, templates, inj_idx, t0 = (
        _make_reduced_basis_fixtures(num_templates=8, rank=3, size=1024, device=device)
    )

    policy = SelectionPolicy(snr_threshold=5.5)
    dense_engine = SearchEngine(dense_bank, selection_policy=policy, device=device)
    rb_engine = ReducedBasisSearchEngine(basis_plan, selection_policy=policy, device=device)

    valid_interval = (50, 450)
    t_dense = dense_engine.submit(data, dense_psd, valid_interval=valid_interval)
    t_rb = rb_engine.submit(data, rb_psd, valid_interval=valid_interval)

    assert not t_dense.aborted and not t_dense.overflow
    assert not t_rb.aborted and not t_rb.overflow
    assert len(t_dense.results) > 0
    assert len(t_rb.results) > 0

    cands_dense = t_dense.results[0]
    cands_rb = t_rb.results[0]

    peak_dense_idx = int(np.argmax(cands_dense["snr"]))
    peak_rb_idx = int(np.argmax(cands_rb["snr"]))

    dense_sample = int(cands_dense["sample_idx"][peak_dense_idx])
    rb_sample = int(cands_rb["sample_idx"][peak_rb_idx])
    dense_snr = float(np.abs(cands_dense["snr"][peak_dense_idx]))
    rb_snr = float(np.abs(cands_rb["snr"][peak_rb_idx]))

    assert abs(rb_sample - t0) <= 1
    assert abs(dense_sample - rb_sample) <= 1
    assert abs(dense_snr - rb_snr) / dense_snr < 0.01

    assert cands_rb["template_id"][peak_rb_idx] == templates[inj_idx].id
    assert cands_dense["template_id"][peak_dense_idx] == templates[inj_idx].id

    dense_engine.close()
    rb_engine.close()


@pytest.mark.parametrize("device", DEVICES)
def test_reduced_basis_veto_evaluation(device):
    """Verify power chi-squared veto evaluation on reduced-basis candidates."""
    basis_plan, rb_psd, dense_bank, dense_psd, data, templates, inj_idx, t0 = (
        _make_reduced_basis_fixtures(num_templates=8, rank=3, size=1024, device=device)
    )

    num_bins = 4
    power_plan = prepare_power_chisq_plan(
        dense_bank, dense_psd, num_bins=num_bins, device=device
    )
    veto_mgr = VetoManager(power_chisq_plan=power_plan)

    policy = SelectionPolicy(snr_threshold=5.5)
    rb_engine = ReducedBasisSearchEngine(
        basis_plan,
        selection_policy=policy,
        veto_manager=veto_mgr,
        device=device,
    )

    valid_interval = (50, 450)
    ticket = rb_engine.submit(data, rb_psd, valid_interval=valid_interval)

    assert len(ticket.results) > 0
    cands = ticket.results[0]
    assert "chisq" in cands
    assert "chisq_dof" in cands

    peak_idx = int(np.argmax(cands["snr"]))
    reduced_chisq = float(cands["chisq"][peak_idx]) / float(cands["chisq_dof"][peak_idx])
    assert reduced_chisq < 1.0

    rb_engine.close()


@pytest.mark.parametrize("device", DEVICES)
def test_reduced_basis_overflow_and_abort(device):
    """Verify overflow and abort triggers in ReducedBasisSearchEngine."""
    basis_plan, rb_psd, _, _, data, _, _, _ = _make_reduced_basis_fixtures(
        num_templates=8, rank=3, size=1024, device=device
    )

    policy_overflow = SelectionPolicy(snr_threshold=0.1)
    engine_overflow = ReducedBasisSearchEngine(
        basis_plan,
        selection_policy=policy_overflow,
        candidate_capacity=2,
        device=device,
    )
    t_over = engine_overflow.submit(data, rb_psd, valid_interval=(50, 450))
    assert t_over.overflow
    engine_overflow.close()

    policy_abort = SelectionPolicy(snr_threshold=5.0, snr_abort_threshold=10.0)
    engine_abort = ReducedBasisSearchEngine(
        basis_plan,
        selection_policy=policy_abort,
        device=device,
    )
    t_abort = engine_abort.submit(data, rb_psd, valid_interval=(50, 450))
    assert t_abort.aborted
    engine_abort.close()


@pytest.mark.parametrize("device", DEVICES)
def test_reduced_basis_plan_migration(device):
    """Verify ReducedBasisPlan and ReducedBasisPSDPlan migration across devices."""
    basis_plan, rb_psd, _, _, _, _, _, _ = _make_reduced_basis_fixtures(
        num_templates=4, rank=2, size=1024, device="cpu"
    )

    p_dev = basis_plan.to(device)
    psd_dev = rb_psd.to(device)

    assert p_dev.rank == basis_plan.rank
    assert p_dev.num_templates == basis_plan.num_templates
    assert p_dev.device == str(device)
    assert psd_dev.device == str(device)
