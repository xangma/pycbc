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
    dense_engine.drain()
    rb_engine.drain()

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
    rb_engine.drain()

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
        experimental_approximation=True,
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


def _host(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _orthogonal_templates():
    rows = np.zeros((2, 17), dtype=np.complex64)
    rows[0, 1] = 1
    rows[1, 2] = 0.5
    templates = [FrequencySeries(row, delta_f=1) for row in rows]
    for i, template in enumerate(templates):
        template.id = 40 + i
    return templates


@pytest.mark.parametrize("device", ["numpy"] + DEVICES)
def test_truncation_retains_original_norms_and_reports_residuals(device):
    templates = _orthogonal_templates()
    plan = compute_reduced_basis(templates, max_rank=1, tolerance=1e-6, device=device)
    psd = FrequencySeries(np.ones(17, dtype=np.float32), delta_f=1)
    bound = bind_reduced_basis_psd(plan, psd, device=device)
    np.testing.assert_array_equal(_host(bound.tile_sigmasqs), [4, 1])
    # FP32 SVD may perturb even the retained orthogonal vector on CUDA.
    # Measure the actual stored factors independently in double precision.
    reconstructed = (_host(plan.coefficients).astype(np.complex128)
                     @ _host(plan.basis_data).astype(np.complex128))
    error = _host(plan.original_data).astype(np.complex128) - reconstructed
    residuals = 4 * np.sum(abs(error[:, 1:-1])**2, axis=1)
    np.testing.assert_allclose(_host(bound.residual_sigmasqs), residuals,
                               rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(_host(bound.relative_error),
                               np.sqrt(residuals / [4, 1]), atol=1e-15)
    assert _host(bound.relative_error)[0] <= plan.tolerance
    np.testing.assert_allclose(_host(bound.relative_error)[1], 1, atol=1e-12)
    np.testing.assert_array_equal(_host(bound.tolerance_met), [True, False])
    assert not plan.svd_tolerance_met
    assert not bound.certifies_reference_decisions
    templates[1][:] = 0
    assert _host(plan.original_data)[1, 2] == 0.5


@pytest.mark.parametrize("device", ["numpy"] + DEVICES)
def test_psd_binding_tracks_content_and_rejects_stale_state(device):
    plan = compute_reduced_basis(_orthogonal_templates(), max_rank=1, device=device)
    psd = FrequencySeries(np.ones(17, dtype=np.float32), delta_f=1)
    first = bind_reduced_basis_psd(plan, psd, psd_version="reused", device=device)
    psd[2] = 4
    second = bind_reduced_basis_psd(plan, psd, psd_version="reused", device=device)
    assert first.psd_version == second.psd_version
    assert first.version_hash != second.version_hash
    np.testing.assert_array_equal(_host(second.tile_sigmasqs), [4, 0.25])
    np.testing.assert_allclose(_host(second.residual_sigmasqs), [0, 0.25], atol=1e-12)
    first.validate(plan)  # Changing the caller's PSD did not mutate its snapshot.
    first.psd_data[2] = 8
    with pytest.raises(ValueError, match="PSD changed"):
        first.validate(plan)
    plan.coefficients[0, 0] += 1
    with pytest.raises(ValueError, match="plan changed"):
        second.validate(plan)


@pytest.mark.parametrize("device", ["numpy"] + DEVICES)
@pytest.mark.parametrize("cluster_policy", ["live_peak", "symmetric", "threshold_only"])
def test_uncertified_basis_defaults_to_original_reference_decisions(device, cluster_policy):
    templates = _orthogonal_templates()
    # Four adjacent frequencies produce a distinct peak; a single frequency
    # has constant magnitude and therefore no strict symmetric-cluster peak.
    templates[1][:] = 0
    templates[1][2:6] = 0.25
    plan = compute_reduced_basis(templates, max_rank=1, tolerance=1e-6, device=device)
    psd = FrequencySeries(np.ones(17, dtype=np.float32), delta_f=1)
    bound = bind_reduced_basis_psd(plan, psd, device=device)
    phase = np.exp(-2j * np.pi * np.arange(17) * 11 / 32)
    data = FrequencySeries((templates[1].numpy() * 20 * phase).astype(np.complex64),
                           delta_f=1)
    bank = prepare_bank(templates, tile_size=2, device=device)
    dense_psd = bind_psd(bank, psd, device=device)
    policy = SelectionPolicy(snr_threshold=5.5, cluster_policy=cluster_policy, cluster_window=4)
    dense = SearchEngine(bank, policy, device=device)
    safe = ReducedBasisSearchEngine(plan, policy, device=device)
    lossy = ReducedBasisSearchEngine(plan, policy, device=device, experimental_approximation=True)
    try:
        dense.submit(data, dense_psd, (0, 32))
        safe.submit(data, bound, (0, 32))
        expected = dense.drain()[0]
        actual = safe.drain()[0]
        assert safe.fallback_reason is not None
        assert len(actual.results) == len(expected.results) == 1
        for key in ("template_id", "sample_idx", "snr", "sigmasq"):
            np.testing.assert_array_equal(actual.results[0][key], expected.results[0][key])
        assert 41 in actual.results[0]["template_id"]
        assert 11 in actual.results[0]["sample_idx"]
        assert lossy.submit(data, bound, (0, 32)).results == []
    finally:
        dense.close()
        safe.close()
        lossy.close()


@pytest.mark.parametrize("device", ["numpy"] + DEVICES)
def test_weighted_residual_estimate_bounds_complex_snr_in_exact_arithmetic(device):
    rng = np.random.default_rng(779)
    rows = (rng.normal(size=(3, 17)) + 1j * rng.normal(size=(3, 17))).astype(np.complex64)
    templates = [FrequencySeries(row, delta_f=0.5) for row in rows]
    plan = compute_reduced_basis(templates, max_rank=1, f_lower=1, f_upper=7, device=device)
    psd = FrequencySeries(np.geomspace(0.01, 10, 17).astype(np.float32), delta_f=0.5)
    bound = bind_reduced_basis_psd(plan, psd, device=device)
    data = rng.normal(size=17) + 1j * rng.normal(size=17)
    weights = np.zeros(17)
    weights[2:14] = 2 / psd.numpy()[2:14]
    original = _host(plan.original_data).astype(np.complex128)
    reconstructed = (_host(plan.coefficients).astype(np.complex128)
                     @ _host(plan.basis_data).astype(np.complex128))
    sigma = np.sqrt(np.sum(abs(original)**2 * weights, axis=1))
    data_norm = np.sqrt(np.sum(abs(data)**2 * weights))
    # Independent double-precision all-time calculation of the mathematical
    # residual. This does not establish bounds versus a float32 FFT backend.
    corr_error = (original - reconstructed).conj() * data * weights
    snr_error = np.fft.ifft(corr_error, n=32, axis=-1) * 32 / sigma[:, None]
    estimate = data_norm * _host(bound.relative_error)
    assert np.all(abs(snr_error) <= estimate[:, None] + 1e-12)


def test_zero_bank_and_invalid_geometry_are_reported():
    zero = FrequencySeries(np.zeros(17, dtype=np.complex64), delta_f=1)
    plan = compute_reduced_basis([zero], device="numpy")
    bound = bind_reduced_basis_psd(plan, np.ones(17), device="numpy")
    assert plan.rank == 1
    assert plan.unweighted_relative_error == 0
    assert not bool(bound.tolerance_met[0])  # Undefined normalized SNR.
    assert np.isinf(bound.relative_error[0])
    with pytest.raises(ValueError, match="max_rank"):
        compute_reduced_basis([zero], max_rank=0, device="numpy")
    with pytest.raises(ValueError, match="tolerance"):
        compute_reduced_basis([zero], tolerance=-1, device="numpy")
    with pytest.raises(ValueError, match="PSD must cover"):
        bind_reduced_basis_psd(plan, np.ones(3), device="numpy")
    with pytest.raises(ValueError, match="delta_f"):
        bind_reduced_basis_psd(plan, FrequencySeries(np.ones(17), delta_f=0.5))


def test_reference_fallback_preserves_owned_template_metadata():
    templates = _orthogonal_templates()
    templates[0].params = {"mass1": 23.0}
    plan = compute_reduced_basis(templates, max_rank=1, device="numpy")
    templates[0].params["mass1"] = 90.0
    engine = ReducedBasisSearchEngine(plan, SelectionPolicy(5.5), device="numpy")
    try:
        retained = engine._reference_engine.bank_plan.templates[0]
        assert retained.params == {"mass1": 23.0}
    finally:
        engine.close()
