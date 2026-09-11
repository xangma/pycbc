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
Unit tests for GPU search engine batched full-statistic vetoes (Power chisq & SG chisq).
"""

import types
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
from pycbc.filter.matchedfilter import (
    sigmasq_series,
    get_cutoff_indices,
    matched_filter_core,
)
from pycbc.vetoes.chisq import (
    power_chisq_bins_from_sigmasq_series,
    power_chisq_at_points_from_precomputed,
)
from pycbc.filter.gpu_search.plans import prepare_bank, bind_psd
from pycbc.filter.gpu_search.candidates import SelectionPolicy
from pycbc.filter.gpu_search.vetoes import (
    prepare_power_chisq_plan,
    batched_power_chisq,
    VetoManager,
    SineGaussianPlan,
)
from pycbc.filter.gpu_search.engine import SearchEngine


def _make_test_fixtures(filter_length=513, delta_f=0.5, num_templates=2):
    np.random.seed(42)
    N = (filter_length - 1) * 2

    templates = []
    for i in range(num_templates):
        h_data = (
            np.random.randn(filter_length) + 1j * np.random.randn(filter_length)
        ).astype(np.complex64)
        h_data[0] = 0.0
        htilde = FrequencySeries(h_data, delta_f=delta_f)
        htilde.id = i
        templates.append(htilde)

    s_data = (
        np.random.randn(filter_length) + 1j * np.random.randn(filter_length)
    ).astype(np.complex64)
    s_data[0] = 0.0
    stilde = FrequencySeries(s_data, delta_f=delta_f)

    psd_data = np.ones(filter_length, dtype=np.float32) * 2.0
    psd = FrequencySeries(psd_data, delta_f=delta_f)

    return templates, stilde, psd, N


@pytest.mark.parametrize("device", DEVICES)
def test_power_chisq_plan_creation_and_bin_edges(device):
    templates, _, psd, N = _make_test_fixtures(num_templates=3)
    bank_plan = prepare_bank(
        templates, tile_size=3, f_lower=20.0, f_upper=200.0, device=device
    )
    psd_plan = bind_psd(bank_plan, psd, device=device)

    num_bins = 16
    plan = prepare_power_chisq_plan(
        bank_plan, psd_plan, num_bins=num_bins, snr_threshold=5.5, device=device
    )

    assert plan.num_bins == num_bins
    assert plan.snr_threshold == 5.5
    assert plan.dof == 2 * num_bins - 2
    assert 0 in plan.tile_bin_edges

    tile_bins = plan.tile_bin_edges[0]
    if hasattr(tile_bins, "cpu"):
        tile_bins = tile_bins.cpu().numpy()

    assert tile_bins.shape == (3, num_bins + 1)

    kmin, kmax = get_cutoff_indices(20.0, 200.0, bank_plan.geometry.delta_f, N)

    # Check each template bin edges match reference power_chisq_bins_from_sigmasq_series
    for i in range(3):
        h = templates[i]
        s_series = sigmasq_series(
            h, psd=psd, low_frequency_cutoff=20.0, high_frequency_cutoff=200.0
        )
        ref_bins = power_chisq_bins_from_sigmasq_series(s_series, num_bins, kmin, kmax)
        np.testing.assert_array_equal(tile_bins[i], ref_bins)


@pytest.mark.parametrize("device", DEVICES)
def test_batched_power_chisq_numerical_parity(device):
    templates, stilde, psd, N = _make_test_fixtures(num_templates=1)
    htilde = templates[0]

    bank_plan = prepare_bank(
        [htilde], tile_size=1, f_lower=20.0, f_upper=200.0, device=device
    )
    psd_plan = bind_psd(bank_plan, psd, device=device)

    num_bins = 16
    plan = prepare_power_chisq_plan(
        bank_plan, psd_plan, num_bins=num_bins, device=device
    )

    ref_snr_series, ref_corr, ref_norm = matched_filter_core(
        htilde, stilde, psd=psd, low_frequency_cutoff=20.0, high_frequency_cutoff=200.0
    )

    points = np.array([120, 250, 480], dtype=np.int64)
    ref_snrv = np.asarray(ref_snr_series[points])
    tile_bins = plan.tile_bin_edges[0]
    if hasattr(tile_bins, "cpu"):
        tile_bins_np = tile_bins.cpu().numpy()[0]
    else:
        tile_bins_np = tile_bins[0]

    ref_chisq = power_chisq_at_points_from_precomputed(
        ref_corr, ref_snrv, ref_norm, tile_bins_np, points
    )
    ref_chisq_vals = np.asarray(ref_chisq)

    # Setup candidate dict for batched_power_chisq
    cand_snr_normed = ref_snrv * ref_norm
    candidates = {
        "template_idx": np.zeros(len(points), dtype=np.int64),
        "sample_idx": points,
        "snr": cand_snr_normed,
    }

    flen = bank_plan.geometry.filter_length
    corr_np = np.zeros((1, N), dtype=np.complex64)
    corr_np[0, :flen] = np.asarray(ref_corr.numpy()[:flen])

    if torch is not None and device != "numpy":
        corr_mem = torch.as_tensor(corr_np, device=torch.device(device))
        norms_mem = psd_plan.tile_norms[0]
        bins_mem = plan.tile_bin_edges[0]
    else:
        corr_mem = corr_np
        norms_mem = psd_plan.tile_norms[0]
        bins_mem = plan.tile_bin_edges[0]

    chisq, chisq_dof = batched_power_chisq(
        corr_tile=corr_mem,
        candidates=candidates,
        tile_bin_edges=bins_mem,
        tile_norms=norms_mem,
        num_bins=num_bins,
        snr_threshold=None,
        transform_length=N,
    )

    assert chisq.shape == (len(points),)
    assert (chisq_dof == 2 * num_bins - 2).all()
    np.testing.assert_allclose(chisq, ref_chisq_vals, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_power_chisq_activation_threshold(device):
    templates, stilde, psd, N = _make_test_fixtures(num_templates=1)
    bank_plan = prepare_bank(templates, tile_size=1, device=device)
    psd_plan = bind_psd(bank_plan, psd, device=device)
    plan = prepare_power_chisq_plan(bank_plan, psd_plan, num_bins=16, device=device)

    snr_series, ref_corr, ref_norm = matched_filter_core(
        templates[0], stilde * 10.0, psd=psd
    )
    snr_vals = np.asarray(snr_series.numpy() * ref_norm)

    # Trigger 0 has low SNR (e.g. 2.0), Trigger 1 has high SNR from the filtered series
    candidates = {
        "template_idx": np.array([0, 0], dtype=np.int64),
        "sample_idx": np.array([150, 250], dtype=np.int64),
        "snr": np.array([2.0 + 0j, snr_vals[250]], dtype=np.complex64),
    }

    flen = bank_plan.geometry.filter_length
    corr_np = np.zeros((1, N), dtype=np.complex64)
    corr_np[0, :flen] = ref_corr.numpy()[:flen]

    corr_mem = (
        torch.as_tensor(corr_np, device=torch.device(device))
        if (torch is not None and device != "numpy")
        else corr_np
    )

    chisq, chisq_dof = batched_power_chisq(
        corr_tile=corr_mem,
        candidates=candidates,
        tile_bin_edges=plan.tile_bin_edges[0],
        tile_norms=psd_plan.tile_norms[0],
        num_bins=16,
        snr_threshold=5.0,
        transform_length=N,
    )

    # Below threshold: 0.0; Above threshold: > 0.0
    assert chisq[0] == 0.0
    assert chisq[1] > 0.0
    assert chisq_dof[0] == 30
    assert chisq_dof[1] == 30


@pytest.mark.parametrize("device", DEVICES)
def test_power_chisq_large_sample_stability(device):
    # Test large time offset (e.g. 500,000) for phase stability
    large_N = 1048576  # 2^20
    flen = large_N // 2 + 1
    kmin, kmax = 40, 400
    num_bins = 16

    corr_np = np.zeros((1, flen), dtype=np.complex64)
    corr_np[0, kmin:kmax] = 1.0 + 0.5j

    bins_np = np.linspace(kmin, kmax, num_bins + 1, dtype=np.int64)[np.newaxis, :]
    norms_np = np.array([0.05], dtype=np.float32)

    # Large time indices
    large_points = np.array([500000, 999999], dtype=np.int64)
    candidates = {
        "template_idx": np.array([0, 0], dtype=np.int64),
        "sample_idx": large_points,
        "snr": np.array([8.0 + 2j, 9.0 - 1j], dtype=np.complex64),
    }

    if torch is not None and device != "numpy":
        corr_mem = torch.as_tensor(corr_np, device=torch.device(device))
        bins_mem = torch.as_tensor(bins_np, device=torch.device(device))
        norms_mem = torch.as_tensor(norms_np, device=torch.device(device))
    else:
        corr_mem = corr_np
        bins_mem = bins_np
        norms_mem = norms_np

    chisq, _ = batched_power_chisq(
        corr_tile=corr_mem,
        candidates=candidates,
        tile_bin_edges=bins_mem,
        tile_norms=norms_mem,
        num_bins=num_bins,
        snr_threshold=None,
        transform_length=large_N,
    )

    # Check finite and non-negative
    assert np.isfinite(chisq).all()
    assert (chisq >= 0.0).all()


@pytest.mark.parametrize("device", DEVICES)
def test_engine_integration_with_power_chisq(device):
    templates, stilde, psd, N = _make_test_fixtures(num_templates=2)
    bank_plan = prepare_bank(templates, tile_size=2, device=device)
    psd_plan = bind_psd(bank_plan, psd, device=device)

    chisq_plan = prepare_power_chisq_plan(
        bank_plan, psd_plan, num_bins=16, snr_threshold=0.0, device=device
    )

    policy = SelectionPolicy(
        snr_threshold=0.0,
        cluster_policy="live_peak",
    )

    engine = SearchEngine(
        bank_plan, policy, veto_manager=chisq_plan, device=device
    )

    ticket = engine.submit(stilde, psd_plan, valid_interval=(100, 900))
    assert ticket.completed
    assert not ticket.aborted

    ready = engine.drain()
    assert len(ready) == 1
    cands = ready[0].results[0]

    assert "chisq" in cands
    assert "chisq_dof" in cands
    assert len(cands["chisq"]) == len(cands["sample_idx"])
    assert (cands["chisq_dof"] == 30).all()

    engine.close()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("delta_f", [0.01, 0.005])
def test_prepare_power_chisq_plan_arbitrary_delta_f_parity(device, delta_f):
    from pycbc.filter.matchedfilter import sigmasq_series
    from pycbc.vetoes.chisq import power_chisq_bins_from_sigmasq_series

    kmin, kmax = int(30.0 / delta_f), int(1000.0 / delta_f)
    flen = kmax + 100
    freqs = np.arange(flen) * delta_f
    amp = np.where(freqs > 30, (freqs / 30.0) ** (-7 / 6), 0.0).astype(np.complex64)
    tmpl = FrequencySeries(amp, delta_f=delta_f)
    psd = FrequencySeries(
        (1.0 + (freqs / 200.0) ** 2).astype(np.float32), delta_f=delta_f
    )

    s_ref = sigmasq_series(tmpl, psd, 30.0, 1000.0)
    bins_ref = power_chisq_bins_from_sigmasq_series(s_ref, 16, kmin, kmax)

    bank_plan = prepare_bank(
        [tmpl], tile_size=1, f_lower=30.0, f_upper=1000.0, device=device
    )
    psd_plan = bind_psd(bank_plan, psd, device=device)
    plan = prepare_power_chisq_plan(bank_plan, psd_plan, num_bins=16, device=device)

    actual_bins = plan.tile_bin_edges[0][0]
    if hasattr(actual_bins, "cpu"):
        actual_bins = actual_bins.cpu().numpy()
    np.testing.assert_array_equal(actual_bins, bins_ref)


def test_veto_manager_sg_evaluator_invocation():
    from unittest.mock import MagicMock
    from pycbc.filter.gpu_search.vetoes import SineGaussianPlan, VetoManager

    mock_evaluator = MagicMock()
    mock_evaluator.do = True
    mock_evaluator.return_value = np.array([4.2, 7.8], dtype=np.float32)

    sg_plan = SineGaussianPlan(sg_chisq=mock_evaluator)
    veto_mgr = VetoManager(sg_plan=sg_plan)

    candidates = {
        "sample_idx": np.array([100, 200], dtype=np.int64),
        "template_idx": np.array([0, 0], dtype=np.int64),
        "snr": np.array([6.5 + 1j, 8.2 - 2j], dtype=np.complex64),
    }

    res = veto_mgr.evaluate(
        corr_tile=None,
        candidates=candidates,
        tile_id=0,
        tile_norms=np.array([1.0], dtype=np.float32),
        transform_length=1024,
    )

    assert mock_evaluator.called
    assert "sg_chisq" in res
    np.testing.assert_allclose(res["sg_chisq"], [4.2, 7.8])


@pytest.mark.parametrize("device", DEVICES)
def test_prepare_power_chisq_plan_flat_power_small_fixture(device):
    """Verify exact bin edge parity on a small flat-power fixture without edge shifting."""
    delta_f = 1.0
    flen = 65
    tmpl = FrequencySeries(np.ones(flen, dtype=np.complex64), delta_f=delta_f)
    psd = FrequencySeries(np.ones(flen, dtype=np.float32) * 3.0, delta_f=delta_f)

    s_ref = sigmasq_series(tmpl, psd, 5.0, 30.0)
    kmin, kmax = int(5.0 / delta_f), int(30.0 / delta_f)
    bins_ref = power_chisq_bins_from_sigmasq_series(s_ref, 8, kmin, kmax)

    bank_plan = prepare_bank(
        [tmpl], tile_size=1, f_lower=5.0, f_upper=30.0, device=device
    )
    psd_plan = bind_psd(bank_plan, psd, device=device)
    plan = prepare_power_chisq_plan(bank_plan, psd_plan, num_bins=8, device=device)

    actual_bins = plan.tile_bin_edges[0][0]
    if hasattr(actual_bins, "cpu"):
        actual_bins = actual_bins.cpu().numpy()
    np.testing.assert_array_equal(actual_bins, bins_ref)


def test_veto_manager_singledet_sgchisq_compatibility():
    """Verify VetoManager calls SingleDetSGChisq.values() with complete context and propagates failures."""
    from unittest.mock import MagicMock, NonCallableMagicMock
    from pycbc.filter.gpu_search.vetoes import SineGaussianPlan, VetoManager
    from pycbc.types import FrequencySeries

    mock_evaluator = NonCallableMagicMock(spec=["do", "values"])
    mock_evaluator.do = True

    def dummy_values(
        stilde, template, psd, snrv, snr_norm, bchisq, bchisq_dof, indices
    ):
        return np.array([2.5], dtype=np.float32)

    mock_evaluator.values = MagicMock(side_effect=dummy_values)

    sg_plan = SineGaussianPlan(sg_chisq=mock_evaluator)
    veto_mgr = VetoManager(sg_plan=sg_plan)

    stilde = FrequencySeries(np.ones(513, dtype=np.complex64), delta_f=1.0)
    psd = FrequencySeries(np.ones(513, dtype=np.float32) * 2.0, delta_f=1.0)
    tmpl = FrequencySeries(np.ones(513, dtype=np.complex64), delta_f=1.0)
    tmpl.params = type(
        "Params",
        (),
        {"template_hash": "hash_123", "mass1": 20.0, "mass2": 1.4},
    )()
    templates = [tmpl]

    candidates = {
        "sample_idx": np.array([100], dtype=np.int64),
        "template_idx": np.array([0], dtype=np.int64),
        "snr": np.array([8.0 + 2j], dtype=np.complex64),
        "chisq": np.array([1.2], dtype=np.float32),
        "chisq_dof": np.array([30], dtype=np.uint32),
    }

    res = veto_mgr.evaluate(
        corr_tile=None,
        candidates=candidates,
        tile_id=0,
        tile_norms=np.array([0.5], dtype=np.float32),
        transform_length=1024,
        stilde=stilde,
        templates=templates,
        psd_plan=type(
            "PSDPlan",
            (),
            {"psd": psd, "delta_f": 1.0, "psd_data": psd.numpy()},
        )(),
    )

    assert mock_evaluator.values.called
    call_args = mock_evaluator.values.call_args[0]
    # Signature: (stilde, template, psd, snrv, snr_norm, bchisq, bchisq_dof, indices)
    assert call_args[0] is not None and hasattr(
        call_args[0], "delta_f"
    )  # stilde
    assert call_args[1] is tmpl  # template
    np.testing.assert_allclose(call_args[2].numpy(), psd.numpy())
    assert call_args[2].dtype == np.float32
    assert "sg_chisq" in res
    np.testing.assert_allclose(res["sg_chisq"], [2.5])

    # Verify that evaluator failures are propagated rather than silently swallowed
    mock_evaluator.values.side_effect = RuntimeError("SG evaluation failed")
    with pytest.raises(RuntimeError, match="SG evaluation failed"):
        veto_mgr.evaluate(
            corr_tile=None,
            candidates=candidates,
            tile_id=0,
            tile_norms=np.array([0.5], dtype=np.float32),
            transform_length=1024,
            stilde=stilde,
            templates=templates,
            psd_plan=type(
                "PSDPlan",
                (),
                {"psd": psd, "delta_f": 1.0, "psd_data": psd.numpy()},
            )(),
        )


def test_veto_manager_sgchisq_overwhitened_psd4():
    """Verify SingleDetSGChisq receives prepared overwhitened buffer (PSD=4 test: avoids 16x error)."""
    from unittest.mock import MagicMock, NonCallableMagicMock

    mock_evaluator = NonCallableMagicMock(spec=["do", "values"])
    mock_evaluator.do = True

    def dummy_values_psd4(
        stilde, template, psd, snrv, snr_norm, bchisq, bchisq_dof, indices
    ):
        return np.array([44.13423], dtype=np.float32)

    mock_evaluator.values = MagicMock(side_effect=dummy_values_psd4)

    sg_plan = SineGaussianPlan(evaluator=mock_evaluator)
    veto_mgr = VetoManager(sg_plan=sg_plan)

    raw_stilde = FrequencySeries(np.ones(513, dtype=np.complex64) * 4.0, delta_f=1.0)
    # Overwhitened strain: raw / PSD = 4.0 / 4.0 = 1.0
    overwhitened_stilde = FrequencySeries(np.ones(513, dtype=np.complex64) * 1.0, delta_f=1.0)
    psd = FrequencySeries(np.ones(513, dtype=np.float32) * 4.0, delta_f=1.0)

    tmpl = FrequencySeries(np.ones(513, dtype=np.complex64), delta_f=1.0)
    tmpl.id = 0
    tmpl.params = types.SimpleNamespace(template_hash=0)

    candidates = {
        "sample_idx": np.array([100], dtype=np.int64),
        "template_idx": np.array([0], dtype=np.int64),
        "snr": np.array([10.0], dtype=np.complex64),
        "chisq": np.array([1.0], dtype=np.float32),
        "chisq_dof": np.array([30], dtype=np.uint32),
    }

    tile = types.SimpleNamespace(
        tile_id=0,
        template_ids=[0],
        templates=[tmpl],
        template_by_id={0: tmpl},
    )

    # Pass raw stilde AND prepared stilde_buf
    res = veto_mgr.evaluate(
        corr_tile=None,
        candidates=candidates,
        tile_id=0,
        tile_norms=np.array([0.5], dtype=np.float32),
        transform_length=1024,
        stilde=raw_stilde,
        stilde_buf=overwhitened_stilde,
        tile=tile,
        psd_plan=type("PSDPlan", (), {"psd": psd, "delta_f": 1.0, "psd_data": psd.numpy()})(),
    )

    assert mock_evaluator.values.called
    call_args = mock_evaluator.values.call_args[0]
    # Verify that stilde passed to values() is the overwhitened buffer (value 1.0), NOT raw strain (4.0)
    passed_stilde = call_args[0]
    passed_stilde_np = passed_stilde.numpy() if hasattr(passed_stilde, "numpy") else np.asarray(passed_stilde)
    np.testing.assert_allclose(passed_stilde_np, 1.0)
    assert res["sg_chisq"][0] == pytest.approx(44.13423, rel=1e-4)


def test_veto_manager_sgchisq_template_id_mapping():
    """Verify template IDs [100, 101] are looked up correctly by ID in SG veto evaluation."""
    from unittest.mock import MagicMock, NonCallableMagicMock

    mock_evaluator = NonCallableMagicMock(spec=["do", "values"])
    mock_evaluator.do = True

    def dummy_values_id(
        stilde, template, psd, snrv, snr_norm, bchisq, bchisq_dof, indices
    ):
        return np.array([1.5], dtype=np.float32)

    mock_evaluator.values = MagicMock(side_effect=dummy_values_id)

    sg_plan = SineGaussianPlan(evaluator=mock_evaluator)
    veto_mgr = VetoManager(sg_plan=sg_plan)

    stilde = FrequencySeries(np.ones(513, dtype=np.complex64), delta_f=1.0)
    psd = FrequencySeries(np.ones(513, dtype=np.float32), delta_f=1.0)

    tmpl100 = FrequencySeries(np.ones(513, dtype=np.complex64) * 1.0, delta_f=1.0)
    tmpl100.id = 100
    tmpl100.params = types.SimpleNamespace(template_hash=100)

    tmpl101 = FrequencySeries(np.ones(513, dtype=np.complex64) * 2.0, delta_f=1.0)
    tmpl101.id = 101
    tmpl101.params = types.SimpleNamespace(template_hash=101)

    tile = types.SimpleNamespace(
        tile_id=0,
        template_ids=[100, 101],
        templates=[tmpl100, tmpl101],
        template_by_id={100: tmpl100, 101: tmpl101},
    )
    bank_plan = types.SimpleNamespace(
        template_by_id={100: tmpl100, 101: tmpl101},
        templates=[tmpl100, tmpl101],
    )

    candidates = {
        "sample_idx": np.array([50, 60], dtype=np.int64),
        "template_idx": np.array([0, 1], dtype=np.int64),
        "snr": np.array([8.0, 9.0], dtype=np.complex64),
        "chisq": np.array([1.1, 1.2], dtype=np.float32),
        "chisq_dof": np.array([30, 30], dtype=np.uint32),
    }

    # Should evaluate without ValueError: template 100/101 not found
    res = veto_mgr.evaluate(
        corr_tile=None,
        candidates=candidates,
        tile_id=0,
        tile_norms=np.array([0.5, 0.5], dtype=np.float32),
        transform_length=1024,
        stilde=stilde,
        stilde_buf=stilde,
        tile=tile,
        bank_plan=bank_plan,
        psd_plan=type("PSDPlan", (), {"psd": psd, "delta_f": 1.0, "psd_data": psd.numpy()})(),
    )
    assert "sg_chisq" in res
    assert len(res["sg_chisq"]) == 2


def test_prepare_power_chisq_plan_template_id_mapping():
    """Verify prepare_power_chisq_plan correctly resolves templates by ID for out-of-order and non-contiguous IDs."""
    flen = 513
    delta_f = 1.0

    # 1. Non-contiguous IDs: [100, 101]
    tmpl100 = FrequencySeries(np.ones(flen, dtype=np.complex64), delta_f=delta_f)
    tmpl100.id = 100
    tmpl100.params = types.SimpleNamespace(mass1=15.0, mass2=1.4)

    tmpl101 = FrequencySeries(np.ones(flen, dtype=np.complex64), delta_f=delta_f)
    tmpl101.id = 101
    tmpl101.params = types.SimpleNamespace(mass1=25.0, mass2=1.4)

    bp1 = prepare_bank([tmpl100, tmpl101], tile_size=2, device="cpu")
    psd = FrequencySeries(np.ones(flen, dtype=np.float32), delta_f=delta_f)
    psd_plan1 = bind_psd(bp1, psd, device="cpu")

    num_bins_expr = "4 if params.mass1 <= 20 else 8"
    plan1 = prepare_power_chisq_plan(
        bp1, psd_plan1, num_bins=num_bins_expr, device="cpu"
    )
    nbins1 = plan1.tile_num_bins[0]
    if hasattr(nbins1, "numpy"):
        nbins1 = nbins1.numpy()
    np.testing.assert_array_equal(nbins1, [4, 8])

    # 2. Permuted IDs: [1, 0] with template at index 0 having ID 1 (mass 15 -> 4 bins)
    # and template at index 1 having ID 0 (mass 25 -> 8 bins)
    tmpl_id1 = FrequencySeries(np.ones(flen, dtype=np.complex64), delta_f=delta_f)
    tmpl_id1.id = 1
    tmpl_id1.params = types.SimpleNamespace(mass1=15.0, mass2=1.4)

    tmpl_id0 = FrequencySeries(np.ones(flen, dtype=np.complex64), delta_f=delta_f)
    tmpl_id0.id = 0
    tmpl_id0.params = types.SimpleNamespace(mass1=25.0, mass2=1.4)

    bp2 = prepare_bank([tmpl_id1, tmpl_id0], tile_size=2, device="cpu")
    psd_plan2 = bind_psd(bp2, psd, device="cpu")

    plan2 = prepare_power_chisq_plan(
        bp2, psd_plan2, num_bins=num_bins_expr, device="cpu"
    )
    nbins2 = plan2.tile_num_bins[0]
    if hasattr(nbins2, "numpy"):
        nbins2 = nbins2.numpy()
    # Template ID 1 (mass1=15 -> 4 bins), Template ID 0 (mass1=25 -> 8 bins)
    np.testing.assert_array_equal(nbins2, [4, 8])


def test_engine_sgchisq_preserves_frequencies_above_cutoff():
    """Verify engine passes un-truncated overwhitened strain above f_upper to SG evaluator."""
    from unittest.mock import MagicMock, NonCallableMagicMock

    flen = 513
    delta_f = 1.0
    f_lower = 20.0
    f_upper = 128.0  # matched-filter band is capped at 128 Hz (bin 128)

    # Template
    h = np.zeros(flen, dtype=np.complex64)
    h[20:128] = 1.0
    tmpl = FrequencySeries(h, delta_f=delta_f)
    tmpl.id = 0
    tmpl.params = types.SimpleNamespace(mass1=1.4, mass2=1.4, template_hash=0)

    # Strain with in-band signal shifted to sample 512 and high-frequency content above 128 Hz (at 200 Hz)
    k_vec = np.arange(flen)
    shift = np.exp(-2j * np.pi * k_vec * 512 / 1024).astype(np.complex64)
    data = np.zeros(flen, dtype=np.complex64)
    data[20:128] = shift[20:128] * 10.0
    data[200] = 5.0 + 3.0j
    stilde = FrequencySeries(data, delta_f=delta_f)

    psd = FrequencySeries(np.ones(flen, dtype=np.float32), delta_f=delta_f)

    bank_plan = prepare_bank(
        [tmpl], tile_size=1, f_lower=f_lower, f_upper=f_upper, device="cpu"
    )
    psd_plan = bind_psd(bank_plan, psd, device="cpu")

    captured_stilde = []

    def mock_values(st, template, psd, snrv, snr_norm, bchisq, bchisq_dof, indices):
        captured_stilde.append(np.array(st))
        return np.array([47.8434], dtype=np.float32)

    mock_evaluator = NonCallableMagicMock(spec=["do", "values"])
    mock_evaluator.do = True
    mock_evaluator.values = MagicMock(side_effect=mock_values)

    sg_plan = SineGaussianPlan(evaluator=mock_evaluator)
    veto_mgr = VetoManager(sg_plan=sg_plan)

    engine = SearchEngine(
        bank_plan,
        selection_policy=SelectionPolicy(snr_threshold=5.0, cluster_policy="none"),
        veto_manager=veto_mgr,
        device="cpu",
    )

    valid_interval = (256, 768)
    engine.submit(stilde, psd_plan, valid_interval, block_id=0)
    ready = engine.drain()
    engine.close()

    assert len(ready) == 1
    assert len(captured_stilde) > 0
    st_passed = captured_stilde[0]
    # Verify frequencies above f_upper (128 Hz) were NOT zeroed out for SG evaluation!
    assert np.abs(st_passed[200]) > 0.0, "High frequency content above f_upper was wiped out!"


def test_sgchisq_uses_bound_psd_precision_and_cache_identity():
    """Verify VetoManager uses the bound float32 PSD for SG chisq and maintains cache identity."""
    from pycbc.vetoes.sgchisq import SingleDetSGChisq

    flen = 1025
    delta_f = 1.0
    raw_psd_data = np.full(flen, 4e-28, dtype=np.float64)
    raw_psd1 = FrequencySeries(raw_psd_data, delta_f=delta_f)
    raw_psd2 = FrequencySeries(raw_psd_data * 2.0, delta_f=delta_f)

    tmpl = FrequencySeries(np.ones(flen, dtype=np.complex64), delta_f=delta_f, epoch=0.0)
    tmpl.id = 0
    tmpl.f_lower = 20.0
    tmpl.params = types.SimpleNamespace(template_hash=0, mass1=25.0, mass2=1.4)

    bank_plan = prepare_bank([tmpl], tile_size=1, f_lower=20.0, f_upper=500.0, device="cpu")
    psd_plan1 = bind_psd(bank_plan, raw_psd1, device="cpu")
    psd_plan2 = bind_psd(bank_plan, raw_psd2, device="cpu")

    class MockBankTable:
        def parse_boolargs(self, args):
            return [np.array([1])]

        def __getitem__(self, key):
            if key == "template_hash":
                return np.array([0])
            raise KeyError(key)

    mock_bank = types.SimpleNamespace(table=MockBankTable())
    evaluator = SingleDetSGChisq(
        mock_bank, num_bins="8", snr_threshold=5.0, chisq_locations=["mtotal>20:10-0"]
    )
    sg_plan = SineGaussianPlan(evaluator=evaluator)
    veto_mgr = VetoManager(sg_plan=sg_plan)

    cand_dict = {
        "template_id": np.array([0]),
        "template_idx": np.array([0]),
        "sample_idx": np.array([512]),
        "snr": np.array([20.0]),
        "chisq": np.array([10.0]),
        "chisq_dof": np.array([16]),
    }
    stilde = FrequencySeries(np.zeros(flen, dtype=np.complex64), delta_f=delta_f)

    # First evaluation with psd_plan1: should succeed without float64 precision error
    res1 = veto_mgr.evaluate(
        corr_tile=None,
        candidates=cand_dict,
        tile_id=0,
        tile_norms=np.array([1.0], dtype=np.float32),
        transform_length=2048,
        stilde=stilde,
        psd_plan=psd_plan1,
        tile=bank_plan.tiles[0],
        bank_plan=bank_plan,
        templates=[tmpl],
    )
    assert "sg_chisq" in res1
    bound_psd1 = psd_plan1._bound_psd_fseries
    assert bound_psd1 is not None
    assert bound_psd1.dtype == np.float32

    # Second evaluation with the same psd_plan1: must preserve cache identity
    res1_again = veto_mgr.evaluate(
        corr_tile=None,
        candidates=cand_dict,
        tile_id=0,
        tile_norms=np.array([1.0], dtype=np.float32),
        transform_length=2048,
        stilde=stilde,
        psd_plan=psd_plan1,
        tile=bank_plan.tiles[0],
        bank_plan=bank_plan,
        templates=[tmpl],
    )
    assert "sg_chisq" in res1_again
    assert psd_plan1._bound_psd_fseries is bound_psd1

    # Switch to psd_plan2: must update cache identity
    res2 = veto_mgr.evaluate(
        corr_tile=None,
        candidates=cand_dict,
        tile_id=0,
        tile_norms=np.array([1.0], dtype=np.float32),
        transform_length=2048,
        stilde=stilde,
        psd_plan=psd_plan2,
        tile=bank_plan.tiles[0],
        bank_plan=bank_plan,
        templates=[tmpl],
    )
    assert "sg_chisq" in res2
    bound_psd2 = psd_plan2._bound_psd_fseries
    assert bound_psd2 is not None
    assert bound_psd2 is not bound_psd1


def test_sgchisq_fallback_paths_native_evaluator():
    """Verify both previously supported PSD fallback paths work with native SingleDetSGChisq."""
    from pycbc.vetoes.sgchisq import SingleDetSGChisq

    flen = 1025
    delta_f = 1.0
    raw_psd = FrequencySeries(np.full(flen, 4e-28, dtype=np.float64), delta_f=delta_f)

    h = np.zeros(flen, dtype=np.complex64)
    h[20:200] = 1.0 / np.sqrt(180)
    tmpl = FrequencySeries(h, delta_f=delta_f, epoch=0.0)
    tmpl.id = 0
    tmpl.f_lower = 20.0
    tmpl.params = types.SimpleNamespace(template_hash=0, mass1=25.0, mass2=1.4)

    class MockBankTable:
        def parse_boolargs(self, args):
            return [np.array([1])]

        def __getitem__(self, key):
            if key == "template_hash":
                return np.array([0])
            raise KeyError(key)

    mock_bank = types.SimpleNamespace(table=MockBankTable())
    evaluator = SingleDetSGChisq(
        mock_bank, num_bins="8", snr_threshold=5.0, chisq_locations=["mtotal>20:10-0"]
    )
    sg_plan = SineGaussianPlan(evaluator=evaluator)
    veto_mgr = VetoManager(sg_plan=sg_plan)

    cand_dict = {
        "template_id": np.array([0]),
        "template_idx": np.array([0]),
        "sample_idx": np.array([512]),
        "snr": np.array([20.0 + 0j]),
        "chisq": np.array([16.0]),
        "chisq_dof": np.array([16]),
    }

    # Fallback Path 1: stilde.psd is provided, psd_plan is None
    stilde1 = FrequencySeries(np.zeros(flen, dtype=np.complex64), delta_f=delta_f)
    stilde1.psd = raw_psd
    res1 = veto_mgr.evaluate(
        corr_tile=None,
        candidates=cand_dict,
        tile_id=0,
        tile_norms=np.array([1.0], dtype=np.float32),
        transform_length=2048,
        stilde=stilde1,
        psd_plan=None,
        templates=[tmpl],
    )
    assert "sg_chisq" in res1
    assert len(res1["sg_chisq"]) == 1
    assert np.isfinite(res1["sg_chisq"][0])

    # Fallback Path 2: psd_plan has .psd but no .psd_data, stilde.psd is None
    stilde2 = FrequencySeries(np.zeros(flen, dtype=np.complex64), delta_f=delta_f)
    fake_psd_plan = types.SimpleNamespace(psd=raw_psd, delta_f=delta_f)
    res2 = veto_mgr.evaluate(
        corr_tile=None,
        candidates=cand_dict,
        tile_id=0,
        tile_norms=np.array([1.0], dtype=np.float32),
        transform_length=2048,
        stilde=stilde2,
        psd_plan=fake_psd_plan,
        templates=[tmpl],
    )
    assert "sg_chisq" in res2
    assert len(res2["sg_chisq"]) == 1
    assert np.isfinite(res2["sg_chisq"][0])


def test_sgchisq_scale_consistency_with_dynamic_range():
    """Verify that SG chisq evaluation with dynamic-range scaling produces physical nonzero values."""
    from pycbc.vetoes.sgchisq import SingleDetSGChisq

    flen = 1025
    delta_f = 1.0

    raw_psd_data = np.full(flen, 4e-28, dtype=np.float64)
    raw_psd = FrequencySeries(raw_psd_data, delta_f=delta_f)

    h = np.zeros(flen, dtype=np.complex64)
    h[20:200] = 1.0 / np.sqrt(180)
    tmpl = FrequencySeries(h, delta_f=delta_f, epoch=0.0)
    tmpl.id = 0
    tmpl.f_lower = 20.0
    tmpl.params = types.SimpleNamespace(template_hash=0, mass1=25.0, mass2=1.4)

    bank_plan = prepare_bank([tmpl], tile_size=1, f_lower=20.0, f_upper=500.0, device="cpu")
    psd_plan = bind_psd(bank_plan, raw_psd, device="cpu")
    dyn_range = float(psd_plan.dyn_range_factor)

    class MockBankTable:
        def parse_boolargs(self, args):
            return [np.array([1])]

        def __getitem__(self, key):
            if key == "template_hash":
                return np.array([0])
            raise KeyError(key)

    mock_bank = types.SimpleNamespace(table=MockBankTable())
    evaluator = SingleDetSGChisq(
        mock_bank, num_bins="8", snr_threshold=5.0, chisq_locations=["mtotal>20:10-0"]
    )
    sg_plan = SineGaussianPlan(evaluator=evaluator)
    veto_mgr = VetoManager(sg_plan=sg_plan)

    cand_dict = {
        "template_id": np.array([0]),
        "template_idx": np.array([0]),
        "sample_idx": np.array([512]),
        "snr": np.array([20.0 + 0j]),
        "chisq": np.array([16.0]),
        "chisq_dof": np.array([16]),
    }

    # Physical scaled strain (s / (Sn * D)) with raw strain ~ 1e-14
    raw_strain = np.zeros(flen, dtype=np.complex64)
    raw_strain[20:200] = 1e-14
    scaled_stilde = FrequencySeries(
        (raw_strain / (raw_psd_data * dyn_range)).astype(np.complex64),
        delta_f=delta_f,
    )

    res = veto_mgr.evaluate(
        corr_tile=None,
        candidates=cand_dict,
        tile_id=0,
        tile_norms=np.array([1.0], dtype=np.float32),
        transform_length=2048,
        stilde=scaled_stilde,
        psd_plan=psd_plan,
        tile=bank_plan.tiles[0],
        bank_plan=bank_plan,
        templates=[tmpl],
    )

    sg_val = float(res["sg_chisq"][0])
    assert np.isfinite(sg_val)
    # Physical reference value is O(1e-3), NOT suppressed to 10^-31
    assert sg_val > 1e-6, f"SG chisq unexpectedly suppressed: {sg_val}"
    assert sg_val == pytest.approx(0.0053437, rel=0.05)
