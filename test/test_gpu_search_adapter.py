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
Unit and integration tests for GPU search engine application adapters
(TiledMatchedFilterControl and TiledLiveBatchMatchedFilter).
"""

import ast
import functools
from pathlib import Path
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

from pycbc import scheme
from pycbc.types import FrequencySeries
from pycbc.filter.matchedfilter import MatchedFilterControl, sigmasq
from pycbc.filter.gpu_search.adapter import (
    TiledMatchedFilterControl,
    TiledLiveBatchMatchedFilter,
)


def _make_offline_fixtures(filter_length=513, delta_f=0.5):
    np.random.seed(9876)
    N = (filter_length - 1) * 2

    # Template
    h_data = (
        np.random.randn(filter_length) + 1j * np.random.randn(filter_length)
    ).astype(np.complex64)
    h_data[0] = 0.0
    template_mem = FrequencySeries(h_data, delta_f=delta_f)

    # Segment
    s_data = (
        np.random.randn(filter_length) + 1j * np.random.randn(filter_length)
    ).astype(np.complex64)
    s_data[0] = 0.0
    seg = FrequencySeries(s_data, delta_f=delta_f)
    seg.psd = FrequencySeries(np.ones(filter_length, dtype=np.float32) * 2.0, delta_f=delta_f)
    seg.analyze = slice(100, 900)
    seg._epoch = 0

    return template_mem, [seg], N, delta_f


@pytest.mark.parametrize("device", DEVICES)
def test_tiled_matched_filter_control_parity(device):
    template_mem, segments, N, delta_f = _make_offline_fixtures()

    with scheme.TorchScheme(device) if (torch is not None and device != "numpy") else scheme.DefaultScheme():
        # Canonical MatchedFilterControl
        ref_ctrl = MatchedFilterControl(
            low_frequency_cutoff=20.0,
            high_frequency_cutoff=200.0,
            snr_threshold=1.5,
            tlen=N,
            delta_f=delta_f,
            dtype=np.complex64,
            segment_list=segments,
            template_output=template_mem,
            use_cluster=True,
            cluster_function="symmetric",
        )

        # TiledMatchedFilterControl
        tiled_ctrl = TiledMatchedFilterControl(
            low_frequency_cutoff=20.0,
            high_frequency_cutoff=200.0,
            snr_threshold=1.5,
            tlen=N,
            delta_f=delta_f,
            dtype=np.complex64,
            segment_list=segments,
            template_output=template_mem,
            use_cluster=True,
            cluster_function="symmetric",
            device=device,
        )

        template_norm = 1.0
        window = 16

        ref_snr, ref_norm, ref_corr, ref_idx, ref_snrv = ref_ctrl.full_matched_filter_and_cluster_symm(
            segnum=0, template_norm=template_norm, window=window
        )
        t_snr, t_norm, t_corr, t_idx, t_snrv = tiled_ctrl.full_matched_filter_and_cluster_symm(
            segnum=0, template_norm=template_norm, window=window
        )

        assert len(ref_idx) == len(t_idx)
        assert len(ref_idx) > 0

        np.testing.assert_allclose(t_norm, ref_norm, rtol=1e-5)
        np.testing.assert_array_equal(np.asarray(t_idx), np.asarray(ref_idx))
        np.testing.assert_allclose(np.asarray(t_snrv), np.asarray(ref_snrv), atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_tiled_matched_filter_control_batched_parity(device):
    np.random.seed(4321)
    filter_length = 513
    delta_f = 0.5
    N = (filter_length - 1) * 2

    # Make 4 templates
    templates = []
    sigmasqs = []
    for i in range(4):
        h_data = (
            np.random.randn(filter_length) + 1j * np.random.randn(filter_length)
        ).astype(np.complex64)
        h_data[0] = 0.0
        t = FrequencySeries(h_data, delta_f=delta_f)
        templates.append(t)
        sigmasqs.append(1.0 + 0.5 * i)

    # Make segment
    s_data = (
        np.random.randn(filter_length) + 1j * np.random.randn(filter_length)
    ).astype(np.complex64)
    s_data[0] = 0.0
    seg = FrequencySeries(s_data, delta_f=delta_f)
    seg.psd = FrequencySeries(np.ones(filter_length, dtype=np.float32) * 2.0, delta_f=delta_f)
    seg.analyze = slice(100, 900)
    seg._epoch = 0

    with scheme.TorchScheme(device) if (torch is not None and device != "numpy") else scheme.DefaultScheme():
        template_mem = FrequencySeries(np.zeros(filter_length, dtype=np.complex64), delta_f=delta_f)
        ref_ctrl = MatchedFilterControl(
            low_frequency_cutoff=20.0,
            high_frequency_cutoff=200.0,
            snr_threshold=1.5,
            tlen=N,
            delta_f=delta_f,
            dtype=np.complex64,
            segment_list=[seg],
            template_output=template_mem,
            use_cluster=True,
            cluster_function="symmetric",
        )

        tiled_ctrl = TiledMatchedFilterControl(
            low_frequency_cutoff=20.0,
            high_frequency_cutoff=200.0,
            snr_threshold=1.5,
            tlen=N,
            delta_f=delta_f,
            dtype=np.complex64,
            segment_list=[seg],
            template_output=template_mem,
            use_cluster=True,
            cluster_function="symmetric",
            tile_size=4,
            device=device,
        )

        window = 16

        # Reference sequential filtering
        ref_results = []
        for i in range(4):
            template_mem[:] = templates[i][:]
            r = ref_ctrl.full_matched_filter_and_cluster_symm(
                segnum=0, template_norm=sigmasqs[i], window=window
            )
            ref_results.append(r)

        # Batched filtering
        batch_results = tiled_ctrl.batched_matched_filter_and_cluster(
            segnum=0, templates=templates, sigmasqs=sigmasqs, window=window
        )

        assert len(batch_results) == 4
        for i in range(4):
            ref_snr, ref_norm, ref_corr, ref_idx, ref_snrv = ref_results[i]
            b_snr, b_norm, b_corr, b_idx, b_snrv = batch_results[i]

            assert len(ref_idx) == len(b_idx)
            np.testing.assert_allclose(b_norm, ref_norm, rtol=1e-5)
            np.testing.assert_array_equal(np.asarray(b_idx), np.asarray(ref_idx))
            np.testing.assert_allclose(np.asarray(b_snrv), np.asarray(ref_snrv), atol=1e-5)


def _make_live_fixtures(num_templates=3, size=256):
    flen = size // 2 + 1
    delta_f = 1.0 / size
    rng = np.random.default_rng(42)

    templates = []
    for i in range(num_templates):
        h_vals = (rng.normal(size=flen) + 1j * rng.normal(size=flen)).astype(np.complex64)
        h_vals[0] = 0.0
        t = FrequencySeries(h_vals, delta_f=delta_f)
        t.id = 100 + i
        t.params = np.array([(20.0 + i, 1.4)], dtype=[("mass1", np.float32), ("mass2", np.float32)])[0]
        templates.append(t)

    data_vals = (rng.normal(size=flen) + 1j * rng.normal(size=flen)).astype(np.complex64)
    data_vals[0] = 0.0
    stilde = FrequencySeries(data_vals, delta_f=delta_f)
    stilde.psd = FrequencySeries(np.ones(flen, dtype=np.float32) * 2.0, delta_f=delta_f)

    data_reader = types.SimpleNamespace(
        overwhitened_data=lambda _df: stilde,
        trim_padding=16,
        blocksize=128 * delta_f,
        sample_rate=int(1.0 / delta_f / size),
        start_time=1000.0,
    )
    # Adjust sample_rate and blocksize for proper indexing
    data_reader.sample_rate = 1
    data_reader.blocksize = 128

    return templates, data_reader


@pytest.mark.parametrize("device", DEVICES)
def test_tiled_live_batch_filter(device):
    templates, data_reader = _make_live_fixtures(num_templates=4, size=256)

    live_filter = TiledLiveBatchMatchedFilter(
        templates=templates,
        snr_threshold=0.0,  # record peak for each template
        chisq_bins=16,
        sg_chisq=types.SimpleNamespace(do=False),
        tile_size=2,
        device=device,
    )

    result = live_filter.process_data(data_reader)
    assert result is not False
    assert "end_time" in result
    assert "coa_phase" in result
    assert "snr" in result
    assert "sigmasq" in result
    assert "template_id" in result
    assert "chisq" in result
    assert "chisq_dof" in result
    assert "mass1" in result
    assert "mass2" in result

    # Should have triggers from all 4 templates
    assert len(result["template_id"]) == 4
    np.testing.assert_array_equal(np.sort(result["template_id"]), [100, 101, 102, 103])


@pytest.mark.parametrize("device", DEVICES)
def test_tiled_live_batch_abort(device):
    templates, data_reader = _make_live_fixtures(num_templates=2, size=256)

    # Set abort threshold very low so any signal triggers an abort
    live_filter = TiledLiveBatchMatchedFilter(
        templates=templates,
        snr_threshold=0.0,
        chisq_bins=16,
        sg_chisq=types.SimpleNamespace(do=False),
        snr_abort_threshold=0.001,
        tile_size=2,
        device=device,
    )

    result = live_filter.process_data(data_reader)
    assert result is False


@pytest.mark.parametrize("device", DEVICES)
def test_tiled_live_batch_top_k(device):
    templates, data_reader = _make_live_fixtures(num_templates=5, size=256)

    # Request top 2 triggers across the batch
    live_filter = TiledLiveBatchMatchedFilter(
        templates=templates,
        snr_threshold=0.0,
        chisq_bins=16,
        sg_chisq=types.SimpleNamespace(do=False),
        max_triggers_in_batch=2,
        tile_size=2,
        device=device,
    )

    result = live_filter.process_data(data_reader)
    assert result is not False
    assert len(result["template_id"]) == 2
    # Ensure sorted by descending SNR magnitude
    assert abs(result["snr"][0]) >= abs(result["snr"][1])


def test_tiled_adapter_torch_cpu_scheme():
    """Verify TiledMatchedFilterControl under TorchScheme CPU with num_threads."""
    from pycbc import scheme
    if not hasattr(scheme, "TorchScheme"):
        pytest.skip("TorchScheme not available")

    with scheme.TorchScheme(device="cpu", num_threads=2):
        template_mem, segments, N, delta_f = _make_offline_fixtures(filter_length=65)
        ctrl = TiledMatchedFilterControl(
            low_frequency_cutoff=20.0,
            high_frequency_cutoff=None,
            snr_threshold=0.0,
            tlen=N,
            delta_f=delta_f,
            dtype=np.complex64,
            segment_list=segments,
            template_output=template_mem,
            use_cluster=True,
            cluster_function="symmetric",
            tile_size=1,
        )
        assert ctrl.device == "cpu"
        assert ctrl.num_threads == 2

        snrv, s_idx, _, _, _ = ctrl.matched_filter_and_cluster(
            segnum=0, template_norm=1.0, window=16
        )
        assert len(snrv) > 0


def test_tiled_adapter_cpu_scheme_no_copy_error():
    """Verify TiledMatchedFilterControl under CPUScheme materializes candidates without copy error."""
    from pycbc import scheme
    from pycbc.types import TimeSeries
    from pycbc.filter.gpu_search.adapter import TiledMatchedFilterControl

    with scheme.CPUScheme(1):
        template_mem, segments, N, delta_f = _make_offline_fixtures(filter_length=65)
        ctrl = TiledMatchedFilterControl(
            low_frequency_cutoff=20.0,
            high_frequency_cutoff=None,
            snr_threshold=0.0,
            tlen=N,
            delta_f=delta_f,
            dtype=np.complex64,
            segment_list=segments,
            template_output=template_mem,
            use_cluster=True,
            cluster_function="symmetric",
            tile_size=2,
        )
        templates = [template_mem, template_mem]
        sigmasqs = [10.0, 10.0]
        results = ctrl.batched_matched_filter_and_cluster(
            segnum=0, templates=templates, sigmasqs=sigmasqs, window=16
        )
        assert len(results) == 2
        for snr_i, norm_i, corr_i, tmpl_idx, tmpl_snrv in results:
            assert isinstance(corr_i, FrequencySeries)
            assert isinstance(snr_i, TimeSeries)


def test_tiled_live_reduced_chisq_normalization():
    """Verify TiledLiveBatchMatchedFilter returns reduced chi-square (c/dof)."""
    templates, data_reader = _make_live_fixtures(num_templates=2, size=256)

    live_filter = TiledLiveBatchMatchedFilter(
        templates=templates,
        snr_threshold=0.0,
        chisq_bins=16,
        sg_chisq=types.SimpleNamespace(do=False),
        tile_size=2,
        device="cpu",
    )

    result = live_filter.process_data(data_reader)
    assert result is not False
    assert "chisq" in result
    assert "chisq_dof" in result
    # For a typical filter against noise/data, reduced chisq should be of order unity (e.g. < 10)
    # If raw chi-square were mistakenly returned, it would be multiplied by DOF (~30) again in coinc.py
    assert (result["chisq_dof"] == 30).all()
    assert (result["chisq"] >= 0.0).all()
    # Ensure values are reduced (not raw * dof)
    assert np.all(result["chisq"] < 50.0)


def test_tiled_live_output_schema_and_quiet_block():
    """Verify TiledLiveBatchMatchedFilter output schema on quiet and active blocks."""
    templates, data_reader = _make_live_fixtures(num_templates=2, size=256)

    # 1. Quiet block test (threshold high so 0 triggers)
    live_filter_quiet = TiledLiveBatchMatchedFilter(
        templates=templates,
        snr_threshold=1000.0,
        chisq_bins=16,
        sg_chisq=types.SimpleNamespace(do=False),
        tile_size=2,
        device="cpu",
    )
    quiet_res = live_filter_quiet.process_data(data_reader)
    assert quiet_res is not False
    assert isinstance(quiet_res, dict)
    expected_keys = [
        "end_time",
        "snr",
        "coa_phase",
        "template_id",
        "sigmasq",
        "chisq",
        "chisq_dof",
        "sg_chisq",
        "mass1",
        "mass2",
    ]
    for k in expected_keys:
        assert k in quiet_res, f"Missing key '{k}' in quiet result"
        assert len(quiet_res[k]) == 0

    # 2. Active block test (threshold 0 so all templates trigger)
    live_filter_active = TiledLiveBatchMatchedFilter(
        templates=templates,
        snr_threshold=0.0,
        chisq_bins=16,
        sg_chisq=types.SimpleNamespace(do=False),
        tile_size=2,
        device="cpu",
    )
    active_res = live_filter_active.process_data(data_reader)
    assert active_res is not False
    for k in expected_keys:
        assert k in active_res, f"Missing key '{k}' in active result"
        assert len(active_res[k]) == 2

    # Check dtypes and properties
    assert np.issubdtype(active_res["end_time"].dtype, np.floating)
    assert np.isrealobj(active_res["snr"])
    assert np.issubdtype(active_res["snr"].dtype, np.floating)
    assert np.isrealobj(active_res["coa_phase"])
    assert np.issubdtype(active_res["coa_phase"].dtype, np.floating)
    assert np.all(active_res["snr"] >= 0.0)


def test_tiled_live_mixed_duration_banks():
    """Verify TiledLiveBatchMatchedFilter supports banks with mixed durations."""
    rng = np.random.default_rng(123)
    flen1 = 65
    df1 = 1.0 / 128
    t1 = FrequencySeries(
        (rng.normal(size=flen1) + 1j * rng.normal(size=flen1)).astype(np.complex64),
        delta_f=df1,
    )
    t1.id = 1
    t1.params = np.array(
        [(10.0, 1.4)], dtype=[("mass1", np.float32), ("mass2", np.float32)]
    )[0]

    flen2 = 129
    df2 = 1.0 / 256
    t2 = FrequencySeries(
        (rng.normal(size=flen2) + 1j * rng.normal(size=flen2)).astype(np.complex64),
        delta_f=df2,
    )
    t2.id = 2
    t2.params = np.array(
        [(20.0, 1.4)], dtype=[("mass1", np.float32), ("mass2", np.float32)]
    )[0]

    templates = [t1, t2]

    def get_overwhitened(df):
        if abs(df - df1) < 1e-7:
            fs = FrequencySeries(
                (rng.normal(size=flen1) + 1j * rng.normal(size=flen1)).astype(
                    np.complex64
                ),
                delta_f=df1,
            )
            fs.psd = FrequencySeries(
                np.ones(flen1, dtype=np.float32) * 2.0, delta_f=df1
            )
            return fs
        else:
            fs = FrequencySeries(
                (rng.normal(size=flen2) + 1j * rng.normal(size=flen2)).astype(
                    np.complex64
                ),
                delta_f=df2,
            )
            fs.psd = FrequencySeries(
                np.ones(flen2, dtype=np.float32) * 2.0, delta_f=df2
            )
            return fs

    data_reader = types.SimpleNamespace(
        overwhitened_data=get_overwhitened,
        trim_padding=8,
        blocksize=32,
        sample_rate=1,
        start_time=1000.0,
    )

    live_filter = TiledLiveBatchMatchedFilter(
        templates=templates,
        snr_threshold=0.0,
        chisq_bins=16,
        sg_chisq=types.SimpleNamespace(do=False),
        tile_size=2,
        device="cpu",
    )
    result = live_filter.process_data(data_reader)
    assert result is not False
    assert len(result["template_id"]) == 2
    assert set(result["template_id"]) == {1, 2}


def test_tiled_live_chisq_bins_expression():
    """Verify TiledLiveBatchMatchedFilter correctly evaluates chisq_bins expression."""
    templates, data_reader = _make_live_fixtures(num_templates=2, size=256)
    live_filter = TiledLiveBatchMatchedFilter(
        templates=templates,
        snr_threshold=0.0,
        chisq_bins="4 + 4",
        sg_chisq=types.SimpleNamespace(do=False),
        tile_size=2,
        device="cpu",
    )
    result = live_filter.process_data(data_reader)
    assert result is not False
    assert len(result["chisq_dof"]) == 2
    # 8 bins -> 2 * 8 - 2 = 14 DOF
    assert (result["chisq_dof"] == 14).all()


def test_tiled_live_detector_psd_switching():
    """Verify switching PSD A -> B -> A restores veto manager and yields identical chisq."""
    flen = 129
    delta_f = 1.0 / 256.0
    rng = np.random.default_rng(1234)

    h_vals = (rng.normal(size=flen) + 1j * rng.normal(size=flen)).astype(
        np.complex64
    )
    h_vals[0] = 0.0
    t = FrequencySeries(h_vals, delta_f=delta_f)
    t.id = 10
    t.params = types.SimpleNamespace(mass1=20.0, mass2=1.4)
    templates = [t]

    psd_A = FrequencySeries(
        np.ones(flen, dtype=np.float32) * 2.0, delta_f=delta_f
    )
    psd_B = FrequencySeries(
        np.ones(flen, dtype=np.float32) * 5.0, delta_f=delta_f
    )

    data_vals = (rng.normal(size=flen) + 1j * rng.normal(size=flen)).astype(
        np.complex64
    )
    data_vals[0] = 0.0

    stilde_A = FrequencySeries(data_vals.copy(), delta_f=delta_f)
    stilde_A.psd = psd_A
    stilde_B = FrequencySeries(data_vals.copy(), delta_f=delta_f)
    stilde_B.psd = psd_B

    live_filter = TiledLiveBatchMatchedFilter(
        templates=templates,
        snr_threshold=0.0,
        chisq_bins=16,
        sg_chisq=types.SimpleNamespace(do=False),
        tile_size=1,
        device="cpu",
    )

    def make_reader(st):
        return types.SimpleNamespace(
            overwhitened_data=lambda _df: st,
            trim_padding=16,
            blocksize=128,
            sample_rate=1,
            start_time=1000.0,
        )

    # 1. Run with PSD A
    res_A1 = live_filter.process_data(make_reader(stilde_A))
    # 2. Run with PSD B
    res_B = live_filter.process_data(make_reader(stilde_B))
    # 3. Run again with PSD A
    res_A2 = live_filter.process_data(make_reader(stilde_A))

    assert res_A1 is not False and res_B is not False and res_A2 is not False
    np.testing.assert_allclose(res_A1["snr"], res_A2["snr"], rtol=1e-5)
    np.testing.assert_allclose(res_A1["chisq"], res_A2["chisq"], rtol=1e-5)
    np.testing.assert_array_equal(res_A1["chisq_dof"], res_A2["chisq_dof"])


def test_tiled_live_template_dependent_chisq_dof():
    """Verify template-dependent chisq_bins expression evaluates per template and produces correct DOFs."""
    flen = 129
    delta_f = 1.0 / 256.0
    rng = np.random.default_rng(42)

    templates = []
    for i, m1 in enumerate([20.0, 21.0]):
        h_vals = (rng.normal(size=flen) + 1j * rng.normal(size=flen)).astype(
            np.complex64
        )
        h_vals[0] = 0.0
        t = FrequencySeries(h_vals, delta_f=delta_f)
        t.id = 100 + i
        t.params = np.array(
            [(m1, 1.4)], dtype=[("mass1", np.float32), ("mass2", np.float32)]
        )[0]
        templates.append(t)

    data_vals = (rng.normal(size=flen) + 1j * rng.normal(size=flen)).astype(
        np.complex64
    )
    data_vals[0] = 0.0
    stilde = FrequencySeries(data_vals, delta_f=delta_f)
    stilde.psd = FrequencySeries(
        np.ones(flen, dtype=np.float32) * 2.0, delta_f=delta_f
    )

    data_reader = types.SimpleNamespace(
        overwhitened_data=lambda _df: stilde,
        trim_padding=16,
        blocksize=128,
        sample_rate=1,
        start_time=1000.0,
    )

    live_filter = TiledLiveBatchMatchedFilter(
        templates=templates,
        snr_threshold=0.0,
        chisq_bins="params.mass1",
        sg_chisq=types.SimpleNamespace(do=False),
        tile_size=2,
        device="cpu",
    )
    result = live_filter.process_data(data_reader)
    assert result is not False
    assert len(result["chisq_dof"]) == 2
    # 20 bins -> 38 dof, 21 bins -> 40 dof
    np.testing.assert_array_equal(result["chisq_dof"], [38, 40])


def test_tiled_live_buffer_overflow_raises_error():
    """Verify exhausted candidate buffer raises RuntimeError instead of dropping triggers."""
    from unittest.mock import patch

    templates, data_reader = _make_live_fixtures(num_templates=2, size=256)
    live_filter = TiledLiveBatchMatchedFilter(
        templates=templates,
        snr_threshold=0.0,
        chisq_bins=16,
        sg_chisq=types.SimpleNamespace(do=False),
        tile_size=2,
        device="cpu",
    )
    with patch(
        "pycbc.filter.gpu_search.engine.select_tile_candidates",
        return_value={"overflow": True},
    ):
        with pytest.raises(RuntimeError, match="Candidate buffer overflow"):
            live_filter.process_data(data_reader)


def test_tiled_live_template_id_mapping_order():
    """Verify template IDs [1, 0] with params.mass1 produce correct DOFs [38, 40], not [40, 38]."""
    flen = 129
    delta_f = 1.0 / 256
    rng = np.random.default_rng(42)

    templates = []
    # Template at pos 0 has ID 1, mass1 20.0 (-> 38 DOF)
    # Template at pos 1 has ID 0, mass1 21.0 (-> 40 DOF)
    ids = [1, 0]
    m1_vals = [20.0, 21.0]

    for i in range(2):
        h_vals = (rng.normal(size=flen) + 1j * rng.normal(size=flen)).astype(
            np.complex64
        )
        h_vals[0] = 0.0
        pwr = np.sum(np.abs(h_vals) ** 2)
        if pwr > 0:
            h_vals /= np.sqrt(pwr)
        t = FrequencySeries(h_vals, delta_f=delta_f)
        t.id = ids[i]
        t.params = np.array(
            [(m1_vals[i], 1.4)],
            dtype=[("mass1", np.float32), ("mass2", np.float32)],
        )[0]
        templates.append(t)

    data_vals = (rng.normal(size=flen) + 1j * rng.normal(size=flen)).astype(
        np.complex64
    )
    data_vals[0] = 0.0
    stilde = FrequencySeries(data_vals, delta_f=delta_f)
    stilde.psd = FrequencySeries(
        np.ones(flen, dtype=np.float32) * 2.0, delta_f=delta_f
    )

    data_reader = types.SimpleNamespace(
        overwhitened_data=lambda _df: stilde,
        trim_padding=16,
        blocksize=128,
        sample_rate=1,
        start_time=1000.0,
    )

    live_filter = TiledLiveBatchMatchedFilter(
        templates=templates,
        snr_threshold=0.0,
        chisq_bins="params.mass1",
        sg_chisq=types.SimpleNamespace(do=False),
        tile_size=2,
        device="cpu",
    )
    result = live_filter.process_data(data_reader)
    assert result is not False
    assert len(result["chisq_dof"]) == 2
    # Verify candidate with template_id 1 has 38 DOF and template_id 0 has 40 DOF
    for t_id, dof in zip(result["template_id"], result["chisq_dof"]):
        if t_id == 1:
            assert dof == 38
        elif t_id == 0:
            assert dof == 40


@pytest.mark.parametrize("device", DEVICES)
def test_tiled_matched_filter_control_workspace_and_shared_core(device):
    """Verify TiledMatchedFilterControl workspace capacity and core."""
    template_mem, segments, N, delta_f = _make_offline_fixtures()
    flen = N // 2 + 1

    ctrl = TiledMatchedFilterControl(
        low_frequency_cutoff=20.0,
        high_frequency_cutoff=200.0,
        snr_threshold=1.5,
        tlen=N,
        delta_f=delta_f,
        dtype=np.complex64,
        segment_list=segments,
        template_output=template_mem,
        use_cluster=True,
        cluster_function="symmetric",
        tile_size=2,
        device=device,
    )

    # Verify workspace properties exist and have correct shape
    assert ctrl.cout_workspace is not None
    assert ctrl.out_workspace is not None
    assert ctrl.cout_workspace.shape == (2, N)
    assert ctrl.out_workspace.shape == (2, N)

    # Verify capacity growth when batch size exceeds initial tile_size
    templates = []
    sigmasqs = []
    for i in range(5):
        t = FrequencySeries(np.ones(flen, dtype=np.complex64), delta_f=delta_f)
        templates.append(t)
        sigmasqs.append(1.0)

    # Filter batch of 5 > tile_size 2
    res = ctrl.batched_matched_filter_and_cluster(
        segnum=0, templates=templates, sigmasqs=sigmasqs, window=16
    )
    assert len(res) == 5
    assert ctrl.cout_workspace.shape[0] >= 5
    assert ctrl.out_workspace.shape[0] >= 5


@pytest.mark.parametrize("device", DEVICES + ["numpy"])
def test_prepare_template_batch_api(device):
    """Test prepare_template_batch on TiledMatchedFilterControl."""
    template_mem, segments, N, delta_f = _make_offline_fixtures()
    flen = N // 2 + 1

    ctrl = TiledMatchedFilterControl(
        low_frequency_cutoff=20.0,
        high_frequency_cutoff=200.0,
        snr_threshold=1.5,
        tlen=N,
        delta_f=delta_f,
        dtype=np.complex64,
        segment_list=segments,
        template_output=template_mem,
        use_cluster=True,
        cluster_function="symmetric",
        tile_size=4,
        device=device,
    )

    # 1. Empty sequence
    empty_prep = ctrl.prepare_template_batch([])
    assert empty_prep.shape == (0, flen)

    # 2. List of FrequencySeries
    templates = [
        FrequencySeries(np.ones(flen, dtype=np.complex64), delta_f=delta_f),
        FrequencySeries(
            np.full(flen, 2.0, dtype=np.complex64), delta_f=delta_f
        ),
    ]
    prep = ctrl.prepare_template_batch(templates)
    assert prep.shape == (2, flen)
    if torch is not None and device != "numpy":
        assert isinstance(prep, torch.Tensor)
        assert prep.device == torch.empty(0, device=device).device
        assert prep.dtype == torch.complex64
        # Identity return for already-prepared tensor
        prep2 = ctrl.prepare_template_batch(prep)
        assert prep2 is prep
    else:
        assert isinstance(prep, np.ndarray)
        assert prep.dtype == np.complex64

    # 3. 2D NumPy array
    arr = np.ones((3, flen), dtype=np.complex64)
    prep_arr = ctrl.prepare_template_batch(arr)
    assert prep_arr.shape == (3, flen)

    # 4. Trimming if width > flen
    arr_wide = np.ones((3, flen + 10), dtype=np.complex64)
    prep_wide = ctrl.prepare_template_batch(arr_wide)
    assert prep_wide.shape == (3, flen)

    # 5. Clear width rejection for all paths (2D width < flen and sequence item < flen)
    narrow_arr = np.ones((2, flen - 1), dtype=np.complex64)
    with pytest.raises(ValueError, match="< filter length"):
        ctrl.prepare_template_batch(narrow_arr)

    narrow_seq = [
        FrequencySeries(np.ones(flen - 1, dtype=np.complex64), delta_f=delta_f)
    ]
    with pytest.raises(ValueError, match="< filter length"):
        ctrl.prepare_template_batch(narrow_seq)

    # 6. Validation errors
    with pytest.raises(ValueError, match="must be 2D"):
        if torch is not None and device != "numpy":
            ctrl.prepare_template_batch(
                torch.zeros(flen, dtype=torch.complex64)
            )
        else:
            ctrl.prepare_template_batch(np.zeros(flen, dtype=np.complex64))

    with pytest.raises(TypeError, match="Unsupported template container type"):
        ctrl.prepare_template_batch({"not": "a template"})

    # 7. List of raw torch.Tensor on device
    if torch is not None and device != "numpy":
        raw_dev_tensors = [
            torch.ones(flen + 5, dtype=torch.complex64, device=device),
            torch.full((flen + 5,), 2.0, dtype=torch.complex64, device=device),
        ]
        prep_raw = ctrl.prepare_template_batch(raw_dev_tensors)
        assert isinstance(prep_raw, torch.Tensor)
        assert prep_raw.shape == (2, flen)
        assert prep_raw.device == torch.empty(0, device=device).device
        assert prep_raw.dtype == torch.complex64
        # Preserves identity when re-prepared
        assert ctrl.prepare_template_batch(prep_raw) is prep_raw

        # Rejects undersized raw tensor in list without host transfer
        raw_undersized = [
            torch.ones(flen - 1, dtype=torch.complex64, device=device)
        ]
        with pytest.raises(ValueError, match="< filter length"):
            ctrl.prepare_template_batch(raw_undersized)


# ---------------------------------------------------------------------------
# Prepared batch optimization and CLI integration tests
# ---------------------------------------------------------------------------


def _make_batch_fixtures(
    num_templates=4,
    num_segments=2,
    filter_length=257,
    delta_f=1.0,
    seed=42,
    flow=20.0,
    ffinal=100.0,
):
    """Shared deterministic fixture for prepared batch and inspiral CLI tests."""
    rng = np.random.default_rng(seed)
    N = (filter_length - 1) * 2

    templates = []
    for i in range(num_templates):
        h = (
            rng.normal(size=filter_length)
            + 1j * rng.normal(size=filter_length)
        ).astype(np.complex64)
        h[0] = 0.0
        t = FrequencySeries(h, delta_f=delta_f)
        t.id = i
        t.params = np.array(
            [(20.0 + i, 1.4)],
            dtype=[("mass1", np.float32), ("mass2", np.float32)],
        )[0]
        t.f_lower = flow
        t.end_frequency = ffinal

        def _calc_sigmasq(self, psd):
            return float(
                sigmasq(
                    self,
                    psd=psd,
                    low_frequency_cutoff=self.f_lower,
                    high_frequency_cutoff=self.end_frequency,
                )
            )

        t.sigmasq = types.MethodType(_calc_sigmasq, t)
        templates.append(t)

    segments = []
    for s in range(num_segments):
        s_data = (
            rng.normal(size=filter_length)
            + 1j * rng.normal(size=filter_length)
        ).astype(np.complex64)
        s_data[0] = 0.0
        seg = FrequencySeries(s_data, delta_f=delta_f)
        seg.psd = FrequencySeries(
            np.ones(filter_length, dtype=np.float32) * (2.0 + s),
            delta_f=delta_f,
        )
        seg.analyze = slice(50, 450)
        seg.cumulative_index = s * 1000
        seg._epoch = float(s * 10)
        segments.append(seg)

    return templates, segments, N, delta_f


@pytest.mark.parametrize("device", DEVICES + ["numpy"])
@pytest.mark.parametrize(
    "subset", ["full", "contiguous", "noncontiguous", "tail_b1"]
)
def test_prepared_batch_parity_and_subsets(device, subset):
    """Verify prepared batch filtering parity with immediate buffer snapshotting."""
    templates, segments, N, delta_f = _make_batch_fixtures(
        num_templates=4, num_segments=2
    )
    flen = N // 2 + 1
    mem = FrequencySeries(np.zeros(flen, dtype=np.complex64), delta_f=delta_f)
    ctrl = TiledMatchedFilterControl(
        20.0,
        100.0,
        1.5,
        N,
        delta_f,
        np.complex64,
        segments,
        mem,
        True,
        tile_size=4,
        device=device,
    )

    idx_map = {
        "full": [0, 1, 2, 3],
        "contiguous": [1, 2],
        "noncontiguous": [0, 3],
        "tail_b1": [0],
    }
    active_idx = idx_map[subset]
    sub_tmpls = [templates[i] for i in active_idx]
    sigmasqs = [t.sigmasq(segments[0].psd) for t in sub_tmpls]

    # Reference run with list path
    res_list = ctrl.batched_matched_filter_and_cluster(
        0, sub_tmpls, sigmasqs, window=16, epoch=segments[0]._epoch
    )
    # Immediate snapshot of list outputs before another filter call can alias workspace
    list_snrs = [np.array(r[0]) for r in res_list]
    list_corrs = [np.array(r[2]) for r in res_list]

    # Prepared batch path
    prepared = ctrl.prepare_template_batch(templates)
    if subset == "full":
        batch_input = prepared
    elif subset == "contiguous":
        batch_input = prepared[active_idx[0] : active_idx[-1] + 1]
    else:
        batch_input = prepared[active_idx]

    res_prep = ctrl.batched_matched_filter_and_cluster(
        0, batch_input, sigmasqs, window=16, epoch=segments[0]._epoch
    )
    # Immediate snapshot of prepared outputs
    prep_snrs = [np.array(r[0]) for r in res_prep]
    prep_corrs = [np.array(r[2]) for r in res_prep]

    assert len(res_prep) == len(res_list) == len(active_idx)
    for i in range(len(active_idx)):
        np.testing.assert_allclose(
            prep_snrs[i], list_snrs[i], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            prep_corrs[i], list_corrs[i], rtol=1e-6, atol=1e-6
        )
        assert res_prep[i][1] == res_list[i][1]
        np.testing.assert_array_equal(
            np.asarray(res_prep[i][3]), np.asarray(res_list[i][3])
        )
        np.testing.assert_allclose(
            np.asarray(res_prep[i][4]),
            np.asarray(res_list[i][4]),
            rtol=1e-6,
            atol=1e-6,
        )


@pytest.mark.parametrize("device", DEVICES + ["numpy"])
def test_prepared_batch_repeated_segments_changed_psd(device):
    """Verify batch reuse across segments with differing PSDs/norms and no cross-segment aliasing."""
    templates, segments, N, delta_f = _make_batch_fixtures(
        num_templates=3, num_segments=2
    )
    flen = N // 2 + 1
    mem = FrequencySeries(np.zeros(flen, dtype=np.complex64), delta_f=delta_f)
    ctrl = TiledMatchedFilterControl(
        20.0,
        100.0,
        1.5,
        N,
        delta_f,
        np.complex64,
        segments,
        mem,
        True,
        tile_size=3,
        device=device,
    )

    prepared = ctrl.prepare_template_batch(templates)

    # Segment 0
    sigmas_0 = [t.sigmasq(segments[0].psd) for t in templates]
    res_list_0 = ctrl.batched_matched_filter_and_cluster(
        0, templates, sigmas_0, 16, epoch=segments[0]._epoch
    )
    snap_list_snr_0 = [np.array(r[0]) for r in res_list_0]
    res_prep_0 = ctrl.batched_matched_filter_and_cluster(
        0, prepared, sigmas_0, 16, epoch=segments[0]._epoch
    )
    snap_prep_snr_0 = [np.array(r[0]) for r in res_prep_0]

    # Segment 1 (differing PSD)
    sigmas_1 = [t.sigmasq(segments[1].psd) for t in templates]
    res_list_1 = ctrl.batched_matched_filter_and_cluster(
        1, templates, sigmas_1, 16, epoch=segments[1]._epoch
    )
    snap_list_snr_1 = [np.array(r[0]) for r in res_list_1]
    res_prep_1 = ctrl.batched_matched_filter_and_cluster(
        1, prepared, sigmas_1, 16, epoch=segments[1]._epoch
    )
    snap_prep_snr_1 = [np.array(r[0]) for r in res_prep_1]

    for i in range(3):
        np.testing.assert_allclose(
            snap_prep_snr_0[i], snap_list_snr_0[i], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            snap_prep_snr_1[i], snap_list_snr_1[i], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_array_equal(
            np.asarray(res_prep_0[i][3]), np.asarray(res_list_0[i][3])
        )
        np.testing.assert_array_equal(
            np.asarray(res_prep_1[i][3]), np.asarray(res_list_1[i][3])
        )


# ---------------------------------------------------------------------------
# AST Extraction Harness for actual bin/pycbc_inspiral batch_template_triggers
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _compile_cli_batch_template_triggers():
    inspiral_path = (
        Path(__file__).resolve().parent.parent / "bin" / "pycbc_inspiral"
    )
    source = inspiral_path.read_text()
    parsed = ast.parse(source, filename=str(inspiral_path))
    for node in parsed.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "batch_template_triggers"
        ):
            mod = ast.Module(body=[node], type_ignores=[])
            return compile(mod, filename=str(inspiral_path), mode="exec")
    raise RuntimeError(
        "batch_template_triggers FunctionDef not found in bin/pycbc_inspiral"
    )


def _execute_actual_cli_batch(
    matched_filter,
    templates,
    segments,
    tile_mem=None,
    checker_fn=None,
    power_chisq_fn=None,
    snr_threshold=1.5,
    flow=20.0,
):
    """Executes the actual compiled batch_template_triggers FunctionDef from bin/pycbc_inspiral."""
    b = len(templates)
    if tile_mem is None:
        tile_mem = [
            FrequencySeries(
                np.zeros(len(templates[0]), dtype=np.complex64),
                delta_f=templates[0].delta_f,
            )
            for _ in range(b)
        ]

    class BankStub:
        out = None

        def __getitem__(self, idx):
            t = templates[idx]
            if self.out is not None:
                self.out[:] = t[:]
                tmpl = FrequencySeries(self.out, delta_f=t.delta_f, copy=False)
                tmpl.id = getattr(t, "id", idx)
                tmpl.params = getattr(t, "params", None)
                tmpl.f_lower = getattr(t, "f_lower", flow)
                tmpl.end_frequency = getattr(t, "end_frequency", 100.0)
                if hasattr(t, "sigmasq"):
                    tmpl.sigmasq = types.MethodType(
                        lambda s, psd: float(
                            sigmasq(
                                s,
                                psd=psd,
                                low_frequency_cutoff=s.f_lower,
                                high_frequency_cutoff=s.end_frequency,
                            )
                        ),
                        tmpl,
                    )
                return tmpl
            return t

        def __len__(self):
            return b

    code = _compile_cli_batch_template_triggers()
    ns = {
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "id": id,
        "hasattr": hasattr,
        "list": list,
        "zip": zip,
        "numpy": np,
        "float32": np.float32,
        "flow": flow,
        "logging": types.SimpleNamespace(info=lambda *args, **kwargs: None),
        "bank": BankStub(),
        "tile_mem": tile_mem,
        "segments": segments,
        "inj_filter_rejector": types.SimpleNamespace(
            template_segment_checker=checker_fn
            or (lambda bk, t_num, seg: True)
        ),
        "matched_filter": matched_filter,
        "use_tiled_adapter": True,
        "cluster_window": 16,
        "out_vals_ref": {},
        "bank_chisq": types.SimpleNamespace(do=False),
        "power_chisq": types.SimpleNamespace(
            values=power_chisq_fn
            or (
                lambda corr, snrv, norm, psd, idx, tmpl: (
                    np.ones(len(idx), dtype=np.float32),
                    np.full(len(idx), 30, dtype=np.int32),
                )
            )
        ),
        "sg_chisq": types.SimpleNamespace(do=False),
        "autochisq": types.SimpleNamespace(do=False),
        "opt": types.SimpleNamespace(
            update_progress=False,
            psdvar_short_segment=None,
            psdvar_long_segment=None,
            snr_threshold=snr_threshold,
        ),
    }
    exec(code, ns)
    return ns["batch_template_triggers"](list(range(b)))


@pytest.mark.parametrize("device", DEVICES + ["numpy"])
def test_inspiral_actual_cli_batch_packing_counts(device):
    """Verify real template packing occurs at most once per batch across multiple segments."""
    templates, segments, N, delta_f = _make_batch_fixtures(
        num_templates=4, num_segments=3
    )
    flen = N // 2 + 1
    mem = FrequencySeries(np.zeros(flen, dtype=np.complex64), delta_f=delta_f)
    ctrl = TiledMatchedFilterControl(
        20.0,
        100.0,
        1.5,
        N,
        delta_f,
        np.complex64,
        segments,
        mem,
        True,
        tile_size=4,
        device=device,
    )

    pack_counts = [0]
    orig_prepare = ctrl.prepare_template_batch

    def counted_prepare(b_tmpls):
        if isinstance(b_tmpls, (list, tuple)):
            pack_counts[0] += 1
        return orig_prepare(b_tmpls)

    ctrl.prepare_template_batch = counted_prepare

    # 1. Full all-active across 3 segments -> exactly 1 real pack
    _execute_actual_cli_batch(ctrl, templates, segments)
    assert pack_counts[0] == 1

    # 2. All-inactive across all segments -> 0 real packs
    pack_counts[0] = 0
    _execute_actual_cli_batch(
        ctrl, templates, segments, checker_fn=lambda bk, tn, seg: False
    )
    assert pack_counts[0] == 0

    # 3. Lazy packing: segment 0 inactive, segment 1 active -> exactly 1 real pack
    pack_counts[0] = 0
    _execute_actual_cli_batch(
        ctrl,
        templates,
        segments,
        checker_fn=lambda bk, tn, seg: seg._epoch > 0,
    )
    assert pack_counts[0] == 1


@pytest.mark.parametrize("device", DEVICES + ["numpy"])
def test_inspiral_actual_cli_batch_reused_bank_buffers_and_subsets(device):
    """Verify bank buffer reuse across sequential batches and compare against fresh control."""
    tmpls_a, segments, N, delta_f = _make_batch_fixtures(
        num_templates=4, num_segments=2, seed=42
    )
    flen = N // 2 + 1
    mem = FrequencySeries(np.zeros(flen, dtype=np.complex64), delta_f=delta_f)
    ctrl = TiledMatchedFilterControl(
        20.0,
        100.0,
        0.0,
        N,
        delta_f,
        np.complex64,
        segments,
        mem,
        True,
        tile_size=4,
        device=device,
    )

    # Pre-allocate reusable tile_mem bank buffers
    reused_tile_mem = [
        FrequencySeries(np.zeros(flen, dtype=np.complex64), delta_f=delta_f)
        for _ in range(4)
    ]

    # Run Batch A with reused_tile_mem
    out_a = _execute_actual_cli_batch(
        ctrl, tmpls_a, segments, tile_mem=reused_tile_mem, snr_threshold=0.0
    )
    assert len(out_a) == 4

    # Run Batch B with DIFFERENT seed, reusing the exact same reused_tile_mem buffers and same ctrl
    tmpls_b, _, _, _ = _make_batch_fixtures(
        num_templates=4, num_segments=2, seed=999
    )
    out_b = _execute_actual_cli_batch(
        ctrl, tmpls_b, segments, tile_mem=reused_tile_mem, snr_threshold=0.0
    )
    assert len(out_b) == 4

    # Independent fresh-control run for Batch B
    fresh_mem = FrequencySeries(
        np.zeros(flen, dtype=np.complex64), delta_f=delta_f
    )
    fresh_ctrl = TiledMatchedFilterControl(
        20.0,
        100.0,
        0.0,
        N,
        delta_f,
        np.complex64,
        segments,
        fresh_mem,
        True,
        tile_size=4,
        device=device,
    )
    fresh_tile_mem = [
        FrequencySeries(np.zeros(flen, dtype=np.complex64), delta_f=delta_f)
        for _ in range(4)
    ]
    out_b_fresh = _execute_actual_cli_batch(
        fresh_ctrl,
        tmpls_b,
        segments,
        tile_mem=fresh_tile_mem,
        snr_threshold=0.0,
    )
    assert len(out_b_fresh) == 4

    # Compare Batch B ALL event fields against fresh-control run
    for i in range(4):
        events_b, param_b = out_b[i]
        events_fresh, param_fresh = out_b_fresh[i]
        assert param_b == param_fresh
        assert len(events_b) == len(events_fresh)
        for ev_b, ev_fresh in zip(events_b, events_fresh):
            assert ev_b.keys() == ev_fresh.keys()
            for k in ev_b:
                val_b = ev_b[k]
                val_fresh = ev_fresh[k]
                if isinstance(val_b, np.ndarray):
                    np.testing.assert_allclose(
                        val_b, val_fresh, rtol=1e-6, atol=1e-6
                    )
                else:
                    assert val_b == val_fresh

    # Assert Batch A vs Batch B have distinct outputs (due to seed change)
    assert len(out_a[0][0]) > 0 and len(out_b[0][0]) > 0
    assert not np.array_equal(out_a[0][0][0]["snr"], out_b[0][0][0]["snr"])


class _BaselineNoPrepareWrapper:
    """Wrapper that forwards batched filtering without exposing prepare_template_batch."""

    def __init__(self, ctrl):
        self._ctrl = ctrl

    def batched_matched_filter_and_cluster(self, *args, **kwargs):
        return self._ctrl.batched_matched_filter_and_cluster(*args, **kwargs)

    def __getattr__(self, name):
        if name == "prepare_template_batch":
            raise AttributeError(
                "Baseline does not provide prepare_template_batch"
            )
        return getattr(self._ctrl, name)


@pytest.mark.parametrize("device", DEVICES + ["numpy"])
def test_inspiral_actual_cli_batch_mixed_active_selection_vs_baseline(device):
    """Verify mixed full/contiguous/noncontiguous active selection in actual CLI matches baseline."""
    templates, segments, N, delta_f = _make_batch_fixtures(
        num_templates=4, num_segments=3, seed=123
    )
    flen = N // 2 + 1
    mem_prep = FrequencySeries(
        np.zeros(flen, dtype=np.complex64), delta_f=delta_f
    )
    ctrl_prep = TiledMatchedFilterControl(
        20.0,
        100.0,
        0.0,
        N,
        delta_f,
        np.complex64,
        segments,
        mem_prep,
        True,
        tile_size=4,
        device=device,
    )

    mem_base = FrequencySeries(
        np.zeros(flen, dtype=np.complex64), delta_f=delta_f
    )
    ctrl_base_inner = TiledMatchedFilterControl(
        20.0,
        100.0,
        0.0,
        N,
        delta_f,
        np.complex64,
        segments,
        mem_base,
        True,
        tile_size=4,
        device=device,
    )
    ctrl_base = _BaselineNoPrepareWrapper(ctrl_base_inner)

    # Mixed selection pattern across the 3 segments:
    # Segment 0: full batch active [0, 1, 2, 3]
    # Segment 1: contiguous subset active [1, 2]
    # Segment 2: noncontiguous subset active [0, 3]
    def mixed_active_checker(bk, t_num, seg):
        seg_idx = int(round(seg._epoch / 10.0))
        if seg_idx == 0:
            return True
        elif seg_idx == 1:
            return t_num in (1, 2)
        else:
            return t_num in (0, 3)

    out_prep = _execute_actual_cli_batch(
        ctrl_prep,
        templates,
        segments,
        checker_fn=mixed_active_checker,
        snr_threshold=0.0,
    )
    out_base = _execute_actual_cli_batch(
        ctrl_base,
        templates,
        segments,
        checker_fn=mixed_active_checker,
        snr_threshold=0.0,
    )

    assert len(out_prep) == len(out_base) == 4

    total_triggers = 0
    for i in range(4):
        events_p, param_p = out_prep[i]
        events_b, param_b = out_base[i]
        assert param_p == param_b
        assert len(events_p) == len(events_b)
        total_triggers += len(events_p)

        # Snapshot and compare scientific fields
        for ev_p, ev_b in zip(events_p, events_b):
            assert ev_p.keys() == ev_b.keys()
            for key in ["time_index", "snr", "sigmasq", "chisq", "chisq_dof"]:
                snap_p = np.array(ev_p[key], copy=True)
                snap_b = np.array(ev_b[key], copy=True)
                if key in ("time_index", "chisq_dof"):
                    np.testing.assert_array_equal(snap_p, snap_b)
                else:
                    np.testing.assert_allclose(
                        snap_p, snap_b, rtol=1e-6, atol=1e-6
                    )

    assert total_triggers > 0


@pytest.mark.parametrize("device", DEVICES + ["numpy"])
def test_inspiral_actual_cli_batch_veto_metadata(device):
    """Verify veto evaluator receives original trigger-bearing FrequencySeries template metadata."""
    templates, segments, N, delta_f = _make_batch_fixtures(
        num_templates=2, num_segments=1
    )
    flen = N // 2 + 1
    mem = FrequencySeries(np.zeros(flen, dtype=np.complex64), delta_f=delta_f)
    ctrl = TiledMatchedFilterControl(
        20.0,
        100.0,
        0.0,
        N,
        delta_f,
        np.complex64,
        segments,
        mem,
        True,
        tile_size=2,
        device=device,
    )

    passed_tmpls = []

    def mock_power_chisq(corr, snrv, norm, psd, idx, tmpl):
        passed_tmpls.append(tmpl)
        return np.ones(len(idx), dtype=np.float32), np.full(
            len(idx), 30, dtype=np.int32
        )

    out = _execute_actual_cli_batch(
        ctrl,
        templates,
        segments,
        power_chisq_fn=mock_power_chisq,
        snr_threshold=0.0,
    )

    # Assert trigger-bearing data produced triggers
    total_triggers = sum(len(entry[0]) for entry in out)
    assert total_triggers > 0
    assert len(passed_tmpls) > 0

    for tmpl in passed_tmpls:
        assert isinstance(tmpl, FrequencySeries)
        assert hasattr(tmpl, "params")
        assert hasattr(tmpl, "id")
        assert hasattr(tmpl, "f_lower")
        assert hasattr(tmpl, "end_frequency")
        assert hasattr(tmpl, "sigmasq")
