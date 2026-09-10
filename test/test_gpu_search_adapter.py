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
from pycbc.filter.matchedfilter import MatchedFilterControl
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
