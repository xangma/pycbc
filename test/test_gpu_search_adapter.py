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
    assert "time" in result
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

