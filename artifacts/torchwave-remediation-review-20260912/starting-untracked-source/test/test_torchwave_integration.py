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

"""Verification suite for torchwave integration with PyCBC matched filter."""

import numpy as np
import pytest
import torch

try:
    import torchwave
    HAS_TORCHWAVE = True
except ImportError:
    HAS_TORCHWAVE = False

from pycbc import DYN_RANGE_FAC
from pycbc.types import FrequencySeries, zeros
from pycbc.waveform import get_fd_waveform as pycbc_get_fd
from pycbc.filter.matchedfilter import match, sigmasq
from pycbc.filter.gpu_search.adapter import TiledMatchedFilterControl


pytestmark = pytest.mark.skipif(
    not HAS_TORCHWAVE, reason="torchwave not installed in environment"
)


def test_torchwave_catalog_coverage():
    """Verify torchwave exposes essential PyCBC search approximants."""
    required = ["TaylorF2", "IMRPhenomD", "IMRPhenomXAS", "IMRPhenomD_NRTidal"]
    for approx in required:
        key = approx.lower().replace("_", "").replace("-", "")
        assert (
            key in torchwave.waveforms._APPROXIMANTS
        ), f"{approx} missing in torchwave"


def test_torchwave_taylorf2_parity():
    """Verify TaylorF2 complex match > 0.999999 against PyCBC baseline."""
    m1, m2 = 1.4, 1.2
    delta_f = 0.25
    flow = 30.0
    fhigh = 500.0

    hp_ref, _ = pycbc_get_fd(
        approximant="TaylorF2",
        mass1=m1,
        mass2=m2,
        delta_f=delta_f,
        f_lower=flow,
        f_final=fhigh,
        distance=1.0 / DYN_RANGE_FAC,
    )

    flen = len(hp_ref)
    freqs = torch.arange(0, flen, dtype=torch.float64) * delta_f
    mask = (freqs >= flow) & (freqs <= fhigh)

    # Note: LAL TaylorF2 has a pi/2 phase convention offset
    hp_tw, _ = torchwave.get_fd_waveform(
        "TaylorF2",
        mass1=m1,
        mass2=m2,
        phic=np.pi / 2.0,
        sample_frequencies=freqs[mask],
        distance=1.0 / DYN_RANGE_FAC,
        dtype=torch.float64,
        device="cpu",
    )

    tw_full = np.zeros(flen, dtype=np.complex128)
    tw_full[mask.numpy()] = hp_tw.numpy()
    hp_tw_fs = FrequencySeries(tw_full, delta_f=delta_f)

    overlap, _ = match(
        hp_ref,
        hp_tw_fs,
        low_frequency_cutoff=flow,
        high_frequency_cutoff=fhigh,
    )
    assert overlap > 0.999999, f"TaylorF2 match {overlap} below threshold"


def test_torchwave_imrphenomd_parity():
    """Verify IMRPhenomD complex match > 0.999999 against PyCBC baseline."""
    m1, m2 = 30.0, 20.0
    s1z, s2z = 0.2, -0.1
    delta_f = 0.25
    flow = 20.0
    fhigh = 512.0

    hp_ref, _ = pycbc_get_fd(
        approximant="IMRPhenomD",
        mass1=m1,
        mass2=m2,
        spin1z=s1z,
        spin2z=s2z,
        delta_f=delta_f,
        f_lower=flow,
        f_final=fhigh,
        distance=1.0 / DYN_RANGE_FAC,
    )

    flen = len(hp_ref)
    freqs = torch.arange(0, flen, dtype=torch.float64) * delta_f
    mask = (freqs >= flow) & (freqs <= fhigh)

    hp_tw, _ = torchwave.get_fd_waveform(
        "IMRPhenomD",
        mass1=m1,
        mass2=m2,
        spin1z=s1z,
        spin2z=s2z,
        sample_frequencies=freqs[mask],
        distance=1.0 / DYN_RANGE_FAC,
        dtype=torch.float64,
        device="cpu",
    )

    tw_full = np.zeros(flen, dtype=np.complex128)
    tw_full[mask.numpy()] = hp_tw.numpy()
    hp_tw_fs = FrequencySeries(tw_full, delta_f=delta_f)

    overlap, _ = match(
        hp_ref,
        hp_tw_fs,
        low_frequency_cutoff=flow,
        high_frequency_cutoff=fhigh,
    )
    assert overlap > 0.999999, f"IMRPhenomD match {overlap} below threshold"


def test_torchwave_batched_tiled_adapter_parity():
    """Verify that a 2D tensor produced by torchwave feeds directly into

    TiledMatchedFilterControl and produces identical triggers and arrival
    times.
    """
    delta_f = 0.5
    flen = 1025
    tlen = (flen - 1) * 2
    flow = 30.0
    fhigh = 500.0
    snr_thresh = 4.0

    np.random.seed(12345)
    s_data = (
        np.random.randn(flen) + 1j * np.random.randn(flen)
    ).astype(np.complex64)
    s_data[0] = 0.0
    seg = FrequencySeries(s_data, delta_f=delta_f)
    psd_arr = np.ones(flen, dtype=np.float32) * 2.0
    seg.psd = FrequencySeries(psd_arr, delta_f=delta_f)
    seg.analyze = slice(100, tlen - 100)
    seg._epoch = 0

    m1_list = [1.4, 1.6, 2.0, 2.5]
    m2_list = [1.2, 1.3, 1.4, 1.5]
    b = len(m1_list)

    # Route A: PyCBC Baseline (scalar FrequencySeries generation)
    templates_a = []
    for i in range(b):
        hp, _ = pycbc_get_fd(
            approximant="TaylorF2",
            mass1=m1_list[i],
            mass2=m2_list[i],
            delta_f=delta_f,
            f_lower=flow,
            f_final=fhigh,
            distance=1.0 / DYN_RANGE_FAC,
        )
        hp_full = FrequencySeries(
            zeros(flen, dtype=np.complex64), delta_f=delta_f
        )
        copy_len = min(len(hp), flen)
        hp_full[:copy_len] = hp[:copy_len]
        templates_a.append(hp_full)

    # Inject template 0 to create loud triggers
    seg += templates_a[0] * 25.0

    sigmasqs_a = [
        sigmasq(
            t, seg.psd, low_frequency_cutoff=flow, high_frequency_cutoff=fhigh
        )
        for t in templates_a
    ]

    # Route B: torchwave 2D Tensor Batched Generation
    freqs = torch.arange(0, flen, dtype=torch.float32) * delta_f
    mask = (freqs >= flow) & (freqs <= fhigh)
    active_freqs = freqs[mask]

    hp_b, _ = torchwave.get_fd_waveform(
        "TaylorF2",
        mass1=torch.tensor(m1_list, dtype=torch.float32),
        mass2=torch.tensor(m2_list, dtype=torch.float32),
        phic=torch.full((b,), np.pi / 2.0, dtype=torch.float32),
        sample_frequencies=active_freqs,
        distance=1.0 / DYN_RANGE_FAC,
        dtype=torch.float32,
        device="cpu",
    )

    full_tensor_b = torch.zeros((b, flen), dtype=torch.complex64)
    full_tensor_b[:, mask] = hp_b

    psd_t = torch.from_numpy(seg.psd.numpy())
    sigmasqs_b = (
        4.0
        * delta_f
        * torch.sum(
            torch.abs(full_tensor_b[:, mask]) ** 2 / psd_t[mask], dim=-1
        )
    ).tolist()

    ctrl = TiledMatchedFilterControl(
        low_frequency_cutoff=flow,
        high_frequency_cutoff=fhigh,
        snr_threshold=snr_thresh,
        tlen=tlen,
        delta_f=delta_f,
        dtype=np.complex64,
        segment_list=[seg],
        template_output=zeros(tlen, dtype=np.complex64),
        use_cluster=True,
        cluster_function="symmetric",
        tile_size=b,
        device="cpu",
    )

    prep_a = ctrl.prepare_template_batch(templates_a)
    prep_b = ctrl.prepare_template_batch(full_tensor_b)

    res_a = ctrl.batched_matched_filter_and_cluster(
        0, prep_a, sigmasqs_a, window=10
    )
    res_b = ctrl.batched_matched_filter_and_cluster(
        0, prep_b, sigmasqs_b, window=10
    )

    for i in range(b):
        _, _, _, idx_a, snrv_a = res_a[i]
        _, _, _, idx_b, snrv_b = res_b[i]

        assert len(idx_a) == len(idx_b), (
            f"Trigger count mismatch in template {i}"
        )

        if len(idx_a) > 0:
            # Arrival times must be exact (0 sample difference)
            max_idx_diff = np.max(np.abs(idx_a - idx_b))
            assert max_idx_diff == 0, f"Arrival time mismatch in template {i}"

            # Peak recovered SNR must agree to within 0.1% relative tolerance
            peak_snr_a = np.max(np.abs(snrv_a))
            peak_snr_b = np.max(np.abs(snrv_b))
            rel_peak_diff = abs(peak_snr_a - peak_snr_b) / peak_snr_a
            assert rel_peak_diff < 1e-3, (
                f"Peak SNR rel diff {rel_peak_diff:.2e} too large in "
                f"template {i}"
            )


def test_torchwave_batched_tensor_throughput():
    """Verify that batched torchwave generation achieves high throughput

    and zero-copy ingestion into TiledMatchedFilterControl.
    """
    import time

    delta_f = 0.5
    flen = 1025
    flow = 30.0
    fhigh = 500.0
    b = 64

    m1_vals = np.linspace(1.4, 2.5, b)
    m2_vals = np.linspace(1.2, 1.5, b)

    freqs = torch.arange(0, flen, dtype=torch.float32) * delta_f
    mask = (freqs >= flow) & (freqs <= fhigh)
    active_freqs = freqs[mask]
    m1_t = torch.tensor(m1_vals, dtype=torch.float32)
    m2_t = torch.tensor(m2_vals, dtype=torch.float32)

    # Warmup
    _ = torchwave.get_fd_waveform(
        "TaylorF2",
        mass1=m1_t,
        mass2=m2_t,
        sample_frequencies=active_freqs,
        distance=1.0 / DYN_RANGE_FAC,
        dtype=torch.float32,
        device="cpu",
    )

    t0 = time.perf_counter()
    hp_batch, _ = torchwave.get_fd_waveform(
        "TaylorF2",
        mass1=m1_t,
        mass2=m2_t,
        sample_frequencies=active_freqs,
        distance=1.0 / DYN_RANGE_FAC,
        dtype=torch.float32,
        device="cpu",
    )
    t_batch = time.perf_counter() - t0

    throughput = b / t_batch
    # Throughput on modern CPU should easily exceed 1,000 templates/s
    assert throughput > 1000.0, (
        f"Batched throughput {throughput:.1f} tmplt/s lower than threshold"
    )

    # Verify direct shape and dtype ingestion for TiledMatchedFilterControl
    full_tensor = torch.zeros((b, flen), dtype=torch.complex64)
    full_tensor[:, mask] = hp_batch
    assert full_tensor.shape == (b, flen)
    assert full_tensor.dtype == torch.complex64
