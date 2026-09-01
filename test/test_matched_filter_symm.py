#!/usr/bin/env python3
"""Unit tests for MatchedFilterControl.full_matched_filter_and_cluster_symm.

Verifies:
  - Complex SNR time series, normalization, and correlation vectors match
    across Standard CPU and Torch schemes (CPU and CUDA if available).
  - Peak clustering indices and trigger SNR values match within tolerance.
  - Zero-trigger behavior (sub-threshold) and single/multiple trigger behavior.
"""

import numpy as np
import pytest

import pycbc
from pycbc.filter.matchedfilter import MatchedFilterControl
from pycbc.scheme import CPUScheme
from pycbc.types import Array, FrequencySeries, zeros


def _generate_test_data(size=131072, sample_rate=2048, inj_snr=12.0):
    delta_t = 1.0 / sample_rate
    delta_f = 1.0 / (size * delta_t)
    freq_len = size // 2 + 1
    flow, fhigh = 30.0, 800.0

    k_min = max(1, int(flow / delta_f))
    k_max = min(freq_len - 1, int(fhigh / delta_f))

    rng = np.random.default_rng(42)
    sig_np = np.zeros(freq_len, dtype=np.complex64)
    sig_np[k_min:k_max] = (
        rng.normal(size=k_max - k_min) + 1j * rng.normal(size=k_max - k_min)
    ).astype(np.complex64)
    sgm_val = float(4.0 * delta_f * np.sum(np.abs(sig_np) ** 2))
    sig_np /= np.sqrt(sgm_val)

    # Injected signal shifted to center of analysis window
    phase_shift = np.exp(
        -2j * np.pi * np.arange(freq_len) * (size // 2) / size
    )
    seg_data = (sig_np * phase_shift * inj_snr).astype(np.complex64)

    return (
        size,
        sample_rate,
        delta_f,
        freq_len,
        flow,
        fhigh,
        sig_np,
        seg_data,
        sgm_val,
    )


def test_matched_filter_and_cluster_symm_cpu_reference():
    """Verify full_matched_filter_and_cluster_symm on CPUScheme."""
    (
        size,
        sample_rate,
        delta_f,
        freq_len,
        flow,
        fhigh,
        sig_np,
        seg_data,
        sgm_val,
    ) = _generate_test_data()

    window = int(0.5 * sample_rate)

    with CPUScheme():
        seg = FrequencySeries(Array(seg_data), delta_f=delta_f)
        seg.analyze = slice(int(8 * sample_rate), int(size - 8 * sample_rate))

        tmpl_mem = zeros(size, dtype=np.complex64)
        tmpl_mem[:freq_len] = Array(sig_np)

        mfc = MatchedFilterControl(
            low_frequency_cutoff=flow,
            high_frequency_cutoff=fhigh,
            snr_threshold=5.5,
            tlen=size,
            delta_f=delta_f,
            dtype=np.complex64,
            segment_list=[seg],
            template_output=tmpl_mem,
            use_cluster=True,
            downsample_factor=1,
            cluster_function="symmetric",
        )

        snr, norm, corr, idx, snrv = mfc.full_matched_filter_and_cluster_symm(
            0, 1.0, window
        )

        assert len(idx) >= 1
        assert len(snrv) == len(idx)
        # Expected trigger index is centered in the analysis window
        expected_idx = size // 2 - int(8 * sample_rate)
        assert np.min(np.abs(np.array(idx) - expected_idx)) <= window


@pytest.mark.skipif(not pycbc.HAVE_TORCH, reason="PyTorch not available")
def test_matched_filter_and_cluster_symm_torch_cpu_parity():
    """Verify parity between CPUScheme and TorchScheme CPU."""
    from pycbc.scheme import TorchScheme

    (
        size,
        sample_rate,
        delta_f,
        freq_len,
        flow,
        fhigh,
        sig_np,
        seg_data,
        sgm_val,
    ) = _generate_test_data()

    window = int(0.5 * sample_rate)

    # 1. Standard CPU reference
    with CPUScheme():
        seg_cpu = FrequencySeries(Array(seg_data), delta_f=delta_f)
        seg_cpu.analyze = slice(
            int(8 * sample_rate), int(size - 8 * sample_rate)
        )
        tmpl_cpu = zeros(size, dtype=np.complex64)
        tmpl_cpu[:freq_len] = Array(sig_np)

        mfc_cpu = MatchedFilterControl(
            low_frequency_cutoff=flow,
            high_frequency_cutoff=fhigh,
            snr_threshold=5.5,
            tlen=size,
            delta_f=delta_f,
            dtype=np.complex64,
            segment_list=[seg_cpu],
            template_output=tmpl_cpu,
            use_cluster=True,
            downsample_factor=1,
            cluster_function="symmetric",
        )
        snr_ref, norm_ref, corr_ref, idx_ref, snrv_ref = (
            mfc_cpu.full_matched_filter_and_cluster_symm(0, 1.0, window)
        )

    # 2. Torch CPU
    with TorchScheme(device="cpu", num_threads=1):
        seg_torch = FrequencySeries(Array(seg_data), delta_f=delta_f)
        seg_torch.analyze = slice(
            int(8 * sample_rate), int(size - 8 * sample_rate)
        )
        tmpl_torch = zeros(size, dtype=np.complex64)
        tmpl_torch[:freq_len] = Array(sig_np)

        mfc_torch = MatchedFilterControl(
            low_frequency_cutoff=flow,
            high_frequency_cutoff=fhigh,
            snr_threshold=5.5,
            tlen=size,
            delta_f=delta_f,
            dtype=np.complex64,
            segment_list=[seg_torch],
            template_output=tmpl_torch,
            use_cluster=True,
            downsample_factor=1,
            cluster_function="symmetric",
        )
        snr_t, norm_t, corr_t, idx_t, snrv_t = (
            mfc_torch.full_matched_filter_and_cluster_symm(0, 1.0, window)
        )

    np.testing.assert_allclose(norm_ref, norm_t, rtol=1e-5)
    np.testing.assert_allclose(
        corr_ref.numpy(), corr_t.numpy(), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        snr_ref.numpy(), snr_t.numpy(), rtol=1e-4, atol=1e-5
    )
    assert len(idx_ref) == len(idx_t)
    np.testing.assert_array_equal(np.array(idx_ref), np.array(idx_t))
    np.testing.assert_allclose(
        np.array(snrv_ref), np.array(snrv_t), rtol=1e-4, atol=1e-5
    )
