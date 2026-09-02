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

"""Unit tests for Milestone 1 signal processing and batched filter optimizations."""

import numpy as np
import pytest
import torch

from pycbc import scheme
from pycbc.filter.matchedfilter import BatchCorrelator
from pycbc.filter import matchedfilter_torch
from pycbc.types import Array, FrequencySeries


def _available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@pytest.mark.parametrize("device", _available_devices())
def test_torch_batch_peak_and_threshold_single_reduction_parity(device):
    """Verify single-reduction max SNR comparison across all threshold branches."""
    template_count = 64
    seg_len = 256
    values = torch.zeros(
        (template_count, seg_len), dtype=torch.complex64, device=device
    )
    # Row 3: sub-threshold peak
    values[3, 20] = 4.0 + 3.0j  # |val| = 5.0
    # Row 12: above threshold
    values[12, 100] = 8.0 + 6.0j  # |val| = 10.0
    # Row 45: high peak
    values[45, 200] = -15.0j  # |val| = 15.0

    norms = np.ones(template_count, dtype=np.float64)

    # 1. High threshold (no triggers survive)
    surv, peak_idx, peak_val, aborted = (
        matchedfilter_torch._torch_batch_peak_and_threshold_gpu(
            values, norms, snr_threshold=20.0
        )
    )
    assert not aborted
    assert len(surv) == 0
    assert len(peak_idx) == 0
    assert len(peak_val) == 0

    # 2. Moderate threshold (rows 12 and 45 survive)
    surv, peak_idx, peak_val, aborted = (
        matchedfilter_torch._torch_batch_peak_and_threshold_gpu(
            values, norms, snr_threshold=8.0
        )
    )
    assert not aborted
    np.testing.assert_array_equal(surv, [12, 45])
    np.testing.assert_array_equal(peak_idx, [100, 200])
    np.testing.assert_allclose(peak_val, [8.0 + 6.0j, -15.0j])

    # 3. Abort threshold triggered (row 45 has |val|=15 > 12)
    surv, peak_idx, peak_val, aborted = (
        matchedfilter_torch._torch_batch_peak_and_threshold_gpu(
            values, norms, snr_threshold=8.0, snr_abort_threshold=12.0
        )
    )
    assert aborted
    assert len(surv) == 0


@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize("num_vectors", [4, 16, 64, 128])
def test_batch_correlate_stacked_and_strided_2d_parity(device, num_vectors):
    """Verify 2D batched correlation parity against reference CPU elementwise multiply."""
    size = 1024
    delta_f = 1.0 / (size * 2)
    rng = np.random.RandomState(42 + num_vectors)

    with scheme.TorchScheme(device):
        templates = []
        outputs = []
        raw_x = []
        for i in range(num_vectors):
            data_x = (
                rng.randn(size).astype(np.float32)
                + 1j * rng.randn(size).astype(np.float32)
            )
            fs_x = FrequencySeries(data_x, delta_f=delta_f)
            fs_z = FrequencySeries(np.zeros(size, dtype=np.complex64), delta_f=delta_f)
            templates.append(fs_x)
            outputs.append(fs_z)
            raw_x.append(data_x)

        data_y = (
            rng.randn(size).astype(np.float32)
            + 1j * rng.randn(size).astype(np.float32)
        )
        fs_y = FrequencySeries(data_y, delta_f=delta_f)

        correlator = BatchCorrelator(templates, outputs, size)
        correlator.execute(fs_y)

        # Reference expectation: z[i] = conj(x[i]) * y
        for i in range(num_vectors):
            expected = np.conj(raw_x[i]) * data_y
            actual = outputs[i].numpy()
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

        # Execute a second time to verify cached conjugate reuse
        data_y2 = (
            rng.randn(size).astype(np.float32)
            + 1j * rng.randn(size).astype(np.float32)
        )
        fs_y2 = FrequencySeries(data_y2, delta_f=delta_f)
        correlator.execute(fs_y2)

        for i in range(num_vectors):
            expected2 = np.conj(raw_x[i]) * data_y2
            actual2 = outputs[i].numpy()
            np.testing.assert_allclose(actual2, expected2, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("device", _available_devices())
def test_array_torch_squared_norm_and_inner_parity(device):
    """Verify array_torch ufuncs maintain strict numerical parity."""
    size = 2048
    rng = np.random.RandomState(123)
    dtype = np.float32 if device == "mps" else np.float64
    cdtype = np.complex64 if device == "mps" else np.complex128
    tol = 1e-4 if device == "mps" else 1e-12

    a_np = (
        rng.randn(size).astype(dtype)
        + 1j * rng.randn(size).astype(dtype)
    ).astype(cdtype)
    b_np = (
        rng.randn(size).astype(dtype)
        + 1j * rng.randn(size).astype(dtype)
    ).astype(cdtype)
    w_np = (np.abs(rng.randn(size).astype(dtype)) + 0.1).astype(dtype)

    with scheme.TorchScheme(device):
        a = Array(a_np)
        b = Array(b_np)
        w = Array(w_np)

        # squared_norm
        sq = a.squared_norm()
        expected_sq = a_np.real**2 + a_np.imag**2
        np.testing.assert_allclose(sq.numpy(), expected_sq, rtol=tol, atol=tol)

        # inner (self with self)
        inner_self = a.inner(a)
        expected_inner_self = np.sum(a_np.real**2 + a_np.imag**2)
        assert abs(inner_self - expected_inner_self) / expected_inner_self < tol

        # inner (a with b)
        inner_ab = a.inner(b)
        expected_inner_ab = np.sum(np.conj(a_np) * b_np)
        assert abs(inner_ab - expected_inner_ab) / abs(expected_inner_ab) < tol

        # weighted_inner (self with self)
        w_self = a.weighted_inner(a, w)
        expected_w_self = np.sum((a_np.real**2 + a_np.imag**2) / w_np)
        assert abs(w_self - expected_w_self) / expected_w_self < tol

        # weighted_inner (a with b)
        w_ab = a.weighted_inner(b, w)
        expected_w_ab = np.sum(np.conj(a_np) * b_np / w_np)
        assert abs(w_ab - expected_w_ab) / abs(expected_w_ab) < tol
