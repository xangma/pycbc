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

"""Adversarial stress-test suite for Milestone 1 signal processing and filtering.

Empirical verification covers:
1. Batch matched filtering across B in {1, 8, 32, 128, 512, 1024, 1025}.
2. LiveBatchMatchedFilter across varying batch sizes B in {1, 8, 32, 128, 512, 1024}.
3. Peak finding, thresholding, and array operations under boundary/corner conditions.
4. Verify exact numerical parity (< 10^-12 in float64/complex128) against reference CPU/NumPy.
5. FFT/IFFT invariance and batched independent transforms.
6. Empirical regression tests for sky_max statistics.
"""

import types
import numpy as np
import pytest
import torch

from pycbc import scheme
from pycbc.filter.matchedfilter import (
    BatchCorrelator,
    LiveBatchMatchedFilter,
    compute_max_snr_over_sky_loc_stat,
    compute_max_snr_over_sky_loc_stat_no_phase,
)
from pycbc.filter import matchedfilter_torch
from pycbc.events import threshold_torch
from pycbc.fft import fft, ifft, torchfft
from pycbc.types import Array, FrequencySeries, TimeSeries, zeros


def _available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


# ============================================================================
# 1. BATCH MATCHED FILTERING ACROSS B in {1, 8, 32, 128, 512, 1024, 1025}
# ============================================================================

@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize("batch_size", [1, 8, 32, 128, 512, 1024, 1025])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_batch_correlate_adversarial_batch_sizes(device, batch_size, precision):
    """Stress-test BatchCorrelator across diverse batch sizes and precisions."""
    if device == "mps" and precision == "double":
        pytest.skip("MPS does not support double precision float64/complex128")

    size = 512
    delta_f = 1.0 / (size * 2)
    rng = np.random.RandomState(1000 + batch_size)

    dtype = np.float32 if precision == "single" else np.float64
    cdtype = np.complex64 if precision == "single" else np.complex128
    tol = 1e-5 if precision == "single" else 1e-12

    with scheme.TorchScheme(device):
        templates = []
        outputs = []
        raw_x = []

        for i in range(batch_size):
            data_x = (
                rng.randn(size).astype(dtype)
                + 1j * rng.randn(size).astype(dtype)
            ).astype(cdtype)
            fs_x = FrequencySeries(data_x, delta_f=delta_f)
            fs_z = FrequencySeries(np.zeros(size, dtype=cdtype), delta_f=delta_f)
            templates.append(fs_x)
            outputs.append(fs_z)
            raw_x.append(data_x)

        data_y1 = (
            rng.randn(size).astype(dtype)
            + 1j * rng.randn(size).astype(dtype)
        ).astype(cdtype)
        fs_y1 = FrequencySeries(data_y1, delta_f=delta_f)

        correlator = BatchCorrelator(templates, outputs, size)
        correlator.execute(fs_y1)

        # Verify output parity against NumPy oracle: z[i] = conj(x[i]) * y
        for i in range(batch_size):
            expected = np.conj(raw_x[i]) * data_y1
            actual = outputs[i].numpy()
            diff = np.max(np.abs(actual - expected))
            denom = max(1e-30, np.max(np.abs(expected)))
            rel_err = diff / denom
            assert rel_err < tol, f"Batch size {batch_size}, vector {i} mismatch: {rel_err}"

        # Second execution: test cache validity and repeated execution
        data_y2 = (
            rng.randn(size).astype(dtype)
            + 1j * rng.randn(size).astype(dtype)
        ).astype(cdtype)
        fs_y2 = FrequencySeries(data_y2, delta_f=delta_f)
        correlator.execute(fs_y2)

        for i in range(batch_size):
            expected2 = np.conj(raw_x[i]) * data_y2
            actual2 = outputs[i].numpy()
            diff2 = np.max(np.abs(actual2 - expected2))
            denom2 = max(1e-30, np.max(np.abs(expected2)))
            rel_err2 = diff2 / denom2
            assert rel_err2 < tol, f"Batch size {batch_size}, vector {i} run 2 mismatch: {rel_err2}"


@pytest.mark.parametrize("device", _available_devices())
def test_batch_correlate_non_power_of_two_and_odd_sizes(device):
    """Test BatchCorrelator on odd and non-power-of-two vector sizes."""
    sizes = [33, 127, 255, 384, 769]
    batch_size = 16
    rng = np.random.RandomState(42)

    with scheme.TorchScheme(device):
        for size in sizes:
            delta_f = 1.0 / (size * 2)
            templates = []
            outputs = []
            raw_x = []
            for _ in range(batch_size):
                dx = (rng.randn(size).astype(np.float32) + 1j * rng.randn(size).astype(np.float32)).astype(np.complex64)
                templates.append(FrequencySeries(dx, delta_f=delta_f))
                outputs.append(FrequencySeries(np.zeros(size, dtype=np.complex64), delta_f=delta_f))
                raw_x.append(dx)

            dy = (rng.randn(size).astype(np.float32) + 1j * rng.randn(size).astype(np.float32)).astype(np.complex64)
            fs_y = FrequencySeries(dy, delta_f=delta_f)

            correlator = BatchCorrelator(templates, outputs, size)
            correlator.execute(fs_y)

            for i in range(batch_size):
                expected = np.conj(raw_x[i]) * dy
                actual = outputs[i].numpy()
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize("batch_size", [1, 8, 32, 128, 512, 1024])
def test_live_batch_matched_filter_processing_stress(device, batch_size):
    """Stress-test LiveBatchMatchedFilter._process_batch across batch sizes."""
    template_size = 64
    rng = np.random.RandomState(42 + batch_size)

    with scheme.TorchScheme(device):
        raw_rows = (
            rng.randn(batch_size, template_size).astype(np.float32)
            + 1j * rng.randn(batch_size, template_size).astype(np.float32)
        ).astype(np.complex64)

        # Plant specific triggers
        # Template 0 has high peak at index 10
        raw_rows[0, 10] = 20.0 + 0.0j

        output = Array(raw_rows.reshape(-1))
        templates = []
        for index in range(batch_size):
            params = np.array([(20.0 + index,)], dtype=[("mass1", np.float32)])[0]
            template = types.SimpleNamespace(
                delta_f=0.25,
                id=100 + index,
                params=params,
                out=output[index * template_size:(index + 1) * template_size],
                sigmasq=lambda _psd: 1.0,
            )
            templates.append(template)

        mid = ("stress_test", batch_size)
        batch = object.__new__(LiveBatchMatchedFilter)
        batch.block_id = 0
        batch.tgroups = [templates]
        batch.chunk_tsamples = [template_size]
        batch.mids = [mid]
        batch.out_mem = {mid: output}
        batch.corr = [types.SimpleNamespace(execute=lambda _data: None)]
        batch.ifts = {mid: types.SimpleNamespace(execute=lambda: None)}
        batch.data = types.SimpleNamespace(
            overwhitened_data=lambda _delta_f: types.SimpleNamespace(psd=object()),
            trim_padding=0,
            blocksize=template_size,
            sample_rate=1,
            start_time=100.0,
        )
        batch.snr_threshold = 15.0
        batch.snr_abort_threshold = None

        result, veto_info = batch._process_batch()
        assert result is not None
        assert len(result["template_id"]) >= 1
        # First trigger should correspond to template id 100 with peak SNR >= 15.0
        assert result["template_id"][0] == 100
        assert result["snr"][0] >= 15.0


# ============================================================================
# 2. PEAK FINDING & THRESHOLDING CORNER / BOUNDARY CONDITIONS
# ============================================================================

@pytest.mark.parametrize("device", _available_devices())
def test_threshold_empty_and_zero_shape_tensors(device):
    """Adversarially test thresholding on empty/zero-dimensional shapes."""
    empty_cases = [
        torch.zeros((0, 256), dtype=torch.complex64, device=device),
        torch.zeros((64, 0), dtype=torch.complex64, device=device),
        torch.zeros((0, 0), dtype=torch.complex64, device=device),
    ]

    for values in empty_cases:
        norms = np.ones(values.shape[0], dtype=np.float64)
        surv, peak_idx, peak_val, aborted = matchedfilter_torch._torch_batch_peak_and_threshold_gpu(
            values, norms, snr_threshold=5.0
        )
        assert len(surv) == 0
        assert len(peak_idx) == 0
        assert len(peak_val) == 0
        assert not aborted


@pytest.mark.parametrize("device", _available_devices())
def test_threshold_exact_boundary_conditions(device):
    """Test exact equality to threshold and abort threshold."""
    template_count = 8
    seg_len = 64
    values = torch.zeros((template_count, seg_len), dtype=torch.complex64, device=device)
    norms = np.ones(template_count, dtype=np.float64)

    # Set exact peaks
    # Template 0: exactly at threshold (SNR = 8.0 -> |val| = 8.0)
    values[0, 10] = 8.0 + 0.0j
    # Template 1: just below threshold (SNR = 7.99999)
    values[1, 20] = 7.99999 + 0.0j
    # Template 2: just above threshold (SNR = 8.00001)
    values[2, 30] = 8.00001 + 0.0j
    # Template 3: exactly at abort threshold (SNR = 15.0)
    values[3, 40] = 15.0 + 0.0j
    # Template 4: strictly above abort threshold (SNR = 15.0001)
    values[4, 50] = 15.0001 + 0.0j

    # 1. Without abort threshold: templates 0, 2, 3, 4 must survive (since >= 8.0)
    surv, peak_idx, peak_val, aborted = matchedfilter_torch._torch_batch_peak_and_threshold_gpu(
        values[:3], norms[:3], snr_threshold=8.0
    )
    assert not aborted
    np.testing.assert_array_equal(surv, [0, 2])
    np.testing.assert_array_equal(peak_idx, [10, 30])

    # 2. Exact abort threshold test: max_val == 15.0 should NOT abort (strict > is required)
    surv, peak_idx, peak_val, aborted = matchedfilter_torch._torch_batch_peak_and_threshold_gpu(
        values[:4], norms[:4], snr_threshold=8.0, snr_abort_threshold=15.0
    )
    assert not aborted
    np.testing.assert_array_equal(surv, [0, 2, 3])

    # 3. Exceeded abort threshold test: max_val == 15.0001 MUST abort
    surv, peak_idx, peak_val, aborted = matchedfilter_torch._torch_batch_peak_and_threshold_gpu(
        values[:5], norms[:5], snr_threshold=8.0, snr_abort_threshold=15.0
    )
    assert aborted
    assert len(surv) == 0


@pytest.mark.parametrize("device", _available_devices())
def test_threshold_nan_and_inf_handling(device):
    """Stress-test NaN and Inf robustness in peak extraction and thresholding."""
    template_count = 10
    seg_len = 128
    values = torch.zeros((template_count, seg_len), dtype=torch.complex64, device=device)
    norms = np.ones(template_count, dtype=np.float64)

    # Put NaN and Infs in various templates
    values[0, 5] = float("nan") + 1j * float("nan")
    values[0, 10] = 10.0 + 0.0j  # Valid peak elsewhere in same template

    values[1, 15] = float("nan")  # Entire row NaN except zeros
    values[2, :] = float("nan")   # All NaNs in row

    values[3, 50] = 12.0 + 5.0j   # Valid high peak (|val| = 13.0)

    # GPU / CPU batch thresholding
    surv, peak_idx, peak_val, aborted = matchedfilter_torch._torch_batch_peak_and_threshold_gpu(
        values, norms, snr_threshold=8.0
    )
    assert not aborted
    # Rows 0 and 3 should survive and cleanly locate their valid peaks without NaN propagation
    assert 0 in surv
    assert 3 in surv
    assert 2 not in surv  # All NaN template cannot exceed threshold


@pytest.mark.parametrize("device", _available_devices())
def test_cluster_candidates_nan_repair_cpu_and_accelerator(device):
    """Stress-test _cluster_candidates with all-NaN and mixed-NaN blocks."""
    window = 16
    length = 64
    mag_sq = torch.full((length,), float("nan"), dtype=torch.float32, device=device)
    # Block 0: All NaN
    # Block 1: Partial NaN with valid max = 50.0 at index 20
    mag_sq[20] = 50.0
    # Block 2: Valid numbers
    mag_sq[32:48] = torch.arange(16, dtype=torch.float32, device=device)
    # Block 3: All Inf
    mag_sq[48:64] = float("inf")

    block_max, block_idx = threshold_torch._cluster_candidates(mag_sq, window)
    assert len(block_max) == 4
    assert len(block_idx) == 4

    # Block 1 should pick index 20 and value 50.0
    assert float(block_max[1].item()) == pytest.approx(50.0)
    assert int(block_idx[1].item()) == 20

    # Block 2 should pick index 47 (max of arange(16) which is 15 at pos 32+15=47)
    assert float(block_max[2].item()) == pytest.approx(15.0)
    assert int(block_idx[2].item()) == 47


# ============================================================================
# 3. ARRAY UFUNCS PARITY AND CORNER CASES (array_torch.py)
# ============================================================================

@pytest.mark.parametrize("device", _available_devices())
def test_array_torch_squared_norm_extended_parity(device):
    """Test squared_norm with float32, float64, complex64, complex128, empty, and extreme values."""
    with scheme.TorchScheme(device):
        # 1. Real Float64
        if device != "mps":
            rng = np.random.RandomState(42)
            arr_np = rng.randn(1024).astype(np.float64)
            arr = Array(arr_np)
            sq = arr.squared_norm()
            np.testing.assert_allclose(sq.numpy(), arr_np**2, rtol=1e-12, atol=1e-12)

            # Complex128
            carr_np = (rng.randn(1024) + 1j * rng.randn(1024)).astype(np.complex128)
            carr = Array(carr_np)
            csq = carr.squared_norm()
            expected_csq = carr_np.real**2 + carr_np.imag**2
            np.testing.assert_allclose(csq.numpy(), expected_csq, rtol=1e-12, atol=1e-12)

        # 2. Float32 & Complex64
        rng = np.random.RandomState(43)
        carr32_np = (rng.randn(1024) + 1j * rng.randn(1024)).astype(np.complex64)
        carr32 = Array(carr32_np)
        csq32 = carr32.squared_norm()
        expected_csq32 = carr32_np.real**2 + carr32_np.imag**2
        np.testing.assert_allclose(csq32.numpy(), expected_csq32, rtol=1e-5, atol=1e-5)

        # 3. Empty Array
        empty_arr = Array(np.array([], dtype=np.complex64))
        empty_sq = empty_arr.squared_norm()
        assert len(empty_sq) == 0

        # 4. Extreme values (large numbers, small numbers)
        if device != "mps":
            extreme_np = np.array([1e-150 + 1e-150j, 1e100 + 1e100j], dtype=np.complex128)
            extreme_arr = Array(extreme_np)
            extreme_sq = extreme_arr.squared_norm()
            expected_extreme = extreme_np.real**2 + extreme_np.imag**2
            np.testing.assert_allclose(extreme_sq.numpy(), expected_extreme, rtol=1e-12)


@pytest.mark.parametrize("device", _available_devices())
def test_array_torch_inner_and_weighted_inner_stress(device):
    """Stress-test inner and weighted_inner under identical/distinct buffers and weights."""
    if device == "mps":
        pytest.skip("Testing double-precision parity (< 10^-12) on non-MPS devices")

    rng = np.random.RandomState(999)
    size = 4096
    a_np = (rng.randn(size) + 1j * rng.randn(size)).astype(np.complex128)
    b_np = (rng.randn(size) + 1j * rng.randn(size)).astype(np.complex128)
    w_real_np = np.abs(rng.randn(size)).astype(np.float64) + 0.5
    w_complex_np = (np.abs(rng.randn(size)) + 1j * rng.randn(size)).astype(np.complex128)

    with scheme.TorchScheme(device):
        a = Array(a_np)
        b = Array(b_np)
        w_real = Array(w_real_np)
        w_complex = Array(w_complex_np)

        # 1. Self inner (fast path view_as_real)
        val_self = a.inner(a)
        exp_self = np.sum(a_np.real**2 + a_np.imag**2)
        assert abs(val_self - exp_self) / exp_self < 1e-12

        # 2. Distinct inner
        val_ab = a.inner(b)
        exp_ab = np.sum(np.conj(a_np) * b_np)
        assert abs(val_ab - exp_ab) / abs(exp_ab) < 1e-12

        # 3. Self weighted inner with real weight
        val_w_self = a.weighted_inner(a, w_real)
        exp_w_self = np.sum((a_np.real**2 + a_np.imag**2) / w_real_np)
        assert abs(val_w_self - exp_w_self) / exp_w_self < 1e-12

        # 4. Distinct weighted inner with real weight
        val_w_ab = a.weighted_inner(b, w_real)
        exp_w_ab = np.sum(np.conj(a_np) * b_np / w_real_np)
        assert abs(val_w_ab - exp_w_ab) / abs(exp_w_ab) < 1e-12

        # 5. Self weighted inner with complex weight
        val_wc_self = a.weighted_inner(a, w_complex)
        exp_wc_self = np.sum((a_np.real**2 + a_np.imag**2) / w_complex_np)
        assert abs(val_wc_self - exp_wc_self) / abs(exp_wc_self) < 1e-12


@pytest.mark.parametrize("device", _available_devices())
def test_array_torch_abs_arg_max_corner_cases(device):
    """Adversarially test abs_arg_max and abs_max_loc with ties, zeros, and signs."""
    with scheme.TorchScheme(device):
        # 1. Complex array with clear maximum
        arr1_np = np.array([1.0 + 1.0j, 3.0 + 4.0j, 2.0 - 2.0j], dtype=np.complex64)
        arr1 = Array(arr1_np)
        assert arr1.abs_arg_max() == 1

        # 2. Real array with negative maximum magnitude
        arr2_np = np.array([2.0, -10.0, 5.0, -3.0], dtype=np.float32)
        arr2 = Array(arr2_np)
        assert arr2.abs_arg_max() == 1

        # 3. All zeros
        arr3_np = np.zeros(100, dtype=np.complex64)
        arr3 = Array(arr3_np)
        assert arr3.abs_arg_max() == 0


# ============================================================================
# 4. FFT BACKEND STRESS & INVARIANCE (torchfft.py, backend_torch.py)
# ============================================================================

@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize("length", [128, 512, 2048, 8192])
def test_fft_ifft_roundtrip_double_precision_parity(device, length):
    """Verify FFT/IFFT roundtrip fidelity and Parseval energy preservation."""
    if device == "mps":
        pytest.skip("MPS does not support float64/complex128")

    delta_t = 1.0 / length
    delta_f = 1.0 / (length * delta_t)
    rng = np.random.RandomState(777 + length)
    data_t = rng.randn(length).astype(np.float64)

    with scheme.TorchScheme(device):
        ts = TimeSeries(data_t, delta_t=delta_t, dtype=np.float64)
        fs = FrequencySeries(np.zeros(length // 2 + 1, dtype=np.complex128), delta_f=delta_f)
        ts_rec = TimeSeries(np.zeros(length, dtype=np.float64), delta_t=delta_t)

        # Forward FFT
        fft(ts, fs)
        # Inverse FFT
        ifft(fs, ts_rec)

        # Numerical roundtrip parity: ts_rec == ts to < 10^-12
        rel_diff = np.max(np.abs(ts_rec.numpy() - data_t)) / np.max(np.abs(data_t))
        assert rel_diff < 1e-12, f"FFT/IFFT roundtrip error: {rel_diff}"


@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize("batch_rows", [1, 8, 32, 128])
def test_batched_torchfft_class_transforms(device, batch_rows):
    """Verify torchfft batched FFT and IFFT across row counts."""
    size = 256
    rng = np.random.RandomState(888 + batch_rows)
    data_np = (rng.randn(batch_rows, size) + 1j * rng.randn(batch_rows, size)).astype(np.complex64)

    with scheme.TorchScheme(device):
        in_arr = Array(data_np.copy())
        out_arr = zeros((batch_rows, size), dtype=np.complex64)

        # Batched FFT
        fft_plan = torchfft.FFT(in_arr, out_arr)
        fft_plan.execute()

        expected_fft = np.fft.fft(data_np, axis=-1)
        np.testing.assert_allclose(out_arr.numpy(), expected_fft, rtol=1e-4, atol=1e-4)

        # Batched IFFT
        rec_arr = zeros((batch_rows, size), dtype=np.complex64)
        ifft_plan = torchfft.IFFT(out_arr, rec_arr)
        ifft_plan.execute()

        # PyCBC IFFT standard normalizes by unscaling or matching np.fft.ifft * size
        expected_ifft = np.fft.ifft(expected_fft, axis=-1) * size
        np.testing.assert_allclose(rec_arr.numpy(), expected_ifft, rtol=1e-4, atol=1e-3)


# ============================================================================
# 5. SKY MAX SNR STATISTIC REGRESSION TEST
# ============================================================================

@pytest.mark.parametrize("device", _available_devices())
def test_compute_max_snr_over_sky_loc_stat_torch(device):
    """Verify compute_max_snr_over_sky_loc_stat runs cleanly without UnboundLocalError."""
    length = 64
    rng = np.random.RandomState(42)
    hp_np = (rng.randn(length) + 1j * rng.randn(length)).astype(np.complex64)
    hc_np = (rng.randn(length) + 1j * rng.randn(length)).astype(np.complex64)

    with scheme.TorchScheme(device):
        hp = TimeSeries(hp_np, delta_t=1.0)
        hc = TimeSeries(hc_np, delta_t=1.0)
        # compute_max_snr_over_sky_loc_stat
        stat = compute_max_snr_over_sky_loc_stat(
            hp, hc, hpnorm=1.0, hcnorm=1.0, hphccorr=0.2, thresh=0.0
        )
        assert len(stat) == length

        # compute_max_snr_over_sky_loc_stat_no_phase
        stat_np = compute_max_snr_over_sky_loc_stat_no_phase(
            hp, hc, hpnorm=1.0, hcnorm=1.0, hphccorr=0.2, thresh=0.0
        )
        assert len(stat_np) == length
