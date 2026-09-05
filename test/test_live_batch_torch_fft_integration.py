# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Production-path integration tests for Torch live-batch filtering."""

import types
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import pycbc  # noqa: E402
from pycbc import scheme  # noqa: E402
from pycbc.filter import matchedfilter  # noqa: E402
from pycbc.types import FrequencySeries  # noqa: E402


if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without Torch support", allow_module_level=True)


def _available_torch_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


TORCH_DEVICES = _available_torch_devices()


def _live_batch_inputs():
    rows, size = 3, 32
    frequency_size = size // 2 + 1
    rng = np.random.default_rng(9182)
    templates = (
        rng.normal(size=(rows, frequency_size))
        + 1j * rng.normal(size=(rows, frequency_size))
    ).astype(np.complex64)
    data = (
        rng.normal(size=frequency_size)
        + 1j * rng.normal(size=frequency_size)
    ).astype(np.complex64)
    return templates, data, size


def _build_live_batch(template_values, data_values, size):
    templates = []
    for index, values in enumerate(template_values):
        template = FrequencySeries(values, delta_f=1.0 / size)
        template.id = 10 + index
        template.params = np.array(
            [(20.0 + index,)], dtype=[("mass1", np.float32)]
        )[0]
        template.sigmasq = lambda _psd: 1.0
        templates.append(template)

    batch = matchedfilter.LiveBatchMatchedFilter(
        templates,
        snr_threshold=0.0,
        chisq_bins=0,
        sg_chisq=types.SimpleNamespace(),
        maxelements=len(templates) * size,
    )
    data = FrequencySeries(data_values, delta_f=1.0 / size)
    data.psd = object()
    batch.set_data(types.SimpleNamespace(
        overwhitened_data=lambda _delta_f: data,
        trim_padding=0,
        blocksize=size,
        sample_rate=1,
        start_time=100.0,
    ))
    return batch, templates, data


@pytest.mark.parametrize("device", TORCH_DEVICES)
def test_live_batch_uses_real_batched_ifft_without_resizing(device):
    template_values, data_values, size = _live_batch_inputs()
    frequency_size = data_values.size
    rows = len(template_values)

    with scheme.TorchScheme(device):
        batch, templates, data = _build_live_batch(
            template_values, data_values, size
        )
        mid = batch.mids[0]
        engine = batch.ifts[mid]
        assert engine.__class__.__module__ == "pycbc.fft.torchfft"
        assert engine.nbatch == rows
        assert engine.size == size

        template_before = [
            template._data.tensor.clone() for template in templates
        ]
        data_before = data._data.tensor.clone()
        output_pointers = [template.cout.ptr for template in templates]

        with warnings.catch_warnings(record=True) as caught:
            result, veto_info = batch._process_batch()

        assert not any(
            "was resized" in str(item.message) for item in caught
        )
        assert [len(template.cout) for template in templates] == [size] * rows
        assert [template.cout.ptr for template in templates] == output_pointers
        assert all(
            torch.equal(template._data.tensor, before)
            for template, before in zip(templates, template_before)
        )
        assert torch.equal(data._data.tensor, data_before)

        correlation = batch.cout_mem[mid].numpy().reshape(rows, size).copy()
        output = batch.out_mem[mid].numpy().reshape(rows, size).copy()

    expected_correlation = np.zeros((rows, size), dtype=np.complex64)
    expected_correlation[:, :frequency_size] = (
        np.conj(template_values) * data_values
    )
    expected_output = (
        np.fft.ifft(
            expected_correlation.astype(np.complex128), axis=-1
        ) * size
    )

    np.testing.assert_allclose(
        correlation[:, :frequency_size],
        expected_correlation[:, :frequency_size],
        rtol=2e-7,
        atol=2e-7,
    )
    np.testing.assert_array_equal(
        correlation[:, frequency_size:], 0
    )
    np.testing.assert_allclose(
        output, expected_output, rtol=4e-7, atol=2e-6
    )

    relative_l2 = np.linalg.norm(
        (output.astype(np.complex128) - expected_output).ravel()
    ) / np.linalg.norm(expected_output.ravel())
    assert relative_l2 < 1.5e-7

    peak_indices = np.abs(expected_output).argmax(axis=1)
    peaks = expected_output[np.arange(rows), peak_indices]
    norm = 4.0 / size
    np.testing.assert_array_equal(result["template_id"], [10, 11, 12])
    np.testing.assert_allclose(
        result["snr"], np.abs(peaks) * norm, rtol=4e-7, atol=2e-7
    )
    np.testing.assert_allclose(
        result["coa_phase"], np.angle(peaks), rtol=4e-7, atol=2e-7
    )
    np.testing.assert_array_equal(result["end_time"], 100 + peak_indices)
    assert len(veto_info) == rows


@pytest.mark.parametrize("device", TORCH_DEVICES)
def test_live_batch_correlator_preserves_autograd_rejection(device):
    template_values, data_values, size = _live_batch_inputs()

    with scheme.TorchScheme(device):
        batch, templates, data = _build_live_batch(
            template_values, data_values, size
        )
        templates[0]._data.tensor.requires_grad_(True)
        with pytest.raises(RuntimeError, match="automatic differentiation"):
            batch.corr[0].execute(data)


def test_live_batch_maxelements_cuda_and_cpu_defaults(monkeypatch):
    size = 131072
    frequency_size = size // 2 + 1
    templates = []
    for i in range(128):
        t = FrequencySeries(
            np.zeros(frequency_size, dtype=np.complex64),
            delta_f=1.0 / size,
        )
        t.id = i
        templates.append(t)

    # 1. On CPU scheme (without PYCBC_BATCH_MAXELEMENTS):
    # default maxelements = 2**27 = 134217728.
    # grabs = 2**27 // 131072 = 1024 -> all 128 templates in 1 chunk.
    monkeypatch.delenv("PYCBC_BATCH_MAXELEMENTS", raising=False)
    monkeypatch.setattr(
        "pycbc.hardware.get_optimal_batch_maxelements",
        lambda is_cuda=False, **kwargs: 2**23 if is_cuda else 2**27,
    )
    with scheme.CPUScheme():
        batch = matchedfilter.LiveBatchMatchedFilter(
            templates,
            snr_threshold=0.0,
            chisq_bins=0,
            sg_chisq=types.SimpleNamespace(),
        )
        assert len(batch.chunks) == 1
        assert batch.chunks[0] == 128

    # 2. On CUDA (simulated via _is_cuda_scheme):
    # default maxelements = 2**23 = 8388608.
    # grabs = 2**23 // 131072 = 64 (B_tile = 64 templates per chunk).
    # 128 templates -> 2 chunks of 64.
    monkeypatch.setattr(
        matchedfilter, "_is_cuda_scheme", lambda *args, **kwargs: True
    )
    batch_cuda = matchedfilter.LiveBatchMatchedFilter(
        templates,
        snr_threshold=0.0,
        chisq_bins=0,
        sg_chisq=types.SimpleNamespace(),
    )
    assert len(batch_cuda.chunks) == 2
    assert list(batch_cuda.chunks) == [64, 64]

    # 3. Environment override via PYCBC_BATCH_MAXELEMENTS:
    monkeypatch.setenv("PYCBC_BATCH_MAXELEMENTS", str(32 * size))
    batch_env = matchedfilter.LiveBatchMatchedFilter(
        templates,
        snr_threshold=0.0,
        chisq_bins=0,
        sg_chisq=types.SimpleNamespace(),
    )
    assert list(batch_env.chunks) == [32, 32, 32, 32]


@pytest.mark.parametrize("device", TORCH_DEVICES)
def test_torch_batched_ifft_inplace_shares_storage(device):
    from pycbc.fft import torchfft
    from pycbc.types import Array

    rows, size = 4, 64
    rng = np.random.default_rng(42)
    dtype = np.complex64 if device == "mps" else np.complex128
    values = (
        rng.normal(size=(rows, size))
        + 1j * rng.normal(size=(rows, size))
    ).astype(dtype)

    with scheme.TorchScheme(device):
        arr = Array(values.ravel())
        orig_ptr = arr._data.tensor.data_ptr()
        engine = torchfft.IFFT(arr, arr, nbatch=rows, size=size)
        engine.execute()

        # In-place execution should preserve the buffer pointer
        assert arr._data.tensor.data_ptr() == orig_ptr

        expected = np.fft.ifft(values, axis=-1) * size
        tol = 1e-5 if dtype == np.complex64 else 1e-12
        np.testing.assert_allclose(
            arr.numpy().reshape(rows, size),
            expected,
            rtol=tol,
            atol=tol,
        )
