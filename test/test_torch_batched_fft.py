# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Parity tests for independent transforms in Torch class FFT batches."""

import numpy as np
import pytest

import pycbc
from pycbc import scheme
from pycbc.types import Array, zeros


torch = pytest.importorskip("torch")

if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without torch support", allow_module_level=True)


DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")
SINGLE_PRECISION_DEVICES = list(DEVICES)
if torch.backends.mps.is_available():
    SINGLE_PRECISION_DEVICES.append("mps")

COMPLEX_CASES = [
    (device, dtype)
    for device in DEVICES
    for dtype in (np.complex64, np.complex128)
]
if "mps" in SINGLE_PRECISION_DEVICES:
    COMPLEX_CASES.append(("mps", np.complex64))


def _complex_values(rows, size, dtype, seed):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(rows, size))
        + 1j * rng.normal(size=(rows, size))
    ).astype(dtype)


@pytest.mark.parametrize("device,dtype", COMPLEX_CASES)
@pytest.mark.parametrize("inverse", [False, True])
def test_batched_complex_transforms_are_independent(device, dtype, inverse):
    from pycbc.fft import torchfft

    rows, size = 3, 31
    values = _complex_values(rows, size, dtype, seed=1701)
    if inverse:
        expected = np.fft.ifft(values, axis=-1) * size
        engine_type = torchfft.IFFT
    else:
        expected = np.fft.fft(values, axis=-1)
        engine_type = torchfft.FFT

    with scheme.TorchScheme(device):
        source = Array(values.ravel(), dtype=dtype)
        target = zeros(rows * size, dtype=dtype)
        source_before = source._data.tensor.clone()
        engine_type(source, target, nbatch=rows, size=size).execute()

        assert torch.equal(source._data.tensor, source_before)
        assert target._data.tensor.device.type == device
        assert target.dtype == np.dtype(dtype)
        got = target.numpy().reshape(rows, size)

    tolerance = 2e-6 if dtype == np.complex64 else 2e-13
    np.testing.assert_allclose(got, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("device", SINGLE_PRECISION_DEVICES)
@pytest.mark.parametrize("size", [15, 16])
def test_batched_real_transforms_preserve_row_boundaries(device, size):
    from pycbc.fft import torchfft

    rows = 3
    rng = np.random.default_rng(1702 + size)
    real_values = rng.normal(size=(rows, size)).astype(np.float32)
    frequency_values = _complex_values(
        rows, size // 2 + 1, np.complex64, seed=1802 + size
    )

    with scheme.TorchScheme(device):
        real_input = Array(real_values.ravel())
        frequency_output = zeros(
            rows * (size // 2 + 1), dtype=np.complex64
        )
        torchfft.FFT(
            real_input, frequency_output, nbatch=rows, size=size
        ).execute()

        frequency_input = Array(frequency_values.ravel())
        real_output = zeros(rows * size, dtype=np.float32)
        torchfft.IFFT(
            frequency_input, real_output, nbatch=rows, size=size
        ).execute()

        got_forward = frequency_output.numpy().reshape(rows, -1)
        got_inverse = real_output.numpy().reshape(rows, size)

    expected_forward = np.fft.rfft(real_values, n=size, axis=-1)
    expected_inverse = (
        np.fft.irfft(frequency_values, n=size, axis=-1) * size
    )
    np.testing.assert_allclose(
        got_forward, expected_forward, rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(
        got_inverse, expected_inverse, rtol=2e-6, atol=2e-6
    )


@pytest.mark.parametrize("device", SINGLE_PRECISION_DEVICES)
def test_batched_complex_ifft_handles_inplace_and_partial_aliases(device):
    from pycbc.fft import torchfft

    rows, size = 3, 32
    values = _complex_values(rows, size, np.complex64, seed=1703)
    expected = np.fft.ifft(values, axis=-1) * size

    with scheme.TorchScheme(device):
        inplace = Array(values.ravel())
        torchfft.IFFT(
            inplace, inplace, nbatch=rows, size=size
        ).execute()
        got_inplace = inplace.numpy().reshape(rows, size)

        storage = Array(
            np.concatenate((values.ravel(), np.zeros(1, np.complex64)))
        )
        source = storage[:-1]
        target = storage[1:]
        torchfft.IFFT(
            source, target, nbatch=rows, size=size
        ).execute()
        got_overlap = target.numpy().reshape(rows, size)

    np.testing.assert_allclose(
        got_inplace, expected, rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(
        got_overlap, expected, rtol=2e-6, atol=2e-6
    )


@pytest.mark.parametrize("device", SINGLE_PRECISION_DEVICES)
@pytest.mark.parametrize("size", [15, 16])
def test_batched_inplace_real_transforms_skip_per_row_padding(device, size):
    from pycbc.fft import torchfft

    rows = 3
    padded_size = 2 * (size // 2 + 1)
    rng = np.random.default_rng(1704 + size)
    real_values = rng.normal(size=(rows, size)).astype(np.float32)
    frequency_values = _complex_values(
        rows, size // 2 + 1, np.complex64, seed=1804 + size
    )

    with scheme.TorchScheme(device):
        forward_storage = zeros(rows * padded_size, dtype=np.float32)
        forward_rows = forward_storage._data.tensor.view(rows, padded_size)
        forward_rows[:, :size].copy_(
            torch.as_tensor(real_values, device=device)
        )
        forward_output = forward_storage.view(np.complex64)
        torchfft.FFT(
            forward_storage, forward_output, nbatch=rows, size=size
        ).execute()
        got_forward = forward_output.numpy().reshape(rows, -1)

        inverse_storage = zeros(rows * padded_size, dtype=np.float32)
        inverse_input = inverse_storage.view(np.complex64)
        inverse_input._data.tensor.copy_(
            torch.as_tensor(frequency_values.ravel(), device=device)
        )
        torchfft.IFFT(
            inverse_input, inverse_storage, nbatch=rows, size=size
        ).execute()
        got_inverse = inverse_storage.numpy().reshape(rows, padded_size)[
            :, :size
        ]

    expected_forward = np.fft.rfft(real_values, n=size, axis=-1)
    expected_inverse = (
        np.fft.irfft(frequency_values, n=size, axis=-1) * size
    )
    np.testing.assert_allclose(
        got_forward, expected_forward, rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(
        got_inverse, expected_inverse, rtol=2e-6, atol=2e-6
    )


def test_batched_fft_requires_and_accepts_explicit_transform_size():
    from pycbc.fft import torchfft

    with scheme.TorchScheme("cpu"):
        source = zeros(32, dtype=np.complex64)
        target = zeros(32, dtype=np.complex64)
        with pytest.raises(ValueError, match="size cannot be 'None'"):
            torchfft.FFT(source, target, nbatch=2)
        with pytest.raises(ValueError, match="size cannot be 'None'"):
            torchfft.IFFT(source, target, nbatch=2)

        torchfft.FFT(source, target, nbatch=2, size=16).execute()
        torchfft.IFFT(source, target, nbatch=2, size=16).execute()


def test_batched_ifft_preserves_out_autograd_rejection():
    from pycbc.fft import torchfft

    with scheme.TorchScheme("cpu"):
        source = zeros(32, dtype=np.complex64)
        target = zeros(32, dtype=np.complex64)
        source._data.tensor.requires_grad_(True)
        engine = torchfft.IFFT(source, target, nbatch=2, size=16)

        with pytest.raises(RuntimeError):
            engine.execute()


def _single_precision_case(kind, rows, size):
    if kind.startswith("c2c"):
        values = _complex_values(
            rows, size, np.complex64, seed=1900 + size
        )
        output_dtype = np.complex64
        output_size = size
        if kind == "c2c_fft":
            expected = np.fft.fft(
                values.astype(np.complex128), axis=-1
            )
        else:
            expected = (
                np.fft.ifft(values.astype(np.complex128), axis=-1)
                * size
            )
    elif kind == "r2c_fft":
        rng = np.random.default_rng(2000 + size)
        values = rng.normal(size=(rows, size)).astype(np.float32)
        output_dtype = np.complex64
        output_size = size // 2 + 1
        expected = np.fft.rfft(values.astype(np.float64), axis=-1)
    else:
        values = _complex_values(
            rows, size // 2 + 1, np.complex64, seed=2100 + size
        )
        output_dtype = np.float32
        output_size = size
        expected = (
            np.fft.irfft(
                values.astype(np.complex128), n=size, axis=-1
            )
            * size
        )
    return values, output_dtype, output_size, expected


@pytest.mark.parametrize("device", SINGLE_PRECISION_DEVICES)
@pytest.mark.parametrize(
    "kind", ["c2c_fft", "c2c_ifft", "r2c_fft", "c2r_ifft"]
)
def test_single_precision_batches_reuse_promoted_workspaces(device, kind):
    """The released batch path improves on native float32 FFT precision."""
    from pycbc.fft import torchfft

    rows, size = 2, 4096
    values, output_dtype, output_size, expected = _single_precision_case(
        kind, rows, size
    )
    engine_type = torchfft.FFT if kind.endswith("fft") and not kind.endswith(
        "ifft"
    ) else torchfft.IFFT

    with scheme.TorchScheme(device):
        source = Array(values.ravel(), dtype=values.dtype)
        target = zeros(rows * output_size, dtype=output_dtype)
        engine = engine_type(
            source, target, nbatch=rows, size=size
        )
        plan = engine._promoted_batch_plan
        assert plan is not None
        assert plan.source.dtype in (torch.float64, torch.complex128)
        assert plan.target.dtype in (torch.float64, torch.complex128)
        expected_workspace_device = "cpu" if device == "mps" else device
        assert plan.source.device.type == expected_workspace_device

        workspace_ptrs = (plan.source.data_ptr(), plan.target.data_ptr())
        source_before = source._data.tensor.clone()
        engine.execute()
        assert torch.equal(source._data.tensor, source_before)

        # The class API promises fixed buffers, but a defensive same-contract
        # storage replacement must neither use stale pointers nor allocate a
        # fresh promoted workspace.
        source._data._set_tensor(source._data.tensor.clone())
        target._data._set_tensor(torch.empty_like(target._data.tensor))
        target_version = target._data.tensor._version
        source_before = source._data.tensor.clone()
        engine.execute()
        assert torch.equal(source._data.tensor, source_before)
        assert target._data.tensor._version == target_version + 1
        assert workspace_ptrs == (
            plan.source.data_ptr(), plan.target.data_ptr()
        )
        got = target.numpy().reshape(rows, output_size)

    difference = got.astype(expected.dtype) - expected
    relative_l2 = np.linalg.norm(difference.ravel()) / np.linalg.norm(
        expected.ravel()
    )
    max_abs = np.max(np.abs(difference))
    assert relative_l2 < 4e-8
    assert max_abs < (
        2 * np.finfo(np.float32).eps * np.max(np.abs(expected))
    )


@pytest.mark.parametrize("device", SINGLE_PRECISION_DEVICES)
@pytest.mark.parametrize(
    "kind", ["c2c_fft", "c2c_ifft", "r2c_fft", "c2r_ifft"]
)
def test_promoted_batches_match_or_exceed_legacy_fftw(device, kind):
    """Guard the precision promise against the established CPU backend."""
    try:
        from pycbc.types import array_cpu  # noqa: F401
        from pycbc.fft import fftw, torchfft
    except (ImportError, OSError, RuntimeError) as exc:
        pytest.skip(f"compiled CPU FFT backend unavailable: {exc}")

    rows, size = 2, 4096
    values, output_dtype, output_size, expected = _single_precision_case(
        kind, rows, size
    )
    engine_type = torchfft.FFT if kind.endswith("fft") and not kind.endswith(
        "ifft"
    ) else torchfft.IFFT
    legacy_type = fftw.FFT if engine_type is torchfft.FFT else fftw.IFFT

    with scheme.CPUScheme(1):
        legacy_source = Array(values.ravel(), dtype=values.dtype)
        legacy_target = zeros(rows * output_size, dtype=output_dtype)
        legacy_type(
            legacy_source, legacy_target, nbatch=rows, size=size
        ).execute()
        legacy = legacy_target.numpy().reshape(rows, output_size).copy()

    with scheme.TorchScheme(device):
        source = Array(values.ravel(), dtype=values.dtype)
        target = zeros(rows * output_size, dtype=output_dtype)
        engine_type(source, target, nbatch=rows, size=size).execute()
        candidate = target.numpy().reshape(rows, output_size)

    def error(result):
        difference = result.astype(expected.dtype) - expected
        return (
            np.linalg.norm(difference.ravel()) / np.linalg.norm(
                expected.ravel()
            ),
            np.max(np.abs(difference)),
        )

    candidate_rel, candidate_max = error(candidate)
    legacy_rel, legacy_max = error(legacy)
    assert candidate_rel <= legacy_rel
    assert candidate_max <= legacy_max
    direct_rel = np.linalg.norm(
        (candidate.astype(expected.dtype) - legacy).ravel()
    ) / np.linalg.norm(legacy.ravel())
    assert direct_rel < 3e-7


def test_promoted_workspace_chunks_rows_and_handles_final_partial(
    monkeypatch,
):
    from pycbc.fft import torchfft

    rows, size = 5, 64
    monkeypatch.setitem(
        torchfft._PROMOTED_BATCH_MAX_ELEMENTS, "cpu", 2 * size
    )
    values = _complex_values(rows, size, np.complex64, seed=2200)
    expected = np.fft.ifft(
        values.astype(np.complex128), axis=-1
    ) * size

    with scheme.TorchScheme("cpu"):
        source = Array(values.ravel())
        target = zeros(rows * size, dtype=np.complex64)
        engine = torchfft.IFFT(
            source, target, nbatch=rows, size=size
        )
        assert engine._promoted_batch_plan.rows == 2
        assert engine._promoted_batch_plan.source.shape == (2, size)
        engine.execute()
        got = target.numpy().reshape(rows, size)

        inplace = Array(values.ravel())
        inplace_engine = torchfft.IFFT(
            inplace, inplace, nbatch=rows, size=size
        )
        assert inplace_engine._promoted_batch_plan.rows == 2
        inplace_engine.execute()
        got_inplace = inplace.numpy().reshape(rows, size)

    np.testing.assert_allclose(got, expected, rtol=5e-7, atol=5e-7)
    np.testing.assert_allclose(
        got_inplace, expected, rtol=5e-7, atol=5e-7
    )


@pytest.mark.parametrize("target_offset", [-1, 1])
def test_promoted_chunks_snapshot_cross_pointer_overlap(
    monkeypatch, target_offset
):
    """Earlier chunk writes must not corrupt later overlapping source rows."""
    from pycbc.fft import torchfft

    rows, size = 5, 64
    monkeypatch.setitem(
        torchfft._PROMOTED_BATCH_MAX_ELEMENTS, "cpu", 2 * size
    )
    values = _complex_values(rows, size, np.complex64, seed=2300)
    expected = np.fft.ifft(
        values.astype(np.complex128), axis=-1
    ) * size

    with scheme.TorchScheme("cpu"):
        padding = np.zeros(1, dtype=np.complex64)
        if target_offset > 0:
            storage = Array(np.concatenate((values.ravel(), padding)))
            source, target = storage[:-1], storage[1:]
        else:
            storage = Array(np.concatenate((padding, values.ravel())))
            source, target = storage[1:], storage[:-1]
        engine = torchfft.IFFT(
            source, target, nbatch=rows, size=size
        )
        assert engine._promoted_batch_plan.rows == 2
        engine.execute()
        got = target.numpy().reshape(rows, size)

    np.testing.assert_allclose(got, expected, rtol=5e-7, atol=5e-7)


def test_promoted_overlap_snapshot_allocation_failure_propagates(
    monkeypatch,
):
    """An overlap snapshot failure must not select a lower-precision path."""
    from pycbc.fft import torchfft

    rows, size = 5, 64
    monkeypatch.setitem(
        torchfft._PROMOTED_BATCH_MAX_ELEMENTS, "cpu", 2 * size
    )
    values = _complex_values(rows, size, np.complex64, seed=2400)

    with scheme.TorchScheme("cpu"):
        storage = Array(
            np.concatenate((values.ravel(), np.zeros(1, np.complex64)))
        )
        source, target = storage[:-1], storage[1:]
        source_pointer = source._data.tensor.data_ptr()
        engine = torchfft.IFFT(
            source, target, nbatch=rows, size=size
        )
        original_clone = torch.Tensor.clone

        def fail_source_snapshot(tensor, *args, **kwargs):
            if tensor.data_ptr() == source_pointer:
                raise RuntimeError("forced overlap snapshot allocation failure")
            return original_clone(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "clone", fail_source_snapshot)
        with pytest.raises(
            RuntimeError, match="forced overlap snapshot allocation failure"
        ):
            engine.execute()


@pytest.mark.parametrize("device", SINGLE_PRECISION_DEVICES)
def test_direct_batch_ifft_execution_and_env_gate(device, monkeypatch):
    """Test direct single-precision 2D batched IFFT and env gate behavior."""
    from pycbc.fft import torchfft

    rows, size = 4, 128
    values = _complex_values(rows, size, np.complex64, seed=3100)
    expected = np.fft.ifft(values, axis=-1) * size

    # 1. Explicitly enabled via PYCBC_TORCH_DIRECT_BATCH_IFFT=1
    monkeypatch.setenv("PYCBC_TORCH_DIRECT_BATCH_IFFT", "1")
    with scheme.TorchScheme(device):
        source = Array(values.ravel())
        target = zeros(rows * size, dtype=np.complex64)
        engine = torchfft.IFFT(source, target, nbatch=rows, size=size)
        assert torchfft._can_use_direct_batch_ifft(engine)
        engine.execute()
        got = target.numpy().reshape(rows, size)
        np.testing.assert_allclose(got, expected, rtol=2e-6, atol=2e-6)

        # In-place test
        inplace = Array(values.ravel())
        engine_inplace = torchfft.IFFT(
            inplace, inplace, nbatch=rows, size=size
        )
        engine_inplace.execute()
        got_inplace = inplace.numpy().reshape(rows, size)
        np.testing.assert_allclose(got_inplace, expected, rtol=2e-6, atol=2e-6)

    # 2. Explicitly disabled via PYCBC_TORCH_DIRECT_BATCH_IFFT=0 (falls back)
    monkeypatch.setenv("PYCBC_TORCH_DIRECT_BATCH_IFFT", "0")
    with scheme.TorchScheme(device):
        source = Array(values.ravel())
        target = zeros(rows * size, dtype=np.complex64)
        engine = torchfft.IFFT(source, target, nbatch=rows, size=size)
        assert not torchfft._can_use_direct_batch_ifft(engine)
        engine.execute()
        got_fallback = target.numpy().reshape(rows, size)
        np.testing.assert_allclose(
            got_fallback, expected, rtol=2e-6, atol=2e-6
        )
