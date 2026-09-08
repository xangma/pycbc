# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch fft writes regression tests."""

import numpy as np
import pytest
from pycbc import scheme
from pycbc.types import Array
import pycbc


torch = pytest.importorskip("torch")

if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without torch support", allow_module_level=True)


@pytest.mark.parametrize(
    ("operation", "itype", "otype", "inverse"),
    [
        ("fft", "complex", "complex", False),
        ("rfft", "real", "complex", False),
        ("ifft", "complex", "complex", True),
        ("irfft", "complex", "real", True),
    ],
)
def test_fft_writes_out_of_place_results_directly(
    monkeypatch, operation, itype, otype, inverse
):
    from pycbc.fft import torchfft

    rng = np.random.default_rng(1701)
    size = 32
    if operation == "rfft":
        input_values = rng.normal(size=size).astype(np.float32)
        output_values = np.empty(size // 2 + 1, np.complex64)
    elif operation == "irfft":
        input_values = (
            rng.normal(size=size // 2 + 1) + 1j * rng.normal(size=size // 2 + 1)
        ).astype(np.complex64)
        output_values = np.empty(size, np.float32)
    else:
        input_values = (rng.normal(size=size) + 1j * rng.normal(size=size)).astype(
            np.complex64
        )
        output_values = np.empty(size, np.complex64)

    with scheme.TorchScheme("cpu"):
        input_array = Array(input_values)
        output_array = Array(output_values)
        original_transform = getattr(torch.fft, operation)
        expected = original_transform(
            input_array._data.tensor,
            n=size,
            norm="forward" if inverse else None,
        )
        direct_outputs = []
        normalizations = []

        def record_transform(input_tensor, n=None, dim=-1, norm=None, *, out=None):
            direct_outputs.append(out)
            normalizations.append(norm)
            return original_transform(input_tensor, n=n, dim=dim, norm=norm, out=out)

        monkeypatch.setattr(getattr(torchfft.torch, "fft"), operation, record_transform)
        transform = torchfft.ifft if inverse else torchfft.fft
        transform(input_array, output_array, None, itype, otype)

        assert direct_outputs == [output_array._data.tensor]
        assert normalizations == (["forward"] if inverse else [None])
        assert torch.equal(output_array._data.tensor, expected)


@pytest.mark.parametrize(("operation", "inverse"), [("fft", False), ("ifft", True)])
@pytest.mark.parametrize("partial_overlap", (False, True))
def test_aliased_complex_fft_preserves_allocation_before_copy(
    monkeypatch, operation, inverse, partial_overlap
):
    from pycbc.fft import torchfft

    rng = np.random.default_rng(2701)
    values = (rng.normal(size=32) + 1j * rng.normal(size=32)).astype(np.complex64)

    with scheme.TorchScheme("cpu"):
        if partial_overlap:
            storage = Array(np.append(values, np.complex64(0)))
            input_array = storage[:-1]
            output_array = storage[1:]
        else:
            input_array = Array(values)
            output_array = input_array
        original_transform = getattr(torch.fft, operation)
        expected = original_transform(
            input_array._data.tensor.clone(),
            n=len(input_array),
            norm="forward" if inverse else None,
        )
        direct_outputs = []
        normalizations = []

        def record_transform(input_tensor, n=None, dim=-1, norm=None, *, out=None):
            direct_outputs.append(out)
            normalizations.append(norm)
            if out is None:
                return original_transform(input_tensor, n=n, dim=dim, norm=norm)
            return original_transform(input_tensor, n=n, dim=dim, norm=norm, out=out)

        monkeypatch.setattr(getattr(torchfft.torch, "fft"), operation, record_transform)
        transform = torchfft.ifft if inverse else torchfft.fft
        transform(input_array, output_array, None, "complex", "complex")

        assert direct_outputs == [None]
        assert normalizations == (["forward"] if inverse else [None])
        assert torch.equal(output_array._data.tensor, expected)


@pytest.mark.parametrize("size", (30, 32))
@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
@pytest.mark.parametrize(
    ("operation", "otype"),
    (("ifft", "complex"), ("irfft", "real")),
)
def test_ifft_preserves_unnormalized_contract(size, dtype, operation, otype):
    """The optimized inverse path retains PyCBC's backend normalization."""
    from pycbc.fft import torchfft

    rng = np.random.default_rng(3701)
    input_size = size if operation == "ifft" else size // 2 + 1
    input_values = (
        rng.normal(size=input_size) + 1j * rng.normal(size=input_size)
    ).astype(dtype)
    output_dtype = dtype if otype == "complex" else np.empty((), dtype).real.dtype
    output_values = np.empty(size, output_dtype)
    if operation == "ifft":
        expected = np.fft.ifft(input_values, n=size) * size
    else:
        expected = np.fft.irfft(input_values, n=size) * size

    with scheme.TorchScheme("cpu"):
        input_array = Array(input_values)
        output_array = Array(output_values)
        torchfft.ifft(input_array, output_array, None, "complex", otype)
        actual = output_array._data.tensor.detach().cpu().numpy().copy()

    tolerance = 2e-6 if dtype == np.complex64 else 1e-12
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
