# Copyright (C) 2026  The PyCBC team
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Native Torch CPU linear decompression and its conservative fallbacks."""

from unittest import mock

import numpy
import pytest

torch = pytest.importorskip("torch")

# These imports require the optional Torch dependency checked above.
from pycbc.scheme import CPUScheme, TorchScheme  # noqa: E402
from pycbc.types import FrequencySeries  # noqa: E402
from pycbc.types.array_torch import TorchArrayData  # noqa: E402
from pycbc.waveform import (  # noqa: E402
    decompress_cpu_cython, decompress_torch,
)
from pycbc.waveform.compress import fd_decompress  # noqa: E402


def _inputs(dtype=numpy.float64):
    frequencies = numpy.array([0., 16.3, 32.9, 80.], dtype=dtype)
    amp = (numpy.exp(-0.01 * frequencies) + 0.1).astype(dtype)
    phase = (0.31 * frequencies + 0.011 * frequencies**2).astype(dtype)
    return amp, phase, frequencies


def _linear(amp, phase, frequencies, output, df=0.125, imin=0, start=0):
    return decompress_torch.inline_linear_interp(
        amp, phase, frequencies, output, df, 0., imin, start,
    )


@pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
@pytest.mark.parametrize("frequencies,df,lower,length", [
    ([0., 0.43, 1.17, 2.05, 3.4, 4.8, 6.35, 8.], 0.2, 0.71, 48),
    ([0., 0.3, 0.7, 1.5], 0.1, 0.2, 24),
    ([0., 16.3, 32.9, 80.], 0.125, 0., 651),
    ([0., 16.3, 32.9, 80.], 0.125, 17., 230),
    ([0.5, 0.6, 0.7, 1.1, 1.2, 3.], 0.25, None, 20),
    ([0., 80.], 0.125, 0., 651),
])
def test_native_grid_parity_and_output_views(
    dtype, frequencies, df, lower, length,
):
    """Cover rounded df, dense knots, reseeding, truncation and both tails."""
    frequencies = numpy.array(frequencies, dtype=dtype)
    amp = (numpy.exp(-0.01 * frequencies) + 0.1).astype(dtype)
    phase = (0.31 * frequencies + 0.011 * frequencies**2).astype(dtype)
    complex_dtype = (numpy.complex64 if dtype is numpy.float32
                     else numpy.complex128)
    tensor_dtype = (torch.complex64 if dtype is numpy.float32
                    else torch.complex128)
    epoch = 1234567890.125
    with CPUScheme():
        reference = FrequencySeries(
            numpy.full(length, 17. - 4.j, dtype=complex_dtype),
            delta_f=df, epoch=epoch, copy=False,
        )
        fd_decompress(amp, phase, frequencies, out=reference, f_lower=lower)
        expected = reference.numpy().copy()

    with TorchScheme("cpu"):
        backing = torch.full((length + 7,), 17. - 4.j, dtype=tensor_dtype)
        storage = backing[3:-4]
        output = FrequencySeries(
            TorchArrayData(storage), delta_f=df, epoch=epoch, copy=False,
        )
        pointer = storage.data_ptr()
        version = backing._version
        with mock.patch.object(
            decompress_torch, "_inline_interp",
            side_effect=AssertionError("native linear dispatch was skipped"),
        ):
            returned = fd_decompress(
                amp, phase, frequencies, out=output, f_lower=lower,
            )
        assert returned is output
        assert output.delta_f == df
        assert float(output.epoch) == epoch
        assert storage.data_ptr() == pointer
        assert backing._version > version
        assert storage._version == backing._version
        numpy.testing.assert_array_equal(storage.numpy(), expected)
        assert torch.all(backing[:3] == 17. - 4.j)
        assert torch.all(backing[-4:] == 17. - 4.j)


@pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
@pytest.mark.parametrize("tensor_inputs", [False, True])
def test_native_shares_inputs_and_observes_host_mutations(
    dtype, tensor_inputs,
):
    inputs = _inputs(dtype)
    tensor_dtype = (torch.complex64 if dtype is numpy.float32
                    else torch.complex128)
    out = torch.empty(651, dtype=tensor_dtype)
    values = tuple(torch.from_numpy(value) for value in inputs)
    arguments = values if tensor_inputs else inputs
    name = ("decomp_ccode_float" if dtype is numpy.float32
            else "decomp_ccode_double")
    with mock.patch.object(
        decompress_cpu_cython, name,
        wraps=getattr(decompress_cpu_cython, name),
    ) as kernel:
        assert _linear(*arguments, out) is out
        call = kernel.call_args.args
        assert call[0].__array_interface__["data"][0] == out.data_ptr()
        expected_inputs = (inputs[2], *inputs[:2])
        for actual, original in zip(call[4:7], expected_inputs):
            assert numpy.shares_memory(actual, original)
        first = out.clone()
        versions = tuple(value._version for value in values)
        inputs[0][:] *= 2
        assert tuple(value._version for value in values) == versions
        _linear(*arguments, out)
        assert kernel.call_count == 2
    # No copied/cached amplitudes may survive a write through a NumPy alias.
    torch.testing.assert_close(out, first * 2, rtol=0, atol=0)


@pytest.mark.parametrize("case", [
    "strided_output", "conjugate_output", "negative_output",
    "strided_input", "readonly_input", "different_precision", "subclass",
])
def test_unsupported_storage_keeps_torch_route(case):
    amp, phase, frequencies = _inputs()
    out = torch.full((651,), 17. - 4.j, dtype=torch.complex128)
    if case == "strided_output":
        out = torch.full((1302,), 17. - 4.j, dtype=out.dtype)[::2]
    elif case == "conjugate_output":
        out = out.conj()
    elif case == "negative_output":
        out = torch._neg_view(out)
    elif case == "strided_input":
        amp = numpy.repeat(amp, 2)[::2]
    elif case == "readonly_input":
        amp.flags.writeable = False
    elif case == "different_precision":
        amp = torch.from_numpy(amp).to(torch.bfloat16)
    elif case == "subclass":
        class TensorSubclass(torch.Tensor):
            pass
        amp = torch.from_numpy(amp).as_subclass(TensorSubclass)
    expected = torch.empty_like(out)
    decompress_torch._inline_interp(
        amp, phase, frequencies, expected, 0.125, 0, 0, 1,
    )
    with mock.patch.object(
        decompress_torch, "_inline_interp",
        wraps=decompress_torch._inline_interp,
    ) as fallback:
        _linear(amp, phase, frequencies, out)
        fallback.assert_called_once()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_output_input_alias_stays_in_torch():
    _, phase, frequencies = _inputs()
    out = torch.ones(651, dtype=torch.complex128)
    amp = out.real[:4]
    with mock.patch.object(
        decompress_torch, "_inline_interp",
        wraps=decompress_torch._inline_interp,
    ) as fallback:
        _linear(amp, phase, frequencies, out)
        fallback.assert_called_once()
    # Preserve the existing alias behavior: zeroing out also zeros amplitude.
    assert torch.count_nonzero(out) == 0


def test_sub_bin_first_interval_preserves_lower_cutoff():
    frequencies = numpy.array([0., 0.11, 0.12, 1.])
    amp = numpy.ones(4)
    phase = numpy.zeros(4)
    out = torch.ones(16, dtype=torch.complex128)
    with mock.patch.object(
        decompress_torch, "_inline_interp",
        wraps=decompress_torch._inline_interp,
    ) as fallback:
        _linear(amp, phase, frequencies, out, df=0.25, imin=1, start=1)
        fallback.assert_called_once()
    assert out[0] == 0
    assert torch.all(out[1:5] == 1)
    assert torch.count_nonzero(out[5:]) == 0


@pytest.mark.parametrize("sample_count", [0, 1])
def test_too_few_samples_zero_output_without_native_pointers(sample_count):
    samples = numpy.ones(sample_count)
    out = torch.ones(16, dtype=torch.complex128)
    with mock.patch.object(
        decompress_torch, "_inline_interp",
        wraps=decompress_torch._inline_interp,
    ) as fallback:
        _linear(samples, samples, samples, out)
        fallback.assert_called_once()
    assert torch.count_nonzero(out) == 0


@pytest.mark.parametrize("missing", ["extension", "version_api"])
def test_missing_native_support_keeps_torch_route(missing, monkeypatch):
    import sys

    args = _inputs()
    out = torch.empty(651, dtype=torch.complex128)
    expected = torch.empty_like(out)
    decompress_torch._inline_interp(*args, expected, 0.125, 0, 0, 1)
    if missing == "extension":
        monkeypatch.setitem(
            sys.modules, "pycbc.waveform.decompress_cpu_cython", None,
        )
    else:
        monkeypatch.delattr(torch.autograd.graph, "increment_version")
    with mock.patch.object(
        decompress_torch, "_inline_interp",
        wraps=decompress_torch._inline_interp,
    ) as fallback:
        _linear(*args, out)
        fallback.assert_called_once()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_autograd_and_inference_keep_torch_semantics():
    amp, phase, frequencies = map(torch.from_numpy, _inputs())
    out = torch.empty(651, dtype=torch.complex128)
    with mock.patch.object(
        decompress_torch, "_inline_interp",
        wraps=decompress_torch._inline_interp,
    ) as fallback:
        amp.requires_grad_()
        _linear(amp, phase, frequencies, out)
        out.real.sum().backward()
        assert amp.grad is not None
        assert torch.all(torch.isfinite(amp.grad))
        fallback.assert_called_once()

    amp = amp.detach()
    leaf = torch.empty(651, dtype=torch.complex128, requires_grad=True)
    with pytest.raises(RuntimeError, match="leaf Variable"):
        _linear(amp, phase, frequencies, leaf)

    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(amp, torch.ones_like(amp))
        with mock.patch.object(
            decompress_torch, "_inline_interp",
            wraps=decompress_torch._inline_interp,
        ) as fallback:
            result = _linear(dual, phase, frequencies, torch.empty_like(out))
            dual_result = torch.autograd.forward_ad.unpack_dual(result)
            assert dual_result.tangent is not None
            fallback.assert_called_once()

    with torch.inference_mode():
        result = torch.empty(651, dtype=torch.complex128)
        with mock.patch.object(
            decompress_torch, "_inline_interp",
            wraps=decompress_torch._inline_interp,
        ) as fallback:
            _linear(amp, phase, frequencies, result)
            fallback.assert_called_once()
            assert torch.is_inference(result)


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_device_output_does_not_enter_cpu_kernel(device):
    available = (torch.cuda.is_available() if device == "cuda"
                 else torch.backends.mps.is_available())
    if not available:
        pytest.skip(f"Torch {device} is unavailable")
    args = _inputs(numpy.float32)
    out = torch.empty(651, dtype=torch.complex64, device=device)
    with mock.patch.object(
        decompress_torch, "_cpu_numpy_view",
        side_effect=AssertionError("device data entered the host ABI"),
    ):
        _linear(*args, out)
    assert out.device.type == device
    assert torch.all(torch.isfinite(out))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_device_inputs_keep_torch_transfer_semantics():
    args = _inputs()
    out = torch.empty(651, dtype=torch.complex128)
    expected = torch.empty_like(out)
    decompress_torch._inline_interp(*args, expected, 0.125, 0, 0, 1)
    args = tuple(torch.as_tensor(value, device="cuda") for value in args)
    with mock.patch.object(
        decompress_torch, "_inline_interp",
        wraps=decompress_torch._inline_interp,
    ) as fallback:
        _linear(*args, out)
        fallback.assert_called_once()
    assert out.device.type == "cpu"
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
