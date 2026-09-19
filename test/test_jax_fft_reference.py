# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Selected FFT validation routes reproduce the original CPU backend exactly."""

import argparse
import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from pycbc import scheme  # noqa: E402
from pycbc.fft import FFT, IFFT, fft, ifft  # noqa: E402
from pycbc.fft import backend_cpu, jaxfft  # noqa: E402
from pycbc.types import Array, FrequencySeries, TimeSeries, zeros  # noqa: E402
from pycbc.types.array_jax import JAXArrayData  # noqa: E402


@pytest.fixture
def device_spec():
    specification = os.environ.get("PYCBC_TEST_SCHEME", "jax:cpu")
    if specification == "jax":
        return "cpu"
    if not specification.startswith("jax:"):
        pytest.skip("FFT reference tests require a JAX PYCBC_TEST_SCHEME")
    return specification.split(':', 1)[1]


def _geometry(operation, family, precision, size, nbatch=1):
    real = np.float32 if precision == "single" else np.float64
    complex_ = np.complex64 if precision == "single" else np.complex128
    half = size // 2 + 1
    if operation == "fft":
        input_dtype = real if family == "real" else complex_
        input_length = size
        output_dtype = complex_
        output_length = half if family == "real" else size
    else:
        input_dtype = complex_
        input_length = half if family == "real" else size
        output_dtype = real if family == "real" else complex_
        output_length = size
    rng = np.random.default_rng(1048)
    values = rng.normal(size=input_length * nbatch)
    if np.issubdtype(input_dtype, np.complexfloating):
        values = values + 1j * rng.normal(size=values.size)
    return values.astype(input_dtype), output_dtype, output_length * nbatch


def _vectors(operation, values, output_dtype, output_length, series=False):
    # Native planning can choose a different algorithm for unaligned buffers.
    # Use the same aligned allocator that ordinary CPU callers use.
    input_ = zeros(values.size, dtype=values.dtype)
    output = zeros(output_length, dtype=output_dtype)
    if series:
        if operation == "fft":
            input_ = TimeSeries(input_, delta_t=.03, epoch=1234567890.125,
                                copy=False)
            output = FrequencySeries(output, delta_f=13., epoch=17.25,
                                     copy=False)
        else:
            input_ = FrequencySeries(input_, delta_f=.125,
                                     epoch=1234567890.125, copy=False)
            output = TimeSeries(output, delta_t=.03, epoch=17.25, copy=False)
    input_.data[:] = values
    return input_, output


def _execute(operation, input_, output, api, nbatch=1, size=None):
    if api == "functional":
        {"fft": fft, "ifft": ifft}[operation](input_, output)
        return None
    plan = {"fft": FFT, "ifft": IFFT}[operation](
        input_, output, nbatch=nbatch, size=size)
    plan.execute()
    return plan


def _assert_bytes(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def _reject_jax_calculation(*args, **kwargs):
    raise AssertionError("selected CPU reference operation executed a JAX FFT")


@pytest.mark.parametrize("operation", ["fft", "ifft"])
@pytest.mark.parametrize("family", ["real", "complex"])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("size", [16, 21])
def test_functional_reference_fft_preserves_exact_bytes_and_output_alias(
        monkeypatch, device_spec, operation, family, precision, size):
    values, output_dtype, output_length = _geometry(
        operation, family, precision, size)
    with scheme.CPUScheme():
        input_, output = _vectors(operation, values, output_dtype, output_length)
        _execute(operation, input_, output, "functional")
        expected = output.numpy().copy()

    names = ("jax_fft", "jax_rfft") if operation == "fft" else (
        "jax_ifft", "jax_irfft")
    for name in names:
        monkeypatch.setattr(jaxfft, name, _reject_jax_calculation)
    with scheme.JAXScheme(device_spec, reference_operations=(operation,)) as ctx:
        input_ = zeros(values.size, dtype=values.dtype)
        input_.data[:] = values
        parent = Array(np.full(output_length + 6, -17, dtype=output_dtype))
        output = parent[3:-3]
        alias = parent[3:-3]
        storage = output._data
        _execute(operation, input_, output, "functional")
        assert scheme.mgr.state is ctx
        assert output._data is storage
        assert isinstance(storage, JAXArrayData)
        assert storage.device == ctx.jax_device
        _assert_bytes(output.numpy(), expected)
        _assert_bytes(alias.numpy(), expected)
        np.testing.assert_array_equal(parent.numpy()[:3], -17)
        np.testing.assert_array_equal(parent.numpy()[-3:], -17)


@pytest.mark.parametrize("operation", ["fft", "ifft"])
@pytest.mark.parametrize("family", ["real", "complex"])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("size", [16, 21])
@pytest.mark.parametrize("nbatch", [1, 3])
def test_class_reference_fft_matches_original_cpu_plan(
        monkeypatch, device_spec, operation, family, precision, size, nbatch):
    values, output_dtype, output_length = _geometry(
        operation, family, precision, size, nbatch)
    with scheme.CPUScheme():
        input_, output = _vectors(operation, values, output_dtype, output_length)
        _execute(operation, input_, output, "class", nbatch, size)
        expected = output.numpy().copy()

    compiler = "_compile_jax_fwd" if operation == "fft" else "_compile_jax_inv"
    monkeypatch.setattr(jaxfft, compiler, _reject_jax_calculation)
    with scheme.JAXScheme(device_spec, reference_operations=(operation,)) as ctx:
        input_, output = _vectors(operation, values, output_dtype, output_length)
        storage = output._data
        plan = _execute(operation, input_, output, "class", nbatch, size)
        assert plan.invec is input_ and plan.outvec is output
        assert scheme.mgr.state is ctx
        assert output._data is storage
        assert storage.device == ctx.jax_device
        _assert_bytes(output.numpy(), expected)


@pytest.mark.parametrize("operation", ["fft", "ifft"])
@pytest.mark.parametrize("family", ["real", "complex"])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("api", ["functional", "class"])
def test_reference_fft_keeps_original_series_metadata_and_normalization(
        device_spec, operation, family, precision, api):
    values, output_dtype, output_length = _geometry(
        operation, family, precision, 21)
    with scheme.CPUScheme():
        input_, output = _vectors(
            operation, values, output_dtype, output_length, series=True)
        original_plan = _execute(operation, input_, output, api)
        expected = output.numpy().copy()
        expected_epoch = (output.epoch if operation == "fft"
                          else output.start_time)
        expected_spacing = (output.delta_f if operation == "fft"
                            else output.delta_t)

    with scheme.JAXScheme(device_spec, reference_operations=(operation,)) as ctx:
        input_, output = _vectors(
            operation, values, output_dtype, output_length, series=True)
        plan = _execute(operation, input_, output, api)
        epoch = output.epoch if operation == "fft" else output.start_time
        assert epoch == expected_epoch
        spacing = output.delta_f if operation == "fft" else output.delta_t
        assert spacing == expected_spacing
        if api == "class":
            assert plan.scale == original_plan.scale
        assert output._data.device == ctx.jax_device
        _assert_bytes(output.numpy(), expected)


@pytest.mark.parametrize("operation", ["fft", "ifft"])
def test_reference_fft_plan_reuses_buffers_after_input_changes(device_spec, operation):
    values, output_dtype, output_length = _geometry(
        operation, "complex", "double", 16)
    with scheme.CPUScheme():
        input_, output = _vectors(operation, values, output_dtype, output_length)
        cpu_plan = _execute(operation, input_, output, "class")
        first = output.numpy().copy()
        input_.data[:] = values + 2
        cpu_plan.execute()
        second = output.numpy().copy()
    with scheme.JAXScheme(device_spec, reference_operations=(operation,)):
        input_, output = _vectors(operation, values, output_dtype, output_length)
        plan = _execute(operation, input_, output, "class")
        _assert_bytes(output.numpy(), first)
        input_.data[:] = values + 2
        plan.execute()
        _assert_bytes(output.numpy(), second)


@pytest.mark.parametrize("operation", ["fft", "ifft"])
def test_fft_reference_switches_are_independent(monkeypatch, device_spec, operation):
    calls = []
    for name in ("fft", "ifft"):
        original = getattr(jaxfft, "jax_" + name)

        def record(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(jaxfft, "jax_" + name, record)
    with scheme.JAXScheme(device_spec, reference_operations=(operation,)):
        input_, output = _vectors("fft", *_geometry("fft", "complex", "double", 16))
        fft(input_, output)
        ifft(output, input_)
    assert calls == ["ifft" if operation == "fft" else "fft"]


def test_default_fft_path_does_not_select_cpu_backend(monkeypatch, device_spec):
    def reject_cpu_backend():
        raise AssertionError("default JAX FFT selected the CPU backend")

    monkeypatch.setattr(backend_cpu, "get_backend", reject_cpu_backend)
    with scheme.JAXScheme(device_spec):
        input_, output = _vectors("fft", *_geometry("fft", "complex", "double", 16))
        fft(input_, output)
        ifft(output, input_)
        FFT(input_, output).execute()
        IFFT(output, input_).execute()


def test_fft_reference_operation_names_parse_from_cli():
    parser = argparse.ArgumentParser()
    scheme.insert_processing_option_group(parser)
    options = parser.parse_args([
        "--processing-scheme", "jax:cpu", "--jax-reference-operations", "fft,ifft",
    ])
    assert scheme.from_cli(options).jax_reference_operations == frozenset({"fft", "ifft"})
