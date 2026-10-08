# Copyright (C) 2026 PyCBC contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Original division routes preserve CPU arithmetic and mutable JAX views."""

import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from pycbc import scheme  # noqa: E402
from pycbc.types import Array, FrequencySeries  # noqa: E402
from pycbc.types.array_jax import JAXArrayData, _divide  # noqa: E402


@pytest.fixture
def device_spec():
    specification = os.environ.get("PYCBC_TEST_SCHEME", "jax:cpu")
    if specification == "jax":
        return "cpu"
    if not specification.startswith("jax:"):
        pytest.skip("Division reference tests require a JAX scheme")
    return specification.split(":", 1)[1]


def _values(dtype):
    values = np.asarray([1.234567, -2.71828, .000137, 3.14159], dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        values += np.asarray([.71j, -2.35j, 3.54j, -.11j], dtype=dtype)
    denominator = np.asarray([.1317, 1.234567, 4.1931, .97853],
                             dtype=values.real.dtype)
    return values, denominator


@pytest.mark.parametrize("dtype", [np.float32, np.float64,
                                   np.complex64, np.complex128])
@pytest.mark.parametrize("operation", ["array", "scalar", "reverse", "inplace"])
def test_division_reference_matches_original(device_spec, dtype, operation):
    values, denominator = _values(dtype)

    def calculate(left, right):
        if operation == "array":
            return left / right
        if operation == "scalar":
            return left / 1.234567
        if operation == "reverse":
            return 1.234567 / left
        left /= right
        return left

    with scheme.CPUScheme():
        expected = calculate(Array(values), Array(denominator)).numpy().copy()
    with scheme.JAXScheme(device_spec, reference_operations=("divide",)) as ctx:
        left, right = Array(values), Array(denominator)
        result = calculate(left, right)
        assert result.numpy().dtype == expected.dtype
        assert result.numpy().tobytes() == expected.tobytes()
        assert result.data.array.devices() == {ctx.jax_device}
        assert scheme.mgr.state is ctx
        np.testing.assert_array_equal(right.numpy(), denominator)
        if operation == "inplace":
            assert result is left
        else:
            assert result.data is not left.data
            np.testing.assert_array_equal(left.numpy(), values)


def test_division_reference_preserves_nested_aliases_and_copy(device_spec):
    values = (np.arange(12) + .1234567j).astype(np.complex64)
    with scheme.CPUScheme():
        expected = Array(values)
        view = expected[1:11][::2]
        view /= expected[2:12:2]
        expected = expected.numpy().copy()
    with scheme.JAXScheme(device_spec, reference_operations=("divide",)):
        parent = Array(values)
        independent = parent.copy()
        sibling = parent[1:11]
        view = sibling[::2]
        original_wrapper = view.data
        view /= parent[2:12:2]
        assert view.data is original_wrapper
        assert parent.numpy().tobytes() == expected.tobytes()
        assert sibling.numpy().tobytes() == expected[1:11].tobytes()
        np.testing.assert_array_equal(independent.numpy(), values)


def test_division_reference_keeps_frequency_metadata(device_spec):
    values, denominator = _values(np.complex64)
    with scheme.CPUScheme():
        expected = (FrequencySeries(values, delta_f=.125, epoch=123.25)
                    / FrequencySeries(denominator, delta_f=.125)).numpy().copy()
    with scheme.JAXScheme(device_spec, reference_operations=("divide",)):
        source = FrequencySeries(values, delta_f=.125, epoch=123.25)
        result = source / FrequencySeries(denominator, delta_f=.125)
        assert isinstance(result, FrequencySeries)
        assert result.numpy().tobytes() == expected.tobytes()
        assert result.delta_f == source.delta_f
        assert result.epoch == source.epoch


def test_default_division_stays_on_device(device_spec, monkeypatch):
    numpy_array, numpy_asarray = np.array, np.asarray

    def reject_download(function):
        def checked(value, *args, **kwargs):
            if isinstance(value, (jax.Array, JAXArrayData)):
                raise AssertionError("default division downloaded samples")
            return function(value, *args, **kwargs)
        return checked

    with scheme.JAXScheme(device_spec):
        left = Array(np.array([1.234567 + 2.345678j], np.complex64))
        right = Array(np.array([.1317], np.float32))
        expected = left.data.array / right.data.array
        with monkeypatch.context() as guard:
            guard.setattr(np, "array", reject_download(numpy_array))
            guard.setattr(np, "asarray", reject_download(numpy_asarray))
            assert isinstance((left / right).data.array, jax.Array)
            assert isinstance((1.234567 / left).data.array, jax.Array)
            left /= right
        np.testing.assert_array_equal(left.numpy(), np.asarray(expected))


def test_raw_division_reference_and_jit_guard(device_spec):
    values, denominator = _values(np.complex64)
    expected = values / denominator
    with scheme.JAXScheme(device_spec, reference_operations=("divide",)) as ctx:
        left = JAXArrayData(jax.device_put(values, ctx.jax_device))
        right = JAXArrayData(jax.device_put(denominator, ctx.jax_device))
        actual = _divide(left, right)
        assert np.asarray(actual).tobytes() == expected.tobytes()
        assert actual.devices() == {ctx.jax_device}
        with pytest.raises(RuntimeError, match="cannot run inside jax.jit"):
            jax.jit(_divide)(left.array, right.array)
