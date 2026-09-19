"""Individual validation switches run the original PyCBC array kernels."""

import os
import numpy as np
import pytest

jax = pytest.importorskip("jax")

from pycbc import scheme  # noqa: E402
from pycbc.types import Array  # noqa: E402
from pycbc.types import array_jax  # noqa: E402


OPERATIONS = ("sum", "cumsum", "dot", "inner", "weighted_inner",
              "multiply_and_add", "abs_max_loc", "abs_arg_max",
              "squared_norm", "min", "max", "max_loc")
DEVICE = os.environ.get("PYCBC_TEST_SCHEME", "jax:cpu").removeprefix("jax:")


def _inputs(dtype):
    values = np.asarray([1.234567, 1e4, -1e4, 1e-4, -1e-4,
                         3.14, -2.718281, .7], dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        values += np.asarray([.13j, -2j, 5j, .03j, -7j, .6j, 2j, -.4j],
                             dtype=dtype)
    return values, values[::-1].copy(), np.full(len(values), 1.234567, dtype)


def _calculate(operation, values, other, weight):
    if operation in ("dot", "inner"):
        return getattr(values, operation)(other)
    if operation == "weighted_inner":
        return values.weighted_inner(other, weight)
    if operation == "multiply_and_add":
        return values.multiply_and_add(other, .1234567)
    return getattr(values, operation)()


def _snapshot(value):
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, Array):
        value = value.numpy()
    return np.array(value, copy=True)


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64,
                                   np.complex64, np.complex128])
def test_reference_operation_runs_original_kernel_exactly(operation, dtype,
                                                          monkeypatch):
    if operation in ("min", "max", "max_loc") and np.issubdtype(
            dtype, np.complexfloating):
        pytest.skip("PyCBC extrema reject complex arrays")
    inputs = _inputs(dtype)
    with scheme.CPUScheme():
        expected = _snapshot(_calculate(operation, *(Array(x) for x in inputs)))

    original = getattr(array_jax._array_cpu, operation)
    calls = []

    def recorded(*args):
        calls.append(operation)
        return original(*args)

    monkeypatch.setattr(array_jax._array_cpu, operation, recorded)
    with scheme.JAXScheme(DEVICE, reference_operations=(operation,)) as context:
        actual = _snapshot(_calculate(operation, *(Array(x) for x in inputs)))
        assert scheme.mgr.state is context
    np.testing.assert_array_equal(actual, expected)
    assert calls == [operation]


def test_reference_switch_is_independent_and_defaults_to_jax(monkeypatch):
    raw = np.asarray([1.234567], np.float32)
    with scheme.CPUScheme():
        values = Array(raw)
        expected = values.inner(values)
    with scheme.JAXScheme(DEVICE):
        values = Array(raw)
        default = values.inner(values)
    assert default != expected

    original = array_jax._array_cpu.inner
    calls = []

    def recorded(*args):
        calls.append("inner")
        return original(*args)

    def reject_unselected(*args):
        raise AssertionError("unselected CPU kernel was called")

    monkeypatch.setattr(array_jax._array_cpu, "inner", recorded)
    monkeypatch.setattr(array_jax._array_cpu, "weighted_inner", reject_unselected)
    with scheme.JAXScheme(DEVICE, reference_operations=("inner",)):
        values = Array(raw)
        assert values.inner(values) == expected
        values.weighted_inner(values, Array(np.ones(1, np.float32)))
    with scheme.JAXScheme(DEVICE):
        values = Array(raw)
        assert values.inner(values) == default
    assert calls == ["inner"]


def test_reference_multiply_and_add_preserves_slice_and_copy():
    raw = np.arange(8, dtype=np.float32)
    rhs = np.asarray([1.234567, 2.718281, 3.14159], np.float32)
    with scheme.CPUScheme():
        expected = Array(raw)
        expected[1:4].multiply_and_add(Array(rhs), .1234567)
        expected = expected.numpy().copy()
    with scheme.JAXScheme(DEVICE, reference_operations=("multiply_and_add",)):
        parent = Array(raw)
        copied = parent.copy()
        view = parent[1:4]
        result = view.multiply_and_add(Array(rhs), .1234567)
        np.testing.assert_array_equal(parent.numpy(), expected)
        np.testing.assert_array_equal(copied.numpy(), raw)
        np.testing.assert_array_equal(result.numpy(), view.numpy())
        view[:] = 9
        np.testing.assert_array_equal(result.numpy(), [9, 9, 9])


@pytest.mark.parametrize("raw", [[complex(np.nan, 0), 0j],
                                  [complex(np.nan, 0), 1j],
                                  [0j, complex(np.nan, 0)], [0j, 0j]])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_abs_arg_max_reference_preserves_compiled_behavior(raw, dtype):
    raw = np.asarray(raw, dtype)
    with scheme.CPUScheme():
        expected = Array(raw).abs_arg_max()
    with scheme.JAXScheme(DEVICE, reference_operations=("abs_arg_max",)):
        assert Array(raw).abs_arg_max() == expected


def test_complex_abs_arg_max_default_preserves_initial_zero_maximum():
    with scheme.JAXScheme(DEVICE):
        values = Array(np.asarray([complex(np.nan, 0), 0j], np.complex64))
        assert values.abs_arg_max() == 0


def test_reference_operation_requires_original_cpu_kernel(monkeypatch):
    monkeypatch.setattr(array_jax, "_array_cpu", None)
    with scheme.JAXScheme(DEVICE, reference_operations=("sum",)):
        with pytest.raises(RuntimeError, match="CPU array kernels are unavailable"):
            Array([1.0]).sum()
    with scheme.JAXScheme(DEVICE):
        assert Array([1.0]).sum() == 1.0
