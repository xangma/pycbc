import numpy as np
import pytest
torch = pytest.importorskip("torch")

from pycbc import scheme  # noqa: E402
from pycbc.types import Array  # noqa: E402


def _torch_array_result(values, operation):
    """Evaluate one operation on Torch and return detached NumPy storage."""
    ctx = scheme.TorchScheme("cpu")
    try:
        with ctx:
            result = operation(Array(values))
            return result.numpy()
    finally:
        del ctx
        scheme.Scheme._single = None


@pytest.mark.parametrize(
    ("dtype", "values"),
    (
        (np.int32, (2_000_000_000, -2_000_000_000)),
        (np.int64, (5_000_000_000_000_000_000, -5_000_000_000_000_000_000)),
    ),
)
@pytest.mark.parametrize(
    "operation",
    (
        lambda array: array + 2,
        lambda array: 2 + array,
        lambda array: array - 2,
        lambda array: 2 - array,
        lambda array: array * 2,
        lambda array: 2 * array,
    ),
)
def test_torch_integer_scalar_arithmetic_retains_legacy_float64(
    dtype, values, operation
):
    """Do not overflow by changing PyCBC's scalar coercion semantics."""
    values = np.asarray(values, dtype=dtype)
    expected = operation(Array(values)).numpy()
    actual = _torch_array_result(values, operation)

    assert expected.dtype == np.dtype(np.float64)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


_TORCH_INT_DTYPES = (np.int32, np.int64, np.bool_) + (
    (np.uint32,) if getattr(torch, "uint32", None) is not None else ()
)


@pytest.mark.parametrize("dtype", _TORCH_INT_DTYPES)
@pytest.mark.parametrize(
    "operation",
    (
        lambda array: array / 2,
        lambda array: 2 / array,
    ),
)
def test_torch_integer_true_division_retains_pycbc_float64(dtype, operation):
    """Do not inherit Torch's float32 integer true-division default."""
    values = np.asarray((1, 2, 3), dtype=dtype)
    expected = operation(Array(values)).numpy()
    actual = _torch_array_result(values, operation)

    assert expected.dtype == np.dtype(np.float64)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", _TORCH_INT_DTYPES)
@pytest.mark.parametrize("scalar", (2, True))
@pytest.mark.parametrize(
    "ufunc",
    (
        np.bitwise_and,
        np.bitwise_or,
        np.bitwise_xor,
        np.equal,
        np.less,
        np.greater_equal,
    ),
)
@pytest.mark.parametrize("reverse", (False, True))
def test_torch_integer_scalar_ufuncs_match_numpy_pycbc(
    dtype, scalar, ufunc, reverse
):
    """Cover scalar bitwise/comparison paths in both operand orders."""
    values = np.asarray((1, 2, 3), dtype=dtype)

    def operation(array):
        operands = (scalar, array) if reverse else (array, scalar)
        return ufunc(*operands)

    expected = operation(Array(values)).numpy()
    actual = _torch_array_result(values, operation)

    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    if ufunc in (np.equal, np.less, np.greater_equal):
        assert actual.dtype == np.dtype(np.bool_)
    elif np.dtype(dtype).kind == "b" and scalar is not True:
        assert actual.dtype == np.dtype(np.int64)
    else:
        assert actual.dtype == np.dtype(dtype)


@pytest.mark.parametrize("dtype", (np.int32, np.int64))
def test_torch_integer_inplace_python_scalar_matches_numpy(dtype):
    expected = np.array([2, 5, 9], dtype=dtype)
    expected += 3
    expected -= 1
    expected *= 2

    ctx = scheme.TorchScheme("cpu")
    try:
        with ctx:
            actual = Array(np.array([2, 5, 9], dtype=dtype))
            actual += 3
            actual -= 1
            actual *= 2

            assert actual._data.tensor.dtype in (torch.int32, torch.int64)
            np.testing.assert_array_equal(actual.numpy(), expected)

            with pytest.raises(RuntimeError):
                actual += 0.5
    finally:
        del ctx
        scheme.Scheme._single = None


@pytest.mark.parametrize("device", ("cpu", "cuda", "mps"))
def test_torch_reductions_use_supported_device_accumulators(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device unavailable")

    values = np.array([1 + 2j, -3 + 4j, 5 - 6j], dtype=np.complex64)
    other = np.array([2 - 1j, 4 + 3j, -2 + 5j], dtype=np.complex64)
    weights = np.array([1.0, 2.0, 4.0], dtype=np.float32)

    ctx = scheme.TorchScheme(device)
    try:
        with ctx:
            actual_values = Array(values)
            actual_other = Array(other)
            actual_weights = Array(weights)

            assert actual_values._data.tensor.device.type == device
            assert actual_values.sum() == pytest.approx(
                values.sum(dtype=np.complex64), rel=2e-6, abs=2e-6
            )
            assert actual_values.inner(actual_other) == pytest.approx(
                np.vdot(values, other), rel=2e-6, abs=2e-6
            )
            assert actual_values.weighted_inner(
                actual_other, actual_weights
            ) == pytest.approx(
                np.sum(np.conj(values) * other / weights, dtype=np.complex64),
                rel=2e-6,
                abs=2e-6,
            )
            assert actual_values.abs_arg_max() == int(
                np.argmax(np.abs(values))
            )
            assert actual_values.inner(actual_values) == pytest.approx(
                np.vdot(values, values).real, rel=2e-6, abs=2e-6
            )
            assert actual_values.weighted_inner(
                actual_values, actual_weights
            ) == pytest.approx(
                np.sum(np.abs(values) ** 2 / weights),
                rel=2e-6,
                abs=2e-6,
            )
    finally:
        del ctx
        scheme.Scheme._single = None


def test_torch_self_inner_and_abs_arg_max():
    """Verify abs_arg_max, inner and weighted_inner self-optimizations."""
    ctx = scheme.TorchScheme("cpu")
    try:
        with ctx:
            # Complex array
            c_vals = np.array(
                [3 + 4j, 1 + 2j, 6 + 8j, 2 - 1j], dtype=np.complex64
            )
            c_arr = Array(c_vals)
            w_vals = np.array([1.0, 2.0, 4.0, 0.5], dtype=np.float32)
            w_arr = Array(w_vals)

            assert c_arr.abs_arg_max() == 2
            expected_inner = np.sum(np.abs(c_vals) ** 2)
            assert c_arr.inner(c_arr) == pytest.approx(
                expected_inner, rel=1e-6
            )

            expected_weighted = np.sum(np.abs(c_vals) ** 2 / w_vals)
            assert c_arr.weighted_inner(c_arr, w_arr) == pytest.approx(
                expected_weighted, rel=1e-6
            )

            # Real array
            r_vals = np.array([-2.5, 7.1, -9.3, 4.0], dtype=np.float32)
            r_arr = Array(r_vals)
            assert r_arr.abs_arg_max() == 2
            assert r_arr.inner(r_arr) == pytest.approx(
                np.sum(r_vals ** 2), rel=1e-6
            )
            assert r_arr.weighted_inner(r_arr, w_arr) == pytest.approx(
                np.sum(r_vals ** 2 / w_vals), rel=1e-6
            )
    finally:
        del ctx
        scheme.Scheme._single = None
