import warnings

import numpy as np
import pytest

import pycbc
from pycbc import scheme
from pycbc.types import Array, FrequencySeries, TimeSeries


def test_array_numpy_protocol_accepts_copy_kwarg():
    ary = Array([1.0, 2.0, 3.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = np.array(ary, copy=False)
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep == []
    np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0]))


@pytest.mark.parametrize("operation", (np.isfinite, np.isnan, np.isinf))
@pytest.mark.parametrize(
    "series",
    (
        TimeSeries([0.0, np.nan, np.inf], delta_t=0.25, epoch=10.0),
        FrequencySeries(
            [0.0j, complex(np.nan, 1.0), complex(1.0, np.inf)],
            delta_f=0.5,
            epoch=20.0,
        ),
    ),
)
def test_numpy_predicates_preserve_series_metadata(operation, series):
    expected = operation(series.numpy())
    actual = operation(series)

    assert type(actual) is type(series)
    assert actual.dtype == np.dtype(np.bool_)
    if isinstance(series, TimeSeries):
        assert actual.start_time == series.start_time
        assert actual.delta_t == series.delta_t
    else:
        assert actual.epoch == series.epoch
        assert actual.delta_f == series.delta_f
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize(
    "operation", (np.logical_not, np.logical_and, np.logical_or, np.logical_xor)
)
def test_numpy_logical_ufuncs_preserve_series_metadata(operation):
    left = TimeSeries([0.0, 1.0, np.nan], delta_t=0.25, epoch=10.0)
    right = TimeSeries([1.0, 0.0, 2.0], delta_t=0.25, epoch=10.0)
    inputs = (left,) if operation is np.logical_not else (left, right)
    expected = operation(*(value.numpy() for value in inputs))
    actual = operation(*inputs)

    assert isinstance(actual, TimeSeries)
    assert actual.dtype == np.dtype(np.bool_)
    assert actual.start_time == left.start_time
    assert actual.delta_t == left.delta_t
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize(
    "operation",
    (
        np.equal,
        np.not_equal,
        np.less,
        np.less_equal,
        np.greater,
        np.greater_equal,
    ),
)
def test_numpy_comparison_ufuncs_preserve_series_metadata(operation):
    left = TimeSeries([-1.0, 0.5, 2.0], delta_t=0.25, epoch=10.0)
    right = TimeSeries([0.0, 0.5, 1.0], delta_t=0.25, epoch=10.0)
    expected = operation(left.numpy(), right.numpy())
    actual = operation(left, right)

    assert isinstance(actual, TimeSeries)
    assert actual.dtype == np.dtype(np.bool_)
    assert actual.start_time == left.start_time
    assert actual.delta_t == left.delta_t
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize(
    "dtype", (np.bool_, np.int32, np.int64, np.uint32)
)
def test_bitwise_operators_preserve_series_metadata(dtype):
    values = np.array([0, 1, 2, 3], dtype=dtype)
    others = np.array([1, 0, 3, 2], dtype=dtype)
    left = TimeSeries(values, delta_t=0.25, epoch=10.0)
    right = TimeSeries(others, delta_t=0.25, epoch=10.0)
    cases = (
        (~left, np.invert(values)),
        (left & right, np.bitwise_and(values, others)),
        (left | right, np.bitwise_or(values, others)),
        (left ^ right, np.bitwise_xor(values, others)),
        (1 | left, np.bitwise_or(1, values)),
        (np.bitwise_and(left, right), np.bitwise_and(values, others)),
    )

    for actual, expected in cases:
        assert isinstance(actual, TimeSeries)
        assert actual.dtype == expected.dtype
        assert actual.start_time == left.start_time
        assert actual.delta_t == left.delta_t
        np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("dtype", (np.float32, np.complex64))
def test_bitwise_operators_reject_nonintegral_arrays(dtype):
    array = Array(np.array([0, 1], dtype=dtype))

    with pytest.raises(TypeError):
        ~array
    with pytest.raises(TypeError):
        array & 1


@pytest.mark.parametrize("method", ("any", "all"))
@pytest.mark.parametrize("axis", (None, 0, -1))
@pytest.mark.parametrize(
    "values",
    (
        np.array([True, False], dtype=np.bool_),
        np.array([0, 2], dtype=np.int32),
        np.array([0.0, np.nan], dtype=np.float32),
        np.array([], dtype=np.bool_),
    ),
)
def test_array_boolean_reduction_methods_match_numpy(method, axis, values):
    array = Array(values)
    expected = getattr(values, method)(axis=axis)

    actual = getattr(array, method)(axis=axis)

    assert isinstance(actual, np.bool_)
    assert actual == expected


@pytest.mark.skipif(not pycbc.HAVE_TORCH, reason="torch support unavailable")
def test_torch_array_data_numpy_protocol_accepts_copy_kwarg():
    old_scheme = scheme.mgr.state
    old_single = scheme.Scheme._single
    try:
        scheme.Scheme._single = None
        scheme.mgr.state = scheme.TorchScheme()
        ary = Array([1.0, 2.0, 3.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = np.array(ary._data, copy=False)
    finally:
        scheme.mgr.state = old_scheme
        scheme.Scheme._single = old_single
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep == []
    np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0]))
