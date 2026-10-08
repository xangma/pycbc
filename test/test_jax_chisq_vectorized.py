"""Independent-lane geometry checks for ordered JAX chi-square kernels.

CPU Pallas interpretation checks indexing and frequency order. CUDA tests
separately require exact equality with the established scalar programs;
CPU interpretation has different scalar/vector floating-point contractions.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from pycbc.vetoes import chisq_jax


@pytest.fixture(autouse=True)
def _enable_x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _assert_same_bits(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    dtype = np.uint32 if actual.dtype.itemsize == 4 else np.uint64
    np.testing.assert_array_equal(actual.view(dtype), expected.view(dtype))


@pytest.mark.parametrize("shape", ((5, 19), (3, 65)))
@pytest.mark.parametrize("unroll", (1, 16))
def test_interpreted_scan_keeps_each_frequency_addition(
        shape, unroll):
    values = np.random.default_rng(73).normal(size=shape).astype(np.float32)
    # A reassociated/tree sum changes this row's third cumulative value.
    values[0, :5] = [2**24, 1, -2**24, 0, 0]
    values[-1, :2] = [0.0, 0.0]
    got = chisq_jax._ordered_cumsum_rows_pallas(
        jnp.asarray(values), unroll=unroll, interpret=True)
    _assert_same_bits(got, np.cumsum(values, axis=1))


@pytest.mark.parametrize("shape", ((0, 7), (3, 0), (0, 0)))
def test_interpreted_scan_empty_axes(shape):
    values = jnp.empty(shape, dtype=jnp.float32)
    got = chisq_jax._ordered_cumsum_rows_pallas(
        values, unroll=16, interpret=True)
    _assert_same_bits(got, np.zeros(shape, dtype=np.float32))


@pytest.mark.parametrize("width", (1, 3, 16))
@pytest.mark.parametrize("unroll", (1, 16))
def test_scalar_scan_unroll_handles_short_and_exact_groups(width, unroll):
    values = np.arange(3 * width, dtype=np.float32).reshape(3, width)
    got = chisq_jax._ordered_cumsum_rows_pallas(
        jnp.asarray(values), unroll=unroll, interpret=True)
    _assert_same_bits(got, np.cumsum(values, axis=1))


def _recurrence_case(count=5, width=47):
    rng = np.random.default_rng(53)
    values = (rng.normal(size=(3, width))
              + 1j * rng.normal(size=(3, width))).astype(np.complex64)
    base, n_time = 7, 2097152
    bins = np.asarray([
        [0, 0, 11, base + width - 3, base + width + 13],
        [0, 9, 9, base + width, base + width + 13],
        [0, 1, 13, base + width - 5, base + width + 13],
    ], dtype=np.int32)
    rows = np.resize(np.asarray([0, 1, 0, 2, 1], dtype=np.int32), count)
    points = np.resize(np.asarray(
        [0, 3, n_time - 1, 2**24 + 1, -7], dtype=np.int64), count)
    return values, rows, points, bins, n_time, base


@pytest.mark.parametrize("lanes,unroll", ((1, 1), (1, 8),
                                          (4, 1), (8, 1)))
def test_interpreted_recurrence_geometry_preserves_native_oracle(
        lanes, unroll):
    values, rows, points, bins, n_time, base = _recurrence_case()
    expected = np.asarray([
        chisq_jax._point_chisq_cpu_compatible(
            values[row], points[i:i + 1], bins[row], n_time, base)[0]
        for i, row in enumerate(rows)
    ])
    got = chisq_jax._compatible_shift_sum_pallas(
        *(jnp.asarray(value) for value in (values, rows, points, bins)),
        n_time, base, lanes=lanes, unroll=unroll, interpret=True)
    assert got.dtype == jnp.float32
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("lanes,unroll", ((1, 8), (4, 1), (8, 1)))
def test_interpreted_recurrence_order_is_exact_at_zero_shift(lanes, unroll):
    # No rotating phase: exact values make reordered frequency/bin sums visible
    # without scalar/vector interpreter contraction differences.
    values = np.asarray([
        [2**24, 1, -2**24, 3, -3, 2, -2, 1, 1],
        [1, 2**24, -2**24, -1, 4, -4, 0, 3, -3],
    ], dtype=np.complex64)
    args = (jnp.asarray(values), jnp.asarray([0, 1, 0, 1, 1]),
            jnp.zeros(5, dtype=jnp.int32),
            jnp.asarray([[0, 0, 3, 9], [0, 2, 2, 9]]), 64, 0)
    baseline = chisq_jax._compatible_shift_sum_pallas(
        *args, lanes=1, interpret=True)
    got = chisq_jax._compatible_shift_sum_pallas(
        *args, lanes=lanes, unroll=unroll, interpret=True)
    _assert_same_bits(got, baseline)


@pytest.mark.parametrize("width,count,edges", (
    (0, 3, [0, 0, 0]), (16, 0, [0, 8, 16]),
    (16, 3, [0]), (16, 3, [7, 7, 7]), (16, 3, [7, 4, 4]),
))
def test_interpreted_recurrence_empty_and_reversed_bins(width, count, edges):
    got = chisq_jax._compatible_shift_sum_pallas(
        jnp.ones((1, width), dtype=jnp.complex64),
        jnp.zeros(count, dtype=jnp.int32), jnp.arange(count),
        jnp.asarray([edges]), 64, 0, lanes=8, interpret=True)
    _assert_same_bits(got, np.zeros(count, dtype=np.float32))


def _gpu():
    try:
        device = jax.devices("gpu")[0]
    except RuntimeError:
        pytest.skip("JAX GPU backend unavailable")
    if not chisq_jax._use_gpu_ordered_scan(jax.device_put(
            np.zeros((1, 1), dtype=np.float32), device)):
        pytest.skip("Pallas Triton ordered kernels unavailable")
    return device


@pytest.mark.parametrize("unroll", (1, 16))
def test_cuda_scan_is_exact_to_scalar_program(
        unroll):
    device = _gpu()
    values = np.random.default_rng(31).random((19, 16385), dtype=np.float32)
    values[0, :5] = 0.0
    array = jax.device_put(values, device)
    baseline = chisq_jax._ordered_cumsum_rows_pallas(
        array, unroll=1)
    got = chisq_jax._ordered_cumsum_rows_pallas(
        array, unroll=unroll)
    _assert_same_bits(got, baseline)
    _assert_same_bits(got, np.cumsum(values, axis=1))
    assert got.devices() == {device}


@pytest.mark.parametrize("width", (173, 56326))
@pytest.mark.parametrize("lanes,unroll", ((1, 8), (4, 1), (8, 1)))
def test_cuda_recurrence_is_exact_to_scalar_program(
        width, lanes, unroll):
    device = _gpu()
    values, rows, points, bins, n_time, base = _recurrence_case(17, width)
    args = tuple(jax.device_put(value, device)
                 for value in (values, rows, points, bins)) + (n_time, base)
    baseline = chisq_jax._compatible_shift_sum_pallas(*args, lanes=1)
    got = chisq_jax._compatible_shift_sum_pallas(
        *args, lanes=lanes, unroll=unroll)
    _assert_same_bits(got, baseline)
    assert got.devices() == {device}


@pytest.mark.parametrize("count", (4, 64, 127, 128, 129, 511, 512, 513))
def test_cuda_production_dispatch_keeps_scalar_rounding(count):
    device = _gpu()
    values, rows, points, bins, n_time, base = _recurrence_case(count, 173)
    args = tuple(jax.device_put(value, device)
                 for value in (values, rows, points, bins)) + (n_time, base)
    baseline = chisq_jax._compatible_shift_sum_pallas(*args)
    got = chisq_jax._compatible_shift_sum(*args)
    _assert_same_bits(got, baseline)
    assert got.devices() == {device}


def test_private_scan_geometry_validation():
    with pytest.raises(ValueError, match="ordered kernel"):
        chisq_jax._ordered_cumsum_rows_pallas(
            jnp.ones((1, 3)), interpret=True, unroll=3)


@pytest.mark.parametrize("options", ({"lanes": 3}, {"unroll": 3}))
def test_private_recurrence_geometry_validation(options):
    with pytest.raises(ValueError, match="ordered kernel"):
        chisq_jax._compatible_shift_sum_pallas(
            jnp.ones((1, 3), dtype=jnp.complex64),
            jnp.zeros(1, dtype=jnp.int32),
            jnp.zeros(1), jnp.asarray([[0, 3]]), 8, 0,
            interpret=True, **options)
