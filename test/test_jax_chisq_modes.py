"""Mode dispatch and pure-JAX recurrence coverage for JAX chi-square."""

import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from pycbc import scheme
from pycbc.vetoes import chisq_jax
from pycbc.vetoes.chisq_jax_compat import shift_sum as compat_shift_sum


@pytest.fixture(autouse=True)
def _enable_x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _row(dtype, n=173, seed=4):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(dtype)


@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
def test_default_cpu_mode_matches_compiled_reference(dtype):
    row = _row(dtype)
    bins = np.array([0, 0, 29, len(row)], dtype=np.uint32)
    points = np.array([0, 3, 127, 4097], dtype=np.int32)
    with scheme.JAXScheme(device="cpu"):
        assert scheme.mgr.state.jax_chisq_mode == "cpu-compatible"
        got = chisq_jax.shift_sum(row, points, bins)
    expected = chisq_jax._point_chisq_cpu_compatible(row, points, bins, len(row))
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_cpu_mode_never_calls_native_point_chisq(monkeypatch):
    row = _row(np.complex64, n=32)
    fail = lambda *args, **kwargs: pytest.fail("native point chi-square used")
    monkeypatch.setattr(chisq_jax, "_point_chisq_cpu_compatible", fail)
    monkeypatch.setattr(chisq_jax, "_point_chisq_cpu", fail)
    with scheme.JAXScheme(device="cpu", chisq_mode="cpu-compatible"):
        got = chisq_jax.shift_sum(row, np.array([1, 7]),
                                  np.array([0, 8, 32], dtype=np.uint32))
    assert got.devices()
    assert got.devices() == {jax.devices("cpu")[0]}


def test_precomputed_chisq_keeps_snr_normalization_on_device(monkeypatch):
    """The final SNR subtraction must not materialize a NumPy array."""
    corr = _row(np.complex64, n=32, seed=19)
    bins = np.array([0, 8, 32], dtype=np.uint32)
    points = np.array([1, 7], dtype=np.int32)
    snr = jnp.asarray([1.0 + 2.0j, 2.0 - 1.0j], dtype=jnp.complex64)
    original_asarray = chisq_jax.np.asarray

    def reject_snr_conversion(value, *args, **kwargs):
        if value is snr:
            raise AssertionError("SNR was converted through NumPy")
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(chisq_jax.np, "asarray", reject_snr_conversion)
    with scheme.JAXScheme(device="cpu"):
        got = chisq_jax.power_chisq_at_points_from_precomputed(
            corr, snr, 0.5, bins, points
        )
    assert got.devices() == {jax.devices("cpu")[0]}
    assert isinstance(got, jax.Array)


def _normalization_devices():
    try:
        return ["cpu", "cuda"] if jax.devices("gpu") else ["cpu"]
    except RuntimeError:
        return ["cpu"]


@pytest.mark.parametrize("device", _normalization_devices())
@pytest.mark.parametrize("chisq_dtype", (jnp.float32, jnp.float64))
def test_precomputed_chisq_squares_normalization_in_float64(
        monkeypatch, chisq_dtype, device):
    """Match native scalar-square then float32-array promotion."""
    monkeypatch.setattr(
        chisq_jax,
        "shift_sum",
        lambda *args, **kwargs: jnp.asarray([3.5], dtype=chisq_dtype),
    )
    bins = np.array([0, 1], dtype=np.uint32)
    points = np.array([0], dtype=np.int32)
    with scheme.JAXScheme(device=device):
        snr = jnp.asarray([1.25 + 0.0j], dtype=jnp.complex64)
        got = chisq_jax.power_chisq_at_points_from_precomputed(
            None, snr, 0.1, bins, points
        )
        assert got.devices() == {scheme.mgr.state.jax_device}
    # NumPy 1.26 promotes the scalar square to float64, then casts that
    # scalar to the array dtype before multiplication. Make both boundaries
    # explicit so this oracle is also valid under NumPy 2 scalar rules.
    term = np.asarray([3.5 - 1.25 ** 2], dtype=chisq_dtype)
    factor = np.asarray(np.float64(0.1) ** 2, dtype=chisq_dtype)
    expected = term * factor
    assert got.dtype == chisq_dtype
    np.testing.assert_array_equal(got, expected)


def test_mode_changes_are_isolated_in_scheme_backend_key(monkeypatch):
    row = _row(np.complex64, n=32)
    points = np.array([1, 7], dtype=np.int32)
    bins = np.array([0, 8, 32], dtype=np.uint32)
    calls = []
    compatible = chisq_jax._point_chisq_cpu_compatible
    direct = chisq_jax._point_chisq_cpu
    monkeypatch.setattr(chisq_jax, "_point_chisq_cpu_compatible",
                        lambda *a, **k: (calls.append("compatible") or
                                         compatible(*a, **k)))
    monkeypatch.setattr(chisq_jax, "_point_chisq_cpu",
                        lambda *a, **k: (calls.append("direct") or direct(*a, **k)))
    with scheme.JAXScheme(device="cpu", chisq_mode="cpu-compatible"):
        cpu_key = scheme.current_backend_key()
        chisq_jax.shift_sum(row, points, bins)
    with scheme.JAXScheme(device="cpu", chisq_mode="direct-phase"):
        phase_key = scheme.current_backend_key()
        chisq_jax.shift_sum(row, points, bins)
    assert cpu_key != phase_key
    assert calls == []


@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
def test_default_cpu_mode_supports_lazy_cropped_correlation(dtype):
    class Lazy:
        def __init__(self, tensor):
            self._batch_tensor = tensor
            self._batch_pos = 1
            self._kmin = 41
            self._tlen = 1024

    tensor = np.stack([_row(dtype, n=91, seed=12),
                       _row(dtype, n=91, seed=13)])
    bins = np.array([41, 41, 63, 132], dtype=np.uint32)
    points = np.array([2, 77, 2049], dtype=np.int32)
    corr = Lazy(tensor)
    with scheme.JAXScheme(device="cpu"):
        got = chisq_jax.shift_sum(corr, points, bins)
    expected = chisq_jax._point_chisq_cpu_compatible(
        tensor[1], points, bins, corr._tlen, base_k=corr._kmin)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
def test_pure_jax_recurrence_matches_reference_on_cropped_rows(dtype):
    n_time, base_k = 4096, 37
    rows = np.stack([_row(dtype, n=173, seed=8), _row(dtype, n=173, seed=9)])
    # The first bins precede the cropped correlation.  This exercises the
    # phase evolution and omitted-zero handling before ``base_k``.
    bins = np.array([[0, 0, 29, base_k + 173],
                     [0, 11, base_k + 173, base_k + 173]],
                    dtype=np.int32)
    row_indices = np.array([0, 1, 0, 1], dtype=np.int32)
    points = np.array([0, 3, 127, 4095], dtype=np.int32)
    got = compat_shift_sum(jnp.asarray(rows), jnp.asarray(row_indices),
                           jnp.asarray(points), jnp.asarray(bins), n_time,
                           base_k)
    expected = np.concatenate([
        chisq_jax._point_chisq_cpu_compatible(
            rows[r], points[i:i + 1], bins[r], n_time, base_k=base_k)
        for i, r in enumerate(row_indices)
    ])
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
def test_cpu_batch_mode_handles_empty_and_thresholded_points(dtype):
    class Corr:
        _kmin = 0
        _tlen = 64

    class Template:
        pass

    psd = object()
    tmpl = Template()
    tmpl._bin_cache = {id(psd): np.array([0, 0, 8, 16], dtype=np.uint32)}
    empty = np.array([], dtype=np.uint32)
    gated = np.array([1], dtype=np.uint32)
    snrv_dtype = dtype
    results = [
        (None, 1.0, Corr(), empty, empty.astype(snrv_dtype)),
        (None, 1.0, Corr(), gated,
         np.array([1 + 0j], dtype=snrv_dtype)),
    ]
    with scheme.JAXScheme(device="cpu"):
        got = chisq_jax.batch_power_chisq_jax(
            np.ones((2, 16), dtype=dtype), results, [tmpl, tmpl], psd, 0,
            snr_threshold=5.0)
    assert got[0][0].size == 0
    np.testing.assert_array_equal(got[1][0], [0.0])


@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
@pytest.mark.parametrize("width,count,edges", (
    (0, 2, [7, 7, 7]), (16, 0, [0, 8, 16]),
    (16, 2, [0, 0, 0]), (16, 2, [0]),
))
def test_pure_jax_empty_inputs_have_zero_power(dtype, width, count, edges):
    got = compat_shift_sum(
        jnp.ones((1, width), dtype=dtype), jnp.zeros(count, dtype=jnp.int32),
        jnp.arange(count, dtype=jnp.int32), jnp.asarray([edges]), 64, 0)
    np.testing.assert_array_equal(np.asarray(got), np.zeros(count))
    assert got.dtype == np.empty((), dtype=dtype).real.dtype


def test_cuda_mode_uses_pure_jax_kernel_without_cpu_fallback(monkeypatch):
    if "cuda" not in os.environ.get("PYCBC_TEST_SCHEME", "").lower():
        pytest.skip("CUDA chi-square tests require PYCBC_TEST_SCHEME=cuda")
    try:
        jax.devices("cuda")
    except RuntimeError:
        pytest.skip("CUDA device unavailable")
    row = jax.device_put(jnp.asarray(_row(np.complex64, n=64)), jax.devices("cuda")[0])
    points = np.array([1, 7], dtype=np.int32)
    bins = np.array([0, 8, 64], dtype=np.uint32)
    monkeypatch.setattr(chisq_jax, "_point_chisq_cpu_compatible",
                        lambda *a, **k: pytest.fail("CPU fallback used"))
    original = chisq_jax._compatible_shift_sum
    seen = []

    def spy(*args, **kwargs):
        tensor = args[0]
        assert tensor.devices() == {jax.devices("cuda")[0]}
        result = original(*args, **kwargs)
        assert result.devices() == {jax.devices("cuda")[0]}
        seen.append(True)
        return result

    monkeypatch.setattr(chisq_jax, "_compatible_shift_sum", spy)
    with scheme.JAXScheme(device="cuda", chisq_mode="cpu-compatible"):
        got = chisq_jax.shift_sum(row, points, bins)
    assert seen
    assert got.devices() == {jax.devices("cuda")[0]}
