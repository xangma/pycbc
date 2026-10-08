"""Exact immutable Welch preparation contracts, not search qualification."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
enable_x64 = getattr(jax, "enable_x64", None)
if enable_x64 is None:
    from jax.experimental import enable_x64

from pycbc import scheme  # noqa: E402
from pycbc.psd import estimate_jax as estimate  # noqa: E402
from pycbc.types import TimeSeries  # noqa: E402


def _devices():
    devices = list(jax.devices("cpu")[:1])
    try:
        devices.extend(jax.devices("gpu")[:1])
    except RuntimeError:
        pass
    return devices


@pytest.fixture(autouse=True)
def clear_preparation_caches():
    """Each regression starts cold and leaves no arrays for another test."""
    estimate._cached_hann_window.cache_clear()
    estimate._cached_welch_norm.cache_clear()
    yield
    estimate._cached_hann_window.cache_clear()
    estimate._cached_welch_norm.cache_clear()


def _cache_info():
    return (estimate._cached_hann_window.cache_info(),
            estimate._cached_welch_norm.cache_info())


def _assert_exact(actual, expected, device):
    assert isinstance(actual, jax.Array)
    assert actual.devices() == {device}
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()


def _samples(length=64, dtype=np.float32):
    return np.random.default_rng(170817).normal(size=length * 3).astype(dtype)


def _welch(samples, length=64, **kwargs):
    return estimate.welch_jax(samples, seg_len=length, seg_stride=length,
                              num_segments=3, require_exact_data_fit=True,
                              **kwargs)


def _uncached(samples, length=64, **kwargs):
    # Keep the original JAX construction and cast. NumPy hanning is not an
    # exact oracle, and wide FFTs still use a window rounded to input dtype.
    window = jnp.hanning(length).astype(samples.dtype)
    return _welch(samples, length, window=window, **kwargs)


@pytest.mark.parametrize("device", _devices(), ids=str)
@pytest.mark.parametrize("length", [63, 64])
@pytest.mark.parametrize("dtype,wide", [(np.float32, False),
                                        (np.float32, True),
                                        (np.float64, False)])
@pytest.mark.parametrize("average", ["mean", "median", "median-mean"])
def test_welch_cold_and_warm_cache_match_uncached_bytes(
        device, length, dtype, wide, average):
    with enable_x64(True), jax.default_device(device):
        samples = jax.device_put(_samples(length, dtype), device)
        kwargs = dict(avg_method=average, wide_fft=wide)
        cold = _welch(samples, length, **kwargs)
        initial = _cache_info()
        warm = _welch(samples, length, **kwargs)
        repeated = _cache_info()
        expected = _uncached(samples, length, **kwargs)
        _assert_exact(cold, expected, device)
        _assert_exact(warm, expected, device)
        assert _cache_info() == repeated  # Custom windows bypass both caches.
        assert initial[0].misses == initial[1].misses == 1
        for first, second in zip(initial, repeated):
            assert second.misses == first.misses
            assert second.hits > first.hits


@pytest.mark.parametrize("device", _devices(), ids=str)
def test_geometry_input_and_output_precision_have_distinct_keys(device):
    """Alternating narrow/wide calls must not reuse the wrong energy sum."""
    with enable_x64(True), jax.default_device(device):
        configurations = [(64, np.float32, False), (64, np.float32, True),
                          (63, np.float32, False), (64, np.float64, False),
                          (64, np.float32, False)]
        for length, dtype, wide in configurations:
            samples = jax.device_put(_samples(length, dtype), device)
            actual = _welch(samples, length, wide_fft=wide)
            _assert_exact(actual, _uncached(samples, length, wide_fft=wide),
                          device)
        window_info, norm_info = _cache_info()
        assert window_info.misses == 3
        assert norm_info.misses == 4
        assert norm_info.hits >= 1


@pytest.mark.parametrize("device", _devices(), ids=str)
def test_sample_rate_changes_norm_and_keeps_series_metadata(device):
    """The same window geometry at another rate needs another full norm."""
    selected = "cuda:0" if device.platform == "gpu" else "cpu"
    with enable_x64(True), scheme.JAXScheme(selected):
        first = None
        for delta_t, epoch in [(1 / 2048, 10), (1 / 1024, 20), (1 / 2048, 30)]:
            samples = TimeSeries(_samples(), delta_t=delta_t, epoch=epoch)
            actual = _welch(samples)
            window = jnp.hanning(64).astype(samples.dtype)
            expected = _welch(samples, window=window)
            _assert_exact(actual._data.array, expected._data.array, device)
            assert actual.delta_f == 1 / (64 * delta_t)
            assert actual.start_time == samples.start_time
            if first is None:
                first = np.asarray(actual._data.array)
            elif delta_t == 1 / 1024:
                values = np.asarray(actual._data.array)
                assert not np.array_equal(values, first)
        window_info, norm_info = _cache_info()
        assert window_info.misses == 1
        assert norm_info.misses == 2
        assert norm_info.hits == 1


@pytest.mark.parametrize("device", _devices(), ids=str)
def test_mutable_custom_window_is_never_cached(device):
    with enable_x64(True), jax.default_device(device):
        samples = jax.device_put(_samples(), device)
        window = np.hanning(64).astype(np.float64)
        before = _welch(samples, window=window)
        # Uniform rescaling cancels in normalized PSDs; change the shape.
        window[8:24] *= 0.125
        after = _welch(samples, window=window)
        fresh = _welch(samples, window=window.copy())
        _assert_exact(after, fresh, device)
        assert not np.array_equal(np.asarray(before), np.asarray(after))
        assert all(info.currsize == info.misses == info.hits == 0
                   for info in _cache_info())


@pytest.mark.parametrize("device", _devices(), ids=str)
def test_preparation_reuse_does_not_reuse_psd_values(device):
    with enable_x64(True), jax.default_device(device):
        original = _samples()
        outputs = []
        buffers = (original, original[::-1].copy(), original + np.float32(2))
        for data in buffers:
            samples = jax.device_put(data, device)
            actual = _welch(samples)
            _assert_exact(actual, _uncached(samples), device)
            outputs.append(np.asarray(actual))
        assert not np.array_equal(outputs[0], outputs[1])
        assert not np.array_equal(outputs[0], outputs[2])
        assert all(info.misses == 1 for info in _cache_info())
        assert _cache_info()[1].hits == 2


def test_caches_are_bounded_and_eviction_preserves_values():
    device = jax.devices("cpu")[0]
    with enable_x64(True), jax.default_device(device):
        window_cache = estimate._cached_hann_window
        norm_cache = estimate._cached_welch_norm
        first_window = window_cache(32, np.dtype("float32"), device, True)
        for index in range(window_cache.cache_info().maxsize):
            window_cache(33 + index, np.dtype("float32"), device, True)
        info = window_cache.cache_info()
        assert info.currsize == info.maxsize
        _assert_exact(window_cache(32, np.dtype("float32"), device, True),
                      first_window, device)
        assert window_cache.cache_info().misses == info.misses + 1

        # Norm construction itself hits the window cache, so test its own
        # eviction independently instead of assuming the same LRU ordering.
        first_norm = norm_cache(32, np.dtype("float32"), np.dtype("float64"),
                                1 / 32, device, True)
        for index in range(norm_cache.cache_info().maxsize):
            norm_cache(32, np.dtype("float32"), np.dtype("float64"),
                       (index + 2) / 32, device, True)
        info = norm_cache.cache_info()
        assert info.currsize == info.maxsize
        repeated = norm_cache(32, np.dtype("float32"), np.dtype("float64"),
                              1 / 32, device, True)
        _assert_exact(repeated, first_norm, device)
        assert norm_cache.cache_info().misses == info.misses + 1


@pytest.mark.parametrize("device", _devices(), ids=str)
@pytest.mark.parametrize("closed_over", [False, True])
@pytest.mark.parametrize("warm", [False, True])
def test_tracing_never_leaks_cached_tracers(device, closed_over, warm):
    with enable_x64(True), jax.default_device(device):
        samples = jax.device_put(_samples(), device)
        if warm:
            _welch(samples)
        before = _cache_info()
        if closed_over:
            actual = jax.jit(lambda: _welch(samples))()
            expected = jax.jit(lambda: _uncached(samples))()
        else:
            actual = jax.jit(_welch)(samples)
            expected = jax.jit(_uncached)(samples)
            assert _cache_info() == before
        _assert_exact(actual, expected, device)
        # A leaked tracer would fail an ordinary later hit, even if JIT worked.
        _assert_exact(_welch(samples), _uncached(samples), device)
        caches = (estimate._cached_hann_window, estimate._cached_welch_norm)
        for cached in caches:
            assert cached.cache_info().currsize == 1


@pytest.mark.filterwarnings(
    "ignore:.*dtype float64.*not available.*:UserWarning")
def test_x64_mode_separates_preparation(monkeypatch):
    monkeypatch.setenv("PYCBC_JAX_ENABLE_X64", "0")
    device = jax.devices("cpu")[0]
    with jax.default_device(device):
        for x64 in (True, False, True):
            with enable_x64(x64):
                samples = jax.device_put(_samples(), device)
                _assert_exact(_welch(samples), _uncached(samples), device)
        assert _cache_info()[0].misses == 2
        assert _cache_info()[1].misses == 2


@pytest.mark.skipif(len(_devices()) < 2, reason="requires CPU and CUDA")
def test_device_caches_do_not_reuse_another_device():
    with enable_x64(True):
        for device in _devices():
            with jax.default_device(device):
                samples = jax.device_put(_samples(), device)
                _assert_exact(_welch(samples), _uncached(samples), device)
        assert _cache_info()[0].misses == 2
        assert _cache_info()[1].misses == 2


@pytest.mark.skipif(len(_devices()) < 2, reason="requires CPU and CUDA")
def test_explicit_nondefault_device_preserves_uncached_behavior():
    cpu, gpu = _devices()
    with enable_x64(True), jax.default_device(cpu):
        samples = jax.device_put(_samples(), gpu)
        assert estimate._hann_cache_key(64, samples) is None
        # Depending on JAX version, the original separately committed window
        # may reject cross-device arguments. Preserve that boundary as well.
        try:
            expected = _uncached(samples, device=gpu)
        except ValueError as error:
            with pytest.raises(type(error)):
                _welch(samples, device=gpu)
        else:
            _assert_exact(_welch(samples, device=gpu), expected, gpu)
        assert all(info.currsize == info.misses == info.hits == 0
                   for info in _cache_info())
