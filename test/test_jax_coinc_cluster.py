# Copyright (C) 2026 The PyCBC Collaboration
# Licensed under GPLv3; see the repository COPYING file.
"""Regression tests for bounded JAX coincidence clustering."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

import pycbc
from pycbc import scheme
from pycbc.events import coinc, coinc_jax
from pycbc.types import Array
from pycbc.types.array_jax import is_jax_array

if not getattr(pycbc, "HAVE_JAX", False):
    pytest.skip("PyCBC built without JAX support", allow_module_level=True)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")
    return request.param


def _eager_geometry(times, slides, slide, window, multiifo):
    """The materialized operation boundaries used before bounded fusion."""
    if multiifo:
        stacked = jnp.stack(times)
        participating = stacked > 0
        count = participating.sum(axis=0)
        first = jnp.argmax(participating.astype(jnp.int64), axis=0)
        anchor = jnp.take_along_axis(stacked, first[None, :], axis=0).squeeze(0)
        relative = jnp.where(participating, stacked - anchor,
                             jnp.zeros_like(stacked)).sum(axis=0)
        time = anchor - anchor[:1] + relative / count.astype(stacked.dtype)
        if np.isfinite(slide):
            time += ((count - 1).astype(stacked.dtype) * slides.astype(stacked.dtype)
                     * slide / count.astype(stacked.dtype))
    else:
        anchor = times[0][:1]
        time = (times[0] - anchor) + (times[1] - anchor)
        if np.isfinite(slide):
            time += slides.astype(time.dtype) * slide
        time *= .5
    span = time.max() - time.min() + window * 10
    return time + span * slides.astype(time.dtype)


@pytest.mark.parametrize("multiifo", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cluster_fusion_preserves_geometry_bytes(device, multiifo, dtype):
    """Fusion may not contract arithmetic and move a clustering boundary."""
    rng = np.random.default_rng(90210)
    host = [(1e6 + rng.normal(size=72) * 1e4).astype(dtype) for _ in range(3)]
    if multiifo:
        host[2][::3] = -1.
        host[1][::5] = 0.
    with scheme.JAXScheme(device):
        times = tuple(jnp.asarray(value) for value in host[:3 if multiifo else 2])
        slides = jnp.asarray(rng.integers(-50, 50, 72), dtype=jnp.int32)
        geometry = jax.jit(coinc_jax._coincidence_cluster_time,
                           static_argnames=("multiifo", "sliding"))
        for slide, window in ((.37, 8.00003), (np.float32(.37), np.float32(.173)),
                              (np.inf, .03125), (np.nan, np.inf)):
            want = _eager_geometry(times, slides, slide, window, multiifo)
            got = geometry(times, slides, slide, window * 10, multiifo,
                           bool(np.isfinite(slide)))
            want, got = np.asarray(want), np.asarray(got)
            assert got.dtype == want.dtype and got.tobytes() == want.tobytes()


@pytest.mark.parametrize("multiifo", [False, True])
@pytest.mark.parametrize("method", ["python", "cython"])
def test_cluster_fusion_matches_ordered_eager_selection(device, multiifo, method):
    """Retain tiny shapes, NaN/tie choices, missing detectors and endpoints."""
    rng = np.random.default_rng(101)
    with scheme.JAXScheme(device):
        for size in (1, 3, 8, 72):
            stat = jnp.asarray(rng.integers(0, 4, size), dtype=jnp.float32)
            if size > 3:
                stat = stat.at[2].set(jnp.nan)
            time = 1e9 + np.arange(size) * .25
            times = (jnp.asarray(time), jnp.asarray(time + .03125))
            if multiifo:
                times += (jnp.asarray(np.where(np.arange(size) % 3, time, -1.)),)
            slides = jnp.asarray(rng.integers(-2, 3, size), dtype=jnp.int32)
            for slide in (.37, np.inf):
                for window in (.25, np.nextafter(.25, 0.), np.nextafter(.25, 1.)):
                    geometry = _eager_geometry(times, slides, slide, window, multiifo)
                    want = coinc_jax.cluster_over_time(stat, geometry, window,
                                                       method=method)
                    if multiifo:
                        got = coinc.cluster_coincs_multiifo(
                            stat, times, slides, slide, window, method=method)
                    else:
                        got = coinc.cluster_coincs(stat, *times, slides, slide,
                                                   window, method=method)
                    assert is_jax_array(got) and got.dtype == jnp.int64
                    assert got.devices() == stat.devices()
                    assert np.asarray(got).tobytes() == np.asarray(want).tobytes()


@pytest.mark.parametrize("multiifo", [False, True])
def test_cluster_fusion_preserves_wrappers_and_cpu_results(device, multiifo):
    stat = np.array([1., 3., 2., 4.], np.float32)
    time = np.array([1e9, 1e9 + .125, 1e9 + 1., 1e9 + 2.])
    slides = np.array([0, 0, 0, 1], np.int32)

    def run(stat, time, slides):
        if multiifo:
            return coinc.cluster_coincs_multiifo(stat, (time, time), slides, .5,
                                                .25, method="cython")
        return coinc.cluster_coincs(stat, time, time, slides, .5, .25,
                                   method="cython")

    want = run(stat, time, slides)
    with scheme.JAXScheme(device):
        for wrapped in ("stat", "time", "slides"):
            args = [stat, time, slides]
            args[("stat", "time", "slides").index(wrapped)] = Array(
                args[("stat", "time", "slides").index(wrapped)])
            got = run(*args)
            assert isinstance(got, Array) and got.dtype == np.int64
            np.testing.assert_array_equal(got.numpy(), want)


def test_cluster_fusion_retains_mixed_time_precision(device):
    with scheme.JAXScheme(device):
        times = (jnp.array([1e6, 1e6 + .25], jnp.float32),
                 jnp.array([1e6 + .12500001, 1e6 + .50000001], jnp.float64))
        stat, slides = jnp.array([2., 3.], jnp.float32), jnp.array([1, 1], jnp.int32)
        want = _eager_geometry(times, slides, .37, .1, False)
        geometry = jax.jit(coinc_jax._coincidence_cluster_time,
                           static_argnames=("multiifo", "sliding"))
        got = geometry(times, slides, .37, 1., False, True)
        assert got.dtype == jnp.float64
        assert np.asarray(got).tobytes() == np.asarray(want).tobytes()
        actual = coinc.cluster_coincs(stat, *times, slides, .37, .1)
        expected = coinc_jax.cluster_over_time(stat, want, .1)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cluster_fusion_preserves_nonparticipating_ifo_geometry(device, dtype):
    with scheme.JAXScheme(device):
        stat = jnp.arange(4, dtype=jnp.float32)
        slides = jnp.array([0, 1, -1, 0], jnp.int32)
        geometry = jax.jit(coinc_jax._coincidence_cluster_time,
                           static_argnames=("multiifo", "sliding"))
        for values in ([0., 1e6, 0., 1e6 + 1], [1e6, 0., 1e6 + 1, 0.]):
            times = (jnp.asarray(values, dtype=dtype),) * 3
            for slide in (.37, np.inf):
                want = _eager_geometry(times, slides, slide, .3, True)
                got = geometry(times, slides, slide, 3., True,
                               bool(np.isfinite(slide)))
                assert np.asarray(got).tobytes() == np.asarray(want).tobytes()
                for method in ("python", "cython"):
                    actual = coinc.cluster_coincs_multiifo(
                        stat, times, slides, slide, .3, method=method)
                    expected = coinc_jax.cluster_over_time(stat, want, .3,
                                                           method=method)
                    np.testing.assert_array_equal(np.asarray(actual),
                                                  np.asarray(expected))


def test_cluster_fusion_preserves_extreme_supported_scalars(monkeypatch, device):
    with scheme.JAXScheme(device):
        stat = jnp.array([1., 2., 3.], jnp.float32)
        slides = jnp.array([0, 1, 2], jnp.int32)
        for dtype in (jnp.float32, jnp.float64):
            times = (jnp.array([1e6, 1e6 + .25, 1e6 + .5], dtype),) * 2
            for slide, window in ((.3, 1e300), (1e300, .3), (2 ** 62, .3),
                                  (.3, np.int64(2 ** 62))):
                with np.errstate(over="ignore", invalid="ignore"):
                    with monkeypatch.context() as reference:
                        reference.setattr(coinc_jax, "_can_cluster_coincs",
                                          lambda *_: False)
                        expected = coinc.cluster_coincs(stat, *times, slides,
                                                        slide, window)
                    actual = coinc.cluster_coincs(stat, *times, slides, slide,
                                                 window)
                assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()


def test_cluster_fusion_reads_only_count_and_reuses_numeric_geometry(monkeypatch, device):
    traces = []
    geometry = coinc_jax._coincidence_cluster_time

    def recorded(*args, **kwargs):
        traces.append(args[4:])
        return geometry(*args, **kwargs)

    coinc_jax._coincidence_cluster_core.clear_cache()
    monkeypatch.setattr(coinc_jax, "_coincidence_cluster_time", recorded)
    try:
        with scheme.JAXScheme(device):
            stat = jnp.array([1., 3., 2., 4.], jnp.float32)
            time = jnp.array([1e9, 1e9 + .125, 1e9 + 1., 1e9 + 2.])
            slides = jnp.array([0, 0, 0, 1], jnp.int32)
            array_type, original = type(stat), type(stat).__array__

            def reject_vector_copy(value, *args, **kwargs):
                if value.ndim:
                    raise AssertionError("clustering copied a vector to host")
                return original(value, *args, **kwargs)

            with monkeypatch.context() as resident:
                resident.setattr(array_type, "__array__", reject_vector_copy)
                for window, slide in ((.25, .5), (.5, .7), (.0625, .3)):
                    got = coinc.cluster_coincs(stat, time, time, slides, slide,
                                               window, method="cython")
                    assert got.dtype == jnp.int64 and got.devices() == stat.devices()
        assert traces == [(False, True)]
    finally:
        coinc_jax._coincidence_cluster_core.clear_cache()


@pytest.mark.parametrize("case", ["large", "zero_window", "nan_window",
                                 "device_scalar", "custom_argmax", "method",
                                 "unknown_keyword", "empty", "longdouble_slide",
                                 "integer_margin_overflow"])
def test_cluster_fusion_preserves_fallback_and_validation(monkeypatch, case):
    with scheme.JAXScheme("cpu"):
        size = 129 if case == "large" else 4
        stat = jnp.arange(size, dtype=jnp.float32)
        times = (jnp.arange(size, dtype=jnp.float64),) * 2
        slides = jnp.zeros(size, dtype=jnp.int32)
        window, slide, kwargs = .25, .5, {}
        if case == "zero_window":
            window = 0.
        elif case == "nan_window":
            window = np.nan
        elif case == "device_scalar":
            window = jnp.asarray(window)
        elif case == "longdouble_slide":
            slide = np.longdouble(np.inf)
        elif case == "integer_margin_overflow":
            window = 2 ** 62
        elif case == "custom_argmax":
            kwargs["argmax"] = np.argmin
        elif case == "method":
            kwargs["method"] = "unknown"
        elif case == "unknown_keyword":
            kwargs["other"] = 1
        elif case == "empty":
            times = tuple(time[:0] for time in times)
            kwargs["method"] = "unknown"
        errors, results = [], []
        for eager in (True, False):
            with monkeypatch.context() as reference:
                if eager:
                    reference.setattr(coinc_jax, "_can_cluster_coincs", lambda *_: False)
                try:
                    results.append(coinc.cluster_coincs(stat, *times, slides, slide,
                                                        window, **kwargs))
                except Exception as error:
                    errors.append((type(error), str(error)))
                else:
                    errors.append(None)
        assert errors[0] == errors[1]
        if not errors[0]:
            assert np.asarray(results[0]).tobytes() == np.asarray(results[1]).tobytes()
