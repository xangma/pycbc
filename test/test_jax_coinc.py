# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Tests for JAX coincidence construction and clustering."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
import pycbc
from pycbc import scheme
from pycbc.events import coinc
from pycbc.types import Array
from pycbc.types.array_jax import JAXArrayData, is_jax_array


if not getattr(pycbc, "HAVE_JAX", False):
    pytest.skip("PyCBC built without JAX support", allow_module_level=True)


@pytest.mark.parametrize("method", ("python", "cython"))
def test_cluster_over_time_stays_on_jax_device(monkeypatch, method):
    """Verify clustering stays on JAX device without host fallbacks."""
    host_stat = np.array([5.0, 2.0, 7.0, 7.0, -1.0, 4.0, 8.0, 3.0])
    host_time = np.array([4.0, 0.0, 1.0, 1.25, 8.0, 4.25, 7.75, 12.0])
    expected = coinc.cluster_over_time(
        host_stat, host_time, window=0.5, method=method
    )

    def reject_host_path(*_args, **_kwargs):
        raise AssertionError("clustering used the NumPy/Cython host path")

    with scheme.JAXScheme():
        jax_stat = Array(host_stat)
        jax_time = Array(host_time)
        monkeypatch.setattr(coinc, "timecluster_cython", reject_host_path)

        actual = coinc.cluster_over_time(
            jax_stat, jax_time, window=0.5, method=method
        )

    assert isinstance(actual, Array)
    assert isinstance(actual._data, JAXArrayData)
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("method", ("python", "cython"))
@pytest.mark.parametrize("jax_stat", (False, True))
def test_cluster_over_time_preserves_gps_time_precision(method, jax_stat):
    """A float32 ranking statistic must not round float64 GPS times."""
    stat = np.array([1.0, 3.0, 2.0], dtype=np.float32)
    time = np.array([1e9, 1e9 + 0.2, 1e9 + 1.0], dtype=np.float64)
    expected = coinc.cluster_over_time(stat, time, 0.5, method=method)
    np.testing.assert_array_equal(expected, [1, 2])

    with scheme.JAXScheme():
        actual = coinc.cluster_over_time(
            jnp.asarray(stat) if jax_stat else stat,
            time, 0.5, method=method,
        )

    assert is_jax_array(actual)
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize("multiifo", (False, True))
@pytest.mark.parametrize("jax_stat", (False, True))
def test_cluster_coincs_preserves_mixed_input_dtypes(multiifo, jax_stat):
    """Float32 rankings must not round distinct float64 GPS times together."""
    stat = np.array([2.0, 3.0], dtype=np.float32)
    time = np.array([1e9, 1e9 + 1.0], dtype=np.float64)
    slides = np.zeros(2, dtype=np.int32)

    def cluster(ranking):
        if multiifo:
            absent = np.full(2, -1.0, dtype=np.float64)
            return coinc.cluster_coincs_multiifo(
                ranking, (time, time, absent), slides, 0.0, 0.1
            )
        return coinc.cluster_coincs(ranking, time, time, slides, 0.0, 0.1)

    expected = cluster(stat)
    np.testing.assert_array_equal(expected, [0, 1])
    with scheme.JAXScheme("cpu"):
        actual = cluster(jnp.asarray(stat) if jax_stat else stat)

    assert is_jax_array(actual)
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_cluster_over_time_raw_jax_nan_and_validation():
    """Verify NaN handling and tie-breaking matching PyCBC specifications."""
    times = jnp.array([0.0, 0.1, 0.2, 2.0], dtype=jnp.float64)
    cases = (
        ([1.0, np.nan, 3.0, 4.0], [1, 3], [2, 3]),
        ([np.nan, 4.0, 3.0, 2.0], [0, 3], [0, 3]),
        ([1.0, 2.0, np.nan, np.nan], [2, 3], [1, 3]),
    )
    for statistics, python_expected, cython_expected in cases:
        stat = jnp.array(statistics, dtype=jnp.float64)
        for method, expected in (
            ("python", python_expected),
            ("cython", cython_expected),
        ):
            actual = coinc.cluster_over_time(
                stat, times, window=0.5, method=method
            )
            assert is_jax_array(actual)
            np.testing.assert_array_equal(np.asarray(actual), expected)

    empty = coinc.cluster_over_time(
        jnp.empty(0, dtype=jnp.float64),
        jnp.empty(0, dtype=jnp.float64),
        window=0.5,
    )
    assert is_jax_array(empty)
    assert empty.size == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cluster_over_time_reuses_numeric_geometry(monkeypatch, device):
    """A changed window/data reuses the fused core; methods retain their ties."""
    from pycbc.events import coinc_jax

    if device == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")
    cases = [(np.array([1., np.nan, 3., 4.]), .5, "python"),
             (np.array([4., 3., 3., 2.]), .15, "python"),
             (np.array([1., 3., 3., 4.]), .5, "cython")]
    times = np.array([1e9, 1e9 + .1, 1e9 + .2, 1e9 + 2.])
    expected = [coinc.cluster_over_time(stat, times, window, method=method)
                for stat, window, method in cases]
    traces = []
    maxima = coinc_jax._cluster_window_maxima

    def counted(stat, left, right, method):
        traces.append(method)
        return maxima(stat, left, right, method)

    coinc_jax._cluster_over_time_core.clear_cache()
    monkeypatch.setattr(coinc_jax, "_cluster_window_maxima", counted)
    try:
        with scheme.JAXScheme(device):
            for (stat, window, method), want in zip(cases, expected):
                actual = coinc.cluster_over_time(jnp.asarray(stat),
                                                jnp.asarray(times), window,
                                                method=method)
                assert is_jax_array(actual)
                np.testing.assert_array_equal(np.asarray(actual), want)
        assert traces == ["python", "cython"]
    finally:
        coinc_jax._cluster_over_time_core.clear_cache()


def test_cluster_over_time_keeps_wrapper_validation(monkeypatch):
    """Validation and data-dependent empty handling precede the compiled core."""
    from pycbc.events import coinc_jax

    def reject_core(*args, **kwargs):
        raise AssertionError("invalid/empty input reached clustering core")

    monkeypatch.setattr(coinc_jax, "_cluster_over_time_core", reject_core)
    with scheme.JAXScheme("cpu"):
        stat, times = jnp.ones(2), jnp.arange(2, dtype=jnp.float64)
        for values, clock, kwargs, error in (
                (stat, times, {"method": "unknown"}, ValueError),
                (stat[None, :], times, {}, TypeError),
                (stat, times[:1], {}, ValueError),
                (stat.astype(jnp.complex64), times, {}, TypeError),
                (stat, times.astype(jnp.int64), {}, TypeError),
                (stat, times, {"argmax": np.argmin}, NotImplementedError),
                (stat, times, {"window": 0.}, ValueError)):
            with pytest.raises(error):
                coinc.cluster_over_time(values, clock,
                                        **({"window": .5} | kwargs))
        empty = coinc.cluster_over_time(stat[:0], times[:0], window=0.,
                                       argmax=np.argmin)
        assert empty.shape == (0,) and empty.dtype == jnp.int64


def test_time_coincidence_jax_parity():
    """Verify time coincidence matching Cython/NumPy with and without slides."""
    time1 = np.array([0.0, 0.6, 1.2, 2.5, 4.0, 6.1], dtype=np.float64)
    time2 = np.array([0.1, 0.8, 1.1, 3.0, 3.9, 6.05], dtype=np.float64)

    # Without time slides
    expected = coinc.time_coincidence(time1, time2, 0.25)
    actual = coinc.time_coincidence(
        jnp.asarray(time1), jnp.asarray(time2), 0.25
    )
    for jax_val, numpy_val in zip(actual, expected):
        assert is_jax_array(jax_val)
        np.testing.assert_array_equal(np.asarray(jax_val), numpy_val)

    # With time slides
    expected_slide = coinc.time_coincidence(time1, time2, 0.25, slide_step=1.0)
    actual_slide = coinc.time_coincidence(
        jnp.asarray(time1), jnp.asarray(time2), 0.25, slide_step=1.0
    )
    for jax_val, numpy_val in zip(actual_slide, expected_slide):
        assert is_jax_array(jax_val)
        np.testing.assert_array_equal(np.asarray(jax_val), numpy_val)


def test_time_coincidence_preserves_mixed_gps_precision():
    """A float32 JAX detector must not narrow the other detector's GPS times."""
    time1 = np.array([1e9, 1e9 + 1.0], dtype=np.float64)
    time2 = np.array([1e9], dtype=np.float32)
    expected = coinc.time_coincidence(time1, time2, 0.1)
    np.testing.assert_array_equal(expected[0], [0])

    with scheme.JAXScheme("cpu"):
        actual = coinc.time_coincidence(time1, jnp.asarray(time2), 0.1)

    for jax_val, numpy_val in zip(actual, expected):
        assert is_jax_array(jax_val)
        np.testing.assert_array_equal(np.asarray(jax_val), numpy_val)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_time_coincidence_single_matches_general_order_and_endpoints(device):
    """Singleton fusion retains folding, duplicate pairs and rounding bytes."""
    from pycbc.events import coinc_jax

    if device == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")
    cases = [
        ([1e9 + .75, 1e9 + .25], [1e9 + .125], .2, 1.),
        ([-.8, -.2], [-.1], .2, 1.),
        ([.2, .1], [.1], 1., 1.),
        ([.1, .1], [.1], 2., 1.),
        ([0., .5], [.25], .25, 0.),
        ([np.nextafter(0., 1.), np.nextafter(.5, 0.)], [.25], .25, 0.),
        ([np.nextafter(0., -1.), np.nextafter(.5, 1.)], [.25], .25, 0.),
        ([-.5, .5], [0.], .75, 1.),
        ([np.nextafter(.5, 0.)], [0.], 1., 1.),
        ([.2, .1], [.1], 0., 1.),
        ([.2, .1], [.9], .01, 0.),
        ([np.nan, .2], [.1], .3, 1.),
        ([np.inf, .2], [.1], .3, 1.),
        ([.2, .1], [np.nan], .3, 1.),
        (np.array([1e9, 1e9 + 1.]), np.array([1e9], "f4"), .1, 0.),
        (np.array([.5, -.5], "f4"), np.array([0.], "f4"), .75, 1.),
        (np.arange(64)[::-1] / 64., [.5], .1, 1.),
    ]
    # Preserve the independent standard CPU oracle where its double-precision
    # Cython input contract and half-away-from-zero rounding match the backend.
    cpu_expected = [coinc.time_coincidence(
        np.asarray(t1, dtype="f8"), np.asarray(t2, dtype="f8"), window, step)
        for t1, t2, window, step in cases[:8]]
    with scheme.JAXScheme(device):
        for pos, (t1, t2, window, step) in enumerate(cases):
            arr1, arr2 = jnp.asarray(t1), jnp.asarray(t2)
            dtype = jnp.result_type(arr1.dtype, arr2.dtype)
            expected = coinc_jax._time_coincidence_general(
                arr1.astype(dtype), arr2.astype(dtype), window, step)
            actual = coinc.time_coincidence(arr1, arr2, window, step)
            for got, want, output_dtype in zip(
                    actual, expected, (np.int64, np.int64, np.int32)):
                assert is_jax_array(got)
                assert got.dtype == output_dtype
                assert got.devices() == arr1.devices()
                assert np.asarray(got).tobytes() == np.asarray(want).tobytes()
            if pos < len(cpu_expected):
                for got, want in zip(actual, cpu_expected[pos]):
                    np.testing.assert_array_equal(np.asarray(got), want)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("wrapped", ["none", "first", "second", "both"])
def test_time_coincidence_single_preserves_mixed_wrappers(device, wrapped):
    """A host/device mixture preserves GPS precision, placement and wrappers."""
    if device == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")
    times = np.array([1e9 + 1., 1e9], dtype="f8")
    fixed = np.array([1e9], dtype="f4")
    expected = coinc.time_coincidence(times, fixed, .1)
    with scheme.JAXScheme(device):
        first = Array(times) if wrapped in ("first", "both") else times
        second = (Array(fixed) if wrapped in ("second", "both") else
                  jnp.asarray(fixed))
        actual = coinc.time_coincidence(first, second, .1)
        for got, want in zip(actual, expected):
            if wrapped != "none":
                assert isinstance(got, Array)
                assert isinstance(got._data, JAXArrayData)
                got = got._data._array
            else:
                assert is_jax_array(got)
            assert all(d.platform == ("gpu" if device == "cuda" else "cpu")
                       for d in got.devices())
            np.testing.assert_array_equal(np.asarray(got), want)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_time_coincidence_single_reuses_core_and_compacts_once(monkeypatch,
                                                            device):
    """Changing numeric times/windows/slides avoids eager dynamic repeats."""
    from pycbc.events import coinc_jax

    if device == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")
    traces, compactions = [], []
    nonzero = coinc_jax.jnp.nonzero
    compact = coinc_jax._compact_time_coincidence

    def counted_nonzero(mask, **kwargs):
        traces.append(mask.shape)
        return nonzero(mask, **kwargs)

    def counted_compact(*args, count):
        compactions.append(count)
        return compact(*args, count=count)

    def reject_repeat(*args, **kwargs):
        raise AssertionError("singleton coincidence dispatched dynamic repeat")

    coinc_jax._time_coincidence_single_core.clear_cache()
    monkeypatch.setattr(coinc_jax.jnp, "nonzero", counted_nonzero)
    monkeypatch.setattr(coinc_jax.jnp, "repeat", reject_repeat)
    monkeypatch.setattr(coinc_jax, "_compact_time_coincidence", counted_compact)
    try:
        with scheme.JAXScheme(device):
            times, fixed = jnp.array([.2, .1]), jnp.array([.1])
            for window, step in ((.01, 1.), (.2, 2.), (2., 1.),
                                 (0., 1.), (.01, 0.)):
                outputs = coinc.time_coincidence(times, fixed, window, step)
                assert outputs[0].size == compactions[-1]
                assert all(is_jax_array(value) for value in outputs)
        assert traces == [(2, 3), (2, 1)]
        assert compactions == [1, 2, 6, 0, 1]
    finally:
        coinc_jax._time_coincidence_single_core.clear_cache()


def test_time_coincidence_single_keeps_general_fallback_and_validation(
        monkeypatch):
    """The fused path leaves larger, empty and unusual inputs unchanged."""
    from pycbc.events import coinc_jax

    def reject_core(*args, **kwargs):
        raise AssertionError("ineligible coincidence reached singleton core")

    monkeypatch.setattr(coinc_jax, "_time_coincidence_single_core", reject_core)
    with scheme.JAXScheme("cpu"):
        cases = [(np.arange(65, dtype="f8"), [.1], .01, 1.),
                 ([.2, .1], [.1, .2], .1, 1.),
                 ([], [.1], .1, 1.), ([.1], [], .1, 1.),
                 ([.2, .1], [.1], .1, -1.),
                 ([.2, .1], [.1], np.nan, 1.),
                 ([.2, .1], [.1], np.inf, 1.)]
        for times, fixed, window, step in cases:
            times = jnp.asarray(times, dtype=jnp.float64)
            fixed = jnp.asarray(fixed, dtype=jnp.float64)
            expected = coinc_jax._time_coincidence_general(
                times, fixed, window, step)
            actual = coinc.time_coincidence(times, fixed, window, step)
            for got, want in zip(actual, expected):
                assert np.asarray(got).tobytes() == np.asarray(want).tobytes()
        with pytest.raises(TypeError):
            coinc.time_coincidence(jnp.array([.2, .1]), jnp.array([.1]),
                                   -.1, 1.)
        for times, fixed in ((jnp.ones((1, 1)), jnp.ones(1)),
                             (jnp.ones(1, dtype=jnp.int64), jnp.ones(1)),
                             (jnp.ones(1), jnp.ones(1, dtype=jnp.int64))):
            with pytest.raises(TypeError):
                coinc.time_coincidence(times, fixed, .1, 1.)


def test_numpy_coincidence_inputs_use_jax_under_jax_scheme():
    """Live host buffers enter the JAX backend at the scheme boundary."""
    time1 = np.array([0.0, 0.6, 1.2, 2.5, 4.0, 6.1], dtype=np.float64)
    time2 = np.array([0.1, 0.8, 1.1, 3.0, 3.9, 6.05], dtype=np.float64)
    stat = np.array([2.0, 7.0, 4.0, 8.0, 3.0, 6.0], dtype=np.float64)
    slide_ids = np.array([0, 0, 1, 1, -1, 2], dtype=np.int32)

    expected_coinc = coinc.time_coincidence(
        time1, time2, 0.25, slide_step=1.0
    )
    expected_cluster = coinc.cluster_coincs(
        stat, time1, time2, slide_ids, 1.0, 0.5
    )
    expected_multi = coinc.cluster_coincs_multiifo(
        stat,
        (time1, time2, time1 + 10.0),
        slide_ids,
        1.0,
        0.5,
    )
    expected_over_time = coinc.cluster_over_time(
        stat, time1, 0.5, method="python"
    )

    with scheme.JAXScheme():
        actual_coinc = coinc.time_coincidence(
            time1, time2, 0.25, slide_step=1.0
        )
        actual_cluster = coinc.cluster_coincs(
            stat, time1, time2, slide_ids, 1.0, 0.5
        )
        actual_multi = coinc.cluster_coincs_multiifo(
            stat,
            (time1, time2, time1 + 10.0),
            slide_ids,
            1.0,
            0.5,
        )
        actual_over_time = coinc.cluster_over_time(
            stat, time1, 0.5, method="python"
        )

    for actual, expected in zip(actual_coinc, expected_coinc):
        assert is_jax_array(actual)
        np.testing.assert_array_equal(np.asarray(actual), expected)
    for actual, expected in (
        (actual_cluster, expected_cluster),
        (actual_multi, expected_multi),
        (actual_over_time, expected_over_time),
    ):
        assert is_jax_array(actual)
        np.testing.assert_array_equal(np.asarray(actual), expected)


def test_cluster_coincs_jax_parity():
    """Verify 2-detector and multi-detector coincidence clustering."""
    time1 = np.array([0.0, 0.6, 1.2, 2.5], dtype=np.float64)
    time2 = np.array([0.1, 0.8, 1.1, 3.0], dtype=np.float64)
    stat = np.array([2.0, 7.0, 4.0, 8.0], dtype=np.float64)
    slide_ids = np.array([0, 0, 1, 1], dtype=np.int32)

    expected = coinc.cluster_coincs(stat, time1, time2, slide_ids, 1.0, 0.5)
    actual = coinc.cluster_coincs(
        jnp.asarray(stat),
        jnp.asarray(time1),
        jnp.asarray(time2),
        jnp.asarray(slide_ids),
        1.0,
        0.5,
    )
    assert is_jax_array(actual)
    np.testing.assert_array_equal(np.asarray(actual), expected)

    # Multi-detector
    detector_times = (
        time1 + 10.0,
        time2 + 10.0,
        np.array([-1.0, 10.7, 11.0, 12.9], dtype=np.float64),
    )
    expected_multi = coinc.cluster_coincs_multiifo(
        stat, detector_times, slide_ids, 1.0, 0.5
    )
    actual_multi = coinc.cluster_coincs_multiifo(
        jnp.asarray(stat),
        tuple(jnp.asarray(v) for v in detector_times),
        jnp.asarray(slide_ids),
        1.0,
        0.5,
    )
    assert is_jax_array(actual_multi)
    np.testing.assert_array_equal(np.asarray(actual_multi), expected_multi)


