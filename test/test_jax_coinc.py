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
import copy

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


def test_jax_expiring_background_buffer_is_device_resident():
    """The live background statistic buffer performs updates with JAX arrays."""
    from pycbc.events.coinc_jax import JAXCoincExpireBuffer

    with scheme.JAXScheme():
        buffer = JAXCoincExpireBuffer(4, ["H1", "L1"], initial_size=2)
        buffer.add(
            np.array([2.0, 5.0]),
            {"H1": np.array([0, 0]), "L1": np.array([0, 0])},
            ["H1", "L1"],
        )
        assert is_jax_array(buffer.data)
    np.testing.assert_array_equal(np.asarray(buffer.data), [2.0, 5.0])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_jax_singles_buffer_keeps_approximant_metadata_host_side(device):
    """Live trigger metadata may be strings beside device numeric columns."""
    from pycbc.events.coinc_jax import JAXMultiRingBuffer

    if device == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")
    values = {"snr": np.array([3.0, 4.0]),
              "template_id": np.array([0, 1], dtype=np.int32),
              "approximant": np.array(["TaylorF2", "TaylorF2"])}
    with scheme.JAXScheme(device):
        buffer = JAXMultiRingBuffer(2, 2)
        buffer.add(values["template_id"], values)
        buffer.add(np.array([0]), {"snr": np.array([5.0]),
                                   "template_id": np.array([0], dtype=np.int32),
                                   "approximant": np.array(["TaylorF2"])})
        stored = buffer.data(0)
        # An empty block advances expiration without converting metadata.
        buffer.add(np.array([], dtype=np.int32),
                   {key: value[:0] for key, value in values.items()})
        expired = buffer.data(0)
        np.testing.assert_array_equal(np.asarray(expired["snr"]), [5.0])
        np.testing.assert_array_equal(expired["approximant"], ["TaylorF2"])
        # Expiration compacts the backing columns as well as the public view.
        assert buffer.buffer[0]["snr"].size == 1
        assert buffer.buffer_expire[0].size == 1
        assert buffer.valid_ends[0] == 1
        assert all(d.platform == ("gpu" if device == "cuda" else "cpu")
                   for d in stored["snr"].devices())
    assert is_jax_array(stored["snr"])
    assert is_jax_array(stored["template_id"])
    assert np.asarray(stored["snr"]).tolist() == [3.0, 5.0]
    assert np.asarray(stored["approximant"]).tolist() == ["TaylorF2", "TaylorF2"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_jax_singles_helper_preserves_metadata_and_device_numeric(device):
    from pycbc.events.coinc_jax import add_singles_to_buffer_jax, JAXMultiRingBuffer
    if device == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")

    class Statistic:
        single_dtype = np.dtype("float32")

        @staticmethod
        def single(trigs):
            assert is_jax_array(trigs["snr"])
            assert isinstance(trigs["approximant"], np.ndarray)
            return jnp.abs(trigs["snr"])

    class Estimator:
        stat_calculator = Statistic()
        singles = {}

        def set_singles_buffer(self, results):
            self.singles = {"H1": JAXMultiRingBuffer(2, 1)}

    estimator = Estimator()
    class Logger:
        @staticmethod
        def info(*args):
            pass

    results = {"H1": {"snr": np.array([3.]), "chisq": np.array([1.]),
                       "chisq_dof": np.array([2.]),
                       "template_id": np.array([0], dtype=np.int32),
                       "approximant": np.array(["TaylorF2"]),
                       "end_time": np.array([1.])}}
    with scheme.JAXScheme(device):
        add_singles_to_buffer_jax(estimator, results, ["H1"], Logger())
        stored = estimator.singles["H1"].data(0)
    assert is_jax_array(stored["stat"])
    assert np.asarray(stored["approximant"]).tolist() == ["TaylorF2"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("sngl_ranking", ["snr", "newsnr"])
def test_live_estimator_jax_empty_foreground_and_expiration_parity(
        device, sngl_ranking):
    from pycbc.events.coinc import LiveCoincTimeslideBackgroundEstimator

    def triggers(time, snr, template=0):
        return {"snr": np.array([snr], dtype=np.float32),
                "chisq": np.array([2.], dtype=np.float32),
                "chisq_dof": np.array([2.], dtype=np.float32),
                "template_id": np.array([template], dtype=np.int32),
                "end_time": np.array([time], dtype=np.float64),
                "mass1": np.array([10.], dtype=np.float64),
                "mass2": np.array([10.], dtype=np.float64),
                "approximant": np.array(["TaylorF2"])}

    kwargs = dict(ifar_limit=1, timeslide_interval=.1,
                  return_background=True)
    cpu = LiveCoincTimeslideBackgroundEstimator(
        2, 10, "single_ranking_only", sngl_ranking, [], ["H1", "L1"], **kwargs)
    if device == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")
    with scheme.JAXScheme(device):
        jax_estimator = LiveCoincTimeslideBackgroundEstimator(
            2, 10, "single_ranking_only", sngl_ranking, [], ["H1", "L1"], **kwargs)
    for estimator in (cpu, jax_estimator):
        estimator.buffer_size = 2
        estimator.coincs.expiration = 2
    empty = {ifo: {key: value[:0] for key, value in
                   triggers(100., 5.).items()} for ifo in ("H1", "L1")}
    blocks = [empty,
              {"H1": triggers(100., 5.), "L1": triggers(100.001, 6., 1)},
              {"H1": triggers(101., 7.), "L1": triggers(101.001, 8.)},
              {"H1": triggers(102., 5.), "L1": triggers(102.101, 6.)},
              empty, {"H1": triggers(110., 5.), "L1": False},
              empty, empty, empty]
    saw_foreground = saw_background = False
    for block in blocks:
        expected = cpu.add_singles(copy.deepcopy(block))
        original_keys = {
            ifo: set(data) for ifo, data in block.items() if data
        }
        with scheme.JAXScheme(device):
            actual = jax_estimator.add_singles(block)
        assert {ifo: set(data) for ifo, data in block.items() if data} == \
            original_keys
        assert actual.keys() == expected.keys()
        for key in expected:
            want, got = np.asarray(expected[key]), np.asarray(actual[key])
            assert got.shape == want.shape, key
            if want.dtype.kind in "biufc":
                np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7,
                                           err_msg=key)
            else:
                np.testing.assert_array_equal(got, want, err_msg=key)
        saw_foreground |= "foreground/stat" in actual
        saw_background |= actual["background/count"] > 0
    assert saw_foreground and saw_background
    assert actual["background/count"] == 0
    assert type(cpu.singles["H1"]).__name__ == "MultiRingBuffer"
    assert type(jax_estimator.singles["H1"]).__name__ == "JAXMultiRingBuffer"
    with scheme.JAXScheme(device):
        for ifo in ("H1", "L1"):
            stored = jax_estimator.singles[ifo].data(0)
            assert stored["snr"].size == 0
            assert is_jax_array(stored["snr"])
            assert all(d.platform == ("gpu" if device == "cuda" else "cpu")
                       for d in stored["snr"].devices())


def test_pick_best_coinc_jax_foreground_stat_shape():
    candidates = [{"coinc_possible": True, "foreground/ifar": 3.,
                   "foreground/stat": jnp.array([2.]),
                   "foreground/type": "H1-L1"},
                  {"coinc_possible": True, "foreground/ifar": 3.,
                   "foreground/stat": jnp.array([4.]),
                   "foreground/type": "H1-L1"}]
    with scheme.JAXScheme():
        result = coinc.LiveCoincTimeslideBackgroundEstimator.pick_best_coinc(
            candidates)
    assert np.asarray(result["foreground/stat"]).shape == (1,)
    assert np.asarray(result["foreground/stat"]).item() == 4.
    assert float(np.asarray(result["foreground/ifar"])) == 1.5


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_live_hdf_serialization_preserves_native_contract(tmp_path, device):
    import ast
    from pathlib import Path
    import h5py

    if device == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")
    # Execute the actual nested output adapter without starting the live CLI.
    executable = Path(__file__).parents[1] / "bin" / "pycbc_live"
    tree = ast.parse(executable.read_text())
    adapter = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef)
                   and node.name == "h5py_unicode_workaround")
    namespace = {"scheme": scheme}
    module = ast.Module(body=[adapter], type_ignores=[])
    exec(compile(module, str(executable), "exec"), namespace)
    serialize = namespace["h5py_unicode_workaround"]
    values = {
        "metadata_scalar": np.str_("TaylorF2"),
        "metadata_vector": np.array(["TaylorF2", "IMRPhenomD"]),
        "python_string": "H1-L1",
        "numeric_scalar": np.float32(4.5),
        "numeric_vector": np.array([1., 2.], dtype=np.float32),
    }
    with h5py.File(tmp_path / "native.hdf", "w") as native:
        for key, value in values.items():
            native[key] = serialize(value)
    with scheme.JAXScheme(device):
        with h5py.File(tmp_path / "jax.hdf", "w") as candidate:
            for key, value in values.items():
                if key.startswith("numeric"):
                    value = jnp.asarray(value)
                candidate[key] = serialize(value)
    with h5py.File(tmp_path / "native.hdf") as native, \
            h5py.File(tmp_path / "jax.hdf") as candidate:
        for key in values:
            assert candidate[key].shape == native[key].shape
            assert candidate[key].dtype == native[key].dtype
            np.testing.assert_array_equal(candidate[key][()], native[key][()])


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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_jax_single_ranking_only_foreground_stat_dtype_matches_cpu(device):
    """JAX keeps native float64 dtype for the returned coincident statistic."""
    from pycbc.events.coinc import LiveCoincTimeslideBackgroundEstimator

    jax.config.update("jax_enable_x64", True)
    if device == "cuda":
        try:
            jax.devices("gpu")
        except RuntimeError:
            pytest.skip("CUDA unavailable")

    def _trigger(time, snr):
        return {
            "snr": np.array([snr], dtype=np.float32),
            "chisq": np.array([2.0], dtype=np.float32),
            "chisq_dof": np.array([2.0], dtype=np.float32),
            "template_id": np.array([0], dtype=np.int32),
            "end_time": np.array([time], dtype=np.float64),
            "mass1": np.array([10.0], dtype=np.float64),
            "mass2": np.array([10.0], dtype=np.float64),
            "approximant": np.array(["TaylorF2"]),
        }

    kwargs = dict(ifar_limit=1, timeslide_interval=0.1, return_background=True)
    block = {"H1": _trigger(100.0, 5.0), "L1": _trigger(100.001, 6.0)}

    cpu_estimator = LiveCoincTimeslideBackgroundEstimator(
        2, 10, "single_ranking_only", "snr", [], ["H1", "L1"], **kwargs
    )
    expected = cpu_estimator.add_singles(
        {ifo: dict(values) for ifo, values in block.items()}
    )

    with scheme.JAXScheme(device):
        jax_estimator = LiveCoincTimeslideBackgroundEstimator(
            2, 10, "single_ranking_only", "snr", [], ["H1", "L1"], **kwargs
        )
        actual = jax_estimator.add_singles(
            {ifo: dict(values) for ifo, values in block.items()}
        )

    assert expected["foreground/stat"].dtype == np.dtype("float64")
    assert np.asarray(actual["foreground/stat"]).dtype == np.dtype("float64")
    np.testing.assert_allclose(
        np.asarray(actual["foreground/stat"]), expected["foreground/stat"]
    )

