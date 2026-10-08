# Copyright (C) 2026 The PyCBC Collaboration
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

"""Exact Live result assembly contracts; these do not qualify a search."""

from types import MethodType, SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
enable_x64 = getattr(jax, "enable_x64", None)
if enable_x64 is None:
    from jax.experimental import enable_x64

from pycbc import scheme  # noqa: E402
from pycbc.filter import matchedfilter_jax as module  # noqa: E402
from pycbc.filter.matchedfilter import LiveBatchMatchedFilter  # noqa: E402
from pycbc.filter.matchedfilter_jax import (  # noqa: E402
    combine_live_results_jax,
)


def _devices():
    devices = ["cpu"]
    try:
        if jax.devices("gpu"):
            devices.append("cuda:0")
    except RuntimeError:
        pass
    return devices


def _assert_numeric_exact(actual, expected, device):
    assert isinstance(actual, jax.Array)
    assert actual.device == device
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    # Array equality alone misses signed zero and integer-to-float rounding.
    assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()


_HOST_COLUMNS = [
    pytest.param([np.array([2**53 + 1, 2**62 + 3], np.int64),
                  np.array([-2**53 - 1], np.int64)], id="large-int64"),
    pytest.param([np.array([2**63 + 1], np.uint64),
                  np.array([2**64 - 1], np.uint64)], id="large-uint64"),
    pytest.param([np.array([2**53 + 1, -1], np.int64),
                  np.array([2**63 + 1], np.uint64)], id="signed-unsigned"),
    pytest.param([np.array([2**53 + 1, -1], np.int64),
                  np.array([2**63 + 1], np.uint64),
                  np.array([1.25], np.float32)], id="wide-integers-float32"),
    pytest.param([np.array([2**32 - 1], np.uint32),
                  np.array([-3], np.int32),
                  np.array([0.25], np.float32)], id="integer-float32"),
    pytest.param([np.array([True, False]), np.array([3], np.int16)],
                 id="bool-integer"),
    pytest.param([[2**53 + 1, -3], [4]], id="python-integer-lists"),
    pytest.param([[1, 2], [0.25]], id="python-mixed-lists"),
    pytest.param([np.array([-0.0, 0.0], np.float32),
                  np.array([-0.0, 0.0], np.float64)], id="signed-zero"),
    pytest.param([np.array([1, 0x80000001, 0x00800000], np.uint32)
                  .view(np.float32), np.array([0.0], np.float64)],
                 id="float32-subnormal-widening"),
    pytest.param([np.array([1, 0x80000001], np.uint32)
                  .view(np.complex64), np.array([0j], np.complex128)],
                 id="complex64-subnormal-widening"),
    pytest.param([np.array([0x80000000, 0, 0x7fc00041], np.uint32)
                  .view(np.float32), np.array([np.inf], np.float32)],
                 id="float-bits"),
    pytest.param([np.array([-0.0 + 1j], np.complex64),
                  np.array([2.5], np.float32)], id="complex-real"),
    pytest.param([np.empty(0, np.float64), np.array([1.25], np.float32),
                  np.empty(0, np.int64)], id="part-empty-promotion"),
    pytest.param([np.empty(0, np.float64), np.empty(0, np.float64)],
                 id="all-empty-approximant"),
    pytest.param([np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2],
                  np.array([[1, 2]], np.int32)], id="strided-matrix"),
]


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("x64", [True, False])
@pytest.mark.parametrize("values", _HOST_COLUMNS)
@pytest.mark.filterwarnings("ignore:.*not available.*:UserWarning")
def test_combine_host_numeric_columns_matches_legacy_bytes(
        device, x64, values):
    """NumPy promotion must not replace JAX's canonicalization/promotion."""
    with scheme.JAXScheme(device) as active, enable_x64(x64):
        try:
            expected = jnp.concatenate(
                [jnp.asarray(value) for value in values])
        except OverflowError:
            # Large Python integers can overflow before concatenation when
            # x64 is disabled; an optimized path must retain that rejection.
            with pytest.raises(OverflowError):
                combine_live_results_jax(
                    [{"column": value} for value in values])
            return
        actual = combine_live_results_jax(
            [{"column": value} for value in values])["column"]
        _assert_numeric_exact(actual, expected, active.jax_device)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("x64", [True, False])
def test_combine_homogeneous_host_column_uses_one_upload(
        device, x64, monkeypatch):
    """Join host groups without an upload per input and device concat."""
    with scheme.JAXScheme(device) as active, enable_x64(x64):
        values = [np.array([-0.0, 1.25], np.float64),
                  np.empty(0, np.float64), np.array([2.5], np.float64)]
        expected = jnp.concatenate([jnp.asarray(value) for value in values])
        original = jnp.asarray
        uploads = []

        def upload(value, *args, **kwargs):
            uploads.append(value)
            return original(value, *args, **kwargs)

        def reject_device_concat(*args, **kwargs):
            raise AssertionError("host column dispatched JAX concat")

        with monkeypatch.context() as patch:
            patch.setattr(jnp, "asarray", upload)
            patch.setattr(jnp, "concatenate", reject_device_concat)
            actual = combine_live_results_jax(
                [{"column": value} for value in values])["column"]
        assert len(uploads) == 1
        _assert_numeric_exact(actual, expected, active.jax_device)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("mixed_host", [False, True])
def test_combine_resident_numeric_columns_never_download(
        device, mixed_host, monkeypatch):
    """Resident columns keep their device path, including mixed host data."""
    with scheme.JAXScheme(device) as active:
        values = [jnp.asarray([2**53 + 1, -3], jnp.int64),
                  np.array([0.25, -0.0], np.float32)]
        if not mixed_host:
            values[1] = jnp.asarray(values[1])
        expected = jnp.concatenate([jnp.asarray(value) for value in values])

        def reject_download(self, *args, **kwargs):
            raise AssertionError("assembly downloaded a resident column")

        with monkeypatch.context() as patch:
            patch.setattr(type(values[0]), "__array__", reject_download)
            actual = combine_live_results_jax(
                [{"column": value} for value in values])["column"]
        _assert_numeric_exact(actual, expected, active.jax_device)


@pytest.mark.parametrize("values", [
    pytest.param([np.array(["TaylorF2", "IMRPhenomD"]),
                  np.array(["SEOBNRv4_ROM"])], id="strings"),
    pytest.param([np.array(["TaylorF2", None], object),
                  np.array([{"name": "IMRPhenomD"}], object)], id="objects"),
    pytest.param([np.empty(0, object), np.array(["TaylorF2"], object),
                  np.empty(0, object)], id="part-empty-object"),
])
def test_combine_nonnumeric_metadata_stays_host(values):
    with scheme.JAXScheme("cpu"):
        actual = combine_live_results_jax(
            [{"approximant": value} for value in values])["approximant"]
    expected = np.concatenate(values)
    assert isinstance(actual, np.ndarray)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    if actual.dtype.kind == "O":
        for item, reference in zip(actual, expected):
            assert item is reference


@pytest.mark.parametrize("values", [
    [np.int64(3), np.int64(4)], [np.ones(2), np.ones((1, 2))],
    [np.array(3, np.int64), np.array(4, np.int64)],
    [np.ones((1, 2)), np.ones((2, 3))],
])
def test_combine_rejects_invalid_numeric_shapes_like_legacy(values):
    with scheme.JAXScheme("cpu"):
        with pytest.raises((TypeError, ValueError)) as legacy_error:
            jnp.concatenate([jnp.asarray(value) for value in values])
        with pytest.raises(type(legacy_error.value)):
            combine_live_results_jax([{"column": value} for value in values])


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("x64", [True, False])
@pytest.mark.parametrize("template_ids", [[1, 2], [2**63 + 1, 2**63 + 2]])
@pytest.mark.parametrize("threshold,selected", [
    (0.0, [0, 1]), (2.0, [0]), (6.0, []),
])
@pytest.mark.filterwarnings("ignore:.*not available.*:UserWarning")
def test_live_batch_keeps_host_metadata_until_combination(
        device, x64, template_ids, threshold, selected, monkeypatch):
    """Sparse host metadata needs one publication, not one upload per group.

    Filtering/selection columns remain resident. This tests the real batch
    assembly boundary with fixed peaks, not numerical filtering qualification.
    """
    from pycbc.filter import matchedfilter_jax as module

    with scheme.JAXScheme(device) as active, enable_x64(x64):
        params = np.array(
            [(2**53 + 1, "TaylorF2", 4.0),
             (2**53 + 3, "IMRPhenomD", 4.0)],
            dtype=[("template_hash", np.int64), ("approximant", object),
                   ("template_duration", np.float32)])
        templates = [SimpleNamespace(
            delta_f=0.25, params=params[i], id=template_ids[i],
            time_offset=np.float32(i / 8)) for i in range(2)]
        data = SimpleNamespace(psd=object())
        matrix = jnp.zeros((2, 257), jnp.complex64)
        control = SimpleNamespace(
            block_id=0, tgroups=[templates], chunk_tsamples=[512], mids=[0],
            data=SimpleNamespace(overwhitened_data=lambda delta_f: data,
                                 trim_padding=0, blocksize=1, sample_rate=128,
                                 start_time=100.0),
            corr=[SimpleNamespace(_jax_template_matrix=matrix,
                                  execute=lambda stilde: None)],
            ifts={0: SimpleNamespace(execute=lambda: None)}, out_mem={0: None},
            cout_mem={0: jnp.zeros(1024, jnp.complex64)},
            snr_threshold=threshold, snr_abort_threshold=None)
        peaks = jnp.asarray([3 + 4j, -3 + 4j], jnp.complex64)
        norms = jnp.asarray([1.0, 0.25], jnp.float64)
        native = jnp.asarray([4.0, 16.0], jnp.float64)
        peak_indices = np.array([20, 30], np.uint32)
        monkeypatch.setattr(module, "_live_cached_template_norms_jax",
                            lambda *args: (native, norms))
        monkeypatch.setattr(module, "batch_peak_values",
                            lambda *args: (jnp.asarray(peak_indices), peaks))
        try:
            legacy_ids = jnp.asarray([template_ids[i] for i in selected],
                                     dtype=jnp.uint64)
        except OverflowError:
            # The original Python-int constructor rejects oversized IDs
            # before any combined-column canonicalization when x64 is off.
            with pytest.raises(OverflowError):
                module.live_process_batch_jax(control)
            return
        result, candidates = module.live_process_batch_jax(control)
        for key in ("end_time", "template_id", "template_hash",
                    "template_duration", "approximant"):
            assert isinstance(result[key], np.ndarray), key
        if not selected:
            assert "time_offset" not in result
        for key in ("snr", "coa_phase", "sigmasq"):
            assert isinstance(result[key], jax.Array), key
            assert result[key].device == active.jax_device
        if selected:
            assert isinstance(result["time_offset"], jax.Array)
            assert result["time_offset"].device == active.jax_device
        assert [info.metadata[4] for info in candidates] == selected
        assert [info.metadata[0] for info in candidates] == [
            int(peak_indices[i]) + 384 for i in selected]
        legacy_columns = {
            "end_time": jnp.asarray(
                100.0 + peak_indices[selected] / 128, dtype=jnp.float64),
            "template_id": legacy_ids,
        }
        if selected:
            legacy_columns["time_offset"] = jnp.asarray(
                [templates[i].time_offset for i in selected])
        for key in params.dtype.names:
            values = np.array([params[i][key] for i in selected])
            legacy_columns[key] = (jnp.asarray(values)
                                   if values.dtype.kind in "biufc" else values)
        combined = combine_live_results_jax([result])
        for key, expected in legacy_columns.items():
            if isinstance(expected, jax.Array):
                _assert_numeric_exact(combined[key], expected,
                                      active.jax_device)
            else:
                np.testing.assert_array_equal(combined[key], expected)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("limit", [None, 3])
def test_process_all_preserves_ties_truncation_and_veto_alignment(
        device, limit):
    """Join sparse duration groups and select one column/candidate order."""
    with scheme.JAXScheme(device) as active:
        snrs = np.array([8, 10, 10, 12, 10], np.float32)
        ids = np.arange(5, dtype=np.uint64) + np.uint64(2**63 + 1)
        hashes = np.arange(5, dtype=np.int64) + 2**53 + 1
        durations = np.array([4, 4, 8, 4, 4], np.float32)
        approximants = np.array(["A", "B", "C", "D", "E"], object)
        results, infos = [], []
        # Include an empty group and interleave the two source durations.
        for indices in ([], [0, 1], [2], [3, 4]):
            indices = np.asarray(indices, np.int32)
            results.append({
                "snr": jnp.asarray(snrs[indices]),
                "template_id": ids[indices],
                "template_hash": hashes[indices],
                "template_duration": durations[indices],
                "approximant": approximants[indices],
            })
            infos.append([SimpleNamespace(template_id=ids[i], index=int(i))
                          for i in indices])
        pending = iter([*zip(results, infos), (None, None)])
        control = LiveBatchMatchedFilter.__new__(LiveBatchMatchedFilter)
        control.max_triggers_in_batch = limit
        control._process_batch = lambda: next(pending)
        selected = (np.arange(5) if limit is None
                    else np.array([3, 4, 2]))

        def vetoes(self, result, candidates):
            assert ([candidate.index for candidate in candidates]
                    == list(selected))
            np.testing.assert_array_equal(
                result["template_id"], [candidate.template_id
                                        for candidate in candidates])
            # Stand-in veto values expose a result/candidate permutation.
            result["chisq"] = jnp.asarray(
                [candidate.index + 0.25 for candidate in candidates],
                jnp.float32)
            return result

        control._process_vetoes = MethodType(vetoes, control)
        actual = control.process_all()
        for key, original in (("snr", snrs), ("template_id", ids),
                              ("template_hash", hashes),
                              ("template_duration", durations)):
            expected = jnp.asarray(original[selected])
            _assert_numeric_exact(actual[key], expected, active.jax_device)
        np.testing.assert_array_equal(actual["approximant"],
                                      approximants[selected])
        np.testing.assert_array_equal(actual["chisq"], selected + 0.25)


def _resident_columns():
    """Three groups, including an empty one, with distinct column storage."""
    columns = {
        "snr": np.array([0x80000000, 0, 0x7fc00041], np.uint32)
        .view(np.float32),
        "template_id": np.array([2**63 + 1, 2**64 - 1, 0], np.uint64),
        "phase": np.array([1, 0x80000001, 0, 0, 0x80000000, 0], np.uint32)
        .view(np.complex64),
        "matrix": np.array([[-0.0, 1], [2, 3], [4, 5]], np.float64),
    }
    return {key: [jnp.asarray(value[:0]), jnp.asarray(value[:2]),
                  jnp.asarray(value[2:])]
            for key, value in columns.items()}


def _column_arguments(columns):
    return tuple(value for values in columns.values() for value in values)


def _assert_columns_exact(actual, columns, device):
    assert len(actual) == len(columns)
    for value, parts in zip(actual, columns.values()):
        expected = jnp.concatenate([jnp.asarray(part) for part in parts])
        _assert_numeric_exact(value, expected, device)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("x64", [True, False])
@pytest.mark.parametrize("all_empty", [False, True])
@pytest.mark.filterwarnings("ignore:.*not available.*:UserWarning")
def test_fused_resident_helper_preserves_all_column_bytes(
        device, x64, all_empty):
    """Fusion copies columns without changing precision or their ordering."""
    with scheme.JAXScheme(device) as active, enable_x64(x64):
        columns = _resident_columns()
        if all_empty:
            columns = {key: [value[:0] for value in parts]
                       for key, parts in columns.items()}
        result = module._live_concat_resident_columns(
            *_column_arguments(columns), groups=3)
        _assert_columns_exact(result, columns, active.jax_device)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("x64", [True, False])
@pytest.mark.parametrize("values", [
    case for case in _HOST_COLUMNS
    if all(type(value) is np.ndarray and value.dtype.isnative
           for value in case.values[0])
])
@pytest.mark.filterwarnings("ignore:.*not available.*:UserWarning")
def test_fused_helper_mixed_dtypes_preserve_legacy_bytes(device, x64, values):
    """Fusion must retain the old conversion and promotion operation order."""
    with scheme.JAXScheme(device) as active, enable_x64(x64):
        parts = tuple(jnp.asarray(value) for value in values)
        expected = jnp.concatenate(parts)
        result = module._live_concat_resident_columns(
            *parts, groups=len(parts))
        assert len(result) == 1
        _assert_numeric_exact(result[0], expected, active.jax_device)


_RESIDENT_PROMOTION_COLUMNS = [
    pytest.param([np.array([2**53 + 1, -1], np.int64),
                  np.array([2**63 + 1], np.uint64),
                  np.array([0.25], np.float32)], id="wide-integer-promotion"),
    pytest.param([np.array([1, 0x80000001], np.uint32).view(np.float32),
                  np.array([-0.0], np.float64)], id="subnormal-widening"),
    pytest.param([np.array([1, 0x80000001], np.uint32).view(np.complex64),
                  np.array([0j], np.complex128)], id="complex-widening"),
    pytest.param([np.empty(0, np.float64), np.array([1.25], np.float32),
                  np.empty(0, np.int64)], id="empty-promotion"),
]


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("x64", [True, False])
@pytest.mark.parametrize("mixed_host", [False, True])
@pytest.mark.parametrize("values", _RESIDENT_PROMOTION_COLUMNS)
@pytest.mark.filterwarnings("ignore:.*not available.*:UserWarning")
def test_resident_and_mixed_promotion_matches_legacy_bytes(
        device, x64, mixed_host, values, monkeypatch):
    with scheme.JAXScheme(device) as active, enable_x64(x64):
        parts = [value if mixed_host and i == 1 else jnp.asarray(value)
                 for i, value in enumerate(values)]
        expected = jnp.concatenate([jnp.asarray(part) for part in parts])

        def reject_download(self, *args, **kwargs):
            raise AssertionError("resident promotion downloaded an input")

        with monkeypatch.context() as patch:
            patch.setattr(type(parts[0]), "__array__", reject_download)
            actual = combine_live_results_jax(
                [{"column": part} for part in parts])["column"]
        _assert_numeric_exact(actual, expected, active.jax_device)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("values", [
    [np.array(3), np.array(4)], [np.ones(2), np.ones((1, 2))],
    [np.ones((1, 2)), np.ones((2, 3))],
])
def test_invalid_resident_shapes_keep_legacy_exception_type(device, values):
    with scheme.JAXScheme(device):
        parts = [jnp.asarray(value) for value in values]
        with pytest.raises((TypeError, ValueError)) as legacy_error:
            jnp.concatenate(parts)
        with pytest.raises(type(legacy_error.value)):
            combine_live_results_jax([{"column": part} for part in parts])


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("closed_over", [False, True])
@pytest.mark.parametrize("warm", [False, True])
def test_resident_combination_traces_without_leaking_arrays(
        device, closed_over, warm):
    with scheme.JAXScheme(device) as active:
        cached = module._live_concat_resident_executable
        cached.cache_clear()
        columns = _resident_columns()
        parts = _column_arguments(columns)

        def combine(*values):
            keys = tuple(columns)
            batches = [{key: values[column * 3 + row]
                        for column, key in enumerate(keys)}
                       for row in range(3)]
            combined = combine_live_results_jax(batches)
            return tuple(combined[key] for key in keys)

        try:
            if warm:
                _assert_columns_exact(combine(*parts), columns,
                                      active.jax_device)
            actual = (jax.jit(lambda: combine(*parts))() if closed_over
                      else jax.jit(combine)(*parts))
            _assert_columns_exact(actual, columns, active.jax_device)
            _assert_columns_exact(combine(*parts), columns, active.jax_device)
        finally:
            cached.cache_clear()


def test_cpu_resident_columns_bypass_gpu_compile_cache(monkeypatch):
    with scheme.JAXScheme("cpu") as active:
        columns = _resident_columns()

        def reject_gpu_compile(*args, **kwargs):
            raise AssertionError("CPU combination used CUDA compilation")

        monkeypatch.setattr(module, "_live_concat_resident_executable",
                            reject_gpu_compile)
        results = []
        for row in range(3):
            keys = list(columns) if row != 1 else list(reversed(columns))
            result = {key: columns[key][row] for key in keys}
            if row:
                result["ignored_later_key"] = jnp.array([99])
            results.append(result)
        combined = combine_live_results_jax(results)
        assert list(combined) == list(results[0])
        _assert_columns_exact(tuple(combined.values()), columns,
                              active.jax_device)


@pytest.mark.parametrize("device", _devices())
def test_resident_executable_reuses_geometry_and_reads_new_values(device):
    with scheme.JAXScheme(device) as active:
        cached = module._live_concat_resident_executable
        cached.cache_clear()
        try:
            parts = (jnp.array([1, 2], jnp.float32),
                     jnp.array([3], jnp.float32))
            shapes = tuple((tuple(value.shape), np.dtype(value.dtype).str)
                           for value in parts)
            first = cached(shapes, 2, active.jax_device, True)
            _assert_numeric_exact(first(*parts)[0], jnp.concatenate(parts),
                                  active.jax_device)
            changed = (jnp.array([-1, -2], jnp.float32),
                       jnp.array([-3], jnp.float32))
            repeated = cached(shapes, 2, active.jax_device, True)
            assert repeated is first
            _assert_numeric_exact(repeated(*changed)[0],
                                  jnp.concatenate(changed), active.jax_device)
            assert cached.cache_info().misses == 1
            assert cached.cache_info().hits == 1
            resized = (parts[0], jnp.array([3, 4], jnp.float32))
            new_shapes = tuple((tuple(value.shape), np.dtype(value.dtype).str)
                               for value in resized)
            new = cached(new_shapes, 2, active.jax_device, True)
            assert new is not first
            _assert_numeric_exact(new(*resized)[0], jnp.concatenate(resized),
                                  active.jax_device)
            assert cached.cache_info().misses == 2
            split = cached(shapes, 1, active.jax_device, True)
            assert split is not first
            for actual, expected in zip(split(*parts), parts):
                _assert_numeric_exact(actual, expected, active.jax_device)
            assert cached.cache_info().misses == 3
            wide = tuple(value.astype(jnp.float64) for value in parts)
            wide_shapes = tuple((tuple(value.shape), np.dtype(value.dtype).str)
                                for value in wide)
            changed_dtype = cached(wide_shapes, 2, active.jax_device, True)
            _assert_numeric_exact(changed_dtype(*wide)[0],
                                  jnp.concatenate(wide), active.jax_device)
            assert cached.cache_info().misses == 4
        finally:
            cached.cache_clear()


@pytest.mark.parametrize("device", _devices())
def test_resident_executable_cache_separates_x64_modes(device):
    with scheme.JAXScheme(device) as active:
        cached = module._live_concat_resident_executable
        cached.cache_clear()
        try:
            handles = []
            for x64 in (True, False, True):
                with enable_x64(x64):
                    parts = (jnp.array([1, 2], jnp.float32),
                             jnp.array([3], jnp.float32))
                    shapes = tuple((tuple(value.shape),
                                    np.dtype(value.dtype).str)
                                   for value in parts)
                    execute = cached(shapes, 2, active.jax_device, x64)
                    handles.append(execute)
                    _assert_numeric_exact(execute(*parts)[0],
                                          jnp.concatenate(parts),
                                          active.jax_device)
            assert handles[0] is handles[2]
            assert handles[0] is not handles[1]
            assert cached.cache_info().misses == 2
            assert cached.cache_info().hits == 1
        finally:
            cached.cache_clear()


@pytest.mark.skipif("cuda:0" not in _devices(), reason="requires CUDA")
def test_gpu_combines_resident_columns_in_one_executable_call(monkeypatch):
    """Count executable invocation, not Python concat calls while tracing."""
    with scheme.JAXScheme("cuda:0") as active:
        columns = _resident_columns()
        ids = [np.empty(0, np.uint64), np.array([11, 12], np.uint64),
               np.array([13], np.uint64)]
        names = [np.empty(0, object), np.array(["A", "B"], object),
                 np.array(["C"], object)]
        results = []
        for row in range(3):
            keys = list(columns) if row != 1 else list(reversed(columns))
            result = {key: columns[key][row] for key in keys}
            result["host_id"] = ids[row]
            result["approximant"] = names[row]
            if row:
                result["ignored_later_key"] = np.array([99])
            results.append(result)
        lookup = module._live_concat_resident_executable
        original_asarray = jnp.asarray
        calls, uploads = [], []

        def executable(*args, **kwargs):
            execute = lookup(*args, **kwargs)

            def invoke(*values):
                calls.append(len(values))
                return execute(*values)
            return invoke

        def upload(value, *args, **kwargs):
            if type(value) is np.ndarray:
                uploads.append(value)
            return original_asarray(value, *args, **kwargs)

        def reject_download(self, *args, **kwargs):
            raise AssertionError("fused combination downloaded an input")

        with monkeypatch.context() as patch:
            patch.setattr(module, "_live_concat_resident_executable",
                          executable)
            patch.setattr(jnp, "asarray", upload)
            patch.setattr(type(columns["snr"][0]), "__array__",
                          reject_download)
            actual = combine_live_results_jax(results)
        # The three vector columns fuse; the matrix keeps legacy concatenation.
        assert calls == [3 * 2]
        assert len(uploads) == 1
        assert list(actual) == list(results[0])
        _assert_columns_exact(tuple(actual[key] for key in columns), columns,
                              active.jax_device)
        expected_ids = jnp.asarray(np.concatenate(ids))
        _assert_numeric_exact(actual["host_id"], expected_ids,
                              active.jax_device)
        np.testing.assert_array_equal(actual["approximant"],
                                      np.concatenate(names))


@pytest.mark.skipif("cuda:0" not in _devices(), reason="requires CUDA")
@pytest.mark.parametrize("x64", [True, False])
def test_gpu_fuses_heterogeneous_columns_with_interleaved_metadata(
        x64, monkeypatch):
    """Two mixed columns exercise promotion inside the public fused path."""
    with scheme.JAXScheme("cuda:0") as active, enable_x64(x64):
        columns = {
            "integer_promotion": [
                jnp.asarray(np.array([2**53 + 1, -1], np.int64)),
                jnp.asarray(np.array([2**63 + 1], np.uint64)),
                jnp.asarray(np.array([0.25], np.float32)),
            ],
            "subnormal_promotion": [
                np.array([1, 0x80000001], np.uint32).view(np.float32),
                jnp.asarray(np.array([-0.0], np.float64)),
                np.array([0.0], np.float32),
            ],
        }
        expected = {key: jnp.concatenate([jnp.asarray(part)
                                         for part in parts])
                    for key, parts in columns.items()}
        ids = [np.array([11, 12], np.uint64), np.array([13], np.uint64),
               np.array([14], np.uint64)]
        names = [np.array(["A", "B"], object), np.array(["C"], object),
                 np.array(["D"], object)]
        results = [
            {"integer_promotion": columns["integer_promotion"][row],
             "host_id": ids[row], "approximant": names[row],
             "subnormal_promotion": columns["subnormal_promotion"][row]}
            for row in range(3)
        ]
        results[1] = dict(reversed(list(results[1].items())))
        lookup = module._live_concat_resident_executable
        calls = []

        def executable(*args, **kwargs):
            execute = lookup(*args, **kwargs)

            def invoke(*values):
                calls.append(len(values))
                return execute(*values)
            return invoke

        def reject_download(self, *args, **kwargs):
            raise AssertionError("mixed fusion downloaded a resident input")

        with monkeypatch.context() as patch:
            patch.setattr(module, "_live_concat_resident_executable",
                          executable)
            patch.setattr(type(columns["integer_promotion"][0]), "__array__",
                          reject_download)
            actual = combine_live_results_jax(results)
        assert calls == [len(columns) * 3]
        assert list(actual) == list(results[0])
        for key, value in expected.items():
            _assert_numeric_exact(actual[key], value, active.jax_device)
        _assert_numeric_exact(actual["host_id"],
                              jnp.asarray(np.concatenate(ids)),
                              active.jax_device)
        np.testing.assert_array_equal(actual["approximant"],
                                      np.concatenate(names))


def _assert_gpu_placement_fallback(columns, device, monkeypatch):
    expected = {key: jnp.concatenate([jnp.asarray(part) for part in parts])
                for key, parts in columns.items()}
    results = [{key: parts[row] for key, parts in columns.items()}
               for row in range(2)]

    def reject_retained_executable(*args, **kwargs):
        raise AssertionError("nondefault placement used retained execution")

    with monkeypatch.context() as patch:
        patch.setattr(module, "_live_concat_resident_executable",
                      reject_retained_executable)
        actual = combine_live_results_jax(results)
    assert list(actual) == list(results[0])
    for key, value in expected.items():
        _assert_numeric_exact(actual[key], value, device)


@pytest.mark.skipif("cuda:0" not in _devices(), reason="requires CUDA")
def test_gpu_pinned_host_vectors_keep_legacy_placement_fallback(monkeypatch):
    """Pinned host columns differ from the compiled device-memory buffers."""
    from jax.sharding import SingleDeviceSharding

    with scheme.JAXScheme("cuda:0") as active:
        sharding = SingleDeviceSharding(active.jax_device,
                                        memory_kind="pinned_host")
        values = {
            "snr": [np.array([-0.0, 1.25], np.float32),
                    np.array([2.5], np.float32)],
            "template_id": [np.array([2**63 + 1, 0], np.uint64),
                            np.array([2**64 - 1], np.uint64)],
        }
        columns = {key: [jax.device_put(part, sharding) for part in parts]
                   for key, parts in values.items()}
        for parts in columns.values():
            assert all(not part.weak_type and part.ndim == 1
                       and part.sharding.memory_kind == "pinned_host"
                       for part in parts)
        _assert_gpu_placement_fallback(columns, active.jax_device, monkeypatch)


@pytest.mark.skipif("cuda:0" not in _devices(), reason="requires CUDA")
def test_gpu_custom_matrix_layout_keeps_legacy_fallback(monkeypatch):
    """Column-major matrices retain the legacy concatenate layout contract."""
    layout = pytest.importorskip("jax.experimental.layout")
    Layout = getattr(layout, "Layout", None)
    Format = getattr(layout, "Format", None)
    if Layout is None or Format is None:
        pytest.skip("JAX does not expose Layout/Format")
    from jax.sharding import SingleDeviceSharding

    with scheme.JAXScheme("cuda:0") as active:
        placement = Format(Layout((1, 0)),
                           SingleDeviceSharding(active.jax_device))
        values = {
            "snr": [np.array([[-0.0, 1.25], [2.5, 3.75]], np.float32),
                    np.array([[4.0, 5.0]], np.float32)],
            "template_id": [np.array([[2**63 + 1, 0], [1, 2]], np.uint64),
                            np.array([[2**64 - 1, 3]], np.uint64)],
        }
        columns = {key: [jax.device_put(part, placement) for part in parts]
                   for key, parts in values.items()}
        for parts in columns.values():
            assert all(not part.weak_type and part.ndim == 2
                       and part.format.layout.major_to_minor == (1, 0)
                       for part in parts)
        _assert_gpu_placement_fallback(columns, active.jax_device, monkeypatch)


@pytest.mark.skipif("cuda:0" not in _devices(), reason="requires CUDA")
def test_gpu_warmed_standard_promotion_preserves_strict_errors(monkeypatch):
    """A retained standard-promotion handle must not bypass strict errors."""
    with scheme.JAXScheme("cuda:0") as active:
        columns = {
            "snr": [jnp.array([1], jnp.int32), jnp.array([2], jnp.float32)],
            "phase": [jnp.array([0.5], jnp.float32),
                      jnp.array([-0.5], jnp.float64)],
        }
        results = [{key: parts[row] for key, parts in columns.items()}
                   for row in range(2)]
        cached = module._live_concat_resident_executable
        cached.cache_clear()
        try:
            with jax.numpy_dtype_promotion("standard"):
                expected = {key: jnp.concatenate(parts)
                            for key, parts in columns.items()}
                actual = combine_live_results_jax(results)
                for key, value in expected.items():
                    _assert_numeric_exact(actual[key], value,
                                          active.jax_device)
                assert cached.cache_info().misses == 1

            def reject_retained_executable(*args, **kwargs):
                raise AssertionError("strict promotion used retained handle")

            with monkeypatch.context() as patch:
                patch.setattr(module, "_live_concat_resident_executable",
                              reject_retained_executable)
                with jax.numpy_dtype_promotion("strict"):
                    with pytest.raises(jax.dtypes.TypePromotionError):
                        jnp.concatenate([jnp.asarray(part)
                                         for part in columns["snr"]])
                    with pytest.raises(jax.dtypes.TypePromotionError):
                        combine_live_results_jax(results)
        finally:
            cached.cache_clear()


_EMPTY_PROMOTION_COLUMNS = [
    pytest.param([
        np.empty(0, np.float64),
        np.array([1, 0x80000001, 0x80000000], np.uint32).view(np.float32),
        np.empty(0, np.float32), np.empty(0, np.float64),
    ], id="empty-float64-widens-subnormals"),
    pytest.param([
        np.empty(0, np.complex128),
        np.array([1, 0x80000001], np.uint32).view(np.complex64),
        np.empty(0, np.complex128),
    ], id="empty-complex128-widens-subnormals"),
    pytest.param([
        np.empty(0, np.uint64), np.array([2**53 + 1, -1], np.int64),
        np.empty(0, np.uint64),
    ], id="empty-unsigned-promotes-signed"),
    pytest.param([
        np.empty(0, np.int64), np.array([2**63 + 1, 2**64 - 1], np.uint64),
        np.empty(0, np.int64),
    ], id="empty-signed-promotes-unsigned"),
    pytest.param([
        np.empty(0, np.float32), np.empty(0, np.float64),
        np.empty(0, np.int64), np.empty(0, np.float32),
        np.empty(0, np.float64),
    ], id="all-empty-mixed-promotion"),
    pytest.param([np.empty(0, np.float64)] * 4,
                 id="all-empty-uniform"),
]


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("x64", [True, False])
@pytest.mark.parametrize("values", _EMPTY_PROMOTION_COLUMNS)
def test_pruned_empty_columns_preserve_promotion_and_tuple_group_bytes(
        device, x64, values):
    """Empty inputs contribute dtype even when their geometry is redundant."""
    with scheme.JAXScheme(device) as active, enable_x64(x64):
        original = tuple(jnp.asarray(value) for value in values)
        parts = module._live_prune_empty_column(original)
        nonempty = [value for value in original if value.size]
        assert [id(value) for value in parts if value.size] == [
            id(value) for value in nonempty]
        assert len(parts) < len(original)
        assert parts
        if nonempty:
            assert {value.dtype for value in parts if not value.size} == (
                {value.dtype for value in original if not value.size}
                - {value.dtype for value in nonempty})
        else:
            assert len(parts) == len({value.dtype for value in original})
        second = tuple(jnp.asarray(
            [-0.0] if row == 0 else [0.0, 2.0]
            if row == len(original) - 1 else [], jnp.float32)
            for row in range(len(original)))
        second_parts = module._live_prune_empty_column(second)
        actual = module._live_concat_resident_columns(
            *parts, *second_parts, groups=(len(parts), len(second_parts)))
        _assert_numeric_exact(actual[0], jnp.concatenate(original),
                              active.jax_device)
        _assert_numeric_exact(actual[1], jnp.concatenate(second),
                              active.jax_device)
        public = combine_live_results_jax([
            {"first": original[row], "second": second[row]}
            for row in range(len(original))])
        for key, values in (("first", original), ("second", second)):
            _assert_numeric_exact(public[key], jnp.concatenate(values),
                                  active.jax_device)


@pytest.mark.skipif("cuda:0" not in _devices(), reason="requires CUDA")
def test_gpu_empty_group_normalization_reuses_handle_and_reads_fresh_values(
        monkeypatch):
    """Empty position/count changes reuse equivalent join geometry."""
    with scheme.JAXScheme("cuda:0") as active, enable_x64(True):
        cached = module._live_concat_resident_executable
        cached.cache_clear()
        handles, argument_counts = [], []

        def executable(*args, **kwargs):
            execute = cached(*args, **kwargs)
            handles.append(execute)

            def invoke(*values):
                argument_counts.append(len(values))
                return execute(*values)
            return invoke

        try:
            geometries = ((0, 0, 2, 0, 1, 0), (0, 2, 0, 0, 0, 1),
                          (2, 0, 1, 0))
            for iteration, sizes in enumerate(geometries):
                snr = jnp.asarray([-0.0, 1.0 + iteration, -2.0 - iteration],
                                  jnp.float32)
                ids = jnp.asarray([2**63 + iteration, 2**64 - 1, iteration],
                                  jnp.uint64)
                results, offset = [], 0
                for count in sizes:
                    results.append({"snr": snr[offset:offset + count],
                                    "approximant": np.full(count, "A", object),
                                    "template_id": ids[offset:offset + count]})
                    offset += count
                expected = {key: jnp.concatenate([batch[key]
                                                 for batch in results])
                            for key in ("snr", "template_id")}

                def reject_download(self, *args, **kwargs):
                    raise AssertionError("empty normalization downloaded data")

                with monkeypatch.context() as patch:
                    patch.setattr(module, "_live_concat_resident_executable",
                                  executable)
                    patch.setattr(type(snr), "__array__", reject_download)
                    actual = combine_live_results_jax(results)
                assert list(actual) == list(results[0])
                for key, reference in expected.items():
                    _assert_numeric_exact(actual[key], reference,
                                          active.jax_device)
                np.testing.assert_array_equal(actual["approximant"],
                                              ["A"] * 3)
            assert argument_counts == [4, 4, 4]
            assert all(handle is handles[0] for handle in handles)
            assert cached.cache_info().misses == 1
            assert cached.cache_info().hits == 2
        finally:
            cached.cache_clear()


@pytest.mark.parametrize("device", _devices())
def test_resident_executable_cache_separates_tuple_column_boundaries(device):
    with scheme.JAXScheme(device) as active:
        cached = module._live_concat_resident_executable
        cached.cache_clear()
        try:
            values = tuple(jnp.asarray([value], jnp.float32)
                           for value in (-0.0, 1.0, -2.0))
            shapes = tuple((tuple(value.shape), np.dtype(value.dtype).str)
                           for value in values)
            handles = []
            for groups in ((1, 2), (2, 1), (1, 2)):
                execute = cached(shapes, groups, active.jax_device, True)
                handles.append(execute)
                actual = execute(*values)
                split = groups[0]
                expected = (jnp.concatenate(values[:split]),
                            jnp.concatenate(values[split:]))
                for value, reference in zip(actual, expected):
                    _assert_numeric_exact(value, reference, active.jax_device)
            assert handles[0] is handles[2]
            assert handles[0] is not handles[1]
            assert cached.cache_info().misses == 2
            assert cached.cache_info().hits == 1
        finally:
            cached.cache_clear()


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("x64", [True, False])
def test_combine_all_empty_resident_and_metadata_columns(device, x64):
    with scheme.JAXScheme(device) as active, enable_x64(x64):
        columns = {
            "snr": [jnp.empty(0, dtype) for dtype in
                    (jnp.float32, jnp.float64, jnp.float32)],
            "template_id": [jnp.empty(0, dtype) for dtype in
                            (jnp.int64, jnp.uint64, jnp.int64)],
            "approximant": [np.empty(0, np.float64) for _ in range(3)],
            "names": [np.empty(0, object) for _ in range(3)],
        }
        actual = combine_live_results_jax([
            {key: parts[row] for key, parts in columns.items()}
            for row in range(3)])
        assert list(actual) == list(columns)
        for key in ("snr", "template_id", "approximant"):
            _assert_numeric_exact(actual[key], jnp.concatenate(columns[key]),
                                  active.jax_device)
        assert isinstance(actual["names"], np.ndarray)
        assert actual["names"].shape == (0,)
        assert actual["names"].dtype == np.dtype(object)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("invalid", [
    [np.empty((0, 2)), np.ones(1)],
    [np.empty(0), np.array(1.0)],
    [np.empty((0, 2)), np.ones((1, 3))],
])
@pytest.mark.parametrize("missing_first", [False, True])
def test_empty_normalization_preserves_first_key_errors(
        device, invalid, missing_first, monkeypatch):
    """Malformed empty inputs cannot disappear or defer earlier errors."""
    with scheme.JAXScheme(device):
        valid = [jnp.asarray([1.0]), jnp.empty(0)]
        keys = (["missing", "invalid"] if missing_first
                else ["invalid", "missing"])
        results = []
        for row in range(2):
            result = {"snr": valid[row], "phase": valid[row]}
            for key in keys:
                if key == "invalid":
                    result[key] = jnp.asarray(invalid[row])
                elif row == 0:
                    result[key] = valid[row]
            results.append(result)

        def legacy():
            return {key: jnp.concatenate([jnp.asarray(batch[key])
                                          for batch in results])
                    for key in results[0]}

        with pytest.raises((KeyError, TypeError, ValueError)) as expected:
            legacy()

        def reject_retained_execution(*args, **kwargs):
            raise AssertionError("later error dispatched retained results")

        monkeypatch.setattr(module, "_live_concat_resident_executable",
                            reject_retained_execution)
        with pytest.raises(type(expected.value)):
            combine_live_results_jax(results)


def _finished_partial_metadata_batches(sizes, iteration):
    """Use the real candidate assembler, including empty metadata dtypes."""
    params = np.empty(3, dtype=[
        ("template_hash", np.int64), ("template_duration", np.float32),
        ("float_bits", np.float32), ("complex_bits", np.complex64),
        ("approximant", object),
    ])
    params["template_hash"] = np.array(
        [2**53 + 1, -2**53 - 1, 2**62 + 3], np.int64) + iteration
    params["template_duration"] = [4 + iteration, 8, 4]
    params["float_bits"] = np.array(
        [1, 0x80000001, 0x80000000], np.uint32).view(np.float32)
    params["complex_bits"] = np.array(
        [1, 0x80000001, 0x80000000, 0, 0x7fc00041, 0], np.uint32
    ).view(np.complex64)
    params["approximant"] = ["TaylorF2", "IMRPhenomD", "TaylorF2"]
    control = SimpleNamespace(data=SimpleNamespace(
        start_time=100.0 + iteration, sample_rate=128))
    batches, offset = [], 0
    for size in sizes:
        count = max(size, 1)
        templates = [SimpleNamespace(
            params=params[(offset + row) % 3], id=11 + iteration + offset + row
        ) for row in range(count)]
        peaks = jnp.asarray(np.arange(count) + 3 + iteration + 4j,
                            jnp.complex64)
        norms = jnp.ones(count, jnp.float64)
        prepared = SimpleNamespace(
            templates=templates, stilde=SimpleNamespace(psd=object()),
            template_matrix=jnp.zeros((count, 1), jnp.complex64),
            native_norms=jnp.full(count, 4, jnp.float64), norms=norms,
            valid_start=0, peak_values=peaks, scaled_peaks=peaks * norms)
        selection = (np.arange(count, dtype=np.uint32),
                     np.arange(count) < size, np.zeros(count, bool))
        result, _ = module._live_finish_batch_jax(control, prepared, selection)
        batches.append(result)
        offset += size
    return batches


def _legacy_numeric_result_columns(batches):
    return {key: jnp.concatenate([jnp.asarray(batch[key]) for batch in batches])
            for key in batches[0]
            if all(batch[key].dtype.kind in "biufc" for batch in batches)}


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("x64", [True, False])
@pytest.mark.filterwarnings("ignore:.*not available.*:UserWarning")
def test_partial_empty_finished_metadata_preserves_legacy_bytes(device, x64):
    """Actual empty groups widen hashes and floating metadata as before."""
    with scheme.JAXScheme(device) as active, enable_x64(x64):
        for iteration, sizes in enumerate(((0, 2, 1, 0), (2, 0, 0, 1))):
            batches = _finished_partial_metadata_batches(sizes, iteration)
            for key in ("template_hash", "template_duration"):
                assert batches[sizes.index(0)][key].dtype == np.float64
            expected = _legacy_numeric_result_columns(batches)
            actual = combine_live_results_jax(batches)
            assert list(actual) == list(batches[0])
            for key, value in expected.items():
                _assert_numeric_exact(actual[key], value, active.jax_device)
            np.testing.assert_array_equal(
                actual["approximant"], np.concatenate(
                    [batch["approximant"] for batch in batches]))


@pytest.mark.skipif("cuda:0" not in _devices(), reason="requires CUDA")
@pytest.mark.parametrize("x64", [True, False])
@pytest.mark.filterwarnings("ignore:.*not available.*:UserWarning")
def test_gpu_partial_empty_metadata_uploads_once_and_reuses_fresh_join(
        x64, monkeypatch):
    """Empty metadata contributes dtype without repeating nonempty uploads."""
    with scheme.JAXScheme("cuda:0") as active, enable_x64(x64):
        cached = module._live_concat_resident_executable
        cached.cache_clear()
        handles = []

        def executable(*args, **kwargs):
            handle = cached(*args, **kwargs)
            handles.append(handle)
            return handle

        try:
            for iteration, sizes in enumerate(((0, 2, 1, 0), (2, 0, 0, 1))):
                batches = _finished_partial_metadata_batches(sizes, iteration)
                expected = _legacy_numeric_result_columns(batches)
                resident_ids = {id(value) for batch in batches
                                for value in batch.values()
                                if isinstance(value, jax.Array)}
                host_keys = [key for key in expected
                             if all(type(batch[key]) is np.ndarray
                                    for batch in batches)]
                original_asarray, uploads = jnp.asarray, []

                def upload(value, *args, **kwargs):
                    assert id(value) not in resident_ids, (
                        "canonical resident input was reconverted")
                    if type(value) is np.ndarray and value.size:
                        uploads.append(value)
                    return original_asarray(value, *args, **kwargs)

                def reject_download(self, *args, **kwargs):
                    raise AssertionError("result preparation downloaded input")

                with monkeypatch.context() as patch:
                    patch.setattr(jnp, "asarray", upload)
                    patch.setattr(module, "_live_concat_resident_executable",
                                  executable)
                    patch.setattr(type(batches[0]["snr"]), "__array__",
                                  reject_download)
                    actual = combine_live_results_jax(batches)
                assert len(host_keys) >= 2
                assert len(uploads) == len(host_keys)
                assert all(value.shape == (3,) for value in uploads)
                for key, value in expected.items():
                    _assert_numeric_exact(actual[key], value, active.jax_device)
            assert len(handles) == 2
            assert handles[0] is handles[1]
            assert cached.cache_info().misses == 1
            assert cached.cache_info().hits == 1
        finally:
            cached.cache_clear()


@pytest.mark.parametrize("device", _devices())
def test_existing_wide_resident_arrays_keep_cross_x64_legacy_conversion(
        device, monkeypatch):
    """Existing wide arrays and new host uploads have different contracts."""
    with scheme.JAXScheme(device) as active:
        with enable_x64(True):
            wide = {
                "integer": jnp.asarray([2**53 + 1, -1], jnp.int64),
                "float": jnp.asarray([1 + 2**-30, -0.0], jnp.float64),
                "complex": jnp.asarray([1 + 2**-30 + 1j, -0j], jnp.complex128),
            }
        host = {key: np.asarray(value).copy() for key, value in wide.items()}
        with enable_x64(False):
            expected = {key: jnp.concatenate(
                [jnp.asarray(value), jnp.asarray(host[key]),
                 jnp.asarray(value[:0])]) for key, value in wide.items()}
            batches = [wide, host,
                       {key: value[:0] for key, value in wide.items()}]
            original_asarray, converted = jnp.asarray, []

            def convert(value, *args, **kwargs):
                converted.append(id(value))
                return original_asarray(value, *args, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(jnp, "asarray", convert)
                actual = combine_live_results_jax(batches)
            assert all(id(value) in converted for value in wide.values())
            for key, value in expected.items():
                _assert_numeric_exact(actual[key], value, active.jax_device)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("promotion", ["standard", "strict"])
def test_weak_resident_vectors_keep_native_promotion_fallback(
        device, promotion, monkeypatch):
    """A weak float64 vector must not force a strong float32 column wider."""
    with scheme.JAXScheme(device) as active, enable_x64(True):
        weak = jnp.broadcast_to(jnp.asarray(2.718281828), (2,))
        assert weak.weak_type and weak.dtype == jnp.float64
        strong = jnp.asarray([3.1415925], jnp.float32)
        batches = [{"first": weak, "second": strong},
                   {"first": strong, "second": weak}]
        with jax.numpy_dtype_promotion(promotion):
            expected = _legacy_numeric_result_columns(batches)
            original_asarray, converted = jnp.asarray, []

            def convert(value, *args, **kwargs):
                converted.append(id(value))
                return original_asarray(value, *args, **kwargs)

            def reject_retained(*args, **kwargs):
                raise AssertionError("weak promotion used retained execution")

            with monkeypatch.context() as patch:
                patch.setattr(jnp, "asarray", convert)
                patch.setattr(module, "_live_concat_resident_executable",
                              reject_retained)
                actual = combine_live_results_jax(batches)
            assert id(weak) in converted
            for key, value in expected.items():
                assert value.dtype == jnp.float32
                _assert_numeric_exact(actual[key], value, active.jax_device)
