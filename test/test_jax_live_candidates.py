# Copyright (C) 2026 The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.

"""Bulk Live candidate publication contracts against the retained assembler.

These fixtures isolate post-peak assembly, not complete-search qualification.
CUDA-only cases exercise real resident arrays and skip without a GPU.
"""

from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
enable_x64 = getattr(jax, "enable_x64", None)
if enable_x64 is None:
    from jax.experimental import enable_x64

from pycbc import scheme  # noqa: E402
from pycbc.filter import matchedfilter_jax as module  # noqa: E402


def _gpu_devices():
    try:
        if jax.devices("gpu"):
            return ["cuda:0"]
    except RuntimeError:
        pass
    return [pytest.param("cuda:0", marks=pytest.mark.skip(
        reason="bulk candidate publication requires CUDA"))]


def _inputs(selected=((0, 2), (1,), (0, 1)), iteration=0, wide_ids=False):
    """Distinct original-grid groups with exact metadata and sparse peaks."""
    params = np.empty(8, dtype=[
        ("template_hash", np.int64), ("template_duration", np.float32),
        ("float_bits", np.float32), ("complex_bits", np.complex64),
        ("signed_zero", np.float64), ("flag", np.bool_),
        ("approximant", object),
    ])
    params["template_hash"] = np.array(
        [2**53 + 1, -2**53 - 1, 2**62 + 3, 17, -19, 23, 29, -31],
        np.int64) + iteration
    params["template_duration"] = [4, 4, 4, 8, 8, 16, 16, 16]
    params["float_bits"] = np.array(
        [1, 0x80000001, 0x80000000, 0, 0x7fc00041, 0x00800000, 0, 1],
        np.uint32).view(np.float32)
    params["complex_bits"] = np.array(
        [1, 0x80000001, 0x80000000, 0, 0x7fc00041, 0, 0, 0,
         0x3f800000, 0, 0, 0x80000000, 1, 0, 0, 1],
        np.uint32).view(np.complex64)
    params["signed_zero"] = [-0.0, 0.0, -0.0, 0.0, 1.25, -0.0, 0, -2.5]
    params["flag"] = [True, False, True, True, False, False, True, False]
    params["approximant"] = ["TaylorF2", "IMRPhenomD", "A", "B",
                             "C", "D", "E", "F"]
    prepared, selections, groups = [], [], []
    offset = 0
    for group_index, (size, accepted) in enumerate(zip((3, 2, 3), selected)):
        templates = [SimpleNamespace(
            params=params[offset + row],
            id=(2**63 + offset + row + 1 if wide_ids else 101 + offset + row),
        ) for row in range(size)]
        groups.append(templates)
        peaks = jnp.asarray(
            np.array([3 + 4j, -3 + 4j, -0.0 - 2j], np.complex64)[:size]
            + np.complex64(iteration), jnp.complex64)
        norms = jnp.asarray([0.25, 0.5, 2.0][:size], jnp.float64)
        prepared.append(module._LivePreparedBatch(
            group_index, templates, SimpleNamespace(psd=object()),
            jnp.zeros((size, 1), jnp.complex64),
            jnp.asarray([1.25, 2.5, 5.0][:size], jnp.float64), norms,
            4096 * (group_index + 1), peaks, peaks * norms, None))
        mask = np.zeros(size, bool)
        mask[list(accepted)] = True
        selections.append((
            np.array([1, 127, 1023][:size], np.uint32) + iteration,
            mask, np.zeros(size, bool)))
        offset += size
    control = SimpleNamespace(
        tgroups=groups, power_chisq=SimpleNamespace(),
        data=SimpleNamespace(start_time=1234567890.123456 + iteration,
                             sample_rate=2048),
    )
    return control, prepared, selections


def _legacy_finish(control, prepared, selections):
    """The existing per-group API is the independent publication oracle."""
    results, veto_info = [], []
    for batch, selection in zip(prepared, selections):
        result, veto = module._live_finish_batch_jax(control, batch, selection)
        if result is False:
            return False, []
        results.append(result)
        veto_info.extend(veto)
    return module.combine_live_results_jax(results), veto_info


def _assert_exact(actual, expected, device):
    result, veto_info = actual
    reference, reference_veto = expected
    assert list(result) == list(reference)
    for key, value in reference.items():
        output = result[key]
        assert type(output) is type(value), key
        assert output.shape == value.shape, key
        assert output.dtype == value.dtype, key
        if isinstance(value, jax.Array):
            assert output.device == device, key
            assert (np.asarray(output).tobytes()
                    == np.asarray(value).tobytes()), key
        else:
            np.testing.assert_array_equal(output, value, err_msg=key)
    assert len(veto_info) == len(reference_veto)
    for info, old in zip(veto_info, reference_veto):
        assert info.row == old.row
        assert info.snrs is old.snrs
        assert info.norms is old.norms
        assert info.metadata[0] == old.metadata[0]
        for index in (1, 2, 3):
            assert info.metadata[index] is old.metadata[index]
        assert info.metadata[4] == old.metadata[4]


@pytest.mark.parametrize("device", _gpu_devices())
@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("selected", [
    ((0, 1, 2), (0, 1), (0, 1, 2)),
    ((0, 2), (1,), (0, 1)),
    ((), (0, 1), (2,)),
    ((2,), (), ()),
])
@pytest.mark.filterwarnings("ignore:.*not available.*:UserWarning")
def test_bulk_candidate_columns_match_existing_publication(
        device, x64, selected):
    """Preserve empty promotion, group order, payload bits and end times."""
    with scheme.JAXScheme(device) as active, enable_x64(x64):
        control, prepared, selections = _inputs(selected)
        actual = module._live_finish_batches_jax(control, prepared, selections)
        assert actual is not None
        expected = _legacy_finish(control, prepared, selections)
        _assert_exact(actual, expected, active.jax_device)
        assert len(actual[1]) == sum(len(rows) for rows in selected)
        # Metadata dictionaries retain the original lazy creation behavior.
        for batch, rows in zip(prepared, selected):
            assert [hasattr(template, "dict_params")
                    for template in batch.templates] == [
                        row in rows for row in range(len(batch.templates))]


@pytest.mark.parametrize("device", _gpu_devices())
def test_bulk_candidates_preserve_per_group_complex_precision(device):
    """Promotion follows each group's phase and magnitude calculation."""
    with scheme.JAXScheme(device) as active, enable_x64(True):
        control, prepared, selections = _inputs()
        for group_index, batch in enumerate(prepared):
            real = jnp.float32 if group_index == 1 else jnp.float64
            complex_dtype = (jnp.complex64 if group_index == 1
                             else jnp.complex128)
            peaks = batch.peak_values.astype(complex_dtype) + complex_dtype(
                0.01 + 0.03j)
            norms = jnp.asarray(
                [0.17, 0.27, 0.43][:len(batch.templates)], real)
            prepared[group_index] = batch._replace(
                peak_values=peaks, norms=norms, scaled_peaks=peaks * norms)
        actual = module._live_finish_batches_jax(control, prepared, selections)
        assert actual is not None
        _assert_exact(actual, _legacy_finish(control, prepared, selections),
                      active.jax_device)


@pytest.mark.parametrize("device", _gpu_devices())
def test_bulk_candidates_preserve_wide_ids_and_cache_mutations(device):
    """Cache stable bank columns while honoring the authoritative snapshots."""
    with scheme.JAXScheme(device) as active, enable_x64(True):
        control, prepared, selections = _inputs(wide_ids=True)
        actual = module._live_finish_batches_jax(control, prepared, selections)
        assert actual is not None
        _assert_exact(actual, _legacy_finish(control, prepared, selections),
                      active.jax_device)
        cached = control._jax_live_candidate_metadata
        repeat = module._live_finish_batches_jax(control, prepared, selections)
        assert control._jax_live_candidate_metadata is cached
        _assert_exact(repeat, actual, active.jax_device)

        template = prepared[0].templates[0]
        template.params["template_hash"] = 41
        # Once dict_params exists, raw record mutation is deliberately stale.
        unchanged = module._live_finish_batches_jax(
            control, prepared, selections)
        _assert_exact(unchanged, _legacy_finish(control, prepared, selections),
                      active.jax_device)
        assert control._jax_live_candidate_metadata is cached

        for mutation in ("dict-value", "dict-replacement", "id", "template"):
            if mutation == "dict-value":
                template.dict_params["template_hash"] = np.int64(2**53 + 7)
            elif mutation == "dict-replacement":
                template.dict_params = dict(template.dict_params)
                template.dict_params["signed_zero"] = np.float64(0.0)
            elif mutation == "id":
                template.id = 2**64 - 1
            else:
                template = SimpleNamespace(
                    params=template.params.copy(), id=2**63 + 97,
                    dict_params=dict(template.dict_params))
                prepared[0].templates[0] = template
            actual = module._live_finish_batches_jax(
                control, prepared, selections)
            assert actual is not None
            expected = _legacy_finish(control, prepared, selections)
            _assert_exact(actual, expected, active.jax_device)

        # A previously rejected template has no dict_params snapshot. Its
        # first acceptance must observe the current record, not an old table.
        newly_selected = prepared[0].templates[1]
        assert not hasattr(newly_selected, "dict_params")
        newly_selected.params["template_hash"] = 2**53 + 11
        newly_selected.params["approximant"] = "newly-selected"
        selections[0][1][1] = True
        actual = module._live_finish_batches_jax(control, prepared, selections)
        assert actual is not None
        _assert_exact(actual, _legacy_finish(control, prepared, selections),
                      active.jax_device)
        assert newly_selected.dict_params["template_hash"] == 2**53 + 11


@pytest.mark.parametrize("device", _gpu_devices())
def test_bulk_candidates_reuse_loaded_kernel_for_sparse_counts(
        device, monkeypatch):
    """Changing sparse counts reuses bucketed execution and fresh peak data."""
    with scheme.JAXScheme(device) as active, enable_x64(True):
        control, prepared, selections = _inputs(((0, 2), (1,), (0,)))
        calls = []
        original = jax.stages.Compiled.__call__

        def execute(compiled, *args, **kwargs):
            calls.append(compiled)
            return original(compiled, *args, **kwargs)

        monkeypatch.setattr(jax.stages.Compiled, "__call__", execute)
        first = module._live_finish_batches_jax(control, prepared, selections)
        assert first is not None
        jax.block_until_ready(first[0])
        initial = tuple(calls)
        assert initial
        calls.clear()
        for batch_index, batch in enumerate(prepared):
            new_peaks = batch.peak_values + jnp.complex64(batch_index + 1)
            prepared[batch_index] = batch._replace(
                peak_values=new_peaks, scaled_peaks=new_peaks * batch.norms)
        selections[0][1][0] = False
        second = module._live_finish_batches_jax(control, prepared, selections)
        assert second is not None
        jax.block_until_ready(second[0])
        # The count-specific publication slice can be new; preparation itself
        # must use the already-loaded handle for the same four-row bucket.
        assert any(handle is previous
                   for handle in calls for previous in initial)
        _assert_exact(second, _legacy_finish(control, prepared, selections),
                      active.jax_device)
        assert not np.array_equal(np.asarray(first[0]["snr"])[1:],
                                  np.asarray(second[0]["snr"]))


@pytest.mark.parametrize("device", _gpu_devices())
@pytest.mark.parametrize("case", ["empty", "offset", "abort", "strict",
                                  "heterogeneous"])
def test_bulk_candidates_fall_back_for_existing_special_contracts(
        device, case):
    with scheme.JAXScheme(device), enable_x64(True):
        selected = ((0,), (0,), (0,))
        if case == "empty":
            selected = ((), (), ())
        control, prepared, selections = _inputs(selected)
        if case == "offset":
            prepared[0].templates[0].time_offset = np.float32(0.125)
        elif case == "abort":
            selections[1][2][0] = True
        elif case == "heterogeneous":
            prepared[0].templates[0].dict_params = {
                name: prepared[0].templates[0].params[name]
                for name in prepared[0].templates[0].params.dtype.names}
            prepared[0].templates[0].dict_params["template_hash"] = 1.25
        if case == "strict":
            with jax.numpy_dtype_promotion("strict"):
                assert module._live_finish_batches_jax(
                    control, prepared, selections) is None
        else:
            assert module._live_finish_batches_jax(
                control, prepared, selections) is None


@pytest.mark.parametrize("device", _gpu_devices())
@pytest.mark.parametrize("mutation", ["offset", "scalar", "numeric-object"])
def test_bulk_candidates_recheck_metadata_after_cache_load(device, mutation):
    """A loaded cache cannot bypass a new unsupported metadata contract."""
    with scheme.JAXScheme(device) as active, enable_x64(True):
        control, prepared, selections = _inputs()
        actual = module._live_finish_batches_jax(control, prepared, selections)
        assert actual is not None
        _assert_exact(actual, _legacy_finish(control, prepared, selections),
                      active.jax_device)
        cached = control._jax_live_candidate_metadata
        # This row has never been selected, so the old assembler has not
        # created its dict_params snapshot yet.
        template = prepared[0].templates[1]
        if mutation == "offset":
            template.time_offset = np.float32(0.125)
            unaffected = module._live_finish_batches_jax(
                control, prepared, selections)
            assert unaffected is not None
            assert control._jax_live_candidate_metadata is cached
            _assert_exact(unaffected,
                          _legacy_finish(control, prepared, selections),
                          active.jax_device)
            selections[0][1][1] = True
            assert module._live_finish_batches_jax(
                control, prepared, selections) is None
            return
        template.dict_params = {key: template.params[key]
                                for key in template.params.dtype.names}
        if mutation == "scalar":
            template.dict_params["float_bits"] = np.float64(1.25)
        else:
            template.dict_params["float_bits"] = "non-numeric"
        unaffected = module._live_finish_batches_jax(
            control, prepared, selections)
        assert unaffected is not None
        assert control._jax_live_candidate_metadata is cached
        expected = _legacy_finish(control, prepared, selections)
        _assert_exact(unaffected, expected, active.jax_device)
        selections[0][1][1] = True
        assert module._live_finish_batches_jax(
            control, prepared, selections) is None


@pytest.mark.parametrize("device", _gpu_devices())
@pytest.mark.parametrize("missing", ["snapshot", "id"])
def test_bulk_candidates_ignore_missing_unselected_metadata(device, missing):
    """Caching the whole bank must not invent an error on a rejected row."""
    with scheme.JAXScheme(device), enable_x64(True):
        control, prepared, selections = _inputs()
        if missing == "snapshot":
            prepared[0].templates[1].dict_params = {}
        else:
            del prepared[0].templates[1].id
        expected = _legacy_finish(control, prepared, selections)
        assert len(expected[1]) == 5
        assert module._live_finish_batches_jax(
            control, prepared, selections) is None


@pytest.mark.parametrize("device", _gpu_devices())
def test_bulk_object_numeric_metadata_reclassifies_when_admitted(device):
    """Object fields follow selected scalar inference, including new dtypes."""
    with scheme.JAXScheme(device) as active, enable_x64(True):
        control, prepared, selections = _inputs()
        for batch in prepared:
            for template in batch.templates:
                dtype = np.dtype([
                    (key, object if key == "float_bits"
                     else template.params.dtype[key])
                    for key in template.params.dtype.names])
                replacement = np.empty(1, dtype=dtype)
                for key in dtype.names:
                    replacement[key][0] = template.params[key]
                template.params = replacement[0]
        actual = module._live_finish_batches_jax(control, prepared, selections)
        assert actual is not None
        _assert_exact(actual, _legacy_finish(control, prepared, selections),
                      active.jax_device)
        template = prepared[0].templates[1]
        template.dict_params = {key: template.params[key]
                                for key in template.params.dtype.names}
        template.dict_params["float_bits"] = np.float64(1.25)
        selections[0][1][1] = True
        selections[1][1][:] = False
        actual = module._live_finish_batches_jax(control, prepared, selections)
        assert actual is not None
        _assert_exact(actual, _legacy_finish(control, prepared, selections),
                      active.jax_device)


def test_bulk_candidate_fast_path_does_not_take_over_jax_cpu():
    with scheme.JAXScheme("cpu"):
        control, prepared, selections = _inputs()
        assert module._live_finish_batches_jax(
            control, prepared, selections) is None


def test_candidate_metadata_scalar_reuse_keeps_bitwise_invalidation():
    """Immutable scalar reuse must still observe replacements and mutations."""
    with scheme.JAXScheme("cpu"), enable_x64(True):
        control, prepared, selections = _inputs()
        rows = [np.flatnonzero(selection[1]) for selection in selections]
        cached = module._live_candidate_metadata(control, prepared, rows)
        assert cached is not None
        template = prepared[0].templates[0]
        first = template.dict_params["float_bits"]
        assert cached["numeric_scalars"]["float_bits"][0] is first
        assert module._live_candidate_metadata(control, prepared, rows) is cached

        # Equivalent replacement preserves the table and becomes reusable.
        replacement = np.frombuffer(first.tobytes(), dtype=first.dtype)[0]
        assert replacement is not first
        template.dict_params["float_bits"] = replacement
        assert module._live_candidate_metadata(control, prepared, rows) is cached
        assert cached["numeric_scalars"]["float_bits"][0] is replacement

        # NaN payloads and signed zero are changes even when value comparisons
        # would be inconclusive or equal. Check the actual uploaded payload.
        for key, value in (
                ("float_bits", np.uint32(0x7fc00042).view(np.float32)),
                ("float_bits", np.uint32(0x7fc00043).view(np.float32)),
                ("signed_zero", np.float64(0.0))):
            template.dict_params[key] = value
            updated = module._live_candidate_metadata(control, prepared, rows)
            assert updated is not cached
            assert updated["numeric_host"][key][0].tobytes() == value.tobytes()
            cached = updated


def test_candidate_metadata_subclass_and_none_do_not_bypass_validation():
    """A user scalar subclass is not covered by NumPy's immutability proof."""
    class MutableScalar(np.float32):
        def tobytes(self):
            return self.payload

    with scheme.JAXScheme("cpu"), enable_x64(True):
        control, prepared, selections = _inputs()
        rows = [np.flatnonzero(selection[1]) for selection in selections]
        template = prepared[0].templates[0]
        template.dict_params = {key: template.params[key]
                                for key in template.params.dtype.names}
        scalar = MutableScalar(template.dict_params["float_bits"])
        scalar.payload = np.float32(scalar).tobytes()
        template.dict_params["float_bits"] = scalar
        cached = module._live_candidate_metadata(control, prepared, rows)
        assert cached["numeric_scalars"]["float_bits"][0] is None
        assert module._live_candidate_metadata(control, prepared, rows) is cached
        scalar.payload = np.float32(1.0).tobytes()
        assert module._live_candidate_metadata(control, prepared, rows) is not cached
        template.dict_params["float_bits"] = None
        assert module._live_candidate_metadata(control, prepared, rows) is None


@pytest.mark.parametrize("device", ["cpu", *_gpu_devices()])
@pytest.mark.parametrize("selected", [
    ((0, 2), (1,), (0, 1)),
    ((), (0, 1), (2,)),
    ((2,), (), ()),
    ((), (), ()),
])
def test_resident_candidate_publication_preserves_payload_bits(device, selected):
    """The terminal snapshot retains wide IDs, NaNs and empty promotion."""
    with scheme.JAXScheme(device), enable_x64(True):
        control, prepared, selections = _inputs(selected, wide_ids=True)
        metadata = module._live_resident_metadata(control, prepared)
        assert metadata is not None
        groups = len(prepared)
        values = (*(batch.scaled_peaks for batch in prepared),
                  *(batch.native_norms for batch in prepared),
                  *(jnp.asarray(selection[0]) for selection in selections),
                  *(jnp.asarray(selection[1]) for selection in selections),
                  *(jnp.asarray(selection[2]) for selection in selections),
                  np.asarray(control.data.start_time, np.float64),
                  np.asarray(control.data.sample_rate, np.float64))
        snr, phase, sigma, times, order, keep, original, counts, abort = (
            module._live_resident_candidate_plan(*values, groups=groups,
                                                 limit=None))
        resident = module._LiveResidentResults(
            metadata, (snr, phase, sigma, times),
            (order, original, counts, abort),
            (jnp.zeros_like(snr, dtype=jnp.float32),
             jnp.zeros_like(snr, dtype=jnp.uint32),
             jnp.zeros_like(snr, dtype=jnp.float32), keep))
        actual = resident.materialize()
        for key in ("chisq", "chisq_dof", "sg_chisq"):
            actual.pop(key)
        expected = jax.device_get(_legacy_finish(control, prepared, selections)[0])
        assert list(actual) == list(expected)
        for key in expected:
            assert actual[key].dtype == expected[key].dtype, key
            assert actual[key].shape == expected[key].shape, key
            assert actual[key].tobytes() == expected[key].tobytes(), key


@pytest.mark.parametrize("device", _gpu_devices())
@pytest.mark.parametrize("limit", [None, 3])
def test_bulk_candidate_process_preserves_limit_and_veto_order(
        device, limit, monkeypatch):
    """The process hook sorts columns and lazy veto references together."""
    with scheme.JAXScheme(device) as active, enable_x64(True):
        control, prepared, selections = _inputs()
        expected, infos = _legacy_finish(control, prepared, selections)
        if limit:
            order = np.asarray(expected["snr"].argsort()[::-1][:limit])
            expected = {key: column[order] for key, column in expected.items()}
            infos = [infos[index] for index in order]
        batches = [batch._replace(selection=tuple(jnp.asarray(column)
                                                  for column in selected))
                   for batch, selected in zip(prepared, selections)]
        control.block_id = 0
        control.max_triggers_in_batch = limit
        control.set_data = lambda reader: None
        control.combine_results = module.combine_live_results_jax
        control._process_vetoes = lambda result, veto_info: (result, veto_info)

        def prepare(instance):
            batch = batches[instance.block_id]
            instance.block_id += 1
            return batch

        monkeypatch.setattr(module, "_live_prepare_batch_jax", prepare)
        actual = module.process_live_data_jax(control, control.data)
        _assert_exact(actual, (expected, infos), active.jax_device)
