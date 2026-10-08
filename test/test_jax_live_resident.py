# Copyright (C) 2026 The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Resident Live plans against the retained sparse publication/veto path.

CPU cases exercise the retained CUDA functional path and API ownership. Linux
CPU contraction of the fused raw chi-square differs by at most two ULP in
these fixtures, so only that surrogate column has a numerical comparison.
Every CUDA column remains byte-exact against the actual retained CUDA path.
These fixtures do not replace complete-search scientific qualification.
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
from pycbc.filter.matchedfilter import LiveBatchMatchedFilter  # noqa: E402
from pycbc.types import FrequencySeries, zeros  # noqa: E402
from pycbc.vetoes.chisq import SingleDetPowerChisq  # noqa: E402


def _devices():
    try:
        if jax.devices("gpu"):
            return ["cpu", "cuda:0"]
    except RuntimeError:
        pass
    return ["cpu", pytest.param("cuda:0", marks=pytest.mark.skip(
        reason="resident Live CUDA qualification requires a GPU"))]


def _fixture(selected=((0, 2), (1,), (0, 1)), limit=None,
             newsnr_threshold=None, iteration=0):
    rng = np.random.default_rng(402)
    groups, prepared = [], []
    offset = 0
    for index, (count, n_time, selected_rows) in enumerate(zip(
            (3, 2, 3), (32, 64, 128), selected)):
        delta_f = 2048.0 / n_time
        width = n_time // 2 + 1
        source = jnp.asarray(rng.normal(size=(count, width))
                             + 1j * rng.normal(size=(count, width)),
                             dtype=jnp.complex64)
        params = np.empty(count, dtype=[
            ("template_hash", np.int64), ("mass1", np.float32),
            ("signed_zero", np.float64), ("flag", np.bool_),
            ("approximant", object)])
        params["template_hash"] = np.arange(count) + 2**53 + 1 + offset
        params["mass1"] = np.arange(count) + 2.25
        params["signed_zero"] = -0.0
        params["flag"] = np.arange(count) % 2 == 0
        params["approximant"] = ["TaylorF2"] * count
        templates = []
        for row in range(count):
            template = FrequencySeries(source[row], delta_f=delta_f)
            template.f_lower = 2 * delta_f
            template.params = params[row]
            template.id = 2**63 + 1 + offset + row
            template.cout = zeros(n_time, dtype=np.complex64)
            templates.append(template)
        psd = FrequencySeries(jnp.linspace(0.75, 2.5, width),
                              delta_f=delta_f)
        stilde = FrequencySeries(jnp.asarray(
            rng.normal(size=width) + 1j * rng.normal(size=width),
            jnp.complex64), delta_f=delta_f)
        stilde.psd = psd
        correlations = jnp.pad(jnp.conj(source) * module.to_jax(stilde),
                               ((0, 0), (0, n_time - width)))
        peaks = jnp.asarray([3 + 4j, -3 + 4j, -0.0 - 2j][:count],
                            jnp.complex64) + np.complex64(iteration)
        norms = jnp.asarray([0.25, 0.5, 2.0][:count], jnp.float64)
        mask = np.zeros(count, bool)
        mask[list(selected_rows)] = True
        selection = (jnp.asarray([1, 5, 11][:count], jnp.uint32),
                     jnp.asarray(mask), jnp.zeros(count, bool))
        prepared.append(module._LivePreparedBatch(
            index, templates, stilde, source,
            jnp.asarray([1.25, 2.5, 5.0][:count], jnp.float64), norms,
            3, peaks, peaks * norms, selection,
            correlations=correlations.reshape(-1), n_time=n_time))
        groups.append(templates)
        offset += count
    reader = SimpleNamespace(start_time=1234567890.123456,
                             sample_rate=2048)
    control = SimpleNamespace(
        tgroups=groups, power_chisq=SingleDetPowerChisq("4", None),
        sg_chisq=SimpleNamespace(do=False, values=lambda *args: None),
        data=reader, max_triggers_in_batch=limit,
        newsnr_threshold=newsnr_threshold)
    control.combine_results = module.combine_live_results_jax
    control._process_vetoes = lambda result, veto: module.process_live_vetoes_jax(
        control, result, veto)
    return control, prepared, reader


def _oracle(control, prepared):
    results, veto_info = [], []
    selections = jax.device_get(tuple(batch.selection for batch in prepared))
    for batch, selection in zip(prepared, selections):
        result, veto = module._live_finish_batch_jax(control, batch, selection)
        if result is False:
            return False
        results.append(result)
        veto_info.extend(veto)
    result = module.combine_live_results_jax(results)
    if control.max_triggers_in_batch:
        order = result["snr"].argsort()[::-1][:control.max_triggers_in_batch]
        result = {key: value[order] for key, value in result.items()}
        veto_info = [veto_info[index] for index in order]
    if veto_info and prepared[0].peak_values.device.platform == "cpu":
        # The changed production path is CUDA-only. Exercise its retained
        # functional oracle on CPU too; scalar JAX CPU division has a separate
        # rounding boundary and remains unchanged by the resident CUDA path.
        bins = [control.power_chisq.cached_chisq_bins(info[3], info[4].psd)
                for info in veto_info]
        result = module._bucketed_live_vetoes(
            control, result, veto_info, bins, control.power_chisq)
    else:
        result = control._process_vetoes(result, veto_info)
    return jax.device_get(result)


def _resident(control, prepared, reader):
    metadata = module._live_resident_metadata(control, prepared)
    assert metadata is not None
    token = module._LiveEnqueuedData(control, prepared, reader, metadata)
    return module.finish_live_data_jax(control, token)


def _assert_exact(actual, expected, cpu_surrogate=False):
    assert list(actual) == list(expected)
    for key in expected:
        assert type(actual[key]) is np.ndarray, key
        assert actual[key].shape == expected[key].shape, key
        assert actual[key].dtype == expected[key].dtype, key
        if cpu_surrogate and key == "chisq":
            np.testing.assert_array_max_ulp(actual[key], expected[key], maxulp=2)
        else:
            assert actual[key].tobytes() == expected[key].tobytes(), key


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("selected", [
    ((0, 1, 2), (0, 1), (0, 1, 2)),
    ((0, 2), (1,), (0, 1)),
    ((), (0, 1), (2,)),
    ((2,), (), ()),
    ((), (), ()),
])
@pytest.mark.parametrize("limit", [None, 3, -1])
@pytest.mark.parametrize("reference_operations", [(), ("waveform",)])
def test_resident_candidates_vetoes_match_sparse_path(
        device, selected, limit, reference_operations):
    """Retain every field, dtype, order, bin grid and arithmetic boundary."""
    with scheme.JAXScheme(
            device, reference_operations=reference_operations), enable_x64(True):
        control, prepared, reader = _fixture(selected, limit=limit)
        resident = _resident(control, prepared, reader)
        assert isinstance(resident, module._LiveResidentResults)
        actual = resident.materialize()
        expected = _oracle(control, prepared)
        _assert_exact(actual, expected, cpu_surrogate=device == "cpu")


def test_waveform_reference_keeps_resident_admission(monkeypatch):
    """Upstream waveform validation keeps prepared filtering inputs resident."""
    with scheme.JAXScheme("cpu", reference_operations=("waveform",)), enable_x64(True):
        control, prepared, reader = _fixture()
        control.block_id = 0
        control.set_data = lambda data: None

        def prepare(owner):
            batch = prepared[owner.block_id]
            owner.block_id += 1
            return batch

        # Exercise the CUDA admission guard with real prepared arrays; the
        # numerical comparison above runs the actual retained device kernels.
        monkeypatch.setattr(scheme.mgr.state, "jax_device",
                            SimpleNamespace(platform="gpu"))
        monkeypatch.setattr(module, "_live_prepare_batch_jax", prepare)
        token = module.enqueue_live_data_jax(control, reader)
        assert token.metadata is not None


def test_other_reference_controls_keep_resident_compatibility(monkeypatch):
    """Every other validation selector retains both conservative guards."""
    with scheme.JAXScheme("cpu"), enable_x64(True):
        control, prepared, reader = _fixture()
        metadata = module._live_resident_metadata(control, prepared)
        control.set_data = lambda data: None
        sentinel = object()

        def prepare(owner):
            batch = prepared[owner.block_id]
            owner.block_id += 1
            return batch

        monkeypatch.setattr(scheme.mgr.state, "jax_device",
                            SimpleNamespace(platform="gpu"))
        monkeypatch.setattr(module, "_live_prepare_batch_jax", prepare)
        monkeypatch.setattr(module, "_finish_live_data_compat",
                            lambda owner, token: sentinel)
        for operation in scheme.JAX_REFERENCE_OPERATIONS - {"waveform"}:
            for references in ((operation,), ("waveform", operation)):
                monkeypatch.setattr(scheme.mgr.state, "jax_reference_operations",
                                    frozenset(references))
                control.block_id = 0
                token = module.enqueue_live_data_jax(control, reader)
                assert token.metadata is None, references
                # The finish guard must also apply if metadata was admitted
                # before the caller selected a filtering validation route.
                admitted = module._LiveEnqueuedData(control, prepared, reader, metadata)
                assert (module.finish_live_data_jax(control, admitted)
                        is sentinel), references


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("threshold", [1.5, 3.0, 1e9])
def test_resident_newsnr_selection_stays_on_device(device, threshold):
    with scheme.JAXScheme(device), enable_x64(True):
        control, prepared, reader = _fixture(newsnr_threshold=threshold)
        actual = _resident(control, prepared, reader).materialize()
        _assert_exact(actual, _oracle(control, prepared),
                      cpu_surrogate=device == "cpu")


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("threshold", [1.5, 100.0])
def test_resident_veto_activation_skips_inactive_recurrences(device, threshold):
    with scheme.JAXScheme(device), enable_x64(True):
        control, prepared, reader = _fixture()
        control.power_chisq.snr_threshold = threshold
        actual = _resident(control, prepared, reader).materialize()
        _assert_exact(actual, _oracle(control, prepared),
                      cpu_surrogate=device == "cpu")


@pytest.mark.parametrize("device", _devices())
def test_resident_veto_interleaved_bin_count_groups(device):
    """Bin rows follow original template rows for sparse geometry groups."""
    with scheme.JAXScheme(device), enable_x64(True):
        control, prepared, reader = _fixture(
            ((0, 1, 2), (0, 1), (0, 1, 2)))
        control.power_chisq.num_bins = "resident_num_bins"
        for batch in prepared:
            for row, template in enumerate(batch.templates):
                template.resident_num_bins = 5 if row == 0 else 4
        actual = _resident(control, prepared, reader).materialize()
        _assert_exact(actual, _oracle(control, prepared),
                      cpu_surrogate=device == "cpu")


def test_resident_finish_has_no_scientific_readback(monkeypatch):
    """Candidate, bin, veto and ranking dispatch must not collect decisions."""
    with scheme.JAXScheme("cpu"), enable_x64(True):
        control, prepared, reader = _fixture(newsnr_threshold=1.5, limit=3)
        metadata = module._live_resident_metadata(control, prepared)
        token = module._LiveEnqueuedData(control, prepared, reader, metadata)
        with monkeypatch.context() as patch:
            patch.setattr(jax, "device_get", lambda *args, **kwargs: (
                pytest.fail("scientific tensor collected before terminal snapshot")))
            resident = module.finish_live_data_jax(control, token)
        actual = resident.materialize()
        _assert_exact(actual, _oracle(control, prepared), cpu_surrogate=True)


def test_resident_inputs_survive_reader_and_workspace_replacement():
    """Two queued detectors retain epochs and their original array roots."""
    with scheme.JAXScheme("cpu"), enable_x64(True):
        control, prepared, reader = _fixture()
        expected = _oracle(control, prepared)
        metadata = module._live_resident_metadata(control, prepared)
        token = module._LiveEnqueuedData(control, prepared, reader, metadata)
        reader.start_time += 100
        reader.sample_rate = 4096
        control.data = SimpleNamespace(start_time=-100, sample_rate=8)
        for batch in prepared:
            batch.stilde.psd = batch.stilde.psd.copy()
            for template in batch.templates:
                template.cout[:] = 0
        actual = module.finish_live_data_jax(control, token).materialize()
        _assert_exact(actual, expected, cpu_surrogate=True)


def test_resident_metadata_snapshot_and_late_dictionary_creation():
    """Mutable host metadata is frozen while lazy selection is preserved."""
    with scheme.JAXScheme("cpu"), enable_x64(True):
        control, prepared, reader = _fixture()
        metadata = module._live_resident_metadata(control, prepared)
        assert all(not hasattr(template, "dict_params")
                   for batch in prepared for template in batch.templates)
        resident = module.finish_live_data_jax(
            control, module._LiveEnqueuedData(control, prepared, reader, metadata))
        first = prepared[0].templates[0]
        first.params["signed_zero"] = 1.0
        first.params["approximant"] = "mutated"
        actual = resident.materialize()
        assert actual["signed_zero"][0].tobytes() == np.float64(-0.0).tobytes()
        assert actual["approximant"][0] == "TaylorF2"
        assert first.dict_params["signed_zero"].tobytes() == (
            np.float64(-0.0).tobytes())
        assert not hasattr(prepared[0].templates[1], "dict_params")


def test_resident_abort_and_recursive_terminal_tree():
    with scheme.JAXScheme("cpu"), enable_x64(True):
        control, prepared, reader = _fixture()
        selection = prepared[1].selection
        prepared[1] = prepared[1]._replace(selection=(
            selection[0], selection[1], jnp.asarray([False, True])))
        resident = _resident(control, prepared, reader)
        tree = ({"H1": resident, "L1": False}, 12.5)
        assert module.materialize_live_results_jax(tree) == (
            {"H1": False, "L1": False}, 12.5)


def test_terminal_snapshot_collects_both_detectors_once(monkeypatch):
    """The final tree includes both resident and compatibility device leaves."""
    with scheme.JAXScheme("cpu"), enable_x64(True):
        control, prepared, reader = _fixture()
        first = _resident(control, prepared, reader)
        control2, prepared2, reader2 = _fixture(iteration=1)
        second = _resident(control2, prepared2, reader2)
        collect = jax.device_get
        calls = []

        def counted(tree):
            calls.append(tree)
            return collect(tree)

        with monkeypatch.context() as patch:
            patch.setattr(jax, "device_get", counted)
            actual = module.materialize_live_results_jax(
                ({"H1": first, "L1": second, "V1": False},
                 [jnp.asarray(4.0), 12.5]))
        assert len(calls) == 1
        _assert_exact(actual[0]["H1"], _oracle(control, prepared),
                      cpu_surrogate=True)
        _assert_exact(actual[0]["L1"], _oracle(control2, prepared2),
                      cpu_surrogate=True)
        assert actual[0]["V1"] is False
        assert actual[1][0] == 4.0 and actual[1][1] == 12.5


@pytest.mark.parametrize("device", _devices())
def test_static_bank_metadata_stays_in_owned_host_tables(device, monkeypatch):
    """Admission casts once; steady reuse and terminal payload omit metadata."""
    from pycbc.events import live_collect_jax

    with scheme.JAXScheme(device), enable_x64(True):
        control, prepared, reader = _fixture()
        get = jax.device_get
        admission_calls = []

        def admission(tree):
            admission_calls.append(tree)
            return get(tree)

        with monkeypatch.context() as patch:
            patch.setattr(jax, "device_get", admission)
            snapshot = module._live_resident_metadata(control, prepared)
        assert len(admission_calls) == 1
        for key, (native, wide) in snapshot["numeric"].items():
            dtype = np.result_type(native.dtype, np.float64)
            expected = get(jnp.asarray(native).astype(dtype))
            assert wide.dtype == expected.dtype, key
            assert wide.tobytes() == expected.tobytes(), key
        with monkeypatch.context() as patch:
            patch.setattr(jax, "device_get", lambda *args, **kwargs: (
                pytest.fail("steady metadata reuse collected values")))
            patch.setattr(jnp, "asarray", lambda *args, **kwargs: (
                pytest.fail("steady metadata reuse uploaded bank columns")))
            assert module._live_resident_metadata(control, prepared) is snapshot
        owned = (snapshot["ids"], *(value for pair in snapshot["numeric"].values()
                                    for value in pair))
        assert all(type(value) is np.ndarray and value.flags.owndata
                   and not value.flags.writeable for value in owned)
        resident = module.finish_live_data_jax(
            control, module._LiveEnqueuedData(control, prepared, reader, snapshot))
        collect = live_collect_jax.collect_live_arrays
        calls = []

        def checked(tree):
            calls.append(tree)
            leaves = jax.tree_util.tree_leaves(tree)
            assert all(type(value) is not np.ndarray for value in leaves)
            assert all(all(value is not column for column in owned)
                       for value in leaves)
            return collect(tree)

        with monkeypatch.context() as patch:
            patch.setattr(live_collect_jax, "collect_live_arrays", checked)
            actual = resident.materialize()
        assert len(calls) == 1
        _assert_exact(actual, _oracle(control, prepared),
                      cpu_surrogate=device == "cpu")


@pytest.mark.parametrize("device", _devices())
def test_host_bank_tables_survive_later_metadata_and_id_changes(device):
    """Queued publication owns IDs/native/promoted values before bank changes."""
    with scheme.JAXScheme(device), enable_x64(True):
        control, prepared, reader = _fixture()
        snapshot = module._live_resident_metadata(control, prepared)
        resident = module.finish_live_data_jax(
            control, module._LiveEnqueuedData(control, prepared, reader, snapshot))
        first = prepared[0].templates[0]
        old_id = snapshot["ids"][0]
        old_mass = snapshot["numeric"]["mass1"][0][0]
        first.params["mass1"] = np.float32(99.0)
        first.id = 7
        first.dict_params = {key: snapshot["host"][key][0]
                             for key in snapshot["keys"]}
        first.dict_params["mass1"] = np.float32(25.0)
        first.dict_params["approximant"] = "later"
        changed = module._live_resident_metadata(control, prepared)
        assert changed is not snapshot
        assert changed["ids"][0] == 7
        assert changed["numeric"]["mass1"][0][0] == np.float32(25.0)
        actual = resident.materialize()
        assert actual["template_id"][0].tobytes() == old_id.tobytes()
        assert actual["mass1"][0].tobytes() == old_mass.tobytes()
        assert actual["approximant"][0] == "TaylorF2"
        assert snapshot["ids"][0] == old_id
        assert snapshot["numeric"]["mass1"][0][0] == old_mass


@pytest.mark.parametrize("device", _devices())
def test_native_metadata_admission_needs_no_device_collection(device, monkeypatch):
    with scheme.JAXScheme(device), enable_x64(True):
        control, prepared, _ = _fixture()
        for batch in prepared:
            for row, template in enumerate(batch.templates):
                template.params = np.asarray([(row + 0.5, "TaylorF2")], dtype=[
                    ("mass1", np.float64), ("approximant", object)])[0]
        with monkeypatch.context() as patch:
            patch.setattr(jax, "device_get", lambda *args, **kwargs: (
                pytest.fail("native metadata collected values")))
            patch.setattr(jnp, "asarray", lambda *args, **kwargs: (
                pytest.fail("native metadata uploaded values")))
            snapshot = module._live_resident_metadata(control, prepared)
            assert module._live_resident_metadata(control, prepared) is snapshot
        native, promoted = snapshot["numeric"]["mass1"]
        assert promoted is native
        assert type(native) is np.ndarray and not native.flags.writeable






def test_resident_metadata_rejects_mutable_custom_values():
    with scheme.JAXScheme("cpu"), enable_x64(True):
        control, prepared, _ = _fixture()
        prepared[0].templates[0].params["approximant"] = ["mutable"]
        assert module._live_resident_metadata(control, prepared) is None


@pytest.mark.parametrize("device", [value for value in _devices()
                                  if value != "cpu"])
def test_two_detectors_enqueue_before_collection(device, monkeypatch):
    """A shared control queues both IFOS without reading or aliasing the first."""
    from pycbc.events.live_pipeline_jax import (
        enqueue_live_detector_jax, finish_live_detectors_jax,
    )

    with scheme.JAXScheme(device), enable_x64(True):
        fixture_control, groups, _ = _fixture()
        templates = [template for group in groups for template in group.templates]
        control = LiveBatchMatchedFilter(
            templates, 0.0, "4", fixture_control.sg_chisq,
            maxelements=2**12, max_triggers_in_batch=5)
        sources = {batch.templates[0].delta_f: batch.stilde for batch in groups}

        def reader(start, scale):
            strain = {}
            for delta_f, value in sources.items():
                converted = FrequencySeries(module.to_jax(value) * scale,
                                            delta_f=delta_f)
                converted.psd = value.psd.copy()
                strain[delta_f] = converted
            return SimpleNamespace(
                start_time=start, sample_rate=2048, trim_padding=3,
                blocksize=8 / 2048,
                overwhitened_data=lambda delta_f: strain[delta_f])

        h1, l1 = reader(1234567890.125, 1.0), reader(1234568000.25, 1.25)
        expected = [jax.device_get(control.process_data(data)) for data in (h1, l1)]
        # Admit static bank promotion semantics once, before testing steady
        # detector dispatch. No peak/candidate decision is collected here.
        module._live_resident_metadata(control, [SimpleNamespace(
            templates=group, peak_values=jnp.empty(0, dtype=jnp.complex64))
            for group in control.tgroups])
        with monkeypatch.context() as patch:
            patch.setattr(jax, "device_get", lambda *args, **kwargs: (
                pytest.fail("queued detector collected before terminal snapshot")))
            tokens = {ifo: enqueue_live_detector_jax(control, data)
                      for ifo, data in (("H1", h1), ("L1", l1))}
            assert all(token.metadata is not None for token in tokens.values())
            # Advance the shared control and both mutable reader epochs before
            # finish; each token still owns the detector's actual input roots.
            control.set_data(reader(1.0, 10.0))
            h1.start_time += 64
            l1.start_time += 64
            h1.sample_rate = l1.sample_rate = 4096
            results = finish_live_detectors_jax(control, tokens)
        actual = module.materialize_live_results_jax(results)
        for output, reference in zip(actual.values(), expected):
            _assert_exact(output, reference)


def test_resident_metadata_invalidates_authoritative_mutations():
    """Bank configuration reuse must observe replacements and payload bits."""
    with scheme.JAXScheme("cpu"), enable_x64(True):
        control, prepared, _ = _fixture()
        cached = module._live_resident_metadata(control, prepared)
        assert module._live_resident_metadata(control, prepared) is cached
        first = prepared[0].templates[0]
        first.params["mass1"] = np.float32(3.5)
        updated = module._live_resident_metadata(control, prepared)
        assert updated is not cached
        assert updated["host"]["mass1"][0] == np.float32(3.5)
        first.dict_params = {key: first.params[key]
                             for key in first.params.dtype.names}
        first.dict_params["signed_zero"] = np.float64(0.0)
        final = module._live_resident_metadata(control, prepared)
        assert final["host"]["signed_zero"][0].tobytes() == (
            np.float64(0.0).tobytes())


def _authoritative_metadata(control, prepared):
    snapshot = module._live_resident_metadata(control, prepared)
    for row, template in enumerate(snapshot["templates"]):
        template.dict_params = {key: snapshot["host"][key][row]
                                for key in snapshot["keys"]}
    return module._live_resident_metadata(control, prepared)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dictionary_source", ["snapshot", "record"])
def test_lazy_dictionary_publication_reuses_owned_metadata(
        device, dictionary_source, monkeypatch):
    """Incremental lazy publication changes validation state without uploads."""
    with scheme.JAXScheme(device), enable_x64(True):
        control, prepared, reader = _fixture()
        snapshot = module._live_resident_metadata(control, prepared)
        resident = module.finish_live_data_jax(
            control, module._LiveEnqueuedData(control, prepared, reader, snapshot))
        columns, ids, host = snapshot["numeric"], snapshot["ids"], snapshot["host"]
        for row in (0, 3, 7):
            template = snapshot["templates"][row]
            template.dict_params = {
                key: (snapshot["host"][key][row] if dictionary_source == "snapshot"
                      else template.params[key]) for key in snapshot["keys"]}
            with monkeypatch.context() as patch:
                patch.setattr(jax, "device_get", lambda *args, **kwargs: (
                    pytest.fail("lazy publication collected device values")))
                patch.setattr(jnp, "asarray", lambda *args, **kwargs: (
                    pytest.fail("lazy publication reuploaded bank columns")))
                patch.setattr(module.np, "iinfo", lambda *args, **kwargs: (
                    pytest.fail("equal lazy publication entered full validation")))
                assert module._live_resident_metadata(control, prepared) is snapshot
            assert snapshot["numeric"] is columns
            assert snapshot["ids"] is ids
            assert snapshot["host"] is host
            assert snapshot["validated"][row] is not None
        _assert_exact(resident.materialize(), _oracle(control, prepared),
                      cpu_surrogate=device == "cpu")


def test_lazy_dictionary_changed_payload_keeps_deferred_snapshot():
    """A real authoritative change rebuilds while queued results stay frozen."""
    with scheme.JAXScheme("cpu"), enable_x64(True):
        control, prepared, reader = _fixture()
        snapshot = module._live_resident_metadata(control, prepared)
        resident = module.finish_live_data_jax(
            control, module._LiveEnqueuedData(control, prepared, reader, snapshot))
        first = prepared[0].templates[0]
        first.dict_params = {key: snapshot["host"][key][0]
                             for key in snapshot["keys"]}
        first.dict_params["signed_zero"] = np.float64(0.0)
        updated = module._live_resident_metadata(control, prepared)
        assert updated is not snapshot
        assert updated["host"]["signed_zero"][0].tobytes() == (
            np.float64(0.0).tobytes())
        assert resident.materialize()["signed_zero"][0].tobytes() == (
            np.float64(-0.0).tobytes())


def test_lazy_metadata_scalar_comparison_preserves_payload_bits():
    same_nan = np.asarray([0x7ff8000000000123], np.uint64).view(np.float64)
    other_nan = np.asarray([0x7ff8000000000456], np.uint64).view(np.float64)
    equal = module._live_immutable_metadata_equal
    assert equal(same_nan[0], same_nan.copy()[0])
    assert not equal(same_nan[0], other_nan[0])
    assert equal(same_nan[0].item(), same_nan.copy()[0].item())
    assert not equal(same_nan[0].item(), other_nan[0].item())
    assert not equal(np.float64(-0.0), np.float64(0.0))
    assert not equal(-0.0, 0.0)
    assert not equal(complex(1.0, -0.0), complex(1.0, 0.0))
    assert not equal(np.float32(1.0), np.float64(1.0))


def test_resident_metadata_identity_reuse_and_equal_bit_replacements(monkeypatch):
    """Stable immutable fields skip byte signatures; replacements revalidate."""
    with scheme.JAXScheme("cpu"), enable_x64(True):
        control, prepared, _ = _fixture()
        cached = _authoritative_metadata(control, prepared)
        first = prepared[0].templates[0]

        def unexpected(*args, **kwargs):
            pytest.fail("stable immutable metadata entered full signature validation")

        with monkeypatch.context() as patch:
            patch.setattr(module.np, "iinfo", unexpected)
            assert module._live_resident_metadata(control, prepared) is cached
        original = first.dict_params["mass1"]
        replacement = np.array(original).copy()[()]
        assert replacement is not original
        first.dict_params["mass1"] = replacement
        assert module._live_resident_metadata(control, prepared) is cached
        assert cached["host"]["mass1"][0] is original
        with monkeypatch.context() as patch:
            patch.setattr(module.np, "iinfo", unexpected)
            assert module._live_resident_metadata(control, prepared) is cached
        # Raw record mutations remain irrelevant after dict_params becomes
        # authoritative, exactly as in the retained sparse assembler.
        first.params["mass1"] = 99.0
        assert module._live_resident_metadata(control, prepared) is cached


@pytest.mark.parametrize("change", ["bits", "dtype", "missing", "mutable", "dict"])
def test_resident_metadata_identity_reuse_rejects_authoritative_changes(change):
    with scheme.JAXScheme("cpu"), enable_x64(True):
        control, prepared, _ = _fixture()
        cached = _authoritative_metadata(control, prepared)
        first = prepared[0].templates[0]
        if change == "bits":
            first.dict_params["signed_zero"] = np.float64(0.0)
        elif change == "dtype":
            first.dict_params["mass1"] = np.float64(first.dict_params["mass1"])
        elif change == "missing":
            del first.dict_params["mass1"]
        elif change == "mutable":
            first.dict_params["approximant"] = ["TaylorF2"]
        else:
            first.dict_params = dict(first.dict_params)
        updated = module._live_resident_metadata(control, prepared)
        if change in ("dtype", "missing", "mutable"):
            assert updated is None
        else:
            assert updated is not cached
            if change == "bits":
                assert updated["host"]["signed_zero"][0].tobytes() == (
                    np.float64(0.0).tobytes())
            # The previous result snapshot remains immutable for its worker.
            assert cached["host"]["signed_zero"][0].tobytes() == (
                np.float64(-0.0).tobytes())


def test_resident_metadata_group_boundary_invalidates_snapshot():
    with scheme.JAXScheme("cpu"), enable_x64(True):
        control, prepared, _ = _fixture()
        cached = module._live_resident_metadata(control, prepared)
        flat = cached["templates"]
        changed = [prepared[0]._replace(templates=flat[:2]),
                   prepared[1]._replace(templates=flat[2:5]), prepared[2]]
        updated = module._live_resident_metadata(control, changed)
        assert updated is not cached
        assert updated["sizes"] == (2, 3, 3)


def test_resident_override_keeps_captured_compatibility_epoch(monkeypatch):
    """Unqualified features retain the original assembler with frozen time."""
    with scheme.JAXScheme("cpu"), enable_x64(True):
        control, prepared, reader = _fixture()
        expected = _oracle(control, prepared)
        token = module._LiveEnqueuedData(control, prepared, reader, None)
        reader.start_time += 100
        control._process_vetoes = lambda result, veto: module._bucketed_live_vetoes(
            control, result, veto,
            [control.power_chisq.cached_chisq_bins(info[3], info[4].psd)
             for info in veto], control.power_chisq)
        actual = jax.device_get(module.finish_live_data_jax(control, token))
        _assert_exact(actual, expected)


def test_resident_reuse_rejects_legacy_mixed_precision_veto_inputs():
    """A rounded FFT root cannot replace a higher-precision veto product."""
    with scheme.JAXScheme("cpu"), enable_x64(True):
        _, prepared, _ = _fixture()
        batch = prepared[0]
        assert module._live_resident_batch_qualified(batch)
        assert not module._live_resident_batch_qualified(batch._replace(
            template_matrix=batch.template_matrix.astype(jnp.complex128)))
        assert not module._live_resident_batch_qualified(batch._replace(
            correlations=None))
        assert not module._live_resident_batch_qualified(batch._replace(
            n_time=batch.n_time + 1))


def test_actual_jax_cpu_enqueue_retains_original_processing(monkeypatch):
    """The portable resident fixtures never select production JAX CPU code."""
    control, reader, output = object(), object(), {"original": np.arange(3)}
    calls = []

    def original(owner, data):
        calls.append((owner, data))
        return output

    monkeypatch.setattr(module, "process_live_data_jax", original)
    with scheme.JAXScheme("cpu"):
        assert module.enqueue_live_data_jax(control, reader) is output
    assert calls == [(control, reader)]
