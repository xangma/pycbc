# Copyright (C) 2026 The PyCBC Collaboration
"""Live fused filter parity, public-buffer ownership and fallback contracts."""

from types import SimpleNamespace

import numpy as np
import pytest

from pycbc import scheme
from pycbc.fft import IFFT
from pycbc.filter.matchedfilter import BatchCorrelator
from pycbc.types import Array

jax = pytest.importorskip("jax")
jnp = jax.numpy
from pycbc.filter import matchedfilter_jax as backend  # noqa: E402


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    if request.param == "cuda:0":
        try:
            available = jax.devices("gpu")
        except RuntimeError:
            available = []
        if not available:
            pytest.skip("CUDA JAX device unavailable")
    return request.param


def _values(count, size, case, precision=np.complex64):
    rng = np.random.default_rng(2835)
    length = size // 2 + 1
    templates = (rng.normal(size=(count, length))
                 + 1j * rng.normal(size=(count, length))).astype(precision)
    strain = (rng.normal(size=length)
              + 1j * rng.normal(size=length)).astype(precision)
    parent = (rng.normal(size=count * size)
              + 1j * rng.normal(size=count * size)).astype(np.complex64)
    norms = np.linspace(0.2, 1.3, count, dtype=np.float64)
    if case == "tie":
        templates[:] = 0
        templates[:, 0] = 1
        strain[:] = 1
        parent[:] = 0
    elif case == "nan":
        templates[0, 0] = np.nan
    elif case == "empty":
        norms[:] = 0
    return tuple(jnp.asarray(value) for value in
                 (templates, strain, parent, norms))


def _reference(values, count, size, start, stop, threshold, abort_threshold):
    templates, strain, parent, norms = values
    correlation = backend._batch_correlate_update(
        templates, strain, parent, 0, size)
    output = backend._live_ifft_flat_jax(correlation, count, size)
    indices, peaks = backend._batch_peak_core(output, count, start, stop)
    scaled, accepted, abort = backend._live_select_peaks(
        peaks, norms, threshold, abort_threshold)
    return correlation, output, indices, peaks, scaled, accepted, abort


def _assert_bytes(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(np.asarray(actual).tobytes(),
                                  np.asarray(expected).tobytes())


@pytest.mark.parametrize("count,size", [(1, 12), (3, 16), (2, 30), (3, 49152)])
@pytest.mark.parametrize("case", ["ordinary", "tie", "nan", "empty"])
def test_fused_filter_retains_exact_stage_outputs(device, count, size, case):
    """Fusion preserves buffers, peak ties/NaNs and threshold masks."""
    with scheme.JAXScheme(device):
        values = _values(count, size, case)
        before = [np.asarray(value).copy() for value in values]
        start, stop, threshold, abort_threshold = 2, size - 1, 0.75, 10.0
        expected = _reference(values, count, size, start, stop,
                              threshold, abort_threshold)
        actual = backend._live_fused_batch_core_jax(
            *values, threshold, abort_threshold,
            count=count, size=size, start=start, stop=stop)
        for value, reference in zip(actual, expected):
            _assert_bytes(value, reference)
        for value, previous in zip(values, before):
            _assert_bytes(value, previous)
        if case == "tie":
            np.testing.assert_array_equal(np.asarray(actual[2]), 0)
        if case == "nan":
            assert bool(actual[-2][0])  # not (NaN < threshold)
            assert not bool(actual[-1][0])
        if case == "empty":
            assert not np.any(np.asarray(actual[-2]))


def test_fused_filter_retains_mixed_precision_storage_boundary(device):
    with scheme.JAXScheme(device):
        count, size = 3, 30
        values = _values(count, size, "ordinary", np.complex128)
        expected = _reference(values, count, size, 1, 24, 0.5, jnp.inf)
        actual = backend._live_fused_batch_core_jax(
            *values, 0.5, jnp.inf, count=count, size=size, start=1, stop=24)
        for value, reference in zip(actual, expected):
            _assert_bytes(value, reference)


def _workspace(count=3, size=16):
    values = _values(count, size, "ordinary")
    templates, strain, parent, norms = values
    source, target = Array(parent), Array(np.zeros(count * size, np.complex64))
    rows = [source[row * size:(row + 1) * size] for row in range(count)]
    correlator = BatchCorrelator(
        [Array(row) for row in np.asarray(templates)], rows,
        templates.shape[1], immutable_templates=True)
    plan = IFFT(source, target, nbatch=count, size=size)
    backend._bind_live_correlate_workspace_jax(correlator, source)
    backend._bind_live_ifft_workspace_jax(plan)
    return SimpleNamespace(correlator=correlator, plan=plan, source=source,
                           target=target, strain=Array(strain), norms=norms,
                           count=count, size=size)


def _capture(ws):
    return backend._live_fused_batch_inputs_jax(
        ws.correlator, ws.plan, ws.strain, ws.norms,
        slice(2, ws.size - 1), 0.75, jnp.inf)


def test_fused_filter_owned_publication_keeps_snapshots_and_row_views(device):
    """Repeated publication replaces roots while preserving public aliases."""
    with scheme.JAXScheme(device):
        ws = _workspace()
        if device == "cpu":
            assert _capture(ws) is None
            return
        row = ws.target[ws.size:2 * ws.size]
        for scale in (np.complex64(1), np.complex64(0.25 - 0.5j)):
            ws.strain._data.set_array(ws.strain._data.array * scale)
            inputs = _capture(ws)
            assert inputs is not None
            raw_source = ws.source._data.array
            raw_target = ws.target._data.array
            previous = [np.asarray(value).copy()
                        for value in (raw_source, raw_target)]
            outputs = backend._live_launch_fused_batch_jax(inputs)
            # Kernel launch alone cannot update externally visible wrappers.
            _assert_bytes(ws.source._data.array, previous[0])
            _assert_bytes(ws.target._data.array, previous[1])
            compact = backend._live_publish_fused_batch_jax(inputs, outputs)
            for value, expected in zip(compact, outputs[2:]):
                assert value is expected
            _assert_bytes(ws.source._data.array, outputs[0])
            _assert_bytes(ws.target._data.array, outputs[1])
            _assert_bytes(row._data.array,
                          np.asarray(outputs[1])[ws.size:2 * ws.size])
            for value, expected in zip((raw_source, raw_target), previous):
                _assert_bytes(value, expected)


@pytest.mark.parametrize("change", [
    "correlate_reference", "ifft_reference", "peak_reference",
    "selection_reference", "ifft_target", "correlation_matrix", "norms",
    "segment", "instrumented_peaks",
])
def test_fused_filter_rejects_unqualified_paths_without_publication(
        device, monkeypatch, change):
    reference = {
        "correlate_reference": "correlate", "ifft_reference": "ifft",
        "peak_reference": "abs_arg_max", "selection_reference": "live_selection",
    }.get(change)
    operations = () if reference is None else (reference,)
    with scheme.JAXScheme(device, reference_operations=operations):
        ws = _workspace()
        before = [np.asarray(array).copy() for array in (ws.source, ws.target)]
        if change == "ifft_target":
            ws.plan.outvec = Array(ws.target, copy=False)
        elif change == "correlation_matrix":
            ws.correlator._jax_template_matrix = (
                ws.correlator._jax_template_matrix * 2)
        elif change == "norms":
            ws.norms = jnp.ones(ws.count + 1)
        elif change == "instrumented_peaks":
            monkeypatch.setattr(backend, "_live_select_peaks", lambda *a: a)
        inputs = backend._live_fused_batch_inputs_jax(
            ws.correlator, ws.plan, ws.strain, ws.norms,
            slice(2, ws.size - 1, 2) if change == "segment"
            else slice(2, ws.size - 1), 0.75, jnp.inf)
        assert inputs is None
        roots = ws.source._data.array, ws.target._data.array
        for value, expected in zip(roots, before):
            _assert_bytes(value, expected)


def test_changed_workspace_rejects_pending_publication(device):
    """Validate both destinations before updating either public root."""
    if device == "cpu":
        pytest.skip("Owned fusion is CUDA only")
    with scheme.JAXScheme(device):
        ws = _workspace()
        inputs = _capture(ws)
        outputs = backend._live_launch_fused_batch_jax(inputs)
        before = [np.asarray(value).copy() for value in (ws.source, ws.target)]
        ws.plan.outvec = Array(np.zeros(ws.count * ws.size, np.complex64))
        with pytest.raises(ValueError, match="changed before publication"):
            backend._live_publish_fused_batch_jax(inputs, outputs)
        roots = ws.source._data.array, ws.target._data.array
        for value, expected in zip(roots, before):
            _assert_bytes(value, expected)


def test_two_phase_preparation_captures_reader_without_advancing_cursor(
        device, monkeypatch):
    """A shared control can capture inputs before ordered publication."""
    with scheme.JAXScheme(device):
        ws = _workspace()
        ws.strain.psd = object()
        for template in ws.correlator.xs:
            template.delta_f = 1.0
        control = SimpleNamespace(
            block_id=0, tgroups=[ws.correlator.xs], corr=[ws.correlator],
            chunk_tsamples=[ws.size], mids=[0], ifts={0: ws.plan},
            out_mem={0: ws.target}, cout_mem={0: ws.source},
            snr_threshold=0.75,
            snr_abort_threshold=None,
            data=SimpleNamespace(
                overwhitened_data=lambda delta_f: ws.strain,
                trim_padding=1, blocksize=1, sample_rate=8),
        )
        monkeypatch.setattr(backend, "_live_cached_template_norms_jax",
                            lambda *a: (jnp.ones(ws.count), ws.norms))
        before = [np.asarray(value).copy() for value in (ws.source, ws.target)]
        inputs = backend._live_prepare_batch_inputs_jax(control, 0)
        assert control.block_id == 0
        assert inputs.stilde is ws.strain
        assert inputs.segment == slice(7, 15)
        roots = ws.source._data.array, ws.target._data.array
        for value, expected in zip(roots, before):
            _assert_bytes(value, expected)
        # Completion must consume the captured reader, not the current one.
        control.data = None
        prepared = backend._live_complete_batch_jax(control, inputs)
        assert control.block_id == 0
        assert prepared.group_index == 0
        assert prepared.stilde is inputs.stilde
        assert prepared.norms is inputs.norms
        if device == "cpu":
            assert prepared.correlations is None
        else:
            assert prepared.correlations is ws.source._data.array
        expected = _reference(
            (ws.correlator._jax_template_matrix, ws.strain._data.array,
             jnp.asarray(before[0]), ws.norms),
            ws.count, ws.size, 7, 15, 0.75, jnp.inf)
        for value, reference in zip(
                (ws.source._data.array, ws.target._data.array,
                 prepared.selection[0], prepared.peak_values,
                 prepared.scaled_peaks, prepared.selection[1],
                 prepared.selection[2]), expected):
            _assert_bytes(value, reference)


def test_shared_workspace_ordered_publication_keeps_each_peak_snapshot(device):
    """Compact results survive later detectors publishing the same roots."""
    if device == "cpu":
        pytest.skip("Owned fusion is CUDA only")
    with scheme.JAXScheme(device):
        ws = _workspace()
        first = _capture(ws)
        ws.strain._data.set_array(
            ws.strain._data.array * np.complex64(0.25 - 0.5j))
        second = _capture(ws)
        outputs = [backend._live_launch_fused_batch_jax(inputs)
                   for inputs in (first, second)]
        snapshots = [tuple(np.asarray(value).copy() for value in values[2:])
                     for values in outputs]
        for inputs, values in zip((first, second), outputs):
            backend._live_publish_fused_batch_jax(inputs, values)
        for values, expected in zip(outputs, snapshots):
            for value, previous in zip(values[2:], expected):
                _assert_bytes(value, previous)
        _assert_bytes(ws.source._data.array, outputs[-1][0])
        _assert_bytes(ws.target._data.array, outputs[-1][1])
