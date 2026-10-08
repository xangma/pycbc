"""Ownership, ordering and backpressure for terminal JAX Live snapshots."""

import importlib.util
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import test_jax_live_mpi as transport_tests

_PACKAGE = '_jax_live_pipeline_test'
_ROOT = Path(__file__).parents[1]
_package = ModuleType(_PACKAGE)
_package.__path__ = [str(_ROOT / 'pycbc/events')]
sys.modules[_PACKAGE] = _package
sys.modules[_PACKAGE + '.live_mpi_jax'] = transport_tests._TRANSPORT_MODULE
_spec = importlib.util.spec_from_file_location(
    _PACKAGE + '.live_pipeline_jax',
    _ROOT / 'pycbc/events/live_pipeline_jax.py',
)
pipeline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pipeline_module)
JAXLiveResultPipeline = pipeline_module.JAXLiveResultPipeline


@pytest.fixture
def threaded_mpi(monkeypatch):
    mpi = transport_tests._FAKE_MPI
    monkeypatch.setattr(transport_tests._TRANSPORT_MODULE, '_MPI', mpi)
    monkeypatch.setattr(mpi, 'THREAD_MULTIPLE', 3, raising=False)
    monkeypatch.setattr(mpi, 'Query_thread', lambda: 3, raising=False)
    monkeypatch.delenv('PYCBC_JAX_LIVE_RESULT_QUEUE_DEPTH', raising=False)
    monkeypatch.delenv('PYCBC_BENCHMARK_STAGES', raising=False)
    monkeypatch.setattr(
        pipeline_module,
        '_materialize_payload',
        lambda payload, device: payload,
    )
    return mpi


def test_gpu_handoff_returns_before_snapshot_and_owns_mutable_leaves(
    threaded_mpi, monkeypatch
):
    entered, release = threading.Event(), threading.Event()
    worker_ids = []

    def materialize(payload, device):
        worker_ids.append(threading.get_ident())
        entered.set()
        assert release.wait(5)
        return payload

    monkeypatch.setattr(pipeline_module, '_materialize_payload', materialize)
    comm = transport_tests._Comm()
    pipeline = JAXLiveResultPipeline(
        comm, device=SimpleNamespace(platform='gpu')
    )
    values = np.arange(5, dtype=np.float32)
    payload = ({'H1': {'snr': values}}, 100.0)
    try:
        pipeline.commit(payload)
        assert entered.wait(5)
        assert pipeline.asynchronous and pipeline.pending_results == 1
        values[:] = -1
        payload[0]['H1']['new'] = True
        release.set()
        pipeline.drain()
        assert worker_ids == [pipeline._transport._thread]
        assert worker_ids[0] != threading.get_ident()
        np.testing.assert_array_equal(
            comm.sent[0][1][0]['H1']['snr'], np.arange(5)
        )
        assert 'new' not in comm.sent[0][1][0]['H1']
        assert pipeline.pending_results == pipeline.pending_bytes == 0
    finally:
        release.set()
        pipeline.close()
    assert comm.events.count('free') == 1


def test_gpu_queue_bounds_device_snapshots_and_preserves_order(
    threaded_mpi, monkeypatch
):
    entered, release = threading.Event(), threading.Event()

    def materialize(payload, device):
        if payload == 0:
            entered.set()
            assert release.wait(5)
        return payload

    monkeypatch.setattr(pipeline_module, '_materialize_payload', materialize)
    comm = transport_tests._Comm()
    pipeline = JAXLiveResultPipeline(
        comm, queue_depth=2, device=SimpleNamespace(platform='cuda')
    )
    try:
        pipeline.commit(0)
        assert entered.wait(5)
        pipeline.commit(1)
        assert pipeline.pending_results == 2
        release.set()
        pipeline.commit(2)
        assert pipeline.pending_results <= 2
        pipeline.drain()
        assert comm.sent == [(1, 0), (2, 1), (3, 2)]
        assert comm.events.count('ack:wait') == 3
    finally:
        release.set()
        pipeline.close()


@pytest.mark.parametrize('platform,level', [('cpu', 3), ('gpu', 2)])
def test_restricted_mpi_and_cpu_use_synchronous_handoff(
    threaded_mpi, platform, level
):
    threaded_mpi.Query_thread = lambda: level
    comm = transport_tests._Comm()
    pipeline = JAXLiveResultPipeline(
        comm, device=SimpleNamespace(platform=platform)
    )
    assert not pipeline.asynchronous
    pipeline.commit(({}, 100.0))
    assert comm.sent == [(1, ({}, 100.0))]
    pipeline.close()


def test_snapshot_failure_is_reported_and_thread_is_joined(
    threaded_mpi, monkeypatch
):
    def fail(payload, device):
        raise ValueError('snapshot failed')

    monkeypatch.setattr(pipeline_module, '_materialize_payload', fail)
    comm = transport_tests._Comm()
    pipeline = JAXLiveResultPipeline(
        comm, device=SimpleNamespace(platform='gpu')
    )
    pipeline.commit({})
    with pytest.raises(ValueError, match='snapshot failed'):
        pipeline.drain()
    pipeline.close()
    assert not comm.sent
    assert not any(
        t.name.startswith('jax-live-handoff') and t.is_alive()
        for t in threading.enumerate()
    )


@pytest.mark.parametrize('platform', ['gpu', 'cpu'])
def test_fatal_snapshot_reports_cause_and_aborts_on_mpi_owner(
    threaded_mpi, monkeypatch, caplog, platform
):
    def fail(payload, device):
        raise ValueError('snapshot failed before posting a result')

    monkeypatch.setattr(pipeline_module, '_materialize_payload', fail)
    comm = transport_tests._Comm()
    aborts = []
    comm.Abort = lambda code: aborts.append((code, threading.get_ident()))
    pipeline = JAXLiveResultPipeline(
        comm, device=SimpleNamespace(platform=platform)
    )
    with pytest.raises(ValueError, match='snapshot failed before posting'):
        if platform == 'gpu':
            pipeline.commit({})
            pipeline.drain()
        else:
            pipeline.commit({})
    pipeline.close()
    assert aborts == [(1, pipeline._transport._thread)]
    assert 'Fatal JAX Live result handoff failure on rank 1' in caplog.text
    assert 'ValueError: snapshot failed before posting a result' in caplog.text
    assert not comm.sent


def test_fatal_transport_error_aborts_once_after_completed_requests(
    threaded_mpi, caplog
):
    comm = transport_tests._Comm()
    comm.ack_sequence = -1
    aborts = []
    comm.Abort = lambda code: aborts.append(code)
    pipeline = JAXLiveResultPipeline(
        comm, device=SimpleNamespace(platform='gpu')
    )
    pipeline.commit({})
    with pytest.raises(RuntimeError, match='acknowledgement'):
        pipeline.drain()
    assert pipeline.pending_results == pipeline.pending_bytes == 0
    pipeline.close()
    assert aborts == [1]
    assert 'acknowledgement is out of order' in caplog.text


def test_pipeline_rejects_external_thread_and_repeated_close_is_safe(
    threaded_mpi,
):
    pipeline = JAXLiveResultPipeline(
        transport_tests._Comm(), device=SimpleNamespace(platform='gpu')
    )
    errors = []

    def wrong_thread():
        try:
            pipeline.commit({})
        except RuntimeError as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=wrong_thread)
    thread.start()
    thread.join()
    assert errors == ['JAX Live result pipeline used from another thread']
    pipeline.close()
    pipeline.close()


def test_close_failure_still_joins_worker_and_is_idempotent(threaded_mpi):
    comm = transport_tests._Comm()
    pipeline = JAXLiveResultPipeline(
        comm, device=SimpleNamespace(platform='gpu')
    )

    def fail_free():
        raise OSError('communicator free failed')

    comm.Free = fail_free
    with pytest.raises(OSError, match='communicator free failed'):
        pipeline.close()
    assert pipeline._closed
    assert not any(
        t.name.startswith('jax-live-handoff') and t.is_alive()
        for t in threading.enumerate()
    )
    pipeline.close()


def test_close_keeps_snapshot_error_when_cleanup_also_fails(
    threaded_mpi, monkeypatch
):
    comm = transport_tests._Comm()

    def fail_free():
        raise OSError('communicator free failed')

    def fail_snapshot(payload, device):
        raise ValueError('original snapshot failure')

    comm.Free = fail_free
    monkeypatch.setattr(pipeline_module, '_materialize_payload', fail_snapshot)
    pipeline = JAXLiveResultPipeline(
        comm, device=SimpleNamespace(platform='gpu')
    )
    pipeline.commit({})
    with pytest.raises(
        ValueError, match='original snapshot failure'
    ) as caught:
        pipeline.close()
    assert any(
        'communicator free failed' in note for note in caught.value.__notes__
    )
    assert pipeline._closed
    pipeline.close()


def test_constructor_failure_releases_duplicated_communicator(
    threaded_mpi, monkeypatch
):
    comm = transport_tests._Comm()

    def fail_pool(**kwargs):
        raise RuntimeError('handoff worker unavailable')

    monkeypatch.setattr(pipeline_module, 'ThreadPoolExecutor', fail_pool)
    with pytest.raises(RuntimeError, match='handoff worker unavailable'):
        JAXLiveResultPipeline(comm, device=SimpleNamespace(platform='gpu'))
    assert comm.events == ['dup', 'free']


def test_materialization_preserves_host_unicode_and_original_scalar_types():
    import jax
    from pycbc import scheme
    from pycbc.events.live_pipeline_jax import _materialize_payload

    labels = np.array(['H1', 'L1'])
    scalar = np.str_('HL')
    values = np.array([0x80000000, 0x7FC01234], np.uint32).view(np.float32)
    with scheme.JAXScheme('cpu') as context:
        resident = jax.device_put(values, context.jax_device)
        actual = _materialize_payload(
            {'labels': labels, 'scalar': scalar, 'science': resident},
            context.jax_device,
        )
    assert actual['labels'] is labels
    assert actual['scalar'] is scalar
    assert actual['science'].dtype == values.dtype
    assert actual['science'].tobytes() == values.tobytes()


@pytest.mark.parametrize('fails', [False, True])
def test_result_scope_closes_and_removes_transport_on_every_exit(
    threaded_mpi, fails
):
    manager = SimpleNamespace(comm=transport_tests._Comm())
    captured = []

    def run():
        with pipeline_module.live_result_scope(
            manager, device=SimpleNamespace(platform='gpu')
        ) as pipeline:
            captured.append(pipeline)
            assert manager._jax_result_transport is pipeline
            pipeline.commit(({}, 100.0))
            if fails:
                raise ValueError('analysis failed')

    if fails:
        with pytest.raises(ValueError, match='analysis failed'):
            run()
    else:
        run()
    assert not hasattr(manager, '_jax_result_transport')
    assert captured[0]._closed
    assert manager.comm.events.count('free') == 1
    assert manager.comm.sent == [(1, ({}, 100.0))]


def test_result_scope_preserves_analysis_error_and_reports_cleanup(
    threaded_mpi, monkeypatch
):
    manager = SimpleNamespace(comm=transport_tests._Comm())

    def fail_snapshot(payload, device):
        raise OSError('device snapshot failed')

    monkeypatch.setattr(pipeline_module, '_materialize_payload', fail_snapshot)
    with pytest.raises(ValueError, match='analysis failed') as caught:
        with pipeline_module.live_result_scope(
            manager, device=SimpleNamespace(platform='gpu')
        ):
            manager._jax_result_transport.commit({})
            raise ValueError('analysis failed')
    assert any(
        'device snapshot failed' in note for note in caught.value.__notes__
    )
    assert not hasattr(manager, '_jax_result_transport')
    assert manager.comm.events.count('free') == 1


def test_result_scope_resolves_active_device_after_scheme_entry(threaded_mpi):
    from pycbc import scheme

    manager = SimpleNamespace(comm=transport_tests._Comm())
    with scheme.JAXScheme('cpu') as context:
        with pipeline_module.live_result_scope(manager) as pipeline:
            assert pipeline._device is context.jax_device
            assert not pipeline.asynchronous
    assert not hasattr(manager, '_jax_result_transport')
