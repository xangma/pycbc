# Copyright (C) 2026 The PyCBC Collaboration
# SPDX-License-Identifier: GPL-3.0-or-later
"""Device work submission and ordered terminal handoff for JAX Live.

Scientific arrays stay immutable while detector work and terminal snapshots
overlap. The serialized MPI boundary still requires host buffers. Only MPI
implementations providing THREAD_MULTIPLE use the dedicated handoff thread;
CPU and restricted-thread MPI retain the synchronous transport contract.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import logging
import threading

import numpy as np

from .live_mpi_jax import JAXLiveResultTransport, host_stage


def enqueue_live_detector_jax(control, reader):
    """Submit a detector without collecting its candidate decisions."""
    from pycbc.filter.matchedfilter_jax import enqueue_live_data_jax

    return enqueue_live_data_jax(control, reader)


def finish_live_detectors_jax(control, results):
    """Submit every detector's veto work before terminal materialization."""
    from pycbc.filter.matchedfilter_jax import finish_live_data_jax

    return {
        ifo: (
            finish_live_data_jax(control, value)
            if getattr(value, '_jax_live_prepared', False)
            else value
        )
        for ifo, value in results.items()
    }


def live_result_column_backend_jax(key, column):
    """Keep routing IDs on the host and numerical science on the device."""
    if key == 'template_id' or column.dtype.kind in 'OUS':
        return np
    import jax.numpy as jnp

    return jnp


def _own_payload(value):
    """Copy host leaves while immutable device arrays retain their owners."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _own_payload(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_own_payload(item) for item in value)
    if isinstance(value, list):
        return [_own_payload(item) for item in value]
    return value


def _materialize_payload(value, device):
    import jax
    from pycbc.filter.matchedfilter_jax import materialize_live_results_jax

    with jax.default_device(device), host_stage('device_to_host'):
        return materialize_live_results_jax(value)


@contextmanager
def live_result_scope(event_manager, queue_depth=None, *, device=None):
    """Own result handoff through normal completion and exceptional exits."""
    if device is None:
        from pycbc import scheme

        device = getattr(scheme.mgr.state, 'jax_device', None)
    pipeline = JAXLiveResultPipeline(
        event_manager.comm, queue_depth=queue_depth, device=device
    )
    event_manager._jax_result_transport = pipeline
    error = None
    try:
        yield pipeline
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            pipeline.close()
        except BaseException as exc:
            if error is None:
                raise
            note = 'JAX Live result cleanup also failed: ' + repr(exc)
            if hasattr(error, 'add_note'):
                error.add_note(note)
            else:
                logging.getLogger(__name__).error(note)
        finally:
            del event_manager._jax_result_transport


class JAXLiveResultPipeline:
    """Bound device snapshots by coordinator receipts, with ordered sends.

    A single worker owns both materialization and its MPI requests. Each job
    completes only after acknowledgement, so the queue bounds all retained
    device results as well as serialized host buffers. Science state remains
    on the calling thread. ``drain`` includes snapshots, sends and receipts.
    """

    def __init__(self, comm, queue_depth=None, *, device=None):
        self._owner = threading.get_ident()
        self._transport = JAXLiveResultTransport(
            comm, queue_depth=queue_depth, device=device
        )
        self.rank = self._transport.rank
        self.size = self._transport.size
        self._device = device
        self._jobs = deque()
        self._closed = False
        self._error = None
        self._abort_requested = False
        mpi = self._transport._mpi
        query_thread = getattr(mpi, 'Query_thread', lambda: None)
        multiple = getattr(mpi, 'THREAD_MULTIPLE', object())
        gpu = getattr(device, 'platform', None) in ('gpu', 'cuda')
        self._executor = None
        if self.rank > 0 and gpu and query_thread() == multiple:
            try:
                self._executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix='jax-live-handoff'
                )
                self._executor.submit(self._bind_worker).result()
            except BaseException as exc:
                if self._executor is not None:
                    self._executor.shutdown(wait=True)
                self._fail_handoff(exc)
                self._transport.close()
                self._closed = True
                raise

    def _bind_worker(self):
        self._transport._thread = threading.get_ident()

    @property
    def queue_depth(self):
        return self._transport.queue_depth

    @property
    def pending_results(self):
        return (
            len(self._jobs)
            if self._executor is not None
            else self._transport.pending_results
        )

    @property
    def pending_bytes(self):
        return self._transport.pending_bytes

    @property
    def asynchronous(self):
        return self._executor is not None

    def _check(self, allow_failed=False):
        if threading.get_ident() != self._owner:
            raise RuntimeError(
                'JAX Live result pipeline used from another thread'
            )
        if self._closed:
            raise RuntimeError('JAX Live result pipeline is closed')
        if self._error is not None and not allow_failed:
            raise RuntimeError(
                'JAX Live result pipeline failed'
            ) from self._error

    def _wait_job(self):
        job = self._jobs[0]
        try:
            job.result()
        except BaseException as exc:
            self._error = exc
            raise
        finally:
            self._jobs.popleft()

    def _fail_handoff(self, exc):
        """Terminate peers that cannot finish a missing or rejected cohort.

        An asynchronous failure bypasses mpi4py's main-thread exception
        handler. Continuing to drain or free would leave another rank waiting
        for a result or receipt that will never arrive. Abort only the isolated
        Live communicator after reporting the original failure. Test transports
        without Abort retain their ordinary exception and cleanup contract.
        """
        self._error = exc
        abort = getattr(self._transport.comm, 'Abort', None)
        if abort is None or self._abort_requested:
            return
        self._abort_requested = True
        try:
            logging.getLogger(__name__).error(
                'Fatal JAX Live result handoff failure on rank %s',
                self.rank,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
        finally:
            abort(1)

    def _handoff(self, payload):
        if self._error is not None:
            raise RuntimeError(
                'JAX Live result pipeline failed'
            ) from self._error
        try:
            host_payload = _materialize_payload(payload, self._device)
            self._transport.commit(host_payload)
            self._transport.drain()
        except BaseException as exc:
            self._fail_handoff(exc)
            raise

    def commit(self, payload):
        self._check()
        if self.rank == 0:
            raise RuntimeError('Only filtering ranks can commit Live results')
        if self._executor is None:
            try:
                self._transport.commit(
                    _materialize_payload(payload, self._device)
                )
            except BaseException as exc:
                self._fail_handoff(exc)
                raise
            return
        while self._jobs and self._jobs[0].done():
            self._wait_job()
        if len(self._jobs) >= self.queue_depth:
            with host_stage('mpi_snapshot_wait', queue_depth=self.queue_depth):
                self._wait_job()
        self._jobs.append(
            self._executor.submit(self._handoff, _own_payload(payload))
        )

    def gather(self):
        self._check()
        try:
            return self._transport.gather()
        except BaseException as exc:
            self._fail_handoff(exc)
            raise

    def drain(self):
        self._check(allow_failed=True)
        if self._executor is None:
            try:
                self._transport.drain()
            except BaseException as exc:
                self._fail_handoff(exc)
                raise
        else:
            while self._jobs:
                self._wait_job()

    def _close_transport(self):
        try:
            self._transport.close()
        except BaseException as exc:
            self._fail_handoff(exc)
            raise

    def close(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError(
                'JAX Live result pipeline used from another thread'
            )
        if self._closed:
            return
        error = None
        try:
            self.drain()
        except BaseException as exc:
            error = exc
        try:
            if self._executor is None:
                self._close_transport()
            else:
                self._executor.submit(self._close_transport).result()
        except BaseException as exc:
            if error is None:
                error = exc
            else:
                note = 'JAX Live communicator cleanup also failed: ' + repr(
                    exc
                )
                if hasattr(error, 'add_note'):
                    error.add_note(note)
                else:
                    logging.getLogger(__name__).error(note)
        finally:
            try:
                if self._executor is not None:
                    self._executor.shutdown(wait=True)
            finally:
                self._jobs.clear()
                self._closed = True
        if error is not None:
            raise error
