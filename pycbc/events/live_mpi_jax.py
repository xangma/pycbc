# Copyright (C) 2026 The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Bounded, ordered host result handoff for the JAX Live MPI pipeline.

Each worker may leave a bounded number of results awaiting the coordinator
while it filters the next block. A receipt acknowledgement, rather than MPI
send completion, provides the bound: eager sends can otherwise queue arbitrary
numbers of blocks. Serialized snapshots own their buffers until both requests
complete, so reusing a host result array cannot change an outstanding block.
Only the pipeline thread uses MPI; conditioning and coordinator science state
retain their existing order and ownership.
"""

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import os
import threading

import numpy as np

_MPI = None


@contextmanager
def host_stage(stage, **metadata):
    """Profile coordinator host work without fencing outstanding GPU work."""
    if os.environ.get('PYCBC_BENCHMARK_STAGES') != '1':
        yield
        return
    from pycbc.benchmark import stage_event

    stage_event(stage, 'start', synchronize=False, **metadata)
    try:
        yield
    finally:
        stage_event(stage, 'end', synchronize=False, **metadata)


@dataclass
class _PendingResult:
    sequence: int
    snapshot: bytes
    acknowledgement: np.ndarray
    receive: object
    send: object


class JAXLiveResultTransport:
    """Ordered pending results on an isolated MPI communicator.

    Construct collectively on the supplied communicator.  ``commit`` accepts
    host results, ``gather`` returns the same rank-ordered list as MPI gather,
    and ``drain`` completes the final handoff before measurement ends.

    The default is four outstanding blocks on GPUs and one on CPU or when no
    device is supplied. Set
    ``PYCBC_JAX_LIVE_RESULT_QUEUE_DEPTH`` or pass ``queue_depth`` for depths
    1--4. The bound applies to unacknowledged blocks even if MPI eagerly copies
    their bytes. Coordinator science processing consumes one block at a time
    in rank order. Profiling uses optional host-only stage markers.
    """

    _RESULT_TAG = 0
    _ACK_TAG = 1
    _MAX_QUEUE_DEPTH = 4

    def __init__(self, comm, queue_depth=None, *, device=None):
        if queue_depth is None:
            default_depth = (
                4
                if getattr(device, 'platform', None) in ('gpu', 'cuda')
                else 1
            )
            raw_depth = os.environ.get(
                'PYCBC_JAX_LIVE_RESULT_QUEUE_DEPTH', str(default_depth)
            )
            try:
                queue_depth = int(raw_depth)
            except ValueError:
                raise ValueError("JAX Live result queue depth must be 1--4")
        if (
            type(queue_depth) is not int
            or not 1 <= queue_depth <= self._MAX_QUEUE_DEPTH
        ):
            raise ValueError("JAX Live result queue depth must be 1--4")
        global _MPI
        if _MPI is None:
            from mpi4py import MPI

            _MPI = MPI
        self._mpi = _MPI
        self._stage_event = None
        if os.environ.get('PYCBC_BENCHMARK_STAGES') == '1':
            from pycbc.benchmark import stage_event

            self._stage_event = stage_event
        self._thread = threading.get_ident()
        self.comm = comm.Dup()
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self._queue_depth = queue_depth
        self._sequence = 0
        self._received = [0] * self.size
        self._pending = deque()
        self._pending_bytes = 0
        self._closed = False
        self._error = None

    @property
    def queue_depth(self):
        return self._queue_depth

    @property
    def pending_results(self):
        return len(self._pending)

    @property
    def pending_bytes(self):
        return self._pending_bytes

    def _event(self, stage, event, **metadata):
        if self._stage_event is not None:
            self._stage_event(
                stage,
                event,
                synchronize=False,
                queue_depth=self.queue_depth,
                pending_results=self.pending_results,
                pending_bytes=self.pending_bytes,
                **metadata
            )

    def _check(self, allow_failed=False):
        if threading.get_ident() != self._thread:
            raise RuntimeError(
                "JAX Live result transport used from another thread"
            )
        if self._closed:
            raise RuntimeError("JAX Live result transport is closed")
        if self._error is not None and not allow_failed:
            raise RuntimeError(
                "JAX Live result transport failed"
            ) from self._error

    def _wait_pending(self):
        if not self._pending:
            return
        pending = self._pending[0]
        metadata = dict(
            sequence=pending.sequence,
            peer_rank=0,
            payload_bytes=len(pending.snapshot),
        )
        self._event('mpi_ack_wait', 'start', **metadata)
        try:
            if pending.send is None:
                # A failed send post may have left its ACK receive alive if
                # cancellation also failed. Keep its buffer until cancellation
                # completes; no result exists for the coordinator to receive.
                pending.receive.Cancel()
            pending.receive.Wait()
        except BaseException as exc:
            self._error = exc
            raise
        finally:
            self._event('mpi_ack_wait', 'end', **metadata)
        if pending.send is not None:
            self._event('mpi_send_completion', 'start', **metadata)
            try:
                pending.send.Wait()
            except BaseException as exc:
                self._error = exc
                raise
            finally:
                self._event('mpi_send_completion', 'end', **metadata)
        # Both requests are finished before releasing their owned buffers.
        self._pending.popleft()
        self._pending_bytes -= len(pending.snapshot)
        if (
            pending.send is not None
            and int(pending.acknowledgement[0]) != pending.sequence
        ):
            self._error = RuntimeError(
                "JAX Live result acknowledgement is out of order"
            )
            raise self._error

    def commit(self, host_payload):
        """Snapshot one worker block, waiting only when the queue is full."""
        self._check()
        if self.rank == 0:
            raise RuntimeError("Only filtering ranks can commit Live results")
        if len(self._pending) >= self.queue_depth:
            self._wait_pending()
        sequence = self._sequence + 1
        metadata = dict(sequence=sequence, peer_rank=0)
        self._event('mpi_serialize', 'start', **metadata)
        try:
            snapshot = self._mpi.pickle.dumps((sequence, host_payload))
            metadata['payload_bytes'] = len(snapshot)
        finally:
            self._event('mpi_serialize', 'end', **metadata)
        acknowledgement = np.empty(1, dtype=np.int64)
        self._event('mpi_post', 'start', **metadata)
        try:
            receive = self.comm.Irecv(
                acknowledgement, source=0, tag=self._ACK_TAG
            )
            pending = _PendingResult(
                sequence, snapshot, acknowledgement, receive, None
            )
            # Own the preposted receive before trying to post the send. MPI
            # cleanup can itself raise; its buffer must survive that failure.
            self._pending.append(pending)
            self._pending_bytes += len(snapshot)
            try:
                pending.send = self.comm.Isend(
                    [snapshot, self._mpi.BYTE], dest=0, tag=self._RESULT_TAG
                )
            except BaseException:
                try:
                    receive.Cancel()
                    receive.Wait()
                except BaseException as exc:
                    self._error = exc
                    raise
                self._pending.pop()
                self._pending_bytes -= len(snapshot)
                raise
        finally:
            self._event('mpi_post', 'end', **metadata)
        self._sequence = sequence

    def gather(self):
        """Receive one block in rank order without advancing science state."""
        self._check()
        if self.rank != 0:
            raise RuntimeError("Only the coordinator can gather Live results")
        gathered = [None]
        first_error = None
        for rank in range(1, self.size):
            expected = self._received[rank] + 1
            metadata = dict(sequence=expected, peer_rank=rank)
            # Probe the actual byte count; a fixed object-Irecv buffer can
            # truncate large trigger blocks. No worker changes its snapshot.
            status = self._mpi.Status()
            self._event('mpi_receive', 'start', **metadata)
            try:
                self.comm.Probe(
                    source=rank, tag=self._RESULT_TAG, status=status
                )
                count = status.Get_count(self._mpi.BYTE)
                if (
                    count < 0
                    or status.Get_source() != rank
                    or status.Get_tag() != self._RESULT_TAG
                ):
                    raise RuntimeError("JAX Live result message is invalid")
                snapshot = bytearray(count)
                self.comm.Recv(
                    [snapshot, self._mpi.BYTE],
                    source=rank,
                    tag=self._RESULT_TAG,
                )
                metadata['payload_bytes'] = count
            finally:
                self._event('mpi_receive', 'end', **metadata)
            self._event('mpi_unpickle', 'start', **metadata)
            try:
                envelope = self._mpi.pickle.loads(snapshot)
                valid = (
                    isinstance(envelope, tuple)
                    and len(envelope) == 2
                    and type(envelope[0]) is int
                    and envelope[0] == expected
                )
                error = (
                    None
                    if valid
                    else RuntimeError(
                        "JAX Live result sequence is out of order"
                    )
                )
            except Exception as exc:
                valid = False
                error = RuntimeError("JAX Live result cannot be unpickled")
                error.__cause__ = exc
            finally:
                self._event('mpi_unpickle', 'end', **metadata)
            acknowledgement = np.array(
                [expected if valid else -1], dtype=np.int64
            )
            # Every worker posts its ACK receive before its result send.
            self._event('mpi_ack_send', 'start', **metadata)
            try:
                self.comm.Send(acknowledgement, dest=rank, tag=self._ACK_TAG)
            finally:
                self._event('mpi_ack_send', 'end', **metadata)
            if valid:
                self._received[rank] = expected
                gathered.append(envelope[1])
            elif first_error is None:
                first_error = error
            # Release this block's other workers before reporting an invalid
            # envelope. No partial block is returned to coordinator science.
        if first_error is not None:
            self._error = first_error
            raise first_error
        return gathered

    def drain(self):
        """Complete an outstanding worker send and receipt acknowledgement."""
        self._check(allow_failed=True)
        while self._pending:
            self._wait_pending()

    def close(self):
        """Drain and free the communicator; repeated closes are safe."""
        if threading.get_ident() != self._thread:
            raise RuntimeError(
                "JAX Live result transport used from another thread"
            )
        if self._closed:
            return
        try:
            self.drain()
        except BaseException:
            # Do not free live request buffers after an MPI failure. If all
            # requests finished, communicator cleanup is still safe.
            if not self._pending:
                self.comm.Free()
                self._closed = True
            raise
        self.comm.Free()
        self._closed = True
