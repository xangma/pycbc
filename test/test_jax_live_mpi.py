"""Ordered MPI handoff, backpressure and cleanup for JAX Live results."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

_MODULE = Path(__file__).parents[1] / "pycbc/events/live_mpi_jax.py"
_SPEC = importlib.util.spec_from_file_location("live_mpi_jax", _MODULE)
_TRANSPORT_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TRANSPORT_MODULE)
JAXLiveResultTransport = _TRANSPORT_MODULE.JAXLiveResultTransport
host_stage = _TRANSPORT_MODULE.host_stage


class _Request:
    def __init__(self, events, name, complete=None):
        self.events, self.name, self.complete = events, name, complete

    def Wait(self):
        self.events.append(self.name + ":wait")
        if self.complete is not None:
            self.complete()

    def Cancel(self):
        self.events.append(self.name + ":cancel")
        self.complete = None


class _Status:
    def Get_count(self, datatype):
        assert datatype is _FAKE_MPI.BYTE
        return self.count

    def Get_source(self):
        return self.source

    def Get_tag(self):
        return self.tag


_FAKE_MPI = SimpleNamespace(BYTE=object(), Status=_Status, pickle=pickle)


@pytest.fixture(autouse=True)
def _fake_mpi(monkeypatch):
    monkeypatch.setattr(_TRANSPORT_MODULE, '_MPI', _FAKE_MPI)
    monkeypatch.delenv('PYCBC_JAX_LIVE_RESULT_QUEUE_DEPTH', raising=False)
    monkeypatch.delenv('PYCBC_BENCHMARK_STAGES', raising=False)


class _Comm:
    def __init__(self, rank=1, size=3, incoming=None):
        self.rank, self.size = rank, size
        self.events, self.sent, self.acks = [], [], []
        self.incoming = incoming or {}
        self.ack_sequence = None
        self.send_error = None
        self.receive_error = None

    def Dup(self):
        self.events.append("dup")
        return self

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def Irecv(self, buffer, source, tag):
        self.events.append(("ack:post", source, tag))
        sequence = len(self.sent) + 1

        def complete():
            buffer[0] = (
                sequence if self.ack_sequence is None else self.ack_sequence
            )

        return _Request(self.events, "ack", complete)

    def Isend(self, buffer, dest, tag):
        self.events.append(("result:post", dest, tag))
        if self.send_error is not None:
            raise self.send_error
        snapshot, datatype = buffer
        assert datatype is _FAKE_MPI.BYTE and isinstance(snapshot, bytes)
        self.sent.append(pickle.loads(snapshot))
        return _Request(self.events, "result")

    def Probe(self, source, tag, status):
        self.events.append(("result:probe", source, tag))
        snapshot = self.incoming[source][0]
        status.count, status.source, status.tag = len(snapshot), source, tag

    def Recv(self, buffer, source, tag):
        self.events.append(("result:recv", source, tag))
        if self.receive_error is not None:
            raise self.receive_error
        snapshot, datatype = buffer
        assert datatype is _FAKE_MPI.BYTE
        snapshot[:] = self.incoming[source].pop(0)

    def Send(self, buffer, dest, tag):
        assert buffer.dtype == np.dtype("int64") and buffer.shape == (1,)
        self.events.append(("ack:send", dest, tag))
        self.acks.append((dest, int(buffer[0])))

    def Free(self):
        self.events.append("free")


def _incoming(envelopes):
    return {
        rank: [pickle.dumps(envelope) for envelope in sequence]
        for rank, sequence in envelopes.items()
    }


def test_worker_waits_for_receipt_before_posting_second_result():
    comm = _Comm()
    transport = JAXLiveResultTransport(comm)
    payload = ({"H1": {"snr": np.ones(4, dtype=np.float32)}}, 100.0)
    transport.commit(payload)
    assert comm.events == ["dup", ("ack:post", 0, 1), ("result:post", 0, 0)]
    assert transport.queue_depth == transport.pending_results == 1
    assert transport.pending_bytes > 0
    assert transport._pending[0].snapshot is not payload
    transport.commit(({"H1": False}, 108.0))
    assert comm.events[3:] == [
        "ack:wait",
        "result:wait",
        ("ack:post", 0, 1),
        ("result:post", 0, 0),
    ]
    assert [envelope[0] for envelope in comm.sent] == [1, 2]
    transport.drain()
    assert transport.pending_results == transport.pending_bytes == 0
    transport.close()
    transport.close()
    assert comm.events.count("free") == 1


def test_root_preserves_rank_order_empty_and_invalid_detector_payloads():
    first = ({"H1": False, "L1": {}}, 100.0)
    second = ({"H1": {"snr": np.empty(0, dtype=np.float32)}}, 100.0)
    comm = _Comm(
        rank=0, incoming=_incoming({1: [(1, first)], 2: [(1, second)]})
    )
    transport = JAXLiveResultTransport(comm)
    gathered = transport.gather()
    assert gathered[0] is None
    assert gathered[1] == first
    assert gathered[2][1] == second[1]
    assert gathered[2][0]['H1']['snr'].shape == (0,)
    assert comm.acks == [(1, 1), (2, 1)]
    assert comm.events[1:] == [
        ("result:probe", 1, 0),
        ("result:recv", 1, 0),
        ("ack:send", 1, 1),
        ("result:probe", 2, 0),
        ("result:recv", 2, 0),
        ("ack:send", 2, 1),
    ]
    transport.close()


@pytest.mark.parametrize(
    "envelope", [(2, {}), (0, {}), (1,), "invalid", (True, {})]
)
def test_root_rejects_bad_sequence_and_releases_waiting_worker(envelope):
    comm = _Comm(rank=0, size=2, incoming=_incoming({1: [envelope]}))
    transport = JAXLiveResultTransport(comm)
    with pytest.raises(RuntimeError, match="sequence is out of order"):
        transport.gather()
    assert comm.acks == [(1, -1)]
    assert transport._received == [0, 0]
    transport.close()


def test_worker_rejects_wrong_ack_after_completing_requests():
    comm = _Comm()
    transport = JAXLiveResultTransport(comm)
    transport.commit(({}, 100.0))
    comm.ack_sequence = -1
    with pytest.raises(RuntimeError, match="acknowledgement is out of order"):
        transport.drain()
    assert comm.events[-2:] == ["ack:wait", "result:wait"]
    assert transport.pending_results == transport.pending_bytes == 0
    transport.close()


def test_send_failure_cancels_preposted_ack():
    comm = _Comm()
    transport = JAXLiveResultTransport(comm)
    comm.send_error = ValueError("cannot post")
    with pytest.raises(ValueError, match="cannot post"):
        transport.commit(({}, 100.0))
    assert comm.events[-2:] == ["ack:cancel", "ack:wait"]
    assert transport.pending_results == transport._sequence == 0
    transport.close()


def test_rank_thread_and_closed_lifecycle_contracts():
    worker = JAXLiveResultTransport(_Comm())
    root = JAXLiveResultTransport(_Comm(rank=0))
    with pytest.raises(RuntimeError, match="Only the coordinator"):
        worker.gather()
    with pytest.raises(RuntimeError, match="Only filtering ranks"):
        root.commit({})
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(RuntimeError, match="another thread"):
            executor.submit(worker.drain).result()
    worker.close()
    root.close()
    for method in (worker.drain, worker.gather):
        with pytest.raises(RuntimeError, match="closed"):
            method()


@pytest.mark.parametrize('depth', [1, 2, 4])
def test_queue_is_bounded_by_receipt_not_eager_send_completion(depth):
    comm = _Comm()
    transport = JAXLiveResultTransport(comm, queue_depth=depth)
    for sequence in range(depth):
        transport.commit({'sequence': sequence})
    assert transport.pending_results == depth
    assert not any(event == 'ack:wait' for event in comm.events)
    transport.commit({'sequence': depth})
    assert transport.pending_results == depth
    assert comm.events.count('ack:wait') == 1
    assert comm.events.count('result:wait') == 1
    assert [item[0] for item in comm.sent] == list(range(1, depth + 2))
    transport.drain()
    assert comm.events.count('ack:wait') == depth + 1
    assert transport.pending_results == transport.pending_bytes == 0
    transport.close()


@pytest.mark.parametrize('depth', [1, 2, 4])
def test_host_reuse_does_not_mutate_outstanding_payloads(depth):
    comm = _Comm()
    transport = JAXLiveResultTransport(comm, queue_depth=depth)
    array = np.arange(32, dtype=np.float32)
    payload = ({'H1': {'snr': array, 'view': array[::2]}}, 100.0)
    for sequence in range(depth + 2):
        array[:] = sequence
        transport.commit(payload)
        array[:] = -999
    for index, envelope in enumerate(comm.sent):
        columns = envelope[1][0]['H1']
        np.testing.assert_array_equal(columns['snr'], np.full(32, index))
        np.testing.assert_array_equal(columns['view'], np.full(16, index))
        assert columns['snr'].dtype == columns['view'].dtype == np.float32
    payload[0].clear()
    transport.close()


@pytest.mark.parametrize('depth', [False, True, 0, -1, 5, '2', 1.0])
def test_invalid_depth_is_rejected_before_communicator_duplication(depth):
    comm = _Comm()
    with pytest.raises(ValueError, match='queue depth'):
        JAXLiveResultTransport(comm, queue_depth=depth)
    assert comm.events == []


@pytest.mark.parametrize(
    'platform, expected',
    [
        ('gpu', 4),
        ('cuda', 4),
        ('cpu', 1),
        ('unknown', 1),
        (None, 1),
    ],
)
def test_queue_depth_device_default(monkeypatch, platform, expected):
    monkeypatch.delenv('PYCBC_JAX_LIVE_RESULT_QUEUE_DEPTH', raising=False)
    device = (
        SimpleNamespace(platform=platform) if platform is not None else None
    )
    transport = JAXLiveResultTransport(_Comm(), device=device)
    assert transport.queue_depth == expected
    transport.close()


@pytest.mark.parametrize('platform', ['gpu', 'cuda', 'cpu'])
def test_queue_depth_environment_and_explicit_override(monkeypatch, platform):
    device = SimpleNamespace(platform=platform)
    monkeypatch.setenv('PYCBC_JAX_LIVE_RESULT_QUEUE_DEPTH', '2')
    transport = JAXLiveResultTransport(_Comm(), device=device)
    assert transport.queue_depth == 2
    transport.close()
    transport = JAXLiveResultTransport(_Comm(), queue_depth=1, device=device)
    assert transport.queue_depth == 1
    transport.close()
    monkeypatch.setenv('PYCBC_JAX_LIVE_RESULT_QUEUE_DEPTH', 'invalid')
    transport = JAXLiveResultTransport(_Comm(), queue_depth=2, device=device)
    assert transport.queue_depth == 2
    transport.close()
    for raw_depth in ('', 'invalid', '2.0', '-1', '0', '5'):
        monkeypatch.setenv('PYCBC_JAX_LIVE_RESULT_QUEUE_DEPTH', raw_depth)
        with pytest.raises(ValueError, match='queue depth'):
            JAXLiveResultTransport(_Comm(), device=device)


def test_serialization_failure_leaves_existing_requests_and_sequence_intact():
    transport = JAXLiveResultTransport(_Comm(), queue_depth=2)
    transport.commit({'valid': np.arange(16)})
    original_bytes = transport.pending_bytes
    with pytest.raises((AttributeError, pickle.PicklingError)):
        transport.commit({'bad': lambda: None})
    assert transport.pending_results == transport._sequence == 1
    assert transport.pending_bytes == original_bytes
    transport.commit({'next': np.arange(3)})
    assert transport.pending_results == 2
    transport.close()


def test_post_failure_preserves_older_request_and_retries_same_sequence():
    comm = _Comm()
    transport = JAXLiveResultTransport(comm, queue_depth=2)
    transport.commit({'valid': 1})
    original_bytes = transport.pending_bytes
    comm.send_error = ValueError('cannot post')
    with pytest.raises(ValueError, match='cannot post'):
        transport.commit({'next': 2})
    assert transport.pending_results == transport._sequence == 1
    assert transport.pending_bytes == original_bytes
    assert comm.events[-2:] == ['ack:cancel', 'ack:wait']
    comm.send_error = None
    transport.commit({'next': 2})
    assert [item[0] for item in comm.sent] == [1, 2]
    transport.close()


def test_post_and_cancel_failure_retains_preposted_ack_buffer():
    comm = _Comm()
    transport = JAXLiveResultTransport(comm)
    original_receive = comm.Irecv
    requests = []

    def receive(*args, **kwargs):
        request = original_receive(*args, **kwargs)
        original_cancel = request.Cancel

        def cancel():
            if not requests:
                requests.append(request)
                raise RuntimeError('cannot cancel')
            original_cancel()

        request.Cancel = cancel
        return request

    comm.Irecv = receive
    comm.send_error = RuntimeError('cannot post')
    with pytest.raises(RuntimeError, match='cannot cancel'):
        transport.commit({'snr': np.arange(1024)})
    assert transport.pending_results == 1 and transport.pending_bytes > 0
    assert transport._pending[0].receive is requests[0]
    assert transport._pending[0].send is None
    assert transport._sequence == 0
    with pytest.raises(RuntimeError, match='transport failed'):
        transport.commit({})
    transport.close()
    assert transport.pending_results == transport.pending_bytes == 0
    assert comm.events[-3:] == ['ack:cancel', 'ack:wait', 'free']


def test_bad_ack_retains_later_owned_requests_and_blocks_new_commits():
    comm = _Comm()
    transport = JAXLiveResultTransport(comm, queue_depth=2)
    transport.commit({'first': 1})
    transport.commit({'second': 2})
    comm.ack_sequence = -1
    with pytest.raises(RuntimeError, match='acknowledgement'):
        transport.drain()
    assert transport.pending_results == 1 and transport.pending_bytes > 0
    with pytest.raises(RuntimeError, match='transport failed'):
        transport.commit({'third': 3})
    comm.ack_sequence = None
    transport.drain()
    assert transport.pending_results == transport.pending_bytes == 0
    transport.close()


def test_root_unpickle_failure_acknowledges_other_workers_before_raising():
    comm = _Comm(rank=0, incoming={1: [b''], 2: [pickle.dumps((1, {}))]})
    transport = JAXLiveResultTransport(comm)
    with pytest.raises(RuntimeError, match='cannot be unpickled'):
        transport.gather()
    assert comm.acks == [(1, -1), (2, 1)]
    assert transport._received == [0, 0, 1]
    with pytest.raises(RuntimeError, match='transport failed'):
        transport.gather()
    transport.close()


def test_request_failure_does_not_free_owned_buffers():
    comm = _Comm()
    transport = JAXLiveResultTransport(comm)
    transport.commit({'snr': np.arange(1024)})

    def fail():
        raise RuntimeError('request failed')

    transport._pending[0].send.complete = fail
    with pytest.raises(RuntimeError, match='request failed'):
        transport.close()
    assert transport.pending_results == 1 and transport.pending_bytes > 0
    assert not transport._closed and 'free' not in comm.events
    transport._pending[0].send.complete = None
    transport.close()
    assert comm.events[-1] == 'free'


def test_host_profile_stages_are_paired_and_record_queue_state(monkeypatch):
    records = []

    def emit(stage, event, **metadata):
        records.append((stage, event, metadata))

    monkeypatch.setenv('PYCBC_BENCHMARK_STAGES', '1')
    monkeypatch.setitem(
        sys.modules, 'pycbc.benchmark', SimpleNamespace(stage_event=emit)
    )
    worker = JAXLiveResultTransport(_Comm(), queue_depth=2)
    worker.commit({'snr': np.arange(10)})
    worker.close()
    root = JAXLiveResultTransport(
        _Comm(
            rank=0,
            size=2,
            incoming=_incoming({1: [(1, {'snr': np.arange(10)})]}),
        )
    )
    root.gather()
    root.close()
    assert {stage for stage, _, _ in records} == {
        'mpi_ack_wait',
        'mpi_send_completion',
        'mpi_serialize',
        'mpi_post',
        'mpi_receive',
        'mpi_unpickle',
        'mpi_ack_send',
    }
    for start, end in zip(records[::2], records[1::2]):
        assert start[0] == end[0]
        assert start[1] == 'start' and end[1] == 'end'
        assert start[2]['sequence'] == end[2]['sequence'] == 1
    assert all(record[2]['synchronize'] is False for record in records)
    assert all(
        'pending_results' in record[2] and 'pending_bytes' in record[2]
        for record in records
    )
    assert (
        next(
            record[2]
            for record in records
            if record[:2] == ('mpi_post', 'end')
        )['pending_results']
        == 1
    )


def test_host_stage_balances_failure_and_keeps_profiling_optional(monkeypatch):
    records = []

    def emit(stage, event, **metadata):
        records.append((stage, event, metadata))

    monkeypatch.setitem(
        sys.modules, 'pycbc.benchmark', SimpleNamespace(stage_event=emit)
    )
    with host_stage('coincidence', sequence=3):
        pass
    assert records == []
    monkeypatch.setenv('PYCBC_BENCHMARK_STAGES', '1')
    with pytest.raises(ValueError, match='science failed'):
        with host_stage('coincidence', sequence=3):
            raise ValueError('science failed')
    assert records == [
        ('coincidence', 'start', {'synchronize': False, 'sequence': 3}),
        ('coincidence', 'end', {'synchronize': False, 'sequence': 3}),
    ]


def test_backend_and_disabled_host_profiler_do_not_import_jax():
    # The byte transport needs MPI, but no JAX import is necessary to load its
    # module or use the disabled host-only profiler in CPU tooling.
    code = '''
import importlib.abc
import importlib.util
import os
import sys
class NoJAX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "jax" or fullname.startswith("jax."):
            raise RuntimeError("Unexpected JAX dependency")
sys.meta_path.insert(0, NoJAX())
os.environ.pop("PYCBC_BENCHMARK_STAGES", None)
spec = importlib.util.spec_from_file_location("transport", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
with module.host_stage("coincidence"):
    pass
assert "jax" not in sys.modules
assert "mpi4py" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, '-c', code, str(_MODULE)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _payload(block, rank):
    if block % 3 == 0:
        detector = False
    elif block % 3 == 1:
        detector = {
            "snr": np.empty(0, dtype=np.float32),
            "template_id": np.empty(0, dtype=np.int64),
        }
    else:
        # Exceed object-Irecv defaults and ordinary eager-message limits.
        detector = {
            "snr": np.arange(100000, dtype=np.float32) + rank,
            "template_id": np.arange(100000, dtype=np.int64) + block,
            "label": np.array(["H1", "L1"]),
        }
    return {"H1": detector}, 100.0 + block * 8


def _check_payload(got, want):
    assert got[1] == want[1]
    if want[0]["H1"] is False:
        assert got[0]["H1"] is False
    else:
        assert set(got[0]["H1"]) == set(want[0]["H1"])
        for key, column in want[0]["H1"].items():
            actual = got[0]["H1"][key]
            assert actual.dtype == column.dtype
            assert actual.shape == column.shape
            assert actual.tobytes() == column.tobytes()


def _mutate_payload(payload):
    detector = payload[0]['H1']
    if isinstance(detector, dict):
        for column in detector.values():
            column[:] = '-1' if column.dtype.kind in 'SU' else -1
    payload[0].clear()


def _mpi_worker(depth):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 3
    transport = JAXLiveResultTransport(comm, queue_depth=depth)
    total_blocks = 12
    markers = (depth - 1, depth, depth + 1)
    try:
        if comm.Get_rank() > 0:
            controls = []
            # Existing pipeline messages using the same tags stay isolated
            # from this transport's result and acknowledgement channels.
            noise = np.array([comm.Get_rank()], dtype=np.int64)
            controls.append((noise, comm.Isend(noise, dest=0, tag=0)))
            world_ack = np.empty(1, dtype=np.int64)
            original_ack = comm.Irecv(world_ack, source=0, tag=1)
            for block in range(total_blocks):
                payload = _payload(block, comm.Get_rank())
                transport.commit(payload)
                _mutate_payload(payload)
                if block in markers:
                    marker = np.array([block], dtype=np.int64)
                    controls.append(
                        (marker, comm.Isend(marker, dest=0, tag=200 + block))
                    )
                assert transport.pending_results <= depth
            transport.drain()
            assert transport.pending_results == transport.pending_bytes == 0
            for marker, request in controls:
                request.Wait()
            original_ack.Wait()
            assert world_ack[0] == 101
        else:
            # Workers can fill the queue before any receipt. An additional
            # commit remains blocked even if the MPI implementation sends
            # every small result eagerly.
            for rank in (1, 2):
                noise = np.empty(1, dtype=np.int64)
                comm.Recv(noise, source=rank, tag=0)
                assert noise[0] == rank
                comm.Send(np.array([101], dtype=np.int64), dest=rank, tag=1)
                marker = np.empty(1, dtype=np.int64)
                comm.Recv(marker, source=rank, tag=200 + depth - 1)
                assert marker[0] == depth - 1
            until = time.monotonic() + 0.05
            while time.monotonic() < until:
                assert not comm.Iprobe(source=MPI.ANY_SOURCE, tag=200 + depth)
            for block in range(total_blocks):
                gathered = transport.gather()
                assert len(gathered) == 3 and gathered[0] is None
                for rank in (1, 2):
                    _check_payload(gathered[rank], _payload(block, rank))
                if block == 0:
                    # One coordinator receipt admits exactly one new block.
                    for rank in (1, 2):
                        marker = np.empty(1, dtype=np.int64)
                        comm.Recv(marker, source=rank, tag=200 + depth)
                        assert marker[0] == depth
                    until = time.monotonic() + 0.05
                    while time.monotonic() < until:
                        assert not comm.Iprobe(
                            source=MPI.ANY_SOURCE, tag=201 + depth
                        )
                elif block == 1:
                    for rank in (1, 2):
                        marker = np.empty(1, dtype=np.int64)
                        comm.Recv(marker, source=rank, tag=201 + depth)
                        assert marker[0] == depth + 1
                if block == 6:
                    time.sleep(0.025)  # Slow coordinator output is bounded.
        transport.close()
        transport.close()
        completed = comm.gather(True, root=0)
        if comm.Get_rank() == 0:
            receipt = {
                "ranks": len(completed),
                "blocks": total_blocks,
                "depth": depth,
                "bounded_overlap": True,
                "exact_payloads": True,
                "host_reuse": True,
            }
            print(json.dumps(receipt), flush=True)
    except BaseException:
        import traceback

        traceback.print_exc()
        comm.Abort(1)
        raise


def _launch_native(arguments):
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    if launcher is None or importlib.util.find_spec("mpi4py") is None:
        pytest.skip("MPI launcher and mpi4py are required")
    if int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1")) > 1:
        pytest.skip("nested MPI launch")
    environment = dict(
        os.environ,
        OMPI_ALLOW_RUN_AS_ROOT="1",
        OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1",
    )
    return subprocess.run(
        [
            launcher,
            "-n",
            "3",
            sys.executable,
            str(Path(__file__).resolve()),
            *arguments,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )


@pytest.mark.parametrize('depth', [1, 2, 4])
def test_native_three_rank_large_payload_and_bounded_overlap(depth):
    result = _launch_native(['--mpi-worker', str(depth)])
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    assert receipt == {
        "ranks": 3,
        "blocks": 12,
        "depth": depth,
        "bounded_overlap": True,
        "exact_payloads": True,
        "host_reuse": True,
    }


def _mpi_bad_worker(kind):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 3
    transport = JAXLiveResultTransport(comm)
    try:
        if comm.Get_rank() == 1:
            if kind == 'sequence':
                transport._sequence = 1
            else:
                transport._mpi = SimpleNamespace(
                    BYTE=MPI.BYTE, pickle=SimpleNamespace(dumps=lambda _: b'')
                )
        if comm.Get_rank() > 0:
            transport.commit(_payload(2, comm.Get_rank()))
            if comm.Get_rank() == 1:
                try:
                    transport.drain()
                except RuntimeError as exc:
                    assert 'acknowledgement is out of order' in str(exc)
                else:
                    raise AssertionError('Malformed worker was accepted')
            else:
                transport.drain()
        else:
            try:
                transport.gather()
            except RuntimeError as exc:
                assert (
                    'sequence is out of order'
                    if kind == 'sequence'
                    else 'cannot be unpickled'
                ) in str(exc)
            else:
                raise AssertionError('Malformed block reached science state')
            assert transport._received == [0, 0, 1]
        transport.close()
        completed = comm.gather(True, root=0)
        if comm.Get_rank() == 0:
            print(
                json.dumps(
                    {
                        'ranks': len(completed),
                        'kind': kind,
                        'other_worker_released': True,
                    }
                ),
                flush=True,
            )
    except BaseException:
        import traceback

        traceback.print_exc()
        comm.Abort(1)
        raise


@pytest.mark.parametrize('kind', ['sequence', 'empty'])
def test_native_malformed_envelope_releases_other_rank_and_requests(kind):
    result = _launch_native(['--mpi-bad-worker', kind])
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        'ranks': 3,
        'kind': kind,
        'other_worker_released': True,
    }


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--mpi-worker':
        _mpi_worker(int(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == '--mpi-bad-worker':
        _mpi_bad_worker(sys.argv[2])
