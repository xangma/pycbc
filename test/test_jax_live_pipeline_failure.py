"""Native MPI coverage for fatal handoff failures and ordinary termination.

Run with an MPI-enabled Python. Fake communicators cannot expose a coordinator
blocked in Probe, or another worker's live ACK wait after a failed cohort.
"""

import importlib.util
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

import test_jax_live_pipeline as pipeline_tests


def _launch(case, failed_rank, depth, platform='gpu'):
    launcher = shutil.which('mpiexec') or shutil.which('mpirun')
    if launcher is None or importlib.util.find_spec('mpi4py') is None:
        pytest.skip('native MPI launcher and mpi4py are required')
    environment = os.environ.copy()
    environment.update(
        OMPI_ALLOW_RUN_AS_ROOT='1', OMPI_ALLOW_RUN_AS_ROOT_CONFIRM='1'
    )
    environment.pop('PYCBC_BENCHMARK_STAGES', None)
    command = [
        launcher,
        '-n',
        '3',
        sys.executable,
        str(Path(__file__).resolve()),
        '--mpi-worker',
        case,
        str(failed_rank),
        str(depth),
        platform,
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env=environment,
    )
    try:
        output = process.communicate(timeout=15)[0]
    except subprocess.TimeoutExpired:
        # Kill only this test's launcher and ranks, then reap their output.
        os.killpg(process.pid, signal.SIGKILL)
        output = process.communicate()[0]
        pytest.fail(
            'MPI handoff did not terminate within 15 seconds:\n' + output
        )
    return process.returncode, output


@pytest.mark.parametrize('depth', [1, 4])
@pytest.mark.parametrize('failed_rank', [1, 2])
@pytest.mark.parametrize(
    'case', ['before_send', 'after_published', 'bad_sequence']
)
def test_native_fatal_handoff_terminates_all_ranks(case, failed_rank, depth):
    code, output = _launch(case, failed_rank, depth)
    assert code != 0, output
    assert 'Fatal JAX Live result handoff failure on rank ' in output
    expected = (
        'out of order'
        if case == 'bad_sequence'
        else f'injected snapshot failure rank={failed_rank}'
    )
    assert expected in output
    assert 'ALL_RANKS_FINISHED' not in output


def test_native_synchronous_snapshot_failure_also_terminates_peers():
    code, output = _launch('before_send', 2, 1, 'cpu')
    assert code != 0, output
    assert 'injected snapshot failure rank=2' in output


@pytest.mark.parametrize('depth', [1, 4])
def test_native_success_preserves_order_and_frees_all_ranks(depth):
    code, output = _launch('success', 0, depth)
    assert code == 0, output
    assert 'ALL_RANKS_FINISHED' in output
    assert 'Fatal JAX Live' not in output


def _mpi_worker(case, failed_rank, depth, platform):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    assert comm.Get_size() == 3
    if platform == 'gpu':
        assert MPI.Query_thread() == MPI.THREAD_MULTIPLE
    module = pipeline_tests.pipeline_module
    pipeline_tests.transport_tests._TRANSPORT_MODULE._MPI = MPI

    def materialize(payload, device):
        block = payload[1]
        fail_block = 1 if case == 'after_published' else 0
        if (
            rank == failed_rank
            and case != 'bad_sequence'
            and block == fail_block
        ):
            raise ValueError(
                f'injected snapshot failure rank={rank} block={block}'
            )
        return payload

    module._materialize_payload = materialize
    pipeline = module.JAXLiveResultPipeline(
        comm, queue_depth=depth, device=SimpleNamespace(platform=platform)
    )
    if case == 'bad_sequence' and rank == failed_rank:
        pipeline._transport._sequence = 1
    # Queued blocks cover a healthy worker already posting a later cohort when
    # root detects the failed one. Large messages require live owned requests.
    try:
        if rank:
            for block in range(3):
                pipeline.commit(
                    ({'rank': rank, 'data': bytes(1024 * 1024)}, block)
                )
            pipeline.drain()
        else:
            for block in range(3):
                got = pipeline.gather()
                assert got[0] is None
                for peer in (1, 2):
                    assert got[peer][1] == block
                    assert got[peer][0]['rank'] == peer
                    assert got[peer][0]['data'] == bytes(1024 * 1024)
        assert pipeline.pending_results == pipeline.pending_bytes == 0
    finally:
        pipeline.close()
    comm.Barrier()
    if rank == 0:
        print('ALL_RANKS_FINISHED', flush=True)


if __name__ == '__main__' and sys.argv[1:2] == ['--mpi-worker']:
    _mpi_worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
