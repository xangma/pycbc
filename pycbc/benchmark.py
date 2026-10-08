"""Optional timing markers for JAX profiling."""
from contextlib import contextmanager
import json
import os
import sys
import time


@contextmanager
def host_stage(stage, **metadata):
    """Time host work without waiting for outstanding device work."""
    if os.environ.get('PYCBC_BENCHMARK_STAGES') != '1':
        yield
        return
    stage_event(stage, 'start', synchronize=False, **metadata)
    try:
        yield
    finally:
        stage_event(stage, 'end', synchronize=False, **metadata)


def _mpi_rank():
    """Return an MPI rank from launcher environment without importing MPI."""
    for name in ('OMPI_COMM_WORLD_RANK', 'PMI_RANK', 'PMIX_RANK'):
        value = os.environ.get(name)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                pass
    return None


def synchronize_jax_stage():
    """Fence live backend arrays at opted-in profile stage boundaries.

    This includes outstanding prefetch work on the active backend. The effect
    barrier alone does not wait for ordinary asynchronous array computations.
    """
    if os.environ.get('PYCBC_BENCHMARK_SYNCHRONIZE_STAGES') != '1':
        return False
    try:
        from pycbc import scheme
        if not isinstance(scheme.mgr.state, scheme.JAXScheme):
            return False
        import jax
        jax.block_until_ready(jax.live_arrays(
            platform=scheme.mgr.state.jax_device.platform))
        jax.effects_barrier()
    except (AttributeError, ImportError):
        return False
    return True


def stage_event(stage, event, *, synchronize=True, **metadata):
    """Emit stage boundaries, optionally synchronized for profiles."""
    if os.environ.get('PYCBC_BENCHMARK_STAGES') == '1':
        synchronized = (synchronize and event == 'end'
                        and synchronize_jax_stage())
        rank = _mpi_rank()
        record = dict(
            stage=stage, event=event, label=stage.replace('_', ' '),
            pid=os.getpid(), monotonic_ns=time.monotonic_ns(), **metadata)
        if rank is not None:
            record['rank'] = rank
        if synchronized:
            record['synchronized'] = True
        print('PYCBC_STAGE_EVENT ' + json.dumps(record), file=sys.stderr,
              flush=True)
