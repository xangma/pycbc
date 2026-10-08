# Copyright (C) 2026 The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Owned host output snapshots and bounded HDF writing for JAX Live.

Device readback happens once at the terminal writer boundary. Queued snapshots
retain immutable device arrays and owned host metadata, preserving the HDF
schema and PSD compression settings. Completion is a timing and lifecycle
boundary: submitting an output does not imply that its file is complete.
"""

from contextlib import contextmanager
from dataclasses import dataclass, replace
import os
import queue
import threading
import logging

import h5py
import jax
import numpy as np


@contextmanager
def _output_stage(name):
    """Expose readback/write overlap without adding synchronization fences."""
    if os.environ.get('PYCBC_BENCHMARK_STAGES') != '1':
        yield
        return
    from pycbc.benchmark import stage_event

    stage_event(name, 'start', synchronize=False)
    try:
        yield
    finally:
        stage_event(name, 'end', synchronize=False)


def _own_host_value(value):
    """Detach mutable host leaves without changing scalar serialization."""
    if isinstance(value, np.ndarray):
        value = value.copy()
        value.flags.writeable = False
        return value
    if isinstance(value, dict):
        return {key: _own_host_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_own_host_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_own_host_value(item) for item in value)
    return value


def _hdf_value(value):
    # Match LiveEventManager.dump's existing Unicode conversion, including
    # metadata scalars; converting these to zero-dimensional arrays changes
    # their HDF representation.
    if hasattr(value, 'dtype') and value.dtype.kind == 'U':
        return [item.encode() for item in value]
    return value


def _loudest_indices(snr, chisq, count):
    from pycbc.types.array_jax import _reference_enabled

    if _reference_enabled('live_output_selection'):
        from pycbc.events import ranking

        # Ranking retains its independent numerical control; this option
        # restores only the original output's NumPy sorting and index union.
        nsnr = ranking.newsnr(snr, chisq) if len(snr) else []
        snr, nsnr = jax.device_get((snr, nsnr))
        return np.union1d(
            np.argsort(nsnr)[::-1][:count], np.argsort(snr)[::-1][:count]
        )
    from pycbc.events.eventmgr_jax import loudest_event_indices

    return loudest_event_indices(snr, chisq, count)


def collect_live_background_jax(values, count=None):
    """Collect background values with the original top-N output control.

    The default partitions on device. ``live_output_selection`` restores
    NumPy's original partition order; membership alone does not establish
    equal HDF columns. Full-background collection preserves all input order.
    """
    if count and count < len(values) - 1:
        assert count > 0, 'We can only store positive int loudest triggers.'
        from pycbc.types.array_jax import _reference_enabled

        if _reference_enabled('live_output_selection'):
            host = np.asarray(jax.device_get(values))
            return -np.partition(-host, count)[:count]
        import jax.numpy as jnp

        values = -jnp.partition(-jnp.asarray(values), count)[:count]
    return jax.device_get(values)


_GATE_DTYPE = [
    ('center_time', float),
    ('zero_half_width', float),
    ('taper_width', float),
]


def _snapshot_gate_values(values):
    """Retain the explicit immutable resident-gate snapshot protocol."""
    if callable(getattr(values, 'snapshot', None)) and callable(
        getattr(values, 'materialize', None)
    ):
        return values.snapshot()
    return _own_host_value(np.array(values, dtype=_GATE_DTYPE))


@dataclass(frozen=True)
class LiveOutputSnapshot:
    """A block's output data, independent of later pipeline mutation."""

    filename: str
    attributes: dict
    results: dict
    raw_results: dict
    loudest: dict
    gates: dict
    psds: dict
    psd_attributes: dict
    device_pending: bool = False


def snapshot_live_output(
    filename,
    attributes,
    results,
    raw_results=None,
    store_psd=False,
    store_loudest_index=False,
    gates=None,
    defer_readback=False,
):
    """Own mutable leaves; optionally retain device output until the writer."""
    loudest = {}
    if store_loudest_index:
        loudest = {
            ifo: _loudest_indices(
                columns['snr'], columns['chisq'], store_loudest_index
            )
            for ifo, columns in results.items()
            if 'snr' in columns
        }
    psds = {}
    psd_attributes = {}
    for ifo, psd in (store_psd or {}).items():
        if psd is None:
            continue
        # Unwrap FrequencySeries storage without routing a host PSD through
        # the GPU or invoking its synchronous numpy() boundary separately.
        data = getattr(psd, '_data', psd)
        psds[ifo] = getattr(data, 'array', data)
        psd_attributes[ifo] = {'delta_f': float(psd.delta_f)}
        if psd.epoch is not None:
            psd_attributes[ifo]['epoch'] = float(psd.epoch)

    leaves, structure = jax.tree_util.tree_flatten(
        {
            'results': results,
            'raw_results': raw_results or {},
            'loudest': loudest,
            'psds': psds,
        }
    )
    # device_get also coerces some native leaves (notably np.str_) to arrays.
    # Transfer only actual JAX leaves and preserve host scalar serialization.
    device_indices = [
        index
        for index, value in enumerate(leaves)
        if isinstance(value, jax.Array)
    ]
    if not defer_readback:
        from pycbc.events.live_collect_jax import collect_live_arrays

        host_arrays = collect_live_arrays(
            [leaves[index] for index in device_indices]
        )
        for index, value in zip(device_indices, host_arrays):
            leaves[index] = value
    tree = _own_host_value(jax.tree_util.tree_unflatten(structure, leaves))
    owned_gates = {
        ifo: _snapshot_gate_values(values)
        for ifo, values in (gates or {}).items()
    }
    return LiveOutputSnapshot(
        os.fspath(filename),
        _own_host_value(attributes),
        tree['results'],
        tree['raw_results'],
        tree['loudest'],
        owned_gates,
        tree['psds'],
        psd_attributes,
        bool(defer_readback and device_indices),
    )


def _materialize_live_output(snapshot):
    """Collect immutable device columns at the terminal HDF boundary."""
    if not isinstance(snapshot, LiveOutputSnapshot):
        return snapshot
    gates = {
        ifo: (
            _own_host_value(np.array(values.materialize(), dtype=_GATE_DTYPE))
            if callable(getattr(values, 'materialize', None))
            else values
        )
        for ifo, values in snapshot.gates.items()
    }
    if not snapshot.device_pending:
        return replace(snapshot, gates=gates)
    with _output_stage('output_readback'):
        leaves, structure = jax.tree_util.tree_flatten(
            (
                snapshot.results,
                snapshot.raw_results,
                snapshot.loudest,
                snapshot.psds,
            )
        )
        indices = [
            index
            for index, value in enumerate(leaves)
            if isinstance(value, jax.Array)
        ]
        from pycbc.events.live_collect_jax import collect_live_arrays

        arrays = collect_live_arrays([leaves[index] for index in indices])
        for index, value in zip(indices, arrays):
            leaves[index] = value
        results, raw_results, loudest, psds = _own_host_value(
            jax.tree_util.tree_unflatten(structure, leaves)
        )
        return replace(
            snapshot,
            results=results,
            raw_results=raw_results,
            loudest=loudest,
            psds=psds,
            gates=gates,
            device_pending=False,
        )


def write_live_output(snapshot):
    """Write a complete block with one HDF handle and native PSD encoding."""
    snapshot = _materialize_live_output(snapshot)
    with _output_stage('output_write'), h5py.File(
        snapshot.filename, 'w'
    ) as output:
        for key, value in snapshot.attributes.items():
            output.attrs[key] = value
        for ifo, columns in snapshot.results.items():
            for key, value in columns.items():
                output[f'{ifo}/{key}'] = _hdf_value(value)
        for key, value in snapshot.raw_results.items():
            output[key] = _hdf_value(value)
        for ifo, indices in snapshot.loudest.items():
            output[ifo]['loudest'] = indices
        for ifo, values in snapshot.gates.items():
            output[f'{ifo}/gates'] = values
        for ifo, values in snapshot.psds.items():
            dataset = output.create_dataset(
                f'{ifo}/psd',
                data=values,
                compression='gzip',
                compression_opts=9,
                shuffle=True,
            )
            for key, value in snapshot.psd_attributes[ifo].items():
                dataset.attrs[key] = value


class JAXLiveOutputWriter:
    """One ordered writer with bounded queued snapshots and visible failures.

    The queue holds at most ``queue_depth`` snapshots in addition to the one
    being written. Only the pipeline thread submits and closes the writer;
    the background thread collects output and never calls MPI. A failure
    prevents subsequent files from being written and is raised by submit,
    drain or close, rather than being lost in a background thread.
    """

    def __init__(self, queue_depth=2, write=None):
        if type(queue_depth) is not int or queue_depth < 1:
            raise ValueError('JAX Live output queue depth must be positive')
        self._queue = queue.Queue(maxsize=queue_depth)
        self._write = write_live_output if write is None else write
        self._error = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name='pycbc-jax-output', daemon=True
        )
        self._thread.start()

    def _raise_error(self):
        if self._error is not None:
            filename, error = self._error
            raise RuntimeError(
                f'JAX Live output failed: {filename}'
            ) from error

    def _run(self):
        while True:
            snapshot = self._queue.get()
            try:
                if snapshot is None:
                    return
                if self._error is None:
                    try:
                        self._write(_materialize_live_output(snapshot))
                    except BaseException as error:
                        self._error = (snapshot.filename, error)
            finally:
                self._queue.task_done()

    def submit(self, snapshot):
        """Queue a snapshot, applying bounded backpressure when necessary."""
        if self._closed:
            raise RuntimeError('JAX Live output writer is closed')
        self._raise_error()
        while True:
            try:
                self._queue.put(snapshot, timeout=0.05)
                break
            except queue.Full:
                self._raise_error()
        self._raise_error()

    def drain(self):
        """Wait until submitted files are complete, then raise any failure."""
        self._queue.join()
        self._raise_error()

    def close(self):
        """Join the writer even on failure; no output outlives its scope."""
        if not self._closed:
            self._closed = True
            self._queue.put(None)
            self._thread.join()
        self._raise_error()


@contextmanager
def live_output_scope(event_manager, enabled=None, queue_depth=2):
    """Own the root writer for one Live run, including exception cleanup."""
    if enabled is None:
        from pycbc import scheme

        default = scheme.mgr.state.jax_device.platform == 'gpu'
        raw = os.environ.get('PYCBC_JAX_ASYNC_OUTPUT', '1' if default else '0')
        if raw.lower() not in ('0', '1', 'false', 'true', 'no', 'yes'):
            raise ValueError('PYCBC_JAX_ASYNC_OUTPUT must be a Boolean')
        enabled = raw.lower() in ('1', 'true', 'yes')
    writer = (
        JAXLiveOutputWriter(queue_depth)
        if enabled and event_manager.rank == 0
        else None
    )
    event_manager._jax_output_writer = writer
    pipeline_error = None
    try:
        yield
    except BaseException as error:
        pipeline_error = error
        raise
    finally:
        try:
            if writer is not None:
                try:
                    writer.close()
                except BaseException as error:
                    if pipeline_error is None:
                        raise
                    # Keep the original pipeline failure while exposing a
                    # concurrent write failure. Cleanup must still join.
                    note = f'JAX Live output also failed: {error!r}'
                    if error.__cause__ is not None:
                        note += f'; cause: {error.__cause__!r}'
                    if hasattr(pipeline_error, 'add_note'):
                        pipeline_error.add_note(note)
                    else:
                        logging.exception(note)
        finally:
            del event_manager._jax_output_writer


def drain_live_output_jax(event_manager):
    """Complete output before the final barrier and benchmark timing fence."""
    writer = getattr(event_manager, '_jax_output_writer', None)
    if writer is not None:
        with _output_stage('output_drain'):
            writer.drain()


def dump_live_results_jax(
    event_manager,
    results,
    name,
    store_psd=False,
    time_index=None,
    store_loudest_index=False,
    raw_results=None,
    gates=None,
):
    """JAX-only Live dump dispatch; native dump remains the CPU reference."""
    import sys
    from pycbc import version

    filename = os.path.join(
        event_manager.get_out_dir_path(time_index), name + '.hdf'
    )
    writer = getattr(event_manager, '_jax_output_writer', None)
    with _output_stage('output_snapshot'):
        snapshot = snapshot_live_output(
            filename,
            {
                'pycbc_version': version.git_verbose_msg,
                'command_line': sys.argv,
                'num_live_detectors': len(event_manager.live_detectors),
            },
            results,
            raw_results=raw_results,
            store_psd=store_psd,
            store_loudest_index=store_loudest_index,
            gates=gates,
            defer_readback=writer is not None,
        )
    if writer is None:
        write_live_output(snapshot)
    else:
        writer.submit(snapshot)
