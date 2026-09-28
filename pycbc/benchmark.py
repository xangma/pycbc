"""Opt-in evidence and stage markers for executable performance campaigns."""
import atexit
import json
import os
import sys
import threading
import time


_JAX_AUDIT_LOCK = threading.Lock()
_JAX_AUDIT_STATE = None


def _jax_chisq_mode():
    """Return the recorded JAX mode, if this run declares one."""
    raw = os.environ.get('PYCBC_BENCHMARK_SCIENCE_CONFIG')
    if not raw:
        return None
    try:
        config = json.loads(raw)
    except (TypeError, ValueError):
        return None
    mode = config.get('--jax-chisq-mode')
    return mode if mode in ('cpu-compatible', 'direct-phase') else None


def stage_event(stage, event, **metadata):
    """Emit host submission boundaries without synchronizing a device."""
    if os.environ.get('PYCBC_BENCHMARK_STAGES') == '1':
        print('PYCBC_STAGE_EVENT ' + json.dumps(dict(
            stage=stage, event=event, label=stage.replace('_', ' '),
            pid=os.getpid(), monotonic_ns=time.monotonic_ns(), **metadata)),
            file=sys.stderr, flush=True)


def enable_jax_compilation_audit(jax):
    """Record persistent-cache requests made by an opted-in JAX process.

    The campaign runner sets ``PYCBC_JAX_COMPILATION_AUDIT_DIR``.  Each MPI
    rank/process writes its own receipt at normal interpreter shutdown, so a
    timed benchmark can prove it loaded cached executables instead of silently
    recompiling them.
    """
    global _JAX_AUDIT_STATE
    output_dir = os.environ.get('PYCBC_JAX_COMPILATION_AUDIT_DIR')
    if not output_dir or _JAX_AUDIT_STATE is not None:
        return

    events = {
        'compile_requests_use_cache': 0,
        'cache_hits': 0,
        'cache_writes': 0,
    }
    names = {
        '/jax/compilation_cache/compile_requests_use_cache':
            'compile_requests_use_cache',
        '/jax/compilation_cache/cache_hits': 'cache_hits',
        '/jax/compilation_cache/cache_misses': 'cache_writes',
    }

    def listener(event, **metadata):
        del metadata
        name = names.get(event)
        if name is not None:
            with _JAX_AUDIT_LOCK:
                events[name] += 1

    state = {
        'schema_version': 1,
        'pid': os.getpid(),
        'started_monotonic_ns': time.monotonic_ns(),
        'cache_dir': os.environ.get('JAX_COMPILATION_CACHE_DIR'),
        'cache_enabled': os.environ.get('JAX_ENABLE_COMPILATION_CACHE'),
        'jax_config': {
            'enable_compilation_cache': bool(
                getattr(jax.config, 'jax_enable_compilation_cache', False)),
            'compilation_cache_dir': getattr(
                jax.config, 'jax_compilation_cache_dir', None),
            'min_compile_time_secs': float(getattr(
                jax.config, 'jax_persistent_cache_min_compile_time_secs', -1)),
            'min_entry_size_bytes': int(getattr(
                jax.config, 'jax_persistent_cache_min_entry_size_bytes', -1)),
            'raise_persistent_cache_errors': bool(getattr(
                jax.config, 'jax_raise_persistent_cache_errors', False)),
        },
        'events': events,
    }
    _JAX_AUDIT_STATE = state
    jax.monitoring.register_event_listener(listener)

    def write_receipt():
        with _JAX_AUDIT_LOCK:
            receipt = dict(state)
            receipt['events'] = dict(events)
        requests = receipt['events']['compile_requests_use_cache']
        hits = receipt['events']['cache_hits']
        receipt['uncached_compile_requests'] = max(0, requests - hits)
        receipt['finished_monotonic_ns'] = time.monotonic_ns()
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'process-%d.json' % os.getpid())
        temporary = path + '.tmp'
        with open(temporary, 'w') as out:
            json.dump(receipt, out, indent=2, sort_keys=True)
            out.write('\n')
        os.replace(temporary, path)

    atexit.register(write_receipt)


def save_inspiral_evidence(strain, segments):
    """Save complete search inputs only in a separate qualification process."""
    path = os.environ.get('PYCBC_BENCHMARK_EVIDENCE')
    if not path:
        return
    import h5py
    with h5py.File(path, 'w') as out:
        out.attrs['science_config'] = os.environ['PYCBC_BENCHMARK_SCIENCE_CONFIG']
        mode = _jax_chisq_mode()
        if mode is not None:
            out.attrs['jax_chisq_mode'] = mode
        out.create_dataset('conditioned_strain', data=strain.numpy())
        out.attrs['strain_epoch'] = float(strain.start_time)
        out.attrs['strain_delta_t'] = float(strain.delta_t)
        for i, seg in enumerate(segments):
            group = out.create_group('segments/%d' % i)
            group.create_dataset('strain', data=seg.numpy())
            group.create_dataset('psd', data=seg.psd.numpy())
            group.attrs['delta_f'] = float(seg.delta_f)
            group.attrs['epoch'] = float(seg.epoch)
            group.attrs['analyze_start'] = int(seg.analyze.start)
            group.attrs['analyze_stop'] = int(seg.analyze.stop)
            group.attrs['segment_start'] = int(seg.seg_slice.start)
            group.attrs['segment_stop'] = int(seg.seg_slice.stop)


def save_inspiral_work(segments, completed_templates, sample_rate):
    """Count the union of valid intervals; overlapping segments count once."""
    path = os.environ.get('PYCBC_BENCHMARK_WORK')
    if not path:
        return
    intervals = sorted((int(seg.cumulative_index),
                        int(seg.cumulative_index + seg.analyze.stop - seg.analyze.start))
                       for seg in segments)
    merged = []
    for start, stop in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    seconds = sum(stop - start for start, stop in merged) / sample_rate
    with open(path, 'w') as out:
        receipt = dict(completed_templates=completed_templates,
                       valid_detector_seconds=seconds,
                       valid_sample_intervals=merged, sample_rate=sample_rate,
                       signal_dtypes=sorted({str(seg.dtype) for seg in segments}),
                       completed_template_seconds=completed_templates * seconds)
        mode = _jax_chisq_mode()
        if mode is not None:
            receipt['jax_chisq_mode'] = mode
        json.dump(receipt, out)
