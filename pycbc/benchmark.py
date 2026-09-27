"""Opt-in evidence and stage markers for executable performance campaigns."""
import json
import os
import sys
import time


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
