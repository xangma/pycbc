"""Continue descriptive timings while retaining the stopped full-PSD verdict."""
import datetime
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('acquisition', ROOT / 'campaign.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
SAVE = C.save


def save(path, value):
    path = Path(path)
    SAVE(ROOT / 'timing-status.json' if path == ROOT / 'status.json' else path, value)


C.save = save
C.STATE = dict(state='starting', pid=os.getpid(), pgid=os.getpgrp(), completed=[],
               started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               qualification_status='status.json', continuation_policy='continuation-policy.json')


def conditioning(left, right):
    a, b = [C.read(p / 'qualification.json')['observations'] for p in (left, right)]
    rows = []
    assert len(a['psd_arrays']) == len(b['psd_arrays']) > 0
    for x, y in zip(a['psd_arrays'], b['psd_arrays']):
        av, bv = np.load(left / x['relative_path']), np.load(right / y['relative_path'])
        assert av.shape == bv.shape and av.dtype == bv.dtype
        lo, hi = x['validity']['filter_bin_start'], x['validity']['filter_bin_stop']
        assert (lo, hi) == (y['validity']['filter_bin_start'], y['validity']['filter_bin_stop']) == (15360, 1048576)
        finite = np.isfinite(av) & np.isfinite(bv)
        af, bf = av[finite].astype(float), bv[finite].astype(float)
        bad = np.abs(af - bf) > 1e-4 * np.maximum(abs(af), abs(bf))
        masks = all(np.array_equal(f(av), f(bv)) for f in (np.isnan, np.isposinf, np.isneginf))
        rows.append(dict(full_psd_budget_pass=bool(masks and not np.any(bad)),
            full_psd_finite_violations=int(np.count_nonzero(bad)), nonfinite_masks_equal=masks,
            used_psd_exact=av[lo:hi].tobytes() == bv[lo:hi].tobytes(), filter_bins=[lo, hi],
            relative_budget=1e-4, absolute_floor=0))
    return dict(conditioned_strain_exact=a['conditioned_strain'] == b['conditioned_strain'],
                geometry_exact=a['segment_geometry'] == b['segment_geometry'], psds=rows)


def main():
    stopped = C.read(ROOT / 'status.json')
    assert stopped['state'] == 'failed' and 'violations' in stopped['error']
    assert stopped['completed'] == ['qual-' + arm for arm in C.ROUTES]
    assert not (ROOT / 'timing-status.json').exists(), 'Never reuse an attempted timing output'
    assert not list((ROOT / 'runs').glob('timing-*'))
    C.pins_unchanged()
    policy = dict(
        purpose='Complete descriptive fixed-workload timings; full-PSD failures remain failures.',
        initial_stop=stopped, initial_status_sha256=C.sha(ROOT / 'status.json'),
        change='After observing excluded-bin PSD differences, allow timings only after all corrected-baseline and proposed-backend trigger comparisons, exact conditioned strain/geometry, and exact used PSD bins pass. Preserve full-array checks and prohibit full-equivalence speedup claims. No source, input, tolerance, worker command or ordering changes.',
        post_qualification_amendment=True, full_psd_failures_reclassified_as_pass=False,
        acquisition_script_sha256=C.sha(ROOT / 'campaign.py'),
        continuation_script_sha256=C.sha(__file__), decided_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    SAVE(ROOT / 'continuation-policy.json', policy)
    extra_pins = {name: C.sha(ROOT / name) for name in ('resume-timings.py', 'continuation-policy.json', 'status.json')}
    SAVE(ROOT / 'continuation-pins.json', extra_pins)
    qualified = {arm: ROOT / 'runs' / ('qual-' + arm) for arm in C.ROUTES}
    for arm, folder in qualified.items():
        C.qualify(folder, arm)
    comparisons, conditions = {}, {}
    pairs = [('corrected-cpu', arm, 'corrected-vs-' + arm) for arm in list(C.ROUTES)[1:]]
    pairs += [('proposed-cpu', arm, 'proposed-cpu-vs-' + arm) for arm in ('torch-cpu', 'torch-cuda')]
    for left, right, name in pairs:
        prior = ROOT / 'comparisons' / (name + '.json')
        comparisons[name] = C.read(prior)['result'] if prior.exists() else C.compare(qualified[left], qualified[right], name)
        if not (ROOT / 'comparisons' / (name + '-conditioning.json')).exists():
            C.conditioning_compare(qualified[left], qualified[right], name)
        conditions[name] = conditioning(qualified[left], qualified[right])
        if left in ('corrected-cpu', 'proposed-cpu'):
            assert comparisons[name]['status'] == 'pass'
            assert conditions[name]['conditioned_strain_exact'] and conditions[name]['geometry_exact']
            assert all(x['used_psd_exact'] for x in conditions[name]['psds'])
    SAVE(ROOT / 'qualification-summary.json', comparisons)
    SAVE(ROOT / 'continuation-conditioning.json', conditions)
    measurements = {arm: [] for arm in C.ROUTES}
    with C.LOCK.open() as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        C.STATE['lock'] = dict(path=str(C.LOCK), inode=os.fstat(lock.fileno()).st_ino)
        for repeat, order in enumerate(C.CONFIG['ordering'], 1):
            for arm in order:
                assert all(C.sha(ROOT / p) == h for p, h in extra_pins.items())
                folder = C.run_case(arm, 'timing', lock, repeat)
                measurements[arm].append(C.read(folder / 'receipt.json')['elapsed_wall_seconds'])
        C.pins_unchanged()
    rows = {}
    for arm, values in measurements.items():
        median = statistics.median(values)
        rows[arm] = dict(source=C.CONFIG['source_commits'][C.ROUTES[arm][0]], scheme=C.ROUTES[arm][1],
            samples_seconds=values, median_seconds=median, min_seconds=min(values), max_seconds=max(values),
            template_seconds_per_wall_second=384 * 1904 / median,
            trigger_count=C.COMPARATOR.load(qualified[arm] / 'triggers.hdf')['detectors']['H1']['count'])
    all_full = all(v['conditioned_strain_exact'] and v['geometry_exact'] and
                   all(x['full_psd_budget_pass'] for x in v['psds']) for v in conditions.values())
    SAVE(ROOT / 'summary.json', dict(scope=C.CONFIG['scope'], timing_boundary=C.CONFIG['timing_boundary'],
        arms=rows, tolerances=C.COMPARATOR.DEFAULTS,
        corrected_comparisons={k: v['status'] for k, v in comparisons.items() if k.startswith('corrected-')},
        proposed_comparisons={k: v['status'] for k, v in comparisons.items() if k.startswith('proposed-')},
        equal_output_speedup_eligible=all_full and all(v['status'] == 'pass' for v in comparisons.values()),
        full_psd_and_conditioning_pass=all_full, continuation_policy='continuation-policy.json'))
    C.STATE.update(state='complete', finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(ROOT / 'status.json', C.STATE)


if __name__ == '__main__':
    def terminate(*_):
        if C.ACTIVE_CHILD is not None:
            try:
                os.killpg(C.ACTIVE_CHILD, signal.SIGTERM)
            except ProcessLookupError:
                pass
        sys.exit(143)
    signal.signal(signal.SIGTERM, terminate)
    try:
        main()
    except BaseException as error:
        C.STATE.update(state='failed', error=repr(error))
        save(ROOT / 'status.json', C.STATE)
        raise
