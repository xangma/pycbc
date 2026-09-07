"""Read-only terminal verification of the approved len experiment; JSON stdout."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path('/home/xangma/pycbc-torch-python-optimization-r4-20260907')


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    campaign = load('optimization', 'optimize-campaign.py')
    gates = load('gates', 'profile-gates.py')
    compare = load('compare', 'compare-candidate.py')
    control = campaign.control
    status = control.read(ROOT / 'executable-status.json')
    assert status['state'] == 'complete' and status['finished']
    assert status['source_unchanged'] and status['child_pid'] is None
    groups = {status['pid'], status['pgid']}
    for phase in ('api', 'executable'):
        item = control.read(ROOT / (phase + '-status.json'))
        groups.update((item['pid'], item['pgid']))
    for path in (ROOT / 'stages').glob('*.json'):
        stage = control.read(path)
        assert stage['state'] == 'complete' and stage['returncode'] == 0
        groups.update((stage['pid'], stage['pgid']))
    active = [p for p in gates.process_table()
              if p['pid'] in groups or p['pgid'] in groups]
    assert not active, active
    with campaign.LOCK.open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pins = campaign.pins()
        campaign.gates()
        review = campaign.api_review()
        assert status['helper_sha256'] == pins
        assert status['api_review'] == review
        before = control.read(ROOT / 'executable-before.json')
        after = control.read(ROOT / 'executable-after.json')
        fresh = compare.take_snapshot(ROOT / 'source-baseline',
                                      ROOT / 'source-candidate')
        compare.validate_window(before, after)
        compare.validate_window(after, fresh)
        summary = control.read(ROOT / 'executable-summary.json')
        assert status['result_sha256'] == control.digest(
            ROOT / 'executable-summary.json')
        assert summary['science_parity'] == 'pass'
        frozen = compare.frozen_comparator()
        bank_hash = compare.INPUTS[str(compare.BANK)]
        loaded, receipts, runtime_pids, metrics = {}, {}, [], {}
        qualifications = {}
        for role in ('baseline', 'candidate'):
            qualification = control.qualify(
                ROOT / 'runs' / ('qualify-' + role) / 'qualification.json',
                384, bank_hash)
            qualifications[role] = qualification
            names = ['qualify-' + role] + [role + '-r' + str(i)
                                          for i in (1, 2, 3)]
            metrics[role] = []
            for name in names:
                folder = ROOT / 'runs' / name
                receipt = control.read(folder / 'receipt.json')
                receipts[name] = receipt
                runtime = compare.validate_receipt(receipt, role, before, after)
                runtime_pids.append(runtime['pid'])
                assert receipt['trigger_sha256'] == control.digest(
                    folder / 'triggers.hdf')
                loaded[name] = frozen.load(folder / 'triggers.hdf')
                if name != names[0]:
                    metrics[role].append(control.timing(
                        folder, 384, compare.REVISIONS[role], bank_hash,
                        qualification))
                    assert frozen.compare(loaded[names[0]], loaded[name],
                                          frozen.DEFAULTS)['status'] == 'pass'
            # Match the recorded JSON representation of interval tuples.
            metrics[role] = json.loads(json.dumps(metrics[role]))
            assert metrics[role] == summary['metrics'][role]
            seconds = [row['full_wall_seconds'] for row in metrics[role]]
            assert summary['full_wall'][role] == dict(
                seconds=seconds, median=statistics.median(seconds),
                minimum=min(seconds), maximum=max(seconds))
        assert len(set(runtime_pids)) == 8
        pairs = [('qualification-parity', 'qualify-baseline', 'qualify-candidate')]
        pairs += [('timing-parity-r' + str(i), 'baseline-r' + str(i),
                   'candidate-r' + str(i)) for i in (1, 2, 3)]
        science = []
        for name, base, candidate in pairs:
            raw = frozen.compare(loaded[base], loaded[candidate], frozen.DEFAULTS)
            assert raw['status'] == 'fail'
            assert raw == control.read(ROOT / name / 'raw-strict-comparison.json')
            window = control.read(ROOT / 'executable-after-qualification.json') \
                if name == 'qualification-parity' else after
            result = compare.cross_revision(
                frozen, loaded[base], loaded[candidate],
                [receipts[base], receipts[candidate]], before, window)
            assert result['status'] == 'pass'
            assert result == control.read(
                ROOT / name / 'cross-revision-comparison.json')
            science.append(dict(case=name, raw_strict='fail', cross_revision='pass'))
        assert summary['ratio'] == (summary['full_wall']['baseline']['median'] /
                                    summary['full_wall']['candidate']['median'])
        hashes = {}
        for directory, folders, files in os.walk(ROOT):
            folders[:] = [n for n in folders if n not in
                          ('source-baseline', 'source-candidate', '__pycache__',
                           '.pytest_cache')]
            for name in files:
                if name in ('transfer-approved.tar.gz', 'source-candidate.bundle'):
                    continue
                path = Path(directory) / name
                assert path.is_file() and not path.is_symlink()
                hashes[str(path.relative_to(ROOT))] = control.digest(path)
        output = dict(
            host='len', checked_epoch=time.time(), state='complete',
            owned_groups=sorted(groups), active_owned=[],
            shared_lock_reacquired=True, source_unchanged=True,
            science_reverification='pass', runtime_pids=runtime_pids,
            qualifications=qualifications, cross_revision_comparisons=science,
            within_revision_comparisons_passed=6,
            full_wall=summary['full_wall'], ratio=summary['ratio'],
            fresh_source_snapshot=fresh, evidence_sha256=hashes)
        print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
