#!/usr/bin/env python3
"""Freeze the measured CPU winner and the subsequent matched campaign plans."""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parent
commit = '837f38d493420043e45fb1ad210a0ccf68bacbaa'
inputs = {}


def read(name):
    path = root / name
    inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def bind(name):
    path = root / name
    inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, data):
    with (root / name).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


bind(Path(__file__).name)
bind('compare-triggers.py')
spec = importlib.util.spec_from_file_location('compare', root / 'compare-triggers.py')
comparator = importlib.util.module_from_spec(spec)
exec(compile((root / 'compare-triggers.py').read_bytes(), 'compare-triggers.py', 'exec'), comparator.__dict__)
assert comparator.DEFAULTS == dict(rtol=1e-4, atol=1e-5, sigmasq_rtol=1e-5,
                                   phase_atol=1e-4, max_examples=12)
assert read('source-v5.json')['commit'] == commit
unit = read('unit-tests-v5.json')
assert unit['passed'] and unit['state'] == 'complete' and unit['source_info']['commit'] == commit
campaign = read('precision5-reference-campaign.status.json')
assert campaign['state'] == 'complete' and campaign['returncode'] == 0
assert campaign['input_sha256'] == campaign['input_sha256_after']
for name in ('waveform-validation-precision5.json', 'boundary-injections-precision5.json'):
    proof = read(name)
    assert proof['state'] == 'complete' and proof['passed'] is True
    assert proof['source_before'] == proof['source_after']
    assert proof['source_before']['commit'] == commit and proof['source_before']['status'] == ''
    assert proof['input_sha256'] == proof['input_sha256_after']
for stem in ('precision5-reference-qualifications-plan', 'precision5-reference-timings-plan'):
    plan = read(stem + '.json')
    state = read(stem + '.status.json')
    assert state['state'] == 'complete' and state['returncode'] == 0
    assert state['completed'] == plan and state['current'] is None
    assert state['plan_sha256'] == inputs[str(root / (stem + '.json'))]

grid = []
for length in (256, 512, 1024):
    qcase = f'qual-precision5-cpu-l{length}'
    q = read(f'runs/{qcase}/qualification.json')
    assert q['status'] == 'success' and all(v is True for v in q['checks'].values())
    read(f'runs/{qcase}/receipt.json')
    bind(f'runs/{qcase}/triggers.hdf')
    baseline = comparator.load(root / 'runs' / qcase / 'triggers.hdf')
    values = []
    for rep in (1, 2, 3):
        case = f'tune-precision5-cpu-l{length}-r{rep}'
        receipt = read(f'runs/{case}/receipt.json')
        assert receipt['state'] == 'complete' and receipt['returncode'] == 0
        assert receipt['source_info'] == dict(commit=commit, status='', tracked_diff='')
        assert receipt['source_status_after'] == '' and receipt['mode'] == 'timing'
        assert receipt['scheme'] == 'cpu:1' and receipt['segment_length'] == length
        assert receipt['input_sha256'] == receipt['input_sha256_after']
        bind(f'runs/{case}/triggers.hdf')
        assert receipt['trigger_sha256'] == inputs[str(root / 'runs' / case / 'triggers.hdf')]
        result = comparator.compare(baseline, comparator.load(root / 'runs' / case / 'triggers.hdf'), comparator.DEFAULTS)
        assert result['status'] == 'pass', result
        wall = receipt['elapsed_wall_seconds']
        assert isinstance(wall, (float, int)) and 0 < wall < float('inf')
        values.append(wall)
    grid.append(dict(segment_length_seconds=length, wall_seconds=values,
                     median_wall_seconds=statistics.median(values)))
selected = min(grid, key=lambda row: (row['median_wall_seconds'], row['segment_length_seconds']))['segment_length_seconds']


def case(name, mode, scheme):
    return ['--case', name, '--mode', mode, '--scheme', scheme,
            '--segment-length', str(selected)]


backends = [('cpu', 'cpu:1'), ('torch-cpu', 'torch:cpu:1'), ('torch-cuda', 'torch:cuda:0')]
plans = {
    'precision5-final-qualifications-plan.json': [case(f'qual-selected5-{label}-l{selected}', 'qualify', scheme) for label, scheme in backends],
    'precision5-reference-profiles-plan.json': [case(f'reference-precision5-cpu-l{selected}-{mode}', mode, 'cpu:1') for mode in ('cprofile', 'perf')],
    'precision5-matched-backends-plan.json': [case(f'matched-precision5-{label}-l{selected}-r{rep}', 'timing', scheme)
        for rep in (1, 2, 3) for label, scheme in backends[rep - 1:] + backends[:rep - 1]],
    'precision5-torch-profiles-plan.json': [case(f'profile-precision5-torch-{device}-l{selected}-{mode}', mode, scheme)
        for device, scheme in [('cpu', 'torch:cpu:1'), ('cuda', 'torch:cuda:0')] for mode in ('cprofile', 'perf')]
        + [case(f'profile-precision5-torch-cuda-l{selected}-torchprofile', 'torchprofile', 'torch:cuda:0')],
}
decision_name = 'precision5-reference-tuning-decision.json'
assert not any((root / name).exists() for name in [decision_name, *plans])
assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest for name, digest in inputs.items())
decision = dict(schema_version=1, source_commit=commit,
                selected_segment_length_seconds=selected,
                selected_start_pad_seconds=112, selected_end_pad_seconds=16,
                basis='Choose the lowest median full-process wall time from three unprofiled repetitions at each declared segment length, all using the corrected source and the 1e-5 compressed bank. Retain 112-second start padding and 16-second end padding; the new independent boundary model checks both 96- and 112-second starts. Earlier source timing results are not used to select this winner.',
                selection_grid=grid, input_sha256=inputs,
                plan_sha256={name: hashlib.sha256(
                    (json.dumps(plan, indent=2, allow_nan=False) + '\n').encode()).hexdigest()
                    for name, plan in plans.items()})
for name, plan in plans.items():
    save(name, plan)
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == decision['plan_sha256'][name]
save(decision_name, decision)
print(json.dumps(dict(selected_segment_length_seconds=selected, selection_grid=grid, plans=list(plans))))
