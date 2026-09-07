"""Primary offline audit of frozen pass-two receipts and actual HDF outputs."""
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import median
import sys
from types import SimpleNamespace

root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2])
remote = Path('/home/xangma/pycbc-torch-python-optimization-pass2-20260907')
def read(path):
    return json.loads(Path(path).read_text())
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

manifest = read(root / 'evidence-manifest.json')
for name, expected in manifest['files'].items():
    assert sha(root / name) == expected, name
assert len(manifest['files']) == 202
assert sha(root / 'compare-candidate.py') == '3464f728b5297452a3172e5cbbc8ba92fc237eb0a5b48a1da89cb79d89ea31b0'
wrapper = load_module('pass2_wrapper', root / 'compare-candidate.py')
frozen = wrapper.frozen_comparator()
# Only filesystem reads are relocated. Archived commands, receipts, environment,
# source identities and the two bounded substitutions remain unchanged.
def local(path):
    path = Path(path)
    return root / path.relative_to(remote) if path.is_relative_to(remote) else path
wrapper.read = lambda path: read(local(path))
wrapper.digest = lambda path: sha(local(path))
before = read(root / 'executable-before.json')
after = read(root / 'executable-after.json')
wrapper.validate_window(before, after)
# Preserve the recorded acquisition path when the helper compares its own
# bundled threadpoolctl identity; its bytes still come from the local archive.
class ReceiptLoader:
    def __init__(self, original):
        self.original = original
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        self.original.exec_module(module)
        expected = module.expected_threadpoolctl
        module.expected_threadpoolctl = lambda: dict(
            expected(), path=str(remote / 'threadpoolctl.py'))
def receipt_spec(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert name == 'checked_inspiral'
    spec.loader = ReceiptLoader(spec.loader)
    return spec
wrapper.importlib = SimpleNamespace(util=SimpleNamespace(
    spec_from_file_location=receipt_spec,
    module_from_spec=importlib.util.module_from_spec))
cases = ['qualify-baseline', 'qualify-candidate', 'baseline-r1', 'candidate-r1',
         'candidate-r2', 'baseline-r2', 'baseline-r3', 'candidate-r3']
runs, receipts, runtimes = {}, {}, []
previous_stop = None
for case in cases:
    role = 'baseline' if 'baseline' in case else 'candidate'
    receipt = read(root / 'runs' / case / 'receipt.json')
    runtime = wrapper.validate_receipt(receipt, role, before, after)
    stage = read(root / 'stages' / (case + '.json'))
    assert stage['state'] == 'complete' and stage['returncode'] == 0
    assert previous_stop is None or previous_stop <= stage['started']
    previous_stop = stage['finished']
    receipts[case] = receipt
    runtimes.append(runtime['pid'])
    runs[case] = frozen.load(root / 'runs' / case / 'triggers.hdf')
assert len(set(runtimes)) == 8
summary = read(root / 'executable-summary.json')
walls = {}
for role in ('baseline', 'candidate'):
    values = [receipts[f'{role}-r{i}']['elapsed_wall_seconds'] for i in (1, 2, 3)]
    assert values == summary['full_wall'][role]['seconds']
    walls[role] = dict(seconds=values, median=median(values), minimum=min(values), maximum=max(values))
    assert walls[role] == summary['full_wall'][role]
ratio = walls['baseline']['median'] / walls['candidate']['median']
assert ratio == summary['ratio']
comparisons = []
def exact(result):
    assert result['status'] == 'pass', result.get('failures')
    h = result['detectors']['H1']
    assert h['baseline_count'] == h['candidate_count'] == h['matched_count'] == 1991
    assert len(h['metrics']) == 11
    assert all(m['max_absolute_error'] == m['violations'] == 0 for m in h['metrics'].values())
    return dict(status=result['status'], matched=1991, fields=11, max_absolute_error=0)
for suffix in ('qualification', 'r1', 'r2', 'r3'):
    a, b = ('qualify-baseline', 'qualify-candidate') if suffix == 'qualification' else (f'baseline-{suffix}', f'candidate-{suffix}')
    raw = frozen.compare(runs[a], runs[b], frozen.DEFAULTS)
    assert raw['status'] == 'fail'
    assert set(raw['failures']) == {'configuration mismatch: consumed_input_sha256', 'configuration mismatch: source_snapshot'}
    result = wrapper.cross_revision(frozen, runs[a], runs[b], [receipts[a], receipts[b]], before, after)
    comparisons.append(dict(case=suffix, kind='bounded-cross-revision', **exact(result['comparison'])))
for role in ('baseline', 'candidate'):
    for i in (1, 2, 3):
        case = f'{role}-r{i}'
        comparisons.append(dict(case=case, kind='strict-within-revision', **exact(frozen.compare(runs['qualify-' + role], runs[case], frozen.DEFAULTS))))
result = dict(state='pass', archived_file_hashes=202, fresh_serial_workers=8, comparisons=comparisons, full_wall=walls, ratio=ratio, source_window_validated=True, archived_evidence_root=str(root))
with out.open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
print(json.dumps(dict(state='pass', hashes=202, fresh_workers=8, actual_hdf_comparisons=len(comparisons), full_wall=walls, ratio=ratio)))
