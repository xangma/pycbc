"""Exercise final-mode evidence gates on private copies of the final acquisition."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location('corrected_verifier', HERE / 'verify-results.py')
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)
final = json.loads((HERE / 'final-verification-2.json').read_text())
inventory = final['input_evidence_sha256']
checks = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


with tempfile.TemporaryDirectory(prefix='final-negative-fixture-', dir=HERE) as tmp:
    root = Path(tmp)
    for name, expected in inventory.items():
        source = ROOT / name
        assert sha(source) == expected
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    control = V.Verifier(root).run()
    assert control['evidence_status'] == 'PASS_WITH_LIMITATIONS'
    assert control['scientific_status'] == 'FAIL' and control['equal_output_speedup_eligible'] is False
    assert len(control['timing_comparisons']) == 16
    assert control['comparisons']['corrected-vs-proposed-cpu']['eligible'] is True
    checks.append(dict(name='complete evidence preserves scientific failure and passing CPU pair', status='PASS'))

    def change_json(path, mutate):
        value = json.loads(path.read_text())
        mutate(value)
        path.write_text(json.dumps(value))

    def rejects(name, mutations, expected):
        saved = {file: (root / file).read_bytes() for file, _ in mutations}
        try:
            for file, mutate in mutations:
                mutate(root / file)
            try:
                V.Verifier(root).run()
            except Exception as error:
                assert expected in str(error), (name, type(error).__name__, str(error))
                checks.append(dict(name=name, status='REJECTED_AS_EXPECTED',
                                   error_type=type(error).__name__, error=str(error)))
            else:
                raise AssertionError('Unexpectedly accepted ' + name)
        finally:
            for file, data in saved.items():
                (root / file).write_bytes(data)

    diag = 'torch-environment-diagnostic.json'
    rejects('missing timing receipt', [('runs/timing-torch-cpu-r4/receipt.json', lambda p: p.unlink())],
            'receipt.json')
    rejects('positive corrupted final timing duration', [('runs/timing-torch-cpu-r4/receipt.json',
            lambda p: change_json(p, lambda d: d.update(elapsed_wall_seconds=10.0)))],
            'Perf duration disagrees with GNU time')
    rejects('altered timing own-qualification comparison',
            [('comparisons/timing-torch-cpu-r4-vs-own-qualification.json',
              lambda p: change_json(p, lambda d: d['result'].update(status='review')))],
            'Recomputed comparison differs from archive')
    rejects('invented summary median', [('summary.json',
            lambda p: change_json(p, lambda d: d['arms']['torch-cuda'].update(median_seconds=1.0)))],
            'Acquired summary disagrees')
    rejects('summary promotes full-PSD failure to equal-output eligibility', [('summary.json',
            lambda p: change_json(p, lambda d: d.update(equal_output_speedup_eligible=True)))],
            'Acquired summary disagrees')
    rejects('missing diagnostic JSON while its script is present', [(diag, lambda p: p.unlink())], diag)
    for key in ('dependency_manifest_sha256', 'timing_status_sha256'):
        rejects('diagnostic bound to wrong ' + key, [(diag,
                lambda p, key=key: change_json(p, lambda d: d.update({key: '0' * 64})))],
                'Dependency diagnostic is not bound')
    rejects('diagnostic observed on a different host', [(diag,
            lambda p: change_json(p, lambda d: d.update(hostname='other-host')))],
            'Worker host differs from post-run diagnostic host')
    rejects('diagnostic predates timing completion', [(diag,
            lambda p: change_json(p, lambda d: d.update(observed_at_utc='2026-09-08T15:00:00+00:00')))],
            'Dependency diagnostic must postdate')
    rejects('diagnostic selected file mismatches RECORD', [(diag,
            lambda p: change_json(p, lambda d: d['distributions'][0]['selected_file_checks'][0].update(recorded='sha256=bad')))],
            'Torch installation selected file disagrees')
    rejects('unreviewed diagnostic script', [('inspect-torch-environment.py',
            lambda p: p.write_bytes(p.read_bytes() + b'\n# changed\n'))],
            'Unreviewed post-run dependency diagnostic script')
    rejects('missing timing completion with consistently rebound diagnostic', [
            ('timing-status.json', lambda p: change_json(p, lambda d: d['completed'].pop())),
            (diag, lambda p: change_json(p, lambda d: d.update(timing_status_sha256=sha(root / 'timing-status.json'))))],
            'Incomplete/out-of-order timing continuation status')

for name, expected in inventory.items():
    assert sha(ROOT / name) == expected
print(json.dumps(dict(status='PASS', acquisition_bytes_unchanged=True,
                     verifier_sha256=sha(HERE / 'verify-results.py'),
                     checks=checks, count=len(checks)), indent=2))
