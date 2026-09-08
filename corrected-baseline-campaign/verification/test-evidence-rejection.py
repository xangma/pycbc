"""Negative integration checks on private copies; acquisition remains read-only."""
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
prior = json.loads((HERE / 'qualification-verification-1.json').read_text())
inventory = prior['input_evidence_sha256']
checks = []
with tempfile.TemporaryDirectory(prefix='negative-fixture-', dir=HERE) as tmp:
    root = Path(tmp)
    for name, expected in inventory.items():
        source = ROOT / name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    control = V.Verifier(root, qualification_only=True).run()
    assert control['scientific_status'] == 'FAIL' and control['equal_output_speedup_eligible'] is False
    checks.append(dict(name='unmodified copied qualification control', status='PASS'))
    def rejects(name, file, mutate, expected, qualification_only=True):
        target = root / file
        data = target.read_bytes() if target.exists() else None
        try:
            mutate(target)
            try:
                V.Verifier(root, qualification_only=qualification_only).run()
            except Exception as error:
                assert expected in str(error), (name, type(error).__name__, str(error))
                checks.append(dict(name=name, status='REJECTED_AS_EXPECTED', error_type=type(error).__name__, error=str(error)))
            else:
                raise AssertionError('Unexpectedly accepted ' + name)
        finally:
            if data is not None:
                target.write_bytes(data)
    def change_json(path, mutate):
        value = json.loads(path.read_text())
        mutate(value)
        path.write_text(json.dumps(value))
    rejects('missing timing status prevents final verdict', 'status.json', lambda _: None,
            'timing-status.json', qualification_only=False)
    rejects('corrupt HDF bytes', 'runs/qual-corrected-cpu/triggers.hdf',
            lambda p: p.write_bytes(p.read_bytes() + b'corruption'), 'Trigger HDF bytes differ')
    rejects('missing PSD array', 'runs/qual-corrected-cpu/arrays/qualification-psd-000.npy',
            lambda p: p.unlink(), 'qualification-psd-000.npy')
    rejects('wrong source receipt', 'runs/qual-corrected-cpu/receipt.json',
            lambda p: change_json(p, lambda d: d['source_info'].update(commit='0'*40)), 'Run identity/state/source mismatch')
    rejects('wrong reused native binary hash', 'corrected-build.json',
            lambda p: change_json(p, lambda d: d['native'].update({next(iter(d['native'])): '0'*64})), 'Build/source pins disagree')
    rejects('extra native worker threads', 'runs/qual-corrected-cpu/runtime.json',
            lambda p: change_json(p, lambda d: d['at_first_bank']['threadpools'][0].update(num_threads=2)), 'not single-threaded')
    rejects('CPU imports Torch', 'runs/qual-corrected-cpu/runtime.json',
            lambda p: change_json(p, lambda d: d['at_first_bank'].update(torch_imported=True)), 'Torch import state')
    rejects('positive but inconsistent elapsed time', 'runs/qual-corrected-cpu/receipt.json',
            lambda p: change_json(p, lambda d: d.update(elapsed_wall_seconds=6.8)), 'Perf duration disagrees with GNU time')
    rejects('reclassified full PSD failure', 'continuation-conditioning.json',
            lambda p: change_json(p, lambda d: d['corrected-vs-torch-cpu']['psds'][0].update(full_psd_budget_pass=True)),
            'Continuation conditioning diagnostics differ')
    rejects('missing one of five trigger comparisons', 'comparisons/proposed-cpu-vs-torch-cuda.json',
            lambda p: p.unlink(), 'proposed-cpu-vs-torch-cuda.json')
    rejects('changed scientific input config', 'config.json',
            lambda p: change_json(p, lambda d: d.update(core=9)), 'Frozen acquisition file mismatch: config.json')
for name, expected in inventory.items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected
print(json.dumps(dict(status='PASS', acquisition_bytes_unchanged=True, checks=checks, count=len(checks)), indent=2))
