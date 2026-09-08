"""Adapt completed, pinned CPU qualification to the frozen comparator schema."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / 'linux-validation-v3'


def read(name):
    return json.loads((FOLDER / name).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


status = read('status.json')
comparison = read('comparison.json')
source = read('source-pins.json')
qualification = read('qualification.json')
runtime = read('runtime.json')
assert status['state'] == runtime['state'] == 'complete'
assert comparison['status'] == 'pass' and comparison['source_and_input_pins_unchanged']
assert qualification['status'] == 'success' and all(qualification['checks'].values())
config = json.loads(Path('/home/xangma/pycbc-torch-baseline-final-20260908/config.json').read_text())
cli = qualification['argv']
assert cli[0] == str(ROOT / 'cpu-corrections/bin/pycbc_inspiral')
assert source['commit'] == '66789ac4a7468094b0cc3ca1498a1de67e0311f6'
hashes = dict(config['input_pins'], **{cli[0]: source['tracked']['bin/pycbc_inspiral']})
assert all(sha(p) == value for p, value in hashes.items())
record = dict(state='complete', returncode=0, executable_cli=cli,
              source_info=dict(commit=source['commit'], status='', tracked_diff=''),
              source_status_after='', input_sha256=hashes, input_sha256_after=hashes,
              trigger_sha256=sha(FOLDER / 'triggers.hdf'),
              provenance='Derived from the completed qualification controller receipts; no new execution.',
              derived_from={p: sha(FOLDER / p) for p in
                            ['status.json', 'comparison.json', 'source-pins.json',
                             'qualification.json', 'runtime.json']})
with (FOLDER / 'receipt.json').open('x') as output:
    json.dump(record, output, indent=2)
    output.write('\n')
print('Comparator receipt created from verified completed qualification')
