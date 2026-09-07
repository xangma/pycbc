"""Rebuild scientific comparisons and timing summaries from the copied raw evidence."""
import argparse
import copy
import subprocess
import types
import json
import math
from pathlib import Path
import statistics
import sys

BASE = Path(__file__).resolve().parent
ROOT = None
OUTPUT = None
PREVIOUS = None
REMOTE = Path('/home/xangma/pycbc-torch-profile-20260907-r3')
REV = 'd2647addb884ead3249914ebc980f3c132076d93'
BANK = '26050d48322a1d71092bb0e024e71a89ace56b3e7b1c5e4cf20c7b769213fb7f'


def load_module(name, file):
    module = types.ModuleType(name)
    module.__file__ = str(file)
    sys.modules[name] = module
    exec(compile(file.read_bytes(), str(file), 'exec'), module.__dict__)
    return module


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    control = load_module('campaign_controls', ROOT / 'campaign_controls.py')
    compare = load_module('frozen_compare', ROOT / 'compare-triggers.py')
    trace = load_module('trace_summary', ROOT / 'summarize_torch_trace.py')
    status = control.read(ROOT / 'profile-status.json')
    audit = control.read(ROOT / 'terminal-audit.json')
    require(status['state'] == 'complete' and status['timing_workers'] == 9 and
            audit['state'] == 'validated-complete' and audit['independent_lock_reacquired'] and
            audit['profile_status_sha256'] == control.digest(ROOT / 'profile-status.json'),
            'Terminal evidence differs')
    for path, sha in audit['stages'].items():
        require(control.digest(ROOT / 'stages' / path) == sha, 'Stage hash differs')
    for path, sha in audit['comparisons'].items():
        require(control.digest(ROOT / path) == sha, 'Comparison hash differs')
    for name, sha in control.read(ROOT / 'staged-helpers.json').items():
        require(control.digest(ROOT / name) == sha, 'Helper hash differs: ' + name)
    # The archive contains the full source/native snapshot. External frame and bank
    # pins are independently validated by the terminal audit and each worker receipt.
    for path, sha in control.read(ROOT / 'provenance.json')['sha256'].items():
        p = Path(path)
        if p.is_relative_to(REMOTE):
            require(control.digest(ROOT / p.relative_to(REMOTE)) == sha, 'Provenance pin differs')
    baseline = compare.load(ROOT / 'runs/qualify-cpu/triggers.hdf')
    comparisons = {}
    rows = {n: [] for n in ('cpu', 'torchcpu', 'cuda')}
    qualifications = {n: control.qualify(ROOT / f'runs/qualify-{n}/qualification.json', 384, BANK)
                      for n in rows}
    for folder in sorted((ROOT / 'runs').iterdir()):
        receipt = control.read(folder / 'receipt.json')
        expected = audit['science_runs'][folder.name]
        require(control.digest(folder / 'receipt.json') == expected['receipt_sha256'] and
                control.digest(folder / 'runtime.json') == expected['runtime_sha256'] and
                control.digest(folder / 'triggers.hdf') == expected['trigger_sha256'],
                'Run receipt or output differs: ' + folder.name)
        runtime = control.read(folder / 'runtime.json')
        for name, sha in runtime['source_modules'].items():
            require(control.digest(ROOT / Path(name).relative_to(REMOTE)) == sha,
                    'Imported source/native hash differs: ' + name)
        if folder.name != 'qualify-cpu':
            result = compare.compare(baseline, compare.load(folder / 'triggers.hdf'), compare.DEFAULTS)
            require(result['status'] == 'pass', 'Scientific comparison failed: ' + folder.name)
            comparisons[folder.name] = result
        if receipt['mode'] == 'timing':
            route = {'cpu:1': 'cpu', 'torch:cpu:1': 'torchcpu', 'torch:cuda:0': 'cuda'}[receipt['scheme']]
            row = control.timing(folder, 384, REV, BANK, qualifications[route])
            row['internal_post_setup_seconds'] = row['internal_seconds'] - row['setup_seconds']
            rows[route].append(row)
    require(len(comparisons) == 20 and all(len(v) == 3 for v in rows.values()), 'Run count differs')
    remote_summary = control.read(ROOT / 'timing-summary.json')
    summary = {}
    for name, samples in rows.items():
        walls = [r['full_wall_seconds'] for r in samples]
        median_row = sorted(samples, key=lambda r: r['full_wall_seconds'])[1]
        summary[name] = dict(
            samples=samples, median_wall_seconds=statistics.median(walls),
            min_wall_seconds=min(walls), max_wall_seconds=max(walls),
            median_template_seconds_per_wall_second=731136 / statistics.median(walls),
            median_wall_sample=median_row,
            relative_range=(max(walls) - min(walls)) / statistics.median(walls))
        require(math.isclose(summary[name]['median_wall_seconds'],
                             remote_summary['routes'][name]['median_wall_seconds'], rel_tol=1e-12),
                'Timing summary does not reproduce')
    raw_trace = (ROOT / 'runs/cuda-trace/trace/trace.json').read_bytes()
    reconstructed = trace.summarize(json.loads(raw_trace))
    saved = control.read(ROOT / 'device-attribution.json')
    require(saved['trace_sha256'] == control.digest(ROOT / 'runs/cuda-trace/trace/trace.json'),
            'Trace hash differs')
    require(all(saved[k] == v for k, v in reconstructed.items()), 'Device attribution does not reproduce')
    profiles = {}
    for script, saved_name in (('summarize-profiles.py', 'attribution-early-loops.json'),
                               ('summarize-callers.py', 'attribution-focused-callers.json')):
        run = subprocess.run([sys.executable, '-I', str(BASE / script), str(ROOT)],
                             capture_output=True, text=True, check=True, timeout=60)
        reproduced = json.loads(run.stdout)
        require(reproduced == control.read(BASE / saved_name), 'cProfile summary differs: ' + script)
        profiles[script] = 'pass'
    previous = None
    if PREVIOUS is not None:
        prior_receipt = PREVIOUS.with_name('receipt.json')
        pins = control.read(BASE / 'previous-cpu-manifest.json')['sha256']
        for name, sha in pins.items():
            require(control.digest(PREVIOUS.parent / name) == sha, 'Prior CPU file differs')
        originals = [compare.load(PREVIOUS), baseline]
        raw = compare.compare(*originals, compare.DEFAULTS)
        items = copy.deepcopy(originals)
        revisions = ['9578a710479b924e882857c4dffab6ed372a634b', REV]
        substitutions = []
        for receipt_path, item, revision in zip(
                (prior_receipt, ROOT / 'runs/qualify-cpu/receipt.json'), items, revisions):
            receipt = control.read(receipt_path)
            source = item['metadata']['source_snapshot']
            require(source == dict(commit=revision, status='', tracked_diff='') and
                    not item['metadata']['issues'] and receipt['scheme'] == 'cpu:1',
                    'Previous/current source pair differs')
            executable = receipt['executable_cli'][0]
            hashes = item['metadata']['consumed_input_sha256']
            sha = hashes.pop(executable)
            require(sha == 'd4af378d77aa5f66bcc018db32fe372360e53b22542e539ea3a2e53d31d8fa8a',
                    'Previous/current executable bytes differ')
            hashes['byte-identical executable'] = sha
            item['metadata']['source_snapshot'] = {'authorized_revisions': revisions}
            substitutions.append(dict(original_source=source, executable_path=executable,
                                      executable_sha256=sha))
        checked = compare.compare(*items, compare.DEFAULTS)
        require(checked['status'] == 'pass', 'Previous corrected science differs')
        previous = dict(raw_strict=raw, substitutions=substitutions, status=checked['status'],
                        tolerances=compare.DEFAULTS, comparison=checked)
    result = dict(state='pass', revision=REV, qualifications=qualifications,
                  numerical_budgets=compare.DEFAULTS, scientific_comparisons=comparisons,
                  timing=summary, archived_source_native_helper_pins_verified=True,
                  external_input_bytes_rechecked=False, external_input_receipt_pins_verified=True,
                  profiles_reproduced=profiles, previous_corrected_comparison=previous,
                  device_attribution_reproduced=True,
                  uncertainty='Three fresh workers per route; observed min/max, not confidence intervals',
                  classification='Shared-host 384-template finite workload; no old-versus-new speedup claim')
    with OUTPUT.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(state='pass', timing={n: {k: v for k, v in s.items()
                         if k not in ('samples', 'median_wall_sample')} for n, s in summary.items()},
                     comparisons=len(comparisons), trace='reproduced', profiles=profiles,
                     previous_comparison=previous['status'] if previous else 'not requested'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous', type=Path)
    args = parser.parse_args()
    ROOT = args.evidence.resolve(strict=True)
    OUTPUT = args.output.absolute()
    PREVIOUS = args.previous.resolve(strict=True) if args.previous else None
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), 'Output must be new')
    require(OUTPUT.parent.is_dir(), 'Output parent must exist')
    main()
