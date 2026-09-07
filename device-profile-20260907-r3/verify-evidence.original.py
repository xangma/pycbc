"""Rebuild scientific comparisons and timing summaries from the copied raw evidence."""
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE / 'remote-evidence'
REMOTE = Path('/home/xangma/pycbc-torch-profile-20260907-r3')
REV = 'd2647addb884ead3249914ebc980f3c132076d93'
BANK = '26050d48322a1d71092bb0e024e71a89ace56b3e7b1c5e4cf20c7b769213fb7f'


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    control = load_module('campaign_controls', BASE / 'acquisition/campaign_controls.py')
    compare = load_module('frozen_compare', BASE / 'acquisition/compare-triggers.py')
    trace = load_module('trace_summary', BASE / 'acquisition/summarize_torch_trace.py')
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
    result = dict(state='pass', revision=REV, qualifications=qualifications,
                  numerical_budgets=compare.DEFAULTS, scientific_comparisons=comparisons,
                  timing=summary, native_input_pins_verified=True,
                  device_attribution_reproduced=True,
                  uncertainty='Three fresh workers per route; observed min/max, not confidence intervals',
                  classification='Shared-host 384-template finite workload; no old-versus-new speedup claim')
    with (BASE / 'evidence-verification.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(state='pass', timing={n: {k: v for k, v in s.items()
                         if k not in ('samples', 'median_wall_sample')} for n, s in summary.items()},
                     comparisons=len(comparisons), trace='reproduced'), indent=2))


if __name__ == '__main__':
    main()
