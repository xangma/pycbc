"""Archive completed and independently verified fresh benchmark results."""
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess

O = Path(__file__).resolve().parent
E = O.parent / 'cpu-precision-evidence-20260908'
D = E / 'torch-fresh-benchmark'
s = json.loads((O / 'remote/summary.json').read_text())
v = json.loads((O / 'independent-verification/verification.json').read_text())
d = json.loads((O / 'additional-source-metadata/dependencies-actual.json').read_text())
assert d['torch']['version'] == '2.13.0+cu130' and d['torch']['cuda'] == '13.0'
assert len(d['all_torch_distributions']) > 1
assert json.loads((O / 'remote/status.json').read_text())['state'] == 'complete'
assert s['scientific_gates_pass'] and s['cpu_preserved'] and s['timing_outputs_pass']
assert s['equal_output_speedup_eligible'] and not s['sustained_capacity_established']
assert v['status'] == 'pass', v
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=E)
labels = {'original-cpu': 'Original CPU', 'proposed-cpu': 'Candidate CPU',
          'torch-cpu': 'Torch CPU', 'torch-cuda': 'Torch CUDA'}
table = ['| Route | Median (s) | Range (s) | Original CPU / route | Template-seconds / wall second |',
         '|---|---:|---:|---:|---:|']
baseline = s['arms']['original-cpu']['median_seconds']
for arm, label in labels.items():
    r = s['arms'][arm]
    assert len(r['samples_seconds']) == 4
    assert statistics.median(r['samples_seconds']) == r['median_seconds']
    table.append(f"| {label} | {r['median_seconds']:.2f} | {r['min_seconds']:.2f}–{r['max_seconds']:.2f} | {baseline / r['median_seconds']:.2f}× | {r['template_seconds_per_wall_second']:,.0f} |")
D.mkdir()
for name in ['remote', 'independent-verification', 'additional-source-metadata', 'manifest.json', 'launch-campaign.py',
             'transfer-verification.json', 'prepare-evidence.py', 'collect-dependencies.py', 'driver.log']:
    p = O / name
    assert p.exists(), p
    if p.is_dir():
        # Keep every remote manifest entry, including the recorded bytecode files.
        shutil.copytree(p, D / name)
    else:
        shutil.copy2(p, D / name)
pins = json.loads((D / 'remote/output-manifest.json').read_text())
for name, sha in pins.items():
    assert hashlib.sha256((D / 'remote' / name).read_bytes()).hexdigest() == sha, name
(D / 'README.md').write_text(f'''# Fresh Torch complete-executable benchmark — 2026-09-08

Four new scientific qualifications followed by **16 fresh unprofiled timed processes**, four per route in balanced rotating order. Original CPU `{s['source_commits']['original']}` is unchanged; candidate main `{s['source_commits']['proposed']}` is the exact published main at launch. Optional FFT and native CPU optimization leaves are outside this measurement.

{chr(10).join(table)}

Workload: **384 compressed templates × 1904 unique H1 seconds = 731,136 template-seconds**, five analysis segments and 1920 template/segment pairs. All four routes produce **1988 triggers**. The complete commands, input hashes and 512/112/16-second geometry are in [configuration](remote/config.json) and qualification receipts.

Host `len`: AMD Ryzen Threadripper PRO 3995WX, affinity fixed to logical CPU 8 (SMT sibling 72), numerical thread pools fixed to one, and NVIDIA GeForce RTX 4090 for CUDA. The Torch routes additionally fix intra/inter-op counts to one. Host and GPU were shared and unreserved. These are observed finite-workload medians and ranges, not confidence intervals, sustained capacity or a full-machine comparison.

Timing starts immediately before launching the checked executable process and stops at its exit. It includes imports, frame I/O, conditioning, PSD estimation, template preparation, filtering, vetoes, HDF output, runtime receipts and loaded-module verification. Qualification instrumentation, parent source/input hashing and output comparisons are outside the timing boundary. Qualification durations are excluded from the table. CUDA graphs were disabled. Raw samples, source/native/module hashes, process environment, resource snapshots and `/usr/bin/time` output are retained for every run.

## Scientific qualification and independent verification

**PASS:** all five cross-route trigger and complete-PSD comparisons under unchanged tolerances. No missing or extra trigger identities or numerical violations. Original versus candidate CPU has byte-exact equality for all 18 scientific H1 datasets and complete PSD arrays; conditioned strain matches through full-array hashes and metadata, and geometry matches. Raw conditioned-strain arrays were not archived. Four elapsed-time-derived H1/search fields are excluded from scientific byte equality. Raw provenance verdicts and the two explicitly verified source/executable provenance substitutions are retained.

Each of the 16 timed outputs passed comparison with its own route's fresh qualification. [Independent verification](independent-verification/verification.json) recomputes scientific comparisons, timing statistics and source/receipt checks. [Summary](remote/summary.json), [all comparisons](remote/comparisons/), [run receipts and outputs](remote/runs/), [transfer verification](transfer-verification.json).

The source bundle requires original CPU `{s['source_commits']['original']}` as a prerequisite. Generated version metadata matches each measured commit; its bytes are included in `additional-source-metadata`. Actual Torch run receipts report **Torch 2.13.0+cu130 / CUDA 13.0**. The original flat package inventory incorrectly lists Torch 2.1.1 because it collapses duplicate distribution names. [Supplemental dependency inspection](additional-source-metadata/dependencies-actual.json), collected after timing, records the selected import and every Torch distribution path/version. The original inventory is retained as collected. Input data and native binaries are referenced by location and hash; their large payloads are not duplicated in this archive. Documentation and final branch mappings will be appended separately; benchmark records are immutable.
''')
p = E / 'README.md'
p.write_text('# Fresh Torch benchmark, 2026-09-08\n\nLatest results: [fresh complete-executable timings](torch-fresh-benchmark/README.md). Four qualifications and 16 timed runs passed; original CPU is unchanged. Earlier sections below are historical snapshots.\n\n---\n\n' + p.read_text())
files = {str(p.relative_to(E)): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in E.rglob('*') if p.is_file() and '.git' not in p.parts and p.name != 'SHA256SUMS.json'}
(E / 'SHA256SUMS.json').write_text(json.dumps(files, indent=2, sort_keys=True) + '\n')
print('Prepared', len(files), 'checksummed files; fresh archive bytes', sum(p.stat().st_size for p in D.rglob('*') if p.is_file()))
