#!/usr/bin/env python3
"""Derive compact, gated tables from a completed len acquisition."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--prefix', default='qualified')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root, prefix = args.root, args.prefix
inputs = {}


def read(path):
    data = (root / path).read_bytes()
    inputs[path] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def stats(values):
    return {'count': len(values), 'median': statistics.median(values),
            'minimum': min(values), 'maximum': max(values),
            'mean': statistics.mean(values),
            'population_stddev': statistics.pstdev(values), 'raw': values}


summary = {'scope': 'Finite fixtures only; timing boundaries are separate',
           'providers': [], 'generation': [], 'offline': []}
lines = ['# Measured results on len', '',
         'Generated from receipts by `summarize_results.py`. All times below are milliseconds unless marked otherwise.', '',
         '## Provider preparation and prepared filtering', '',
         'B=16, N=2048, 1024 Hz, reference-generated fixed strain. Five fresh processes per route, each with one first drain and twenty prepared drains. Every actual drain was checked outside its timer. Medians below are across workers; the prepared column is the median of worker medians.', '',
         '| Execution | Provider | Generation ready | Engine preparation | Generation through first drain | Prepared submit/drain |',
         '|---|---|---:|---:|---:|---:|']
commits = set()
for mode in ('cpu1', 'cpu4', 'cuda'):
    data = read(f'logs/{prefix}-provider-{mode}.json')
    assert data['status'] == 'passed'
    for provider in ('reference', 'torchwave'):
        runs = [r for r in data['runs'] if r['measurement']['provider'] == provider]
        assert len(runs) == 5
        assert all(r['returncode'] == 0 and r['child_qualification']['passed'] for r in runs)
        measurements = [r['measurement'] for r in runs]
        for measurement in measurements:
            assert measurement['status'] == 'passed'
            assert len(measurement['result_qualification']) == 21
            assert all(q['passed'] for q in measurement['result_qualification'])
            commits.add(measurement['provenance']['repositories']['pycbc']['git_commit'])
        entry = {'mode': mode, 'provider': provider, 'unit': 'seconds',
                 'generation': stats([m['generation_to_device_ready_sec'] for m in measurements]),
                 'setup': stats([m['bank_psd_bins_engine_setup_sec'] for m in measurements]),
                 'through_first_drain': stats([m['generation_through_first_drain_sec'] for m in measurements]),
                 'prepared_worker_medians': stats([statistics.median(m['raw_prepared_submit_drain_sec']) for m in measurements]),
                 'prepared_samples_per_worker': [m['raw_prepared_submit_drain_sec'] for m in measurements],
                 'tool_process_wall': stats([r['process_wall_time_sec'] for r in runs]),
                 'first_drain_candidate_count': sum(q['candidate_count'] for q in measurements[0]['result_qualification'][0]['rows'])}
        summary['providers'].append(entry)
        columns = [entry[k]['median'] * 1000 for k in ('generation', 'setup', 'through_first_drain', 'prepared_worker_medians')]
        lines.append(f'| {mode} | {provider} | ' + ' | '.join(f'{v:.3f}' for v in columns) + ' |')
summary['source_commits'] = sorted(commits)
assert len(commits) == 1, 'Mixed source commits in primary provider acquisition'
lines += ['', 'Generation includes bank construction in this experiment. Prepared filtering excludes generation and setup. Outer process wall times also include imports, fixture construction, independent qualification, provenance and output; they are retained in JSON and are not operational latency.', '',
          '## Warm generation only', '',
          'Prepared bank/arguments to a completed device tensor, five rotated samples per route; all rows pass unaligned complex-L2, norm, exact support, dtype and device gates before and after timing. PyCBC native prebuilds argument tensors and generates both polarizations; reference/TorchWave include FilterBank metadata. These are explicit API cost differences.', '',
          '| Execution | N | Reference | PyCBC native TaylorF2 | TorchWave |',
          '|---|---:|---:|---:|---:|']
for mode in ('cpu1', 'cpu4', 'cuda'):
    for n in (2048, 131072):
        data = read(f'logs/{prefix}-generation-{mode}-n{n}.json')
        assert data['status'] == 'passed'
        assert data['provenance']['repositories']['pycbc']['git_commit'] in commits
        entry = {'mode': mode, 'transform_length': n, 'batch_size': 16,
                 'source_commit': data['provenance']['repositories']['pycbc']['git_commit'],
                 'unit': 'seconds', 'providers': {p: stats([s['generation_to_device_ready_sec'] for s in rows])
                                                  for p, rows in data['measurements'].items()}}
        summary['generation'].append(entry)
        lines.append(f'| {mode} | {n} | ' + ' | '.join(f"{entry['providers'][p]['median'] * 1000:.3f}" for p in ('reference', 'pycbc_native', 'torchwave')) + ' |')
lines += ['', '## Actual offline executable', '',
          'One persistent worker per route, two identical injected-frame shards, four templates, N=65536 at 1024 Hz. Campaign wall time is seconds; each shard is elapsed executable time within that worker. A single campaign is dispatch/reuse evidence, not a statistically qualified speed comparison.', '',
          '| Route | Campaign seconds | First shard seconds | Second shard seconds | Second-shard batch hits |',
          '|---|---:|---:|---:|---:|']
for route in ('native-cuda', 'reference-cuda', 'reference-cpu'):
    data = read(f'{prefix}-cli-{route}/receipt.json')
    assert data['status'] == 'completed'
    entry = {'route': route, 'campaign_seconds': data['total_campaign_s'], 'shards': data['shards']}
    summary['offline'].append(entry)
    shards = data['shards']
    assert len(shards) == 2 and shards[0]['pid'] == shards[1]['pid']
    lines.append(f"| {route} | {data['total_campaign_s']:.3f} | {shards[0]['elapsed_s']:.3f} | {shards[1]['elapsed_s']:.3f} | {shards[1]['cache_delta']['batch_hits']} |")
comparison = read(f'logs/{prefix}-cli-contract-comparison.json')
assert comparison['status'] == 'pass' and not comparison['fatal_errors']
summary['offline_comparison'] = comparison
live = read(f'{prefix}-live/manifest.json')
assert live['status'] == 'passed' and all(live['checks'].values())
summary['live'] = live
lines += ['', f"Live: two MPI ranks completed the two-detector 32-second noise fixture in {live['wall_seconds']:.3f} seconds. All seven dispatch, device and HDF checks passed. Its SNR threshold is 1e6; this checks the empty-trigger executable path, not detection efficiency or live trigger parity."]
engine = read(f'logs/{prefix}-engine.json')
graph = engine['benchmarks']['cuda_graph_comparison']
stream = engine['benchmarks']['live_streaming_latency']
assert engine['status'] == 'measured'
assert graph['status'] == 'passed'
assert graph['output_equivalence']['passed'] and graph['graph_execution_gate']['passed']
summary['graph'] = {k:v for k,v in graph.items() if k != 'input_hashes'}
summary['streaming'] = stream
lines += ['', '## Graph and streaming diagnostics', '',
          f"N={graph['transform_length']}, templates={graph['num_templates']}, tile={graph['tile_size']}, {graph['iterations']} repetitions. Mean synchronized eager/graph submit-drain: {statistics.mean(graph['raw_eager_ms']):.3f}/{statistics.mean(graph['raw_graph_ms']):.3f} ms. Graph coverage is correlation/IFFT only. This normalized synthetic fixture has no selected candidates; positive-candidate qualification is separate.", '',
          f"Streaming measured {stream['num_blocks']} blocks after warmup, allocated memory {stream['vram_initial_mb']:.4f} to {stream['vram_final_mb']:.4f} MiB, peak {stream['vram_peak_mb']:.4f} MiB. Raw latency and memory samples are retained; unchanged endpoints do not prove absence of leaks."]
profile = read(f'logs/{prefix}-profile/candidate_pipeline_profile.json')
assert profile['status'] == 'passed'
assert profile['small_direct_gate']['passed'] and profile['preprofile_gate']['passed']
assert len(profile['profiled_result_checks']) == 3 and all(c['passed'] for c in profile['profiled_result_checks'])
summary['profile'] = {k:profile[k] for k in ('small_direct_gate', 'preprofile_gate', 'isolated_veto_memory', 'trace')}
veto = read(f'logs/{prefix}-power-chisq-before-after.json')
assert veto['all_final_scientific_gates_passed'] and veto['source']['final_commit'] in commits
summary['veto'] = veto
lines += ['', '## Identical-input veto comparison', '',
          'N=131072, eight candidate times, four correlation rows, sixteen bins; five synchronized calls per implementation after two warmups. Times include the common NumPy output boundary. Allocation peaks exclude caller inputs, host allocations, non-Torch allocations and allocator caches.', '',
          '| Input case | Implementation | FP64 gate | Median ms | Peak new Torch bytes | Maximum absolute error |',
          '|---|---|---|---:|---:|---:|']
for case in veto['cases']:
    for provider, values in case['implementations'].items():
        lines.append(f"| {case['name']} | {provider} | {values['scientific_gate_passed']} | {values['median_seconds'] * 1000:.3f} | {values['maximum_peak_new_torch_allocation_bytes']} | {values['max_absolute_error']:.6g} |")
lines += ['', 'Any case with a failed scientific gate has diagnostic timings only; no equivalent-output speed ratio is inferred for that case.', '']
tests_path = root / f'logs/{prefix}-tests.xml'
tests = ET.parse(tests_path).getroot()
summary['tests'] = [{k:s.attrib[k] for k in ('tests', 'errors', 'failures', 'skipped', 'time')} for s in tests.iter('testsuite')]
assert all(s['errors'] == s['failures'] == s['skipped'] == '0' for s in summary['tests'])
inputs[str(tests_path.relative_to(root))] = hashlib.sha256(tests_path.read_bytes()).hexdigest()
summary['inputs_sha256'] = inputs
args.output.mkdir(parents=True, exist_ok=True)
(args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
(args.output / 'measurements.md').write_text('\n'.join(lines))
print(args.output)
