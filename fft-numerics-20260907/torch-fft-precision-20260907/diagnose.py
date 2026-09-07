#!/usr/bin/env python3
"""Reproduce frozen public-filter outputs; attribute error using complex128.

Diagnostic evidence only. This cannot qualify or authorize campaign timings.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--campaign-root', type=Path, required=True)
p.add_argument('--output-root', type=Path, required=True)
p.add_argument('--route', required=True)
p.add_argument('--batch', type=int, default=1)
opt = p.parse_args()
root = opt.campaign_root.resolve()
out = opt.output_root.resolve()
out.mkdir(parents=True, exist_ok=False)
spec = importlib.util.spec_from_file_location('frozen_worker', root / 'batch-worker.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
frozen_path = root / f'runs/qual-b{opt.batch}-{opt.route}/result.json'
frozen = json.loads(frozen_path.read_text())
reference_path = root / 'runs/qual-b1-branch_standard/result.json'
reference = json.loads(reference_path.read_text())
selected = {}
for b, block in enumerate(frozen['pointwise_blocks']):
    rows = sorted(block['rows'], key=lambda r: r['max_tolerance_ratio'], reverse=True)[:8]
    selected[b] = {r['template_id'] - 1000 for r in rows} | {
        inj['template_id'] - 1000 for inj in frozen['inputs']['injections'][b]}
material = {}
original_workload = w.workload


def workload(args):
    values = original_workload(args)
    material.update(bank=values[0], blocks=values[3])
    return values


w.workload = workload
observations = []


class DiagnosticCapture(w.Capture):
    def capture(self, group_index, triggers, veto):
        np = w.np
        group = self.filt.tgroups[group_index]
        for template in group:
            row_id = int(template.id) - 1000
            if row_id not in selected[self.block]:
                continue
            actual = template.out.numpy().copy()
            corr = template.cout.numpy().copy()
            ref = self.output_map[self.block, row_id].copy()
            bank = material['bank'][row_id].astype(np.complex128)
            strain = material['blocks'][self.block].astype(np.complex128)
            exact_corr = np.zeros(self.args.size, dtype=np.complex128)
            exact_corr[:len(bank)] = bank.conj() * strain
            exact = np.fft.ifft(exact_corr) * self.args.size
            same_corr_fft = np.fft.ifft(corr.astype(np.complex128)) * self.args.size
            pair = w.row_metrics(actual, ref)
            limit = w.POLICY['pointwise_complex']['atol'] + w.POLICY['pointwise_complex']['rtol'] * np.abs(ref.astype(np.complex128))
            failures = np.flatnonzero(np.abs(actual.astype(np.complex128) - ref) > limit)
            details = []
            freq = np.arange(len(bank))
            for i in failures[:5]:
                # Independent O(N) direct DFT at selected failed samples.
                direct = np.sum(exact_corr[:len(bank)] * np.exp(2j * np.pi * freq * int(i) / self.args.size), dtype=np.complex128)
                details.append({'sample': int(i), 'reference_magnitude': float(abs(ref[i])),
                                'pair_error': float(abs(actual[i] - ref[i])),
                                'reference_error_vs_exact': float(abs(ref[i] - exact[i])),
                                'actual_error_vs_exact': float(abs(actual[i] - exact[i])),
                                'actual_fft_error_on_same_correlation': float(abs(actual[i] - same_corr_fft[i])),
                                'correlation_error_in_output': float(abs(same_corr_fft[i] - exact[i])),
                                'direct_dft_vs_double_fft': float(abs(direct - exact[i]))})
            expected = frozen['pointwise_blocks'][self.block]['rows'][row_id]['sha256']
            expected_ref = reference['pointwise_blocks'][self.block]['rows'][row_id]['sha256']
            obs = {'block': self.block, 'template_id': int(template.id),
                   'actual_matches_frozen': w.digest_bytes(actual) == expected,
                   'reference_matches_frozen': w.digest_bytes(ref) == expected_ref,
                   'actual_vs_reference': pair,
                   'reference_vs_exact': w.row_metrics(ref, exact),
                   'actual_vs_exact': w.row_metrics(actual, exact),
                   'actual_fft_vs_same_correlation': w.row_metrics(actual, same_corr_fft),
                   'rounded_correlation_vs_exact': w.row_metrics(same_corr_fft, exact),
                   'maximum_normalized_snr_error': pair['max_abs_error'] * 4 * frozen['inputs']['geometry']['delta_f'] / material['sigma'][row_id] ** .5,
                   'failures': details}
            observations.append(obs)
            np.savez(out / f'block{self.block}-template{template.id}.npz', actual=actual, reference=ref, correlation=corr, exact_correlation=exact_corr)
        for _snr, _norm, peak, template, _strain in veto:
            self.indices[int(template.id)] = int(peak)

    def finish_block(self):
        self.result['pointwise_blocks'].append({'block': self.block, 'diagnostic_only': True})
        print(json.dumps({'block': self.block, 'observations': len(observations)}), flush=True)


def workload_with_sigma(args):
    values = workload(args)
    material['sigma'] = values[2]
    return values


w.workload = workload_with_sigma
w.Capture = DiagnosticCapture
args = argparse.Namespace(**frozen['arguments'])
args.source_root = root / 'source'
args.reference_dir = reference_path.parent
args.output_dir = out
result = {'worker_sha256': hashlib.sha256((root / 'batch-worker.py').read_bytes()).hexdigest(), 'failures': []}
w.execute(args, result)
report = {'purpose': __doc__, 'route': opt.route, 'batch': opt.batch,
          'campaign_source': w.EXPECTED_HEAD, 'diagnostic_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'frozen_result_sha256': hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
          'runtime': result['runtime'], 'routing': result['routing'], 'input_sha256': result['input_sha256'],
          'observations': observations, 'trigger_comparisons': result['trigger_comparisons'],
          'all_actual_hashes_match': all(r['actual_matches_frozen'] for r in observations),
          'all_reference_hashes_match': all(r['reference_matches_frozen'] for r in observations),
          'pid': os.getpid(), 'affinity': sorted(os.sched_getaffinity(0)),
          'status': 'diagnostic_complete', 'timing_authorized': False}
w.atomic_json(out / 'diagnostic.json', report)
print(json.dumps({k: report[k] for k in ['status', 'all_actual_hashes_match', 'all_reference_hashes_match']}), flush=True)
sys.exit(0 if report['all_actual_hashes_match'] and report['all_reference_hashes_match'] else 1)
