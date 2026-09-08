"""Report integrity gates, boundary distinctions, and analytic lambda checks."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from bin_accuracy import audit
from aggregate_bin_report import MODULE_SHA, aggregate, classify, edge_stats, signal_imbalance


def fixture():
    rows = []
    for index in range(2):
        rng = np.random.default_rng(index+30)
        h = (rng.normal(size=96)+1j*rng.normal(size=96)).astype(np.complex64)
        for label in ('original', 'proposed'):
            row = audit(h, np.ones(96, np.float32), .25, 0, 96, 4,
                        dict(index=index, template_hash=100+index, audit_psd=label))
            # This is a synthetic serialized-report fixture for aggregation.
            # No extended-precision execution is claimed by these unit tests.
            row['runtime']['longdouble_wider_than_float64'] = True
            row['runtime']['longdouble_significand_bits'] = 64
            rows.append(row)
    summary = dict(schema='pycbc-bin-accuracy-v1', status='complete',
                   inputs_unchanged_after_analysis=True, changed_inputs=[], capture_mismatches=[],
                   module_sha256=MODULE_SHA, rows=4, template_count=2, input_sha256={}, by_psd={})
    for label in ('original', 'proposed'):
        selected = [r for r in rows if r['metadata']['audit_psd'] == label]
        stats = dict(max_old_current_edge_shift=max(r['old_current_max_edge_difference'] for r in selected),
                     old_total_error_closer_templates=0, current_total_error_closer_templates=0,
                     tied_total_error_templates=0)
        for key, suffix in [('old_closer', 'old_closer'), ('current_closer', 'current_closer'), ('same_edge', 'same'), ('unresolved', 'unresolved')]:
            stats['input_reference_'+suffix+'_edges'] = sum(
                r['references']['input_power']['old_current_edge_comparison']['internal_uncertainty_aware_counts'][key] for r in selected)
        for row in selected:
            old = abs(row['paths']['old']['scan_error_vs_input_power']['total_signed'])
            current = abs(row['paths']['current']['scan_error_vs_input_power']['total_signed'])
            key = 'old_total_error_closer_templates' if old < current else 'current_total_error_closer_templates' if current < old else 'tied_total_error_templates'
            stats[key] += 1
        summary['by_psd'][label] = stats
    return rows, summary


class AggregateTests(unittest.TestCase):
    def test_classification_retains_uncertainty(self):
        self.assertEqual(classify(8, 9, 8, 8), 'old_closer')
        self.assertEqual(classify(8, 9, 9, 9), 'current_closer')
        self.assertEqual(classify(8, 9, 8, 9), 'unresolved')
        self.assertEqual(classify(8, 8, 7, 9), 'same_edge')
        self.assertEqual(classify(8, 10, 9, 9), 'unresolved')

    def test_one_cell_off_is_not_exact(self):
        rows, _ = fixture()
        row = copy.deepcopy(rows[0])
        oracle = row['references']['input_power']
        self.assertTrue(all(b['stable'] for b in oracle['boundaries'][1:]))
        row['paths']['old']['edges'] = list(oracle['edges'])
        row['paths']['current']['edges'] = list(oracle['edges'])
        row['paths']['current']['edges'][1] += 1
        oracle['old_current_edge_comparison']['internal_uncertainty_aware_counts'] = dict(old_closer=1, current_closer=0, same_edge=2, unresolved=0)
        result = edge_stats([row], 'input_power', [])
        self.assertEqual(result['paths']['current']['stable_exact'], 2)
        self.assertEqual(result['paths']['current']['stable_one_cell_off'], 1)
        self.assertEqual(result['paths']['current']['stable_more_than_one_cell_off'], 0)

    def test_signal_lambda_formula_and_zero_oracle(self):
        quality = dict(relative_to_target=[.1, -.1, 0., 0.])
        row = dict(inputs=dict(num_bins=4), metadata=dict(index=9),
                   paths={p: dict(input_reference_bin_quality=quality) for p in ('old', 'current')},
                   references=dict(input_power=dict(own_bin_quality=dict(relative_to_target=[0.]*4))))
        result = signal_imbalance([row])
        self.assertAlmostEqual(result['old']['by_snr']['20']['max'], 2.)
        self.assertAlmostEqual(result['old']['by_snr']['100']['max'], 50.)
        self.assertAlmostEqual(result['old']['by_snr']['5.5']['max'], .15125)
        self.assertEqual(result['oracle']['by_snr']['100']['max'], 0.)

    def test_summary_coverage_precision_and_pairing_gates(self):
        rows, summary = fixture()
        report, templates, boundaries = aggregate(rows, summary, 2)
        self.assertEqual(len(templates), 4)
        self.assertEqual(len(boundaries), 24)
        self.assertEqual(report['paired_psd_effects_fixed_waveform']['current']['changed_internal_edges'], 0)
        with self.assertRaises(ValueError):
            aggregate(rows[:-1], summary, 2)
        bad = copy.deepcopy(summary)
        bad['by_psd']['original']['input_reference_current_closer_edges'] += 1
        with self.assertRaises(ValueError):
            aggregate(rows, bad, 2)
        bad_rows = copy.deepcopy(rows)
        bad_rows[0]['runtime']['longdouble_wider_than_float64'] = False
        with self.assertRaises(ValueError):
            aggregate(bad_rows, summary, 2)
        bad_rows = copy.deepcopy(rows)
        bad_rows[0]['inputs']['active_htilde_sha256'] = 'different'
        with self.assertRaises(ValueError):
            aggregate(bad_rows, summary, 2)

    def test_cli_artifacts_and_immutability(self):
        rows, summary = fixture()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'input'
            root.mkdir()
            metrics = '\n'.join(json.dumps(row) for row in rows)+'\n'
            (root/'metrics.jsonl').write_text(metrics)
            (root/'summary.json').write_text(json.dumps(summary))
            command = [sys.executable, '-B', str(Path(__file__).with_name('aggregate_bin_report.py')),
                       str(root), '--output-dir', str(Path(folder)/'output'), '--expected-templates', '2']
            subprocess.run(command, check=True, capture_output=True)
            out = Path(folder)/'output'
            self.assertEqual({p.name for p in out.iterdir()},
                             {'bin-aggregate.json', 'bin-report.md', 'bin-templates.csv', 'bin-boundaries.csv'})
            text = (out/'bin-report.md').read_text()
            self.assertIn('One cell off', text)
            self.assertIn('not an injected-strain recovery result', text)
            self.assertIn('SNR 100', text)
            self.assertEqual((root/'metrics.jsonl').read_text(), metrics)


if __name__ == '__main__':
    unittest.main()
