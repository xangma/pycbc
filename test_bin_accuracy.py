"""Analytic, exact-rational, capture-contract, and isolation checks."""
from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from bin_accuracy import audit, array_sha


class AuditTests(unittest.TestCase):
    def test_uniform_and_plateau_right_boundary(self):
        h = np.ones(64, np.complex64)
        r = audit(h, None, 0.25, 0, 64)
        self.assertEqual(r['paths']['old']['edges'], list(range(0, 65, 4)))
        self.assertEqual(r['references']['input_power']['own_bin_quality']['signed_errors'], [0]*16)
        h = np.array([0, 1, 0, 0, 1, 0], np.complex64)
        r = audit(h, None, 0.25, 0, 6, 2)
        self.assertEqual(r['references']['input_power']['edges'], [1, 4, 6])
        self.assertGreater(r['references']['input_power']['boundaries'][1]['computed_equal_prefix_cells'], 0)
        self.assertFalse(r['references']['input_power']['boundaries'][1]['stable'])

    def test_float32_scan_loses_small_tail(self):
        h = np.ones(4097, np.complex64)
        h[0] = 4096
        r = audit(h, None, 0.25, 0, len(h))
        self.assertEqual(r['paths']['old']['stagnated_positive_cells'], 4096)
        self.assertEqual(r['paths']['old']['final_plateau_rounded_power'], 4096)
        self.assertEqual(r['paths']['old']['scan_error_vs_rounded_power_scan_only']['total_signed'], -4096)
        self.assertEqual(r['paths']['current']['scan_error_vs_rounded_power_scan_only']['total_signed'], 0)

    def test_input_rounding_separate_from_scan(self):
        h = np.array([1 + 1j*2**-12]*8, np.complex64)
        r = audit(h, None, 0.25, 0, len(h), 2)
        self.assertEqual(r['input_power_rounding']['total_signed_error'], -8*2**-24)
        self.assertEqual(r['paths']['current']['scan_error_vs_rounded_power_scan_only']['total_signed'], 0)
        self.assertLess(r['paths']['current']['scan_error_vs_input_power']['total_signed'], 0)

    def test_public_rounding_and_capture_mismatch(self):
        h = np.arange(1, 20, dtype=np.float32).astype(np.complex64)
        arrays = {}
        r = audit(h, None, 0.1, 2, 18, 4, arrays=arrays)
        mag = np.float32(h[2:18].real**2 + h[2:18].imag**2)
        expected = np.cumsum(mag, dtype=np.float64).astype(np.float32)*np.float32(4*0.1)
        self.assertEqual(array_sha(expected), r['paths']['current']['public_prefix_sha256'])
        self.assertEqual(r['normalization']['applied_float32'], float(np.float32(0.4)))
        r = audit(h, None, 0.1, 2, 18, 4, mag_numpy=mag,
                  captured_current_prefix=expected, captured_current_bins=[2]*5)
        self.assertTrue(r['observed']['captured_current_prefix_byte_exact'])
        self.assertEqual(r['status'], 'capture_mismatch')

    def test_native_mag_authoritative_but_reference_independent(self):
        h = np.ones(32, np.complex64)
        mag = np.full(32, 2, np.float32)
        r = audit(h, None, 0.25, 0, 32, mag_numpy=mag)
        self.assertFalse(r['observed']['native_mag_equal_numpy'])
        self.assertEqual(r['references']['rounded_power_scan_only']['total'], 64)
        self.assertEqual(r['references']['input_power']['total'], 32)

    def test_exact_rational_oracle_and_cell_budget(self):
        rng = np.random.default_rng(4201)
        for _ in range(12):
            h = (rng.normal(size=37) + 1j*rng.normal(size=37)).astype(np.complex64)
            psd = np.exp(rng.uniform(-3, 3, 37)).astype(np.float32)
            r = audit(h, psd, 0.3, 2, 35, 7)
            weights = [(F(float(z.real))**2 + F(float(z.imag))**2)/F(float(p))
                       for z, p in zip(h[2:35], psd[2:35])]
            prefix, total = [], F(0)
            for weight in weights:
                total += weight
                prefix.append(total)
            oracle = r['references']['input_power']
            for i, boundary in enumerate(oracle['boundaries']):
                threshold = i*total/7
                edge = sum(value <= threshold for value in prefix) + 2
                self.assertLessEqual(boundary['possible_edge_min'], edge)
                self.assertGreaterEqual(boundary['possible_edge_max'], edge)
                if boundary['stable']:
                    self.assertEqual(boundary['edge'], edge)
            self.assertEqual(oracle['own_bin_quality']['beyond_discrete_cell_and_uncertainty'], 0)
            observed_total = F(oracle['total_decimal'])
            bound = F(oracle['total_uncertainty_absolute_decimal'])
            self.assertLessEqual(abs(observed_total - total), bound)

    def test_excluded_invalid_values_and_no_mutation(self):
        h = np.ones(40, np.complex64)
        psd = np.ones(40, np.float32)
        h[0] = np.nan
        psd[0] = np.inf
        before = h.tobytes(), psd.tobytes()
        audit(h, psd, 0.25, 2, 38)
        self.assertEqual(before, (h.tobytes(), psd.tobytes()))
        with self.assertRaises(ValueError):
            audit(h, psd, 0.25, 0, 38)
        with self.assertRaises(ValueError):
            audit(np.ones(10, np.complex128), None, 1, 0, 10)
        with self.assertRaises(ValueError):
            audit(np.zeros(10, np.complex64), None, 1, 0, 10)

    def test_cli_exact_capture_schema(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'capture'
            root.mkdir()
            h = np.ones(64, np.complex64)
            p = np.ones(64, np.float32)
            captured = dict(status='complete', arrays={})
            for name, value in [('template-000', h), ('psd-original', p), ('psd-proposed', p),
                                ('power-000', p), ('prefix-000', np.arange(1, 65, dtype=np.float32))]:
                path = root/(name+'.npy')
                np.save(path, value)
                captured['arrays'][path.name] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                                    data_sha256=array_sha(value), shape=list(value.shape), dtype=str(value.dtype))
            (root/'capture.json').write_text(json.dumps(captured))
            row = dict(index=0, template_hash=44, delta_f=0.25, kmin=0, kmax=64, num_bins=16,
                       file='template-000.npy', native_power_file='power-000.npy', current_prefix_file='prefix-000.npy',
                       captured_current_bins=list(range(0, 65, 4)))
            (root/'templates.json').write_text(json.dumps([row]))
            command = [sys.executable, '-B', str(Path(__file__).with_name('bin_accuracy.py')), str(root),
                       '--output-dir', str(Path(tmp)/'output')]
            subprocess.run(command, check=True, capture_output=True, text=True)
            results = [json.loads(x) for x in (Path(tmp)/'output/metrics.jsonl').read_text().splitlines()]
            self.assertFalse(results[0]['observed']['native_mag_supplied'])
            self.assertTrue(results[1]['observed']['native_mag_supplied'])
            self.assertTrue(results[1]['observed']['captured_current_bins_exact'])
            self.assertTrue(results[1]['observed']['captured_current_prefix_byte_exact'])
            self.assertTrue(json.loads((Path(tmp)/'output/summary.json').read_text())['inputs_unchanged_after_analysis'])


if __name__ == '__main__':
    unittest.main()
