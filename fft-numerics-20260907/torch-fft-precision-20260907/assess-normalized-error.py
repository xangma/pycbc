"""Offline audit using frozen row maxima; not campaign qualification."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
root = Path(__file__).resolve().parent
campaign = root.parent / 'torch-current-batch-sweep-20260907-r3'
spec = importlib.util.spec_from_file_location('worker', campaign / 'batch-worker.py')
w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w); w.np = np
ref = json.loads((campaign / 'runs/qual-b1-branch_standard/result.json').read_text())
args = SimpleNamespace(**ref['arguments'])
bank, psd, sigma, blocks, geometry, injections, identity = w.workload(args)
assert all(v == ref['inputs'][k] for k, v in identity.items())
cells = []
for batch in w.BATCHES:
 for route in w.ROUTES:
  path = campaign / f'runs/qual-b{batch}-{route}/result.json'
  d = json.loads(path.read_text())
  rows = [r for b in d['pointwise_blocks'] for r in b['rows']]
  for r in rows:
   r['maximum_normalized_snr_error'] = r['max_abs_error'] * 4 * geometry['delta_f'] / np.sqrt(sigma[r['template_id']-1000])
  cells.append({'batch': batch, 'route': route, 'original_status': d['status'],
                'sample_count': sum(r['samples'] for r in rows),
                'raw_policy_failures': sum(r['failed_samples'] for r in rows),
                'max_raw_tolerance_ratio': max(r['max_tolerance_ratio'] for r in rows),
                'max_complex_snr_error': max(r['maximum_normalized_snr_error'] for r in rows),
                'max_relative_l2_error': max(r['relative_l2_error'] for r in rows),
                'all_triggers_pass': all(c['passed'] for c in d['trigger_comparisons']),
                'nonfinite_samples': sum(r['nonfinite_actual'] for r in rows),
                'all_rows_processed_once': all(b['processed_once'] for b in d['pointwise_blocks']),
                'result_sha256': __import__('hashlib').sha256(path.read_bytes()).hexdigest()})
report = {'purpose': __doc__, 'source': w.EXPECTED_HEAD, 'sigmasq_sha256': w.digest_bytes(sigma),
          'normalization': '4 * delta_f / sqrt(sigmasq[template_id])',
          'scope': 'All frozen complex samples: exact maxima derived from per-row maximum absolute error and independently regenerated hash-matching sigmasq.',
          'timing_authorized': False, 'cells': cells}
(root/'normalized-error-audit.json').write_text(json.dumps(report,indent=2)+'\n')
for c in cells:
 print(c['batch'],c['route'],c['raw_policy_failures'],c['max_complex_snr_error'],c['max_relative_l2_error'])
