"""Bounded negative controls for asymmetric field sets and trigger identity."""
import copy
import numpy as np
from verify_evidence import field_comparison, independent_metrics, read, O, datasets

folder = O / 'campaign-v2-results'
a = datasets(folder / 'runs/qual-original-cpu')
b = datasets(folder / 'runs/qual-proposed-cpu')
r = read(folder / 'comparisons/original-cpu-vs-proposed-cpu.json')['result']
checks = []
for label, mutate in [
    ('right-only scientific field', lambda x: x.update(extra_scientific=np.zeros(1988))),
    ('missing scientific field', lambda x: x.pop('snr')),
    ('duplicate identity', lambda x: (x['template_hash'].__setitem__(1, x['template_hash'][0]), x['end_time'].__setitem__(1, x['end_time'][0]))),
    ('numerical violation', lambda x: x['snr'].__setitem__(0, x['snr'][0] + 1)),
]:
    altered = copy.deepcopy(b)
    mutate(altered)
    try:
        independent_metrics(a, altered, r)
    except ValueError:
        checks.append(label)
    else:
        raise AssertionError('Verifier accepted ' + label)
changed = copy.deepcopy(b)
changed['snr'] = changed['snr'].astype(np.float64)
assert not field_comparison(a, changed)['datasets']['snr']['dtype_equal']
checks.append('dtype difference detected')
print('PASS:', ', '.join(checks))
