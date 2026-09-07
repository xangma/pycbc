"""Read recorded cProfile caller/callee edges for focused ownership analysis."""
import hashlib
import json
from pathlib import Path
import pstats
import sys

root = Path(sys.argv[1])
result = {}
for mode in ('cprofile', 'filter-cprofile'):
    folder = root / 'runs' / (mode + '-torchcpu')
    if json.loads((root / (folder.name + '-parity.json')).read_text())['status'] != 'pass':
        raise ValueError('Profile did not pass parity')
    path = folder / ('profile.pstats' if mode == 'cprofile' else 'filtering.pstats')
    stats = pstats.Stats(str(path))
    stats.calc_callees()

    def edges(values):
        rows = []
        for key, value in values.items():
            rows.append(dict(file=key[0], line=key[1], function=key[2],
                             raw_cprofile_edge=list(value)))
        return sorted(rows, key=lambda r: -r['raw_cprofile_edge'][3])[:16]

    selected = []
    for key, value in stats.stats.items():
        if not (key[2] in ('deepcopy', 'squared_norm') or
                key[2] == "<method 'copy_' of 'torch._C.TensorBase' objects>"):
            continue
        selected.append(dict(file=key[0], line=key[1], function=key[2],
                             primitive_calls=value[0], calls=value[1],
                             self_seconds=value[2], cumulative_seconds=value[3],
                             callers=edges(value[4]), callees=edges(stats.all_callees[key])))
    result[str(path.relative_to(root))] = dict(
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        total_seconds=stats.total_tt, selected=selected)
print(json.dumps(dict(
    scope='Recorded cProfile only; nested/recursive cumulative edge durations must not be summed',
    edge_fields=['total_calls', 'primitive_calls', 'self_seconds', 'cumulative_seconds'],
    profiles=result), indent=2))
