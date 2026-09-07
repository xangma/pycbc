"""Read saved pstats without executing profiled code; retain nested timing semantics."""
import json
from pathlib import Path
import pstats
import sys

root = Path(sys.argv[1])
result = {}
for path in sorted(root.glob('runs/*/*.pstats')):
    parity = root / (path.parent.name + '-parity.json')
    if not parity.exists() or json.loads(parity.read_text()).get('status') != 'pass':
        continue
    stats = pstats.Stats(str(path))
    rows = []
    for (file, line, name), (primitive, calls, own, cumulative, callers) in stats.stats.items():
        row = dict(file=file, line=line, function=name, primitive_calls=primitive,
                   calls=calls, self_seconds=own, cumulative_seconds=cumulative)
        rows.append(row)
    semantic = [r for r in rows if any(term in r['function'].lower() for term in
                ('squared_norm', 'ifft', 'chisq', 'correlat', 'threshold', 'decompress',
                 'numpy', 'sigmasq', 'save_performance', 'first_bank'))]
    result[str(path.relative_to(root))] = dict(total_seconds=stats.total_tt,
        total_calls=stats.total_calls, primitive_calls=stats.prim_calls,
        top_self=sorted(rows, key=lambda r: -r['self_seconds'])[:40],
        top_cumulative=sorted(rows, key=lambda r: -r['cumulative_seconds'])[:40],
        semantic=sorted(semantic, key=lambda r: -r['cumulative_seconds']))
print(json.dumps(dict(scope='Instrumented attribution only; cumulative rows nest and must not be summed',
                      profiles=result), indent=2, allow_nan=False))
