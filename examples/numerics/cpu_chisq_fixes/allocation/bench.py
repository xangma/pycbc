"""PR 5452 CPU chi-square allocation comparison; exact committed source snapshots."""
import csv
import importlib
import json
import platform
import random
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from pycbc.types import Array

here = Path(__file__).resolve().parent
labels = ['base', 'malloc', 'malloc1', 'cache', 'numpy']
modules = {label: importlib.import_module('chisq_' + label) for label in labels}
with np.load(here.parent.parent / 'data' / 'point-input-05.npz', allow_pickle=False) as saved:
    capture = {name: saved[name].copy() for name in saved.files}
rng = np.random.default_rng(183)
order_rng = random.Random(183)
rows = []
paired = []
for shape in ['short', 'uniform', 'captured', 'wide']:
    n = {'short': 4096, 'uniform': 2**20, 'captured': 2**20, 'wide': 2**22}[shape]
    bins = (capture['bins'] if shape == 'captured' else
            np.linspace(0, n // 2, 17, dtype=np.uint32))
    data = (capture['corr'] if shape == 'captured' else
            (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64))
    for dtype in [np.complex64, np.complex128]:
        corr = Array(data.astype(dtype), copy=False)
        for npoints in [1, 2, 5]:
            points = (np.resize(capture['indices'], npoints) if shape == 'captured'
                      else np.linspace(n // 7, 6*n // 7, npoints, dtype=np.int64))
            values = {label: modules[label].shift_sum(corr, points, bins) for label in labels}
            for label in ['malloc', 'malloc1', 'cache']:
                np.testing.assert_allclose(values['numpy'], values[label], rtol=1e-12, atol=1e-10,
                                           err_msg=f'{shape} {dtype} {npoints} {label}')
            repeats = 30 if shape == 'short' else 2
            times = {label: [] for label in labels}
            for batch in range(9):
                order = labels[:]
                order_rng.shuffle(order)
                for label in order:
                    start = perf_counter()
                    for _ in range(repeats):
                        modules[label].shift_sum(corr, points, bins)
                    times[label].append((perf_counter() - start) / repeats)
            for label in labels:
                elapsed = times[label]
                rows.append(dict(shape=shape, n=n, npoints=npoints,
                                 dtype=np.dtype(dtype).name, variant=label,
                                 median_ms=1000*np.median(elapsed),
                                 q25_ms=1000*np.quantile(elapsed, .25),
                                 q75_ms=1000*np.quantile(elapsed, .75)))
            paired.append(dict(shape=shape, dtype=np.dtype(dtype).name,
                               npoints=npoints,
                               numpy_vs_malloc1=float(np.median(np.array(times['numpy']) / times['malloc1'])),
                               numpy_vs_malloc=float(np.median(np.array(times['numpy']) / times['malloc'])),
                               numpy_vs_cache=float(np.median(np.array(times['numpy']) / times['cache'])),
                               numpy_vs_base=float(np.median(np.array(times['numpy']) / times['base']))))
            print(shape, np.dtype(dtype).name, npoints, 'complete', flush=True)
with (here / 'timings.csv').open('w', newline='') as out:
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
with (here / 'ratios.csv').open('w', newline='') as out:
    writer = csv.DictWriter(out, fieldnames=list(paired[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(paired)
info = dict(platform=platform.platform(), python=sys.version, numpy=np.__version__,
            compiler=subprocess.check_output(['c++', '--version'], text=True).splitlines()[0],
            shapes=['short:4096', 'uniform:1048576', 'captured:1048576', 'wide:4194304'],
            variants={'base':'454ee900e6faca88a066e71ade489fca9fb221a5',
                      'malloc':'698baa686ff1f5a19a7140c5c4492958b16c3c67',
                      'malloc1':'scratch variant of 8c96aff: one contiguous checked C allocation',
                      'cache':'42218e0c7',
                      'numpy':'8c96affb842687557fbb5b26f293e03b1e6b9ba0'},
            batches=9, repeats_short=30, repeats_other=2,
            statistic='median and IQR of 9 interleaved batches; ratios are median paired ratios',
            scope='shift_sum CPU kernel; construction excluded; source outputs checked to rtol=1e-12, atol=1e-10 across double variants')
(here / 'environment.json').write_text(json.dumps(info, indent=2) + '\n')
