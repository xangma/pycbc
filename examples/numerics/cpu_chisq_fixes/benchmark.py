"""Reproduce CPU-kernel timings separately from notebook execution."""
import argparse
import csv
import json
import platform
import random
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from pycbc.types import Array

from cpu_chisq_review import kernels, load_capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', nargs='+')
    parser.add_argument('--output', default='benchmark.csv')
    args = parser.parse_args()
    implementations = kernels(tuple(args.variants) if args.variants else None)
    capture = load_capture()
    rng = np.random.default_rng(183)
    order_rng = random.Random(183)
    rows = []
    for shape in ['short', 'uniform', 'captured', 'wide']:
        n = {'short': 4096, 'uniform': 2**20,
             'captured': 2**20, 'wide': 2**22}[shape]
        bins = (capture['bins'] if shape == 'captured' else
                np.linspace(0, n // 2, 17, dtype=np.uint32))
        data = (capture['corr'] if shape == 'captured' else
                (rng.normal(size=n) +
                 1j*rng.normal(size=n)).astype(np.complex64))
        for dtype in [np.complex64, np.complex128]:
            corr = Array(data.astype(dtype), copy=False)
            for npoints in [1, 2, 5]:
                points = (np.resize(capture['indices'], npoints)
                          if shape == 'captured' else
                          np.linspace(n // 7, 6*n // 7, npoints,
                                      dtype=np.int64))
                repeats = 30 if shape == 'short' else 2
                times = {label: [] for label in implementations}
                for module in implementations.values():
                    module.shift_sum(corr, points, bins)
                for batch in range(9):
                    labels = list(implementations)
                    order_rng.shuffle(labels)
                    for label in labels:
                        start = perf_counter()
                        for _ in range(repeats):
                            implementations[label].shift_sum(
                                corr, points, bins)
                        times[label].append((perf_counter() - start) / repeats)
                baseline = np.median(times['upstream'])
                for label, elapsed in times.items():
                    ratio = np.median(elapsed)/baseline
                    rows.append(dict(shape=shape, n=n, npoints=npoints,
                                     dtype=np.dtype(dtype).name, variant=label,
                                     median_ms=1000*np.median(elapsed),
                                     q25_ms=1000*np.quantile(elapsed, .25),
                                     q75_ms=1000*np.quantile(elapsed, .75),
                                     ratio_to_upstream=ratio))
                print(shape, np.dtype(dtype).name, npoints, 'complete',
                      flush=True)
    folder = Path(__file__).resolve().parent
    output = folder / args.output
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]),
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    compiler = subprocess.check_output(['c++', '--version'], text=True)
    info = dict(platform=platform.platform(), machine=platform.machine(),
                python=sys.version, numpy=np.__version__,
                compiler=compiler.splitlines()[0],
                batches=9, repeats_short=30, repeats_other=2,
                statistic='median and IQR of interleaved batches',
                scope='kernel only; input construction excluded; one host')
    output.with_name(output.stem + '_environment.json').write_text(
        json.dumps(info, indent=2)+'\n')


if __name__ == '__main__':
    main()
