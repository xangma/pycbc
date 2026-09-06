"""Render qualified independent-process FFT measurements; never pool samples."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics as stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SHAS = {'main': '607bce53ead14f12af32552a5b2441d3bc667267',
        'fft': 'e6073eaf1a89cfed69af53707f52321eadf129f1'}
CONFIGS = [('main','off',1), ('fft','off',1), ('fft','cold',1),
           ('fft','warm',1), ('main','off',4), ('fft','off',4)]
LABELS = ['Main · off · 1T', 'Optional · off · 1T', 'Optional · cold · 1T',
          'Optional · warm · 1T', 'Main · off · 4T', 'Optional · off · 4T']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    rows, hashes = [], {}
    for (head, mode, threads), label in zip(CONFIGS, LABELS):
        records = []
        for rep in range(1, 4):
            path = a.input_dir/f'fft-{head}-{mode}-t{threads}-r{rep}.json'
            d = json.loads(path.read_text())
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            assert d['source']['revision'] == SHAS[head]
            assert d['source']['tracked_clean']
            assert d['environment']['torch_threads'] == threads
            assert d['cache']['mode'] == mode
            assert d['workload']['size'] == 131072 and d['workload']['dtype'] == 'complex64'
            records.append(d)
        row = dict(head=head, cache_mode=mode, threads=threads, label=label,
                   statuses=[d['status'] for d in records],
                   eligible=all(d['status']=='passed' and d['parity']['passed']
                     and d['cache']['confirmed'] and d['plan']['expected_route_confirmed']
                     for d in records))
        if row['eligible']:
            for key, values in [('plan_ms',[d['plan']['construction_seconds']*1e3 for d in records]),
                                ('execute_ms',[stats.median(d['timing']['raw_samples'])*1e3 for d in records])]:
                row[key] = dict(replicate_values=values, median=stats.median(values),
                                minimum=min(values), maximum=max(values))
            row['max_relative_l2'] = max(d['parity']['relative_l2'] for d in records)
            row['native_successes'] = [d['plan']['untimed_dispatch']['native_successes'] for d in records]
            row['cache_imports'] = [d['cache']['successful_imports'] for d in records]
            row['cache_exports'] = [d['cache']['successful_exports'] for d in records]
        rows.append(row)
    result = dict(source=SHAS, rows=rows, input_sha256=hashes,
        method='Three independent processes; median and min/max of process medians; range is not a confidence interval',
        scope='Public Torch CPU IFFT, one complex64 vector of length 131072; plan construction measured separately; four-thread path uses Torch fallback')
    (a.output_dir/'fft-summary.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# FFTW wisdom and IFFT — 6 September 2026', '', result['scope']+'.', '',
             result['method']+'.', '',
             '| Route | Plan construction ms (range) | Warm IFFT ms (range) | Status |',
             '| --- | ---: | ---: | --- |']
    for row in rows:
        values = []
        for key in ['plan_ms', 'execute_ms']:
            x = row.get(key)
            values.append(f"{x['median']:.3f} ({x['minimum']:.3f}–{x['maximum']:.3f})" if x else 'excluded')
        lines.append(f"| {row['label']} | {values[0]} | {values[1]} | {'passed' if row['eligible'] else row['statuses']} |")
    lines += ['', 'Cache-off and warm-cache workers use FFTW ESTIMATE; cold-cache workers use MEASURE and export wisdom. Warm workers import the corresponding cold worker’s isolated cache. The one-thread direct FFTW route and the four-thread Torch fallback are observed separately. All 18 workers must pass the numerical, dispatch and cache checks to support a complete comparison.', '',
              'Parity covers the full output against a NumPy complex128 unnormalized IFFT (relative L2 and maximum error relative to reference peak ≤2e-6), input preservation and repeat-output equality. Timing excludes verification and instrumentation. Five samples of ten IFFTs follow three warmups in each process. Plan times are one construction per fresh process, after imports; they are not complete process startup times.', '',
              'This single-vector case measures automatic cache behavior. It does not measure every batch-layout correction in PR #16. CPU affinity 8–11 on a shared Threadripper PRO 3995WX; execution is sequential and host telemetry is retained.', '']
    (a.output_dir/'fft-summary.md').write_text('\n'.join(lines))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1,2,figsize=(13,5.6), layout='constrained')
    colors = ['#607080','#236c9e','#c47524','#26926c','#85929e','#5991b6']
    for ax, key, title in zip(axes,['plan_ms','execute_ms'],['Plan construction (ms) — lower is faster','Warm IFFT (ms) — lower is faster']):
        for i,row in enumerate(rows):
            if not row['eligible']:
                ax.text(.03,i,'Excluded: failed qualification', transform=ax.get_yaxis_transform())
                continue
            d=row[key]; m=d['median']
            ax.errorbar(m,i,xerr=[[m-d['minimum']],[d['maximum']-m]],fmt='o',color=colors[i],capsize=4,markersize=7)
            ax.annotate(f' {m:.3f}',(m,i),xytext=(7,7),textcoords='offset points',fontsize=9)
        ax.set_yticks(np.arange(6), LABELS if ax is axes[0] else [])
        ax.set_ylim(5.6,-.65); ax.set_xscale('log'); ax.grid(axis='x',alpha=.18)
        ax.set_title(title,fontsize=12,pad=12)
        lo,hi=ax.get_xlim(); ax.set_xlim(lo*.8,hi*1.8)
    fig.suptitle('FFTW wisdom: creation cost and repeated execution\n6 Sep 2026 · main 607bce5 / optional e6073ea · N=131072, complex64',fontsize=14)
    fig.supxlabel('Median of 3 processes; whiskers = process range, not confidence intervals. 1T: direct FFTW; 4T: Torch fallback.',fontsize=9)
    for suffix in ['png','svg']:
        fig.savefig(a.output_dir/f'fft.{suffix}',dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
