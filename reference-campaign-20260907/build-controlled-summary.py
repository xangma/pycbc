"""Summarize accepted receipts without mixing timings with profiling runs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import pstats
import re
import statistics

import h5py


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def summary(values):
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values))


def read_run(root, case):
    folder = root / 'runs' / case
    r = json.loads((folder / 'receipt.json').read_text())
    assert r['state'] == 'complete' and r['returncode'] == 0, case
    assert r['input_sha256'] == r['input_sha256_after'], case
    assert r['source_info']['status'] == r['source_status_after'] == '', case
    assert r['trigger_sha256'] == digest(folder/'triggers.hdf'), case
    with h5py.File(folder/'triggers.hdf') as f:
        g = f['H1/search']
        starts, ends = g['start_time'][:], g['end_time'][:]
        assert all(ends > starts) and all(starts[1:] >= ends[:-1]), case
        valid = float(sum(ends-starts))
        assert valid == 1904, case
        internal = float(g['run_time'][0])
        setup = internal * float(g['setup_time_fraction'][0])
        templates = float(g['templates_per_core'][0]) * internal / valid
        assert math.isclose(templates, 384, abs_tol=1e-6), (case, templates)
        segments = float(g['filter_rate_per_core'][0]) * internal
        assert math.isclose(segments, round(segments), abs_tol=1e-6), case
        triggers = len(f['H1/snr'])
    wall = r['elapsed_wall_seconds']
    assert 0 <= setup <= internal <= wall, case
    return dict(case=case, mode=r['mode'], scheme=r['scheme'], source_commit=r['source_info']['commit'],
                segment_length=r['segment_length'], start_pad=r['start_pad'], end_pad=r['end_pad'],
                valid_seconds=valid, templates=384, segments_per_template=round(segments), triggers=triggers,
                wall_seconds=wall, internal_seconds=internal, internal_setup_seconds=setup,
                internal_postsetup_seconds=internal-setup, outside_internal_timer_seconds=wall-internal,
                templates_per_core=384*valid/wall,
                peak_rss_kib=r['peak_child_rss_kib'],
                input_sha256=r['input_sha256'], receipt_sha256=digest(folder/'receipt.json'))


def profile_group(key):
    filename, _, name = key
    if filename.endswith('/fft/mkl.py') and name == 'execute':
        return 'FFT execution'
    if filename.endswith('/fft/mkl.py') and name == 'create_descriptor':
        return 'FFT planning'
    if 'point_chisq_code' in name:
        return 'Chi-square kernel'
    if filename.endswith('/events/threshold_cpu.py') and name == 'threshold_and_cluster':
        return 'Threshold/cluster wrapper'
    if filename.endswith('/waveform/decompress_cpu.py') and name == 'inline_linear_interp':
        return 'Decompression wrapper'
    if 'lalframe' in name:
        return 'Frame API'
    return 'Other measured functions'


def summarize_profiles(root):
    records = []
    for path in sorted((root/'runs').glob('*/*.pstats')):
        if not (path.parent/'receipt.json').exists():
            continue
        receipt = json.loads((path.parent/'receipt.json').read_text())
        if receipt['state'] != 'complete':
            continue
        stats = pstats.Stats(str(path))
        groups = {}
        for key, value in stats.stats.items():
            group = profile_group(key)
            groups[group] = groups.get(group, 0) + value[2]
        records.append(dict(case=path.parent.name, scope='template loop' if path.name=='filtering.pstats' else 'full executable',
                            total_exclusive_seconds=stats.total_tt,
                            groups={k:dict(seconds=v, percent=100*v/stats.total_tt) for k,v in groups.items()},
                            top_functions=[dict(file=k[0],line=k[1],function=k[2],exclusive_seconds=v[2])
                                           for k,v in sorted(stats.stats.items(),key=lambda item:-item[1][2])[:20]],
                            source_path=str(path.relative_to(root)),sha256=digest(path)))
    return records


def summarize_native_profiles(root):
    records = []
    for path in sorted((root/'runs').glob('*/perf-report-filtering.txt')):
        text = path.read_text()
        assert "event 'cycles:u'" in text and 'time slices:' in text, path
        groups = {}
        reported = 0
        for line in text.splitlines():
            match = re.match(r'\s*(\d+\.\d+)%\s+\S+\s+(\S+)\s+\[.\]\s+(.*)', line)
            if not match:
                continue
            percent, dso, symbol = match.groups()
            percent = float(percent)
            reported += percent
            if 'mkl_' in symbol and ('dft' in symbol or 'fft' in symbol.lower()):
                group = 'MKL FFT'
            elif 'chisq_cpu' in dso:
                group = 'Chi-square kernel'
            elif 'simd_threshold' in dso:
                group = 'Threshold kernel'
            elif 'matchedfilter_cpu' in dso:
                group = 'Correlation kernel'
            elif 'decompress_cpu' in dso:
                group = 'Decompression kernel'
            elif '_multiarray_umath' in dso:
                group = 'NumPy kernels'
            else:
                continue
            groups[group] = groups.get(group, 0) + percent
        assert 97 <= reported <= 101, (path, reported)
        groups['Other / rounding remainder'] = 100 - sum(groups.values())
        assert groups['Other / rounding remainder'] >= 0
        records.append(dict(case=path.parent.name, scope='template loop',
                            groups=groups, source_path=str(path.relative_to(root)),
                            sha256=digest(path), reported_percent_sum=reported,
                            denominator='rounded sampled cycles:u event-period percentages'))
    return records


def host_load(root):
    path = root/'comparison-host-samples.jsonl'
    if not path.exists():
        return None
    samples = [json.loads(line) for line in path.read_text().splitlines()]
    intervals = []
    def counters(sample):
        return {line.split()[0]:list(map(int,line.split()[1:9])) for line in sample['proc_stat'].splitlines() if line.startswith('cpu')}
    for before,after in zip(samples,samples[1:]):
        a,b=counters(before),counters(after)
        usage={}
        for name in a.keys() & b.keys():
            delta=[v-u for u,v in zip(a[name],b[name])]
            total=sum(delta)
            if total>0 and all(d>=0 for d in delta):
                usage[name]=100*(total-delta[3]-delta[4])/total
        intervals.append(dict(start_utc=before['utc'],end_utc=after['utc'],busy_percent=usage))
    return dict(label='shared host; no reservation',interval_count=len(intervals),
                aggregate_busy_percent=summary([x['busy_percent']['cpu'] for x in intervals]) if intervals else None,
                pinned_cpu='cpu8',smt_sibling='cpu72',
                sibling_busy_percent=summary([x['busy_percent']['cpu72'] for x in intervals]) if intervals else None,
                note='Utilization includes benchmark processes; ps percent is lifetime average. Full per-CPU time series retained separately.',
                sha256=digest(path))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--partial',action='store_true')
    args=p.parse_args();root=args.root
    sweep=json.loads((root/'tuning-decision.json').read_text())
    tuning=[]
    for row in sweep['rows']:
        runs=[read_run(root,item['case']) for item in row['samples']]
        tuning.append(dict(segment_length=row['length'],start_pad=row['start_pad'],end_pad=16,
                           wall_seconds=summary([r['wall_seconds'] for r in runs]),runs=runs))
    campaign=json.loads((root/'comparison-status.json').read_text())
    if not args.partial:
        assert campaign['state']=='complete',campaign
        required = {'corrected-trigger-comparison', *(
            f'{backend}-{kind}-parity' for backend in
            ('original-cpu', 'corrected-cpu', 'torch-cpu', 'torch-cuda')
            for kind in ('repeat', 'profile'))}
        assert required <= campaign['comparison_returncodes'].keys()
        assert all(campaign['comparison_returncodes'][k] == 0 for k in required)
    groups=[]
    for backend in ('original-cpu','corrected-cpu','torch-cpu','torch-cuda'):
        cases=[f'matched-{backend}-r{r}' for r in range(1,4)]
        if all(case in campaign['completed'] for case in cases):
            runs=[read_run(root,case) for case in cases]
            assert all(r['mode']=='timing' for r in runs)
            groups.append(dict(backend=backend,runs=runs,
                               wall_seconds=summary([r['wall_seconds'] for r in runs]),
                               capacity=summary([r['templates_per_core'] for r in runs])))
    qualification={}
    for backend in ('original','corrected-cpu','torch-cpu','torch-cuda'):
        path=root/'runs'/f'qual-selected-{backend}'/'qualification.json'
        q=json.loads(path.read_text())
        assert all(q['checks'].values()), backend
        qualification[backend]=dict(checks=q['checks'],sha256=digest(path))
    parity={}
    for path in sorted(root.glob('*-parity.json'))+sorted(root.glob('*-trigger-comparison.json')):
        parity[path.stem]=json.loads(path.read_text())
    if not args.partial:
        assert len(groups) == 4
        for name in required:
            assert parity[name]['status'] == 'pass', name
    intervals={}
    for backend in ('original-cpu','corrected-cpu','torch-cpu','torch-cuda'):
        path=root/'runs'/f'interval-{backend}'/'filtering-window.json'
        if path.exists():
            receipt=json.loads(path.read_text())
            assert receipt['state']=='complete' and receipt['completed_templates']==384
            receipt.pop('indices');intervals[backend]=receipt
    if not args.partial:
        assert len(intervals) == 4
    result=dict(schema_version=1,status='partial' if args.partial else 'complete',
                host='len',scope='384-template finite workload on shared host; not sustained capacity',
                templates=384,valid_seconds=1904,template_seconds=731136,
                selected_geometry=[512,112,16],selection_rule=sweep['selection_rule'],
                tuning=tuning,matched=groups,qualification=qualification,intervals=intervals,
                profiles=summarize_profiles(root), native_profiles=summarize_native_profiles(root),
                parity=parity,host_load=host_load(root),
                limitations=['No exclusive host reservation or full-physical-core saturation measurement.',
                             '384-template workload has not demonstrated capacity convergence.',
                             'End padding fixed at16s; no global geometry optimality claim.',
                             'Sweep and selected original profiles predate the periodic host-load record.',
                             'Original/final output differences retained; parity status applies only to its named reference.'])
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],matched_groups=len(groups),profiles=len(result['profiles']),output=str(args.output))))


if __name__=='__main__':
    main()
