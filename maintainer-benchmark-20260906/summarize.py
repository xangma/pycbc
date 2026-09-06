#!/usr/bin/env python3
"""Verify raw run receipts and derive the maintainer comparison."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import pstats
import statistics
import sys
import h5py

ROOT = Path(sys.argv[1]).resolve()

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    status = json.loads((ROOT / 'status.json').read_text())
    assert status['state'] == 'complete'
    source = json.loads((ROOT / 'source.json').read_text())
    assert source['original_status'] == source['final_status'] == ''
    assert source['native_sources_unchanged'] and source['native_build_definitions_unchanged']
    assert len(source['reused_native_sha256']) == 11
    assert source['original_import_probe']['version'] == source['original']
    plan = json.loads((ROOT / 'corrected-plan.json').read_text())
    assert len(plan) == 15 and len(status['completed']) == 15
    rows, profiles, inputs = [], [], {}
    identities = set()
    for item in plan:
        folder = ROOT / 'runs' / item['case']
        receipt_path = folder / 'receipt.json'
        r = json.loads(receipt_path.read_text())
        assert r['state'] == 'complete' and r['returncode'] == 0
        assert r['source_info']['status'] == r['source_status_after'] == ''
        assert r['input_sha256'] == r['input_sha256_after']
        assert r['scheme'] == item['scheme']
        assert r['source_info']['commit'] == source['original' if item['case'].startswith('original-') else 'final']
        assert r['segment_length'] == 512 and r['start_pad'] == 112 and r['end_pad'] == 16
        assert r['environment']['OMP_NUM_THREADS'] == r['environment']['MKL_NUM_THREADS'] == '1'
        assert r['command'][r['command'].index('taskset')+1:r['command'].index('taskset')+3] == ['-c','8']
        hdf = folder / 'triggers.hdf'
        assert digest(hdf) == r['trigger_sha256']
        cli = r['executable_cli']
        for opt in ('--bank-file', '--frame-files'):
            value = cli[cli.index(opt)+1]; identities.add((opt,r['input_sha256'][value]))
        with h5py.File(hdf) as f:
            start, end = f['H1/search/start_time'][:], f['H1/search/end_time'][:]
            valid = float(sum(end-start)); assert valid == 1904
            row = dict(case=item['case'], backend=item['case'].rsplit('-',1)[0], mode=item['mode'], scheme=r['scheme'], source=r['source_info']['commit'], wall_seconds=r['elapsed_wall_seconds'], templates=96, valid_detector_seconds=valid, capacity_templates_per_host_core=96*valid/r['elapsed_wall_seconds'], triggers=len(f['H1/snr']), peak_rss_mib=r['peak_child_rss_kib']/1024)
            row['internal_seconds'] = float(f['H1/search/run_time'][0])
            row['setup_fraction'] = float(f['H1/search/setup_time_fraction'][0])
        assert row['wall_seconds'] > 0
        inputs[str(receipt_path.relative_to(ROOT))] = digest(receipt_path)
        inputs[str(hdf.relative_to(ROOT))] = digest(hdf)
        if item['mode'] == 'timing':
            rows.append(row)
        else:
            if item['mode'] == 'cprofile':
                p = pstats.Stats(str(folder/'profile.pstats'))
                row['exclusive_total_seconds'] = p.total_tt
                row['top_self_functions'] = [dict(file=k[0],line=k[1],function=k[2],self_seconds=v[2],cumulative_seconds=v[3]) for k,v in sorted(p.stats.items(),key=lambda kv:kv[1][2],reverse=True)[:20]]
                inputs[str((folder/'profile.pstats').relative_to(ROOT))] = digest(folder/'profile.pstats')
            profiles.append(row)
    assert len(identities) == 2
    summaries=[]
    for backend in ('original-cpu','torch-cpu','torch-cuda'):
        samples = [r for r in rows if r['backend']==backend]
        assert len(samples)==3
        summaries.append(dict(backend=backend, source=samples[0]['source'], samples=samples, **{key:dict(median=statistics.median(r[key] for r in samples),minimum=min(r[key] for r in samples),maximum=max(r[key] for r in samples)) for key in ('wall_seconds','capacity_templates_per_host_core','peak_rss_mib','internal_seconds')}))
    for s in summaries:
        s['capacity_relative_to_original'] = summaries[0]['wall_seconds']['median']/s['wall_seconds']['median']
    comparison=json.loads((ROOT/'original-trigger-comparison.json').read_text())
    for name in ('source.json','config.json','corrected-plan.json','status.json','original-trigger-comparison.json','campaign.py','resume.py','run-case.py','compare-triggers.py'):
        inputs[name] = digest(ROOT/name)
    final_check=json.loads((ROOT/'final-trigger-validation.json').read_text())
    assert final_check['status']=='pass' and len(final_check['comparisons'])==6
    for name in ('final-trigger-validation.json','environment.json','profile-transport.json','package-profiles.py','summarize.py','corrected-reference/receipt.json','corrected-reference/triggers.hdf'):
        inputs[name]=digest(ROOT/name)
    for record in json.loads((ROOT/'profile-transport.json').read_text()):
        path=ROOT/record['gzip_path']
        assert digest(path)==record['gzip_sha256']
        with gzip.open(path,'rb') as stream: assert hashlib.file_digest(stream,'sha256').hexdigest()==record['raw_sha256']
        assert digest(ROOT/record['report_path'])==record['report_sha256']
        inputs[record['gzip_path']]=record['gzip_sha256']
        inputs[record['report_path']]=record['report_sha256']
    result=dict(schema_version=1, final_corrected_reference_comparisons='pass', scope='Complete pycbc_inspiral, original upstream CPU versus final Torch, one host core; CUDA adds one GPU', original_source=source['original'], final_source=source['final'], summaries=summaries, profiles=profiles, trigger_comparison_returncode=status['comparison_returncode'], trigger_comparison_status=comparison.get('status'), source_and_workload_checks='pass', input_sha256=inputs)
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (ROOT/'timings.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n"); w.writeheader(); w.writerows(rows)
    print(json.dumps({k:result[k] for k in ('source_and_workload_checks','trigger_comparison_returncode','trigger_comparison_status')}))
    for s in summaries: print(s['backend'],s['wall_seconds'],s['capacity_relative_to_original'])

if __name__=='__main__': main()
