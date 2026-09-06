"""Sequential paired workers; profiling is excluded from throughput timing."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

R = Path('/home/xangma/pycbc-torch-performance-fix-20260906')
P = '/home/xangma/pycbc-torch-split-20260905/venv/bin/python'
sys.path.insert(0, str(R/'baseline'))
from tools import bench_production_live_batch as live


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


sources = {}
for name in ['baseline', 'candidate']:
    root = R/name
    sources[name] = dict(root=str(root), sha=git(root, 'rev-parse', 'HEAD'),
                         tree=git(root, 'rev-parse', 'HEAD^{tree}'),
                         status=git(root, 'status', '--porcelain'))
    assert not sources[name]['status']
status = dict(pid=os.getpid(), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              sources=sources, completed=[], state='running')
status_path = R/'comparison-status.json'
save(status_path, status)


def record_live(name, route, batch, threads, rep):
    label=f'{route}-b{batch}-t{threads}-r{rep}-{name}'
    existing = R/'comparison/live'/f'{label}.json'
    if existing.exists():
        result=json.loads(existing.read_text())
        assert result['comparison_source']==sources[name]
        assert result['route']==route and result['batch']==batch and result['threads']==threads
        assert result['comparison_replicate']==rep
        status['completed'].append(label)
        save(status_path,status)
        return result
    status['current'] = label
    save(status_path, status)
    started=time.monotonic()
    result=live._run_child(P, R/name/'tools/bench_production_live_batch.py',
        route, R/name, batch, 131072, 3, threads, 3, 1, 5.5, 0, 7101,
        '8-11', 'public')
    result.update(comparison_source=sources[name], comparison_label=label,
                  comparison_replicate=rep, worker_wall_seconds=time.monotonic()-started)
    save(R/'comparison/live'/f'{label}.json', result)
    print(json.dumps(dict(label=label, seconds=result['worker_wall_seconds'],
        throughput=result['throughput_wps_summary'])), flush=True)
    status['completed'].append(label)
    save(status_path, status)
    return result


try:
    for batch in [1024, 1, 32, 128, 512, 8]:
        for threads in [1, 4]:
            routes=['torch_cuda_native', 'torch_cuda', 'torch_cpu_native', 'torch_cpu'] if threads==1 else ['torch_cpu_native', 'torch_cpu']
            for rep in range(1,4):
                control=record_live('baseline','branch_standard',batch,threads,rep)
                for route in routes:
                    pair={}
                    for name in (['baseline','candidate'] if rep%2 else ['candidate','baseline']):
                        pair[name]=record_live(name,route,batch,threads,rep)
                    parity=live._verify_parity({batch: {'branch_standard':control,
                        route:pair['baseline'], route+'_fixed':pair['candidate']}})
                    save(R/'comparison/parity'/f'{route}-b{batch}-t{threads}-r{rep}.json',parity)
                    if not parity['all_passed_globally']:
                        raise RuntimeError(f'live parity failed: {route} {batch} {threads} {rep}')
    for batch in [1024, 1, 32, 128, 512, 8]:
        for route,threads in [('torch-cuda-batch',1),('torch-cpu-batch',1),('torch-cpu-batch',4)]:
            for rep in range(1,4):
                for name in (['baseline','candidate'] if rep%2 else ['candidate','baseline']):
                    label=f'{route}-b{batch}-t{threads}-r{rep}-{name}'
                    status['current']=label
                    save(status_path,status)
                    command=['taskset','-c','8-11',P,str(R/'waveform-worker.py'),
                        '--root',str(R/name),'--out',str(R/'comparison/waveform'/f'{label}.json'),
                        '--route',route,'--batch',str(batch),'--threads',str(threads),
                        '--replicate',str(rep),'--samples','5','--expected-sha',sources[name]['sha']]
                    subprocess.run(command,cwd=R/name,check=True,timeout=240)
                    status['completed'].append(label)
                    save(status_path,status)
    status['state']='complete'
except BaseException as exc:
    status['state']='failed'
    status['error']=repr(exc)
    raise
finally:
    status['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    save(status_path,status)
