"""Run the published live drivers serially, retaining logs and resource samples."""
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

ROOT = Path('/home/xangma/pycbc-torch-benchmark-20260906')
PYTHON = '/home/xangma/pycbc-torch-split-20260905/venv/bin/python'
OUT = ROOT / 'live'
OUT.mkdir(exist_ok=True)
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
           NUMEXPR_NUM_THREADS='1', OMP_DYNAMIC='FALSE')
env.pop('PYTHONPATH', None)
active = None
stop = threading.Event()


def terminate(signum, frame):
    stop.set()
    if active is not None and active.poll() is None:
        os.killpg(active.pid, signal.SIGTERM)
    raise SystemExit(128 + signum)


signal.signal(signal.SIGTERM, terminate)
signal.signal(signal.SIGINT, terminate)


def gpu():
    return subprocess.check_output(['nvidia-smi',
        '--query-gpu=timestamp,name,utilization.gpu,memory.used,memory.free',
        '--format=csv,noheader,nounits'], text=True).strip()


def monitor():
    with (OUT / 'telemetry.jsonl').open('w') as out:
        while not stop.is_set():
            try:
                record = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                              gpu=gpu(), load=Path('/proc/loadavg').read_text().strip())
                out.write(json.dumps(record)+'\n')
                out.flush()
            except Exception as exc:
                out.write(json.dumps({'error': str(exc)})+'\n')
                out.flush()
            stop.wait(5)


assert (ROOT / 'prepared').exists()
thread = threading.Thread(target=monitor, daemon=True)
thread.start()
with (OUT / 'environment.txt').open('w') as out:
    for command in [['date','-u'], ['hostname'], ['uname','-a'], ['lscpu'], ['nvidia-smi']]:
        subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, check=True)
jobs = [('main',1), ('cpu',1), ('main',4), ('cpu',4)]
results = []
try:
    for head, threads in jobs:
        label = f'{head}-t{threads}'
        repo = ROOT / head
        routes = ['branch_standard', 'torch_cpu', 'torch_cpu_native']
        if head == 'main' and threads == 1:
            routes += ['torch_cuda', 'torch_cuda_native']
            row = gpu().split(',')
            assert float(row[-1]) >= 10240 and float(row[-3]) <= 10, row
        command = [PYTHON, str(repo / 'tools/bench_production_live_batch.py'),
            'orchestrate', '--root', str(repo), '--python', PYTHON,
            '--output', str(OUT / f'{label}.json'), '--routes', *routes,
            '--batches', '1', '8', '32', '--size', '131072', '--num-blocks', '3',
            '--threads', str(threads), '--replicates', '3', '--samples', '5',
            '--warmups', '2', '--cuda-device', '0', '--affinity', '8-11',
            '--seed', '7101', '--call-surface', 'public']
        started = time.time()
        result = dict(label=label, cwd=str(repo), command=command,
                      started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        print(json.dumps(result), flush=True)
        (OUT / 'active.json').write_text(json.dumps(result, indent=2)+'\n')
        with (OUT / f'{label}.log').open('w') as out:
            active = subprocess.Popen(command, cwd=repo, env=env, stdout=out,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            result['child_pid'] = active.pid
            (OUT / 'active.json').write_text(json.dumps(result, indent=2)+'\n')
            try:
                result['returncode'] = active.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGTERM)
                result['returncode'] = active.wait(timeout=30)
                result['timeout'] = True
        result['seconds'] = time.time() - started
        results.append(result)
        (OUT / 'runs.json').write_text(json.dumps(results, indent=2)+'\n')
        print(json.dumps(result), flush=True)
finally:
    stop.set()
    thread.join(timeout=10)
(OUT / 'status.json').write_text(json.dumps(dict(finished=True,
    passed=all(r['returncode']==0 for r in results), jobs=len(results)))+'\n')
