"""Run independent supplementary measurements serially after passing smoke."""
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

ROOT = Path('/home/xangma/pycbc-torch-benchmark-20260906')
PY = '/home/xangma/pycbc-torch-split-20260905/venv/bin/python'
OUT = ROOT / 'supplement'
H = ROOT / 'harness'
SHAS = {'main': '607bce53ead14f12af32552a5b2441d3bc667267',
        'fft': 'e6073eaf1a89cfed69af53707f52321eadf129f1',
        'cpu': '1a2ebea088d9e0a31cbb22c19ad24f96ffea2b7c'}
assert json.loads((ROOT/'smoke/status.json').read_text())['passed']
assert not (OUT/'runs.json').exists(), 'Never overwrite a previous run'
OUT.mkdir(exist_ok=True)
for part in ['fft', 'probes', 'waveform', 'inference']:
    (OUT/part).mkdir(exist_ok=True)
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
           OMP_DYNAMIC='FALSE')
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


def monitor():
    with (OUT/'telemetry.jsonl').open('w') as out:
        while not stop.is_set():
            row = subprocess.check_output(['nvidia-smi',
                '--query-gpu=timestamp,utilization.gpu,memory.free',
                '--format=csv,noheader,nounits'], text=True).strip()
            out.write(json.dumps(dict(gpu=row, load=Path('/proc/loadavg').read_text().strip()))+'\n')
            out.flush()
            stop.wait(5)


jobs = []
for rep in range(1, 4):
    for head, mode, threads in [('main','off',1),('fft','off',1),('fft','cold',1),
                                ('fft','warm',1),('main','off',4),('fft','off',4)]:
        label = f'fft-{head}-{mode}-t{threads}-r{rep}'
        jobs.append((label, [PY, str(H/'fft/fft_worker.py'), '--source-root', str(ROOT/head),
            '--expected-revision', SHAS[head], '--label', label, '--cache-mode', mode,
            '--cache-dir', str(OUT/'fft'/f'cache-r{rep}'), '--threads', str(threads),
            '--output', str(OUT/'fft'/(label+'.json'))], 180))
for head in ['main', 'cpu']:
    for threads in [1, 4]:
        for route in ['torch_cpu', 'torch_cpu_native']:
            for batch in [8, 32]:
                label = f'probe-{head}-{route}-t{threads}-b{batch}'
                jobs.append((label, [PY, str(H/'fft/live_dispatch_probe.py'),
                    '--source-root', str(ROOT/head), '--expected-revision', SHAS[head],
                    '--route', route, '--threads', str(threads), '--batch', str(batch),
                    '--output', str(OUT/'probes'/(label+'.json'))], 180))
for route in ['torch_cuda', 'torch_cuda_native']:
    for batch in [8, 32]:
        label = f'probe-main-{route}-t1-b{batch}'
        jobs.append((label, [PY, str(H/'fft/live_dispatch_probe.py'),
            '--source-root', str(ROOT/'main'), '--expected-revision', SHAS['main'],
            '--route', route, '--threads', '1', '--batch', str(batch),
            '--output', str(OUT/'probes'/(label+'.json'))], 180))
jobs.append(('waveform', [PY, str(H/'waveform/run.py'), '--root', str(ROOT/'main'),
    '--python', PY, '--out', str(OUT/'waveform')], 2400))
jobs.append(('inference', [PY, str(H/'inference/benchmark.py'), '--mode', 'orchestrate',
    '--root', str(ROOT/'main'), '--python', PY, '--output', str(OUT/'inference')], 1800))
gpu = subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.free',
    '--format=csv,noheader,nounits'], text=True).strip().splitlines()[0].split(',')
assert int(gpu[0]) <= 10 and int(gpu[1]) >= 10240, gpu
thread = threading.Thread(target=monitor, daemon=True)
thread.start()
results = []
try:
    for label, command, timeout in jobs:
        result = dict(label=label, command=['taskset','-c','8-11',*command],
                      cwd=str(ROOT/'main'),
                      started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        start = time.time()
        print(json.dumps(result), flush=True)
        with (OUT/(label+'.log')).open('w') as log:
            active = subprocess.Popen(result['command'], cwd=ROOT/'main', env=env,
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            result['pid'] = active.pid
            (OUT/'active.json').write_text(json.dumps(result,indent=2)+'\n')
            try:
                result['returncode'] = active.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGTERM)
                try:
                    active.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(active.pid, signal.SIGKILL)
                    active.wait()
                result['returncode'] = active.returncode
                result['timeout'] = True
        result['seconds'] = time.time()-start
        results.append(result)
        (OUT/'runs.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(result), flush=True)
finally:
    stop.set()
    thread.join(timeout=10)
(OUT/'status.json').write_text(json.dumps(dict(finished=True,
    passed=all(r['returncode']==0 for r in results), jobs=len(results)))+'\n')
