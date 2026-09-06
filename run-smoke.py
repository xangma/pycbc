"""Wait for the serial live run, then validate supplementary harnesses."""
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
OUT = ROOT / 'smoke'
OUT.mkdir(exist_ok=True)
H = ROOT / 'harness'
SHAS = {'main': '607bce53ead14f12af32552a5b2441d3bc667267',
        'fft': 'e6073eaf1a89cfed69af53707f52321eadf129f1',
        'cpu': '1a2ebea088d9e0a31cbb22c19ad24f96ffea2b7c'}
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
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
started = time.time()
while not (ROOT / 'live/status.json').exists():
    if time.time() - started > 1800:
        raise RuntimeError('Live run did not finish within 30 minutes')
    time.sleep(5)


def monitor():
    with (OUT / 'telemetry.jsonl').open('w') as out:
        while not stop.is_set():
            row = subprocess.check_output(['nvidia-smi',
                '--query-gpu=timestamp,utilization.gpu,memory.free',
                '--format=csv,noheader,nounits'], text=True).strip()
            out.write(json.dumps(dict(gpu=row, load=Path('/proc/loadavg').read_text().strip()))+'\n')
            out.flush()
            stop.wait(5)


jobs = []
for head, mode, threads in [('main','off',1),('fft','off',1),('fft','cold',1),
                            ('fft','warm',1),('fft','off',4)]:
    label = f'fft-{head}-{mode}-t{threads}'
    jobs.append((label, [PY, str(H/'fft/fft_worker.py'), '--source-root', str(ROOT/head),
        '--expected-revision', SHAS[head], '--label', label, '--cache-mode', mode,
        '--cache-dir', str(OUT/'cache'), '--threads', str(threads),
        '--output', str(OUT/(label+'.json'))]))
for head, route in [('main','torch_cpu_native'),('cpu','torch_cpu_native'),('main','torch_cuda_native')]:
    label = f'probe-{head}-{route}'
    jobs.append((label, [PY, str(H/'fft/live_dispatch_probe.py'), '--source-root', str(ROOT/head),
        '--expected-revision', SHAS[head], '--route', route, '--threads','1','--batch','8',
        '--output', str(OUT/(label+'.json'))]))
for route in ['standard-cpu','torch-cpu-scalar','torch-cpu-batch','torch-cuda-scalar','torch-cuda-batch']:
    label = 'waveform-'+route
    jobs.append((label, [PY, str(H/'waveform/worker.py'), '--root', str(ROOT/'main'),
        '--route', route, '--batch','8','--threads','1','--replicate','1','--precision','double',
        '--samples','3','--sample-ms','5','--max-inner','2','--out', str(OUT/(label+'.json'))]))
base = [PY, str(H/'inference/benchmark.py'), '--root', str(ROOT/'main'),
        '--output', str(OUT/'inference')]
jobs.append(('inference-prepare', base+['--mode','prepare','--groups','3','--evaluations','2']))
for model in ['gaussian','relative']:
    for route in ['standard_cpu','torch_cpu','cuda']:
        label = f'inference-{model}-{route}'
        jobs.append((label, base+['--mode','worker','--model',model,'--route',route,
                                '--threads','1','--replicate','0']))
thread = threading.Thread(target=monitor, daemon=True)
thread.start()
results = []
try:
    for label, command in jobs:
        result = dict(label=label, command=['taskset','-c','8-11',*command],
                      started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        start = time.time()
        print(json.dumps(result), flush=True)
        with (OUT/(label+'.log')).open('w') as log:
            active = subprocess.Popen(result['command'], cwd=ROOT/'main', env=env,
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            result['pid'] = active.pid
            (OUT/'active.json').write_text(json.dumps(result,indent=2)+'\n')
            try:
                result['returncode'] = active.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGTERM)
                result['returncode'] = active.wait(timeout=30)
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
