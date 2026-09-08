"""Check CUDA regressions and Qlty in a separate pinned source worktree."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

R = Path(__file__).resolve().parent
O = R/'linux-checks'
O.mkdir()
W = R/'quality-source'
PY = sys.executable
HEAD = json.loads((R/'config.json').read_text())['source_commits']['proposed']
assert json.loads((R/'summary.json').read_text())['scientific_gates_pass']
assert json.loads((R/'status.json').read_text())['state']=='complete'
subprocess.run(['git','-C',str(R/'repo'),'worktree','add','--detach',str(W),HEAD],check=True)
for p in (R/'proposed/pycbc').rglob('*.so'):
    target = W/p.relative_to(R/'proposed')
    shutil.copy2(p,target)
    assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(target.read_bytes()).digest()
shutil.copy2(R/'proposed/pycbc/version.py',W/'pycbc/version.py')
ENV = dict(os.environ,PYTHONPATH=str(W),PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
           PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1',MKL_DYNAMIC='FALSE',MKL_THREADING_LAYER='GNU')
STATE = dict(host=os.uname().nodename,cwd=str(W),pid=os.getpid(),pgid=os.getpgrp(),head=HEAD,results=[])
CHILD = None

def save():
    (O/'status.json').write_text(json.dumps(STATE,indent=2)+'\n')

def stop(*args):
    if CHILD is not None and CHILD.poll() is None: os.killpg(CHILD.pid,signal.SIGTERM)
    raise SystemExit(143)

signal.signal(signal.SIGTERM,stop)
signal.signal(signal.SIGINT,stop)
print(json.dumps(STATE),flush=True)
paths=['test/test_torch_cpu_compat.py','test/test_torch_search_power_scan.py',
       'test/test_torch_chisq_cpu_compat.py','test/test_torch_chisq_precision.py',
       'test/test_torch_chisq_sparse_dispatch.py','test/test_torch_chisq_cuda_normalization.py',
       'test/test_torch_chisq_cpu_optimization.py','test/test_torch_strain_psd_precision.py',
       'test/test_torch_sigmasq_series_precision.py','test/test_torch_psd_protocol.py']
qlty='/home/xangma/pycbc-torch-finish-20260904/qlty/qlty-x86_64-unknown-linux-gnu/qlty'
commands=[('cuda-regressions',['taskset','-c','8',PY,'-B','-m','pytest','-q','-ra',*paths,f'--junitxml={O}/cuda-regressions.xml']),
          ('qlty',['taskset','-c','9',qlty,'check','--no-upgrade-check','--no-fix','--sarif','--no-progress','--jobs','1','--upstream','40e94792b3edf59f39b18b65102b28a4f74433a7'])]
with Path('/home/xangma/pycbc-torch-performance-coordination-20260907/len-benchmark.lock').open() as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name,command in commands:
        STATE['current']=name
        save()
        with (O/f'{name}.stdout').open('w') as out,(O/f'{name}.stderr').open('w') as err:
            CHILD=subprocess.Popen(command,cwd=W,env=ENV,stdout=out,stderr=err,start_new_session=True)
            code=CHILD.wait()
        STATE['results'].append(dict(name=name,command=command,returncode=code))
        save()
        print(json.dumps(STATE['results'][-1]),flush=True)
        if name=='cuda-regressions' and code: raise SystemExit(code)
assert not subprocess.check_output(['git','-C',str(W),'diff','HEAD','--'])
STATE.update(state='complete',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
save()
