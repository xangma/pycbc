"""Run targeted Linux runtime regressions and static analysis after qualification."""
from pathlib import Path
import datetime, fcntl, json, os, subprocess, sys
R=Path(__file__).resolve().parent; W=R/'proposed'; O=R/'linux-checks'; O.mkdir(exist_ok=True)
LOCK=Path('/home/xangma/pycbc-torch-performance-coordination-20260907/len-benchmark.lock')
ENV=dict(os.environ, PYTHONPATH=str(W), PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()
state=dict(pid=os.getpid(),head=head,cwd=str(W),state='running',results=[])
def save(): (O/'status.json').write_text(json.dumps(state,indent=2)+'\n')
save()
with LOCK.open() as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 commands=[('runtime', ['taskset','-c','8',sys.executable,'-B','-m','pytest','-q','-ra','test/test_scheme_runtime.py','test/test_scheme_selection.py','test/test_torch_optional.py','test/test_torch_fft_cpu_native.py','test/test_torch_fft_writes.py','test/test_torch_large_ifft.py','test/test_torch_large_batches.py','test/test_fft_cpu_preservation.py',f'--junitxml={O}/runtime.xml']),('qlty',['taskset','-c','9','/home/xangma/pycbc-torch-finish-20260904/qlty/qlty-x86_64-unknown-linux-gnu/qlty','check','--no-upgrade-check','--no-fix','--sarif','--no-progress','--jobs','1','--upstream','40e94792b3edf59f39b18b65102b28a4f74433a7'])]
 for name,command in commands:
  state['current']=name;save()
  with (O/(name+'.log')).open('w') as out,(O/(name+'.stderr')).open('w') as err:
   r=subprocess.run(command,cwd=W,env=ENV,stdout=out,stderr=err)
  state['results'].append(dict(name=name,command=command,returncode=r.returncode));save()
state.update(state='complete',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
