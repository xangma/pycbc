"""Validate immutable owner and assembled heads with pinned local binaries."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

O = Path(__file__).resolve().parent
R = Path('/Users/xangma/repos/pycbc')
W = Path('/private/tmp/pycbc-torch-parity-validation-20260908')
P = '/private/tmp/pycbc-cpu-precision-env-20260908/bin/python'
OUT = O/'tests'
OUT.mkdir(exist_ok=True)
M = json.loads((O/'manifest.json').read_text())
HEADS = {r['pr']:r['new_head'] for r in M['prs']}
ENV = dict(os.environ, PYTHONPATH=str(W), PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           PYTHONDONTWRITEBYTECODE='1')
CHILD = None

def stop(*args):
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGTERM)
    raise SystemExit(143)

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)

def git(*args):
    return subprocess.check_output(['git','-C',str(W if W.exists() else R),*args],text=True).strip()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

cases = [
    (5, 'runtime-owner', ['test/test_torch_cpu_compat.py','test/test_scheme_runtime.py','test/test_scheme_selection.py','test/test_torch_optional.py']),
    (7, 'psd-owner', ['test/test_psd.py','test/test_torch_psd_pipeline.py','test/test_torch_psd_protocol.py','test/test_torch_versioned_data_psd.py']),
    (8, 'filter-owner', ['test/test_chisq_torch.py','test/test_torch_chisq_precision.py','test/test_torch_chisq_sparse_dispatch.py','test/test_torch_chisq_cuda_normalization.py','test/test_torch_chisq_cpu_optimization.py','test/test_torch_search_power_scan.py','test/test_torch_sigmasq_series_precision.py']),
    (9, 'search-owner', ['test/test_torch_strain_psd_precision.py','test/test_torch_search_kernels.py','test/test_cpu_strain_cache.py']),
    (15, 'main', ['test/test_torch_cpu_compat.py','test/test_torch_search_power_scan.py','test/test_matchedfilter.py','test/test_torch_chisq_precision.py','test/test_torch_sigmasq_series_precision.py','test/test_torch_strain_psd_precision.py','test/test_psd.py','test/test_torch_psd_pipeline.py','test/test_torch_psd_protocol.py','test/test_torch_versioned_data_psd.py','test/test_chisq_torch.py','test/test_torch_filter_pipeline.py','test/test_torch_chisq_sparse_dispatch.py','test/test_torch_chisq_cpu_optimization.py','test/test_torch_chisq_cuda_normalization.py','test/test_torch_matchedfilter_cpu_optimization.py','test/test_torch_search_kernels.py','test/test_torch_fft_writes.py','test/test_torch_fft_cpu_native.py','test/test_fft_cpu_preservation.py','test/test_cpu_array_copy.py','test/test_cpu_psd_variation.py','test/test_cpu_skymax_chisq.py','test/test_cpu_match_cache.py','test/test_cpu_strain_cache.py','test/test_numpy_array_protocol.py','test/test_strain.py','test/test_scheme_runtime.py','test/test_scheme_selection.py','test/test_torch_optional.py','test/test_torch_large_ifft.py']),
    (16, 'fft-leaf', ['test/test_fft_batched_backends.py','test/test_fftw_wisdom_cache.py','test/test_fft_cli_wisdom.py','test/test_torch_batched_fft.py','test/test_live_batch_torch_fft_integration.py']),
    (17, 'cpu-leaf', ['test/test_torch_chisq_cpu_optimization.py','test/test_torch_cpu_fft_tuning.py','test/test_torch_cpu_native_peaks.py','test/test_cpu_batch_peaks.py','test/test_torch_large_ifft.py']),
]
if len(sys.argv)>1:
    cases = [c for c in cases if c[1] in sys.argv[1:]]
if not W.exists(): git('worktree','add','--detach',str(W),HEADS[15])
state = dict(host=os.uname().nodename,cwd=str(W),pid=os.getpid(),command=[P,str(Path(__file__).resolve()),*sys.argv[1:]],log=str(OUT/'status.json'),results=[])
print(json.dumps(state),flush=True)
for n,name,paths in cases:
    assert not git('status','--porcelain','--untracked-files=no')
    git('checkout','--detach',HEADS[n])
    source = Path('/private/tmp/pycbc-original-cpu-docs-runtime-20260908') if n not in (16,17) else Path(f'/private/tmp/pycbc-original-cpu-followup-validation-20260908-pr{n}')
    source_head = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    assert not git('diff','--name-only',source_head,HEADS[n],'--','*.pyx','*.pxd','*.pxi','*.c','*.cc','*.cpp','*.h','*.hpp','*.cu','pycbc/lib','setup.py')
    native = {}
    for item in (source/'pycbc').rglob('*.so'):
        rel = item.relative_to(source)
        shutil.copy2(item,W/rel)
        native[str(rel)] = sha(item)
        assert sha(W/rel)==native[str(rel)]
    assert len(native)==11
    version_code = "import ast; from pathlib import Path; t=ast.parse(Path('setup.py').read_text()); f=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='get_version_info'); d={}; exec(compile(ast.Module(body=[f],type_ignores=[]),'setup.py','exec'),d); d['get_version_info']()"
    subprocess.run([P,'-c',version_code],cwd=W,env=ENV,check=True,capture_output=True)
    # Include the newly added independent pointwise CPU-oracle file, if present.
    if n in (8,15):
        for item in sorted((W/'test').glob('*chisq*compat*.py')):
            relative = str(item.relative_to(W))
            if relative not in paths: paths.append(relative)
    assert all((W/p).exists() for p in paths), paths
    command = [P,'-B','-m','pytest','-q','-ra',*paths,f'--junitxml={OUT/name}.xml']
    result = dict(pr=n,head=HEADS[n],case=name,command=command,native_reference=source_head,native=native,version_sha256=sha(W/'pycbc/version.py'))
    state['current']=name
    (OUT/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    start=time.time()
    with (OUT/f'{name}.log').open('w') as log:
        CHILD=subprocess.Popen(command,cwd=W,env=ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(case=name,pid=CHILD.pid,log=str(OUT/f'{name}.log'))),flush=True)
        result['returncode']=CHILD.wait()
    result['seconds']=time.time()-start
    xml=OUT/f'{name}.xml'
    if xml.exists():
        suites=list(ET.parse(xml).getroot().iter('testsuite'))
        result['counts']={k:sum(int(s.get(k,'0')) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert not git('status','--porcelain','--untracked-files=no')
    state['results'].append(result)
    (OUT/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('case','returncode','seconds','counts')}),flush=True)
    if result['returncode']: raise SystemExit(result['returncode'])
state['state']='complete'
(OUT/'status.json').write_text(json.dumps(state,indent=2)+'\n')
