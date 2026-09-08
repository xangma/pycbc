"""Sequential tests on immutable staged heads, without installing anything."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parent
WT = Path('/private/tmp/pycbc-original-cpu-validation-20260908')
PY = '/private/tmp/pycbc-cpu-precision-env-20260908/bin/python'
MANIFEST = OUT / 'manifest.json'
ENV = dict(os.environ, PYTHONPATH=str(WT), PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           PYTHONDONTWRITEBYTECODE='1')
CHILD = None


def stop(*args):
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGTERM)
    raise SystemExit(143)


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
cpu = ['test/test_matchedfilter.py']
torch_precision = ['test/test_torch_chisq_precision.py',
                   'test/test_torch_sigmasq_series_precision.py',
                   'test/test_torch_strain_psd_precision.py']
psd = ['test/test_psd.py', 'test/test_torch_psd_pipeline.py',
       'test/test_torch_psd_protocol.py', 'test/test_torch_versioned_data_psd.py']
filtering = ['test/test_chisq_torch.py', 'test/test_torch_filter_pipeline.py',
             'test/test_torch_chisq_sparse_dispatch.py',
             'test/test_torch_chisq_cpu_optimization.py',
             'test/test_torch_chisq_cuda_normalization.py',
             'test/test_torch_matchedfilter_cpu_optimization.py']
cases = [
    (5, 'pr5-preservation', ['test/test_cpu_array_copy.py','test/test_scheme_runtime.py','test/test_scheme_selection.py','test/test_torch_optional.py'], False),
    (6, 'pr6-preservation', ['test/test_fft_cpu_preservation.py','test/test_torch_fft_cpu_native.py','test/test_torch_fft_writes.py','test/test_torch_large_ifft.py'], False),
    (7, 'pr7-psd', cpu + psd + ['test/test_cpu_psd_variation.py'], False),
    (8, 'pr8-filtering', cpu + torch_precision[:2] + filtering + ['test/test_cpu_skymax_chisq.py','test/test_cpu_match_cache.py'], False),
    (9, 'pr9-strain', cpu + torch_precision + ['test/test_torch_search_kernels.py','test/test_cpu_strain_cache.py'], False),
    (15, 'pr15-main', cpu + torch_precision + psd + filtering +
     ['test/test_torch_search_kernels.py', 'test/test_torch_fft_writes.py', 'test/test_torch_fft_cpu_native.py', 'test/test_fft_cpu_preservation.py', 'test/test_cpu_array_copy.py', 'test/test_cpu_psd_variation.py', 'test/test_cpu_skymax_chisq.py', 'test/test_cpu_match_cache.py', 'test/test_cpu_strain_cache.py', 'test/test_numpy_array_protocol.py', 'test/test_strain.py', 'test/test_scheme_runtime.py', 'test/test_scheme_selection.py', 'test/test_torch_optional.py', 'test/test_torch_large_ifft.py'], False),
    (15, 'pr15-no-torch', cpu + ['test/test_psd.py'], True),
]
if len(sys.argv) > 1:
    cases = [c for c in cases if c[1] in sys.argv[1:]]
print(json.dumps(dict(host=os.uname().nodename, cwd=str(WT), pid=os.getpid(),
                     command=[PY, str(Path(__file__).resolve()), *sys.argv[1:]],
                     log=str(OUT / 'validation-supervisor.log'),
                     stop=f'kill -TERM {os.getpid()}', next_check='30 seconds')), flush=True)
for pr, name, paths, no_torch in cases:
    manifest = json.loads(MANIFEST.read_text())
    row = next(r for r in manifest['prs'] if r['pr'] == pr)
    subprocess.run(['git', '-C', str(WT), 'switch', row['staging_ref']], check=True)
    subprocess.run([PY, str(OUT / 'prepare-runtime.py')], cwd=WT, env=ENV, check=True)
    provenance = subprocess.check_output([PY, '-c',
        'import json,sys,pycbc,pycbc.version,pycbc.vetoes.chisq_cpu,numpy,scipy; '
        'print(json.dumps(dict(python=sys.executable, pycbc=pycbc.__file__, '
        'head=pycbc.version.git_hash, native=pycbc.vetoes.chisq_cpu.__file__, '
        'numpy=numpy.__version__, scipy=scipy.__version__)))'], cwd=WT, env=ENV, text=True)
    provenance = json.loads(provenance)
    assert provenance['head'] == row['new_head']
    assert provenance['pycbc'] == str(WT / 'pycbc/__init__.py')
    xml = OUT / f'{name}.xml'
    command = [PY, str(OUT / 'cpu-without-torch.py')] if no_torch else [PY, '-m', 'pytest', '-q', '-ra']
    command += [*paths, f'--junitxml={xml}']
    started = time.time()
    with (OUT / f'{name}.log').open('w') as log:
        CHILD = subprocess.Popen(command, cwd=WT, env=ENV, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(case=name, pid=CHILD.pid, log=str(OUT / f'{name}.log'), command=command)), flush=True)
        code = CHILD.wait()
    result = dict(case=name, pr=pr, head=row['new_head'], command=command,
                  cwd=str(WT), environment={k: ENV[k] for k in ('PYTHONPATH', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE')},
                  provenance=provenance, exit_code=code, seconds=time.time()-started,
                  log=str(OUT / f'{name}.log'), junit=str(xml))
    if xml.exists():
        suites = list(ET.parse(xml).getroot().iter('testsuite'))
        result['counts'] = {k: sum(int(s.get(k, '0')) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    manifest = json.loads(MANIFEST.read_text())
    manifest['validation'].append(result)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    if code:
        break
