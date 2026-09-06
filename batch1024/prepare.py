"""Build isolated published source revisions for the batch-1024 campaign."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/xangma/pycbc-torch-batch1024-20260906')
PYTHON = '/home/xangma/pycbc-torch-split-20260905/venv/bin/python'
HEADS = {
    'main': '4885b64560e9f39b740e85b6a976898869dd360e',
    'fft': 'd840198592a0ca128f4f89d90987a7469ec37c8f',
    'cpu': 'd544420232428225c214a4be84fbe1262a6d307b',
}
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
env.pop('PYTHONPATH', None)
results = []
for name, sha in HEADS.items():
    repo = ROOT / name
    if repo.exists():
        raise RuntimeError(f'Will not reuse checkout {repo}')
    subprocess.run(['git', 'clone', '--quiet', '--no-hardlinks', '--no-checkout',
                    '/home/xangma/pycbc-taylorf2-triton-20260906/validation', str(repo)], check=True)
    subprocess.run(['git', 'fetch', '--quiet', str(ROOT / 'published.bundle'),
                    'refs/heads/*:refs/remotes/batch-published/*'],
                   cwd=repo, check=True)
    subprocess.run(['git', 'checkout', '--quiet', '--detach', sha], cwd=repo, check=True)
    started = time.time()
    with (ROOT / f'logs/build-{name}.log').open('w') as out:
        subprocess.run([PYTHON, 'setup.py', 'build_ext', '--inplace', '-j', '4'],
                       cwd=repo, env=env, stdout=out, stderr=subprocess.STDOUT, check=True)
    status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True)
    assert not status, status
    probe = subprocess.check_output([PYTHON, '-c',
        'import json,pycbc,torch,sys; from pycbc.filter import matchedfilter_cpu as m; '
        'from pycbc.vetoes import chisq_cpu as c; '
        'print(json.dumps(dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,'
        'pycbc=pycbc.__file__,matchedfilter_extension=m.__file__,chisq_extension=c.__file__,'
        'cpu_peak_kernel=hasattr(m,"_batch_abs_arg_max_complex64"))))'], cwd=repo, env=env, text=True)
    (ROOT / f'logs/runtime-{name}.log').write_text(probe)
    runtime = json.loads(next(line for line in reversed(probe.splitlines()) if line.startswith('{')))
    result = dict(name=name, sha=sha, build_seconds=time.time()-started,
                  runtime=runtime, status=status,
                  binaries={str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (repo / 'pycbc').rglob('*.so')})
    assert runtime['pycbc'].startswith(str(repo)+'/')
    results.append(result)
    (ROOT / 'preparation.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(dict(name=name, sha=sha, build_seconds=result['build_seconds'])), flush=True)
(ROOT / 'prepared').write_text('complete\n')
