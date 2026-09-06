"""Prepare isolated, exact published checkouts for the fresh campaign."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/xangma/pycbc-torch-benchmark-20260906')
BUNDLE = Path('/home/xangma/pycbc-torch-format-20260906/final/candidate.bundle')
PYTHON = '/home/xangma/pycbc-torch-split-20260905/venv/bin/python'
HEADS = {
    'main': '607bce53ead14f12af32552a5b2441d3bc667267',
    'fft': 'e6073eaf1a89cfed69af53707f52321eadf129f1',
    'cpu': '1a2ebea088d9e0a31cbb22c19ad24f96ffea2b7c',
}
ROOT.mkdir(exist_ok=True)
(ROOT / 'logs').mkdir(exist_ok=True)
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
env.pop('PYTHONPATH', None)
results = []
for name, sha in HEADS.items():
    repo = ROOT / name
    if not repo.exists():
        subprocess.run(['git', 'clone', '--quiet', '--no-hardlinks', '--no-checkout',
                        '/home/xangma/pycbc-torch-format-20260906/validation', str(repo)], check=True)
    subprocess.run(['git', 'fetch', '--quiet', str(BUNDLE),
                    '+refs/heads/codex/torch-format-20260906/*:refs/remotes/publish/*'],
                   cwd=repo, check=True)
    subprocess.run(['git', 'checkout', '--quiet', '--detach', sha], cwd=repo, check=True)
    started = time.time()
    with (ROOT / f'logs/build-{name}.log').open('w') as out:
        subprocess.run([PYTHON, 'setup.py', 'build_ext', '--inplace', '-j', '4'],
                       cwd=repo, env=env, stdout=out, stderr=subprocess.STDOUT, check=True)
    status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=repo, text=True)
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
                  runtime=runtime, tracked_status=status,
                  binaries={str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (repo / 'pycbc').rglob('*.so')})
    assert result['runtime']['pycbc'].startswith(str(repo)+'/')
    results.append(result)
    (ROOT / 'preparation.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(result), flush=True)
(ROOT / 'prepared').write_text('complete\n')
