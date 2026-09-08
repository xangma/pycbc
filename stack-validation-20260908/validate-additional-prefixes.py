"""Run owned regressions at each rebuilt feature prefix, retaining all logs."""
import json
import os
import pathlib
import re
import socket
import subprocess
import time

OUT = pathlib.Path('/Users/xangma/repos/pycbc/artifacts/torch-publication-20260908')
TREE = pathlib.Path('/private/tmp/pycbc-torch-publication-20260908')
PY = '/private/tmp/pycbc-torch-publication-20260908-venv/bin/python'
heads = json.loads((OUT / 'final-heads.json').read_text())
owners = json.loads((OUT / 'ownership.json').read_text())
prs = {str(p['number']): p for p in json.loads((OUT / 'pr-before.json').read_text())}
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYCBC_TEST_SCHEME='torch:cpu', PYTHONPATH=str(TREE))
receipts = []
for n in range(5, 16):
    key = str(n)
    subprocess.run(['git', 'checkout', '--detach', heads[key]], cwd=TREE, check=True, stdout=subprocess.DEVNULL)
    tests = json.loads((OUT/'additional-owned-tests.json').read_text()).get(key, [])
    if not tests:
        continue
    cmd = [PY, '-m', 'pytest', '-q', '-ra', '-p', 'no:cacheprovider', '--tb=short', '--junitxml=' + str(OUT / f'additional-pr{n}-tests.xml'), *tests]
    started = time.time()
    with (OUT / f'additional-pr{n}-tests.log').open('w') as log:
        child = subprocess.Popen(cmd, cwd=TREE, env=env, stdout=log, stderr=subprocess.STDOUT)
        live = dict(host=socket.gethostname(), cwd=str(TREE), command=cmd, pid=child.pid, controller_pid=os.getpid(), log=str(OUT / f'additional-pr{n}-tests.log'), head=heads[key], started=started)
        (OUT / 'active-prefix-test.json').write_text(json.dumps(live, indent=2) + '\n')
        result = child.wait()
    receipts.append(dict(**live, returncode=result, finished=time.time()))
    (OUT / 'additional-prefix-test-results.json').write_text(json.dumps(receipts, indent=2) + '\n')
    print(n, result, len(tests), flush=True)
    if result:
        break
subprocess.run(['git', 'checkout', '--detach', heads['15']], cwd=TREE, check=True, stdout=subprocess.DEVNULL)
