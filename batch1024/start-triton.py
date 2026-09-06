import json
from pathlib import Path
import subprocess

root = Path('/home/xangma/pycbc-torch-batch1024-20260906')
command = ['taskset', '-c', '8-11', '/home/xangma/pycbc-torch-split-20260905/venv/bin/python',
           '-B', str(root/'triton-original/run.py'), '--root', str(root/'main'),
           '--python', '/home/xangma/pycbc-torch-split-20260905/venv/bin/python',
           '--expected-sha', '4885b64560e9f39b740e85b6a976898869dd360e',
           '--batches', '1', '8', '32', '128', '512', '1024', '--out', str(root/'triton')]
with (root/'logs/triton.log').open('w') as log:
    proc = subprocess.Popen(command, cwd=root/'main', stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
record = dict(pid=proc.pid, cwd=str(root/'main'), command=command,
              log=str(root/'logs/triton.log'))
(root/'triton-launch.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record))
