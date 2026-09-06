import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
root=Path(__file__).resolve().parent
assert json.loads((root/'status.json').read_text())['state']=='complete'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
rows=[]
for source in sorted((root/'runs').glob('*/perf.data')):
 target=source.with_suffix('.data.gz')
 with source.open('rb') as f, target.open('xb') as out:
  with gzip.GzipFile(fileobj=out,mode='wb',filename='',mtime=0) as compressed: shutil.copyfileobj(f,compressed)
 report=source.with_name('perf-report.txt')
 with report.open('w') as out, source.with_name('perf-report.stderr.log').open('w') as err:
  result=subprocess.run(['perf','report','--stdio','--no-children','--percent-limit','0.1','-i',str(source)],stdout=out,stderr=err)
 assert result.returncode==0
 rows.append(dict(raw_path=str(source.relative_to(root)),raw_sha256=sha(source),raw_bytes=source.stat().st_size,gzip_path=str(target.relative_to(root)),gzip_sha256=sha(target),gzip_bytes=target.stat().st_size,report_path=str(report.relative_to(root)),report_sha256=sha(report)))
assert len(rows)==3
(root/'profile-transport.json').write_text(json.dumps(rows,indent=2)+'\n')
packages={}
for name in ('numpy','scipy','torch','mkl','lalsuite','h5py'):
 try: packages[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError: packages[name]=None
(root/'environment.json').write_text(json.dumps(dict(hostname=platform.node(),platform=platform.platform(),python=sys.version,packages=packages,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total,uuid','--format=csv,noheader'],text=True),affinity=os.sched_getaffinity(0).__repr__()),indent=2)+'\n')
print(json.dumps(rows,indent=2))
