import hashlib,json,shlex,subprocess,tarfile
from pathlib import Path
O=Path(__file__).resolve().parent
R='/home/xangma/pycbc-torch-parity-fix-v2-20260908'
P='/home/xangma/pycbc-torch-split-20260905/venv/bin/python'
assert json.loads((O/'remote/summary.json').read_text())['scientific_gates_pass']
subprocess.run(['scp',str(O/'linux-checks.py'),'len:'+R+'/linux-checks.py'],check=True)
code="import os;os.setsid() if os.getpgrp()!=os.getpid() else None;os.execv("+repr(P)+",["+repr(P)+",'-u',"+repr(R+'/linux-checks.py')+"])"
with (O/'linux-driver.log').open('w') as log:
 p=subprocess.Popen(['ssh','len',P+' -c '+shlex.quote(code)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 for line in p.stdout:print(line,end='',flush=True);log.write(line);log.flush()
 ret=p.wait()
collect="import hashlib,json,pathlib,tarfile;r=pathlib.Path("+repr(R)+");d=r/'linux-checks';files=[p for p in d.rglob('*') if p.is_file()];pins={str(p.relative_to(d)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};(d/'SHA256SUMS.json').write_text(json.dumps(pins,indent=2));a=r/'linux-checks.tar.gz';t=tarfile.open(a,'w:gz');t.add(d,arcname='linux-checks');t.close();print(hashlib.sha256(a.read_bytes()).hexdigest())"
sha=subprocess.check_output(['ssh','len','python3 -c '+shlex.quote(collect)],text=True).strip()
subprocess.run(['scp','len:'+R+'/linux-checks.tar.gz',str(O/'linux-checks.tar.gz')],check=True)
a=O/'linux-checks.tar.gz';assert hashlib.sha256(a.read_bytes()).hexdigest()==sha
with tarfile.open(a) as t:t.extractall(O,filter='data')
pins=json.loads((O/'linux-checks/SHA256SUMS.json').read_text())
for n,h in pins.items():assert hashlib.sha256((O/'linux-checks'/n).read_bytes()).hexdigest()==h
(O/'linux-transfer-verification.json').write_text(json.dumps(dict(sha256=sha,files=len(pins),exit_code=ret),indent=2)+'\n')
print('Verified Linux outputs',len(pins),'exit',ret,flush=True)
raise SystemExit(ret)
