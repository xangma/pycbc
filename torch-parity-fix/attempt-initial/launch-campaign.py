"""Transfer a pinned isolated qualification and collect verified evidence."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

O = Path(__file__).resolve().parent
C = O/'campaign'
R = Path('/Users/xangma/repos/pycbc')
REMOTE = '/home/xangma/pycbc-torch-parity-fix-20260908'
PYTHON = '/home/xangma/pycbc-torch-split-20260905/venv/bin/python'
manifest = json.loads((O/'manifest.json').read_text())
main = next(r for r in manifest['prs'] if r['pr']==15)
config = json.loads((C/'config.json').read_text())
config.update(description='Production Torch-only CPU arithmetic compatibility versus unchanged original CPU.',
              source_commits=dict(original=manifest['cpu_base'],proposed=main['new_head']))
(C/'config.json').write_text(json.dumps(config,indent=2)+'\n')
subprocess.run(['git','bundle','create',str(C/'sources.bundle'),'refs/heads/'+main['staging_ref'],'^'+manifest['cpu_base']],cwd=R,check=True)
driver = '''import os,json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
os.chdir(root)
if os.getpgrp()!=os.getpid(): os.setsid()
print(json.dumps(dict(host=os.uname().nodename,cwd=str(root),pid=os.getpid(),pgid=os.getpgrp(),log=str(root/'status.json'),command=sys.executable+' -u campaign.py')),flush=True)
with (root/'setup.log').open('x') as log:
 p=subprocess.run([sys.executable,'-u',str(root/'setup-remote.py')],stdout=log,stderr=subprocess.STDOUT)
assert p.returncode==0,'Setup failed; see setup.log'
os.execv(sys.executable,[sys.executable,'-u',str(root/'campaign.py')])
'''
(C/'driver.py').write_text(driver)
archive=O/'campaign-upload.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for path in C.iterdir():
        tar.add(path,arcname=path.name)
subprocess.run(['ssh','len','mkdir -p '+shlex.quote(REMOTE)],check=True)
subprocess.run(['scp',str(archive),'len:'+REMOTE+'/upload.tar.gz'],check=True)
remote_setup=f'import hashlib,tarfile,pathlib; r=pathlib.Path({REMOTE!r}); a=r/"upload.tar.gz"; assert hashlib.sha256(a.read_bytes()).hexdigest()=={hashlib.sha256(archive.read_bytes()).hexdigest()!r}; tarfile.open(a).extractall(r,filter="data")'
subprocess.run(['ssh','len','python3 -c '+shlex.quote(remote_setup)],check=True)
command = shlex.quote(PYTHON)+' -u '+shlex.quote(REMOTE+'/driver.py')
with (O/'driver.log').open('x') as log:
    proc=subprocess.Popen(['ssh','len',command],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in proc.stdout:
        print(line,end='',flush=True)
        log.write(line);log.flush()
    code=proc.wait()
print('Remote exit:',code,flush=True)
collect=f'''import hashlib,json,pathlib,tarfile
r=pathlib.Path({REMOTE!r})
files=[p for p in r.rglob('*') if p.is_file() and p.parts[len(r.parts)] not in ('original','proposed','repo') and p.name not in ('upload.tar.gz','evidence.tar.gz','output-manifest.json')]
pins={{str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(r/'output-manifest.json').write_text(json.dumps(pins,indent=2)+'\\n')
with tarfile.open(r/'evidence.tar.gz','w:gz') as tar:
 for p in files+[r/'output-manifest.json']:tar.add(p,arcname=str(p.relative_to(r)))
print(hashlib.sha256((r/'evidence.tar.gz').read_bytes()).hexdigest())
'''
checksum=subprocess.check_output(['ssh','len','python3 -c '+shlex.quote(collect)],text=True).strip()
subprocess.run(['scp','len:'+REMOTE+'/evidence.tar.gz',str(O/'evidence.tar.gz')],check=True)
assert hashlib.sha256((O/'evidence.tar.gz').read_bytes()).hexdigest()==checksum
out=O/'remote'
out.mkdir()
with tarfile.open(O/'evidence.tar.gz') as tar:tar.extractall(out,filter='data')
pins=json.loads((out/'output-manifest.json').read_text())
for relative,expected in pins.items():assert hashlib.sha256((out/relative).read_bytes()).hexdigest()==expected,relative
(O/'transfer-verification.json').write_text(json.dumps(dict(archive_sha256=checksum,verified_files=len(pins),remote_exit=code),indent=2)+'\n')
print('Verified files:',len(pins),flush=True)
if (out/'summary.json').exists():
 s=json.loads((out/'summary.json').read_text());print(json.dumps({k:s[k] for k in ('cpu_preserved','trigger_counts','comparison_status','scientific_gates_pass')},indent=2),flush=True)
raise SystemExit(code)
