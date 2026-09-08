import os,json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
os.chdir(root)
if os.getpgrp()!=os.getpid(): os.setsid()
print(json.dumps(dict(host=os.uname().nodename,cwd=str(root),pid=os.getpid(),pgid=os.getpgrp(),log=str(root/'status.json'),command=sys.executable+' -u campaign.py')),flush=True)
with (root/'setup.log').open('x') as log:
 p=subprocess.run([sys.executable,'-u',str(root/'setup-remote.py')],stdout=log,stderr=subprocess.STDOUT)
assert p.returncode==0,'Setup failed; see setup.log'
os.execv(sys.executable,[sys.executable,'-u',str(root/'campaign.py')])
