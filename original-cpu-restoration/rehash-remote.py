from pathlib import Path
import hashlib,json,subprocess,datetime
R=Path('/home/xangma/pycbc-torch-original-cpu-restoration-v2-20260908')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
c=read(R/'config.json');pins=read(R/'source-pins.json');report={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'inputs':{},'sources':{},'native':{}}
for p,expected in c['input_pins'].items():assert sha(p)==expected;report['inputs'][p]=expected
for name,pin in pins.items():
 W=R/name
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip();assert head==c['source_commits'][name]
 assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=W,text=True).strip()
 for p,expected in pin['tracked'].items():assert sha(W/p)==expected,(name,p)
 for p,expected in pin['native'].items():assert sha(W/p)==expected,(name,p)
 report['sources'][name]={'head':head,'tracked_files_rehashed':len(pin['tracked']),'tracked_clean':True,'post_qualification_untracked':subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).splitlines()};report['native'][name]=pin['native']
print(json.dumps(report,indent=2))
