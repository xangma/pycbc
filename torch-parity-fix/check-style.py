"""Run CI unused-import groups and record changed production Ruff checks."""
import json
from pathlib import Path
import subprocess
import sys

O=Path(__file__).resolve().parent
W=Path(sys.argv[1])
PY='/private/tmp/pycbc-cpu-precision-env-20260908/bin/python'
RUFF='/Users/xangma/miniconda3/bin/ruff'
FLAKE='/Users/xangma/miniconda3/bin/flake8'
groups={
    'modules':[str(p.relative_to(W)) for p in (W/'pycbc').rglob('*.py') if p.name not in ('__init__.py','version.py')],
    'tests':[str(p.relative_to(W)) for p in (W/'test').rglob('*.py') if 'test_schemes' not in str(p)],
    'executables':[str(p.relative_to(W)) for p in (W/'bin').rglob('pycbc_*') if p.is_file()],
}
results=[]
for name,files in groups.items():
    command=[FLAKE,'--select=F401',*files]
    p=subprocess.run(command,cwd=W,text=True,capture_output=True)
    (O/f'final-f401-{name}.log').write_text(p.stdout+p.stderr)
    results.append(dict(group=name,files=len(files),returncode=p.returncode))
(O/'final-f401.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results),flush=True)
assert all(r['returncode']==0 for r in results)
files=['pycbc/types/torch_compat.py','pycbc/psd/__init__.py','pycbc/filter/matchedfilter.py','pycbc/strain/strain.py','pycbc/vetoes/chisq_torch.py']
for name,command in [('ruff-check',[RUFF,'check','--output-format=json',*files]),('ruff-format',[RUFF,'format','--check',*files])]:
    p=subprocess.run(command,cwd=W,text=True,capture_output=True)
    (O/f'final-{name}.log').write_text(p.stdout+p.stderr)
    print(name,p.returncode,flush=True)
    if name=='ruff-format':assert p.returncode==0,p.stdout+p.stderr
