"""Read actual Torch dependency identity after all timed processes finish."""
import json
from pathlib import Path
import shlex
import subprocess

O = Path(__file__).resolve().parent
REMOTE = '/home/xangma/pycbc-torch-fresh-benchmark-20260908'
PYTHON = '/home/xangma/pycbc-torch-split-20260905/venv/bin/python'
inner = r'''
import datetime,hashlib,importlib.metadata as md,json,os,pathlib,sys,torch
def distribution_record(d):
    return dict(name=d.metadata['Name'],version=d.version,metadata_path=str(d._path),root=str(d.locate_file('')))
files = {str(pathlib.Path(p).resolve()):hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
         for p in (torch.__file__,torch.version.__file__)}
print(json.dumps(dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    python=sys.version,executable=sys.executable,cwd=os.getcwd(),sys_path=sys.path,
    torch=dict(file=torch.__file__,version=torch.__version__,cuda=torch.version.cuda,module_sha256=files),
    selected_distribution=distribution_record(md.distribution('torch')),
    all_torch_distributions=[distribution_record(d) for d in md.distributions()
                             if d.metadata['Name'].lower().replace('_','-')=='torch']),indent=2))
'''
outer = f'''
import importlib.util,json,os,pathlib,subprocess,sys
r=pathlib.Path({REMOTE!r})
assert json.loads((r/'status.json').read_text())['state']=='complete'
spec=importlib.util.spec_from_file_location('checked',r/'checked-inspiral.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
env=m.clean_environment(os.environ,json.loads((r/'config.json').read_text()),r/'proposed')
subprocess.run([sys.executable,'-B','-c',{inner!r}],env=env,cwd=r,check=True)
'''
command = ['ssh', 'len', shlex.quote(PYTHON) + ' -B -c ' + shlex.quote(outer)]
result = json.loads(subprocess.check_output(command, text=True))
inventory = json.loads((O / 'remote/dependencies.json').read_text())['packages']['torch']
versions = {json.loads(p.read_text())['torch_version'] for p in (O / 'remote/runs').glob('*torch*/runtime.json')}
assert versions == {result['torch']['version']}, versions
result.update(timing_boundary='Supplemental read-only inspection after all timings completed.',
              original_flat_inventory_version=inventory,
              inventory_policy='The original flat inventory collapses duplicate distribution names; actual run receipts and this selected import identify the runtime. All duplicate Torch distributions are retained here.')
path = O / 'additional-source-metadata/dependencies-actual.json'
path.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k:result[k] for k in ('torch','selected_distribution','all_torch_distributions','original_flat_inventory_version')},indent=2))
