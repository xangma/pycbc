"""Validate the isolated CPU correction split with target MKL and workload."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = Path('/home/xangma/pycbc-torch-baseline-final-20260908')
SOURCE = ROOT/'cpu-corrections'
OUT = ROOT/'linux-validation-v2'
OUT.mkdir()
CONFIG = json.loads((BASE/'config.json').read_text())
spec = importlib.util.spec_from_file_location('checked',BASE/'checked-inspiral.py')
checked = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checked)
state = dict(state='starting',host=os.uname().nodename,cwd=str(ROOT),pid=os.getpid(),completed=[])
child = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n')


def pins():
    assert subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain']) == b''
    paths = subprocess.check_output(['git','-C',str(SOURCE),'ls-files','-z']).split(b'\0')
    return dict(commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),
        tracked={p.decode():sha(SOURCE/p.decode()) for p in paths if p and (SOURCE/p.decode()).is_file()},
        native={str(p.relative_to(SOURCE)):sha(p) for p in (SOURCE/'pycbc').rglob('*.so')})


def terminate(*_):
    if child is not None:
        os.killpg(child.pid,signal.SIGTERM)
    raise SystemExit(143)


signal.signal(signal.SIGTERM,terminate)
signal.signal(signal.SIGINT,terminate)
try:
    with checked.LOCK.open() as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial = pins()
        assert initial['commit'] == '66789ac4a7468094b0cc3ca1498a1de67e0311f6'
        assert {p:sha(p) for p in CONFIG['input_pins']} == CONFIG['input_pins']
        save('source-pins.json',initial)
        cli = [str(SOURCE/'bin/pycbc_inspiral'),*CONFIG['common_args'],'--bank-file',CONFIG['bank'],
               '--processing-scheme','cpu:1','--segment-length','512','--segment-start-pad','112',
               '--segment-end-pad','16','--output',str(OUT/'triggers.hdf')]
        replay = [str(BASE/'checked-inspiral.py'),'--receipt',str(OUT/'runtime.json'),'--config',str(BASE/'config.json'),
                  '--source',str(SOURCE),'--scheme','cpu:1','--lock-fd',str(lock.fileno()),'--',
                  str(BASE/'qualify-inspiral.py'),'--receipt',str(OUT/'qualification.json'),'--',*cli]
        for label, tail in [('tests',[str(ROOT/'linux-cpu-tests-v2.py')]),('replay',replay)]:
            command = ['taskset','-c','8',sys.executable,'-B',*tail]
            state.update(state='running',current=label,command=command,started=time.time())
            with (OUT/(label+'.log')).open('x') as log:
                child = subprocess.Popen(command,cwd=SOURCE,env=checked.clean_environment(os.environ,CONFIG,SOURCE),
                    stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),),start_new_session=True)
                state['child_pid'] = child.pid
                save('status.json',state)
                print(json.dumps(state),flush=True)
                code = child.wait(timeout=600)
                child = None
            assert code == 0, (label,code)
            state['completed'].append(label)
        q = json.loads((OUT/'qualification.json').read_text())
        runtime = json.loads((OUT/'runtime.json').read_text())
        assert q['status'] == 'success' and all(q['checks'].values())
        assert runtime['state'] == 'complete'
        assert all(not runtime[key]['torch_imported'] for key in ('before_executable','at_first_bank','after_executable'))
        fields = {}
        with h5py.File(BASE/'runs/qual-proposed-cpu/triggers.hdf') as expected,h5py.File(OUT/'triggers.hdf') as observed:
            for name,value in expected['H1'].items():
                if isinstance(value,h5py.Dataset):
                    fields[name] = bool(np.array_equal(value[:],observed['H1/'+name][:],equal_nan=True))
        psd_exact = bool(np.array_equal(np.load(OUT/'arrays/qualification-psd-000.npy'),
            np.load(BASE/'runs/qual-proposed-cpu/arrays/qualification-psd-000.npy')))
        assert len(fields) == 13 and all(fields.values()) and psd_exact
        unchanged = pins() == initial and {p:sha(p) for p in CONFIG['input_pins']} == CONFIG['input_pins']
        assert unchanged
        save('comparison.json',dict(status='pass',comparison='CPU split versus frozen proposed CPU',
             trigger_fields_exact=fields,psd_exact=psd_exact,source_and_input_pins_unchanged=unchanged,
             no_torch_imported=True,qualification_checks=q['checks']))
        state.update(state='complete',finished=time.time())
except BaseException as error:
    if child is not None and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        child.wait(timeout=20)
    state.update(state='failed',error=repr(error))
    raise
finally:
    save('status.json',state)
