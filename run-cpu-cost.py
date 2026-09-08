"""Alternating original/corrected CPU timing with frozen inputs and checks."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
BASE = Path('/home/xangma/pycbc-torch-baseline-final-20260908')
OUT = ROOT / 'cpu-cost-v1'
SOURCES = {'original': BASE / 'original', 'corrected': ROOT / 'cpu-corrections'}
HEADS = {'original': '40e94792b3edf59f39b18b65102b28a4f74433a7',
         'corrected': '66789ac4a7468094b0cc3ca1498a1de67e0311f6'}
QUAL = {'original': BASE / 'runs/qual-original-cpu',
        'corrected': ROOT / 'linux-validation-v3'}
CONFIG = json.loads((BASE / 'config.json').read_text())
ORDER = [('original', 'corrected'), ('corrected', 'original'),
         ('original', 'corrected'), ('corrected', 'original')]
STATE = dict(state='starting', host=os.uname().nodename, cwd=str(ROOT),
             pid=os.getpid(), completed=[], ordering=ORDER)
CHILD = None


def module(name):
    spec = importlib.util.spec_from_file_location(name, BASE / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


CHECKED = module('checked-inspiral')
COMPARE = module('compare-triggers')
MONITOR = module('benchmark_cpu_campaign')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def pins():
    result = {'inputs': {p: sha(p) for p in CONFIG['input_pins']},
              'harness': {str(p): sha(p) for p in
                          [Path(__file__), BASE / 'config.json',
                           BASE / 'checked-inspiral.py', BASE / 'compare-triggers.py',
                           BASE / 'benchmark_cpu_campaign.py', BASE / 'threadpoolctl.py']},
              'sources': {}}
    assert result['inputs'] == CONFIG['input_pins']
    for name, source in SOURCES.items():
        def git(*args):
            return subprocess.check_output(['git', '-C', str(source), *args])
        assert git('status', '--porcelain') == b''
        assert git('rev-parse', 'HEAD').decode().strip() == HEADS[name]
        tracked = git('ls-files', '-z').split(b'\0')
        result['sources'][name] = dict(head=HEADS[name],
            tracked={p.decode(): sha(source / p.decode()) for p in tracked
                     if p and (source / p.decode()).is_file()},
            native={str(p.relative_to(source)): sha(p)
                    for p in (source / 'pycbc').rglob('*.so')},
            version=sha(source / 'pycbc/version.py'))
    return result


def terminate(*_):
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGTERM)
    raise SystemExit(143)


signal.signal(signal.SIGTERM, terminate)
signal.signal(signal.SIGINT, terminate)
OUT.mkdir()
try:
    assert json.loads((QUAL['corrected'] / 'comparison.json').read_text())['status'] == 'pass'
    references = {name: COMPARE.load(path / 'triggers.hdf') for name, path in QUAL.items()}
    with CHECKED.LOCK.open() as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial = pins()
        save(OUT / 'pins.json', initial)
        samples = {name: [] for name in SOURCES}
        for repeat, order in enumerate(ORDER, 1):
            for name in order:
                source = SOURCES[name]
                case = f'r{repeat}-{name}'
                folder = OUT / case
                folder.mkdir()
                cli = [str(source / 'bin/pycbc_inspiral'), *CONFIG['common_args'],
                       '--bank-file', CONFIG['bank'], '--processing-scheme', 'cpu:1',
                       '--segment-length', '512', '--segment-start-pad', '112',
                       '--segment-end-pad', '16', '--output', str(folder / 'triggers.hdf')]
                command = ['/usr/bin/time', '-v', '-o', str(folder / 'time.txt'),
                           'taskset', '-c', '8', sys.executable, str(BASE / 'checked-inspiral.py'),
                           '--receipt', str(folder / 'runtime.json'), '--config', str(BASE / 'config.json'),
                           '--source', str(source), '--scheme', 'cpu:1', '--lock-fd', str(lock.fileno()),
                           '--', *cli]
                assert pins() == initial
                env = CHECKED.clean_environment(os.environ, CONFIG, source)
                sampler = MONITOR.Sampler(folder / 'host-samples.jsonl')
                sampler.take('before')
                stop = threading.Event()
                monitor_errors = []

                def monitor():
                    try:
                        while not stop.wait(5):
                            sampler.take('during')
                    except BaseException as error:
                        monitor_errors.append(repr(error))

                thread = threading.Thread(target=monitor, daemon=True)
                thread.start()
                STATE.update(state='running', current=case, command=command, started=time.time())
                record = dict(case=case, source=str(source), head=HEADS[name], command=command,
                              host=STATE['host'], cwd=str(ROOT), started=STATE['started'])
                with (folder / 'stdout.log').open('x') as log, (folder / 'stderr.log').open('x') as error:
                    start = time.perf_counter()
                    CHILD = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=error,
                                             pass_fds=(lock.fileno(),), start_new_session=True)
                    STATE['child_pid'] = CHILD.pid
                    save(OUT / 'status.json', STATE)
                    code = CHILD.wait(timeout=600)
                    elapsed = time.perf_counter() - start
                    CHILD = None
                stop.set()
                thread.join(timeout=20)
                assert not thread.is_alive() and not monitor_errors
                sampler.take('after')
                record.update(returncode=code, elapsed_wall_seconds=elapsed,
                              finished=time.time(), trigger_sha256=sha(folder / 'triggers.hdf'))
                save(folder / 'receipt.json', record)
                assert code == 0
                runtime = json.loads((folder / 'runtime.json').read_text())
                assert runtime['state'] == 'complete'
                assert all(not runtime[k]['torch_imported'] for k in
                           ('before_executable', 'at_first_bank', 'after_executable'))
                verdict = COMPARE.compare(references[name], COMPARE.load(folder / 'triggers.hdf'), COMPARE.DEFAULTS)
                save(folder / 'comparison.json', verdict)
                assert verdict['status'] == 'pass', verdict
                assert pins() == initial
                samples[name].append(elapsed)
                STATE['completed'].append(case)
                save(OUT / 'status.json', STATE)
                print(json.dumps(dict(case=case, seconds=elapsed, output_verdict='pass')), flush=True)
        rows = {name: dict(head=HEADS[name], samples_seconds=values,
                          median_seconds=statistics.median(values),
                          min_seconds=min(values), max_seconds=max(values))
                for name, values in samples.items()}
        ratio = rows['corrected']['median_seconds'] / rows['original']['median_seconds']
        save(OUT / 'summary.json', dict(status='complete', arms=rows,
             corrected_over_original_wall_ratio=ratio, wall_change_percent=(ratio - 1) * 100,
             ordering=ORDER, source_and_input_pins_unchanged=True,
             own_qualification_comparisons='all pass',
             scope='Shared host, one core, four trials per arm; descriptive cost of changed numerical behavior.',
             timing_boundary=CONFIG['timing_boundary'],
             equal_output_speedup_eligible=False))
        STATE.update(state='complete', finished=time.time())
except BaseException as error:
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGTERM)
        CHILD.wait(timeout=20)
    STATE.update(state='failed', error=repr(error))
    raise
finally:
    save(OUT / 'status.json', STATE)
