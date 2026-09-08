"""Corrected CPU baseline versus rebuilt Torch stack executable experiment."""
import copy
import datetime
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
CONFIG = json.loads((ROOT / 'config.json').read_text())
LOCK = Path('/home/xangma/pycbc-torch-performance-coordination-20260907/len-benchmark.lock')
ROUTES = {
    'corrected-cpu': ('corrected', 'cpu:1'),
    'proposed-cpu': ('proposed', 'cpu:1'),
    'torch-cpu': ('proposed', 'torch:cpu:1'),
    'torch-cuda': ('proposed', 'torch:cuda:0'),
}
STATE = dict(state='starting', pid=os.getpid(), pgid=os.getpgrp(), completed=[],
             started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
ACTIVE_CHILD = None


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def module(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), ROOT / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


CHECKER = module('checked-inspiral')
COMPARATOR = module('compare-triggers')
MONITOR = module('benchmark_cpu_campaign')


def source_info(name):
    source = ROOT / name
    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args], text=True)
    result = dict(commit=git('rev-parse', 'HEAD').strip(), status=git('status', '--porcelain'),
                  tracked_diff=git('diff', 'HEAD', '--'))
    assert result == dict(commit=CONFIG['source_commits'][name], status='', tracked_diff=''), result
    return result


def source_pins():
    result = {}
    for name in ('corrected', 'proposed'):
        source = ROOT / name
        result[name] = dict(info=source_info(name))
        files = subprocess.check_output(['git', '-C', str(source), 'ls-files', '-z']).split(b'\0')
        result[name]['tracked'] = {p.decode(): sha(source / p.decode()) for p in files
                                   if p and (source / p.decode()).is_file()}
        result[name]['native'] = {str(p.relative_to(source)): sha(p)
                                 for p in (source / 'pycbc').rglob('*.so')}
        result[name]['generated_version'] = sha(source / 'pycbc/version.py')
        assert result[name]['native'] == read(ROOT / (name + '-build.json'))['native']
    return result


def pins_unchanged():
    assert source_pins() == read(ROOT / 'source-pins.json'), 'Source/build mutation'
    for path, expected in read(ROOT / 'harness-pins.json').items():
        assert sha(ROOT / path) == expected, path
    for path, expected in CONFIG['input_pins'].items():
        assert sha(path) == expected, path


def qualify(folder, arm):
    import numpy as np
    source_name, scheme = ROUTES[arm]
    q = read(folder / 'qualification.json')
    assert q['status'] == 'success' and q['executable_exit_code'] == 0
    assert q['checks'] and all(q['checks'].values()), q['checks']
    assert q['source_root'] == str(ROOT / source_name)
    assert q['executable']['sha256'] == sha(ROOT / source_name / 'bin/pycbc_inspiral')
    obs = q['observations']
    assert len(obs['banks']) == 1 and obs['banks'][0]['selected_template_count'] == 384
    assert obs['banks'][0]['file']['sha256'] == CONFIG['input_pins'][CONFIG['bank']]
    assert obs['banks'][0]['expected_filter_calls'] == 1920
    assert obs['parsed_options']['chisq_bins'] == '16'
    assert obs['parsed_options']['cluster_window'] == 1.0
    control, = obs['matched_filter_controllers']
    assert control['segment_count'] == 5
    assert [control['filter_bin_start'], control['filter_bin_stop']] == [15360, 1048576]
    geometry, = obs['segment_geometry']
    assert geometry['fft_samples'] == 2097152 and geometry['sample_rate_hz'] == 4096
    assert geometry['unique_analyzed_seconds'] == 1904 and geometry['unique_analyzed_samples'] == 7798784
    assert geometry['gap_samples'] == geometry['overlap_samples'] == 0
    assert geometry['union_analyzed_sample_intervals'] == [[458752, 8257536]]
    assert float(geometry['strain_start_time']) == 1187007048
    hdf = COMPARATOR.load(folder / 'triggers.hdf')
    assert hdf['detectors']['H1']['intervals'] == [[1187007160.0, 1187009064.0]]
    for row in obs['psd_arrays']:
        path = folder / row['relative_path']
        assert str(path) == row['path'] and sha(path) == row['sha256']
        array = np.load(path, allow_pickle=False)
        assert list(array.shape) == row['shape'] and array.dtype.str == row['dtype_str']
        assert hashlib.sha256(array.tobytes()).hexdigest() == row['data_sha256']
        assert row['validity']['valid_for_filter']
    return dict(templates=384, segments=5, valid_seconds=1904, checks=q['checks'])


def compare(left, right, name):
    base = COMPARATOR.load(left / 'triggers.hdf')
    candidate = COMPARATOR.load(right / 'triggers.hdf')
    raw = COMPARATOR.compare(base, candidate, COMPARATOR.DEFAULTS)
    save(ROOT / 'comparisons' / (name + '-raw.json'), raw)
    substitutions = []
    for folder, loaded in ((left, base), (right, candidate)):
        receipt = read(folder / 'receipt.json')
        source_name = ROUTES[receipt['arm']][0]
        assert receipt['source_info'] == source_info(source_name)
        assert not loaded['metadata']['issues'], loaded['metadata']['issues']
        executable = str(ROOT / source_name / 'bin/pycbc_inspiral')
        expected = dict(CONFIG['input_pins'], **{executable: sha(executable)})
        assert loaded['metadata']['consumed_input_sha256'] == expected
    # Code/executable provenance is intentionally different in this experiment.
    # Both revisions and executable contents are independently pinned above.
    # Preserve raw verdicts and normalize only these declared provenance fields.
    fixed = copy.deepcopy(candidate)
    for key in ('source_snapshot', 'consumed_input_sha256'):
        if fixed['metadata'][key] != base['metadata'][key]:
            substitutions.append(dict(field=key, before=fixed['metadata'][key], after=base['metadata'][key]))
            fixed['metadata'][key] = copy.deepcopy(base['metadata'][key])
    result = COMPARATOR.compare(base, fixed, COMPARATOR.DEFAULTS)
    save(ROOT / 'comparisons' / (name + '.json'), dict(result=result,
         provenance_substitutions=substitutions, tolerances=COMPARATOR.DEFAULTS))
    return result


def conditioning_compare(left, right, name):
    import numpy as np
    a, b = [read(p / 'qualification.json')['observations'] for p in (left, right)]
    results = dict(conditioned_strain_exact=a['conditioned_strain'] == b['conditioned_strain'],
                   geometry_exact=a['segment_geometry'] == b['segment_geometry'], psds=[])
    assert len(a['psd_arrays']) == len(b['psd_arrays'])
    for x, y in zip(a['psd_arrays'], b['psd_arrays']):
        av, bv = np.load(left / x['relative_path']), np.load(right / y['relative_path'])
        assert av.shape == bv.shape
        finite = np.isfinite(av) & np.isfinite(bv)
        d = np.abs(av[finite].astype(float) - bv[finite].astype(float))
        budget = 1e-4 * np.maximum(np.abs(av[finite].astype(float)), np.abs(bv[finite].astype(float)))
        results['psds'].append(dict(exact=av.dtype == bv.dtype and av.tobytes() == bv.tobytes(),
            nonfinite_masks_equal=bool(np.array_equal(np.isposinf(av), np.isposinf(bv))),
            relative_budget=1e-4, absolute_floor=0, violations=int(np.count_nonzero(d > budget)),
            max_absolute_difference=float(d.max(initial=0))))
    save(ROOT / 'comparisons' / (name + '-conditioning.json'), results)
    return results


def gpu():
    command = ['nvidia-smi', '--query-gpu=name,uuid,utilization.gpu,memory.used,memory.total',
               '--format=csv,noheader,nounits']
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    return dict(command=command, returncode=result.returncode, output=result.stdout.strip())


def run_case(arm, mode, lock, repeat=None):
    global ACTIVE_CHILD
    source_name, scheme = ROUTES[arm]
    source = ROOT / source_name
    case = f'qual-{arm}' if mode == 'qualify' else f'timing-{arm}-r{repeat}'
    folder = ROOT / 'runs' / case
    folder.mkdir()
    STATE.update(state='running', current=case)
    save(ROOT / 'status.json', STATE)
    pins_unchanged()
    cli = [str(source / 'bin/pycbc_inspiral'), *CONFIG['common_args'],
           '--bank-file', CONFIG['bank'], '--processing-scheme', scheme,
           '--segment-length', '512', '--segment-start-pad', '112', '--segment-end-pad', '16',
           '--output', str(folder / 'triggers.hdf')]
    inner = cli if mode == 'timing' else [str(ROOT / 'qualify-inspiral.py'),
            '--receipt', str(folder / 'qualification.json'), '--', *cli]
    command = ['/usr/bin/time', '-v', '-o', str(folder / 'time.txt'), 'taskset', '-c', '8',
        sys.executable, str(ROOT / 'checked-inspiral.py'), '--receipt', str(folder / 'runtime.json'),
        '--config', str(ROOT / 'config.json'), '--source', str(source), '--scheme', scheme,
        '--lock-fd', str(lock.fileno()), '--', *inner]
    env = CHECKER.clean_environment(os.environ, CONFIG, source)
    input_paths = [*CONFIG['input_pins'], str(source / 'bin/pycbc_inspiral'),
                   str(ROOT / 'config.json'), str(ROOT / 'checked-inspiral.py')]
    record = dict(arm=arm, case=case, mode=mode, repeat=repeat, scheme=scheme,
        state='running', source_info=source_info(source_name), source=str(source),
        executable_cli=cli, command=command, environment=CHECKER.relevant_environment(env),
        cwd=str(ROOT), hostname=os.uname().nodename, parent_pid=os.getpid(),
        input_sha256={p: sha(p) for p in input_paths}, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(folder / 'receipt.json', record)
    sampler = MONITOR.Sampler(folder / 'host-samples.jsonl')
    sampler.take('before')
    save(folder / 'gpu-before.json', gpu())
    stop = threading.Event()
    monitor_errors = []
    def monitor():
        try:
            while not stop.wait(5):
                sampler.take('during')
                with (folder / 'gpu-during.jsonl').open('a') as output:
                    output.write(json.dumps(dict(at=time.time(), **gpu())) + '\n')
        except BaseException as error:
            monitor_errors.append(repr(error))
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    with (folder / 'stdout.log').open('x') as output, (folder / 'stderr.log').open('x') as error:
        start = time.perf_counter()
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output, stderr=error,
                                 pass_fds=(lock.fileno(),), start_new_session=True)
        ACTIVE_CHILD = child.pid
        record['child_pid'] = child.pid
        save(folder / 'receipt.json', record)
        try:
            code = child.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            raise
        elapsed = time.perf_counter() - start
        ACTIVE_CHILD = None
    stop.set()
    thread.join(timeout=20)
    assert not thread.is_alive() and not monitor_errors, monitor_errors
    sampler.take('after')
    save(folder / 'gpu-after.json', gpu())
    record.update(state='complete' if code == 0 else 'failed', returncode=code,
        elapsed_wall_seconds=elapsed, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        input_sha256_after={p: sha(p) for p in input_paths}, source_status_after=source_info(source_name)['status'])
    if (folder / 'triggers.hdf').exists():
        record['trigger_sha256'] = sha(folder / 'triggers.hdf')
    save(folder / 'receipt.json', record)
    assert code == 0, (case, code)
    assert record['input_sha256'] == record['input_sha256_after']
    runtime = read(folder / 'runtime.json')
    assert runtime['state'] == 'complete'
    for key in ('before_executable', 'at_first_bank', 'after_executable'):
        CHECKER.check(runtime[key], record['environment'], bank=key == 'at_first_bank', scheme=scheme)
    if mode == 'qualify':
        save(folder / 'workload-check.json', qualify(folder, arm))
    else:
        verdict = compare(ROOT / 'runs' / f'qual-{arm}', folder, f'{case}-vs-own-qualification')
        assert verdict['status'] == 'pass', verdict
    pins_unchanged()
    STATE['completed'].append(case)
    save(ROOT / 'status.json', STATE)
    print(json.dumps(dict(case=case, elapsed_wall_seconds=elapsed, returncode=code)), flush=True)
    return folder


def main():
    assert read(ROOT / 'setup-status.json')['state'] == 'complete'
    (ROOT / 'runs').mkdir()
    (ROOT / 'comparisons').mkdir()
    save(ROOT / 'source-pins.json', source_pins())
    save(ROOT / 'harness-pins.json', {p.name: sha(p) for p in ROOT.glob('*.py')} |
         {'config.json': sha(ROOT / 'config.json'), 'dependencies.json': sha(ROOT / 'dependencies.json')})
    save(ROOT / 'machine.json', dict(uname=list(os.uname()), topology=MONITOR.discover_topology(),
         cpuinfo=Path('/proc/cpuinfo').read_text(), gpu=gpu(),
         gpu_processes=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
             '--format=csv,noheader'], text=True), reservation='none; shared host and GPU',
         governor={str(c): (Path(f'/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_governor').read_text().strip())
                   for c in (8, 72)}))
    with LOCK.open() as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        STATE['lock'] = dict(path=str(LOCK), inode=os.fstat(lock.fileno()).st_ino)
        qualified = {arm: run_case(arm, 'qualify', lock) for arm in ROUTES}
        comparisons = {}
        for arm in ('proposed-cpu', 'torch-cpu', 'torch-cuda'):
            name = 'corrected-vs-' + arm
            comparisons[name] = compare(qualified['corrected-cpu'], qualified[arm], name)
            condition = conditioning_compare(qualified['corrected-cpu'], qualified[arm], name)
            assert comparisons[name]['status'] == 'pass', comparisons[name]
            assert condition['conditioned_strain_exact'] and condition['geometry_exact'], condition
            assert all(x['nonfinite_masks_equal'] and x['violations'] == 0 for x in condition['psds']), condition
        for arm in ('torch-cpu', 'torch-cuda'):
            name = 'proposed-cpu-vs-' + arm
            comparisons[name] = compare(qualified['proposed-cpu'], qualified[arm], name)
            condition = conditioning_compare(qualified['proposed-cpu'], qualified[arm], name)
            assert comparisons[name]['status'] == 'pass', comparisons[name]
            assert condition['conditioned_strain_exact'] and condition['geometry_exact'], condition
            assert all(x['nonfinite_masks_equal'] and x['violations'] == 0 for x in condition['psds']), condition
        save(ROOT / 'qualification-summary.json', comparisons)
        measurements = {arm: [] for arm in ROUTES}
        for repeat, order in enumerate(CONFIG['ordering'], 1):
            for arm in order:
                folder = run_case(arm, 'timing', lock, repeat)
                measurements[arm].append(read(folder / 'receipt.json')['elapsed_wall_seconds'])
        rows = {}
        for arm, samples in measurements.items():
            median = statistics.median(samples)
            rows[arm] = dict(source=CONFIG['source_commits'][ROUTES[arm][0]], scheme=ROUTES[arm][1],
                samples_seconds=samples, median_seconds=median, min_seconds=min(samples), max_seconds=max(samples),
                template_seconds_per_wall_second=384 * 1904 / median,
                trigger_count=COMPARATOR.load(qualified[arm] / 'triggers.hdf')['detectors']['H1']['count'])
        save(ROOT / 'summary.json', dict(scope=CONFIG['scope'], timing_boundary=CONFIG['timing_boundary'],
            arms=rows, tolerances=COMPARATOR.DEFAULTS,
            corrected_comparisons={k: v['status'] for k, v in comparisons.items() if k.startswith('corrected-')},
            proposed_comparisons={k: v['status'] for k, v in comparisons.items() if k.startswith('proposed-')},
            equal_output_speedup_eligible=all(v['status'] == 'pass' for v in comparisons.values())))
        pins_unchanged()
    STATE.update(state='complete', finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(ROOT / 'status.json', STATE)


if __name__ == '__main__':
    def terminate(*_):
        if ACTIVE_CHILD is not None:
            try:
                os.killpg(ACTIVE_CHILD, signal.SIGTERM)
            except ProcessLookupError:
                pass
        sys.exit(143)
    signal.signal(signal.SIGTERM, terminate)
    try:
        main()
    except BaseException as error:
        STATE.update(state='failed', error=repr(error))
        save(ROOT / 'status.json', STATE)
        raise
