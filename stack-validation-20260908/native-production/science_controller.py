"""Three bounded production science checks; no performance measurements."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source'
PYTHON = '/home/xangma/pycbc-torch-split-20260905/venv/bin/python'
LOCK = Path('/home/xangma/pycbc-torch-performance-coordination-20260907/len-benchmark.lock')
BANK = '/home/xangma/pycbc-torch-reference-protocol-20260907/inputs/bank-compressed.hdf'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def source_record():
    def git(*args):
        return subprocess.check_output(['git', '-C', str(SOURCE), *args], text=True)
    return dict(commit=git('rev-parse', 'HEAD').strip(), status=git('status', '--porcelain'),
                tracked_diff=git('diff', 'HEAD', '--'))


def pins():
    source = read(ROOT / 'source-pins.json')
    assert source_record() == dict(commit=source['commit'], status='', tracked_diff='')
    for name, expected in source['tracked_files'].items():
        assert sha(SOURCE / name) == expected, name
    for name, record in read(ROOT / 'native-extension-pins.json').items():
        assert sha(SOURCE / name) == record['sha256'], name
    for name, expected in read(ROOT / 'science-manifest.json').items():
        assert sha(ROOT / name) == expected, name
    for name, expected in read(ROOT / 'input-pins.json').items():
        assert sha(name) == expected, name
    for record in read(ROOT / 'science-reference-pins.json').values():
        for name, expected in record['sha256'].items():
            assert sha(Path(record['remote']) / name) == expected, name


def qualify(folder, scheme):
    import numpy as np
    q = read(folder / 'qualification.json')
    assert q['status'] == 'success' and q['executable_exit_code'] == 0
    assert q['checks'] and all(q['checks'].values()), q['checks']
    assert q['source_root'] == str(SOURCE)
    assert q['executable']['sha256'] == sha(SOURCE / 'bin/pycbc_inspiral')
    obs = q['observations']
    assert len(obs['banks']) == 1 and obs['banks'][0]['selected_template_count'] == 384
    assert obs['banks'][0]['file']['sha256'] == read(ROOT / 'input-pins.json')[BANK]
    assert obs['banks'][0]['expected_filter_calls'] == 1920
    assert obs['parsed_options']['chisq_bins'] == '16' and obs['parsed_options']['cluster_window'] == 1.0
    control, = obs['matched_filter_controllers']
    assert control['segment_count'] == 5
    assert [control['filter_bin_start'], control['filter_bin_stop']] == [15360, 1048576]
    psds = []
    for row in obs['psd_arrays']:
        path = folder / row['relative_path']
        assert str(path) == row['path'] and sha(path) == row['sha256']
        array = np.load(path, allow_pickle=False)
        assert list(array.shape) == row['shape'] and array.dtype.str == row['dtype_str']
        assert array.nbytes == row['nbytes'] and path.stat().st_size == row['bytes']
        assert hashlib.sha256(array.tobytes()).hexdigest() == row['data_sha256']
        assert row['validity']['valid_for_filter']
        psds.append({key: value for key, value in row.items() if key != 'path'})
    science = dict(conditioned_strain=obs['conditioned_strain'], psd_arrays=psds,
                   segment_geometry=obs['segment_geometry'])
    assert science == read(ROOT / 'science-reference.json')['by_scheme'][scheme]
    if scheme == 'torch:cuda:0':
        graph = obs['offline_graph']
        assert len(graph['calls']) == graph['replays'] == 1920
        assert len(graph['captures']) == len(graph['final_bindings']) == 5
        assert graph['capture_contexts'] == list(range(5))
        assert [(row['template'], row['segment']) for row in graph['calls']] == [
            ([0, template], segment) for template in range(384) for segment in range(5)]
        assert all(all(row[key] for key in ('full_bytes_equal', 'sparse_bytes_equal',
            'inputs_unchanged', 'binding_unchanged')) for row in graph['calls'])
        assert graph['retained_nonempty'] == sum(not row['empty'] for row in graph['calls']) > 1
        offsets = [v['analyzed_sample_interval'][0] for v in obs['segment_geometry'][0]['segments']]
        for row in graph['calls']:
            if row['empty']:
                continue
            consumer = row['consumer']
            assert consumer['verified'] and consumer['offset'] == offsets[row['segment']]
            assert consumer['stage'] == 'complete' and consumer['transition_count'] == 1
            assert consumer['observed_after'] == consumer['expected_after']
            assert consumer['observed_versions_after'] == consumer['expected_versions_after']
            assert consumer['expected_versions_after'] == [consumer['versions_before'][0] + 1,
                                                           consumer['versions_before'][1]]
            assert consumer['before'] == row['sparse']
    return dict(templates=384, segments=5, valid_seconds=1904,
                exact_conditioning_psd_geometry=True, checks=q['checks'])


def compare(name, left, right, exact):
    import h5py
    comparator = module('production_comparator', 'compare-triggers.py')
    result = comparator.compare(comparator.load(left / 'triggers.hdf'),
                                comparator.load(right / 'triggers.hdf'), comparator.DEFAULTS)
    certificates = None
    if exact:
        certificates = {}
        with h5py.File(left / 'triggers.hdf') as a, h5py.File(right / 'triggers.hdf') as b:
            datasets = []
            for handle in (a, b):
                paths = {}
                handle['H1'].visititems(lambda key, value: paths.update({key: value})
                                       if isinstance(value, h5py.Dataset) else None)
                datasets.append(paths)
            assert set(datasets[0]) == set(datasets[1])
            telemetry = {'search/run_time', 'search/filter_rate_per_core',
                         'search/setup_time_fraction', 'search/templates_per_core'}
            assert telemetry <= datasets[0].keys()
            keys = sorted(datasets[0].keys() - telemetry)
            assert len(keys) == 18
            for key in keys:
                av, bv = datasets[0][key][()], datasets[1][key][()]
                assert av.dtype == bv.dtype and av.shape == bv.shape and av.tobytes() == bv.tobytes(), key
                certificates[key] = dict(shape=list(av.shape), dtype=av.dtype.str,
                                          sha256=hashlib.sha256(av.tobytes()).hexdigest())
    path = ROOT / (name + '.json')
    save(path, dict(result=result, tolerances=comparator.DEFAULTS, exact_H1=certificates))
    assert result['status'] == 'pass', (name, result)
    return dict(path=str(path), sha256=sha(path), exact_science_datasets=18 if exact else None)


def main():
    status_path = ROOT / 'science-status.json'
    assert not status_path.exists()
    status = dict(state='running', host=os.uname().nodename, cwd=str(ROOT),
                  pid=os.getpid(), pgid=os.getpgrp(), started=time.time(), runs=[], comparisons={})
    save(status_path, status)
    def terminated(signum, _frame):
        raise SystemExit(f'Terminated by signal {signum}')
    signal.signal(signal.SIGTERM, terminated)
    checker = module('production_runtime', 'checked-inspiral.py')
    try:
        with LOCK.open('r') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            status['lock'] = dict(path=str(LOCK), fd=lock.fileno(), inode=os.fstat(lock.fileno()).st_ino)
            pins()
            refs = read(ROOT / 'science-reference-pins.json')
            for name, scheme in [('standard', 'cpu:1'), ('cpu', 'torch:cpu:1'), ('cuda', 'torch:cuda:0')]:
                pins()
                folder = ROOT / 'runs' / name
                folder.mkdir(parents=True, exist_ok=False)
                config_path = ROOT / ('config-graph.json' if name == 'cuda' else 'config.json')
                cfg = read(config_path)
                env = checker.clean_environment(os.environ, cfg, SOURCE)
                cli = [str(SOURCE / 'bin/pycbc_inspiral'), *cfg['common_args'],
                       '--bank-file', BANK, '--processing-scheme', scheme,
                       '--segment-length', '512', '--segment-start-pad', '112', '--segment-end-pad', '16',
                       '--output', str(folder / 'triggers.hdf')]
                qualifier = 'qualify_production_graph.py' if name == 'cuda' else 'qualify-inspiral.py'
                command = ['taskset', '-c', '8', PYTHON, str(ROOT / 'checked-inspiral.py'),
                           '--receipt', str(folder / 'runtime.json'), '--config', str(config_path),
                           '--source', str(SOURCE), '--scheme', scheme, '--lock-fd', str(lock.fileno()),
                           '--', str(ROOT / qualifier), '--receipt', str(folder / 'qualification.json'), '--', *cli]
                inputs = [*read(ROOT / 'input-pins.json'), str(SOURCE / 'bin/pycbc_inspiral')]
                record = dict(name=name, scheme=scheme, command=command, executable_cli=cli,
                              source_info=source_record(), environment=checker.fixed_environment(cfg, SOURCE),
                              input_sha256={path: sha(path) for path in inputs}, state='running',
                              cwd=str(ROOT), log=str(folder / 'stderr.log'), started=time.time())
                status['runs'].append(record)
                save(status_path, status)
                save(folder / 'receipt.json', record)
                with (folder / 'stdout.log').open('x') as stdout, (folder / 'stderr.log').open('x') as stderr:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                             stdout=stdout, stderr=stderr, pass_fds=(lock.fileno(),))
                    record['pid'] = child.pid
                    save(status_path, status)
                    try:
                        code = child.wait(timeout=1800)
                    finally:
                        if child.poll() is None:
                            child.kill()
                            child.wait()
                record.update(returncode=code, finished=time.time(), state='complete' if code == 0 else 'failed',
                              source_status_after=source_record()['status'],
                              input_sha256_after={path: sha(path) for path in inputs})
                for filename, key in [('runtime.json', 'runtime_sha256'), ('triggers.hdf', 'trigger_sha256')]:
                    if (folder / filename).exists():
                        record[key] = sha(folder / filename)
                save(folder / 'receipt.json', record)
                save(status_path, status)
                assert code == 0, name
                pins()
                assert record['input_sha256'] == record['input_sha256_after']
                runtime = read(folder / 'runtime.json')
                assert runtime['state'] == 'complete' and runtime['source'] == runtime['imported_source'] == str(SOURCE)
                assert runtime['environment'] == record['environment']
                for stage in ('before_executable', 'at_first_bank', 'after_executable'):
                    checker.check(runtime[stage], record['environment'], bank=stage == 'at_first_bank', scheme=scheme)
                record['qualification'] = qualify(folder, scheme)
                key = name + '-vs-own-reference'
                status['comparisons'][key] = compare(key, Path(refs[name]['remote']), folder, True)
                if name != 'standard':
                    key = name + '-vs-standard'
                    status['comparisons'][key] = compare(key, ROOT / 'runs/standard', folder, False)
                save(status_path, status)
            pins()
            status.update(state='complete', source_unchanged=True)
    except BaseException as error:
        status.update(state='failed', error=repr(error))
        raise
    finally:
        status['finished'] = time.time()
        save(status_path, status)


if __name__ == '__main__':
    main()
