#!/usr/bin/env python3
"""Run final-v6 scientific checks sequentially; --check only reads inputs.

The extra L256/L1024 CPU qualifications are diagnostic evidence. They do not
extend the timing campaign. The unchanged validators retain all original gates.
This runner never writes the aggregate scientific-validation-v6.json receipt.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback


PARENT = '837f38d493420043e45fb1ad210a0ccf68bacbaa'
PINNED = {
    'config.json': 'e29df1ee26ad1a513982a81b55d1f8b648126baacc7b509bc918e5c8e912d36a',
    'source-v5.json': 'db00a8b27cd8a8986aa6c55d403e1d396f89ca686f80badd0755b29847925ae8',
    'run-case.py': '243eef653386a76d1ead4b92542bc84088e882df9011faab5a8154154da9c456',
    'qualify-inspiral.py': '234ff35cc4d54ab2e9dfd5d475d825a328346183adaf516f46df4653e50450b7',
    'validate-waveforms.py': '9c761995585d27364d97960fd74f2efb3b64f7e978178f348072901b6f4489c6',
    'check-boundary-injections.py': '60fbc9ab4c3a67cd038b081d5f5d1367e39ed7e4efc5a0f9b65f5337738a7b1b',
    'qualify-compressed-bank-v6.py': '150db2eba01bd699959b9a851c9bbe06f6b828a6c3ab134cc7e0d86912bf1460',
    'compression-1e5.json': '9d5724fbc1961fe20ff346b8ba39959f4c7fab1aae949126957aece2499279ee',
    'inputs/bank-metadata.json': '565488d9e10981c47a8f4cf42d074b9385efb8214a89b8cbd6a58412a9ce8538',
    'inputs/bank-compressed-1e5.hdf': '161820754117cd51b24a7bb672ea456cf7b2a3808e69690c23235d024b1a2916',
}
WAVEFORM_LIMITS = dict(fixed_time_phase_optimized_mismatch=1e-3,
                       relative_weighted_amplitude_error=1e-2,
                       relative_snr_norm_error=1e-2)
BOUNDARY_LIMITS = dict(maximum_relative_complex_error=1e-3,
                       relative_peak_snr_error=1e-3,
                       peak_location_error_samples=1, edge_energy_fraction=1e-6)
QUALIFIERS = {256: 'qual-science6-cpu-l256',
              512: 'qual-selected6-cpu-l512',
              1024: 'qual-science6-cpu-l1024'}
BACKENDS = ('cpu', 'torch-cpu', 'torch-cuda')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def snapshot(paths):
    return {str(path): digest(path) if Path(path).is_file() else None for path in paths}


def git(source, *args):
    return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()


def source_info(source):
    return dict(commit=git(source, 'rev-parse', 'HEAD'),
                status=git(source, 'status', '--porcelain'))


def bind(context, path, expected=None):
    path = Path(path)
    require(path.is_absolute(), f'Expected absolute input: {path}')
    value = digest(path)
    require(expected is None or value == expected, f'Input hash differs: {path}')
    old = context['inputs'].setdefault(str(path), value)
    require(old == value, f'Input changed after first use: {path}')
    return path


def read(context, path, expected=None):
    path = bind(context, path, expected)
    value = json.loads(path.read_text())
    bind(context, path, context['inputs'][str(path)])
    return value


def manifest(context, value):
    hashes = value['input_sha256']
    require(hashes and hashes == value['input_sha256_after'],
            'Missing or changed recorded inputs')
    for path, expected in hashes.items():
        bind(context, path, expected)


def idle(root, own_status=None):
    paths = {*root.glob('*.status.json'), *root.glob('unit-tests-v6*.json')}
    for path in sorted(paths):
        if path != own_status:
            require(json.loads(path.read_text()).get('state') != 'running',
                    f'Another job is running: {path}')


def plan(root, python):
    source = root / 'source-v6'
    prefix = ['taskset', '-c', '8', python, '-u']
    result = []
    for length in (256, 1024):
        name = QUALIFIERS[length]
        result.append(dict(name=name, kind='qualification', length=length,
                           output=str(root / 'runs' / name), command=[
            *prefix, str(root / 'run-case.py'), '--config', str(root / 'config.json'),
            '--source', str(source), '--bank', str(root / 'inputs/bank-compressed-1e5.hdf'),
            '--case', name, '--mode', 'qualify', '--scheme', 'cpu:1',
            '--segment-length', str(length), '--start-pad', '112', '--end-pad', '16']))
    common = ['--bank-metadata', str(root / 'inputs/bank-metadata.json'),
              '--compression-receipt', str(root / 'compression-1e5.json'),
              '--source', str(source)]
    for name, script, lengths in (
        ('waveform-validation-v6', 'validate-waveforms.py', (256, 512, 1024)),
        ('boundary-injections-v6', 'check-boundary-injections.py', (1024,)),
    ):
        output = root / f'{name}.json'
        qualifiers = [arg for length in lengths for arg in (
            '--qualification', str(root / 'runs' / QUALIFIERS[length] / 'qualification.json'))]
        result.append(dict(name=name, kind='diagnostic', output=str(output), command=[
            *prefix, str(root / script), *common, '--output', str(output), *qualifiers]))
    result.append(dict(name='compressed-bank-v6', kind='diagnostic',
                       output=str(root / 'compressed-bank-v6.json'), command=[
        *prefix, str(root / 'qualify-compressed-bank-v6.py')]))
    for step in result:
        step['log'] = str(root / f"scientific-checks-v6-{step['name']}.log")
    return result


def qualification(context, name, length, scheme='cpu:1'):
    root, source = context['root'], context['source']
    directory = root / 'runs' / name
    run = read(context, directory / 'receipt.json')
    expected_source = dict(context['before'], tracked_diff='')
    require(run['state'] == 'complete' and run['returncode'] == 0 and
            run.get('finished_utc') and run['source_info'] == expected_source and
            run['source_status_after'] == '' and run['source'] == str(source) and
            (run['case'], run['mode'], run['scheme']) == (name, 'qualify', scheme) and
            (run['segment_length'], run['start_pad'], run['end_pad']) == (length, 112, 16),
            f'Incomplete or wrong qualification run: {name}')
    manifest(context, run)
    bind(context, directory / 'triggers.hdf', run['trigger_sha256'])
    expected_cli = [str(source / 'bin/pycbc_inspiral'), *context['config']['common_args'],
                    '--bank-file', str(root / 'inputs/bank-compressed-1e5.hdf'),
                    '--processing-scheme', scheme, '--segment-length', str(length),
                    '--segment-start-pad', '112', '--segment-end-pad', '16',
                    '--output', str(directory / 'triggers.hdf')]
    require(run['executable_cli'] == expected_cli and run['environment'] == context['environment'],
            f'Qualification command or environment differs: {name}')
    value = read(context, directory / 'qualification.json')
    require(value['status'] == 'success' and value['executable_exit_code'] == 0 and
            value['source_root'] == str(source) and value['checks'] and
            all(check is True for check in value['checks'].values()),
            f'Qualification checks failed: {name}')
    require(Path(value['python']) == Path(sys.executable), 'Qualification interpreter differs')
    for record in [value['executable'], value['wrapper'], *value['source_modules'].values()]:
        bind(context, record['path'], record['sha256'])
    observations = value['observations']
    require(len(observations['banks']) == len(observations['segment_geometry']) == 1,
            'Expected one bank and one geometry')
    bank, geometry = observations['banks'][0], observations['segment_geometry'][0]
    require(bank['selected_template_count'] == len(bank['templates']) == 96 and
            set(str(row['template_hash']) for row in bank['templates'].values()) == context['template_hashes'],
            'Qualified bank population differs')
    require(bank['file']['path'] == str(root / 'inputs/bank-compressed-1e5.hdf'), 'Wrong qualified bank')
    bind(context, bank['file']['path'], bank['file']['sha256'])
    require(geometry['delta_f_hz'] == 1 / length and geometry['sample_rate_hz'] == 4096 and
            geometry['unique_analyzed_seconds'] == 1904, 'Qualified grid or analyzed duration differs')
    psds = observations['psd_arrays']
    require(len(psds) == 1, 'Expected one measured shared PSD per qualification')
    require(sorted(i for row in psds for i in row['segment_indices']) == list(range(len(geometry['segments']))),
            'PSD mapping omits or duplicates segments')
    for record in psds:
        require(record['delta_f_hz'] == 1 / length and record['n_samples'] == length * 4096 // 2 + 1 and
                record['scaling'] == 'DYN_RANGE_FAC**2', 'PSD grid or scaling differs')
        bind(context, record['path'], record['sha256'])
    return {str(path): digest(path) for path in sorted(directory.rglob('*')) if path.is_file()}


def prepare(root, source_manifest_sha256):
    require(re.fullmatch('[0-9a-f]{64}', source_manifest_sha256 or '') is not None,
            'Supply the reviewed source-v6.json SHA256')
    context = dict(root=root, source=root / 'source-v6', inputs={})
    for name, expected in PINNED.items():
        bind(context, root / name, expected)
    bind(context, Path(__file__).resolve())
    cfg = read(context, root / 'config.json')
    require(cfg['core'] == 8, 'Expected frozen CPU core 8')
    context.update(config=cfg, environment=dict(cfg['environment'],
                   PYTHONPATH=str(context['source']), PYTHONDONTWRITEBYTECODE='1'))
    current = read(context, root / 'source-v6.json', source_manifest_sha256)
    before = source_info(context['source'])
    require(current['schema_version'] == 1 and current['source'] == str(context['source']) and
            re.fullmatch('[0-9a-f]{40}', current['commit']) and current['parent'] == PARENT and
            before == dict(commit=current['commit'], status='') and
            git(context['source'], 'rev-parse', 'HEAD^') == PARENT, 'Wrong or modified final source')
    context.update(before=before, tree=git(context['source'], 'rev-parse', 'HEAD^{tree}'))
    manifest(context, current)
    for name, expected in current['changed_files_sha256'].items():
        bind(context, context['source'] / name, expected)
    old_source = root / 'source-v5'
    require(source_info(old_source) == dict(commit=PARENT, status=''), 'Frozen v5 CUDA reference changed')
    previous = read(context, root / 'source-v5.json')
    require(current['native_modules_sha256'] == previous['native_modules_sha256'], 'Native modules changed')
    for name, expected in current['native_modules_sha256'].items():
        bind(context, context['source'] / name, expected)
        bind(context, old_source / name, expected)
    bind(context, old_source / 'pycbc/waveform/decompress_torch.py')
    for name in ('bin/pycbc_inspiral', 'pycbc/waveform/bank.py', 'pycbc/waveform/compress.py',
                 'pycbc/waveform/decompress_cpu.py', 'pycbc/waveform/decompress_torch.py'):
        bind(context, context['source'] / name)
    compression = read(context, root / 'compression-1e5.json')
    require(compression['state'] == 'complete' and compression['returncode'] == 0, 'Compression failed')
    manifest(context, compression)
    bind(context, root / 'inputs/bank-compressed-1e5.hdf',
         compression['output_sha256'][str(root / 'inputs/bank-compressed-1e5.hdf')])
    metadata = read(context, root / 'inputs/bank-metadata.json')
    context['template_hash_order'] = [str(row['template_hash']) for row in metadata['templates']]
    context['template_hashes'] = {str(row['template_hash']) for row in metadata['templates']}
    require(len(metadata['templates']) == len(context['template_hashes']) == 96, 'Wrong metadata population')
    unit = read(context, root / 'unit-tests-v6.json')
    require(unit['state'] == 'complete' and unit['passed'] is True and unit['returncode'] == 0 and
            unit['source_info'] == unit['source_after'] == before and unit.get('finished_utc'), 'Unit checks failed')
    manifest(context, unit)
    bind(context, root / 'unit-tests-v6.log', unit['log_sha256'])
    stage = read(context, root / 'campaign-v6-qualifications.status.json')
    expected_cases = [dict(case=f'qual-selected6-{label}-l512', mode='qualify', scheme=scheme)
                      for label, scheme in zip(BACKENDS, ('cpu:1', 'torch:cpu:1', 'torch:cuda:0'))]
    require(stage['state'] == 'complete' and stage['returncode'] == 0 and stage.get('finished_utc') and
            stage['source_info'] == stage['source_after'] == before and stage['current'] is None and
            stage['completed'] == expected_cases, 'Selected qualification stage is incomplete')
    manifest(context, stage)
    require(stage['input_sha256'].get(str(root / 'source-v6.json')) == source_manifest_sha256,
            'Qualification stage does not bind the reviewed final source')
    require(stage['output_sha256'], 'Qualification output manifest is missing')
    for path, expected in stage['output_sha256'].items():
        bind(context, path, expected)
    for case in expected_cases:
        outputs = qualification(context, case['case'], 512, case['scheme'])
        require(all(stage['output_sha256'].get(path) == value for path, value in outputs.items()),
                'Selected qualification outputs are not bound by their stage')
    context['plan'] = plan(root, sys.executable)
    context['status'] = root / 'scientific-checks-v6.status.json'
    outputs = [context['status'], context['status'].with_suffix('.tmp')]
    outputs += [Path(step[key]) for step in context['plan'] for key in ('output', 'log')]
    outputs += [root / 'compressed-bank-v6.tmp']
    require(not any(path.exists() for path in outputs), 'Scientific output already exists; preserve prior evidence')
    idle(root)
    require(snapshot(context['inputs']) == context['inputs'], 'Inputs changed during preflight')
    return context


def validate_diagnostic(context, step):
    value = read(context, Path(step['output']))
    require(value['state'] == 'complete' and value['passed'] is True and value.get('finished_utc'),
            f"Scientific diagnostic failed: {step['name']}")
    expected_source = context['before']
    if step['name'] == 'compressed-bank-v6':
        require(value['source_info'] == value['source_after'] == expected_source and value['returncode'] == 0,
                'Compressed-bank source or completion differs')
        require(value['expected_cases'] == value['completed_cases'] == len(value['cases']) == 576 and
                value['native_calls'] == value['cuda_comparisons'] == 288, 'Incomplete compressed-bank matrix')
        keys = {(row['device'], row['length_seconds'], row['index']) for row in value['cases']}
        require(keys == {(device, length, index) for device in ('cpu', 'cuda')
                         for length in (256, 512, 1024) for index in range(96)} and
                all(row['bitwise_equal'] is True for row in value['cases']), 'Compressed-bank cases differ')
        for row in value['cases']:
            require(row['template_hash'] == context['template_hash_order'][row['index']] and
                    re.fullmatch('[0-9a-f]{64}', row['output_sha256']) and
                    row['reference'] == ('normal CPU' if row['device'] == 'cpu' else 'frozen v5 Torch CUDA') and
                    (row['device'] != 'cpu' or row['native_used'] is True), 'Compressed-bank identity differs')
    else:
        source = dict(expected_source, root=str(context['source']), tree=context['tree'])
        require(value['source_before'] == value['source_after'] == source, 'Diagnostic source differs')
        if step['name'] == 'waveform-validation-v6':
            require(value['tolerances'] == WAVEFORM_LIMITS and
                    value['expected_template_psd_pairs'] == value['completed_template_psd_pairs'] == 288 and
                    len(value['cases']) == 3 and all(case['passed'] is True for case in value['cases']),
                    'Waveform gates or coverage differ')
            require({case['qualification'] for case in value['cases']} == {
                str(context['root'] / 'runs' / name / 'qualification.json') for name in QUALIFIERS.values()
            }, 'Waveform qualification set differs')
            for case in value['cases']:
                require(case['state'] == 'complete' and len(case['templates']) == 96 and
                        {row['template_hash'] for row in case['templates']} == context['template_hashes'],
                        'Waveform population differs')
                for row in case['templates']:
                    require(row['passed'] is True and row['decompression_succeeded_without_generation'] is True and
                            len(row['psd_results']) == 1 and row['psd_results'][0]['passed'] is True and
                            all(0 <= row['psd_results'][0]['metrics'][key] <= limit
                                for key, limit in WAVEFORM_LIMITS.items()), 'Waveform numerical gate failed')
        else:
            require(value['tolerances'] == BOUNDARY_LIMITS and value['target_reference_peak_snr'] == 8 and
                    value['expected_cases'] == value['completed_cases'] == 36 and
                    value['segment_lengths_seconds'] == [256, 512, 1024] and
                    value['start_pads_seconds'] == [96, 112] and value['end_pad_seconds'] == 16 and
                    value['long_reference_seconds'] == 4096 and len(value['templates']) == 2 and
                    all(row['passed'] is True and len(row['cases']) == 18 for row in value['templates']),
                    'Boundary gates or coverage differ')
            require(value['qualification'] == str(context['root'] / 'runs' / QUALIFIERS[1024] / 'qualification.json') and
                    {row['kind'] for row in value['templates']} == {'longest', 'shortest'}, 'Boundary inputs differ')
            for row in value['templates']:
                require(row['reference_edge_budget_passed'] is True and all(
                    0 <= row['reference']['tails'][key] <= BOUNDARY_LIMITS['edge_energy_fraction']
                    for key in ('injection_edge_energy_fraction', 'kernel_edge_energy_fraction')),
                    'Boundary reference edge energy failed')
                keys = {(case['segment_length_seconds'], case['start_pad_seconds'], case['placement'])
                        for case in row['cases']}
                require(keys == {(length, start, placement) for length in (256, 512, 1024)
                                 for start in (96, 112) for placement in ('first_valid', 'midpoint', 'last_valid')},
                        'Boundary case matrix differs')
                require(all(case['passed'] is True and all(0 <= case[key] <= limit
                            for key, limit in BOUNDARY_LIMITS.items() if key != 'edge_energy_fraction')
                            for case in row['cases']), 'Boundary numerical gate failed')
    manifest(context, value)
    return value


def run(context):
    root, status, inputs = context['root'], context['status'], context['inputs']
    record = dict(schema_version=1, state='running', passed=False, returncode=None,
                  pid=os.getpid(), host=os.uname().nodename, cwd=str(root),
                  command=[sys.executable, *sys.argv], started_utc=utc(), source_info=context['before'],
                  environment=context['environment'], cpu_affinity=[8], input_sha256=inputs,
                  input_capture_policy='Inputs are bound at first use; new diagnostic qualifier outputs are bound before validation.',
                  plan=context['plan'], completed=[], commands=[], current=None, child_pid=None,
                  output_sha256={}, expected_coverage=dict(waveform_template_psd_pairs=288,
                  boundary_cases=36, compressed_bank_cases=576))
    child = None
    started = time.monotonic()

    def save(first=False):
        temporary = status if first else status.with_suffix('.tmp')
        with temporary.open('x') as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write('\n')
        if not first:
            temporary.replace(status)

    def unchanged():
        require(source_info(context['source']) == context['before'], 'Final source changed')
        require(source_info(root / 'source-v5') == dict(commit=PARENT, status=''), 'Frozen reference source changed')
        require(snapshot(inputs) == inputs, 'Scientific input changed')
        require(snapshot(record['output_sha256']) == record['output_sha256'], 'Completed scientific output changed')

    def stop(signum, frame):
        raise InterruptedError(f'Received signal {signum}')

    previous_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    save(first=True)
    try:
        env = dict(os.environ, **context['environment'])
        for step in context['plan']:
            idle(root, status)
            unchanged()
            record.update(current=step['command'], child_pid=None)
            save()
            with Path(step['log']).open('x') as log:
                child = subprocess.Popen(step['command'], cwd=root, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                record['child_pid'] = child.pid
                save()
                code = child.wait()
                child = None
            record['commands'].append(dict(name=step['name'], command=step['command'],
                                            returncode=code, log=step['log']))
            record.update(current=None, child_pid=None)
            record['output_sha256'][step['log']] = digest(step['log'])
            require(code == 0, f"Child failed ({code}): {step['name']}")
            unchanged()
            if step['kind'] == 'qualification':
                outputs = qualification(context, step['name'], step['length'])
                record['output_sha256'].update(outputs)
            else:
                validate_diagnostic(context, step)
                record['output_sha256'][step['output']] = digest(step['output'])
            record['completed'].append(step['name'])
            save()
        unchanged()
        record.update(state='complete', passed=True, returncode=0)
    except BaseException as error:
        record.update(state='failed', passed=False, returncode=1,
                      error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
    finally:
        if child is not None and child.poll() is None:
            for sig in previous_handlers:
                signal.signal(sig, signal.SIG_IGN)
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        try:
            record['source_after'] = source_info(context['source'])
            record['input_sha256_after'] = snapshot(inputs)
            unchanged()
        except Exception as error:
            record.update(state='invalid-input-mutation', passed=False, returncode=1,
                          provenance_error=f'{type(error).__name__}: {error}')
            record['input_sha256_after'] = snapshot(inputs)
        record.update(current=None, child_pid=None, finished_utc=utc(),
                      wall_seconds=time.monotonic() - started)
        partial_paths = []
        for step in context['plan']:
            output = Path(step['output'])
            if output.is_dir():
                partial_paths.extend(p for p in output.rglob('*') if p.is_file())
            else:
                partial_paths.append(output)
            partial_paths.append(Path(step['log']))
        record['all_created_output_sha256'] = {str(p): digest(p) for p in partial_paths if p.is_file()}
        save()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    print(json.dumps(dict(status=str(status), state=record['state'], passed=record['passed'],
                          completed=record['completed'])))
    return record['returncode']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest-sha256')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.describe:
        print(json.dumps(plan(root, sys.executable), indent=2))
        return 0
    if not args.source_manifest_sha256:
        parser.error('--source-manifest-sha256 is required')
    context = prepare(root, args.source_manifest_sha256)
    if args.check:
        print(json.dumps(dict(state='ready', source_info=context['before'],
                              plan=context['plan'], input_sha256=context['inputs']), indent=2))
        return 0
    return run(context)


if __name__ == '__main__':
    sys.exit(main())
