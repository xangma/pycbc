#!/usr/bin/env python3
"""Run one frozen v6 stage; --check validates without writing or running cases."""
import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


PARENT = '837f38d493420043e45fb1ad210a0ccf68bacbaa'
ALLOWED = (
    'pycbc/fft/torchfft.py', 'pycbc/waveform/decompress_torch.py',
    'test/test_torch_decompress_cpu.py', 'test/test_torch_large_ifft.py',
)
CPU_PATHS = (
    'bin/pycbc_inspiral', 'pycbc/scheme.py', 'pycbc/fft/backend_support.py',
    'pycbc/fft/backend_cpu.py', 'pycbc/fft/class_api.py', 'pycbc/fft/mkl.py',
    'pycbc/waveform/compress.py', 'pycbc/waveform/decompress_cpu.py',
)
DISPATCH = dict(scheme_prefix='cpu', fft='pycbc.fft.mkl',
                decompression='pycbc.waveform.decompress_cpu')
SCIENCE_CHECKS = {
    'waveform_reference', 'compressed_bank_backend_parity',
    'boundary_injections', 'qualification_trigger_parity',
}
BACKENDS = (('cpu', 'cpu:1'), ('torch-cpu', 'torch:cpu:1'),
            ('torch-cuda', 'torch:cuda:0'))
PINNED = {
    'config.json': 'e29df1ee26ad1a513982a81b55d1f8b648126baacc7b509bc918e5c8e912d36a',
    'source-v5.json': 'db00a8b27cd8a8986aa6c55d403e1d396f89ca686f80badd0755b29847925ae8',
    'precision5-reference-tuning-decision.json':
        'f9ee357a9191b858dabe445f2498cb4db4e4a69910f6dbf491cf95c325816c93',
    'run-case.py': '243eef653386a76d1ead4b92542bc84088e882df9011faab5a8154154da9c456',
    'run-case-profiling.py': '722db103e1e51929de7c1951a67ceac24be61cc6e63f33f56e21df3a11cd9362',
    'qualify-inspiral.py': '234ff35cc4d54ab2e9dfd5d475d825a328346183adaf516f46df4653e50450b7',
    'profile-inspiral-torch.py': '2d2b4e0284138e211afaab111a34a0254480238a612c1cb1187be4c3311e67d8',
    'summarize-profiles.py': '7175caade9ab07a39f2db33b346650f73f68e7f6d548bed085418689072d69d9',
    'inputs/bank-compressed-1e5.hdf':
        '161820754117cd51b24a7bb672ea456cf7b2a3808e69690c23235d024b1a2916',
}
FRAME_SHA256 = '580e238054474fd09be900c47217bbcd0497ab84d1756f886647e934352e4865'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def git(source, *args):
    return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()


def source_info(source):
    return dict(commit=git(source, 'rev-parse', 'HEAD'),
                status=git(source, 'status', '--porcelain'))


def snapshot(paths):
    return {name: digest(name) if Path(name).is_file() else None for name in paths}


def case_plan(stage):
    require(stage in ('qualifications', 'measurements'), 'Unknown stage')
    if stage == 'qualifications':
        return [dict(case=f'qual-selected6-{label}-l512', mode='qualify', scheme=scheme)
                for label, scheme in BACKENDS]
    return [dict(case=f'matched-optimized6-{label}-l512-r{rep}', mode='timing', scheme=scheme)
            for rep in (1, 2, 3)
            for label, scheme in BACKENDS[rep - 1:] + BACKENDS[:rep - 1]] + [
        dict(case=f'profile-optimized6-{label}-l512-{mode}', mode=mode, scheme=scheme)
        for label, scheme in BACKENDS for mode in ('cprofile', 'perf')
    ] + [dict(case='profile-optimized6-torch-cuda-l512-torchprofile',
              mode='torchprofile', scheme='torch:cuda:0')]


def cpu_ast_checks(source):
    """Check the unchanged dispatch syntax without importing scientific code."""
    scheme = ast.parse((source / 'pycbc/scheme.py').read_text())
    maps = [n.value for n in scheme.body if isinstance(n, ast.Assign) and
            any(isinstance(t, ast.Name) and t.id == 'scheme_prefix' for t in n.targets)]
    require(len(maps) == 1 and isinstance(maps[0], ast.Dict), 'Cannot verify scheme prefix map')
    prefixes = {k.id: ast.literal_eval(v) for k, v in zip(maps[0].keys, maps[0].values)}
    require(prefixes['CPUScheme'] == 'cpu' and prefixes['TorchScheme'] == 'torch',
            'CPU and Torch scheme dispatch changed')
    backend = ast.parse((source / 'pycbc/fft/backend_cpu.py').read_text())
    mappings = [ast.literal_eval(n.value) for n in backend.body if isinstance(n, ast.Assign) and
                any(isinstance(t, ast.Name) and t.id == '_backend_dict' for t in n.targets)]
    require(len(mappings) == 1 and mappings[0]['mkl'] == 'mkl', 'Normal CPU MKL mapping changed')
    compress = ast.parse((source / 'pycbc/waveform/compress.py').read_text())
    functions = [n for n in compress.body if isinstance(n, ast.FunctionDef) and
                 n.name == 'inline_linear_interp']
    require(len(functions) == 1 and any(
        isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == 'schemed' and
        len(d.args) == 1 and ast.literal_eval(d.args[0]) == 'pycbc.waveform.decompress_'
        for d in functions[0].decorator_list), 'Waveform scheme dispatch changed')
    return dict(DISPATCH)


def validate_pass_receipt(value, commit, label):
    expected = dict(commit=commit, status='')
    require(value['state'] == 'complete' and value['passed'] is True and
            value['returncode'] == 0 and value.get('finished_utc') and
            value['source_info'] == value['source_after'] == expected,
            f'{label} must pass on unchanged v6 source')


def prepare(root, stage, source_manifest_sha256):
    """Read-only preflight; returned hashes are rechecked around every child."""
    root = root.resolve()
    require(re.fullmatch('[0-9a-f]{64}', source_manifest_sha256) is not None,
            'Supply the reviewed source manifest SHA256')
    inputs = {}

    def bind(path, expected=None):
        path = Path(path)
        require(path.is_absolute(), f'Expected absolute input: {path}')
        value = digest(path)
        require(expected is None or value == expected, f'Input hash mismatch: {path}')
        require(str(path) not in inputs or inputs[str(path)] == value, f'Input changed: {path}')
        inputs[str(path)] = value

    def read(name, expected=None):
        path = root / name
        bind(path, expected)
        value = json.loads(path.read_text())
        require(digest(path) == inputs[str(path)], f'Input changed while reading: {path}')
        return value

    def manifest(value, require_after=True):
        hashes = value['input_sha256']
        require(hashes and (not require_after or hashes == value['input_sha256_after']),
                'Receipt has missing or changed input hashes')
        for name, expected in hashes.items():
            bind(name, expected)

    for name, expected in PINNED.items():
        bind(root / name, expected)
    cfg, previous = read('config.json'), read('source-v5.json')
    current = read('source-v6.json', source_manifest_sha256)
    source, old = root / 'source-v6', root / 'source-v5'
    commit = current['commit']
    require(current['schema_version'] == 1 and re.fullmatch('[0-9a-f]{40}', commit) and
            current['parent'] == previous['commit'] == PARENT and
            current['source'] == str(source) and previous['source'] == str(old),
            'Source manifest has wrong revision, parent or path')
    require(source_info(old) == dict(commit=PARENT, status='') and
            source_info(source) == dict(commit=commit, status='') and
            git(source, 'rev-parse', 'HEAD^') == PARENT, 'Source or direct parent is not clean and pinned')
    changed = git(source, 'diff', '--no-renames', '--name-only', PARENT, commit, '--').splitlines()
    require(changed == current['changed_paths'] == list(ALLOWED) and
            set(current['changed_files_sha256']) == set(ALLOWED), 'Unreviewed source delta')
    for name, expected in current['changed_files_sha256'].items():
        blob = subprocess.check_output(['git', '-C', str(source), 'show', f'{commit}:{name}'])
        require(hashlib.sha256(blob).hexdigest() == expected, f'Changed blob differs: {name}')
        bind(source / name, expected)
    cpu_blobs = {}
    for name in CPU_PATHS:
        parent_blob = subprocess.check_output(['git', '-C', str(source), 'show', f'{PARENT}:{name}'])
        current_blob = subprocess.check_output(['git', '-C', str(source), 'show', f'{commit}:{name}'])
        require(parent_blob == current_blob, f'Normal CPU source changed: {name}')
        cpu_blobs[name] = hashlib.sha256(parent_blob).hexdigest()
        bind(source / name, cpu_blobs[name])
    audit = current['normal_cpu_path_audit']
    require(audit['status'] == 'pass' and audit['source_commit'] == commit and
            audit['parent_source_commit'] == PARENT and audit['dispatch'] == cpu_ast_checks(source) and
            isinstance(audit['rationale'], str) and audit['rationale'].strip(),
            'Missing reviewed normal CPU dispatch and side-effect audit')
    native = previous['native_modules_sha256']
    require(len(native) == 11 and current['native_modules_sha256'] == native,
            'Native module manifest differs from v5')
    for tree in (old, source):
        require({str(p.relative_to(tree)) for p in (tree / 'pycbc').rglob('*.so')} == set(native),
                f'Unexpected native module set: {tree}')
        for name, expected in native.items():
            bind(tree / name, expected)
    if 'input_sha256' in current:
        manifest(current)
    decision = read('precision5-reference-tuning-decision.json')
    require(decision['schema_version'] == 1 and decision['source_commit'] == PARENT and
            (decision['selected_segment_length_seconds'], decision['selected_start_pad_seconds'],
             decision['selected_end_pad_seconds']) == (512, 112, 16), 'Reference geometry changed')
    manifest(decision, require_after=False)
    require(len(cfg['input_files']) == 1 and cfg['core'] == 8 and
            cfg['environment']['OMP_NUM_THREADS'] == cfg['environment']['MKL_NUM_THREADS'] == '1',
            'Reference CPU allocation changed')
    bind(cfg['input_files'][0], FRAME_SHA256)
    for name in ('campaign-v6.py', 'launch-campaign-v6.py', 'campaign-v6-contract.md'):
        bind(root / name)
    unit = read('unit-tests-v6.json')
    validate_pass_receipt(unit, commit, 'Unit tests')
    manifest(unit)
    require(str(root / 'source-v6.json') in unit['input_sha256'], 'Unit tests do not bind source manifest')
    expected_environment = dict(cfg['environment'], PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1')
    require(unit['environment'] == expected_environment and sys.executable in unit['command'],
            'Campaign must use the qualified Python environment')
    bind(root / 'unit-tests-v6.log', unit['log_sha256'])
    if stage == 'measurements':
        ledger = read('campaign-v6-qualifications.status.json')
        require(ledger['state'] == 'complete' and ledger['returncode'] == 0 and
                ledger['source_info'] == ledger['source_after'] == dict(commit=commit, status='') and
                ledger['current'] is None and ledger['completed'] == case_plan('qualifications') and
                ledger.get('finished_utc'), 'Qualification stage is incomplete')
        manifest(ledger)
        require(ledger['output_sha256'], 'Missing qualification outputs')
        for name, expected in ledger['output_sha256'].items():
            bind(name, expected)
        for case in case_plan('qualifications'):
            hashes = validate_case(dict(root=root, source=source, commit=commit, config=cfg), case)
            require(all(ledger['output_sha256'].get(name) == expected
                        for name, expected in hashes.items()), 'Qualification outputs are unbound')
        science = read('scientific-validation-v6.json')
        validate_pass_receipt(science, commit, 'Scientific validation')
        require(science['schema_version'] == 1 and set(science['checks']) == SCIENCE_CHECKS and
                all(value is True for value in science['checks'].values()) and
                set(science['evidence']) == SCIENCE_CHECKS, 'Scientific coverage is incomplete')
        manifest(science)
        required_science = {root / 'source-v6.json', root / 'unit-tests-v6.json',
                            root / 'campaign-v6-qualifications.status.json'}
        for case in case_plan('qualifications'):
            required_science.update(root / 'runs' / case['case'] / name
                                    for name in ('qualification.json', 'triggers.hdf'))
        for evidence in science['evidence'].values():
            bind(evidence['path'], evidence['sha256'])
            required_science.add(Path(evidence['path']))
        require({str(p) for p in required_science} <= science['input_sha256'].keys(),
                'Science does not bind all prerequisite evidence')
    plan = case_plan(stage)
    prefix = f'campaign-v6-{stage}'
    outputs = [root / f'{prefix}.status.json', root / f'{prefix}.status.tmp']
    outputs += [root / 'runs' / case['case'] for case in plan]
    outputs += [root / f"campaign-v6-{case['case']}.log" for case in plan]
    if stage == 'measurements':
        outputs += [root / 'profiles-v6.json', root / 'campaign-v6-summary.log']
        outputs += [root / f"campaign-v6-{case['case']}-perf-export.stderr.log"
                    for case in plan if case['mode'] == 'perf']
    require(not any(p.exists() for p in outputs), 'Stage output or run directory already exists')
    for path in root.glob('*.status.json'):
        require(json.loads(path.read_text()).get('state') != 'running', f'Another job is running: {path}')
    require(snapshot(inputs) == inputs, 'Inputs changed during preparation')
    return dict(root=root, stage=stage, source=source, commit=commit, config=cfg, inputs=inputs,
                plan=plan, normal_cpu_equivalence=dict(dispatch=DISPATCH, unchanged_sha256=cpu_blobs,
                native_modules_sha256=native, reviewed_audit=audit, tuning_source_commit=PARENT))


def validate_case(context, case):
    root, commit = context['root'], context['commit']
    path = root / 'runs' / case['case']
    receipt = json.loads((path / 'receipt.json').read_text())
    require(receipt['state'] == 'complete' and receipt['returncode'] == 0 and
            receipt['source_info'] == dict(commit=commit, status='', tracked_diff='') and
            receipt['source_status_after'] == '' and receipt.get('finished_utc') and
            receipt['source'] == str(context['source']) and
            all(receipt[key] == value for key, value in case.items()) and
            (receipt['segment_length'], receipt['start_pad'], receipt['end_pad']) == (512, 112, 16) and
            receipt['input_sha256'] == receipt['input_sha256_after'] == snapshot(receipt['input_sha256']),
            f'Incomplete or changed case: {case["case"]}')
    require(receipt['trigger_sha256'] == digest(path / 'triggers.hdf'), 'Trigger output changed')
    expected_cli = [str(context['source'] / 'bin/pycbc_inspiral'), *context['config']['common_args'],
                    '--bank-file', str(root / 'inputs/bank-compressed-1e5.hdf'),
                    '--processing-scheme', case['scheme'], '--segment-length', '512',
                    '--segment-start-pad', '112', '--segment-end-pad', '16',
                    '--output', str(path / 'triggers.hdf')]
    expected_environment = dict(context['config']['environment'], PYTHONPATH=str(context['source']),
                                PYTHONDONTWRITEBYTECODE='1')
    require(receipt['executable_cli'] == expected_cli and receipt['environment'] == expected_environment,
            'Case command or environment differs from the frozen workload')
    if case['mode'] == 'qualify':
        qualification = json.loads((path / 'qualification.json').read_text())
        require(qualification['status'] == 'success' and qualification['executable_exit_code'] == 0 and
                qualification['source_root'] == str(context['source']) and qualification['checks'] and
                all(value is True for value in qualification['checks'].values()), 'Qualification failed')
    elif case['mode'] == 'torchprofile':
        trace = json.loads((path / 'torch-profile/receipt.json').read_text())
        require(trace['status'] == 'success' and trace['wrapper_exit_code'] == 0 and
                trace['executable_exit_code'] == 0 and trace['source_root'] == str(context['source']),
                'Torch profiler wrapper failed')
        for key in ('trace', 'key_averages'):
            require(trace[key]['sha256'] == digest(trace[key]['path']), 'Torch trace output changed')
    elif case['mode'] in ('cprofile', 'perf'):
        raw = path / ('profile.pstats' if case['mode'] == 'cprofile' else 'perf.data')
        require(raw.is_file() and raw.stat().st_size > 0, 'Raw profile output is missing')
    return {str(p): digest(p) for p in sorted(path.rglob('*')) if p.is_file()}


def run_stage(context):
    root, source, inputs = (context[key] for key in ('root', 'source', 'inputs'))
    before = dict(commit=context['commit'], status='')
    env_fixed = dict(context['config']['environment'], PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1')
    env = dict(os.environ, **env_fixed)
    status = root / f"campaign-v6-{context['stage']}.status.json"
    record = dict(schema_version=1, state='running', returncode=None, stage=context['stage'],
                  pid=os.getpid(), host=os.uname().nodename, cwd=str(root), started_utc=utc(),
                  source_info=before, environment=env_fixed, input_sha256=inputs,
                  normal_cpu_equivalence=context['normal_cpu_equivalence'],
                  plan=context['plan'], completed=[], commands=[], current=None,
                  child_pid=None, output_sha256={})

    def save(first=False):
        temporary = status if first else status.with_suffix('.tmp')
        with temporary.open('x') as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write('\n')
        if not first:
            temporary.replace(status)

    def unchanged():
        require(source_info(source) == before, 'Campaign source changed')
        require(snapshot(inputs) == inputs, 'Campaign input changed')
        require(snapshot(record['output_sha256']) == record['output_sha256'], 'Completed output changed')

    def execute(command, log, stderr_path=None):
        unchanged()
        record.update(current=command, child_pid=None)
        save()
        with log.open('x') as stdout:
            stderr = stderr_path.open('x') if stderr_path else subprocess.STDOUT
            try:
                child = subprocess.Popen(command, cwd=root, env=env, stdout=stdout, stderr=stderr)
                record['child_pid'] = child.pid
                save()
                code = child.wait()
            finally:
                if stderr_path:
                    stderr.close()
        record['commands'].append(dict(command=command, returncode=code, log=str(log),
                                       stderr=str(stderr_path) if stderr_path else None))
        record.update(current=None, child_pid=None)
        save()
        require(code == 0, f'Child failed with return code {code}: {command}')
        unchanged()
        record['output_sha256'][str(log)] = digest(log)
        if stderr_path:
            record['output_sha256'][str(stderr_path)] = digest(stderr_path)

    save(first=True)
    try:
        for case in context['plan']:
            runner = 'run-case.py' if case['mode'] in ('qualify', 'timing') else 'run-case-profiling.py'
            command = [sys.executable, '-u', str(root / runner), '--config', str(root / 'config.json'),
                       '--source', str(source), '--bank', str(root / 'inputs/bank-compressed-1e5.hdf'),
                       '--case', case['case'], '--mode', case['mode'], '--scheme', case['scheme'],
                       '--segment-length', '512', '--start-pad', '112', '--end-pad', '16']
            execute(command, root / f"campaign-v6-{case['case']}.log")
            record['output_sha256'].update(validate_case(context, case))
            record['completed'].append(case)
            save()
        if context['stage'] == 'measurements':
            profiles = []
            for case in context['plan']:
                path = root / 'runs' / case['case']
                if case['mode'] in ('cprofile', 'perf'):
                    profiles.append(path)
                if case['mode'] == 'perf':
                    execute(['perf', 'report', '--stdio', '--no-children', '--call-graph', 'none',
                             '--percent-limit', '0.1', '--show-nr-samples', '--show-total-period',
                             '--sort', 'comm,dso,symbol', '-i', str(path / 'perf.data')],
                            path / 'perf-report.txt',
                            root / f"campaign-v6-{case['case']}-perf-export.stderr.log")
            summary = root / 'profiles-v6.json'
            execute([sys.executable, '-u', str(root / 'summarize-profiles.py'), '--output', str(summary),
                     *map(str, profiles)], root / 'campaign-v6-summary.log')
            record['output_sha256'][str(summary)] = digest(summary)
        unchanged()
        record.update(state='complete', returncode=0)
    except Exception as error:
        record.update(state='failed', returncode=1, error=f'{type(error).__name__}: {error}')
    finally:
        record.update(current=None, child_pid=None, finished_utc=utc(),
                      source_after=source_info(source), input_sha256_after=snapshot(inputs))
        if record['source_after'] != before or record['input_sha256_after'] != inputs:
            record.update(state='invalid-input-mutation', returncode=1)
        save()
    print(json.dumps(dict(status=str(status), state=record['state'], returncode=record['returncode'],
                         completed=len(record['completed']), source_commit=context['commit'])))
    return record['returncode']


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('qualifications', 'measurements'))
    parser.add_argument('--source-manifest-sha256')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args()
    if not args.describe and (not args.stage or not args.source_manifest_sha256):
        parser.error('--stage and --source-manifest-sha256 are required')
    return args


def main():
    args = parse_args()
    if args.describe:
        print(json.dumps({stage: case_plan(stage) for stage in ('qualifications', 'measurements')}, indent=2))
        return 0
    context = prepare(Path(__file__).resolve().parent, args.stage, args.source_manifest_sha256)
    if args.check:
        print(json.dumps(dict(status='ready', stage=args.stage, source_commit=context['commit'],
                             cases=context['plan'], input_sha256=context['inputs'])))
        return 0
    return run_stage(context)


if __name__ == '__main__':
    sys.exit(main())
