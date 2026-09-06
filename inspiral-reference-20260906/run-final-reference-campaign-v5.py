#!/usr/bin/env python3
"""Run the frozen final campaign once, after the CPU selector has completed."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


COMMIT = '837f38d493420043e45fb1ad210a0ccf68bacbaa'
PREFIX = 'precision5-final-campaign'
STAGES = (
    ('qualifications', 'precision5-final-qualifications-plan.json', 'run-series-v5.py'),
    ('reference-profiles', 'precision5-reference-profiles-plan.json', 'run-series-profiling-v5.py'),
    ('matched-timings', 'precision5-matched-backends-plan.json', 'run-series-v5.py'),
    ('torch-profiles', 'precision5-torch-profiles-plan.json', 'run-series-profiling-v5.py'),
)


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


def snapshot(inputs):
    return {name: digest(name) if Path(name).is_file() else None for name in inputs}


def source_info(source):
    return {key: subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()
            for key, args in (('commit', ['rev-parse', 'HEAD']), ('status', ['status', '--porcelain']))}


def prepare(root):
    inputs = {}

    def bind(path, expected=None):
        path = Path(path)
        value = digest(path)
        require(expected is None or value == expected, f'Input hash mismatch: {path}')
        require(str(path) not in inputs or inputs[str(path)] == value, f'Input changed: {path}')
        inputs[str(path)] = value

    def read(name):
        path = root / name
        bind(path)
        value = json.loads(path.read_text())
        require(digest(path) == inputs[str(path)], f'Input changed while reading: {path}')
        return value

    def manifest(value):
        before = value['input_sha256']
        require(before and before == value.get('input_sha256_after', before), 'Changed receipt inputs')
        for name, expected in before.items():
            require(Path(name).is_absolute(), f'Nonabsolute recorded input: {name}')
            bind(name, expected)

    cfg, source = read('config.json'), read('source-v5.json')
    require(source['source'] == str(root / 'source-v5') and source['commit'] == COMMIT,
            'Wrong final source')
    unit = read('unit-tests-v5.json')
    require(unit['state'] == 'complete' and unit['passed'] is True and unit['returncode'] == 0 and
            unit['source_info'] == unit['source_after'] == dict(commit=COMMIT, status='') and
            unit.get('finished_utc') and unit['input_sha256'] == unit['input_sha256_after'],
            'Final unit tests have not passed on unchanged source')
    manifest(unit)
    bind(root / 'unit-tests-v5.log', unit['log_sha256'])
    campaign = read('precision5-reference-campaign.status.json')
    require(campaign['state'] == 'complete' and campaign['returncode'] == 0 and
            campaign['current'] is None and campaign.get('finished_utc') and
            campaign['input_sha256'] == campaign['input_sha256_after'], 'CPU campaign is incomplete')
    manifest(campaign)
    decision = read('precision5-reference-tuning-decision.json')
    length = decision['selected_segment_length_seconds']
    require(decision['schema_version'] == 1 and decision['source_commit'] == COMMIT and
            type(length) is int and length in (256, 512, 1024) and
            decision['selected_start_pad_seconds'] == 112 and decision['selected_end_pad_seconds'] == 16,
            'Wrong final tuning decision')
    manifest(decision)
    require(set(decision['plan_sha256']) == {name for _, name, _ in STAGES}, 'Wrong decision plan set')
    backends = [('cpu', 'cpu:1'), ('torch-cpu', 'torch:cpu:1'), ('torch-cuda', 'torch:cuda:0')]

    def case(name, mode, scheme):
        return ['--case', name, '--mode', mode, '--scheme', scheme, '--segment-length', str(length)]

    expected = [
        [case(f'qual-selected5-{label}-l{length}', 'qualify', scheme) for label, scheme in backends],
        [case(f'reference-precision5-cpu-l{length}-{mode}', mode, 'cpu:1') for mode in ('cprofile', 'perf')],
        [case(f'matched-precision5-{label}-l{length}-r{rep}', 'timing', scheme)
         for rep in (1, 2, 3) for label, scheme in backends[rep - 1:] + backends[:rep - 1]],
        [case(f'profile-precision5-torch-{device}-l{length}-{mode}', mode, scheme)
         for device, scheme in [('cpu', 'torch:cpu:1'), ('cuda', 'torch:cuda:0')]
         for mode in ('cprofile', 'perf')] +
        [case(f'profile-precision5-torch-cuda-l{length}-torchprofile', 'torchprofile', 'torch:cuda:0')],
    ]
    plans = {}
    for (_, name, _), wanted in zip(STAGES, expected):
        bind(root / name, decision['plan_sha256'][name])
        plans[name] = read(name)
        require(plans[name] == wanted, f'Unexpected case order, mode, scheme or geometry: {name}')
    for name in (Path(__file__).name, 'run-series-v5.py', 'run-series-profiling-v5.py',
                 'run-case.py', 'run-case-profiling.py', 'qualify-inspiral.py',
                 'profile-inspiral-torch.py', 'summarize-profiles.py', 'select-precision-reference-v5.py'):
        bind(root / name)
    for name in cfg['input_files']:
        bind(name)
    bind(root / 'inputs/bank-compressed-1e5.hdf')
    profiles = [root / 'runs' / args[1] for plan in plans.values() for args in plan
                if args[3] in ('cprofile', 'perf')]
    require(len(profiles) == 6, 'Expected six cProfile/perf cases')
    outputs = [root / f'{PREFIX}{suffix}' for suffix in ('.status.json', '.status.tmp', '.log')]
    outputs += [root / 'profiles-precision5.json', root / f'{PREFIX}-summary.log']
    for stage, name, _ in STAGES:
        outputs += [(root / name).with_suffix('.status.json'), (root / name).with_suffix('.status.tmp'),
                    root / f'{PREFIX}-{stage}.log']
        outputs += [root / 'runs' / args[1] for args in plans[name]]
    outputs += [root / f'{PREFIX}-{path.name}.stderr.log' for path in profiles if path.name.endswith('-perf')]
    require(not any(path.exists() for path in outputs), 'Final output or run directory already exists')
    for path in root.glob('*.status.json'):
        require(json.loads(path.read_text()).get('state') != 'running', f'Another campaign is running: {path}')
    require(snapshot(inputs) == inputs, 'Inputs changed during preparation')
    return cfg, inputs, plans, profiles


def main():
    root = Path(__file__).resolve().parent
    cfg, inputs, plans, profiles = prepare(root)
    source = root / 'source-v5'
    before = source_info(source)
    require(before == dict(commit=COMMIT, status=''), 'Final source must be clean at the frozen commit')
    fixed_env = dict(cfg['environment'], PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1')
    env = dict(os.environ, **fixed_env)
    status = root / f'{PREFIX}.status.json'
    record = dict(state='running', returncode=None, pid=os.getpid(), host=os.uname().nodename,
                  cwd=str(root), started_utc=utc(), source_info=before, environment=fixed_env,
                  input_sha256=inputs, completed=[], current=None, output_sha256={})

    def save():
        temporary = status.with_suffix('.tmp')
        with temporary.open('x') as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write('\n')
        temporary.replace(status)

    def unchanged():
        require(snapshot(inputs) == inputs, 'Campaign inputs changed')
        require(snapshot(record['output_sha256']) == record['output_sha256'], 'Completed output changed')
        require(source_info(source) == before, 'Campaign source changed')

    def execute(name, command, stdout_path, stderr_path=None):
        unchanged()
        record.update(current_stage=name, current=command, child_pid=None)
        save()
        with stdout_path.open('x') as stdout:
            stderr = stderr_path.open('x') if stderr_path else subprocess.STDOUT
            try:
                child = subprocess.Popen(command, cwd=root, env=env, stdout=stdout, stderr=stderr)
                record['child_pid'] = child.pid
                save()
                code = child.wait()
            finally:
                if stderr_path:
                    stderr.close()
        record['completed'].append(dict(name=name, command=command, returncode=code,
                                        stdout=str(stdout_path), stderr=str(stderr_path) if stderr_path else None))
        save()
        require(code == 0, f'Stage {name} failed with return code {code}')
        unchanged()

    with (root / f'{PREFIX}.log').open('x') as log:
        save()
        try:
            for stage, name, runner in STAGES:
                execute(stage, [sys.executable, '-u', str(root / runner), str(root / name)],
                        root / f'{PREFIX}-{stage}.log')
                ledger = (root / name).with_suffix('.status.json')
                value = json.loads(ledger.read_text())
                require(value['state'] == 'complete' and value['returncode'] == 0 and
                        value['current'] is None and value['completed'] == plans[name] and
                        value['plan_sha256'] == inputs[str(root / name)], f'Incomplete stage ledger: {stage}')
                record['output_sha256'][str(ledger)] = digest(ledger)
                for args in plans[name]:
                    receipt_path = root / 'runs' / args[1] / 'receipt.json'
                    receipt = json.loads(receipt_path.read_text())
                    require(receipt['state'] == 'complete' and receipt['returncode'] == 0 and
                            receipt['source_info'] == dict(commit=COMMIT, status='', tracked_diff='') and
                            receipt['source_status_after'] == '' and
                            receipt['input_sha256'] == receipt['input_sha256_after'],
                            f'Incomplete or changed run: {args[1]}')
                    record['output_sha256'][str(receipt_path)] = digest(receipt_path)
                log.write(f'{utc()} {stage} complete\n')
                log.flush()
                save()
            for path in profiles:
                for name in ('receipt.json', 'triggers.hdf',
                             'perf.data' if path.name.endswith('-perf') else 'profile.pstats'):
                    record['output_sha256'][str(path / name)] = digest(path / name)
            for path in profiles:
                if path.name.endswith('-perf'):
                    output = path / 'perf-report.txt'
                    execute('perf-export-' + path.name,
                            ['perf', 'report', '--stdio', '--no-children', '--call-graph', 'none',
                             '--percent-limit', '0.1', '--show-nr-samples', '--show-total-period',
                             '--sort', 'comm,dso,symbol', '-i', str(path / 'perf.data')],
                            output, root / f'{PREFIX}-{path.name}.stderr.log')
                    record['output_sha256'][str(output)] = digest(output)
            summary = root / 'profiles-precision5.json'
            execute('summary', [sys.executable, '-u', str(root / 'summarize-profiles.py'),
                               '--output', str(summary), *map(str, profiles)], root / f'{PREFIX}-summary.log')
            record['output_sha256'][str(summary)] = digest(summary)
            record.update(state='complete', returncode=0)
        except Exception as error:
            record.update(state='failed', returncode=1, error=f'{type(error).__name__}: {error}')
            log.write(record['error'] + '\n')
        finally:
            record.update(input_sha256_after=snapshot(inputs), source_after=source_info(source),
                          current=None, child_pid=None, finished_utc=utc())
            if record['input_sha256_after'] != inputs or record['source_after'] != before:
                record.update(state='invalid-input-mutation', returncode=1)
            save()
    print(json.dumps(record))
    return record['returncode']


if __name__ == '__main__':
    sys.exit(main())
