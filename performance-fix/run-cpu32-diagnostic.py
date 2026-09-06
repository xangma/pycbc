#!/usr/bin/env python3
"""Serial B32/T4 CPU-native diagnostics after timing; --plan-only never writes."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
ROOT = Path('/home/xangma/pycbc-torch-performance-fix-20260906')
PYTHON = '/home/xangma/pycbc-torch-split-20260905/venv/bin/python'
SOURCES = {'baseline': 'dfd42bf76766cadca0eecf609a1eaeac73534676',
           'candidate-v2': '0d00581251e642a5d6b56b2497a9adad93069e6b'}
PINNED = {'run-postchecks.py': '6ed676f8aeb7b478f5a5de27c7fa6b697129c5efe198eb26a03606f8ab4cf22c',
          'run-accelerator-refinement.py': 'a69ef8762c585c2040e5d6a3e91e16b371c606a2a63353b8ee57fdc83844d795',
          'profile-live.py': '2023c5f9d91f3bfe9eeb34dae3fa734e3094be0ec67495cba06673404a01750b'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def interrupted(number, _frame):
    raise InterruptedError(f'Received signal {number}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    root, out = args.root.absolute(), args.root.absolute() / 'cpu32-t4-diagnostic'
    commands = {key: ['taskset', '-c', '8-11', PYTHON, str(root / 'profile-live.py'),
                     '--root', str(root / key), '--out', str(out / key),
                     '--route', 'torch_cpu_native', '--batch', '32', '--threads', '4'] for key in SOURCES}
    if args.plan_only:
        print(json.dumps(dict(output=str(out), sources=SOURCES, commands=commands,
                              prerequisite=str(root / 'accelerator-refinement/status.json'), scope=__doc__), indent=2))
        return
    prerequisite = root / 'accelerator-refinement/status.json'
    completed = json.loads(prerequisite.read_text())
    require(completed['state'] == 'complete' and completed.get('finished_utc')
            and len(completed['completed']) == 36, 'Accelerator timings are incomplete')
    require(not out.exists(), 'Diagnostic output already exists; inspect before retry')
    for name, expected in PINNED.items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, 'Helper/profile changed: ' + name)
    sys.dont_write_bytecode = True
    helper = load(root / 'run-postchecks.py', 'cpu32_postcheck_helpers')
    guard = load(root / 'run-accelerator-refinement.py', 'cpu32_source_helpers')
    before = {key: guard.snapshot(root / key, head) for key, head in SOURCES.items()}
    frozen_path = root / 'accelerator-refinement/sources-after.json'
    frozen = json.loads(frozen_path.read_text())
    require(all(before[key] == frozen[key] for key in SOURCES), 'Source differs from completed timing receipt')
    for relative in ('tools/bench_production_live_batch.py', 'tools/benchmark_artifact.py'):
        require(helper.digest(root / 'baseline' / relative) == helper.digest(root / 'candidate-v2' / relative), 'Harness differs: ' + relative)
    inputs = {str(p): helper.digest(p) for p in [prerequisite, frozen_path, Path(__file__), *(root / name for name in PINNED)]}
    out.mkdir()
    status = dict(state='running', scope='Diagnostic attribution only; instrumented profiles are not throughput evidence',
                  host=socket.gethostname(), pid=os.getpid(), cwd=str(Path.cwd()), command=[sys.executable, *sys.argv],
                  started_utc=helper.utc(), stop_command=f'kill -TERM {os.getpid()}', expected_next_check_seconds=30,
                  inputs_sha256=inputs, commands=[])
    helper.save(out / 'sources-before.json', before)
    helper.save(out / 'status.json', status)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for key, command in commands.items():
            environment = helper.load_harness(root / key, 'cpu32_' + key.replace('-', '_')).route_environment(
                'torch_cpu_native', helper.base_environment(root / key, 4))
            record = dict(source=key, command=command, cwd=str(root / key), host=status['host'],
                          log=str(out / (key + '.log')), environment=helper.recorded_environment(environment), started_utc=helper.utc())
            status['commands'].append(record)
            process = None
            try:
                with Path(record['log']).open('x') as log:
                    process = subprocess.Popen(command, cwd=root / key, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    record.update(pid=process.pid, stop_command=f'kill -TERM -- -{process.pid}')
                    helper.save(out / 'status.json', status)
                    print(json.dumps(record), flush=True)
                    process.wait(timeout=3600)
            except BaseException:
                if process is not None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise
            finally:
                record.update(returncode=process.poll() if process else None, finished_utc=helper.utc())
                helper.save(out / (key + '-command.json'), record)
            require(record['returncode'] == 0, 'Profile failed: ' + key)
            files = ('python.pstats', 'python.txt', 'operators.txt', 'trace.json', 'status.json', 'worker.log')
            require(all((out / key / name).is_file() and (out / key / name).stat().st_size for name in files), 'Incomplete profiles: ' + key)
            profile = json.loads((out / key / 'status.json').read_text())
            require(profile['completed'] and profile['calls'] >= 8, 'Instrumented calls missing: ' + key)
            helper.save(out / (key + '-profiles-sha256.json'), {name: helper.digest(out / key / name) for name in files})
        status['state'] = 'complete'
    except BaseException as exc:
        status.update(state='failed', error=repr(exc))
        raise
    finally:
        try:
            after = {key: guard.snapshot(root / key, head) for key, head in SOURCES.items()}
            helper.save(out / 'sources-after.json', after)
            require(before == after and all(helper.digest(Path(p)) == h for p, h in inputs.items()), 'Source or input changed')
        except BaseException as exc:
            status.update(state='failed', source_audit_error=repr(exc))
            raise
        finally:
            status['finished_utc'] = helper.utc()
            helper.save(out / 'status.json', status)


if __name__ == '__main__':
    main()
