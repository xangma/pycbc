#!/usr/bin/env python3
"""Run one frozen inspiral workload; every invocation has its own directory."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--mode', choices=['timing', 'qualify', 'cprofile', 'perf', 'filter-cprofile', 'filter-perf', 'filter-timing'],
                        default='timing')
    parser.add_argument('--scheme', default='cpu:1')
    parser.add_argument('--segment-length', type=int, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--start-pad', type=int, default=112)
    parser.add_argument('--end-pad', type=int, default=16)
    args = parser.parse_args()
    if Path(args.case).name != args.case:
        parser.error('--case must be a directory basename')
    cfg = json.loads(args.config.read_text())
    root = args.config.resolve().parent
    out = root / 'runs' / args.case
    out.mkdir(parents=True, exist_ok=False)
    source = args.source.resolve()
    executable = source / 'bin' / 'pycbc_inspiral'
    env = os.environ.copy()
    fixed_env = dict(cfg['environment'], PYTHONPATH=str(source),
                     PYTHONDONTWRITEBYTECODE='1')
    env.update(fixed_env)
    cli = [str(executable), *cfg['common_args'],
           '--bank-file', str(args.bank.resolve()),
           '--processing-scheme', args.scheme,
           '--segment-length', str(args.segment_length),
           '--segment-start-pad', str(args.start_pad),
           '--segment-end-pad', str(args.end_pad),
           '--output', str(out / 'triggers.hdf')]
    command = [sys.executable, *cli]
    if args.mode == 'qualify':
        command = [sys.executable, str(root / 'qualify-inspiral.py'),
                   '--receipt', str(out / 'qualification.json'), '--', *cli]
    elif args.mode == 'cprofile':
        command = [sys.executable, '-m', 'cProfile', '-o',
                   str(out / 'profile.pstats'), *cli]
    elif args.mode == 'perf':
        command = ['perf', 'record', '-F', '199', '-e', 'cycles:u',
                   '--call-graph', 'dwarf,16384', '-o', str(out / 'perf.data'),
                   '--', *command]
    elif args.mode in ('filter-cprofile', 'filter-perf', 'filter-timing'):
        command = [sys.executable, str(root / 'profile-filtering.py'),
                   '--receipt', str(out / 'filtering-window.json')]
        if args.mode == 'filter-cprofile':
            command += ['--pstats', str(out / 'filtering.pstats')]
        command += ['--', *cli]
        if args.mode == 'filter-perf':
            command = ['perf', 'record', '-F', '199', '-e', 'cycles:u',
                       '--clockid', 'mono', '--call-graph', 'dwarf,16384',
                       '-o', str(out / 'perf.data'), '--', *command]
    command = ['taskset', '-c', str(cfg['core']), *command]
    command = ['/usr/bin/time', '-v', '-o', str(out / 'time.txt'), *command]
    source_info = {
        'commit': subprocess.check_output(
            ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
        'status': subprocess.check_output(
            ['git', '-C', str(source), 'status', '--porcelain'], text=True),
        'tracked_diff': subprocess.check_output(
            ['git', '-C', str(source), 'diff', 'HEAD', '--'], text=True),
    }
    if source_info['status']:
        raise RuntimeError('Source must be clean before running')
    inputs = [args.config.resolve(), args.bank.resolve(), executable,
              Path(__file__).resolve(), *map(Path, cfg['input_files'])]
    if args.mode == 'qualify':
        inputs.append(root / 'qualify-inspiral.py')
    if args.mode in ('filter-cprofile', 'filter-perf', 'filter-timing'):
        inputs.append(root / 'profile-filtering.py')
    record = dict(case=args.case, mode=args.mode, scheme=args.scheme,
                  segment_length=args.segment_length, start_pad=args.start_pad,
                  end_pad=args.end_pad, source=str(source), source_info=source_info,
                  command=command, executable_cli=cli, environment=fixed_env,
                  hostname=os.uname().nodename, cwd=str(root), parent_pid=os.getpid(),
                  input_sha256={str(p): digest(p) for p in inputs},
                  started_utc=utc(), state='running')
    receipt = out / 'receipt.json'

    def save():
        temporary = receipt.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(receipt)

    save()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    with open(out / 'stdout.log', 'w') as stdout, open(out / 'stderr.log', 'w') as stderr:
        child = subprocess.Popen(command, cwd=root, env=env,
                                 stdout=stdout, stderr=stderr)
        record['child_pid'] = child.pid
        save()
        code = child.wait()
    elapsed = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    record.update(state='complete' if code == 0 else 'failed', returncode=code,
                  finished_utc=utc(), elapsed_wall_seconds=elapsed,
                  child_user_cpu_seconds=after.ru_utime - before.ru_utime,
                  child_system_cpu_seconds=after.ru_stime - before.ru_stime,
                  peak_child_rss_kib=after.ru_maxrss)
    record['input_sha256_after'] = {str(p): digest(p) for p in inputs}
    record['source_status_after'] = subprocess.check_output(
        ['git', '-C', str(source), 'status', '--porcelain'], text=True)
    if record['input_sha256_after'] != record['input_sha256'] or record['source_status_after']:
        record['state'] = 'invalid-input-mutation'
        code = 1
    if (out / 'triggers.hdf').exists():
        record['trigger_sha256'] = digest(out / 'triggers.hdf')
    save()
    print(json.dumps(record))
    return code


if __name__ == '__main__':
    sys.exit(main())
