#!/usr/bin/env python3
"""Preflight and detach a single frozen v6 campaign stage."""
import argparse
import datetime
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('qualifications', 'measurements'), required=True)
    parser.add_argument('--source-manifest-sha256', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    harness = runpy.run_path(str(root / 'campaign-v6.py'), run_name='campaign_v6_library')
    context = harness['prepare'](root, args.stage, args.source_manifest_sha256)
    prefix = f'campaign-v6-{args.stage}-launch'
    log, receipt = root / f'{prefix}.log', root / f'{prefix}.json'
    temporary = receipt.with_suffix('.tmp')
    harness['require'](not any(path.exists() for path in (log, receipt, temporary)),
                       'Launch output already exists')
    command = [sys.executable, '-u', str(root / 'campaign-v6.py'), '--stage', args.stage,
               '--source-manifest-sha256', args.source_manifest_sha256]
    record = dict(state='launching', host=os.uname().nodename, cwd=str(root), command=command,
                  pid=None, log=str(log), source_commit=context['commit'],
                  source_manifest_sha256=args.source_manifest_sha256,
                  input_sha256=context['inputs'], started_utc=harness['utc'](),
                  expected_next_check_seconds=60)
    # Reserve the receipt before detaching so simultaneous launchers cannot both start.
    with receipt.open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    try:
        with log.open('x') as stream:
            child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        record.update(state='launched', pid=child.pid, stop=f'kill -TERM -- -{child.pid}',
                      expected_next_check_utc=(datetime.datetime.now(datetime.timezone.utc) +
                                               datetime.timedelta(seconds=60)).isoformat())
    except Exception as error:
        record.update(state='launch-failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        with temporary.open('x') as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write('\n')
        temporary.replace(receipt)
    print(json.dumps(record))


if __name__ == '__main__':
    main()
