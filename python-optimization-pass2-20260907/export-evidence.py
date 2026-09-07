"""Bounded, locked export of completed pass-two evidence; no scientific jobs."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tarfile
import time

ROOT = Path('/home/xangma/pycbc-torch-python-optimization-pass2-20260907')
EXPORT = ROOT.with_name(ROOT.name + '-export')
SKIP = {'source-baseline', 'source-candidate', '__pycache__', '.pytest_cache'}


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(name, value):
    path = EXPORT / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def files():
    result = {}
    for directory, folders, names in os.walk(ROOT):
        folders[:] = sorted(n for n in folders if n not in SKIP)
        for name in sorted(names):
            path = Path(directory) / name
            assert path.is_file() and not path.is_symlink(), path
            result[str(path.relative_to(ROOT))] = digest(path)
    return result


def main():
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))
    assert os.getpid() == os.getpgrp()
    assert Path(__file__).resolve() == EXPORT / 'export-evidence.py'
    spec = importlib.util.spec_from_file_location('campaign', ROOT / 'optimize-campaign.py')
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    compare = campaign.load('compare', 'compare-candidate.py')
    gates = campaign.load('gates', 'profile-gates.py')
    state = dict(state='running', pid=os.getpid(), pgid=os.getpgrp(),
                 started=time.time(), host='len', cwd=str(EXPORT), timeout_seconds=120,
                 stop_command=f'kill -TERM {os.getpid()}', next_check_seconds=30)
    save('export-status.json', state)

    def stop(signum, frame):
        raise TimeoutError(f'Export stopped by signal {signum}')

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, stop)
    signal.setitimer(signal.ITIMER_REAL, 120)
    try:
        with campaign.LOCK.open('r') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pins = campaign.pins()
            release = campaign.gates()
            campaign.api_review()
            groups = set(compare.read(ROOT / 'source-staging-terminal-audit.json')['owned_groups'])
            for phase in ('api', 'executable'):
                status = compare.read(ROOT / (phase + '-status.json'))
                launch = compare.read(ROOT / (phase + '-launch.json'))
                assert status['state'] == 'complete' and status['finished']
                assert status['child_pid'] is None and status['source_unchanged']
                assert status['pid'] == status['pgid'] == launch['pid'] == launch['pgid']
                assert status['result_sha256'] == digest(ROOT / (phase + '-summary.json'))
                groups.update((status['pid'], status['pgid']))
            stages = list((ROOT / 'stages').glob('*.json'))
            for path in stages:
                stage = compare.read(path)
                assert stage['state'] == 'complete' and stage['returncode'] == 0
                groups.update((stage['pid'], stage['pgid']))
            assert not [p for p in gates.process_table()
                        if p['pid'] in groups or p['pgid'] in groups]
            before = compare.take_snapshot(ROOT / 'source-baseline', ROOT / 'source-candidate')
            compare.validate_window(compare.read(ROOT / 'executable-after.json'), before)
            pinned = files()
            manifest = dict(campaign_root=str(ROOT), files=pinned, source_snapshot=before,
                            scientific_owned_groups=sorted(groups), helper_sha256=pins,
                            profile_release_sha256=release['release_sha256'])
            save('evidence-manifest.json', manifest)
            archive = EXPORT / 'evidence.tar.gz'
            with tarfile.open(archive, 'x:gz') as output:
                output.add(EXPORT / 'evidence-manifest.json', arcname='evidence-manifest.json')
                for name in sorted(pinned):
                    output.add(ROOT / name, arcname=name, recursive=False)
            assert files() == pinned
            assert campaign.pins() == pins
            after = compare.take_snapshot(ROOT / 'source-baseline', ROOT / 'source-candidate')
            compare.validate_window(before, after)
            state.update(state='complete', finished=time.time(), files=len(pinned),
                         archive=str(archive), archive_bytes=archive.stat().st_size,
                         archive_sha256=digest(archive), all_pins_unchanged=True,
                         manifest_sha256=digest(EXPORT / 'evidence-manifest.json'),
                         scientific_owned_groups=sorted(groups))
            save('export-status.json', state)
    except BaseException as error:
        state.update(state='failed', finished=time.time(), error=repr(error))
        save('export-status.json', state)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps(state))


if __name__ == '__main__':
    raise SystemExit(main())
