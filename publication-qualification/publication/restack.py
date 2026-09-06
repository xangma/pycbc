#!/usr/bin/env python3
"""Freeze local inputs by default; --execute assembles four guarded candidates.

No network, push, PR edit, or test execution. An existing execution journal blocks
automatic retries. The original source worktree and its branch are never edited.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys


OUT = Path(__file__).resolve().parent
FIX = 'a4d77a6d1863c0515e8dace64c5609b63d40b51e'
FIX_SOURCE_CHAIN = [
    '450ab3f96ccea2783abb943b47f698578a507d59',
    '0d00581251e642a5d6b56b2497a9adad93069e6b',
    'fb4b335eeeeaeaa907c1143b45e0191e2d977761',
    '968bcd558117262af0d603710b054174659adb51',
    '6c82155044d58f3344b281869d87745f71ba2285',
    'f2c0abe61e787a26f41208f489c62c877bbd5667',
    '837f38d493420043e45fb1ad210a0ccf68bacbaa',
    FIX,
]
INPUT_BRANCH = 'codex/torch-inspiral-hotpaths-20260906'
PREFIX = 'codex/torch-performance-fix-20260906-'
BACKUPS = 'refs/codex-backups/torch-performance-fix-20260906/'
SHARED = {'pycbc/fft/torchfft.py', 'pycbc/filter/matchedfilter.py', 'pycbc/vetoes/chisq_torch.py',
          '.github/workflows/basic-tests.yml'}
WORKFLOWS = {'.github/workflows/basic-tests.yml', '.github/workflows/torch-gpu.yml'}
CODE = sorted([
    'pycbc/fft/torchfft.py', 'pycbc/waveform/decompress_torch.py',
    'test/test_torch_large_ifft.py', 'test/test_torch_decompress_cpu.py',
    'bin/pycbc_inspiral',
    'pycbc/filter/matchedfilter.py', 'pycbc/filter/matchedfilter_torch.py',
    'pycbc/psd/__init__.py', 'pycbc/scheme.py',
    'pycbc/strain/strain.py', 'pycbc/vetoes/chisq.py', 'pycbc/vetoes/chisq_torch.py',
    'test/test_scheme_runtime.py', 'test/test_chisq_precision.py',
    'test/test_sigmasq_series_precision.py', 'test/test_strain_psd_precision.py',
    'pycbc/waveform/taylorf2_torch.py', 'test/test_live_batch_torch_peaks.py',
    'test/test_torch_batch_overlap_scaling.py',
    'test/waveform/test_taylorf2_phase_evaluation.py',
])
STACK = [
    dict(key='pr11', number=15, branch='torch-pr11-performance-evidence',
         base='torch-pr10-inference',
         sha='dfd42bf76766cadca0eecf609a1eaeac73534676',
         parent='78d99b5e0f540abd438e01a221de77fc02109b3f'),
    dict(key='format-fft', number=19, branch='torch-fft-formatting-base',
         base='torch-pr11-performance-evidence',
         sha='fa38dbca79f4e2e662f079e56b4d9dbb73fd9ef4',
         parent='dfd42bf76766cadca0eecf609a1eaeac73534676'),
    dict(key='fft-followup', number=16,
         branch='torch-followup-fft-optimizations',
         base='torch-fft-formatting-base',
         sha='37d3c6b4ac1ec74a2dcd42558f44f21b92218ca5',
         parent='fa38dbca79f4e2e662f079e56b4d9dbb73fd9ef4'),
    dict(key='cpu-followup', number=17,
         branch='torch-followup-cpu-optimizations',
         base='torch-pr11-performance-evidence',
         sha='bd53914be6d2e4324cc867d52b3842b77cc6729a',
         parent='dfd42bf76766cadca0eecf609a1eaeac73534676'),
]
UNCHANGED = [
    (8, 'torch-pr4-filtering', 'torch-pr3-psd',
     '0d6160c3ed22d87e63de4a0c663ac61038799cf4'),
    (9, 'torch-pr5-search', 'torch-pr4-filtering',
     '9db3c9779c45b71d130f6f56e9be84a385780b86'),
    (11, 'torch-pr7-taylorf2', 'torch-pr6-waveform',
     'a4ba198e7c4d2fa1b0238de946f1367a98e7a2f6'),
]
DIFF = ['--no-ext-diff', '--no-textconv', '--no-color', '--no-renames',
        '--binary', '--full-index', '--unified=3', '--diff-algorithm=myers',
        '--no-indent-heuristic', '--src-prefix=a/', '--dst-prefix=b/']


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data, exclusive=False):
    if exclusive:
        with path.open('x') as stream:
            stream.write(json.dumps(data, indent=2) + '\n')
    else:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(data, indent=2) + '\n')
        temporary.replace(path)


def git(root, *args, data=None, check=True):
    result = subprocess.run(['git', '-C', str(root), *args], input=data,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    require(not check or result.returncode == 0,
            'git ' + ' '.join(args) + ': ' + result.stderr.decode().strip())
    return result


def value(root, *args):
    return git(root, *args).stdout.decode().strip()


def paths(root, old, new):
    return sorted(value(root, 'diff', '--no-renames', '--name-only',
                        old, new).splitlines())


def patch(root, old, new, path=None):
    args = ['diff', *DIFF]
    if path == 'pycbc/fft/torchfft.py':
        # The independent wisdom-cache patch borders the changed MKL factory
        # signature. Compare every changed line without that shared context;
        # quality checks additionally reconstruct the complete tested file.
        args += ['--unified=0']
    args += [old, new]
    if path is not None:
        args += ['--', path]
    return git(root, *args).stdout


def normalized(data):
    """Remove only blob IDs and hunk starts; preserve counts and all text."""
    data = re.sub(rb'(?m)^index [0-9a-f]{40}\.\.[0-9a-f]{40}(.*)$',
                  rb'index <old>..<new>\1', data)
    return re.sub(rb'(?m)^@@ -\d+(,\d+)? \+\d+(,\d+)? @@',
                  rb'@@ -START\1 +START\2 @@', data)


def fingerprints(root, old, new, selected):
    result = {}
    for path in selected:
        data = patch(root, old, new, path)
        result[path] = digest(normalized(data) if path in SHARED else data)
    return result


def entries(root, revision, selected):
    result = {}
    for path in selected:
        item = value(root, 'ls-tree', revision, '--', path)
        result[path] = item.split('\t', 1)[0] if item else None
    return result


def parent_check(root, head, parent):
    require(value(root, 'rev-list', '--parents', '-n', '1', head)
            == head + ' ' + parent, 'Wrong parent or merge commit: ' + head)
    require(value(root, 'rev-list', '--count', parent + '..' + head) == '1',
            'More than one commit above base: ' + head)


def no_operation(root):
    for marker in ('CHERRY_PICK_HEAD', 'MERGE_HEAD', 'REBASE_HEAD',
                   'sequencer', 'rebase-apply', 'rebase-merge'):
        location = Path(value(root, 'rev-parse', '--git-path', marker))
        require(not (location if location.is_absolute() else root / location)
                .exists(), 'Unfinished Git operation: ' + marker)


def snapshot_files(root, selected):
    result = {}
    for relative in selected:
        path = root / relative
        require(path.is_file() and not path.is_symlink()
                and path.resolve().is_relative_to(root.resolve()),
                'Missing, linked, or escaping input: ' + relative)
        mode = stat.S_IMODE(path.stat().st_mode)
        require(mode in (0o644, 0o755), 'Unexpected input mode: ' + relative)
        result[relative] = dict(sha256=digest(path.read_bytes()), mode=mode)
    return result


def inspect(args):
    root = args.root.resolve()
    raw_allowlist = args.allowlist.read_bytes()
    allowlist = json.loads(raw_allowlist)
    require(set(allowlist) == {'schema', 'paths'}
            and allowlist['schema'] == 'torch-performance-fix-inputs-v1',
            'Expected explicit supplementary allowlist schema and paths')
    docs = allowlist['paths']
    require(isinstance(docs, list) and docs and all(isinstance(p, str)
            for p in docs) and len(set(docs)) == len(docs),
            'Supplementary allowlist must contain unique explicit paths')
    for path in docs:
        require((path.startswith('docs/') or path in WORKFLOWS) and '\\' not in path
                and '\n' not in path and '\t' not in path
                and '..' not in PurePosixPath(path).parts
                and str(PurePosixPath(path)) == path,
                'Invalid supplementary path: ' + path)
    docs = sorted(docs)
    require(value(root, 'rev-parse', 'HEAD') == FIX
            and value(root, 'branch', '--show-current') == INPUT_BRANCH,
            'Input HEAD or branch changed')
    no_operation(root)
    status = [row for row in git(root, 'status', '--porcelain=v1', '-z',
              '--untracked-files=all').stdout.decode().split('\0') if row]
    require(all(row[:2] in (' M', '??') for row in status)
            and sorted(row[3:] for row in status) == docs,
            'Only the exact unstaged/untracked supplementary allowlist is allowed')
    git(root, 'diff', '--check')
    for parent, head in zip([STACK[0]['sha'], *FIX_SOURCE_CHAIN[:-1]],
                            FIX_SOURCE_CHAIN):
        parent_check(root, head, parent)
    require(paths(root, STACK[0]['sha'], FIX) == CODE,
            'Committed fix differs from 20-path allowlist')
    snapshot_bytes = args.snapshot.read_bytes()
    prs = json.loads(snapshot_bytes)
    expected = UNCHANGED + [(e['number'], e['branch'], e['base'], e['sha'])
                            for e in STACK]
    require(isinstance(prs, list) and len(prs) == len(expected),
            'Expected seven PR snapshot records')
    by_number = {p['number']: p for p in prs}
    require(set(by_number) == {e[0] for e in expected}, 'Wrong snapshot PRs')
    for number, branch, base, head in expected:
        pr = by_number[number]
        require((pr['headRefName'], pr['baseRefName'], pr['headRefOid'])
                == (branch, base, head) and pr['state'] == 'OPEN'
                and pr['isDraft'] is True and pr['author']['login'] == 'xangma'
                and 'agent-assisted' in {x['name'] for x in pr['labels']},
                'Snapshot metadata changed: #' + str(number))
    source = []
    for item in STACK:
        parent_check(root, item['sha'], item['parent'])
        ref = 'refs/heads/' + PREFIX + item['key']
        require(git(root, 'show-ref', '--verify', '--quiet', ref,
                    check=False).returncode == 1, 'Candidate ref exists: ' + ref)
        selected = paths(root, item['parent'], item['sha'])
        if item != STACK[0]:
            require(set(selected) & set(CODE + docs) <= SHARED,
                    'New shared path requires explicit review: ' + item['key'])
        source.append({**item, 'files': selected,
                       'patches': fingerprints(root, item['parent'],
                                               item['sha'], selected),
                       'entries': entries(root, item['sha'], selected)})
    for key in [e['key'] for e in STACK] + ['input-fix']:
        require(git(root, 'show-ref', '--verify', '--quiet', BACKUPS + key,
                    check=False).returncode == 1, 'Backup ref already exists')
    require(not args.assembly.exists(), 'Assembly path already exists')
    return dict(schema='torch-performance-fix-restack-v1', root=str(root),
                assembly=str(args.assembly.resolve()), input_head=FIX,
                fix_source_chain=FIX_SOURCE_CHAIN,
                input_branch=INPUT_BRANCH, candidate_prefix=PREFIX,
                script_sha256=digest(Path(__file__).read_bytes()),
                snapshot_path=str(args.snapshot.resolve()),
                snapshot_sha256=digest(snapshot_bytes),
                allowlist_path=str(args.allowlist.resolve()),
                allowlist_sha256=digest(raw_allowlist),
                files=snapshot_files(root, docs), code_files=CODE,
                input_index_sha256=digest(git(root, 'ls-files', '--stage',
                                              '-z').stdout),
                input_status_sha256=digest(git(root, 'status', '--porcelain=v1',
                                              '-z', '--untracked-files=all').stdout),
                fix_patch_sha256=digest(patch(root, STACK[0]['sha'], FIX)),
                fix_entries=entries(root, FIX, CODE), source_stack=source,
                normalized_paths=sorted(SHARED),
                notice='Local identity only; no scientific or remote qualification.')


def execute(args, plan):
    root, assembly = Path(plan['root']), Path(plan['assembly'])
    journal_path = OUT / 'restack-execution.json'
    candidate_path = OUT / 'candidate-stack.json'
    bundle = OUT / 'old-heads.bundle'
    require(not any(p.exists() for p in (journal_path, candidate_path, bundle)),
            'Execution output exists; inspect before any retry')
    journal = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                   state='started', plan_sha256=digest(args.plan.read_bytes()),
                   candidates=[])
    write_json(journal_path, journal, exclusive=True)
    try:
        refs = [(BACKUPS + e['key'], e['sha']) for e in STACK]
        refs.append((BACKUPS + 'input-fix', FIX))
        transaction = 'start\n' + ''.join('create ' + ref + ' ' + head + '\n'
                                         for ref, head in refs) + 'prepare\ncommit\n'
        git(root, 'update-ref', '--stdin', data=transaction.encode())
        git(root, 'bundle', 'create', str(bundle), *(ref for ref, _ in refs))
        git(root, 'bundle', 'verify', str(bundle))
        journal['backup_bundle_sha256'] = digest(bundle.read_bytes())
        write_json(journal_path, journal)
        git(root, 'worktree', 'add', '-b', PREFIX + 'pr11', str(assembly),
            STACK[0]['sha'])
        git(assembly, 'apply', '--index', '-',
            data=patch(root, STACK[0]['sha'], FIX))
        require(snapshot_files(root, list(plan['files'])) == plan['files'],
                'Supplementary inputs changed before copying')
        for path, info in plan['files'].items():
            target = assembly / path
            require(not target.is_symlink()
                    and target.resolve().is_relative_to(assembly.resolve()),
                    'Assembly destination escapes worktree: ' + path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / path).read_bytes())
            target.chmod(info['mode'])
        git(assembly, 'add', '--', *plan['files'])
        updated = sorted(CODE + list(plan['files']))
        require(paths(assembly, 'HEAD', '--cached') == updated,
                'Staged paths differ from frozen update')
        for path, info in plan['files'].items():
            require(digest(git(assembly, 'show', ':' + path).stdout)
                    == info['sha256'], 'Staged supplementary input differs: ' + path)
            staged = value(assembly, 'ls-files', '--stage', '--', path)
            expected_mode = '100755' if info['mode'] == 0o755 else '100644'
            require(staged.startswith(expected_mode + ' '),
                    'Staged supplementary input mode differs: ' + path)
        git(assembly, 'diff', '--cached', '--check')
        git(assembly, 'commit', '--amend', '--no-edit')
        new_heads = {}
        main_update = None
        for index, old in enumerate(plan['source_stack']):
            parent = old['parent'] if index == 0 else new_heads[old['base']]
            if index:
                git(assembly, 'switch', '-c', PREFIX + old['key'], parent)
                git(assembly, 'cherry-pick', old['sha'])
            head = value(assembly, 'rev-parse', 'HEAD')
            parent_check(assembly, head, parent)
            require(not value(assembly, 'status', '--porcelain=v1',
                              '--untracked-files=all'), 'Dirty candidate worktree')
            no_operation(assembly)
            require(paths(assembly, old['sha'], head) == updated,
                    'Main update paths changed: ' + old['key'])
            delta = fingerprints(assembly, old['sha'], head, updated)
            if not index:
                require(entries(assembly, head, CODE) == plan['fix_entries'],
                        'Squashed source differs from tested fix commit')
                main_update = delta
            else:
                require(paths(assembly, parent, head) == old['files'],
                        'Feature path set changed: ' + old['key'])
                require(fingerprints(assembly, parent, head, old['files'])
                        == old['patches'], 'Feature patch changed: ' + old['key'])
                nonshared = [p for p in old['files'] if p not in SHARED]
                require(entries(assembly, head, nonshared)
                        == {p: old['entries'][p] for p in nonshared},
                        'Feature bytes or modes changed: ' + old['key'])
                require(delta == main_update,
                        'Main update differs on dependent head: ' + old['key'])
            new_heads[old['branch']] = head
            journal['candidates'].append(dict(
                number=old['number'], branch=old['branch'], base=old['base'],
                candidate_branch=PREFIX + old['key'], old_head=old['sha'],
                head=head, parent=parent, tree=value(assembly, 'rev-parse',
                                                    head + '^{tree}')))
            write_json(journal_path, journal)
        require(value(root, 'rev-parse', 'HEAD') == FIX
                and value(root, 'branch', '--show-current') == INPUT_BRANCH
                and snapshot_files(root, list(plan['files'])) == plan['files']
                and digest(git(root, 'ls-files', '--stage', '-z').stdout)
                == plan['input_index_sha256']
                and digest(git(root, 'status', '--porcelain=v1', '-z',
                               '--untracked-files=all').stdout)
                == plan['input_status_sha256'],
                'Original worktree changed during assembly')
        write_json(candidate_path, dict(
            schema='torch-performance-fix-candidates-v1',
            plan_sha256=journal['plan_sha256'], candidates=journal['candidates'],
            main_update_patches=main_update, backup_bundle=str(bundle),
            backup_bundle_sha256=journal['backup_bundle_sha256'],
            notice='Structural checks passed; tests and publication remain separate.'),
            exclusive=True)
        journal['state'] = 'complete'
    except BaseException as exc:
        journal['state'], journal['error'] = 'failed', str(exc)
        raise
    finally:
        journal['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(journal_path, journal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
                        default=Path('/private/tmp/pycbc-torch-inspiral-hotpaths-20260906'))
    parser.add_argument('--allowlist', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, default=OUT / 'prs-current.json')
    parser.add_argument('--plan', type=Path, default=OUT / 'frozen-plan.json')
    parser.add_argument('--assembly', type=Path,
                        default=Path('/private/tmp/pycbc-torch-performance-fix-20260906-assembly'))
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    inspected = inspect(args)
    if args.execute:
        require(args.plan.exists(), 'Freeze and review a dry-run plan first')
        require(json.loads(args.plan.read_bytes()) == inspected,
                'Frozen inputs changed; do not execute')
        execute(args, inspected)
        print('Created four local candidates; no publication or test run performed.')
    else:
        if args.plan.exists():
            require(json.loads(args.plan.read_bytes()) == inspected,
                    'Existing plan differs; review before replacing it')
        else:
            write_json(args.plan, inspected, exclusive=True)
        print('Frozen local plan:', args.plan)
        print('Supplementary paths:', len(inspected['files']),
              '| source paths:', len(CODE),
              '| candidates: 4 | Git state unchanged')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError, TypeError) as error:
        print('Refused:', error, file=sys.stderr)
        sys.exit(1)
