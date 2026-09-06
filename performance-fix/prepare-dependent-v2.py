"""One-shot local v2 qualification assembly; no builds, tests or remote calls."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


REPO = Path('/Users/xangma/repos/pycbc')
HERE = Path(__file__).resolve().parent
BASE = 'dfd42bf76766cadca0eecf609a1eaeac73534676'
V1 = '450ab3f96ccea2783abb943b47f698578a507d59'
V2 = '0d00581251e642a5d6b56b2497a9adad93069e6b'
MAIN_REF = 'refs/heads/codex/torch-performance-fix-20260906'
SHARED = 'pycbc/filter/matchedfilter.py'
SOURCES = (
    ('fft', 'd6407d32742a57e4f026c461a26ef7b3929c5849',
     '37d3c6b4ac1ec74a2dcd42558f44f21b92218ca5'),
    ('cpu', '1514327669fc7be125523b991c847868c3a2a17e',
     'bd53914be6d2e4324cc867d52b3842b77cc6729a'),
)
CODE = {
    SHARED, 'pycbc/filter/matchedfilter_torch.py',
    'pycbc/waveform/taylorf2_torch.py', 'test/test_live_batch_torch_peaks.py',
    'test/test_torch_batch_overlap_scaling.py',
    'test/waveform/test_taylorf2_phase_evaluation.py',
}
DIFF = (
    '--no-ext-diff', '--no-textconv', '--no-color', '--no-renames',
    '--binary', '--full-index', '--unified=3', '--diff-algorithm=myers',
    '--no-indent-heuristic', '--src-prefix=a/', '--dst-prefix=b/',
)
BUNDLE = HERE / 'dependent-sources-v2.bundle'
MANIFEST = HERE / 'dependent-sources-v2.json'
JOURNAL = HERE / 'dependent-sources-v2-journal.json'


def git(*args, cwd=REPO):
    return subprocess.check_output(['git', '-C', str(cwd), *args])


def rev(ref):
    return git('rev-parse', ref).decode().strip()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def paths(before, after):
    return set(git('diff', '--name-only', '--no-renames', before, after)
               .decode().splitlines())


def patch(before, after, path):
    data = git('diff', *DIFF, before, after, '--', path)
    if path == SHARED:
        data = re.sub(rb'^index [0-9a-f]{40}\.\.[0-9a-f]{40}(.*)$',
                      rb'index <old>..<new>\1', data, flags=re.M)
        data = re.sub(rb'^@@ -\d+(,\d+)? \+\d+(,\d+)? @@',
                      rb'@@ -<line>\1 +<line>\2 @@', data, flags=re.M)
    return data


def files(ref):
    output = {}
    for row in git('ls-tree', '-rz', ref).split(b'\0'):
        if not row:
            continue
        meta, raw_path = row.split(b'\t', 1)
        mode, kind, blob = meta.decode().split()
        output[raw_path.decode()] = dict(mode=mode, kind=kind, git_blob=blob)
    return output


def details(ref, path):
    result = dict(files_cache[ref][path])
    data = git('show', f'{ref}:{path}')
    result.update(sha256=digest(data), bytes=len(data))
    return result


def native_paths(mapping):
    suffixes = {'.c', '.cc', '.cpp', '.h', '.hpp', '.pyx', '.pxd',
                '.cu', '.f', '.f90'}
    roots = {'setup.py', 'setup.cfg', 'pyproject.toml', 'MANIFEST.in'}
    return {p for p in mapping if Path(p).suffix.lower() in suffixes or p in roots}


def saved_json(path, data, exclusive=False):
    with path.open('x' if exclusive else 'w') as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write('\n')


require(rev(V2 + '^') == V1 and rev(V1 + '^') == BASE, 'main ancestry')
require(rev(MAIN_REF) == V2, 'main ref moved')
require(rev(V2 + '^{tree}') == 'fa7a7c09df6d93133324f762d756842de172433d',
        'main tree differs')
require(paths(V1, V2) == {SHARED}, 'refinement scope differs')
require(paths(BASE, V2) == CODE, 'aggregate fix scope differs')
require(all(not p.exists() for p in (BUNDLE, MANIFEST, JOURNAL)),
        'v2 output already exists; do not retry ambiguous assembly')
old_worktrees = {}
for key, old, _ in SOURCES:
    stem = f'pycbc-torch-performance-fix-{key}-test-20260906'
    worktree = Path('/private/tmp') / stem
    new = Path(str(worktree) + '-v2')
    branch = f'refs/heads/codex/torch-performance-fix-{key}-test-20260906-v2'
    require(not new.exists(), f'worktree already exists: {new}')
    require(subprocess.run(['git', '-C', str(REPO), 'show-ref', '--verify',
                            '--quiet', branch]).returncode == 1,
            f'candidate ref already exists: {branch}')
    require(git('rev-parse', 'HEAD', cwd=worktree).decode().strip() == old,
            f'old worktree HEAD moved: {worktree}')
    require(not git('status', '--porcelain', cwd=worktree),
            f'old worktree is dirty: {worktree}')
    old_worktrees[key] = dict(path=str(worktree), head=old)

files_cache = {ref: files(ref) for ref in (BASE, V1, V2)}
refinement = patch(V1, V2, SHARED)
aggregate = {p: digest(patch(BASE, V2, p)) for p in sorted(CODE)}
state = dict(schema='torch_performance_fix_dependent_sources_v2',
             started_utc=datetime.now(timezone.utc).isoformat(), state='running',
             script_sha256=digest(Path(__file__).read_bytes()), sources=[])
saved_json(JOURNAL, state, exclusive=True)
try:
    for key, old, published in SOURCES:
        branch = f'codex/torch-performance-fix-{key}-test-20260906-v2'
        worktree = Path(old_worktrees[key]['path'] + '-v2')
        git('worktree', 'add', '-b', branch, str(worktree), old)
        git('cherry-pick', V2, cwd=worktree)
        new = rev('refs/heads/' + branch)
        require(rev(new + '^') == old, f'{key}: unexpected parent')
        require(paths(old, new) == {SHARED}, f'{key}: refinement scope')
        require(patch(old, new, SHARED) == refinement,
                f'{key}: refinement patch differs')
        files_cache.update({ref: files(ref) for ref in (old, new, published)})
        feature_paths = paths(V1, old)
        require(paths(V2, new) == feature_paths == paths(BASE, published),
                f'{key}: feature path set changed')
        checks = {}
        for path in sorted(feature_paths):
            old_patch = patch(V1, old, path)
            new_patch = patch(V2, new, path)
            require(old_patch == new_patch == patch(BASE, published, path),
                    f'{key}: feature patch changed: {path}')
            if path != SHARED:
                require(files_cache[old][path] == files_cache[new][path]
                        == files_cache[published][path],
                        f'{key}: feature file bytes/mode changed: {path}')
            checks[path] = dict(
                patch_sha256=digest(new_patch),
                comparison='blob IDs/hunk starts normalized' if path == SHARED
                else 'exact patch and file bytes/mode', preserved=True)
        require(paths(published, new) == CODE, f'{key}: aggregate update scope')
        require({p: digest(patch(published, new, p)) for p in sorted(CODE)}
                == aggregate, f'{key}: aggregate update differs')
        for path in CODE - {SHARED}:
            require(files_cache[new][path] == files_cache[V2][path],
                    f'{key}: fix file differs from main: {path}')
        native = native_paths(files_cache[new])
        require(native == native_paths(files_cache[old])
                == native_paths(files_cache[published]), f'{key}: native paths')
        for path in native:
            require(files_cache[new][path] == files_cache[old][path]
                    == files_cache[published][path], f'{key}: native input {path}')
        git('diff', '--check', old, new)
        require(not git('status', '--porcelain', cwd=worktree),
                f'{key}: new worktree dirty')
        state['sources'].append(dict(
            key=key, ref='refs/heads/' + branch, worktree=str(worktree),
            old=old, published=published, new=new, parent=old,
            tree=rev(new + '^{tree}'), worktree_clean=True,
            feature_base_before=V1, feature_base_after=V2,
            refinement_patch_sha256=digest(refinement),
            aggregate_update_sha256=aggregate, feature_checks=checks,
            fix_files={p: details(new, p) for p in sorted(CODE)},
            native_build_inputs={p: details(new, p) for p in sorted(native)}))
        saved_json(JOURNAL, state)
    require(rev(MAIN_REF) == V2, 'main ref moved during preparation')
    refs = [MAIN_REF] + [s['ref'] for s in state['sources']]
    git('bundle', 'create', str(BUNDLE), *refs, '^' + BASE)
    verify = subprocess.run(['git', '-C', str(REPO), 'bundle', 'verify',
                             str(BUNDLE)], check=True, capture_output=True,
                            text=True)
    header = BUNDLE.read_bytes().split(b'\n\n', 1)[0].decode()
    prereqs = [line[1:].split()[0] for line in header.splitlines()
               if line.startswith('-')]
    require(prereqs == [BASE], 'unexpected bundle prerequisites')
    advertised = {line.split()[1]: line.split()[0] for line in
                  git('bundle', 'list-heads', str(BUNDLE)).decode().splitlines()}
    require(advertised == {ref: rev(ref) for ref in refs}, 'bundle refs differ')
    for old in old_worktrees.values():
        require(git('rev-parse', 'HEAD', cwd=old['path']).decode().strip()
                == old['head'] and not git('status', '--porcelain', cwd=old['path']),
                'old qualification worktree changed')
    state.update(
        state='complete', finished_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Local temporary qualification sources; no builds/tests/remote '
        'actions. Existing source/test worktrees were not modified.',
        main_fix=dict(commit=V2, parent=V1, tree=rev(V2 + '^{tree}'),
                      files={p: details(V2, p) for p in sorted(CODE)}),
        feature_comparison='Only matchedfilter.py blob IDs and hunk start '
        'coordinates normalized. Context, additions, deletions, counts, modes '
        'and all other paths remain exact.',
        old_worktrees=old_worktrees,
        bundle=dict(path=str(BUNDLE), bytes=BUNDLE.stat().st_size,
                    sha256=digest(BUNDLE.read_bytes()), prerequisites=prereqs,
                    advertised_refs=advertised, git_bundle_verify_passed=True,
                    verify_output=verify.stdout + verify.stderr))
    saved_json(MANIFEST, state, exclusive=True)
    saved_json(JOURNAL, state)
    print(json.dumps(dict(main=V2, sources=[{k: s[k] for k in
                    ('key', 'new', 'tree', 'worktree')} for s in state['sources']],
                    bundle=state['bundle']), indent=2))
except Exception as error:
    state.update(state='failed', error=str(error))
    saved_json(JOURNAL, state)
    raise
