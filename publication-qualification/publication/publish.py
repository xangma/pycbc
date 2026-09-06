#!/usr/bin/env python3
"""Freeze reviewed publication inputs; --execute updates four refs and seven PRs.

Default invocation performs read-only Git/GitHub checks and writes a new local
plan. --execute repeats that preflight and requires byte-identical frozen inputs.
--resume-bodies requires this publisher's existing journal and all four new refs;
it never retries a push. It accepts only its own frozen old/intended PR bodies.
No archive commits/pushes, body generation, tests, review requests or comments.

Required --context JSON (all paths relative to its directory or absolute):
  schema: torch-performance-fix-publication-context-v2
  reviewed: true (set by the primary operator after reviewing actual evidence)
  inputs: {ROLE: {path: PATH, sha256: SHA256}, ...}, with exactly these roles:
    restack_plan, restack_execution, restack_script, candidate_stack, snapshot,
    body_context, body_generator, bodies_manifest, template
  archive: {root: PATH, branch: codex/torch-benchmarks-20260906,
    a: FULL_SHA, b: FULL_SHA, a_changes: [PATH,...], b_changes: [PATH,...]}
    A must descend the pinned prior archive; B must descend A and be published.
    Both exact change inventories are reviewed; B may append only
    publication-qualification/ plus README.md and SHA256SUMS. A's science bytes
    and all earlier historical bytes remain preserved except those root indices.
  checks: {ROLE: {archive: "b", path: REPO_PATH, sha256: SHA256,
                  assertions: [{pointer: JSON_POINTER, equals: VALUE}, ...]}}
    Required roles: archive_finalization, docs, quality, and for each key
    pr11/format-fft/fft-followup/cpu-followup:
      KEY-source-before, KEY-source-after, KEY-reviewed-source.
    Checks must be committed regular JSON files under publication-qualification/.
    Every check requires actual reviewed assertions; no finalization filename is
    assumed. Add other committed receipts/logs as body-context evidence as needed.
    Candidate snapshots must identify the exact candidate head/tree, be clean and
    unchanged, and be bound to completed passing quality commands. Docs validation
    must include all thirteen pages, twenty images and four manifests.

The body manifest must come from the reviewed write-bodies.py and bind its
snapshot, candidate stack, template, reviewed context and every cited file.
Body-context evidence URLs must resolve to exact matching blobs in archive A/B.
Six frozen baseline and eleven optimized science records are checked for
completion/source identity; the optimized report binds every archived input/output.
Review numerical claims, scientific scope, root index diffs and every preserved
failure independently. This script checks identities and receipts, not science.

All outputs stay beside this script, outside both Git worktrees and archive B.
Default plan: publish-plan.json; journal: publish-execution.json; per-operation
logs: publish-NNN-NAME.log. Never erase a journal to bypass uncertain state.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import difflib
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import socket
import stat
import statistics
import subprocess
import sys
import time
from urllib.parse import unquote, urlparse

OUT = Path(__file__).resolve().parent
REPOSITORY = 'xangma/pycbc'
GH_REPOSITORY = 'github.com/' + REPOSITORY
WEB = 'https://github.com/' + REPOSITORY
ARCHIVE_BASE = '7f1ea7a7aa05b4fdafe2995a43a754984a2c5817'
ARCHIVE_BRANCH = 'codex/torch-benchmarks-20260906'
NUMBERS = (8, 9, 11, 15, 19, 16, 17)
DOC_PAGES = ('torch', 'torch_benchmark_details', 'torch_filtering',
             'torch_inspiral_optimized', 'torch_inspiral_reference', 'torch_optimization_results',
             'torch_optimizations', 'torch_parity', 'torch_performance',
             'torch_runtime', 'torch_search', 'torch_testing', 'torch_workflows')
COLLECTIONS = {
    'torch-benchmarks-20260906': (
        'main-live.png', 'waveform.png', 'taylorf2-throughput.png',
        'taylorf2-cold.png', 'inference.png', 'inference-cold.png',
        'fft.png', 'optional-cpu.png'),
    'torch-performance-fix-20260906': (
        'live-before-after.png', 'waveform-before-after.png', 'cuda-before-v1-v2.png'),
    'torch-inspiral-reference-20260906': (
        'tuning-capacity.png', 'matched-capacity.png', 'wall-and-internal-times.png',
        'profile-self-time.png', 'native-symbols.png', 'cuda-events.png'),
    'torch-inspiral-optimized-20260906': (
        'capacity-before-after.png', 'wall-breakdown.png', 'hotpath-profile-self-time.png'),
}
RECEIPT = OUT / 'publish-execution.json'
LOCK = OUT / '.publish-lock'



BASELINE_SOURCE = '837f38d493420043e45fb1ad210a0ccf68bacbaa'
BASELINE_REPORT_SHA = '11f17262ab7ec11c21cb50bdb8dd884379414a56fb22a4bbfd8ff934365f8b14'
FIX = 'a4d77a6d1863c0515e8dace64c5609b63d40b51e'
OPTIMIZED_BUILDER_SHA = '1da505f098d065e483e6ab475a9fbf15e2c071d8f163b9056a64788b5cc0a151'
OPTIMIZED_PATHS = sorted([
    'pycbc/fft/torchfft.py', 'pycbc/waveform/decompress_torch.py',
    'test/test_torch_decompress_cpu.py', 'test/test_torch_large_ifft.py',
])
CPU_DISPATCH = dict(scheme_prefix='cpu', fft='pycbc.fft.mkl',
                    decompression='pycbc.waveform.decompress_cpu')
FIX_SOURCE_CHAIN = [
    '450ab3f96ccea2783abb943b47f698578a507d59',
    '0d00581251e642a5d6b56b2497a9adad93069e6b',
    'fb4b335eeeeaeaa907c1143b45e0191e2d977761',
    '968bcd558117262af0d603710b054174659adb51',
    '6c82155044d58f3344b281869d87745f71ba2285',
    'f2c0abe61e787a26f41208f489c62c877bbd5667',
    BASELINE_SOURCE,
    FIX,
]
INPUT_BRANCH = 'codex/torch-inspiral-hotpaths-20260906'
PREFIX = 'codex/torch-performance-fix-20260906-'
SHARED = {'pycbc/filter/matchedfilter.py', 'pycbc/vetoes/chisq_torch.py',
          '.github/workflows/basic-tests.yml', 'pycbc/fft/torchfft.py'}
CODE = sorted(OPTIMIZED_PATHS + [
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


FIELDS = ('number', 'url', 'author', 'title', 'state', 'isDraft', 'headRefName',
          'headRefOid', 'baseRefName', 'body', 'labels', 'updatedAt')
HEADINGS = ('Standard information about the request', 'Motivation', 'Contents',
            'Links to any issues or associated PRs', 'Testing performed', 'Additional notes')
COC = ('- [ ] The author of this pull request confirms they will adhere to the '
       '[code of conduct](https://github.com/gwastro/pycbc/blob/master/CODE_OF_CONDUCT.md)')
AI_NOTE = ('*AI Agent Note: Unchecked by default. @xangma, please review this PR and '
           'check the Code of Conduct box above to confirm your agreement before requesting review.*')


FINAL_SOURCE = FIX
SHARED_PATHS = SHARED


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_file(path):
    require(path.is_file() and not path.is_symlink(), 'Missing/non-regular file: ' + str(path))
    require(path.stat().st_size <= 100_000_000, 'Oversized file: ' + str(path))
    return path.read_bytes()


def record(path, data=None):
    if data is None:
        data = read_file(path)
    return {'path': str(path.resolve()), 'sha256': digest(data), 'bytes': len(data),
            'mode': stat.S_IMODE(path.stat().st_mode)}


def read_json(path):
    return json.loads(read_file(path))


def write_json(path, obj, exclusive=False):
    data = (json.dumps(obj, indent=2) + '\n').encode()
    if exclusive:
        with path.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        temporary = path.with_name(path.name + '.tmp')
        with temporary.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)


def environment():
    env = dict(os.environ)
    env.update(GH_HOST='github.com', GH_PROMPT_DISABLED='1', GIT_TERMINAL_PROMPT='0',
               GIT_OPTIONAL_LOCKS='0', GIT_PAGER='cat')
    return env


def stop_child(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run(command, cwd, log=None, input_bytes=None):
    """Bounded command, never shell text; mutation output is persisted immediately."""
    stream = log.open('xb') if log else None
    try:
        process = subprocess.Popen(
            command, cwd=cwd, env=environment(), start_new_session=True,
            stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
            stdout=stream if stream else subprocess.PIPE,
            stderr=subprocess.STDOUT if stream else subprocess.PIPE)
        if log:
            print(json.dumps({'host': socket.gethostname(), 'cwd': str(cwd),
                              'command': command, 'pid': process.pid, 'log': str(log),
                              'next_check': 'completion or 45-second timeout',
                              'stop_command': f'kill -TERM -- -{process.pid}'}), flush=True)
        try:
            stdout, stderr = process.communicate(input=input_bytes, timeout=45)
        except BaseException:
            stop_child(process)
            raise
        require(process.returncode == 0,
                f'Command failed ({process.returncode}): {command!r}; '
                + (f'see {log}' if log else (stderr or b'').decode(errors='replace').strip()))
        return stdout or b''
    finally:
        if stream:
            stream.close()


def git(root, *args):
    return run(['git', '-C', str(root), *args], root)


def value(root, *args):
    return git(root, *args).decode().strip()


def clean_root(root):
    require(root.is_dir(), 'Missing checkout: ' + str(root))
    require(Path(value(root, 'rev-parse', '--show-toplevel')).resolve() == root,
            'Expected checkout root: ' + str(root))
    require(not value(root, 'status', '--porcelain=v1', '--untracked-files=all'),
            'Checkout is not clean: ' + str(root))
    for marker in ('CHERRY_PICK_HEAD', 'MERGE_HEAD', 'REBASE_HEAD',
                   'sequencer', 'rebase-apply', 'rebase-merge'):
        path = Path(value(root, 'rev-parse', '--git-path', marker))
        if not path.is_absolute():
            path = root / path
        require(not path.exists(), 'Unfinished Git operation: ' + str(path))


def remote_urls(root):
    result = {kind: value(root, 'remote', 'get-url', *option, '--all', 'origin').splitlines()
              for kind, option in (('fetch', ()), ('push', ('--push',)))}
    allowed = {f'git@github.com:{REPOSITORY}.git', f'ssh://git@github.com/{REPOSITORY}.git',
               WEB, WEB + '.git'}
    require(all(len(urls) == 1 and urls[0] in allowed for urls in result.values()),
            'Origin must have one GitHub xangma/pycbc fetch URL and one push URL')
    return result


def remote_refs(root, url, branches):
    require(len(branches) == len(set(branches)), 'Duplicate remote branch')
    refs = {'refs/heads/' + name for name in branches}
    output = value(root, 'ls-remote', '--heads', url, *sorted(refs))
    result = {}
    for line in output.splitlines():
        sha, ref = line.split()
        require(ref in refs and ref not in result and re.fullmatch('[0-9a-f]{40}', sha),
                'Unexpected remote reference: ' + line)
        result[ref] = sha
    require(set(result) == refs, 'Required remote branch is missing')
    return result


def changed(root, old, new):
    return sorted(value(root, 'diff', '--no-renames', '--name-only', old, new).splitlines())


def patch_hash(root, old, new, relative, context=None):
    data = git(root, 'diff', '--no-ext-diff', '--no-textconv', '--no-color',
               '--no-renames', '--binary', '--full-index',
               '--unified=' + str(context if context is not None else
                                  (0 if relative == 'pycbc/fft/torchfft.py' else 3)),
               '--diff-algorithm=myers', '--no-indent-heuristic',
               '--src-prefix=a/', '--dst-prefix=b/', old, new, '--', relative)
    if relative in SHARED_PATHS:
        data = re.sub(rb'(?m)^index [0-9a-f]{40}\.\.[0-9a-f]{40}(.*)$',
                      rb'index <old>..<new>\1', data)
        data = re.sub(rb'(?m)^@@ -\d+(,\d+)? \+\d+(,\d+)? @@',
                      rb'@@ -START\1 +START\2 @@', data)
    return digest(data)


def reverse_file_patch(data, patch):
    """Reconstruct one parent file in memory from a verified Git text patch."""
    lines, output, cursor = data.splitlines(keepends=True), [], 0
    hunks = re.split(rb'(?m)^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@[^\n]*\n', patch)
    require(len(hunks) > 1 and (len(hunks) - 1) % 3 == 0,
            'Invalid single-file feature patch')
    for index in range(1, len(hunks), 3):
        start, count, body = hunks[index:index + 3]
        count = int(count) if count else 1
        start = int(start) - (1 if count else 0)
        changes = body.splitlines(keepends=True)
        require(all(line[:1] in (b' ', b'+', b'-') for line in changes),
                'Unexpected feature-patch record')
        old = [line[1:] for line in changes if line[:1] != b'+']
        new = [line[1:] for line in changes if line[:1] != b'-']
        require(cursor <= start and count == len(new)
                and lines[start:start + count] == new,
                'Feature patch does not exactly match candidate content')
        output.extend(lines[cursor:start])
        output.extend(old)
        cursor = start + count
    return b''.join(output + lines[cursor:])


def metadata(raw):
    result = {name: raw[name] for name in FIELDS}
    result['labels'] = sorted(raw['labels'], key=lambda row: row['name'])
    return result


def live_pr(number):
    raw = json.loads(run(['gh', 'pr', 'view', str(number), '--repo', GH_REPOSITORY,
                          '--json', ','.join(FIELDS)], OUT))
    return metadata(raw)


def edit_command(number):
    # gh --body-file - reads reviewed bytes from stdin, so later file edits cannot change the request.
    return ['gh', 'pr', 'edit', str(number), '--repo', GH_REPOSITORY,
            '--body-file', '-', '--add-label', 'agent-assisted']


def compare_pr(actual, expected, number, allow_old_head=None):
    # Pushes/edits may advance updatedAt. All other captured fields must match exactly.
    for field in FIELDS:
        if field == 'updatedAt':
            continue
        if field == 'headRefOid' and allow_old_head is not None:
            require(actual[field] in (expected[field], allow_old_head), f'PR #{number}: unexpected head')
        else:
            require(actual[field] == expected[field], f'PR #{number}: changed {field}')


def read_expected(number, expected, old_head=None):
    for attempt in range(4):
        observed = live_pr(number)
        compare_pr(observed, expected, number, old_head)
        if observed['headRefOid'] == expected['headRefOid']:
            return observed
        require(attempt < 3, f'PR #{number}: new head is not visible; stop for inspection')
        time.sleep(2)
    raise RuntimeError('Unreachable read-back state')


def interrupted(signum, _frame):
    raise KeyboardInterrupt(f'Interrupted by signal {signum}')


def json_pointer(value, pointer):
    require(isinstance(pointer, str) and (pointer == '' or pointer.startswith('/')),
            'Invalid JSON pointer')
    for token in pointer.split('/')[1:] if pointer else []:
        key = token.replace('~1', '/').replace('~0', '~')
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def verify_primary_reference(context, evidence, documents):
    primary = context['primary_reference']
    kinds = dict(source_manifest='qualification', provenance='qualification',
                 report='report', unit_tests='tests', waveform='science',
                 boundary='science')
    require(set(primary) == {'source_head', *kinds}
            and primary['source_head'] == BASELINE_SOURCE
            and BASELINE_SOURCE in context['source_identities'],
            'Primary reference must identify the frozen baseline source')
    ids = [primary[role] for role in kinds]
    require(len(ids) == len(set(ids)) and all(key in evidence for key in ids),
            'Primary reference requires six distinct evidence records')
    require(all(evidence[primary[role]]['kind'] == kind
                for role, kind in kinds.items()), 'Wrong primary evidence kind')
    require(evidence[primary['report']]['sha256'] == BASELINE_REPORT_SHA,
            'Frozen baseline report changed')
    records = {role: documents[primary[role]] for role in kinds}
    require(records['source_manifest']['commit'] == BASELINE_SOURCE,
            'Primary source manifest identifies an earlier revision')
    proof = records['provenance']
    require(proof['status'] == 'pass' and proof['new_source_commit'] == BASELINE_SOURCE
            and proof['normal_cpu_outputs_changed'] is True
            and proof['prior_science_reused'] is False,
            'Primary source provenance is incomplete or stale')
    report = records['report']
    require(report['schema_version'] == 1 and report['status'] == 'pass'
            and report['source_phases'] == {'precision': BASELINE_SOURCE}
            and report['gates'] and all(v is True for v in report['gates'].values())
            and report['unit_tests']['source_commit'] == BASELINE_SOURCE
            and report['parity']['status'] == 'pass'
            and report['source_provenance']['normal_cpu_outputs_changed'] is True
            and report['source_provenance']['prior_science_reused'] is False,
            'Primary report is incomplete, failed or from an earlier source')
    tests = records['unit_tests']
    require(tests['state'] == 'complete' and tests['passed'] is True
            and type(tests['returncode']) is int and tests['returncode'] == 0
            and tests['source_info'] == tests['source_after'] ==
            {'commit': BASELINE_SOURCE, 'status': ''},
            'Baseline source unit-test receipt is incomplete or stale')
    for role, field, count in (('waveform', 'template_psd_pairs', 288),
                               ('boundary', 'cases', 36)):
        data = records[role]
        require(data['state'] == 'complete' and data['passed'] is True
                and data['completed_' + field] == data['expected_' + field] == count
                and data['source_before'] == data['source_after']
                and data['source_before']['commit'] == BASELINE_SOURCE
                and not data['source_before']['status'],
                'Baseline scientific validation is incomplete or stale: ' + role)
    claims = context['prs']['15']['testing']
    cited = {key for claim in claims + context['prs']['15'].get('notes', [])
             for key in claim['evidence']}
    require(set(ids) <= cited, '#15 must cite all primary reference evidence')


def verify_optimized_reference(context, evidence, documents, locations, archive):
    """Bind the separately reviewed v6 report to its final source and archive."""
    optimized = context['optimized_reference']
    kinds = dict(source_manifest='qualification', report='report', unit_tests='tests',
                 ifft_matrix='qualification', ifft_decision='qualification', science='science',
                 waveform='science', boundary='science', compressed='science',
                 qualifications='qualification', measurements='qualification')
    require(set(optimized) == {'source_head', 'parent_source_head', *kinds}
            and optimized['source_head'] == FINAL_SOURCE
            and optimized['parent_source_head'] == BASELINE_SOURCE
            and FINAL_SOURCE in context['source_identities'], 'Wrong optimized reference sources')
    ids = [optimized[role] for role in kinds]
    require(len(set(ids)) == 11 and all(key in evidence for key in ids)
            and not set(ids) & {context['primary_reference'][role] for role in
                               ('source_manifest', 'provenance', 'report', 'unit_tests', 'waveform', 'boundary')},
            'Optimized reference requires eleven distinct fresh evidence records')
    require(all(evidence[optimized[role]]['kind'] == kind for role, kind in kinds.items()),
            'Wrong optimized evidence kind')
    records = {role: documents[optimized[role]] for role in kinds}
    source, report = records['source_manifest'], records['report']
    version, source_path = locations[optimized['source_manifest']]
    artifact_root = PurePosixPath(source_path).parent
    require(version == 'a' and PurePosixPath(source_path).name == 'source-v6.json',
            'Optimized science must be preserved in archive A')
    require(all(locations[key][0] == version for key in ids), 'Mixed optimized evidence archives')
    report_path = PurePosixPath(locations[optimized['report']][1])
    require(report_path.is_relative_to(artifact_root) and report_path.parent != artifact_root,
            'Optimized report must have its own artifact directory')
    filenames = dict(source_manifest='source-v6.json', unit_tests='unit-tests-v6.json',
        ifft_matrix='large-ifft-v6.json', ifft_decision='large-ifft-v6-decision.json',
        science='scientific-validation-v6.json', waveform='waveform-validation-v6.json',
        boundary='boundary-injections-v6.json', compressed='compressed-bank-v6.json',
        qualifications='campaign-v6-qualifications.status.json',
        measurements='campaign-v6-measurements.status.json')
    for role, name in filenames.items():
        require(locations[optimized[role]][1] == str(artifact_root / name),
                'Optimized evidence path differs: ' + role)
    gates = {'source_provenance', 'unit_tests', 'large_ifft', 'campaign_complete',
             'trigger_parity', 'qualified_work_and_timings', 'waveform_validation',
             'boundary_validation', 'scientific_integration', 'profiles'}
    require(report['schema_version'] == 1 and report['status'] == 'pass'
            and report['source_commit'] == FINAL_SOURCE and report['parent_source_commit'] == BASELINE_SOURCE
            and gates <= set(report['gates']) and all(v is True for v in report['gates'].values())
            and report['source_manifest_sha256'] == evidence[optimized['source_manifest']]['sha256'],
            'Optimized report is incomplete, failed or stale')
    hashes = report['input_sha256']
    require(isinstance(hashes, dict) and hashes
            and hashes.get('build-hotpath-report-v6.py') == OPTIMIZED_BUILDER_SHA
            and hashes.get('final-report/report.json') == BASELINE_REPORT_SHA,
            'Optimized report lacks the frozen builder or baseline report')
    for role, name in filenames.items():
        require(hashes.get(name) == evidence[optimized[role]]['sha256'],
                'Optimized report does not bind its evidence: ' + role)
    blobs = {}
    for name, sha256 in hashes.items():
        relative_path(name)
        require(re.fullmatch('[0-9a-f]{64}', sha256 or ''), 'Malformed optimized input hash')
        raw = archive_blob(archive, version, str(artifact_root / name), sha256)
        if name.endswith('.json') or name == 'unit-tests-v6.log':
            blobs[name] = raw
    expected_outputs = {'runs.csv', 'groups.csv', 'before-after.csv', 'profiles.csv', 'native-symbols.csv',
                        *COLLECTIONS['torch-inspiral-optimized-20260906']}
    require(set(report['output_sha256']) == expected_outputs
            and report['figures'] == list(COLLECTIONS['torch-inspiral-optimized-20260906']),
            'Optimized report output inventory differs')
    for name, sha256 in report['output_sha256'].items():
        archive_blob(archive, version, str(report_path.parent / name), sha256)

    def read(name):
        require(name in blobs, 'Report omitted required evidence: ' + name)
        return json.loads(blobs[name])

    remote = PurePosixPath(source['source']).parent
    omitted = report['archived_remote_input_sha256']
    require(isinstance(omitted, dict), 'Missing portable source/frame hash manifest')

    def recorded(mapping, after=None):
        require(isinstance(mapping, dict) and mapping and (after is None or mapping == after),
                'Missing or changed optimized receipt inputs')
        for name, sha256 in mapping.items():
            path = PurePosixPath(name)
            if path.is_absolute():
                name = str(path.relative_to(remote)) if path.is_relative_to(remote) else None
            if name in hashes:
                require(hashes[name] == sha256, 'Receipt/report input mismatch: ' + name)
            else:
                require(omitted.get(str(path)) == sha256, 'Unbound portable receipt input: ' + str(path))

    def passed(row, label):
        require(row['state'] == 'complete' and row['passed'] is True
                and type(row['returncode']) is int and row['returncode'] == 0 and row.get('finished_utc')
                and row['source_info'] == row['source_after'] == {'commit': FINAL_SOURCE, 'status': ''},
                'Optimized receipt is incomplete or stale: ' + label)
        recorded(row['input_sha256'], row['input_sha256_after'])

    require(source['schema_version'] == 1 and source['commit'] == FINAL_SOURCE
            and source['parent'] == BASELINE_SOURCE
            and source['changed_paths'] == report['changed_paths'] == OPTIMIZED_PATHS
            and set(source['changed_files_sha256']) == set(OPTIMIZED_PATHS), 'Wrong optimized source delta')
    previous = read('source-v5.json')
    require(previous['commit'] == BASELINE_SOURCE and len(source['native_modules_sha256']) == 11
            and source['native_modules_sha256'] == previous['native_modules_sha256'],
            'Optimized native modules differ from the baseline')
    source_root = Path(archive['_source_root'])
    require(changed(source_root, BASELINE_SOURCE, FINAL_SOURCE) == OPTIMIZED_PATHS,
            'Optimized source changed outside its reviewed paths')
    for name, sha256 in source['changed_files_sha256'].items():
        require(digest(git(source_root, 'show', FINAL_SOURCE + ':' + name)) == sha256,
                'Optimized source blob differs: ' + name)
    require(hashes.get('inspiral-source-v6.bundle') == source['bundle_sha256'],
            'Optimized source bundle is unbound')
    audit = source['normal_cpu_path_audit']
    require(audit == report['normal_cpu_path_audit'] and audit['status'] == 'pass'
            and audit['source_commit'] == FINAL_SOURCE and audit['parent_source_commit'] == BASELINE_SOURCE
            and audit['dispatch'] == CPU_DISPATCH and audit['rationale'].strip(), 'Missing normal CPU audit')
    recorded(source['input_sha256'], source['input_sha256_after'])
    for role in ('unit_tests', 'ifft_decision', 'science', 'compressed'):
        passed(records[role], role)
    units = records['unit_tests']
    require(hashes.get('unit-tests-v6.log') == units['log_sha256']
            and report['unit_tests']['receipt'] == 'unit-tests-v6.json'
            and report['unit_tests']['command'] == units['command'], 'Optimized unit-test report differs')
    lines = blobs['unit-tests-v6.log'].decode().strip().splitlines()
    require(lines and report['unit_tests']['summary'] == lines[-1].strip()
            and re.search(r'\b[1-9][0-9]* passed\b', lines[-1])
            and not re.search(r'\b[1-9][0-9]* (?:failed|errors?)\b', lines[-1]), 'No passing unit-test summary')
    for role, field, count in (('waveform', 'template_psd_pairs', 288), ('boundary', 'cases', 36)):
        row = records[role]
        require(row['state'] == 'complete' and row['passed'] is True
                and row['expected_' + field] == row['completed_' + field] == count
                and row['source_before'] == row['source_after']
                and row['source_before']['commit'] == FINAL_SOURCE and row['source_before']['status'] == '',
                'Optimized scientific validation is incomplete or stale: ' + role)
        recorded(row['input_sha256'], row['input_sha256_after'])
    science = records['science']
    checks = {'waveform_reference', 'compressed_bank_backend_parity',
              'boundary_injections', 'qualification_trigger_parity'}
    require(set(science['checks']) == set(science['evidence']) == checks
            and all(v is True for v in science['checks'].values()), 'Optimized science checks are incomplete')
    for item in science['evidence'].values():
        recorded({item['path']: item['sha256']})
    runner = read('scientific-checks-v6.status.json')
    passed(runner, 'scientific runner')
    require(runner['current'] is None and runner['child_pid'] is None
            and len(runner['plan']) == len(runner['commands']) == 5
            and runner['completed'] == [row['name'] for row in runner['plan']]
            and all(row['returncode'] == 0 for row in runner['commands'])
            and runner['expected_coverage'] == dict(waveform_template_psd_pairs=288,
                boundary_cases=36, compressed_bank_cases=576)
            and runner['output_sha256'] == runner['all_created_output_sha256'],
            'Optimized scientific runner is incomplete')
    recorded(runner['output_sha256'])
    compressed = records['compressed']
    require(compressed['expected_cases'] == compressed['completed_cases'] == len(compressed['cases']) == 576
            and compressed['native_calls'] == compressed['cuda_comparisons'] == 288
            and {(r['device'], r['length_seconds'], r['index']) for r in compressed['cases']} ==
            set(itertools.product(('cpu', 'cuda'), (256, 512, 1024), range(96)))
            and all(r['bitwise_equal'] is True and
                    (r['reference'] == 'normal CPU' and r['native_used'] is True if r['device'] == 'cpu'
                     else r['reference'] == 'frozen v5 Torch CUDA') for r in compressed['cases']),
            'Optimized compressed waveform coverage/parity failed')
    summary = report['scientific_validation']
    require(summary['waveform_template_psd_pairs'] == 288 and summary['boundary_cases'] == 36
            and summary['compressed_backend_cases'] == 576, 'Optimized scientific report counts differ')
    matrix = records['ifft_matrix']
    require(records['ifft_decision']['matrix'] == dict(
                path=str(remote / 'large-ifft-v6.json'), sha256=hashes['large-ifft-v6.json'])
            and matrix['source_before']['harness_sha256'] == hashes['qualify-large-ifft-v6b.py'],
            'Optimized IFFT decision/harness binding differs')
    verify_optimized_ifft(matrix, records['ifft_decision'], source, report)
    stages = []
    for stage in ('qualifications', 'measurements'):
        row = records[stage]
        require(row['schema_version'] == 1 and row['state'] == 'complete' and row['stage'] == stage
                and type(row['returncode']) is int and row['returncode'] == 0 and row.get('finished_utc')
                and row['source_info'] == row['source_after'] == {'commit': FINAL_SOURCE, 'status': ''}
                and row['current'] is None and row['child_pid'] is None
                and row['plan'] == row['completed'] and len(row['plan']) == (3 if stage == 'qualifications' else 16)
                and len(row['commands']) == (3 if stage == 'qualifications' else 20)
                and all(c['returncode'] == 0 for c in row['commands']), 'Incomplete optimized campaign: ' + stage)
        recorded(row['input_sha256'], row['input_sha256_after'])
        recorded(row['output_sha256'])
        launch = read('campaign-v6-' + stage + '-launch.json')
        require(launch['state'] == 'launched' and launch['pid'] == row['pid']
                and launch['source_commit'] == FINAL_SOURCE
                and launch['source_manifest_sha256'] == report['source_manifest_sha256'],
                'Optimized campaign launch differs')
        recorded(launch['input_sha256'])
        require('campaign-v6-' + stage + '-launch.log' in hashes, 'Missing campaign launch log')
        equivalence = row['normal_cpu_equivalence']
        require(equivalence['dispatch'] == CPU_DISPATCH and equivalence['reviewed_audit'] == audit
                and equivalence['native_modules_sha256'] == source['native_modules_sha256']
                and equivalence['tuning_source_commit'] == BASELINE_SOURCE, 'Missing campaign CPU equivalence')
        stages.extend(row['plan'])
    expected_cases = {row['case'] for row in stages}
    require(len(expected_cases) == 19 and report['required_cases'] == sorted(expected_cases)
            and len(report['rows']) == 19 and {row['case'] for row in report['rows']} == expected_cases
            and all(row['source_commit'] == FINAL_SOURCE for row in report['rows']), 'Optimized report omitted runs')
    schemes = {'cpu:1', 'torch:cpu:1', 'torch:cuda:0'}
    timings = [row for row in report['rows'] if row['mode'] == 'timing']
    require(len(timings) == 9 and all(sum(r['scheme'] == s for r in timings) == 3 for s in schemes)
            and all((r['segment_length'], r['start_pad'], r['end_pad']) == (512, 112, 16) for r in timings)
            and len(report['matched']) == 3 and {r['scheme'] for r in report['matched']} == schemes
            and all(r['n'] == 3 and r['source_commit'] == FINAL_SOURCE for r in report['matched']),
            'Optimized timings mix sources, profiles, repetitions or workloads')
    before = {r['scheme']: r for r in documents[context['primary_reference']['report']]['matched']}
    after = {r['scheme']: r for r in report['matched']}
    require(len(report['before_after']) == 3 and {r['scheme'] for r in report['before_after']} == schemes
            and all(r['before_source_commit'] == BASELINE_SOURCE and r['after_source_commit'] == FINAL_SOURCE
                    and r['before'] == before[r['scheme']] and r['after'] == after[r['scheme']]
                    for r in report['before_after']), 'Before/after report changed or pooled the baseline')
    parity = report['parity']
    require(parity['status'] == 'pass' and len(parity['comparisons']) == 18
            and {r['candidate'] for r in parity['comparisons']} == expected_cases - {'qual-selected6-cpu-l512'}
            and all(r['status'] == 'pass' and r['baseline'] == 'qual-selected6-cpu-l512'
                    for r in parity['comparisons'])
            and parity['tolerances'] == documents[context['primary_reference']['report']]['parity']['tolerances'],
            'Optimized trigger parity is incomplete or has changed tolerances')
    claims = context['prs']['15']['testing']
    require(claims and optimized['report'] in claims[0]['evidence']
            and FINAL_SOURCE in claims[0]['source_heads'], '#15 must lead with the optimized report and source')
    cited = {key for claim in claims + context['prs']['15'].get('notes', []) for key in claim['evidence']}
    require(set(ids) <= cited, '#15 must cite all optimized reference evidence')


def verify_optimized_ifft(matrix, decision, source, report):
    sizes = {2**20, 2**21, 2**22}
    dispatch = {str(size): 'mkl_double_workspace' for size in sizes}
    require(decision['schema_version'] == 1 and decision['supported_threads'] == [1]
            and decision['promoted_native_dtype'] == 'complex128'
            and decision['direct_single_sizes_unchanged'] == [32768]
            and decision['dispatch_by_size'] == dispatch and decision['rationale'].strip()
            and report['large_ifft']['dispatch_by_size'] == dispatch
            and report['large_ifft']['cases'] == 108, 'Optimized IFFT dispatch decision differs')
    if 'enabled_sizes' in decision:
        require(set(decision['enabled_sizes']) == sizes and len(decision['enabled_sizes']) == 3,
                'Optimized IFFT enabled-size list differs')
    before = matrix['source_before']
    require(matrix['schema'] == 'torch-large-ifft-qualification-v6b' and matrix['state'] == 'complete'
            and matrix.get('finished_utc') and before == matrix['source_after']
            and before['head'] == FINAL_SOURCE and before['status'] == '' and before['root'] == source['source']
            and before['torchfft_sha256'] == source['changed_files_sha256']['pycbc/fft/torchfft.py']
            and matrix['threads'] == 1 and len(matrix['sizes']) == 3
            and {r['size'] for r in matrix['sizes']} == sizes, 'Optimized IFFT matrix is incomplete or stale')
    grid = set(itertools.product((7, 91, 812, 20260906), ('dense', 'banded', 'impulse'), (1e-12, 1., 1e12)))
    for row in matrix['sizes']:
        require(len(row['cases']) == 36 and {(c['seed'], c['pattern'], c['scale']) for c in row['cases']} == grid,
                'Optimized IFFT precision matrix omitted a case')
        for name in ('mkl_double_workspace', 'fftw_double_workspace_one_native_thread'):
            timing = row['timings'][name]
            samples = timing['steady_seconds']
            require(len(samples) == 9 and timing['native_library_threads'] == 1
                    and timing['bound_buffers_verified'] is True and timing['input_storage_preserved'] is True
                    and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in samples)
                    and timing['steady_median_seconds'] == statistics.median(samples)
                    and timing['steady_min_seconds'] == min(samples) and timing['steady_max_seconds'] == max(samples),
                    'Invalid optimized IFFT timing')
        require(row['timings']['mkl_double_workspace']['steady_median_seconds'] <
                row['timings']['fftw_double_workspace_one_native_thread']['steady_median_seconds'],
                'Enabled IFFT route lacks measured speedup')
        for case in row['cases']:
            gate = case['gates']['mkl_double_workspace']
            require(gate['passed'] is True and gate['bitwise_mkl_parity'] is True,
                    'Optimized IFFT failed native parity')
            for key in ('l2', 'max_abs'):
                candidate = case['errors']['mkl_double_workspace'][key]
                baseline = case['errors']['legacy_fftw_single'][key]
                require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                            for v in (candidate, baseline)) and candidate <= baseline,
                        'Optimized IFFT worsens legacy FFTW error')


def relative_path(text):
    require(isinstance(text, str), 'Repository path must be text')
    path = PurePosixPath(text)
    require(text and not path.is_absolute()
            and str(path) == text and '..' not in path.parts
            and not any(c in text for c in '\\\x00\n\r\t'),
            'Unsafe repository path: ' + repr(text))
    return text


def commit_sha(text):
    require(isinstance(text, str) and re.fullmatch('[0-9a-f]{40}', text),
            'Expected full Git SHA: ' + repr(text))
    return text


def bound(path, expected, inputs):
    path = path.resolve()
    item = record(path)
    require(item['sha256'] == expected, 'Reviewed input changed: ' + str(path))
    inputs[str(path)] = item
    return read_file(path)


def entries(root, head, paths):
    return {p: value(root, 'ls-tree', head, '--', p).split('\t', 1)[0]
            for p in paths}


def candidate_checks(root, files):
    plan = read_json(files['restack_plan'])
    stack = read_json(files['candidate_stack'])
    receipt = read_json(files['restack_execution'])
    if 'polish' in stack:
        info = stack['polish']
        require(info == receipt['polish'], 'Polish receipts differ')
        proof_path, script_path = Path(info['path']).resolve(), Path(info['script']).resolve()
        proof = read_json(proof_path)
        require(digest(read_file(proof_path)) == info['sha256']
                and digest(read_file(script_path)) == info['script_sha256'] == proof['script_sha256']
                and proof['schema'] == 'torch-publication-polish-v1' and proof['reviewed'] is True,
                'Polish proof is unbound or unreviewed')
        prior_files = dict(files)
        for role, record in proof['prior_inputs'].items():
            path = Path(record['path']).resolve()
            require(digest(read_file(path)) == record['sha256'], 'Prior polish input changed: ' + role)
            prior_files[role] = path
        require('polish' not in read_json(prior_files['candidate_stack']), 'Recursive polish is forbidden')
        prior, additional = candidate_checks(root, prior_files)
        previous_stack = read_json(prior_files['candidate_stack'])
        previous_receipt = read_json(prior_files['restack_execution'])
        require({k: v for k, v in stack.items() if k not in ('polish', 'candidates')}
                == {k: v for k, v in previous_stack.items() if k != 'candidates'}
                and {k: v for k, v in receipt.items() if k not in ('polish', 'candidates')}
                == {k: v for k, v in previous_receipt.items() if k != 'candidates'}
                and receipt['candidates'] == stack['candidates']
                and len(prior) == len(stack['candidates']) == len(proof['rows']) == 4,
                'Polish changed unrelated stack metadata')
        spec = importlib.util.spec_from_file_location('reviewed_polish', script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result, heads = [], {}
        for old, row, reviewed in zip(prior, stack['candidates'], proof['rows']):
            number = row['number']
            require(number == old['number'] == reviewed['number']
                    and all(row[k] == old[k] for k in ('branch', 'base', 'old_head'))
                    and row['candidate_branch'] == old['candidate_branch'] + '-final',
                    'Polished candidate metadata differs')
            parent = old['parent'] if number == 15 else heads[row['base']]
            require(row['parent'] == parent
                    and value(root, 'rev-list', '--parents', '-n', '1', row['head']) == row['head'] + ' ' + parent
                    and value(root, 'rev-parse', row['head'] + '^{tree}') == row['tree']
                    and value(root, 'rev-parse', 'refs/heads/' + row['candidate_branch']) == row['head']
                    and reviewed['before_head'] == old['head'] and reviewed['after_head'] == row['head']
                    and reviewed['before_tree'] == old['tree'] and reviewed['after_tree'] == row['tree'],
                    'Polished candidate identity differs')
            require(module.validate_delta(root, old['head'], row['head'], number) == reviewed['files'],
                    'Polish differs from the reviewed AST-equivalent cleanup')
            heads[row['branch']] = row['head']
            result.append(dict(row, key=old['key']))
        return result, additional + [proof_path, script_path, *prior_files.values()]
    plan_hash = digest(read_file(files['restack_plan']))
    require(plan['schema'] == 'torch-performance-fix-restack-v1'
            and stack['schema'] == 'torch-performance-fix-candidates-v1'
            and receipt['state'] == 'complete'
            and receipt['plan_sha256'] == stack['plan_sha256'] == plan_hash
            and receipt['candidates'] == stack['candidates'],
            'Restack is incomplete or does not match its plan')
    require(Path(plan['assembly']).resolve() == root
            and plan['input_head'] == FIX and plan['fix_source_chain'] == FIX_SOURCE_CHAIN
            and plan['input_branch'] == INPUT_BRANCH and plan['candidate_prefix'] == PREFIX
            and plan['code_files'] == CODE and plan['normalized_paths'] == sorted(SHARED)
            and plan['script_sha256'] == digest(read_file(files['restack_script']))
            and Path(plan['snapshot_path']).resolve() == files['snapshot']
            and plan['snapshot_sha256'] == digest(read_file(files['snapshot'])),
            'Wrong restack source, assembly or reviewed inputs')
    require(len(stack['candidates']) == len(plan['source_stack']) == len(STACK),
            'Expected four candidates')
    allow = read_json(Path(plan['allowlist_path']))
    require(digest(read_file(Path(plan['allowlist_path']))) == plan['allowlist_sha256']
            and allow['schema'] == 'torch-performance-fix-inputs-v1'
            and sorted(allow['paths']) == sorted(plan['files']), 'Supplementary allowlist changed')
    for path in plan['files']:
        relative_path(path)
        require(path.startswith('docs/') or path in
                {'.github/workflows/basic-tests.yml', '.github/workflows/torch-gpu.yml'},
                'Unexpected supplementary file: ' + path)
    updated = sorted(CODE + list(plan['files']))
    require(len(updated) == len(set(updated)), 'Duplicate updated file')
    require(changed(root, STACK[0]['sha'], FIX) == CODE
            and entries(root, FIX, CODE) == plan['fix_entries'], 'Corrected source differs')
    previous = STACK[0]['sha']
    for head in FIX_SOURCE_CHAIN:
        require(value(root, 'rev-list', '--parents', '-n', '1', head) == head + ' ' + previous,
                'Corrected source commit chain differs')
        previous = head
    integration_info = stack['integration']
    require(integration_info == receipt['integration'], 'CPU integration receipt differs')
    integration_path = Path(integration_info['path']).resolve()
    completion = Path(integration_info['completion_script']).resolve()
    failure = integration_path.with_name('restack-execution-conflict-v6.json')
    integration = read_json(integration_path)
    exceptions = {'.github/workflows/basic-tests.yml', 'docs/torch_optimizations.rst',
                  'pycbc/fft/torchfft.py', 'test/test_torch_cpu_fft_tuning.py',
                  'test/test_torch_large_ifft.py'}
    require(digest(read_file(integration_path)) == integration_info['sha256']
            and digest(read_file(completion)) == integration_info['completion_script_sha256']
            and digest(read_file(failure)) == integration['failed_execution_sha256']
            and integration['schema'] == 'torch-cpu-hotpath-integration-v1'
            and integration['reviewed'] is True and integration['source_head'] == FIX
            and integration['plan_sha256'] == plan_hash
            and integration['exception_paths'] == sorted(exceptions),
            'Unbound or unreviewed CPU integration')
    heads, result = {}, []
    for spec, prior, row in zip(STACK, plan['source_stack'], stack['candidates']):
        key, old = spec['key'], spec['sha']
        require(all(prior[k] == v for k, v in spec.items()), 'Historical stack differs: ' + key)
        require(all(row[k] == spec[k] for k in ('number', 'branch', 'base'))
                and row['old_head'] == old and row['candidate_branch'] == PREFIX + key,
                'Candidate metadata differs: ' + key)
        head = commit_sha(row['head'])
        parent = spec['parent'] if not result else heads[spec['base']]
        require(head != old and value(root, 'rev-parse', 'refs/heads/' + row['candidate_branch']) == head,
                'Candidate branch differs: ' + key)
        for actual, expected in ((head, parent), (old, spec['parent'])):
            require(value(root, 'rev-list', '--parents', '-n', '1', actual) == actual + ' ' + expected,
                    'Commit parent differs: ' + key)
        require(row['parent'] == parent and value(root, 'rev-parse', head + '^{tree}') == row['tree'],
                'Candidate tree differs: ' + key)
        cpu = spec['number'] == 17
        update_paths = integration['update_paths'] if cpu else updated
        update_patches = integration['main_update_patches'] if cpu else stack['main_update_patches']
        require(changed(root, old, head) == update_paths
                and {p: patch_hash(root, old, head, p) for p in update_paths} == update_patches,
                'Candidate update differs: ' + key)
        if cpu:
            require(integration['old_head'] == old and integration['parent'] == parent
                    and integration['tree'] == row['tree']
                    and update_paths == sorted(set(updated) | {'docs/torch_optimizations.rst',
                                                              'test/test_torch_cpu_fft_tuning.py'})
                    and all(update_patches[p] == stack['main_update_patches'][p]
                            for p in updated if p not in exceptions),
                    'CPU integration exceeds reviewed update scope')
        require(changed(root, spec['parent'], old) == prior['files']
                and {p: patch_hash(root, spec['parent'], old, p) for p in prior['files']} == prior['patches']
                and entries(root, old, prior['files']) == prior['entries'],
                'Frozen historical feature differs: ' + key)
        if result:
            feature_paths = integration['feature_paths'] if cpu else prior['files']
            feature_patches = integration['feature_patches'] if cpu else prior['patches']
            require(changed(root, parent, head) == feature_paths
                    and {p: patch_hash(root, parent, head, p) for p in feature_paths} == feature_patches,
                    'Dependent feature patch changed: ' + key)
            if cpu:
                require(feature_paths == sorted(set(prior['files']) | {'test/test_torch_large_ifft.py'})
                        and all(feature_patches[p] == prior['patches'][p]
                                for p in prior['files'] if p not in exceptions)
                        and entries(root, head, feature_paths) == integration['feature_entries']
                        and patch_hash(root, parent, head, '.github/workflows/basic-tests.yml', 0)
                        == patch_hash(root, spec['parent'], old, '.github/workflows/basic-tests.yml', 0)
                        == integration['workflow_feature_u0_sha256'],
                        'CPU feature integration differs from reviewed changes')
            nonshared = [p for p in prior['files'] if p not in SHARED
                         and (not cpu or p not in exceptions)]
            require(entries(root, old, nonshared) == entries(root, head, nonshared),
                    'Dependent feature bytes/modes changed: ' + key)
        else:
            require(entries(root, head, CODE) == plan['fix_entries'], 'Main corrected source bytes differ')
            for path, info in plan['files'].items():
                data = git(root, 'show', head + ':' + path)
                mode = '100755' if info['mode'] == 0o755 else '100644'
                require(info['mode'] in (0o644, 0o755) and digest(data) == info['sha256']
                        and entries(root, head, [path])[path].startswith(mode + ' blob '),
                        'Supplementary candidate input changed: ' + path)
        reverse_paths = (['pycbc/fft/torchfft.py'] if spec['number'] == 16 else
                         ['pycbc/fft/torchfft.py', 'test/test_torch_large_ifft.py',
                          'pycbc/filter/matchedfilter.py', 'pycbc/vetoes/chisq_torch.py'] if cpu else [])
        for path in reverse_paths:
            feature = git(root, 'diff', '--no-ext-diff', '--no-textconv', '--no-color',
                          '--no-renames', '--binary', '--full-index', '--unified=0',
                          '--diff-algorithm=myers', '--no-indent-heuristic',
                          '--src-prefix=a/', '--dst-prefix=b/', parent, head, '--', path)
            reconstructed = reverse_file_patch(git(root, 'show', head + ':' + path), feature)
            require(reconstructed == git(root, 'show', FINAL_SOURCE + ':' + path),
                    'Feature reversal does not recover the complete tested source: ' + path)
        heads[spec['branch']] = head
        result.append(dict(row, key=key))
    bundle = Path(stack['backup_bundle']).resolve()
    require(digest(read_file(bundle)) == stack['backup_bundle_sha256'] == receipt['backup_bundle_sha256'],
            'Historical backup bundle changed')
    git(root, 'bundle', 'verify', str(bundle))
    return result, [Path(plan['allowlist_path']).resolve(), bundle,
                    integration_path, completion, failure]


def archive_blob(archive, version, path, sha256):
    require(version in ('a', 'b'), 'Evidence must be in archive A or B')
    relative_path(path)
    root, commit = Path(archive['root']), archive[version]
    entry = value(root, 'ls-tree', commit, '--', path)
    require(re.fullmatch(r'100(?:644|755) blob [0-9a-f]{40}\t' + re.escape(path), entry),
            'Evidence must identify an exact committed regular file: ' + path)
    data = git(root, 'show', commit + ':' + path)
    require(digest(data) == sha256, 'Archive evidence hash differs: ' + path)
    return data


def asserted(data, checks, name):
    require(isinstance(checks, list) and checks, 'Missing reviewed assertions: ' + name)
    for assertion in checks:
        actual = json_pointer(data, assertion['pointer'])
        require(type(actual) is type(assertion['equals']) and actual == assertion['equals'],
                'Failed reviewed assertion: ' + name + ' ' + assertion['pointer'])


def archive_checks(context, context_path, updates):
    archive = dict(context['archive'])
    root = (context_path.parent / archive['root']).resolve()
    archive['root'] = str(root)
    clean_root(root)
    require(archive['branch'] == ARCHIVE_BRANCH, 'Wrong archive branch')
    a, b = commit_sha(archive['a']), commit_sha(archive['b'])
    require(len({ARCHIVE_BASE, a, b}) == 3, 'Archive A and B must be new distinct commits')
    git(root, 'merge-base', '--is-ancestor', ARCHIVE_BASE, a)
    git(root, 'merge-base', '--is-ancestor', a, b)
    for before, after, field, prefixes in (
            (ARCHIVE_BASE, a, 'a_changes', ('performance-fix/', 'inspiral-reference-20260906/')),
            (a, b, 'b_changes', ('publication-qualification/',))):
        paths = archive[field]
        require(isinstance(paths, list) and paths and sorted(set(paths)) == paths
                and changed(root, before, after) == paths, 'Archive change inventory differs: ' + field)
        for path in paths:
            relative_path(path)
            require(path in ('README.md', 'SHA256SUMS') or path.startswith(prefixes),
                    'Archive changed outside its supplement: ' + path)
            require(bool(value(root, 'ls-tree', after, '--', path)), 'Archive removed a reviewed path: ' + path)
    # No overwrites of previous supplements: only root indices may change.
    historical = set(value(root, 'ls-tree', '-r', '--name-only', ARCHIVE_BASE).splitlines())
    require(not (historical & set(archive['a_changes'])) - {'README.md', 'SHA256SUMS'},
            'Archive A overwrites historical evidence')
    a_paths = set(value(root, 'ls-tree', '-r', '--name-only', a).splitlines())
    require(not (a_paths & set(archive['b_changes'])) - {'README.md', 'SHA256SUMS'},
            'Archive B overwrites existing evidence')
    require(any(p.startswith('performance-fix/') for p in archive['a_changes'])
            and any(p.startswith('inspiral-reference-20260906/') for p in archive['a_changes']),
            'Archive A lacks both scientific supplements')
    require(value(root, 'rev-parse', 'HEAD') == b
            and value(root, 'branch', '--show-current') == ARCHIVE_BRANCH, 'Archive checkout is not at B')
    archive['trees'] = {k: value(root, 'rev-parse', archive[k] + '^{tree}') for k in ('a', 'b')}
    archive['origin_urls'] = remote_urls(root)
    archive['remote_refs'] = remote_refs(root, archive['origin_urls']['push'][0], [ARCHIVE_BRANCH])
    require(archive['remote_refs'] == {'refs/heads/' + ARCHIVE_BRANCH: b}, 'Archive B is not published')
    checks = context['checks']
    needed = {'archive_finalization', 'docs', 'quality', 'cpu_integration'} | {
        row['key'] + '-' + suffix for row in updates
        for suffix in ('source-before', 'source-after', 'reviewed-source')}
    require(needed <= set(checks), 'Missing final archive/quality checks: ' + str(sorted(needed - set(checks))))
    documents = {}
    for name, check in checks.items():
        require(check['archive'] == 'b' and check['path'].startswith('publication-qualification/'),
                'Final qualification must be archived under B: ' + name)
        data = archive_blob(archive, 'b', check['path'], check['sha256'])
        documents[name] = json.loads(data)
        asserted(documents[name], check['assertions'], name)
    quality, docs = documents['quality'], documents['docs']
    cpu_tests = documents['cpu_integration']
    cpu = next(row for row in updates if row['number'] == 17)
    require(cpu_tests['state'] == 'complete' and cpu_tests['passed'] is True
            and cpu_tests['returncode'] == 0 and cpu_tests.get('finished_utc')
            and cpu_tests['source_before'] == cpu_tests['source_after']
            and cpu_tests['source_before']['head'] == cpu['head']
            and cpu_tests['source_before']['tree'] == cpu['tree']
            and not cpu_tests['source_before']['status']
            and checks['cpu_integration']['sha256'] in quality['inputs_sha256'].values(),
            'CPU integration tests are incomplete or not bound to quality qualification')
    require(quality['state'] == 'complete' and quality['passed'] is True and quality.get('finished_utc')
            and quality['unit_source_info'] == {'commit': FIX, 'status': ''}
            and quality['commands'] and all(type(c['returncode']) is int and c['returncode'] == 0
                                           for c in quality['commands']), 'Quality qualification is incomplete')
    candidate_hash = context['inputs']['candidate_stack']['sha256']
    require(candidate_hash in quality['inputs_sha256'].values(),
            'Quality qualification did not bind this candidate stack')
    labels = {c['label'] for c in quality['commands']}
    require({row['key'] + '-qlty' for row in updates} <= labels
            and {row['key'] + '-units' for row in updates} <= labels
            and {'main-flake8-bin', 'main-flake8-pycbc', 'main-flake8-test'} <= labels,
            'Missing required candidate lint commands')
    for row in updates:
        before = documents[row['key'] + '-source-before']
        after = documents[row['key'] + '-source-after']
        require(before == after and before['head'] == row['head']
                and before['tree'] == row['tree'] and not before['status'],
                'Quality candidate snapshot differs: ' + row['key'])
        candidate = next(p for p in quality['candidates'] if p['number'] == row['number'])
        tests = next(c for c in quality['commands'] if c['label'] == row['key'] + '-units')
        require(candidate['head'] == row['head'] and candidate['tree'] == row['tree']
                and len(candidate['test_files']) >= 23
                and len(set(candidate['test_files'])) == len(candidate['test_files'])
                and tests['command'][9:] == candidate['test_files']
                and tests['command'][:3] == ['taskset', '-c', '8']
                and tests['command'][4:9] == ['-m', 'pytest', '-q', '-p', 'no:cacheprovider']
                and tests['returncode'] == 0 and tests['finished_utc'],
                'Fresh final-commit unit checks are missing: ' + row['key'])
    require(type(docs['returncode']) is int and docs['returncode'] == 0 and docs.get('finished_utc')
            and docs['pages'] == list(DOC_PAGES) and docs['images'] == sum(map(len, COLLECTIONS.values()))
            and set(docs['page_sha256']) == set(DOC_PAGES)
            and set(docs['manifest_sha256']) == set(COLLECTIONS)
            and docs['image_hashes_verified'] is True and docs['downloadable_manifests_verified'] is True,
            'Documentation qualification is incomplete')
    main = updates[0]['head']
    for field in ('page_sha256', 'manifest_sha256'):
        for path, sha in docs[field].items():
            relative_path(path)
            path = 'docs/' + path + '.rst' if field == 'page_sha256' else 'docs/images/' + path + '/manifest.json'
            require(digest(git(Path(context['_root']), 'show', main + ':' + path)) == sha,
                    'Documentation receipt source differs: ' + path)
    for collection, names in COLLECTIONS.items():
        folder = 'docs/images/' + collection + '/'
        manifest = json.loads(git(Path(context['_root']), 'show', main + ':' + folder + 'manifest.json'))
        require(len(manifest['figures']) == len(names)
                and {r['file'] for r in manifest['figures']} == set(names),
                'Documentation figure inventory differs: ' + collection)
        if collection != 'torch-benchmarks-20260906':
            require(manifest['evidence_commit'] == a, 'New documentation must pin archive A')
        if collection == 'torch-inspiral-reference-20260906':
            require(manifest['measured_source_commit'] == BASELINE_SOURCE,
                    'Reference figure manifest changed its frozen baseline source')
        if collection == 'torch-inspiral-optimized-20260906':
            require(manifest['measured_source_commit'] == FINAL_SOURCE
                    and manifest['baseline_source_commit'] == BASELINE_SOURCE,
                    'Optimized figure manifest has wrong before/after sources')
        for figure in manifest['figures']:
            require(digest(git(Path(context['_root']), 'show', main + ':' + folder + figure['file']))
                    == figure['sha256'], 'Candidate figure bytes differ: ' + figure['file'])
    return archive


def body_check(text, prior, number, head):
    require(text.endswith('\n') and '\r' not in text and '\x00' not in text
            and len(text.encode()) <= 60_000, f'PR #{number}: invalid body encoding/size')
    require(not re.search(r'\b(?:TODO|TBD|FIXME|PLACEHOLDER|NEW_MAIN_HEAD|NEW_HEAD|EVIDENCE_COMMIT)\b'
                          r'|\{\{|\}\}|<\s*(?:parent|head|sha|username)\s*>', text, re.I),
            f'PR #{number}: unresolved placeholder')
    for heading in HEADINGS:
        require(text.count('## ' + heading + '\n') == 1, f'PR #{number}: missing/duplicate template heading')
    for field in ('This is a:', 'This change affects:', 'This change changes:', 'This change:'):
        require(field in text, f'PR #{number}: missing standard information')
    require(text.count(COC) == text.count(AI_NOTE) == 1
            and 'This PR was created by AI Gareth' in text and '`agent-assisted`' in text,
            f'PR #{number}: required operator/AI/Code of Conduct text missing')
    require(not any(re.match(r'\s*[-*]\s*\[[xX]\]', line) and 'conduct' in line.lower()
                    for line in text.splitlines()), f'PR #{number}: Code of Conduct was checked')
    require(text.count('<!-- stack-index -->') == prior['body'].count('<!-- stack-index -->') == 1
            and text.split('<!-- stack-index -->')[1] == prior['body'].split('<!-- stack-index -->')[1],
            f'PR #{number}: stack index/footer changed')
    line = (f"Head branch: `{prior['headRefName']}`. Head commit: `{head}`. "
            f"Base branch: `{prior['baseRefName']}`.")
    require(line in text and not re.search(r'!\[[^\]]*\]\(|<img\b', text, re.I),
            f'PR #{number}: head/base attribution differs or embedded image remains')
    require(text.split('## Testing performed\n')[1].split('<!-- stack-index -->')[0].strip(),
            f'PR #{number}: missing testing content')


def body_checks(files, archive, updates, before, inputs):
    context_path = files['body_context']
    context = read_json(context_path)
    manifest = read_json(files['bodies_manifest'])
    require(manifest['schema'] == 'torch-performance-fix-bodies-v1'
            and context['schema'] == 'torch-performance-fix-body-context-v1'
            and context['reviewed'] is True, 'Wrong/unreviewed body inputs')
    expected_bound = {files[k] for k in ('body_context', 'body_generator', 'snapshot', 'candidate_stack', 'template')}
    body_inputs = manifest['inputs_sha256']
    require(expected_bound <= {Path(p).resolve() for p in body_inputs}, 'Body manifest lacks required input bindings')
    for path, sha in body_inputs.items():
        bound(Path(path), sha, inputs)
    require(context['snapshot_sha256'] == inputs[str(files['snapshot'])]['sha256']
            and context['candidate_stack_sha256'] == inputs[str(files['candidate_stack'])]['sha256'],
            'Body context snapshot/candidates differ')
    documents, evidence, locations = {}, {}, {}
    for item in context['evidence']:
        require(item['id'] not in evidence, 'Duplicate evidence identifier')
        evidence[item['id']] = item
        path = (context_path.parent / item['path']).resolve()
        raw = bound(path, item['sha256'], inputs)
        require(body_inputs.get(str(path)) == item['sha256'], 'Evidence is not bound by body manifest')
        url = urlparse(item['url'])
        prefix = '/' + REPOSITORY + '/'
        parts = url.path.removeprefix(prefix).split('/', 2)
        require(url.scheme == 'https' and url.netloc == 'github.com' and url.path.startswith(prefix)
                and len(parts) == 3 and parts[0] in ('blob', 'tree') and not url.query and not url.fragment,
                'Body evidence URL must identify an immutable archive file')
        version = {archive['a']: 'a', archive['b']: 'b'}.get(parts[1])
        require(version is not None, 'Body evidence must pin reviewed archive A or B')
        require(archive_blob(archive, version, unquote(parts[2]), item['sha256']) == raw,
                'Local/archived body evidence differs')
        locations[item['id']] = (version, unquote(parts[2]))
        if item['kind'] != 'artifact' or item.get('assertions'):
            documents[item['id']] = json.loads(raw)
            asserted(documents[item['id']], item.get('assertions'), item['id'])
    verify_primary_reference(context, evidence, documents)
    verify_optimized_reference(context, evidence, documents, locations, archive)
    for head, identity in context['source_identities'].items():
        require(value(Path(archive['_source_root']), 'rev-parse', commit_sha(head) + '^{tree}')
                == commit_sha(identity['tree']), 'Measured source tree differs: ' + head)
    require([r['number'] for r in manifest['prs']] == list(NUMBERS), 'Wrong body inventory')
    directory = files['bodies_manifest'].parent
    require({p.name for p in directory.iterdir()} == {'manifest.json'} | {
        f'pr-{n}.{ext}' for n in NUMBERS for ext in ('md', 'diff')}, 'Unexpected bodies directory inventory')
    heads = {row['number']: row['head'] for row in updates}
    bodies = {}
    for row in manifest['prs']:
        n, prior = row['number'], before[row['number']]
        head = heads.get(n, prior['headRefOid'])
        expected = dict(number=n, url=prior['url'], title=prior['title'], base=prior['baseRefName'],
                        branch=prior['headRefName'], isDraft=prior['isDraft'], head=head,
                        old_head=prior['headRefOid'], body_file=f'pr-{n}.md',
                        body_sha256=row['body_sha256'], required_labels=['agent-assisted'])
        require(row == expected, f'PR #{n}: body manifest metadata differs')
        path = directory / row['body_file']
        raw = bound(path, row['body_sha256'], inputs)
        text = raw.decode('utf-8')
        body_check(text, prior, n, head)
        require(text != prior['body'], f'PR #{n}: body has no update')
        diff = ''.join(difflib.unified_diff(prior['body'].splitlines(True), text.splitlines(True),
                                          fromfile=f'PR-{n}-snapshot', tofile=f'pr-{n}.md')).encode()
        diff_path = directory / f'pr-{n}.diff'
        require(read_file(diff_path) == diff, f'PR #{n}: review diff differs')
        inputs[str(diff_path)] = record(diff_path)
        bodies[str(n)] = record(path, raw)
    return bodies


def push_command(root, url, updates):
    return ['git', '-C', str(root), '-c', 'push.followTags=false', 'push', '--atomic',
            '--porcelain', '--no-follow-tags', '--recurse-submodules=no',
            *[f"--force-with-lease=refs/heads/{r['branch']}:{r['old_head']}" for r in updates],
            url, *[f"{r['head']}:refs/heads/{r['branch']}" for r in updates]]


def inspect(args):
    inputs = {str(Path(__file__).resolve()): record(Path(__file__).resolve()),
              str(args.context): record(args.context)}
    context = read_json(args.context)
    require(context['schema'] == 'torch-performance-fix-publication-context-v2'
            and context['reviewed'] is True, 'Explicit reviewed publication context required')
    roles = {'restack_plan', 'restack_execution', 'restack_script', 'candidate_stack',
             'snapshot', 'body_context', 'body_generator', 'bodies_manifest', 'template'}
    require(set(context['inputs']) == roles, 'Wrong publication input role inventory')
    files = {}
    for role, info in context['inputs'].items():
        files[role] = (args.context.parent / info['path']).resolve()
        bound(files[role], info['sha256'], inputs)
    root = Path(read_json(files['restack_plan'])['assembly']).resolve()
    clean_root(root)
    updates, additional = candidate_checks(root, files)
    for path in additional:
        inputs[str(path)] = record(path)
    source_urls = remote_urls(root)
    rows = read_json(files['snapshot'])
    require([r['number'] for r in rows] == list(NUMBERS), 'Wrong PR snapshot inventory')
    before = {r['number']: metadata(r) for r in rows}
    identities = [(r['number'], r['branch'], r['base'], r['sha']) for r in STACK] + UNCHANGED
    for n, branch, base, head in identities:
        prior = before[n]
        require((prior['headRefName'], prior['baseRefName'], prior['headRefOid']) == (branch, base, head)
                and prior['state'] == 'OPEN' and prior['author']['login'] == 'xangma'
                and isinstance(prior['isDraft'], bool) and prior['url'] == WEB + '/pull/' + str(n)
                and 'agent-assisted' in [label['name'] for label in prior['labels']],
                f'PR #{n}: frozen identity/state/label differs')
    expected_refs = {'refs/heads/' + row['headRefName']: row['headRefOid'] for row in before.values()}
    expected_refs['refs/heads/torch-pr10-inference'] = STACK[0]['parent']
    actual = remote_refs(root, source_urls['push'][0], [p.removeprefix('refs/heads/') for p in expected_refs])
    require(actual == expected_refs, 'Origin refs differ from reviewed historical heads')
    context['_root'] = str(root)
    archive = archive_checks(context, args.context, updates)
    require(root != Path(archive['root']) and not OUT.is_relative_to(root)
            and not OUT.is_relative_to(Path(archive['root'])), 'Publication outputs must be outside both checkouts')
    archive['_source_root'] = str(root)
    bodies = body_checks(files, archive, updates, before, inputs)
    del archive['_source_root']
    live = {str(n): live_pr(n) for n in NUMBERS}
    require(live == {str(n): before[n] for n in NUMBERS}, 'Live PR metadata/body differs from frozen snapshot')
    plan = dict(schema='torch-performance-fix-publication-inputs-v1', repository=GH_REPOSITORY,
                root=str(root), root_head=value(root, 'rev-parse', 'HEAD'),
                root_branch=value(root, 'branch', '--show-current'), inputs=inputs, bodies=bodies,
                source_origin_urls=source_urls, source_remote_refs=actual, updates=updates,
                evidence=archive, live_before=live, push_command=push_command(root, source_urls['push'][0], updates),
                pr_edit_commands={str(n): edit_command(n) for n in NUMBERS})
    assert_local_inputs(plan)
    return plan


def assert_local_inputs(plan):
    for group in ('inputs', 'bodies'):
        for name, frozen in plan[group].items():
            require(record(Path(frozen['path'])) == frozen, 'Frozen local input changed: ' + name)
    root, archive = Path(plan['root']), plan['evidence']
    require(remote_urls(root) == plan['source_origin_urls'], 'Source origin URL changed')
    require(value(root, 'rev-parse', 'HEAD') == plan['root_head']
            and value(root, 'branch', '--show-current') == plan['root_branch'], 'Source checkout moved')
    for row in plan['updates']:
        require(value(root, 'rev-parse', 'refs/heads/' + row['candidate_branch']) == row['head'],
                'Candidate branch moved: ' + row['candidate_branch'])
    evidence_root = Path(archive['root'])
    require(value(evidence_root, 'rev-parse', 'HEAD') == archive['b']
            and value(evidence_root, 'branch', '--show-current') == archive['branch']
            and remote_urls(evidence_root) == archive['origin_urls'], 'Archive checkout/origin moved')
    clean_root(root)
    clean_root(evidence_root)


def assert_remote_inputs(plan, refs):
    actual = remote_refs(Path(plan['root']), plan['source_origin_urls']['push'][0],
                         [p.removeprefix('refs/heads/') for p in refs])
    require(actual == refs, 'Remote source refs differ')
    archive = plan['evidence']
    actual = remote_refs(Path(archive['root']), archive['origin_urls']['push'][0], [archive['branch']])
    require(actual == archive['remote_refs'], 'Published archive B moved')


def final_refs(plan):
    result = dict(plan['source_remote_refs'])
    for row in plan['updates']:
        result['refs/heads/' + row['branch']] = row['head']
    return result


def intended_pr(plan, number):
    expected = dict(plan['live_before'][str(number)])
    for row in plan['updates']:
        if row['number'] == number:
            expected['headRefOid'] = row['head']
    expected['body'] = read_file(Path(plan['bodies'][str(number)]['path'])).decode('utf-8')
    return expected


def body_states(plan, journal):
    """Accept only this attempt's old/intended bodies, preserving other metadata."""
    observed = {}
    for number in NUMBERS:
        prior, expected = plan['live_before'][str(number)], intended_pr(plan, number)
        actual = live_pr(number)
        require(actual['body'] in (prior['body'], expected['body']),
                f'PR #{number}: unrelated body edit; manual inspection required')
        if actual['body'] == expected['body']:
            require(any(op['name'] == f'pr-{number}' and op['command'] == edit_command(number)
                        and op.get('stdin_sha256') == plan['bodies'][str(number)]['sha256']
                        for op in journal['operations']),
                    f'PR #{number}: intended body appeared outside this journal')
        expected['body'] = actual['body']
        compare_pr(actual, expected, number)
        observed[str(number)] = actual
    return observed


def mutation(journal, name, command, cwd, input_bytes=None):
    log = OUT / f"publish-{len(journal['operations']) + 1:03d}-{name}.log"
    entry = dict(name=name, command=command, cwd=str(cwd), log=str(log), started_utc=now(), state='attempting')
    if input_bytes is not None:
        entry.update(stdin_sha256=digest(input_bytes), stdin_bytes=len(input_bytes))
    journal['operations'].append(entry)
    journal['state'] = 'attempting-' + name
    write_json(RECEIPT, journal)
    try:
        run(command, cwd, log=log, input_bytes=input_bytes)
        entry.update(state='command-returned-success', finished_utc=now(), log_sha256=digest(read_file(log)))
    except BaseException as error:
        entry.update(state='failed-or-uncertain', error=repr(error), finished_utc=now())
        if log.is_file():
            entry['log_sha256'] = digest(read_file(log))
        raise
    finally:
        write_json(RECEIPT, journal)


def execute(args, frozen, resume=False):
    plan, plan_record = frozen['inputs'], record(args.plan)
    require(read_json(args.plan) == frozen, 'Reviewed publication plan changed')
    if resume:
        journal = read_json(RECEIPT)
        require(journal['schema'] == 'torch-performance-fix-publication-execution-v1'
                and journal['plan'] == plan_record and journal['reviewed_plan'] == frozen,
                'Resume journal belongs to different reviewed inputs')
        pushes = [op for op in journal['operations'] if op['name'] == 'atomic-push']
        require(len(pushes) == 1 and pushes[0]['command'] == plan['push_command'],
                'Resume requires exactly one original atomic push attempt')
    else:
        journal = dict(schema='torch-performance-fix-publication-execution-v1', started_utc=now(),
                       host=socket.gethostname(), pid=os.getpid(), plan=plan_record, reviewed_plan=frozen,
                       state='checking-before-push', operations=[], verified_prs=[])
        write_json(RECEIPT, journal, exclusive=True)
    try:
        assert_local_inputs(plan)
        if not resume:
            assert_remote_inputs(plan, plan['source_remote_refs'])
            for number in NUMBERS:
                require(live_pr(number) == plan['live_before'][str(number)],
                        f'PR #{number}: pre-push metadata changed')
            require(record(args.plan) == plan_record, 'Reviewed plan changed before push')
            mutation(journal, 'atomic-push', plan['push_command'], Path(plan['root']))
            assert_remote_inputs(plan, final_refs(plan))
            journal['metadata_after_push'] = {}
            for number in NUMBERS:
                expected = intended_pr(plan, number)
                expected['body'] = plan['live_before'][str(number)]['body']
                journal['metadata_after_push'][str(number)] = read_expected(
                    number, expected, plan['live_before'][str(number)]['headRefOid'])
            journal['state'] = 'push-verified'
            write_json(RECEIPT, journal)
        # Resume can recover an uncertain push only if all four refs already
        # match exactly. It never issues the push command a second time.
        assert_remote_inputs(plan, final_refs(plan))
        states = body_states(plan, journal)
        if resume:
            journal.setdefault('resumptions', []).append(dict(utc=now(), metadata=states,
                                                              host=socket.gethostname(), pid=os.getpid()))
            write_json(RECEIPT, journal)
        for number in NUMBERS:
            assert_local_inputs(plan)
            assert_remote_inputs(plan, final_refs(plan))
            states = body_states(plan, journal)
            intended = intended_pr(plan, number)
            if states[str(number)]['body'] != intended['body']:
                body_bytes = read_file(Path(plan['bodies'][str(number)]['path']))
                require(digest(body_bytes) == plan['bodies'][str(number)]['sha256']
                        and record(args.plan) == plan_record, 'Reviewed body/plan changed before edit')
                mutation(journal, f'pr-{number}', edit_command(number), OUT, input_bytes=body_bytes)
            observed = read_expected(number, intended)
            journal['verified_prs'] = [row for row in journal['verified_prs'] if row['number'] != number]
            journal['verified_prs'].append(dict(number=number, metadata=observed,
                                               body_sha256=digest(observed['body'].encode()), verified_utc=now()))
            journal['state'] = f'verified-pr-{number}'
            write_json(RECEIPT, journal)
            print(f'Verified PR #{number}: exact body, label, title, base, head and draft status.', flush=True)
        assert_local_inputs(plan)
        assert_remote_inputs(plan, final_refs(plan))
        require(record(args.plan) == plan_record, 'Reviewed plan changed during publication')
        journal['final_metadata'] = {str(n): read_expected(n, intended_pr(plan, n)) for n in NUMBERS}
        journal.update(state='completed', completed_utc=now())
        write_json(RECEIPT, journal)
        print('Published and verified four branch heads and seven PR bodies. Receipt:', RECEIPT)
    except BaseException as error:
        journal.update(state='failed', error=repr(error), failed_utc=now(),
                       recovery='Inspect receipt/logs and live state. --resume-bodies accepts only this plan, '
                       'all four already-published new refs, and this journal\'s old/intended bodies. '
                       'No automatic push retry or rollback.')
        write_json(RECEIPT, journal)
        raise


@contextmanager
def lock():
    with LOCK.open('x') as stream:
        stream.write(json.dumps(dict(pid=os.getpid(), host=socket.gethostname(), started_utc=now())) + '\n')
    try:
        yield
    finally:
        LOCK.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--context', type=Path, required=True, help='Explicit reviewed archive/body/input context')
    parser.add_argument('--plan', type=Path, default=OUT / 'publish-plan.json')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--execute', action='store_true')
    actions.add_argument('--resume-bodies', action='store_true')
    args = parser.parse_args()
    args.context, args.plan = args.context.resolve(), args.plan.resolve()
    require(args.plan.parent == OUT and args.plan.name.startswith('publish-plan')
            and args.plan.suffix == '.json', 'Plan must be publication/publish-plan*.json')
    with lock():
        if args.resume_bodies:
            require(RECEIPT.is_file(), 'No original publication journal exists')
            frozen = read_json(args.plan)
            require(frozen['schema'] == 'torch-performance-fix-publication-plan-v1'
                    and frozen['context'] == str(args.context), 'Wrong publication plan/context')
            execute(args, frozen, resume=True)
            return
        require(not RECEIPT.exists() and not list(OUT.glob('publish-*.log')),
                'A publication attempt exists; inspect it before explicit --resume-bodies')
        if args.execute:
            frozen = read_json(args.plan)
            require(frozen['schema'] == 'torch-performance-fix-publication-plan-v1'
                    and frozen['context'] == str(args.context), 'Wrong publication plan/context')
            current = inspect(args)
            require(current == frozen['inputs'], 'Preflight differs from reviewed plan; no mutation performed')
            execute(args, frozen)
        else:
            require(not args.plan.exists(), 'Plan exists; preserve it and choose a new plan path')
            current = inspect(args)
            frozen = dict(schema='torch-performance-fix-publication-plan-v1', prepared_utc=now(),
                          context=str(args.context), inputs=current,
                          execution_order=['atomic four-ref push', *[f'PR #{n}: body and required label' for n in NUMBERS]])
            write_json(args.plan, frozen, exclusive=True)
            print('Read-only preflight passed. Review plan, bodies and actual validation before --execute:', args.plan)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        print('STOP:', repr(error), file=sys.stderr)
        sys.exit(2)
