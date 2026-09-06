#!/usr/bin/env python3
"""Qualify four final publication heads after timing; --plan-only never writes.

No builds, fixes, source edits, publication, or baseline attribution. Every tool
exit code is retained; completed checks with findings yield a nonzero exit.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import time
import traceback

ROOT = Path('/home/xangma/pycbc-torch-performance-fix-20260906')
PYTHON = '/home/xangma/pycbc-torch-split-20260905/venv/bin/python'
QLTY = '/home/xangma/pycbc-torch-finish-20260904/qlty/qlty-x86_64-unknown-linux-gnu/qlty'
BASELINE = 'dfd42bf76766cadca0eecf609a1eaeac73534676'
REFERENCE = 'a4d77a6d1863c0515e8dace64c5609b63d40b51e'
INSPIRAL_SHA256 = 'd4af378d77aa5f66bcc018db32fe372360e53b22542e539ea3a2e53d31d8fa8a'
UNIT_TESTS = (
    'test/test_torch_decompress_cpu.py', 'test/test_torch_large_ifft.py',
    'test/test_decompress.py', 'test/test_torch_fft_cpu_native.py',
    'test/test_torch_fft_writes.py', 'test/test_torch_fft_cuda_workspace.py',
    'test/test_scheme_runtime.py', 'test/test_scheme_selection.py',
    'test/test_matchedfilter.py', 'test/test_chisq.py',
    'test/test_psd.py', 'test/test_strain.py',
    'test/test_sigmasq_series_precision.py', 'test/test_chisq_precision.py',
    'test/test_strain_psd_precision.py', 'test/test_torch_chisq_cpu_optimization.py',
    'test/test_torch_chisq_sparse_dispatch.py', 'test/test_torch_filter_pipeline.py',
    'test/test_torch_psd_pipeline.py', 'test/test_torch_psd_protocol.py',
    'test/test_torch_versioned_data_psd.py',
    'test/test_torch_matchedfilter_cpu_optimization.py',
)
REVIEWED_SHA256 = {
    'pycbc/fft/torchfft.py': '9123222316047a11652e2207e1b006f9f4a17e850d89d2d8b4b3cd081072ca5f',
    'pycbc/waveform/decompress_torch.py': '9710a30767edeacb68f70178534c215a6668e2d6ec12c389dbb50d73f52953b4',
    'test/test_torch_decompress_cpu.py': '90c9f428dee12ac9d0d1fb3f3ac9a9866e9ee52637e626da6cea3e2110b9199a',
    'test/test_torch_large_ifft.py': 'fd65930d7ee5786b91a7a21f4daf42591860fb098e279f7490f9290e72d82e50',
    'test/test_decompress.py': '0dde129f3c8168f3789bc63f45aa2971b0d2b6fc006c9873c0eed7fefafefa49',
    'test/test_torch_fft_cpu_native.py': 'bba8b779eb5f70055a678962482ff33a757fcfb1a7711fbac8ba2ae3a9ebeece',
    'test/test_torch_fft_writes.py': '288123508aeedf242c6515968013fe77af06a8251ad2898b08d897d4547f75a5',
    'test/test_torch_fft_cuda_workspace.py': 'f06f3400362e68f1684f3d2271a120476fe36b7c269877ae5bea8a623755bc82',
    'bin/pycbc_inspiral': INSPIRAL_SHA256,
    'pycbc/scheme.py': '27a0704bea2156e946e29e4437363036da21a05a8a98b403f93e605b75e9f5f0',
    'test/test_scheme_runtime.py': '45ac62167a6a3c503391f56bbbfc21dd982fbb58f05448b22c3dc52b3aca02cb',
    'test/test_scheme_selection.py': '2e2c754335a18fc18619d2271466948535f84c53313e0a2d96881042e3c59b36',
    'pycbc/filter/matchedfilter.py': '5941534911809626e15ffde451759129e74b520630efc4a50ba168652963f786',
    'pycbc/vetoes/chisq_torch.py': '6f61bd09869c86441b717ce28a62ba1532b24f273bce73a226268f269644cf84',
    'pycbc/vetoes/chisq.py': 'b27366b495a6fc46f0b742930ab1d74caf44286c99715fb30712f755526ac71c',
    'pycbc/psd/__init__.py': '01c9eb05c8729e4362f356d2722fd25e3475e4485a9da574a9439b29309c47ab',
    'pycbc/strain/strain.py': '889a3c1adb2a5e391ec94355a2a789b7b21f46759af7134ff906668b760da10f',
    'test/test_sigmasq_series_precision.py': 'a294cbbe29ea4f4120b3ef38e2c696bdc4708391c813c3356cbac310dcbc691a',
    'test/test_chisq_precision.py': '917d98d5b2db0920f76d22d74db8152fbebfde646c2db04b2ca887d3add55be6',
    'test/test_strain_psd_precision.py': 'cc6380df5aad6b8b25f88cd194c9786c30e4d9eaaffe3f756ff9a129f95a8134',
}
KEYS = {15: 'pr11', 19: 'format-fft', 16: 'fft-followup', 17: 'cpu-followup'}
PREFIX = 'codex/torch-performance-fix-20260906-'
LINKS = ('logs', 'out', 'plugin_cachedir', 'results')
CPU_FEATURE_PATCH_SHA256 = {
    'pycbc/fft/torchfft.py': '1ea6322b454a84ef3f80a90ec6268af774fb6ebba57daa02711a7001386de20f',
    'test/test_torch_large_ifft.py': 'a6b4bd3e9750a4756736328e1f18546a919616c80e6799cbd8ee3b1b3064faf2',
    'pycbc/filter/matchedfilter.py': '387cd17317aba8c1e5857adfab64efd93c0a109832387d069d14d085bf35de6a',
    'pycbc/vetoes/chisq_torch.py': 'b73f4a0b5a2b5ca3c110680dfc91bb5241d773d36fac1816b4b8330b8b0010f7',
}
FFT_FEATURE_PATCH_SHA256 = {'pycbc/fft/torchfft.py': 'c9fc0888e95fc96e397890587927312531f7fe3dbf7a8d50b0a913b2d35e011b'}
DIFF = ['--no-ext-diff', '--no-textconv', '--no-color', '--no-renames',
        '--binary', '--full-index', '--unified=3', '--diff-algorithm=myers',
        '--no-indent-heuristic', '--src-prefix=a/', '--dst-prefix=b/']


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args]).decode().strip()


def snapshot(repo):
    files = {}
    entries = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', '-rz', 'HEAD'])
    for entry in entries.split(b'\0'):
        if not entry:
            continue
        info, name = entry.split(b'\t', 1)
        mode, kind, blob = info.decode().split()
        path = repo / os.fsdecode(name)
        require(kind == 'blob' and mode in ('100644', '100755', '120000'), str(path))
        require(path.is_symlink() == (mode == '120000'), 'Wrong file type: ' + str(path))
        data = os.fsencode(os.readlink(path)) if path.is_symlink() else path.read_bytes()
        actual = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        require(actual == blob, 'Tracked content changed: ' + str(path))
        actual_mode = stat.S_IMODE(path.lstat().st_mode)
        require(mode == '120000' or bool(actual_mode & 0o111) == (mode == '100755'),
                'Tracked executable mode changed: ' + str(path))
        files[os.fsdecode(name)] = dict(mode=mode, filesystem_mode=actual_mode,
                                       blob=blob, sha256=hashlib.sha256(data).hexdigest())
    return dict(head=git(repo, 'rev-parse', 'HEAD'), tree=git(repo, 'rev-parse', 'HEAD^{tree}'),
                status=git(repo, 'status', '--porcelain', '--untracked-files=all'), files=files)


def flake_selection(repo):
    def found(*args):
        return subprocess.check_output(['find', *args], cwd=repo).decode().splitlines()
    return {'bin': found('bin', '-type', 'f', '-name', 'pycbc_*'),
            'pycbc': [p for p in found('pycbc') if p.endswith('.py')
                      and '__init__' not in p and not re.search('version.py', p)],
            'test': [p for p in found('test') if p.endswith('.py') and 'test_schemes' not in p]}


def interrupted(number, _frame):
    raise InterruptedError(f'Received signal {number}')


def path_suffix(path, relative):
    suffix = Path(relative).parts
    return Path(path).parts[-len(suffix):] == suffix


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


def reviewed_candidate(repo, part, before):
    """Reconstruct the tested source through exact #16/#17 feature deltas."""
    evidence = {}
    for path, expected in REVIEWED_SHA256.items():
        actual = before['files'][path]['sha256']
        feature_hashes = {16: FFT_FEATURE_PATCH_SHA256, 17: CPU_FEATURE_PATCH_SHA256}.get(part['number'], {})
        if path in feature_hashes:
            context = ['--unified=0'] if path in FFT_FEATURE_PATCH_SHA256 else []
            patch = subprocess.check_output([
                'git', '-C', str(repo), 'diff', *DIFF, *context,
                part['parent'], part['head'], '--', path])
            normalized = re.sub(rb'(?m)^index [0-9a-f]{40}\.\.[0-9a-f]{40}(.*)$',
                                rb'index <old>..<new>\1', patch)
            normalized = re.sub(rb'(?m)^@@ -\d+(,\d+)? \+\d+(,\d+)? @@',
                                rb'@@ -START\1 +START\2 @@', normalized)
            feature_hash = hashlib.sha256(normalized).hexdigest()
            require(feature_hash == feature_hashes[path],
                    'Candidate feature patch differs from the reviewed patch')
            reconstructed = reverse_file_patch((repo / path).read_bytes(), patch)
            actual = hashlib.sha256(reconstructed).hexdigest()
            evidence[path] = dict(candidate_sha256=before['files'][path]['sha256'],
                                  normalized_feature_patch_sha256=feature_hash,
                                  reconstructed_reference_sha256=actual)
        require(actual == expected,
                'Candidate differs from the unit-tested source: '
                + KEYS[part['number']] + ': ' + path)
    return evidence


def unit_evidence(path):
    """Verify existing unit evidence; the fresh quality clone never runs tests."""
    receipt = json.loads(path.read_text())
    require(receipt.get('state') == 'complete' and receipt.get('passed') is True
            and receipt.get('returncode') == 0 and receipt.get('finished_utc'),
            'Relevant unit checks are not complete and passing')
    require(receipt.get('source_info') == dict(commit=REFERENCE, status='')
            and receipt.get('source_after') == receipt['source_info'],
            'Unit source differs from the frozen reference')
    command = receipt.get('command')
    require(isinstance(command, list) and all(isinstance(p, str) for p in command)
            and len(command) == 9 + len(UNIT_TESTS)
            and command[:2] == ['taskset', '-c'] and command[2].isdecimal()
            and Path(command[3]).is_absolute()
            and command[4:9] == ['-m', 'pytest', '-q', '-p', 'no:cacheprovider']
            and all(path_suffix(p, test) for p, test in zip(command[9:], UNIT_TESTS)),
            'Unit command does not include all 22 complete test files')
    hashes = receipt.get('input_sha256')
    require(isinstance(hashes, dict) and hashes
            and receipt.get('input_sha256_after') == hashes
            and all(isinstance(p, str) and Path(p).is_absolute()
                    and isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value)
                    for p, value in hashes.items()),
            'Unit inputs are missing, invalid, or changed')
    for relative, expected in REVIEWED_SHA256.items():
        matches = [value for p, value in hashes.items()
                   if path_suffix(p, relative)]
        require(len(matches) == 1, 'Missing or ambiguous unit input: ' + relative)
        require(matches[0] == expected,
                'Unit-tested source differs from the reviewed revision: ' + relative)
    require(all(Path(p).is_file() and sha(Path(p)) == value
                for p, value in hashes.items()), 'Unit-check inputs changed after testing')
    log = path.with_name('unit-tests-v6.log')
    require(log.is_file() and sha(log) == receipt.get('log_sha256'),
            'Unit log differs from its receipt')
    return receipt, dict(hashes, **{str(path): sha(path), str(log): sha(log)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--stack', type=Path)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--unit-receipt', type=Path,
                        help='Completed relevant-unit receipt from the frozen reference source')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--timeout', type=int, default=7200)
    args = parser.parse_args()
    root = args.root.resolve()
    stack_path = args.stack or root / 'publication/candidate-stack.json'
    bundle = args.bundle or root / 'publication/publication.bundle'
    unit_path = (args.unit_receipt or root.parent
                 / 'pycbc-torch-inspiral-reference-20260906/unit-tests-v6.json').resolve()
    repo, out = root / 'quality-checkout', root / 'quality'
    stack = json.loads(stack_path.read_text()) if stack_path.exists() else None
    plan = dict(checkout=str(repo), output=str(out), stack=str(stack_path), bundle=str(bundle),
                prerequisite=str(root / 'accelerator-refinement/status.json'),
                unit_receipt=str(unit_path), unit_source_commit=REFERENCE,
                unit_test_files=list(UNIT_TESTS), inspiral_sha256=INSPIRAL_SHA256,
                reviewed_source_sha256=REVIEWED_SHA256,
                cpu_feature_patch_sha256=CPU_FEATURE_PATCH_SHA256,
                fft_feature_patch_sha256=FFT_FEATURE_PATCH_SHA256,
                candidates=stack['candidates'] if stack else None,
                qlty_command=[QLTY, 'check', '--no-upgrade-check', '--no-fix', '--sarif',
                              '--no-progress', '--upstream', '<actual candidate parent>'],
                flake8_command=[PYTHON, '-m', 'flake8', '--select=F401', '<CI-selected paths>'])
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0
    require(stack and stack['schema'] == 'torch-performance-fix-candidates-v1', 'Wrong stack schema')
    parts = stack['candidates']
    require(len(parts) == 4 and {p['number'] for p in parts} == set(KEYS), 'Expected four final heads')
    for part in parts:
        require(part['candidate_branch'] == PREFIX + KEYS[part['number']]
                and all(re.fullmatch('[0-9a-f]{40}', part[k]) for k in ('head', 'parent', 'tree')),
                'Invalid candidate identity')
    parts = sorted(parts, key=lambda p: list(KEYS).index(p['number']))
    prerequisite = root / 'accelerator-refinement/status.json'
    done = json.loads(prerequisite.read_text())
    require(done['state'] == 'complete' and done.get('finished_utc')
            and len(done['completed']) == 36, 'Accelerator timing is not complete')
    units, unit_inputs = unit_evidence(unit_path)
    integration_path = root / 'cpu-integration-v6-tests/result.json'
    integration = json.loads(integration_path.read_bytes())
    cpu = next(p for p in parts if p['number'] == 17)
    before = integration['source_before']
    require(integration['state'] == 'complete' and integration['passed'] is True
            and integration['returncode'] == 0 and integration.get('finished_utc')
            and before == integration['source_after'] and not before['status']
            and before['head'] == cpu['head'] and before['tree'] == cpu['tree'],
            'Integrated CPU unit tests are incomplete or source differs')
    runner = root / 'publication/test-cpu-integration-v6.py'
    # Read the declared test list without executing the test runner.
    declaration = next(node.value for node in ast.parse(runner.read_text()).body
                       if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'tests' for t in node.targets))
    tests = ast.literal_eval(declaration)
    require(len(tests) == 17 and integration['test_command'][9:] == tests
            and integration['test_command'][:3] == ['taskset', '-c', '8']
            and integration['test_command'][4:9] == ['-m', 'pytest', '-q', '-p', 'no:cacheprovider'],
            'Integrated CPU test selection differs')
    integration_log = integration_path.with_name('tests.log')
    require(sha(integration_log) == integration['log_sha256']
            and all(sha(Path(p)) == h for p, h in integration['input_sha256'].items()),
            'Integrated CPU test log or inputs changed')
    require(not repo.exists() and not out.exists(), 'Quality checkout/output exists; inspect before retry')
    inputs = {str(p): sha(p) for p in (stack_path, bundle, prerequisite, Path(__file__))}
    inputs.update(unit_inputs)
    inputs.update(integration['input_sha256'])
    inputs.update({str(p): sha(p) for p in (integration_path, integration_log)})
    out.mkdir()
    status = dict(state='running', passed=False, host=socket.gethostname(), pid=os.getpid(),
                  cwd=str(Path.cwd()), command=[sys.executable, *sys.argv], started_utc=utc(),
                  stop_command=f'kill -TERM {os.getpid()}', expected_next_check_seconds=30,
                  unit_source_info=units['source_info'], unit_command=units['command'],
                  inputs_sha256=inputs, commands=[])
    save(out / 'plan.json', plan)

    def execute(label, command, cwd=repo, *, check=False, sarif=False):
        stdout = out / (label + ('.sarif' if sarif else '.log'))
        stderr = out / (label + '-stderr.log')
        record = dict(label=label, command=command, cwd=str(cwd), host=status['host'],
                      started_utc=utc(), stdout=str(stdout), stderr=str(stderr))
        status['commands'].append(record)
        process = None
        started = time.monotonic()
        try:
            with stdout.open('x') as output, stderr.open('x') as errors:
                process = subprocess.Popen(command, cwd=cwd, stdout=output, stderr=errors,
                                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                                           start_new_session=True)
                record.update(pid=process.pid, stop_command=f'kill -TERM -- -{process.pid}')
                save(out / 'result.json', status)
                print(json.dumps(record), flush=True)
                while process.poll() is None:
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        require(time.monotonic() - started < args.timeout, 'Command timeout: ' + label)
                        save(out / 'result.json', status)
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
            record.update(returncode=process.poll() if process else None, finished_utc=utc(),
                          wall_seconds=time.monotonic() - started)
            save(out / (label + '-result.json'), record)
            save(out / 'result.json', status)
        require(not check or record['returncode'] == 0, 'Setup command failed: ' + label)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        require(git(root / 'baseline', 'rev-parse', 'HEAD') == BASELINE, 'Baseline head changed')
        execute('clone', ['git', 'clone', '--no-hardlinks', '--no-checkout', str(root / 'baseline'), str(repo)], root, check=True)
        refs = [f"refs/heads/{p['candidate_branch']}:refs/remotes/publication/{KEYS[p['number']]}" for p in parts]
        execute('fetch', ['git', 'fetch', str(bundle), *refs], check=True)
        for part in parts:
            key = KEYS[part['number']]
            require(git(repo, 'rev-parse', 'refs/remotes/publication/' + key) == part['head'], 'Bundle head differs')
            require(git(repo, 'rev-list', '--parents', '-n', '1', part['head']) == part['head'] + ' ' + part['parent'], 'Candidate parent differs')
            execute(key + '-checkout', ['git', 'checkout', '--detach', part['head']], check=True)
            before = snapshot(repo)
            require(before['tree'] == part['tree'] and not before['status'], 'Candidate checkout differs')
            reviewed = reviewed_candidate(repo, part, before)
            save(out / (key + '-reviewed-source.json'), reviewed)
            save(out / (key + '-source-before.json'), before)
            try:
                execute(key + '-qlty', [QLTY, 'check', '--no-upgrade-check', '--no-fix', '--sarif', '--no-progress', '--upstream', part['parent']], sarif=True)
                if part['number'] == 15:
                    selection = flake_selection(repo)
                    save(out / 'flake8-selection.json', dict(workflow_sha256=sha(repo / '.github/workflows/check_code.yml'), paths=selection))
                    for group, paths in selection.items():
                        require(paths, 'Empty CI F401 selection: ' + group)
                        execute('main-flake8-' + group, [PYTHON, '-m', 'flake8', '--select=F401', *paths])
            finally:
                links = out / (key + '-qlty-generated-links')
                links.mkdir()
                targets = {}
                for name in LINKS:
                    path = repo / '.qlty' / name
                    if path.is_symlink():
                        require(str(path.relative_to(repo)) not in before['files'], 'Refusing to move a tracked link')
                        targets[name] = dict(target=os.readlink(path), resolved_before=str(path.resolve()))
                        path.rename(links / name)
                save(out / (key + '-qlty-links.json'), targets)
                after = snapshot(repo)
                save(out / (key + '-source-after.json'), after)
                require(before == after, 'Quality tool changed checkout: ' + key)
        require(all(sha(Path(p)) == value for p, value in inputs.items()), 'Qualification input changed')
        status.update(state='complete', passed=all(r['returncode'] == 0 for r in status['commands']))
    except BaseException as exc:
        status.update(state='failed', error=repr(exc))
        (out / 'failure.log').write_text(traceback.format_exc())
        raise
    finally:
        status['finished_utc'] = utc()
        save(out / 'result.json', status)
    return 0 if status['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
