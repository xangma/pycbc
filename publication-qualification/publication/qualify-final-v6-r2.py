"""Run units and lint on each exact final head; preserve all prior failures."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path('/home/xangma/pycbc-torch-performance-fix-20260906')
PUB = ROOT / 'publication'
OUT = ROOT / 'quality-final-v6-r2'
REFERENCE = ROOT.parent / 'pycbc-torch-inspiral-reference-20260906/source-v6'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


Q = module('quality', PUB / 'qualify-quality.py')
P = module('polish', PUB / 'polish-publication.py')
stack = json.loads((PUB / 'candidate-stack-final.json').read_bytes())
prior = json.loads((PUB / 'candidate-stack.json').read_bytes())
proof = json.loads((PUB / 'polish-proof-v6.json').read_bytes())
assert Q.sha(PUB / 'polish-proof-v6.json') == stack['polish']['sha256']
assert Q.sha(PUB / 'polish-publication.py') == proof['script_sha256'] == stack['polish']['script_sha256']
assert len(stack['candidates']) == len(prior['candidates']) == len(proof['rows']) == 4
assert Q.git(REFERENCE, 'rev-parse', 'HEAD') == Q.REFERENCE
units, inputs = Q.unit_evidence(REFERENCE.parent / 'unit-tests-v6.json')
for name in ('candidate-stack-final.json', 'candidate-stack.json', 'publication-final.bundle',
             'publication.bundle', 'polish-proof-v6.json', 'polish-publication.py',
             'qualify-quality.py', 'qualify-final-v6.py', 'qualify-final-v6-r2.py', 'test-cpu-integration-v6.py'):
    inputs[str(PUB / name)] = Q.sha(PUB / name)
for name in ('quality/result.json', 'quality-final-v6/result.json', 'cpu-integration-v6-tests/result.json'):
    inputs[str(ROOT / name)] = Q.sha(ROOT / name)
assert not OUT.exists()
OUT.mkdir()
status = dict(state='running', passed=False, host=socket.gethostname(), pid=os.getpid(),
    cwd=str(ROOT), command=[sys.executable, *sys.argv], started_utc=Q.utc(),
    stop_command=f'kill -TERM {os.getpid()}', expected_next_check_seconds=30,
    inputs_sha256=inputs, unit_source_info=units['source_info'], commands=[], candidates=[])
Q.save(OUT / 'result.json', status)
env = {k: v for k, v in os.environ.items() if not k.startswith('PYCBC_')
       and k not in {'PYTEST_ADDOPTS', 'PYTEST_PLUGINS', 'PYTHONSTARTUP', 'PYTHONPATH'}}
env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', OMP_DYNAMIC='FALSE')


def execute(label, command, cwd, check=False, sarif=False):
    record = dict(label=label, command=command, cwd=str(cwd), host=status['host'], started_utc=Q.utc(),
                  stdout=str(OUT / (label + ('.sarif' if sarif else '.log'))),
                  stderr=str(OUT / (label + '-stderr.log')))
    status['commands'].append(record)
    process = None
    started = time.monotonic()
    try:
        with open(record['stdout'], 'x') as out, open(record['stderr'], 'x') as err:
            process = subprocess.Popen(command, cwd=cwd, env=dict(env, PYTHONPATH=str(cwd)),
                stdout=out, stderr=err, start_new_session=True)
            record.update(pid=process.pid, stop_command=f'kill -TERM -- -{process.pid}')
            Q.save(OUT / 'result.json', status)
            process.wait(timeout=7200)
    except BaseException:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    finally:
        record.update(returncode=process.poll() if process else None, finished_utc=Q.utc(),
                      wall_seconds=time.monotonic() - started)
        for stream in ('stdout', 'stderr'):
            if Path(record[stream]).is_file():
                record[stream + '_sha256'] = Q.sha(Path(record[stream]))
        Q.save(OUT / (label + '-result.json'), record)
        Q.save(OUT / 'result.json', status)
    assert not check or record['returncode'] == 0, label
    return record


signal.signal(signal.SIGTERM, Q.interrupted)
signal.signal(signal.SIGINT, Q.interrupted)
try:
    for part, old, reviewed in zip(stack['candidates'], prior['candidates'], proof['rows']):
        n = part['number']
        assert n == old['number'] == reviewed['number']
        key = Q.KEYS[n]
        repo = ROOT / ('quality-final-v6-r2-' + key + '-checkout')
        assert not repo.exists()
        execute(key + '-clone', ['git', 'clone', '--no-hardlinks', '--no-checkout', str(REFERENCE), str(repo)], ROOT, True)
        for kind, row in (('', old), ('-final', part)):
            execute(key + '-fetch' + kind, ['git', 'fetch', str(PUB / ('publication' + kind + '.bundle')),
                'refs/heads/' + row['candidate_branch']], repo, True)
        execute(key + '-prior-checkout', ['git', 'checkout', '--detach', old['head']], repo, True)
        old_source = Q.snapshot(repo)
        original_review = Q.reviewed_candidate(repo, old, old_source)
        details = P.validate_delta(repo, old['head'], part['head'], n)
        # AST dumps vary across Python versions; source hashes are portable.
        # validate_delta independently proves AST equality on this interpreter.
        assert set(details) == set(reviewed['files'])
        assert all(details[p][k] == reviewed['files'][p][k] for p in details
                   for k in ('before_sha256', 'after_sha256'))
        assert reviewed['after_head'] == part['head']
        assert reviewed['before_head'] == old['head']
        execute(key + '-checkout', ['git', 'checkout', '--detach', part['head']], repo, True)
        assert Q.git(repo, 'rev-list', '--parents', '-n', '1', part['head']) == part['head'] + ' ' + part['parent']
        binary_source = ROOT / 'cpu-integration-v6-checkout' if n == 17 else REFERENCE
        expected_binary_head = old['head'] if n == 17 else Q.REFERENCE
        assert Q.git(binary_source, 'rev-parse', 'HEAD') == expected_binary_head
        assert not Q.git(binary_source, 'status', '--porcelain', '--untracked-files=all')
        native_inputs = {}
        for relative in Q.git(repo, 'ls-files').splitlines():
            p = Path(relative)
            if p.suffix.lower() in {'.c', '.cc', '.cpp', '.h', '.hpp', '.pyx', '.pxd', '.cu', '.f', '.f90'} or relative in {'setup.py', 'setup.cfg', 'pyproject.toml', 'MANIFEST.in'}:
                assert (repo / p).read_bytes() == (binary_source / p).read_bytes(), relative
                native_inputs[relative] = Q.sha(repo / p)
        binaries = {}
        for p in sorted((binary_source / 'pycbc').rglob('*.so')):
            assert p.is_file() and not p.is_symlink()
            target = repo / p.relative_to(binary_source)
            assert not target.exists()
            shutil.copy2(p, target)
            assert Q.sha(target) == Q.sha(p)
            binaries[str(target.relative_to(repo))] = Q.sha(target)
        assert len(binaries) == 11
        before = Q.snapshot(repo)
        before['native_binaries'] = binaries
        assert before['head'] == part['head'] and before['tree'] == part['tree'] and not before['status']
        Q.save(OUT / (key + '-source-before.json'), before)
        Q.save(OUT / (key + '-reviewed-source.json'), dict(original_review=original_review,
            polish=reviewed, runtime_polish_ast=details, python_version=sys.version, binary_source=str(binary_source), binary_source_head=expected_binary_head,
            native_inputs=native_inputs, native_binaries=binaries))
        tests = list(Q.UNIT_TESTS) + ['test/waveform/test_taylorf2_phase_evaluation.py']
        if n == 16:
            tests += ['test/test_fft_batched_backends.py', 'test/test_fft_cli_wisdom.py', 'test/test_fftw_wisdom_cache.py']
        if n == 17:
            declaration = next(node.value for node in ast.parse((PUB / 'test-cpu-integration-v6.py').read_text()).body
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'tests' for t in node.targets))
            tests += [p for p in ast.literal_eval(declaration) if p not in tests]
        assert all((repo / p).is_file() for p in tests)
        test = execute(key + '-units', ['taskset', '-c', '8', sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *tests], repo)
        execute(key + '-qlty', [Q.QLTY, 'check', '--no-upgrade-check', '--no-fix', '--sarif', '--no-progress', '--upstream', part['parent']], repo, sarif=True)
        if n == 15:
            selection = Q.flake_selection(repo)
            Q.save(OUT / 'flake8-selection.json', dict(workflow_sha256=Q.sha(repo / '.github/workflows/check_code.yml'), paths=selection))
            for group, paths in selection.items():
                assert paths
                execute('main-flake8-' + group, [sys.executable, '-m', 'flake8', '--select=F401', *paths], repo)
        links = OUT / (key + '-qlty-generated-links')
        links.mkdir()
        targets = {}
        for name in Q.LINKS:
            path = repo / '.qlty' / name
            if path.is_symlink():
                assert str(path.relative_to(repo)) not in before['files']
                targets[name] = dict(target=os.readlink(path), resolved_before=str(path.resolve()))
                path.rename(links / name)
        Q.save(OUT / (key + '-qlty-links.json'), targets)
        after = Q.snapshot(repo)
        after['native_binaries'] = {p: Q.sha(repo / p) for p in binaries}
        assert before == after
        Q.save(OUT / (key + '-source-after.json'), after)
        status['candidates'].append(dict(number=n, head=part['head'], tree=part['tree'], test_files=tests))
        if n == 17:
            integration = dict(state='complete', passed=test['returncode'] == 0, returncode=test['returncode'],
                finished_utc=Q.utc(), source_before=before, source_after=after, test_command=test['command'],
                log_sha256=test['stdout_sha256'], input_sha256=dict(inputs))
            Q.save(OUT / 'cpu-integration-final.json', integration)
            inputs[str(OUT / 'cpu-integration-final.json')] = Q.sha(OUT / 'cpu-integration-final.json')
        Q.save(OUT / 'result.json', status)
    assert all(Q.sha(Path(p)) == h for p, h in inputs.items())
    status.update(state='complete', passed=all(c['returncode'] == 0 for c in status['commands']))
except BaseException as exc:
    status.update(state='failed', error=repr(exc))
    (OUT / 'failure.log').write_text(traceback.format_exc())
    raise
finally:
    status['finished_utc'] = Q.utc()
    Q.save(OUT / 'result.json', status)
sys.exit(0 if status['passed'] else 1)
