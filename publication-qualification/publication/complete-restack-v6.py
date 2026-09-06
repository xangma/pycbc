"""Complete only the reviewed CPU cherry-pick left by the preserved failure."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('restack', HERE / 'restack.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
plan = json.loads((HERE / 'frozen-plan.json').read_bytes())
proof_path = HERE / 'cpu-integration-v6.json'
proof = json.loads(proof_path.read_bytes())
journal_path = HERE / 'restack-execution.json'
journal = json.loads(journal_path.read_bytes())
root, assembly = Path(plan['root']), Path(plan['assembly'])
old = r.STACK[-1]
assert proof['reviewed'] is True and proof['schema'] == 'torch-cpu-hotpath-integration-v1'
assert r.digest(journal_path.read_bytes()) == proof['failed_execution_sha256']
assert (HERE / 'restack-execution-conflict-v6.json').read_bytes() == journal_path.read_bytes()
assert r.digest((HERE / 'frozen-plan.json').read_bytes()) == proof['plan_sha256']
assert journal['state'] == 'failed' and len(journal['candidates']) == 3
assert r.value(assembly, 'rev-parse', 'HEAD') == proof['parent']
assert r.value(assembly, 'rev-parse', 'CHERRY_PICK_HEAD') == old['sha']
assert r.value(assembly, 'branch', '--show-current') == r.PREFIX + old['key']
assert not r.value(assembly, 'diff', '--name-only')
assert r.value(assembly, 'write-tree') == proof['tree']
assert r.snapshot_files(root, list(plan['files'])) == plan['files']
for part in journal['candidates']:
    assert r.value(root, 'rev-parse', part['candidate_branch']) == part['head']
r.git(assembly, 'diff', '--cached', '--check')
r.git(assembly, '-c', 'core.editor=true', 'cherry-pick', '--continue')
head = r.value(assembly, 'rev-parse', 'HEAD')
r.parent_check(assembly, head, proof['parent'])
assert r.value(assembly, 'rev-parse', 'HEAD^{tree}') == proof['tree']
assert not r.value(assembly, 'status', '--porcelain=v1', '--untracked-files=all')
r.no_operation(assembly)
assert r.paths(assembly, proof['parent'], head) == proof['feature_paths']
assert r.fingerprints(assembly, proof['parent'], head, proof['feature_paths']) == proof['feature_patches']
assert r.paths(assembly, old['sha'], head) == proof['update_paths']
assert r.fingerprints(assembly, old['sha'], head, proof['update_paths']) == proof['main_update_patches']
assert r.value(root, 'rev-parse', 'HEAD') == r.FIX
assert r.snapshot_files(root, list(plan['files'])) == plan['files']
assert r.digest(r.git(root, 'ls-files', '--stage', '-z').stdout) == plan['input_index_sha256']
assert r.digest(r.git(root, 'status', '--porcelain=v1', '-z', '--untracked-files=all').stdout) == plan['input_status_sha256']
journal['candidates'].append(dict(number=old['number'], branch=old['branch'], base=old['base'],
    candidate_branch=r.PREFIX + old['key'], old_head=old['sha'], head=head,
    parent=proof['parent'], tree=proof['tree']))
integration = dict(path=str(proof_path), sha256=r.digest(proof_path.read_bytes()),
    completion_script=str(Path(__file__).resolve()),
    completion_script_sha256=r.digest(Path(__file__).read_bytes()))
main = journal['candidates'][0]['head']
stack = dict(schema='torch-performance-fix-candidates-v1', plan_sha256=journal['plan_sha256'],
    candidates=journal['candidates'],
    main_update_patches=r.fingerprints(root, r.STACK[0]['sha'], main, sorted(r.CODE + list(plan['files']))),
    backup_bundle=str(HERE / 'old-heads.bundle'), backup_bundle_sha256=journal['backup_bundle_sha256'],
    integration=integration,
    notice='Structural preservation verified except the explicit reviewed CPU IFFT integration; fresh integration tests are required.')
r.write_json(HERE / 'candidate-stack.json', stack, exclusive=True)
journal.update(state='complete', integration=integration,
    prior_failure='restack-execution-conflict-v6.json',
    finished_utc=r.datetime.now(r.timezone.utc).isoformat())
r.write_json(journal_path, journal)
print(json.dumps({'state': 'complete', 'candidates': journal['candidates']}, indent=2))
