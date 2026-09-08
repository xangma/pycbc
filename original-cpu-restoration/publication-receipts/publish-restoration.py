"""Guard and atomically publish the original CPU restoration, and withdraw PR20."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

V = Path(__file__).resolve().parent
ROOT = V.parents[2]
GH = '/opt/homebrew/bin/gh'
REPO = 'xangma/pycbc'
REMOTE = 'https://github.com/' + REPO + '.git'
GIT = ['git', '-c', 'credential.helper=', '-c', 'credential.helper=!/opt/homebrew/bin/gh auth git-credential']


def run(command, **kw):
    return subprocess.check_output(command, cwd=ROOT, text=True, **kw)


def get_pr(number):
    return json.loads(run([GH, 'api', f'repos/{REPO}/pulls/{number}']))


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    updates = json.loads((V / 'final-pr-bodies/updates.json').read_text())
    old = {r['number']: r for r in json.loads((V / 'live-prs-before.json').read_text())}
    assert len(updates) == 15 and len({u['number'] for u in updates}) == 15
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        current = list(pool.map(get_pr, [u['number'] for u in updates]))
    save(V / 'restack-prepublication-snapshot.json', current)
    for row, now in zip(updates, current):
        before = old[row['number']]
        assert now['number'] == row['number'] and now['state'] == 'open'
        assert now['head']['sha'] == row['old_head'] == before['head']['sha']
        assert now['head']['ref'] == row['head_ref']
        assert now['base']['ref'] == row['old_base']
        assert now['body'] == before['body'] and now['title'] == before['title']
        assert now['draft'] == row['draft'] is True
        assert 'agent-assisted' in [x['name'] for x in now['labels']]
        body = Path(row['body_file']).read_text()
        assert 'This PR was created by AI Gareth' in body and '- [x]' not in body
        assert '*AI Agent Note: Unchecked by default. @xangma, please review this PR and check the Code of Conduct box above to confirm your agreement before requesting review.*' in body
        assert row['new_head'] in body
        assert run(['git', 'rev-parse', row['new_head']]).strip() == row['new_head']
        run(['git', 'merge-base', '--is-ancestor', '40e94792b3edf59f39b18b65102b28a4f74433a7', row['new_head']])
        assert subprocess.run(['git', 'merge-base', '--is-ancestor', '66789ac4a7468094b0cc3ca1498a1de67e0311f6', row['new_head']], cwd=ROOT).returncode == 1
        assert all(h in body for h in ['## Standard information about the request', '## Motivation', '## Contents', '## Links to any issues or associated PRs', '## Testing performed', '## Additional notes'])
        assert not any(p in body for p in ['FINAL_', 'PENDING_PRIMARY', 'Final-source qualification is pending'])
    cpu = get_pr(20)
    assert cpu['head']['sha'] == '66789ac4a7468094b0cc3ca1498a1de67e0311f6'
    assert cpu['head']['ref'] == 'codex/cpu-precision-corrections-20260908'
    assert cpu['base']['ref'] == 'torch-stack-base' and cpu['state'] == 'open'
    assert cpu['body'] == old[20]['body'] and cpu['title'] == old[20]['title']
    assert cpu['draft'] is True and 'agent-assisted' in [x['name'] for x in cpu['labels']]
    withdrawn = (V / 'final-pr-bodies' / 'pr-20.md').read_text()
    assert 'withdrawn' in withdrawn.lower() and 'This PR was created by AI Gareth' in withdrawn
    assert '- [x]' not in withdrawn and '*AI Agent Note: Unchecked by default. @xangma, please review this PR and check the Code of Conduct box above to confirm your agreement before requesting review.*' in withdrawn
    original_ref = run(['git', 'ls-remote', REMOTE, 'refs/heads/torch-stack-base']).split()[0]
    assert original_ref == '40e94792b3edf59f39b18b65102b28a4f74433a7'
    command = GIT + ['push', '--atomic']
    command += ['--force-with-lease=refs/heads/' + u['head_ref'] + ':' + u['old_head'] for u in updates]
    command += [REMOTE] + [u['new_head'] + ':refs/heads/' + u['head_ref'] for u in updates]
    plan = dict(status='preflight_pass', observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                push_command=command, updates=[dict(u, body_sha256=hashlib.sha256(Path(u['body_file']).read_bytes()).hexdigest()) for u in updates])
    save(V / 'restack-publication-plan.json', plan)
    if not args.execute:
        print('PASS: 15 heads, descriptions, labels and draft states unchanged; publication plan ready')
        return
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    save(V / 'restack-push-result.json', dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
    result.check_returncode()
    completed = []
    for row in updates:
        path = V / 'final-pr-bodies' / f'pr-{row["number"]}-request.json'
        payload = dict(body=Path(row['body_file']).read_text())
        if row['new_base'] != row['old_base']:
            payload['base'] = row['new_base']
        save(path, payload)
        updated = json.loads(run([GH, 'api', '--method', 'PATCH', f'repos/{REPO}/pulls/{row["number"]}', '--input', str(path)]))
        assert updated['body'] == payload['body']
        completed.append(row['number'])
        save(V / 'restack-publication-progress.json', dict(branches_pushed=True, bodies_updated=completed))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(get_pr, [u['number'] for u in updates]))
    for row, now in zip(updates, verified):
        assert now['head']['sha'] == row['new_head']
        assert now['base']['ref'] == row['new_base']
        assert now['base']['sha'] == row['new_base_sha']
        assert now['body'] == Path(row['body_file']).read_text()
        assert now['draft'] is True and now['state'] == 'open'
        assert 'agent-assisted' in [x['name'] for x in now['labels']]
    save(V / 'restack-published-verification.json', verified)
    withdraw_path = V / 'final-pr-bodies' / 'pr-20-request.json'
    save(withdraw_path, dict(body=withdrawn, state='closed'))
    run([GH, 'api', '--method', 'PATCH', f'repos/{REPO}/pulls/20', '--input', str(withdraw_path)])
    closed = get_pr(20)
    assert closed['state'] == 'closed' and closed['merged'] is False
    assert closed['head']['sha'] == cpu['head']['sha'] and closed['body'] == withdrawn
    assert 'agent-assisted' in [x['name'] for x in closed['labels']]
    save(V / 'pr20-withdrawn-verification.json', closed)
    print('PASS: all 15 draft PR heads, dependencies, descriptions and labels published and verified; PR20 withdrawn and closed')


if __name__ == '__main__':
    main()
