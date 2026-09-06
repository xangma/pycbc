"""Archive completed publication qualification without changing scientific evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent.parent
PUB = ROOT / 'publication'
ARCH = ROOT.parent / 'torch-benchmark-20260906/evidence'
A = '639154f7ef359eb473db79660942a2cd88dd59e6'
PREFIX = 'publication-qualification'


def git(*args):
    return subprocess.check_output(['git', '-C', str(ARCH), *args]).decode().strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as f:
        f.write(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    quality_path = ROOT / 'quality-final-v6-r2/result.json'
    quality = json.loads(quality_path.read_bytes())
    assert quality['state'] == 'complete' and quality['passed'] is True
    assert all(c['returncode'] == 0 for c in quality['commands'])
    assert len(quality['candidates']) == 4
    for directory in ('docs-validation-v6', 'docs-validation-cpu-v6'):
        docs = json.loads((ROOT / directory / 'docs-validation.json').read_bytes())
        assert docs['returncode'] == 0 and docs['finished_utc']
    assert git('rev-parse', 'HEAD') == A and not git('status', '--porcelain')
    assert not (ARCH / PREFIX).exists()
    sources = sorted(p for p in PUB.iterdir() if p.is_file() and not p.is_symlink()
        and p.suffix in {'.py', '.json', '.md', '.patch', '.log', '.bundle'}
        and not p.name.startswith('archive-qualification-plan'))
    omitted = []
    for directory in ('quality', 'quality-final-v6', 'quality-final-v6-r2',
                      'cpu-integration-v6-tests', 'docs-validation-v6', 'docs-validation-cpu-v6'):
        assert (ROOT / directory).is_dir(), directory
        for p in sorted((ROOT / directory).rglob('*')):
            if p.is_symlink():
                omitted.append(str(p.relative_to(ROOT)))
            elif p.is_file() and not {'.doctrees', '__pycache__'} & set(p.parts):
                sources.append(p)
    inventory = {str(p.relative_to(ROOT)): dict(sha256=sha(p), bytes=p.stat().st_size)
                 for p in sorted(sources)}
    assert len(inventory) == len(sources)
    plan = dict(schema='torch-publication-qualification-archive-v1', archive_a=A,
                files=inventory, omitted_host_links=omitted,
                total_bytes=sum(v['bytes'] for v in inventory.values()),
                max_file_bytes=max(v['bytes'] for v in inventory.values()))
    assert plan['max_file_bytes'] < 80_000_000
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    plan_path = PUB / 'archive-qualification-plan-v6.json'
    assert json.loads(plan_path.read_bytes()) == plan
    for name, item in inventory.items():
        source, target = ROOT / name, ARCH / PREFIX / name
        assert sha(source) == item['sha256'] and not target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha(target) == item['sha256']
    receipt = dict(plan, state='complete', passed=True, finished_utc=datetime.now(timezone.utc).isoformat(),
        scientific_archive_unchanged=True, final_candidates=quality['candidates'],
        quality_sha256=sha(quality_path), staging_plan_sha256=sha(plan_path),
        note='Qualification evidence only. Retained failed lint and runner setup attempts are superseded by quality-final-v6-r2. Remote qlty symlinks are represented by the captured link manifests.')
    save(ARCH / PREFIX / 'archive-finalization.json', receipt)
    original = subprocess.check_output(['git', '-C', str(ARCH), 'show', A + ':README.md'])
    prefix = ('# Final publication qualification\n\n'
        'Fresh unit tests and qlty checks pass on all four final publication commits; '
        'CI-selected F401 checks and both documentation builds also pass. '
        'The measured scientific sources and raw campaign evidence remain unchanged.\n\n'
        '[Final check results](publication-qualification/quality-final-v6-r2/result.json), '
        '[candidate identities and bounded lint cleanup](publication-qualification/publication/polish-proof-v6.json), '
        '[archive inventory](publication-qualification/archive-finalization.json), '
        '[retained earlier lint findings](publication-qualification/quality/result.json), and '
        '[retained setup failure](publication-qualification/quality-final-v6/result.json).\n\n---\n\n').encode()
    (ARCH / 'README.md').write_bytes(prefix + original)
    assert subprocess.check_output(['git', '-C', str(ARCH), 'diff', '--name-only']).decode().splitlines() == ['README.md']
    paths = sorted(p for p in ARCH.rglob('*') if p.is_file() and not p.is_symlink()
                   and '.git' not in p.relative_to(ARCH).parts and p.name != 'SHA256SUMS')
    # Existing nested SHA256SUMS files are evidence too; exclude only the root index.
    paths += sorted(p for p in ARCH.rglob('SHA256SUMS') if p != ARCH / 'SHA256SUMS' and p.is_file())
    paths = sorted(set(paths), key=lambda p: str(p.relative_to(ARCH)))
    (ARCH / 'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(ARCH)}\n' for p in paths))
    for name, item in inventory.items():
        assert sha(ROOT / name) == sha(ARCH / PREFIX / name) == item['sha256']
    git('add', '--', PREFIX, 'README.md', 'SHA256SUMS')
    expected = sorted(['README.md', 'SHA256SUMS', PREFIX + '/archive-finalization.json']
                      + [PREFIX + '/' + p for p in inventory])
    assert git('diff', '--cached', '--name-only').splitlines() == expected
    assert not git('diff', '--name-only')
    git('commit', '-m', 'Archive final publication qualification and retained check failures')
    assert not git('status', '--porcelain')
    save(PUB / 'archive-qualification-commit-v6.json', dict(a=A, b=git('rev-parse', 'HEAD'),
        tree=git('rev-parse', 'HEAD^{tree}'), changes=expected, receipt_sha256=sha(ARCH / PREFIX / 'archive-finalization.json')))
    print(git('log', '-1', '--format=%H %s'))


if __name__ == '__main__':
    main()
