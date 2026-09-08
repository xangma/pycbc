"""Finalize local sidecar receipts without changing any source or Git ref."""
import collections
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = Path('/Users/xangma/repos/pycbc')
manifest_path = OUT / 'manifest.json'
m = json.loads(manifest_path.read_text())
audit = json.loads((OUT / 'structural-audit.json').read_text())
lint = json.loads((OUT / 'lint-audit.json').read_text())
main = next(p for p in m['prs'] if p['pr'] == 15)


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def receipt(path):
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


assert audit['status'] == 'pass'
assert audit['main_new_head'] == main['new_head']
assert all(v['exit_code'] == 0 for v in m['validation'])
assert all(v['exit_code'] == 0 for v in lint['ci'])
assert subprocess.check_output(['git', '-C', m['worktree'], 'status', '--porcelain'], text=True) == ''
assert subprocess.check_output(['git', '-C', m['worktree'], 'rev-parse', 'HEAD'], text=True).strip() == main['new_head']

before = dict(row.split(' ', 1) for row in m['refs_before'])
after = dict(row.split(' ', 1) for row in git('for-each-ref', '--format=%(refname) %(objectname)').splitlines())
observed_changes = [dict(ref=ref, before=sha, after=after.get(ref))
                    for ref, sha in before.items() if after.get(ref) != sha]
for pr in m['prs']:
    assert after['refs/heads/' + pr['staging_ref']] == pr['new_head']
    assert git('merge-base', m['cpu_base'], pr['new_head']) == m['cpu_base']

live_path = OUT.parent / 'live-prs-before-cpu-restack.json'
live = {row['number']: row for row in json.loads(live_path.read_text())}
for pr in m['prs']:
    row = live[pr['pr']]
    assert row['head']['sha'] == pr['old_head']
    assert row['head']['ref'] == pr['published_ref']
    assert row['base']['sha'] == pr['old_base']
    assert row['base']['ref'] == pr['published_base_ref']

m.update(
    status='complete_local_validation',
    completed_at=datetime.now(timezone.utc).isoformat(),
    stable_main_head=main['new_head'],
    structural_audit=dict(status='pass', file='structural-audit.json'),
    lint=lint,
    live_snapshot=receipt(live_path),
    ref_audit=dict(
        existing_refs_observed_changed=observed_changes,
        authorized_staging_refs=[p['staging_ref'] for p in m['prs']],
        note='This sidecar only created/updated its 15 staging refs. Other actors may independently change repository refs; this is an observation of shared state.'),
    scope=dict(
        source_changes='Only four production functions and the precision test layout differ from prior main.',
        native='All native source Git blobs match the corresponding prior published head, on all 15 PRs.',
        cpu_tests='All three standalone CPU precision test files match CPU base bytes on every staged head.',
        documentation='docs/ and examples/ match corresponding prior heads byte-for-byte. All other main tracked files also match prior main.',
        mutations='Only the new staging worktree/refs and this artifact directory. No network, remote operations, publication, further delegation, or shared environment installs.'),
    limitations=[
        'Local validation used macOS arm64 and existing Python 3.13 dependencies. CUDA was unavailable; hardware-dependent skips are retained.',
        'Optional leaves #16/#17 were replayed and structurally audited, but were not separately executed in this sidecar.',
        'Full style lint has inherited findings; qlty CLI is unavailable. All repository CI F401 selections pass.',
        'No new Linux qualification, timing, sensitivity or ranking claims. Primary owns independent source review, Linux qualification and publication.',
        'Previously reported standalone Linux 32 passes / 2 baseline-identical MKL failures are primary-provided context and were not rerun here.'
    ],
)

pr_rows = '\n'.join(
    f"| #{p['pr']} | {'CPU' if p['parent_pr'] is None else '#' + str(p['parent_pr'])} | `{p['old_head']}` | `{p['new_head']}` | {len(p['commits'])} |"
    for p in m['prs'])
test_rows = []
for v in m['validation']:
    c = v['counts']
    passed = c['tests'] - c['skipped'] - c['failures'] - c['errors']
    test_rows.append(f"| {v['case']} | `{v['head'][:12]}` | {passed} | {c['skipped']} | {c['failures'] + c['errors']} | [{v['case']}.log]({v['case']}.log) |")
test_rows = '\n'.join(test_rows)
changes_text = ('All refs present before the sidecar still resolve to the same objects.'
                if not observed_changes else
                f'{len(observed_changes)} pre-existing ref changes were observed in shared state; see `ref_audit` in the manifest. This sidecar did not change them.')
report = f'''# CPU-base Torch stack restack

All 15 published PR heads were rebuilt locally onto standalone CPU `{m['cpu_base']}`. The stable #15 head is **`{main['new_head']}`**, replacing `{main['old_head']}`. Its tracked worktree is clean. No source head changed after this stable SHA was communicated to the primary.

The reviewed main diff changes only four production functions and six precision test files. Every other tracked file matches old #15, including the latest documentation/evidence. All native source blobs match the corresponding old head across all 15 PRs. There are no new C, Cython, CUDA or `pycbc/lib` edits.

## Staging and ancestry

Worktree: `{m['worktree']}`. Each ref is `codex/cpu-base-restack-20260908-prNN`, with `NN` replaced by the PR number. The main chain is CPU → #18 → #5 → #6 → #7 → #8 → #9 → #10 → #11 → #12 → #13 → #14 → #15; optional branches are #15 → #19 → #16 and #15 → #17.

| PR | New parent | Old head | New head | Replayed commits |
| --- | --- | --- | --- | ---: |
{pr_rows}

All {sum(len(p['commits']) for p in m['prs'])} commits in the verified old per-PR ranges have an explicit old/new mapping. [manifest.json](manifest.json) contains the full old/new heads, bases, published/staging refs, ordered commit maps, conflict decisions, commands, runtime provenance and logs. Old heads/bases were checked against the primary's supplied [live snapshot](../live-prs-before-cpu-restack.json); this sidecar made no network request. {changes_text}

## Functional review and conflicts

- `pycbc.psd.from_cli`: CPU restores input dtype only when float32 input was actually promoted. Published non-CPU return casts and non-MPS Torch promotion are retained.
- `pycbc.filter.matchedfilter.sigmasq_series`: CPU promotion is conditional on float32 magnitude; the published non-MPS Torch cast path is retained.
- `StrainSegments.fourier_segments`: CPU casts FFT output back only after promotion. Published non-CPU casting and Torch promotion remain in place.
- `power_chisq_at_points_from_precomputed`: the standalone CPU body is retained with the published Torch early dispatch. Splitting the `shifts` assignment and restoring CPU comments/formatting does not alter arithmetic.

The [structural audit](structural-audit.json) verifies that each affected module's AST outside these four function bodies is identical to old main. It additionally verifies the chi-squared CPU body against standalone CPU, the Torch dispatch against old main, and arithmetic identity after inlining the split assignment. Guards and dtype handling in the other three functions were reviewed directly. See [production diff](final-main-production.diff), [complete diff](final-main.diff) and [diff statistics](final-main-stat.txt).

Conflicts occurred in #18 formatting, #7 PSD, #8 filtering/chi-squared/tests, and #9 strain/tests. They were resolved within the affected functions/imports/test cases. A duplicate strain `_scheme` import found during review was removed before the stable #15 SHA was communicated; descendants were rebuilt from that corrected #9. No native conflict was resolved by editing native code. The six latest #15 documentation commits were replayed in order and their final files remain byte-identical to old #15, as historical evidence.

`test_chisq_precision.py`, `test_sigmasq_series_precision.py` and `test_strain_psd_precision.py` are byte-identical to standalone CPU on all 15 heads. They contain no Torch imports or Torch-dependent skips. Published additional Torch coverage lives in `test_torch_chisq_precision.py`, `test_torch_sigmasq_series_precision.py` and `test_torch_strain_psd_precision.py`; duplicate CPU cases were removed from those moved files, retaining Torch reference calculations, assertions and CUDA/Triton checks.

## Local validation

Existing interpreter: `/private/tmp/pycbc-cpu-precision-env-20260908/bin/python`; Python 3.13, NumPy 2.3.5, SciPy 1.16.3, macOS arm64. No shared environment install was performed. Each run set `PYTHONPATH` to the staging worktree, disabled unrelated pytest plugin autoload, and set OMP/OpenBLAS/MKL thread counts to one. Imports, Git head, generated version and native module origin were checked before tests. Eleven existing compiled modules were copied only after verifying unchanged native source blobs, and their SHA-256 hashes were checked. [prepare-runtime.py](prepare-runtime.py) and `runtime-*.json` record this preparation.

| Case | Tested head | Passed | Skipped | Failed/errors | Log |
| --- | --- | ---: | ---: | ---: | --- |
{test_rows}

#7 covers CPU precision, PSD and Torch PSD pipelines/protocols; #8 covers CPU/Torch precision, filtering, sparse and optimized chi-squared paths; #9 covers strain/PSD precision and search kernels. #15 runs their union plus FFT write and CPU-native tests. The separate #15 CPU run blocks Torch imports/discovery and passes all 19 standalone tests without skips. Commands and JUnit XML paths are recorded per run in the manifest. These are overlapping suites; counts should not be summed as unique coverage. CUDA is unavailable locally and capability-dependent CUDA/MPS skips are visible in the logs.

The supervisor and test processes have exited. Optional leaves #16/#17 received ancestry, exact-range replay, CPU test, documentation and native blob audits; they were not separately executed. The primary's previously reported standalone Linux result (32 passes; two legacy MKL failures reproduced identically on frozen original) is context only and was not rerun in this sidecar.

## Lint and remaining ownership

All three exact CI F401 file selections pass: 225 executables, 292 modules and 139 test files. [lint-audit.json](lint-audit.json) records commands and logs. Unfiltered `flake8 pycbc/ test/ --select F401` reports 130 diagnostics, verified unchanged against old main for tracked files; generated `pycbc/version.py` is excluded by CI. Full `flake8 pycbc/ test/` also reports inherited style findings. The changed-file comparison is 272 old versus 269 new diagnostics; eight apparently added E501 findings are verbatim historical lines relocated into Torch test files, with provenance in [lint-relocated-lines.json](lint-relocated-lines.json). The `qlty` CLI is unavailable and was not installed.

This completes the bounded local implementation and validation. Primary retains independent source review, Linux corrected-base/rebuilt CPU/Torch qualification and timing, and all publication. This report makes no new performance, sensitivity or trigger-ranking claim.
'''
(OUT / 'report.md').write_text(report)
m['artifacts'] = [receipt(p) for p in sorted(OUT.iterdir())
                  if p.is_file() and p.name != 'manifest.json']
manifest_path.write_text(json.dumps(m, indent=2) + '\n')
print(json.dumps(dict(status=m['status'], main=main['new_head'], prs=len(m['prs']),
                     replayed_commits=sum(len(p['commits']) for p in m['prs']),
                     existing_refs_observed_changed=observed_changes,
                     report=str(OUT / 'report.md'), manifest=str(manifest_path)), indent=2))
