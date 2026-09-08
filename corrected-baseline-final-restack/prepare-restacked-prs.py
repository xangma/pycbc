"""Prepare complete PR descriptions from final, verified publication metadata."""
import argparse
import json
from pathlib import Path
import re

V = Path(__file__).resolve().parent
REPO = 'https://github.com/xangma/pycbc'
CPU = '66789ac4a7468094b0cc3ca1498a1de67e0311f6'
CPU_EVIDENCE = 'dcd123cade49b75ce312d0a8342a8c9c5c9b6abd'
OLD_VALIDATION = REPO + '/tree/06c77a19432bce561f5cbeb6c2da1dedde98274b/stack-validation-20260908'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--evidence-commit', required=True)
    parser.add_argument('--validation-commit', required=True)
    args = parser.parse_args()
    assert all(re.fullmatch('[a-f0-9]{40}', x) for x in (args.evidence_commit, args.validation_commit))
    manifest = json.loads(args.manifest.read_text())
    old = {p['number']: p for p in json.loads((V / 'live-prs-before-cpu-restack.json').read_text())}
    rows = manifest['prs']
    heads = {p['pr']: p['new_head'] for p in rows}
    evidence = REPO + '/tree/' + args.evidence_commit + '/corrected-baseline-campaign'
    review = REPO + '/blob/' + args.evidence_commit + '/corrected-baseline-campaign/README.md'
    validation = REPO + '/tree/' + args.validation_commit + '/cpu-base-restack'
    out = V / 'restacked-pr-bodies'
    out.mkdir(exist_ok=True)
    changes = []
    for row in rows:
        number = row['pr']
        previous = old[number]
        sections = dict(re.findall(r'^## ([^\n]+)\n\n(.*?)(?=^## |\Z)', previous['body'], re.M | re.S))
        intro = previous['body'].split('\n\n## ', 1)[0]
        standard = sections['Standard information about the request'].strip()
        motivation = sections['Motivation'].strip()
        contents = sections['Contents'].strip()
        if number == 7:
            contents = contents.replace('Shared PSD generation retains the precision needed by downstream strain and normalization calculations.',
                'CPU Welch/PSD precision corrections are inherited from #20. This PR adds Torch PSD dispatch and retains its device-specific precision handling; standalone CPU regressions remain Torch-free.')
        elif number == 8:
            contents = contents.replace('Spectral-power accumulation and signal-consistency calculations retain the corrected precision across CPU and CUDA.',
                'CPU template-power accumulation and point chi-square precision corrections are inherited from #20. This PR adds Torch dispatch and backend implementations; additional Torch precision tests are separate from the unchanged standalone CPU tests.')
        elif number == 9:
            intro = 'Connect Torch filtering to event processing and offline and live search, including shared input-loading improvements.'
            contents = contents.replace('The offline executable retains the corrected strain/PSD precision and thread reporting.',
                'The CPU strain-segment FFT precision correction is inherited from #20. This PR adds Torch FFT handling, keeps Torch precision coverage in a separate test file, and retains executable thread reporting.')
        elif number == 15:
            intro = 'Provide reproducible Torch validation and measurement tools, with backend comparisons against the standalone corrected CPU baseline. Trigger comparisons pass; full-PSD failures remain visible alongside descriptive execution costs.'
            motivation = 'Reviewers need to assess CPU numerical corrections separately from Torch support. The comparison uses the corrected CPU prerequisite and measures the rebuilt normal CPU, Torch CPU and Torch CUDA routes with fixed inputs, resources and a complete-process timing boundary.'
            contents = '''Controlled campaign, profile acquisition, summary, artifact-validation and offline plotting tools; CPU/GPU test selectors; usage, parity and comparison guides. Runtime implementations and their regressions live in the preceding owning PRs.

The completed comparison fixes 384 compressed templates, five analysis segments and 1904 unique H1 seconds. It compares standalone CPU `66789ac4a7468094b0cc3ca1498a1de67e0311f6` with rebuilt main `f582b6fd250d0b82612492979e01e645d5c07afc`. Four rotating fresh-process repetitions per arm use one CPU core/thread on shared host `len`; CUDA additionally uses RTX 4090 GPU 0. Timings cover launch through completed HDF output and runtime verification. Existing native builds were reused only after source and binary hashes matched; no fresh compilation is claimed for this campaign.

All five cross-route trigger comparisons pass the unchanged budgets. Conditioned strain, geometry and the PSD bins used by filtering are exact. Full PSD arrays differ below 30 Hz and fail the unchanged full-array budget. The initial strict controller stop is preserved. A disclosed post-qualification continuation collected descriptive timings after the trigger and used-bin checks passed; it does not turn the full-PSD failure into a pass. The verifier reports evidence completeness separately from scientific equivalence. No full-equivalence speedup or sustained-capacity claim is made.

Current guides describe the corrected CPU prerequisite, backend results, reproduction and remaining scientific limitation. The earlier unchanged-CPU versus combined-proposal comparison is retained as historical evidence. CPU correction cost and intentional trigger changes are reviewed in #20.'''
        parent = 20 if number == 18 else row['parent_pr']
        base_ref = 'codex/cpu-precision-corrections-20260908' if number == 18 else row['published_base_ref']
        links = f'''Depends on [#{parent}]({REPO}/pull/{parent}). The main chain starts with the standalone [CPU corrections in #20]({REPO}/pull/20), then #18 and #5–#15. Optional leaves remain #15 → #19 → #16 and #15 → #17.

Head branch: `{row['published_ref']}`. Head commit: `{row['new_head']}`. Base branch: `{base_ref}`.

[CPU scientific review]({REPO}/blob/{CPU_EVIDENCE}/cpu-review.md); [corrected-baseline results]({review}); [raw benchmark and reproduction]({evidence}); [restack validation]({validation}); [current documentation]({REPO}/blob/{heads[15]}/docs/torch_performance.rst).'''
        testing = '''The rebuilt main source `f582b6fd250d0b82612492979e01e645d5c07afc` passed the relevant local CPU/Torch suite: **377 passed, 139 capability-dependent skips**, with no failures. The separate CPU run blocks Torch imports and discovery: **19 passed, no skips**. The #7, #8 and #9 prefixes were also tested: 92/17, 213/110 and 77/21 passed/skipped respectively. These overlapping counts are not a unique-test total. Local validation used macOS arm64; unavailable CUDA/MPS paths retain their skips.

The current restack audit verifies all 15 ancestry relationships, exact old/new commit mappings, unchanged native source blobs, and byte-identical standalone CPU precision tests. CI F401 selections pass across 225 executables, 292 modules and 139 test files. Full flake8 retains inherited diagnostics. Linux Qlty found two formatting-only issues in the rebuilt filtering wrappers; the owner PR #8 contains their correction. Both entire Python module ASTs are identical to the measured source, and Qlty passes after formatting. See the validation archive for the exact scope and receipts.

The Linux comparison ran four executable qualifications and 16 timed processes against the corrected CPU baseline. All timed trigger outputs pass their own qualification; all five cross-route trigger checks pass. Full-PSD equivalence still fails below the filtering cutoff. Raw receipts and the independent verifier are linked above. Optional #16/#17 are structurally audited descendants and are outside this benchmark.'''
        if number in (18,19):
            testing = ('Formatting and native-blob audits are recorded in the restack validation. The four corrected CPU function bodies in #18 match the CPU prerequisite after AST normalization of docstring whitespace. Optional #19 replays its original formatting commit.\n\n' + testing)
        if number == 15:
            testing += '\n\nThe final documentation build uses Sphinx `-E -a -W --keep-going` for all retained Torch guides and their real waveform/plugin/installation dependencies. Its exact command, result and source mapping are included in the validation archive. Relative to the measured source, the final head changes five guides and formatting in two Python wrappers; their entire ASTs are identical. Native sources, executables, tests, tools and CI files remain byte-identical.'
        testing += f'\n\n[Earlier feature-prefix tests and quality checks]({OLD_VALIDATION}) remain historical evidence tied to their recorded revisions. They are not presented as reruns on these new heads. No new native kernels were introduced by this restack.'
        notes = sections['Additional notes'].strip()
        body = intro + '\n\n' + '\n\n'.join('## ' + name + '\n\n' + value for name, value in (
            ('Standard information about the request', standard), ('Motivation', motivation),
            ('Contents', contents), ('Links to any issues or associated PRs', links),
            ('Testing performed', testing), ('Additional notes', notes))) + '\n'
        assert body.count('This PR was created by AI Gareth') == 1
        assert '*AI Agent Note: Unchecked by default. @xangma, please review this PR and check the Code of Conduct box above to confirm your agreement before requesting review.*' in body
        assert '- [ ] The author' in body and '- [x]' not in body
        assert row['old_head'] == previous['head']['sha']
        path = out / f'pr-{number}.md'
        path.write_text(body)
        changes.append(dict(number=number, old_head=row['old_head'], new_head=row['new_head'],
            old_base=previous['base']['ref'], new_base=base_ref, head_ref=row['published_ref'],
            title=previous['title'], body_file=str(path), draft=previous['draft']))
    (out / 'updates.json').write_text(json.dumps(changes, indent=2) + '\n')
    print(json.dumps(dict(prepared=len(changes), path=str(out))))


if __name__ == '__main__':
    main()
