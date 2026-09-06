#!/usr/bin/env python3
"""Freeze both campaigns; --execute copies two new sibling supplements.
No Git/network/test/report runs. Failed copies remain for inspection, never overwrite.
"""
import argparse
import hashlib
import json
import posixpath
import re
import shutil
import stat
from pathlib import Path
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent
DIRS = ('comparison profiles-baseline wave-profiles-baseline postchecks cpu32-t4-diagnostic '
        'accelerator-refinement accelerator-refinement-setup-failed '
        'report cpu-followup-report cuda-v2-followup-report').split()
FILES = ('''comparison-status.json source-environment.json source.bundle
performance-fix.patch accelerator-fastpath.patch final-performance-fix.diff
dependent-sources.bundle dependent-sources.json dependent-sources-v2.bundle
dependent-sources-v2.json dependent-sources-v2-journal.json dependent-v2-root-review.json
optional-cpu-baseline.bundle run-comparison.py waveform-worker.py run-postchecks.py
setup-dependent.py prepare-dependent-v2.py run-accelerator-refinement.py profile-live.py
profile-waveform.py probe-cpu-peaks.py build-report.py build-cpu-followup-report.py
build-cuda-v2-followup-report.py summarize-profiles.py profile-summary.json profile-summary.md
test-command.json tests-candidate.log test_taylorf2_phase_evaluation.py
test_torch_batch_overlap_scaling.py lint-f401.json lint-f401.log
lint-f401-v2.json lint-f401-v2.log lint-baseline-equivalence.json workflow-validation.json
postchecks-launch.json accelerator-refinement-launch.json run-cpu32-diagnostic.py''').split()
EXCLUDE = {'.git', '__pycache__', '.pytest_cache', '.DS_Store', '.venv', 'venv', 'build', 'dist'}
DESTINATIONS = ('performance-fix', 'inspiral-reference-20260906')
REFERENCE_DIRS = {'inputs', 'runs', 'final-report', 'raw', 'tuning', 'accuracy',
                  'parity', 'profiles', 'unit-tests', 'snr-psd-diagnostic-v1',
                  'snr-psd-diagnostic-v2', 'chisq-input-capture-cpu', 'chisq-input-capture-cuda',
                  'chisq-capture-v4-cpu', 'chisq-capture-v4b-cpu', 'chisq-capture-v4b-torch-cuda'}
REFERENCE_EXCLUDE = EXCLUDE | {'source', 'source-v2', 'source-v3', 'source-v4', 'source-v5',
                             'native', 'native-modules', 'publication',
                             'render', 'render-cache', 'docs-validation'}
REFERENCE_SUFFIXES = {'.py', '.json', '.csv', '.md', '.rst', '.log', '.txt', '.xml',
                      '.bundle', '.patch', '.diff', '.png', '.svg', '.pdf'}
REFERENCE_REQUIRED = '''config.json environment.json source.json inspiral-source.bundle
source-v2.json inspiral-source-v2.bundle reference-source-equivalence.json setup-source-v2.py
unit-tests.json unit-tests.log run-unit-checks-v2.py deepcopy-before-tests.json
deepcopy-before-tests.log test_scheme_deepcopy_regression.py
matched-backends-plan.json matched-backends-plan.status.json matched-backends-launch.log
final-qualifications-v2-plan.json final-qualifications-v2-plan.status.json final-qualifications-v2-launch.log
summarize-profiles-v1.py
prepare-bank.py compress-bank.py compress-bank-1e5.py compression.json compression-1e5.json
compression-refinement-decision.json qualification-wrapper-v1.py qualify-inspiral.py
qualification-v1-failure-analysis.json waveform-validation.json waveform-validation-1e5.json
boundary-injections-1e5.json scientific-validation.status.json
run-case.py run-case-profiling.py profile-inspiral-torch.py summarize-profiles.py
validate-waveforms.py check-boundary-injections.py compare-triggers.py
publication-integration-plan.md build-reference-report.py
inputs/bank.hdf inputs/bank-compressed.hdf inputs/bank-metadata.json
inputs/bank-pilot.hdf inputs/bank-pilot-compressed.hdf
inputs/bank-compressed-1e5.hdf inputs/bank-pilot-compressed-1e5.hdf
runs/qual-cpu-l256/qualification.json runs/qual-v2-cpu-l256/qualification.json
accuracy-campaign.status.json reference-tuning-decision.json
final-report/report.json final-report/runs.csv final-report/groups.csv
final-report/profiles.csv final-report/native-symbols.csv'''.split()
REFERENCE_REQUIRED += '''source-v3.json source-v4.json inspiral-source-v3.bundle
inspiral-source-v4.bundle source-precision-provenance.json source-precision-provenance-v4.json
setup-source-v3.py setup-source-v4.py unit-tests-v3.json unit-tests-v3.log
unit-tests-v4.json unit-tests-v4.log run-unit-checks-v3.py run-unit-checks-v4.py
build-reference-report-v3.py test-build-reference-report-v3.py
run-precision-reference-campaign.py run-precision-reference-campaign-v4.py
precision-backend-smoke-plan.json precision-backend-smoke-plan.status.json
runs/qual-precision-torch-cpu-l512/receipt.json
runs/qual-precision-torch-cpu-l512/qualification.json
runs/qual-precision-torch-cuda-l512/receipt.json
runs/qual-precision-torch-cuda-l512/qualification.json
numerical-investigation-v1.json numerical-investigation-v1.md
snr-psd-diagnostic-v1/report.json snr-psd-diagnostic-v2/report.json
chisq-input-capture-cpu/status.json chisq-input-capture-cpu/capture-verification.json
chisq-input-capture-cuda/status.json chisq-input-capture-cuda/capture-verification.json
chisq-identical-input-diagnostic.json chisq-identical-input-binning-diagnostic.json
chisq-captured-bin-swap-diagnostic.json prefix-reference-diagnostic.json
prefix-reference-diagnostic.log'''.split()
REFERENCE_REQUIRED += '''source-v5.json inspiral-source-v5.bundle
source-precision-provenance-v5.json setup-source-v5.py unit-tests-v5.json unit-tests-v5.log
run-unit-checks-v5.py build-reference-report-v4.py test-build-reference-report-v4.py
run-precision-reference-campaign-v5.py waveform-validation-precision5.json
boundary-injections-precision5.json profiles-precision5.json
precision5-reference-tuning-decision.json precision-backend-smoke-parity.json
precision-backend-smoke-parity-v2.json chisq-coefficient-v4.json
chisq-coefficient-v4.log chisq-coefficient-v4b.log diagnose-chisq-coefficient-v4.py
chisq-capture-v4-cpu/status.json chisq-capture-v4b-cpu/status.json
chisq-capture-v4b-torch-cuda/status.json capture-chisq-inputs-v4.py
capture-chisq-inputs-v4b.py run-chisq-captures-v4.py run-chisq-captures-v4b.py
chisq-captures-v4.status.json chisq-captures-v4b.status.json
runs/qual-precision-cpu-l512/receipt.json runs/qual-precision-cpu-l512/qualification.json'''.split()
REFERENCE_REQUIRED += '''precision5-backend-smoke-parity.json
precision5-backend-smoke-plan.json precision5-backend-smoke-plan.status.json
runs/qual-precision5-torch-cpu-l512/receipt.json
runs/qual-precision5-torch-cpu-l512/qualification.json
runs/qual-precision5-torch-cuda-l512/receipt.json
runs/qual-precision5-torch-cuda-l512/qualification.json
run-final-reference-campaign-v5.py precision5-final-campaign.status.json
precision5-final-campaign.log precision5-final-campaign-qualifications.log
precision5-final-campaign-reference-profiles.log precision5-final-campaign-matched-timings.log
precision5-final-campaign-torch-profiles.log precision5-final-campaign-summary.log'''.split()
REFERENCE_DUPLICATES = {'qualification-v2-sample.json':
                        'runs/qual-v2-cpu-l256/qualification.json'}
SOURCE_CHAIN = ('dfd42bf76766cadca0eecf609a1eaeac73534676',
                '0d00581251e642a5d6b56b2497a9adad93069e6b',
                'fb4b335eeeeaeaa907c1143b45e0191e2d977761',
                '968bcd558117262af0d603710b054174659adb51',
                '6c82155044d58f3344b281869d87745f71ba2285',
                'f2c0abe61e787a26f41208f489c62c877bbd5667',
                '837f38d493420043e45fb1ad210a0ccf68bacbaa')
REFERENCE_PLANS = ('precision5-reference-qualifications-plan', 'precision5-reference-timings-plan',
                   'precision5-final-qualifications-plan', 'precision5-reference-profiles-plan',
                   'precision5-matched-backends-plan', 'precision5-torch-profiles-plan')
PRECISION_PATHS = ('pycbc/filter/matchedfilter.py', 'pycbc/psd/__init__.py',
                   'pycbc/strain/strain.py', 'pycbc/vetoes/chisq.py', 'pycbc/vetoes/chisq_torch.py',
                   'test/test_chisq_precision.py', 'test/test_sigmasq_series_precision.py',
                   'test/test_strain_psd_precision.py')
PARENT_PRECISION_PATHS = tuple(name for name in PRECISION_PATHS
                               if name != 'pycbc/vetoes/chisq_torch.py')
PARENT_CHANGED_PATHS = ('pycbc/vetoes/chisq_torch.py', 'test/test_chisq_precision.py')
UNIT_TESTS = '''test/test_scheme_runtime.py test/test_scheme_selection.py
test/test_matchedfilter.py test/test_chisq.py test/test_psd.py test/test_strain.py
test/test_sigmasq_series_precision.py test/test_chisq_precision.py
test/test_strain_psd_precision.py test/test_torch_chisq_cpu_optimization.py
test/test_torch_chisq_sparse_dispatch.py test/test_torch_filter_pipeline.py
test/test_torch_psd_pipeline.py test/test_torch_psd_protocol.py
test/test_torch_versioned_data_psd.py test/test_torch_matchedfilter_cpu_optimization.py'''.split()
REFERENCE_FIGURES = {'tuning-capacity.png', 'matched-capacity.png', 'wall-and-internal-times.png',
                     'profile-self-time.png', 'native-symbols.png', 'cuda-events.png'}
REFERENCE_GATES = {'source_provenance', 'unit_tests', 'campaign_complete', 'source_and_inputs_bound',
                   'qualifications', 'waveform_validation', 'boundary_validation',
                   'timing_counts_and_geometry', 'tuning_decision', 'trigger_parity', 'profiles'}


def reference_cases(selected):
    require(type(selected) is int and selected in (256, 512, 1024), 'Invalid selected FFT length')
    schemes = {'cpu': 'cpu:1', 'torch-cpu': 'torch:cpu:1', 'torch-cuda': 'torch:cuda:0'}
    return (
        {f'qual-precision5-cpu-l{n}': ('qualify', 'cpu:1', n) for n in (256, 512, 1024)},
        {f'tune-precision5-cpu-l{n}-r{r}': ('timing', 'cpu:1', n)
         for n in (256, 512, 1024) for r in (1, 2, 3)},
        {f'qual-selected5-{label}-l{selected}': ('qualify', scheme, selected)
         for label, scheme in schemes.items()},
        {f'reference-precision5-cpu-l{selected}-{mode}': (mode, 'cpu:1', selected)
         for mode in ('cprofile', 'perf')},
        {f'matched-precision5-{label}-l{selected}-r{r}': ('timing', scheme, selected)
         for label, scheme in schemes.items() for r in (1, 2, 3)},
        {**{f'profile-precision5-{label}-l{selected}-{mode}': (mode, scheme, selected)
             for label, scheme in schemes.items() if label != 'cpu'
             for mode in ('cprofile', 'perf')},
         f'profile-precision5-torch-cuda-l{selected}-torchprofile':
             ('torchprofile', 'torch:cuda:0', selected)},
    )


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def record(path):
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not mode & 0o7000, f'Not a plain file: {path}')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(sha256=digest, bytes=path.stat().st_size, mode=stat.S_IMODE(mode))


def walk(path, prune=False):
    mode = path.lstat().st_mode
    require(stat.S_ISDIR(mode) or stat.S_ISREG(mode), f'Symlink/special file: {path}')
    if prune and (path.name in EXCLUDE or path.suffix in {'.pyc', '.tmp', '.so', '.dylib', '.pyd'}
                  or path.name.endswith('-temp') or (path.name.startswith('.') and 'report' in path.name)):
        return
    if stat.S_ISREG(mode):
        yield path
    else:
        for child in sorted(path.iterdir()):
            yield from walk(child, prune)


def inventory(art):
    sources = {name: art / name for name in FILES}
    require(not (art / 'archive-docs').is_symlink(), 'Symlink documentation directory')
    sources.update({name: art / 'archive-docs' / name for name in ('README.md', 'REPRODUCE.md')})
    for name in DIRS:
        require((art / name).is_dir(), f'Missing required directory: {name}')
        entries = list(walk(art / name, True))
        require(entries, f'Empty required directory: {name}')
        sources.update({str(p.relative_to(art)): p for p in entries})
    require(all(not any(c in name for c in '\n\r\\') for name in sources), 'Unsafe filename')
    return {name: dict(source=str(path), **record(path)) for name, path in sorted(sources.items())}


def history(archive):
    return {str(p.relative_to(archive)): record(p) for entry in sorted(archive.iterdir())
            if entry.name not in {'.git', *DESTINATIONS} for p in walk(entry)}


def reference_inventory(art):
    sources, excluded = {}, {}
    for entry in sorted(art.iterdir()):
        mode = entry.lstat().st_mode
        require(stat.S_ISDIR(mode) or stat.S_ISREG(mode), f'Symlink/special file: {entry}')
        if (entry.name in REFERENCE_EXCLUDE or entry.suffix in {'.pyc', '.tmp', '.so', '.dylib', '.pyd'}
                or entry.name.endswith('-temp')
                or (entry.name.startswith('.') and 'report' in entry.name)):
            continue
        if entry.name in REFERENCE_DUPLICATES:
            canonical = art / REFERENCE_DUPLICATES[entry.name]
            original, duplicate = record(canonical), record(entry)
            require(all(original[k] == duplicate[k] for k in ('sha256', 'bytes')),
                    f'Incidental duplicate differs from canonical evidence: {entry.name}')
            excluded[entry.name] = dict(canonical=REFERENCE_DUPLICATES[entry.name], **duplicate)
            continue
        if stat.S_ISDIR(mode):
            require(entry.name in REFERENCE_DIRS, f'Unreviewed reference directory: {entry.name}')
            entries = list(walk(entry, True))
            require(entries, f'Empty reference directory: {entry.name}')
        else:
            require(entry.suffix in REFERENCE_SUFFIXES,
                    f'Unreviewed reference file type: {entry.name}')
            entries = [entry]
        sources.update({str(p.relative_to(art)): p for p in entries})
    require(all(not any(c in name for c in '\n\r\\') for name in sources), 'Unsafe filename')
    require(set(REFERENCE_REQUIRED) <= sources.keys(),
            'Missing reference evidence: ' + ', '.join(sorted(set(REFERENCE_REQUIRED) - sources.keys())))
    return ({name: dict(source=str(path), **record(path)) for name, path in sorted(sources.items())},
            excluded)


def bundle_header(path):
    with path.open('rb') as stream:
        raw = stream.read(65536)
    require(b'\n\n' in raw, f'Missing bounded bundle header: {path}')
    lines = raw.split(b'\n\n', 1)[0].decode('ascii').splitlines()
    require(lines[0] == '# v2 git bundle', f'Unexpected bundle format: {path}')
    prerequisites, heads = [], []
    for line in lines[1:]:
        require(re.fullmatch(r'-?[0-9a-f]{40} .+', line), f'Unexpected bundle header: {path}')
        (prerequisites if line.startswith('-') else heads).append(line.split(' ', 1)[0].lstrip('-'))
    return dict(prerequisites=sorted(prerequisites), heads=sorted(heads))


def qualify(art, files):
    indexed = {item['source']: item for item in files.values()}
    statuses = [read(art / p) for p in ('comparison-status.json',
                'postchecks/postcheck-status.json', 'accelerator-refinement/status.json')]
    require(all(s['state'] == 'complete' and not s.get('error') for s in statuses), 'Incomplete run')
    require(len(set(statuses[0]['completed'])) == 360 and len(set(statuses[2]['completed'])) == 36,
            'Incomplete main/v2 ledger')
    require(not statuses[1]['skipped_stages'], 'Postcheck stages skipped')
    folder = 'accelerator-refinement-setup-failed'
    failed, retry = (read(art / folder / name) for name in ('status.json', 'retry-preservation.json'))
    old_runner = 'a478f5b0b5865534c79d8dacf0fa4b27c592971b53d4faf442976c5b976c86d4'
    require(failed['state'] == 'failed' and failed['completed'] == failed['completed_tests'] == []
            and failed['timing_started_utc'] is None and failed.get('finished_utc')
            and failed['error'] == "RuntimeError('Harness differs: cpu-candidate: "
            "tools/bench_production_live_batch.py')",
            'Preserved attempt was not the expected setup-only failure')
    launch = read(art / folder / 'accelerator-refinement-launch.json')
    require(failed['runner_sha256'] == launch['runner_sha256']
            == files[f'{folder}/run-accelerator-refinement.py']['sha256'] == old_runner, 'Old runner changed')
    require(all(f'{folder}/{n}' in files for n in '''accelerator-refinement-launch.log runner.log failure.log
            original-status-snapshot.json postcheck-status-snapshot.json setup-environment.json
            setup-environment.partial.json sources-after.json plan.json accelerator-refinement-plan-len.json'''.split())
            and any(n.startswith(f'{folder}/commands/') for n in files)
            and not any(n.startswith(f'{folder}/{d}/') for n in files for d in ('live', 'parity', 'tests')),
            'Preserved setup evidence missing or contains test/timing output')
    remote = Path(failed['cwd'])
    require(retry['failed_status_sha256'] == files[f'{folder}/status.json']['sha256']
            and retry['preserved_source_moves'] == {str(remote / n): str(remote / f'{folder}-sources' / n)
            for n in ('candidate-v2', 'fft-candidate-v2', 'cpu-candidate-v2')}
            and retry['reason'] == 'Harness guard incorrectly compared optional CPU route harness to main. '
            'No tests or timings ran.', 'Retry preservation receipt differs')
    diagnostic = read(art / 'cpu32-t4-diagnostic/status.json')
    require(diagnostic['state'] == 'complete' and diagnostic.get('finished_utc')
            and not diagnostic.get('error') and not diagnostic.get('source_audit_error'), 'Diagnostic incomplete')
    require([c['source'] for c in diagnostic['commands']] == ['baseline', 'candidate-v2']
            and all(c['returncode'] == 0 for c in diagnostic['commands']), 'Diagnostic commands incomplete')
    for side in ('baseline', 'candidate-v2'):
        hashes = read(art / 'cpu32-t4-diagnostic' / f'{side}-profiles-sha256.json')
        require(set(hashes) == set('python.pstats python.txt operators.txt trace.json status.json worker.log'.split())
                and all(files[f'cpu32-t4-diagnostic/{side}/{n}']['sha256'] == h for n, h in hashes.items()),
                'Diagnostic profile changed')
    post = read(art / 'postchecks/plan.json')['steps']
    require(statuses[1]['completed'] == [s['label'] for s in post], 'Postcheck ledger differs')
    tests = {'postchecks': [s['label'] for s in post if s['stage'] == 'tests'],
             'accelerator-refinement': [s['label'] for s in read(art / 'accelerator-refinement/plan.json')['tests']]}
    require(all(len(names) == 3 for names in tests.values()), 'Test coverage differs')
    require(statuses[2]['completed_tests'] == tests['accelerator-refinement'], 'V2 tests incomplete')
    for folder, names in tests.items():
        for name in names:
            summary = read(art / folder / 'tests' / f'{name}-summary.json')
            require(summary['total'] > summary['skipped'] and not summary['failures']
                    and not summary['errors'] and f'{folder}/tests/{name}.xml' in files, f'Failed test: {name}')
    reports = []
    for folder in DIRS[-3:]:
        report = read(art / folder / 'report.json')
        manifest = read(art / folder / 'input-manifest.json')
        require(report['input_manifest'] == dict(path='input-manifest.json', files=len(manifest),
                sha256=files[f'{folder}/input-manifest.json']['sha256']), f'Changed manifest: {folder}')
        require(manifest and all(p in indexed and indexed[p]['sha256'] == h for p, h in manifest.items()),
                f'Missing/changed report input: {folder}')
        require(f'{folder}/report.md' in files and all(f'{folder}/{p}' in files
                for p in report.get('figures', [])), f'Missing report output: {folder}')
        reports.append(report)
    main, cpu, cuda = reports
    require(main['counts'] == dict(live_workers=252, standard_controls=36, waveform_workers=108,
            live_parity_records=108) and all(main['validation'].values()), 'Main validation differs')
    require(cpu['counts']['worker_processes'] == 36 and cpu['counts']['parity_documents'] == 12
            and cpu['numerical_parity']['all_passed'] is True, 'CPU validation differs')
    require(cuda['counts']['fresh_v2_cuda_workers'] == cuda['counts']['v2_parity_documents'] == 36
            and cuda['validation']['complete'] is True and cuda['validation']['all_v2_parity_passed'] is True,
            'V2 validation differs')
    require(main['sources'] == statuses[0]['sources'] and cuda['sources']
            == {**main['sources'], 'candidate-v2': statuses[2]['source']}, 'Report source differs')
    before = read(art / 'postchecks/sources-before.json')
    require(cpu['source_identities'] == {n: {k: before[n][k] for k in ('root', 'sha', 'tree', 'status')}
            for n in ('cpu-baseline', 'cpu-candidate')}, 'CPU report source differs')
    profile = read(art / 'profile-summary.json')
    require(profile['complete'] is True and profile['pair_count'] == len(profile['pairs']) == 12,
            'Profile summary incomplete')
    provenance = profile['provenance']
    receipts = [provenance['source_environment']['file'], provenance['candidate_script_manifest'],
                provenance['summarizer'], *provenance['profiling_scripts'].values()]
    receipts += [f for p in profile['pairs'] for side in ('baseline', 'candidate') for f in p[side]['files'].values()]
    for item in receipts:
        require(item['path'] in indexed and all(indexed[item['path']][k] == item[k]
                for k in ('sha256', 'bytes')), 'Changed profile input')
    for name in ('README.md', 'REPRODUCE.md', 'profile-summary.md', *(f'{d}/report.md' for d in DIRS[-3:])):
        for link in re.findall(r'\[[^\]\n]*\]\(([^)\s]+)\)', Path(files[name]['source']).read_text()):
            url = urlsplit(link.removeprefix('<').removesuffix('>'))
            if url.scheme or url.netloc or not url.path:
                continue
            target = posixpath.normpath(str(Path(name).parent / unquote(url.path)))
            require(target in files or any(p.startswith(target.rstrip('/') + '/') for p in files),
                    f'Unresolved archive link: {name}: {link}')


def qualify_reference_history(art, files, supporting_art):
    """Preserve independently qualified earlier evidence, never reuse it as final science."""
    umbrella = read(art / 'accuracy-campaign.status.json')
    stages = ['compression-1e5', 'accuracy-qualifications', 'waveform-validation-1e5',
              'boundary-injections-1e5', 'accuracy-timings']
    require(umbrella['state'] == 'complete' and umbrella['returncode'] == 0
            and umbrella.get('finished_utc') and not umbrella.get('error')
            and umbrella['completed'] == [dict(name=name, returncode=0) for name in stages]
            and umbrella['input_sha256'] == umbrella['input_sha256_after'],
            'Refined-compression campaign is incomplete or its inputs changed')
    for name in ('waveform-validation-1e5.json', 'boundary-injections-1e5.json'):
        validation = read(art / name)
        require(validation['state'] == 'complete' and validation['passed'] is True
                and validation.get('finished_utc')
                and validation['input_sha256'] == validation['input_sha256_after']
                and validation['source_before'] == validation['source_after']
                and validation['source_before']['commit'] == SOURCE_CHAIN[2]
                and validation['source_before']['status'] == '', f'Reference validation failed: {name}')
    for directory in sorted((art / 'runs').iterdir()):
        if directory.name in EXCLUDE:
            continue
        require(directory.is_dir() and f'runs/{directory.name}/receipt.json' in files,
                f'Missing attempted-run receipt: {directory.name}')
        receipt = read(directory / 'receipt.json')
        require(receipt['state'] in ('complete', 'failed') and receipt.get('finished_utc')
                and isinstance(receipt.get('returncode'), int),
                f'Attempt still running or unresolved: {directory.name}')
    for name in files:
        if '/' not in name and name.endswith('status.json'):
            require(read(art / name).get('state') != 'running', f'Campaign still running: {name}')
    rejected = read(art / 'waveform-validation.json')
    decision = read(art / 'compression-refinement-decision.json')
    require(rejected['state'] == 'complete' and rejected['passed'] is False
            and decision['failed_receipt_sha256'] == files['waveform-validation.json']['sha256']
            and rejected['tolerances'] == decision['budgets_unchanged']
            == read(art / 'waveform-validation-1e5.json')['tolerances'],
            'Original compression rejection or unchanged validation budgets differ')
    failure = read(art / 'qualification-v1-failure-analysis.json')
    original = read(art / 'runs/qual-cpu-l256/qualification.json')
    require(original['status'] == 'failed' and original['executable_exit_code'] == 0
            and failure['sha256'] == files['runs/qual-cpu-l256/qualification.json']['sha256']
            and failure['original_wrapper_sha256'] == original['wrapper']['sha256']
            == files['qualification-wrapper-v1.py']['sha256'], 'Original wrapper failure changed')
    chain = [bundle_header(supporting_art / 'source.bundle'),
             bundle_header(supporting_art / 'dependent-sources-v2.bundle'),
             bundle_header(art / 'inspiral-source.bundle'),
             bundle_header(art / 'inspiral-source-v2.bundle')]
    require(chain[0]['prerequisites'] == [] and SOURCE_CHAIN[0] in chain[0]['heads']
            and chain[1]['prerequisites'] == [SOURCE_CHAIN[0]] and SOURCE_CHAIN[1] in chain[1]['heads']
            and chain[2] == dict(prerequisites=[SOURCE_CHAIN[1]], heads=[SOURCE_CHAIN[2]])
            and chain[3] == dict(prerequisites=[SOURCE_CHAIN[2]], heads=[SOURCE_CHAIN[3]]),
            'Source bundle dependency chain differs')
    source = read(art / 'source.json')
    require(source['commit'] == SOURCE_CHAIN[2] and source['parent'] == SOURCE_CHAIN[1]
            and source['bundle_sha256'] == files['inspiral-source.bundle']['sha256'],
            'Reference source identity differs')
    revised = read(art / 'source-v2.json')
    proof = read(art / 'reference-source-equivalence.json')
    changed = ['pycbc/scheme.py', 'test/test_scheme_runtime.py']
    checks = {'only_TorchScheme_deepcopy_added', 'all_other_tracked_blobs_identical',
              'native_hashes_equal', 'both_sources_clean'}
    require(revised['commit'] == SOURCE_CHAIN[3] and revised['parent'] == SOURCE_CHAIN[2]
            and revised['changed_paths'] == changed
            and revised['bundle_sha256'] == files['inspiral-source-v2.bundle']['sha256']
            and source['native_modules_sha256']
            and source['native_modules_sha256'] == revised['native_modules_sha256']
            and proof['schema_version'] == 1 and proof['status'] == 'pass'
            and proof['old_source_commit'] == SOURCE_CHAIN[2]
            and proof['new_source_commit'] == SOURCE_CHAIN[3]
            and proof['changed_paths'] == changed and checks <= proof['checks'].keys()
            and all(value is True for value in proof['checks'].values()),
            'Source phase equivalence is incomplete or changed')
    remote = Path(source['source']).parent
    proof_inputs = {str(Path(name).relative_to(remote)): digest
                    for name, digest in proof['input_sha256'].items()}
    require({'source.json', 'source-v2.json', 'inspiral-source.bundle',
             'inspiral-source-v2.bundle', 'setup-source-v2.py'} <= proof_inputs.keys()
            and all(name in files and files[name]['sha256'] == digest
                    for name, digest in proof_inputs.items()), 'Source equivalence inputs changed')
    tests = read(art / 'unit-tests.json')
    before_tests = read(art / 'deepcopy-before-tests.json')
    require(tests['state'] == 'complete' and tests['passed'] is True and tests['returncode'] == 0
            and tests.get('finished_utc') and tests['source_info'] == tests['source_after']
            == dict(commit=SOURCE_CHAIN[3], status='')
            and tests['input_sha256'] == tests['input_sha256_after']
            and tests['log_sha256'] == files['unit-tests.log']['sha256']
            and re.search(r'\n31 passed in [0-9.]+s\s*\Z', (art / 'unit-tests.log').read_text())
            and tests['command'][-2:] == ['test/test_scheme_runtime.py', 'test/test_scheme_selection.py']
            and before_tests['expected_failure'] is True and before_tests['returncode'] == 1
            and before_tests.get('finished_utc') and before_tests['source_commit'] == SOURCE_CHAIN[2]
            and before_tests['log_sha256'] == files['deepcopy-before-tests.log']['sha256']
            and re.search(r'\n2 failed, 20 passed in [0-9.]+s\s*\Z',
                          (art / 'deepcopy-before-tests.log').read_text())
            and before_tests['test_sha256'] == files['test_scheme_deepcopy_regression.py']['sha256']
            == tests['input_sha256'][str(remote / 'source-v2/test/test_scheme_runtime.py')],
            'Final scheme tests or preserved pre-fix regression changed')
    for name in ('run-unit-checks-v2.py', 'config.json'):
        require(tests['input_sha256'][str(remote / name)] == files[name]['sha256'],
                f'Unit test input changed: {name}')
    for name in ('matched-backends-plan', 'final-qualifications-v2-plan'):
        plan_name, status_name = name + '.json', name + '.status.json'
        plan, status = read(art / plan_name), read(art / status_name)
        log_name = name.removesuffix('-plan') + '-launch.log'
        require(plan and status['state'] == 'failed' and status['returncode'] == 1
                and status.get('finished_utc') and status['completed'] == []
                and status['plan_sha256'] == files[plan_name]['sha256']
                and status['current'][-len(plan[0]):] == plan[0] and log_name in files,
                f'Preserved failed campaign changed: {name}')
    failed_run = read(art / 'runs/qual-torch-cpu-l512/receipt.json')
    failed_qual = read(art / 'runs/qual-torch-cpu-l512/qualification.json')
    require(failed_run['state'] == 'failed' and failed_run['returncode'] == 1
            and failed_run['source_info']['commit'] == SOURCE_CHAIN[2]
            and failed_qual['status'] == 'failed' and failed_qual['executable_exit_code'] == 1
            and failed_qual['error']['type'] == 'TypeError'
            and failed_qual['error']['message'] == "cannot pickle 'module' object",
            'Preserved Torch qualification failure differs')
    collision = read(art / 'final-qualifications-v2-plan.status.json')
    existing = read(art / 'runs/qual-v2-cpu-l512/receipt.json')
    collision_log = (art / 'final-qualifications-v2-launch.log').read_text()
    require('FileExistsError' in collision_log and 'out.mkdir(parents=True, exist_ok=False)' in collision_log
            and str(remote / 'runs/qual-v2-cpu-l512') in collision_log
            and existing['source_info']['commit'] == SOURCE_CHAIN[2]
            and existing['finished_utc'] < collision['started_utc'],
            'Preserved preflight collision changed or replaced an earlier run')
    return chain


def bound_inputs(inputs, remote, files):
    """Check archived receipt inputs; source trees and external frames retain recorded hashes."""
    archived = set()
    require(isinstance(inputs, dict) and inputs, 'Empty receipt input manifest')
    for name, digest in inputs.items():
        require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest),
                f'Invalid receipt input hash: {name}')
        path = Path(name)
        if path.is_absolute() and not path.is_relative_to(remote):
            continue
        relative = str(path.relative_to(remote)) if path.is_absolute() else name
        require(relative == Path(relative).as_posix() and '..' not in Path(relative).parts,
                f'Unsafe receipt input: {name}')
        if Path(relative).parts[0] in ('source', 'source-v2', 'source-v3', 'source-v4', 'source-v5'):
            continue
        require(relative in files and files[relative]['sha256'] == digest,
                f'Missing/changed archived receipt input: {relative}')
        archived.add(relative)
    return archived


def qualify_precision_parent(art, files):
    source, parent = (read(art / name) for name in ('source-v4.json', 'source-v3.json'))
    previous = read(art / 'source-v2.json')
    remote = Path(source['source']).parent
    require(Path(source['source']).is_absolute() and Path(source['source']).name == 'source-v4'
            and source['commit'] == SOURCE_CHAIN[5] and source['parent'] == SOURCE_CHAIN[4]
            and source['parent_source_commit'] == SOURCE_CHAIN[4]
            and source['reference_base'] == SOURCE_CHAIN[3]
            and parent['source'] == str(remote / 'source-v3')
            and parent['commit'] == SOURCE_CHAIN[4] and parent['parent'] == SOURCE_CHAIN[3]
            and previous['source'] == str(remote / 'source-v2')
            and source['changed_paths'] == parent['changed_paths'] == list(PARENT_PRECISION_PATHS),
            'Final precision source identity differs')
    native = source['native_modules_sha256']
    require(len(native) == 11 and native == parent['native_modules_sha256']
            == previous['native_modules_sha256'], 'Precision native module hashes changed')
    require(set(source['changed_files_sha256']) == set(PARENT_PRECISION_PATHS)
            and all(re.fullmatch('[0-9a-f]{64}', digest)
                    for digest in source['changed_files_sha256'].values())
            and all(source['changed_files_sha256'][name] == parent['changed_files_sha256'][name]
                    for name in PARENT_PRECISION_PATHS if name != 'test/test_sigmasq_series_precision.py'),
            'Final source changed reviewed runtime blobs')
    proof = read(art / 'source-precision-provenance-v4.json')
    checks = {'only_reviewed_python_and_test_paths_changed', 'native_hashes_equal',
              'both_sources_clean', 'all_final_runs_require_new_source', 'tests_only_parent_delta'}
    require(proof['schema_version'] == 1 and proof['status'] == 'pass'
            and proof['old_source_commit'] == proof['reference_base'] == SOURCE_CHAIN[3]
            and proof['parent_source_commit'] == SOURCE_CHAIN[4]
            and proof['new_source_commit'] == SOURCE_CHAIN[5]
            and proof['changed_paths'] == list(PARENT_PRECISION_PATHS)
            and proof['normal_cpu_outputs_changed'] is True and proof['prior_science_reused'] is False
            and set(proof['checks']) == checks and all(v is True for v in proof['checks'].values())
            and isinstance(proof['scope'], str) and proof['scope'].strip()
            and re.findall(r'^diff --git a/(\S+) b/(\S+)$', proof['exact_git_diff'], re.MULTILINE)
            == [(name, name) for name in PARENT_PRECISION_PATHS], 'Precision provenance is incomplete')
    inputs = bound_inputs(proof['input_sha256'], remote, files)
    require({'setup-source-v4.py', 'source-v2.json', 'source-v3.json', 'source-v4.json',
             'inspiral-source-v2.bundle', 'inspiral-source-v3.bundle', 'inspiral-source-v4.bundle'}
            <= inputs, 'Precision provenance omits source inputs')
    chain = []
    for version, manifest, index in ((3, parent, 4), (4, source, 5)):
        name = f'inspiral-source-v{version}.bundle'
        header = bundle_header(art / name)
        require(header == dict(prerequisites=[SOURCE_CHAIN[index - 1]], heads=[SOURCE_CHAIN[index]])
                and files[name]['sha256'] == manifest['bundle_sha256'],
                f'Precision source bundle differs: {name}')
        chain.append(header)
    old_proof = read(art / 'source-precision-provenance.json')
    require(old_proof['new_source_commit'] == SOURCE_CHAIN[4]
            and old_proof['old_source_commit'] == SOURCE_CHAIN[3]
            and old_proof['normal_cpu_outputs_changed'] is True
            and old_proof['prior_science_reused'] is False,
            'Preserved first precision provenance differs')
    bound_inputs(old_proof['input_sha256'], remote, files)
    return source, remote, inputs, chain


def qualify_precision_source(art, files, report):
    parent, remote, _, chain = qualify_precision_parent(art, files)
    source = read(art / 'source-v5.json')
    require(source['source'] == str(remote / 'source-v5')
            and source['commit'] == SOURCE_CHAIN[6]
            and source['parent'] == source['parent_source_commit'] == SOURCE_CHAIN[5]
            and source['reference_base'] == SOURCE_CHAIN[3]
            and source['changed_paths'] == list(PRECISION_PATHS)
            and source['native_modules_sha256'] == parent['native_modules_sha256']
            and set(source['changed_files_sha256']) == set(PRECISION_PATHS)
            and all(re.fullmatch('[0-9a-f]{64}', digest)
                    for digest in source['changed_files_sha256'].values())
            and all(source['changed_files_sha256'][name] == parent['changed_files_sha256'][name]
                    for name in PRECISION_PATHS if name not in PARENT_CHANGED_PATHS),
            'Final precision source identity or reviewed blobs differ')
    proof = read(art / 'source-precision-provenance-v5.json')
    checks = {'only_reviewed_python_and_test_paths_changed', 'native_hashes_equal',
              'both_sources_clean', 'all_final_runs_require_new_source',
              'host_launch_and_regression_only_parent_delta'}
    require(proof['schema_version'] == 1 and proof['status'] == 'pass'
            and proof['old_source_commit'] == proof['reference_base'] == SOURCE_CHAIN[3]
            and proof['parent_source_commit'] == SOURCE_CHAIN[5]
            and proof['new_source_commit'] == SOURCE_CHAIN[6]
            and proof['changed_paths'] == list(PRECISION_PATHS)
            and proof['parent_changed_paths'] == list(PARENT_CHANGED_PATHS)
            and proof['normal_cpu_outputs_changed'] is True and proof['prior_science_reused'] is False
            and set(proof['checks']) == checks and all(v is True for v in proof['checks'].values())
            and isinstance(proof['scope'], str) and proof['scope'].strip()
            and re.findall(r'^diff --git a/(\S+) b/(\S+)$', proof['exact_git_diff'], re.MULTILINE)
            == [(name, name) for name in PRECISION_PATHS]
            and re.findall(r'^diff --git a/(\S+) b/(\S+)$', proof['exact_parent_git_diff'], re.MULTILINE)
            == [(name, name) for name in PARENT_CHANGED_PATHS], 'Final precision provenance is incomplete')
    launch_diff = proof['exact_parent_git_diff'].split('diff --git a/test/', 1)[0]
    changed_lines = [line for line in launch_diff.splitlines()
                     if line.startswith(('+', '-')) and not line.startswith(('+++', '---'))
                     and not line[1:].lstrip().startswith('#')]
    require(len(re.findall(r'^@@ ', launch_diff, re.MULTILINE)) == 1
            and 'def _accelerator_batched_bin_sums(corr, pts, bins):' in launch_diff
            and changed_lines == ['-            two_pi_over_N,',
                                  '+            tl.constexpr(two_pi_over_N),'],
            'Final parent runtime delta exceeds the reviewed host launch')
    require(report['source_provenance'] == dict(receipt='source-precision-provenance-v5.json',
            changed_paths=proof['changed_paths'], checks=proof['checks'],
            normal_cpu_outputs_changed=True, prior_science_reused=False),
            'Report source provenance differs')
    inputs = bound_inputs(proof['input_sha256'], remote, files)
    require({'setup-source-v5.py', *(f'source-v{version}.json' for version in (2, 3, 4, 5)),
             *(f'inspiral-source-v{version}.bundle' for version in (2, 3, 4, 5))} <= inputs,
            'Final precision provenance omits source inputs')
    header = bundle_header(art / 'inspiral-source-v5.bundle')
    require(header == dict(prerequisites=[SOURCE_CHAIN[5]], heads=[SOURCE_CHAIN[6]])
            and files['inspiral-source-v5.bundle']['sha256'] == source['bundle_sha256'],
            'Final precision source bundle differs')
    return source, remote, inputs, chain + [header]


def qualify_precision_tests(art, files, source, remote, report=None, version=5):
    require(version in (4, 5), 'Unreviewed unit-test source version')
    source_name, unit_name = f'source-v{version}', f'unit-tests-v{version}'
    config, tests = (read(art / name) for name in ('config.json', unit_name + '.json'))
    command = tests['command']
    require(tests['state'] == 'complete' and tests['passed'] is True and tests['returncode'] == 0
            and tests.get('finished_utc') and tests['source_info'] == tests['source_after']
            == dict(commit=SOURCE_CHAIN[version + 1], status='')
            and tests['cwd'] == source['source'] and tests['host'] == 'len'
            and tests['environment'] == dict(config['environment'], PYTHONPATH=source['source'],
                                             PYTHONDONTWRITEBYTECODE='1')
            and command[:3] == ['taskset', '-c', str(config['core'])]
            and Path(command[3]).is_absolute()
            and command[4:] == ['-m', 'pytest', '-q', '-p', 'no:cacheprovider', *UNIT_TESTS]
            and tests['input_sha256'] == tests['input_sha256_after']
            and tests['log_sha256'] == files[unit_name + '.log']['sha256'],
            'Final source unit tests are incomplete or changed')
    inputs = bound_inputs(tests['input_sha256'], remote, files)
    require({f'run-unit-checks-v{version}.py', 'config.json', source_name + '.json'} <= inputs
            and all(tests['input_sha256'].get(str(remote / source_name / name)) == digest
                    for name, digest in source['changed_files_sha256'].items())
            and all(str(remote / source_name / name) in tests['input_sha256']
                    for name in ('bin/pycbc_inspiral', 'pycbc/scheme.py', *source['changed_paths'], *UNIT_TESTS)),
            'Final unit tests omit required reviewed source/test hashes')
    lines = [line.strip() for line in (art / (unit_name + '.log')).read_text().splitlines() if line.strip()]
    require(lines and re.search(r'\b[1-9][0-9]* passed\b', lines[-1])
            and not re.search(r'\b[1-9][0-9]* (?:failed|errors?)\b', lines[-1]),
            'Final unit-test log differs')
    if report is not None:
        require(report['unit_tests']['receipt'] == unit_name + '.json'
                and report['unit_tests']['log'] == unit_name + '.log'
                and report['unit_tests']['source_commit'] == SOURCE_CHAIN[version + 1]
                and report['unit_tests']['command'] == command
                and report['unit_tests']['summary'] == lines[-1], 'Final unit-test report differs')
    failed = read(art / 'unit-tests-v3.json')
    require(failed['state'] == 'complete' and failed['passed'] is False and failed['returncode'] == 1
            and failed.get('finished_utc') and failed['source_info'] == failed['source_after']
            == dict(commit=SOURCE_CHAIN[4], status='')
            and failed['input_sha256'] == failed['input_sha256_after']
            and failed['log_sha256'] == files['unit-tests-v3.log']['sha256']
            and re.search(r'\b2 failed, 363 passed, 68 skipped,.*8 subtests passed in [0-9.]+s\s*\Z',
                          (art / 'unit-tests-v3.log').read_text()),
            'Preserved v3 unit-test failure changed')
    bound_inputs(failed['input_sha256'], remote, files)
    return inputs


def qualify_precision_smoke(art, files, remote, version):
    require(version in (4, 5), 'Unreviewed smoke source version')
    prefix = 'precision' if version == 4 else 'precision5'
    filename = 'precision-backend-smoke-parity-v2.json' if version == 4 else 'precision5-backend-smoke-parity.json'
    smoke = read(art / filename)
    require(smoke['schema_version'] == 1 and smoke['status'] == ('fail' if version == 4 else 'pass')
            and len(smoke['comparisons']) == 2
            and smoke['tolerances'] == dict(rtol=1e-4, atol=1e-5, sigmasq_rtol=1e-5,
                                             phase_atol=1e-4, max_examples=12),
            'Smoke parity changed')
    expected = {f'qual-{prefix}-torch-cpu-l512': 'pass',
                f'qual-{prefix}-torch-cuda-l512': 'fail' if version == 4 else 'pass'}
    require({Path(item['candidate']).parent.name: item['status']
             for item in smoke['comparisons']} == expected,
            'Smoke outcomes changed')
    for item in smoke['comparisons']:
        require(item['baseline'] == str(remote / f'runs/qual-{prefix}-cpu-l512/triggers.hdf'),
                'Smoke CPU baseline changed')
        for role in ('baseline', 'candidate'):
            trigger = Path(item[role])
            require(trigger.is_relative_to(remote / 'runs') and trigger.name == 'triggers.hdf',
                    'Smoke path differs')
            bound_inputs({str(trigger): item[role + '_sha256'],
                          str(trigger.with_name('receipt.json')): item[role + '_receipt_sha256']},
                         remote, files)
            receipt = read(art / trigger.relative_to(remote).with_name('receipt.json'))
            require(receipt['state'] == 'complete' and receipt['returncode'] == 0
                    and receipt['source_info'] == dict(commit=SOURCE_CHAIN[version + 1], status='', tracked_diff='')
                    and receipt['source_status_after'] == '' and receipt.get('finished_utc')
                    and receipt['input_sha256'] == receipt['input_sha256_after'],
                    'Smoke execution differs')
            bound_inputs(receipt['input_sha256'], remote, files)


def qualify_precision_history(art, files, remote):
    """Preserve the passing v4 units and failed CUDA smoke as supplemental evidence."""
    qualify_precision_tests(art, files, read(art / 'source-v4.json'), remote, version=4)
    require(files['precision-backend-smoke-parity.json']['bytes'] == 0,
            'Preserved unsuccessful parity output was overwritten')
    qualify_precision_smoke(art, files, remote, version=4)
    for dirname, failed in (('chisq-capture-v4-cpu', True), ('chisq-capture-v4b-cpu', False),
                            ('chisq-capture-v4b-torch-cuda', False)):
        capture = read(art / dirname / 'status.json')
        require(capture['state'] == ('failed' if failed else 'complete')
                and capture['returncode'] == int(failed) and capture.get('finished_utc')
                and capture['source_info'] == dict(commit=SOURCE_CHAIN[5], status='', tracked_diff='')
                and capture['source_status_after'] == ''
                and capture['input_sha256'] == capture['input_sha256_after']
                and len(capture['captures']) == 3
                and (failed or capture['observed_filter_segments'] == list(range(5))),
                f'Preserved coefficient capture differs: {dirname}')
        bound_inputs(capture['input_sha256'], remote, files)
        arrays = [capture['conditioned_strain'],
                  *(array for item in capture['captures'] for array in item['arrays'].values())]
        require(all(Path(array['path']).is_relative_to(remote / dirname) for array in arrays),
                'Preserved capture array path differs')
        bound_inputs({array['path']: array['sha256'] for array in arrays}, remote, files)
    coefficient = read(art / 'chisq-coefficient-v4.json')
    require(coefficient['state'] == 'complete' and coefficient['source_commit'] == SOURCE_CHAIN[5]
            and coefficient['source_status_after'] == ''
            and coefficient['input_sha256'] == coefficient['input_sha256_after']
            and coefficient['specialization_plain'] == 'fp32'
            and coefficient['specialization_constexpr'] == 'constexpr'
            and len(coefficient['rows']) == 6
            and {(row['origin'], row['segment']['number']) for row in coefficient['rows']}
            == {(origin, segment) for origin in ('cpu', 'torch-cuda') for segment in (1, 2, 3)}
            and sum(row['violations']['original'] for row in coefficient['rows']) == 4
            and all(row['violations']['constexpr_coefficient'] == row['violations']['torch_fallback'] == 0
                    for row in coefficient['rows']), 'Preserved coefficient diagnosis differs')
    bound_inputs(coefficient['input_sha256'], remote, files)


def qualify_precision_orchestration(art, files, remote, cases):
    """Bind final orchestration and smoke evidence outside the 31 report cases."""
    qualify_precision_smoke(art, files, remote, version=5)
    prefix = 'precision5-final-campaign'
    receipt = read(art / (prefix + '.status.json'))
    profiles = {case: mode for group in cases for case, (mode, _, _) in group.items()
                if mode in ('cprofile', 'perf')}
    perf = sorted(case for case, mode in profiles.items() if mode == 'perf')
    stages = ['qualifications', 'reference-profiles', 'matched-timings', 'torch-profiles',
              *('perf-export-' + case for case in perf), 'summary']
    require(receipt['state'] == 'complete' and receipt['returncode'] == 0
            and receipt.get('finished_utc') and receipt.get('current') is None
            and not receipt.get('error') and receipt['host'] == 'len' and receipt['cwd'] == str(remote)
            and receipt['source_info'] == receipt['source_after'] == dict(commit=SOURCE_CHAIN[6], status='')
            and receipt['input_sha256'] == receipt['input_sha256_after']
            and len(receipt['completed']) == 8
            and {item['name'] for item in receipt['completed']} == set(stages)
            and [item['name'] for item in receipt['completed'][:4]] == stages[:4]
            and receipt['completed'][-1]['name'] == 'summary'
            and all(item['returncode'] == 0 for item in receipt['completed']),
            'Final orchestration is incomplete or changed')
    inputs = bound_inputs(receipt['input_sha256'], remote, files)
    require({'run-final-reference-campaign-v5.py', 'select-precision-reference-v5.py',
             'precision5-reference-tuning-decision.json', 'precision5-reference-campaign.status.json',
             'unit-tests-v5.json', 'unit-tests-v5.log',
             *(name + '.json' for name in REFERENCE_PLANS[2:])} <= inputs,
            'Final orchestration omits prerequisites')
    outputs = bound_inputs(receipt['output_sha256'], remote, files)
    expected = {name + '.status.json' for name in REFERENCE_PLANS[2:]}
    expected.update(f'runs/{case}/receipt.json' for group in cases[2:] for case in group)
    expected.update(f'runs/{case}/{name}' for case, mode in profiles.items()
                    for name in ('triggers.hdf', 'perf.data' if mode == 'perf' else 'profile.pstats'))
    expected.update(f'runs/{case}/perf-report.txt' for case in perf)
    expected.add('profiles-precision5.json')
    require(outputs == expected, 'Final orchestration output manifest differs')
    for item in receipt['completed']:
        for field in ('stdout', 'stderr'):
            if item[field] is not None:
                path = Path(item[field])
                require(path.is_relative_to(remote) and str(path.relative_to(remote)) in files,
                        'Final orchestration log was not archived')


def qualify_reference(art, files, supporting_art):
    report = read(art / 'final-report/report.json')
    require(report['schema_version'] == 1 and report['status'] == 'pass'
            and set(report['gates']) == REFERENCE_GATES
            and all(value is True for value in report['gates'].values())
            and report['source_phases'] == dict(precision=SOURCE_CHAIN[6]),
            'Final precision report is not completely qualified')
    manifest = report['input_sha256']
    require(isinstance(manifest, dict) and manifest and all(name in files
            and files[name]['sha256'] == digest for name, digest in manifest.items()),
            'Missing/changed final reference report input')
    decision = read(art / 'precision5-reference-tuning-decision.json')
    selected = decision['selected_segment_length_seconds']
    cases = reference_cases(selected)
    require(decision['source_commit'] == SOURCE_CHAIN[6]
            and decision['selected_start_pad_seconds'] == 112 and decision['selected_end_pad_seconds'] == 16
            and report['selected_geometry'] == dict(segment_length_seconds=selected,
            start_pad_seconds=112, end_pad_seconds=16, basis=decision['basis']),
            'Selected precision geometry differs')
    source, remote, source_inputs, chain = qualify_precision_source(art, files, report)
    required_inputs = source_inputs | qualify_precision_tests(art, files, source, remote, report) | {
        'source-v2.json', 'source-v3.json', 'source-v4.json', 'source-v5.json', 'inspiral-source-v2.bundle',
        'inspiral-source-v3.bundle', 'inspiral-source-v4.bundle', 'inspiral-source-v5.bundle', 'source-precision-provenance-v5.json',
        'setup-source-v5.py', 'config.json', 'environment.json', 'build-reference-report-v4.py',
        'waveform-validation-precision5.json', 'boundary-injections-precision5.json',
        'compression-1e5.json', 'precision5-reference-tuning-decision.json', 'profiles-precision5.json',
        'unit-tests-v5.json', 'unit-tests-v5.log', 'run-unit-checks-v5.py'}
    required_inputs.update(bound_inputs(decision['input_sha256'], remote, files))
    all_cases = {}
    for name, expected in zip(REFERENCE_PLANS, cases):
        plan_name, status_name = name + '.json', name + '.status.json'
        required_inputs.update((plan_name, status_name))
        plan, status = read(art / plan_name), read(art / status_name)
        require(isinstance(plan, list) and plan and status['state'] == 'complete'
                and status['returncode'] == 0 and status.get('finished_utc') and not status.get('error')
                and status.get('current') is None and status['completed'] == plan
                and status['plan_sha256'] == files[plan_name]['sha256'],
                f'Incomplete/changed precision campaign ledger: {name}')
        observed = []
        for command in plan:
            require(isinstance(command, list) and len(command) % 2 == 0
                    and len(set(command[::2])) == len(command[::2]), f'Ambiguous command: {name}')
            options = dict(zip(command[::2], command[1::2]))
            require({'--case', '--mode', '--scheme', '--segment-length'} <= options.keys()
                    and options.keys() <= {'--case', '--mode', '--scheme', '--segment-length',
                                           '--start-pad', '--end-pad'}, f'Unexpected options: {name}')
            case = options['--case']
            require(case in expected and expected[case] == (options['--mode'], options['--scheme'],
                    int(options['--segment-length'])) and int(options.get('--start-pad', 112)) == 112
                    and int(options.get('--end-pad', 16)) == 16, f'Unexpected precision case: {case}')
            observed.append(case)
            receipt_name = f'runs/{case}/receipt.json'
            required_inputs.add(receipt_name)
            receipt = read(art / receipt_name)
            require(receipt['state'] == 'complete' and receipt['returncode'] == 0
                    and receipt.get('finished_utc') and receipt['case'] == case
                    and receipt['source_info']['commit'] == SOURCE_CHAIN[6]
                    and receipt['source_info']['status'] == '' and receipt['source_info']['tracked_diff'] == ''
                    and receipt['source_status_after'] == '' and receipt['source'] == source['source']
                    and receipt['cwd'] == str(remote) and receipt['hostname'] == 'len'
                    and (receipt['mode'], receipt['scheme'], receipt['segment_length']) == expected[case]
                    and (receipt['start_pad'], receipt['end_pad']) == (112, 16)
                    and receipt['input_sha256'] == receipt['input_sha256_after'],
                    f'Incomplete/changed final precision run: {case}')
            required_inputs.update(bound_inputs(receipt['input_sha256'], remote, files))
        require(len(observed) == len(expected) and set(observed) == set(expected),
                f'Unexpected precision plan case set: {name}')
        all_cases.update(expected)
    require(len(all_cases) == 31 and report['required_cases'] == sorted(all_cases),
            'Final reference case set differs')
    expected_pairs = {(f'qual-selected5-cpu-l{length}' if length == selected
                       else f'qual-precision5-cpu-l{length}', case)
                      for case, (_, _, length) in all_cases.items()}
    expected_pairs = {(baseline, case) for baseline, case in expected_pairs if baseline != case}
    parity = report['parity']
    require(parity['status'] == 'pass' and len(parity['comparisons']) == 28
            and {(item['baseline'], item['candidate']) for item in parity['comparisons']} == expected_pairs
            and all(item['status'] == 'pass' and item['source_phase'] == 'precision'
                    for item in parity['comparisons'])
            and json.dumps(parity['tolerances'], sort_keys=True) == json.dumps(dict(
                rtol=1e-4, atol=1e-5, sigmasq_rtol=1e-5, phase_atol=1e-4, max_examples=12), sort_keys=True),
            'Final strict trigger parity is incomplete or changed')
    outputs = report['output_sha256']
    require(len(report['figures']) == 6 and set(report['figures']) == REFERENCE_FIGURES
            and set(outputs) == REFERENCE_FIGURES | {'runs.csv', 'groups.csv', 'profiles.csv', 'native-symbols.csv'}
            and all('final-report/' + name in files and files['final-report/' + name]['sha256'] == digest
                    for name, digest in outputs.items()), 'Missing/changed final reference outputs')
    for name, field, count, old_name in (
            ('waveform-validation-precision5.json', 'template_psd_pairs', 288, 'waveform-validation-1e5.json'),
            ('boundary-injections-precision5.json', 'cases', 36, 'boundary-injections-1e5.json')):
        validation = read(art / name)
        require(validation['state'] == 'complete' and validation['passed'] is True
                and validation['completed_' + field] == validation['expected_' + field] == count
                and validation.get('finished_utc') and validation['input_sha256'] == validation['input_sha256_after']
                and validation['source_before'] == validation['source_after']
                and validation['source_before']['commit'] == SOURCE_CHAIN[6]
                and validation['source_before']['status'] == ''
                and json.dumps(validation['tolerances'], sort_keys=True)
                == json.dumps(read(art / old_name)['tolerances'], sort_keys=True),
                f'Final precision validation failed or changed budgets: {name}')
        required_inputs.update(bound_inputs(validation['input_sha256'], remote, files))
    require(required_inputs <= manifest.keys(), 'Final report omits required precision inputs: '
            + ', '.join(sorted(required_inputs - manifest.keys())))
    qualify_precision_history(art, files, remote)
    qualify_precision_orchestration(art, files, remote, cases)
    historical_chain = qualify_reference_history(art, files, supporting_art)
    return dict(report='final-report/report.json', input_count=len(manifest),
                required_cases=sorted(all_cases), parity_comparisons=28,
                selected_segment_length_seconds=selected, source_bundle_chain=historical_chain + chain,
                source_phases=report['source_phases'], prior_science_reused=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--art', type=Path, default=HERE.parent)
    parser.add_argument('--reference', type=Path,
                        default=HERE.parent.parent / 'torch-inspiral-reference-20260906')
    parser.add_argument('--archive', type=Path, default=HERE.parent.parent / 'torch-benchmark-20260906/evidence')
    parser.add_argument('--plan', type=Path, default=HERE / 'evidence-staging-plan.json')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    require(all(not p.is_symlink() for p in (args.art, args.reference, args.archive)), 'Symlink root')
    art, reference, archive = (p.resolve(strict=True) for p in (args.art, args.reference, args.archive))
    roots = (art, reference, archive)
    require(all(not left.is_relative_to(right) for i, left in enumerate(roots)
                for j, right in enumerate(roots) if i != j), 'Roots overlap')
    require(not args.plan.is_symlink() and not args.plan.resolve().is_relative_to(archive)
            and not args.plan.resolve().is_relative_to(reference),
            'Plan must be a local file outside the archive and reference evidence')
    destinations = [archive / name for name in DESTINATIONS]
    require(all(not dest.exists() and not dest.is_symlink() for dest in destinations),
            'A destination already exists; inspect any partial copy before retrying')
    files = inventory(art)
    qualify(art, files)
    reference_files, duplicates = reference_inventory(reference)
    qualification = qualify_reference(reference, reference_files, art)
    supplements = {
        DESTINATIONS[0]: dict(source=str(art), destination=str(destinations[0]), files=files),
        DESTINATIONS[1]: dict(source=str(reference), destination=str(destinations[1]),
                             files=reference_files, excluded_duplicates=duplicates,
                             qualification=qualification),
    }
    plan = dict(schema='torch-reference-and-performance-fix-stage-v4',
                generator=record(Path(__file__)), supplements=supplements,
                historical_files=history(archive))
    if args.plan.exists():
        require(read(args.plan) == plan, 'Frozen plan differs; inspect before choosing a new plan')
    else:
        require(not args.execute, 'Run inventory mode first')
        with args.plan.open('x') as stream:
            stream.write(json.dumps(plan, indent=2, sort_keys=True) + '\n')
    if args.execute:
        for supplement in supplements.values():
            dest = Path(supplement['destination'])
            dest.mkdir()
            for name, item in supplement['files'].items():
                target = dest / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(item['source'], target)
                target.chmod(item['mode'])
                require(record(target) == {k: item[k] for k in ('sha256', 'bytes', 'mode')},
                        f'Copy changed: {dest.name}/{name}')
            with (dest / 'SHA256SUMS').open('x') as stream:
                stream.write(''.join(f'{item["sha256"]}  {name}\n'
                                    for name, item in supplement['files'].items()))
        require(inventory(art) == files and reference_inventory(reference) == (reference_files, duplicates)
                and history(archive) == plan['historical_files'], 'Inputs/history changed')
    print(json.dumps(dict(mode='copied' if args.execute else 'planned', plan=str(args.plan),
                          supplements={name: dict(destination=item['destination'], files=len(item['files']),
                          bytes=sum(v['bytes'] for v in item['files'].values()))
                          for name, item in supplements.items()})))


if __name__ == '__main__':
    main()
