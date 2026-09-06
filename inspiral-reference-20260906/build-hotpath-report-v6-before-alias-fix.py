#!/usr/bin/env python3
"""Validate frozen v6 evidence and write a new optimized-report-v6 directory.

Requires --source-manifest-sha256 from the reviewed final source-v6.json.
The archive may omit remote checkouts/native binaries and the original frame:
their hashes remain bound by the source, unit and campaign receipts. No
workloads execute here. The v5 report is immutable comparison evidence only.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import re
import statistics
import sys


HERE = Path(__file__).resolve().parent
HELPER_SHA = 'ad303e00008dd4792353ae23cb0856087bf7a643388d1e9b4391de2f8801ccda'
CAMPAIGN_SHA = 'dae70e88e37c537d82559e9bd6fd975419c6c3a191d0a128b77ae98386d28197'
OLD_REPORT_SHA = '11f17262ab7ec11c21cb50bdb8dd884379414a56fb22a4bbfd8ff934365f8b14'
COMPARATOR_SHA = 'f0af115a2bf2d3a5a85152cb1f61d9b7570a460efd6d420566c5a2b74ac8f1b5'
SCIENCE_SHA = '512a5511b3fda9d93b9f13e008c9316a9c8daecd238130b3ba4d621be1d308ea'
FIGURES = ('capacity-before-after.png', 'wall-breakdown.png', 'hotpath-profile-self-time.png')
# Exact dependencies recorded by both accepted v5 scientific diagnostics.
# Their remote bytes are not represented as local archived files.
EXTERNAL_PREFIX = '/home/xangma/miniconda3/envs/pycbc3g/lib/python3.11/site-packages/'
EXTERNAL_INPUTS = {EXTERNAL_PREFIX + name: digest for name, digest in {
    'lal/__init__.py': '5229a365707f57bac4b377b0f03f62f4edf52d90a351b3a3eac6c38ee8bfb087',
    'lal/_lal.cpython-311-x86_64-linux-gnu.so':
        'baa61ab6b4b9dd9bb0666bfced783ec958132bfed223686b2af70a7c30860b5e',
    'lalsimulation/__init__.py': 'c3e1826c9321eb2f0485b322737ac292e98396363f58fb2af38b520e2ea8b0ee',
    'lalsimulation/_lalsimulation.cpython-311-x86_64-linux-gnu.so':
        'ae3b2766e7d2d0192062cf5e7df11b34c0b84e5eeebccf24ba6984f539432bf2',
}.items()}


def load_helper(root, name, expected):
    contents = (root / name).read_bytes()
    if hashlib.sha256(contents).hexdigest() != expected:
        raise ValueError('Unreviewed helper: ' + name)
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), root / name)
    module = importlib.util.module_from_spec(spec)
    exec(compile(contents, str(root / name), 'exec'), module.__dict__)
    return module


V4 = load_helper(HERE, 'build-reference-report-v4.py', HELPER_SHA)
CAMP = load_helper(HERE, 'campaign-v6.py', CAMPAIGN_SHA)
require, sha, positive, canonical = V4.require, V4.sha, V4.positive, V4.canonical
UNIT_TESTS = ('test/test_torch_decompress_cpu.py', 'test/test_torch_large_ifft.py',
              'test/test_decompress.py', 'test/test_torch_fft_cpu_native.py',
              'test/test_torch_fft_writes.py', 'test/test_torch_fft_cuda_workspace.py',
              *V4.UNIT_TESTS)
SCIENCE_ALIASES = {
    'waveform-validation-precision5.json': 'waveform-validation-v6.json',
    'boundary-injections-precision5.json': 'boundary-injections-v6.json',
    'runs/qual-precision5-cpu-l1024/qualification.json':
        'runs/qual-science6-cpu-l1024/qualification.json',
    'profiles-precision5.json': 'profiles-v6.json',
}


class Evidence(V4.Evidence):
    def __init__(self, root):
        super().__init__(root)
        self.archived_remote_inputs = {}
        self.commit = None
        self.science_aliases = False

    def local_name(self, recorded):
        name = super().local_name(recorded)
        require(name is None or '..' not in Path(name).parts,
                'Recorded path escapes archive: ' + str(recorded))
        if self.science_aliases:
            inverse = {value: key for key, value in SCIENCE_ALIASES.items()}
            return inverse.get(name, name)
        return name

    def bind(self, relative, expected=None):
        require(relative is not None, 'A required receipt path is outside the archive')
        return super().bind(SCIENCE_ALIASES.get(str(relative), relative), expected)

    def recorded_inputs(self, before, after=None):
        require(isinstance(before, dict) and before and (after is None or before == after),
                'Missing or changed recorded inputs')
        for name, digest in before.items():
            require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest),
                    'Malformed recorded hash: ' + name)
            relative = self.local_name(name)
            source_match = re.match(r'^(source(?:-v[2-6])?)/(.+)$', relative or '')
            if source_match or relative is None:
                if source_match:
                    if source_match[1] == 'source-v6' and self.sources:
                        manifest = self.sources['optimized']
                        expected = {**manifest['changed_files_sha256'],
                                    **manifest['native_modules_sha256']}.get(source_match[2])
                        require(expected is None or digest == expected,
                                'Recorded source content differs: ' + name)
                else:
                    require((name == self.frame_path and digest == CAMP.FRAME_SHA256) or
                            EXTERNAL_INPUTS.get(name) == digest,
                            'Unrecognized external input: ' + name)
                previous = self.archived_remote_inputs.get(name)
                require(previous is None or previous == digest,
                        'Inconsistent remote input hash: ' + name)
                self.archived_remote_inputs[name] = digest
            else:
                self.bind(relative, digest)


def required(evidence, inputs, names):
    V4.required_inputs(evidence, inputs, names)


def passed(evidence, value, label):
    require(value['state'] == 'complete' and value['passed'] is True and
            value['returncode'] == 0 and value.get('finished_utc') and
            value['source_info'] == value['source_after'] ==
            dict(commit=evidence.commit, status=''), label + ': incomplete, failed or stale source')
    evidence.recorded_inputs(value['input_sha256'], value['input_sha256_after'])


def read_source(evidence, source_manifest_sha256, config):
    require(re.fullmatch('[0-9a-f]{64}', source_manifest_sha256 or ''),
            'A reviewed source manifest SHA256 is required')
    evidence.bind('source-v6.json', source_manifest_sha256)
    source = evidence.read('source-v6.json')
    source_path = Path(source['source'])
    require(source_path.is_absolute() and source_path.name == 'source-v6', 'Wrong source location')
    evidence.remote, evidence.commit = source_path.parent, source['commit']
    require(len(config['input_files']) == 1, 'Wrong frame selection')
    evidence.frame_path = config['input_files'][0]
    previous = evidence.read('source-v5.json')
    require(source['schema_version'] == 1 and re.fullmatch('[0-9a-f]{40}', evidence.commit) and
            evidence.commit != CAMP.PARENT and source['parent'] == previous['commit'] == CAMP.PARENT and
            source['changed_paths'] == list(CAMP.ALLOWED) and
            set(source['changed_files_sha256']) == set(CAMP.ALLOWED), 'Wrong reviewed source delta')
    native = source['native_modules_sha256']
    require(len(native) == 11 and native == previous['native_modules_sha256'] and
            all(name.startswith('pycbc/') and name.endswith('.so') and '..' not in Path(name).parts
                for name in native), 'Native module set differs from v5')
    require(all(re.fullmatch('[0-9a-f]{64}', digest) for digest in
                [*native.values(), *source['changed_files_sha256'].values()]), 'Malformed source hash')
    audit = source['normal_cpu_path_audit']
    require(audit['status'] == 'pass' and audit['source_commit'] == evidence.commit and
            audit['parent_source_commit'] == CAMP.PARENT and audit['dispatch'] == CAMP.DISPATCH and
            isinstance(audit['rationale'], str) and audit['rationale'].strip(), 'Missing normal CPU audit')
    paths = re.findall(r'^diff --git a/(\S+) b/(\S+)$', source['exact_git_diff'], re.MULTILINE)
    require(paths == [(name, name) for name in CAMP.ALLOWED], 'Source diff includes unreviewed paths')
    evidence.sources['optimized'] = source
    evidence.bind('inspiral-source-v6.bundle', source['bundle_sha256'])
    evidence.recorded_inputs(source['input_sha256'], source['input_sha256_after'])
    required(evidence, source['input_sha256'],
             ('setup-source-v6.py', 'inspiral-source-v6.bundle', 'source-v5.json'))
    evidence.gates['source_provenance'] = True
    return source


def unit_tests(evidence, config, source):
    receipt = evidence.read('unit-tests-v6.json')
    passed(evidence, receipt, 'Final unit tests')
    require(receipt['host'] == 'len' and receipt['cwd'] == source['source'] and
            receipt['environment'] == dict(config['environment'], PYTHONPATH=source['source'],
                                           PYTHONDONTWRITEBYTECODE='1'), 'Wrong unit-test environment')
    command = receipt['command']
    require(command[:3] == ['taskset', '-c', str(config['core'])] and Path(command[3]).is_absolute() and
            command[4:] == ['-m', 'pytest', '-q', '-p', 'no:cacheprovider', *UNIT_TESTS],
            'Final tests omit the required regression suite')
    required(evidence, receipt['input_sha256'], ('run-unit-checks-v6.py', 'source-v6.json', 'config.json'))
    require(all(str(Path(source['source']) / name) in receipt['input_sha256']
                for name in (*UNIT_TESTS, *CAMP.ALLOWED)), 'Missing tested-source hashes')
    log = evidence.bind('unit-tests-v6.log', receipt['log_sha256'])
    terminal = [line.strip() for line in log.read_text().splitlines() if line.strip()][-1]
    require(re.search(r'\b[1-9][0-9]* passed\b', terminal) and
            not re.search(r'\b[1-9][0-9]* (?:failed|errors?)\b', terminal), 'No passing unit-test summary')
    evidence.gates['unit_tests'] = True
    return dict(receipt='unit-tests-v6.json', summary=terminal, command=command)


def read_stages(evidence, config, source):
    expected = {}
    for stage in ('qualifications', 'measurements'):
        name = f'campaign-v6-{stage}'
        ledger = evidence.read(name + '.status.json')
        plan = CAMP.case_plan(stage)
        require(ledger['schema_version'] == 1 and ledger['state'] == 'complete' and
                ledger['returncode'] == 0 and ledger['stage'] == stage and ledger.get('finished_utc') and
                ledger['source_info'] == ledger['source_after'] == dict(commit=evidence.commit, status='') and
                ledger['plan'] == ledger['completed'] == plan and
                ledger['current'] is None and ledger['child_pid'] is None,
                'Incomplete or incorrect campaign stage: ' + stage)
        require(ledger['host'] == 'len' and ledger['cwd'] == str(evidence.remote) and
                ledger['environment'] == dict(config['environment'], PYTHONPATH=source['source'],
                                               PYTHONDONTWRITEBYTECODE='1'), 'Wrong campaign environment')
        launch = evidence.read(name + '-launch.json')
        require(launch['state'] == 'launched' and launch['host'] == ledger['host'] and
                launch['cwd'] == ledger['cwd'] and launch['pid'] == ledger['pid'] and
                launch['source_commit'] == evidence.commit and
                launch['source_manifest_sha256'] == evidence.input_sha256['source-v6.json'],
                'Launch does not identify this completed stage')
        evidence.recorded_inputs(launch['input_sha256'])
        require(evidence.local_name(launch['log']) == name + '-launch.log', 'Wrong launch log')
        evidence.bind(name + '-launch.log')
        evidence.recorded_inputs(ledger['input_sha256'], ledger['input_sha256_after'])
        required(evidence, ledger['input_sha256'], (*CAMP.PINNED, 'source-v6.json', 'unit-tests-v6.json',
                  'campaign-v6.py', 'launch-campaign-v6.py', 'campaign-v6-contract.md'))
        if stage == 'measurements':
            required(evidence, ledger['input_sha256'],
                     ('scientific-validation-v6.json', 'campaign-v6-qualifications.status.json'))
        equivalence = ledger['normal_cpu_equivalence']
        require(equivalence['dispatch'] == CAMP.DISPATCH and
                equivalence['native_modules_sha256'] == source['native_modules_sha256'] and
                equivalence['reviewed_audit'] == source['normal_cpu_path_audit'] and
                equivalence['tuning_source_commit'] == CAMP.PARENT and
                set(equivalence['unchanged_sha256']) == set(CAMP.CPU_PATHS), 'Missing CPU equivalence proof')
        for path, digest in equivalence['unchanged_sha256'].items():
            require(ledger['input_sha256'].get(str(Path(source['source']) / path)) == digest,
                    'Unbound unchanged CPU path: ' + path)
        outputs = ledger['output_sha256']
        require(outputs, 'Missing stage output hashes')
        evidence.recorded_inputs(outputs)
        for row in plan:
            prefix = f"runs/{row['case']}"
            files = [str(path.relative_to(evidence.root)) for path in
                     sorted((evidence.root / prefix).rglob('*')) if path.is_file()]
            require(files, 'Missing run outputs: ' + row['case'])
            required(evidence, outputs, [*files, f"campaign-v6-{row['case']}.log"])
            expected[row['case']] = dict(row, segment_length=512, start_pad=112,
                                          end_pad=16, source_phase='optimized')
        for entry in ledger['commands']:
            require(entry['returncode'] == 0 and isinstance(entry['command'], list) and entry['command'],
                    'Failed stage command')
            required(evidence, outputs, [evidence.local_name(entry['log'])])
            if entry.get('stderr'):
                required(evidence, outputs, [evidence.local_name(entry['stderr'])])
        expected_commands = len(plan) + (4 if stage == 'measurements' else 0)
        require(len(ledger['commands']) == expected_commands, 'Missing campaign child commands')
        if stage == 'measurements':
            required(evidence, outputs, ('profiles-v6.json', 'campaign-v6-summary.log'))
    require(len(expected) == 19, 'Campaign requires exactly 19 distinct cases')
    evidence.gates['campaign_complete'] = True
    return expected


def large_ifft(evidence, config, source):
    decision = evidence.read('large-ifft-v6-decision.json')
    passed(evidence, decision, 'Large IFFT decision')
    dispatch = decision['dispatch_by_size']
    routes = {'mkl_double_workspace', 'fftw_double_workspace_one_native_thread'}
    require(decision['schema_version'] == 1 and decision['supported_threads'] == [1] and
            decision['promoted_native_dtype'] == 'complex128' and
            decision['direct_single_sizes_unchanged'] == [32768] and
            set(dispatch) == {str(2**power) for power in (20, 21, 22)} and
            set(dispatch.values()) <= routes and isinstance(decision['rationale'], str) and
            decision['rationale'].strip(), 'Unreviewed IFFT dispatch policy')
    matrix_name = evidence.local_name(decision['matrix']['path'])
    require(matrix_name == 'large-ifft-v6.json', 'Final matrix must be separate from candidate history')
    evidence.bind(matrix_name, decision['matrix']['sha256'])
    required(evidence, decision['input_sha256'],
             ('source-v6.json', 'qualify-large-ifft-v6b.py', matrix_name, 'large-ifft-v6.log'))
    matrix = evidence.read(matrix_name)
    before = matrix['source_before']
    require(matrix['schema'] == 'torch-large-ifft-qualification-v6b' and matrix['state'] == 'complete' and
            matrix.get('finished_utc') and before == matrix['source_after'] and
            before['head'] == evidence.commit and before['status'] == '' and
            before['root'] == source['source'] and
            before['torchfft_sha256'] == source['changed_files_sha256']['pycbc/fft/torchfft.py'] and
            matrix['threads'] == 1 and matrix['host'] == 'len' and matrix['cpu_affinity'] == [config['core']],
            'Incomplete, stale or differently threaded final IFFT matrix')
    evidence.bind('qualify-large-ifft-v6b.py', before['harness_sha256'])
    require(all(value == config['environment'][key] for key, value in matrix['environment'].items()),
            'Wrong IFFT numerical environment')
    sizes = matrix['sizes']
    require(len(sizes) == 3 and {str(row['size']) for row in sizes} == set(dispatch), 'Wrong IFFT size coverage')
    grid = set(itertools.product((7, 91, 812, 20260906), ('dense', 'banded', 'impulse'), (1e-12, 1., 1e12)))
    enabled = []
    for row in sizes:
        require(len(row['cases']) == 36 and {(case['seed'], case['pattern'], case['scale'])
                for case in row['cases']} == grid, 'Incomplete IFFT precision matrix')
        for name, timing in row['timings'].items():
            samples = timing['steady_seconds']
            require(len(samples) == 9 and timing['native_library_threads'] == 1 and
                    all(positive(value, 'IFFT steady time') for value in samples) and
                    timing['steady_median_seconds'] == statistics.median(samples) and
                    timing['steady_min_seconds'] == min(samples) and timing['steady_max_seconds'] == max(samples),
                    'Invalid IFFT timing summary: ' + name)
        if dispatch[str(row['size'])] == 'mkl_double_workspace':
            enabled.append(row['size'])
            require(row['timings']['mkl_double_workspace']['steady_median_seconds'] <
                    row['timings']['fftw_double_workspace_one_native_thread']['steady_median_seconds'],
                    'Enabled IFFT size has no measured steady gain')
            for case in row['cases']:
                gate = case['gates']['mkl_double_workspace']
                require(gate['passed'] is True and gate['bitwise_mkl_parity'] is True,
                        'Enabled IFFT size failed strict MKL parity')
                for key in ('l2', 'max_abs'):
                    require(positive(case['errors']['mkl_double_workspace'][key], key, zero=True) <=
                            positive(case['errors']['legacy_fftw_single'][key], key, zero=True),
                            'Enabled IFFT size worsens legacy FFTW error')
    if 'enabled_sizes' in decision:
        require(sorted(decision['enabled_sizes']) == sorted(enabled), 'Enabled-size decision disagrees with routes')
    for key in ('candidate_matrix', 'failed_historical_matrix'):
        if key in decision:
            item = decision[key]
            evidence.bind(evidence.local_name(item['path']), item['sha256'])
    evidence.gates['large_ifft'] = True
    return dict(receipt='large-ifft-v6-decision.json', matrix=matrix_name, cases=108,
                dispatch_by_size=dispatch, supported_threads=[1], promoted_native_dtype='complex128')


def science(evidence, bank_hashes, config):
    receipt = evidence.read('scientific-validation-v6.json')
    passed(evidence, receipt, 'Scientific integration')
    require(receipt['schema_version'] == 1 and set(receipt['checks']) == CAMP.SCIENCE_CHECKS and
            all(value is True for value in receipt['checks'].values()) and
            set(receipt['evidence']) == CAMP.SCIENCE_CHECKS, 'Incomplete scientific integration')
    names = ['source-v6.json', 'unit-tests-v6.json', 'campaign-v6-qualifications.status.json']
    names += [f"runs/{case['case']}/{name}" for case in CAMP.case_plan('qualifications')
              for name in ('qualification.json', 'triggers.hdf')]
    for item in receipt['evidence'].values():
        name = evidence.local_name(item['path'])
        require(name is not None, 'Science receipt is outside the archive')
        evidence.bind(name, item['sha256'])
        names.append(name)
    required(evidence, receipt['input_sha256'], names)
    runner = load_helper(evidence.root, 'run-scientific-checks-v6.py', SCIENCE_SHA)
    status = evidence.read('scientific-checks-v6.status.json')
    passed(evidence, status, 'Scientific runner')
    require(status['schema_version'] == 1 and status['host'] == 'len' and
            status['cwd'] == str(evidence.remote) and status['cpu_affinity'] == [config['core']] and
            status['current'] is None and status['child_pid'] is None and
            status['environment'] == dict(config['environment'],
                PYTHONPATH=evidence.sources['optimized']['source'], PYTHONDONTWRITEBYTECODE='1'),
            'Wrong scientific runner environment or completion')
    require(Path(status['command'][0]).is_absolute(), 'Missing scientific interpreter')
    plan = runner.plan(evidence.remote, status['command'][0])
    require(status['plan'] == plan and status['completed'] == [row['name'] for row in plan] and
            len(status['commands']) == 5 and
            status['expected_coverage'] == dict(waveform_template_psd_pairs=288,
                                               boundary_cases=36, compressed_bank_cases=576),
            'Scientific runner omitted a planned check')
    required(evidence, receipt['input_sha256'], ('scientific-checks-v6.status.json',))
    required(evidence, status['input_sha256'], ('run-scientific-checks-v6.py', 'source-v6.json'))
    require(status['output_sha256'] == status['all_created_output_sha256'],
            'Scientific runner has unbound or missing outputs')
    evidence.recorded_inputs(status['output_sha256'])
    for step, command in zip(plan, status['commands']):
        require(command == dict(name=step['name'], command=step['command'], returncode=0,
                                log=step['log']), 'Scientific command differs from the frozen plan')
        output = evidence.root / evidence.local_name(step['output'])
        files = ([str(path.relative_to(evidence.root)) for path in sorted(output.rglob('*'))
                  if path.is_file()] if output.is_dir() else [str(output.relative_to(evidence.root))])
        required(evidence, status['output_sha256'], [*files, evidence.local_name(step['log'])])
    expected_names = dict(waveform_reference='waveform-validation-v6.json',
                          boundary_injections='boundary-injections-v6.json',
                          compressed_bank_backend_parity='compressed-bank-v6.json')
    require(all(evidence.local_name(receipt['evidence'][key]['path']) == name
                for key, name in expected_names.items()), 'Scientific receipts use unexpected sources')
    # Reuse the exact frozen numerical validator with explicit path aliases and
    # a separate module instance, so the historical helper is never edited.
    validator = load_helper(evidence.root, 'build-reference-report-v4.py', HELPER_SHA)
    validator.COMMIT = evidence.commit
    qualifier_cases = ['qual-science6-cpu-l256', 'qual-selected6-cpu-l512', 'qual-precision5-cpu-l1024']
    evidence.science_aliases = True
    try:
        result = validator.validate_science(evidence, qualifier_cases, bank_hashes)
    finally:
        evidence.science_aliases = False
    compressed = evidence.read('compressed-bank-v6.json')
    passed(evidence, compressed, 'Compressed waveform backend parity')
    required(evidence, compressed['input_sha256'],
             ('qualify-compressed-bank-v6.py', 'source-v6.json', 'config.json', 'inputs/bank-metadata.json',
              'inputs/bank-compressed-1e5.hdf', 'compression-1e5.json'))
    require(compressed['expected_cases'] == compressed['completed_cases'] == len(compressed['cases']) == 576 and
            compressed['native_calls'] == compressed['cuda_comparisons'] == 288,
            'Incomplete compressed waveform backend matrix')
    grid = set(itertools.product(('cpu', 'cuda'), (256, 512, 1024), range(96)))
    require({(row['device'], row['length_seconds'], row['index']) for row in compressed['cases']} == grid,
            'Missing or duplicate compressed waveform case')
    hashes = [str(row['template_hash']) for row in evidence.read('inputs/bank-metadata.json')['templates']]
    require(len(hashes) == 96 and {int(value) for value in hashes} == bank_hashes, 'Wrong bank metadata')
    for row in compressed['cases']:
        require(row['bitwise_equal'] is True and str(row['template_hash']) == hashes[row['index']] and
                re.fullmatch('[0-9a-f]{64}', row['output_sha256']) and len(row['metadata']) == 5 and
                row['metadata'][:2] == [row['length_seconds'] * 4096 // 2 + 1, 1 / row['length_seconds']],
                'Compressed waveform identity, geometry or bitwise gate failed')
        if row['device'] == 'cpu':
            require(row['native_used'] is True and row['reference'] == 'normal CPU', 'CPU native path not exercised')
        else:
            require(row['reference'] == 'frozen v5 Torch CUDA', 'CUDA reference changed')
    result['compressed_backend_cases'] = 576
    result['compressed_backend_gate'] = 'bitwise CPU/normal CPU and CUDA/frozen v5 CUDA'
    evidence.gates['scientific_integration'] = True
    return result


def strict_parity(comparator, data):
    require(comparator.DEFAULTS == V4.PARITY_TOLERANCES, 'Trigger parity tolerances changed')
    baseline = 'qual-selected6-cpu-l512'
    require(baseline in data and len(data) == 19, 'Strict parity requires every v6 run')
    comparisons = []
    for case in sorted(data):
        if case != baseline:
            result = comparator.compare(data[baseline], data[case], comparator.DEFAULTS)
            comparisons.append(dict(result, baseline=baseline, candidate=case))
    require(len(comparisons) == 18 and all(row['status'] == 'pass' for row in comparisons),
            'Strict trigger parity failed or needs review', comparisons)
    return dict(status='pass', tolerances=comparator.DEFAULTS, comparisons=comparisons)


def before_after(evidence, matched):
    evidence.bind('final-report/report.json', OLD_REPORT_SHA)
    previous = evidence.read('final-report/report.json')
    require(previous['status'] == 'pass' and previous['source_phases'] == {'precision': CAMP.PARENT} and
            all(previous['gates'].values()), 'Historical report is not the accepted v5 source')
    old = {row['scheme']: row for row in previous['matched']}
    require(set(old) == set(V4.SCHEMES), 'Historical matched backends are incomplete')
    comparisons = []
    for row in matched:
        prior = old[row['scheme']]
        require(prior['source_commit'] == CAMP.PARENT and prior['n'] == 3 and
                (prior['segment_length'], prior['start_pad'], prior['end_pad']) == (512, 112, 16),
                'Historical timings have different geometry')
        comparisons.append(dict(scheme=row['scheme'], before_source_commit=CAMP.PARENT,
            after_source_commit=evidence.commit, before=prior, after=row,
            median_capacity_speedup=row['templates_per_core_at_real_time']['median'] /
                prior['templates_per_core_at_real_time']['median'],
            median_wall_speedup=prior['wall_seconds']['median'] / row['wall_seconds']['median']))
    return comparisons


def build(evidence, source_manifest_sha256):
    for name, digest in {**CAMP.PINNED, 'build-reference-report-v4.py': HELPER_SHA,
                         'campaign-v6.py': CAMPAIGN_SHA, 'compare-triggers.py': COMPARATOR_SHA,
                         'run-scientific-checks-v6.py': SCIENCE_SHA}.items():
        evidence.bind(name, digest)
    evidence.bind(Path(__file__).name)
    config = evidence.read('config.json')
    source = read_source(evidence, source_manifest_sha256, config)
    units = unit_tests(evidence, config, source)
    fft = large_ifft(evidence, config, source)
    decision = evidence.read('precision5-reference-tuning-decision.json')
    require(decision['source_commit'] == CAMP.PARENT and
            tuple(decision['selected_' + name + '_seconds'] for name in
                  ('segment_length', 'start_pad', 'end_pad')) == (512, 112, 16), 'Reference tuning changed')
    evidence.recorded_inputs(decision['input_sha256'])
    hardware = evidence.read('environment.json')
    require(hardware['hostname'] == 'len' and hardware['affinity'] == [config['core']], 'Wrong host/core')
    with V4.h5py.File(evidence.bind('inputs/bank-compressed-1e5.hdf', V4.BANK_SHA), 'r') as handle:
        bank_hashes = {int(value) for value in handle['template_hash'][:]}
    require(len(bank_hashes) == 96, 'Wrong bank population')
    expected = read_stages(evidence, config, source)
    comparator = load_helper(evidence.root, 'compare-triggers.py', COMPARATOR_SHA)
    receipts, data, work = {}, {}, {}
    for case, row in expected.items():
        receipts[case], data[case] = V4.read_run(evidence, row, config, comparator)
        expected_cli = [str(Path(source['source']) / 'bin/pycbc_inspiral'), *config['common_args'],
            '--bank-file', str(evidence.remote / 'inputs/bank-compressed-1e5.hdf'),
            '--processing-scheme', row['scheme'], '--segment-length', '512',
            '--segment-start-pad', '112', '--segment-end-pad', '16',
            '--output', str(evidence.remote / 'runs' / case / 'triggers.hdf')]
        require(receipts[case]['executable_cli'] == expected_cli and receipts[case].get('finished_utc'),
                'Run command differs from frozen workload or did not finish: ' + case)
        if row['mode'] == 'qualify':
            work[row['scheme']] = V4.qualification(evidence, receipts[case], bank_hashes, config)
    rows = []
    for case, receipt in receipts.items():
        qualification = work[receipt['scheme']]
        baseline = receipts[Path(qualification['qualification']).parent.name]
        profile = receipt['mode'] in ('cprofile', 'perf', 'torchprofile')
        require(V4.analysis_signature(receipt, comparator, profile=profile) ==
                V4.analysis_signature(baseline, comparator, profile=profile),
                'Run differs from qualified work: ' + case)
        rows.append(V4.timing_row(receipt, qualification, data[case]))
    matched_rows = [row for row in rows if row['mode'] == 'timing']
    require(len(matched_rows) == 9 and Counter(row['scheme'] for row in matched_rows) ==
            Counter({scheme: 3 for scheme in V4.SCHEMES}), 'Wrong matched repetition counts')
    require(len({V4.analysis_signature(receipts[row['case']], comparator,
                omit=('--processing-scheme', '--fft-backends', '--verbose'))
                for row in matched_rows}) == 1, 'Matched backends have different workload inputs')
    matched = V4.grouped(matched_rows, 'optimized-v6')
    parity = strict_parity(comparator, data)
    evidence.gates['trigger_parity'] = evidence.gates['qualified_work_and_timings'] = True
    scientific = science(evidence, bank_hashes, config)
    profile_cases = {case for case, receipt in receipts.items()
                     if receipt['mode'] in ('cprofile', 'perf', 'torchprofile')}
    profiles = V4.profiles(evidence, profile_cases, receipts)
    evidence.gates['profiles'] = True
    return dict(schema_version=1, status='pass', gates=evidence.gates, figures=list(FIGURES),
        input_sha256=evidence.input_sha256, archived_remote_input_sha256=evidence.archived_remote_inputs,
        required_cases=sorted(expected), source_commit=evidence.commit, parent_source_commit=CAMP.PARENT,
        source_manifest_sha256=source_manifest_sha256, changed_paths=source['changed_paths'],
        normal_cpu_path_audit=source['normal_cpu_path_audit'], unit_tests=units, large_ifft=fft,
        hardware=hardware, bank_sha256=V4.BANK_SHA, selected_geometry=dict(segment_length_seconds=512,
            start_pad_seconds=112, end_pad_seconds=16, tuning_source_commit=CAMP.PARENT,
            basis='Reuse only the verified unchanged normal CPU tuning choice; every v6 timing is fresh.'),
        accounting=dict(formula='96 templates * 1904 unique valid detector seconds / full-process wall seconds / 1 allocated host core',
            units='templates per core at real time', repetitions=3,
            summary='Median and full observed min/max range; no confidence interval.',
            profile_timing='Seven separate profiled runs excluded from the nine matched capacity measurements.',
            internal_timing='HDF run_time excludes early imports/startup and final output. Setup and postsetup use setup_time_fraction.'),
        rows=rows, matched=matched, before_after=before_after(evidence, matched), parity=parity,
        scientific_validation=scientific, profiles=profiles,
        limitations=['Finite 1904-second interval and 96 deterministic aligned-spin templates; no steady-state or bank-coverage claim.',
            'CUDA capacity additionally consumes one GPU; all backends consume one allocated host core.',
            'Before measurements use source 837f38d; after measurements use the declared v6 source. Samples are never pooled.',
            'Only the v5 normal CPU tuning choice is reused. New backend waveforms, triggers, timings and profiles are validated separately.',
            'Profile percentages have separate denominators: exclusive cProfile self time, perf cycles:u samples and profiler-visible CUDA events.',
            'Remote source/native/frame bytes may be absent from this portable archive; their hashes remain bound by reviewed source, unit and campaign receipts.'])


def figures(report, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 160, 'savefig.facecolor': 'white'})
    pairs = sorted(report['before_after'], key=lambda row: V4.SCHEMES.index(row['scheme']))
    positions = np.arange(3)
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    for side, offset, color, label in (('before', -.19, '#a3aebc', 'Before · source 837f38d'),
                                     ('after', .19, '#187b91', 'After · source ' + report['source_commit'][:7])):
        stats = [row[side]['templates_per_core_at_real_time'] for row in pairs]
        centers = np.array([row['median'] for row in stats])
        errors = np.array([[row['median'] - row['min'] for row in stats],
                           [row['max'] - row['median'] for row in stats]])
        bars = ax.bar(positions + offset, centers, .35, color=color, label=label,
                      yerr=errors, capsize=4)
        ax.bar_label(bars, labels=[f'{value:.0f}' for value in centers], padding=5)
    ax.set(xticks=positions, xticklabels=[V4.LABELS[row['scheme']] for row in pairs],
           ylabel='Templates per allocated host core at real time',
           title='Fresh matched timings at 512 s FFT length\nMedian and full observed range · three runs per backend and source')
    ax.legend(loc='upper left')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    fig.savefig(out / FIGURES[0]); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), layout='constrained', sharey=True)
    for ax, pair in zip(axes, pairs):
        for x, side in enumerate(('before', 'after')):
            group = pair[side]
            # Each column uses one actual run, selected by median wall time;
            # independently rounded medians would not necessarily add to wall.
            if side == 'after':
                rows = [row for row in report['rows'] if row['case'] in group['cases']]
                row = sorted(rows, key=lambda row: row['wall_seconds'])[1]
                values = [row['internal_setup_seconds'], row['internal_postsetup_seconds'],
                          row['wall_seconds'] - row['internal_runtime_seconds']]
                bottom = 0
                for value, color, label in zip(values, ('#efb366', '#187b91', '#a3aebc'),
                                               ('Internal setup', 'Internal postsetup', 'Outside internal timer')):
                    ax.bar(x, value, .6, bottom=bottom, color=color, label=label)
                    bottom += value
            else:
                ax.bar(x, group['wall_seconds']['median'], .6, color='#d5dbe1', label='Before full wall')
        ax.set(title=V4.LABELS[pair['scheme']], xticks=[0, 1], xticklabels=['Before', 'After'])
    axes[0].set_ylabel('Full-process wall seconds')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4)
    fig.suptitle('Before median wall; after breakdown of the run with median wall\nInternal postsetup includes analysis housekeeping')
    fig.savefig(out / FIGURES[1]); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.3), layout='constrained', sharex=True)
    for ax, scheme in zip(axes, V4.SCHEMES):
        rows = sorted((row for row in report['profiles']['groups']
                       if row['scheme'] == scheme and row['mode'] == 'cprofile'),
                      key=lambda row: row['share_percent'], reverse=True)[:7]
        ax.barh([row['group'] for row in rows][::-1], [row['share_percent'] for row in rows][::-1], color='#187b91')
        ax.set(title=V4.LABELS[scheme], xlabel='Share of exclusive self time (%)', xlim=(0, 100))
    fig.suptitle('Separate cProfile executions · seven largest exclusive categories per backend\nThese shares do not describe CUDA device occupancy or add to perf samples')
    fig.savefig(out / FIGURES[2]); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE)
    parser.add_argument('--source-manifest-sha256', required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    out = args.output_dir or args.root / 'optimized-report-v6'
    require(not out.exists(), 'Use a new output directory; previous reports are immutable')
    evidence = Evidence(args.root)
    try:
        report = build(evidence, args.source_manifest_sha256)
        for name, digest in list(evidence.input_sha256.items()):
            evidence.bind(name, digest)
    except Exception as error:
        out.mkdir(parents=True, exist_ok=False)
        (out / 'report.json').write_text(json.dumps(dict(schema_version=1, status='fail',
            error=str(error), error_type=type(error).__name__, details=getattr(error, 'details', None),
            gates=evidence.gates, input_sha256=evidence.input_sha256), indent=2, allow_nan=False) + '\n')
        print('Report withheld: ' + str(error), file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=False)
    try:
        for name, rows in (('runs.csv', report['rows']), ('groups.csv', report['matched']),
                           ('before-after.csv', report['before_after']),
                           ('profiles.csv', report['profiles']['groups']),
                           ('native-symbols.csv', report['profiles']['native_symbols'])):
            V4.write_csv(out / name, rows)
        figures(report, out)
        for name, digest in list(evidence.input_sha256.items()):
            evidence.bind(name, digest)
        report['output_sha256'] = {path.name: sha(path) for path in sorted(out.iterdir())}
    except Exception as error:
        report = dict(schema_version=1, status='fail', error=str(error), error_type=type(error).__name__)
    with (out / 'report.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(out.resolve())
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    sys.exit(main())
