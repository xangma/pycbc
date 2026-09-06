#!/usr/bin/env python3
"""Build the final single-host-core inspiral report from frozen local evidence.

Usage: python build-reference-report.py --output-dir final-report
Requires the six final plans/status ledgers, reference-tuning-decision.json,
profiles.json, all named runs, and the two final scientific validation receipts.
Reads evidence only; writes a new output directory. Failure writes diagnostics
with status=fail and exits nonzero, without publishing performance figures.
"""

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys

import h5py
import numpy as np


COMMIT = 'fb4b335eeeeaeaa907c1143b45e0191e2d977761'
MATCHED_COMMIT = '968bcd558117262af0d603710b054174659adb51'
BANK_SHA = '161820754117cd51b24a7bb672ea456cf7b2a3808e69690c23235d024b1a2916'
SCHEMES = ('cpu:1', 'torch:cpu:1', 'torch:cuda:0')
LABELS = dict(zip(SCHEMES, ('Normal CPU (MKL)', 'Torch CPU', 'Torch CUDA')))
PLANS = ('accuracy-qualifications-plan', 'accuracy-timings-plan',
         'final-qualifications-v3-plan', 'reference-profiles-v2-plan',
         'matched-backends-v2-plan', 'torch-profiles-v2-plan')
PLAN_CASES = {
    'accuracy-qualifications-plan': {
        f'qual-1e5-cpu-l{length}': ('qualify', 'cpu:1', length)
        for length in (256, 512, 1024)},
    'accuracy-timings-plan': {
        f'tune-1e5-cpu-l{length}-r{rep}': ('timing', 'cpu:1', length)
        for length in (256, 512, 1024) for rep in (1, 2, 3)},
    'final-qualifications-v3-plan': {
        f'qual-final-{label}-l512': ('qualify', scheme, 512)
        for label, scheme in zip(('cpu', 'torch-cpu', 'torch-cuda'), SCHEMES)},
    'reference-profiles-v2-plan': {
        f'reference-v2-cpu-l512-{mode}': (mode, 'cpu:1', 512)
        for mode in ('cprofile', 'perf')},
    'matched-backends-v2-plan': {
        f'matched-v2-{label}-l512-r{rep}': ('timing', scheme, 512)
        for label, scheme in zip(('cpu', 'torch-cpu', 'torch-cuda'), SCHEMES)
        for rep in (1, 2, 3)},
    'torch-profiles-v2-plan': {
        **{f'profile-v2-torch-{device}-l512-{mode}': (mode, scheme, 512)
           for device, scheme in zip(('cpu', 'cuda'), SCHEMES[1:])
           for mode in ('cprofile', 'perf')},
        'profile-v2-torch-cuda-l512-torchprofile': ('torchprofile', 'torch:cuda:0', 512)},
}
METRICS = ('wall_seconds', 'templates_per_core_at_real_time',
           'internal_runtime_seconds', 'internal_setup_seconds',
           'internal_postsetup_seconds', 'peak_rss_mib')
FIGURES = ('tuning-capacity.png', 'matched-capacity.png',
           'wall-and-internal-times.png', 'profile-self-time.png',
           'native-symbols.png', 'cuda-events.png')


class InvalidEvidence(ValueError):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details


def require(condition, message, details=None):
    if not condition:
        raise InvalidEvidence(message, details)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def positive(value, name, zero=False):
    require(isinstance(value, (int, float)) and math.isfinite(value) and
            (value >= 0 if zero else value > 0), f'Invalid {name}: {value}')
    return value


def helper(root, name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), root / name)
    module = importlib.util.module_from_spec(spec)
    # Do not create __pycache__ inside the frozen evidence directory.
    exec(compile((root / name).read_bytes(), str(root / name), 'exec'), module.__dict__)
    return module


class Evidence:
    def __init__(self, root):
        self.root, self.input_sha256, self.gates = root.resolve(), {}, {}
        self.remote = None
        self.sources = {}

    def bind(self, relative, expected=None):
        path = (self.root / relative).resolve()
        name = str(path.relative_to(self.root))
        digest = sha(path)
        require(expected is None or digest == expected, f'Hash mismatch: {name}')
        previous = self.input_sha256.get(name)
        require(previous is None or previous == digest, f'Input changed while reading: {name}')
        self.input_sha256[name] = digest
        return path

    def read(self, relative):
        path = self.bind(relative)
        value = json.loads(path.read_text())
        canonical(value)
        require(sha(path) == self.input_sha256[str(path.relative_to(self.root))],
                f'JSON changed while reading: {relative}')
        return value

    def local_name(self, recorded):
        path = Path(recorded)
        if not path.is_absolute():
            require('..' not in path.parts, f'Invalid relative path: {recorded}')
            return str(path)
        for prefix in (self.root, self.remote):
            if prefix is not None and path.is_relative_to(prefix):
                return str(path.relative_to(prefix))
        return None

    def recorded_inputs(self, before, after=None):
        require(isinstance(before, dict) and before, 'Missing input hashes')
        require(after is None or before == after, 'Recorded inputs changed during execution')
        for name, digest in before.items():
            require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest),
                    f'Malformed SHA256: {name}')
            relative = self.local_name(name)
            # Source and external dependencies are bound by their archived
            # receipts and source bundle; do not pretend their binaries are here.
            if relative is not None and not relative.startswith(('source/', 'source-v2/')):
                self.bind(relative, digest)


def read_plans(evidence, comparator):
    plans, all_cases = {}, {}
    for name in PLANS:
        plan = evidence.read(name + '.json')
        require(isinstance(plan, list) and plan, f'Empty/malformed plan: {name}')
        status = evidence.read(name + '.status.json')
        require(status['state'] == 'complete' and status['returncode'] == 0 and
                status['completed'] == plan and status.get('current') is None and
                status['plan_sha256'] == evidence.input_sha256[name + '.json'],
                f'Incomplete or unbound plan ledger: {name}')
        rows = []
        for argv in plan:
            opts = comparator.options(['run-case', *argv])
            require(set(opts) <= {'--case', '--mode', '--scheme', '--segment-length',
                                 '--start-pad', '--end-pad'} and
                    all(len(value) == 1 for value in opts.values()),
                    f'Unexpected plan options: {name}')
            case = opts['--case'][0]
            require(Path(case).name == case and case not in all_cases,
                    f'Duplicate or malformed case: {case}')
            row = dict(case=case, mode=opts.get('--mode', ['timing'])[0],
                       scheme=opts.get('--scheme', ['cpu:1'])[0],
                       segment_length=int(opts['--segment-length'][0]),
                       start_pad=int(opts.get('--start-pad', ['112'])[0]),
                       end_pad=int(opts.get('--end-pad', ['16'])[0]))
            row['source_phase'] = 'reference' if name.startswith('accuracy-') else 'matched'
            all_cases[case] = row
            rows.append(row)
        require({row['case']: (row['mode'], row['scheme'], row['segment_length'])
                 for row in rows} == PLAN_CASES[name] and
                all((row['start_pad'], row['end_pad']) == (112, 16) for row in rows),
                f'Unexpected final case names, modes, schemes or geometry: {name}')
        plans[name] = rows
    require(len(all_cases) == 31, 'Final campaign must contain exactly 31 distinct runs')
    umbrella = evidence.read('accuracy-campaign.status.json')
    require(umbrella['state'] == 'complete' and umbrella['returncode'] == 0 and
            umbrella.get('current') is None and len(umbrella['completed']) == 5,
            'Final accuracy campaign is incomplete')
    evidence.recorded_inputs(umbrella['input_sha256'], umbrella['input_sha256_after'])
    evidence.gates['campaign_complete'] = True
    return plans, all_cases


def analysis_signature(receipt, comparator, omit=(), profile=False):
    opts = comparator.options(receipt['executable_cli'])
    for name in ('--output', *omit):
        opts.pop(name, None)
    hashes = dict(receipt['input_sha256'])
    cwd = Path(receipt['cwd'])
    hashes.pop(str(cwd / 'qualify-inspiral.py'), None)
    if profile:
        for name in ('run-case.py', 'run-case-profiling.py', 'profile-inspiral-torch.py'):
            hashes.pop(str(cwd / name), None)
    return canonical(dict(source=receipt['source_info'], host=receipt['hostname'],
                          environment=receipt['environment'], executable=receipt['executable_cli'][0],
                          options=opts, hashes=hashes))


def read_run(evidence, expected, config, comparator):
    case = expected['case']
    receipt = evidence.read(f'runs/{case}/receipt.json')
    require(all(receipt[key] == value for key, value in expected.items() if key != 'source_phase'),
            f'{case}: receipt disagrees with the final plan')
    phase = expected['source_phase']
    source = evidence.sources[phase]
    require(receipt['state'] == 'complete' and receipt['returncode'] == 0 and
            receipt['source_info']['commit'] == source['commit'] and
            not receipt['source_info']['status'] and not receipt['source_info']['tracked_diff'] and
            not receipt['source_status_after'], f'{case}: incomplete or changed source')
    evidence.recorded_inputs(receipt['input_sha256'], receipt['input_sha256_after'])
    require(receipt['hostname'] == 'len' and receipt['cwd'] == str(evidence.remote) and
            receipt['source'] == source['source'], f'{case}: wrong source location')
    env = dict(config['environment'], PYTHONPATH=receipt['source'], PYTHONDONTWRITEBYTECODE='1')
    require(receipt['environment'] == env, f'{case}: unexpected numerical environment')
    command = receipt['command']
    require(command.count('taskset') == 1, f'{case}: missing unique CPU affinity command')
    index = command.index('taskset')
    require(command[index + 1:index + 3] == ['-c', str(config['core'])],
            f'{case}: run is not pinned to the declared single host core')
    opts = comparator.options(receipt['executable_cli'])
    for name, field in (('--processing-scheme', 'scheme'), ('--segment-length', 'segment_length'),
                        ('--segment-start-pad', 'start_pad'), ('--segment-end-pad', 'end_pad')):
        require(opts[name] == [str(receipt[field])], f'{case}: inconsistent executable {name}')
    bank = str(evidence.remote / 'inputs/bank-compressed-1e5.hdf')
    require(opts['--bank-file'] == [bank] and receipt['input_sha256'][bank] == BANK_SHA,
            f'{case}: wrong bank')
    trigger = evidence.bind(f'runs/{case}/triggers.hdf', receipt['trigger_sha256'])
    data = comparator.load(trigger)
    require(not data['metadata']['issues'] and set(data['detectors']) == {'H1'},
            f'{case}: malformed trigger evidence', data['metadata']['issues'])
    require(data['detectors']['H1']['intervals'] == [[1187007160, 1187009064]],
            f'{case}: wrong valid detector interval')
    receipt['report_source_phase'] = phase
    return receipt, data


def qualification(evidence, receipt, bank_hashes, config):
    case = receipt['case']
    q = evidence.read(f'runs/{case}/qualification.json')
    require(q['status'] == 'success' and q['executable_exit_code'] == 0 and q['checks'] and
            all(v is True for v in q['checks'].values()), f'{case}: qualification failed')
    require(q['argv'] == q['executed_argv'] == receipt['executable_cli'] and
            q['host'] == receipt['hostname'] and q['source_root'] == receipt['source'],
            f'{case}: qualification is not bound to this run')
    for record in (q['executable'], q['wrapper']):
        require(receipt['input_sha256'][record['path']] == record['sha256'],
                f'{case}: unbound executable or qualification wrapper')
    observed = q['observations']
    require(len(observed['banks']) == len(observed['segment_geometry']) == 1,
            f'{case}: ambiguous qualified work')
    bank, geometry = observed['banks'][0], observed['segment_geometry'][0]
    templates, segments = bank['templates'], len(geometry['segments'])
    require(len(templates) == 96 and bank['file']['sha256'] == BANK_SHA and
            {int(t['template_hash']) for t in templates.values()} == bank_hashes,
            f'{case}: template identities/count disagree with the bank')
    require(segments > 0 and geometry['gap_samples'] == geometry['overlap_samples'] == 0 and
            geometry['unique_analyzed_seconds'] == 1904 and
            geometry['fft_samples'] == receipt['segment_length'] * 4096 and
            geometry['sample_rate_hz'] == 4096 and
            sum(e['execute_successes'] for e in observed['fft_engines']) == 96 * segments and
            observed['decompression_success_count'] == 96,
            f'{case}: incomplete work or wrong FFT geometry')
    origin = float(geometry['strain_start_time'])
    require([[origin + lo / 4096, origin + hi / 4096]
             for lo, hi in geometry['union_analyzed_sample_intervals']] ==
            [[1187007160, 1187009064]], f'{case}: qualification analyzed different samples')
    runtime = observed['runtime']['inside_filter_context']
    expected_schemes = {
        'cpu:1': dict(class_name='pycbc.scheme.CPUScheme', device='None', num_threads=1),
        'torch:cpu:1': dict(class_name='pycbc.scheme.TorchScheme', device='cpu', num_threads=1),
        'torch:cuda:0': dict(class_name='pycbc.scheme.TorchScheme', device='cuda:0', num_threads=None),
    }
    expected_runtime = expected_schemes[receipt['scheme']]
    require(runtime['affinity'] == [config['core']] and
            runtime['scheme'] == {'class': expected_runtime['class_name'],
                                  'device': expected_runtime['device'],
                                  'num_threads': expected_runtime['num_threads']},
            f'{case}: wrong qualified scheme, device, thread setting or host affinity')
    if receipt['scheme'].startswith('torch:'):
        require(runtime['torch']['num_threads'] == 1,
                f'{case}: Torch numerical operations did not use one host thread')
    for psd in observed['psd_arrays']:
        relative = evidence.local_name(psd['path'])
        require(relative and relative.startswith(f'runs/{case}/arrays/'),
                f'{case}: PSD path escapes its run')
        evidence.bind(relative, psd['sha256'])
    return dict(templates=96, segments_per_template=segments, fft_samples=geometry['fft_samples'],
                valid_data_seconds=1904, total_template_seconds=96 * 1904,
                qualification=f'runs/{case}/qualification.json')


def timing_row(receipt, work, data):
    with h5py.File(data['path'], 'r') as handle:
        search = handle['H1/search']
        runtime = positive(float(search['run_time'][0]), 'internal runtime')
        fraction = float(search['setup_time_fraction'][0])
    wall = positive(receipt['elapsed_wall_seconds'], 'full wall time')
    require(math.isfinite(fraction) and 0 <= fraction <= 1 and runtime <= wall,
            f"{receipt['case']}: inconsistent internal/wall timing")
    row = {key: receipt[key] for key in ('case', 'mode', 'scheme', 'segment_length', 'start_pad', 'end_pad')}
    row.update(work, source_phase=receipt['report_source_phase'],
               source_commit=receipt['source_info']['commit'],
               allocated_cpu_cores=1, wall_seconds=wall,
               templates_per_core_at_real_time=work['total_template_seconds'] / wall,
               internal_runtime_seconds=runtime, internal_setup_seconds=runtime * fraction,
               internal_postsetup_seconds=runtime * (1 - fraction),
               peak_rss_mib=positive(receipt['peak_child_rss_kib'], 'peak RSS') / 1024,
               user_cpu_seconds=positive(receipt['child_user_cpu_seconds'], 'user CPU time', zero=True),
               system_cpu_seconds=positive(receipt['child_system_cpu_seconds'], 'system CPU time', zero=True),
               triggers=data['detectors']['H1']['count'], receipt=f"runs/{receipt['case']}/receipt.json")
    return row


def grouped(rows, cohort):
    groups = []
    for scheme, length in sorted({(r['scheme'], r['segment_length']) for r in rows}):
        values = [r for r in rows if (r['scheme'], r['segment_length']) == (scheme, length)]
        require(len(values) == 3 and all(r['mode'] == 'timing' for r in values),
                f'{cohort}: require exactly three unprofiled repetitions per group')
        keys = ('templates', 'segments_per_template', 'fft_samples', 'valid_data_seconds',
                'total_template_seconds', 'start_pad', 'end_pad', 'allocated_cpu_cores',
                'source_phase', 'source_commit')
        require(len({canonical({k: r[k] for k in keys}) for r in values}) == 1,
                f'{cohort}: group contains different work or padding')
        group = dict(cohort=cohort, scheme=scheme, segment_length=length, n=3,
                     cases=[r['case'] for r in values], **{k: values[0][k] for k in keys})
        for metric in METRICS:
            samples = [r[metric] for r in values]
            group[metric] = dict(median=statistics.median(samples), min=min(samples), max=max(samples))
        groups.append(group)
    return groups


def validate_science(evidence, qualification_cases):
    waveform = evidence.read('waveform-validation-1e5.json')
    boundary = evidence.read('boundary-injections-1e5.json')
    for name, value, field, count in (
            ('waveform', waveform, 'template_psd_pairs', 288), ('boundary', boundary, 'cases', 36)):
        require(value['state'] == 'complete' and value['passed'] is True and
                value['completed_' + field] == value['expected_' + field] == count,
                f'{name}: incomplete or failed scientific validation')
        require(value['source_before'] == value['source_after'] and
                value['source_before']['commit'] == COMMIT and not value['source_before']['status'],
                f'{name}: source changed during validation')
        evidence.recorded_inputs(value['input_sha256'], value['input_sha256_after'])
        evidence.gates[name + '_validation'] = True
    cases = waveform['cases']
    require({Path(c['qualification']).parent.name for c in cases} == set(qualification_cases),
            'Waveform validation does not cover every final CPU qualification')
    pairs = []
    for case in cases:
        require(case['passed'] is True and case['state'] == 'complete' and len(case['templates']) == 96,
                'Incomplete waveform-validation case')
        for template in case['templates']:
            require(template['passed'] is True and template['decompression_succeeded_without_generation'] is True,
                    'Waveform template validation failed')
            pairs.extend(template['psd_results'])
    require(len(pairs) == 288 and all(p['passed'] is True for p in pairs),
            'Waveform per-pair count/pass mismatch')
    boundary_rows = [c for template in boundary['templates'] for c in template['cases']]
    require(len(boundary_rows) == 36 and all(t['passed'] is True and
            t['reference_edge_budget_passed'] is True for t in boundary['templates']) and
            all(c['passed'] is True for c in boundary_rows), 'Boundary per-case count/pass mismatch')
    return dict(waveform_template_psd_pairs=288, boundary_cases=36,
                waveform_tolerances=waveform['tolerances'], boundary_tolerances=boundary['tolerances'],
                maximum_waveform_errors={key: max(p['metrics'][key] for p in pairs)
                                         for key in waveform['tolerances']})


def profiles(evidence, selected, receipts):
    summary = evidence.read('profiles.json')
    require(summary['schema_version'] == 1, 'Wrong profile summary schema')
    evidence.bind('summarize-profiles.py', summary['summarizer']['sha256'])
    expected = {case for case in selected if receipts[case]['mode'] in ('cprofile', 'perf')}
    measured = {item['run']['case']: item for item in summary['runs']}
    require(set(measured) == expected and len(measured) == len(summary['runs']) == 6,
            'Profile summary must contain exactly the six final cProfile/perf runs')
    group_rows, native_rows, compact = [], [], []
    for case, item in measured.items():
        receipt = receipts[case]
        evidence.bind(f'runs/{case}/receipt.json', item['receipt']['sha256'])
        record = dict(case=case, scheme=receipt['scheme'], mode=receipt['mode'])
        if receipt['mode'] == 'cprofile':
            data = item['cprofile']
            evidence.bind(f'runs/{case}/profile.pstats', data['input']['sha256'])
            total = positive(data['total_profile_self_seconds'], 'cProfile denominator')
            require(data['exclusive_groups_reconcile'] is True and
                    math.isclose(math.fsum(g['self_seconds'] for g in data['groups']), total,
                                 rel_tol=1e-9, abs_tol=1e-9), f'{case}: cProfile self times do not reconcile')
            for group in data['groups']:
                numerator = positive(group['self_seconds'], 'cProfile self time', zero=True)
                require(math.isclose(100 * numerator / total, group['share_of_profile_self_percent'],
                                    rel_tol=1e-9, abs_tol=1e-9), f'{case}: wrong profile percentage')
                group_rows.append(dict(**record, group=group['group'], numerator=numerator,
                                       denominator=total, unit='seconds', share_percent=100 * numerator / total,
                                       denominator_kind='sum of cProfile exclusive self time'))
            record.update(total_profile_self_seconds=total, accounting=data['accounting'])
        else:
            data = item['perf']
            evidence.bind(f'runs/{case}/perf-report.txt', data['input']['sha256'])
            evidence.bind(f'runs/{case}/perf.data')
            require(data['event'] == 'cycles:u' and data['native_functions'],
                    f'{case}: missing/wrong native sampling event')
            for row in data['native_functions']:
                require(0 <= row['self_event_percent'] <= 100, f'{case}: invalid perf share')
                native_rows.append(dict(**record, event=data['event'],
                                        symbol=row['symbol'], shared_object=row['shared_object'],
                                        self_event_percent=row['self_event_percent'],
                                        samples=row.get('samples'), event_period=row.get('event_period'),
                                        extra_columns=row.get('extra_columns', {}),
                                        classification=row['classification']))
            record.update(event=data['event'], denominator_note=data['denominator_note'])
        compact.append(record)
    cuda_cases = [case for case in selected if receipts[case]['mode'] == 'torchprofile']
    require(len(cuda_cases) == 1, 'Expected exactly one CUDA operator/device profile')
    case = cuda_cases[0]
    prefix = f'runs/{case}/torch-profile'
    trace_receipt = evidence.read(prefix + '/receipt.json')
    receipt = receipts[case]
    require(trace_receipt['status'] == 'success' and trace_receipt['wrapper_exit_code'] == 0 and
            trace_receipt['executable_exit_code'] == 0 and
            trace_receipt['argv'] == trace_receipt['executed_argv'] == receipt['executable_cli'] and
            trace_receipt['host'] == receipt['hostname'] and
            trace_receipt['source_root'] == receipt['source'],
            'Incomplete or unbound Torch device trace')
    for key in ('wrapper', 'executable'):
        require(receipt['input_sha256'][trace_receipt[key]['path']] == trace_receipt[key]['sha256'],
                'Torch trace executable/wrapper hash mismatch')
    evidence.bind(prefix + '/trace.json', trace_receipt['trace']['sha256'])
    evidence.bind(prefix + '/key-averages.json', trace_receipt['key_averages']['sha256'])
    averages = evidence.read(prefix + '/key-averages.json')
    require(averages['units'] == 'microseconds', 'Unexpected Torch profile units')
    cuda_rows = []
    for kind, field, total_field in (
            ('CPU', 'self_cpu_time_us', 'cpu_event_self_time_us'),
            ('CUDA', 'self_device_time_us', 'cuda_device_event_self_time_us')):
        rows = [r for r in averages['rows'] if r['device_type'].endswith('.' + kind)]
        total = positive(averages[total_field], kind + ' trace denominator')
        require(math.isclose(math.fsum(positive(r[field], 'Torch event time', zero=True)
                                      for r in rows), total, rel_tol=1e-9, abs_tol=1e-6),
                kind + ' trace times do not reconcile')
        for row in sorted(rows, key=lambda r: r[field], reverse=True):
            item = dict(case=case, scheme=receipt['scheme'], mode='torchprofile', group=row['key'],
                        numerator=row[field], denominator=total, unit='microseconds',
                        share_percent=100 * row[field] / total,
                        denominator_kind=kind + ' profiler-visible event self-time sum')
            group_rows.append(item)
            if kind == 'CUDA':
                cuda_rows.append(item)
    return dict(runs=compact, groups=group_rows, native_symbols=native_rows,
                cuda_device_events=cuda_rows, torch_note=averages['note'])


def build(evidence):
    evidence.bind(Path(__file__).name)
    for name in ('compare-triggers.py', 'summarize-runs.py'):
        evidence.bind(name)
    comparator = helper(evidence.root, 'compare-triggers.py')
    source, config = evidence.read('source.json'), evidence.read('config.json')
    require(source['commit'] == COMMIT, 'Unexpected final source revision')
    evidence.remote = Path(source['source']).parent
    evidence.bind('inspiral-source.bundle', source['bundle_sha256'])
    revised = evidence.read('source-v2.json')
    require(revised['commit'] == MATCHED_COMMIT and
            revised['source'] == str(evidence.remote / 'source-v2'), 'Wrong matched source revision/location')
    evidence.bind('inspiral-source-v2.bundle', revised['bundle_sha256'])
    equivalence = evidence.read('reference-source-equivalence.json')
    required_checks = {'only_TorchScheme_deepcopy_added', 'all_other_tracked_blobs_identical',
                       'native_hashes_equal'}
    require(equivalence['status'] == 'pass' and equivalence['old_source_commit'] == COMMIT and
            equivalence['new_source_commit'] == revised['commit'] and
            equivalence['changed_paths'] == ['pycbc/scheme.py', 'test/test_scheme_runtime.py'] and
            required_checks <= set(equivalence['checks']) and
            all(v is True for v in equivalence['checks'].values()) and
            source['native_modules_sha256'] == revised['native_modules_sha256'],
            'Source-phase equivalence is missing or failed')
    evidence.recorded_inputs(equivalence['input_sha256'])
    require({'source.json', 'source-v2.json', 'inspiral-source.bundle', 'inspiral-source-v2.bundle'} <=
            {evidence.local_name(name) for name in equivalence['input_sha256']},
            'Source-equivalence proof does not bind both manifests/bundles')
    evidence.sources = dict(reference=source, matched=revised)
    evidence.gates['source_phase_equivalence'] = True
    hardware = evidence.read('environment.json')
    require(hardware['hostname'] == 'len' and hardware['affinity'] == [config['core']],
            'Hardware inventory does not describe the configured host/core')
    bank_path = evidence.bind('inputs/bank-compressed-1e5.hdf', BANK_SHA)
    with h5py.File(bank_path, 'r') as handle:
        bank_hashes = {int(value) for value in handle['template_hash'][:]}
    require(len(bank_hashes) == 96, 'Final bank must contain 96 unique template hashes')
    compression = evidence.read('compression-1e5.json')
    require(compression['state'] == 'complete' and compression['returncode'] == 0 and
            compression['output_sha256'][str(evidence.remote / 'inputs/bank-compressed-1e5.hdf')] == BANK_SHA,
            'Final bank compression is incomplete or unbound')
    evidence.recorded_inputs(compression['input_sha256'], compression['input_sha256_after'])
    plans, expected = read_plans(evidence, comparator)
    receipts, data = {}, {}
    for case, item in expected.items():
        receipts[case], data[case] = read_run(evidence, item, config, comparator)
    evidence.gates['source_and_inputs_bound'] = True
    qualified = {}
    for case, receipt in receipts.items():
        if receipt['mode'] == 'qualify':
            work = qualification(evidence, receipt, bank_hashes, config)
            key = analysis_signature(receipt, comparator, profile=True)
            require(key not in qualified, 'Duplicate final qualification configuration')
            qualified[key] = work
    require(len(qualified) == 6, 'Require three reference CPU and three matched-phase qualifications')
    rows = []
    for case, receipt in receipts.items():
        key = analysis_signature(receipt, comparator, profile=True)
        require(key in qualified, f'{case}: no matching final qualification')
        # Unprofiled timings must also retain the exact qualified timing runner.
        if receipt['mode'] == 'timing':
            qcase = Path(qualified[key]['qualification']).parent.name
            require(analysis_signature(receipt, comparator) ==
                    analysis_signature(receipts[qcase], comparator),
                    f'{case}: timing runner differs from qualification')
        rows.append(timing_row(receipt, qualified[key], data[case]))
    evidence.gates['qualifications'] = True
    scientific = validate_science(evidence, [r['case'] for r in plans['accuracy-qualifications-plan']])
    tuning_cases = {r['case'] for r in plans['accuracy-timings-plan']}
    matched_cases = {r['case'] for r in plans['matched-backends-v2-plan'] if r['mode'] == 'timing'}
    require(len(tuning_cases) == len(matched_cases) == 9 and
            all(case.startswith('matched-') for case in matched_cases), 'Wrong timing cohort membership')
    tuning_rows, matched_rows = ([r for r in rows if r['case'] in cases]
                                 for cases in (tuning_cases, matched_cases))
    require(Counter((r['scheme'], r['segment_length']) for r in tuning_rows) ==
            Counter({('cpu:1', n): 3 for n in (256, 512, 1024)}), 'Wrong CPU tuning grid')
    require(Counter(r['scheme'] for r in matched_rows) == Counter({s: 3 for s in SCHEMES}),
            'Matched cohort must contain three fresh timings per backend')
    require(len({analysis_signature(receipts[r['case']], comparator,
                                    omit=('--segment-length',)) for r in tuning_rows}) == 1,
            'CPU length tuning changes another workload input')
    routing = ('--processing-scheme', '--fft-backends', '--verbose')
    require(len({analysis_signature(receipts[r['case']], comparator, omit=routing)
                 for r in matched_rows}) == 1, 'Matched backend workloads differ')
    tuning, matched = grouped(tuning_rows, 'tuning'), grouped(matched_rows, 'matched')
    evidence.gates['timing_counts_and_geometry'] = True
    decision = evidence.read('reference-tuning-decision.json')
    evidence.recorded_inputs(decision['input_sha256'])
    selected = (decision['selected_segment_length_seconds'], decision['selected_start_pad_seconds'],
                decision['selected_end_pad_seconds'])
    require(isinstance(decision['basis'], str) and decision['basis'].strip() and
            selected in {(r['segment_length'], r['start_pad'], r['end_pad']) for r in tuning_rows} and
            all((r['segment_length'], r['start_pad'], r['end_pad']) == selected for r in matched_rows),
            'Tuning decision disagrees with measured or matched geometry')
    required_decision_inputs = {f'runs/{case}/{name}' for case in tuning_cases
                                for name in ('receipt.json', 'triggers.hdf')}
    require(required_decision_inputs <= {evidence.local_name(p) for p in decision['input_sha256']},
            'Tuning decision does not bind all nine final repetitions')
    evidence.gates['tuning_decision'] = True
    comparisons = []
    # Source-phase equivalence justifies carrying normal-CPU scientific/tuning
    # evidence forward. It does not waive the comparator's exact source check.
    for phase, length in sorted({(r['report_source_phase'], r['segment_length'])
                                 for r in receipts.values()}):
        cases = sorted(case for case, receipt in receipts.items()
                       if (receipt['report_source_phase'], receipt['segment_length']) == (phase, length))
        baseline = next((case for case in cases if case in matched_cases and
                         receipts[case]['scheme'] == 'cpu:1'), None)
        if baseline is None:
            baseline = next(case for case in cases if receipts[case]['mode'] == 'qualify' and
                            receipts[case]['scheme'] == 'cpu:1')
        for case in cases:
            if case != baseline:
                result = comparator.compare(data[baseline], data[case], comparator.DEFAULTS)
                result['baseline'], result['candidate'] = baseline, case
                result['source_phase'] = phase
                comparisons.append(result)
    require(len(comparisons) == 27 and all(r['status'] == 'pass' for r in comparisons),
            'Strict trigger parity failed or needs review', comparisons)
    evidence.gates['trigger_parity'] = True
    profile_cases = {r['case'] for name in ('reference-profiles-v2-plan', 'torch-profiles-v2-plan')
                     for r in plans[name]}
    require(all((receipts[c]['segment_length'], receipts[c]['start_pad'], receipts[c]['end_pad']) == selected
                for c in profile_cases), 'Profiles do not use the selected reference geometry')
    profile_data = profiles(evidence, profile_cases, receipts)
    evidence.gates['profiles'] = True
    old = evidence.read('compression-refinement-decision.json')
    evidence.bind('waveform-validation.json', old['failed_receipt_sha256'])
    cpu = next(g for g in matched if g['scheme'] == 'cpu:1')
    for group in matched:
        group['capacity_ratio_to_normal_cpu'] = (group['templates_per_core_at_real_time']['median'] /
                                                cpu['templates_per_core_at_real_time']['median'])
    return dict(schema_version=1, status='pass', gates=evidence.gates,
                required_cases=sorted(expected), input_sha256=evidence.input_sha256,
                figures=list(FIGURES), source_phases={name: value['commit'] for name, value in evidence.sources.items()},
                source_equivalence=dict(receipt='reference-source-equivalence.json',
                                        changed_paths=equivalence['changed_paths'], checks=equivalence['checks']),
                bank_sha256=BANK_SHA,
                hardware={k: hardware[k] for k in ('hostname', 'cpu', 'gpu', 'versions', 'affinity')},
                selected_geometry=dict(segment_length_seconds=selected[0], start_pad_seconds=selected[1],
                                       end_pad_seconds=selected[2], basis=decision['basis']),
                accounting=dict(formula='96 templates * 1904 unique valid detector seconds / full-process wall seconds / 1 allocated host core',
                                units='templates per core at real time', allocated_host_cores=1,
                                cuda_device_count=1, repetition_summary='median and full observed min/max range; three repetitions; no confidence interval',
                                internal_timing='Executable run_time excludes early imports/startup and final HDF writing. Setup and postsetup derive from setup_time_fraction; postsetup includes analysis housekeeping.',
                                profile_timing='Profiled executions are separate from all capacity groups; each profiler retains its own denominator.'),
                scientific_validation=scientific, rows=rows, tuning=tuning, matched=matched,
                parity=dict(status='pass', tolerances=comparator.DEFAULTS, comparisons=comparisons),
                profiles=profile_data, supplemental=dict(rejected_original_template_psd_pairs=len(old['failed_pairs']),
                                                        decision=old['decision']),
                limitations=[
                    'Finite 1904-second valid interval; no steady-state throughput claim.',
                    'Three FFT lengths and one final padding configuration; no global optimum claim.',
                    'CPU-normalized CUDA capacity additionally consumes one GPU; it is not a GPU-free resource comparison.',
                    '96 deterministic point-particle aligned-spin templates do not establish search-bank coverage, tidal/disruption accuracy, or population-weighted throughput.',
                    'Length changes can alter PSD bins and triggers; trigger parity is tested only within the same FFT geometry.',
                    'Reference tuning/science and matched backend timings use two explicitly bound source phases. The bridge verifies a TorchScheme-only deepcopy method plus tests; strict trigger parity remains within each source phase.',
                    'Hardware inventory was captured on the parent checkout; each run receipt binds the actual final source revision.',
                    'Native perf percentages, cProfile seconds and CUDA event durations must not be added or interpreted as one timeline.'])


def write_csv(path, rows):
    require(bool(rows), f'Cannot publish an empty table: {path.name}')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: canonical(v) if isinstance(v, (dict, list)) else v
                          for k, v in row.items()} for row in rows)


def figures(report, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 220})
    colors = ('#2b6f8e', '#b47a22', '#74539c')

    def finish(fig, name):
        fig.savefig(out / name, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    def dots(ax, groups, metric, labels):
        for x, group in enumerate(groups):
            values = [r[metric] for r in report['rows'] if r['case'] in group['cases']]
            summary = group[metric]
            ax.errorbar(x, summary['median'], yerr=[[summary['median'] - summary['min']],
                         [summary['max'] - summary['median']]], fmt='o', color=colors[x % 3],
                         markersize=8, capsize=6, linewidth=2)
            ax.scatter(x + np.linspace(-.08, .08, len(values)), values,
                       color=colors[x % 3], s=20, alpha=.65)
        ax.set_xticks(range(len(groups)), labels)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=.2)

    fig, ax = plt.subplots(figsize=(7.6, 4.3), layout='constrained')
    dots(ax, report['tuning'], 'templates_per_core_at_real_time',
         [f"{g['segment_length']} s\n{g['segments_per_template']} segments/template" for g in report['tuning']])
    ax.set(ylabel='Templates per core at real time', xlabel='Normal CPU FFT length',
           title='Final-bank reference tuning · full-process wall time')
    fig.supxlabel('96 templates × 1904 valid seconds · one host core · median and observed range (n=3)', fontsize=9)
    finish(fig, 'tuning-capacity.png')
    fig, ax = plt.subplots(figsize=(7.6, 4.3), layout='constrained')
    dots(ax, report['matched'], 'templates_per_core_at_real_time', [LABELS[g['scheme']] for g in report['matched']])
    ax.set(ylabel='Templates per core at real time', title='Matched backend capacity · fresh unprofiled repetitions')
    fig.supxlabel('One allocated host core; Torch CUDA additionally uses one GPU · median and observed range (n=3)', fontsize=9)
    finish(fig, 'matched-capacity.png')
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    for ax, metric, title in zip(axes.flat,
            ('wall_seconds', 'internal_runtime_seconds', 'internal_setup_seconds', 'internal_postsetup_seconds'),
            ('Full-process wall time', 'Executable internal runtime', 'Executable setup', 'Executable postsetup')):
        dots(ax, report['matched'], metric, [LABELS[g['scheme']] for g in report['matched']])
        ax.set(ylabel='Seconds', title=title)
    fig.supxlabel('Separate measured intervals; postsetup includes housekeeping and is not steady-state kernel time.', fontsize=9)
    finish(fig, 'wall-and-internal-times.png')
    rows = [r for r in report['profiles']['groups'] if r['mode'] == 'cprofile']
    groups = list(dict.fromkeys(r['group'] for r in rows))
    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    left = np.zeros(3)
    for index, group in enumerate(groups):
        values = [sum(r['share_percent'] for r in rows if r['scheme'] == s and r['group'] == group) for s in SCHEMES]
        ax.barh(range(3), values, left=left, label=group, color=plt.get_cmap('tab10')(index))
        left += values
    ax.set(yticks=range(3), yticklabels=[LABELS[s] for s in SCHEMES], xlim=(0, 100),
           xlabel='Percent of this cProfile run’s exclusive self time', title='Profile ownership groups · instrumented runs')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.2), ncol=2, frameon=False, fontsize=9)
    finish(fig, 'profile-self-time.png')
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), layout='constrained')
    for ax, scheme in zip(axes, SCHEMES):
        rows = [r for r in report['profiles']['native_symbols'] if r['scheme'] == scheme][:8][::-1]
        labels = [r['symbol'] if len(r['symbol']) <= 82 else r['symbol'][:79] + '…' for r in rows]
        ax.barh(range(len(rows)), [r['self_event_percent'] for r in rows], color=colors[SCHEMES.index(scheme)])
        ax.set(yticks=range(len(rows)), yticklabels=labels, xlabel='Direct cycles:u event-period percentage',
               title=LABELS[scheme] + ' · native CPU sampling')
        ax.tick_params(axis='y', labelsize=8)
    fig.supxlabel('Top reported symbols; percentages retain the full perf event denominator. CPU sampling does not measure CUDA kernels.', fontsize=9)
    finish(fig, 'native-symbols.png')
    rows = report['profiles']['cuda_device_events'][:10][::-1]
    fig, ax = plt.subplots(figsize=(11, 6), layout='constrained')
    labels = [r['group'] if len(r['group']) <= 95 else r['group'][:92] + '…' for r in rows]
    ax.barh(range(len(rows)), [r['numerator'] / 1000 for r in rows], color=colors[2])
    ax.set(yticks=range(len(rows)), yticklabels=labels, xlabel='Summed CUDA device-event self duration (ms)',
           title='Torch CUDA device events · separate instrumented execution')
    ax.tick_params(axis='y', labelsize=8)
    fig.supxlabel('Device-event rows only; event durations can overlap across streams. Their sum is neither wall time nor utilization.', fontsize=9)
    finish(fig, 'cuda-events.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_dir.exists(), 'Use a new output directory to preserve earlier results')
    evidence = Evidence(args.root)
    try:
        report = build(evidence)
        # A final reread catches evidence mutations during report construction.
        for name, digest in list(evidence.input_sha256.items()):
            evidence.bind(name, digest)
    except Exception as error:
        report = dict(schema_version=1, status='fail', gates=evidence.gates,
                      input_sha256=evidence.input_sha256, error=str(error),
                      error_type=type(error).__name__, details=getattr(error, 'details', None))
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(f'Report withheld: {error}', file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        write_csv(args.output_dir / 'runs.csv', report['rows'])
        flat = []
        for group in report['tuning'] + report['matched']:
            item = {k: v for k, v in group.items() if k not in METRICS}
            for metric in METRICS:
                item.update({metric + '_' + stat: value for stat, value in group[metric].items()})
            flat.append(item)
        write_csv(args.output_dir / 'groups.csv', flat)
        write_csv(args.output_dir / 'profiles.csv', report['profiles']['groups'])
        write_csv(args.output_dir / 'native-symbols.csv', report['profiles']['native_symbols'])
        figures(report, args.output_dir)
        report['output_sha256'] = {p.name: sha(p) for p in sorted(args.output_dir.iterdir())}
        # The success receipt is written last; partial figure generation cannot
        # leave an apparently complete report for downstream archive staging.
        with (args.output_dir / 'report.json').open('x') as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    except Exception as error:
        with (args.output_dir / 'report.json').open('x') as stream:
            stream.write(json.dumps(dict(schema_version=1, status='fail', error=str(error),
                                        error_type=type(error).__name__), indent=2) + '\n')
        raise
    print(args.output_dir.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
