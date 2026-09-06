#!/usr/bin/env python3
"""Build the final single-host-core inspiral report from frozen local evidence.

Usage: python build-reference-report-v4.py --output-dir final-report
Requires the six final plans/status ledgers, precision5-reference-tuning-decision.json,
profiles-precision5.json, all named runs, and the two final scientific validation receipts.
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


COMMIT = '837f38d493420043e45fb1ad210a0ccf68bacbaa'
PARENT_COMMIT = 'f2c0abe61e787a26f41208f489c62c877bbd5667'
PRECISION_COMMIT = '6c82155044d58f3344b281869d87745f71ba2285'
PREVIOUS_COMMIT = '968bcd558117262af0d603710b054174659adb51'
BANK_SHA = '161820754117cd51b24a7bb672ea456cf7b2a3808e69690c23235d024b1a2916'
SCHEMES = ('cpu:1', 'torch:cpu:1', 'torch:cuda:0')
LABELS = dict(zip(SCHEMES, ('Normal CPU (MKL)', 'Torch CPU', 'Torch CUDA')))
PLANS = ('precision5-reference-qualifications-plan', 'precision5-reference-timings-plan',
         'precision5-final-qualifications-plan', 'precision5-reference-profiles-plan',
         'precision5-matched-backends-plan', 'precision5-torch-profiles-plan')
WAVEFORM_TOLERANCES = dict(fixed_time_phase_optimized_mismatch=0.001,
                           relative_snr_norm_error=0.01,
                           relative_weighted_amplitude_error=0.01)
BOUNDARY_TOLERANCES = dict(edge_energy_fraction=1e-6,
                          maximum_relative_complex_error=0.001,
                          peak_location_error_samples=1, relative_peak_snr_error=0.001)
PARITY_TOLERANCES = dict(rtol=1e-4, atol=1e-5, sigmasq_rtol=1e-5,
                         phase_atol=1e-4, max_examples=12)
CHANGED_PATHS = (
    'pycbc/filter/matchedfilter.py', 'pycbc/psd/__init__.py',
    'pycbc/strain/strain.py', 'pycbc/vetoes/chisq.py',
    'pycbc/vetoes/chisq_torch.py',
    'test/test_chisq_precision.py', 'test/test_sigmasq_series_precision.py',
    'test/test_strain_psd_precision.py',
)
PARENT_CHANGED_PATHS = ('pycbc/vetoes/chisq_torch.py', 'test/test_chisq_precision.py')
PRECISION_CHANGED_PATHS = tuple(name for name in CHANGED_PATHS
                                if name != 'pycbc/vetoes/chisq_torch.py')
UNIT_TESTS = (
    'test/test_scheme_runtime.py', 'test/test_scheme_selection.py',
    'test/test_matchedfilter.py', 'test/test_chisq.py',
    'test/test_psd.py', 'test/test_strain.py',
    'test/test_sigmasq_series_precision.py', 'test/test_chisq_precision.py',
    'test/test_strain_psd_precision.py', 'test/test_torch_chisq_cpu_optimization.py',
    'test/test_torch_chisq_sparse_dispatch.py', 'test/test_torch_filter_pipeline.py',
    'test/test_torch_psd_pipeline.py', 'test/test_torch_psd_protocol.py',
    'test/test_torch_versioned_data_psd.py', 'test/test_torch_matchedfilter_cpu_optimization.py',
)


def plan_cases(selected):
    require(type(selected) is int and selected in (256, 512, 1024),
            'Selected FFT length must be 256, 512 or 1024')
    return {
        PLANS[0]: {f'qual-precision5-cpu-l{length}': ('qualify', 'cpu:1', length)
                   for length in (256, 512, 1024)},
        PLANS[1]: {f'tune-precision5-cpu-l{length}-r{rep}': ('timing', 'cpu:1', length)
                   for length in (256, 512, 1024) for rep in (1, 2, 3)},
        PLANS[2]: {f'qual-selected5-{label}-l{selected}': ('qualify', scheme, selected)
                   for label, scheme in zip(('cpu', 'torch-cpu', 'torch-cuda'), SCHEMES)},
        PLANS[3]: {f'reference-precision5-cpu-l{selected}-{mode}': (mode, 'cpu:1', selected)
                   for mode in ('cprofile', 'perf')},
        PLANS[4]: {f'matched-precision5-{label}-l{selected}-r{rep}': ('timing', scheme, selected)
                   for label, scheme in zip(('cpu', 'torch-cpu', 'torch-cuda'), SCHEMES)
                   for rep in (1, 2, 3)},
        PLANS[5]: {
            **{f'profile-precision5-torch-{device}-l{selected}-{mode}': (mode, scheme, selected)
               for device, scheme in zip(('cpu', 'cuda'), SCHEMES[1:])
               for mode in ('cprofile', 'perf')},
            f'profile-precision5-torch-cuda-l{selected}-torchprofile':
                ('torchprofile', 'torch:cuda:0', selected)},
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
    require(type(value) in (int, float) and math.isfinite(value) and
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
            if relative is not None and not relative.startswith(
                    ('source/', 'source-v2/', 'source-v3/', 'source-v4/', 'source-v5/')):
                self.bind(relative, digest)


def read_plans(evidence, comparator, selected):
    plans, all_cases = {}, {}
    expected_cases = plan_cases(selected)
    for name in PLANS:
        plan = evidence.read(name + '.json')
        require(isinstance(plan, list) and plan, f'Empty/malformed plan: {name}')
        status = evidence.read(name + '.status.json')
        require(status['state'] == 'complete' and status['returncode'] == 0 and
                status['completed'] == plan and status.get('current') is None and
                bool(status.get('finished_utc')) and
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
            row['source_phase'] = 'precision'
            all_cases[case] = row
            rows.append(row)
        require({row['case']: (row['mode'], row['scheme'], row['segment_length'])
                 for row in rows} == expected_cases[name] and
                all((row['start_pad'], row['end_pad']) == (112, 16) for row in rows),
                f'Unexpected final case names, modes, schemes or geometry: {name}')
        plans[name] = rows
    require(len(all_cases) == 31, 'Final campaign must contain exactly 31 distinct runs')
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


def validate_science(evidence, qualification_cases, bank_hashes):
    waveform = evidence.read('waveform-validation-precision5.json')
    boundary = evidence.read('boundary-injections-precision5.json')
    for name, value, field, count in (
            ('waveform', waveform, 'template_psd_pairs', 288), ('boundary', boundary, 'cases', 36)):
        require(value['state'] == 'complete' and value['passed'] is True and
                value['completed_' + field] == value['expected_' + field] == count,
                f'{name}: incomplete or failed scientific validation')
        require(value['source_before'] == value['source_after'] and
                value['source_before']['commit'] == COMMIT and not value['source_before']['status'],
                f'{name}: source changed during validation')
        evidence.recorded_inputs(value['input_sha256'], value['input_sha256_after'])
    require(canonical(waveform['tolerances']) == canonical(WAVEFORM_TOLERANCES) and
            canonical(boundary['tolerances']) == canonical(BOUNDARY_TOLERANCES),
            'Scientific validation tolerances changed')
    cases = waveform['cases']
    require(len(cases) == 3 and
            {evidence.local_name(c['qualification']) for c in cases} ==
            {f'runs/{case}/qualification.json' for case in qualification_cases},
            'Waveform validation does not cover every final CPU qualification')
    required_inputs(evidence, waveform['input_sha256'],
                    [f'runs/{case}/qualification.json' for case in qualification_cases])
    boundary_qualification = 'runs/qual-precision5-cpu-l1024/qualification.json'
    require(evidence.local_name(boundary['qualification']) == boundary_qualification,
            'Boundary validation uses a different qualification')
    required_inputs(evidence, boundary['input_sha256'], [boundary_qualification])
    pairs = []
    for case in cases:
        require(case['passed'] is True and case['state'] == 'complete' and len(case['templates']) == 96,
                'Incomplete waveform-validation case')
        require({int(template['template_hash']) for template in case['templates']} == bank_hashes,
                'Waveform validation does not cover the final bank template identities')
        for template in case['templates']:
            require(template['passed'] is True and template['decompression_succeeded_without_generation'] is True,
                    'Waveform template validation failed')
            pairs.extend(template['psd_results'])
    require(len(pairs) == 288 and all(p['passed'] is True for p in pairs),
            'Waveform per-pair count/pass mismatch')
    for pair in pairs:
        for name, budget in WAVEFORM_TOLERANCES.items():
            require(positive(pair['metrics'][name], name, zero=True) <= budget,
                    'Waveform metric exceeds its frozen scientific budget')
    boundary_rows = [c for template in boundary['templates'] for c in template['cases']]
    require(len(boundary_rows) == 36 and all(t['passed'] is True and
            t['reference_edge_budget_passed'] is True for t in boundary['templates']) and
            all(c['passed'] is True for c in boundary_rows), 'Boundary per-case count/pass mismatch')
    require(len(boundary['templates']) == 2 and
            {t['kind'] for t in boundary['templates']} == {'longest', 'shortest'},
            'Boundary validation must include the longest and shortest templates')
    boundary_grid = {(length, pad, 16, placement)
                     for length in (256, 512, 1024) for pad in (96, 112)
                     for placement in ('first_valid', 'midpoint', 'last_valid')}
    for template in boundary['templates']:
        require(int(template['reference']['template_hash']) in bank_hashes and
                len(template['cases']) == 18 and
                {(case['segment_length_seconds'], case['start_pad_seconds'],
                  case['end_pad_seconds'], case['placement']) for case in template['cases']} == boundary_grid,
                'Boundary validation does not cover the declared template/geometry grid')
        for field in ('injection_edge_energy_fraction', 'kernel_edge_energy_fraction'):
            require(positive(template['reference']['tails'][field], field, zero=True) <=
                    BOUNDARY_TOLERANCES['edge_energy_fraction'], 'Boundary reference edge budget failed')
        for case in template['cases']:
            for name, budget in BOUNDARY_TOLERANCES.items():
                if name != 'edge_energy_fraction':
                    require(positive(case[name], name, zero=True) <= budget,
                            'Boundary metric exceeds its frozen scientific budget')
    evidence.gates['waveform_validation'] = evidence.gates['boundary_validation'] = True
    return dict(waveform_template_psd_pairs=288, boundary_cases=36,
                waveform_tolerances=waveform['tolerances'], boundary_tolerances=boundary['tolerances'],
                maximum_waveform_errors={key: max(p['metrics'][key] for p in pairs)
                                         for key in waveform['tolerances']})


def profiles(evidence, selected, receipts):
    summary = evidence.read('profiles-precision5.json')
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


def required_inputs(evidence, recorded, names):
    """Require named local evidence to be covered by a receipt's manifest."""
    evidence.recorded_inputs(recorded)
    local = {evidence.local_name(name): digest for name, digest in recorded.items()}
    for name in names:
        require(name in local, f'Missing required input hash: {name}')
        evidence.bind(name, local[name])


def read_source(evidence):
    source = evidence.read('source-v5.json')
    source_path = Path(source['source'])
    require(source_path.is_absolute() and source_path.name == 'source-v5' and
            source['commit'] == COMMIT and source['parent'] == PARENT_COMMIT and
            source['parent_source_commit'] == PARENT_COMMIT and
            source['reference_base'] == PREVIOUS_COMMIT and
            source['changed_paths'] == list(CHANGED_PATHS),
            'Wrong final source revision, parent or changed paths')
    evidence.remote = source_path.parent
    previous = evidence.read('source-v2.json')
    require(previous['commit'] == PREVIOUS_COMMIT and
            previous['source'] == str(evidence.remote / 'source-v2'),
            'Wrong previous source provenance')
    precision = evidence.read('source-v3.json')
    require(precision['commit'] == PRECISION_COMMIT and precision['parent'] == PREVIOUS_COMMIT and
            precision['source'] == str(evidence.remote / 'source-v3') and
            precision['changed_paths'] == list(PRECISION_CHANGED_PATHS), 'Wrong precision source ancestor')
    parent = evidence.read('source-v4.json')
    require(parent['commit'] == PARENT_COMMIT and parent['parent'] == PRECISION_COMMIT and
            parent['parent_source_commit'] == PRECISION_COMMIT and
            parent['reference_base'] == PREVIOUS_COMMIT and
            parent['source'] == str(evidence.remote / 'source-v4') and
            parent['changed_paths'] == list(PRECISION_CHANGED_PATHS), 'Wrong precision source parent')
    native = source['native_modules_sha256']
    require(len(native) == 11 and
            native == previous['native_modules_sha256'] == precision['native_modules_sha256'] ==
            parent['native_modules_sha256'] and
            all(name.startswith('pycbc/') and name.endswith('.so') and
                '..' not in Path(name).parts and re.fullmatch('[0-9a-f]{64}', digest)
                for name, digest in native.items()), 'Native module hashes changed')
    for record, paths in ((source, CHANGED_PATHS), (parent, PRECISION_CHANGED_PATHS),
                          (precision, PRECISION_CHANGED_PATHS)):
        require(set(record['changed_files_sha256']) == set(paths) and
                all(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest)
                    for digest in record['changed_files_sha256'].values()),
                'Missing or malformed changed-file hashes')
    require(all(parent['changed_files_sha256'][name] == precision['changed_files_sha256'][name]
                for name in PRECISION_CHANGED_PATHS if name != 'test/test_sigmasq_series_precision.py'),
            'Parent revision changes files beyond the reviewed oracle correction')
    require(all(source['changed_files_sha256'][name] == parent['changed_files_sha256'][name]
                for name in CHANGED_PATHS if name not in PARENT_CHANGED_PATHS),
            'Final revision changes files beyond the reviewed launch and regression')
    for name, record in (('inspiral-source-v2.bundle', previous),
                         ('inspiral-source-v3.bundle', precision),
                         ('inspiral-source-v4.bundle', parent),
                         ('inspiral-source-v5.bundle', source)):
        evidence.bind(name, record['bundle_sha256'])
    proof = evidence.read('source-precision-provenance-v5.json')
    expected_checks = ('only_reviewed_python_and_test_paths_changed',
                       'native_hashes_equal', 'both_sources_clean',
                       'all_final_runs_require_new_source', 'host_launch_and_regression_only_parent_delta')
    require(proof['schema_version'] == 1 and proof['status'] == 'pass' and
            proof['old_source_commit'] == PREVIOUS_COMMIT and
            proof['new_source_commit'] == COMMIT and
            proof['parent_source_commit'] == PARENT_COMMIT and
            proof['reference_base'] == PREVIOUS_COMMIT and
            proof['changed_paths'] == list(CHANGED_PATHS) and
            proof['parent_changed_paths'] == list(PARENT_CHANGED_PATHS) and
            proof['normal_cpu_outputs_changed'] is True and
            proof['prior_science_reused'] is False and
            set(proof['checks']) == set(expected_checks) and
            all(proof['checks'][name] is True for name in expected_checks) and
            isinstance(proof['scope'], str) and proof['scope'].strip(),
            'Incomplete or incorrect precision source provenance')
    diff_paths = re.findall(r'^diff --git a/(\S+) b/(\S+)$',
                            proof['exact_git_diff'], flags=re.MULTILINE)
    require(diff_paths == [(name, name) for name in CHANGED_PATHS],
            'Precision source diff changes unexpected paths')
    parent_diff = proof['exact_parent_git_diff']
    parent_paths = re.findall(r'^diff --git a/(\S+) b/(\S+)$', parent_diff, flags=re.MULTILINE)
    require(parent_paths == [(name, name) for name in PARENT_CHANGED_PATHS],
            'Parent source diff changes unexpected paths')
    runtime_diff = re.split(r'(?=^diff --git )', parent_diff, flags=re.MULTILINE)[1]
    hunks = re.findall(r'^@@ .*$', runtime_diff, flags=re.MULTILINE)
    changes = {marker: [line[1:] for line in runtime_diff.splitlines()
                       if line.startswith(marker) and not line.startswith(marker * 3)
                       and line[1:].strip() and not line[1:].lstrip().startswith('#')]
               for marker in ('-', '+')}
    require(len(hunks) == 1 and
            hunks[0].endswith('def _accelerator_batched_bin_sums(corr, pts, bins):') and
            changes == {'-': ['            two_pi_over_N,'],
                        '+': ['            tl.constexpr(two_pi_over_N),']},
            'Parent runtime diff is not the reviewed host launch change')
    required_inputs(evidence, proof['input_sha256'], (
        'setup-source-v5.py', 'source-v2.json', 'source-v5.json',
        'source-v3.json', 'source-v4.json', 'inspiral-source-v2.bundle',
        'inspiral-source-v3.bundle', 'inspiral-source-v4.bundle', 'inspiral-source-v5.bundle'))
    evidence.sources['precision'] = source
    evidence.gates['source_provenance'] = True
    return source, proof


def read_unit_tests(evidence, source, config):
    receipt = evidence.read('unit-tests-v5.json')
    require(receipt['state'] == 'complete' and receipt['passed'] is True and
            receipt['returncode'] == 0 and receipt.get('finished_utc') and
            receipt['source_info'] == receipt['source_after'] ==
            dict(commit=COMMIT, status=''), 'Final source unit tests failed or changed source')
    require(receipt['host'] == 'len' and receipt['cwd'] == source['source'] and
            receipt['environment'] == dict(config['environment'], PYTHONPATH=source['source'],
                                           PYTHONDONTWRITEBYTECODE='1'),
            'Wrong unit-test host, source or environment')
    command = receipt['command']
    require(command[:3] == ['taskset', '-c', str(config['core'])] and
            Path(command[3]).is_absolute() and
            command[4:] == ['-m', 'pytest', '-q', '-p', 'no:cacheprovider', *UNIT_TESTS],
            'Unit-test command does not cover the frozen regression suite')
    evidence.recorded_inputs(receipt['input_sha256'], receipt['input_sha256_after'])
    required_inputs(evidence, receipt['input_sha256'],
                    ('run-unit-checks-v5.py', 'config.json', 'source-v5.json'))
    source_inputs = {str(Path(source['source']) / name): digest
                     for name, digest in source['changed_files_sha256'].items()}
    require(all(receipt['input_sha256'].get(name) == digest
                for name, digest in source_inputs.items()),
            'Unit-test source hashes differ from the corrected source manifest')
    required = ('bin/pycbc_inspiral', 'pycbc/scheme.py', *CHANGED_PATHS, *UNIT_TESTS)
    require(all(str(Path(source['source']) / name) in receipt['input_sha256']
                for name in required), 'Unit tests omit required source/test input hashes')
    log = evidence.bind('unit-tests-v5.log', receipt['log_sha256'])
    lines = [line.strip() for line in log.read_text().splitlines() if line.strip()]
    require(lines and re.search(r'\b[1-9][0-9]* passed\b', lines[-1]) and
            not re.search(r'\b[1-9][0-9]* (?:failed|errors?)\b', lines[-1]),
            'Unit-test log has no passing terminal summary')
    evidence.gates['unit_tests'] = True
    return dict(receipt='unit-tests-v5.json', log='unit-tests-v5.log',
                command=command, summary=lines[-1], source_commit=COMMIT,
                wall_seconds=positive(receipt['wall_seconds'], 'unit-test wall time'))


def read_decision(evidence):
    decision = evidence.read('precision5-reference-tuning-decision.json')
    selected = tuple(decision['selected_' + name + '_seconds']
                     for name in ('segment_length', 'start_pad', 'end_pad'))
    require(decision['schema_version'] == 1 and decision['source_commit'] == COMMIT and
            all(type(value) is int for value in selected) and
            selected[0] in (256, 512, 1024) and selected[1:] == (112, 16) and
            isinstance(decision['basis'], str) and decision['basis'].strip(),
            'Invalid or stale precision tuning decision')
    evidence.recorded_inputs(decision['input_sha256'])
    return decision, selected


def collect_qualifications(evidence, receipts, data, bank_hashes, config, comparator, selected):
    expected = set(plan_cases(selected)[PLANS[0]]) | set(plan_cases(selected)[PLANS[2]])
    cases = {case for case, receipt in receipts.items() if receipt['mode'] == 'qualify'}
    require(cases == expected, 'Final campaign must contain all six qualifications')
    qualified, own_work, owners = {}, {}, {}
    duplicate = {f'qual-precision5-cpu-l{selected}', f'qual-selected5-cpu-l{selected}'}
    # Both repeated CPU qualifications retain their own observed work. The
    # selected qualification supplies work to subsequent timings and profiles.
    for case in sorted(cases):
        receipt = receipts[case]
        work = qualification(evidence, receipt, bank_hashes, config)
        own_work[case] = work
        key = analysis_signature(receipt, comparator, profile=True)
        if key in qualified:
            previous = owners[key]
            require({case, previous} == duplicate, 'Unexpected duplicate qualification')
            fields = [name for name in work if name != 'qualification']
            require(all(work[name] == qualified[key][name] for name in fields),
                    'Repeated CPU qualifications observed different work')
            before = evidence.read(f'runs/{previous}/qualification.json')['observations']['segment_geometry']
            after = evidence.read(f'runs/{case}/qualification.json')['observations']['segment_geometry']
            require(before == after, 'Repeated CPU qualification sample coverage differs')
            result = comparator.compare(data[previous], data[case], PARITY_TOLERANCES)
            require(result['status'] == 'pass', 'Repeated CPU qualification trigger parity failed', result)
        qualified[key], owners[key] = work, case
    require(len(qualified) == 5, 'Repeated selected CPU qualifications differ in configuration')
    final = f'qual-selected5-cpu-l{selected}'
    key = analysis_signature(receipts[final], comparator, profile=True)
    qualified[key] = own_work[final]
    return qualified, own_work


def validate_decision(evidence, decision, selected, tuning_rows, matched_rows):
    grid = decision['selection_grid']
    require(len(grid) == 3 and
            [row['segment_length_seconds'] for row in grid] == [256, 512, 1024],
            'Tuning decision must record the three declared FFT lengths in order')
    rows = {row['case']: row for row in tuning_rows}
    expected = plan_cases(selected[0])[PLANS[1]]
    require(set(rows) == set(expected) and len(tuning_rows) == 9,
            'Tuning decision requires the nine distinct unprofiled timings')
    medians = {}
    for item in grid:
        length = item['segment_length_seconds']
        values = [rows[f'tune-precision5-cpu-l{length}-r{rep}']['wall_seconds']
                  for rep in (1, 2, 3)]
        require(all(positive(value, 'tuning wall time') for value in values),
                'Invalid tuning wall times')
        median = statistics.median(values)
        require(item['wall_seconds'] == values and item['median_wall_seconds'] == median,
                f'Tuning decision timings or median differ at {length} seconds')
        medians[length] = median
    require(selected[0] == min(medians, key=lambda length: (medians[length], length)),
            'Selected FFT length is not the lowest median wall time')
    require(all((row['segment_length'], row['start_pad'], row['end_pad']) == selected
                for row in matched_rows), 'Matched runs differ from the selected geometry')
    names = [name + suffix for name in PLANS[:2] for suffix in ('.json', '.status.json')]
    names.extend(f'runs/{case}/{file}' for case in expected for file in ('receipt.json', 'triggers.hdf'))
    required_inputs(evidence, decision['input_sha256'], names)


def build(evidence):
    evidence.bind(Path(__file__).name)
    for name in ('compare-triggers.py', 'summarize-runs.py'):
        evidence.bind(name)
    comparator = helper(evidence.root, 'compare-triggers.py')
    require(comparator.DEFAULTS == PARITY_TOLERANCES, 'Trigger comparator budgets changed')
    config = evidence.read('config.json')
    source, provenance = read_source(evidence)
    unit_tests = read_unit_tests(evidence, source, config)
    decision, selected = read_decision(evidence)
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
    plans, expected = read_plans(evidence, comparator, selected[0])
    receipts, data = {}, {}
    for case, item in expected.items():
        receipts[case], data[case] = read_run(evidence, item, config, comparator)
    evidence.gates['source_and_inputs_bound'] = True
    qualified, own_work = collect_qualifications(evidence, receipts, data,
                                                bank_hashes, config, comparator, selected[0])
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
        rows.append(timing_row(receipt, own_work.get(case, qualified[key]), data[case]))
    evidence.gates['qualifications'] = True
    scientific = validate_science(evidence, [r['case'] for r in plans[PLANS[0]]], bank_hashes)
    tuning_cases = {r['case'] for r in plans[PLANS[1]]}
    matched_cases = {r['case'] for r in plans[PLANS[4]] if r['mode'] == 'timing'}
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
    validate_decision(evidence, decision, selected, tuning_rows, matched_rows)
    evidence.gates['tuning_decision'] = True
    comparisons = []
    for length in (256, 512, 1024):
        cases = sorted(case for case, receipt in receipts.items()
                       if receipt['segment_length'] == length)
        baseline = (f'qual-selected5-cpu-l{length}' if length == selected[0]
                    else f'qual-precision5-cpu-l{length}')
        require(baseline in cases, 'Missing CPU parity baseline')
        for case in cases:
            if case != baseline:
                result = comparator.compare(data[baseline], data[case], PARITY_TOLERANCES)
                result['baseline'], result['candidate'] = baseline, case
                result['source_phase'] = 'precision'
                comparisons.append(result)
    require(len(comparisons) == 28 and all(r['status'] == 'pass' for r in comparisons),
            'Strict trigger parity failed or needs review', comparisons)
    evidence.gates['trigger_parity'] = True
    profile_cases = {r['case'] for name in (PLANS[3], PLANS[5])
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
                source_provenance=dict(receipt='source-precision-provenance-v5.json',
                                       changed_paths=provenance['changed_paths'],
                                       checks=provenance['checks'],
                                       normal_cpu_outputs_changed=True, prior_science_reused=False),
                unit_tests=unit_tests,
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
                    'All final reference tuning, scientific validation, matched timings and profiles use one corrected source. Earlier source phases are supplemental; their science and timings are not reused.',
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
