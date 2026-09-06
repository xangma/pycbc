#!/usr/bin/env python3
"""Synthetic end-to-end and adversarial tests; never execute a workload."""
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), HERE / name)
    module = importlib.util.module_from_spec(spec)
    exec(compile((HERE / name).read_bytes(), str(HERE / name), 'exec'), module.__dict__)
    return module


REPORT = load('build-hotpath-report-v6.py')
LEGACY = load('test-build-reference-report-v4.py')
RUNNER = load('run-scientific-checks-v6.py')
REMOTE = LEGACY.REMOTE
SOURCE = REMOTE / 'source-v6'
COMMIT = '6' * 40
HASH = LEGACY.HASH
NOW = '2026-09-06T00:00:00Z'
LEGACY.SOURCE = SOURCE
LEGACY.REPORT.COMMIT = COMMIT


class Fixture(LEGACY.Fixture):
    def __init__(self, root):
        self.root, self.selected = root, 512
        root.mkdir()
        helpers = set(REPORT.CAMP.PINNED) - {
            'config.json', 'source-v5.json', 'precision5-reference-tuning-decision.json',
            'inputs/bank-compressed-1e5.hdf'}
        helpers |= {'build-hotpath-report-v6.py', 'build-reference-report-v4.py',
                    'campaign-v6.py', 'launch-campaign-v6.py', 'campaign-v6-contract.md',
                    'compare-triggers.py', 'run-scientific-checks-v6.py', 'qualify-large-ifft-v6b.py'}
        for name in helpers:
            shutil.copyfile(HERE / name, root / name)
        for name in ('setup-source-v6.py', 'run-unit-checks-v6.py', 'qualify-compressed-bank-v6.py'):
            self.write(name, b'# Synthetic fixture only\n')
        self.config = dict(core=8, environment=dict(OMP_NUM_THREADS='1'),
            input_files=['/fixture/frame.gwf'], common_args=[
                '--sample-rate', '4096', '--channel-name', 'H1:STRAIN',
                '--trig-start-time', '1187007160', '--trig-end-time', '1187009064',
                '--snr-threshold', '5.5'])
        self.save('config.json', self.config)
        self.environment = dict(self.config['environment'], PYTHONPATH=str(SOURCE),
                                PYTHONDONTWRITEBYTECODE='1')
        self.save('environment.json', dict(hostname='len', affinity=[8], cpu='Synthetic CPU',
                                         gpu='Synthetic GPU', versions={}))
        self.write('inputs/bank-compressed-1e5.hdf', b'')
        with REPORT.V4.h5py.File(root / 'inputs/bank-compressed-1e5.hdf', 'w') as handle:
            handle['template_hash'] = REPORT.V4.np.arange(96, dtype='int64')
        self.bank_sha = REPORT.sha(root / 'inputs/bank-compressed-1e5.hdf')
        self.save('inputs/bank-metadata.json', dict(templates=[dict(template_hash=i) for i in range(96)]))
        self.save('compression-1e5.json', dict(state='complete', returncode=0))
        self.source()
        self.unit()
        self.fft()
        self.save('precision5-reference-tuning-decision.json', dict(source_commit=REPORT.CAMP.PARENT,
            selected_segment_length_seconds=512, selected_start_pad_seconds=112,
            selected_end_pad_seconds=16, input_sha256=self.manifest('config.json')))
        self.cases, self.receipts = {}, {}
        for row in REPORT.CAMP.case_plan('qualifications') + REPORT.CAMP.case_plan('measurements'):
            case, mode, scheme = (row[key] for key in ('case', 'mode', 'scheme'))
            self.cases[case] = (mode, scheme, 512)
            self.run(case, mode, scheme, 512, 10 + int(case[-1]) if mode == 'timing' else 15)
        for length in (256, 1024):
            self.run(f'qual-science6-cpu-l{length}', 'qualify', 'cpu:1', length, 15)
        self.profiles()
        (root / 'profiles-precision5.json').rename(root / 'profiles-v6.json')
        self.stage('qualifications')
        self.science()
        self.seal_science()
        self.stage('measurements')
        self.write('final-report/report.json', (HERE / 'final-report/report.json').read_bytes())
        self.pins = {name: REPORT.sha(root / name) for name in REPORT.CAMP.PINNED}

    def complete(self, **extra):
        inputs = self.manifest('source-v6.json')
        return dict(state='complete', passed=True, returncode=0, finished_utc=NOW,
            source_info=dict(commit=COMMIT, status=''), source_after=dict(commit=COMMIT, status=''),
            input_sha256=inputs, input_sha256_after=inputs, **extra)

    def source(self):
        native = {f'pycbc/native{i}.so': HASH for i in range(11)}
        self.save('source-v5.json', dict(commit=REPORT.CAMP.PARENT, native_modules_sha256=native))
        self.write('inspiral-source-v6.bundle', b'Synthetic source bundle\n')
        inputs = self.manifest('setup-source-v6.py', 'inspiral-source-v6.bundle', 'source-v5.json')
        self.save('source-v6.json', dict(schema_version=1, source=str(SOURCE), commit=COMMIT,
            parent=REPORT.CAMP.PARENT, changed_paths=list(REPORT.CAMP.ALLOWED),
            changed_files_sha256=dict.fromkeys(REPORT.CAMP.ALLOWED, HASH), native_modules_sha256=native,
            bundle_sha256=REPORT.sha(self.root / 'inspiral-source-v6.bundle'),
            normal_cpu_path_audit=dict(status='pass', source_commit=COMMIT,
                parent_source_commit=REPORT.CAMP.PARENT, dispatch=REPORT.CAMP.DISPATCH,
                rationale='Synthetic unchanged CPU path'),
            exact_git_diff='\n'.join(f'diff --git a/{name} b/{name}' for name in REPORT.CAMP.ALLOWED),
            input_sha256=inputs, input_sha256_after=inputs))

    def unit(self):
        self.write('unit-tests-v6.log', b'300 passed, 2 skipped in 1.00s\n')
        value = self.complete(host='len', cwd=str(SOURCE), environment=self.environment,
            command=['taskset', '-c', '8', '/fixture/python', '-m', 'pytest', '-q', '-p',
                     'no:cacheprovider', *REPORT.UNIT_TESTS],
            log_sha256=REPORT.sha(self.root / 'unit-tests-v6.log'))
        inputs = self.manifest('run-unit-checks-v6.py', 'source-v6.json', 'config.json')
        inputs.update({str(SOURCE / name): HASH for name in (*REPORT.UNIT_TESTS, *REPORT.CAMP.ALLOWED)})
        value.update(input_sha256=inputs, input_sha256_after=inputs)
        self.save('unit-tests-v6.json', value)

    def fft(self):
        before = dict(root=str(SOURCE), head=COMMIT, status='', torchfft_sha256=HASH,
                      harness_sha256=REPORT.sha(self.root / 'qualify-large-ifft-v6b.py'))
        sizes = []
        for size in (2**20, 2**21, 2**22):
            timings = {name: dict(steady_seconds=[seconds] * 9, steady_median_seconds=seconds,
                steady_min_seconds=seconds, steady_max_seconds=seconds, native_library_threads=1)
                for name, seconds in (('mkl_double_workspace', 1.),
                                      ('fftw_double_workspace_one_native_thread', 2.))}
            cases = [dict(seed=seed, pattern=pattern, scale=scale,
                gates=dict(mkl_double_workspace=dict(passed=True, bitwise_mkl_parity=True)),
                errors=dict(mkl_double_workspace=dict(l2=0., max_abs=0.),
                            legacy_fftw_single=dict(l2=1e-7, max_abs=1e-7)))
                for seed, pattern, scale in itertools.product(
                    (7, 91, 812, 20260906), ('dense', 'banded', 'impulse'), (1e-12, 1., 1e12))]
            sizes.append(dict(size=size, timings=timings, cases=cases))
        self.save('large-ifft-v6.json', dict(schema='torch-large-ifft-qualification-v6b',
            state='complete', finished_utc=NOW, source_before=before, source_after=before,
            threads=1, host='len', cpu_affinity=[8], environment=self.config['environment'], sizes=sizes))
        self.write('large-ifft-v6.log', b'Synthetic 108-case matrix\n')
        self.seal_fft()

    def seal_fft(self):
        inputs = self.manifest('source-v6.json', 'qualify-large-ifft-v6b.py',
                               'large-ifft-v6.json', 'large-ifft-v6.log')
        value = self.complete(schema_version=1, supported_threads=[1],
            promoted_native_dtype='complex128', direct_single_sizes_unchanged=[32768],
            rationale='Synthetic promoted path with strict parity and measured gain',
            dispatch_by_size={str(2**power): 'mkl_double_workspace' for power in (20, 21, 22)},
            matrix=dict(path=str(REMOTE / 'large-ifft-v6.json'),
                        sha256=REPORT.sha(self.root / 'large-ifft-v6.json')))
        value.update(input_sha256=inputs, input_sha256_after=inputs)
        self.save('large-ifft-v6-decision.json', value)

    def run(self, case, mode, scheme, length, wall):
        super().run(case, mode, scheme, length, wall)
        self.receipts[case]['finished_utc'] = NOW
        self.save(f'runs/{case}/receipt.json', self.receipts[case])

    def stage(self, stage):
        plan = REPORT.CAMP.case_plan(stage)
        name = 'campaign-v6-' + stage
        source = self.read('source-v6.json')
        names = [*REPORT.CAMP.PINNED, 'source-v6.json', 'unit-tests-v6.json', 'campaign-v6.py',
                 'launch-campaign-v6.py', 'campaign-v6-contract.md']
        if stage == 'measurements':
            names += ['scientific-validation-v6.json', 'campaign-v6-qualifications.status.json']
        inputs = self.manifest(*names)
        inputs.update({str(SOURCE / path): HASH for path in REPORT.CAMP.CPU_PATHS})
        inputs[self.config['input_files'][0]] = REPORT.CAMP.FRAME_SHA256
        outputs, commands = [], []
        for row in plan:
            logfile = f"campaign-v6-{row['case']}.log"
            self.write(logfile, b'Synthetic successful execution\n')
            outputs += [str(path.relative_to(self.root)) for path in
                        (self.root / 'runs' / row['case']).rglob('*') if path.is_file()]
            outputs.append(logfile)
            commands.append(dict(command=['/fixture/python', 'run-case.py'], returncode=0,
                                 log=str(REMOTE / logfile)))
        if stage == 'measurements':
            self.write('campaign-v6-summary.log', b'Synthetic summary\n')
            outputs += ['profiles-v6.json', 'campaign-v6-summary.log']
            commands += [dict(command=['/fixture/python', 'summarize'], returncode=0,
                              log=str(REMOTE / 'campaign-v6-summary.log')) for _ in range(4)]
        self.save(name + '.status.json', dict(schema_version=1, state='complete', returncode=0,
            stage=stage, finished_utc=NOW, source_info=dict(commit=COMMIT, status=''),
            source_after=dict(commit=COMMIT, status=''), plan=plan, completed=plan,
            current=None, child_pid=None, host='len', cwd=str(REMOTE), pid=123,
            environment=self.environment, input_sha256=inputs, input_sha256_after=inputs,
            normal_cpu_equivalence=dict(dispatch=REPORT.CAMP.DISPATCH,
                native_modules_sha256=source['native_modules_sha256'],
                reviewed_audit=source['normal_cpu_path_audit'], tuning_source_commit=REPORT.CAMP.PARENT,
                unchanged_sha256=dict.fromkeys(REPORT.CAMP.CPU_PATHS, HASH)),
            output_sha256=self.manifest(*outputs), commands=commands))
        self.write(name + '-launch.log', b'Synthetic launch\n')
        self.save(name + '-launch.json', dict(state='launched', host='len', cwd=str(REMOTE), pid=123,
            source_commit=COMMIT, source_manifest_sha256=REPORT.sha(self.root / 'source-v6.json'),
            input_sha256=self.manifest('source-v6.json'), log=str(REMOTE / (name + '-launch.log'))))

    def science(self):
        qualifier_names = {length: f'runs/{case}/qualification.json'
                           for length, case in RUNNER.QUALIFIERS.items()}
        inputs = self.manifest('config.json', *qualifier_names.values())
        common = dict(state='complete', passed=True, source_before=dict(commit=COMMIT, status=''),
            source_after=dict(commit=COMMIT, status=''), input_sha256=inputs, input_sha256_after=inputs)
        cases = [dict(qualification=str(REMOTE / name), passed=True, state='complete',
            templates=[dict(passed=True, template_hash=index, decompression_succeeded_without_generation=True,
                psd_results=[dict(passed=True, metrics=dict.fromkeys(REPORT.V4.WAVEFORM_TOLERANCES, 0.))])
                for index in range(96)]) for name in qualifier_names.values()]
        self.save('waveform-validation-v6.json', dict(common, cases=cases,
            expected_template_psd_pairs=288, completed_template_psd_pairs=288,
            tolerances=REPORT.V4.WAVEFORM_TOLERANCES))
        boundary = [dict(passed=True, segment_length_seconds=length, start_pad_seconds=pad,
            end_pad_seconds=16, placement=placement, maximum_relative_complex_error=0.,
            peak_location_error_samples=0, relative_peak_snr_error=0.)
            for length, pad, placement in itertools.product((256, 512, 1024), (96, 112),
                                                          ('first_valid', 'midpoint', 'last_valid'))]
        templates = [dict(kind=kind, passed=True, reference_edge_budget_passed=True,
            reference=dict(template_hash=index, tails=dict(injection_edge_energy_fraction=0.,
                                                          kernel_edge_energy_fraction=0.)),
            cases=copy.deepcopy(boundary)) for index, kind in enumerate(('shortest', 'longest'))]
        self.save('boundary-injections-v6.json', dict(common, templates=templates,
            qualification=str(REMOTE / qualifier_names[1024]), expected_cases=36, completed_cases=36,
            tolerances=REPORT.V4.BOUNDARY_TOLERANCES))
        compressed = self.complete(expected_cases=576, completed_cases=576, native_calls=288,
            cuda_comparisons=288, cases=[dict(device=device, length_seconds=length, index=index,
                template_hash=str(index), output_sha256=HASH, bitwise_equal=True, native_used=device == 'cpu',
                metadata=[length * 4096 // 2 + 1, 1 / length, None, None, None],
                reference='normal CPU' if device == 'cpu' else 'frozen v5 Torch CUDA')
                for device, length, index in itertools.product(('cpu', 'cuda'), (256, 512, 1024), range(96))])
        inputs = self.manifest('qualify-compressed-bank-v6.py', 'source-v6.json', 'config.json',
            'inputs/bank-metadata.json', 'inputs/bank-compressed-1e5.hdf', 'compression-1e5.json')
        compressed.update(input_sha256=inputs, input_sha256_after=inputs)
        self.save('compressed-bank-v6.json', compressed)
        self.save('qualification-parity-v6.json', dict(status='pass'))

    def seal_science(self):
        plan = RUNNER.plan(REMOTE, '/fixture/python')
        outputs = []
        for step in plan:
            logfile = str(Path(step['log']).relative_to(REMOTE))
            self.write(logfile, b'Synthetic successful scientific check\n')
            path = self.root / Path(step['output']).relative_to(REMOTE)
            outputs += ([str(item.relative_to(self.root)) for item in path.rglob('*') if item.is_file()]
                        if path.is_dir() else [str(path.relative_to(self.root))]) + [logfile]
        inputs = self.manifest('run-scientific-checks-v6.py', 'source-v6.json')
        status = self.complete(schema_version=1, host='len', cwd=str(REMOTE), cpu_affinity=[8],
            current=None, child_pid=None, environment=self.environment, command=['/fixture/python'],
            plan=plan, completed=[row['name'] for row in plan], commands=[dict(name=row['name'],
                command=row['command'], returncode=0, log=row['log']) for row in plan],
            expected_coverage=dict(waveform_template_psd_pairs=288, boundary_cases=36, compressed_bank_cases=576),
            output_sha256=self.manifest(*outputs), all_created_output_sha256=self.manifest(*outputs))
        status.update(input_sha256=inputs, input_sha256_after=inputs)
        self.save('scientific-checks-v6.status.json', status)
        mapping = dict(waveform_reference='waveform-validation-v6.json',
                       boundary_injections='boundary-injections-v6.json',
                       compressed_bank_backend_parity='compressed-bank-v6.json',
                       qualification_trigger_parity='qualification-parity-v6.json')
        names = ['source-v6.json', 'unit-tests-v6.json', 'campaign-v6-qualifications.status.json',
                 'scientific-checks-v6.status.json', *mapping.values()]
        names += [f"runs/{row['case']}/{name}" for row in REPORT.CAMP.case_plan('qualifications')
                  for name in ('qualification.json', 'triggers.hdf')]
        inputs = self.manifest(*names)
        aggregate = self.complete(schema_version=1, checks=dict.fromkeys(mapping, True),
            evidence={key: dict(path=str(REMOTE / name), sha256=REPORT.sha(self.root / name))
                      for key, name in mapping.items()})
        aggregate.update(input_sha256=inputs, input_sha256_after=inputs)
        self.save('scientific-validation-v6.json', aggregate)

    def evidence(self):
        value = REPORT.Evidence(self.root)
        source = REPORT.read_source(value, REPORT.sha(self.root / 'source-v6.json'), self.config)
        return value, source

    def build(self):
        with patch.object(REPORT.CAMP, 'PINNED', self.pins), patch.object(REPORT.V4, 'BANK_SHA', self.bank_sha):
            return REPORT.build(REPORT.Evidence(self.root), REPORT.sha(self.root / 'source-v6.json'))


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic-hotpath-report-v6-')
        self.addCleanup(self.temp.cleanup)
        self.fixture = Fixture(Path(self.temp.name) / 'evidence')

    def mutate(self, name, change):
        value = self.fixture.read(name)
        change(value)
        self.fixture.save(name, value)

    def test_full_report_and_artifacts(self):
        report = self.fixture.build()
        self.assertEqual(report['status'], 'pass')
        self.assertTrue(all(report['gates'].values()))
        self.assertEqual((len(report['rows']), len(report['matched']), len(report['parity']['comparisons'])),
                         (19, 3, 18))
        self.assertEqual(report['scientific_validation']['compressed_backend_cases'], 576)
        self.assertEqual(report['large_ifft']['cases'], 108)
        for group in report['matched']:
            self.assertEqual((group['n'], group['source_commit']), (3, COMMIT))
            self.assertAlmostEqual(group['templates_per_core_at_real_time']['median'], 96 * 1904 / 12)
        self.assertEqual({pair['before_source_commit'] for pair in report['before_after']}, {REPORT.CAMP.PARENT})
        out = Path(self.temp.name) / 'synthetic-output'
        args = ['--root', str(self.fixture.root), '--source-manifest-sha256',
                REPORT.sha(self.fixture.root / 'source-v6.json'), '--output-dir', str(out)]
        with patch.object(REPORT.CAMP, 'PINNED', self.fixture.pins), \
                patch.object(REPORT.V4, 'BANK_SHA', self.fixture.bank_sha):
            self.assertEqual(REPORT.main(args), 0)
            with self.assertRaisesRegex(Exception, 'previous reports are immutable'):
                REPORT.main(args)
        saved = json.loads((out / 'report.json').read_text())
        self.assertEqual(len(saved['output_sha256']), 8)
        self.assertEqual(set(saved['figures']), set(REPORT.FIGURES))
        for name, digest in saved['output_sha256'].items():
            self.assertEqual(REPORT.sha(out / name), digest)

    def test_source_and_native_audit_reject_unreviewed_delta(self):
        for field, change in [('parent', lambda v: v.update(parent='0' * 40)),
            ('native', lambda v: v['native_modules_sha256'].update({'pycbc/native0.so': 'b' * 64})),
            ('diff', lambda v: v.update(exact_git_diff=v['exact_git_diff'] + '\ndiff --git a/setup.py b/setup.py'))]:
            original = self.fixture.read('source-v6.json')
            with self.subTest(field=field):
                self.mutate('source-v6.json', change)
                with self.assertRaises(Exception):
                    self.fixture.evidence()
                self.fixture.save('source-v6.json', original)

    def test_unit_tests_reject_stale_failed_or_incomplete_suite(self):
        for change in (lambda v: v['source_info'].update(commit=REPORT.CAMP.PARENT),
                       lambda v: v.update(passed=False), lambda v: v['command'].pop()):
            original = self.fixture.read('unit-tests-v6.json')
            self.mutate('unit-tests-v6.json', change)
            evidence, source = self.fixture.evidence()
            with self.assertRaises(Exception):
                REPORT.unit_tests(evidence, self.fixture.config, source)
            self.fixture.save('unit-tests-v6.json', original)

    def test_incomplete_stage_and_unbound_output_rejected(self):
        name = 'campaign-v6-measurements.status.json'
        original = self.fixture.read(name)
        for change in (lambda v: v['completed'].pop(), lambda v: v['output_sha256'].pop(next(iter(v['output_sha256']))),
                       lambda v: v['normal_cpu_equivalence']['unchanged_sha256'].pop('bin/pycbc_inspiral')):
            self.mutate(name, change)
            evidence, source = self.fixture.evidence()
            with self.assertRaises(Exception):
                REPORT.read_stages(evidence, self.fixture.config, source)
            self.fixture.save(name, original)

    def test_fft_rejects_stale_incomplete_imprecise_slow_or_multithreaded_matrix(self):
        name = 'large-ifft-v6.json'
        original = self.fixture.read(name)
        changes = [lambda v: v['source_before'].update(head=REPORT.CAMP.PARENT),
            lambda v: v['sizes'][0]['cases'].pop(), lambda v: v.update(threads=4),
            lambda v: v['sizes'][0]['cases'][0]['gates']['mkl_double_workspace'].update(bitwise_mkl_parity=False),
            lambda v: v['sizes'][0]['cases'][0]['errors']['mkl_double_workspace'].update(l2=1e-6),
            lambda v: v['sizes'][0]['timings'].update(mkl_double_workspace=
                copy.deepcopy(v['sizes'][0]['timings']['fftw_double_workspace_one_native_thread']))]
        for change in changes:
            self.mutate(name, change)
            self.fixture.seal_fft()
            evidence, source = self.fixture.evidence()
            with self.assertRaises(Exception):
                REPORT.large_ifft(evidence, self.fixture.config, source)
            self.fixture.save(name, original)

    def test_science_rechecks_raw_numerical_results(self):
        mutations = [('compressed-bank-v6.json', lambda v: v['cases'][0].update(bitwise_equal=False)),
            ('compressed-bank-v6.json', lambda v: v['cases'].__setitem__(0, copy.deepcopy(v['cases'][1]))),
            ('waveform-validation-v6.json', lambda v: v['cases'][0]['templates'][0]['psd_results'][0]['metrics']
                .update(relative_snr_norm_error=.02)),
            ('boundary-injections-v6.json', lambda v: v['templates'][0]['cases'][0]
                .update(peak_location_error_samples=2))]
        for name, change in mutations:
            original = self.fixture.read(name)
            with self.subTest(name=name):
                self.mutate(name, change)
                self.fixture.seal_science()
                evidence, _ = self.fixture.evidence()
                with self.assertRaises(Exception):
                    REPORT.science(evidence, set(range(96)), self.fixture.config)
                self.fixture.save(name, original)

    def test_scientific_runner_plan_and_output_must_be_complete(self):
        name = 'scientific-checks-v6.status.json'
        original = self.fixture.read(name)
        for change in (lambda v: v['completed'].pop(), lambda v: v['all_created_output_sha256'].clear()):
            self.mutate(name, change)
            value = self.fixture.read('scientific-validation-v6.json')
            value['input_sha256'][str(REMOTE / name)] = REPORT.sha(self.fixture.root / name)
            value['input_sha256_after'] = value['input_sha256']
            self.fixture.save('scientific-validation-v6.json', value)
            evidence, _ = self.fixture.evidence()
            with self.assertRaises(Exception):
                REPORT.science(evidence, set(range(96)), self.fixture.config)
            self.fixture.save(name, original)

    def test_strict_trigger_phase_failure_survives_resealed_receipts(self):
        case = 'matched-optimized6-torch-cpu-l512-r1'
        path = self.fixture.root / 'runs' / case / 'triggers.hdf'
        with REPORT.V4.h5py.File(path, 'r+') as handle:
            handle['H1/coa_phase'][0] = .5
        self.mutate(f'runs/{case}/receipt.json', lambda v: v.update(trigger_sha256=REPORT.sha(path)))
        self.fixture.stage('measurements')
        with self.assertRaisesRegex(Exception, 'Strict trigger parity failed'):
            self.fixture.build()

    def test_profile_denominator_rechecked_after_resealing(self):
        self.mutate('profiles-v6.json', lambda v: v['runs'][0]['cprofile'].update(total_profile_self_seconds=4.))
        self.fixture.stage('measurements')
        with self.assertRaisesRegex(Exception, 'self times do not reconcile'):
            self.fixture.build()

    def test_qualification_work_and_historical_report_are_verified(self):
        name = 'runs/qual-selected6-torch-cpu-l512/qualification.json'
        self.mutate(name, lambda v: v['observations'].update(decompression_success_count=95))
        self.fixture.stage('qualifications')
        self.fixture.seal_science()
        self.fixture.stage('measurements')
        with self.assertRaisesRegex(Exception, 'incomplete work'):
            self.fixture.build()
        evidence, _ = self.fixture.evidence()
        self.mutate('final-report/report.json', lambda v: v.update(status='fail'))
        with self.assertRaisesRegex(Exception, 'Hash mismatch'):
            REPORT.before_after(evidence, [])

    def test_unrecognized_external_input_and_source_hash_rejected(self):
        evidence, _ = self.fixture.evidence()
        for path, digest in [('/unknown/input', HASH), (str(SOURCE / REPORT.CAMP.ALLOWED[0]), 'b' * 64)]:
            with self.assertRaises(Exception):
                evidence.recorded_inputs({path: digest})

    def test_frozen_lal_dependencies_are_bound_without_archive_bytes(self):
        evidence, _ = self.fixture.evidence()
        evidence.recorded_inputs(REPORT.EXTERNAL_INPUTS, REPORT.EXTERNAL_INPUTS)
        self.assertEqual(evidence.archived_remote_inputs, REPORT.EXTERNAL_INPUTS)
        path = next(iter(REPORT.EXTERNAL_INPUTS))
        with self.assertRaisesRegex(Exception, 'Unrecognized external input'):
            evidence.recorded_inputs({path: 'b' * 64})
        with self.assertRaisesRegex(Exception, 'changed recorded inputs'):
            evidence.recorded_inputs(REPORT.EXTERNAL_INPUTS, {path: 'b' * 64})

    def test_historical_inputs_do_not_alias_optimized_inputs(self):
        evidence, _ = self.fixture.evidence()
        old = 'waveform-validation-precision5.json'
        new = 'waveform-validation-v6.json'
        self.fixture.save(old, {'historical': True})
        historical_hash = REPORT.sha(self.fixture.root / old)
        optimized_hash = REPORT.sha(self.fixture.root / new)
        self.assertNotEqual(historical_hash, optimized_hash)
        evidence.recorded_inputs({str(REMOTE / old): historical_hash})
        self.assertEqual(evidence.bind(old), (self.fixture.root / old).resolve())
        evidence.science_aliases = True
        self.assertEqual(evidence.bind(old), (self.fixture.root / new).resolve())
        evidence.science_aliases = False
        evidence.recorded_inputs({str(REMOTE / old): historical_hash})

    def test_failure_writes_no_success_artifacts(self):
        out = Path(self.temp.name) / 'failed-output'
        with patch.object(REPORT.CAMP, 'PINNED', self.fixture.pins):
            self.assertEqual(REPORT.main(['--root', str(self.fixture.root), '--source-manifest-sha256',
                '0' * 64, '--output-dir', str(out)]), 1)
        self.assertEqual([path.name for path in out.iterdir()], ['report.json'])
        self.assertEqual(json.loads((out / 'report.json').read_text())['status'], 'fail')


if __name__ == '__main__':
    unittest.main()
