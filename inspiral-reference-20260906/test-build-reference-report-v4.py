#!/usr/bin/env python3
"""Isolated synthetic evidence tests; never writes into the campaign evidence."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('reference_report', HERE / 'build-reference-report-v4.py')
REPORT = importlib.util.module_from_spec(SPEC)
exec(compile(SPEC.loader.get_data(SPEC.origin), SPEC.origin, 'exec'), REPORT.__dict__)
REMOTE = Path('/fixture/campaign')
SOURCE = REMOTE / 'source-v5'
HASH = 'a' * 64


class Fixture:
    def __init__(self, root, selected=512):
        self.root, self.selected = root, selected
        root.mkdir()
        for name in ('build-reference-report-v4.py', 'compare-triggers.py', 'summarize-runs.py'):
            shutil.copyfile(HERE / name, root / name)
        for name in ('setup-source-v5.py', 'run-unit-checks-v5.py', 'run-case.py',
                     'run-case-profiling.py', 'qualify-inspiral.py',
                     'profile-inspiral-torch.py', 'summarize-profiles.py'):
            self.write(name, b'# synthetic evidence fixture\n')
        self.config = dict(core=8, environment=dict(OMP_NUM_THREADS='1'))
        self.save('config.json', self.config)
        self.environment = dict(self.config['environment'], PYTHONPATH=str(SOURCE),
                                PYTHONDONTWRITEBYTECODE='1')
        self.save('environment.json', dict(hostname='len', affinity=[8], cpu='Fixture CPU',
                                         gpu='Fixture GPU', versions={}))
        self.write('inputs/bank-compressed-1e5.hdf', b'')
        with h5py.File(root / 'inputs/bank-compressed-1e5.hdf', 'w') as handle:
            handle['template_hash'] = np.arange(96, dtype=np.int64)
        self.bank_sha = REPORT.sha(root / 'inputs/bank-compressed-1e5.hdf')
        self.source()
        self.unit()
        self.save('compression-1e5.json', dict(
            state='complete', returncode=0,
            output_sha256={str(REMOTE / 'inputs/bank-compressed-1e5.hdf'): self.bank_sha},
            input_sha256=self.manifest('config.json'),
            input_sha256_after=self.manifest('config.json')))
        self.save('waveform-validation.json', {'passed': False})
        self.save('compression-refinement-decision.json', dict(
            failed_receipt_sha256=REPORT.sha(root / 'waveform-validation.json'),
            failed_pairs=[{}], decision='Synthetic rejected earlier bank'))
        self.cases, self.receipts = {}, {}
        self.tuning = {}
        for name, cases in REPORT.plan_cases(selected).items():
            plan = []
            for case, (mode, scheme, length) in cases.items():
                plan.append(['--case', case, '--mode', mode, '--scheme', scheme,
                             '--segment-length', str(length), '--start-pad', '112', '--end-pad', '16'])
                self.cases[case] = (mode, scheme, length)
                wall = (10 if length == selected else 20) + int(case[-1]) if name == REPORT.PLANS[1] else 15
                if name == REPORT.PLANS[1]:
                    self.tuning[case] = wall
                self.run(case, mode, scheme, length, wall)
            self.save(name + '.json', plan)
            self.save(name + '.status.json', dict(state='complete', returncode=0, completed=plan,
                current=None, finished_utc='2026-09-06T00:00:00Z',
                plan_sha256=REPORT.sha(root / (name + '.json'))))
        self.science()
        self.profiles()
        names = [name + suffix for name in REPORT.PLANS[:2] for suffix in ('.json', '.status.json')]
        names += [f'runs/{case}/{name}' for case in self.tuning for name in ('receipt.json', 'triggers.hdf')]
        grid = []
        for length in (256, 512, 1024):
            values = [self.tuning[f'tune-precision5-cpu-l{length}-r{rep}'] for rep in (1, 2, 3)]
            grid.append(dict(segment_length_seconds=length, wall_seconds=values, median_wall_seconds=values[1]))
        self.save('precision5-reference-tuning-decision.json', dict(schema_version=1,
            source_commit=REPORT.COMMIT, selected_segment_length_seconds=selected,
            selected_start_pad_seconds=112, selected_end_pad_seconds=16,
            basis='Lowest observed median; shorter length resolves exact ties.',
            selection_grid=grid, input_sha256=self.manifest(*names)))

    def write(self, name, contents):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)

    def save(self, name, value):
        self.write(name, (json.dumps(value, indent=2) + '\n').encode())

    def read(self, name):
        return json.loads((self.root / name).read_text())

    def manifest(self, *names):
        return {str(REMOTE / name): REPORT.sha(self.root / name) for name in names}

    def source(self):
        native = {f'pycbc/native{i}.so': HASH for i in range(11)}
        for version, commit, parent in ((2, REPORT.PREVIOUS_COMMIT, '0' * 40),
                (3, REPORT.PRECISION_COMMIT, REPORT.PREVIOUS_COMMIT),
                (4, REPORT.PARENT_COMMIT, REPORT.PRECISION_COMMIT),
                (5, REPORT.COMMIT, REPORT.PARENT_COMMIT)):
            paths = REPORT.CHANGED_PATHS if version == 5 else REPORT.PRECISION_CHANGED_PATHS
            changed = dict.fromkeys(paths, HASH)
            if version >= 4:
                changed['test/test_sigmasq_series_precision.py'] = 'b' * 64
            if version == 5:
                changed['test/test_chisq_precision.py'] = 'c' * 64
                changed['pycbc/vetoes/chisq_torch.py'] = 'd' * 64
            bundle = f'inspiral-source-v{version}.bundle'
            self.write(bundle, f'synthetic source {version}'.encode())
            self.save(f'source-v{version}.json', dict(source=str(REMOTE / f'source-v{version}'),
                commit=commit, parent=parent, parent_source_commit=parent, reference_base=REPORT.PREVIOUS_COMMIT,
                changed_paths=list(paths), changed_files_sha256=changed,
                native_modules_sha256=native, bundle_sha256=REPORT.sha(self.root / bundle)))
        names = ['setup-source-v5.py'] + [f'{prefix}-v{version}.{suffix}'
            for version in (2, 3, 4, 5) for prefix, suffix in (('source', 'json'), ('inspiral-source', 'bundle'))]
        parent_diff = '\n'.join([
            'diff --git a/pycbc/vetoes/chisq_torch.py b/pycbc/vetoes/chisq_torch.py',
            '--- a/pycbc/vetoes/chisq_torch.py', '+++ b/pycbc/vetoes/chisq_torch.py',
            '@@ -516,3 +516,4 @@ def _accelerator_batched_bin_sums(corr, pts, bins):',
            '             out_im,', '-            two_pi_over_N,',
            '+            # Preserve the phase coefficient precision.',
            '+            tl.constexpr(two_pi_over_N),', '             out_re.stride(0),',
            'diff --git a/test/test_chisq_precision.py b/test/test_chisq_precision.py',
            '--- a/test/test_chisq_precision.py', '+++ b/test/test_chisq_precision.py',
            '@@ -1 +1 @@', '-# Previous regression', '+# Extended regression',
        ])
        self.save('source-precision-provenance-v5.json', dict(schema_version=1, status='pass',
            old_source_commit=REPORT.PREVIOUS_COMMIT, new_source_commit=REPORT.COMMIT,
            parent_source_commit=REPORT.PARENT_COMMIT, reference_base=REPORT.PREVIOUS_COMMIT,
            changed_paths=list(REPORT.CHANGED_PATHS),
            parent_changed_paths=list(REPORT.PARENT_CHANGED_PATHS), exact_parent_git_diff=parent_diff,
            normal_cpu_outputs_changed=True, prior_science_reused=False,
            checks=dict.fromkeys(('only_reviewed_python_and_test_paths_changed', 'native_hashes_equal',
                'both_sources_clean', 'all_final_runs_require_new_source', 'host_launch_and_regression_only_parent_delta'), True),
            scope='Synthetic fixture; no performance or science claim.',
            exact_git_diff='\n'.join(f'diff --git a/{name} b/{name}' for name in REPORT.CHANGED_PATHS),
            input_sha256=self.manifest(*names)))

    def unit(self):
        self.write('unit-tests-v5.log', b'100 passed, 2 skipped in 1.00s\n')
        inputs = self.manifest('run-unit-checks-v5.py', 'config.json', 'source-v5.json')
        inputs.update({str(SOURCE / name): HASH for name in
                      ('bin/pycbc_inspiral', 'pycbc/scheme.py', *REPORT.CHANGED_PATHS, *REPORT.UNIT_TESTS)})
        inputs.update({str(SOURCE / name): digest for name, digest in
                       self.read('source-v5.json')['changed_files_sha256'].items()})
        self.save('unit-tests-v5.json', dict(state='complete', passed=True, returncode=0,
            finished_utc='2026-09-06T00:00:00Z', source_info=dict(commit=REPORT.COMMIT, status=''),
            source_after=dict(commit=REPORT.COMMIT, status=''), host='len', cwd=str(SOURCE),
            environment=self.environment, command=['taskset', '-c', '8', '/fixture/python',
            '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *REPORT.UNIT_TESTS],
            input_sha256=inputs, input_sha256_after=inputs,
            log_sha256=REPORT.sha(self.root / 'unit-tests-v5.log'), wall_seconds=1))

    def run(self, case, mode, scheme, length, wall):
        prefix = f'runs/{case}'
        self.write(prefix + '/triggers.hdf', b'')
        with h5py.File(self.root / prefix / 'triggers.hdf', 'w') as handle:
            group = handle.create_group('H1')
            for name, values in dict(template_hash=np.array([0, 1], dtype=np.int64),
                end_time=[1187007500., 1187007600.], snr=[8., 9.], chisq=[15., 17.],
                chisq_dof=[16., 16.], sigmasq=[100., 200.], coa_phase=[.1, .2]).items():
                group[name] = values
            for name, value in dict(start_time=1187007160, end_time=1187009064,
                                     run_time=5, setup_time_fraction=.2).items():
                group['search/' + name] = [value]
        cli = [str(SOURCE / 'bin/pycbc_inspiral'), '--sample-rate', '4096',
               '--channel-name', 'H1:STRAIN', '--trig-start-time', '1187007160',
               '--trig-end-time', '1187009064', '--snr-threshold', '5.5',
               '--bank-file', str(REMOTE / 'inputs/bank-compressed-1e5.hdf'),
               '--processing-scheme', scheme, '--segment-length', str(length),
               '--segment-start-pad', '112', '--segment-end-pad', '16',
               '--output', str(REMOTE / prefix / 'triggers.hdf')]
        runner = 'run-case.py' if mode in ('timing', 'qualify') else 'run-case-profiling.py'
        names = ['config.json', 'inputs/bank-compressed-1e5.hdf', runner]
        if mode == 'qualify':
            names.append('qualify-inspiral.py')
        if mode == 'torchprofile':
            names.append('profile-inspiral-torch.py')
        inputs = self.manifest(*names)
        inputs[str(SOURCE / 'bin/pycbc_inspiral')] = HASH
        receipt = dict(case=case, mode=mode, scheme=scheme, segment_length=length,
            start_pad=112, end_pad=16, state='complete', returncode=0,
            source_info=dict(commit=REPORT.COMMIT, status='', tracked_diff=''), source_status_after='',
            source=str(SOURCE), hostname='len', cwd=str(REMOTE), environment=self.environment,
            command=['taskset', '-c', '8', '/fixture/python', *cli], executable_cli=cli,
            input_sha256=inputs, input_sha256_after=inputs,
            trigger_sha256=REPORT.sha(self.root / prefix / 'triggers.hdf'),
            elapsed_wall_seconds=wall, peak_child_rss_kib=1024,
            child_user_cpu_seconds=4, child_system_cpu_seconds=1)
        self.save(prefix + '/receipt.json', receipt)
        self.receipts[case] = receipt
        if mode == 'qualify':
            self.qualification(case, receipt)

    def qualification(self, case, receipt):
        scheme, length = receipt['scheme'], receipt['segment_length']
        segments = {256: 15, 512: 5, 1024: 3}[length]
        runtime = dict(affinity=[8], scheme={'class': 'pycbc.scheme.CPUScheme' if scheme == 'cpu:1'
                else 'pycbc.scheme.TorchScheme', 'device': {'cpu:1': 'None', 'torch:cpu:1': 'cpu',
                'torch:cuda:0': 'cuda:0'}[scheme], 'num_threads': None if scheme == 'torch:cuda:0' else 1},
                torch=dict(num_threads=1))
        self.save(f'runs/{case}/qualification.json', dict(status='success', executable_exit_code=0,
            checks=dict(complete=True), argv=receipt['executable_cli'], executed_argv=receipt['executable_cli'],
            host='len', source_root=str(SOURCE), executable=dict(path=str(SOURCE / 'bin/pycbc_inspiral'), sha256=HASH),
            wrapper=dict(path=str(REMOTE / 'qualify-inspiral.py'),
                         sha256=REPORT.sha(self.root / 'qualify-inspiral.py')),
            observations=dict(banks=[dict(file=dict(sha256=self.bank_sha),
                templates={str(i): dict(template_hash=i) for i in range(96)})],
                segment_geometry=[dict(segments=list(range(segments)), gap_samples=0, overlap_samples=0,
                    unique_analyzed_seconds=1904, fft_samples=length * 4096, sample_rate_hz=4096,
                    strain_start_time=1187007160, union_analyzed_sample_intervals=[[0, 1904 * 4096]])],
                fft_engines=[dict(execute_successes=96 * segments)], decompression_success_count=96,
                runtime=dict(inside_filter_context=runtime), psd_arrays=[])))

    def science(self):
        inputs = self.manifest('config.json', *[f'runs/qual-precision5-cpu-l{length}/qualification.json'
                                               for length in (256, 512, 1024)])
        common = dict(state='complete', passed=True, source_before=dict(commit=REPORT.COMMIT, status=''),
            source_after=dict(commit=REPORT.COMMIT, status=''), input_sha256=inputs,
            input_sha256_after=inputs)
        cases = [dict(qualification=str(REMOTE / f'runs/qual-precision5-cpu-l{length}/qualification.json'),
            passed=True, state='complete', templates=[dict(passed=True, template_hash=i,
                decompression_succeeded_without_generation=True,
                psd_results=[dict(passed=True, metrics=dict.fromkeys(REPORT.WAVEFORM_TOLERANCES, 0.))])
                for i in range(96)]) for length in (256, 512, 1024)]
        self.save('waveform-validation-precision5.json', dict(common, cases=cases,
            expected_template_psd_pairs=288, completed_template_psd_pairs=288,
            tolerances=REPORT.WAVEFORM_TOLERANCES))
        boundary_cases = [dict(passed=True, segment_length_seconds=length, start_pad_seconds=pad,
            end_pad_seconds=16, placement=placement, maximum_relative_complex_error=0.,
            peak_location_error_samples=0, relative_peak_snr_error=0.) for length in (256, 512, 1024)
            for pad in (96, 112) for placement in ('first_valid', 'midpoint', 'last_valid')]
        templates = [dict(kind=kind, passed=True, reference_edge_budget_passed=True,
            reference=dict(template_hash=i, tails=dict(injection_edge_energy_fraction=0.,
            kernel_edge_energy_fraction=0.)), cases=boundary_cases) for i, kind in enumerate(('shortest', 'longest'))]
        self.save('boundary-injections-precision5.json', dict(common, templates=templates,
            expected_cases=36, completed_cases=36, tolerances=REPORT.BOUNDARY_TOLERANCES))
        boundary = self.read('boundary-injections-precision5.json')
        boundary['qualification'] = str(REMOTE / 'runs/qual-precision5-cpu-l1024/qualification.json')
        self.save('boundary-injections-precision5.json', boundary)

    def profiles(self):
        runs = []
        for case, (mode, scheme, length) in self.cases.items():
            prefix = f'runs/{case}'
            if mode in ('perf', 'cprofile'):
                row = dict(run=dict(case=case), receipt=dict(sha256=REPORT.sha(self.root / prefix / 'receipt.json')))
                if mode == 'cprofile':
                    self.write(prefix + '/profile.pstats', b'synthetic pstats')
                    row['cprofile'] = dict(input=dict(sha256=REPORT.sha(self.root / prefix / 'profile.pstats')),
                        total_profile_self_seconds=2., exclusive_groups_reconcile=True, accounting='Fixture',
                        groups=[dict(group='filter', self_seconds=2., share_of_profile_self_percent=100.)])
                else:
                    self.write(prefix + '/perf.data', b'synthetic perf data')
                    self.write(prefix + '/perf-report.txt', b'synthetic perf report')
                    row['perf'] = dict(input=dict(sha256=REPORT.sha(self.root / prefix / 'perf-report.txt')),
                        event='cycles:u', denominator_note='Full observed period denominator.', native_functions=[
                            dict(symbol='fixture_filter', shared_object='fixture.so', self_event_percent=50.,
                                 classification='matched filtering')])
                runs.append(row)
            if mode == 'torchprofile':
                prefix += '/torch-profile'
                self.write(prefix + '/trace.json', b'{}')
                self.save(prefix + '/key-averages.json', dict(units='microseconds', note='Fixture',
                    cpu_event_self_time_us=3., cuda_device_event_self_time_us=2., rows=[
                        dict(key='fixture_cpu', device_type='DeviceType.CPU', self_cpu_time_us=3.),
                        dict(key='fixture_cuda', device_type='DeviceType.CUDA', self_device_time_us=2.)]))
                receipt = self.receipts[case]
                self.save(prefix + '/receipt.json', dict(status='success', wrapper_exit_code=0,
                    executable_exit_code=0, argv=receipt['executable_cli'], executed_argv=receipt['executable_cli'],
                    host='len', source_root=str(SOURCE),
                    executable=dict(path=str(SOURCE / 'bin/pycbc_inspiral'), sha256=HASH),
                    wrapper=dict(path=str(REMOTE / 'profile-inspiral-torch.py'),
                                 sha256=REPORT.sha(self.root / 'profile-inspiral-torch.py')),
                    trace=dict(sha256=REPORT.sha(self.root / prefix / 'trace.json')),
                    key_averages=dict(sha256=REPORT.sha(self.root / prefix / 'key-averages.json'))))
        self.save('profiles-precision5.json', dict(schema_version=1, runs=runs,
            summarizer=dict(sha256=REPORT.sha(self.root / 'summarize-profiles.py'))))

    def build(self):
        with patch.object(REPORT, 'BANK_SHA', self.bank_sha):
            return REPORT.build(REPORT.Evidence(self.root))


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='reference-report-fixture-')
        self.addCleanup(self.temp.cleanup)
        self.fixture = Fixture(Path(self.temp.name) / 'evidence')

    def test_all_selected_geometries_and_duplicate_cpu_qualification(self):
        for selected in (256, 512, 1024):
            with self.subTest(selected=selected):
                fixture = Fixture(Path(self.temp.name) / str(selected), selected)
                report = fixture.build()
                self.assertEqual(report['status'], 'pass')
                self.assertEqual(len(report['required_cases']), 31)
                self.assertEqual(len(report['parity']['comparisons']), 28)
                self.assertEqual(report['source_phases'], {'precision': REPORT.COMMIT})
                self.assertEqual(len(report['tuning']), 3)
                self.assertEqual(len(report['matched']), 3)
                for row in report['rows']:
                    if row['mode'] == 'qualify':
                        self.assertEqual(row['qualification'], f"runs/{row['case']}/qualification.json")
                    elif row['scheme'] == 'cpu:1' and row['segment_length'] == selected:
                        self.assertEqual(row['qualification'], f'runs/qual-selected5-cpu-l{selected}/qualification.json')

    def test_six_figures_and_four_csv_contract(self):
        report = self.fixture.build()
        out = Path(self.temp.name) / 'output'
        with patch.object(REPORT, 'BANK_SHA', self.fixture.bank_sha), patch('sys.argv',
                ['builder', '--root', str(self.fixture.root), '--output-dir', str(out)]):
            self.assertEqual(REPORT.main(), 0)
        expected = set(REPORT.FIGURES) | {'runs.csv', 'groups.csv', 'profiles.csv', 'native-symbols.csv'}
        self.assertEqual({path.name for path in out.iterdir()}, expected | {'report.json'})
        saved = json.loads((out / 'report.json').read_text())
        self.assertEqual(set(saved['output_sha256']), expected)
        self.assertEqual(saved['required_cases'], report['required_cases'])

    def reject_mutation(self, name, mutate, pattern=None):
        original = (self.fixture.root / name).read_bytes()
        try:
            value = self.fixture.read(name)
            mutate(value)
            self.fixture.save(name, value)
            with self.assertRaises((REPORT.InvalidEvidence, KeyError, FileNotFoundError)) as raised:
                self.fixture.build()
            if pattern:
                self.assertIn(pattern, str(raised.exception))
        finally:
            (self.fixture.root / name).write_bytes(original)

    def test_bad_source_provenance(self):
        for field, value in [('commit', REPORT.PREVIOUS_COMMIT), ('parent', REPORT.PREVIOUS_COMMIT),
                             ('changed_paths', []), ('native_modules_sha256', {})]:
            with self.subTest(field=field):
                self.reject_mutation('source-v5.json', lambda d: d.update({field: value}))
        for field, value in [('prior_science_reused', True), ('normal_cpu_outputs_changed', False),
                             ('exact_git_diff', ''), ('checks', {})]:
            with self.subTest(field=field):
                self.reject_mutation('source-precision-provenance-v5.json', lambda d: d.update({field: value}))

    def test_source_chain_and_reviewed_hash_deltas(self):
        for name, field, value, message in (
                ('source-v3.json', 'commit', REPORT.PARENT_COMMIT, 'precision source ancestor'),
                ('source-v4.json', 'parent', REPORT.PREVIOUS_COMMIT, 'precision source parent'),
                ('source-v5.json', 'parent_source_commit', REPORT.PRECISION_COMMIT, 'final source revision')):
            with self.subTest(name=name, field=field):
                self.reject_mutation(name, lambda d: d.update({field: value}), message)
        for version, message in ((4, 'reviewed oracle correction'), (5, 'reviewed launch and regression')):
            with self.subTest(version=version):
                self.reject_mutation(f'source-v{version}.json',
                    lambda d: d['changed_files_sha256'].update({'pycbc/filter/matchedfilter.py': 'e' * 64}),
                    message)

    def test_parent_proof_and_all_archived_inputs_required(self):
        name = 'source-precision-provenance-v5.json'
        self.reject_mutation(name, lambda d: d.update(parent_changed_paths=list(REPORT.CHANGED_PATHS)),
                             'incorrect precision source provenance')

        def old_check(d):
            d['checks']['tests_only_parent_delta'] = d['checks'].pop('host_launch_and_regression_only_parent_delta')

        self.reject_mutation(name, old_check, 'incorrect precision source provenance')
        for version in (2, 3, 4, 5):
            for entry in (f'source-v{version}.json', f'inspiral-source-v{version}.bundle'):
                with self.subTest(entry=entry):
                    self.reject_mutation(name, lambda d: d['input_sha256'].pop(str(REMOTE / entry)),
                                         'Missing required input hash')

    def test_parent_runtime_diff_rejects_unreviewed_changes(self):
        name = 'source-precision-provenance-v5.json'
        original = self.fixture.read(name)['exact_parent_git_diff']
        for variant in (
                original.replace('+            tl.constexpr(two_pi_over_N),', '+            two_pi_over_N,'),
                original.replace('+            tl.constexpr(two_pi_over_N),', '+            float(two_pi_over_N),'),
                original.replace('def _accelerator_batched_bin_sums(corr, pts, bins):', 'def _kernel(corr, pts, bins):'),
                original.replace('diff --git a/test/test_chisq_precision.py',
                                 '@@ -12 +12 @@ def _kernel():\n-x = 0\n+x = 1\ndiff --git a/test/test_chisq_precision.py'),
                original.replace('+            tl.constexpr(two_pi_over_N),',
                                 '+            tl.constexpr(two_pi_over_N),\n+            altered = True'),
        ):
            with self.subTest(diff=variant):
                self.reject_mutation(name, lambda d: d.update(exact_parent_git_diff=variant),
                                     'not the reviewed host launch change')

    def test_previous_final_source_cannot_supply_runs_or_science(self):
        name = 'runs/qual-selected5-torch-cuda-l512/receipt.json'
        self.reject_mutation(name, lambda d: d['source_info'].update(commit=REPORT.PARENT_COMMIT),
                             'incomplete or changed source')
        for name in ('waveform-validation-precision5.json', 'boundary-injections-precision5.json'):
            with self.subTest(name=name):
                self.reject_mutation(name, lambda d: d.update(
                    source_before=dict(commit=REPORT.PARENT_COMMIT, status=''),
                    source_after=dict(commit=REPORT.PARENT_COMMIT, status='')), 'source changed')
        self.reject_mutation('waveform-validation-precision5.json', lambda d: d['cases'][0].update(
            qualification=str(REMOTE / 'runs/qual-precision-cpu-l256/qualification.json')),
            'every final CPU qualification')

    def test_plans_reject_previous_final_case_names(self):
        name = REPORT.PLANS[0]
        plan = self.fixture.read(name + '.json')
        plan[0][1] = 'qual-precision-cpu-l256'
        self.fixture.save(name + '.json', plan)
        status = self.fixture.read(name + '.status.json')
        status.update(completed=plan, plan_sha256=REPORT.sha(self.fixture.root / (name + '.json')))
        self.fixture.save(name + '.status.json', status)
        comparator = REPORT.helper(self.fixture.root, 'compare-triggers.py')
        with self.assertRaisesRegex(REPORT.InvalidEvidence, 'Unexpected final case names'):
            REPORT.read_plans(REPORT.Evidence(self.fixture.root), comparator, self.fixture.selected)

    def test_failed_stale_or_incomplete_unit_tests(self):
        for field, value in [('passed', False), ('returncode', 1), ('state', 'running'),
                             ('source_after', dict(commit=REPORT.PREVIOUS_COMMIT, status='')),
                             ('log_sha256', 'b' * 64), ('command', ['taskset', '-c', '8', '/python'])]:
            with self.subTest(field=field):
                self.reject_mutation('unit-tests-v5.json', lambda d: d.update({field: value}))

    def test_invalid_decision_and_wrong_winner(self):
        for field, value in [('selected_segment_length_seconds', 100),
                             ('selected_segment_length_seconds', True), ('selected_start_pad_seconds', 96),
                             ('source_commit', REPORT.PREVIOUS_COMMIT), ('basis', '')]:
            with self.subTest(field=field, value=value):
                self.reject_mutation('precision5-reference-tuning-decision.json', lambda d: d.update({field: value}))
        self.reject_mutation('precision5-reference-tuning-decision.json',
            lambda d: d['selection_grid'][0].update(median_wall_seconds=1), 'median differ')
        self.reject_mutation('precision5-reference-tuning-decision.json',
            lambda d: d['input_sha256'].pop(str(REMOTE / 'runs/tune-precision5-cpu-l256-r1/receipt.json')),
            'Missing required input hash')

    def test_failed_science_and_frozen_budgets(self):
        for name in ('waveform-validation-precision5.json', 'boundary-injections-precision5.json'):
            with self.subTest(name=name):
                self.reject_mutation(name, lambda d: d.update(passed=False))
                self.reject_mutation(name, lambda d: d.update(source_before=dict(commit=REPORT.PREVIOUS_COMMIT, status='')))
                self.reject_mutation(name, lambda d: d['tolerances'].update({next(iter(d['tolerances'])): 1}))
        self.reject_mutation('waveform-validation-precision5.json',
            lambda d: d['cases'][0]['templates'][0]['psd_results'][0]['metrics'].update(relative_snr_norm_error=.02),
            'exceeds its frozen')
        self.reject_mutation('waveform-validation-precision5.json',
            lambda d: d['cases'][0]['templates'][0].update(template_hash=1), 'template identities')
        self.reject_mutation('boundary-injections-precision5.json',
            lambda d: d['templates'][0]['cases'][0].update(maximum_relative_complex_error=.01), 'exceeds its frozen')
        self.reject_mutation('boundary-injections-precision5.json',
            lambda d: d['tolerances'].update(peak_location_error_samples=True), 'tolerances changed')

    def test_qualification_failure_and_duplicate_coverage(self):
        name = 'runs/qual-selected5-cpu-l512/qualification.json'
        self.reject_mutation(name, lambda d: d['checks'].update(complete=False), 'qualification failed')
        self.reject_mutation(name, lambda d: d['observations']['segment_geometry'][0]['segments'].__setitem__(0, -1),
                             'sample coverage differs')
        self.reject_mutation(name, lambda d: d['observations'].update(decompression_success_count=95), 'incomplete work')

    def test_missing_run_incomplete_ledger_and_profile(self):
        self.reject_mutation(REPORT.PLANS[5] + '.status.json', lambda d: d.update(state='running'), 'Incomplete')
        self.reject_mutation(REPORT.PLANS[5] + '.status.json', lambda d: d.update(plan_sha256='b' * 64), 'unbound')
        self.reject_mutation('profiles-precision5.json', lambda d: d['runs'].pop(), 'exactly the six')
        self.reject_mutation('profiles-precision5.json',
            lambda d: next(r['cprofile'] for r in d['runs'] if 'cprofile' in r).update(total_profile_self_seconds=3.),
            'do not reconcile')
        name = 'runs/profile-precision5-torch-cuda-l512-torchprofile/receipt.json'
        self.reject_mutation(name, lambda d: d.update(source_status_after=' M changed.py'), 'changed source')
        path = self.fixture.root / name
        saved = path.read_bytes()
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.fixture.build()
        path.write_bytes(saved)

    def test_winner_recomputed_from_timings(self):
        report = self.fixture.build()
        decision = self.fixture.read('precision5-reference-tuning-decision.json')
        evidence = REPORT.Evidence(self.fixture.root)
        evidence.remote = REMOTE
        tuning = [row for row in report['rows'] if row['case'] in self.fixture.tuning]
        matched = [row for row in report['rows'] if row['case'].startswith('matched-')]
        with self.assertRaisesRegex(REPORT.InvalidEvidence, 'not the lowest median wall time'):
            REPORT.validate_decision(evidence, decision, (256, 112, 16), tuning, matched)

    def test_trigger_mismatch_is_fail_closed(self):
        case = 'qual-selected5-cpu-l512'
        path = self.fixture.root / f'runs/{case}/triggers.hdf'
        with h5py.File(path, 'r+') as handle:
            handle['H1/chisq'][0] += 1
        receipt = self.fixture.read(f'runs/{case}/receipt.json')
        receipt['trigger_sha256'] = REPORT.sha(path)
        self.fixture.save(f'runs/{case}/receipt.json', receipt)
        with self.assertRaisesRegex(REPORT.InvalidEvidence, 'Repeated CPU qualification trigger parity failed'):
            self.fixture.build()
        out = Path(self.temp.name) / 'failed-output'
        with patch.object(REPORT, 'BANK_SHA', self.fixture.bank_sha), patch('sys.argv',
                ['builder', '--root', str(self.fixture.root), '--output-dir', str(out)]):
            self.assertEqual(REPORT.main(), 1)
        self.assertEqual([p.name for p in out.iterdir()], ['report.json'])
        self.assertEqual(json.loads((out / 'report.json').read_text())['status'], 'fail')


if __name__ == '__main__':
    unittest.main()
