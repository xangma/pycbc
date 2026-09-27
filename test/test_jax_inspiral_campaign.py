"""Tests for the JAX multi-arm inspiral benchmark campaign runner."""

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import json
import h5py
import numpy as np

from tools.bench_jax_inspiral_campaign import (
    compare_trigger_parity,
    compare_campaign_parity,
    sample_summary,
    _parse_stderr_phases,
    ARM_NAMES,
    DEFAULT_ARMS,
    BATCHED_ARMS,
    _completed_case,
    _retry_output_dir,
    _run_single_case,
    _validate_benchmark_work,
    _require_pristine_original,
    _require_source_unchanged,
    BENCHMARK_SAMPLE_RATE,
    BENCHMARK_PRECISION,
    input_contract,
    resolve_waveform_mode,
    validate_bank_mode,
    _require_matching_contract,
    _run_profile_campaign,
    main,
)


class TestBenchJaxInspiralCampaign(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def make_bank(self, compressed=True):
        bank = self.path / 'bank.hdf'
        with h5py.File(bank, 'w') as out:
            out['mass1'] = [1.4, 1.6]
            out['mass2'] = [1.2, 1.2]
            out['template_hash'] = np.array([11, 22], dtype=np.uint64)
            out['approximant'] = np.array(['IMRPhenomD', 'IMRPhenomD'], dtype='S16')
            if compressed:
                for template_hash in (11, 22):
                    group = out.create_group('compressed_waveforms/' + str(template_hash))
                    group['sample_points'] = [20., 100., 1024.]
                    group['amplitude'] = [1., 0.5, 0.1]
                    group['phase'] = [0., 1., 2.]
                    group.attrs.update(interpolation='inline_linear', tolerance=0.001,
                                       mismatch=0.0001, precision='double', compression_factor=5.)
        return bank

    def make_contract(self, compressed=True, mode=None):
        bank = self.make_bank(compressed)
        frame = self.path / 'frame.gwf'
        frame.write_bytes(b'fixture frame')
        return input_contract(bank, frame, mode or ('compressed' if compressed else 'generated'),
                              'IMRPhenomD', -1, 'inline_linear')

    def test_waveform_mode_defaults_and_conflicting_aliases(self):
        self.assertEqual(resolve_waveform_mode(), 'compressed')
        self.assertEqual(resolve_waveform_mode(track='track2'), 'generated')
        self.assertEqual(resolve_waveform_mode(uncompressed=True), 'generated')
        self.assertEqual(resolve_waveform_mode('generated', True, 'track2'), 'generated')
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            resolve_waveform_mode('compressed', uncompressed=True)
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            resolve_waveform_mode('generated', track='track1')

    def test_parameter_bank_requires_explicit_generated_mode(self):
        bank = self.make_bank(False)
        with self.assertRaisesRegex(ValueError, 'select --waveform-mode generated'):
            validate_bank_mode(bank, resolve_waveform_mode(), 'IMRPhenomD')
        self.assertEqual(validate_bank_mode(bank, 'generated', 'IMRPhenomD')['templates'], 2)

    def test_compressed_bank_requires_every_row_and_loader_attribute(self):
        bank = self.make_bank()
        with h5py.File(bank, 'a') as out:
            del out['compressed_waveforms/22']
        with self.assertRaisesRegex(ValueError, 'Missing compressed waveform data for 22'):
            validate_bank_mode(bank, 'compressed', 'IMRPhenomD')
        bank = self.make_bank()
        with h5py.File(bank, 'a') as out:
            del out['compressed_waveforms/11'].attrs['precision']
        with self.assertRaisesRegex(ValueError, 'metadata for 11'):
            validate_bank_mode(bank, 'compressed', 'IMRPhenomD')

    def test_invalid_compressed_samples_are_rejected(self):
        for field, values, message in (
                ('phase', [0., float('nan'), 1.], 'arrays'),
                ('amplitude', [1., 2.], 'arrays'),
                ('sample_points', [20., 20., 1024.], 'frequency support'),
                ('sample_points', [40., 100., 1024.], 'frequency support')):
            with self.subTest(field=field, values=values):
                bank = self.make_bank()
                with h5py.File(bank, 'a') as out:
                    group = out['compressed_waveforms/11']
                    del group[field]
                    group[field] = values
                with self.assertRaisesRegex(ValueError, message):
                    validate_bank_mode(bank, 'compressed', 'IMRPhenomD')

    def test_compressed_approximant_mismatch_and_duplicate_rows_rejected(self):
        bank = self.make_bank()
        with self.assertRaisesRegex(ValueError, 'approximants do not match'):
            validate_bank_mode(bank, 'compressed', 'TaylorF2')
        with h5py.File(bank, 'a') as out:
            out['mass1'][1] = out['mass1'][0]
        with self.assertRaisesRegex(ValueError, 'Repeated template rows'):
            validate_bank_mode(bank, 'generated', 'IMRPhenomD')

    def test_contract_binds_mode_inputs_and_forbids_waveform_overrides(self):
        compressed = self.make_contract()
        generated = input_contract(self.path / 'bank.hdf', self.path / 'frame.gwf',
                                   'generated', 'IMRPhenomD', -1, 'inline_linear')
        self.assertEqual(generated['track'], 'track2_generated')
        self.assertIsNone(generated['decompression_method'])
        self.assertEqual(compressed['bank_sha256'], generated['bank_sha256'])
        with self.assertRaisesRegex(ValueError, 'Input contract mismatch'):
            _require_matching_contract(compressed, generated)
        with self.assertRaisesRegex(ValueError, 'invalid input contract'):
            _require_matching_contract(compressed, {**compressed, 'waveform_mode': 'generated'})
        (self.path / 'frame.gwf').write_bytes(b'different frame')
        changed = input_contract(self.path / 'bank.hdf', self.path / 'frame.gwf',
                                 'compressed', 'IMRPhenomD', -1, 'inline_linear')
        self.assertNotEqual(compressed['fingerprint'], changed['fingerprint'])
        for key in ('approximant', 'use_compressed_waveforms', 'sample_rate', 'order'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Use campaign options'):
                input_contract(self.path / 'bank.hdf', self.path / 'frame.gwf',
                               'compressed', 'IMRPhenomD', -1, 'inline_linear', {key: 'changed'})
        with self.assertRaisesRegex(ValueError, 'Duplicate search override'):
            input_contract(self.path / 'bank.hdf', self.path / 'frame.gwf',
                           'generated', 'IMRPhenomD', -1, 'inline_linear',
                           {'low_frequency_cutoff': 30, 'low-frequency-cutoff': 40})

    def fake_process(self, command, **kwargs):
        run_dir = Path(kwargs['cwd'])
        with h5py.File(run_dir / 'triggers.hdf', 'w') as out:
            out['H1/search/run_time'] = [2.]
            out['H1/search/setup_time_fraction'] = [0.25]
            out['H1/snr'] = np.array([8.], dtype=np.float32)
        (run_dir / 'work.json').write_text(json.dumps({
            'sample_rate': 2048, 'signal_dtypes': ['complex64'],
            'completed_templates': 2, 'completed_template_seconds': 2., 'valid_detector_seconds': 1.,
        }))
        return SimpleNamespace(returncode=0)

    def test_runner_preserves_compressed_cli_and_records_explicit_generation(self):
        contract = self.make_contract()
        for mode in ('compressed', 'generated'):
            with self.subTest(mode=mode), patch(
                    'tools.bench_jax_inspiral_campaign.subprocess.run', side_effect=self.fake_process), patch(
                    'tools.bench_jax_inspiral_campaign.physical_cores', return_value=1):
                result = _run_single_case(
                    'branch_cpu', self.path, 'python', self.path / mode,
                    self.path / 'frame.gwf', self.path / 'bank.hdf',
                    use_compressed_waveforms=mode == 'compressed')
                self.assertEqual('--use-compressed-waveforms' in result['command'], mode == 'compressed')
                self.assertEqual(result['input_contract']['waveform_mode'], mode)
                manifest = json.loads((self.path / mode / 'command.json').read_text())
                self.assertEqual(manifest['input_contract'], result['input_contract'])
                config = json.loads(manifest['environment']['PYCBC_BENCHMARK_SCIENCE_CONFIG'])
                self.assertEqual(config['input_contract'], result['input_contract'])
                if mode == 'compressed':
                    self.assertEqual(result['input_contract'], contract)

    def test_changed_bank_rejected_before_subprocess(self):
        contract = self.make_contract()
        with h5py.File(self.path / 'bank.hdf', 'a') as out:
            out['mass1'][0] = 1.5
        with patch('tools.bench_jax_inspiral_campaign.subprocess.run') as launch:
            with self.assertRaisesRegex(ValueError, 'Input contract mismatch'):
                _run_single_case('branch_cpu', self.path, 'python', self.path / 'run',
                                 self.path / 'frame.gwf', self.path / 'bank.hdf',
                                 expected_contract=contract)
            launch.assert_not_called()

    def test_original_cpu_runs_through_observer(self):
        self.make_contract()
        with patch('tools.bench_jax_inspiral_campaign.subprocess.run', side_effect=self.fake_process), patch(
                'tools.bench_jax_inspiral_campaign.physical_cores', return_value=1), patch(
                'tools.bench_jax_inspiral_campaign.validate_observer_source') as preflight:
            result = _run_single_case('original_cpu', self.path, 'python', self.path / 'run',
                                      self.path / 'frame.gwf', self.path / 'bank.hdf')
        executable = str((self.path / 'bin' / 'pycbc_inspiral').resolve())
        index = result['command'].index(executable)
        self.assertEqual(Path(result['command'][index - 1]).name, 'observe_pycbc_inspiral.py')
        preflight.assert_called_once_with(Path(executable))

    def test_failed_qualification_prevents_all_timing_and_track_changes_no_model(self):
        self.make_contract(False)
        argv = ['benchmark', '--original-source', str(self.path), '--reference-revision', 'a' * 40,
                '--branch-source', str(self.path), '--python', 'python', '--frame-file',
                str(self.path / 'frame.gwf'), '--bank-file', str(self.path / 'bank.hdf'),
                '--output', str(self.path / 'receipt.json'), '--output-dir', str(self.path / 'runs'),
                '--track', 'track2', '--approximant', 'IMRPhenomD', '--order', '-1',
                '--arms', 'original_cpu', 'jax_cuda_batched']

        def fake_case(*args, **kwargs):
            self.assertTrue(kwargs['qualification'])
            self.assertEqual(kwargs['approximant'], 'IMRPhenomD')
            self.assertEqual(kwargs['order'], -1)
            self.assertFalse(kwargs['use_compressed_waveforms'])
            return {'input_contract': kwargs['expected_contract'], 'sample_rate': 2048,
                    'triggers_path': str(self.path / 'triggers.hdf')}

        module = 'tools.bench_jax_inspiral_campaign.'
        with patch('sys.argv', argv), patch(module + '_run_single_case', side_effect=fake_case) as run, patch(
                module + '_source_records', return_value={'original': {'revision': 'a' * 40, 'dirty': False}}), patch(
                module + 'validate_reference'), patch(module + 'validate_observer_source'), patch(
                module + '_require_source_unchanged'), patch(module + '_git_commit', return_value='a' * 40), patch(
                module + 'compare_scientific_hdf', return_value={'passed': False}):
            with self.assertRaisesRegex(RuntimeError, 'no timing runs launched'):
                main()
        self.assertEqual(run.call_count, 2)
        receipt = json.loads((self.path / 'receipt.json').read_text())
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(receipt['failure']['phase'], 'qualification_science')
        self.assertTrue(all(not runs for runs in receipt['raw_results'].values()))

    def test_allow_unqualified_timings_keeps_failures_and_runs_all_repeats(self):
        self.make_contract(False)
        argv = ['benchmark', '--original-source', str(self.path),
                '--reference-revision', 'a' * 40,
                '--branch-source', str(self.path), '--python', 'python',
                '--frame-file', str(self.path / 'frame.gwf'), '--bank-file',
                str(self.path / 'bank.hdf'), '--output',
                str(self.path / 'receipt.json'), '--output-dir',
                str(self.path / 'runs'), '--arms', 'original_cpu',
                'jax_cuda_batched', '--waveform-mode', 'generated',
                '--allow-unqualified-timings']

        phases = {name: 0.1 for name in (
            'startup_import_sec', 'conditioning_sec', 'waveform_prep_sec',
            'matched_filter_sec', 'vetoes_clustering_sec',
            'serialization_io_sec')}

        def fake_case(*args, **kwargs):
            return {
                'input_contract': kwargs['expected_contract'],
                'sample_rate': 2048, 'triggers_path': str(self.path / 'triggers.hdf'),
                'evidence_path': str(self.path / 'science.hdf'),
                'elapsed_wall_sec': 1.0, 'calc_time_sec': 0.5,
                'tsetup_sec': 0.25, 'time_info': {'max_rss_kib': 1},
                'phases': phases, 'num_triggers': 1,
                'valid_detector_seconds': 1.0,
                'completed_template_seconds': 2.0,
                'resources': {'physical_cpu_cores': 1, 'gpus': 0},
            }

        science = {'passed': False, 'scope': 'full',
                   'missing_gates': ['retained chi-square']}
        records = {
            'original': {'revision': 'a' * 40, 'dirty': False},
            'branch': {'revision': 'b' * 40, 'dirty': True},
        }
        module = 'tools.bench_jax_inspiral_campaign.'
        with patch('sys.argv', argv), patch(
                module + '_run_single_case', side_effect=fake_case) as run, patch(
                module + '_source_records', return_value=records), patch(
                module + 'validate_reference'), patch(
                module + 'validate_observer_source'), patch(
                module + '_require_source_unchanged'), patch(
                module + '_git_commit', return_value='a' * 40), patch(
                module + 'compare_scientific_hdf', return_value=science), patch(
                module + 'compare_trigger_parity', return_value=science), patch(
                module + 'compare_campaign_parity', return_value=('original_cpu', {})):
            self.assertIsNone(main())
        self.assertEqual(run.call_count, 8)
        receipt = json.loads((self.path / 'receipt.json').read_text())
        self.assertEqual(receipt['status'], 'complete_known_divergence')
        self.assertTrue(receipt['campaign']['process_complete'])
        self.assertFalse(receipt['campaign']['passed'])
        self.assertFalse(receipt['performance_claim'])
        self.assertNotIn('speedup_wall_vs_original', receipt['summaries']['jax_cuda_batched'])

    def test_timing_science_failure_stops_remaining_runs(self):
        self.make_contract()
        argv = ['benchmark', '--original-source', str(self.path), '--reference-revision', 'a' * 40,
                '--branch-source', str(self.path), '--python', 'python', '--frame-file',
                str(self.path / 'frame.gwf'), '--bank-file', str(self.path / 'bank.hdf'),
                '--output', str(self.path / 'receipt.json'), '--output-dir', str(self.path / 'runs'),
                '--arms', 'original_cpu', 'jax_cuda_batched']

        def fake_case(*args, **kwargs):
            return {'input_contract': kwargs['expected_contract'], 'sample_rate': 2048,
                    'triggers_path': str(self.path / 'triggers.hdf')}

        module = 'tools.bench_jax_inspiral_campaign.'
        with patch('sys.argv', argv), patch(module + '_run_single_case', side_effect=fake_case) as run, patch(
                module + '_source_records', return_value={'original': {'revision': 'a' * 40, 'dirty': False}}), patch(
                module + 'validate_reference'), patch(module + 'validate_observer_source'), patch(
                module + '_require_source_unchanged'), patch(module + '_git_commit', return_value='a' * 40), patch(
                module + 'compare_scientific_hdf', return_value={'passed': True, 'scope': 'full', 'missing_gates': []}), patch(
                module + 'compare_trigger_parity', return_value={'passed': False}):
            with self.assertRaisesRegex(RuntimeError, 'remaining runs stopped'):
                main()
        self.assertEqual(run.call_count, 3)  # Two qualifications and one failed timed run.
        receipt = json.loads((self.path / 'receipt.json').read_text())
        self.assertEqual(receipt['failure']['phase'], 'run_science')
        self.assertEqual(sum(len(runs) for runs in receipt['raw_results'].values()), 1)

    def test_qualification_only_launches_no_timing_repetitions(self):
        self.make_contract()
        argv = ['benchmark', '--original-source', str(self.path), '--reference-revision', 'a' * 40,
                '--branch-source', str(self.path), '--python', 'python', '--frame-file',
                str(self.path / 'frame.gwf'), '--bank-file', str(self.path / 'bank.hdf'),
                '--output', str(self.path / 'receipt.json'), '--output-dir', str(self.path / 'runs'),
                '--arms', 'original_cpu', 'branch_cpu', '--qualification-only']

        def fake_case(*args, **kwargs):
            return {'input_contract': kwargs['expected_contract'], 'sample_rate': 2048,
                    'triggers_path': str(self.path / 'triggers.hdf'),
                    'evidence_path': str(self.path / 'science.hdf')}

        module = 'tools.bench_jax_inspiral_campaign.'
        with patch('sys.argv', argv), patch(module + '_run_single_case', side_effect=fake_case) as run, patch(
                module + '_source_records', return_value={'original': {'revision': 'a' * 40, 'dirty': False}}), patch(
                module + 'validate_reference'), patch(module + 'validate_observer_source'), patch(
                module + '_require_source_unchanged'), patch(module + '_git_commit', return_value='a' * 40), patch(
                module + 'compare_scientific_hdf', return_value={'passed': True, 'scope': 'full', 'missing_gates': []}):
            self.assertEqual(main(), 0)
        self.assertEqual(run.call_count, 2)
        receipt = json.loads((self.path / 'receipt.json').read_text())
        self.assertEqual(receipt['status'], 'science_qualification_only')
        self.assertFalse(receipt['performance_claim'])
        self.assertEqual(receipt['timing']['timing_runs_launched'], 0)
        self.assertEqual(receipt['raw_results'], {'original_cpu': [], 'branch_cpu': []})

    def test_profile_campaign_rewrites_fresh_paths_and_marks_exclusion(self):
        qualification_dir = self.path / 'qualification' / 'jax_cuda_batched'
        qualification_dir.mkdir(parents=True)
        old_output = qualification_dir / 'triggers.hdf'
        manifest = {
            'command': ['taskset', '-c', '8', 'python', str(qualification_dir / 'run.py'),
                        '--output', str(old_output)],
            'environment': {'PYCBC_BENCHMARK_WORK': str(qualification_dir / 'work.json')},
        }
        (qualification_dir / 'command.json').write_text(json.dumps(manifest))
        calls = []

        def fake_profile(**kwargs):
            calls.append(kwargs)
            Path(kwargs['output_json']).write_text(json.dumps({'summary': {'sample_count': 1}}))
            run_dir = Path(kwargs['output_json']).parent
            (run_dir / 'triggers.hdf').write_bytes(b'profile triggers')
            (run_dir / 'science.hdf').write_bytes(b'profile evidence')
            (run_dir / 'work.json').write_text(json.dumps({
                'signal_dtypes': ['complex64'], 'sample_rate': 2048.0,
                'completed_template_seconds': 1.0,
                'valid_detector_seconds': 1.0,
            }))
            return {
                'returncode': 0, 'summary': {'sample_count': 1},
                'process_tree': {'completion': {
                    'root_exited': True,
                    'all_observed_processes_exited': True,
                    'remaining_pids': [],
                }},
            }

        with patch('tools.profile_jax_gpu_timeline.run_command_profiling', side_effect=fake_profile):
            results = _run_profile_campaign(
                ['jax_cuda_batched'],
                {'jax_cuda_batched': {'triggers_path': str(old_output),
                                      'input_contract': {'fingerprint': 'x'}}},
                self.path / 'campaign')
        profile_dir = self.path / 'campaign' / 'profiles' / 'jax_cuda_batched'
        self.assertEqual(results['jax_cuda_batched']['excluded_from_unprofiled_timing'], True)
        self.assertEqual(calls[0]['cwd'], profile_dir.resolve())
        self.assertTrue(any(str(profile_dir.resolve()) in item for item in calls[0]['command']))
        self.assertNotIn(str(qualification_dir), json.dumps(calls[0]['command']))
        timeline = json.loads((profile_dir / 'timeline.json').read_text())
        self.assertTrue(timeline['excluded_from_unprofiled_timing'])
        self.assertEqual(timeline['cwd'], str(profile_dir.resolve()))

    def test_profile_campaign_rejects_incomplete_process_tree(self):
        qualification_dir = self.path / 'qualification' / 'jax_cuda_batched'
        qualification_dir.mkdir(parents=True)
        old_output = qualification_dir / 'triggers.hdf'
        (qualification_dir / 'command.json').write_text(json.dumps({
            'command': ['python', 'run.py'], 'environment': {},
        }))

        def fake_profile(**kwargs):
            return {
                'returncode': 0,
                'process_tree': {'completion': {
                    'root_exited': True,
                    'all_observed_processes_exited': False,
                    'remaining_pids': [1234],
                }},
            }

        with patch('tools.profile_jax_gpu_timeline.run_command_profiling',
                   side_effect=fake_profile):
            with self.assertRaisesRegex(RuntimeError, 'all observed process exits'):
                _run_profile_campaign(
                    ['jax_cuda_batched'],
                    {'jax_cuda_batched': {'triggers_path': str(old_output),
                                          'input_contract': {'fingerprint': 'x'}}},
                    self.path / 'campaign')

    def test_profile_campaign_rejects_failed_science_comparison(self):
        qualification_dir = self.path / 'qualification' / 'jax_cuda_batched'
        qualification_dir.mkdir(parents=True)
        old_output = qualification_dir / 'triggers.hdf'
        old_evidence = qualification_dir / 'science.hdf'
        old_output.write_bytes(b'reference triggers')
        old_evidence.write_bytes(b'reference evidence')
        (qualification_dir / 'command.json').write_text(json.dumps({
            'command': ['python', 'run.py'], 'environment': {},
        }))

        def fake_profile(**kwargs):
            run_dir = Path(kwargs['output_json']).parent
            Path(kwargs['output_json']).write_text(json.dumps({'summary': {}}))
            (run_dir / 'triggers.hdf').write_bytes(b'profile triggers')
            (run_dir / 'science.hdf').write_bytes(b'profile evidence')
            (run_dir / 'work.json').write_text(json.dumps({
                'signal_dtypes': ['complex64'], 'sample_rate': 2048.0,
            }))
            return {'returncode': 0, 'process_tree': {'completion': {
                'root_exited': True, 'all_observed_processes_exited': True,
                'remaining_pids': [],
            }}}

        with patch('tools.profile_jax_gpu_timeline.run_command_profiling',
                   side_effect=fake_profile), patch(
                       'tools.bench_jax_inspiral_campaign.compare_scientific_hdf',
                       return_value={'passed': False, 'missing_gates': ['triggers']}):
            with self.assertRaisesRegex(RuntimeError, 'scientific qualification failed'):
                _run_profile_campaign(
                    ['jax_cuda_batched'],
                    {'jax_cuda_batched': {'triggers_path': str(old_output),
                                          'evidence_path': str(old_evidence),
                                          'input_contract': {'fingerprint': 'x'}}},
                    self.path / 'campaign',
                    reference={'triggers_path': str(old_output),
                               'evidence_path': str(old_evidence)})

    def test_profile_mode_uses_one_fresh_run_per_arm_and_no_timing(self):
        self.make_contract()
        argv = ['benchmark', '--original-source', str(self.path), '--reference-revision', 'a' * 40,
                '--branch-source', str(self.path), '--python', 'python', '--frame-file',
                str(self.path / 'frame.gwf'), '--bank-file', str(self.path / 'bank.hdf'),
                '--output', str(self.path / 'receipt.json'), '--output-dir', str(self.path / 'runs'),
                '--arms', 'original_cpu', 'branch_cpu', 'jax_cpu_batched', '--profile-utilization',
                '--replicates', '3']

        def fake_case(*args, **kwargs):
            return {'input_contract': kwargs['expected_contract'], 'sample_rate': 2048,
                    'triggers_path': str(self.path / 'triggers.hdf'),
                    'evidence_path': str(self.path / 'science.hdf')}

        profile = {arm: {'timeline_path': str(self.path / (arm + '.json')),
                         'excluded_from_unprofiled_timing': True,
                         'science': {'passed': True, 'missing_gates': []}}
                   for arm in ('original_cpu', 'branch_cpu', 'jax_cpu_batched')}
        module = 'tools.bench_jax_inspiral_campaign.'
        with patch('sys.argv', argv), patch(module + '_run_single_case', side_effect=fake_case), patch(
                module + '_run_profile_campaign', return_value=profile) as profile_run, patch(
                module + '_source_records', return_value={'original': {'revision': 'a' * 40, 'dirty': False}}), patch(
                module + 'validate_reference'), patch(module + 'validate_observer_source'), patch(
                module + '_require_source_unchanged'), patch(module + '_git_commit', return_value='a' * 40), patch(
                module + 'compare_scientific_hdf', return_value={'passed': True, 'scope': 'full', 'missing_gates': []}):
            self.assertEqual(main(), 0)
        profile_run.assert_called_once()
        receipt = json.loads((self.path / 'receipt.json').read_text())
        self.assertEqual(receipt['status'], 'profiled_science_qualification')
        self.assertFalse(receipt['performance_claim'])
        self.assertTrue(receipt['campaign']['passed'])
        self.assertEqual(receipt['profile_utilization']['effective_replicates'], 1)
        self.assertEqual(receipt['timing']['timing_runs_launched'], 0)
        self.assertEqual(receipt['raw_results']['jax_cpu_batched'], [])

    def test_sample_summary(self):
        samples = [10.0, 12.0, 11.0, 9.0, 13.0]
        res = sample_summary(samples, "seconds")
        self.assertEqual(res["count"], 5)
        self.assertAlmostEqual(res["median"], 11.0)
        self.assertAlmostEqual(res["mean"], 11.0)
        self.assertEqual(res["min"], 9.0)
        self.assertEqual(res["max"], 13.0)
        self.assertTrue("median_ci95" in res)
        self.assertLessEqual(res["median_ci95"]["low"], res["median"])
        self.assertGreaterEqual(res["median_ci95"]["high"], res["median"])

    def test_sample_summary_empty(self):
        res = sample_summary([], "seconds")
        self.assertEqual(res["count"], 0)
        self.assertEqual(res["samples"], [])

    def test_campaign_physics_is_fixed(self):
        self.assertEqual(BENCHMARK_SAMPLE_RATE, 2048.0)
        self.assertEqual(BENCHMARK_PRECISION, "complex64")

    def test_work_receipt_requires_observed_2048_complex64(self):
        work = {"sample_rate": 2048, "signal_dtypes": ["complex64"]}
        self.assertEqual(_validate_benchmark_work(work), ["complex64"])
        with self.assertRaisesRegex(RuntimeError, "sample_rate"):
            _validate_benchmark_work({**work, "sample_rate": 4096})
        with self.assertRaisesRegex(RuntimeError, "signal_dtypes"):
            _validate_benchmark_work({**work, "signal_dtypes": ["float32"]})
        complete = dict(work, completed_templates=2, completed_template_seconds=20., valid_detector_seconds=10.)
        self.assertEqual(_validate_benchmark_work(complete, 2), ['complex64'])
        for changed in ({'completed_templates': 1}, {'completed_template_seconds': 30.},
                        {'valid_detector_seconds': float('nan')}):
            with self.subTest(changed=changed), self.assertRaisesRegex(RuntimeError, 'work receipt'):
                _validate_benchmark_work({**complete, **changed}, 2)

    def test_parse_stderr_phases_fallback(self):
        phases = _parse_stderr_phases([], process_wall_sec=100.0, calc_time_sec=60.0, tsetup_sec=15.0)
        self.assertEqual(phases["matched_filter_sec"], 60.0)
        self.assertEqual(phases["conditioning_sec"], 15.0)
        self.assertEqual(phases["startup_import_sec"], 25.0)

    def test_parse_stderr_phases_parsed(self):
        log_lines = [
            "2026-09-18T10:00:00.000 Reading Frames...",
            "2026-09-18T10:00:10.000 Read in template bank...",
            "2026-09-18T10:00:12.000 Filtering template 0...",
            "2026-09-18T10:00:50.000 We currently have 5 triggers...",
            "2026-09-18T10:00:52.000 Writing out triggers...",
            "2026-09-18T10:00:55.000 Finished",
        ]
        phases = _parse_stderr_phases(log_lines, process_wall_sec=60.0, calc_time_sec=30.0, tsetup_sec=10.0)
        self.assertGreater(phases["conditioning_sec"], 0.0)
        self.assertEqual(phases["matched_filter_sec"], 30.0)
        self.assertGreater(phases["waveform_prep_sec"], 0.0)
        self.assertGreaterEqual(phases["startup_import_sec"], 0.0)

    def test_compare_trigger_parity_exact(self):
        f1 = self.path / "base.hdf"
        f2 = self.path / "cand.hdf"

        with h5py.File(f1, "w") as fb, h5py.File(f2, "w") as fc:
            snrs = np.array([8.0, 9.5, 12.0, 15.5], dtype=np.float32)
            times = np.array([100.0, 105.0, 110.0, 115.0])
            fb.create_dataset("H1/snr", data=snrs)
            fb.create_dataset("H1/end_time", data=times)
            fc.create_dataset("H1/snr", data=snrs)
            fc.create_dataset("H1/end_time", data=times)
            for f in (fb, fc):
                f.create_dataset('H1/template_hash', data=np.arange(len(times), dtype=np.uint64))
                for name in ('chisq', 'chisq_dof', 'sigmasq', 'coa_phase'):
                    f.create_dataset('H1/' + name, data=np.ones(len(times)))

        res = compare_trigger_parity(f1, f2)
        self.assertTrue(res["passed"])
        self.assertEqual(res["detectors"]["H1"]["matched_triggers"], 4)

    def test_compare_trigger_parity_tolerance(self):
        f1 = self.path / "base_tol.hdf"
        f2 = self.path / "cand_tol.hdf"

        with h5py.File(f1, "w") as fb, h5py.File(f2, "w") as fc:
            snrs_b = np.array([8.0, 10.0, 12.0], dtype=np.float32)
            snrs_c = np.array([8.00001, 10.00001, 12.00001], dtype=np.float32)
            times = np.array([100.0, 105.0, 110.0])
            fb.create_dataset("H1/snr", data=snrs_b)
            fb.create_dataset("H1/end_time", data=times)
            fc.create_dataset("H1/snr", data=snrs_c)
            fc.create_dataset("H1/end_time", data=times)
            for f in (fb, fc):
                f.create_dataset('H1/template_hash', data=np.arange(len(times), dtype=np.uint64))
                for name in ('chisq', 'chisq_dof', 'sigmasq', 'coa_phase'):
                    f.create_dataset('H1/' + name, data=np.ones(len(times)))

        res = compare_trigger_parity(f1, f2)
        self.assertTrue(res["passed"])
        self.assertLess(res["detectors"]["H1"]["fields"]["H1/snr"]["max_relative_difference"], 1e-4)

    def test_campaign_parity_cannot_use_branch_as_reference(self):
        contract = self.make_contract()
        raw = {
            "branch_cpu": [{"triggers_path": "current.hdf", 'input_contract': contract}],
            "original_cpu": [{"triggers_path": "original.hdf", 'input_contract': contract}],
            "jax_cuda_batched": [{"triggers_path": "jax.hdf", 'input_contract': contract}],
        }
        with patch(
            "tools.bench_jax_inspiral_campaign.compare_trigger_parity",
            return_value={"passed": True},
        ) as compare:
            reference, results = compare_campaign_parity(raw)
            self.assertEqual(reference, "original_cpu")
            self.assertEqual(set(results), {"original_cpu", "branch_cpu", "jax_cuda_batched"})
            self.assertEqual(compare.call_count, 3)
            self.assertTrue(all(call.args[0] == Path("original.hdf")
                                for call in compare.call_args_list))
        self.assertEqual(compare_campaign_parity({"branch_cpu": raw["branch_cpu"]}),
                         (None, {}))
        self.assertEqual(compare_campaign_parity({"jax_cuda_batched": raw["jax_cuda_batched"]}),
                         (None, {}))
        raw['jax_cuda_batched'][0].pop('input_contract')
        with self.assertRaisesRegex(ValueError, 'input contract'):
            compare_campaign_parity(raw)

    def test_original_source_must_be_clean_and_pinned(self):
        _require_pristine_original({"revision": "abc", "dirty": False})
        with self.assertRaisesRegex(RuntimeError, "clean"):
            _require_pristine_original({"revision": "abc", "dirty": True})
        with self.assertRaisesRegex(RuntimeError, "clean"):
            _require_pristine_original({"revision": "abc", "dirty": None})
        with self.assertRaisesRegex(RuntimeError, "git checkout"):
            _require_pristine_original({"revision": None, "dirty": False})

    def test_original_source_mutation_is_rejected(self):
        expected = {"revision": "abc", "dirty": False, "files_sha256": {}}
        with patch(
            "tools.bench_jax_inspiral_campaign.source_identity",
            return_value={**expected, "revision": "def"},
        ):
            with self.assertRaisesRegex(RuntimeError, "changed"):
                _require_source_unchanged(self.path, expected)

    def test_arm_definitions(self):
        self.assertIn("original_cpu", ARM_NAMES)
        self.assertIn("branch_cpu", ARM_NAMES)
        self.assertNotIn("branch_cpu_batched", ARM_NAMES)
        self.assertIn("jax_cpu", ARM_NAMES)
        self.assertIn("jax_cpu_batched", ARM_NAMES)
        self.assertIn("jax_cuda", ARM_NAMES)
        self.assertIn("jax_cuda_batched", ARM_NAMES)
        self.assertIn("jax_cuda_diffgw", ARM_NAMES)
        self.assertEqual(DEFAULT_ARMS, ("original_cpu", "branch_cpu", "jax_cpu", "jax_cuda"))
        self.assertEqual(
            BATCHED_ARMS,
            ("original_cpu", "branch_cpu", "jax_cpu_batched", "jax_cuda_batched"),
        )

    def test_completed_case_requires_successful_process_receipt(self):
        trigger = self.path / "run" / "triggers.hdf"
        trigger.parent.mkdir()
        trigger.write_bytes(b"placeholder")
        self.assertFalse(_completed_case({"triggers_path": str(trigger)}))
        (trigger.parent / "process.json").write_text(
            json.dumps({"returncode": 0})
        )
        (trigger.parent / "command.json").write_text(json.dumps({"command": ["python"]}))
        (trigger.parent / "work.json").write_text(json.dumps({
            "completed_template_seconds": 1.0, "valid_detector_seconds": 1.0,
        }))
        self.assertTrue(_completed_case({"triggers_path": str(trigger)}))

    def test_retry_output_dir_preserves_existing_logs(self):
        base = self.path / "runs" / "jax_cpu_rep1"
        base.mkdir(parents=True)
        (base / "stderr.log").write_text("failed")
        retry = _retry_output_dir(base)
        self.assertEqual(retry.name, "jax_cpu_rep1.retry1")
        self.assertFalse(retry.exists())

    def test_diffgw_arm_fails_before_launch(self):
        frame = self.path / "frame.gwf"
        bank = self.path / "bank.hdf"
        frame.write_bytes(b"")
        bank.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "Explicit diffgw arms"):
            _run_single_case(
                "jax_cpu_diffgw", self.path, "python", self.path / "run",
                frame, bank, use_compressed_waveforms=False,
            )


if __name__ == "__main__":
    unittest.main()
