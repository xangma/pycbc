"""Verify complete loop coverage and cleanup without CUDA dependencies."""
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    'profile_torch_filtering',
    Path(__file__).resolve().parents[1] / 'tools' /
    'profile_torch_filtering.py'
)
acquire = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acquire)


class AcquisitionTests(unittest.TestCase):
    def exercise(
        self,
        mode,
        route='scalar',
        tile_sizes=None,
        extra_argv=None,
        num_templates=2,
        num_segments=3,
        device_type='cuda',
        template_1d=False,
        positional_batch=False,
        modules_override=None,
        command_argv=None,
    ):
        operations = []
        recorded_activities = []

        class Bank:
            def __init__(self, count=num_templates):
                self.count = count
                self.filename = 'fake_bank.hdf'

            def __len__(self):
                return self.count

            def __getitem__(self, index):
                operations.append(('template', index))
                return index

        class EventManager:
            def save_performance(self, *args):
                operations.append(('performance', args))

        class IFFT:
            def execute(self):
                operations.append('ifft')

        class Correlator:
            def correlate(self):
                pass

        class Threshold:
            def threshold_and_cluster(self):
                pass

        class Core:
            @staticmethod
            def correlate_and_ifft(
                templates, data, cout_workspace, out_workspace, tlen,
                *args, **kwargs
            ):
                b = kwargs.get('batch_size')
                if b is None and len(args) >= 4:
                    b = args[3]
                if b is None:
                    ndim = getattr(templates, 'ndim', None)
                    shape = getattr(templates, 'shape', None)
                    if ndim == 1 or (shape is not None and len(shape) == 1):
                        b = 1
                    elif ndim == 2 or (shape is not None and len(shape) == 2):
                        b = shape[0]
                    else:
                        b = 1
                operations.append(('correlate_and_ifft', int(b)))
                torch.mul(templates, data)
                torch.fft.ifft(cout_workspace)
                return cout_workspace, out_workspace

        class Candidates:
            @staticmethod
            def select_tile_candidates(*args, **kwargs):
                operations.append('select_candidates')
                return {'candidates': {}}

        class Vetoes:
            @staticmethod
            def batched_power_chisq(*args, **kwargs):
                operations.append('batched_power_chisq')
                return {}

        class Range:
            def __init__(self, name):
                self.name = name

            def __enter__(self):
                operations.append(('enter', self.name))

            def __exit__(self, *args):
                operations.append(('exit', self.name))

        class Profiler:
            def start(self):
                operations.append('start')

            def stop(self):
                operations.append('stop')

            def export_chrome_trace(self, filename):
                Path(filename).write_text('{}')

        def namespace(**kwargs):
            return types.SimpleNamespace(**kwargs)

        def make_profiler(**kwargs):
            recorded_activities.extend(kwargs.get('activities', []))
            return Profiler()

        def synchronize(device):
            operations.append('sync')

        def noop(*args):
            pass

        def mul_op(*args, **kwargs):
            operations.append('mul')
            return args[0] if args else None

        def core_ifft_op(*args, **kwargs):
            operations.append('core_ifft')
            return args[0] if args else None

        torch = namespace(
            __version__='fake',
            version=namespace(cuda='fake'),
            get_num_threads=lambda: 2 if mode == 'wrong-threads' else 1,
            get_num_interop_threads=lambda: 1,
            set_num_threads=noop,
            set_num_interop_threads=noop,
            mul=mul_op,
            fft=namespace(ifft=core_ifft_op),
            cuda=namespace(
                get_device_name=lambda _: 'fake',
                synchronize=synchronize,
            ),
            profiler=namespace(
                profile=make_profiler,
                record_function=Range,
                ProfilerActivity=namespace(CPU=1, CUDA=2),
            ),
        )
        array = namespace(squared_norm=noop, inner=noop, numpy=noop)
        chisq = namespace(power_chisq_at_points_from_precomputed=noop)

        if mode == 'wrong-device' and not extra_argv:
            dev_type = 'cpu'
        else:
            dev_type = device_type
        device = namespace(type=dev_type)

        modules = {
            'torch': torch,
            'pycbc': namespace(),
            'pycbc.events': namespace(),
            'pycbc.events.eventmgr': namespace(EventManager=EventManager),
            'pycbc.events.threshold_torch': namespace(
                TorchThresholdCluster=Threshold
            ),
            'pycbc.fft': namespace(),
            'pycbc.fft.torchfft': namespace(IFFT=IFFT),
            'pycbc.filter': namespace(),
            'pycbc.filter.matchedfilter_torch': namespace(
                TorchCorrelator=Correlator
            ),
            'pycbc.filter.gpu_search': namespace(),
            'pycbc.filter.gpu_search.core': namespace(
                correlate_and_ifft=Core.correlate_and_ifft
            ),
            'pycbc.filter.gpu_search.adapter': namespace(
                correlate_and_ifft=Core.correlate_and_ifft,
                select_tile_candidates=Candidates.select_tile_candidates,
                batched_power_chisq=Vetoes.batched_power_chisq,
            ),
            'pycbc.filter.gpu_search.candidates': namespace(
                select_tile_candidates=Candidates.select_tile_candidates,
            ),
            'pycbc.filter.gpu_search.vetoes': namespace(
                batched_power_chisq=Vetoes.batched_power_chisq,
            ),
            'pycbc.scheme': namespace(
                mgr=namespace(state=namespace(device=device))
            ),
            'pycbc.types': namespace(array_torch=array),
            'pycbc.vetoes': namespace(chisq_torch=chisq),
            'pycbc.waveform': namespace(),
            'pycbc.waveform.bank': namespace(FilterBank=Bank),
        }
        if modules_override:
            modules_override(modules)
        original_getitem = Bank.__getitem__
        original_execute = IFFT.execute
        original_correlate = Core.correlate_and_ifft
        original_select = Candidates.select_tile_candidates
        original_chisq = Vetoes.batched_power_chisq
        original_mul = torch.mul
        original_core_ifft = torch.fft.ifft

        def workload(*args, **kwargs):
            if mode == 'threads-changed':
                torch.get_num_interop_threads = lambda: 2
            bank = Bank()
            for index in range(num_templates):
                bank[index if mode != 'repeat' else 0]

            if mode == 'failure':
                raise RuntimeError('workload failed')

            # External mul outside correlate_and_ifft must not record core_mul
            torch.mul(None, None)

            if route == 'scalar':
                fft = IFFT()
                for _ in range(num_templates):
                    for _ in range(
                        2 if mode == 'missed-ifft' else num_segments
                    ):
                        fft.execute()
            elif route == 'tiled':
                core = modules['pycbc.filter.gpu_search.core']
                cand = modules['pycbc.filter.gpu_search.candidates']
                actual_tile_sizes = (
                    tile_sizes if tile_sizes is not None
                    else [1] * num_templates
                )
                missing_dropped = False
                for _ in range(num_segments):
                    for b_size in actual_tile_sizes:
                        if mode == 'missing-work' and not missing_dropped:
                            missing_dropped = True
                            continue
                        if template_1d:
                            t_obj = types.SimpleNamespace(
                                ndim=1, shape=(1024,)
                            )
                            core.correlate_and_ifft(
                                t_obj, None, None, None, 128
                            )
                        elif positional_batch:
                            t_obj = types.SimpleNamespace(
                                ndim=2, shape=(b_size, 100)
                            )
                            core.correlate_and_ifft(
                                t_obj, None, None, None, 128,
                                100, 10, 90, b_size
                            )
                        else:
                            core.correlate_and_ifft(
                                templates=types.SimpleNamespace(
                                    ndim=2, shape=(b_size, 100)
                                ),
                                data=None,
                                cout_workspace=None,
                                out_workspace=None,
                                tlen=128,
                                batch_size=b_size,
                            )
                        cand.select_tile_candidates()

            if mode != 'missing-stop':
                EventManager().save_performance(
                    1, num_segments, num_templates, 1.0, 0.1
                )

        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            output = Path(folder) / 'trace'
            argv = ['profile', '--output', str(output)]
            if extra_argv:
                argv.extend(extra_argv)
            argv.extend(['--', 'pycbc_inspiral'])
            if command_argv:
                argv.extend(command_argv)
            original_argv = sys.argv
            stack.enter_context(patch('sys.argv', argv))
            stack.enter_context(patch.dict(sys.modules, modules))
            stack.enter_context(patch.object(
                acquire.os, 'sched_getaffinity', return_value={8}, create=True
            ))
            stack.enter_context(patch.object(
                acquire.runpy, 'run_path', side_effect=workload
            ))
            if mode == 'success':
                acquire.main()
            else:
                with self.assertRaises((RuntimeError, ValueError)):
                    acquire.main()
            record = json.loads((output / 'receipt.json').read_text())
            self.assertIs(Bank.__getitem__, original_getitem)
            self.assertIs(IFFT.execute, original_execute)
            self.assertIs(Core.correlate_and_ifft, original_correlate)
            self.assertIs(
                Candidates.select_tile_candidates, original_select
            )
            self.assertIs(Vetoes.batched_power_chisq, original_chisq)
            self.assertIs(torch.mul, original_mul)
            self.assertIs(torch.fft.ifft, original_core_ifft)
            self.assertIs(sys.argv, argv)
            self.assertEqual(
                record['state'], 'complete' if mode == 'success' else 'failed'
            )
            self.assertFalse(record['eligible_for_throughput'])
        self.assertIs(sys.argv, original_argv)
        return record, operations, recorded_activities

    def test_complete_workload_only_synchronizes_at_boundaries(self):
        record, operations, _ = self.exercise('success')
        self.assertEqual(record['indices'], [0, 1])
        self.assertEqual(record['semantic_calls']['ifft'], 6)
        self.assertEqual(record['route'], 'scalar')
        for key in (
            'threads_before_executable',
            'threads_at_first_bank',
            'threads_at_loop_end',
        ):
            self.assertEqual(record[key], dict(intra_op=1, inter_op=1))
        self.assertEqual(operations.count('sync'), 2)
        self.assertEqual(operations.count('start'), 1)
        self.assertEqual(operations.count('stop'), 1)
        self.assertLess(
            operations.index('stop'),
            operations.index(('performance', (1, 3, 2, 1.0, 0.1))),
        )

    def test_incomplete_or_wrong_workload_never_gets_success_receipt(self):
        for mode in [
            'failure',
            'repeat',
            'missed-ifft',
            'missing-stop',
            'wrong-device',
            'wrong-threads',
            'threads-changed',
        ]:
            with self.subTest(mode=mode):
                _, operations, _ = self.exercise(mode)
                self.assertEqual(
                    operations.count('start'), operations.count('stop')
                )

    def test_tiled_b1_coverage(self):
        record, operations, _ = self.exercise(
            'success',
            route='tiled',
            tile_sizes=[1, 1],
            num_templates=2,
            num_segments=3,
        )
        self.assertEqual(record['route'], 'tiled')
        self.assertEqual(record['state'], 'complete')
        self.assertEqual(record['executed_template_rows'], 6)
        self.assertEqual(record['tile_calls'], 6)
        self.assertEqual(record['tile_sizes'], [1, 1, 1, 1, 1, 1])
        self.assertEqual(
            record['semantic_calls']['core_correlate_and_ifft'], 6
        )
        self.assertEqual(
            record['semantic_calls']['select_candidates'], 6
        )
        self.assertEqual(
            record['semantic_calls']['core_mul'], 6
        )
        self.assertEqual(
            record['semantic_calls']['core_ifft'], 6
        )
        self.assertEqual(
            operations.count(('enter', 'pycbc::core_mul')), 6
        )
        self.assertEqual(
            operations.count(('enter', 'pycbc::core_ifft')), 6
        )
        self.assertEqual(operations.count('sync'), 2)

    def test_tiled_batch_and_tail_tile_sizes(self):
        record, operations, _ = self.exercise(
            'success',
            route='tiled',
            tile_sizes=[2, 2, 1],
            num_templates=5,
            num_segments=2,
        )
        self.assertEqual(record['route'], 'tiled')
        self.assertEqual(record['state'], 'complete')
        self.assertEqual(record['executed_template_rows'], 10)
        self.assertEqual(record['tile_calls'], 6)
        self.assertEqual(record['tile_sizes'], [2, 2, 1, 2, 2, 1])
        self.assertEqual(
            record['semantic_calls']['core_correlate_and_ifft'], 6
        )

    def test_opt_in_cpu_activity_selection_without_cuda_sync(self):
        record, operations, activities = self.exercise(
            'success',
            route='tiled',
            tile_sizes=[2],
            num_templates=2,
            num_segments=1,
            extra_argv=['--cpu'],
            device_type='cpu',
        )
        self.assertEqual(record['route'], 'tiled')
        self.assertEqual(record['state'], 'complete')
        self.assertEqual(record['device_name'], 'cpu')
        self.assertEqual(operations.count('sync'), 0)
        self.assertEqual(activities, [1])

    def test_tiled_missing_work_fails_receipt(self):
        record, operations, _ = self.exercise(
            'missing-work',
            route='tiled',
            tile_sizes=[1, 1],
            num_templates=2,
            num_segments=2,
        )
        self.assertEqual(record['state'], 'failed')
        self.assertIn('differs from', record['error'])

    def test_tiled_cleanup_on_failure(self):
        record, operations, _ = self.exercise(
            'failure',
            route='tiled',
            tile_sizes=[1, 1],
            num_templates=2,
            num_segments=1,
        )
        self.assertEqual(record['state'], 'failed')

    def test_1d_template_counts_as_one_row(self):
        record, operations, _ = self.exercise(
            'success',
            route='tiled',
            tile_sizes=[1, 1],
            num_templates=2,
            num_segments=3,
            template_1d=True,
        )
        self.assertEqual(record['state'], 'complete')
        self.assertEqual(record['executed_template_rows'], 6)
        self.assertEqual(record['tile_sizes'], [1, 1, 1, 1, 1, 1])
        self.assertEqual(
            record['semantic_calls']['core_correlate_and_ifft'], 6
        )

    def test_positional_batch_size_derived_correctly(self):
        record, operations, _ = self.exercise(
            'success',
            route='tiled',
            tile_sizes=[2, 1],
            num_templates=3,
            num_segments=2,
            positional_batch=True,
        )
        self.assertEqual(record['state'], 'complete')
        self.assertEqual(record['executed_template_rows'], 6)
        self.assertEqual(record['tile_sizes'], [2, 1, 2, 1])

    def test_core_mul_and_core_ifft_ranges_recorded(self):
        record, operations, _ = self.exercise(
            'success',
            route='tiled',
            tile_sizes=[2],
            num_templates=2,
            num_segments=1,
        )
        self.assertEqual(record['state'], 'complete')
        self.assertEqual(record['semantic_calls']['core_mul'], 1)
        self.assertEqual(record['semantic_calls']['core_ifft'], 1)
        self.assertIn(('enter', 'pycbc::core_mul'), operations)
        self.assertIn(('enter', 'pycbc::core_ifft'), operations)

    def test_source_identity_and_input_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_core = Path(tmpdir) / 'fake_core.py'
            fake_core.write_text('# fake core\n')
            expected_core_hash = hashlib.sha256(
                fake_core.read_bytes()
            ).hexdigest()

            cmd_input = Path(tmpdir) / 'input.hdf'
            cmd_input.write_text('input data')
            expected_input_hash = hashlib.sha256(
                cmd_input.read_bytes()
            ).hexdigest()

            def set_core_file(mods):
                mods['pycbc.filter.gpu_search.core'].__file__ = (
                    str(fake_core)
                )

            record, _, _ = self.exercise(
                'success',
                command_argv=['--bank-file', str(cmd_input)],
                modules_override=set_core_file,
            )
            self.assertEqual(record['state'], 'complete')

            self.assertIn('source_identity', record)
            src = record['source_identity']
            self.assertIn('tools.profile_torch_filtering', src)
            self.assertIn('pycbc.filter.gpu_search.core', src)
            self.assertEqual(
                src['pycbc.filter.gpu_search.core']['path'],
                str(fake_core.resolve()),
            )
            self.assertEqual(
                src['pycbc.filter.gpu_search.core']['sha256'],
                expected_core_hash,
            )

            self.assertIn('input_identity', record)
            inp = record['input_identity']
            self.assertIn(str(cmd_input), inp)
            entry = inp[str(cmd_input)]
            self.assertEqual(entry['path'], str(cmd_input.resolve()))
            self.assertEqual(entry['size'], len(b'input data'))
            self.assertEqual(entry['sha256'], expected_input_hash)
            self.assertIn('mtime', entry)

    def test_modifying_input_file_changes_sha256(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd_input = Path(tmpdir) / 'test_data.dat'
            cmd_input.write_text('version 1')

            record1, _, _ = self.exercise(
                'success',
                command_argv=['--data', str(cmd_input)],
            )
            sha1 = record1['input_identity'][str(cmd_input)]['sha256']
            self.assertEqual(
                sha1, hashlib.sha256(b'version 1').hexdigest()
            )

            cmd_input.write_text('version 2 modified')
            record2, _, _ = self.exercise(
                'success',
                command_argv=['--data', str(cmd_input)],
            )
            sha2 = record2['input_identity'][str(cmd_input)]['sha256']
            self.assertEqual(
                sha2, hashlib.sha256(b'version 2 modified').hexdigest()
            )
            self.assertNotEqual(sha1, sha2)


if __name__ == '__main__':
    unittest.main()
