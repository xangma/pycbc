"""Verify complete loop coverage and cleanup without CUDA dependencies."""
from contextlib import ExitStack
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
    'profile_torch_filtering.py')
acquire = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acquire)


class AcquisitionTests(unittest.TestCase):
    def exercise(self, mode):
        operations = []

        class Bank:
            def __len__(self):
                return 2

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

        def synchronize(device):
            operations.append('sync')

        def noop(*args):
            pass

        torch = namespace(
            __version__='fake', version=namespace(cuda='fake'),
            get_num_threads=lambda: 2 if mode == 'wrong-threads' else 1,
            get_num_interop_threads=lambda: 1,
            set_num_threads=noop, set_num_interop_threads=noop,
            cuda=namespace(get_device_name=lambda _: 'fake',
                           synchronize=synchronize),
            profiler=namespace(profile=lambda **_: Profiler(),
                               record_function=Range,
                               ProfilerActivity=namespace(CPU=1, CUDA=2)))
        array = namespace(squared_norm=noop, inner=noop, numpy=noop)
        chisq = namespace(power_chisq_at_points_from_precomputed=noop)
        device = namespace(type='cpu' if mode == 'wrong-device' else 'cuda')
        modules = {
            'torch': torch, 'pycbc': namespace(), 'pycbc.events': namespace(),
            'pycbc.events.eventmgr': namespace(EventManager=EventManager),
            'pycbc.events.threshold_torch': namespace(
                TorchThresholdCluster=Threshold),
            'pycbc.fft': namespace(),
            'pycbc.fft.torchfft': namespace(IFFT=IFFT),
            'pycbc.filter': namespace(),
            'pycbc.filter.matchedfilter_torch': namespace(
                TorchCorrelator=Correlator),
            'pycbc.scheme': namespace(
                mgr=namespace(state=namespace(device=device))),
            'pycbc.types': namespace(array_torch=array),
            'pycbc.vetoes': namespace(chisq_torch=chisq),
            'pycbc.waveform': namespace(),
            'pycbc.waveform.bank': namespace(FilterBank=Bank),
        }
        original_getitem, original_execute = Bank.__getitem__, IFFT.execute

        def workload(*args, **kwargs):
            if mode == 'threads-changed':
                torch.get_num_interop_threads = lambda: 2
            bank = Bank()
            fft = IFFT()
            for index in range(2):
                bank[index if mode != 'repeat' else 0]
                if mode == 'failure':
                    raise RuntimeError('workload failed')
                for _ in range(2 if mode == 'missed-ifft' else 3):
                    fft.execute()
            if mode != 'missing-stop':
                EventManager().save_performance(1, 3, 2, 1.0, 0.1)

        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            output = Path(folder) / 'trace'
            argv = ['profile', '--output', str(output), '--', 'pycbc_inspiral']
            original_argv = sys.argv
            stack.enter_context(patch('sys.argv', argv))
            stack.enter_context(patch.dict(sys.modules, modules))
            stack.enter_context(patch.object(acquire.os, 'sched_getaffinity',
                                             return_value={8}, create=True))
            stack.enter_context(patch.object(
                acquire.runpy, 'run_path', side_effect=workload))
            if mode == 'success':
                acquire.main()
            else:
                with self.assertRaises((RuntimeError, ValueError)):
                    acquire.main()
            record = json.loads((output / 'receipt.json').read_text())
            self.assertIs(Bank.__getitem__, original_getitem)
            self.assertIs(IFFT.execute, original_execute)
            self.assertIs(sys.argv, argv)
            self.assertEqual(
                record['state'], 'complete' if mode == 'success' else 'failed')
            self.assertFalse(record['eligible_for_throughput'])
        self.assertIs(sys.argv, original_argv)
        return record, operations

    def test_complete_workload_only_synchronizes_at_boundaries(self):
        record, operations = self.exercise('success')
        self.assertEqual(record['indices'], [0, 1])
        self.assertEqual(record['semantic_calls']['ifft'], 6)
        for key in ('threads_before_executable', 'threads_at_first_bank',
                    'threads_at_loop_end'):
            self.assertEqual(record[key], dict(intra_op=1, inter_op=1))
        self.assertEqual(operations.count('sync'), 2)
        self.assertEqual(operations.count('start'), 1)
        self.assertEqual(operations.count('stop'), 1)
        self.assertLess(operations.index('stop'),
                        operations.index(('performance', (1, 3, 2, 1.0, 0.1))))

    def test_incomplete_or_wrong_workload_never_gets_success_receipt(self):
        for mode in ['failure', 'repeat', 'missed-ifft',
                     'missing-stop', 'wrong-device', 'wrong-threads',
                     'threads-changed']:
            with self.subTest(mode=mode):
                _, operations = self.exercise(mode)
                self.assertEqual(operations.count('start'),
                                 operations.count('stop'))


if __name__ == '__main__':
    unittest.main()
