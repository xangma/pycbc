"""Check attribution against specified overlapping timelines."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    'summarize_torch_trace',
    Path(__file__).resolve().parents[1] / 'tools' / 'summarize_torch_trace.py')
trace_tools = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trace_tools)


def event(name, category, start, duration, *, thread=1, process=42, **args):
    return dict(name=name, cat=category, ts=start, dur=duration, ph='X',
                tid=thread, pid=process, args=args)


def scope(name, start, duration):
    return event('pycbc::' + name, 'user_annotation', start, duration)


def fixture():
    return dict(traceEvents=[
        scope('filter_loop', 0, 100), scope('ifft', 0, 15),
        scope('chisq', 15, 15), scope('to_numpy', 20, 5),
        event('aten::fft', 'cpu_op', 1, 4, **{'External id': 1}),
        event('cudaLaunchKernel', 'cuda_runtime', 2, 1, correlation=10),
        event('aten::sum', 'cpu_op', 16, 2, **{'External id': 2}),
        event('cudaLaunchKernel', 'cuda_runtime', 16, 1, correlation=20),
        event('cudaMemcpyAsync', 'cuda_runtime', 21, 1, correlation=30),
        event('fft_kernel', 'kernel', 50, 20, process=0, correlation=10,
              **{'External id': 1}),
        event('reduce_kernel', 'kernel', 60, 20, process=0, correlation=20,
              **{'External id': 2}),
        event('Memcpy DtoH', 'gpu_memcpy', 70, 20, process=0,
              correlation=30, bytes=16),
        event('pycbc::ifft', 'gpu_user_annotation', 50, 20, process=0),
    ])


class TraceSummaryTests(unittest.TestCase):
    def test_window_counts_only_positive_intersections(self):
        outside = [dict(ts=0, dur=5), dict(ts=20, dur=3)]
        result = trace_tools.metrics(outside, (10, 15))
        self.assertEqual(result, dict(event_count=0, summed_seconds=0,
                                      union_seconds=0))
        spans = [(8, 4), (11, 3), (13, 3), (5, 5), (15, 3),
                 (10, 0), (12, 0)]
        result = trace_tools.metrics(
            outside + [dict(ts=ts, dur=dur) for ts, dur in spans], (10, 15))
        self.assertEqual(result['event_count'], 3)
        self.assertEqual(result['summed_seconds'], 7e-6)
        self.assertEqual(result['union_seconds'], 5e-6)

    def test_asynchronous_links_nested_ownership_and_overlap(self):
        result = trace_tools.summarize(fixture())
        actual = result['devices']['0']['all_trace']
        self.assertEqual(actual['event_count'], 3)
        self.assertAlmostEqual(actual['summed_seconds'], 60e-6)
        self.assertAlmostEqual(actual['union_seconds'], 40e-6)
        self.assertEqual({row['path'] for row in result['by_semantic_path']},
                         {'ifft', 'chisq', 'chisq/to_numpy'})
        copies = [r for r in result['by_device_event_name']
                  if r['category'] == 'gpu_memcpy']
        self.assertEqual(copies[0]['recorded_bytes'], 16)
        self.assertEqual(result['linkage_counts'],
                         {'external_and_runtime': 2, 'runtime': 1})

    def test_missing_links_and_other_thread_are_unattributed(self):
        trace = fixture()
        trace['traceEvents'] += [
            event('cudaLaunchKernel', 'cuda_runtime', 5, 1,
                  thread=2, correlation=90),
            event('unlinked', 'kernel', 30, 5, process=0),
            event('thread2', 'kernel', 35, 5, process=0, correlation=90),
            event('memset', 'gpu_memset', 40, 5, process=0,
                  **{'External id': 0}),
        ]
        result = trace_tools.summarize(trace)
        self.assertEqual(result['linkage_counts']
                         ['missing_semantic_or_link'], 3)
        self.assertEqual(sum(r['event_count']
                             for r in result['by_semantic_path']
                             if r['path'] == 'unattributed'), 3)

    def test_conflicting_identifiers_are_not_guessed(self):
        trace = fixture()
        trace['traceEvents'] += [event('conflict', 'kernel', 40, 5, process=0,
                                       correlation=20, **{'External id': 1})]
        result = trace_tools.summarize(trace)
        self.assertEqual(result['linkage_counts']['conflicting'], 1)

    def test_window_clipping_and_separate_devices(self):
        trace = dict(traceEvents=[scope('filter_loop', 10, 20),
                                  event('a', 'kernel', 0, 40, process=0),
                                  event('b', 'kernel', 5, 20, process=1)])
        result = trace_tools.summarize(trace)
        self.assertEqual(result['devices']['0']
                         ['inside_loop']['union_seconds'], 20e-6)
        self.assertEqual(result['devices']['1']
                         ['inside_loop']['union_seconds'], 15e-6)
        self.assertEqual(
            result['device_events_outside_or_crossing_loop']['event_count'], 2)

    def test_invalid_duration_or_loop_fails_closed(self):
        for events in [[], [scope('filter_loop', 0, 0)],
                       [scope('filter_loop', 0, 1),
                        scope('filter_loop', 2, 1)],
                       [scope('filter_loop', 0, 1),
                        event('a', 'kernel', 0, -1)],
                       [scope('filter_loop', float('nan'), 1)]]:
            with self.subTest(events=events), self.assertRaises(ValueError):
                trace_tools.summarize(dict(traceEvents=events))

    def test_cli_checks_receipt_hash_and_refuses_empty_device_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / 'trace.json'
            receipt = root / 'receipt.json'
            output = root / 'summary.json'
            argv = ['summary', str(trace), '--receipt', str(receipt),
                    '--output', str(output)]
            trace.write_text(json.dumps(fixture()))
            receipt.write_text(json.dumps(
                dict(state='complete', trace_sha256='wrong')))
            with patch('sys.argv', argv), self.assertRaisesRegex(
                    ValueError, 'receipt'):
                trace_tools.main()
            receipt.write_text(json.dumps(dict(
                state='complete', trace_sha256=hashlib.sha256(
                    trace.read_bytes()).hexdigest())))
            with patch('sys.argv', argv):
                trace_tools.main()
            self.assertTrue(json.loads(output.read_text())['devices'])
            trace.write_text(json.dumps(
                dict(traceEvents=[scope('filter_loop', 0, 100)])))
            receipt.write_text(json.dumps(dict(
                state='complete', trace_sha256=hashlib.sha256(
                    trace.read_bytes()).hexdigest())))
            with patch('sys.argv', argv), self.assertRaisesRegex(
                    ValueError, 'No device'):
                trace_tools.main()


if __name__ == '__main__':
    unittest.main()
