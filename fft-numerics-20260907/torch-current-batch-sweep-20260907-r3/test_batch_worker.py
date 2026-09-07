"""Evidence checks without importing PyCBC, Torch, or accessing a GPU."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


spec = importlib.util.spec_from_file_location('batch_worker', Path(__file__).with_name('batch-worker.py'))
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
w.np = np


def test_workload_identity_does_not_change_with_execution_batch():
    args = SimpleNamespace(size=4096, bank_size=32, seed=7101, num_blocks=3, batch=1)
    first = w.workload(args)
    for batch in w.BATCHES:
        args.batch = batch
        other = w.workload(args)
        assert w.canonical_hash(first[-1]) == w.canonical_hash(other[-1])
        np.testing.assert_array_equal(first[0], other[0])
        for left, right in zip(first[3], other[3]):
            np.testing.assert_array_equal(left, right)


def test_pointwise_check_detects_phase_error_near_zero_and_nonfinite():
    reference = np.array([1+0j, 0j, 3+4j], dtype=np.complex64)
    assert w.row_metrics(reference, reference)['failed_samples'] == 0
    actual = np.array([0+1j, 2e-6+0j, complex('nan')], dtype=np.complex64)
    result = w.row_metrics(actual, reference)
    assert result['failed_samples'] == 3
    assert result['nonfinite_actual'] == 1


def trigger():
    return {'count': 1, 'fields': {'template_id': [1000], 'peak_index': [25],
                                 'end_time': [0.25], 'snr': [8.0], 'coa_phase': [np.pi-1e-5],
                                 'sigmasq': [1.0], 'chisq': [0.5], 'chisq_dof': [30],
                                 'sg_chisq': [0.5]}, 'nonfinite_fields': {'snr': 0}}


def test_circular_phase_and_exact_peak_checks():
    reference = trigger()
    actual = copy.deepcopy(reference)
    actual['fields']['coa_phase'] = [-np.pi+1e-5]
    assert w.compare_triggers(actual, reference)['passed']
    actual['fields']['peak_index'] = [26]
    assert not w.compare_triggers(actual, reference)['passed']


def test_empty_and_nonfinite_trigger_outputs_fail():
    empty = {'count': 0, 'fields': {'template_id': []}}
    assert not w.compare_triggers(empty, empty)['passed']
    actual = trigger()
    actual['fields']['snr'] = [None]
    assert not w.compare_triggers(actual, trigger())['passed']


def test_injection_displacement_is_a_failure():
    record = trigger()
    assert not w.validate_triggers(record, [{'template_id': 1000, 'peak_index': 25}])
    assert w.validate_triggers(record, [{'template_id': 1000, 'peak_index': 26}])


def test_capture_retains_each_group_before_shared_output_reuse():
    buffer = np.arange(16, dtype=np.float32).astype(np.complex64)
    filt = SimpleNamespace(tgroups=[[SimpleNamespace(id=1002), SimpleNamespace(id=1000)],
                                   [SimpleNamespace(id=1003), SimpleNamespace(id=1001)]],
                           mids=[0, 0], out_mem={0: SimpleNamespace(numpy=lambda: buffer)})
    args = SimpleNamespace(size=8, bank_size=4)
    output = np.zeros((1, 4, 8), dtype=np.complex64)
    result = {'failures': [], 'pointwise_blocks': []}
    capture = w.Capture(args, filt, output, None, result)
    capture.start_block(0)
    capture.capture(0, {}, [])
    buffer[:] += 100
    capture.capture(1, {}, [])
    capture.finish_block()
    assert not result['failures']
    np.testing.assert_array_equal(output[0, 2], np.arange(8))
    np.testing.assert_array_equal(output[0, 0], np.arange(8, 16))
    np.testing.assert_array_equal(output[0, 3], np.arange(8)+100)
    np.testing.assert_array_equal(output[0, 1], np.arange(8, 16)+100)
    assert result['pointwise_blocks'][0]['samples'] == 32


def test_capture_detects_skipped_group():
    args = SimpleNamespace(size=8, bank_size=4)
    filt = SimpleNamespace()
    result = {'failures': [], 'pointwise_blocks': []}
    capture = w.Capture(args, filt, None, None, result)
    capture.start_block(0)
    capture.finish_block()
    assert result['failures'] and not result['pointwise_blocks'][0]['processed_once']
