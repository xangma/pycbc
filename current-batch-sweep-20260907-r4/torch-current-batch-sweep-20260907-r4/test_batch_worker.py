"""Pure evidence checks plus opt-in small public Torch CPU integration checks."""
import copy
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


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


def capture_fixture(reference=None):
    bank = np.zeros((4, 5), dtype=np.complex64)
    bank[:, 0] = np.arange(1, 5)
    material = {'bank': bank, 'psd': np.ones(5, np.float32),
                'blocks': [np.ones(5, np.complex64)], 'delta_f': 0.25}
    templates = []
    for i in range(4):
        corr = np.zeros(8, dtype=np.complex64)
        corr[0] = i+1
        templates.append(SimpleNamespace(id=1000+i, cout=SimpleNamespace(numpy=lambda c=corr: c)))
    buffer = np.zeros(16, dtype=np.complex64)
    filt = SimpleNamespace(tgroups=[[templates[2], templates[0]], [templates[3], templates[1]]],
                           mids=[0, 0], out_mem={0: SimpleNamespace(numpy=lambda: buffer)})
    args = SimpleNamespace(size=8, bank_size=4, num_blocks=1)
    output = np.zeros((1, 4, 8), dtype=np.complex64)
    oracle = np.zeros(output.shape, dtype=np.complex128)
    result = {'failures': [], 'pointwise_blocks': []}
    capture = w.Capture(args, filt, output, reference, result, oracle, material)
    return capture, buffer, output, oracle, result


def capture_group(capture, buffer, group):
    norms = []
    for j, template in enumerate(capture.filt.tgroups[group]):
        amplitude = template.id - 999
        buffer[j*8:(j+1)*8] = amplitude
        norms.append(w.normalization_record(1/amplitude, amplitude**2, 'test_observation'))
    capture.capture(group, {}, [], norms)


def test_capture_retains_each_group_before_shared_output_reuse():
    capture, buffer, output, oracle, result = capture_fixture()
    capture.start_block(0)
    capture_group(capture, buffer, 0)
    capture_group(capture, buffer, 1)
    capture.finish_block()
    assert not result['failures']
    for i in range(4):
        np.testing.assert_array_equal(output[0, i], np.full(8, i+1))
        np.testing.assert_array_equal(oracle[0, i], output[0, i])
    assert result['pointwise_blocks'][0]['samples'] == 32


def test_capture_detects_skipped_group():
    capture, buffer, _, _, result = capture_fixture()
    capture.start_block(0)
    capture_group(capture, buffer, 0)
    capture.finish_block()
    assert result['failures'] and not result['pointwise_blocks'][0]['processed_once']


def test_duplicate_group_capture_is_rejected():
    capture, buffer, *_ = capture_fixture()
    capture.start_block(0)
    capture_group(capture, buffer, 0)
    with pytest.raises(ValueError, match='duplicate'):
        capture_group(capture, buffer, 0)


def qualify(actual, oracle, reference=None, norm=None, oracle_norm=None, reference_norm=None):
    norm = norm or w.normalization_record(1, 1, 'test')
    oracle_norm = oracle_norm or w.normalization_record(1, 1, 'test')
    reference_norm = reference_norm or w.normalization_record(1, 1, 'test')
    return w.qualified_row_metrics(np.asarray(actual, np.complex128), np.asarray(oracle, np.complex128),
                                   np.asarray(reference if reference is not None else oracle, np.complex128),
                                   norm, oracle_norm, reference_norm)


@pytest.mark.parametrize('actual', [[-1], [np.exp(0.002j)]])
def test_equal_magnitude_wrong_phase_is_a_v2_failure(actual):
    assert abs(abs(actual[0]) - 1) < 1e-12
    assert qualify(actual, [1])['failed_samples'] == 1


def test_absolute_criterion_boundary_and_union_not_double_counted():
    limit = 1e-3
    actual = [np.nextafter(limit, 0), limit, np.nextafter(limit, np.inf)]
    result = qualify(actual, [0, 0, 0])
    assert result['failed_samples'] == 1
    assert result['normalized_oracle']['failed_samples'] == 1
    assert result['normalized_mkl_compatibility']['failed_samples'] == 1


def test_oracle_and_compatibility_are_both_required_and_raw_only_audit():
    assert qualify([0], [0], [0.002])['failed_samples'] == 1
    assert qualify([0], [0.002], [0])['failed_samples'] == 1
    # This deliberately fails the old raw envelope but passes the new units.
    result = qualify([2e-6], [0])
    assert result['failed_samples'] == 0
    assert result['raw_complex_v1_audit']['failed_samples'] == 1
    assert result['normwise_diagnostics']['classification'] == 'diagnostic_only_no_threshold'


def test_wrong_group_norm_and_wrong_sigma_fail_even_without_trigger():
    wrong_norm = w.normalization_record(0.5, 1, 'observed')
    assert qualify([1], [1], norm=wrong_norm)['failed_samples'] == 1
    wrong_sigma = w.normalization_record(1, 1.002, 'observed')
    result = qualify([0, 0], [0, 0], norm=wrong_sigma)
    assert result['failed_samples'] == 2 and not result['sigmasq_oracle']['passed']


@pytest.mark.parametrize('bad', [0, -1, np.nan, np.inf])
def test_invalid_normalizations_are_rejected(bad):
    with pytest.raises(ValueError):
        w.normalization_record(bad, 1, 'test')
    with pytest.raises(ValueError):
        w.normalization_record(1, bad, 'test')


def test_zero_signal_and_nonfinite_complex_values():
    assert qualify([0], [0])['failed_samples'] == 0
    assert qualify([np.nan, np.inf], [0, 0])['failed_samples'] == 2
    assert qualify([0], [np.nan])['failed_samples'] == 1
    with pytest.raises(ValueError, match='sigmasq'):
        w.oracle_normalization(np.zeros(2, np.complex64), np.ones(2), 1)


def test_independent_oracle_uses_double_components_before_product_and_power():
    bank = np.array([0, 1.0000001+1.0000002j, 0], np.complex64)
    strain = np.array([0, 0.9999999+1.0000001j, 0], np.complex64)
    psd = np.array([1, 3, 1], np.float32)
    norm = w.oracle_normalization(bank, psd, 0.25)
    b, s = complex(bank[1]), complex(strain[1])
    expected_sigma = (b.real*b.real + b.imag*b.imag) / 3
    assert norm['sigmasq'] == expected_sigma
    assert norm['sigmasq'] != float(np.sum(np.abs(bank)**2/psd, dtype=np.float64))
    # A single frequency has an analytic inverse; no second FFT implementation.
    expected = b.conjugate()*s * np.exp(2j*np.pi*np.arange(8)/8)
    np.testing.assert_allclose(w.oracle_row(bank, strain, 8), expected, rtol=1e-15, atol=1e-15)
    rounded = complex((bank.conj()*strain)[1])
    assert abs(expected[0] - rounded) > 1e-9


@pytest.mark.parametrize('name,dtype', [('reference', 'complex64'), ('oracle', 'complex128')])
def test_corrupted_row_hash_rejected(name, dtype):
    array = np.ones((1, 1, 8), dtype=dtype)
    digest = w.digest_bytes(array[0, 0])
    w.validated_row(array, 0, 0, digest, dtype, 8, name)
    array[0, 0, 3] *= -1
    with pytest.raises(ValueError, match=f'{name} row hash mismatch'):
        w.validated_row(array, 0, 0, digest, dtype, 8, name)


def test_normalization_hash_rejected():
    record = w.normalization_record(1, 1, 'test')
    record['norm'] = 2
    with pytest.raises(ValueError, match='hash'):
        w.validate_normalization(record)


@pytest.mark.parametrize('filename,bad', [('outputs', np.zeros((1, 1, 8), np.complex128)),
                                        ('oracle', np.zeros((1, 1, 8), np.complex64)),
                                        ('oracle', np.zeros((1, 1, 7), np.complex128))])
def test_reference_array_shape_and_dtype_rejected(tmp_path, filename, bad):
    np.save(tmp_path/'outputs.npy', np.ones((1, 1, 8), np.complex64))
    np.save(tmp_path/'oracle.npy', np.ones((1, 1, 8), np.complex128))
    np.save(tmp_path/(filename+'.npy'), bad)
    with pytest.raises(ValueError, match='shape/dtype'):
        w.load_reference_arrays(tmp_path, (1, 1, 8))


def test_near_threshold_and_near_tie_still_require_identical_triggers():
    reference = trigger()
    reference['fields']['snr'] = [5.5000001]
    actual = copy.deepcopy(reference)
    actual['fields']['snr'] = [5.4999999]
    assert w.compare_triggers(actual, reference)['passed']  # Numeric field only.
    actual['fields']['peak_index'] = [26]
    assert not w.compare_triggers(actual, reference)['passed']
    missing = {'count': 0, 'fields': {k: [] for k in reference['fields']}}
    assert not w.compare_triggers(missing, reference)['passed']


class ObservedFilter:
    def __init__(self, branch="scalar", fault=None):
        self.tgroups = [[SimpleNamespace(id=1000, delta_f=0.25, sigmasq=lambda psd: 4.0),
                         SimpleNamespace(id=1001, delta_f=0.25, sigmasq=lambda psd: 9.0)]]
        self.block_id = 0
        self.branch, self.fault = branch, fault
        self.norms = np.array([0.5, 1/3], np.float32)
        self.sigmasqs = np.array([4, 9], np.float32)

    def _process_batch(self):
        if self.block_id == len(self.tgroups):
            return None, None
        group_id = self.block_id
        tgroup = self.tgroups[group_id]
        if self.fault == "identity":
            tgroup = list(tgroup)
        self.block_id += 1
        result, veto_info = {}, []
        ondevice_result = ([], [], [], False) if self.branch == "ondevice" else None
        batch_magnitudes = np.ones(2) if self.branch == "bulk" else None
        if self.branch == "scalar":
            psd = object()
            for htilde in tgroup:
                if self.fault == "skip" and htilde.id == 1001:
                    continue
                sgm = htilde.sigmasq(psd)
                assert sgm > 0
        else:
            norms, sigmasqs = self.norms, self.sigmasqs
        if self.fault == "branch":
            del ondevice_result
        if self.fault == "abort":
            return False, []
        return result, veto_info


@pytest.mark.parametrize('route', ['branch_standard', 'torch_cpu'])
def test_scalar_observation_and_restoration_on_exception(route):
    filt = ObservedFilter()
    original = [t.sigmasq for t in filt.tgroups[0]]
    process = filt._process_batch
    with pytest.raises(RuntimeError, match='deliberate'):
        with w.NormalizationCapture(SimpleNamespace(route=route), filt, None) as obs:
            obs.begin_group(0)
            filt._process_batch()
            rows = obs.finish_group()
            assert [r['sigmasq'] for r in rows] == [4, 9]
            assert [r['norm'] for r in rows] == [0.5, 1/3]
            assert all(r['source'] == 'observed_scalar_sigmasq_callback' for r in rows)
            assert filt._process_batch() == (None, None)
            raise RuntimeError('deliberate')
    assert filt._process_batch == process
    assert [t.sigmasq for t in filt.tgroups[0]] == original
    assert sys.getprofile() is None and sys.gettrace() is None


@pytest.mark.parametrize('branch,source', [('bulk', 'observed_bulk_magnitude_return_locals'),
                                         ('ondevice', 'observed_ondevice_return_locals')])
def test_bulk_observation_copies_actual_arrays_without_reconstructing(branch, source):
    filt = ObservedFilter(branch)
    # Deliberately differs from a scalar double calculation to detect guessing.
    assert float(filt.norms[1]) != 1/3
    with w.NormalizationCapture(SimpleNamespace(route='torch_cpu'), filt, None) as obs:
        obs.begin_group(0)
        filt._process_batch()
        rows = obs.finish_group()
        np.testing.assert_array_equal([r['norm'] for r in rows], filt.norms)
        assert [r['sigmasq'] for r in rows] == [4, 9]
        assert all(r['source'] == source for r in rows)
        filt.norms[:] = 0
        assert rows[0]['norm'] == 0.5


@pytest.mark.parametrize('fault,reason', [('identity', 'wrong-group'), ('skip', 'missing scalar'),
                                        ('branch', 'unknown production'), ('abort', 'aborted')])
def test_observation_rejects_unqualified_paths_and_restores(fault, reason):
    filt = ObservedFilter(fault=fault)
    original = [t.sigmasq for t in filt.tgroups[0]]
    with pytest.raises(ValueError, match=reason):
        with w.NormalizationCapture(SimpleNamespace(route='torch_cpu'), filt, None) as obs:
            obs.begin_group(0)
            filt._process_batch()
    assert [t.sigmasq for t in filt.tgroups[0]] == original
    assert sys.getprofile() is None


def test_duplicate_and_missing_observations_rejected():
    filt = ObservedFilter()
    obs = w.NormalizationCapture(SimpleNamespace(route='torch_cpu'), filt, None)
    obs.begin_group(0)
    obs.observe_scalar(filt.tgroups[0][0], 4)
    with pytest.raises(ValueError, match='duplicate'):
        obs.observe_scalar(filt.tgroups[0][0], 4)
    with pytest.raises(ValueError, match='missing'):
        obs.finish_group()
    with pytest.raises(ValueError, match='overlapping'):
        obs.begin_group(0)


@pytest.mark.parametrize('kind', ['profile', 'trace'])
def test_existing_instrumentation_is_refused_without_clobbering(kind):
    filt = ObservedFilter()
    original = [t.sigmasq for t in filt.tgroups[0]]
    def hook(*_args):
        return hook
    setter, getter = (sys.setprofile, sys.getprofile) if kind == 'profile' else (sys.settrace, sys.gettrace)
    setter(hook)
    try:
        with pytest.raises(ValueError, match='existing trace/profile'):
            with w.NormalizationCapture(SimpleNamespace(route='torch_cpu'), filt, None):
                pytest.fail('must refuse existing instrumentation')
        assert getter() is hook
        assert [t.sigmasq for t in filt.tgroups[0]] == original
    finally:
        setter(None)


def reference_fixture():
    cap, buffer, output, oracle, result = capture_fixture()
    cap.start_block(0)
    for group in range(2):
        capture_group(cap, buffer, group)
    cap.finish_block()
    result.update(status='pass', mode='qualify', route='branch_standard', batch=1,
                  input_sha256='input', tolerance_policy=copy.deepcopy(w.POLICY),
                  policy_sha256=w.canonical_hash(w.POLICY), worker_sha256='worker',
                  source={'revision': w.EXPECTED_HEAD, 'file_sha256': {'file': 'hash'}, 'tracked_dirty': False})
    return cap, output, oracle, result


@pytest.mark.parametrize('key', ['status', 'mode', 'route', 'batch', 'input_sha256',
                               'tolerance_policy', 'policy_sha256', 'worker_sha256', 'source'])
def test_reference_identity_changes_rejected(key):
    cap, _, _, reference = reference_fixture()
    candidate = copy.deepcopy(reference)
    w.validate_reference_metadata(reference, candidate, cap.args)
    reference[key] = {} if key == 'source' else None
    with pytest.raises((ValueError, KeyError)):
        w.validate_reference_metadata(reference, candidate, cap.args)


@pytest.mark.parametrize('target', ['reference', 'oracle', 'normalization', 'oracle_norm', 'coverage'])
def test_candidate_capture_rejects_corrupted_reference_evidence(target):
    _, output, oracle, reference = reference_fixture()
    cap, buffer, _, _, _ = capture_fixture(reference)
    cap.output_map, cap.oracle_map = output, oracle
    if target == 'reference':
        output[0, 2, 0] += 1
    elif target == 'oracle':
        oracle[0, 2, 0] *= -1
    elif target == 'normalization':
        reference['pointwise_blocks'][0]['rows'][2]['normalization']['norm'] = 99
    elif target == 'oracle_norm':
        reference['pointwise_blocks'][0]['rows'][2]['oracle']['normalization'] = w.normalization_record(2, 1, 'test')
    else:
        reference['pointwise_blocks'][0]['rows'][2]['template_id'] = 1000
    cap.start_block(0)
    with pytest.raises(ValueError):
        capture_group(cap, buffer, 0)


@pytest.mark.skipif(not os.environ.get('PYCBC_R4_INTEGRATION_SOURCE'),
                    reason='set PYCBC_R4_INTEGRATION_SOURCE to the pinned PyCBC checkout')
@pytest.mark.parametrize('route,batch', [('torch_cpu', 1), ('torch_cpu', 8), ('branch_standard', 1)])
def test_public_default_normalization_path(route, batch, monkeypatch):
    # Runtime observation check; the local scalar case uses FFTW, not MKL qualification.
    root = Path(os.environ['PYCBC_R4_INTEGRATION_SOURCE']).resolve()
    assert w.source_identity(root)['revision'] == w.EXPECTED_HEAD
    monkeypatch.syspath_prepend(str(root))
    from tools.bench_production_live_batch import route_environment
    configured = route_environment(route, dict(os.environ))
    for key in set(os.environ) - set(configured):
        monkeypatch.delenv(key)
    for key, value in configured.items():
        monkeypatch.setenv(key, value)
    import torch
    import pycbc
    from pycbc import scheme
    from pycbc.filter import matchedfilter
    from pycbc.types import FrequencySeries
    from pycbc.io.record import FieldArray
    from pycbc.vetoes.sgchisq import SingleDetSGChisq
    assert Path(pycbc.__file__).resolve().is_relative_to(root)
    assert not matchedfilter._torch_ondevice_peaks_enabled('cpu')
    torch.set_num_threads(1)
    args = SimpleNamespace(size=4096, bank_size=16, seed=7102, num_blocks=2, batch=batch, route=route)
    bank, psd_np, sigma, blocks, geom, injections, _ = w.workload(args)
    table = FieldArray.from_kwargs(mass1=np.full(16, 10, np.float32), mass2=np.full(16, 10, np.float32),
                                   template_hash=np.arange(1000, 1016, dtype=np.int64))
    context = scheme.TorchScheme('cpu') if route == 'torch_cpu' else scheme.CPUScheme(num_threads=1)
    with context:
        if route == 'branch_standard':
            from pycbc.fft.backend_support import get_backend, set_backend
            set_backend(['fftw'])
            assert get_backend().__name__ == 'pycbc.fft.fftw'
        psd = FrequencySeries(psd_np, delta_f=geom['delta_f'], epoch=0)
        templates = []
        for i, values in enumerate(bank):
            template = FrequencySeries(values, delta_f=geom['delta_f'], epoch=0)
            template.id, template.params, template.f_lower = 1000+i, table[i], geom['f_lower']
            template.sigmasq = lambda p, value=float(sigma[i]): value
            templates.append(template)
        sg = SingleDetSGChisq(SimpleNamespace(table=table), '16', 0.0, ['mass1>0:8-20,12-40'])
        filt = matchedfilter.LiveBatchMatchedFilter(templates, snr_threshold=5.5, chisq_bins='16',
                                                   sg_chisq=sg, maxelements=batch*args.size)
        readers = []
        for block, values in enumerate(blocks):
            strain = FrequencySeries(values, delta_f=geom['delta_f'], epoch=0)
            strain.psd = psd
            readers.append(SimpleNamespace(overwhitened_data=lambda _df, s=strain: s,
                                           trim_padding=geom['trim_padding'], blocksize=geom['blocksize'],
                                           sample_rate=geom['sample_rate'],
                                           start_time=geom['reader_start_time'] + block*geom['blocksize']))
        baseline = [w.trigger_record(filt.process_data(reader), geom, b) for b, reader in enumerate(readers)]
        outputs = np.zeros((2, 16, 4096), np.complex64)
        oracle = np.zeros(outputs.shape, np.complex128)
        result = {'failures': [], 'pointwise_blocks': []}
        cap = w.Capture(args, filt, outputs, None, result, oracle,
                        {'bank': bank, 'psd': psd_np, 'blocks': blocks, 'delta_f': geom['delta_f']})
        original = filt._process_batch
        callbacks = [t.sigmasq for t in templates]
        with w.NormalizationCapture(args, filt, matchedfilter) as obs:
            def wrapped():
                group = filt.block_id
                if group == len(filt.tgroups):
                    return original()
                obs.begin_group(group)
                triggers, veto = original()
                cap.capture(group, triggers, veto, obs.finish_group())
                return triggers, veto
            filt._process_batch = wrapped
            for b, reader in enumerate(readers):
                cap.start_block(b)
                raw = filt.process_data(reader)
                cap.finish_block()
                recorded = w.trigger_record(raw, geom, b, cap.indices)
                assert w.compare_triggers(recorded, baseline[b])['passed']
                assert not w.validate_triggers(recorded, injections[b])
                assert w.sg_activation(sg, {t.id: t for t in templates}, psd, recorded)
        assert not result['failures']
        assert filt._process_batch == original and [t.sigmasq for t in templates] == callbacks
        assert sys.getprofile() is None
        expected_source = ('observed_scalar_sigmasq_callback' if route == 'branch_standard'
                           else 'observed_bulk_magnitude_return_locals')
        for block in result['pointwise_blocks']:
            assert block['processing_counts'] == [1]*16
            assert block['samples'] == 16*4096
            assert {row['normalization']['source'] for row in block['rows']} == {expected_source}
            assert all(row['sigmasq_oracle']['passed'] for row in block['rows'])
