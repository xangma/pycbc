#!/usr/bin/env python3
"""Qualified, sparse candidate/veto trace and isolated CUDA allocation evidence.

Run from the final integrated checkout. This is an instrumented fixture, not a
throughput benchmark, production workload, or proof of fully asynchronous work.
"""

import argparse
from collections import Counter
from contextlib import ExitStack
import gc
import hashlib
import json
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from pycbc.events.ranking import newsnr
from pycbc.filter.gpu_search import (
    SearchEngine, SelectionPolicy, VetoManager, bind_psd, prepare_bank,
    prepare_power_chisq_plan,
)
import pycbc.filter.gpu_search.engine as engine_module
import pycbc.filter.gpu_search.vetoes as veto_module
from pycbc.filter.matchedfilter import sigmasq
from pycbc.types import FrequencySeries
from pycbc.vetoes.chisq import power_chisq_bins
from pycbc.waveform import get_fd_waveform
from tools.benchmarking.benchmark_gpu_search import (
    compare_template_candidates, compute_canonical_cpu_reference_template,
    extract_and_validate_gpu_candidates,
)
from tools.benchmarking.evidence import (
    DeviceUnavailable, array_hash, execution_provenance, require_device,
    search_snapshot, write_receipt,
)


def owned(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: owned(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(owned(val) for val in value)
    return value


def fixture(size):
    sample_rate, flow, fhigh = 2048., 30., 800.
    df, flen = sample_rate / size, size // 2 + 1
    psd = FrequencySeries(np.ones(flen, dtype=np.float32), delta_f=df)
    rows, manifest = [], []
    for index, (mass1, mass2) in enumerate(((20., 10.), (23., 11.))):
        params = dict(approximant='TaylorF2', mass1=mass1, mass2=mass2,
                      spin1z=0., spin2z=0., delta_f=df, f_lower=flow,
                      f_final=fhigh, phase_order=-1)
        hp, _ = get_fd_waveform(**params)
        values = np.zeros(flen, dtype=np.complex128)
        length = min(flen, len(hp))
        values[:length] = np.asarray(hp)[:length]
        # Scale in FP64 before complex64 casting to avoid physical underflow.
        lo, hi = int(flow / df), int(fhigh / df)
        scale = np.sqrt(4 * df * np.sum(np.abs(values[lo:hi]) ** 2))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('invalid reference waveform norm')
        row = FrequencySeries((values / scale).astype(np.complex64), delta_f=df)
        row.id, row.f_lower = index, flow
        rows.append(row)
        manifest.append(dict(params, amplitude_divisor=float(scale)))
    rng = np.random.default_rng(20260912)
    data = (.02 * (rng.normal(size=flen) + 1j * rng.normal(size=flen))).astype(np.complex64)
    injections = []
    for row, sample in zip(rows, (size // 3, 2 * size // 3)):
        norm = float(sigmasq(row, psd, low_frequency_cutoff=flow,
                             high_frequency_cutoff=fhigh))
        phase = np.exp(-2j * np.pi * np.arange(flen) * sample / size)
        data += np.asarray(row) * phase * (20. / np.sqrt(norm))
        injections.append({'template_id': row.id, 'sample': sample, 'target_snr': 20.})
    data[0] = data[-1] = 0
    strain = FrequencySeries(data, delta_f=df)
    return rows, strain, psd, flow, fhigh, manifest, injections


def small_direct_gate(device):
    """Independent bin-local NumPy sums, with no shared cumulative algorithm."""
    size, count = 2048, 4
    rng = np.random.default_rng(7103)
    corr = (.01 * (rng.normal(size=(2, size // 2 + 1)) +
                   1j * rng.normal(size=(2, size // 2 + 1)))).astype(np.complex64)
    edges = np.array([[7, 28, 90, 90, 410, 1000],
                      [3, 18, 205, 512, 820, 1010]], dtype=np.int64)
    norms = np.array([.3, .7], dtype=np.float64)
    ti, si = np.array([0, 1, 0, 1]), np.array([5, 727, 1024, 1998])
    expected, snr = [], []
    for template, sample in zip(ti, si):
        bins = []
        for low, high in zip(edges[template, :-1], edges[template, 1:]):
            frequencies = np.arange(low, high, dtype=np.int64)
            phase = np.exp(2j * np.pi * ((sample * frequencies) % size) / size)
            bins.append(np.sum(corr[template, low:high].astype(np.complex128) * phase))
        normalized = np.sum(bins) * norms[template]
        snr.append(normalized)
        expected.append(max(0., 5 * np.sum(np.abs(bins) ** 2) * norms[template] ** 2 - abs(normalized) ** 2))
    got, dof = veto_module.batched_power_chisq(
        torch.as_tensor(corr, device=device),
        {'template_idx': torch.as_tensor(ti, device=device),
         'sample_idx': torch.as_tensor(si, device=device),
         'snr': torch.as_tensor(np.array(snr), device=device)},
        torch.as_tensor(edges, device=device), torch.as_tensor(norms, device=device),
        num_bins=5, transform_length=size, scratch_budget_bytes=64 * 1024,
        return_device=True,
    )
    actual, actual_dof = got.cpu().numpy(), dof.cpu().numpy()
    if not np.allclose(actual, expected, rtol=1e-5, atol=1e-6) or not np.array_equal(actual_dof, np.full(count, 8)):
        raise AssertionError('small independent direct-sum PowerChisq gate failed')
    return {'passed': True, 'transform_length': size, 'candidates': count,
            'max_abs_chisq_error': float(np.max(np.abs(actual - expected))),
            'expected_chisq': expected, 'actual_chisq': actual.tolist(),
            'correlation_hash': array_hash(corr)}


def qualify_result(tickets, reference):
    snapshot = search_snapshot(tickets, 2)
    actual = extract_and_validate_gpu_candidates(tickets, 2)
    result = []
    for index in range(2):
        comparison = compare_template_candidates(actual[index], reference[index], template_idx=index)
        accepted = lambda entries: sorted(row['sample_idx'] for row in entries
            if float(newsnr(abs(row['snr']), row['red_chisq'])) >= 5.)
        if accepted(actual[index]) != accepted(reference[index]):
            raise AssertionError('NewSNR >= 5 accepted identity mismatch')
        result.append(dict(comparison, template_id=index,
                           candidate_count=len(actual[index]),
                           accepted_samples=accepted(actual[index])))
    count = sum(len(values) for values in actual.values())
    if not 0 < count <= 64:
        raise ValueError(f'fixture must remain sparse: observed {count} candidates')
    return {'passed': True, 'candidate_count': count, 'rows': result,
            'result_hashes': {key: array_hash(value) for key, value in snapshot.items()}}


def memory_evidence(inputs, device, samples):
    records, baseline_result = [], None
    for budget in (64 * 1024, 64 * 1024 * 1024):
        call = dict(inputs, scratch_budget_bytes=budget, return_device=True)
        warm = veto_module.batched_power_chisq(**call)
        torch.cuda.synchronize(device)
        values = tuple(value.cpu().numpy().copy() for value in warm)
        if baseline_result is None:
            baseline_result = values
        elif not (np.allclose(values[0], baseline_result[0], rtol=1e-5, atol=1e-3)
                  and np.array_equal(values[1], baseline_result[1])):
            raise AssertionError('scratch configurations disagree')
        del warm
        for repetition in range(samples):
            gc.collect()
            torch.cuda.synchronize(device)
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)
            result = veto_module.batched_power_chisq(**call)
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            reserved_peak = torch.cuda.max_memory_reserved(device)
            actual = tuple(value.cpu().numpy().copy() for value in result)
            if not (np.allclose(actual[0], baseline_result[0], rtol=1e-5, atol=1e-3)
                    and np.array_equal(actual[1], baseline_result[1])):
                raise AssertionError('measured veto output changed')
            records.append({'scratch_budget_bytes': budget, 'repetition': repetition,
                            'baseline_allocated_bytes': allocated,
                            'peak_allocated_bytes': peak,
                            'peak_minus_baseline_allocated_bytes': peak - allocated,
                            'baseline_reserved_bytes': reserved,
                            'peak_reserved_bytes': reserved_peak,
                            'output_hashes': [array_hash(value) for value in actual]})
            del result
    return {'records': records,
            'scope': 'Isolated batched_power_chisq call after warmup, inputs resident before baseline; includes result/index arrays and backend allocations.',
            'budget_exclusions': 'Configured scratch bound excludes caller inputs, O(candidate count) result/index arrays, normalized SNR copies, allocator caches and backend-internal workspace; peak-minus-baseline need not equal that bound.',
            'allocator_policy': 'No empty_cache; reserved memory is reported separately; synchronization brackets the isolated measurement.'}


def instrumented(function, label):
    def wrapped(*args, **kwargs):
        with torch.profiler.record_function(label):
            return function(*args, **kwargs)
    return wrapped


def profiler_summary(profiler, trace_path):
    events = []
    for event in profiler.key_averages():
        events.append({'key': event.key, 'count': int(event.count),
                       'cpu_time_total_us': float(event.cpu_time_total),
                       'self_cpu_time_total_us': float(event.self_cpu_time_total),
                       'device_time_total_us': float(getattr(event, 'device_time_total', 0.)),
                       'self_device_time_total_us': float(getattr(event, 'self_device_time_total', 0.))})
    trace = json.loads(trace_path.read_text())
    copies = Counter()
    for event in trace.get('traceEvents', []):
        name = event.get('name', '')
        if 'memcpy' in name.lower() or 'memset' in name.lower():
            copies[(event.get('cat', ''), name)] += 1
    selected = [entry for entry in events if entry['key'].startswith(('stage::', 'python::'))
                or any(word in entry['key'].lower() for word in
                       ('nonzero', 'item', '_local_scalar_dense', 'copy', 'memcpy'))
                or entry['key'] == 'aten::to']
    return {'selected_operation_counts_and_times': selected,
            'all_operator_counts_and_times': events,
            'chrome_copy_events': [{'category': category, 'name': name, 'count': count}
                                   for (category, name), count in sorted(copies.items())],
            'interpretation': 'Tensor.cpu/numpy/item entries count instrumented Python calls; aten and CUDA entries count observed operators/runtime events. A cpu/to/copy call alone does not establish transfer direction or synchronization. Trace contains stream and memcpy details. Instrumentation adds overhead; timings are not throughput claims.'}


def run(args, receipt):
    device = require_device(args.device, torch)
    if torch.device(device).type != 'cuda':
        raise ValueError('CUDA is required for this allocation/trace artifact')
    receipt['effective_device'] = device
    receipt['small_direct_gate'] = small_direct_gate(device)
    rows, strain, psd, flow, fhigh, manifest, injections = fixture(args.size)
    policy = SelectionPolicy(snr_threshold=5.5, cluster_policy='symmetric', cluster_window=512)
    valid = (args.size // 8, args.size - args.size // 8)
    reference = []
    for index, row in enumerate(rows):
        norm = float(sigmasq(row, psd, low_frequency_cutoff=flow, high_frequency_cutoff=fhigh))
        bins = power_chisq_bins(row, 16, psd, flow, fhigh)
        reference.append(compute_canonical_cpu_reference_template(
            row, strain, psd, norm, flow, fhigh, bins, valid, policy, template_idx=index))
    if not 0 < sum(map(len, reference)) <= 64:
        raise ValueError('canonical fixture is not sparse')
    receipt.update({'physical_manifest': manifest, 'injections': injections,
                    'geometry': {'transform_length': args.size, 'batch_size': 2,
                                 'sample_rate_hz': 2048, 'delta_f': strain.delta_f,
                                 'valid_interval': list(valid), 'flow': flow, 'fhigh': fhigh},
                    'input_hashes': {'templates': array_hash(np.stack([np.asarray(row) for row in rows])),
                                     'strain': array_hash(np.asarray(strain)), 'psd': array_hash(np.asarray(psd))}})
    bank = prepare_bank(rows, tile_size=2, device=device, f_lower=flow, f_upper=fhigh)
    bound = bind_psd(bank, psd, device=device)
    bins = prepare_power_chisq_plan(bank, bound, num_bins=16, device=device)
    engine = SearchEngine(bank, policy, veto_manager=VetoManager(power_chisq_plan=bins),
                          candidate_capacity=128, device=device, use_cuda_graphs=False)
    captured = []
    original_veto = veto_module.batched_power_chisq
    def capture(*positional, **kwargs):
        if positional:
            raise TypeError('expected keyword arguments from VetoManager')
        captured.append(owned(kwargs))
        return original_veto(**kwargs)
    try:
        with patch.object(veto_module, 'batched_power_chisq', capture):
            engine.submit(strain, bound, valid)
            tickets = engine.drain()
        receipt['preprofile_gate'] = qualify_result(tickets, reference)
        if len(captured) != 1:
            raise AssertionError('expected one actual veto call for the single tile')
        receipt['isolated_veto_memory'] = memory_evidence(captured[0], device, args.memory_samples)
        trace_path = Path(args.output_dir) / 'candidate_pipeline.trace.json'
        torch.cuda.synchronize(device)
        with ExitStack() as stack:
            for module, name, label in (
                    (engine_module, 'select_tile_candidates', 'stage::candidate_selection'),
                    (veto_module, 'batched_power_chisq', 'stage::power_chisq'),
                    (engine_module, 'candidates_to_host', 'stage::host_materialization')):
                stack.enter_context(patch.object(module, name, instrumented(getattr(module, name), label)))
            for method in ('cpu', 'numpy', 'item'):
                stack.enter_context(patch.object(torch.Tensor, method,
                    instrumented(getattr(torch.Tensor, method), f'python::Tensor.{method}')))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=True, profile_memory=True) as prof:
                measured = []
                for repetition in range(args.samples):
                    with torch.profiler.record_function(f'stage::submit_{repetition}'):
                        engine.submit(strain, bound, valid)
                    with torch.profiler.record_function(f'stage::public_drain_{repetition}'):
                        measured.append(engine.drain())
                torch.cuda.synchronize(device)
        prof.export_chrome_trace(str(trace_path))
        receipt['profiled_result_checks'] = [qualify_result(result, reference) for result in measured]
        receipt['profiler'] = profiler_summary(prof, trace_path)
        receipt['trace'] = {'path': str(trace_path.resolve()),
                            'sha256': hashlib.sha256(trace_path.read_bytes()).hexdigest()}
        receipt['status'] = 'passed'
    finally:
        engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--size', type=int, default=131072)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--memory-samples', type=int, default=3)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    if args.size != 131072 or not 1 <= args.samples <= 5 or not 1 <= args.memory_samples <= 5:
        parser.error('this bounded artifact requires N=131072 and 1..5 repetitions')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    receipt = {'schema_version': 2, 'status': 'failed', 'configuration': vars(args),
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'scope': 'Two scaled reference TaylorF2 templates, fixed synthetic sparse injection fixture, native PowerChisq, eager single tile. SG veto, graphs, live/offline CLI and waveform generation are outside the traced region. No asynchronous or production qualification claim.'}
    code = 1
    try:
        run(args, receipt)
        code = 0
    except DeviceUnavailable as error:
        receipt.update(status='skipped_unavailable', error=str(error))
        code = 2
    except Exception as error:
        receipt.update(error=str(error), traceback=traceback.format_exc())
    finally:
        receipt['provenance'] = execution_provenance()
        write_receipt(output / 'candidate_pipeline_profile.json', receipt)
    print(json.dumps({'status': receipt['status'], 'receipt': str((output / 'candidate_pipeline_profile.json').resolve())}))
    return code


if __name__ == '__main__':
    sys.exit(main())
