#!/usr/bin/env python3
# Copyright (C) 2026
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Cold-process and prepared native-provider measurements on a fixed fixture."""

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tools import verify_torchwave_gpu_search as verification
from tools.benchmarking.evidence import (
    array_hash, compare_execution_identity, execution_provenance,
    require_device, search_snapshot, synchronize, timed_command, write_receipt,
)
from pycbc.filter.gpu_search import (
    SearchEngine, SelectionPolicy, VetoManager, prepare_bank, bind_psd,
    prepare_power_chisq_plan,
)


def validate_timed_result(batches, reference, args, label):
    """Validate this drain's owned outputs outside its measured interval."""
    result = {'sample': label, 'passed': False}
    try:
        snapshot = search_snapshot(batches, args.batch_size)
        candidates = verification.extract_and_validate_gpu_candidates(batches, args.batch_size)
        rows = []
        for index in range(args.batch_size):
            check = verification.compare_template_candidates(candidates[index], reference[index],
                                                             template_idx=index)
            accepted = lambda entries: sorted(row['sample_idx'] for row in entries
                if float(verification.newsnr(abs(row['snr']), row['red_chisq'])) >= 5.)
            if accepted(candidates[index]) != accepted(reference[index]):
                raise ValueError(f'NewSNR accepted identities differ for template {index}')
            rows.append({'index': index, 'candidate_count': len(candidates[index]), **check})
        result.update(passed=True, rows=rows,
                      output_hashes={name: array_hash(values) for name, values in snapshot.items()})
    except (ValueError, RuntimeError, KeyError, TypeError) as exc:
        result['reason'] = str(exc)
    return result


def device_matches(requested, effective):
    requested, effective = torch.device(requested), torch.device(effective)
    return (requested.type == effective.type and
            (requested.index is None or requested.index == effective.index))


def validate_child_measurement(measurement, parent, provider, samples):
    """Parent qualification cannot substitute for qualification of timed children."""
    try:
        if measurement.get('status') != 'passed' or measurement.get('provider') != provider:
            raise ValueError('child status/route did not pass')
        if measurement.get('requested_device') != parent['requested_device'] or not device_matches(
                parent['requested_device'], measurement['effective_device']):
            raise ValueError('child effective device differs from requested device')
        if measurement.get('effective_dtype') != 'torch.' + parent['requested_dtype']:
            raise ValueError('child effective dtype differs from requested dtype')
        if measurement.get('manifest') != parent.get('physical_manifest'):
            raise ValueError('child physical manifest differs from parent')
        if not parent.get('geometry') or measurement.get('geometry') != parent['geometry']:
            raise ValueError('child filtering geometry differs from parent')
        diagnostics_key = 'provider_diagnostics' if provider == 'torchwave' else 'reference_provider_diagnostics'
        if measurement.get('provider_diagnostics') != parent.get(diagnostics_key):
            raise ValueError('child dispatch/effective parameters differ from parent')
        expected = parent['input_hashes']
        actual = measurement['input_hashes']
        for name in ('reference_waveforms', 'strain', 'psd'):
            if not expected.get(name) or expected[name] != actual.get(name):
                raise ValueError(f'child fixture differs from parent: {name}')
        waveform_key = 'native_waveforms' if provider == 'torchwave' else 'reference_waveforms'
        if actual.get('waveforms') != expected[waveform_key]:
            raise ValueError('child measured waveforms differ from parent qualified waveforms')
        checks = measurement.get('result_qualification', [])
        if len(checks) != samples + 1 or not all(row.get('passed') is True for row in checks):
            raise ValueError('not every timed child result passed qualification')
        count = parent['configuration']['batch_size']
        for check in checks:
            if ([row.get('index') for row in check.get('rows', [])] != list(range(count)) or
                    not check.get('output_hashes')):
                raise ValueError('child result qualification omitted rows or output hashes')
        if len(measurement.get('raw_prepared_submit_drain_sec', [])) != samples:
            raise ValueError('child timing sample count differs')
        compare_execution_identity(parent['provenance'], measurement['provenance'])
        return {'passed': True}
    except (ValueError, KeyError, TypeError) as exc:
        return {'passed': False, 'reason': str(exc)}


def worker(args, provider, samples):
    require_device(args.device, torch)
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / 'bank.hdf')
        manifest = verification.create_fixture_bank(path, args)
        # Independently fixed reference/strain fixture precedes every measured route.
        reference_tensor, reference, _ = verification.generate_bank(path, args, False)
        strain, psd, flow, fhigh, valid = verification.prepare_fixture(reference, args)
        policy = SelectionPolicy(snr_threshold=5.5, cluster_policy='symmetric', cluster_window=10)
        expected = {}
        for index, ref in enumerate(reference):
            norm = verification.sigmasq(ref, psd, low_frequency_cutoff=flow,
                                        high_frequency_cutoff=fhigh)
            ref_bins = verification.power_chisq_bins(ref, 16, psd, flow, fhigh)
            expected[index] = verification.compute_canonical_cpu_reference_template(
                ref, strain, psd, norm, flow, fhigh, ref_bins, valid, policy, template_idx=index)
        synchronize(args.device, torch)
        combined_start = time.perf_counter()
        tensor, rows, diagnostics = verification.generate_bank(path, args, provider == 'torchwave')
        generation_end = time.perf_counter()
        bank = prepare_bank(rows, tile_size=min(16, args.batch_size), device=args.device,
                            f_lower=flow, f_upper=fhigh)
        bound = bind_psd(bank, psd, device=args.device)
        bins = prepare_power_chisq_plan(bank, bound, num_bins=16, device=args.device)
        engine = SearchEngine(bank, policy, veto_manager=VetoManager(power_chisq_plan=bins),
                              device=args.device)
        synchronize(args.device, torch)
        setup_end = time.perf_counter()
        try:
            engine.submit(strain, bound, valid)
            drained = engine.drain()
            synchronize(args.device, torch)
            combined_end = time.perf_counter()
            checks = [validate_timed_result(drained, expected, args, 'first_drain')]
            raw = []
            for sample in range(samples):
                start = time.perf_counter()
                engine.submit(strain, bound, valid)
                drained = engine.drain()
                synchronize(args.device, torch)
                raw.append(time.perf_counter() - start)
                checks.append(validate_timed_result(drained, expected, args, f'prepared_{sample}'))
        finally:
            engine.close()
        dispatch_passed = (len(diagnostics) == args.batch_size and all(
            row.get('provider') == provider for row in diagnostics))
        device_passed = device_matches(args.device, str(tensor.device))
        dtype_passed = tensor.dtype == getattr(torch, args.dtype)
        passed = dispatch_passed and device_passed and dtype_passed and all(row['passed'] for row in checks)
        return {'schema_version': 2, 'provider': provider, 'status': 'passed' if passed else 'failed',
                'experiment_class': 'provider_pipeline', 'manifest': manifest,
                'geometry': {'transform_length': 2 * (args.flen - 1),
                             'sample_rate_hz': 2 * (args.flen - 1) * args.delta_f,
                             'duration_sec': 1 / args.delta_f, 'valid_interval': list(valid),
                             'flow': flow, 'fhigh': fhigh},
                'requested_device': args.device, 'effective_device': str(tensor.device),
                'effective_dtype': str(tensor.dtype), 'provider_diagnostics': diagnostics,
                'dispatch_gate': {'passed': dispatch_passed},
                'device_gate': {'passed': device_passed}, 'dtype_gate': {'passed': dtype_passed},
                'result_qualification': checks,
                'qualification_boundary': 'canonical reference prepared before timing; each actual drain checked after timer stop and before next submission',
                'generation_to_device_ready_sec': generation_end - combined_start,
                'bank_psd_bins_engine_setup_sec': setup_end - generation_end,
                'generation_through_first_drain_sec': combined_end - combined_start,
                'combined_boundary': 'bank construction/generation through first final drain; fixed strain/PSD already prepared',
                'raw_prepared_submit_drain_sec': raw, 'warmup_submissions': 1,
                'prepared_boundary': 'synchronized host submit/drain with candidate selection and PowerChisq',
                'input_hashes': {'waveforms': array_hash(tensor.detach().cpu().numpy()),
                                 'reference_waveforms': array_hash(reference_tensor.detach().cpu().numpy()),
                                 'strain': array_hash(strain.numpy()), 'psd': array_hash(psd.numpy())},
                'provenance': execution_provenance()}


def main(argv=None):
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument('--cold-runs', type=int, default=5)
    options.add_argument('--warm-samples', type=int, default=20)
    options.add_argument('--worker-provider', choices=['reference', 'torchwave'])
    extra, remaining = options.parse_known_args(argv)
    if not any(value == '--output' or value.startswith('--output=') for value in remaining):
        remaining += ['--output', 'artifacts/torchwave_pipeline_v2.json']
    args = verification.parse_args(remaining)
    if extra.cold_runs < 1 or extra.warm_samples < 1:
        raise ValueError('cold-runs and warm-samples must be positive')
    if extra.worker_provider:
        measurement = worker(args, extra.worker_provider, extra.warm_samples)
        write_receipt(args.output, measurement)
        return 0 if measurement['status'] == 'passed' else 1
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    gates = verification.qualify(args)
    report = {'schema_version': 2, 'status': gates['status'], 'verification': gates,
              'runs': [], 'speedup_claim': None,
              'cold_runs_per_provider': extra.cold_runs,
              'warm_samples_per_process': extra.warm_samples,
              'scope': 'provider preparation and prepared synthetic search; no live/offline CLI qualification'}
    if gates['status'] != 'skipped':
        for run in range(extra.cold_runs):
            # Alternate order to reduce order bias; one child at a time.
            providers = ['reference', 'torchwave'] if run % 2 == 0 else ['torchwave', 'reference']
            for provider in providers:
                receipt_path = output.with_name(f'{output.stem}.{provider}.{run}.json')
                log_path = receipt_path.with_suffix('.log')
                command = [sys.executable, str(Path(__file__).resolve()),
                           '--worker-provider', provider, '--warm-samples', str(extra.warm_samples),
                           '--approximant', args.approximant, '--batch-size', str(args.batch_size),
                           '--device', args.device, '--flen', str(args.flen),
                           '--delta-f', str(args.delta_f), '--dtype', args.dtype,
                           '--phase-order', str(args.phase_order), '--output', str(receipt_path)]
                run_result = timed_command(command, log_path)
                run_result['process_wall_time_scope'] = (
                    'whole child tool including fixture creation, canonical qualification, '
                    'per-result checks, provenance and output; not operational pipeline latency')
                if receipt_path.exists():
                    try:
                        measurement = json.loads(receipt_path.read_text())
                        run_result['measurement'] = measurement
                        run_result['child_qualification'] = validate_child_measurement(
                            measurement, gates, provider, extra.warm_samples)
                    except (ValueError, OSError) as exc:
                        run_result['child_qualification'] = {'passed': False, 'reason': str(exc)}
                if run_result['returncode'] != 0 or not run_result.get('child_qualification', {}).get('passed'):
                    report['status'] = 'failed'
                report['runs'].append(run_result)
    write_receipt(output, report)
    print(f"{report['status'].upper()}: {output}")
    return {'passed': 0, 'failed': 1, 'skipped': 2}[report['status']]


if __name__ == '__main__':
    sys.exit(main())
