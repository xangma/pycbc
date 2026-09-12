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
    array_hash, execution_provenance, require_device, synchronize,
    timed_command, write_receipt,
)
from pycbc.filter.gpu_search import (
    SearchEngine, SelectionPolicy, VetoManager, prepare_bank, bind_psd,
    prepare_power_chisq_plan,
)


def worker(args, provider, samples):
    require_device(args.device, torch)
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / 'bank.hdf')
        manifest = verification.create_fixture_bank(path, args)
        # Independently fixed reference/strain fixture precedes every measured route.
        _, reference, _ = verification.generate_bank(path, args, False)
        strain, psd, flow, fhigh, valid = verification.prepare_fixture(reference, args)
        synchronize(args.device, torch)
        combined_start = time.perf_counter()
        tensor, rows, diagnostics = verification.generate_bank(path, args, provider == 'torchwave')
        generation_end = time.perf_counter()
        bank = prepare_bank(rows, tile_size=min(16, args.batch_size), device=args.device,
                            f_lower=flow, f_upper=fhigh)
        bound = bind_psd(bank, psd, device=args.device)
        bins = prepare_power_chisq_plan(bank, bound, num_bins=16, device=args.device)
        policy = SelectionPolicy(snr_threshold=5.5, cluster_policy='symmetric', cluster_window=10)
        engine = SearchEngine(bank, policy, veto_manager=VetoManager(power_chisq_plan=bins),
                              device=args.device)
        synchronize(args.device, torch)
        setup_end = time.perf_counter()
        try:
            engine.submit(strain, bound, valid)
            engine.drain()
            synchronize(args.device, torch)
            combined_end = time.perf_counter()
            raw = []
            for _ in range(samples):
                start = time.perf_counter()
                engine.submit(strain, bound, valid)
                engine.drain()
                synchronize(args.device, torch)
                raw.append(time.perf_counter() - start)
        finally:
            engine.close()
        return {'schema_version': 2, 'provider': provider,
                'experiment_class': 'provider_pipeline', 'manifest': manifest,
                'requested_device': args.device, 'effective_device': str(tensor.device),
                'effective_dtype': str(tensor.dtype), 'provider_diagnostics': diagnostics,
                'generation_to_device_ready_sec': generation_end - combined_start,
                'bank_psd_bins_engine_setup_sec': setup_end - generation_end,
                'generation_through_first_drain_sec': combined_end - combined_start,
                'combined_boundary': 'bank construction/generation through first final drain; fixed strain/PSD already prepared',
                'raw_prepared_submit_drain_sec': raw, 'warmup_submissions': 1,
                'prepared_boundary': 'synchronized host submit/drain with candidate selection and PowerChisq',
                'input_hashes': {'waveforms': array_hash(tensor.detach().cpu().numpy()),
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
        write_receipt(args.output, worker(args, extra.worker_provider, extra.warm_samples))
        return 0
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
                if run_result['returncode'] == 0:
                    run_result['measurement'] = json.loads(receipt_path.read_text())
                else:
                    report['status'] = 'failed'
                report['runs'].append(run_result)
    write_receipt(output, report)
    print(f"{report['status'].upper()}: {output}")
    return {'passed': 0, 'failed': 1, 'skipped': 2}[report['status']]


if __name__ == '__main__':
    sys.exit(main())
