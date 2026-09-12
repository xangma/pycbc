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

"""Fixture-scoped TorchWave provider/search checks; never catalog certification."""

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

# Direct execution must use this checkout, including from an isolated worktree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import torch

from tools.benchmarking.evidence import (
    DeviceUnavailable, array_error, array_hash, execution_provenance,
    require_device, synchronize, write_receipt,
)
from tools.benchmarking.benchmark_gpu_search import (
    compare_template_candidates, compute_canonical_cpu_reference_template,
    extract_and_validate_gpu_candidates,
)
from pycbc.filter.gpu_search import (
    SearchEngine, SelectionPolicy, VetoManager, prepare_bank, bind_psd,
    prepare_power_chisq_plan,
)
from pycbc.filter.matchedfilter import matched_filter_core, sigmasq
from pycbc.types import FrequencySeries
from pycbc.waveform.bank import FilterBank
from pycbc.vetoes.chisq import power_chisq_bins
from pycbc.events.ranking import newsnr
from pycbc.scheme import TorchScheme


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approximant', default='TaylorF2')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--delta-f', type=float, default=0.5)
    parser.add_argument('--flen', type=int, default=1025)
    parser.add_argument('--dtype', choices=['complex64', 'complex128'], default='complex64')
    parser.add_argument('--phase-order', type=int, default=-1)
    parser.add_argument('--output', default='artifacts/torchwave_verification_v2.json')
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.flen < 65 or not np.isfinite(args.delta_f) or args.delta_f <= 0:
        parser.error('positive batch/grid required, with flen >= 65')
    if (args.flen - 1) * args.delta_f <= 40:
        parser.error('Nyquist frequency must exceed 40 Hz')
    return args


def create_fixture_bank(path, args):
    """One immutable physical manifest used by both waveform providers."""
    count = args.batch_size
    parameters = {
        'mass1': np.linspace(10., 30., count),
        'mass2': np.linspace(5., 10., count),
        'spin1z': np.zeros(count), 'spin2z': np.zeros(count),
        'f_lower': np.full(count, 30.),
    }
    with h5py.File(path, 'w') as output:
        for name, values in parameters.items():
            output[name] = values
        output.attrs['parameters'] = list(parameters)
    return {name: values.tolist() for name, values in parameters.items()}


def generate_bank(path, args, native):
    """Construction through device-ready rows; includes metadata/packing."""
    bank = FilterBank(path, args.flen, args.delta_f,
                      dtype=np.dtype(args.dtype), approximant=args.approximant,
                      phase_order=args.phase_order, enable_torchwave=native)
    indices = list(range(args.batch_size))
    if native:
        with TorchScheme(device=args.device):
            tensor, rows = bank.get_batch_tensor(
                indices, device=args.device, dtype=getattr(torch, args.dtype))
        diagnostics = bank.torchwave_diagnostics(indices, device=args.device)
    else:
        rows = [bank[i].copy() for i in indices]
        tensor = torch.as_tensor(np.stack([row.numpy() for row in rows]),
                                 dtype=getattr(torch, args.dtype), device=args.device)
        diagnostics = [{'index': i, 'provider': 'reference', 'reason': 'requested reference'}
                       for i in indices]
    for row in diagnostics:
        params, flen = bank._waveform_parameters(row['index'])
        # The resolved manifest includes global options, row values and bank grid.
        row['effective_parameters'] = json.loads(json.dumps(params, default=lambda value:
            value.item() if isinstance(value, np.generic) else str(value)))
        row['filter_length'] = flen
    synchronize(args.device, torch)
    return tensor, rows, diagnostics


def prepare_fixture(reference, args):
    """Strain is generated once, with an injection from the reference only."""
    size = 2 * (args.flen - 1)
    flow, fhigh = 30., min(500., (args.flen - 1) * args.delta_f)
    real_dtype = np.float64 if args.dtype == 'complex128' else np.float32
    psd = FrequencySeries(np.full(args.flen, 2., real_dtype), delta_f=args.delta_f)
    rng = np.random.default_rng(20260912)
    data = (rng.normal(size=args.flen) + 1j * rng.normal(size=args.flen)).astype(args.dtype)
    data[0] = 0
    norm0 = sigmasq(reference[0], psd, low_frequency_cutoff=flow,
                   high_frequency_cutoff=fhigh)
    shift = np.exp(-2j * np.pi * np.arange(args.flen) * (size // 2) / size)
    data += np.asarray(reference[0]) * shift * (25. / np.sqrt(norm0))
    strain = FrequencySeries(data, delta_f=args.delta_f)
    return strain, psd, flow, fhigh, (size // 8, size - size // 8)


def engine_candidates(rows, strain, psd, args, flow, fhigh, valid):
    policy = SelectionPolicy(snr_threshold=5.5, cluster_policy='symmetric', cluster_window=10)
    bank = prepare_bank(rows, tile_size=min(16, len(rows)), device=args.device,
                        f_lower=flow, f_upper=fhigh)
    bound = bind_psd(bank, psd, device=args.device)
    bins = prepare_power_chisq_plan(bank, bound, num_bins=16, device=args.device)
    engine = SearchEngine(bank, policy, veto_manager=VetoManager(power_chisq_plan=bins),
                          device=args.device)
    try:
        engine.submit(strain, bound, valid)
        candidates = extract_and_validate_gpu_candidates(engine.drain(), len(rows))
    finally:
        engine.close()
    return candidates, policy


def qualify(args):
    """Run every row; return failed gates rather than unconditional success."""
    receipt = {'schema_version': 2, 'status': 'failed',
               'scope': 'one explicit bank/strain fixture; PowerChisq enabled',
               'experiment_class': 'provider_pipeline',
               'requested_device': args.device, 'effective_device': None,
               'requested_provider': 'torchwave', 'requested_dtype': args.dtype,
               'configuration': vars(args).copy(), 'gates': [],
               'provenance': execution_provenance()}
    try:
        receipt['effective_device'] = require_device(args.device, torch)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'bank.hdf')
            receipt['physical_manifest'] = create_fixture_bank(path, args)
            reference_tensor, reference, reference_diagnostics = generate_bank(path, args, False)
            native_tensor, _, diagnostics = generate_bank(path, args, True)
        receipt['provider_diagnostics'] = diagnostics
        receipt['reference_provider_diagnostics'] = reference_diagnostics
        receipt['effective_device'] = str(native_tensor.device)
        receipt['effective_dtype'] = str(native_tensor.dtype)
        receipt['engine_filter_dtype'] = 'complex64'
        # Own host copies for independent reference computations; no live buffer aliases.
        native = [FrequencySeries(row.detach().cpu().numpy().copy(), delta_f=args.delta_f)
                  for row in native_tensor]
        receipt['gates'].append({'name': 'native_dispatch', 'passed':
                                len(diagnostics) == args.batch_size and all(
                                    row.get('provider') == 'torchwave' for row in diagnostics)})
        receipt['gates'].append({'name': 'requested_dtype', 'passed':
                                 native_tensor.dtype == getattr(torch, args.dtype)})
        requested = torch.device(args.device)
        receipt['gates'].append({'name': 'requested_device', 'passed':
                                requested.type == native_tensor.device.type and
                                (requested.index is None or requested.index == native_tensor.device.index)})
        strain, psd, flow, fhigh, valid = prepare_fixture(reference, args)
        receipt['input_hashes'] = {'reference_waveforms': array_hash(reference_tensor.cpu().numpy()),
                                   'native_waveforms': array_hash(native_tensor.cpu().numpy()),
                                   'strain': array_hash(strain.numpy()), 'psd': array_hash(psd.numpy())}
        receipt['geometry'] = {'transform_length': 2 * (args.flen - 1),
                               'sample_rate_hz': 2 * (args.flen - 1) * args.delta_f,
                               'duration_sec': 1 / args.delta_f, 'valid_interval': list(valid),
                               'flow': flow, 'fhigh': fhigh}
        candidates, policy = engine_candidates(native, strain, psd, args, flow, fhigh, valid)
        rows = []
        for index, (ref, got) in enumerate(zip(reference, native)):
            ref_norm = float(sigmasq(ref, psd, low_frequency_cutoff=flow, high_frequency_cutoff=fhigh))
            got_norm = float(sigmasq(got, psd, low_frequency_cutoff=flow, high_frequency_cutoff=fhigh))
            waveform = array_error(ref.numpy(), got.numpy(), rtol=1e-4)
            start, stop = int(flow / args.delta_f), int(fhigh / args.delta_f)
            weight = np.sqrt(psd.numpy()[start:stop])
            weighted_waveform = array_error(ref.numpy()[start:stop] / weight,
                                             got.numpy()[start:stop] / weight, rtol=1e-4)
            norms = array_error([ref_norm], [got_norm], rtol=1e-4)
            norms['passed'] = norms['passed'] and ref_norm > 0 and got_norm > 0
            ref_snr, _, rn = matched_filter_core(ref, strain, psd, flow, fhigh, h_norm=ref_norm)
            got_snr, _, gn = matched_filter_core(got, strain, psd, flow, fhigh, h_norm=got_norm)
            # Apply the established absolute complex-SNR budget to every sample.
            expected, actual = np.asarray(ref_snr) * rn, np.asarray(got_snr) * gn
            series = array_error(expected, actual, rtol=0, atol=1e-3)
            series['passed'] = bool(np.all(np.isfinite(expected)) and
                                    np.all(np.isfinite(actual)) and
                                    np.all(np.abs(actual - expected) <= 1e-3))
            series['criterion'] = 'every complex sample absolute error <= 0.001'
            bins = power_chisq_bins(ref, 16, psd, flow, fhigh)
            ref_candidates = compute_canonical_cpu_reference_template(
                ref, strain, psd, ref_norm, flow, fhigh, bins, valid, policy,
                template_idx=index)
            candidate_gate = {'passed': True}
            try:
                candidate_gate.update(compare_template_candidates(candidates[index], ref_candidates,
                                                                   template_idx=index))
                accepted = lambda entries: [row['sample_idx'] for row in entries
                    if float(newsnr(abs(row['snr']), row['red_chisq'])) >= 5.]
                if sorted(accepted(candidates[index])) != sorted(accepted(ref_candidates)):
                    raise RuntimeError('NewSNR accepted identities differ')
            except (ValueError, RuntimeError) as exc:
                candidate_gate = {'passed': False, 'reason': str(exc)}
            rows.append({'index': index, 'waveform_complex': waveform, 'original_norm': norms,
                         'waveform_psd_weighted': weighted_waveform,
                         'provider_snr_series_cpu_reference': series,
                         'engine_candidates_powerchisq_newsnr': candidate_gate,
                         'reference_candidate_count': len(ref_candidates),
                         'candidate_count': len(candidates[index]),
                         'passed': all(g['passed'] for g in
                                       (waveform, weighted_waveform, norms, series, candidate_gate))})
        receipt['rows'] = rows
        receipt['checked_rows'] = len(rows)
        receipt['gates'].append({'name': 'all_rows', 'passed': len(rows) == args.batch_size
                                and all(row['passed'] for row in rows)})
        receipt['status'] = 'passed' if all(g['passed'] for g in receipt['gates']) else 'failed'
        receipt['limitations'] = ['finite fixture evidence; no universal model or strict-decision certification',
                                 'full SNR series compares providers with canonical CPU filtering; device engine checks candidates/vetoes']
    except DeviceUnavailable as exc:
        receipt.update(status='skipped', reason=str(exc))
    except Exception as exc:
        receipt.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
    # Record late-loaded reference FFT/LAL extensions used by the actual gates.
    receipt['provenance'] = execution_provenance()
    return receipt


def main(argv=None):
    args = parse_args(argv)
    start = time.perf_counter()
    receipt = qualify(args)
    receipt['in_process_verification_seconds'] = time.perf_counter() - start
    write_receipt(args.output, receipt)
    print(f"{receipt['status'].upper()}: {receipt.get('checked_rows', 0)}/{args.batch_size} rows; {args.output}")
    return {'passed': 0, 'failed': 1, 'skipped': 2}[receipt['status']]


if __name__ == '__main__':
    sys.exit(main())
