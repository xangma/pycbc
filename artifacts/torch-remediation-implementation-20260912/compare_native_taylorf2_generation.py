#!/usr/bin/env python3
"""Prepared-bank generation comparison: reference, PyCBC native, TorchWave.

All samples are qualified without fitting phase, time, amplitude or endpoints.
This measures warm synthesis/packing only, not a search or CLI pipeline.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

# Select CPU library thread limits before importing numerical libraries.
bootstrap = argparse.ArgumentParser(add_help=False)
bootstrap.add_argument('--mode', choices=['cpu1', 'cpu4', 'cuda'], default='cpu1')
initial, _ = bootstrap.parse_known_args()
threads = 4 if initial.mode == 'cpu4' else 1
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = str(threads)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from pycbc import DYN_RANGE_FAC
from pycbc.scheme import CPUScheme, TorchScheme
from pycbc.waveform import get_fd_waveform_batch
from pycbc.waveform.bank import FilterBank
from pycbc.waveform import taylorf2_torch as native_module
from pycbc.waveform.waveform import default_args
from tools import verify_torchwave_gpu_search as verification
from tools.benchmarking.evidence import (
    DeviceUnavailable, array_error, array_hash, execution_provenance,
    require_device, synchronize, write_receipt,
)


def jsonable(value):
    return json.loads(json.dumps(value, default=lambda val:
        val.item() if isinstance(val, np.generic) else str(val)))


def native_arguments(resolved, device):
    """Retain all supported resolved options; reject other non-default physics."""
    defaults = native_module._BATCH_PARAMETER_DEFAULTS
    numeric = {'mass1', 'mass2', 'f_lower', *defaults,
               *native_module._BATCH_ZERO_ONLY_PARAMETERS}
    orders = native_module._BATCH_DISCRETE_PARAMETERS
    special = {'approximant', 'delta_f', 'delta_t', 'template_hash',
               'template_duration', 'mode_array', 'numrel_data'}
    ignored_defaults = {}
    for index, params in enumerate(resolved):
        for key, value in params.items():
            if key in numeric or key in orders or key in special:
                continue
            if key not in default_args or value != default_args[key]:
                raise ValueError(f'native batch cannot preserve non-default row {index} option {key}={value!r}')
            ignored_defaults[key] = value
        if params.get('mode_array') is not None or params.get('numrel_data'):
            raise ValueError('explicit mode_array/numrel_data cannot be adapted')
    kwargs = {'delta_f': resolved[0]['delta_f']}
    defaulted_none = {}
    for key in sorted(numeric):
        default = defaults.get(key, 0.)
        values = []
        for index, params in enumerate(resolved):
            value = params.get(key, default)
            if value is None:
                if key not in defaults and key not in native_module._BATCH_ZERO_ONLY_PARAMETERS:
                    raise ValueError(f'missing native physical argument {key}')
                value = default
                defaulted_none.setdefault(key, []).append(index)
            values.append(value)
        kwargs[key] = torch.tensor(values, dtype=torch.float64, device=device)
    for key, (default, _) in orders.items():
        values = [params.get(key, default) for params in resolved]
        if len(set(values)) != 1:
            raise ValueError(f'native batch requires shared {key}')
        kwargs[key] = values[0]
    if any(params['delta_f'] != kwargs['delta_f'] for params in resolved):
        raise ValueError('native batch requires shared delta_f')
    return kwargs, {'ignored_default_options': jsonable(ignored_defaults),
                    'none_to_native_default': defaulted_none,
                    'bank_metadata_omissions': ['delta_t', 'template_hash', 'template_duration'],
                    'rule': 'Resolved distance, phase, inclination, spins and per-row cutoffs are passed directly; no fitted alignment or waveform-dependent rescaling.'}


def qualify(reference, tensor, args, provider, metadata):
    values = tensor.detach().cpu().numpy().copy()
    requested = torch.device(args.device)
    same_device = (tensor.device.type == requested.type and
                   (requested.index is None or tensor.device.index == requested.index))
    rows = []
    if values.shape != reference.shape:
        raise ValueError(f'{provider} returned {values.shape}, expected {reference.shape}')
    for index in range(args.batch_size):
        error = array_error(reference[index], values[index], rtol=1e-4)
        # Direct FP64 full-grid norm, independent of provider normalization/cache.
        norm_ref = float(4 * args.delta_f * np.sum(np.abs(reference[index].astype(np.complex128)) ** 2) / 2.)
        norm_got = float(4 * args.delta_f * np.sum(np.abs(values[index].astype(np.complex128)) ** 2) / 2.)
        norm = array_error([norm_ref], [norm_got], rtol=1e-4)
        norm['passed'] = norm['passed'] and norm_ref > 0 and norm_got > 0
        same_support = np.array_equal(reference[index] != 0, values[index] != 0)
        rows.append({'index': index, 'complex_waveform_error': error,
                     'full_grid_sigmasq_psd_2': {'reference': norm_ref, 'actual': norm_got, **norm},
                     'exact_nonzero_support': bool(same_support),
                     'passed': bool(error['passed'] and norm['passed'] and same_support)})
    dispatch_passed = (provider != 'torchwave' or
                       (len(metadata['row_providers']) == args.batch_size and
                        all(value == 'torchwave' for value in metadata['row_providers'])))
    passed = same_device and tensor.dtype == getattr(torch, args.dtype) and dispatch_passed and all(row['passed'] for row in rows)
    return {'passed': bool(passed), 'effective_device': str(tensor.device),
            'effective_dtype': str(tensor.dtype), 'dispatch_passed': dispatch_passed,
            'rows': rows, 'waveform_hash': array_hash(values), 'metadata': metadata}


def run(args, receipt):
    require_device(args.device, torch)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    receipt['threads'] = {'torch_intraop': torch.get_num_threads(),
                          'torch_interop': torch.get_num_interop_threads(),
                          'environment': {key: os.environ.get(key) for key in
                              ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                               'PYCBC_TAYLORF2_TRITON')}}
    indices = list(range(args.batch_size))
    with tempfile.TemporaryDirectory() as directory:
        bank_path = str(Path(directory) / 'bank.hdf')
        receipt['physical_manifest'] = verification.create_fixture_bank(bank_path, args)
        options = dict(dtype=np.dtype(args.dtype), approximant='TaylorF2',
                       phase_order=args.phase_order, enable_compressed_waveforms=False)
        reference_bank = FilterBank(bank_path, args.flen, args.delta_f,
                                    enable_torchwave=False, **options)
        torchwave_bank = FilterBank(bank_path, args.flen, args.delta_f,
                                    enable_torchwave=True, **options)
        resolved = [reference_bank._waveform_parameters(index)[0] for index in indices]
        if jsonable(resolved) != jsonable([torchwave_bank._waveform_parameters(index)[0] for index in indices]):
            raise ValueError('provider resolved physical manifests disagree')
        receipt['resolved_manifest'] = jsonable(resolved)
        receipt['bank_conventions'] = {
            'distance': 1. / DYN_RANGE_FAC, 'dynamic_range_factor': DYN_RANGE_FAC,
            'output_filter_length': args.flen, 'delta_f': args.delta_f,
            'first_bins': [int(np.ceil(row['f_lower'] / args.delta_f)) for row in resolved],
            'end_bins_exclusive': [int(np.floor(row['f_final'] / args.delta_f)) + 1 for row in resolved],
            'native_regular_grid_epoch': -1. / args.delta_f,
            'alignment': 'No phase, time, amplitude, or support fitting. Native regular-grid time convention is used unchanged; only zero padding and requested storage conversion are applied.'}
        native_kwargs, adaptation = native_arguments(resolved, args.device)
        receipt['native_argument_adaptation'] = adaptation
        receipt['native_arguments'] = {key: value.detach().cpu().tolist()
                                      if isinstance(value, torch.Tensor) else value
                                      for key, value in native_kwargs.items()}
        receipt['torchwave_dispatch'] = torchwave_bank.torchwave_diagnostics(indices, device=args.device)
        def reference_route():
            rows = [reference_bank[index].copy() for index in indices]
            tensor = torch.as_tensor(np.stack([row.numpy() for row in rows]),
                                     device=args.device, dtype=getattr(torch, args.dtype))
            return tensor, {'route': 'FilterBank scalar reference then stack/transfer',
                            'epochs': [float(row.epoch) for row in rows]}
        def torchwave_route():
            tensor, rows = torchwave_bank.get_batch_tensor(indices, device=args.device,
                                                           dtype=getattr(torch, args.dtype))
            return tensor, {'route': 'FilterBank.get_batch_tensor TorchWave',
                            'row_providers': [row.waveform_provider for row in rows],
                            'epochs': [float(row.epoch) for row in rows]}
        def native_route():
            generated = get_fd_waveform_batch('TaylorF2', **native_kwargs)
            values = generated.hplus
            if values.shape[0] != args.batch_size or values.shape[1] > args.flen:
                raise ValueError('native batch exceeds requested bank grid')
            output = torch.zeros((args.batch_size, args.flen), device=args.device,
                                  dtype=getattr(torch, args.dtype))
            output[:, :values.shape[1]] = values
            return output, {'route': 'get_fd_waveform_batch native TaylorF2 hplus',
                            'generation_dtype': str(values.dtype), 'epoch': float(generated.epoch),
                            'generated_length': int(values.shape[1]),
                            'cross_polarization_generated_by_api': True}
        routes = {'reference': reference_route, 'pycbc_native': native_route,
                  'torchwave': torchwave_route}
        def scheme(provider):
            return (CPUScheme(num_threads=args.threads) if provider == 'reference' else
                    TorchScheme(device=args.device, num_threads=args.threads))
        with scheme('reference'):
            original, _ = reference_route()
        synchronize(args.device, torch)
        reference = original.detach().cpu().numpy().copy()
        receipt['reference_hash'] = array_hash(reference)
        del original
        receipt['pre_timing_gates'] = {}
        for provider, route in routes.items():
            with scheme(provider):
                tensor, metadata = route()
                synchronize(args.device, torch)
            receipt['pre_timing_gates'][provider] = qualify(reference, tensor, args, provider, metadata)
            del tensor
        if not all(gate['passed'] for gate in receipt['pre_timing_gates'].values()):
            receipt['status'] = 'failed'
            receipt['reason'] = 'generation gate failed before timing; no speed claim'
            return
        receipt['measurements'] = {provider: [] for provider in routes}
        names = list(routes)
        for repetition in range(5):
            # Rotate order to avoid always placing one route first.
            for provider in names[repetition % 3:] + names[:repetition % 3]:
                with scheme(provider):
                    synchronize(args.device, torch)
                    start = time.perf_counter()
                    tensor, metadata = routes[provider]()
                    synchronize(args.device, torch)
                    elapsed = time.perf_counter() - start
                gate = qualify(reference, tensor, args, provider, metadata)
                receipt['measurements'][provider].append({'repetition': repetition,
                    'generation_to_device_ready_sec': elapsed, 'qualification': gate})
                del tensor
        passed = all(sample['qualification']['passed'] for samples in receipt['measurements'].values() for sample in samples)
        receipt['status'] = 'passed' if passed else 'failed'
        if passed:
            medians = {provider: float(np.median([row['generation_to_device_ready_sec'] for row in samples]))
                       for provider, samples in receipt['measurements'].items()}
            receipt['qualified_generation_summary'] = {'median_seconds': medians,
                'reference_over_torchwave': medians['reference'] / medians['torchwave'],
                'pycbc_native_over_torchwave': medians['pycbc_native'] / medians['torchwave'],
                'reference_over_pycbc_native': medians['reference'] / medians['pycbc_native']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['cpu1', 'cpu4', 'cuda'], default='cpu1')
    parser.add_argument('--batch-size', type=int, choices=[4, 16], default=4)
    parser.add_argument('--size', type=int, choices=[2048, 131072], default=2048)
    parser.add_argument('--dtype', choices=['complex64', 'complex128'], default='complex64')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    args.device = 'cuda' if args.mode == 'cuda' else 'cpu'
    args.threads = 4 if args.mode == 'cpu4' else 1
    args.flen, args.delta_f, args.phase_order = args.size // 2 + 1, 1024. / args.size, -1
    receipt = {'schema_version': 2, 'status': 'failed', 'configuration': vars(args),
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'scope': 'One shared verify_torchwave_gpu_search fixture; 5 warm generation-only samples per route. No filtering, veto, operational pipeline or cold-process speed claim.',
               'timing_boundary': 'Prepared bank/arguments to requested-device complex tensor; bank file I/O, construction, cold imports and qualification are outside timers. Includes synthesis, API metadata, allocation, zero padding, storage conversion, stacking/transfer where required, and device completion synchronization. Scheme entry/exit is outside timers.',
               'route_cost_differences': 'Reference and TorchWave use prepared FilterBank generation including row metadata; PyCBC native uses existing get_fd_waveform_batch and generates both polarizations internally. Native physical argument tensors are prebuilt outside timers. These costs differ and ratios apply only to these stated routes.',
               'gates': 'All rows: unaligned full complex relative L2 <= 1e-4, direct FP64 full-grid sigmasq relative error <= 1e-4, exact nonzero support, requested storage/device, native TorchWave dispatch. Every timed output is independently checked after its timer stops.'}
    code = 1
    try:
        run(args, receipt)
        code = 0 if receipt['status'] == 'passed' else 1
    except DeviceUnavailable as error:
        receipt.update(status='skipped_unavailable', error=str(error))
        code = 2
    except Exception as error:
        receipt.update(error=str(error), traceback=traceback.format_exc())
    finally:
        receipt['provenance'] = execution_provenance()
        write_receipt(args.output, receipt)
    print(json.dumps({'status': receipt['status'], 'receipt': str(Path(args.output).resolve())}))
    return code


if __name__ == '__main__':
    sys.exit(main())
