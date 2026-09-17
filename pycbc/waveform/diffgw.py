# Copyright (C) 2026
# This program is free software under the GNU General Public License,
# version 3 or (at your option) any later version.
"""Explicit, conservative diffgw dispatch for search template banks.

The runtime allowlist admits aligned-spin models natively supported by diffgw,
including TaylorF2, IMRPhenomD, and IMRPhenomXAS with standard 3.5PN phase
conventions and zero reference frequency. Synthesis always uses float64;
storage may use complex64 or complex128. Other requests retain scalar PyCBC
behavior, including compressed waveforms.
"""

import importlib.util
import hashlib
from functools import lru_cache
from pathlib import Path
import math
import operator
import os
import types

import numpy as np

from pycbc.types import FrequencySeries
from pycbc.waveform.waveform import default_args

# Bank bookkeeping is not a waveform option. Unknown explicit options are
# deliberately rejected, even if an upstream generator silently ignores them.
_BOOKKEEPING = {'template_hash', 'template_duration'}
_PHYSICAL = {'mass1', 'mass2', 'spin1z', 'spin2z', 'coa_phase', 'inclination',
             'distance'}
_GEOMETRY = {'approximant', 'f_lower', 'f_final', 'delta_f', 'delta_t'}
_METADATA = ('f_lower', 'min_f_lower', 'end_idx', 'chirp_length',
             'length_in_time', 'approximant', 'end_frequency', 'time_offset',
             'id', 'waveform_provider', 'waveform_provider_reason')
_IDENTITY = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

_ORDERS = {'phase_order': (-1, 7), 'spin_order': (-1, 7),
           'amplitude_order': (-1, 0), 'tidal_order': (-1,)}

_SUPPORTED_APPROXIMANTS = ('TaylorF2', 'IMRPhenomD', 'IMRPhenomXAS')


def _indices(bank, indices):
    indices = list(range(len(bank))) if indices is None else list(indices)
    result = []
    for index in indices:
        index = operator.index(index)
        if not 0 <= index < len(bank):
            raise IndexError('template bank index out of range')
        result.append(index)
    return result


def _is_provider_enabled(bank):
    return (getattr(bank, 'enable_diffgw', None) is True
            or getattr(bank, 'enable_torchwave', None) is True)


def is_available():
    """Return True if diffgw (or legacy torchwave) is available."""
    for mod in ('diffgw', 'torchwave'):
        if importlib.util.find_spec(mod) is not None:
            try:
                __import__(mod)
                return True
            except Exception:
                pass
    return False


def _reason(bank, params, device):
    if not _is_provider_enabled(bank):
        return 'diffgw requires explicit enable_diffgw=True or enable_torchwave=True'
    if (importlib.util.find_spec('diffgw') is None
            and importlib.util.find_spec('torchwave') is None):
        return 'Neither diffgw nor torchwave is installed'
    if device.split(':')[0] not in ('cpu', 'cuda'):
        return 'float64 diffgw generation requires CPU or CUDA'
    if (getattr(bank, 'has_compressed_waveforms', False)
            and getattr(bank, 'enable_compressed_waveforms', False)):
        return 'compressed waveform generation takes precedence'
    approx = params.get('approximant')
    if approx not in _SUPPORTED_APPROXIMANTS:
        return f'approximant {approx} is outside the runtime allowlist'
    max_mass = 200 if approx in ('IMRPhenomD', 'IMRPhenomXAS') else 100
    for key in ('mass1', 'mass2'):
        if not 1 <= params[key] <= max_mass:
            return f'{key} is outside the allowed [1, {max_mass}] solar-mass range'
    for key in ('spin1z', 'spin2z'):
        if not abs(params[key]) <= 0.99:
            return f'{key} is outside the allowed [-0.99, 0.99] range'
    for key in _PHYSICAL | _GEOMETRY - {'approximant'}:
        if not math.isfinite(params[key]):
            return f'{key} must be finite'
    if not 10 <= params['f_lower'] < params['f_final'] <= 4096:
        return 'frequency range is outside the allowed 10–4096 Hz domain'
    for key, value in params.items():
        if key in _PHYSICAL | _GEOMETRY | _BOOKKEEPING:
            continue
        if key in _ORDERS:
            if value not in _ORDERS[key]:
                return f'{key}={value!r} is outside the runtime allowlist'
        elif key in ('lambda1', 'lambda2') and value in (None, 0):
            continue
        elif key == 'mode_array':
            if value is not None:
                return 'explicit mode_array requires reference generation'
        elif key == 'taper':
            # The inspiral CLI passes None even when tapering is disabled.
            # get_waveform_filter applies a taper only for a non-None value.
            if value is not None:
                return 'taper requests require reference generation'
        elif key not in default_args:
            return f'option {key} is outside the runtime allowlist'
        elif value != default_args[key]:
            return f'non-default {key} requires reference generation'
    return None


def diagnostics(bank, indices=None, device='cpu'):
    """Return dispatch decisions without synthesizing waveform samples."""
    results = []
    for index in _indices(bank, indices):
        params, _ = bank._waveform_parameters(index)
        reason = _reason(bank, params, str(device))
        provider_name = getattr(bank, 'waveform_provider_name', 'diffgw')
        results.append({'index': index,
                        'provider': 'reference' if reason else provider_name,
                        'reason': reason or f"{params.get('approximant')} runtime allowlist",
                        'generation_dtype': None if reason else 'float64'})
    return results


def can_use(bank):
    """Whether any row can use the explicitly requested native provider."""
    if not _is_provider_enabled(bank):
        return False
    for index in range(len(bank)):
        params, _ = bank._waveform_parameters(index)
        if _reason(bank, params, 'cpu') is None:
            return True
    return False


def _same_device(left, right):
    import torch
    left, right = torch.device(left), torch.device(right)
    if left.type != right.type:
        return False
    if left.type == 'cuda':
        li = torch.cuda.current_device() if left.index is None else left.index
        ri = (torch.cuda.current_device()
              if right.index is None else right.index)
        return li == ri
    return True


def template_metadata(templates):
    """Extract sample-free records for a session's owned tensor cache."""
    return [dict({name: getattr(t, name) for name in _METADATA
                  if hasattr(t, name)}, delta_f=t.delta_f, epoch=t.epoch)
            for t in templates]


def wrap_batch(bank, indices, data, metadata):
    """Reconstruct independent metadata views of a cloned session tensor."""
    from pycbc.waveform.bank import sigma_cached
    indices = _indices(bank, indices)
    if len(indices) != len(metadata) or data.shape[0] != len(indices):
        raise ValueError('batch indices, metadata and tensor rows must agree')
    templates = []
    for index, row, record in zip(indices, data, metadata):
        series = _series_view(row, record['delta_f'], record['epoch'])
        for name in _METADATA:
            if name in record:
                setattr(series, name, record[name])
        if 'chirp_length' in record:
            bank.table[index].template_duration = record['chirp_length']
        series.params = bank.table[index]
        series.sigmasq = types.MethodType(sigma_cached, series)
        series._sigmasq = {}
        templates.append(series)
    return templates


@lru_cache(maxsize=1)
def provider_identity():
    """Get the provider-owned source digest once per process."""
    if importlib.util.find_spec('diffgw') is not None:
        try:
            import diffgw
            if hasattr(diffgw, 'provenance') and hasattr(diffgw.provenance, 'taylorf2_source_identity'):
                return diffgw.provenance.taylorf2_source_identity()
            if hasattr(diffgw, 'torch') and hasattr(diffgw.torch, 'provenance'):
                return diffgw.torch.provenance.taylorf2_source_identity()
        except Exception:
            pass
    if importlib.util.find_spec('torchwave') is None:
        return None
    from torchwave.provenance import taylorf2_source_identity
    return taylorf2_source_identity()


def batch_key(bank, indices):
    """Hashable identity of complete effective waveform inputs and provider."""
    def freeze(value):
        if isinstance(value, np.ndarray):
            return (str(value.dtype), value.shape, value.tobytes())
        if isinstance(value, dict):
            return tuple(sorted((key, freeze(v)) for key, v in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(freeze(v) for v in value)
        if isinstance(value, np.generic):
            return value.item()
        return value

    rows = []
    for index in _indices(bank, indices):
        params, length = bank._waveform_parameters(index)
        # Scalar generation writes this cache field after the request is made.
        if not (getattr(bank, 'enable_compressed_waveforms', False)
                and getattr(bank, 'has_compressed_waveforms', False)):
            params.pop('template_duration', None)
        rows.append((index, length, freeze(params)))
    enabled = getattr(bank, 'enable_diffgw', getattr(bank, 'enable_torchwave', False))
    return (_IDENTITY, provider_identity(), enabled,
            getattr(bank, 'enable_compressed_waveforms', False),
            getattr(bank, 'has_compressed_waveforms', False),
            getattr(bank, 'waveform_decompression_method', None), tuple(rows))


def _series_view(row, delta_f, epoch=0):
    from pycbc import scheme
    from pycbc.types.array_torch import TorchArrayData
    state = scheme.mgr.state
    if isinstance(state, scheme.TorchScheme):
        if not _same_device(state.torch_device, row.device):
            raise ValueError('tensor metadata requires a matching TorchScheme')
        return FrequencySeries(TorchArrayData(row), delta_f=delta_f,
                               epoch=epoch, copy=False)
    if row.device.type != 'cpu':
        raise ValueError('CUDA tensor metadata requires a matching '
                         'TorchScheme')
    # CPU .numpy() shares the tensor storage; this does not copy samples.
    return FrequencySeries(row.numpy(), delta_f=delta_f, epoch=epoch,
                           copy=False)


def _metadata(bank, index, params, row):
    from pycbc.waveform import get_waveform_filter_length_in_time
    from pycbc.waveform.bank import sigma_cached
    series = _series_view(row, params['delta_f'], -1.0 / params['delta_f'])
    duration = get_waveform_filter_length_in_time(**params)
    bank.table[index].template_duration = duration
    series.f_lower = params['f_lower']
    series.min_f_lower = bank.min_f_lower
    series.end_frequency = params['f_final']
    series.end_idx = int(params['f_final'] / params['delta_f'])
    series.params = bank.table[index]
    series.chirp_length = duration
    series.length_in_time = duration
    series.approximant = params['approximant']
    series.sigmasq = types.MethodType(sigma_cached, series)
    series._sigmasq = {}
    if hasattr(bank, 'id_from_param'):
        p = series.params
        series.id = bank.id_from_param((p.mass1, p.mass2, p.spin1z, p.spin2z))
    return series


_COMPILED_DIFFGW_GENERATOR = None


def _diffgw_compile_requested(bank=None):
    """Return True if torch.compile should be used for diffgw waveform generation."""
    if bank is not None and getattr(bank, 'diffgw_compile', None) is not None:
        return bool(bank.diffgw_compile)
    val = os.environ.get('PYCBC_DIFFGW_COMPILE', os.environ.get('PYCBC_TORCH_COMPILE_DIFFGW', None))
    if val is not None:
        return val.lower() in ('1', 'true', 'yes', 'on')
    return False


def _get_diffgw_generator(compile=False):
    """Import diffgw, optionally wrapping with torch.compile."""
    global _COMPILED_DIFFGW_GENERATOR
    try:
        import diffgw
        raw_gen = diffgw.get_fd_waveform
    except ImportError:
        import torchwave
        raw_gen = torchwave.get_fd_waveform

    if not compile:
        return raw_gen

    if _COMPILED_DIFFGW_GENERATOR is None:
        import torch
        try:
            torch._dynamo.config.capture_scalar_outputs = True
        except Exception:
            pass
        _COMPILED_DIFFGW_GENERATOR = torch.compile(raw_gen, fullgraph=False)
    return _COMPILED_DIFFGW_GENERATOR


def generate_batch(bank, indices, device='cpu', dtype=None, delta_f=None):
    """Return resident samples and metadata views, preserving input order.

    ``device`` must match an active TorchScheme for accelerator requests.
    Scalar consumers can explicitly convert returned FrequencySeries at their
    usual scheme boundary. Unsupported rows use the existing scalar generator;
    their transfer to the batch tensor is part of reference fallback cost.
    """
    import torch
    from pycbc import scheme
    indices = _indices(bank, indices)
    device = torch.device(device)
    if dtype is None:
        bank_dtype = np.dtype(getattr(bank, 'dtype', np.complex64))
        dtype = (torch.complex128 if bank_dtype == np.dtype('complex128')
                 else torch.complex64)
    if dtype not in (torch.complex64, torch.complex128):
        raise TypeError('waveform storage dtype must be complex64 '
                        'or complex128')
    state = scheme.mgr.state
    if isinstance(state, scheme.TorchScheme):
        if not _same_device(state.torch_device, device):
            raise ValueError('tensor metadata requires a matching TorchScheme')
    elif device.type != 'cpu':
        raise ValueError('CUDA tensor metadata requires a matching '
                         'TorchScheme')
    if device.type == 'mps' and dtype == torch.complex128:
        raise ValueError('MPS does not support complex128 waveform storage')
    requests = [bank._waveform_parameters(i, delta_f) for i in indices]
    flen = requests[0][1] if requests else bank.filter_length
    if any(length != flen or p['delta_f'] != requests[0][0]['delta_f']
           for p, length in requests):
        raise ValueError('a waveform tensor batch requires a common grid')
    output = torch.zeros((len(indices), flen), dtype=dtype, device=device)
    templates = [None] * len(indices)
    native = []
    for position, (index, (params, _)) in enumerate(zip(indices, requests)):
        reason = _reason(bank, params, str(device))
        if reason is None:
            native.append(position)
            continue
        reference = (bank[index] if not hasattr(bank, 'get_template') else
                     bank.get_template(index, delta_f=params['delta_f']))
        data = reference.data
        values = data.tensor if hasattr(data, 'tensor') else reference.numpy()
        source = torch.as_tensor(values, device=device, dtype=dtype)
        output[position].copy_(source)
        # Use the batch row even when bank.out is reused by scalar generation.
        series = _series_view(output[position], params['delta_f'],
                              reference.epoch)
        for name in ('f_lower', 'min_f_lower', 'end_idx', 'params',
                     'chirp_length', 'length_in_time', 'approximant',
                     'end_frequency', 'time_offset', 'id'):
            if hasattr(reference, name):
                setattr(series, name, getattr(reference, name))
        from pycbc.waveform.bank import sigma_cached
        series.sigmasq = types.MethodType(sigma_cached, series)
        series._sigmasq = {}
        series.waveform_provider = 'reference'
        series.waveform_provider_reason = reason
        templates[position] = series

    if native:
        use_compile = _diffgw_compile_requested(bank)
        diffgw_gen = _get_diffgw_generator(compile=use_compile)
        provider_name = getattr(bank, 'waveform_provider_name', 'diffgw')

        # Group native requests by approximant
        approx_groups = {}
        for pos in native:
            approx_groups.setdefault(requests[pos][0]['approximant'], []).append(pos)

        bins = torch.arange(flen, device=device)
        for app, pos_list in approx_groups.items():
            sub_params = [requests[i][0] for i in pos_list]
            kwargs = {
                key: torch.tensor([p[key] for p in sub_params], dtype=torch.float64,
                                  device=device)
                for key in _PHYSICAL - {'coa_phase'}
            }
            freqs = torch.arange(flen, device=device, dtype=torch.float64)
            freqs *= sub_params[0]['delta_f']

            if app == 'TaylorF2':
                kwargs['phic'] = torch.tensor(
                    [p['coa_phase'] + np.pi / 2 for p in sub_params],
                    dtype=torch.float64, device=device)
                hp, _ = diffgw_gen(
                    'TaylorF2', sample_frequencies=freqs, dtype=torch.float64,
                    device=device, f_isco_cutoff=False, **kwargs)
            else:
                # IMRPhenomD, IMRPhenomXAS, etc.
                kwargs['phic'] = torch.tensor(
                    [p['coa_phase'] for p in sub_params],
                    dtype=torch.float64, device=device)
                f_lower = min(p['f_lower'] for p in sub_params)
                hp, _ = diffgw_gen(
                    app, sample_frequencies=freqs, dtype=torch.float64,
                    device=device, f_lower=f_lower, **kwargs)

            lows = torch.tensor([math.ceil(p['f_lower'] / p['delta_f'])
                                 for p in sub_params], device=device)
            highs = torch.tensor([int(p['f_final'] / p['delta_f'])
                                  for p in sub_params], device=device)
            hp = torch.where((bins[None, :] >= lows[:, None])
                             & (bins[None, :] <= highs[:, None]), hp, 0)
            output[pos_list] = hp.to(dtype=dtype)

            for position in pos_list:
                series = _metadata(bank, indices[position], requests[position][0],
                                    output[position])
                series.waveform_provider = provider_name
                series.waveform_provider_reason = f"{app} runtime allowlist"
                templates[position] = series

    return output, templates
