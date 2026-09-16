# Copyright (C) 2026
# This program is free software under the GNU General Public License,
# version 3 or (at your option) any later version.
"""Backwards-compatibility shim for TorchWave, delegating to diffgw."""

import importlib.util
from functools import lru_cache
from pathlib import Path
import hashlib

from pycbc.waveform import diffgw as _diffgw
from pycbc.waveform.diffgw import (
    _BOOKKEEPING, _GEOMETRY, _METADATA, _ORDERS, _PHYSICAL,
    _indices, _same_device, _series_view, template_metadata, wrap_batch
)

__all__ = [
    '_BOOKKEEPING', '_GEOMETRY', '_METADATA', '_ORDERS', '_PHYSICAL',
    '_same_device', '_series_view', 'template_metadata', 'wrap_batch',
    'provider_identity', '_reason', 'diagnostics', 'can_use', 'batch_key',
    'generate_batch',
]

_IDENTITY = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def provider_identity():
    """Get the provider-owned TaylorF2 digest once per process."""
    if (importlib.util.find_spec('torchwave') is None
            and importlib.util.find_spec('diffgw') is None):
        return None
    from torchwave.provenance import taylorf2_source_identity
    return taylorf2_source_identity()


def _reason(bank, params, device):
    return _diffgw._reason(bank, params, device)


def diagnostics(bank, indices=None, device='cpu'):
    """Return dispatch decisions without synthesizing waveform samples."""
    results = _diffgw.diagnostics(bank, indices, device)
    for r in results:
        if r['provider'] == 'diffgw':
            r['provider'] = 'torchwave'
    return results


def can_use(bank):
    """Whether any row can use the explicitly requested native provider."""
    return _diffgw.can_use(bank)


def batch_key(bank, indices):
    """Hashable identity of complete effective waveform inputs and provider."""
    def freeze(value):
        import numpy as np
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
        if not (getattr(bank, 'enable_compressed_waveforms', False)
                and getattr(bank, 'has_compressed_waveforms', False)):
            params.pop('template_duration', None)
        rows.append((index, length, freeze(params)))
    enabled = getattr(bank, 'enable_torchwave',
                      getattr(bank, 'enable_diffgw', False))
    return (_IDENTITY, provider_identity(), enabled,
            getattr(bank, 'enable_compressed_waveforms', False),
            getattr(bank, 'has_compressed_waveforms', False),
            getattr(bank, 'waveform_decompression_method', None), tuple(rows))


def generate_batch(bank, indices, device='cpu', dtype=None, delta_f=None):
    """Generate batch using diffgw, ensuring waveform_provider is set."""
    output, templates = _diffgw.generate_batch(
        bank, indices, device=device, dtype=dtype, delta_f=delta_f
    )
    for t in templates:
        if getattr(t, 'waveform_provider', None) == 'diffgw':
            t.waveform_provider = 'torchwave'
    return output, templates
