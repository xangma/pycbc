# Copyright (C) 2026 The PyCBC Collaboration
# SPDX-License-Identifier: GPL-3.0-or-later
"""diffGW generation on each Live template's original frequency grid."""

from pycbc import scheme
from .diffgw_jax import generate_batch


def _waveform_windows(bank, indices, batch_size):
    for start in range(0, len(indices), batch_size):
        window = indices[start:start + batch_size]
        groups = {}
        for index in window:
            groups.setdefault(bank.freq_resolution_for_template(index), []).append(index)
        templates = {}
        for delta_f, group in groups.items():
            _, waveforms = generate_batch(bank, group, delta_f=delta_f)
            templates.update(zip(group, waveforms))
        for index in window:
            yield templates.pop(index)


def iter_live_waveforms_jax(bank, batch_size=None):
    """Iterate in bank order while batching equal grids in bounded windows."""
    if not isinstance(scheme.mgr.state, scheme.JAXScheme):
        raise TypeError('JAX Live generation requires JAXScheme')
    if batch_size is None:
        batch_size = 4 if scheme.mgr.state.jax_device.platform == 'cpu' else 128
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    yield from _waveform_windows(bank, list(range(len(bank))), batch_size)


def generate_live_waveforms_jax(bank, rank, size):
    """Generate the original MPI worker partition, retaining per-row grids."""
    if size < 1 or rank < 0 or (size > 1 and rank >= size):
        raise ValueError('invalid MPI rank or size')
    if size > 1:
        if rank == 0:
            return []
        indices = list(range(rank - 1, len(bank), size - 1))
    else:
        indices = list(range(len(bank)))
    batch_size = 4 if scheme.mgr.state.jax_device.platform == 'cpu' else 128
    return list(_waveform_windows(bank, indices, batch_size))
