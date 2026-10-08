# Copyright (C) 2026 PyCBC contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional diffGW adapter for JAX waveform banks.

Importing the adapter does not load either waveform frontend or JAX.
"""

import importlib.util


def is_available():
    """Whether the public diffGW package can be imported."""
    return importlib.util.find_spec('diffgw') is not None


def can_use(bank):
    from .diffgw_jax import can_use as implementation
    return implementation(bank)


def diagnostics(bank, indices=None, device='jax'):
    from .diffgw_jax import diagnostics as implementation
    return implementation(bank, indices, device)


def generate_batch(bank, indices, device='jax', dtype=None, delta_f=None):
    from .diffgw_jax import generate_batch as implementation
    return implementation(bank, indices, device, dtype, delta_f)


def wrap_batch(bank, indices, data, metadata):
    from .diffgw_jax import wrap_batch as implementation
    return implementation(bank, indices, data, metadata)


def batch_key(bank, indices):
    from .diffgw_jax import batch_key as implementation
    return implementation(bank, indices)


def template_metadata(templates):
    from .diffgw_jax import template_metadata as implementation
    return implementation(templates)
