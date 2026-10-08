# Copyright (C) 2026 PyCBC contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""PSD metadata adapter for isolated original CPU validation."""


def cpu_psd_reference(operation, values=None, spacing=1.0, epoch=None, **kwargs):
    from pycbc.reference_jax import cpu_reference

    return cpu_reference(operation, values, spacing, epoch, **kwargs)
