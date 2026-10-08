# Copyright (C) 2026 The PyCBC Collaboration
# SPDX-License-Identifier: GPL-3.0-or-later
"""Submit detector work and retain host identifiers at result boundaries."""

import numpy as np


def enqueue_live_detector_jax(control, reader):
    """Submit a detector without collecting its candidate decisions."""
    from pycbc.filter.matchedfilter_jax import enqueue_live_data_jax

    return enqueue_live_data_jax(control, reader)


def finish_live_detectors_jax(control, results):
    """Submit every detector's veto work before terminal materialization."""
    from pycbc.filter.matchedfilter_jax import finish_live_data_jax

    return {
        ifo: (
            finish_live_data_jax(control, value)
            if getattr(value, '_jax_live_prepared', False)
            else value
        )
        for ifo, value in results.items()
    }


def live_result_column_backend_jax(key, column):
    """Keep routing IDs on the host and numerical science on the device."""
    if key == 'template_id' or column.dtype.kind in 'OUS':
        return np
    import jax.numpy as jnp

    return jnp
