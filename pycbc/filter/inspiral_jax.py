# Copyright (C) 2026 PyCBC contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Setup boundaries for the dedicated JAX inspiral search."""


def batch_size_from_cli(options, context, parser):
    """Validate a positive batch size and choose the device default."""
    if options.multiprocessing_nprocesses:
        parser.error('JAX uses device batching; multiprocessing is not supported')
    count = options.batch_size
    if count is None:
        return 16 if context.jax_device.platform == 'cpu' else 64
    if count < 1:
        parser.error('--batch-size must be positive')
    return count


def prepare_data(options, context, dyn_range_factor, injection_filter):
    """Load and condition strain with the selected JAX scheme active."""
    from pycbc import strain

    with context:
        series = strain.from_cli(
            options, dyn_range_fac=dyn_range_factor,
            inj_filter_rejector=injection_filter)
        segments = strain.StrainSegments.from_cli(options, series)
    return series, segments
