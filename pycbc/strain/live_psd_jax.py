# Copyright (C) 2026 The PyCBC Collaboration
# Licensed under the GNU General Public License, version 3 or later.

"""Enqueue detector PSD refreshes before their shared control readback."""

from dataclasses import dataclass

import jax
import numpy as np

from pycbc import scheme
from pycbc.psd.estimate_jax import welch_jax
from pycbc.strain.strain import logger
from pycbc.strain.strain_jax import psd_horizon_payload_jax


def _prepare_psd(buffer):
    seg_len = int(buffer.sample_rate * buffer.psd_segment_length)
    end = len(buffer.strain)
    start = end - (buffer.psd_samples + 1) * seg_len // 2
    psd = welch_jax(buffer.strain[start:end], seg_len=seg_len,
                    seg_stride=seg_len // 2)
    if buffer.strain.dtype == np.float32:
        psd = psd.astype(np.float32)
    return psd, psd_horizon_payload_jax(psd, buffer.low_frequency_cutoff)


def _apply_psd(buffer, psd, horizon):
    distance, negative_power = horizon
    if negative_power:
        raise ValueError("math domain error")
    psd.dist = float(distance)
    if buffer.psd and buffer.psd_recalculate_difference:
        if abs(buffer.psd.dist - psd.dist) / buffer.psd.dist < buffer.psd_recalculate_difference:
            logger.info("Skipping recalculation of %s PSD, %s-%s",
                        buffer.detector, buffer.psd.dist, psd.dist)
            return True
    if buffer.psd and buffer.psd_abort_difference:
        if abs(buffer.psd.dist - psd.dist) / buffer.psd.dist > buffer.psd_abort_difference:
            logger.info("%s PSD is CRAZY, aborting!!!!, %s-%s",
                        buffer.detector, buffer.psd.dist, psd.dist)
            buffer.psd = psd
            buffer.psds = {}
            buffer.segments = {}
            return False
    buffer.psd = psd
    buffer.psds = {}
    buffer.segments = {}
    logger.info("Recalculating %s PSD, %s", buffer.detector, psd.dist)
    return True


@dataclass
class _PendingPSDUpdate:
    buffer: object
    psd: object = None
    horizon: object = None
    error: Exception = None
    _done: bool = False
    _status: object = None

    def result(self):
        """Apply this detector's decision only at its original processing turn."""
        if not self._done:
            if self.error is None:
                try:
                    self._status = _apply_psd(self.buffer, self.psd, self.horizon)
                except Exception as error:
                    self.error = error
            self._done = True
        if self.error is not None:
            raise self.error
        return self._status


def prepare_psd_updates_jax(buffers):
    """Enqueue a finite ordered group and collect detached horizon payloads.

    Returned handles retain preparation/domain errors until their ``result``
    turn, allowing an earlier detector's filter error to keep its precedence.
    Buffer PSD/cache identities remain untouched until that handle is applied.
    """
    if not isinstance(scheme.mgr.state, scheme.JAXScheme):
        raise TypeError("Batched PSD refresh requires JAXScheme")
    pending = []
    for buffer in buffers:
        update = _PendingPSDUpdate(buffer)
        try:
            update.psd, update.horizon = _prepare_psd(buffer)
        except Exception as error:
            update.error = error
        pending.append(update)
    prepared = [update for update in pending if update.error is None]
    if prepared:
        try:
            horizons = jax.device_get([update.horizon for update in prepared])
        except Exception:
            # Failure-only fallback identifies per-detector readback errors;
            # a later failed buffer must not preempt an earlier filter turn.
            for update in prepared:
                try:
                    update.horizon = jax.device_get(update.horizon)
                except Exception as error:
                    update.error = error
        else:
            for update, horizon in zip(prepared, horizons):
                update.horizon = horizon
    return tuple(pending)


def recalculate_psds_jax(buffers):
    """Apply refreshed PSDs in order, returning native detector statuses."""
    return tuple(update.result() for update in prepare_psd_updates_jax(buffers))
