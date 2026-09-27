# Copyright (C) 2026 The PyCBC Collaboration
"""JAX kernels for trigger event clustering."""

import jax
import jax.numpy as jnp
import numpy

from . import ranking
from .eventmgr import EventManager, findchirp_cluster_over_window_cython


def loudest_event_indices(snr, chisq, count):
    """Return the union of loudest SNR and NewSNR event indices."""
    snr = jnp.asarray(snr)
    chisq = jnp.asarray(chisq)
    snr_order = jnp.argsort(jnp.abs(snr))[::-1][:int(count)]
    newsnr_order = jnp.argsort(ranking.newsnr(jnp.abs(snr), chisq))[::-1]
    newsnr_order = newsnr_order[:int(count)]
    return jnp.unique(jnp.concatenate((snr_order, newsnr_order)))


def findchirp_cluster_over_window_jax(times, values, window_length):
    """FindChirp greedy clustering with device resident numeric arrays.

    The selected indices are a compact host shaped result, while all time,
    magnitude, and comparison arithmetic is performed by JAX.
    """
    times = jnp.asarray(times)
    values = jnp.asarray(values)
    if times.size == 0:
        return jnp.array([], dtype=jnp.int32)
    out = jnp.zeros(times.size, dtype=jnp.int32)
    out = out.at[0].set(0)

    def step(carry, idx):
        current, count, selected = carry
        new_group = (times[idx] - times[current]) > window_length
        replace = (~new_group) & (
            jnp.abs(values[idx]) > jnp.abs(values[current]))
        selected = jax.lax.cond(
            new_group,
            lambda x: x.at[count].set(idx),
            lambda x: jax.lax.cond(
                replace, lambda y: y.at[count - 1].set(idx), lambda y: y,
                x), selected)
        current = jnp.where(new_group | replace, idx, current)
        return (current, count + new_group.astype(jnp.int32), selected), None

    (_, count, selected), _ = jax.lax.scan(
        step, (jnp.int32(0), jnp.int32(1), out),
        jnp.arange(1, times.size, dtype=jnp.int32))
    return selected[:int(count)]


class JAXEventManager(EventManager):
    """Device-resident event state used by the JAX inspiral path.

    Template metadata remains ordinary Python objects. Trigger columns and
    all filtering/clustering state are JAX arrays; conversion is deferred to
    the caller's output or checkpoint boundary.
    """

    def __init__(self, opt, column, column_types, array_minsize=10000,
                 **kwds):
        self.opt = opt
        self.global_params = kwds
        self.array_minsize = array_minsize
        self.columns = tuple(column)
        self.column_types = tuple(column_types)
        self.event_dtype = [("template_id", int)] + list(
            zip(self.columns, self.column_types))
        self._events = {"template_id": jnp.array([], dtype=jnp.int32)}
        self._events.update({name: jnp.array([], dtype=jnp.dtype(dtype))
                             for name, dtype in zip(column, column_types)})
        self.template_events = {name: value[:0]
                                for name, value in self._events.items()}
        self._pending_event_chunks = []
        self.template_params = []
        self.template_index = -1
        self.write_performance = False

    @property
    def events(self):
        self._flush_event_chunks()
        return self._events

    @property
    def num_events(self):
        self._flush_event_chunks()
        return self._events["template_id"].size

    def _flush_event_chunks(self):
        """Materialize completed template chunks in one linear-time append."""
        if not self._pending_event_chunks:
            return
        chunks = self._pending_event_chunks
        self._events = {
            name: jnp.concatenate(
                [self._events[name]] + [chunk[name] for chunk in chunks]
            )
            for name in self._events
        }
        self._pending_event_chunks = []

    def new_template(self, **kwds):
        self.template_params.append(kwds)
        self.template_index += 1

    def add_template_params(self, **kwds):
        self.template_params[-1].update(kwds)

    def add_template_events(self, columns, vectors):
        length = next((len(value) for value in vectors if value is not None), 0)
        if not length:
            return
        data = {name: jnp.zeros(length, dtype=value.dtype)
                for name, value in self.template_events.items()}
        data["template_id"] = jnp.full(length, self.template_index,
                                        dtype=jnp.int32)
        for name, value in zip(columns, vectors):
            if value is not None:
                data[name] = jnp.asarray(value, dtype=data[name].dtype)
        self.template_events = {
            name: jnp.concatenate((self.template_events[name], data[name]))
            for name in self.template_events
        }

    def cluster_template_events(self, tcolumn, column, window_size):
        if window_size <= 0 or not self.template_events[tcolumn].size:
            return
        # Trigger vectors are small, irregular final products.  The native
        # implementation is both exact and avoids compiling one scan for each
        # distinct trigger count. Transfer the compact event record once;
        # selecting every column separately on device creates thousands of
        # tiny shape-specific launches over a realistic bank.
        names = tuple(self.template_events)
        host_values = jax.device_get(tuple(
            self.template_events[name] for name in names
        ))
        host_events = dict(zip(names, host_values))
        times = host_events[tcolumn]
        values = host_events[column]
        indices = numpy.zeros(len(times), dtype=numpy.int32)
        count = findchirp_cluster_over_window_cython(
            numpy.asarray(times, dtype=numpy.int32),
            numpy.asarray(abs(values)), window_size, indices, len(times)
        )
        indices = indices[:count + 1]
        self.template_events = {
            name: value[indices] for name, value in host_events.items()
        }

    def finalize_template_events(self):
        if self.template_events["template_id"].size:
            self._pending_event_chunks.append(self.template_events)
        # Start the next template with typed device empties. Completed chunks
        # may be NumPy arrays after exact host clustering and are converted
        # only once when the global event record is materialized.
        self.template_events = {
            name: value[:0] for name, value in self._events.items()
        }

    def cut_events_via_mask(self, keep):
        self._flush_event_chunks()
        keep = jnp.asarray(keep, dtype=bool)
        self._events = {name: value[keep]
                        for name, value in self._events.items()}

    def cut_events_via_indices(self, indices):
        self._flush_event_chunks()
        indices = jnp.asarray(indices, dtype=jnp.int32)
        self._events = {name: value[indices]
                        for name, value in self._events.items()}

    def chisq_threshold(self, value, num_bins, delta=0):
        chisq = self._events["chisq"]
        dof = self._events["chisq_dof"]
        snr = self._events["snr"]
        reduced = chisq / (dof + delta * jnp.real(snr.conj() * snr))
        self.cut_events_via_mask(~(reduced > value))

    def newsnr_threshold(self, threshold):
        if not self.opt.chisq_bins:
            raise RuntimeError("Chi-square test must be enabled in order to "
                               "use newsnr threshold")
        trigs = {name: value for name, value in self._events.items()}
        nsnr = ranking.newsnr(jnp.abs(trigs["snr"]),
                              trigs["chisq"] / trigs["chisq_dof"])
        self.cut_events_via_mask(nsnr >= threshold)

    def consolidate_events(self, opt, gwstrain=None):
        self._flush_event_chunks()
        if opt.chisq_threshold and opt.chisq_bins:
            self.chisq_threshold(opt.chisq_threshold, opt.chisq_bins,
                                 opt.chisq_delta)
        if opt.newsnr_threshold and opt.chisq_bins:
            self.newsnr_threshold(opt.newsnr_threshold)
        if opt.keep_loudest_interval:
            self.keep_loudest_in_interval(
                opt.keep_loudest_interval * opt.sample_rate,
                opt.keep_loudest_num, opt.keep_loudest_stat,
                opt.keep_loudest_log_chirp_window)
        if opt.injection_window and hasattr(gwstrain, 'injections'):
            raise NotImplementedError(
                "JAXEventManager does not support injection window "
                "consolidation yet"
            )

    def keep_loudest_in_interval(self, window, num_keep, statname="newsnr",
                                 log_chirp_width=None):
        if log_chirp_width:
            raise NotImplementedError(
                "JAXEventManager does not support chirp-width loudest bins")
        if not self.num_events:
            return
        if statname in ("newsnr", "new_snr"):
            score = ranking.newsnr(
                jnp.abs(self._events["snr"]),
                self._events["chisq"] / self._events["chisq_dof"])
        elif statname == "snr":
            score = jnp.abs(self._events["snr"])
        else:
            raise NotImplementedError(
                "unsupported JAX loudest statistic: %s" % statname)
        bins = (self._events["time_index"] / window).astype(jnp.int32)
        keep = []
        for bucket in jnp.unique(bins):
            indices = jnp.flatnonzero(bins == bucket)
            order = indices[jnp.argsort(score[indices])[-num_keep:]]
            keep.append(order)
        self.cut_events_via_indices(jnp.concatenate(keep))

    def finalize_events(self):
        self._flush_event_chunks()
        return None

    def _host_structured_events(self):
        self._flush_event_chunks()
        dtype = [("template_id", int)] + list(
            zip(self.columns, self.column_types))
        result = numpy.zeros(len(self._events["template_id"]), dtype=dtype)
        names = result.dtype.names
        values = jax.device_get(tuple(self._events[name] for name in names))
        for name, value in zip(names, values):
            result[name] = numpy.asarray(value)
        return result

    def _host_manager(self):
        host_events = self._host_structured_events()
        manager = EventManager(
            self.opt, self.columns, self.column_types,
            array_minsize=max(self.array_minsize, len(host_events)),
            **self.global_params)
        manager._events[:len(host_events)] = host_events
        manager._events_size = len(host_events)
        manager.template_params = self.template_params
        manager.template_index = self.template_index
        manager.write_performance = self.write_performance
        for name in ("run_time", "setup_time", "ncores", "nfilters",
                     "ntemplates"):
            if hasattr(self, name):
                setattr(manager, name, getattr(self, name))
        return manager

    def write_events(self, outname):
        self.make_output_dir(outname)
        if not outname.endswith((".hdf", ".h5")):
            raise ValueError("Unsupported event output file format")
        self.write_to_hdf(outname)

    def write_to_hdf(self, outname):
        """Transfer compact results once and reuse the exact native writer."""
        self._host_manager().write_to_hdf(outname)

    def save_state(self, tnum_finished, filename):
        self._host_manager().save_state(tnum_finished, filename)

    @classmethod
    def restore_state(cls, filename):
        next_template, manager = EventManager.restore_state(filename)
        columns = tuple(name for name, _ in manager.event_dtype
                        if name != "template_id")
        result = cls(manager.opt, columns,
                     [dtype for name, dtype in manager.event_dtype
                      if name != "template_id"],
                     array_minsize=manager.array_minsize,
                     **manager.global_params)
        result._events = {name: jnp.asarray(manager.events[name])
                          for name in manager.events.dtype.names}
        result.template_params = manager.template_params
        result.template_index = manager.template_index
        return next_template, result

    def save_performance(self, ncores, nfilters, ntemplates, run_time,
                         setup_time):
        self.run_time = run_time
        self.setup_time = setup_time
        self.ncores = ncores
        self.ntemplates = ntemplates
        self.nfilters = nfilters
        self.write_performance = True
