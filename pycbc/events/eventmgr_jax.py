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
            name: jnp.asarray(numpy.concatenate(
                [numpy.asarray(self._events[name])] + [numpy.asarray(chunk[name]) for chunk in chunks]
            ), dtype=self._events[name].dtype)
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
        data = {name: numpy.zeros(length, dtype=value.dtype)
                for name, value in self.template_events.items()}
        data["template_id"] = numpy.full(length, self.template_index,
                                          dtype=numpy.int32)
        for name, value in zip(columns, vectors):
            if value is not None:
                data[name] = numpy.asarray(value, dtype=data[name].dtype)
        self.template_events = {
            name: numpy.concatenate((numpy.asarray(self.template_events[name]), data[name]))
            for name in self.template_events
        }

    def cluster_template_events(self, tcolumn, column, window_size):
        if window_size <= 0 or not len(self.template_events[tcolumn]):
            return
        times = numpy.asarray(self.template_events[tcolumn], dtype=numpy.int32)
        values = numpy.asarray(self.template_events[column])
        indices = numpy.zeros(len(times), dtype=numpy.int32)
        count = findchirp_cluster_over_window_cython(
            times, numpy.asarray(abs(values)), window_size, indices, len(times)
        )
        indices = indices[:count + 1]
        self.template_events = {
            name: numpy.asarray(value)[indices] for name, value in self.template_events.items()
        }

    def finalize_template_events(self):
        if len(self.template_events["template_id"]):
            self._pending_event_chunks.append(self.template_events)
        self.template_events = {
            name: numpy.zeros(0, dtype=self._events[name].dtype)
            for name in self._events
        }

    def add_template_events_direct(
        self,
        tmplt_param,
        time_index,
        snr,
        chisq=None,
        chisq_dof=None,
        sigmasq=None,
        bank_chisq=None,
        bank_chisq_dof=None,
        cont_chisq=None,
        cont_chisq_dof=None,
        sg_chisq=None,
        **extra_columns,
    ):
        """Add clustered events for a single template directly without Python dict copying."""
        length = len(time_index)
        if not length:
            return
        self.new_template(tmplt=tmplt_param)
        chunk = {
            name: numpy.zeros(length, dtype=self._events[name].dtype)
            for name in self._events
        }
        chunk["template_id"] = numpy.full(length, self.template_index, dtype=numpy.int32)
        chunk["time_index"] = numpy.asarray(time_index, dtype=chunk["time_index"].dtype)
        chunk["snr"] = numpy.asarray(snr, dtype=chunk["snr"].dtype)
        if chisq is not None and "chisq" in chunk:
            chunk["chisq"] = numpy.asarray(chisq, dtype=chunk["chisq"].dtype)
        if chisq_dof is not None and "chisq_dof" in chunk:
            chunk["chisq_dof"] = numpy.asarray(chisq_dof, dtype=chunk["chisq_dof"].dtype)
        if sigmasq is not None and "sigmasq" in chunk:
            chunk["sigmasq"] = numpy.asarray(sigmasq, dtype=chunk["sigmasq"].dtype)
        if bank_chisq is not None and "bank_chisq" in chunk:
            chunk["bank_chisq"] = numpy.asarray(bank_chisq, dtype=chunk["bank_chisq"].dtype)
        if bank_chisq_dof is not None and "bank_chisq_dof" in chunk:
            chunk["bank_chisq_dof"] = numpy.asarray(bank_chisq_dof, dtype=chunk["bank_chisq_dof"].dtype)
        if cont_chisq is not None and "cont_chisq" in chunk:
            chunk["cont_chisq"] = numpy.asarray(cont_chisq, dtype=chunk["cont_chisq"].dtype)
        if cont_chisq_dof is not None and "cont_chisq_dof" in chunk:
            chunk["cont_chisq_dof"] = numpy.asarray(cont_chisq_dof, dtype=chunk["cont_chisq_dof"].dtype)
        if sg_chisq is not None and "sg_chisq" in chunk:
            chunk["sg_chisq"] = numpy.asarray(sg_chisq, dtype=chunk["sg_chisq"].dtype)
        for col, val in extra_columns.items():
            if col in chunk and val is not None:
                chunk[col] = numpy.asarray(val, dtype=chunk[col].dtype)
        self._pending_event_chunks.append(chunk)

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
