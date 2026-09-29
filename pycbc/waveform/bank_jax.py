# Copyright (C) 2026 The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""JAX-native template bank abstractions and staged decompression."""

import concurrent.futures
import logging
import threading
import types
import jax.numpy as jnp
import numpy as np

from pycbc.types import FrequencySeries
from pycbc.types.array_jax import JAXArrayData
from pycbc.waveform.waveform import props, get_waveform_filter_length_in_time
from pycbc.waveform.decompress_jax import stage_batched_inline_interp_jax


class TemplateBatchList(list):
    """List of templates carrying the underlying 2D batch tensor."""
    _batch_tensor = None
    _host_batch_tensor = None


class LazyFrequencySeries(FrequencySeries):
    """FrequencySeries that lazily materializes a 1D slice from a 2D tensor."""
    def __init__(self, batch_tensor, pos, delta_f):
        self._batch_tensor = batch_tensor
        self._batch_pos = pos
        self._delta_f = delta_f
        self._epoch = 0.0
        from pycbc import scheme as _scheme
        self._scheme = getattr(_scheme.mgr, "state", None)
        self._saved = {}
        self._data_inst = None

    @property
    def _data(self):
        if self._data_inst is None and self._batch_tensor is not None:
            self._data_inst = JAXArrayData(self._batch_tensor[self._batch_pos])
        return self._data_inst

    @_data.setter
    def _data(self, val):
        self._data_inst = val

    @_data.deleter
    def _data(self):
        self._data_inst = None
        self._batch_tensor = None

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        self._data_inst = val

    @data.deleter
    def data(self):
        self._data_inst = None
        self._batch_tensor = None

    @property
    def shape(self):
        if self._data_inst is not None:
            return self._data_inst.shape
        return (self._batch_tensor.shape[1],)

    def __len__(self):
        if self._data_inst is not None:
            return len(self._data_inst)
        return int(self._batch_tensor.shape[1])


def execute_batch_decompression_jax(bank, indices):
    """Perform host decompression and device transfer for indices."""
    from pycbc.waveform.bank import sigma_cached, find_variable_start_frequency

    b = len(indices)
    if b == 0:
        return tuple(), None, None, {}

    logging.info(
        "Loading template batch %d-%d (%d templates) staged decompression",
        indices[0] + 1, indices[-1] + 1, b
    )

    amps_list = []
    phases_list = []
    freqs_list = []
    imins = []
    starts = []
    ends = []
    counts = []
    interpolations = []
    metadata = []

    flen = bank.filter_length
    df = bank.delta_f

    bank_lock = getattr(bank, "_bank_lock", None)
    if bank_lock is None:
        bank._bank_lock = threading.Lock()
        bank_lock = bank._bank_lock

    with bank_lock:
        for index in indices:
            tmplt_hash = bank.table.template_hash[index]
            group = bank.filehandler['compressed_waveforms'][str(tmplt_hash)]
            interpolation = (
                bank.waveform_decompression_method
                if bank.waveform_decompression_method is not None
                else group.attrs['interpolation']
            )
            interpolations.append(interpolation)
            real_dtype = (
                np.float32 if bank.dtype == np.complex64 else np.float64
            )
            amp = np.asarray(group['amplitude'], dtype=real_dtype)
            phase = np.asarray(group['phase'], dtype=real_dtype)
            freq = np.asarray(group['sample_points'], dtype=real_dtype)

            approximant = bank.approximant(index)
            f_end = bank.end_frequency(index)
            if f_end is None or f_end >= (flen * df):
                f_end = (flen - 1) * df

            f_low = find_variable_start_frequency(
                approximant, bank.table[index], bank.f_lower,
                bank.max_template_length, **bank.extra_args
            )

            p = props(bank.table[index])
            p.pop('approximant', None)
            try:
                tmpltdur = bank.table[index].template_duration
            except AttributeError:
                tmpltdur = None
            if tmpltdur is None or tmpltdur == 0.0:
                tmpltdur = get_waveform_filter_length_in_time(approximant, **p)
            bank.table[index].template_duration = tmpltdur

            k = len(freq)
            counts.append(k)
            imin = int(np.searchsorted(freq, f_low, side='right')) - 1
            imins.append(imin)
            s_idx = int(np.ceil(f_low / df))
            starts.append(s_idx)
            ends.append(flen)

            amps_list.append(amp)
            phases_list.append(phase)
            freqs_list.append(freq)

            metadata.append({
                'approximant': approximant,
                'f_low': f_low,
                'f_end': f_end,
                'tmpltdur': tmpltdur,
                'index': index
            })

    dtype = jnp.complex64 if bank.dtype == np.complex64 else jnp.complex128
    host_waveforms, batch_waveforms = stage_batched_inline_interp_jax(
        amps_list, phases_list, freqs_list,
        imins, starts, ends, counts,
        interpolations,
        df, flen, dtype=dtype
    )

    tmpls = {}
    for pos, meta in enumerate(metadata):
        idx = meta['index']
        fs = LazyFrequencySeries(batch_waveforms, pos, df)
        fs.params = bank.table[idx]
        fs.approximant = meta['approximant']
        fs.f_lower = meta['f_low']
        fs.min_f_lower = bank.min_f_lower
        fs.end_idx = int(meta['f_end'] / df)
        fs.end_frequency = meta['f_end']
        fs.chirp_length = meta['tmpltdur']
        fs.length_in_time = meta['tmpltdur']
        fs.sigmasq = types.MethodType(sigma_cached, fs)
        fs._sigmasq = {}
        tmpls[idx] = fs

    return tuple(indices), host_waveforms, batch_waveforms, tmpls


def decompress_batch_jax(bank, indices):
    """Decompress a host batch once, then transfer it to the JAX device."""
    t_indices, host_waveforms, batch_waveforms, tmpls = (
        execute_batch_decompression_jax(bank, indices)
    )
    bank._last_batch_tensor = batch_waveforms
    bank._last_batch_host_tensor = host_waveforms
    bank._last_batch_indices = t_indices
    for idx, fs in tmpls.items():
        bank._template_cache[idx] = fs


def prefetch_batch_jax(bank, indices):
    """Pre-decompress the next template batch in a background thread."""
    compressed_ok = (
        bank.has_compressed_waveforms and bank.enable_compressed_waveforms
    )
    if not indices or not compressed_ok:
        return
    t_indices = tuple(indices)
    if getattr(bank, "_prefetch_indices", None) == t_indices:
        return
    if not hasattr(bank, "_prefetch_executor"):
        bank._prefetch_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1
        )
    bank._prefetch_indices = t_indices
    bank._prefetch_future = bank._prefetch_executor.submit(
        execute_batch_decompression_jax, bank, list(indices)
    )


def clear_batch_cache_jax(bank, indices=None, collect=True):
    """Evict decompressed templates from cache to release GPU VRAM."""
    if hasattr(bank, "_template_cache"):
        if indices is None:
            to_clear = list(bank._template_cache.keys())
        else:
            to_clear = list(indices)
        for idx in to_clear:
            tmpl = bank._template_cache.pop(idx, None)
            if tmpl is not None:
                if hasattr(tmpl, "sigma_view"):
                    try:
                        del tmpl.sigma_view
                    except Exception:
                        pass
                if hasattr(tmpl, "_batch_tensor"):
                    tmpl._batch_tensor = None
                if hasattr(tmpl, "_data_inst"):
                    tmpl._data_inst = None
                elif hasattr(tmpl, "_data"):
                    try:
                        del tmpl._data
                    except Exception:
                        pass
                if hasattr(tmpl, "_sigmasq"):
                    tmpl._sigmasq.clear()
        bank._last_batch_tensor = None
        bank._last_batch_host_tensor = None
        bank._last_batch_indices = None
        if collect:
            import gc
            gc.collect()


def get_batch_jax(bank, indices):
    """Return a list of templates for given indices with JAX batching."""
    if (
        getattr(bank, "_prefetch_indices", None) == tuple(indices)
        and getattr(bank, "_prefetch_future", None) is not None
    ):
        _, host_waveforms, batch_waveforms, tmpls = (
            bank._prefetch_future.result()
        )
        bank._prefetch_indices = None
        bank._prefetch_future = None
        bank._last_batch_tensor = batch_waveforms
        bank._last_batch_host_tensor = host_waveforms
        bank._last_batch_indices = tuple(indices)
        if hasattr(bank, "_template_cache"):
            bank._template_cache.clear()
        else:
            from pycbc.opt import LimitedSizeDict
            bank._template_cache = LimitedSizeDict(size_limit=len(indices) * 2)
        for idx, fs in tmpls.items():
            bank._template_cache[idx] = fs

    if not hasattr(bank, "_template_cache"):
        from pycbc.opt import LimitedSizeDict
        b_size = max(len(indices), 64)
        bank._template_cache = LimitedSizeDict(size_limit=b_size * 2)

    # Evict keys not in current indices without gc.collect() inside search loop
    keys_to_evict = [k for k in bank._template_cache if k not in indices]
    for k in keys_to_evict:
        bank._template_cache.pop(k, None)

    missing = [idx for idx in indices if idx not in bank._template_cache]
    if missing:
        if bank.has_compressed_waveforms and bank.enable_compressed_waveforms:
            decompress_batch_jax(bank, missing)
        else:
            for idx in missing:
                bank._template_cache[idx] = bank[idx]

    res = TemplateBatchList([bank._template_cache[idx] for idx in indices])
    if getattr(bank, "_last_batch_indices", None) == tuple(indices):
        res._batch_tensor = getattr(bank, "_last_batch_tensor", None)
        res._host_batch_tensor = getattr(bank, "_last_batch_host_tensor", None)
    return res
