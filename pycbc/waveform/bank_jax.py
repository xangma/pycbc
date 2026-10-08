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
from pycbc.types.array_jax import JAXArrayData, _reference_enabled
from pycbc.scheme import current_backend_key
from pycbc.waveform.waveform import props, get_waveform_filter_length_in_time
from pycbc.waveform.decompress_jax import (
    stage_batched_inline_interp_jax,
    stage_batched_device_interp_jax,
    _target_device,
)


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


def waveform_parameters_jax(bank, index, delta_f=None, approximant=None,
                            f_lower=None, f_final=None):
    """Resolve the original bank geometry without changing CPU bank behavior."""
    from pycbc import DYN_RANGE_FAC
    from pycbc.waveform.bank import find_variable_start_frequency
    params = props(bank.table[index], **bank.extra_args)
    approximant = bank.approximant(index) if approximant is None else approximant
    if hasattr(bank, 'sample_rate'):
        delta_f = (bank.freq_resolution_for_template(index)
                   if delta_f is None else delta_f)
        flen = round(bank.sample_rate / (2 * delta_f) + 1)
        delta_t = 1.0 / bank.sample_rate
        flow = bank.table[index].f_lower
    else:
        delta_f, delta_t = bank.delta_f, bank.delta_t
        flen = bank.filter_length
        flow = (find_variable_start_frequency(
            approximant, bank.table[index], bank.f_lower,
            bank.max_template_length) if f_lower is None else f_lower)
    if f_final is None:
        f_final = bank.end_frequency(index)
        if f_final is None or f_final >= flen * delta_f:
            f_final = (flen - 1) * delta_f
    params.update(approximant=approximant, f_lower=flow, f_final=f_final,
                  delta_f=delta_f, delta_t=delta_t, distance=1.0 / DYN_RANGE_FAC)
    return params, flen


def get_template_jax(bank, index, delta_f=None):
    """Generate or decompress a single JAX template with original metadata."""
    if (getattr(bank, 'has_compressed_waveforms', False)
            and getattr(bank, 'enable_compressed_waveforms', False)):
        return get_batch_jax(bank, [index])[0]
    from .diffgw_jax import generate_batch
    _, templates = generate_batch(bank, [index], delta_f=delta_f)
    template = templates[0]
    output = getattr(bank, 'out', None)
    if output is not None:
        output[:len(template)] = template
        template._data = output[:len(template)]._data
    return template


def execute_batch_decompression_jax(bank, indices, power_chisq=None, psd=None):
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

    real_dtype = (
        np.float32 if bank.dtype == np.complex64 else np.float64
    )

    with bank_lock:
        cw_group = getattr(bank, "_cw_group", None)
        if cw_group is None:
            cw_group = bank.filehandler['compressed_waveforms']
            bank._cw_group = cw_group

        for index in indices:
            tmplt_hash = bank.table.template_hash[index]
            group = cw_group[str(tmplt_hash)]
            interpolation = (
                bank.waveform_decompression_method
                if bank.waveform_decompression_method is not None
                else group.attrs['interpolation']
            )
            if interpolation == "device_linear":
                interpolation = "inline_linear"
            interpolations.append(interpolation)
            amp = group['amplitude'][()]
            if amp.dtype != real_dtype:
                amp = amp.astype(real_dtype)
            phase = group['phase'][()]
            if phase.dtype != real_dtype:
                phase = phase.astype(real_dtype)
            freq = group['sample_points'][()]
            if freq.dtype != real_dtype:
                freq = freq.astype(real_dtype)

            approximant = bank.approximant(index)
            f_end = bank.end_frequency(index)
            if f_end is None or f_end >= (flen * df):
                f_end = (flen - 1) * df

            f_low = find_variable_start_frequency(
                approximant, bank.table[index], bank.f_lower,
                bank.max_template_length
            )

            try:
                tmpltdur = bank.table[index].template_duration
            except AttributeError:
                tmpltdur = None
            if tmpltdur is None or tmpltdur == 0.0:
                p = props(bank.table[index])
                p.pop('approximant', None)
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

    target_dev = _target_device()
    is_cuda = (
        target_dev is not None
        and getattr(target_dev, "platform", "") in ("cuda", "gpu")
    )
    decomp_method = getattr(bank, "waveform_decompression_method", None)
    use_device_decomp = not _reference_enabled("decompress") and is_cuda and (
        decomp_method == "device_linear"
        or getattr(bank, "enable_device_decompression", False)
    )
    if use_device_decomp and any(method != "inline_linear"
                                 for method in interpolations):
        raise NotImplementedError(
            "Device decompression supports inline_linear interpolation only")

    dtype = jnp.complex64 if bank.dtype == np.complex64 else jnp.complex128
    if use_device_decomp:
        host_waveforms, batch_waveforms = stage_batched_device_interp_jax(
            amps_list, phases_list, freqs_list,
            starts, ends, counts,
            df, flen, dtype=dtype, target_dev=target_dev,
        )
    else:
        host_waveforms, batch_waveforms = stage_batched_inline_interp_jax(
            amps_list, phases_list, freqs_list,
            imins, starts, ends, counts,
            interpolations,
            df, flen, dtype=dtype,
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

    if (
        power_chisq is not None
        and getattr(power_chisq, "do", False)
        and psd is not None
    ):
        try:
            from pycbc.vetoes.chisq_jax import cache_batch_power_chisq_bins_jax
            batch_list = TemplateBatchList([tmpls[idx] for idx in indices])
            batch_list._batch_tensor = batch_waveforms
            cache_batch_power_chisq_bins_jax(power_chisq, batch_list, psd)
        except Exception as e:
            logging.warning("Pre-caching power chisq bins failed: %s", e)

    return tuple(indices), host_waveforms, batch_waveforms, tmpls


def decompress_batch_jax(bank, indices, power_chisq=None, psd=None):
    """Decompress a host batch once, then transfer it to the JAX device."""
    t_indices, host_waveforms, batch_waveforms, tmpls = (
        execute_batch_decompression_jax(
            bank, indices, power_chisq=power_chisq, psd=psd
        )
    )
    bank._template_cache_backend_key = current_backend_key()
    bank._last_batch_tensor = batch_waveforms
    bank._last_batch_host_tensor = host_waveforms
    bank._last_batch_indices = t_indices
    if not hasattr(bank, "_template_cache"):
        from pycbc.opt import LimitedSizeDict
        bank._template_cache = LimitedSizeDict(size_limit=max(64, len(indices)) * 2)
    for idx, fs in tmpls.items():
        bank._template_cache[idx] = fs


def prefetch_batch_jax(bank, indices, power_chisq=None, psd=None):
    """Pre-decompress the next template batch in a background thread."""
    compressed_ok = (
        bank.has_compressed_waveforms and bank.enable_compressed_waveforms
    )
    if not indices or not compressed_ok:
        return
    t_indices = tuple(indices)
    backend_key = current_backend_key()
    if (getattr(bank, "_prefetch_indices", None) == t_indices
            and getattr(bank, "_prefetch_backend_key", None) == backend_key):
        return
    if (
        hasattr(bank, "_template_cache")
        and all(i in bank._template_cache for i in t_indices)
    ):
        return
    if not hasattr(bank, "_prefetch_executor"):
        bank._prefetch_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1
        )
    bank._prefetch_indices = t_indices
    bank._prefetch_backend_key = backend_key
    bank._prefetch_future = bank._prefetch_executor.submit(
        execute_batch_decompression_jax, bank, list(indices), power_chisq, psd
    )


def clear_batch_cache_jax(bank, indices=None, collect=True):
    """Evict decompressed templates from cache to release GPU VRAM."""
    if hasattr(bank, "_template_cache"):
        if indices is None:
            to_clear = list(bank._template_cache.keys())
        else:
            to_clear = list(indices)
        for idx in to_clear:
            bank._template_cache.pop(idx, None)
        bank._last_batch_tensor = None
        bank._last_batch_host_tensor = None
        bank._last_batch_indices = None
        if collect:
            import gc
            gc.collect()


def get_batch_jax(bank, indices):
    """Return a list of templates for given indices with JAX batching."""
    backend_key = current_backend_key()
    if getattr(bank, '_prefetch_backend_key', None) != backend_key:
        future = getattr(bank, '_prefetch_future', None)
        if future is not None:
            future.cancel()
        bank._prefetch_future = None
        bank._prefetch_indices = None
    if getattr(bank, '_template_cache_backend_key', None) != backend_key:
        if hasattr(bank, '_template_cache'):
            bank._template_cache.clear()
        bank._template_cache_backend_key = backend_key
        bank._last_batch_indices = None
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
            from .diffgw_jax import generate_batch
            tensor, templates = generate_batch(bank, missing)
            for index, template in zip(missing, templates):
                bank._template_cache[index] = template
            bank._last_batch_indices = tuple(missing)
            bank._last_batch_tensor = tensor
            bank._last_batch_host_tensor = None

    res = TemplateBatchList([bank._template_cache[idx] for idx in indices])
    if getattr(bank, "_last_batch_indices", None) == tuple(indices):
        res._batch_tensor = getattr(bank, "_last_batch_tensor", None)
        res._host_batch_tensor = getattr(bank, "_last_batch_host_tensor", None)
    return res
