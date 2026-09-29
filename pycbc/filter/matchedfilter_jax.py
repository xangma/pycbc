# Copyright (C) 2026
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

"""JAX backend for matched filtering primitives, correlation,
sigmasq, and match calculations.
"""

import functools
import numpy as np
import jax
import jax.numpy as jnp

from pycbc.events import ranking
from pycbc.filter.matchedfilter import _BaseCorrelator, get_cutoff_indices
from pycbc.types.array_jax import (
    JAXArrayData,
    _ensure_x64,
    to_jax,
)


def batch_template_matrix_jax(templates, size):
    """Pack immutable live templates on the active JAX device."""
    return jnp.stack([to_jax(template)[:size] for template in templates], axis=0)


@jax.jit
def batch_template_power_jax(template_matrix, delta_f):
    """Cache the PSD-independent part of generic live normalization."""
    return (
        template_matrix.real ** 2 + template_matrix.imag ** 2
    ) * (4.0 * delta_f)


@jax.jit
def _live_generic_norms_core(template_power, psd, kmins, kmaxs):
    """Evaluate all generic live template norms in one compiled reduction."""
    from pycbc.psd.estimate_jax import _reciprocal_numpy_compat

    bins = jnp.arange(template_power.shape[1])[None, :]
    support = (bins >= kmins[:, None]) & (bins < kmaxs[:, None])
    weighted = template_power * _reciprocal_numpy_compat(psd)[None, :]
    return jnp.sum(
        jnp.where(support, weighted, 0.0), axis=-1, dtype=jnp.float64
    )


def live_template_norms_jax(
        templates, psd, template_matrix=None, template_power=None):
    """Compute live norms using the same support and accumulation as ``sigma_cached``.

    The native generic cache forms a float32 squared-magnitude view, multiplies
    it by ``4 * delta_f``, weights it by the cached inverse PSD, and accumulates
    in float64.  Keeping those boundaries avoids the former direct float32
    reduction and honors template end-frequency metadata.
    """
    _ensure_x64()
    delta_f = float(templates[0].delta_f)
    psd_j = to_jax(psd)
    rows = [None] * len(templates)
    generic_indices = []
    generic_kmins = []
    generic_kmaxs = []
    for index, template in enumerate(templates):
        flow = getattr(template, "min_f_lower", None) or getattr(
            template, "f_lower", 0.0)
        fhigh = getattr(template, "end_frequency", None)
        template_size = len(getattr(template, "data", template))
        n_pts = (template_size - 1) * 2
        if fhigh is None and not hasattr(template, "end_frequency"):
            # Lightweight array-like templates used by callers/tests do
            # not carry the FrequencySeries end-frequency metadata.
            kmin = int(flow / delta_f) if flow else 1
            kmax = template_size
        else:
            kmin, kmax = get_cutoff_indices(flow, fhigh, delta_f, n_pts)

        # sigma_cached has a special precomputed norm path for SPAtmplt.
        # Reuse that vector here, then perform its same endpoint subtraction.
        from pycbc import waveform
        if waveform.waveform_norm_exists(getattr(template, "approximant", "")):
            vector = waveform.get_waveform_filter_norm(
                template.approximant, psd, len(psd), delta_f, flow)
            if hasattr(template, "sigma_scale"):
                scale = template.sigma_scale
            else:
                from pycbc import DYN_RANGE_FAC
                amp_norm = waveform.get_template_amplitude_norm(
                    template.params, approximant=template.approximant)
                amp_norm = 1 if amp_norm is None else amp_norm
                scale = (DYN_RANGE_FAC * amp_norm) ** 2.0
            rows[index] = scale * (
                to_jax(vector)[template.end_idx - 1]
                - to_jax(vector)[int(float(template.f_lower) / delta_f)])
            continue

        generic_indices.append(index)
        generic_kmins.append(kmin)
        generic_kmaxs.append(kmax)

    if generic_indices:
        if template_power is not None:
            generic_power = to_jax(template_power)[
                jnp.asarray(generic_indices, dtype=jnp.int32)
            ]
        else:
            if template_matrix is None:
                generic_matrix = jnp.stack(
                    [to_jax(templates[index]) for index in generic_indices]
                )
            else:
                generic_matrix = to_jax(template_matrix)[
                    jnp.asarray(generic_indices, dtype=jnp.int32)
                ]
            generic_power = batch_template_power_jax(
                generic_matrix, delta_f)
        generic_norms = _live_generic_norms_core(
            generic_power,
            psd_j,
            jnp.asarray(generic_kmins, dtype=jnp.int32),
            jnp.asarray(generic_kmaxs, dtype=jnp.int32),
        )
        for row, index in enumerate(generic_indices):
            rows[index] = generic_norms[row]
    return jnp.stack(rows)


def live_veto_buffer_jax(size, dtype):
    """Allocate a standalone JAX veto correlation buffer."""
    from pycbc.types import zeros
    return zeros(size, dtype=dtype)


def _cache_live_veto_bins_jax(power_chisq, veto_info):
    """Populate exact bin caches from the live device template matrices."""
    from collections import defaultdict
    from pycbc.benchmark import stage_event
    from pycbc.vetoes.chisq_jax import cache_batch_power_chisq_bins_jax
    from pycbc.waveform.bank import TemplateBatchList

    by_source = defaultdict(list)
    psds = {}
    for info in veto_info:
        _, _, _, template, stilde = info[:5]
        source = info[5] if len(info) > 5 else None
        position = info[6] if len(info) > 6 else None
        key = (id(stilde.psd), id(source))
        psds[key] = stilde.psd
        by_source[key].append((template, source, position))

    for key, entries in by_source.items():
        templates = TemplateBatchList([entry[0] for entry in entries])
        source = entries[0][1]
        positions = [entry[2] for entry in entries]
        if source is not None and all(position is not None
                                      for position in positions):
            rows = to_jax(source)[jnp.asarray(positions, dtype=jnp.int32)]
            templates._batch_tensor = rows
        stage_event("filter_veto_bins", "start", templates=len(templates))
        cache_batch_power_chisq_bins_jax(
            power_chisq, templates, psds[key]
        )
        stage_event("filter_veto_bins", "end", templates=len(templates))


@jax.jit
def _batched_live_chisq_core(corr_tensor, pts, bins_rel_all, kmin_f, n_time_f):
    """JIT-compiled batched power chisq prefix sum for live triggers."""
    from pycbc.vetoes.chisq_jax import _time_shift_phase
    n_slice = corr_tensor.shape[-1]

    def _point_chisq(row, pt, bin_rel):
        phases = _time_shift_phase(
            n_slice, kmin_f, jnp.reshape(pt, (1,)), n_time_f, jnp.complex128
        )[:, 0]
        weighted = row.astype(jnp.complex128) * phases
        C = jnp.cumsum(weighted)
        C_padded = jnp.pad(C, (1, 0))
        edges = jnp.clip(bin_rel, 0, n_slice)
        i0 = edges[:-1]
        i1 = edges[1:]
        zb = C_padded[i1] - C_padded[i0]
        return (jnp.sum(zb.real ** 2 + zb.imag ** 2)).astype(jnp.float32)

    return jax.vmap(_point_chisq)(corr_tensor, pts, bins_rel_all)


def _batched_live_vetoes_gpu(control, results, veto_info, power_chisq):
    """Evaluate power chisq for all batch triggers in a single GPU pass."""
    if not veto_info:
        return None
    try:
        from pycbc.vetoes.chisq_jax import _require_point_chisq_x64

        bins_list = []
        for info in veto_info:
            tmpl = info[3]
            stilde = info[4]
            b_edges = None
            if hasattr(power_chisq, "cached_chisq_bins"):
                b_edges = power_chisq.cached_chisq_bins(tmpl, stilde.psd)
            elif (
                hasattr(tmpl, "_bin_cache")
                and id(stilde.psd) in tmpl._bin_cache
            ):
                b_edges = tmpl._bin_cache[id(stilde.psd)]
            if b_edges is None:
                return None
            bins_list.append(b_edges)

        num_bins_set = set(len(b) for b in bins_list)
        if len(num_bins_set) != 1:
            return None

        bins_arr = np.array(bins_list, dtype=np.int32)
        htilde0 = veto_info[0][3]
        stilde = veto_info[0][4]
        kmin = int(htilde0.f_lower / htilde0.delta_f)
        max_kmax = int(np.max(bins_arr[:, -1]))
        size = len(htilde0.cout) if hasattr(htilde0, "cout") else len(stilde)
        n_time = size

        if all(
            len(info) > 6 and info[5] is not None and info[6] is not None
            for info in veto_info
        ):
            source = veto_info[0][5]
            positions = jnp.asarray(
                [info[6] for info in veto_info], dtype=jnp.int32
            )
            templates_mat = to_jax(source)[positions]
        else:
            templates_mat = jnp.stack([to_jax(info[3]) for info in veto_info])

        stilde_dev = to_jax(stilde)
        corr_tensor = (
            jnp.conj(templates_mat[:, kmin:max_kmax])
            * stilde_dev[None, kmin:max_kmax]
        )

        pts = jnp.asarray(
            [int(info[2]) for info in veto_info], dtype=jnp.int32
        )
        bins_rel_all = jnp.asarray(bins_arr - kmin, dtype=jnp.int32)
        snr_arr = jnp.asarray(
            [
                info[0][0] if hasattr(info[0], "__getitem__") else info[0]
                for info in veto_info
            ]
        )
        norm_arr = jnp.asarray(
            [info[1] for info in veto_info], dtype=jnp.float64
        )

        _require_point_chisq_x64()
        shifts = _batched_live_chisq_core(
            corr_tensor, pts, bins_rel_all, float(kmin), float(n_time)
        )
        num_bins = bins_arr.shape[1] - 1
        dof_val = num_bins * 2 - 2
        chisq_raw = (
            shifts * num_bins - (snr_arr.conj() * snr_arr).real
        ) * (norm_arr ** 2.0)
        chisq_red = (chisq_raw / dof_val).astype(jnp.float32)
        if getattr(power_chisq, "snr_threshold", None):
            above = abs(snr_arr * norm_arr) > power_chisq.snr_threshold
            chisq_red = jnp.where(above, chisq_red, 0.0)
        dof = jnp.full(len(veto_info), dof_val, dtype=jnp.uint32)

        if getattr(control.sg_chisq, "do", False):
            sg_values = []
            for i, info in enumerate(veto_info):
                snrv, norm, l, htilde, stilde_i = info[:5]
                c = chisq_raw[i:i + 1]
                d = dof[i:i + 1]
                sgv = control.sg_chisq.values(
                    stilde_i, htilde, stilde_i.psd, snrv, norm, c, d, [l]
                )
                if sgv is not None:
                    sg_values.append(jnp.asarray(sgv[0], dtype=jnp.float32))
                else:
                    sg_values.append(jnp.asarray(0, dtype=jnp.float32))
            sg_chisq = jnp.stack(sg_values)
        else:
            sg_chisq = jnp.zeros(len(veto_info), dtype=jnp.float32)

        results["chisq"] = chisq_red
        results["chisq_dof"] = dof
        results["sg_chisq"] = sg_chisq

        if control.newsnr_threshold:
            keep_mask = ranking.newsnr(results["snr"], chisq_red) >= (
                control.newsnr_threshold
            )
            keep = jnp.asarray(
                np.flatnonzero(jax.device_get(keep_mask)), dtype=jnp.uint32
            )
            for key in results:
                results[key] = results[key][keep]

        return results
    except Exception:
        return None


def process_live_vetoes_jax(control, results, veto_info):
    """Run live veto reductions while retaining numeric result columns on JAX."""
    from pycbc.benchmark import stage_event
    stage_event("filter_veto", "start", triggers=len(veto_info))
    power_chisq = control.power_chisq
    from pycbc import scheme
    device = getattr(scheme.mgr.state, "jax_device", None)
    platform = getattr(device, "platform", None)
    if (
        veto_info
        and platform in ("cuda", "gpu")
        and getattr(power_chisq, "do", False)
        and hasattr(power_chisq, "cached_chisq_bins")
    ):
        # A live batch has at most one candidate per template. CUDA benefits
        # from replacing several executable loads and dispatches with one
        # exact batched scan. Small CPU batches retain the scalar scans, which
        # have better cache locality.
        _cache_live_veto_bins_jax(power_chisq, veto_info)
        batched_res = _batched_live_vetoes_gpu(
            control, results, veto_info, power_chisq
        )
        if batched_res is not None:
            stage_event("filter_veto", "end", triggers=len(veto_info))
            return batched_res

    chisq_values = []
    dof_values = []
    sg_values = []
    veto_corr = {}
    for info in veto_info:
        snrv, norm, l, htilde, stilde = info[:5]
        size = len(htilde.cout)
        if size not in veto_corr:
            veto_corr[size] = live_veto_buffer_jax(size, htilde.dtype)
        corr = veto_corr[size]
        correlate(htilde, stilde, corr)
        c, d = power_chisq.values(corr, snrv, norm,
                                  stilde.psd, [l], htilde)
        c = jnp.asarray(c)
        d = jnp.asarray(d)
        chisq_values.append(jnp.asarray(c[0] / d[0], dtype=jnp.float32))
        dof_values.append(jnp.asarray(d[0], dtype=jnp.uint32))
        sgv = control.sg_chisq.values(stilde, htilde, stilde.psd,
                                      snrv, norm, c, d, [l])
        if sgv is not None:
            sg_values.append(jnp.asarray(sgv[0], dtype=jnp.float32))
        else:
            sg_values.append(jnp.asarray(0, dtype=jnp.float32))
    if veto_info:
        chisq = jnp.stack(chisq_values)
        dof = jnp.stack(dof_values)
        sg_chisq = jnp.stack(sg_values)
    else:
        chisq = jnp.zeros(0, dtype=jnp.float32)
        dof = jnp.zeros(0, dtype=jnp.uint32)
        sg_chisq = jnp.zeros(0, dtype=jnp.float32)
    results["chisq"] = chisq
    results["chisq_dof"] = dof
    results["sg_chisq"] = sg_chisq
    if control.newsnr_threshold:
        keep_mask = ranking.newsnr(results["snr"], chisq) >= (
            control.newsnr_threshold
        )
        keep = jnp.asarray(
            np.flatnonzero(jax.device_get(keep_mask)), dtype=jnp.uint32
        )
        for key in results:
            results[key] = results[key][keep]
    stage_event("filter_veto", "end", triggers=len(veto_info))
    return results


def combine_live_results_jax(results):
    """Concatenate live batches without moving numeric JAX columns host-side."""
    combined = {}
    for key in results[0]:
        values = [batch[key] for batch in results]
        kinds = []
        for value in values:
            dtype = getattr(value, "dtype", None)
            if dtype is None:
                dtype = np.asarray(value).dtype
            kinds.append(np.dtype(dtype).kind)
        if all(kind in "biufc" for kind in kinds):
            combined[key] = jnp.concatenate([jnp.asarray(value)
                                             for value in values])
        else:
            combined[key] = np.concatenate(values)
    return combined


@jax.jit
def _live_select_peaks(peaks, norms, threshold, abort_threshold):
    """Scale and select a live batch without per-template synchronization."""
    scaled = peaks * norms
    magnitudes = jnp.abs(scaled)
    # Preserve the scalar path's NaN behavior: ``not (s < threshold)``.
    accepted = jnp.logical_not(magnitudes < threshold)
    abort = accepted & (magnitudes > abort_threshold)
    return scaled, accepted, abort


def live_process_batch_jax(self):
    """Process only a single batch group of data"""
    from pycbc.filter.matchedfilter import logger
    from pycbc.benchmark import stage_event
    if self.block_id == len(self.tgroups):
        return None, None

    tgroup = self.tgroups[self.block_id]
    psize = self.chunk_tsamples[self.block_id]
    mid = self.mids[self.block_id]
    stage_event("filter_overwhiten", "start", templates=len(tgroup))
    stilde = self.data.overwhitened_data(tgroup[0].delta_f)
    stage_event("filter_overwhiten", "end", templates=len(tgroup))
    psd = stilde.psd
    template_matrix = getattr(
        self.corr[self.block_id], "_jax_template_matrix", None
    )
    template_power = getattr(
        self.corr[self.block_id], "_jax_template_power", None
    )
    stage_event("filter_norms", "start", templates=len(tgroup))
    native_norms = live_template_norms_jax(
        tgroup, psd, template_matrix=template_matrix,
        template_power=template_power,
    )
    stage_event("filter_norms", "end", templates=len(tgroup))

    valid_end = int(psize - self.data.trim_padding)
    valid_start = int(valid_end - self.data.blocksize * self.data.sample_rate)

    seg = slice(valid_start, valid_end)

    stage_event("filter_correlation", "start", templates=len(tgroup))
    self.corr[self.block_id].execute(stilde)
    stage_event("filter_correlation", "end", templates=len(tgroup))
    # Every transform in the JAX scheme must remain in the JAX backend,
    # including on a CPU device.  A host FFT here would break device
    # residency and make the live trigger path use a different algorithm.
    stage_event("filter_ifft", "start", templates=len(tgroup))
    self.ifts[mid].execute()
    stage_event("filter_ifft", "end", templates=len(tgroup))

    self.block_id += 1

    result = {}
    tkeys = tgroup[0].params.dtype.names
    for key in tkeys:
        result[key] = []

    veto_info = []

    # Reduce, normalize, and select the full group before the only host
    # synchronization.  The host loop below handles sparse metadata and veto
    # objects, not accelerator decisions.
    stage_event("filter_peak_select", "start", templates=len(tgroup))
    jax_peaks = batch_peak_values(self.out_mem[mid], len(tgroup), psize, seg)
    if jax_peaks is None:
        raise ValueError("JAX live peak reduction could not process the batch")
    peak_indices, peak_values = jax_peaks
    norms = (4.0 * tgroup[0].delta_f) / jnp.sqrt(native_norms)
    abort_threshold = (
        jnp.inf if self.snr_abort_threshold is None
        else float(self.snr_abort_threshold)
    )
    scaled_peaks, accepted, abort = _live_select_peaks(
        peak_values, norms, float(self.snr_threshold), abort_threshold
    )
    host_peak_indices, host_accepted, host_abort = jax.device_get(
        (peak_indices, accepted, abort)
    )
    stage_event("filter_peak_select", "end", templates=len(tgroup),
                accepted=int(np.count_nonzero(host_accepted)))

    if np.any(host_abort):
        logger.info("We are seeing some *really* high SNRs, let's "
                    "assume they aren't signals and just give up")
        return False, []

    accepted_indices = np.flatnonzero(host_accepted)
    selected = jnp.asarray(accepted_indices, dtype=jnp.int32)
    selected_scaled = scaled_peaks[selected]
    selected_norms = norms[selected]
    selected_sigmasq = native_norms[selected]

    result.update({
        "snr": jnp.abs(selected_scaled),
        "coa_phase": jnp.angle(selected_scaled),
        "end_time": jnp.asarray(
            float(self.data.start_time)
            + host_peak_indices[accepted_indices] / self.data.sample_rate,
            dtype=jnp.float64,
        ),
        "template_id": jnp.asarray(
            [tgroup[idx].id for idx in accepted_indices], dtype=jnp.uint64
        ),
        "sigmasq": selected_sigmasq.astype(jnp.float32),
    })

    for result_index, idx in enumerate(accepted_indices):
        htilde = tgroup[idx]
        if hasattr(htilde, 'time_offset'):
            if 'time_offset' not in result:
                result['time_offset'] = []

        l = int(host_peak_indices[idx]) + valid_start
        snrv = peak_values[idx:idx + 1]
        norm = selected_norms[result_index]
        veto_info.append((
            snrv, norm, l, htilde, stilde, template_matrix, int(idx)
        ))
        if not hasattr(htilde, 'dict_params'):
            htilde.dict_params = {}
            for key in tkeys:
                htilde.dict_params[key] = htilde.params[key]

        for key in tkeys:
            result[key].append(htilde.dict_params[key])

        if hasattr(htilde, 'time_offset'):
            result['time_offset'].append(htilde.time_offset)

    for key in tkeys:
        values = np.array(result[key])
        if values.dtype.kind in "biufc":
            result[key] = jnp.asarray(values)
        else:
            result[key] = values

    if 'time_offset' in result:
        result['time_offset'] = jnp.asarray(result['time_offset'])

    return result, veto_info

def _set_output_array(z, val):
    """Store result val into output container z."""
    if isinstance(z, JAXArrayData):
        z.set_array(val)
    elif hasattr(z, "_data") and isinstance(z._data, JAXArrayData):
        z._data.set_array(val)
    elif hasattr(z, "data") and isinstance(z.data, JAXArrayData):
        z.data.set_array(val)
    elif hasattr(z, "_data"):
        try:
            z._data[:] = np.asarray(val)
        except Exception:
            z._data = val
    elif hasattr(z, "data"):
        try:
            z.data[:] = np.asarray(val)
        except Exception:
            z.data = val
    else:
        try:
            z[:] = np.asarray(val)
        except Exception:
            pass


@jax.jit
def correlate_jax(x, y):
    """Pure JAX elementwise conjugate multiplication: conj(x) * y."""
    return jnp.conj(x) * y


_fast_conj_mul = correlate_jax
_batch_correlate_products = correlate_jax


def correlate(x, y, z):
    """Elementwise z = conj(x) * y in JAX scheme."""
    _ensure_x64()
    x_arr = to_jax(x)
    y_arr = to_jax(y)
    prod = correlate_jax(x_arr, y_arr)
    _set_output_array(z, prod)


class JAXCorrelator(_BaseCorrelator):
    """JAX correlator engine for persistent/repeated correlation."""

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def correlate(self):
        """Execute elementwise correlation."""
        correlate(self.x, self.y, self.z)


def _correlate_factory(x, y, z):
    """Factory returning the JAX correlator class."""
    return JAXCorrelator


@functools.partial(jax.jit, static_argnums=(4,))
def _batch_correlate_update(templates, y, parent, base, row_stride):
    """Correlate and publish one contiguous sibling-view block."""
    products = correlate_jax(templates, y)
    block_len = templates.shape[0] * row_stride
    block = jax.lax.dynamic_slice(parent, (base,), (block_len,))
    block = block.reshape(templates.shape[0], row_stride)
    block = block.at[:, :templates.shape[1]].set(products)
    return jax.lax.dynamic_update_slice(parent, block.reshape(-1), (base,))


def batch_correlate_execute(self, y):
    """Vectorized batch correlation for BatchCorrelator in JAX scheme."""
    _ensure_x64()
    size = self.size
    y_arr = to_jax(y)[:size]

    # Lazy bank templates retain the complete 2-D device tensor.  When this
    # batch is an aligned contiguous run of rows, use it directly.  Reading
    # each LazyFrequencySeries first would materialize a separate static row
    # slice before stack, which is particularly costly during CUDA startup.
    templates = getattr(self, "_jax_template_matrix", None)
    if templates is not None:
        templates = templates[:, :size]

    if templates is None and self.xs:
        batch_tensor = getattr(self.xs[0], "_batch_tensor", None)
        positions = []
        if batch_tensor is not None:
            for x in self.xs:
                if getattr(x, "_batch_tensor", None) is not batch_tensor:
                    batch_tensor = None
                    break
                pos = getattr(x, "_batch_pos", None)
                if pos is None:
                    batch_tensor = None
                    break
                positions.append(int(pos))
        if batch_tensor is not None and positions:
            start = positions[0]
            aligned = positions == list(range(start, start + len(positions)))
            shape = getattr(batch_tensor, "shape", ())
            if aligned and len(shape) == 2 and start >= 0:
                stop = start + len(positions)
                if stop <= shape[0] and size <= shape[1]:
                    templates = to_jax(batch_tensor)[start:stop, :size]

    if templates is None:
        templates = jnp.stack([to_jax(x)[:size] for x in self.xs], axis=0)
    # LiveBatchMatchedFilter passes sibling views into one contiguous output
    # allocation.  Updating each view separately makes JAX rebuild the whole
    # immutable parent for every template.  Fold the updates into one JAX
    # expression and publish the parent once, preserving any untouched tails.
    zdata = []
    for z in self.zs:
        data = z if isinstance(z, JAXArrayData) else getattr(z, "_data", None)
        if not isinstance(data, JAXArrayData):
            zdata = []
            break
        info = data.slice_info
        parent = data.parent
        if parent is None or not isinstance(info, slice) or info.step not in (None, 1):
            zdata = []
            break
        start = 0 if info.start is None else info.start
        stop = parent.shape[0] if info.stop is None else info.stop
        zdata.append((data, parent, start, stop))

    contiguous = False
    if zdata and all(item[1] is zdata[0][1] for item in zdata):
        parent = zdata[0][1]
        row_stride = zdata[0][3] - zdata[0][2]
        base = zdata[0][2]
        contiguous = (
            row_stride >= size and
            all(stop - start == row_stride for _, _, start, stop in zdata) and
            all(start == base + row * row_stride
                for row, (_, _, start, _) in enumerate(zdata)) and
            base + len(zdata) * row_stride <= parent.shape[0]
        )
        if contiguous:
            updated = _batch_correlate_update(
                templates, y_arr, parent.array, base, row_stride,
            )
            parent.set_array(updated)
            return

    # General BatchCorrelator users may provide unrelated output arrays.
    # Retain the existing per-output behavior for those cases.
    products = _batch_correlate_products(templates, y_arr)
    for row, z in enumerate(self.zs):
        _set_output_array(z[:size], products[row])


# ----------------------------------------------------------------------
# Pure Functional JAX API (JIT-compilable & Autodiff-compatible)
# ----------------------------------------------------------------------


def sigmasq_jax(
    htilde,
    psd=None,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
    delta_f=1.0,
):
    """Pure JAX calculation of template loudness (sigma)^2."""
    _ensure_x64()
    h_arr = to_jax(htilde)
    n_pts = (len(h_arr) - 1) * 2
    flow = low_frequency_cutoff
    fhigh = high_frequency_cutoff

    kmin = int(flow / delta_f) if flow else 1
    if kmin < 0:
        raise ValueError("flow cannot be negative")

    if fhigh:
        kmax = min(int(fhigh / delta_f), int((n_pts + 1) / 2.0))
    else:
        kmax = int((n_pts + 1) / 2.0)

    if kmax <= kmin:
        raise ValueError(f"kmax ({kmax}) must be greater than kmin ({kmin})")

    ht = h_arr[kmin:kmax]
    mag_sq = jnp.abs(ht) ** 2

    if psd is not None:
        psd_arr = to_jax(psd)
        mag_sq = mag_sq / psd_arr[kmin:kmax]

    norm = 4.0 * delta_f
    return jnp.sum(mag_sq) * norm


def sigmasq_series_jax(
    htilde,
    psd=None,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
    delta_f=1.0,
):
    """Pure JAX calculation of cumulative power frequency series."""
    _ensure_x64()
    h_arr = to_jax(htilde)
    n_pts = (len(h_arr) - 1) * 2
    flow = low_frequency_cutoff
    fhigh = high_frequency_cutoff

    kmin = int(flow / delta_f) if flow else 1
    if fhigh:
        kmax = min(int(fhigh / delta_f), int((n_pts + 1) / 2.0))
    else:
        kmax = int((n_pts + 1) / 2.0)

    mag_sq = jnp.abs(h_arr) ** 2
    if psd is not None:
        psd_arr = to_jax(psd)
        mag_sq = mag_sq / psd_arr

    sub = jnp.cumsum(mag_sq[kmin:kmax])
    norm = 4.0 * delta_f

    series = jnp.zeros(len(h_arr), dtype=h_arr.real.dtype)
    series = series.at[kmin:kmax].set(sub * norm)
    return series


def matched_filter_core_jax(
    template,
    data,
    psd=None,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
    h_norm=None,
    delta_f=None,
    delta_t=None,
):
    """Pure JAX matched filter core returning (snr, correlation, norm)."""
    _ensure_x64()
    from pycbc.types import TimeSeries
    from pycbc.filter import make_frequency_series

    if isinstance(template, TimeSeries):
        template = make_frequency_series(template)
    if isinstance(data, TimeSeries):
        data = make_frequency_series(data)

    htilde = to_jax(template)
    stilde = to_jax(data)

    if len(htilde) != len(stilde):
        raise ValueError("Length of template and data must match")

    if delta_f is None:
        delta_f = getattr(data, "delta_f", 1.0)
    if delta_t is None:
        delta_t = getattr(
            data, "delta_t", 1.0 / (2.0 * (len(stilde) - 1) * delta_f)
        )

    n_time = (len(stilde) - 1) * 2
    flow = low_frequency_cutoff
    fhigh = high_frequency_cutoff

    kmin = int(flow / delta_f) if flow else 1
    if fhigh:
        kmax = min(int(fhigh / delta_f), int((n_time + 1) / 2.0))
    else:
        kmax = int((n_time + 1) / 2.0)

    qtilde = jnp.zeros(n_time, dtype=stilde.dtype)
    corr_slice = jnp.conj(htilde[kmin:kmax]) * stilde[kmin:kmax]

    if psd is not None:
        psd_arr = to_jax(psd)
        corr_slice = corr_slice / psd_arr[kmin:kmax]

    qtilde = qtilde.at[kmin:kmax].set(corr_slice)

    # Complex-to-complex IFFT of length n_time
    snr_time = jnp.fft.ifft(qtilde) * n_time

    if h_norm is None:
        h_norm = sigmasq_jax(
            htilde,
            psd=psd,
            low_frequency_cutoff=low_frequency_cutoff,
            high_frequency_cutoff=high_frequency_cutoff,
            delta_f=delta_f,
        )

    norm = (4.0 * delta_f) / jnp.sqrt(h_norm)
    return snr_time, qtilde, norm


def matched_filter_jax(
    template,
    data,
    psd=None,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
    sigmasq=None,
    delta_f=None,
    delta_t=None,
):
    """Pure JAX matched filter returning normalized complex/real SNR."""
    snr_time, _, norm = matched_filter_core_jax(
        template,
        data,
        psd=psd,
        low_frequency_cutoff=low_frequency_cutoff,
        high_frequency_cutoff=high_frequency_cutoff,
        h_norm=sigmasq,
        delta_f=delta_f,
        delta_t=delta_t,
    )
    return snr_time * norm


def overlap_jax(
    vec1,
    vec2,
    psd=None,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
    delta_f=1.0,
    normalized=True,
):
    """Calculate overlap between two frequency series in JAX."""
    from pycbc.types import TimeSeries
    from pycbc.filter import make_frequency_series

    if isinstance(vec1, TimeSeries):
        vec1 = make_frequency_series(vec1)
    if isinstance(vec2, TimeSeries):
        vec2 = make_frequency_series(vec2)

    v1 = to_jax(vec1)
    v2 = to_jax(vec2)
    n_pts = (len(v1) - 1) * 2
    flow = low_frequency_cutoff
    fhigh = high_frequency_cutoff

    kmin = int(flow / delta_f) if flow else 1
    if fhigh:
        kmax = min(int(fhigh / delta_f), int((n_pts + 1) / 2.0))
    else:
        kmax = int((n_pts + 1) / 2.0)

    v1_sub = v1[kmin:kmax]
    v2_sub = v2[kmin:kmax]
    prod = jnp.conj(v1_sub) * v2_sub

    if psd is not None:
        psd_arr = to_jax(psd)
        prod = prod / psd_arr[kmin:kmax]

    inn = jnp.sum(prod).real * (4.0 * delta_f)
    if not normalized:
        return inn

    norm1 = sigmasq_jax(
        v1,
        psd=psd,
        low_frequency_cutoff=flow,
        high_frequency_cutoff=fhigh,
        delta_f=delta_f,
    )
    norm2 = sigmasq_jax(
        v2,
        psd=psd,
        low_frequency_cutoff=flow,
        high_frequency_cutoff=fhigh,
        delta_f=delta_f,
    )
    return inn / jnp.sqrt(norm1 * norm2)


def match_jax(
    vec1,
    vec2,
    psd=None,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
    delta_f=1.0,
    v1_norm=None,
    v2_norm=None,
):
    """Calculate match (overlap maximized over time and phase) in JAX."""
    from pycbc.types import TimeSeries
    from pycbc.filter import make_frequency_series

    if isinstance(vec1, TimeSeries):
        vec1 = make_frequency_series(vec1)
    if isinstance(vec2, TimeSeries):
        vec2 = make_frequency_series(vec2)

    v1 = to_jax(vec1)
    v2 = to_jax(vec2)
    n_pts = (len(v1) - 1) * 2
    flow = low_frequency_cutoff
    fhigh = high_frequency_cutoff

    kmin = int(flow / delta_f) if flow else 1
    if fhigh:
        kmax = min(int(fhigh / delta_f), int((n_pts + 1) / 2.0))
    else:
        kmax = int((n_pts + 1) / 2.0)

    corr = jnp.zeros(n_pts, dtype=v1.dtype)
    prod = jnp.conj(v1[kmin:kmax]) * v2[kmin:kmax]
    if psd is not None:
        psd_arr = to_jax(psd)
        prod = prod / psd_arr[kmin:kmax]

    corr = corr.at[kmin:kmax].set(prod)
    time_series = jnp.fft.ifft(corr) * n_pts

    mag = jnp.abs(time_series)
    max_idx = int(jnp.argmax(mag))
    max_val = mag[max_idx] * (4.0 * delta_f)

    if v1_norm is None:
        v1_norm = sigmasq_jax(
            v1,
            psd=psd,
            low_frequency_cutoff=flow,
            high_frequency_cutoff=fhigh,
            delta_f=delta_f,
        )
    if v2_norm is None:
        v2_norm = sigmasq_jax(
            v2,
            psd=psd,
            low_frequency_cutoff=flow,
            high_frequency_cutoff=fhigh,
            delta_f=delta_f,
        )

    norm = jnp.sqrt(v1_norm * v2_norm)
    return max_val / norm, max_idx


@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def _batch_peak_core(tensor, template_count, segment_start, segment_stop):
    """Reduce a fixed-shape batch segment in one compiled JAX operation."""
    values = tensor.reshape(template_count, -1)[:, segment_start:segment_stop]
    if jnp.iscomplexobj(values):
        sq_mag = values.real ** 2 + values.imag ** 2
    else:
        sq_mag = values ** 2
    indices = jnp.argmax(sq_mag, axis=-1)
    peaks = values[jnp.arange(template_count), indices]
    return indices, peaks


def batch_peak_values(output, template_count, template_size, segment):
    """Materialize one peak index and value per template from contiguous output.

    Avoids per-template host synchronization by reducing the full 2D batch
    allocation in a single vectorized JAX kernel.
    """
    _ensure_x64()

    tensor = to_jax(output)
    template_count = int(template_count)
    template_size = int(template_size)
    if (
        template_count < 1
        or template_size < 1
        or tensor.size != template_count * template_size
    ):
        return None

    if segment.step not in (None, 1):
        return None

    segment_start, segment_stop, _ = segment.indices(template_size)
    if segment_start >= segment_stop:
        return None
    indices, peaks = _batch_peak_core(
        tensor, template_count, segment_start, segment_stop,
    )
    # Keep reductions on the selected JAX device.  The live filter consumes
    # these values directly and must not synchronize each template through a
    # NumPy conversion before veto processing.
    return indices, peaks


def batch_peak_magnitudes(peak_values):
    """Materialize batch peak magnitudes in JAX."""
    _ensure_x64()
    jarr = to_jax(peak_values)
    return jnp.abs(jarr)


def batch_matched_filter_bank(
    templates,
    strain,
    psd=None,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
    delta_f=None,
):
    """High-throughput batched template bank matched filtering in pure JAX.

    Performs 2D overwhitening, batched correlation, and batched IFFT across
    the entire template bank in an XLA-fused execution graph.

    Parameters
    ----------
    templates : jax.Array or FrequencySeries
        2D array of templates with shape (num_templates, n_freq) or list of
        FrequencySeries.
    strain : jax.Array or FrequencySeries
        1D strain frequency series of length n_freq.
    psd : jax.Array or FrequencySeries, optional
        1D PSD frequency series of length n_freq.
    low_frequency_cutoff : float, optional
        Low frequency cutoff for integration in Hz.
    high_frequency_cutoff : float, optional
        High frequency cutoff for integration in Hz.
    delta_f : float, optional
        Frequency spacing in Hz.

    Returns
    -------
    norm_snr : jax.Array
        2D array of normalized complex SNR time series of shape
        (num_templates, n_time).
    sigmasq : jax.Array
        1D array of template variances (sigmasq) of length num_templates.
    """
    _ensure_x64()
    import jax.numpy as jnp

    # Lists of FrequencySeries need row conversion before stacking.
    if isinstance(templates, (list, tuple)):
        templates_j = jnp.stack([to_jax(t) for t in templates], axis=0)
        template_df = getattr(templates[0], "delta_f", None)
    else:
        templates_j = to_jax(templates)
        template_df = getattr(templates, "delta_f", None)
    if templates_j.ndim == 1:
        templates_j = templates_j[None, :]
    df = delta_f if delta_f is not None else getattr(strain, "delta_f", None)
    if df is None:
        df = template_df if template_df is not None else 1.0

    num_templates = templates_j.shape[0]
    strain_j = to_jax(strain)
    n_freq = strain_j.shape[-1]
    n_time = (n_freq - 1) * 2

    # Match the CPU reference's half-open band, including its default
    # exclusion of DC and Nyquist and truncation of fractional cutoffs.
    kmin, kmax = get_cutoff_indices(
        low_frequency_cutoff, high_frequency_cutoff, df, n_time
    )

    # 2D overwhitening
    if psd is not None:
        psd_j = to_jax(psd)
        inv_psd = jnp.where(psd_j > 0, 1.0 / psd_j, 0.0)
    else:
        inv_psd = jnp.ones(n_freq, dtype=strain_j.dtype)

    # Apply frequency cutoffs to inv_psd
    freq_mask = (jnp.arange(n_freq) >= kmin) & (jnp.arange(n_freq) < kmax)
    inv_psd_masked = jnp.where(freq_mask, inv_psd, 0.0)

    # Template normalizations: sigmasq = 4 * df * sum(|h|^2 * inv_psd)
    t_mag_sq = templates_j.real ** 2 + templates_j.imag ** 2
    sigmasq = 4.0 * df * jnp.sum(t_mag_sq * inv_psd_masked[None, :], axis=-1)

    # Batched correlation:
    # qtilde has length n_time = (n_freq - 1) * 2
    # In PyCBC matched_filter_core:
    # correlate(htilde[kmin:kmax], stilde[kmin:kmax], qtilde[kmin:kmax])
    # which is qtilde[kmin:kmax] = conj(htilde[kmin:kmax]) * stilde[kmin:kmax]
    # then qtilde[kmin:kmax] /= psd[kmin:kmax]
    qtilde = jnp.zeros((num_templates, n_time), dtype=templates_j.dtype)
    corr = (
        jnp.conj(templates_j[:, kmin:kmax])
        * (strain_j[kmin:kmax] * inv_psd[kmin:kmax])[None, :]
    )
    qtilde = qtilde.at[:, kmin:kmax].set(corr)

    # Batched IFFT: n_time length
    snr_series = jnp.fft.ifft(qtilde, axis=-1) * n_time

    # Normalize by (4 * df) / sqrt(sigmasq)
    norm = (4.0 * df) / jnp.sqrt(jnp.maximum(sigmasq, 1e-30))[:, None]
    norm_snr = snr_series * norm

    return norm_snr, sigmasq


def _batched_filter_core(
    templates_2d,
    seg_slice,
    kmin,
    kmax,
    tlen,
    valid_start,
    valid_stop,
    full_templates=False,
):
    """Build the shared batched correlation and valid SNR series."""
    if full_templates:
        templates_2d = templates_2d[:, kmin:kmax]
    corr_slice = jnp.conj(templates_2d) * seg_slice[None, :]
    pad_left = kmin
    pad_right = tlen - kmax
    qtilde = jnp.pad(corr_slice, ((0, 0), (pad_left, pad_right)))
    snr_series = jnp.fft.ifft(qtilde, axis=-1) * tlen
    valid_snr = snr_series[:, valid_start:valid_stop]
    return snr_series, valid_snr, corr_slice


def _cluster_filtered_batch(valid_snr, thresh_sq, window):
    """Cluster inside the filter JIT without recomputing SNR magnitudes."""
    from pycbc.events.threshold_jax import _batched_cluster_from_magnitude

    mag_sq = valid_snr.real ** 2 + valid_snr.imag ** 2
    return _batched_cluster_from_magnitude(
        valid_snr, mag_sq, thresh_sq, window
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "kmin", "kmax", "tlen", "valid_start", "valid_stop", "window",
        "full_templates",
    ),
)
def _batched_filter_and_cluster(
    templates_2d,
    seg_slice,
    thresh_sq,
    kmin,
    kmax,
    tlen,
    valid_start,
    valid_stop,
    window,
    full_templates=False,
):
    """JIT correlation, IFFT, and clustering as one compiled boundary."""
    snr_series, valid_snr, corr_slice = _batched_filter_core(
        templates_2d, seg_slice, kmin, kmax, tlen, valid_start,
        valid_stop, full_templates=full_templates,
    )
    clustered = _cluster_filtered_batch(valid_snr, thresh_sq, window)
    return (snr_series, corr_slice, *clustered)


@functools.partial(
    jax.jit,
    static_argnames=(
        "kmin", "kmax", "tlen", "valid_start", "valid_stop", "window",
        "full_templates",
    ),
)
def _batched_filter_and_cluster_lean(
    templates_2d,
    seg_slice,
    thresh_sq,
    kmin,
    kmax,
    tlen,
    valid_start,
    valid_stop,
    window,
    full_templates=False,
):
    """Compiled filter and cluster path without retaining full SNR series."""
    _, valid_snr, corr_slice = _batched_filter_core(
        templates_2d, seg_slice, kmin, kmax, tlen, valid_start,
        valid_stop, full_templates=full_templates,
    )
    clustered = _cluster_filtered_batch(valid_snr, thresh_sq, window)
    return (corr_slice, *clustered)



def batched_matched_filter_and_cluster_jax(
    mf_control,
    segnum,
    templates,
    sigmasqs,
    window,
    epoch=None,
):
    """Batched matched filtering, thresholding, and clustering on JAX device.

    Parameters
    ----------
    mf_control : MatchedFilterControl
        The matched filter control object holding analysis configuration.
    segnum : int
        Index of the segment to filter against.
    templates : list of FrequencySeries
        Templates in the current batch.
    sigmasqs : sequence of float
        Normalization factors for each template in the batch.
    window : int
        Clustering window size in samples.
    epoch : optional
        GPS epoch for the returned TimeSeries.

    Returns
    -------
    list of tuples
        For each template, returns (snr, norm, corr, idx, snrv) matching
        MatchedFilterControl's contract.
    """
    _ensure_x64()
    from pycbc.types import Array, TimeSeries

    b = len(sigmasqs)
    if b == 0:
        return []

    seg = mf_control.segments[segnum]
    kmin, kmax = mf_control.kmin, mf_control.kmax
    tlen = mf_control.tlen
    delta_f = mf_control.delta_f
    delta_t = mf_control.delta_t
    valid_start = seg.analyze.start
    valid_stop = seg.analyze.stop
    threshold = float(mf_control.snr_threshold)

    from pycbc import scheme
    state = getattr(scheme.mgr, "state", None)
    target_dev = getattr(state, "jax_device", None)

    # Pre-stack all segments into a persistent 2D tensor in VRAM during setup / first call
    cached_seg_tensor = getattr(mf_control, "_jax_segments_tensor", None)
    if (
        cached_seg_tensor is None
        or (target_dev is not None
            and cached_seg_tensor.devices() != {target_dev})
    ):
        slices = []
        for s in mf_control.segments:
            s_jax = to_jax(s, device=target_dev)
            slices.append(s_jax[kmin:kmax])
        mf_control._jax_segments_tensor = jnp.stack(slices, axis=0)
        if target_dev is not None:
            mf_control._jax_segments_tensor = jax.device_put(
                mf_control._jax_segments_tensor, target_dev
            )

    seg_slice = mf_control._jax_segments_tensor[segnum]

    # Cache 2D templates on mf_control across segments to avoid re-stacking 5x per batch
    batch_tensor = getattr(templates, "_batch_tensor", None)
    if batch_tensor is not None:
        cache_key = (id(batch_tensor), kmin, kmax, True)
        full_templates = True
    else:
        cache_key = (tuple(id(t) for t in templates), kmin, kmax, False)
        full_templates = False

    if getattr(mf_control, "_cached_templates_key", None) == cache_key:
        templates_2d = mf_control._cached_templates_2d
    else:
        if batch_tensor is not None:
            # Keep the full batch tensor cached. Slicing inside the JIT avoids
            # retaining a second device allocation for the cropped templates.
            templates_2d = to_jax(batch_tensor, device=target_dev)
        elif hasattr(templates, "ndim") and templates.ndim == 2:
            templates_2d = to_jax(templates, device=target_dev)[:, kmin:kmax]
        else:
            templates_2d = jnp.stack(
                [to_jax(t, device=target_dev)[kmin:kmax] for t in templates], axis=0
            )
        if target_dev is not None:
            templates_2d = jax.device_put(templates_2d, target_dev)
        mf_control._cached_templates_key = cache_key
        mf_control._cached_templates_2d = templates_2d

    sigmasqs_np = np.asarray(sigmasqs, dtype=np.float32)
    norms_host = (4.0 * delta_f) / np.sqrt(np.maximum(sigmasqs_np, 1e-30))
    thresh_sq_np = (threshold / norms_host) ** 2
    if target_dev is not None:
        thresh_sq = jax.device_put(thresh_sq_np, target_dev)
    else:
        thresh_sq = jax.device_put(thresh_sq_np)

    # Keep correlation, IFFT, magnitude reduction, thresholding, and
    # clustering in one compiled boundary. This prevents a duplicate pass
    # over the full valid SNR matrix and avoids an intermediate launch.
    need_snr = getattr(mf_control, "need_snr_series", False)
    platform = getattr(target_dev, "platform", "cpu") if target_dev else "cpu"
    is_cuda = platform in ("cuda", "gpu")

    if not is_cuda and b > 1:
        corr_slices, max_idxs, masks, max_snrs = [], [], [], []
        snr_series_list = [] if need_snr else None
        for i in range(b):
            t_slice = templates_2d[i:i + 1]
            th_slice = thresh_sq[i:i + 1]
            if need_snr:
                (snr_s, c_s, m_idx, s_mask, m_snr) = _batched_filter_and_cluster(
                    t_slice,
                    seg_slice,
                    th_slice,
                    kmin,
                    kmax,
                    tlen,
                    valid_start,
                    valid_stop,
                    window,
                    full_templates=full_templates,
                )
                snr_series_list.append(snr_s[0])
            else:
                (c_s, m_idx, s_mask, m_snr) = (
                    _batched_filter_and_cluster_lean(
                        t_slice,
                        seg_slice,
                        th_slice,
                        kmin,
                        kmax,
                        tlen,
                        valid_start,
                        valid_stop,
                        window,
                        full_templates=full_templates,
                    )
                )
            corr_slices.append(c_s[0])
            max_idxs.append(m_idx[0])
            masks.append(s_mask[0])
            max_snrs.append(m_snr[0])
        corr_slice = jnp.stack(corr_slices, axis=0)
        batched_max_idx = jnp.stack(max_idxs, axis=0)
        batched_survivor_mask = jnp.stack(masks, axis=0)
        batched_max_snr = jnp.stack(max_snrs, axis=0)
        snr_series = (
            jnp.stack(snr_series_list, axis=0) if need_snr else None
        )
    else:
        if need_snr:
            (snr_series, corr_slice, batched_max_idx,
             batched_survivor_mask, batched_max_snr) = _batched_filter_and_cluster(
                templates_2d,
                seg_slice,
                thresh_sq,
                kmin,
                kmax,
                tlen,
                valid_start,
                valid_stop,
                window,
                full_templates=full_templates,
            )
        else:
            (corr_slice, batched_max_idx,
             batched_survivor_mask, batched_max_snr) = (
                _batched_filter_and_cluster_lean(
                    templates_2d,
                    seg_slice,
                    thresh_sq,
                    kmin,
                    kmax,
                    tlen,
                    valid_start,
                    valid_stop,
                    window,
                    full_templates=full_templates,
                )
            )
            snr_series = None

    empty_idx = np.empty(0, dtype=np.uint32)
    empty_snrv = np.empty(0, dtype=np.complex64)

    # One bounded device-to-host transfer replaces per-template boolean
    # synchronizations.  These arrays contain one candidate per clustering
    # window, not the full SNR series.
    host_max_idx, host_survivor_mask, host_max_snr = jax.device_get(
        (batched_max_idx, batched_survivor_mask, batched_max_snr)
    )

    results = []
    if not np.any(host_survivor_mask):
        for i in range(b):
            results.append(([], float(norms_host[i]), [], empty_idx, empty_snrv))
        return results

    from pycbc.waveform.bank import LazyFrequencySeries

    for i in range(b):
        norm_i = float(norms_host[i])
        mask = host_survivor_mask[i]
        if not np.any(mask):
            results.append(([], norm_i, [], empty_idx, empty_snrv))
            continue

        survivor_indices = np.asarray(
            host_max_idx[i][mask], dtype=np.uint32
        )
        survivor_values = np.asarray(host_max_snr[i][mask])

        corr = LazyFrequencySeries(corr_slice, i, delta_f)
        corr._kmin = kmin
        corr._tlen = tlen

        if need_snr and snr_series is not None:
            snr = TimeSeries(
                Array(JAXArrayData(snr_series[i]), copy=False),
                epoch=epoch,
                delta_t=delta_t,
                copy=False,
            )
        else:
            snr = None

        results.append((snr, norm_i, corr, survivor_indices, survivor_values))

    del batched_max_idx, batched_survivor_mask, batched_max_snr
    if snr_series is not None:
        del snr_series
    if corr_slice is not None:
        del corr_slice

    return results


@functools.partial(jax.jit, static_argnames=("delta_f",))
def _batched_sigmasq_core(tmpls_stack, psd_j, delta_f):
    """JIT-compiled GPU reduction for template batch sigmasq values."""
    mag_sq = tmpls_stack.real ** 2 + tmpls_stack.imag ** 2
    return jnp.sum(mag_sq / psd_j[None, :], axis=-1) * (4.0 * delta_f)


def batch_sigmasq_jax(templates, psd):
    """Compute sigmasq for a batch of FrequencySeries templates against psd on GPU."""
    cached = getattr(templates, "_cached_sigmasqs", None)
    if cached is not None:
        cached_psd_id = getattr(templates, "_cached_sigmasqs_psd_id", None)
        if cached_psd_id is None or cached_psd_id == id(psd):
            return cached

    from pycbc import scheme

    _ensure_x64()
    b = len(templates)
    if b == 0:
        return np.zeros(0, dtype=np.float32)
    df = float(templates[0].delta_f)
    flow = float(templates[0].f_lower) if hasattr(templates[0], "f_lower") else 30.0
    kmin = int(flow / df)

    state = getattr(scheme.mgr, "state", None)
    target_dev = getattr(state, "jax_device", None)

    batch_tensor = getattr(templates, "_batch_tensor", None)
    if batch_tensor is not None:
        tmpls_stack = batch_tensor[:, kmin:]
    else:
        tmpls_stack = jnp.stack([to_jax(t, device=target_dev)[kmin:] for t in templates], axis=0)
    psd_j = to_jax(psd, device=target_dev)[kmin:]
    ssq_gpu = _batched_sigmasq_core(tmpls_stack, psd_j, df)
    ssq_np = np.asarray(jax.device_get(ssq_gpu), dtype=np.float32)
    try:
        templates._cached_sigmasqs = ssq_np
        templates._cached_sigmasqs_psd_id = id(psd)
    except Exception:
        pass
    return ssq_np


def process_batch_inspiral_jax(
    event_mgr,
    bank,
    batch_tnums,
    segments,
    matched_filter,
    power_chisq,
    cluster_window,
    next_batch_tnums=None,
    bank_chisq=None,
    autochisq=None,
    sg_chisq=None,
    inj_filter_rejector=None,
    opt=None,
):
    """Execute batched filtering, vetoes, and direct event manager insertion for JAX."""
    import logging
    from pycbc.events.eventmgr import findchirp_cluster_over_window_cython
    from pycbc.vetoes.chisq_jax import (
        batch_power_chisq_jax,
        cache_batch_power_chisq_bins_jax,
    )

    b = len(batch_tnums)
    if b == 0:
        return

    if hasattr(bank, "get_batch"):
        batch_templates = bank.get_batch(batch_tnums)
        if next_batch_tnums is not None and hasattr(bank, "prefetch_batch_jax"):
            bank.prefetch_batch_jax(
                next_batch_tnums,
                power_chisq=power_chisq,
                psd=segments[0].psd,
            )
    else:
        batch_templates = [bank[i] for i in batch_tnums]

    # Pre-cache chi-square bins for the batch
    if power_chisq is not None and getattr(power_chisq, "do", False):
        cache_batch_power_chisq_bins_jax(
            power_chisq, batch_templates, segments[0].psd
        )

    sigmasq_caches = {}
    triggered_events = [[] for _ in range(b)]
    flow = getattr(opt, "low_frequency_cutoff", None)

    for s_num, stilde in enumerate(segments):
        active_indices = []
        active_templates = []
        active_sigmasqs = []
        psd_id = id(stilde.psd)

        if psd_id not in sigmasq_caches:
            sigmasq_caches[psd_id] = batch_sigmasq_jax(batch_templates, stilde.psd)
        batch_sigmasqs = sigmasq_caches[psd_id]

        if inj_filter_rejector is None:
            active_indices = list(range(b))
            active_templates = batch_templates
            active_sigmasqs = batch_sigmasqs
        else:
            for i, t_num in enumerate(batch_tnums):
                if not inj_filter_rejector.template_segment_checker(
                    bank, t_num, stilde
                ):
                    continue
                active_indices.append(i)
                active_templates.append(batch_templates[i])
                active_sigmasqs.append(batch_sigmasqs[i])

            if not active_indices:
                continue

            batch_tensor = getattr(batch_templates, "_batch_tensor", None)
            if batch_tensor is not None:
                from pycbc.waveform.bank import TemplateBatchList
                act_list = TemplateBatchList(active_templates)
                if len(active_indices) == len(batch_templates):
                    act_list._batch_tensor = batch_tensor
                else:
                    act_list._batch_tensor = batch_tensor[jnp.asarray(active_indices)]
                active_templates = act_list

        if opt and getattr(opt, "update_progress", None):
            from pycbc.workflow import update_progress
            update_progress(
                (batch_tnums[0] + (s_num / float(len(segments)))) / len(bank),
                opt.update_progress,
                getattr(opt, "update_progress_file", None),
            )

        logging.info(
            "Filtering template batch %d-%d/%d segment %d/%d"
            % (batch_tnums[0] + 1, batch_tnums[-1] + 1, len(bank), s_num + 1, len(segments))
        )

        batch_results = matched_filter.batched_matched_filter_and_cluster(
            s_num,
            active_templates,
            active_sigmasqs,
            cluster_window,
            epoch=stilde._epoch,
        )

        # Check if any template in this segment produced triggers
        has_triggers = any(len(res[3]) > 0 for res in batch_results)
        if not has_triggers:
            continue

        chisq_map = None
        if power_chisq is not None and getattr(power_chisq, "do", False):
            corr_tensor = None
            for res in batch_results:
                if res[2] is not None and hasattr(res[2], "_batch_tensor"):
                    corr_tensor = res[2]._batch_tensor
                    break
            if corr_tensor is not None:
                chisq_map = batch_power_chisq_jax(
                    corr_tensor,
                    batch_results,
                    active_templates,
                    stilde.psd,
                    stilde.analyze.start,
                    snr_threshold=power_chisq.snr_threshold,
                    power_chisq=power_chisq,
                )

        for act_pos, tmpl_idx in enumerate(active_indices):
            snr, norm, corr, idx, snrv = batch_results[act_pos]
            if not len(idx):
                continue

            tmpl = active_templates[act_pos]
            sigmasq = active_sigmasqs[act_pos]
            idx_cum = idx + stilde.cumulative_index
            snr_vals = snrv * norm

            if chisq_map is not None and act_pos in chisq_map:
                chisq_val, chisq_dof_val = chisq_map[act_pos]
            elif power_chisq is not None and getattr(power_chisq, "do", False):
                chisq_val, chisq_dof_val = power_chisq.values(
                    corr, snrv, norm, stilde.psd, idx + stilde.analyze.start, tmpl
                )
            else:
                chisq_val, chisq_dof_val = None, None

            bank_chisq_val, bank_chisq_dof_val = (
                bank_chisq.values(tmpl, stilde.psd, stilde, snrv, norm, idx + stilde.analyze.start)
                if bank_chisq is not None and getattr(bank_chisq, "do", False)
                else (None, None)
            )
            sg_chisq_val = (
                sg_chisq.values(stilde, tmpl, stilde.psd, snrv, norm, chisq_val, chisq_dof_val, idx + stilde.analyze.start)
                if sg_chisq is not None and getattr(sg_chisq, "do", False)
                else None
            )
            cont_chisq_val, cont_chisq_dof_val = (
                autochisq.values(snr, idx + stilde.analyze.start, tmpl, stilde.psd, norm, stilde=stilde, low_frequency_cutoff=flow)
                if autochisq is not None and getattr(autochisq, "do", False)
                else (None, None)
            )

            triggered_events[tmpl_idx].append({
                "time_index": idx_cum,
                "snr": snr_vals,
                "chisq": chisq_val,
                "chisq_dof": chisq_dof_val,
                "sigmasq": sigmasq,
                "bank_chisq": bank_chisq_val,
                "bank_chisq_dof": bank_chisq_dof_val,
                "sg_chisq": sg_chisq_val,
                "cont_chisq": cont_chisq_val,
                "cont_chisq_dof": cont_chisq_dof_val,
            })

    # Add accumulated events directly to event_mgr per template
    for tmpl_idx, ev_list in enumerate(triggered_events):
        if not ev_list:
            continue
        tmpl = batch_templates[tmpl_idx]
        if len(ev_list) == 1:
            ev = ev_list[0]
            times = ev["time_index"]
            snrs = ev["snr"]
            chisqs = ev["chisq"]
            dofs = ev["chisq_dof"]
            sigmasq_val = ev["sigmasq"]
            sigmasqs_arr = np.full(len(times), sigmasq_val, dtype=np.float32)
            b_chisq = ev["bank_chisq"]
            b_dof = ev["bank_chisq_dof"]
            sg = ev["sg_chisq"]
            cont = ev["cont_chisq"]
            cont_dof = ev["cont_chisq_dof"]
        else:
            times = np.concatenate([e["time_index"] for e in ev_list])
            snrs = np.concatenate([e["snr"] for e in ev_list])
            chisqs = np.concatenate([e["chisq"] for e in ev_list]) if ev_list[0]["chisq"] is not None else None
            dofs = np.concatenate([e["chisq_dof"] for e in ev_list]) if ev_list[0]["chisq_dof"] is not None else None
            sigmasqs_arr = np.concatenate([np.full(len(e["time_index"]), e["sigmasq"], dtype=np.float32) for e in ev_list])
            b_chisq = np.concatenate([e["bank_chisq"] for e in ev_list]) if ev_list[0]["bank_chisq"] is not None else None
            b_dof = np.concatenate([e["bank_chisq_dof"] for e in ev_list]) if ev_list[0]["bank_chisq_dof"] is not None else None
            sg = np.concatenate([e["sg_chisq"] for e in ev_list]) if ev_list[0]["sg_chisq"] is not None else None
            cont = np.concatenate([e["cont_chisq"] for e in ev_list]) if ev_list[0]["cont_chisq"] is not None else None
            cont_dof = np.concatenate([e["cont_chisq_dof"] for e in ev_list]) if ev_list[0]["cont_chisq_dof"] is not None else None

        if cluster_window > 0 and len(times) > 1:
            times_i32 = times.astype(np.int32)
            indices = np.zeros(len(times), dtype=np.int32)
            count = findchirp_cluster_over_window_cython(
                times_i32, np.asarray(abs(snrs)), cluster_window, indices, len(times)
            )
            indices = indices[:count + 1]
            times = times[indices]
            snrs = snrs[indices]
            if chisqs is not None:
                chisqs = chisqs[indices]
            if dofs is not None:
                dofs = dofs[indices]
            sigmasqs_arr = sigmasqs_arr[indices]
            if b_chisq is not None:
                b_chisq = b_chisq[indices]
            if b_dof is not None:
                b_dof = b_dof[indices]
            if sg is not None:
                sg = sg[indices]
            if cont is not None:
                cont = cont[indices]
            if cont_dof is not None:
                cont_dof = cont_dof[indices]

        event_mgr.add_template_events_direct(
            tmpl.params,
            times,
            snrs,
            chisq=chisqs,
            chisq_dof=dofs,
            sigmasq=sigmasqs_arr,
            bank_chisq=b_chisq,
            bank_chisq_dof=b_dof,
            cont_chisq=cont,
            cont_chisq_dof=cont_dof,
            sg_chisq=sg,
        )

    if hasattr(bank, "clear_batch_cache"):
        bank.clear_batch_cache(batch_tnums, collect=False)
    if hasattr(matched_filter, "clear_batch_cache"):
        matched_filter.clear_batch_cache()

