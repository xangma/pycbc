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

Functional helpers support JIT with the default JAX kernels. Optional CPU
validation routes perform host work and run outside JAX transformations.
"""

import functools
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp

from pycbc.events import ranking
from pycbc.filter.matchedfilter import _BaseCorrelator, get_cutoff_indices
from pycbc.types.array_jax import (
    JAXArrayData,
    _as_jax_array,
    _ensure_x64,
    _cpu_reference,
    _divide,
    _fast_inner,
    _fast_inner_self,
    _fast_weighted_inner,
    _fast_weighted_inner_self,
    _reference_enabled,
    to_jax,
)


_EMPTY_F32 = np.zeros(0, dtype=np.float32)
_EMPTY_U32 = np.zeros(0, dtype=np.uint32)


def _functional_device(*inputs):
    """Use the active scheme, an existing input device, or JAX's default."""
    from pycbc import scheme

    if isinstance(scheme.mgr.state, scheme.JAXScheme):
        return scheme.mgr.state.jax_device
    arrays = [_as_jax_array(value) for value in inputs]
    # Traced computations inherit their execution device from JAX rather than
    # committing intermediate arrays to the device available during tracing.
    if any(isinstance(value, jax.core.Tracer) for value in arrays):
        return None
    for value in arrays:
        if value is not None:
            return value.device
    return jax.config.jax_default_device or jax.devices()[0]


def _functional_array(value, device):
    """Place eager inputs together while leaving JAX tracers untouched."""
    array = _as_jax_array(value)
    return (array if isinstance(array, jax.core.Tracer)
            else to_jax(value, device=device))


def _live_array_metadata(value):
    """Read fixed geometry without materializing a lazy device row/view."""
    if isinstance(value, JAXArrayData):
        return value.shape[0], value.dtype
    stored = getattr(value, "__dict__", {})
    data = stored.get("_data_inst", stored.get("_data", stored.get("data")))
    if data is not None:
        return _live_array_metadata(data)
    tensor = stored.get("_batch_tensor")
    if tensor is not None:
        return tensor.shape[1], tensor.dtype
    return len(value), getattr(value, "dtype", None)


def _live_veto_executable_cache(control, power_chisq=None):
    """Share the existing retained cache across preparation and veto work."""
    if power_chisq is None:
        power_chisq = getattr(control, "power_chisq", None)
    cache = getattr(power_chisq, "_jax_live_veto_executables", None)
    if cache is None:
        cache = getattr(control, "_jax_live_veto_executables", None)
    if cache is None:
        cache = {}
    control._jax_live_veto_executables = cache
    if power_chisq is not None:
        power_chisq._jax_live_veto_executables = cache
    return cache


@functools.partial(jax.jit, static_argnames=("count",))
def _live_compact_columns(*columns, count):
    """Publish exact-length columns at the sparse trigger API boundary.

    This final slice still specializes by count; the preceding selection and
    order reductions use stable buckets and retained executable handles.
    """
    return tuple(column[:count] for column in columns)


def _live_compact_results(columns, count, cache, device):
    from pycbc.vetoes.chisq_jax import _live_chisq_executable

    if count == columns[0].shape[0]:
        return columns
    execute = _live_chisq_executable(
        _live_compact_columns, columns, dict(count=count), cache, device)
    return execute(*columns)


def batch_template_matrix_jax(templates, size):
    """Pack immutable live templates on the active JAX device."""
    if templates:
        first = templates[0]
        batch_tensor = getattr(first, "_batch_tensor", None)
        if batch_tensor is not None and all(
            getattr(template, "_batch_tensor", None) is batch_tensor
            for template in templates
        ):
            positions = [getattr(t, "_batch_pos", None) for t in templates]
            if None not in positions:
                start = positions[0]
                shape = getattr(batch_tensor, "shape", ())
                if positions == list(range(start, start + len(positions))):
                    if len(shape) == 2 and start >= 0:
                        stop = start + len(positions)
                        if stop <= shape[0] and size <= shape[1]:
                            return to_jax(batch_tensor)[start:stop, :size]
                if (
                    len(shape) == 2
                    and all(0 <= p < shape[0] for p in positions)
                    and size <= shape[1]
                ):
                    pos_idx = jnp.asarray(positions, dtype=jnp.int32)
                    return to_jax(batch_tensor)[pos_idx, :size]
    return jnp.stack(
        [to_jax(template)[:size] for template in templates], axis=0
    )


@jax.jit
def batch_template_power_jax(template_matrix, delta_f):
    """Cache the PSD-independent part of generic live normalization."""
    return (
        template_matrix.real ** 2 + template_matrix.imag ** 2
    ) * (4.0 * delta_f)


def batch_correlator_init_jax(control, immutable_templates=False):
    """Retain JAX rows without materializing native pointer tables."""
    control.x = control.z = None
    control._jax_template_matrix = None
    control._jax_template_power = None
    if immutable_templates:
        control._jax_template_matrix = batch_template_matrix_jax(
            control.xs, control.size)
        if (hasattr(control.xs[0], "delta_f")
                and not _reference_enabled("squared_norm")):
            control._jax_template_power = batch_template_power_jax(
                control._jax_template_matrix, float(control.xs[0].delta_f))


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
    templates, psd, template_matrix=None, template_power=None
):
    """Preserve ``sigma_cached`` frequency support and accumulation precision.

    The native generic cache forms a float32 squared-magnitude view, multiplies
    it by ``4 * delta_f``, weights it by the inverse PSD, and accumulates
    in float64. Selected squared-norm, division and inner-product controls
    independently restore the original CPU implementations.
    """
    _ensure_x64()
    if not templates:
        return jnp.zeros(0, dtype=jnp.float64)
    device = _functional_device(
        *templates, psd, template_matrix, template_power
    )
    delta_f = float(templates[0].delta_f)
    psd_j = getattr(psd, "_jax_psd", None)
    if psd_j is None:
        psd_j = _functional_array(psd, device)
        try:
            psd._jax_psd = psd_j
        except AttributeError:
            pass
    else:
        psd_j = _functional_array(psd_j, device)
    rows = [None] * len(templates)
    generic_indices = []
    generic_kmins = []
    generic_kmaxs = []
    for index, template in enumerate(templates):
        flow = getattr(template, "min_f_lower", None) or getattr(
            template, "f_lower", 0.0
        )
        fhigh = getattr(template, "end_frequency", None)
        template_size = (
            template_matrix.shape[1]
            if template_matrix is not None
            else _live_array_metadata(template)[0]
        )
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
                _functional_array(vector, device)[template.end_idx - 1]
                - _functional_array(vector, device)[
                    int(float(template.f_lower) / delta_f)])
            continue

        generic_indices.append(index)
        generic_kmins.append(kmin)
        generic_kmaxs.append(kmax)

    if generic_indices:
        native_power = _reference_enabled("squared_norm")
        if template_power is not None and not native_power:
            generic_power = _functional_array(template_power, device)[
                jnp.asarray(generic_indices, dtype=jnp.int32)
            ]
        else:
            if template_matrix is None:
                generic_matrix = jnp.stack(
                    [_functional_array(templates[index], device)
                     for index in generic_indices]
                )
            else:
                generic_matrix = _functional_array(template_matrix, device)[
                    jnp.asarray(generic_indices, dtype=jnp.int32)
                ]
            if native_power:
                generic_power = jnp.stack([
                    _functional_array(_cpu_reference(row, "squared_norm"),
                                      device) * 4.0 * delta_f
                    for row in generic_matrix])
            else:
                generic_power = batch_template_power_jax(
                    generic_matrix, delta_f)
        if _reference_enabled("inner") or _reference_enabled("divide"):
            from pycbc.psd.estimate_jax import _reciprocal_numpy_compat
            inverse = (_divide(1.0, psd_j) if _reference_enabled("divide")
                       else _reciprocal_numpy_compat(psd_j))
            generic_norms = jnp.stack([
                (_functional_array(_cpu_reference(
                    power[kmin:kmax], "inner", inverse[kmin:kmax]), device)
                 if _reference_enabled("inner") else jnp.sum(
                     power[kmin:kmax] * inverse[kmin:kmax], dtype=jnp.float64))
                for power, kmin, kmax in zip(
                    generic_power, generic_kmins, generic_kmaxs)])
        else:
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


class _LiveVetoCandidate:
    """Original veto sequence with lazy scalar views of shared vectors.

    Metadata survives ``process_all`` concatenation and trigger sorting. Only
    the scalar fallback or SG veto needs to resolve individual SNR/norm rows.
    """

    def __init__(self, snrs, norms, row, point, template, stilde):
        self.snrs, self.norms, self.row = snrs, norms, row
        self.metadata = (point, template, stilde)

    def __len__(self):
        return 5

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        index = range(len(self))[index]
        if index == 0:
            return self.snrs[self.row:self.row + 1]
        if index == 1:
            return self.norms[self.row]
        return self.metadata[index - 2]


def process_live_vetoes_jax(control, results, veto_info):
    """Run live veto reductions and retain numeric result columns on JAX."""
    power_chisq = control.power_chisq

    chisq_values = []
    dof_values = []
    sg_values = []
    veto_corr = {}
    for info in veto_info:
        snrv, norm, l, htilde, stilde = info[:5]
        if _reference_enabled("live_selection"):
            from pycbc import waveform

            # The original generic sigma_cached returns a Python float;
            # precomputed waveform norms return a NumPy scalar. Preserve
            # that promotion boundary for both signal-consistency vetoes.
            host_norm = np.asarray(jax.device_get(norm))
            norm = (host_norm[()] if waveform.waveform_norm_exists(
                getattr(htilde, "approximant", "")) else host_norm.item())
        size = _live_array_metadata(htilde.cout)[0]
        if size not in veto_corr:
            veto_corr[size] = live_veto_buffer_jax(size, htilde.dtype)
        corr = veto_corr[size]
        correlate(htilde, stilde, corr)
        c, d = power_chisq.values(corr, snrv, norm,
                                  stilde.psd, [l], htilde)
        c = jnp.asarray(c)
        d = jnp.asarray(d)
        chisq_values.append(
            jnp.asarray(_divide(c[0], d[0]), dtype=jnp.float32)
        )
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
        chisq = _EMPTY_F32
        dof = _EMPTY_U32
        sg_chisq = _EMPTY_F32
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
    return results


@functools.partial(jax.jit, static_argnames=("groups",))
def _live_concat_resident_columns(*values, groups):
    """Join canonicalized groups, retaining JAX's per-column promotion."""
    sizes = ((groups,) * len(range(0, len(values), groups))
             if isinstance(groups, (int, np.integer)) else groups)
    columns, start = [], 0
    for size in sizes:
        columns.append(jnp.concatenate(values[start:start + size]))
        start += size
    return tuple(columns)


def _live_prune_empty_column(values):
    """Remove empty-position specialization without changing promotion.

    Empty arrays still contribute their dtype to JAX concatenation. Retain one
    representative per dtype absent from the nonempty inputs, in a stable
    order, while preserving the original order of every nonempty input.
    """
    nonempty = [value for value in values if value.shape[0]]
    represented = {np.dtype(value.dtype).str for value in nonempty}
    empty = {}
    for value in values:
        dtype = np.dtype(value.dtype).str
        if not value.shape[0] and dtype not in represented:
            empty.setdefault(dtype, value)
    return nonempty + [empty[dtype] for dtype in sorted(empty)]


@functools.lru_cache(maxsize=128)
def _live_concat_resident_executable(shapes, groups, device, x64):
    """Retain handles by geometry, device and precision, without results."""
    from pycbc.vetoes.chisq_jax import _live_chisq_executable

    abstracts = tuple(
        jax.ShapeDtypeStruct(shape, np.dtype(dtype)) for shape, dtype in shapes
    )
    return _live_chisq_executable(
        _live_concat_resident_columns,
        abstracts,
        dict(groups=groups),
        {},
        device,
    )


def combine_live_results_jax(results):
    """Upload uniform host columns once and join CUDA columns together."""
    combined = {}
    resident = {}
    device = None
    for key in results[0]:
        values = [batch[key] for batch in results]
        kinds = []
        for value in values:
            dtype = getattr(value, "dtype", None)
            if dtype is None:
                dtype = np.asarray(value).dtype
            kinds.append(np.dtype(dtype).kind)
        if all(kind in "biufc" for kind in kinds):
            if all(type(value) is np.ndarray and value.dtype.isnative
                   and value.ndim > 0
                   and value.shape[1:] == values[0].shape[1:]
                   for value in values):
                dtypes = [jax.dtypes.canonicalize_dtype(value.dtype)
                          for value in values]
                if all(dtype == dtypes[0] for dtype in dtypes):
                    # Canonicalize each group before combining, as the old
                    # per-group uploads did. Keep heterogeneous promotion in
                    # JAX: NumPy widening can preserve subnormals XLA flushes.
                    arrays = [np.asarray(value, dtype=dtypes[0])
                              for value in values]
                    combined[key] = jnp.asarray(np.concatenate(arrays))
                    continue
            arrays = [jnp.asarray(value) for value in values]
            # Keep conversions outside fusion: host canonicalization and JAX
            # promotion differ from NumPy for wide integers and subnormals.
            eligible = (
                jax.config.jax_numpy_dtype_promotion == "standard"
                and all(
                    (type(value) is np.ndarray and value.dtype.isnative)
                    or (
                        isinstance(value, jax.Array)
                        and not isinstance(value, jax.core.Tracer)
                        and not value.weak_type
                        and value.dtype
                        == jax.dtypes.canonicalize_dtype(value.dtype)
                    )
                    for value in values
                )
            )
            eligible = eligible and all(
                isinstance(array, jax.Array)
                and not isinstance(array, jax.core.Tracer)
                and not array.weak_type and array.ndim == 1
                and len(array.devices()) == 1 for array in arrays)
            if eligible:
                column_device = next(iter(arrays[0].devices()))
                sharding = jax.sharding.SingleDeviceSharding(column_device)
                eligible = column_device.platform == "gpu" and all(
                    array.sharding == sharding for array in arrays)
                if eligible:
                    # A retained executable requires its default vector
                    # layout and device memory, not merely the same device.
                    layouts = [getattr(getattr(array, "format", None),
                                       "layout", None) for array in arrays]
                    eligible = all(
                        layout is not None and layout.major_to_minor == (0,)
                        and not layout.tiling
                        and layout.sub_byte_element_size_in_bits == 0
                        for layout in layouts)
                if eligible and device is None:
                    device = column_device
                eligible = eligible and column_device == device
            if eligible:
                combined[key] = None
                resident[key] = arrays
            else:
                combined[key] = jnp.concatenate(arrays)
        else:
            combined[key] = np.concatenate(values)
    if len(resident) > 1:
        columns = [_live_prune_empty_column(column)
                   for column in resident.values()]
        values = tuple(value for column in columns for value in column)
        groups = tuple(len(column) for column in columns)
        shapes = tuple((tuple(value.shape), np.dtype(value.dtype).str)
                       for value in values)
        # Closed-over concrete arrays may also be assembled under an outer
        # trace. Evaluate their retained executable without caching tracers.
        with jax.ensure_compile_time_eval():
            executable = _live_concat_resident_executable(
                shapes, groups, device, bool(jax.config.jax_enable_x64))
            outputs = executable(*values)
        for key, output in zip(resident, outputs):
            combined[key] = output
    else:
        for key, values in resident.items():
            combined[key] = jnp.concatenate(values)
    return combined


@jax.jit
def _live_select_peaks(peaks, norms, threshold, abort_threshold):
    """Scale and select a live batch without per-template synchronization."""
    scaled = (peaks * norms).astype(peaks.dtype)
    magnitudes = jnp.abs(peaks) * norms
    # Preserve the scalar path's NaN behavior: ``not (s < threshold)``.
    accepted = jnp.logical_not(magnitudes < threshold)
    abort = accepted & (magnitudes > abort_threshold)
    return scaled, accepted, abort


@jax.jit
def _live_candidate_columns(scaled_peaks, sigmasq, selected):
    """Assemble numeric columns and veto vectors in one device dispatch."""
    scaled = scaled_peaks[selected]
    return (jnp.abs(scaled), jnp.angle(scaled),
            sigmasq[selected].astype(jnp.float32))


@functools.lru_cache(maxsize=32)
def _live_empty_candidate_columns(device, dtypes, x64):
    """Retain immutable empty outputs by placement and dtype mode."""
    return tuple(jax.device_put(np.empty(0, dtype=np.dtype(dtype)), device)
                 for dtype in dtypes)


def _live_empty_candidate_device(arrays):
    """Qualify ordinary Live vectors without reading their scientific data."""
    if not all(isinstance(array, jax.Array)
               and not isinstance(array, jax.core.Tracer)
               and not array.weak_type and array.ndim == 1
               and array.dtype == jax.dtypes.canonicalize_dtype(array.dtype)
               for array in arrays):
        return None
    if arrays[0].shape[0] == 0 or any(
            array.shape != arrays[0].shape for array in arrays[1:]):
        return None
    if any(array.dtype not in (np.complex64, np.complex128)
           for array in arrays[:2]) or any(
            array.dtype not in (np.float32, np.float64)
            for array in arrays[2:]):
        return None
    if any(len(array.devices()) != 1 for array in arrays):
        return None
    device = next(iter(arrays[1].devices()))
    sharding = jax.sharding.SingleDeviceSharding(device)
    if any(array.sharding != sharding for array in arrays):
        return None
    layouts = [getattr(getattr(array, "format", None), "layout", None)
               for array in arrays]
    if not all(layout is not None and layout.major_to_minor == (0,)
               and not layout.tiling
               and layout.sub_byte_element_size_in_bits == 0
               for layout in layouts):
        return None
    return device


def _live_candidate_columns_bucketed(
        control, scaled_peaks, peaks, norms, sigmasq, selected):
    """Retain selection kernels across sparse CUDA counts in one bucket."""
    count = len(selected)
    if count == 0:
        device = _live_empty_candidate_device(
            (scaled_peaks, peaks, norms, sigmasq))
        if device is not None:
            real = np.dtype(np.float32 if scaled_peaks.dtype == np.complex64
                            else np.float64).str
            dtypes = (real, real, np.dtype(np.float32).str)
            # A concrete closed-over vector can enter under an outer trace.
            # Never retain a tracer in the empty-array cache.
            with jax.ensure_compile_time_eval():
                return _live_empty_candidate_columns(
                    device, dtypes, bool(jax.config.jax_enable_x64))

    from pycbc.vetoes.chisq_jax import (
        _chisq_candidate_bucket, _live_chisq_executable,
    )

    indices = np.zeros(_chisq_candidate_bucket(count), dtype=np.int32)
    indices[:count] = selected
    cache = _live_veto_executable_cache(control)
    args = (scaled_peaks, sigmasq, indices)
    execute = _live_chisq_executable(
        _live_candidate_columns, args, {}, cache, peaks.device)
    columns = execute(*args)
    return _live_compact_results(columns, count, cache, peaks.device)


def live_process_batch_jax(self):
    """Process only a single batch group of data"""
    from pycbc.filter.matchedfilter import logger
    if self.block_id == len(self.tgroups):
        return None, None

    tgroup = self.tgroups[self.block_id]
    psize = self.chunk_tsamples[self.block_id]
    mid = self.mids[self.block_id]
    stilde = self.data.overwhitened_data(tgroup[0].delta_f)
    psd = stilde.psd
    template_matrix = getattr(
        self.corr[self.block_id], "_jax_template_matrix", None
    )
    template_power = getattr(
        self.corr[self.block_id], "_jax_template_power", None
    )
    native_norms = live_template_norms_jax(
        tgroup, psd, template_matrix=template_matrix,
        template_power=template_power,
    )

    valid_end = int(psize - self.data.trim_padding)
    valid_start = int(valid_end - self.data.blocksize * self.data.sample_rate)

    seg = slice(valid_start, valid_end)

    self.corr[self.block_id].execute(stilde)
    # The JAX plan stays on-device by default and honors an explicit native
    # IFFT validation route through the same plan API.
    self.ifts[mid].execute()

    self.block_id += 1

    result = {}
    tkeys = tgroup[0].params.dtype.names
    for key in tkeys:
        result[key] = []

    veto_info = []

    # Reduce, normalize, and select the full group before the only host
    # synchronization.  The host loop below handles sparse metadata and veto
    # objects, not accelerator decisions.
    jax_peaks = batch_peak_values(self.out_mem[mid], len(tgroup), psize, seg)
    if jax_peaks is None:
        raise ValueError("JAX live peak reduction could not process the batch")
    peak_indices, peak_values = jax_peaks
    norms = (4.0 * tgroup[0].delta_f) / jnp.sqrt(native_norms)
    if _reference_enabled("live_selection"):
        from pycbc import waveform
        from pycbc.reference_jax import cpu_reference

        native_result, accepted_indices, accepted_norms = cpu_reference(
            "live_selection", peak_values, spacing=tgroup[0].delta_f,
            sigmasqs=np.asarray(native_norms),
            python_sigmasq=np.asarray([
                not waveform.waveform_norm_exists(
                    getattr(template, "approximant", ""))
                for template in tgroup]),
            snr_threshold=self.snr_threshold,
            snr_abort_threshold=self.snr_abort_threshold)
        if native_result is False:
            return False, []
        host_peak_indices = np.asarray(peak_indices)
        norms = norms.at[accepted_indices].set(to_jax(accepted_norms))
        snr, phase, sigmasq = (to_jax(native_result[key]) for key in (
            "snr", "coa_phase", "sigmasq"))
    else:
        abort_threshold = (
            jnp.inf if self.snr_abort_threshold is None
            else float(self.snr_abort_threshold))
        scaled_peaks, accepted, abort = _live_select_peaks(
            peak_values, norms, float(self.snr_threshold), abort_threshold)
        host_peak_indices, host_accepted, host_abort = jax.device_get(
            (peak_indices, accepted, abort))
        if np.any(host_abort):
            logger.info("We are seeing some *really* high SNRs, let's "
                        "assume they aren't signals and just give up")
            return False, []
        accepted_indices = np.flatnonzero(host_accepted)
        if getattr(peak_values.device, "platform", None) in ("cuda", "gpu"):
            columns = _live_candidate_columns_bucketed(
                self, scaled_peaks, peak_values, norms, native_norms,
                accepted_indices)
        else:
            selected = jnp.asarray(accepted_indices, dtype=jnp.int32)
            columns = _live_candidate_columns(
                scaled_peaks, native_norms, selected)
        snr, phase, sigmasq = columns

    result.update({
        "snr": snr,
        "coa_phase": phase,
        "end_time": np.asarray(
            float(self.data.start_time)
            + host_peak_indices[accepted_indices] / self.data.sample_rate,
            dtype=np.float64,
        ),
        "template_id": np.asarray(
            [tgroup[idx].id for idx in accepted_indices],
            dtype=jax.dtypes.canonicalize_dtype(np.uint64),
        ),
        "sigmasq": sigmasq,
    })

    for idx in accepted_indices:
        htilde = tgroup[idx]
        if hasattr(htilde, 'time_offset'):
            if 'time_offset' not in result:
                result['time_offset'] = []

        l = int(host_peak_indices[idx]) + valid_start
        veto_info.append(_LiveVetoCandidate(
            peak_values, norms, int(idx), l, htilde, stilde))
        if not hasattr(htilde, 'dict_params'):
            htilde.dict_params = {}
            for key in tkeys:
                htilde.dict_params[key] = htilde.params[key]

        for key in tkeys:
            result[key].append(htilde.dict_params[key])

        if hasattr(htilde, 'time_offset'):
            result['time_offset'].append(htilde.time_offset)

    for key in tkeys:
        result[key] = np.array(result[key])

    if 'time_offset' in result:
        result['time_offset'] = jnp.asarray(result['time_offset'])

    return result, veto_info


def live_batch_matched_filter_init_jax(control, templates, maxelements=2**27):
    """Batch original duration groups with unchanged template and PSD grids."""
    from pycbc.fft import IFFT
    from pycbc.types import zeros
    from pycbc.filter.matchedfilter import BatchCorrelator

    durations = np.array([1.0 / t.delta_f for t in templates])

    lsort = durations.argsort()
    durations = durations[lsort]
    templates = [templates[li] for li in lsort]

    # Figure out how to chunk together the templates into groups to process
    _, counts = np.unique(durations, return_counts=True)
    tsamples = [(len(t) - 1) * 2 for t in templates]
    grabs = maxelements / np.unique(tsamples)

    chunks = np.array([])
    num = 0
    for count, grab in zip(counts, grabs):
        chunks = np.append(chunks, np.arange(num, count + num, grab))
        chunks = np.append(chunks, [count + num])
        num += count
    chunks = np.unique(chunks).astype(np.uint32)

    # We now have how many templates to grab at a time.
    control.chunks = chunks[1:] - chunks[0:-1]

    control.out_mem = {}
    control.cout_mem = {}
    control.ifts = {}
    chunk_durations = [durations[i] for i in chunks[:-1]]
    control.chunk_tsamples = [tsamples[int(i)] for i in chunks[:-1]]
    samples = control.chunk_tsamples * control.chunks

    # Create workspace memory for correlate and snr
    mem_ids = [(a, b) for a, b in zip(chunk_durations, control.chunks)]
    mem_types = set(zip(mem_ids, samples))

    control.tgroups, control.mids = [], []
    for i, size in mem_types:
        dur, count = i
        control.out_mem[i] = zeros(size, dtype=np.complex64)
        control.cout_mem[i] = zeros(size, dtype=np.complex64)
        control.ifts[i] = IFFT(
            control.cout_mem[i],
            control.out_mem[i],
            nbatch=count,
            size=len(control.cout_mem[i]) // count,
        )

    # Split the templates into their processing groups
    for dur, count in mem_ids:
        tgroup = templates[0:count]
        control.tgroups.append(tgroup)
        control.mids.append((dur, count))
        templates = templates[count:]

    # Associate the snr and corr memory block to each template
    control.corr = []
    for i, tgroup in enumerate(control.tgroups):
        psize = control.chunk_tsamples[i]
        s = 0
        e = psize
        mid = control.mids[i]
        for htilde in tgroup:
            htilde.out = control.out_mem[mid][s:e]
            htilde.cout = control.cout_mem[mid][s:e]
            s += psize
            e += psize
        control.corr.append(BatchCorrelator(
            tgroup, [t.cout for t in tgroup], len(tgroup[0]),
            immutable_templates=True))

    control.unique_delta_fs = tuple(sorted(
        {group[0].delta_f for group in control.tgroups}, reverse=True))


def set_live_data_jax(control, data):
    """Tell the JAX reader which unchanged template grids are required."""
    try:
        data.required_delta_fs = control.unique_delta_fs
    except AttributeError:
        pass


def _set_output_array(z, val):
    """Store results, preserving output wrappers and exposing write errors."""
    data = z if isinstance(z, JAXArrayData) else getattr(z, "_data", None)
    if isinstance(data, JAXArrayData):
        data.set_array(val)
    elif data is not None:
        data[:] = np.asarray(val)
    else:
        z[:] = np.asarray(val)


def _cpu_correlate(x, y):
    """Execute original compiled correlation on explicit host arrays."""
    from pycbc.filter import matchedfilter_cpu

    # Cython's original writable memoryviews cannot accept read-only JAX views.
    x = np.asarray(x).copy()
    y = np.asarray(y).copy()
    z = np.empty_like(x)
    matchedfilter_cpu.correlate(
        SimpleNamespace(data=x), SimpleNamespace(data=y),
        SimpleNamespace(data=z))
    return to_jax(z)


def _inner_product(x, y, weight=None):
    """Share array kernels and their independent original-CPU selectors."""
    operation = "inner" if weight is None else "weighted_inner"
    if _reference_enabled(operation):
        args = (y,) if weight is None else (y, weight)
        return to_jax(_cpu_reference(x, operation, *args))
    if weight is None:
        return _fast_inner_self(x) if x is y else _fast_inner(x, y)
    return (_fast_weighted_inner_self(x, weight) if x is y
            else _fast_weighted_inner(x, y, weight))


def _inverse_transform(correlation):
    """Use the original CPU IFFT only when its validation route is selected."""
    if not _reference_enabled("ifft"):
        return jnp.fft.ifft(correlation) * len(correlation)
    from pycbc.fft import ifft
    from pycbc.types import Array

    source = Array(JAXArrayData(correlation), copy=False)
    target = Array(JAXArrayData(jnp.zeros_like(correlation)), copy=False)
    ifft(source, target)
    return to_jax(target)


@jax.jit
def correlate_jax(x, y):
    """Pure JAX elementwise conjugate multiplication: conj(x) * y."""
    return jnp.conj(x) * y


def correlate(x, y, z):
    """Elementwise z = conj(x) * y in JAX scheme."""
    _ensure_x64()
    x_arr = to_jax(x)
    y_arr = to_jax(y)
    prod = (_cpu_correlate(x_arr, y_arr) if _reference_enabled("correlate")
            else correlate_jax(x_arr, y_arr))
    _set_output_array(z[:len(prod)], prod)


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
    if _reference_enabled("correlate"):
        for x, z in zip(self.xs, self.zs):
            products = _cpu_correlate(to_jax(x)[:size], y_arr)
            _set_output_array(z[:size], products)
        return

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
        if (parent is None or not isinstance(info, slice)
                or info.step not in (None, 1)):
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
    products = correlate_jax(templates, y_arr)
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
    device = _functional_device(htilde, psd)
    h_arr = _functional_array(htilde, device)
    n_pts = (len(h_arr) - 1) * 2
    flow = low_frequency_cutoff
    fhigh = high_frequency_cutoff

    kmin, kmax = get_cutoff_indices(flow, fhigh, delta_f, n_pts)

    ht = h_arr[kmin:kmax]
    weight = (None if psd is None
              else _functional_array(psd, device)[kmin:kmax])
    return _inner_product(ht, ht, weight).real * (4.0 * delta_f)


def sigmasq_series_jax(
    htilde,
    psd=None,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
    delta_f=1.0,
):
    """Pure JAX calculation of cumulative power frequency series."""
    _ensure_x64()
    device = _functional_device(htilde, psd)
    h_arr = _functional_array(htilde, device)
    n_pts = (len(h_arr) - 1) * 2
    flow = low_frequency_cutoff
    fhigh = high_frequency_cutoff

    kmin, kmax = get_cutoff_indices(flow, fhigh, delta_f, n_pts)

    mag_sq = (to_jax(_cpu_reference(h_arr, "squared_norm"))
              if _reference_enabled("squared_norm")
              else h_arr.real ** 2 + h_arr.imag ** 2)
    if psd is not None:
        psd_arr = _functional_array(psd, device)
        mag_sq = _divide(mag_sq, psd_arr, inplace=True)

    sub = mag_sq[kmin:kmax]
    sub = (to_jax(_cpu_reference(sub, "cumsum"))
           if _reference_enabled("cumsum") else jnp.cumsum(sub))
    norm = 4.0 * delta_f

    series = jnp.zeros(len(h_arr), dtype=h_arr.real.dtype, device=device)
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

    device = _functional_device(template, data, psd, h_norm)
    if isinstance(template, TimeSeries):
        template = make_frequency_series(template)
    if isinstance(data, TimeSeries):
        data = make_frequency_series(data)

    htilde = _functional_array(template, device)
    stilde = _functional_array(data, device)
    psd_arr = None if psd is None else _functional_array(psd, device)

    if len(htilde) != len(stilde):
        raise ValueError("Length of template and data must match")

    if delta_f is None:
        delta_f = getattr(data, "delta_f", 1.0)
    n_time = (len(stilde) - 1) * 2
    flow = low_frequency_cutoff
    fhigh = high_frequency_cutoff

    kmin, kmax = get_cutoff_indices(flow, fhigh, delta_f, n_time)

    qtilde = jnp.zeros(n_time, dtype=stilde.dtype, device=device)
    corr_slice = (_cpu_correlate(htilde[kmin:kmax], stilde[kmin:kmax])
                  if _reference_enabled("correlate")
                  else correlate_jax(htilde[kmin:kmax], stilde[kmin:kmax]))

    if psd_arr is not None:
        corr_slice = _divide(corr_slice, psd_arr[kmin:kmax], inplace=True)

    qtilde = qtilde.at[kmin:kmax].set(corr_slice)

    # Complex-to-complex IFFT of length n_time
    snr_time = _inverse_transform(qtilde)

    if h_norm is None:
        h_norm = sigmasq_jax(
            htilde,
            psd=psd_arr,
            low_frequency_cutoff=low_frequency_cutoff,
            high_frequency_cutoff=high_frequency_cutoff,
            delta_f=delta_f,
        )
    else:
        h_norm = _functional_array(h_norm, device)

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

    device = _functional_device(vec1, vec2, psd)
    if isinstance(vec1, TimeSeries):
        vec1 = make_frequency_series(vec1)
    if isinstance(vec2, TimeSeries):
        vec2 = make_frequency_series(vec2)

    v1 = _functional_array(vec1, device)
    v2 = _functional_array(vec2, device)
    psd_arr = None if psd is None else _functional_array(psd, device)
    n_pts = (len(v1) - 1) * 2
    flow = low_frequency_cutoff
    fhigh = high_frequency_cutoff

    kmin, kmax = get_cutoff_indices(flow, fhigh, delta_f, n_pts)

    v1_sub = v1[kmin:kmax]
    v2_sub = v2[kmin:kmax]
    weight = None if psd_arr is None else psd_arr[kmin:kmax]
    inn = _inner_product(v1_sub, v2_sub, weight).real * (4.0 * delta_f)
    if not normalized:
        return inn

    norm1 = sigmasq_jax(
        v1,
        psd=psd_arr,
        low_frequency_cutoff=flow,
        high_frequency_cutoff=fhigh,
        delta_f=delta_f,
    )
    norm2 = sigmasq_jax(
        v2,
        psd=psd_arr,
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

    device = _functional_device(vec1, vec2, psd, v1_norm, v2_norm)
    if isinstance(vec1, TimeSeries):
        vec1 = make_frequency_series(vec1)
    if isinstance(vec2, TimeSeries):
        vec2 = make_frequency_series(vec2)

    v1 = _functional_array(vec1, device)
    v2 = _functional_array(vec2, device)
    psd_arr = None if psd is None else _functional_array(psd, device)
    n_pts = (len(v1) - 1) * 2
    flow = low_frequency_cutoff
    fhigh = high_frequency_cutoff

    kmin, kmax = get_cutoff_indices(flow, fhigh, delta_f, n_pts)

    corr = jnp.zeros(n_pts, dtype=v1.dtype, device=device)
    prod = (_cpu_correlate(v1[kmin:kmax], v2[kmin:kmax])
            if _reference_enabled("correlate")
            else correlate_jax(v1[kmin:kmax], v2[kmin:kmax]))
    if psd_arr is not None:
        prod = _divide(prod, psd_arr[kmin:kmax], inplace=True)

    corr = corr.at[kmin:kmax].set(prod)
    time_series = _inverse_transform(corr)

    if _reference_enabled("abs_max_loc"):
        max_value, max_idx = _cpu_reference(time_series, "abs_max_loc")
    else:
        magnitude = time_series.real ** 2 + time_series.imag ** 2
        max_idx = int(jnp.argmax(magnitude))
        max_value = jnp.sqrt(magnitude[max_idx])

    if v1_norm is None:
        v1_norm = sigmasq_jax(
            v1,
            psd=psd_arr,
            low_frequency_cutoff=flow,
            high_frequency_cutoff=fhigh,
            delta_f=delta_f,
        )
    else:
        v1_norm = _functional_array(v1_norm, device)
    if v2_norm is None:
        v2_norm = sigmasq_jax(
            v2,
            psd=psd_arr,
            low_frequency_cutoff=flow,
            high_frequency_cutoff=fhigh,
            delta_f=delta_f,
        )
    else:
        v2_norm = _functional_array(v2_norm, device)

    # NumPy scalar arithmetic in the original match retains peak precision.
    peak = _functional_array(max_value, device)
    snr_norm = ((4.0 * delta_f) / jnp.sqrt(v1_norm)).astype(peak.dtype)
    second_norm = jnp.sqrt(v2_norm).astype(peak.dtype)
    return _divide(peak * snr_norm, second_norm), max_idx


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
    """Reduce contiguous output to one peak index and value per template.

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
    if _reference_enabled("abs_arg_max"):
        values = tensor.reshape(template_count, template_size)[
            :, segment_start:segment_stop]
        indices = to_jax(np.asarray([
            _cpu_reference(row, "abs_arg_max") for row in values], np.int64))
        peaks = values[jnp.arange(template_count), indices]
    else:
        indices, peaks = _batch_peak_core(
            tensor, template_count, segment_start, segment_stop,
        )
    # Keep reductions on the selected JAX device.  The live filter consumes
    # these values directly and must not synchronize each template through a
    # NumPy conversion before veto processing.
    return indices, peaks


def batch_matched_filter_bank(
    templates,
    strain,
    psd=None,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
    delta_f=None,
):
    """High-throughput batched template bank matched filtering in pure JAX.

    Uses the single-template calculation's band, correlation/division order,
    normalization and output precision. The default kernels run on-device;
    selected validation operations use their original CPU implementations.

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
    inputs = tuple(templates) if isinstance(templates, (list, tuple)) else (
        templates,)
    device = _functional_device(*inputs, strain, psd)
    if isinstance(templates, (list, tuple)):
        templates_j = jnp.stack([
            _functional_array(template, device) for template in templates])
        template_df = getattr(templates[0], "delta_f", None)
    else:
        templates_j = _functional_array(templates, device)
        template_df = getattr(templates, "delta_f", None)
    if templates_j.ndim == 1:
        templates_j = templates_j[None, :]
    df = delta_f if delta_f is not None else getattr(strain, "delta_f", None)
    if df is None:
        df = template_df if template_df is not None else 1.0

    strain_j = _functional_array(strain, device)
    n_time = (strain_j.shape[-1] - 1) * 2
    kmin, kmax = get_cutoff_indices(
        low_frequency_cutoff, high_frequency_cutoff, df, n_time)
    template_band = templates_j[:, kmin:kmax]
    strain_band = strain_j[kmin:kmax]
    weight = (None if psd is None
              else _functional_array(psd, device)[kmin:kmax])
    if _reference_enabled("weighted_inner" if weight is not None else "inner"):
        sigmasqs = jnp.stack([
            _inner_product(row, row, weight).real for row in template_band])
    elif weight is None:
        sigmasqs = jax.vmap(_fast_inner_self)(template_band)
    else:
        sigmasqs = jax.vmap(_fast_weighted_inner_self, in_axes=(0, None))(
            template_band, weight
        )
    sigmasqs = sigmasqs * (4.0 * df)

    products = (jnp.stack([_cpu_correlate(row, strain_band)
                          for row in template_band])
                if _reference_enabled("correlate") else correlate_jax(
                    template_band, strain_band))
    if weight is not None:
        products = _divide(products, weight[None, :], inplace=True)
    correlation = jnp.zeros((len(templates_j), n_time), dtype=strain_j.dtype,
                            device=device)
    correlation = correlation.at[:, kmin:kmax].set(products)
    snr = (jnp.stack([_inverse_transform(row) for row in correlation])
           if _reference_enabled("ifft") else jnp.fft.ifft(
               correlation, axis=-1) * n_time)
    norm = (4.0 * df) / jnp.sqrt(sigmasqs)
    # Native matched_filter multiplies its existing complex output in place.
    norm = norm.astype(snr.real.dtype)
    return (snr * norm[:, None]).astype(snr.dtype), sigmasqs


def batch_sigmasq_jax(templates, psd):
    """Return live-template norms as a host float32 metadata vector."""
    from pycbc.scheme import current_backend_key

    key = (id(psd), current_backend_key())
    cached = getattr(templates, "_cached_sigmasqs", None)
    if (cached is not None
            and getattr(templates, "_cached_sigmasqs_psd_id", None) == id(psd)
            and getattr(templates, "_cached_sigmasqs_key", None) == key):
        return cached
    values = live_template_norms_jax(
        templates,
        psd,
        template_matrix=getattr(templates, "_batch_tensor", None),
    )
    result = np.asarray(jax.device_get(values), dtype=np.float32)
    try:
        templates._cached_sigmasqs = result
        templates._cached_sigmasqs_psd_id = id(psd)
        templates._cached_sigmasqs_key = key
    except AttributeError:
        pass
    return result
