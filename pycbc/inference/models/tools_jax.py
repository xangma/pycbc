"""JAX implementations for :mod:`pycbc.inference.models.tools`.

This module is imported lazily when a shared inference helper receives a
JAX-backed value. Keeping the import here avoids making JAX a dependency
of the NumPy inference path.
"""

import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy

from pycbc.types.backend import backend_array


def _jax_array(value):
    """Return the public JAX backend array for ``value``."""
    return backend_array(value, "jax")


def _numpy_array(value):
    """Return a host array for a non-JAX operand."""
    value = backend_array(value)
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    return numpy.asarray(value)


def inner(left, right):
    """Return the JAX inner product without scalarizing reductions."""
    left = _jax_array(left)
    right = _jax_array(right)
    dtype = jnp.promote_types(left.dtype, right.dtype)
    left = jnp.asarray(left, dtype=dtype)
    right = jnp.asarray(right, dtype=dtype)
    accumulation_dtype = (
        jnp.complex128
        if jnp.issubdtype(dtype, jnp.complexfloating)
        else jnp.float64
    )

    left_acc = left.astype(accumulation_dtype).reshape(-1)
    right_acc = right.astype(accumulation_dtype).reshape(-1)
    return jnp.vdot(left_acc, right_acc)


def real_inner(left, right):
    """Return the real JAX inner product, optimizing identical storage."""
    left_arr = _jax_array(left)
    right_arr = _jax_array(right)
    is_same = left is right or left_arr is right_arr
    if not is_same:
        return inner(left, right).real

    if jnp.issubdtype(left_arr.dtype, jnp.complexfloating):
        summand = jnp.square(left_arr.real) + jnp.square(left_arr.imag)
    else:
        summand = jnp.square(left_arr)
    return jnp.sum(summand, dtype=jnp.float64)


def _to_jax(value, like, *, real=False):
    """Convert one mixed-backend operand to a JAX array matching ``like``."""
    arr = _jax_array(value)
    if arr is not None:
        return arr
    value = _numpy_array(value)
    if real:
        dtype = like.real.dtype
    else:
        dtype = jnp.complex128 if numpy.iscomplexobj(value) else jnp.float64
    return jnp.asarray(value, dtype=dtype)


def fused_inner_hd_hh(h, d, weight=None):
    """Compute JAX ``<h|d>`` and ``<h|h>`` in a single pass."""
    h_arr = _jax_array(h)
    d_arr = _jax_array(d)
    like = h_arr if h_arr is not None else d_arr
    h_arr = _to_jax(h, like)
    d_arr = _to_jax(d, like)
    if weight is not None:
        h_arr = h_arr * _to_jax(weight, h_arr, real=True)

    dtype = jnp.promote_types(h_arr.dtype, d_arr.dtype)
    h_arr = jnp.asarray(h_arr, dtype=dtype)
    d_arr = jnp.asarray(d_arr, dtype=dtype)
    is_batched = h_arr.ndim > 1 or d_arr.ndim > 1
    is_cplx = jnp.issubdtype(dtype, jnp.complexfloating)

    if is_batched:
        real_acc_dtype = jnp.float64
        accumulation_dtype = (
            jnp.complex128 if is_cplx else jnp.float64
        )
        h_acc = h_arr.astype(accumulation_dtype)
        d_acc = d_arr.astype(accumulation_dtype)
        if is_cplx:
            real_hd = jnp.sum(
                h_acc.real * d_acc.real + h_acc.imag * d_acc.imag,
                axis=-1,
                dtype=real_acc_dtype,
            )
            imag_hd = jnp.sum(
                h_acc.real * d_acc.imag - h_acc.imag * d_acc.real,
                axis=-1,
                dtype=real_acc_dtype,
            )
            cplx_hd = real_hd + 1j * imag_hd
            hh = jnp.sum(
                jnp.square(h_acc.real) + jnp.square(h_acc.imag),
                axis=-1,
                dtype=real_acc_dtype,
            )
        else:
            cplx_hd = jnp.sum(h_acc * d_acc, axis=-1, dtype=accumulation_dtype)
            hh = jnp.sum(jnp.square(h_acc), axis=-1, dtype=real_acc_dtype)
        return cplx_hd, hh

    accumulation_dtype = jnp.complex128 if is_cplx else jnp.float64
    h_acc = h_arr.astype(accumulation_dtype).reshape(-1)
    d_acc = d_arr.astype(accumulation_dtype).reshape(-1)
    if is_cplx:
        real_hd = jnp.sum(
            h_acc.real * d_acc.real + h_acc.imag * d_acc.imag,
            dtype=jnp.float64,
        )
        imag_hd = jnp.sum(
            h_acc.real * d_acc.imag - h_acc.imag * d_acc.real,
            dtype=jnp.float64,
        )
        cplx_hd = real_hd + 1j * imag_hd
        hh = jnp.sum(
            jnp.square(h_acc.real) + jnp.square(h_acc.imag),
            dtype=jnp.float64,
        )
    else:
        cplx_hd = jnp.dot(h_acc, d_acc)
        hh = jnp.sum(jnp.square(h_acc), dtype=jnp.float64)
    return cplx_hd, hh


def selected_values(values, indices, *, host=True):
    """Select values using device-resident integer indices."""
    arr = _jax_array(values)
    indices = jnp.asarray(indices, dtype=jnp.int64)
    selected = arr[indices]
    return numpy.asarray(selected) if host else selected


def add_values(total, values):
    """Add values on the device of the JAX-backed operand."""
    total_arr = _jax_array(total)
    values_arr = _jax_array(values)
    template = total_arr if total_arr is not None else values_arr
    if total_arr is None:
        total_arr = jnp.asarray(total, dtype=template.dtype)
    if values_arr is None:
        values_arr = jnp.asarray(values, dtype=template.dtype)
    return total_arr + values_arr


def last_index_at_or_below(values, upper):
    """Return the last sorted-value index at or below ``upper``."""
    arr = _jax_array(values)
    insertion = int(
        jnp.searchsorted(arr, jnp.asarray(upper, dtype=arr.dtype), side="right")
    )
    if insertion == 0:
        raise IndexError(f"no values are at or below {upper}")
    return int(insertion - 1)


def threshold_extent(values, threshold):
    """Return the first and last JAX indices above ``threshold``."""
    arr = _jax_array(values)
    indices = jnp.flatnonzero(jnp.abs(arr) > threshold)
    return int(indices[0]), int(indices[-1])


def _weighted_cdf(loglr):
    arr = _jax_array(loglr)
    cdf = jnp.cumsum(jnp.exp(arr - jnp.max(arr)), axis=0)
    return arr, cdf / cdf[-1]


def draw_sample(loglr, size=None, *, host=True):
    """Draw weighted indices without unnecessary device transfers."""
    arr, cdf = _weighted_cdf(loglr)
    uniforms = (
        numpy.random.uniform(size=size) if size else numpy.random.uniform()
    )
    uniforms = jnp.asarray(uniforms, dtype=arr.dtype)
    indices = jnp.searchsorted(cdf, uniforms)
    if indices.ndim == 0:
        return int(indices)
    return numpy.asarray(indices) if host else indices


def draw_device_sample_with_host_rng(loglr, size):
    """Draw device indices while retaining NumPy's RNG stream."""
    arr, cdf = _weighted_cdf(loglr)
    uniforms = jnp.asarray(
        numpy.random.uniform(size=size),
        dtype=arr.dtype,
    )
    return jnp.searchsorted(cdf, uniforms)


def device_index_matrix_to_host(indices):
    """Materialize a matrix of device indices at one host boundary."""
    return numpy.asarray(jnp.stack(indices, axis=0))


def same_device(values):
    """Return whether all JAX-backed values share one device."""
    arrays = [_jax_array(value) for value in values]
    return all(arr.device == arrays[0].device for arr in arrays[1:])


def weighted_loglr(values, weights):
    """Add probability weights on the likelihood device."""
    arr = _jax_array(values)
    weights = jnp.asarray(weights, dtype=arr.real.dtype)
    return arr + jnp.log(weights)


def phase_reconstruction_values(sh, hh, sample_count=int(1e4)):
    """Build the phase-reconstruction grid on its input device."""
    sh_arr = _jax_array(sh)
    hh_arr = _jax_array(hh)
    like = sh_arr if sh_arr is not None else hh_arr
    real_dtype = like.real.dtype
    is_cplx_sh = (
        jnp.issubdtype(sh_arr.dtype, jnp.complexfloating)
        if sh_arr is not None
        else numpy.iscomplexobj(sh)
    )
    sh = jnp.asarray(
        sh_arr if sh_arr is not None else sh,
        dtype=(
            jnp.complex128
            if real_dtype == jnp.float64
            else jnp.complex64
        ) if is_cplx_sh else real_dtype,
    )
    hh = jnp.asarray(
        hh_arr if hh_arr is not None else hh,
        dtype=real_dtype,
    )
    phase = jnp.linspace(
        0,
        2.0 * numpy.pi,
        sample_count,
        dtype=real_dtype,
    )
    angle = -2.0 * phase
    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)
    sh_r = sh.real if jnp.issubdtype(sh.dtype, jnp.complexfloating) else sh
    sh_i = sh.imag if jnp.issubdtype(sh.dtype, jnp.complexfloating) else 0.0
    return phase, (cos_angle * sh_r - sin_angle * sh_i) + hh


def selected_scalar(values, index):
    """Return one selected JAX value at the public scalar boundary."""
    return _jax_array(values)[index].item()


def random_permutation(values, size):
    """Return device and host views of one random permutation."""
    choice_host = numpy.random.permutation(len(values))[:size]
    choice_dev = jnp.asarray(choice_host)
    return choice_dev, choice_host


def normalize_logweights(values):
    """Normalize device-resident log weights."""
    arr = _jax_array(values)
    return arr - jsp.logsumexp(arr, axis=0)


def host_indices(values):
    """Materialize device indices for public host calculations."""
    return numpy.asarray(_jax_array(values))


def _bspline_basis(values, knots, degree):
    """Return active B-spline coefficient indices and basis values."""
    coefficient_count = knots.size - degree - 1
    spans = jnp.searchsorted(knots, values, side="right") - 1
    spans = jnp.clip(spans, degree, coefficient_count - 1)

    basis = jnp.ones(values.shape + (1,), dtype=values.dtype)
    left = [None] * (degree + 1)
    right = [None] * (degree + 1)
    for column in range(1, degree + 1):
        left[column] = values - knots[spans + 1 - column]
        right[column] = knots[spans + column] - values
        saved = jnp.zeros_like(values)
        updated = []
        for row in range(column):
            weight = basis[..., row] / (right[row + 1] + left[column - row])
            updated.append(saved + right[row + 1] * weight)
            saved = left[column - row] * weight
        updated.append(saved)
        basis = jnp.stack(updated, axis=-1)

    offsets = jnp.arange(degree + 1, dtype=jnp.int64)
    indices = spans[..., None] - degree + offsets
    return indices, basis


def rect_bivariate_spline_evaluator(interp):
    """Create a device-native evaluator for a SciPy bivariate spline."""
    knots_x, knots_y = interp.get_knots()
    degree_x, degree_y = interp.degrees
    coefficient_count_x = len(knots_x) - degree_x - 1
    coefficient_count_y = len(knots_y) - degree_y - 1
    coefficients = interp.get_coeffs().reshape(
        coefficient_count_x, coefficient_count_y
    )
    cache = {}

    def evaluate(x, y, bounds_check=True):
        x_arr = _jax_array(x)
        y_arr = _jax_array(y)
        like = x_arr if x_arr is not None else y_arr
        dtype = like.real.dtype
        if x_arr is not None:
            dtype = jnp.promote_types(dtype, x_arr.real.dtype)
        if y_arr is not None:
            dtype = jnp.promote_types(dtype, y_arr.real.dtype)
        x_arr = jnp.asarray(
            x_arr if x_arr is not None else x,
            dtype=dtype,
        )
        y_arr = jnp.asarray(
            y_arr if y_arr is not None else y,
            dtype=dtype,
        )
        x_arr, y_arr = jnp.broadcast_arrays(x_arr, y_arr)

        key = dtype
        cached = cache.get(key)
        if cached is None:
            cached = (
                jnp.asarray(knots_x, dtype=dtype),
                jnp.asarray(knots_y, dtype=dtype),
                jnp.asarray(coefficients, dtype=dtype),
            )
            cache[key] = cached
        arr_knots_x, arr_knots_y, arr_coefficients = cached

        indices_x, basis_x = _bspline_basis(x_arr, arr_knots_x, degree_x)
        indices_y, basis_y = _bspline_basis(y_arr, arr_knots_y, degree_y)
        local_coefficients = arr_coefficients[
            indices_x[..., :, None], indices_y[..., None, :]
        ]
        values = (
            local_coefficients * basis_x[..., :, None] * basis_y[..., None, :]
        ).sum(axis=(-2, -1))

        if bounds_check:
            outside = (
                (x_arr < knots_x[degree_x])
                | (x_arr > knots_x[-degree_x - 1])
                | (y_arr < knots_y[degree_y])
                | (y_arr > knots_y[-degree_y - 1])
            )
            values = jnp.where(outside, -jnp.inf, values)
        return values

    return evaluate


def numpy_from_backend(value):
    """Return a detached CPU value for a NumPy-only calculation."""
    arr = _jax_array(value)
    if arr is None:
        return value
    if arr.ndim == 0:
        return arr.item()
    return numpy.asarray(arr)


def marginalize_likelihood(
    sh,
    hh,
    logw,
    phase,
    distance,
    skip_vector,
    return_peak,
    return_complex,
    interpolator=None,
):
    """JAX implementation of explicit likelihood marginalizations."""
    sh_arr = _jax_array(sh)
    hh_arr = _jax_array(hh)
    if sh_arr is None:
        real_dtype = hh_arr.real.dtype
        if numpy.iscomplexobj(sh):
            sh_dtype = (
                jnp.complex128 if real_dtype == jnp.float64 else jnp.complex64
            )
        else:
            sh_dtype = real_dtype
        sh = jnp.asarray(sh, dtype=sh_dtype)
    else:
        sh = sh_arr
    hh = jnp.asarray(
        hh_arr if hh_arr is not None else hh,
        dtype=sh.real.dtype,
    )

    if distance and interpolator is None and sh.ndim:
        raise ValueError(
            "Cannot do vector marginalization and distance at the same time"
        )

    if return_complex:
        pass
    elif phase:
        sh = jnp.abs(sh)
    else:
        sh = sh.real

    if distance and interpolator is None:
        dist_rescale, dist_weights = distance
        dist_rescale = jnp.asarray(dist_rescale, dtype=hh.dtype)
        dist_weights = jnp.asarray(dist_weights, dtype=hh.dtype)
        sh = sh * dist_rescale
        hh = hh * jnp.square(dist_rescale)
        logw = jnp.log(dist_weights)

    if return_complex and interpolator is None:
        return sh, -0.5 * hh

    if interpolator is not None:
        vloglr = interpolator(sh, hh)
    else:
        if phase:
            sh = jnp.log(jsp.i0e(sh)) + sh
        vloglr = sh - 0.5 * hh
    if return_peak:
        if vloglr.ndim:
            maxv = int(vloglr.argmax())
            maxl = vloglr[maxv].item()
        else:
            maxv = 0
            maxl = vloglr.item()

    if not skip_vector:
        if vloglr.ndim:
            if logw is None:
                logw = -jnp.log(jnp.asarray(vloglr.shape[0], dtype=vloglr.dtype))
            else:
                logw_arr = _jax_array(logw)
                if logw_arr is not None:
                    logw = logw_arr
                logw = jnp.asarray(logw, dtype=vloglr.dtype)
            vloglr = jsp.logsumexp(vloglr + logw, axis=0)
        vloglr = vloglr.item()

    if return_peak:
        return vloglr, maxv, maxl
    return vloglr


def batched_project_detector_strain(
    hp, hc, fp, fc, dt, delta_f, batch_size
):
    """Project polarizations onto detector with time delays for a batch in JAX."""
    hp_jax = _jax_array(hp)
    hc_jax = _jax_array(hc)
    fp_jax = _jax_array(fp)
    fc_jax = _jax_array(fc)
    dt_jax = _jax_array(dt)
    jax_arrays = (hp_jax, hc_jax, fp_jax, fc_jax, dt_jax)

    like = next(a for a in jax_arrays if a is not None)
    real_dtype = like.real.dtype
    complex_dtype = (
        jnp.complex128 if real_dtype == jnp.float64 else jnp.complex64
    )

    def _to_jax_arr(x, target_dt):
        arr = _jax_array(x)
        if arr is not None:
            return arr.astype(target_dt)
        if hasattr(x, "numpy") and callable(x.numpy):
            arr = x.numpy()
        elif hasattr(x, "data") and not isinstance(x.data, memoryview):
            arr = numpy.asarray(x.data)
        else:
            arr = numpy.asarray(x)
        return jnp.asarray(arr, dtype=target_dt)

    hp_t = (
        hp_jax
        if hp_jax is not None
        else _to_jax_arr(hp, complex_dtype)
    )
    if hc_jax is None:
        hc_t = _to_jax_arr(hc, complex_dtype)
    else:
        hc_t = hc_jax.astype(complex_dtype)

    if hp_t.ndim != 1 or hc_t.ndim != 1:
        raise ValueError(
            "batched likelihood waveform parameters must be "
            "scalar; only detector-frame extrinsics may be "
            "batched"
        )

    def _sample_column(value, name):
        value_arr = _jax_array(value)
        if value_arr is None:
            value_arr = jnp.asarray(value, dtype=real_dtype)
        else:
            value_arr = value_arr.astype(real_dtype)
        if value_arr.ndim == 0:
            value_arr = jnp.broadcast_to(value_arr, (batch_size,))
        elif value_arr.ndim == 1 and value_arr.shape[0] == 1:
            value_arr = jnp.broadcast_to(value_arr, (batch_size,))
        elif value_arr.ndim != 1 or value_arr.shape[0] != batch_size:
            raise ValueError(
                f"Batched detector-frame parameter {name!r} "
                f"must be scalar or have length {batch_size}"
            )
        return jnp.expand_dims(value_arr, axis=-1)

    fp_t = _sample_column(fp, "fplus")
    fc_t = _sample_column(fc, "fcross")
    dt_t = _sample_column(dt, "tc")

    n_freq = hp_t.shape[-1]
    freqs = jnp.arange(n_freq, dtype=real_dtype) * delta_f

    phase = -2.0 * jnp.pi * freqs * dt_t
    shift_r = jnp.cos(phase)
    shift_i = jnp.sin(phase)
    hp_r = hp_t.real if jnp.issubdtype(hp_t.dtype, jnp.complexfloating) else hp_t
    hp_i = hp_t.imag if jnp.issubdtype(hp_t.dtype, jnp.complexfloating) else jnp.zeros_like(hp_r)
    hc_r = hc_t.real if jnp.issubdtype(hc_t.dtype, jnp.complexfloating) else hc_t
    hc_i = hc_t.imag if jnp.issubdtype(hc_t.dtype, jnp.complexfloating) else jnp.zeros_like(hc_r)
    h_proj_r = fp_t * hp_r + fc_t * hc_r
    h_proj_i = fp_t * hp_i + fc_t * hc_i
    h_det_r = h_proj_r * shift_r - h_proj_i * shift_i
    h_det_i = h_proj_r * shift_i + h_proj_i * shift_r
    return h_det_r + 1j * h_det_i
