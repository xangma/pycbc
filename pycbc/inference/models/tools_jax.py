"""JAX implementations for :mod:`pycbc.inference.models.tools`.

This module is imported lazily when a shared inference helper receives a
JAX-backed value. Keeping the import here avoids making JAX a dependency
of the NumPy inference path.
"""

import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy

import jax
from pycbc.types.backend import backend_array
from pycbc.types.array_jax import _reference_enabled, _cpu_reference, _divide, to_jax


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
    """Inner product on-device, with independently selected native stages."""
    left = _jax_array(left)
    right = _jax_array(right)
    if _reference_enabled("inner") or _reference_enabled("inference_inner"):
        if isinstance(left, jax.core.Tracer) or isinstance(right, jax.core.Tracer):
            raise RuntimeError("Native inner validation cannot run inside jax.jit")
        return to_jax(_cpu_reference(left, "inner", right), device=left.device)
    dtype = jnp.promote_types(left.dtype, right.dtype)
    accumulation = (
        jnp.complex128 if jnp.issubdtype(dtype, jnp.complexfloating) else jnp.float64
    )
    summand = jnp.conj(left.astype(accumulation)) * right.astype(accumulation)
    if _reference_enabled("sum"):
        return to_jax(_cpu_reference(summand.reshape(-1), "sum"), device=left.device)
    return jnp.sum(summand, dtype=accumulation)


def real_inner(left, right):
    """Real part of the same configured inner-product boundary."""
    return inner(left, right).real


def _to_jax(value, like, *, real=False):
    """Place mixed inputs beside their authoritative device array."""
    arr = _jax_array(value)
    if arr is None:
        arr = _numpy_array(value)
    # Preserve weight precision until the original in-place cast boundary.
    device = None if isinstance(like, jax.core.Tracer) else like.device
    return to_jax(arr, device=device)


def fused_inner_hd_hh(h, d, weight=None):
    """Whiten a template and evaluate the two configured inner products."""
    h_arr = _jax_array(h)
    d_arr = _jax_array(d)
    like = h_arr if h_arr is not None else d_arr
    if like is None:
        like = to_jax(_numpy_array(h))
    h_arr = _to_jax(h, like)
    d_arr = _to_jax(d, like)
    if weight is not None:
        h_arr = whiten_template(h_arr, weight)
    if h_arr.ndim == 1 and d_arr.ndim == 1:
        return inner(h_arr, d_arr), real_inner(h_arr, h_arr)
    shape = numpy.broadcast_shapes(h_arr.shape, d_arr.shape)
    hs = jnp.broadcast_to(h_arr, shape).reshape(-1, shape[-1])
    ds = jnp.broadcast_to(d_arr, shape).reshape(-1, shape[-1])
    outputs = [(inner(hr, dr), real_inner(hr, hr)) for hr, dr in zip(hs, ds)]
    hd, hh = zip(*outputs)
    return jnp.stack(hd).reshape(shape[:-1]), jnp.stack(hh).reshape(shape[:-1])


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
    weights = jnp.exp(arr - jnp.max(arr))
    if _reference_enabled("cumsum"):
        cumulative = to_jax(_cpu_reference(weights, "cumsum"), device=arr.device)
    else:
        cumulative = jnp.cumsum(weights, axis=0)
    return arr, _divide(cumulative, cumulative[-1])


def draw_sample(loglr, size=None, *, host=True):
    """Draw weighted indices without unnecessary device transfers."""
    if _reference_enabled("inference_sampling"):
        from .tools import draw_sample as original

        arr = _jax_array(loglr)
        if isinstance(arr, jax.core.Tracer):
            raise RuntimeError("Native sampling cannot run inside jax.jit")
        indices = original(numpy.asarray(arr), size=size)
        return (
            indices
            if host or numpy.ndim(indices) == 0
            else to_jax(indices, device=arr.device)
        )
    arr, cdf = _weighted_cdf(loglr)
    uniforms = numpy.random.uniform(size=size) if size else numpy.random.uniform()
    uniforms = jnp.asarray(uniforms, dtype=arr.dtype)
    indices = jnp.searchsorted(cdf, uniforms)
    if indices.ndim == 0:
        return int(indices)
    return numpy.asarray(indices) if host else indices


def draw_device_sample_with_host_rng(loglr, size):
    """Draw device indices using the caller's NumPy RNG stream."""
    return draw_sample(loglr, size=size, host=False)


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
    if _reference_enabled("inference_sampling"):
        if any(isinstance(value, jax.core.Tracer) for value in (arr, weights)):
            raise RuntimeError("Native sampling weights cannot run inside jax.jit")
        return to_jax(
            numpy.asarray(arr) + numpy.log(numpy.asarray(weights)), device=arr.device
        )
    weights = jnp.asarray(weights, dtype=arr.real.dtype)
    return arr + jnp.log(weights)


def phase_reconstruction_values(sh, hh, sample_count=int(1e4)):
    """Build the phase-reconstruction grid on its input device."""
    sh_arr = _jax_array(sh)
    hh_arr = _jax_array(hh)
    like = sh_arr if sh_arr is not None else hh_arr
    if _reference_enabled("inference_sampling"):
        if any(isinstance(value, jax.core.Tracer) for value in (like, sh, hh)):
            raise RuntimeError("Native phase reconstruction cannot run inside jax.jit")
        phase = numpy.linspace(0, 2.0 * numpy.pi, sample_count)
        values = (numpy.exp(-2.0j * phase) * numpy.asarray(sh)).real + numpy.asarray(hh)
        return to_jax(phase, device=like.device), to_jax(values, device=like.device)
    real_dtype = like.real.dtype
    is_cplx_sh = (
        jnp.issubdtype(sh_arr.dtype, jnp.complexfloating)
        if sh_arr is not None
        else numpy.iscomplexobj(sh)
    )
    sh = jnp.asarray(
        sh_arr if sh_arr is not None else sh,
        dtype=(
            (jnp.complex128 if real_dtype == jnp.float64 else jnp.complex64)
            if is_cplx_sh
            else real_dtype
        ),
    )
    hh = jnp.asarray(
        hh_arr if hh_arr is not None else hh,
        dtype=real_dtype,
    )
    phase = jnp.linspace(
        0,
        2.0 * numpy.pi,
        sample_count,
        dtype=jnp.float64,
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


def random_permutation(values, size, rng):
    """Return device and host views of one random permutation."""
    choice_host = rng.choice(len(values), size=size, replace=False)
    choice_dev = to_jax(choice_host, device=_jax_array(values).device)
    return choice_dev, choice_host


def normalize_logweights(values):
    """Normalize device-resident log weights."""
    arr = _jax_array(values)
    if _reference_enabled("inference_marginalization"):
        from scipy.special import logsumexp

        return to_jax(
            numpy.asarray(arr) - logsumexp(numpy.asarray(arr)), device=arr.device
        )
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
    coefficients = interp.get_coeffs().reshape(coefficient_count_x, coefficient_count_y)
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
        device = None if isinstance(like, jax.core.Tracer) else like.device
        x_arr = to_jax(x_arr if x_arr is not None else x, dtype=dtype, device=device)
        y_arr = to_jax(y_arr if y_arr is not None else y, dtype=dtype, device=device)
        x_arr, y_arr = jnp.broadcast_arrays(x_arr, y_arr)

        device = None if isinstance(x_arr, jax.core.Tracer) else x_arr.device
        key = (dtype, device)
        cached = cache.get(key)
        if cached is None:
            cached = (
                to_jax(knots_x, dtype=dtype, device=device),
                to_jax(knots_y, dtype=dtype, device=device),
                to_jax(coefficients, dtype=dtype, device=device),
            )
            cache[key] = cached
        arr_knots_x, arr_knots_y, arr_coefficients = cached

        if _reference_enabled("inference_interpolant"):
            if isinstance(x_arr, jax.core.Tracer) or isinstance(y_arr, jax.core.Tracer):
                raise RuntimeError("Native interpolation cannot run inside jax.jit")
            values = numpy.asarray(
                interp(numpy.asarray(x_arr), numpy.asarray(y_arr), grid=False)
            )
            if bounds_check:
                outside = (
                    (numpy.asarray(x_arr) < knots_x[degree_x])
                    | (numpy.asarray(x_arr) > knots_x[-degree_x - 1])
                    | (numpy.asarray(y_arr) < knots_y[degree_y])
                    | (numpy.asarray(y_arr) > knots_y[-degree_y - 1])
                )
                values = numpy.where(outside, -numpy.inf, values)
            return to_jax(values, device=device)

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
    if _reference_enabled("inference_marginalization"):
        from pycbc.reference_jax import cpu_reference

        reference = _jax_array(sh)
        if reference is None:
            reference = _jax_array(hh)
        if isinstance(reference, jax.core.Tracer):
            raise RuntimeError("Native marginalization cannot run inside jax.jit")
        projected = sh if return_complex else (jnp.abs(sh) if phase else jnp.real(sh))
        interpolated = (
            None if interpolator is None else numpy.asarray(interpolator(projected, hh))
        )
        result = cpu_reference(
            "inference_marginalization",
            numpy.asarray(sh),
            hh=numpy.asarray(hh),
            logw=None if logw is None else numpy.asarray(logw),
            phase=phase,
            distance=distance,
            skip_vector=skip_vector,
            return_peak=return_peak,
            return_complex=return_complex,
            interpolated=interpolated,
        )

        def placed(value):
            return to_jax(value, device=reference.device) if skip_vector else value

        return (
            tuple(placed(v) for v in result)
            if isinstance(result, tuple)
            else placed(result)
        )
    sh_arr = _jax_array(sh)
    hh_arr = _jax_array(hh)
    if sh_arr is None:
        real_dtype = hh_arr.real.dtype
        if numpy.iscomplexobj(sh):
            sh_dtype = jnp.complex128 if real_dtype == jnp.float64 else jnp.complex64
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
            vloglr = jsp.logsumexp(vloglr, b=jnp.exp(logw), axis=0)
        vloglr = vloglr.item()

    if return_peak:
        return vloglr, maxv, maxl
    return vloglr


def batched_project_detector_strain(hp, hc, fp, fc, dt, delta_f, batch_size):
    """Validate a detector-frame batch and reuse its projection boundary."""
    from pycbc.waveform.utils_jax import fused_detector_strain_fd_jax

    hp = _jax_array(hp)
    if hp is None:
        raise TypeError("a JAX-backed waveform is required")
    if hp.ndim != 1 or _to_jax(hc, hp).ndim != 1:
        raise ValueError(
            "batched likelihood waveform parameters must be scalar; only detector-frame extrinsics may be batched"
        )
    device = None if isinstance(hp, jax.core.Tracer) else hp.device
    values = []
    for name, value in (("fplus", fp), ("fcross", fc), ("tc", dt)):
        arr = to_jax(value, dtype=hp.real.dtype, device=device)
        if arr.ndim == 0 or arr.shape == (1,):
            arr = jnp.broadcast_to(arr, (batch_size,))
        elif arr.shape != (batch_size,):
            raise ValueError(
                f"Batched detector-frame parameter {name!r} must be scalar or have length {batch_size}"
            )
        values.append(arr)
    return fused_detector_strain_fd_jax(
        hp, _to_jax(hc, hp), [values[0]], [values[1]], [values[2]], delta_f
    )[0]


def whiten_template(values, weight):
    """Apply the configured original in-place weighting boundary."""
    h_arr = _jax_array(values)
    if h_arr is None:
        raise TypeError("a JAX-backed template is required")
    like = h_arr
    w = _to_jax(weight, h_arr, real=True)
    if _reference_enabled("inference_whitening"):
        from pycbc.reference_jax import cpu_reference

        if isinstance(h_arr, jax.core.Tracer):
            raise RuntimeError("Native whitening cannot run inside jax.jit")
        shape = numpy.broadcast_shapes(h_arr.shape, w.shape)
        rows = jnp.broadcast_to(h_arr, shape).reshape(-1, shape[-1])
        weights = jnp.broadcast_to(w, shape).reshape(-1, shape[-1])
        h_arr = jnp.stack(
            [
                to_jax(
                    cpu_reference(
                        "inference_weight",
                        numpy.asarray(row),
                        weight=numpy.asarray(weight_row),
                    ),
                    device=like.device,
                )
                for row, weight_row in zip(rows, weights)
            ]
        ).reshape(shape)
    else:
        # CPU Array.__imul__ rounds the weighted result back to storage dtype.
        h_arr = (h_arr * w).astype(h_arr.dtype)
    return h_arr
