"""Torch implementations for :mod:`pycbc.inference.models.tools`.

This module is imported lazily when a shared inference helper receives a
Torch-backed value. Keeping the import here avoids making Torch a dependency
of the NumPy inference path.
"""

import numpy
import torch

from pycbc.types.backend import backend_array


def _tensor(value):
    """Return the public Torch backend array for ``value``."""
    return backend_array(value, "torch")


def _numpy_array(value):
    """Return a host array for a non-Torch operand."""
    value = backend_array(value)
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    return numpy.asarray(value)


def inner(left, right):
    """Return the Torch inner product without scalarizing reductions."""
    left = _tensor(left)
    right = _tensor(right)
    dtype = torch.promote_types(left.dtype, right.dtype)
    left = left.to(dtype=dtype)
    right = right.to(device=left.device, dtype=dtype)
    if left.device.type == "mps":
        accumulation_dtype = dtype
    else:
        accumulation_dtype = torch.complex128 if dtype.is_complex else torch.float64

    if left.device.type != "mps":
        try:
            return torch.vdot(
                left.to(dtype=accumulation_dtype).reshape(-1),
                right.to(dtype=accumulation_dtype).reshape(-1),
            )
        except (NotImplementedError, RuntimeError):
            pass

    if dtype.is_complex:
        left_real = torch.view_as_real(left)
        right_real = torch.view_as_real(right)
        real_dtype = (
            torch.float32 if accumulation_dtype == torch.complex64 else torch.float64
        )
        real_part = torch.sum(left_real * right_real, dtype=real_dtype)
        imag_part = torch.sum(
            left_real[..., 0] * right_real[..., 1]
            - left_real[..., 1] * right_real[..., 0],
            dtype=real_dtype,
        )
        return torch.complex(real_part, imag_part)

    return torch.sum(left * right, dtype=accumulation_dtype)


def real_inner(left, right):
    """Return the real Torch inner product, optimizing identical storage."""
    left_tensor = _tensor(left)
    right_tensor = _tensor(right)
    is_same = (
        left is right
        or left_tensor is right_tensor
        or (
            left_tensor.data_ptr() == right_tensor.data_ptr()
            and left_tensor.shape == right_tensor.shape
            and left_tensor.stride() == right_tensor.stride()
        )
    )
    if not is_same:
        return inner(left, right).real

    if left_tensor.device.type == "mps":
        accumulation_dtype = (
            left_tensor.real.dtype if left_tensor.is_complex() else left_tensor.dtype
        )
    else:
        accumulation_dtype = torch.float64
    if left_tensor.is_complex():
        summand = torch.view_as_real(left_tensor).square()
    else:
        summand = left_tensor.square()
    return torch.sum(summand, dtype=accumulation_dtype)


def _to_tensor(value, like, *, real=False):
    """Convert one mixed-backend operand to the device of ``like``."""
    tensor = _tensor(value)
    if tensor is not None:
        return tensor
    value = _numpy_array(value)
    if real:
        dtype = like.real.dtype
    else:
        dtype = torch.complex128 if numpy.iscomplexobj(value) else torch.float64
    return torch.as_tensor(value, device=like.device, dtype=dtype)


def fused_inner_hd_hh(h, d, weight=None):
    """Compute Torch ``<h|d>`` and ``<h|h>`` in a single pass."""
    h_tensor = _tensor(h)
    d_tensor = _tensor(d)
    like = h_tensor if h_tensor is not None else d_tensor
    h_tensor = _to_tensor(h, like)
    d_tensor = _to_tensor(d, like)
    if weight is not None:
        h_tensor = h_tensor * _to_tensor(weight, h_tensor, real=True)

    dtype = torch.promote_types(h_tensor.dtype, d_tensor.dtype)
    h_tensor = h_tensor.to(dtype=dtype)
    d_tensor = d_tensor.to(device=h_tensor.device, dtype=dtype)
    is_batched = h_tensor.ndim > 1 or d_tensor.ndim > 1

    if is_batched:
        if h_tensor.device.type == "mps":
            real_acc_dtype = h_tensor.real.dtype if dtype.is_complex else dtype
            accumulation_dtype = dtype
        else:
            real_acc_dtype = torch.float64
            accumulation_dtype = torch.complex128 if dtype.is_complex else torch.float64
        h_acc = h_tensor.to(dtype=accumulation_dtype)
        d_acc = d_tensor.to(dtype=accumulation_dtype)
        if dtype.is_complex:
            h_real = torch.view_as_real(h_acc)
            d_real = torch.view_as_real(d_acc)
            real_hd = torch.sum(
                h_real[..., 0] * d_real[..., 0] + h_real[..., 1] * d_real[..., 1],
                dim=-1,
                dtype=real_acc_dtype,
            )
            imag_hd = torch.sum(
                h_real[..., 0] * d_real[..., 1] - h_real[..., 1] * d_real[..., 0],
                dim=-1,
                dtype=real_acc_dtype,
            )
            cplx_hd = torch.complex(real_hd, imag_hd)
            hh = torch.sum(
                h_real[..., 0].square() + h_real[..., 1].square(),
                dim=-1,
                dtype=real_acc_dtype,
            )
        else:
            cplx_hd = torch.sum(h_acc * d_acc, dim=-1, dtype=accumulation_dtype)
            hh = torch.sum(h_acc.square(), dim=-1, dtype=real_acc_dtype)
        return cplx_hd, hh

    if h_tensor.device.type == "mps":
        if dtype.is_complex:
            h_real = torch.view_as_real(h_tensor)
            d_real = torch.view_as_real(d_tensor)
            real_dtype = h_tensor.real.dtype
            real_hd = torch.sum(
                h_real[..., 0] * d_real[..., 0] + h_real[..., 1] * d_real[..., 1],
                dtype=real_dtype,
            )
            imag_hd = torch.sum(
                h_real[..., 0] * d_real[..., 1] - h_real[..., 1] * d_real[..., 0],
                dtype=real_dtype,
            )
            cplx_hd = torch.complex(real_hd, imag_hd)
            hh = torch.sum(h_real.square(), dtype=real_dtype)
        else:
            cplx_hd = torch.sum(h_tensor * d_tensor, dtype=h_tensor.dtype)
            hh = torch.sum(h_tensor.square(), dtype=h_tensor.dtype)
        return cplx_hd, hh

    accumulation_dtype = torch.complex128 if dtype.is_complex else torch.float64
    h_acc = h_tensor.to(dtype=accumulation_dtype).reshape(-1)
    d_acc = d_tensor.to(dtype=accumulation_dtype, device=h_tensor.device).reshape(-1)
    if dtype.is_complex:
        try:
            cplx_hd = torch.vdot(h_acc, d_acc)
        except (NotImplementedError, RuntimeError):
            h_real = torch.view_as_real(h_acc)
            d_real = torch.view_as_real(d_acc)
            real_hd = torch.sum(
                h_real[..., 0] * d_real[..., 0] + h_real[..., 1] * d_real[..., 1],
                dtype=torch.float64,
            )
            imag_hd = torch.sum(
                h_real[..., 0] * d_real[..., 1] - h_real[..., 1] * d_real[..., 0],
                dtype=torch.float64,
            )
            cplx_hd = torch.complex(real_hd, imag_hd)
        hh = torch.sum(torch.view_as_real(h_acc).square(), dtype=torch.float64)
    else:
        cplx_hd = torch.dot(h_acc, d_acc)
        hh = torch.sum(h_acc.square(), dtype=torch.float64)
    return cplx_hd, hh


def selected_values(values, indices, *, host=True):
    """Select values using device-resident integer indices."""
    tensor = _tensor(values)
    indices = torch.as_tensor(indices, device=tensor.device, dtype=torch.int64)
    selected = tensor[indices]
    return selected.detach().cpu().numpy() if host else selected


def add_values(total, values):
    """Add values on the device of the Torch-backed operand."""
    total_tensor = _tensor(total)
    values_tensor = _tensor(values)
    template = total_tensor if total_tensor is not None else values_tensor
    if total_tensor is None:
        total_tensor = torch.as_tensor(
            total, device=template.device, dtype=template.dtype
        )
    if values_tensor is None:
        values_tensor = torch.as_tensor(
            values, device=template.device, dtype=template.dtype
        )
    return total_tensor + values_tensor


def last_index_at_or_below(values, upper):
    """Return the last sorted-value index at or below ``upper``."""
    tensor = _tensor(values)
    insertion = torch.searchsorted(tensor, tensor.new_tensor(upper), right=True).item()
    if insertion == 0:
        raise IndexError(f"no values are at or below {upper}")
    return int(insertion - 1)


def threshold_extent(values, threshold):
    """Return the first and last Torch indices above ``threshold``."""
    indices = torch.nonzero(
        torch.abs(_tensor(values)) > threshold, as_tuple=False
    ).flatten()
    return int(indices[0].item()), int(indices[-1].item())


def _weighted_cdf(loglr):
    tensor = _tensor(loglr)
    cdf = torch.exp(tensor - tensor.max()).cumsum(dim=0)
    return tensor, cdf / cdf[-1]


def draw_sample(loglr, size=None, *, host=True):
    """Draw weighted indices without unnecessary device transfers."""
    tensor, cdf = _weighted_cdf(loglr)
    if size and not host:
        uniforms = torch.rand(size, device=tensor.device, dtype=tensor.dtype)
    else:
        uniforms = numpy.random.uniform(size=size) if size else numpy.random.uniform()
        uniforms = torch.as_tensor(uniforms, device=tensor.device, dtype=tensor.dtype)
    indices = torch.searchsorted(cdf, uniforms)
    if indices.ndim == 0:
        return int(indices.item())
    return indices.detach().cpu().numpy() if host else indices


def draw_device_sample_with_host_rng(loglr, size):
    """Draw device indices while retaining NumPy's RNG stream."""
    tensor, cdf = _weighted_cdf(loglr)
    uniforms = torch.as_tensor(
        numpy.random.uniform(size=size),
        device=tensor.device,
        dtype=tensor.dtype,
    )
    return torch.searchsorted(cdf, uniforms)


def device_index_matrix_to_host(indices):
    """Materialize a matrix of device indices at one host boundary."""
    return torch.stack(indices, dim=0).detach().cpu().numpy()


def same_device(values):
    """Return whether all Torch-backed values share one device."""
    tensors = [_tensor(value) for value in values]
    return all(tensor.device == tensors[0].device for tensor in tensors[1:])


def weighted_loglr(values, weights):
    """Add probability weights on the likelihood device."""
    tensor = _tensor(values)
    weights = torch.as_tensor(weights, device=tensor.device, dtype=tensor.real.dtype)
    return tensor + torch.log(weights)


def phase_reconstruction_values(sh, hh, sample_count=int(1e4)):
    """Build the phase-reconstruction grid on its input device."""
    sh_tensor = _tensor(sh)
    hh_tensor = _tensor(hh)
    like = sh_tensor if sh_tensor is not None else hh_tensor
    real_dtype = like.real.dtype
    sh = torch.as_tensor(
        sh_tensor if sh_tensor is not None else sh,
        device=like.device,
        dtype=(torch.complex128 if real_dtype == torch.float64 else torch.complex64),
    )
    hh = torch.as_tensor(
        hh_tensor if hh_tensor is not None else hh,
        device=like.device,
        dtype=real_dtype,
    )
    phase = torch.linspace(
        0,
        2.0 * numpy.pi,
        sample_count,
        device=like.device,
        dtype=real_dtype,
    )
    angle = -2.0 * phase
    phase_factor = torch.complex(torch.cos(angle), torch.sin(angle))
    return phase, (phase_factor * sh).real + hh


def selected_scalar(values, index):
    """Return one selected Torch value at the public scalar boundary."""
    return _tensor(values)[index].item()


def random_permutation(values, size):
    """Return device and host views of one random permutation."""
    tensor = _tensor(values)
    choice = torch.randperm(len(values), device=tensor.device)[:size]
    return choice, choice.detach().cpu().numpy()


def normalize_logweights(values):
    """Normalize device-resident log weights."""
    tensor = _tensor(values)
    return tensor - torch.logsumexp(tensor, dim=0)


def host_indices(values):
    """Materialize device indices for public host calculations."""
    return _tensor(values).detach().cpu().numpy()


def _bspline_basis(values, knots, degree):
    """Return active B-spline coefficient indices and basis values."""
    coefficient_count = knots.numel() - degree - 1
    spans = torch.searchsorted(knots, values.contiguous(), right=True).sub(1)
    spans = spans.clamp(degree, coefficient_count - 1)

    basis = torch.ones(values.shape + (1,), device=values.device, dtype=values.dtype)
    left = [None] * (degree + 1)
    right = [None] * (degree + 1)
    for column in range(1, degree + 1):
        left[column] = values - knots[spans + 1 - column]
        right[column] = knots[spans + column] - values
        saved = torch.zeros_like(values)
        updated = []
        for row in range(column):
            weight = basis[..., row] / (right[row + 1] + left[column - row])
            updated.append(saved + right[row + 1] * weight)
            saved = left[column - row] * weight
        updated.append(saved)
        basis = torch.stack(updated, dim=-1)

    offsets = torch.arange(degree + 1, device=values.device, dtype=torch.int64)
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
        x_tensor = _tensor(x)
        y_tensor = _tensor(y)
        like = x_tensor if x_tensor is not None else y_tensor
        dtype = like.real.dtype
        if x_tensor is not None:
            dtype = torch.promote_types(dtype, x_tensor.real.dtype)
        if y_tensor is not None:
            dtype = torch.promote_types(dtype, y_tensor.real.dtype)
        x_tensor = torch.as_tensor(
            x_tensor if x_tensor is not None else x,
            device=like.device,
            dtype=dtype,
        )
        y_tensor = torch.as_tensor(
            y_tensor if y_tensor is not None else y,
            device=like.device,
            dtype=dtype,
        )
        x_tensor, y_tensor = torch.broadcast_tensors(x_tensor, y_tensor)

        key = (like.device.type, like.device.index, dtype)
        cached = cache.get(key)
        if cached is None:
            cached = (
                torch.as_tensor(knots_x, device=like.device, dtype=dtype),
                torch.as_tensor(knots_y, device=like.device, dtype=dtype),
                torch.as_tensor(coefficients, device=like.device, dtype=dtype),
            )
            cache[key] = cached
        tensor_knots_x, tensor_knots_y, tensor_coefficients = cached

        indices_x, basis_x = _bspline_basis(x_tensor, tensor_knots_x, degree_x)
        indices_y, basis_y = _bspline_basis(y_tensor, tensor_knots_y, degree_y)
        local_coefficients = tensor_coefficients[
            indices_x[..., :, None], indices_y[..., None, :]
        ]
        values = (
            local_coefficients * basis_x[..., :, None] * basis_y[..., None, :]
        ).sum(dim=(-2, -1))

        if bounds_check:
            outside = (
                (x_tensor < knots_x[degree_x])
                | (x_tensor > knots_x[-degree_x - 1])
                | (y_tensor < knots_y[degree_y])
                | (y_tensor > knots_y[-degree_y - 1])
            )
            values = torch.where(outside, values.new_full((), -torch.inf), values)
        return values

    return evaluate


def numpy_from_backend(value):
    """Return a detached CPU value for a NumPy-only calculation."""
    tensor = _tensor(value)
    if tensor is None:
        return value
    tensor = tensor.detach()
    if tensor.is_conj():
        tensor = tensor.resolve_conj()
    if tensor.ndim == 0:
        return tensor.item()
    return tensor.cpu().numpy()


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
    """Torch implementation of explicit likelihood marginalizations."""
    sh_tensor = _tensor(sh)
    hh_tensor = _tensor(hh)
    like = sh_tensor if sh_tensor is not None else hh_tensor
    if sh_tensor is None:
        real_dtype = hh_tensor.real.dtype
        if numpy.iscomplexobj(sh):
            sh_dtype = (
                torch.complex128 if real_dtype == torch.float64 else torch.complex64
            )
        else:
            sh_dtype = real_dtype
        sh = torch.as_tensor(sh, device=like.device, dtype=sh_dtype)
    else:
        sh = sh_tensor
    hh = torch.as_tensor(
        hh_tensor if hh_tensor is not None else hh,
        device=like.device,
        dtype=sh.real.dtype,
    )

    if distance and interpolator is None and sh.ndim:
        raise ValueError(
            "Cannot do vector marginalization and distance at the same time"
        )

    if return_complex:
        pass
    elif phase:
        sh = torch.abs(sh)
    else:
        sh = sh.real

    if distance and interpolator is None:
        dist_rescale, dist_weights = distance
        dist_rescale = torch.as_tensor(dist_rescale, device=like.device, dtype=hh.dtype)
        dist_weights = torch.as_tensor(dist_weights, device=like.device, dtype=hh.dtype)
        sh = sh * dist_rescale
        hh = hh * dist_rescale.square()
        logw = torch.log(dist_weights)

    if return_complex and interpolator is None:
        return sh, -0.5 * hh

    if interpolator is not None:
        vloglr = interpolator(sh, hh)
    else:
        if phase:
            sh = torch.log(torch.special.i0e(sh)) + sh
        vloglr = sh - 0.5 * hh
    if return_peak:
        if vloglr.ndim:
            maxv = int(vloglr.argmax().item())
            maxl = vloglr[maxv].item()
        else:
            maxv = 0
            maxl = vloglr.item()

    if not skip_vector:
        if vloglr.ndim:
            if logw is None:
                logw = -torch.log(vloglr.new_tensor(vloglr.shape[0]))
            else:
                logw_tensor = _tensor(logw)
                if logw_tensor is not None:
                    logw = logw_tensor
                logw = torch.as_tensor(logw, device=vloglr.device, dtype=vloglr.dtype)
            vloglr = torch.logsumexp(vloglr + logw, dim=0)
        vloglr = vloglr.item()

    if return_peak:
        return vloglr, maxv, maxl
    return vloglr
