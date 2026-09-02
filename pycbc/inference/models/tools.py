""" Common utility functions for calculation of likelihoods
"""

import logging
import warnings
from distutils.util import strtobool

import numpy
import numpy.random
import tqdm

from scipy.special import logsumexp, i0e
from scipy.interpolate import RectBivariateSpline, interp1d
from pycbc.distributions import JointDistribution

from pycbc.detector import Detector


# Earth radius in seconds
EARTH_RADIUS = 0.031


def _torch_tensor(value):
    """Return the tensor backing a Torch/PyCBC value, if present."""
    if type(value).__module__.split('.', 1)[0] == 'torch':
        return value
    tensor = getattr(value, 'tensor', None)
    if tensor is None:
        tensor = getattr(getattr(value, '_data', None), 'tensor', None)
    return tensor


def _inner(left, right):
    """Return an inner product without scalarizing Torch reductions."""
    left_tensor = _torch_tensor(left)
    right_tensor = _torch_tensor(right)
    if left_tensor is None or right_tensor is None:
        return left.inner(right)

    import torch

    dtype = torch.promote_types(left_tensor.dtype, right_tensor.dtype)
    left_tensor = left_tensor.to(dtype=dtype)
    right_tensor = right_tensor.to(
        device=left_tensor.device, dtype=dtype
    )
    if left_tensor.device.type == 'mps':
        accumulation_dtype = dtype
    else:
        accumulation_dtype = (
            torch.complex128 if dtype.is_complex else torch.float64
        )

    if left_tensor.device.type != 'mps':
        try:
            return torch.vdot(
                left_tensor.to(dtype=accumulation_dtype).reshape(-1),
                right_tensor.to(dtype=accumulation_dtype).reshape(-1),
            )
        except (NotImplementedError, RuntimeError):
            pass

    if dtype.is_complex:
        left_real = torch.view_as_real(left_tensor)
        right_real = torch.view_as_real(right_tensor)
        real_dtype = (
            torch.float32
            if accumulation_dtype == torch.complex64
            else torch.float64
        )
        real_part = torch.sum(left_real * right_real, dtype=real_dtype)
        imag_part = torch.sum(
            left_real[..., 0] * right_real[..., 1]
            - left_real[..., 1] * right_real[..., 0],
            dtype=real_dtype,
        )
        return torch.complex(real_part, imag_part)

    return torch.sum(
        left_tensor * right_tensor,
        dtype=accumulation_dtype,
    )


def _real_inner(left, right):
    """Return a real inner product without scalarizing Torch reductions."""
    left_tensor = _torch_tensor(left)
    right_tensor = _torch_tensor(right)
    if left_tensor is None or right_tensor is None:
        return _inner(left, right).real

    is_same = (
        left is right
        or (
            hasattr(left, '_data')
            and hasattr(right, '_data')
            and left._data is right._data
        )
        or left_tensor is right_tensor
        or (
            left_tensor.data_ptr() == right_tensor.data_ptr()
            and left_tensor.shape == right_tensor.shape
            and left_tensor.stride() == right_tensor.stride()
        )
    )
    if is_same:
        import torch

        if left_tensor.device.type == 'mps':
            accumulation_dtype = (
                left_tensor.real.dtype
                if left_tensor.is_complex()
                else left_tensor.dtype
            )
        else:
            accumulation_dtype = torch.float64

        if left_tensor.is_complex():
            summand = torch.view_as_real(left_tensor).square()
        else:
            summand = left_tensor.square()
        return torch.sum(summand, dtype=accumulation_dtype)

    return _inner(left, right).real


def _fused_inner_hd_hh(h, d, weight=None):
    """Compute <h|d> and <h|h> in a single pass with minimal allocations.

    If ``weight`` is provided, evaluates inner products with respect to
    the whitened waveform ``hw = h * weight`` without modifying ``h`` in-place
    or performing separate memory passes.

    Parameters
    ----------
    h : FrequencySeries, Array, numpy.ndarray, or torch.Tensor
        The waveform frequency series.
    d : FrequencySeries, Array, numpy.ndarray, or torch.Tensor
        The whitened data frequency series.
    weight : FrequencySeries, Array, numpy.ndarray, or torch.Tensor, optional
        The inner-product weighting factor.

    Returns
    -------
    cplx_hd : complex float or torch.Tensor
        The complex inner product <h|d>.
    hh : float or torch.Tensor
        The real inner product <h|h>.
    """
    h_tensor = _torch_tensor(h)
    d_tensor = _torch_tensor(d)
    w_tensor = _torch_tensor(weight) if weight is not None else None

    if h_tensor is not None or d_tensor is not None:
        import torch

        like = h_tensor if h_tensor is not None else d_tensor
        device = like.device

        if h_tensor is None:
            h_arr = getattr(h, 'data', getattr(h, '_data', h))
            if hasattr(h_arr, 'numpy') and callable(h_arr.numpy):
                h_arr = h_arr.numpy()
            else:
                h_arr = numpy.asarray(h_arr)
            dtype = (
                torch.complex128
                if numpy.iscomplexobj(h_arr)
                else torch.float64
            )
            h_tensor = torch.as_tensor(h_arr, device=device, dtype=dtype)

        if d_tensor is None:
            d_arr = getattr(d, 'data', getattr(d, '_data', d))
            if hasattr(d_arr, 'numpy') and callable(d_arr.numpy):
                d_arr = d_arr.numpy()
            else:
                d_arr = numpy.asarray(d_arr)
            dtype = (
                torch.complex128
                if numpy.iscomplexobj(d_arr)
                else torch.float64
            )
            d_tensor = torch.as_tensor(d_arr, device=device, dtype=dtype)

        if w_tensor is None and weight is not None:
            w_arr = getattr(weight, 'data', getattr(weight, '_data', weight))
            if hasattr(w_arr, 'numpy') and callable(w_arr.numpy):
                w_arr = w_arr.numpy()
            else:
                w_arr = numpy.asarray(w_arr)
            w_tensor = torch.as_tensor(
                w_arr, device=device, dtype=h_tensor.real.dtype
            )

        if w_tensor is not None:
            hw = h_tensor * w_tensor
        else:
            hw = h_tensor

        dtype = torch.promote_types(hw.dtype, d_tensor.dtype)
        hw = hw.to(dtype=dtype)
        d_tensor = d_tensor.to(device=hw.device, dtype=dtype)

        is_batched = hw.ndim > 1 or d_tensor.ndim > 1

        if is_batched:
            if hw.device.type == 'mps':
                real_acc_dtype = hw.real.dtype if dtype.is_complex else dtype
                accumulation_dtype = dtype
            else:
                real_acc_dtype = torch.float64
                accumulation_dtype = (
                    torch.complex128 if dtype.is_complex else torch.float64
                )

            if dtype.is_complex:
                hw_acc = hw.to(dtype=accumulation_dtype)
                d_acc = d_tensor.to(dtype=accumulation_dtype)
                hw_real = torch.view_as_real(hw_acc)
                d_real = torch.view_as_real(d_acc)
                real_hd = torch.sum(
                    hw_real[..., 0] * d_real[..., 0]
                    + hw_real[..., 1] * d_real[..., 1],
                    dim=-1,
                    dtype=real_acc_dtype,
                )
                imag_hd = torch.sum(
                    hw_real[..., 0] * d_real[..., 1]
                    - hw_real[..., 1] * d_real[..., 0],
                    dim=-1,
                    dtype=real_acc_dtype,
                )
                cplx_hd = torch.complex(real_hd, imag_hd)
                hh = torch.sum(
                    hw_real[..., 0].square() + hw_real[..., 1].square(),
                    dim=-1,
                    dtype=real_acc_dtype,
                )
            else:
                hw_acc = hw.to(dtype=accumulation_dtype)
                d_acc = d_tensor.to(dtype=accumulation_dtype)
                cplx_hd = torch.sum(hw_acc * d_acc, dim=-1, dtype=accumulation_dtype)
                hh = torch.sum(hw_acc.square(), dim=-1, dtype=real_acc_dtype)
            return cplx_hd, hh

        if hw.device.type == 'mps':
            if dtype.is_complex:
                hw_real = torch.view_as_real(hw)
                d_real = torch.view_as_real(d_tensor)
                real_dtype = hw.real.dtype
                real_hd = torch.sum(
                    hw_real[..., 0] * d_real[..., 0]
                    + hw_real[..., 1] * d_real[..., 1],
                    dtype=real_dtype,
                )
                imag_hd = torch.sum(
                    hw_real[..., 0] * d_real[..., 1]
                    - hw_real[..., 1] * d_real[..., 0],
                    dtype=real_dtype,
                )
                cplx_hd = torch.complex(real_hd, imag_hd)
                hh = torch.sum(hw_real.square(), dtype=real_dtype)
            else:
                cplx_hd = torch.sum(hw * d_tensor, dtype=hw.dtype)
                hh = torch.sum(hw.square(), dtype=hw.dtype)
            return cplx_hd, hh

        accumulation_dtype = (
            torch.complex128 if dtype.is_complex else torch.float64
        )
        real_acc_dtype = torch.float64
        hw_acc = hw.to(dtype=accumulation_dtype).reshape(-1)
        d_acc = d_tensor.to(
            dtype=accumulation_dtype, device=hw.device
        ).reshape(-1)

        if dtype.is_complex:
            try:
                cplx_hd = torch.vdot(hw_acc, d_acc)
            except (NotImplementedError, RuntimeError):
                hw_real = torch.view_as_real(hw_acc)
                d_real = torch.view_as_real(d_acc)
                real_hd = torch.sum(
                    hw_real[..., 0] * d_real[..., 0]
                    + hw_real[..., 1] * d_real[..., 1],
                    dtype=real_acc_dtype,
                )
                imag_hd = torch.sum(
                    hw_real[..., 0] * d_real[..., 1]
                    - hw_real[..., 1] * d_real[..., 0],
                    dtype=real_acc_dtype,
                )
                cplx_hd = torch.complex(real_hd, imag_hd)
            hh = torch.sum(
                torch.view_as_real(hw_acc).square(),
                dtype=real_acc_dtype,
            )
        else:
            cplx_hd = torch.dot(hw_acc, d_acc)
            hh = torch.sum(hw_acc.square(), dtype=real_acc_dtype)
        return cplx_hd, hh

    def _unwrap_arr(v):
        if hasattr(v, 'numpy') and callable(v.numpy):
            return numpy.asarray(v.numpy())
        if hasattr(v, 'data') and not isinstance(v.data, memoryview):
            return numpy.asarray(v.data)
        return numpy.asarray(v)

    h_arr = _unwrap_arr(h)
    d_arr = _unwrap_arr(d)
    if weight is not None:
        w_arr = _unwrap_arr(weight)
        hw = h_arr * w_arr
    else:
        hw = h_arr

    if hw.size == 0:
        if hw.ndim > 1:
            return (numpy.zeros(hw.shape[:-1], dtype=complex),
                    numpy.zeros(hw.shape[:-1], dtype=float))
        return 0j, 0.0

    if hw.ndim > 1 or d_arr.ndim > 1:
        if numpy.iscomplexobj(hw) or numpy.iscomplexobj(d_arr):
            hw_real = hw.real
            hw_imag = hw.imag
            d_real = d_arr.real
            d_imag = d_arr.imag
            real_hd = numpy.sum(hw_real * d_real + hw_imag * d_imag, axis=-1)
            imag_hd = numpy.sum(hw_real * d_imag - hw_imag * d_real, axis=-1)
            cplx_hd = real_hd + 1j * imag_hd
            hh = numpy.sum(hw_real ** 2 + hw_imag ** 2, axis=-1)
        else:
            cplx_hd = numpy.sum(hw * d_arr, axis=-1)
            hh = numpy.sum(hw ** 2, axis=-1)
        return cplx_hd, hh

    if numpy.iscomplexobj(hw) or numpy.iscomplexobj(d_arr):
        hw_real = hw.real
        hw_imag = hw.imag
        d_real = d_arr.real
        d_imag = d_arr.imag
        real_hd = float(
            numpy.dot(hw_real, d_real) + numpy.dot(hw_imag, d_imag)
        )
        imag_hd = float(
            numpy.dot(hw_real, d_imag) - numpy.dot(hw_imag, d_real)
        )
        cplx_hd = complex(real_hd, imag_hd)
        hh = float(
            numpy.dot(hw_real, hw_real) + numpy.dot(hw_imag, hw_imag)
        )
    else:
        cplx_hd = float(numpy.dot(hw, d_arr))
        hh = float(numpy.dot(hw, hw))
    return cplx_hd, hh


def _squared_norm_values(snr):
    """Return SNR-squared values without moving Torch series to the host."""
    values = snr.squared_norm()
    tensor = _torch_tensor(values)
    return tensor if tensor is not None else values.numpy()


def _selected_values(values, indices, *, host=True):
    """Return selected proposal values, optionally preserving Torch storage."""
    tensor = _torch_tensor(values)
    if tensor is None:
        return values[indices]

    import torch

    indices = torch.as_tensor(indices, device=tensor.device, dtype=torch.int64)
    selected = tensor[indices]
    if host:
        return selected.detach().cpu().numpy()
    return selected


def _add_values(total, values):
    """Add proposal values without moving a Torch operand to the host."""
    if total is None:
        return values

    total_tensor = _torch_tensor(total)
    values_tensor = _torch_tensor(values)
    if total_tensor is None and values_tensor is None:
        return total + values

    import torch

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


def _last_index_at_or_below(values, upper):
    """Return the last sorted-value index at or below ``upper``."""
    tensor = _torch_tensor(values)
    if tensor is not None:
        import torch

        insertion = torch.searchsorted(
            tensor, tensor.new_tensor(upper), right=True
        ).item()
    else:
        if hasattr(values, "numpy"):
            values = values.numpy()
        insertion = numpy.searchsorted(values, upper, side="right")

    if insertion == 0:
        raise IndexError(f"no values are at or below {upper}")
    return int(insertion - 1)


def _threshold_extent(values, threshold):
    """Return the first and last indices above ``threshold``."""
    tensor = _torch_tensor(values)
    if tensor is not None:
        import torch

        indices = torch.nonzero(
            torch.abs(tensor) > threshold, as_tuple=False
        ).flatten()
        return int(indices[0].item()), int(indices[-1].item())

    indices = numpy.flatnonzero(abs(values) > threshold)
    return int(indices[0]), int(indices[-1])


def str_to_tuple(sval, ftype):
    """ Convenience parsing to convert str to tuple"""
    if sval is None:
        return ()
    return tuple(ftype(x.strip(' ')) for x in sval.split(','))


def str_to_bool(sval):
    """ Ensure value is a bool if it can be converted """
    if isinstance(sval, str):
        return strtobool(sval)
    return sval


def draw_sample(loglr, size=None, *, host=True):
    """Draw a random index from a 1-d vector with loglr weights.

    Vector draws from Torch inputs may be kept on their input device by
    setting ``host=False``. These device-resident vector draws use Torch's
    active device RNG; host-returning and scalar draws retain NumPy's RNG and
    the public return types.
    """
    tensor = _torch_tensor(loglr)
    if tensor is not None:
        import torch

        cdf = torch.exp(tensor - tensor.max()).cumsum(dim=0)
        cdf = cdf / cdf[-1]
        if size and not host:
            x = torch.rand(
                size, device=tensor.device, dtype=tensor.dtype
            )
        else:
            if size:
                x = numpy.random.uniform(size=size)
            else:
                x = numpy.random.uniform()
            x = torch.as_tensor(
                x, device=tensor.device, dtype=tensor.dtype
            )
        xl = torch.searchsorted(cdf, x)
        if xl.ndim == 0:
            return int(xl.item())
        if host:
            return xl.detach().cpu().numpy()
        return xl

    if size:
        x = numpy.random.uniform(size=size)
    else:
        x = numpy.random.uniform()
    loglr = loglr - loglr.max()
    cdf = numpy.exp(loglr).cumsum()
    cdf /= cdf[-1]
    xl = numpy.searchsorted(cdf, x)
    return xl


def _draw_device_sample_with_host_rng(loglr, size):
    """Draw device indices while retaining NumPy's RNG stream.

    Sky/time marginalization historically uses NumPy to generate its random
    uniforms even when the proposal weights live on a Torch device. Keeping
    the uniforms on that stream preserves seeded public results while the
    returned indices can remain device-resident for subsequent gathers.
    """
    tensor = _torch_tensor(loglr)
    if tensor is None:
        raise TypeError("device sampling requires Torch-backed weights")

    import torch

    cdf = torch.exp(tensor - tensor.max()).cumsum(dim=0)
    cdf = cdf / cdf[-1]
    uniforms = numpy.random.uniform(size=size)
    uniforms = torch.as_tensor(
        uniforms, device=tensor.device, dtype=tensor.dtype
    )
    return torch.searchsorted(cdf, uniforms)


def _device_index_matrix_to_host(indices):
    """Materialize device index vectors at one explicit host boundary."""
    import torch

    return torch.stack(indices, dim=0).detach().cpu().numpy()


def _draw_sky_time_indices(weights, offsets, size):
    """Draw detector indices and return their host-relative delays.

    Same-device Torch weights keep each draw on device through proposal
    gathers and integer detector-offset arithmetic. The complete index matrix
    then crosses to the host once for the existing sky-delay dictionary and
    GPS-time construction. NumPy, mixed-backend, and mixed-device inputs keep
    the legacy host-index behavior.
    """
    if len(weights) != len(offsets) or not weights:
        raise ValueError("weights and offsets must have the same nonzero size")

    tensors = [_torch_tensor(weight) for weight in weights]
    use_device_indices = (
        bool(size)
        and all(tensor is not None for tensor in tensors)
        and all(
            tensor.device == tensors[0].device for tensor in tensors[1:]
        )
    )

    indices = []
    selected_weights = []
    for weight, offset in zip(weights, offsets):
        if use_device_indices:
            index = _draw_device_sample_with_host_rng(weight, size)
        else:
            index = draw_sample(weight, size=size)
        selected_weights.append(
            _selected_values(weight, index, host=False)
        )
        indices.append(index + offset)

    if use_device_indices:
        host_indices = _device_index_matrix_to_host(indices)
    else:
        host_indices = numpy.stack(indices, axis=0)

    reference = host_indices[0]
    relative = [reference - index for index in host_indices[1:]]
    return reference, relative, selected_weights


def _weighted_loglr(values, weights):
    """Add probability weights without moving Torch values to the host."""
    tensor = _torch_tensor(values)
    if tensor is None:
        return values + numpy.log(weights)

    import torch

    weights = torch.as_tensor(
        weights, device=tensor.device, dtype=tensor.real.dtype
    )
    return tensor + torch.log(weights)


def _phase_reconstruction_values(sh, hh, sample_count=int(1e4)):
    """Build the phase-reconstruction likelihood grid on its input device."""
    sh_tensor = _torch_tensor(sh)
    hh_tensor = _torch_tensor(hh)
    like = sh_tensor if sh_tensor is not None else hh_tensor
    if like is None:
        phase = numpy.linspace(0, numpy.pi * 2.0, sample_count)
        loglr = (numpy.exp(-2.0j * phase) * sh).real + hh
        return phase, loglr

    import torch

    real_dtype = like.real.dtype
    sh = torch.as_tensor(
        sh_tensor if sh_tensor is not None else sh,
        device=like.device,
        dtype=(
            torch.complex128
            if real_dtype == torch.float64
            else torch.complex64
        ),
    )
    hh = torch.as_tensor(
        hh_tensor if hh_tensor is not None else hh,
        device=like.device,
        dtype=real_dtype,
    )
    phase = torch.linspace(
        0, 2.0 * numpy.pi, sample_count,
        device=like.device, dtype=real_dtype,
    )
    angle = -2.0 * phase
    phase_factor = torch.complex(torch.cos(angle), torch.sin(angle))
    return phase, (phase_factor * sh).real + hh


def _selected_scalar(values, index):
    """Return one selected value at the public scalar boundary."""
    tensor = _torch_tensor(values)
    if tensor is not None:
        return tensor[index].item()
    return values[index]


class DistMarg():
    """Help class to add bookkeeping for likelihood marginalization"""

    def setup_marginalization(self,
                              variable_params,
                              marginalize_phase=False,
                              marginalize_distance=False,
                              marginalize_distance_param='distance',
                              marginalize_distance_samples=int(1e4),
                              marginalize_distance_interpolator=False,
                              marginalize_distance_snr_range=None,
                              marginalize_distance_density=None,
                              marginalize_vector_params=None,
                              marginalize_vector_samples=1e3,
                              marginalize_sky_initial_samples=1e6,
                              **kwargs):
        """ Setup the model for use with distance marginalization

        This function sets up precalculations for distance / phase
        marginalization. For distance margininalization it modifies the
        model to internally remove distance as a parameter.

        Parameters
        ----------
        variable_params: list of strings
            The set of variable parameters
        marginalize_phase: bool, False
            Do analytic marginalization (appopriate only for 22 mode waveforms)
        marginalize_distance: bool, False
            Marginalize over distance
        marginalize_distance_param: str
            Name of the parameter that is used to determine the distance.
            This might be 'distance' or a parameter which can be converted
            to distance by a provided univariate transformation.
        marginalize_distance_interpolator: bool
            Use a pre-calculated interpolating function for the distance
            marginalized likelihood.
        marginalize_distance_snr_range: tuple of floats, (1, 50)
            The SNR range for the interpolating function to be defined in.
            If a sampler goes outside this range, the logl will be returned
            as -numpy.inf.
        marginalize_distance_density: tuple of intes, (1000, 1000)
            The dimensions of the interpolation grid over (sh, hh).

        Returns
        -------
        variable_params: list of strings
            Set of variable params (missing distance-related parameter).
        kwags: dict
            The keyword arguments to the model initialization, may be modified
            from the original set by this function.
        """
        def pop_prior(param):
            variable_params.remove(param)
            old_prior = kwargs['prior']

            dists = [d for d in old_prior.distributions
                     if param not in d.params]
            dprior = [d for d in old_prior.distributions
                      if param in d.params][0]
            prior = JointDistribution(variable_params,
                                      *dists, **old_prior.kwargs)
            kwargs['prior'] = prior
            return dprior

        self.reconstruct_phase = False
        self.reconstruct_distance = False
        self.reconstruct_vector = False
        self.precalc_antenna_factors = False

        # Handle any requested parameter vector / brute force marginalizations
        self.marginalize_vector_params = {}
        self.marginalized_vector_priors = {}
        self.vsamples = int(marginalize_vector_samples)

        self.marginalize_sky_initial_samples = \
            int(float(marginalize_sky_initial_samples))

        for param in str_to_tuple(marginalize_vector_params, str):
            logging.info('Marginalizing over %s, %s points from prior',
                         param, self.vsamples)
            self.marginalized_vector_priors[param] = pop_prior(param)

        # Remove in the future, backwards compatibility
        if 'polarization_samples' in kwargs:
            warnings.warn("use marginalize_vector_samples rather "
                          "than 'polarization_samples'", DeprecationWarning)
            pol_uniform = numpy.linspace(0, numpy.pi * 2.0, self.vsamples)
            self.marginalize_vector_params['polarization'] = pol_uniform
            self.vsamples = int(kwargs['polarization_samples'])
            kwargs.pop('polarization_samples')

        self.reset_vector_params()

        self.marginalize_phase = str_to_bool(marginalize_phase)

        self.distance_marginalization = False
        self.distance_interpolator = None

        marginalize_distance = str_to_bool(marginalize_distance)
        self.marginalize_distance = marginalize_distance
        if not marginalize_distance:
            return variable_params, kwargs

        if isinstance(marginalize_distance_snr_range, str):
            marginalize_distance_snr_range = \
                str_to_tuple(marginalize_distance_snr_range, float)

        if isinstance(marginalize_distance_density, str):
            marginalize_distance_density = \
                str_to_tuple(marginalize_distance_density, int)

        logging.info('Marginalizing over distance')

        # Take distance out of the variable params since we'll handle it
        # manually now
        dprior = pop_prior(marginalize_distance_param)

        if len(dprior.params) != 1 or not hasattr(dprior, 'bounds'):
            raise ValueError('Distance Marginalization requires a '
                             'univariate and bounded prior')

        # Set up distance prior vector and samples

        # (1) prior is using distance
        if dprior.params[0] == 'distance':
            logging.info("Prior is directly on distance, setting up "
                         "%s grid weights", marginalize_distance_samples)
            dmin, dmax = dprior.bounds['distance']
            dist_locs = numpy.linspace(dmin, dmax,
                                       int(marginalize_distance_samples))
            dist_weights = [dprior.pdf(distance=l) for l in dist_locs]
            dist_weights = numpy.array(dist_weights)

        # (2) prior is univariate and can be converted to distance
        elif marginalize_distance_param != 'distance':
            waveform_transforms = kwargs['waveform_transforms']
            pname = dprior.params[0]
            logging.info("Settings up transform,  prior is in terms of"
                         " %s", pname)
            wtrans = [d for d in waveform_transforms
                      if 'distance' not in d.outputs]
            if len(wtrans) == 0:
                wtrans = None
            kwargs['waveform_transforms'] = wtrans
            dtrans = [d for d in waveform_transforms
                      if 'distance' in d.outputs][0]
            v = dprior.rvs(int(1e8))
            d = dtrans.transform({pname: v[pname]})['distance']
            d.sort()
            cdf = numpy.arange(1, len(d)+1) / len(d)
            i = interp1d(d, cdf)
            dmin, dmax = d.min(), d.max()
            logging.info('Distance range %s-%s', dmin, dmax)
            x = numpy.linspace(dmin, dmax,
                               int(marginalize_distance_samples) + 1)
            xl, xr = x[:-1], x[1:]
            dist_locs = 0.5 * (xr + xl)
            dist_weights = i(xr) - i(xl)
        else:
            raise ValueError("No prior seems to determine the distance")

        dist_weights /= dist_weights.sum()
        dist_ref = 0.5 * (dmax + dmin)
        self.dist_locs = dist_locs
        self.distance_marginalization = dist_ref / dist_locs, dist_weights
        self.distance_interpolator = None

        if str_to_bool(marginalize_distance_interpolator):
            setup_args = {}
            if marginalize_distance_snr_range:
                setup_args['snr_range'] = marginalize_distance_snr_range
            if marginalize_distance_density:
                setup_args['density'] = marginalize_distance_density
            i = setup_distance_marg_interpolant(self.distance_marginalization,
                                                phase=self.marginalize_phase,
                                                **setup_args)
            self.distance_interpolator = i

        kwargs['static_params']['distance'] = dist_ref

        # Save marginalized parameters' name into one place,
        # coa_phase will be a static param if been marginalized
        if marginalize_distance:
            self.marginalized_params_name =\
                list(self.marginalize_vector_params.keys()) +\
                [marginalize_distance_param]

        return variable_params, kwargs

    def reset_vector_params(self):
        """ Redraw vector params from their priors
        """
        for param in self.marginalized_vector_priors:
            vprior = self.marginalized_vector_priors[param]
            values = vprior.rvs(self.vsamples)[param]
            self.marginalize_vector_params[param] = values

    def marginalize_loglr(self, sh_total, hh_total,
                          skip_vector=False, return_peak=False):
        """ Return the marginal likelihood

        Parameters
        -----------
        sh_total: float or ndarray
            The total <s|h> inner product summed over detectors
        hh_total: float or ndarray
            The total <h|h> inner product summed over detectors
        skip_vector: bool, False
            If true, and input is a vector, do not marginalize over that
            vector, instead return the likelihood values as a vector.
        """
        interpolator = self.distance_interpolator
        return_complex = False
        distance = self.distance_marginalization

        if self.reconstruct_vector:
            skip_vector = True

        if self.reconstruct_distance:
            interpolator = None
            skip_vector = True

        if self.reconstruct_phase:
            interpolator = None
            distance = False
            skip_vector = True
            return_complex = True

        return marginalize_likelihood(sh_total, hh_total,
                                      logw=self.marginalize_vector_weights,
                                      phase=self.marginalize_phase,
                                      interpolator=interpolator,
                                      distance=distance,
                                      skip_vector=skip_vector,
                                      return_complex=return_complex,
                                      return_peak=return_peak)

    def premarg_draw(self):
        """Choose random samples from a precomputed proposal set.

        Torch-backed proposal weights use their device RNG and remain on the
        device through selection and normalization. Public parameter arrays
        remain NumPy-backed, so only the selected integer indices cross to the
        host for those arrays.
        """

        # Update the current proposed times and the marginalization values
        logw = self.premarg['logw_partial']
        if self.vsamples == len(logw):
            choice = slice(None, None)
            host_choice = choice
        elif _torch_tensor(logw) is not None:
            import torch

            tensor = _torch_tensor(logw)
            choice = torch.randperm(
                len(logw), device=tensor.device
            )[:self.vsamples]
            host_choice = choice.detach().cpu().numpy()
        else:
            choice = numpy.random.choice(len(logw), size=self.vsamples,
                                         replace=False)
            host_choice = choice

        for k in self.snr_params:
            values = self.premarg[k]
            indices = (
                choice if _torch_tensor(values) is not None else host_choice
            )
            self.marginalize_vector_params[k] = _selected_values(
                values, indices, host=False
            )

        self._current_params.update(self.marginalize_vector_params)
        sample_idx = self.premarg['sample_idx']
        indices = (
            choice if _torch_tensor(sample_idx) is not None else host_choice
        )
        self.sample_idx = _selected_values(
            sample_idx, indices, host=False
        )

        # Update the importance weights for each vector sample
        selected_logw = _selected_values(logw, choice, host=False)
        logw = _add_values(self.marginalize_vector_weights, selected_logw)
        tensor = _torch_tensor(logw)
        if tensor is not None:
            import torch

            self.marginalize_vector_weights = (
                logw - torch.logsumexp(logw, dim=0)
            )
        else:
            self.marginalize_vector_weights = logw - logsumexp(logw)
        return self.marginalize_vector_params

    def snr_draw(self, wfs=None, snrs=None, size=None):
        """ Improve the monte-carlo vector marginalization using the SNR time
        series of each detector
        """
        try:
            p = self.current_params
            set_scalar = numpy.isscalar(p['tc'])
        except:
            set_scalar = False

        if not set_scalar:
            if hasattr(self, 'premarg'):
                return self.premarg_draw()

            if snrs is None:
                snrs = self.get_snr(wfs)
            if ('tc' in self.marginalized_vector_priors and
                not ('ra' in self.marginalized_vector_priors
                     or 'dec' in self.marginalized_vector_priors)):
                return self.draw_times(snrs, size=size)
            elif ('tc' in self.marginalized_vector_priors and
                  'ra' in self.marginalized_vector_priors and
                  'dec' in self.marginalized_vector_priors):
                return self.draw_sky_times(snrs, size=size)
        else:
            # OK, we couldn't do anything with the requested monte-carlo
            # marginalizations.
            self.precalc_antenna_factors = None
            return None

    def draw_times(self, snrs, size=None):
        """ Draw times consistent with the incoherent network SNR

        Parameters
        ----------
        snrs: dist of TimeSeries
        """
        if not hasattr(self, 'tinfo'):
            # determine the rough time offsets for this sky location
            tcprior = self.marginalized_vector_priors['tc']
            tcmin, tcmax = tcprior.bounds['tc']
            tcave = (tcmax + tcmin) / 2.0
            ifos = list(snrs.keys())
            if hasattr(self, 'keep_ifos'):
                ifos = self.keep_ifos
            d = {ifo: Detector(ifo, reference_time=tcave) for ifo in ifos}
            self.tinfo = tcmin, tcmax, tcave, ifos, d
            self.snr_params = ['tc']

        tcmin, tcmax, tcave, ifos, d = self.tinfo
        vsamples = size if size is not None else self.vsamples

        # Determine the weights for the valid time range
        ra = self._current_params['ra']
        dec = self._current_params['dec']

        # Determine the common valid time range
        iref = ifos[0]
        dref = d[iref]
        dt = dref.time_delay_from_earth_center(ra, dec, tcave)

        starts = []
        ends = []

        delt = snrs[iref].delta_t
        tmin = tcmin + dt - delt
        tmax = tcmax + dt + delt
        if hasattr(self, 'tstart'):
            tmin = self.tstart[iref]
            tmax = self.tend[iref]

        # Make sure we draw from times within prior and that have enough
        # SNR calculated to do later interpolation
        starts.append(max(tmin, snrs[iref].start_time + delt))
        ends.append(min(tmax, snrs[iref].end_time - delt * 2))

        idels = {}
        for ifo in ifos[1:]:
            dti = d[ifo].time_delay_from_detector(dref, ra, dec, tcave)
            idel = round(dti / snrs[iref].delta_t) * snrs[iref].delta_t
            idels[ifo] = idel

            starts.append(snrs[ifo].start_time - idel)
            ends.append(snrs[ifo].end_time - idel)
        start = max(starts)
        end = min(ends)
        if end <= start:
            return

        # get the weights
        snr = snrs[iref].time_slice(start, end, mode='nearest')
        logweight = _squared_norm_values(snr)
        for ifo in ifos[1:]:
            idel = idels[ifo]
            snrv = snrs[ifo].time_slice(snr.start_time + idel,
                                        snr.end_time + idel,
                                        mode='nearest')
            logweight += _squared_norm_values(snrv)
        logweight /= 2.0
        tensor = _torch_tensor(logweight)
        if tensor is not None:
            import torch
            logweight = logweight - torch.logsumexp(logweight, dim=0)
        else:
            logweight -= logsumexp(logweight)

        # Draw proportional to the incoherent likelihood
        # Draw first which time sample
        tci = draw_sample(logweight, size=vsamples, host=False)
        # Second draw a subsample size offset so that all times are covered
        tct = numpy.random.uniform(-snr.delta_t / 2.0,
                                   snr.delta_t / 2.0,
                                   size=vsamples)
        tci_tensor = _torch_tensor(tci)
        if tci_tensor is not None:
            # Public GPS times need host float64 precision, notably on MPS.
            time_indices = tci_tensor.detach().cpu().numpy()
        else:
            time_indices = tci
        tc = (
            tct + time_indices * snr.delta_t
            + float(snr.start_time) - dt
        )

        # Update the current proposed times and the marginalization values
        # assumes uniform prior!
        logw = (
            -_selected_values(logweight, tci, host=False)
            + numpy.log(1.0 / len(logweight))
        )
        self.marginalize_vector_params['tc'] = tc
        self.marginalize_vector_params['logw_partial'] = logw

        if self._current_params is not None:
            # Update the importance weights for each vector sample
            self._current_params.update(self.marginalize_vector_params)
            self.marginalize_vector_weights = _add_values(
                self.marginalize_vector_weights, logw
            )

        return self.marginalize_vector_params

    def draw_sky_times(self, snrs, size=None):
        """ Draw ra, dec, and tc together using SNR timeseries to determine
        monte-carlo weights.
        """
        # First setup
        # precalculate dense sky grid and make dict and or array of the results
        ifos = list(snrs.keys())
        if hasattr(self, 'keep_ifos'):
            ifos = self.keep_ifos
        ikey = ''.join(ifos)

        vsamples = size if size is not None else self.vsamples

        # No good SNR peaks, go with prior draw
        if len(ifos) == 0:
            self.marginalize_vector_params['logw_partial'] = numpy.zeros(vsamples)
            return

        def make_init():
            self.snr_params = ['tc', 'ra', 'dec']
            size = self.marginalize_sky_initial_samples
            logging.info('drawing samples: %s', size)
            ra = self.marginalized_vector_priors['ra'].rvs(size=size)['ra']
            dec = self.marginalized_vector_priors['dec'].rvs(size=size)['dec']
            tcmin, tcmax = self.marginalized_vector_priors['tc'].bounds['tc']
            tcave = (tcmax + tcmin) / 2.0
            d = {ifo: Detector(ifo, reference_time=tcave) for ifo in self.data}

            # What data structure to hold times? Dict of offset -> list?
            logging.info('sorting into time delay dict')
            dts = []
            for i in range(len(ifos) - 1):
                dt = d[ifos[0]].time_delay_from_detector(d[ifos[i+1]],
                                                         ra, dec, tcave)
                dt = numpy.rint(dt / snrs[ifos[0]].delta_t)
                dts.append(dt)

            fp, fc, dtc = {}, {}, {}
            for ifo in self.data:
                fp[ifo], fc[ifo] = d[ifo].antenna_pattern(ra, dec, 0.0, tcave)
                dtc[ifo] = d[ifo].time_delay_from_earth_center(ra, dec, tcave)

            dmap = {}
            for i, t in enumerate(tqdm.tqdm(zip(*dts))):
                if t not in dmap:
                    dmap[t] = []
                dmap[t].append(i)

            if len(ifos) == 1:
                dmap[()] = numpy.arange(0, size, 1).astype(int)

            # Sky prior by bin
            bin_prior = {t: len(dmap[t]) / size for t in dmap}

            return dmap, tcmin, tcmax, fp, fc, ra, dec, dtc, bin_prior

        if not hasattr(self, 'tinfo'):
            self.tinfo = {}

        if ikey not in self.tinfo:
            logging.info('pregenerating sky pointings')
            self.tinfo[ikey] = make_init()

        dmap, tcmin, tcmax, fp, fc, ra, dec, dtc, bin_prior = self.tinfo[ikey]

        # draw times from each snr time series
        # Is it worth doing this if some detector has low SNR?
        sref = None
        draw_weights = []
        sample_offsets = []
        for ifo in ifos:
            snr = snrs[ifo]
            tmin, tmax = tcmin - EARTH_RADIUS, tcmax + EARTH_RADIUS
            if hasattr(self, 'tstart'):
                tmin = self.tstart[ifo]
                tmax = self.tend[ifo]

            start = max(tmin, snr.start_time + snr.delta_t)
            end = min(tmax, snr.end_time - snr.delta_t * 2)
            snr = snr.time_slice(start, end, mode='nearest')

            w = _squared_norm_values(snr) / 2.0
            if sref is not None:
                delt = float(snr.start_time - sref.start_time)
                sample_offsets.append(round(delt / sref.delta_t))
            else:
                sref = snr
                sample_offsets.append(0)
            draw_weights.append(w)

        iref, dx, selected_weights = _draw_sky_time_indices(
            draw_weights, sample_offsets, vsamples
        )
        mcweight = None
        for selected_weight in selected_weights:
            mcweight = _add_values(mcweight, selected_weight)

        tensor = _torch_tensor(mcweight)
        if tensor is not None:
            import torch

            mcweight = mcweight - torch.logsumexp(mcweight, dim=0)
        else:
            mcweight -= logsumexp(mcweight)

        # check if delay is in dict, if not, throw out
        ti = []
        ix = []
        wi = []
        rand = numpy.random.uniform(0, 1, size=vsamples)
        for i in range(vsamples):
            t = tuple(x[i] for x in dx)
            if t in dmap:
                randi = int(rand[i] * (len(dmap[t])))
                ix.append(dmap[t][randi])
                wi.append(bin_prior[t])
                ti.append(i)

        # If we had really poor efficiency at finding a point, we should
        # give up and just use the original random draws
        if len(ix) < 0.05 * vsamples:
            self.marginalize_vector_params['logw_partial'] = numpy.zeros(vsamples)
            return

        # fill back to fixed size with repeat samples
        # sample order is random, so this should be OK statistically
        ix = numpy.resize(numpy.array(ix, dtype=int), vsamples)
        self.sample_idx = ix
        self.precalc_antenna_factors = fp, fc, dtc
        resize_factor = len(ti) / vsamples

        ra = ra[ix]
        dec = dec[ix]
        dtc = {ifo: dtc[ifo][ix] for ifo in dtc}

        ti = numpy.resize(numpy.array(ti, dtype=int), vsamples)
        wi = numpy.resize(numpy.array(wi), vsamples)

        # Second draw a subsample size offset so that all times are covered
        tct = numpy.random.uniform(-snr.delta_t / 2.0,
                                   snr.delta_t / 2.0,
                                   size=len(ti))

        tc = tct + iref[ti] * snr.delta_t + float(sref.start_time) - dtc[ifos[0]]

        # Update the current proposed times and the marginalization values
        # There's an overall normalization here which may introduce a constant
        # factor at the moment.
        selected_mcweight = _selected_values(mcweight, ti, host=False)
        logw_sky = _add_values(
            -selected_mcweight,
            numpy.log(wi) - numpy.log(resize_factor),
        )

        self.marginalize_vector_params['tc'] = tc
        self.marginalize_vector_params['ra'] = ra
        self.marginalize_vector_params['dec'] = dec
        self.marginalize_vector_params['logw_partial'] = logw_sky

        if self._current_params is not None:
            # Update the importance weights for each vector sample
            self._current_params.update(self.marginalize_vector_params)
            self.marginalize_vector_weights = _add_values(
                self.marginalize_vector_weights, logw_sky
            )

        return self.marginalize_vector_params

    def get_precalc_antenna_factors(self, ifo):
        """ Get the antenna factors for marginalized samples if they exist """
        ix = self.sample_idx
        fp, fc, dtc = self.precalc_antenna_factors
        return fp[ifo][ix], fc[ifo][ix], dtc[ifo][ix]

    def setup_peak_lock(self,
                        sample_rate=4096,
                        snrs=None,
                        peak_lock_snr=None,
                        peak_lock_ratio=1e4,
                        peak_lock_region=4,
                        **kwargs):
        """ Determine where to constrain marginalization based on
        the observed reference SNR peaks.

        Parameters
        ----------
        sample_rate : float
            The SNR sample rate
        snrs : Dict of SNR time series
            Either provide this or the model needs a function
            to get the reference SNRs.
        peak_lock_snr: float
            The minimum SNR to bother restricting from the prior range
        peak_lock_ratio: float
            The likelihood ratio (not log) relative to the peak to
            act as a threshold bounding region.
        peak_lock_region: int
            Number of samples to inclue beyond the strict region
            determined by the relative likelihood
        """

        if 'tc' not in self.marginalized_vector_priors:
            return

        tcmin, tcmax = self.marginalized_vector_priors['tc'].bounds['tc']
        tstart = tcmin - EARTH_RADIUS
        tmax = tcmax - tcmin + EARTH_RADIUS * 2.0
        num_samples = int(tmax * sample_rate)
        self.tstart = {ifo: tstart for ifo in self.data}
        self.num_samples = {ifo: num_samples for ifo in self.data}

        if snrs is None:
            if not hasattr(self, 'ref_snr'):
                raise ValueError("Model didn't have a reference SNR!")
            snrs = self.ref_snr

        # Restrict the time range for constructing SNR time series
        # to identifiable peaks
        if peak_lock_snr is not None:
            peak_lock_snr = float(peak_lock_snr)
            peak_lock_ratio = float(peak_lock_ratio)
            peak_lock_region = int(peak_lock_region)

            for ifo in snrs:
                s = max(tstart, snrs[ifo].start_time)
                e = min(tstart + tmax, snrs[ifo].end_time)
                z = snrs[ifo].time_slice(s, e, mode='nearest')
                peak_snr, imax = z.abs_max_loc()
                start_time = float(z.start_time)
                peak_time = start_time + imax * z.delta_t

                logging.info('%s: Max Ref SNR Peak of %s at %s',
                             ifo, peak_snr, peak_time)

                if peak_snr > peak_lock_snr:
                    target = peak_snr ** 2.0 / 2.0 - numpy.log(peak_lock_ratio)
                    target = (target * 2.0) ** 0.5

                    first, last = _threshold_extent(z, target)
                    ts = (
                        start_time + first * z.delta_t
                        - peak_lock_region / sample_rate
                    )
                    te = (
                        start_time + last * z.delta_t
                        + peak_lock_region / sample_rate
                    )
                    self.tstart[ifo] = ts
                    self.num_samples[ifo] = int((te - ts) * sample_rate)

            # Check times are commensurate with each other
            for ifo in snrs:
                ts = self.tstart[ifo]
                te = ts + self.num_samples[ifo] / sample_rate

                for ifo2 in snrs:
                    if ifo == ifo2:
                        continue
                    ts2 = self.tstart[ifo2]
                    te2 = ts2 + self.num_samples[ifo2] / sample_rate
                    det = Detector(ifo)
                    dt = Detector(ifo2).light_travel_time_to_detector(det)

                    ts = max(ts, ts2 - dt)
                    te = min(te, te2 + dt)

                self.tstart[ifo] = ts
                self.num_samples[ifo] = int((te - ts) * sample_rate) + 1
                logging.info('%s: use region %s-%s, %s points',
                             ifo, ts, te, self.num_samples[ifo])

        self.tend = self.tstart.copy()
        for ifo in snrs:
            self.tend[ifo] += self.num_samples[ifo] / sample_rate

    def draw_ifos(self, snrs, peak_snr_threshold=4.0, log=True,
                  precalculate_marginalization_points=False,
                  **kwargs):
        """ Helper utility to determine which ifos we should use based on the
        reference SNR time series.
        """
        if 'tc' not in self.marginalized_vector_priors:
            return

        peak_snr_threshold = float(peak_snr_threshold)

        tcmin, tcmax = self.marginalized_vector_priors['tc'].bounds['tc']
        ifos = list(snrs.keys())
        keep_ifos = []
        psnrs = []
        for ifo in ifos:
            snr = snrs[ifo]
            start = max(tcmin - EARTH_RADIUS, snr.start_time)
            end = min(tcmax + EARTH_RADIUS, snr.end_time)
            snr = snr.time_slice(start, end, mode='nearest')
            psnr = abs(snr).max()
            if psnr > peak_snr_threshold:
                keep_ifos.append(ifo)
            psnrs.append(psnr)

        if log:
            logging.info("Ifos used for SNR based draws:"
                         " %s, snrs: %s, peak_snr_threshold=%s",
                         keep_ifos, psnrs, peak_snr_threshold)

        self.keep_ifos = keep_ifos

        if precalculate_marginalization_points:
            num_points = int(float(precalculate_marginalization_points))
            self.premarg = self.snr_draw(size=num_points, snrs=snrs).copy()
            self.premarg['sample_idx'] = self.sample_idx

        return keep_ifos

    @property
    def current_params(self):
        """ The current parameters

        If a parameter has been vector marginalized, the likelihood should
        expect an array for the given parameter. This allows transparent
        vectorization for many models.
        """
        params = self._current_params
        for k in self.marginalize_vector_params:
            if k not in params:
                params[k] = self.marginalize_vector_params[k]
        self.marginalize_vector_weights = - numpy.log(self.vsamples)
        return params

    def reconstruct(self, rec=None, seed=None, set_loglr=None):
        """ Reconstruct the distance or vectored marginalized parameter
        of this class.
        """
        if seed:
            numpy.random.seed(seed)

        if rec is None:
            rec = {}

        if set_loglr is None:
            def get_loglr():
                p = self.current_params.copy()
                p.update(rec)
                self.update(**p)
                return self.loglr
        else:
            get_loglr = set_loglr

        if self.marginalize_vector_params:
            logging.debug('Reconstruct vector')
            self.reconstruct_vector = True
            self.reset_vector_params()
            loglr = get_loglr()
            xl = draw_sample(loglr + self.marginalize_vector_weights)
            for k in self.marginalize_vector_params:
                rec[k] = self.marginalize_vector_params[k][xl]
            self.reconstruct_vector = False

        if self.distance_marginalization:
            logging.debug('Reconstruct distance')
            # call likelihood to get vector output
            self.reconstruct_distance = True
            _, weights = self.distance_marginalization
            loglr = get_loglr()
            xl = draw_sample(_weighted_loglr(loglr, weights))
            rec['distance'] = self.dist_locs[xl]
            self.reconstruct_distance = False

        if self.marginalize_phase:
            logging.debug('Reconstruct phase')
            self.reconstruct_phase = True
            s, h = get_loglr()
            # This assumes that the template was conjugated in inner products
            phasev, loglr = _phase_reconstruction_values(s, h)
            xl = draw_sample(loglr)
            rec['coa_phase'] = _selected_scalar(phasev, xl)
            self.reconstruct_phase = False

        rec['loglr'] = _selected_scalar(loglr, xl)
        rec['loglikelihood'] = self.lognl + rec['loglr']
        return rec


def setup_distance_marg_interpolant(dist_marg,
                                    phase=False,
                                    snr_range=(1, 50),
                                    density=(1000, 1000)):
    """ Create the interpolant for distance marginalization

    Parameters
    ----------
    dist_marg: tuple of two arrays
        The (dist_loc, dist_weight) tuple which defines the grid
        for integrating over distance
    snr_range: tuple of (float, float)
        Tuple of min, max SNR that the interpolant is expected to work
        for.
    density: tuple of (float, float)
        The number of samples in either dimension of the 2d interpolant

    Returns
    -------
    interp: function
        Function which returns the precalculated likelihood for a given
        inner product sh/hh.
    """
    dist_rescale, _ = dist_marg
    logging.info("Interpolator valid for SNRs in %s", snr_range)
    logging.info("Interpolator using grid %s", density)
    # approximate maximum shr and hhr values, assuming the true SNR is
    # within the indicated range (and neglecting noise fluctuations)
    snr_min, snr_max = snr_range
    smax = dist_rescale.max()
    smin = dist_rescale.min()
    shr_max = snr_max ** 2.0 / smin
    hhr_max = snr_max ** 2.0 / smin / smin

    shr_min = snr_min ** 2.0 / smax
    hhr_min = snr_min ** 2.0 / smax / smax

    shr = numpy.geomspace(shr_min, shr_max, density[0])
    hhr = numpy.geomspace(hhr_min, hhr_max, density[1])
    lvals = numpy.zeros((len(shr), len(hhr)))
    logging.info('Setup up likelihood interpolator')
    for i, sh in enumerate(tqdm.tqdm(shr)):
        for j, hh in enumerate(hhr):
            lvals[i, j] = marginalize_likelihood(sh, hh,
                                                 distance=dist_marg,
                                                 phase=phase)
    interp = RectBivariateSpline(shr, hhr, lvals)
    torch_interp = _torch_rect_bivariate_spline_evaluator(interp)

    # said once, the first time it happens
    warned = [False]

    def warn_out_of_range():
        warned[0] = True
        logging.warning(
            "A likelihood evaluation asked for a signal to noise ratio "
            "outside marginalize_distance_snr_range %s; beyond it the "
            "distance marginalized likelihood is set to zero, which biases "
            "the result. Widen the range.", snr_range)

    def interp_wrapper(x, y, bounds_check=True):
        if _torch_tensor(x) is not None or _torch_tensor(y) is not None:
            return torch_interp(x, y, bounds_check=bounds_check)

        k = None
        if bounds_check:
            if isinstance(x, float):
                if x > shr_max or x < shr_min or y > hhr_max or y < hhr_min:
                    if not warned[0]:
                        warn_out_of_range()
                    return -numpy.inf
            else:
                k = (x > shr_max) | (x < shr_min)
                k = k | (y > hhr_max) | (y < hhr_min)
                # short circuits, so this costs nothing once said
                if not warned[0] and k.any():
                    warn_out_of_range()

        v = interp(x, y, grid=False)
        if k is not None:
            v[k] = -numpy.inf
        return v
    interp_wrapper._torch_evaluate = torch_interp
    return interp_wrapper


def _torch_bspline_basis(values, knots, degree):
    """Return active B-spline coefficient indices and basis values."""
    import torch

    coefficient_count = knots.numel() - degree - 1
    spans = torch.searchsorted(
        knots, values.contiguous(), right=True
    ).sub(1)
    spans = spans.clamp(degree, coefficient_count - 1)

    basis = torch.ones(
        values.shape + (1,), device=values.device, dtype=values.dtype
    )
    left = [None] * (degree + 1)
    right = [None] * (degree + 1)
    for column in range(1, degree + 1):
        left[column] = values - knots[spans + 1 - column]
        right[column] = knots[spans + column] - values
        saved = torch.zeros_like(values)
        updated = []
        for row in range(column):
            weight = basis[..., row] / (
                right[row + 1] + left[column - row]
            )
            updated.append(saved + right[row + 1] * weight)
            saved = left[column - row] * weight
        updated.append(saved)
        basis = torch.stack(updated, dim=-1)

    offsets = torch.arange(
        degree + 1, device=values.device, dtype=torch.int64
    )
    indices = spans[..., None] - degree + offsets
    return indices, basis


def _torch_rect_bivariate_spline_evaluator(interp):
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
        import torch

        x_tensor = _torch_tensor(x)
        y_tensor = _torch_tensor(y)
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
                torch.as_tensor(
                    coefficients, device=like.device, dtype=dtype
                ),
            )
            cache[key] = cached
        tensor_knots_x, tensor_knots_y, tensor_coefficients = cached

        indices_x, basis_x = _torch_bspline_basis(
            x_tensor, tensor_knots_x, degree_x
        )
        indices_y, basis_y = _torch_bspline_basis(
            y_tensor, tensor_knots_y, degree_y
        )
        local_coefficients = tensor_coefficients[
            indices_x[..., :, None], indices_y[..., None, :]
        ]
        values = (
            local_coefficients
            * basis_x[..., :, None]
            * basis_y[..., None, :]
        ).sum(dim=(-2, -1))

        if bounds_check:
            outside = (
                (x_tensor < knots_x[degree_x])
                | (x_tensor > knots_x[-degree_x - 1])
                | (y_tensor < knots_y[degree_y])
                | (y_tensor > knots_y[-degree_y - 1])
            )
            values = torch.where(
                outside, values.new_full((), -torch.inf), values
            )
        return values

    return evaluate


def _numpy_from_torch(value):
    """Return a detached CPU value for a NumPy-only calculation."""
    tensor = _torch_tensor(value)
    if tensor is None:
        return value

    tensor = tensor.detach()
    if tensor.is_conj():
        tensor = tensor.resolve_conj()
    if tensor.ndim == 0:
        return tensor.item()
    return tensor.cpu().numpy()


def _marginalize_likelihood_torch(sh, hh, logw, phase, distance,
                                  skip_vector, return_peak,
                                  return_complex, interpolator=None):
    """Torch implementation of explicit likelihood marginalizations."""
    import torch

    sh_tensor = _torch_tensor(sh)
    hh_tensor = _torch_tensor(hh)
    like = sh_tensor if sh_tensor is not None else hh_tensor
    if sh_tensor is None:
        real_dtype = hh_tensor.real.dtype
        if numpy.iscomplexobj(sh):
            sh_dtype = (
                torch.complex128
                if real_dtype == torch.float64 else torch.complex64
            )
        else:
            sh_dtype = real_dtype
        sh = torch.as_tensor(sh, device=like.device, dtype=sh_dtype)
    else:
        sh = sh_tensor
    hh = torch.as_tensor(
        hh_tensor if hh_tensor is not None else hh,
        device=like.device, dtype=sh.real.dtype)

    if distance and interpolator is None and sh.ndim:
        raise ValueError("Cannot do vector marginalization "
                         "and distance at the same time")

    if return_complex:
        pass
    elif phase:
        sh = torch.abs(sh)
    else:
        sh = sh.real

    if distance and interpolator is None:
        dist_rescale, dist_weights = distance
        dist_rescale = torch.as_tensor(
            dist_rescale, device=like.device, dtype=hh.dtype)
        dist_weights = torch.as_tensor(
            dist_weights, device=like.device, dtype=hh.dtype)
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
                logw_tensor = _torch_tensor(logw)
                if logw_tensor is not None:
                    logw = logw_tensor
                logw = torch.as_tensor(
                    logw, device=vloglr.device, dtype=vloglr.dtype)
            vloglr = torch.logsumexp(vloglr + logw, dim=0)
        vloglr = vloglr.item()

    if return_peak:
        return vloglr, maxv, maxl
    return vloglr


def marginalize_likelihood(sh, hh,
                           logw=None,
                           phase=False,
                           distance=False,
                           skip_vector=False,
                           interpolator=None,
                           return_peak=False,
                           return_complex=False,
                           ):
    """ Return the marginalized likelihood.

    Apply various marginalizations to the data, including phase, distance,
    and brute-force vector marginalizations. Several options relate
    to how the distance marginalization is approximated and others allow for
    special return products to aid in parameter reconstruction.

    Parameters
    ----------
    sh: complex float or numpy.ndarray
        The data-template inner product
    hh: complex float or numpy.ndarray
        The template-template inner product
    logw:
        log weighting factors if vector marginalization is used, if not
        given, each sample is assumed to be equally weighted
    phase: bool, False
        Enable phase marginalization. Only use if orbital phase can be related
        to just a single overall phase (e.g. not true for waveform with
        sub-dominant modes)
    skip_vector: bool, False
        Don't apply marginalization of vector component of input (i.e. leave
        as vector).
    interpolator: function, None
        If provided, internal calculation is skipped in favor of a
        precalculated interpolating function which takes in sh/hh
        and returns the likelihood.
    return_peak: bool, False
        Return the peak likelihood and index if using passing an array as
        input in addition to the marginalized over the array likelihood.
    return_complex: bool, False
        Return the sh / hh data products before applying phase marginalization.
        This option is intended to aid in reconstucting phase marginalization
        and is unlikely to be useful for other purposes.

    Returns
    -------
    loglr: float
        The marginalized loglikehood ratio
    """
    sh_tensor = _torch_tensor(sh)
    hh_tensor = _torch_tensor(hh)
    if sh_tensor is not None or hh_tensor is not None:
        torch_interpolator = getattr(
            interpolator, '_torch_evaluate', None
        )
        if interpolator is None or torch_interpolator is not None:
            return _marginalize_likelihood_torch(
                sh, hh, logw, phase, distance, skip_vector,
                return_peak, return_complex, torch_interpolator)
        sh = _numpy_from_torch(sh)
        hh = _numpy_from_torch(hh)

    if distance and not interpolator and not numpy.isscalar(sh):
        raise ValueError("Cannot do vector marginalization "
                         "and distance at the same time")

    if logw is None:
        if isinstance(hh, float):
            logw = 0
        else:
            logw = -numpy.log(len(sh))

    if return_complex:
        pass
    elif phase:
        sh = abs(sh)
    else:
        sh = sh.real

    if interpolator:
        # pre-calculated result for this function
        vloglr = interpolator(sh, hh)

        if skip_vector:
            return vloglr
    else:
        # explicit calculation
        if distance:
            # brute force distance path
            dist_rescale, dist_weights = distance
            sh = sh * dist_rescale
            hh = hh * dist_rescale ** 2.0
            logw = numpy.log(dist_weights)

        if return_complex:
            return sh, -0.5 * hh

        # Apply the phase marginalization
        if phase:
            sh = numpy.log(i0e(sh)) + sh

        # Calculate loglikelihood ratio
        vloglr = sh - 0.5 * hh

    if return_peak:
        maxv = vloglr.argmax()
        maxl = vloglr[maxv]

    # Do brute-force marginalization if loglr is a vector
    if isinstance(vloglr, float):
        vloglr = float(vloglr)
    elif not skip_vector:
        vloglr = float(logsumexp(vloglr, b=numpy.exp(logw)))

    if return_peak:
        return vloglr, maxv, maxl
    return vloglr
