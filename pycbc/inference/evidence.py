# Copyright (C) 2019 Steven Reyes
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
"""
This modules provides functions for estimating the marginal
likelihood or evidence of a model.
"""
import numpy
from scipy import integrate

# numpy renamed trapz to trapezoid in 2.0 and removed trapz in 2.x
try:
    from numpy import trapezoid
except ImportError:  # numpy < 2.0
    from numpy import trapz as trapezoid


def _torch_log_likelihood(log_likelihood):
    """Return a Torch likelihood tensor without importing Torch eagerly."""
    if type(log_likelihood).__module__.split(".", 1)[0] != "torch":
        return None

    import torch

    if isinstance(log_likelihood, torch.Tensor):
        if log_likelihood.numel() == 0:
            raise ValueError(
                "zero-size array to reduction operation maximum which has "
                "no identity"
            )
        if not log_likelihood.is_floating_point():
            return log_likelihood.to(dtype=torch.get_default_dtype())
        return log_likelihood
    return None


def _torch_evidence_inputs(*values):
    """Move mixed estimator inputs beside their existing Torch tensor."""
    if not any(
            type(value).__module__.split(".", 1)[0] == "torch"
            for value in values):
        return None

    import torch

    reference = next(
        value for value in values if isinstance(value, torch.Tensor)
    )
    dtype = (
        reference.dtype
        if reference.is_floating_point()
        else torch.get_default_dtype()
    )
    tensors = []
    for value in values:
        is_complex = (
            value.is_complex()
            if isinstance(value, torch.Tensor)
            else numpy.iscomplexobj(value)
        )
        if is_complex:
            raise TypeError("evidence estimator inputs must be real")
        tensor = torch.as_tensor(
            value, device=reference.device, dtype=dtype
        )
        tensors.append(tensor)
    return tensors


def _numpy_simpson_last(values, x, axis=0):
    """Integrate with the legacy SciPy ``simps(even="last")`` rule."""
    values = numpy.asarray(values)
    x = numpy.asarray(x)
    count = values.shape[axis]
    if x.ndim != 1 or len(x) != count:
        raise ValueError("x must be one-dimensional and match the integration axis")
    if count == 0:
        raise ValueError("cannot integrate an empty axis")

    result = values.sum(axis=axis) * 0.0
    start = 0
    if count % 2 == 0:
        first = numpy.take(values, 0, axis=axis)
        second = numpy.take(values, 1, axis=axis)
        result = 0.5 * (x[1] - x[0]) * (first + second)
        start = 1

    if count - start >= 3:
        slices = [slice(None)] * values.ndim
        slices[axis] = slice(start, None)
        simpson = getattr(integrate, "simpson", None)
        if simpson is None:
            simpson = integrate.simps
        result = result + simpson(
            values[tuple(slices)], x=x[start:], axis=axis
        )
    return result


def _torch_simpson_last(values, x, dim=0):
    """Torch implementation of the legacy ``simps(even="last")`` rule."""
    import torch

    values = values.movedim(dim, 0)
    count = values.shape[0]
    if x.ndim != 1 or len(x) != count:
        raise ValueError("x must be one-dimensional and match the integration axis")
    if count == 0:
        raise ValueError("cannot integrate an empty axis")

    result = values.sum(dim=0) * 0.0
    start = 0
    if count % 2 == 0:
        result = 0.5 * (x[1] - x[0]) * (values[0] + values[1])
        start = 1

    if count - start >= 3:
        y0 = values[start:-2:2]
        y1 = values[start + 1:-1:2]
        y2 = values[start + 2::2]
        h0 = x[start + 1:-1:2] - x[start:-2:2]
        h1 = x[start + 2::2] - x[start + 1:-1:2]
        shape = (len(h0),) + (1,) * (values.ndim - 1)
        h0 = h0.reshape(shape)
        h1 = h1.reshape(shape)
        width = h0 + h1
        triples = width / 6.0 * (
            y0 * (2.0 - h1 / h0)
            + y1 * width.square() / (h0 * h1)
            + y2 * (2.0 - h0 / h1)
        )
        result = result + torch.sum(triples, dim=0)
    return result


def arithmetic_mean_estimator(log_likelihood):
    """Returns the log evidence via the prior arithmetic mean estimator (AME).

    The logarithm form of AME is used. This is the most basic
    evidence estimator, and often requires O(billions) of samples
    from the prior.

    Parameters
    ----------
    log_likelihood : 1d array of floats
        The log likelihood of the data sampled from the prior
        distribution.

    Returns
    -------
    float :
        Estimation of the log of the evidence.
    """
    tensor = _torch_log_likelihood(log_likelihood)
    if tensor is not None:
        import torch

        num_samples = tensor.numel()
        return torch.logsumexp(tensor.reshape(-1), dim=0) - torch.log(
            tensor.new_tensor(num_samples)
        )

    num_samples = len(log_likelihood)
    logl_max = numpy.max(log_likelihood)

    log_evidence = 0.
    for i, _ in enumerate(log_likelihood):
        log_evidence += numpy.exp(log_likelihood[i] - logl_max)

    log_evidence = numpy.log(log_evidence)
    log_evidence += logl_max - numpy.log(num_samples)

    return log_evidence


def harmonic_mean_estimator(log_likelihood):
    """Returns the log evidence via posterior harmonic mean estimator (HME).

    The logarithm form of HME is used. This method is not
    recommended for general use. It is very slow to converge,
    formally, has infinite variance, and very error prone.

    Not recommended for general use.

    Parameters
    ----------
    log_likelihood : 1d array of floats
        The log likelihood of the data sampled from the posterior
        distribution.

    Returns
    -------
    float :
        Estimation of the log of the evidence.
    """
    tensor = _torch_log_likelihood(log_likelihood)
    if tensor is not None:
        import torch

        num_samples = tensor.numel()
        return torch.log(tensor.new_tensor(num_samples)) - torch.logsumexp(
            -tensor.reshape(-1), dim=0
        )

    num_samples = len(log_likelihood)
    logl_max = numpy.max(-1.0*log_likelihood)

    log_evidence = 0.
    for i, _ in enumerate(log_likelihood):
        log_evidence += numpy.exp(-1.0*log_likelihood[i] + logl_max)

    log_evidence = -1.0*numpy.log(log_evidence)
    log_evidence += logl_max
    log_evidence += numpy.log(num_samples)

    return log_evidence


# numpy.trapz was renamed to numpy.trapezoid in numpy 2.0.
try:
    from numpy import trapezoid as _trapezoid
except ImportError:  # numpy < 2.0
    from numpy import trapz as _trapezoid


def mean_logl_by_temperature(logls, betas):
    """The mean log likelihood at each distinct inverse temperature.

    Parameters
    ----------
    logls : numpy.ndarray
        Log likelihoods of shape (ntemps, nwalkers, niterations).
    betas : numpy.ndarray
        The inverse temperatures, of shape (ntemps, niterations); a ladder
        that adapts visits more than one temperature per chain.

    Returns
    -------
    betas : numpy.ndarray
        Each distinct inverse temperature.
    mean_logls : numpy.ndarray
        The mean log likelihood at each of them.
    """
    mean_logls = []
    unique_betas = []
    for ti in range(betas.shape[0]):
        ubti, idx = numpy.unique(betas[ti, :], return_inverse=True)
        unique_idx = numpy.unique(idx)
        loglsti = logls[ti, :, :]
        for ii in unique_idx:
            # average over the walkers and iterations at this temperature
            getiters = numpy.where(ii == unique_idx)[0]
            mean_logls.append(loglsti[:, getiters].mean())
            unique_betas.append(ubti[ii])
    return numpy.array(unique_betas), numpy.array(mean_logls)


def ladder_thermodynamic_integration(betas, logls):
    """Thermodynamic integration estimate of the evidence.

    This is the same estimator used by the ``ptemcee`` sampler; see
    :py:func:`pycbc.inference.sampler.ptemcee` for details.

    Parameters
    ----------
    betas : array
        The inverse temperatures to use for the quadrature.
    logls : array
        The mean log-likelihoods corresponding to ``betas``.

    Returns
    -------
    logZ : float
        Estimate of the log-evidence.
    dlogZ : float
        The error associated with the finite number of temperatures at which
        the posterior has been sampled.
    """
    if len(betas) != len(logls):
        raise ValueError("Need the same number of log(L) values as "
                         "temperatures.")
    order = numpy.argsort(betas)[::-1]
    betas = betas[order]
    logls = logls[order]
    betas0 = numpy.copy(betas)
    if betas[-1] != 0:
        betas = numpy.concatenate((betas0, [0]))
        betas2 = numpy.concatenate((betas0[::2], [0]))
        logls2 = numpy.concatenate((logls[::2], [logls[-1]]))
        logls = numpy.concatenate((logls, [logls[-1]]))
    else:
        betas2 = numpy.concatenate((betas0[:-1:2], [0]))
        logls2 = numpy.concatenate((logls[:-1:2], [logls[-1]]))
    logZ = -_trapezoid(logls, betas)
    logZ2 = -_trapezoid(logls2, betas2)
    return logZ, numpy.abs(logZ - logZ2)


def thermodynamic_integration(log_likelihood, betas,
                              method="simpsons"):
    """Returns the log evidence of the model via thermodynamic integration.
    Also returns an estimated standard deviation for the log evidence.

    Current options are integration through the trapezoid rule, a
    first-order corrected trapezoid rule, and Simpson's rule.

    Parameters
    ----------
    log_likelihood : 3d array or torch.Tensor
        The log likelihood for each temperature separated by
        temperature, walker, and iteration. All methods support Torch
        tensors on their current device.

    betas : 1d array or torch.Tensor
        The inverse temperatures used in the MCMC. For Torch methods, mixed
        inputs are moved beside the existing Torch tensor.

    method : {"trapezoid", "trapezoid_corrected", "simpsons"},
             optional.
        The numerical integration method to use for the
        thermodynamic integration. Choices include: "trapezoid",
        "trapezoid_corrected", "simpsons", for the trapezoid rule,
        the first-order correction to the trapezoid rule, and
        Simpson's rule. [Default = "simpsons"]

    Returns
    -------
    log_evidence : float or torch.Tensor
        Estimation of the log of the evidence.

    mcmc_std : float or torch.Tensor
        The standard deviation of the log evidence estimate from
        Monte-Carlo spread. Torch inputs return scalar tensors.
    """
    # Check if the method of integration is in the list of choices
    method_list = ["trapezoid", "trapezoid_corrected", "simpsons"]

    if method not in method_list:
        raise ValueError("Method %s not supported. Expected %s"
                         % (method, method_list))

    torch_inputs = _torch_evidence_inputs(log_likelihood, betas)
    if torch_inputs is not None:
        import torch

        log_likelihood, betas = torch_inputs
        if log_likelihood.numel() == 0:
            raise ValueError("thermodynamic integration requires samples")
        order = torch.argsort(betas)
        betas = betas[order]
        log_likelihood = log_likelihood[order].reshape(len(betas), -1)
        num_samples = log_likelihood.shape[1]

        average_logl = log_likelihood.mean(dim=1)
        if method == "simpsons":
            log_evidence = _torch_simpson_last(average_logl, betas)
        else:
            log_evidence = torch.trapezoid(average_logl, betas)

        if method == "trapezoid_corrected":
            delta_beta = betas[1:] - betas[:-1]
            variances = log_likelihood.var(dim=1, correction=0)
            variance_delta = variances[1:] - variances[:-1]
            log_evidence -= (
                delta_beta.square() * variance_delta / 12.0
            ).sum()

        if method == "simpsons":
            ti_vec = _torch_simpson_last(log_likelihood, betas, dim=0)
        else:
            ti_vec = torch.trapezoid(
                log_likelihood, betas[:, None], dim=0
            )
        mcmc_std = ti_vec.std(correction=0) / torch.sqrt(
            ti_vec.new_tensor(num_samples)
        )
        return log_evidence, mcmc_std

    # Read in the data and ensure ordering of data.
    # Ascending order sort
    order = numpy.argsort(betas)
    betas = betas[order]
    log_likelihood = log_likelihood[order]

    # Assume log likelihood is given in shape of beta, walker,
    # and iteration.
    log_likelihood = numpy.reshape(log_likelihood,
                                   (len(betas),
                                    len(log_likelihood[0].flatten())))

    average_logl = numpy.average(log_likelihood, axis=1)

    if method in ("trapezoid", "trapezoid_corrected"):
        log_evidence = trapezoid(average_logl, betas)

    if method == "trapezoid_corrected":
        # var_correction holds the derivative correction terms
        # See Friel et al. 2014 for expression and derivation.
        # https://link.springer.com/article/10.1007/s11222-013-9397-1
        var_correction = 0
        for i in range(len(betas) - 1):
            delta_beta = betas[i+1] - betas[i]
            pre_fac_var = (1. / 12.) * (delta_beta ** 2.0)
            var_diff = numpy.var(log_likelihood[i+1])
            var_diff -= numpy.var(log_likelihood[i])
            var_correction -= pre_fac_var * var_diff

        # Add the derivative correction term back to the log_evidence
        # from the first if statement.
        log_evidence += var_correction

    elif method == "simpsons":
        # beta -> 0 tends to contribute the least to the integral
        # so we can sacrifice precision there, rather than near
        # beta -> 1. Option even="last" puts trapezoid rule at
        # first few points.
        log_evidence = _numpy_simpson_last(average_logl, betas)

    # Estimate the Monte Carlo variance of the evidence calculation
    # See (Evans, Annis, 2019.)
    # https://www.sciencedirect.com/science/article/pii/S0022249617302651
    ti_vec = numpy.zeros(len(log_likelihood[0]))

    # Get log likelihood chains by sample and not by temperature.
    logl_per_samp = []
    for i, _ in enumerate(log_likelihood[0]):
        logl_per_samp.append([log_likelihood[x][i] for x in range(len(betas))])

    if method in ("trapezoid", "trapezoid_corrected"):
        for i, _ in enumerate(log_likelihood[0]):
            ti_vec[i] = trapezoid(logl_per_samp[i], betas)

    elif method == "simpsons":
        ti_vec[:] = _numpy_simpson_last(log_likelihood, betas, axis=0)

    # Standard error is sample std / sqrt(number of samples)
    mcmc_std = numpy.std(ti_vec) / numpy.sqrt(float(len(log_likelihood[0])))

    return log_evidence, mcmc_std


def stepping_stone_algorithm(log_likelihood, betas):
    """Returns the log evidence of the model via stepping stone algorithm.
    Also returns an estimated standard deviation for the log evidence.

    Parameters
    ----------
    log_likelihood : 3d array or torch.Tensor
        The log likelihood for each temperature separated by
        temperature, walker, and iteration. Torch inputs are evaluated on
        their current device.

    betas : 1d array or torch.Tensor
        The inverse temperatures used in the MCMC. Mixed array and Torch
        inputs are moved to the device and dtype of the Torch input.

    Returns
    -------
    log_evidence : float or torch.Tensor
        Estimation of the log of the evidence.
    mcmc_std : float or torch.Tensor
        The standard deviation of the log evidence estimate from
        Monte-Carlo spread. Torch inputs return scalar Torch tensors.
    """
    torch_inputs = _torch_evidence_inputs(log_likelihood, betas)
    if torch_inputs is not None:
        import torch

        log_likelihood, betas = torch_inputs
        if log_likelihood.numel() == 0:
            raise ValueError(
                "zero-size array to reduction operation maximum which has "
                "no identity"
            )
        order = torch.argsort(betas, descending=True)
        betas = betas[order]
        log_likelihood = log_likelihood[order].reshape(len(betas), -1)
        num_samples = log_likelihood.shape[1]

        delta_beta = betas[:-1] - betas[1:]
        weighted_logl = delta_beta[:, None] * log_likelihood[1:]
        log_rk_pb = torch.logsumexp(weighted_logl, dim=1) - torch.log(
            weighted_logl.new_tensor(num_samples)
        )
        log_evidence = log_rk_pb.sum()

        centered = torch.exp(weighted_logl - log_rk_pb[:, None]) - 1.0
        mcmc_std = torch.sqrt(
            centered.square().sum() / float(num_samples) ** 2.0
        )
        return log_evidence, mcmc_std

    # Reverse order sort
    order = numpy.argsort(betas)[::-1]
    betas = betas[order]
    log_likelihood = log_likelihood[order]

    # Assume log likelihood is given in shape of beta,
    # walker, iteration.
    log_likelihood = numpy.reshape(log_likelihood,
                                   (len(betas),
                                    len(log_likelihood[0].flatten())))

    log_rk_pb = numpy.zeros(len(betas) - 1)
    for i in range(len(betas) - 1):
        delta_beta = betas[i] - betas[i+1]
        # Max log likelihood for beta [i+1]
        max_logl_pb = numpy.max(log_likelihood[i+1])
        val_1 = delta_beta * max_logl_pb
        val_2 = delta_beta * (log_likelihood[i+1] - max_logl_pb)
        val_2 = numpy.log(numpy.average(numpy.exp(val_2)))
        log_rk_pb[i] = val_1 + val_2

    log_rk = numpy.sum(log_rk_pb)
    log_evidence = log_rk

    # Calculate the Monte Carlo variation
    mcmc_std = 0
    for i in range(len(betas) - 1):
        delta_beta = betas[i] - betas[i+1]
        pre_fact = (delta_beta * log_likelihood[i+1]) - log_rk_pb[i]
        pre_fact = numpy.exp(pre_fact) - 1.0
        val = numpy.sum(pre_fact ** 2)

        mcmc_std += val

    mcmc_std /= float(len(log_likelihood[0])) ** 2.0
    mcmc_std = numpy.sqrt(mcmc_std)

    return log_evidence, mcmc_std
