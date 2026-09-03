# Copyright (C) 2017  Christopher M. Biwer
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
""" This modules provides functions for evaluating the Gelman-Rubin convergence
diagnostic statistic.
"""

import numpy


def _torch_chains(chains):
    """Return real Torch chains without importing Torch eagerly."""
    is_tensor = type(chains).__module__.split(".", 1)[0] == "torch"
    is_tensor_list = isinstance(chains, (list, tuple)) and any(
        type(chain).__module__.split(".", 1)[0] == "torch"
        for chain in chains
    )
    if not (is_tensor or is_tensor_list):
        return None

    import torch

    if isinstance(chains, torch.Tensor):
        tensor = chains
        if tensor.is_complex():
            raise TypeError("Gelman-Rubin chains must be real")
        if not tensor.is_floating_point():
            tensor = tensor.to(dtype=torch.get_default_dtype())
        return tensor

    reference = next(
        chain for chain in chains if isinstance(chain, torch.Tensor)
    )
    dtype = (
        reference.dtype
        if reference.is_floating_point()
        else torch.get_default_dtype()
    )
    if any(
            chain.is_complex()
            if isinstance(chain, torch.Tensor)
            else numpy.iscomplexobj(chain)
            for chain in chains):
        raise TypeError("Gelman-Rubin chains must be real")
    return torch.stack([
        torch.as_tensor(chain, device=reference.device, dtype=dtype)
        for chain in chains
    ])


def _torch_sample_covariance(rows):
    """Return row-wise sample covariance, matching ``numpy.cov``."""
    centered = rows - rows.mean(dim=-1, keepdim=True)
    return centered @ centered.transpose(-1, -2) / (rows.shape[-1] - 1)


def _torch_gelman_rubin(chains, auto_burn_in):
    """Torch implementation of the univariate Gelman-Rubin estimator."""
    import torch

    if auto_burn_in:
        niterations = chains.shape[2]
        chains = chains[:, :, niterations // 2 + 1:]

    nchains, _, niterations = chains.shape
    chains_covs = _torch_sample_covariance(chains)
    w = chains_covs.mean(dim=0)

    means = chains.mean(dim=2).transpose(0, 1)
    b = niterations * _torch_sample_covariance(means)
    w_diag = torch.diagonal(w)
    b_diag = torch.diagonal(b)

    var = torch.diagonal(chains_covs, dim1=-2, dim2=-1).transpose(0, 1)
    mu_hat = means.mean(dim=1)
    s = var.var(dim=1, correction=0)

    v = (
        (niterations - 1.0) * w_diag / niterations
        + (1.0 + 1.0 / nchains) * b_diag / niterations
    )
    k = 2.0 * b_diag**2 / (nchains - 1)
    centered_var = var - var.mean(dim=1, keepdim=True)
    centered_means = means - means.mean(dim=1, keepdim=True)
    centered_means_squared = means**2 - (means**2).mean(
        dim=1, keepdim=True
    )
    mid_term = (
        centered_var * centered_means_squared
    ).sum(dim=1) / (nchains - 1)
    end_term = (
        centered_var * centered_means
    ).sum(dim=1) / (nchains - 1)
    wb = niterations / nchains * (mid_term - 2.0 * mu_hat * end_term)

    var_v = (
        (niterations - 1.0) ** 2 * s
        + (1.0 + 1.0 / nchains) ** 2 * k
        + 2.0 * (niterations - 1.0) * (1.0 + 1.0 / nchains) * wb
    ) / niterations**2
    dof = 2.0 * v**2 / var_v
    df_adj = (dof + 3.0) / (dof + 1.0)
    r2_estimate = (
        (niterations - 1.0) / niterations
        + (1.0 + 1.0 / nchains) / niterations * (b_diag / w_diag)
    )
    return torch.sqrt(r2_estimate * df_adj)


def walk(chains, start, end, step):
    """ Calculates Gelman-Rubin conervergence statistic along chains of data.
    This function will advance along the chains and calculate the
    statistic for each step.

    Parameters
    ----------
    chains : iterable or torch.Tensor
        An iterable of arrays, or a Torch tensor, containing the samples for
        each chain. Each chain has shape (nparameters, niterations). Torch
        inputs remain on their current device and produce Torch outputs.
    start : float
        Start index of blocks to calculate all statistics.
    end : float
        Last index of blocks to calculate statistics.
    step : float
        Step size to take for next block.

    Returns
    -------
    starts : numpy.array or torch.Tensor
        1-D array of start indexes of calculations.
    ends : numpy.array or torch.Tensor
        1-D array of end indexes of caluclations.
    stats : numpy.array or torch.Tensor
        Array with convergence statistic. It has
        shape (nparameters, ncalculations).
    """

    tensor = _torch_chains(chains)
    if tensor is not None:
        import torch

        _, nparameters, _ = tensor.shape
        end_values = list(range(start, end, step))
        ends = torch.tensor(
            end_values, device=tensor.device, dtype=torch.int64
        )
        starts = torch.full(
            (len(end_values),), start,
            device=tensor.device, dtype=torch.int64,
        )
        values = [
            _torch_gelman_rubin(tensor[:, :, :e], True)
            for e in end_values
        ]
        stats = (
            torch.stack(values, dim=1)
            if values else tensor.new_empty((nparameters, 0))
        )
        return starts, ends, stats

    # get number of chains, parameters, and iterations
    chains = numpy.array(chains)
    _, nparameters, _ = chains.shape

    # get end index of blocks
    ends = numpy.arange(start, end, step)
    stats = numpy.zeros((nparameters, len(ends)))

    # get start index of blocks
    starts = numpy.array(len(ends) * [start])

    # loop over end indexes and calculate statistic
    for i, e in enumerate(ends):
        tmp = chains[:, :, 0:e]
        stats[:, i] = gelman_rubin(tmp)

    return starts, ends, stats


def gelman_rubin(chains, auto_burn_in=True):
    """ Calculates the univariate Gelman-Rubin convergence statistic
    which compares the evolution of multiple chains in a Markov-Chain Monte
    Carlo process and computes their difference to determine their convergence.
    The between-chain and within-chain variances are computed for each sampling
    parameter, and a weighted combination of the two is used to determine the
    convergence. As the chains converge, the point scale reduction factor
    should go to 1.

    Parameters
    ----------
    chains : iterable or torch.Tensor
        An iterable of arrays, or a Torch tensor, containing the samples for
        each chain. Each chain has shape (nparameters, niterations). Torch
        inputs remain on their current device and produce a Torch output.
    auto_burn_in : bool
        If True, then only use later half of samples provided.

    Returns
    -------
    psrf : numpy.array or torch.Tensor
        An array of shape (nparameters) that has the point estimates of the
        potential scale reduction factor.
    """

    tensor = _torch_chains(chains)
    if tensor is not None:
        return _torch_gelman_rubin(tensor, auto_burn_in)

    # remove first half of samples
    # this will have shape (nchains, nparameters, niterations)
    if auto_burn_in:
        _, _, niterations = numpy.array(chains).shape
        chains = numpy.array([chain[:, niterations // 2 + 1:]
                              for chain in chains])

    # get number of chains, parameters, and iterations
    chains = numpy.array(chains)
    nchains, nparameters, niterations = chains.shape

    # calculate the covariance matrix for each chain
    # this will have shape (nchains, nparameters, nparameters)
    chains_covs = numpy.array([numpy.cov(chain) for chain in chains])
    if nparameters == 1:
        chains_covs = chains_covs.reshape((nchains, 1, 1))

    # calculate W the within-chain variance
    # this will have shape (nparameters, nparameters)
    w = numpy.zeros(chains_covs[0].shape)
    for i, row in enumerate(chains_covs[0]):
        for j, _ in enumerate(row):
            w[i, j] = numpy.mean(chains_covs[:, i, j])
    if nparameters == 1:
        w = w.reshape((1, 1))

    # calculate B the between-chain variance
    # this will have shape (nparameters, nparameters)
    means = numpy.zeros((nparameters, nchains))
    for i, chain in enumerate(chains):
        means[:, i] = numpy.mean(chain, axis=1).transpose()
    b = niterations * numpy.cov(means)
    if nparameters == 1:
        b = b.reshape((1, 1))

    # get diagonal elements of W and B
    # these will have shape (nparameters)
    w_diag = numpy.diag(w)
    b_diag = numpy.diag(b)

    # get variance for each chain
    # this will have shape (nparameters, nchains)
    var = numpy.zeros((nparameters, nchains))
    for i, chain_cov in enumerate(chains_covs):
        var[:, i] = numpy.diag(chain_cov)

    # get mean of means
    # this will have shape (nparameters)
    mu_hat = numpy.mean(means, axis=1)

    # get variance of variances
    # this will have shape (nparameters)
    s = numpy.var(var, axis=1)

    # get V the combined variance of all chains
    # this will have shape (nparameters)
    v = ((niterations - 1.) * w_diag / niterations +
         (1. + 1. / nchains) * b_diag / niterations)

    # get factors in variance of V calculation
    # this will have shape (nparameters)
    k = 2 * b_diag**2 / (nchains - 1)
    mid_term = numpy.cov(
        var, means**2)[nparameters:2*nparameters, 0:nparameters].T
    end_term = numpy.cov(
        var, means)[nparameters:2*nparameters, 0:nparameters].T
    wb = niterations / nchains * numpy.diag(mid_term - 2 * mu_hat * end_term)

    # get variance of V
    # this will have shape (nparameters)
    var_v = (
        (niterations - 1.) ** 2 * s +
        (1. + 1. / nchains) ** 2 * k +
        2. * (niterations - 1.) * (1. + 1. / nchains) * wb
    ) / niterations**2

    # get degrees of freedom
    # this will have shape (nparameters)
    dof = (2. * v**2) / var_v

    # more degrees of freedom factors
    # this will have shape (nparameters)
    df_adj = (dof + 3.) / (dof + 1.)

    # estimate R
    # this will have shape (nparameters)
    r2_fixed = (niterations - 1.) / niterations
    r2_random = (1. + 1. / nchains) * (1. / niterations) * (b_diag / w_diag)
    r2_estimate = r2_fixed + r2_random

    # calculate PSRF the potential scale reduction factor
    # this will have shape (nparameters)
    psrf = numpy.sqrt(r2_estimate * df_adj)

    return psrf
