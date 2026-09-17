"""
Tools for maximum likelihood fits to single trigger statistic values

For some set of values above a threshold, e.g. trigger SNRs, the functions
in this module perform maximum likelihood fits with 1-sigma uncertainties
to various simple functional forms of PDF, all normalized to 1.
You can also obtain the fitted function and its (inverse) CDF and perform
a Kolmogorov-Smirnov test.

Usage:
# call the fit function directly if the threshold is known
alpha, sigma_alpha = fit_exponential(snrs, 5.5)

# apply a threshold explicitly
alpha, sigma_alpha = fit_above_thresh('exponential', snrs, thresh=6.25)

# let the code work out the threshold from the smallest value via the default thresh=None
alpha, sigma_alpha = fit_above_thresh('exponential', snrs)

# or only fit the largest N values, i.e. tail fitting
thresh = tail_threshold(snrs, N=500)
alpha, sigma_alpha = fit_above_thresh('exponential', snrs, thresh)

# obtain the fitted function directly
xvals = numpy.xrange(5.5, 10.5, 20)
exponential_fit = expfit(xvals, alpha, thresh)

# or access function by name
exponential_fit_1 = fit_fn('exponential', xvals, alpha, thresh)

# Use weighting factors to e.g. take decimation into account
alpha, sigma_alpha = fit_above_thresh('exponential', snrs, weights=weights)

# get the KS test statistic and p-value - see scipy.stats.kstest
ks_stat, ks_pval = KS_test('exponential', snrs, alpha, thresh)

"""

# Copyright T. Dent 2015 (thomas.dent@aei.mpg.de)
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.

import logging

import numpy
from scipy.stats import kstest

from pycbc.types import Array
from pycbc.types.backend import (
    backend_array,
    backend_matches_scheme,
    is_backend,
    wrap_backend_array,
)

logger = logging.getLogger("pycbc.events.trigger_fits")


def _torch_fit_tensors(*values):
    """Return compatible Torch tensors for trigger-fit arithmetic."""
    if not values:
        return None

    from pycbc import scheme

    if scheme.current_prefix() != "torch":
        return None

    import torch

    data = [backend_array(value) for value in values]
    torch_data = []
    for value in data:
        if is_backend(value, "torch"):
            torch_data.append(value)
    if not torch_data:
        return None

    first = torch_data[0]
    if not (
        first.is_floating_point()
        and backend_matches_scheme(first)
        and all(
            value.is_floating_point()
            and value.device == first.device
            and backend_matches_scheme(value)
            for value in torch_data
        )
    ):
        return None

    tensors = []
    for value in data:
        if is_backend(value, "torch"):
            tensors.append(value.to(dtype=first.dtype))
        elif isinstance(value, Array):
            return None
        else:
            try:
                host = numpy.asarray(value)
                if host.dtype.kind not in "fiu":
                    return None
                tensors.append(
                    torch.as_tensor(
                        host,
                        dtype=first.dtype,
                        device=first.device,
                    )
                )
            except (TypeError, ValueError, RuntimeError):
                return None

    return tuple(tensors)


def _torch_fit_result(input_value, tensor):
    """Wrap a Torch fit vector when its public input was a PyCBC Array."""
    if isinstance(input_value, Array):
        return Array(wrap_backend_array(tensor), copy=False)
    return tensor


def _torch_fitalpha(distr, vals, thresh, weights):
    """Evaluate a maximum-likelihood slope entirely with Torch."""
    import torch

    average = torch.sum(vals * weights) / torch.sum(weights)
    if distr == "exponential":
        return 1.0 / (average - thresh)
    if distr == "rayleigh":
        square_average = torch.sum(vals.square() * weights) / torch.sum(weights)
        return 2.0 / (square_average - thresh.square())
    if distr == "power":
        log_average = torch.sum(torch.log(vals / thresh) * weights)
        log_average = log_average / torch.sum(weights)
        return log_average.reciprocal() + 1.0
    raise KeyError(distr)


def _torch_fit_values(distr, xvals, alpha, thresh, *, cumulative):
    """Evaluate a fitted density or reverse CDF with Torch."""
    import torch

    below_threshold = xvals < thresh
    fit_xvals = torch.where(below_threshold, thresh, xvals)
    if cumulative:
        if distr == "exponential":
            values = torch.exp(-alpha * (fit_xvals - thresh))
        elif distr == "rayleigh":
            values = torch.exp(-alpha * (fit_xvals.square() - thresh.square()) / 2.0)
        elif distr == "power":
            values = fit_xvals ** (1.0 - alpha) * thresh ** (alpha - 1.0)
        else:
            raise KeyError(distr)
    elif distr == "exponential":
        values = alpha * torch.exp(-alpha * (fit_xvals - thresh))
    elif distr == "rayleigh":
        values = (
            alpha
            * fit_xvals
            * torch.exp(-alpha * (fit_xvals.square() - thresh.square()) / 2.0)
        )
    elif distr == "power":
        values = (alpha - 1.0) * fit_xvals ** (-alpha) * thresh ** (alpha - 1.0)
    else:
        raise KeyError(distr)

    return torch.where(below_threshold, torch.zeros_like(values), values)


def _ks_test_result(statistic, probability, location, sign):
    """Build the richest KS result supported by the installed SciPy."""
    try:
        from scipy.stats._stats_py import KstestResult
    except ImportError:
        return statistic, probability
    return KstestResult(
        statistic,
        probability,
        statistic_location=location,
        statistic_sign=sign,
    )


def _torch_ks_test(distr, vals, alpha, thresh):
    """Evaluate a one-sample, two-sided KS test on a Torch device."""
    torch_inputs = [vals, alpha]
    if thresh is not None:
        torch_inputs.append(thresh)
    tensors = _torch_fit_tensors(*torch_inputs)
    if tensors is None or tensors[0].ndim != 1:
        return None

    import torch
    from scipy.stats import distributions

    vals_t, alpha_t = tensors[:2]
    if alpha_t.numel() != 1:
        return None

    sample_size = vals_t.numel()
    if thresh is None:
        if sample_size == 0:
            raise ValueError("min() arg is an empty sequence")
        thresh_t = torch.min(vals_t)
    else:
        thresh_t = tensors[2]
        if thresh_t.numel() != 1:
            return None
        vals_t = vals_t[vals_t >= thresh_t]
        sample_size = vals_t.numel()

    if sample_size == 0 or torch.isnan(vals_t).any().item():
        nan = float("nan")
        return _ks_test_result(nan, nan, nan, nan)

    sorted_vals = torch.sort(vals_t).values
    reverse_cdf = _torch_fit_values(
        distr,
        sorted_vals,
        alpha_t,
        thresh_t,
        cumulative=True,
    )
    cdf = 1.0 - reverse_cdf
    if torch.isnan(cdf).any().item():
        nan = float("nan")
        return _ks_test_result(nan, nan, nan, nan)

    # SciPy forms its empirical CDF in float64 even when the fitted CDF was
    # evaluated from float32 inputs. MPS does not support float64 tensors.
    work_dtype = torch.float64 if sorted_vals.device.type != "mps" else cdf.dtype
    cdf = cdf.to(dtype=work_dtype)
    ranks = torch.arange(
        sample_size,
        dtype=work_dtype,
        device=sorted_vals.device,
    )
    dplus_values = (ranks + 1.0) / sample_size - cdf
    dminus_values = cdf - ranks / sample_size
    dplus, dplus_index = torch.max(dplus_values, dim=0)
    dminus, dminus_index = torch.max(dminus_values, dim=0)
    if (dplus > dminus).item():
        statistic_t = dplus
        location_index = dplus_index
        sign = 1
    else:
        statistic_t = dminus
        location_index = dminus_index
        sign = -1

    statistic = float(statistic_t.item())
    location = float(sorted_vals[location_index].item())
    probability = distributions.kstwo.sf(statistic, sample_size)
    probability = float(numpy.clip(probability, 0.0, 1.0))
    return _ks_test_result(statistic, probability, location, sign)


def exponential_fitalpha(vals, thresh, w):
    """
    Maximum likelihood estimator for the fit factor for
    an exponential decrease model
    """
    return 1.0 / (numpy.average(vals, weights=w) - thresh)


def rayleigh_fitalpha(vals, thresh, w):
    """
    Maximum likelihood estimator for the fit factor for
    a Rayleigh distribution of events
    """
    return 2.0 / (numpy.average(vals**2.0, weights=w) - thresh**2.0)


def power_fitalpha(vals, thresh, w):
    """
    Maximum likelihood estimator for the fit factor for
    a power law model
    """
    return numpy.average(numpy.log(vals / thresh), weights=w) ** -1.0 + 1.0


fitalpha_dict = {
    "exponential": exponential_fitalpha,
    "rayleigh": rayleigh_fitalpha,
    "power": power_fitalpha,
}

# measurement standard deviation = (-d^2 log L/d alpha^2)^(-1/2)
fitstd_dict = {
    "exponential": lambda weights, alpha: alpha / sum(weights) ** 0.5,
    "rayleigh": lambda weights, alpha: alpha / sum(weights) ** 0.5,
    "power": lambda weights, alpha: (alpha - 1.0) / sum(weights) ** 0.5,
}


def fit_above_thresh(distr, vals, thresh=None, weights=None):
    """
    Maximum likelihood fit for the coefficient alpha

    Fitting a distribution of discrete values above a given threshold.
    Exponential  p(x) = alpha exp(-alpha (x-x_t))
    Rayleigh     p(x) = alpha x exp(-alpha (x**2-x_t**2)/2)
    Power        p(x) = ((alpha-1)/x_t) (x/x_t)**-alpha
    Values below threshold will be discarded.
    If no threshold is specified the minimum sample value will be used.

    Parameters
    ----------
    distr : {'exponential', 'rayleigh', 'power'}
        Name of distribution
    vals : sequence of floats
        Values to fit
    thresh : float
        Threshold to apply before fitting; if None, use min(vals)
    weights: sequence of floats
        Weighting factors to use for the values when fitting.
        Default=None - all the same

    Returns
    -------
    alpha : float
        Fitted value
    sigma_alpha : float
        Standard error in fitted value
    """
    torch_inputs = [vals]
    if thresh is not None:
        torch_inputs.append(thresh)
    if weights is not None:
        torch_inputs.append(weights)
    tensors = _torch_fit_tensors(*torch_inputs)
    if tensors is not None:
        import torch

        vals_t = tensors[0]
        tensor_index = 1
        if thresh is None:
            thresh_t = torch.min(vals_t)
            above_thresh = torch.ones_like(vals_t, dtype=torch.bool)
        else:
            thresh_t = tensors[tensor_index]
            tensor_index += 1
            above_thresh = vals_t >= thresh_t
            vals_t = vals_t[above_thresh]
            if vals_t.numel() == 0:
                return -1.0, -1.0

        weights_t = tensors[tensor_index] if weights is not None else None
        if weights_t is not None:
            weights_t = weights_t[above_thresh]
        else:
            weights_t = torch.ones_like(vals_t)

        alpha = _torch_fitalpha(distr, vals_t, thresh_t, weights_t)
        if distr == "power":
            sigma_alpha = (alpha - 1.0) / torch.sqrt(torch.sum(weights_t))
        else:
            sigma_alpha = alpha / torch.sqrt(torch.sum(weights_t))
        return alpha, sigma_alpha

    vals = numpy.array(vals)
    if thresh is None:
        thresh = min(vals)
        above_thresh = numpy.ones_like(vals, dtype=bool)
    else:
        above_thresh = vals >= thresh
        if numpy.count_nonzero(above_thresh) == 0:
            # Nothing is above threshold - warn and return -1
            logger.warning(
                "No values are above the threshold, %.2f, maximum is %.2f.",
                thresh,
                vals.max(),
            )
            return -1.0, -1.0

        vals = vals[above_thresh]

    # Set up the weights
    if weights is not None:
        weights = numpy.array(weights)
        w = weights[above_thresh]
    else:
        w = numpy.ones_like(vals)

    alpha = fitalpha_dict[distr](vals, thresh, w)
    return alpha, fitstd_dict[distr](w, alpha)


# Variables:
# x: the trigger stat value(s) at which to evaluate the function
# a: slope parameter of the fit
# t: lower threshold stat value
fitfn_dict = {
    "exponential": lambda x, a, t: a * numpy.exp(-a * (x - t)),
    "rayleigh": lambda x, a, t: (a * x * numpy.exp(-a * (x**2 - t**2) / 2.0)),
    "power": lambda x, a, t: (a - 1.0) * x ** (-a) * t ** (a - 1.0),
}


def fit_fn(distr, xvals, alpha, thresh):
    """
    The fitted function normalized to 1 above threshold

    To normalize to a given total count multiply by the count.

    Parameters
    ----------
    xvals : sequence of floats
        Values where the function is to be evaluated
    alpha : float
        The fitted parameter
    thresh : float
        Threshold value applied to fitted values

    Returns
    -------
    fit : array of floats
        Fitted function at the requested xvals
    """
    tensors = _torch_fit_tensors(xvals, alpha, thresh)
    if tensors is not None:
        fit = _torch_fit_values(distr, *tensors, cumulative=False)
        return _torch_fit_result(xvals, fit)

    xvals = numpy.array(xvals)
    fit = fitfn_dict[distr](xvals, alpha, thresh)
    # set fitted values below threshold to 0
    numpy.putmask(fit, xvals < thresh, 0.0)
    return fit


cum_fndict = {
    "exponential": lambda x, alpha, t: numpy.exp(-alpha * (x - t)),
    "rayleigh": lambda x, alpha, t: numpy.exp(-alpha * (x**2.0 - t**2.0) / 2.0),
    "power": lambda x, alpha, t: x ** (1.0 - alpha) * t ** (alpha - 1.0),
}


def cum_fit(distr, xvals, alpha, thresh):
    """
    Integral of the fitted function above a given value (reverse CDF)

    The fitted function is normalized to 1 above threshold

    Parameters
    ----------
    xvals : sequence of floats
        Values where the function is to be evaluated
    alpha : float
        The fitted parameter
    thresh : float
        Threshold value applied to fitted values

    Returns
    -------
    cum_fit : array of floats
        Reverse CDF of fitted function at the requested xvals
    """
    tensors = _torch_fit_tensors(xvals, alpha, thresh)
    if tensors is not None:
        values = _torch_fit_values(distr, *tensors, cumulative=True)
        return _torch_fit_result(xvals, values)

    xvals = numpy.array(xvals)
    cum_fit = cum_fndict[distr](xvals, alpha, thresh)
    # set fitted values below threshold to 0
    numpy.putmask(cum_fit, xvals < thresh, 0.0)
    return cum_fit


def tail_threshold(vals, N=1000):
    """Determine a threshold above which there are N louder values"""
    tensors = _torch_fit_tensors(vals)
    if tensors is not None and tensors[0].ndim == 1:
        vals_t = tensors[0]
        if len(vals_t) < N:
            raise RuntimeError("Not enough input values to determine threshold")
        sorted_vals = vals_t.sort().values
        return sorted_vals[-N if N else 0]

    vals = numpy.array(vals)
    if len(vals) < N:
        raise RuntimeError("Not enough input values to determine threshold")
    vals.sort()
    return min(vals[-N:])


def KS_test(distr, vals, alpha, thresh=None):
    """
    Perform Kolmogorov-Smirnov test for fitted distribution

    Compare the given set of discrete values above a given threshold to the
    fitted distribution function.
    If no threshold is specified, the minimum sample value will be used.
    Returns the KS test statistic and its p-value: lower p means less
    probable under the hypothesis of a perfect fit

    Parameters
    ----------
    distr : {'exponential', 'rayleigh', 'power'}
        Name of distribution
    vals : sequence of floats
        Values to compare to fit
    alpha : float
        Fitted distribution parameter
    thresh : float
        Threshold to apply before fitting; if None, use min(vals)

    Returns
    -------
    D : float
        KS test statistic
    p-value : float
        p-value, assumed to be two-tailed
    """
    torch_result = _torch_ks_test(distr, vals, alpha, thresh)
    if torch_result is not None:
        return torch_result

    vals = numpy.array(vals)
    if thresh is None:
        thresh = min(vals)
    else:
        vals = vals[vals >= thresh]

    def cdf_fn(x):
        return 1 - cum_fndict[distr](x, alpha, thresh)

    return kstest(vals, cdf_fn)


def which_bin(par, minpar, maxpar, nbins, log=False):
    """
    Helper function

    Returns bin index where a parameter value belongs (from 0 through nbins-1)
    when dividing the range between minpar and maxpar equally into bins.

    Parameters
    ----------
    par : float
        Parameter value being binned
    minpar : float
        Minimum parameter value
    maxpar : float
        Maximum parameter value
    nbins : int
        Number of bins to use
    log : boolean
        If True, use log spaced bins

    Returns
    -------
    binind : int
        Bin index
    """
    assert par >= minpar and par <= maxpar
    if log:
        par, minpar, maxpar = numpy.log(par), numpy.log(minpar), numpy.log(maxpar)
    # par lies some fraction of the way between min and max
    if minpar != maxpar:
        frac = float(par - minpar) / float(maxpar - minpar)
    else:
        # if they are equal there is only one size 0 bin
        # must be in that bin
        frac = 0
    # binind then lies between 0 and nbins - 1
    binind = int(frac * nbins)
    # corner case
    if par == maxpar:
        binind = nbins - 1
    return binind
