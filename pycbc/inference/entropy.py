""" The module contains functions for calculating the
Kullback-Leibler divergence.
"""

import operator

import numpy
from scipy import stats


def _torch_entropy(pk, qk=None, base=numpy.e):
    """Compute entropy for Torch PDFs without importing Torch eagerly."""
    values = (pk,) if qk is None else (pk, qk)
    if not any(
            type(value).__module__.split('.', 1)[0] == 'torch'
            for value in values):
        return None

    import torch

    reference = next(
        value for value in values if isinstance(value, torch.Tensor)
    )
    reference_dtype = (
        reference.dtype if reference.is_floating_point()
        else torch.get_default_dtype()
    )
    tensors = []
    for value in values:
        tensor = (
            value if isinstance(value, torch.Tensor) else torch.as_tensor(
                value, device=reference.device, dtype=reference_dtype
            )
        )
        if tensor.is_complex():
            raise TypeError("probability densities must be real")
        if not tensor.is_floating_point():
            tensor = tensor.to(dtype=torch.get_default_dtype())
        tensors.append(tensor)

    if base is not None and base <= 0:
        raise ValueError("`base` must be a positive number or `None`.")

    pk = tensors[0]
    pk = pk / pk.sum(dim=0, keepdim=True)
    if qk is None:
        positive = pk > 0
        safe_pk = torch.where(positive, pk, torch.ones_like(pk))
        terms = torch.where(
            positive, -pk * torch.log(safe_pk), torch.zeros_like(pk)
        )
        terms = torch.where(
            pk < 0, torch.full_like(pk, -torch.inf), terms
        )
        terms = torch.where(
            torch.isnan(pk), torch.full_like(pk, torch.nan), terms
        )
    else:
        pk, qk = torch.broadcast_tensors(pk, tensors[1])
        qk = qk / qk.sum(dim=0, keepdim=True)
        positive = (pk > 0) & (qk > 0)
        safe_pk = torch.where(positive, pk, torch.ones_like(pk))
        safe_qk = torch.where(positive, qk, torch.ones_like(qk))
        terms = torch.full_like(pk, torch.inf)
        terms = torch.where(
            positive, pk * torch.log(safe_pk / safe_qk), terms
        )
        terms = torch.where(
            (pk == 0) & (qk >= 0), torch.zeros_like(terms), terms
        )
        terms = torch.where(
            torch.isnan(pk) | torch.isnan(qk),
            torch.full_like(terms, torch.nan), terms,
        )

    result = terms.sum(dim=0)
    if base is not None:
        result = result / torch.log(result.new_tensor(base))
    return result


def _torch_histogram_pdf(samples, bins, hist_range):
    """Return an on-device equal-width histogram PDF when supported."""
    if type(samples).__module__.split('.', 1)[0] != 'torch':
        return None

    try:
        bins = operator.index(bins)
    except TypeError:
        # NumPy's named/adaptive bin estimators remain the legacy path.
        return None

    import torch

    if not isinstance(samples, torch.Tensor):
        return None
    if samples.is_complex():
        raise TypeError("histogram samples must be real")
    if not samples.is_floating_point():
        samples = samples.to(dtype=torch.get_default_dtype())
    if hist_range is not None:
        hist_range = tuple(float(bound) for bound in hist_range)
    pdf, _ = torch.histogram(
        samples, bins=bins, range=hist_range, density=True
    )
    return pdf


def _torch_kde_evaluate(samples, points, bandwidth,
                        max_elements=2_000_000):
    """Evaluate a one-dimensional Gaussian KDE in bounded-size chunks."""
    import torch

    chunk_size = max(1, max_elements // samples.numel())
    normalization = bandwidth * torch.sqrt(
        bandwidth.new_tensor(2.0 * numpy.pi)
    )
    densities = []
    for start in range(0, points.numel(), chunk_size):
        point_chunk = points[start:start + chunk_size]
        scaled = (
            point_chunk[:, None] - samples[None, :]
        ) / bandwidth
        densities.append(
            torch.exp(-0.5 * scaled.square()).mean(dim=1) / normalization
        )
    return torch.cat(densities)


def _torch_kde_pdf(samples):
    """Sample and evaluate a one-dimensional Scott-bandwidth Torch KDE."""
    if type(samples).__module__.split('.', 1)[0] != 'torch':
        return None

    import torch

    if not isinstance(samples, torch.Tensor):
        return None
    if samples.ndim != 1:
        raise ValueError("KDE samples must be one-dimensional")
    if samples.is_complex():
        raise TypeError("KDE samples must be real")
    if not samples.is_floating_point():
        samples = samples.to(dtype=torch.get_default_dtype())
    if samples.numel() < 2:
        raise ValueError("KDE requires multiple samples")
    if not bool(torch.isfinite(samples).all()):
        raise ValueError("KDE samples must be finite")

    # scipy.stats.gaussian_kde uses Scott's rule and the unbiased sample
    # covariance by default. In one dimension this reduces to a scalar
    # bandwidth of std(samples) * n ** (-1/5).
    scott_factor = samples.new_tensor(samples.numel() ** (-1.0 / 5.0))
    bandwidth = samples.std(unbiased=True) * scott_factor
    if not bool(torch.isfinite(bandwidth)) or not bool(bandwidth > 0):
        raise numpy.linalg.LinAlgError(
            "KDE covariance is singular; samples must have nonzero variance"
        )

    npts = max(10_000, samples.numel())
    indices = torch.randint(
        samples.numel(), (npts,), device=samples.device
    )
    noise = torch.randn(
        npts, device=samples.device, dtype=samples.dtype
    )
    points = samples[indices] + bandwidth * noise
    return _torch_kde_evaluate(samples, points, bandwidth)


def check_hist_params(samples, hist_min, hist_max, hist_bins):
    """ Checks that the bound values given for the histogram are consistent,
    returning the range if they are or raising an error if they are not.
    Also checks that if hist_bins is a str, it corresponds to a method
    available in numpy.histogram

    Parameters
    ----------
    samples : numpy.array
        Set of samples to get the min/max if only one of the bounds is given.
    hist_min : numpy.float64
        Minimum value for the histogram.
    hist_max : numpy.float64
        Maximum value for the histogram.
    hist_bins: int or str
        If int, number of equal-width bins to use in numpy.histogram. If str,
        it should be one of the methods to calculate the optimal bin width
        available in numpy.histogram: ['auto', 'fd', 'doane', 'scott', 'stone',
        'rice', 'sturges', 'sqrt']. Default is 'fd' (Freedman Diaconis
        Estimator). This option will be ignored if `kde=True`.

    Returns
    -------
    hist_range : tuple or None
        The bounds (hist_min, hist_max) or None.
    hist_bins : int or str
        Number of bins or method for optimal width bin calculation.
    """

    hist_methods = ['auto', 'fd', 'doane', 'scott', 'stone', 'rice',
                    'sturges', 'sqrt']
    if not hist_bins:
        hist_bins = 'fd'
    elif isinstance(hist_bins, str) and hist_bins not in hist_methods:
        raise ValueError('Method for calculating bins width must be one of'
                         ' {}'.format(hist_methods))

    # No bounds given, return None
    if not hist_min and not hist_max:
        return None, hist_bins

    # One of the bounds is missing
    if hist_min and not hist_max:
        hist_max = samples.max()
    elif hist_max and not hist_min:
        hist_min = samples.min()
    # Both bounds given
    elif hist_min and hist_max and hist_min >= hist_max:
        raise ValueError('hist_min must be lower than hist_max.')

    hist_range = (hist_min, hist_max)

    return hist_range, hist_bins


def compute_pdf(samples, method, bins, hist_min, hist_max):
    """ Computes the probability density function for a set of samples.

    Parameters
    ----------
    samples : numpy.array or torch.Tensor
        Set of samples to calculate the pdf.
    method : str
        Method to calculate the pdf. Options are 'kde' for the Kernel Density
        Estimator, and 'hist' to use numpy.histogram
    bins : str or int, optional
        This option will be ignored if method is `kde`.
        If int, number of equal-width bins to use when calculating probability
        density function from a set of samples of the distribution. If str, it
        should be one of the methods to calculate the optimal bin width
        available in numpy.histogram: ['auto', 'fd', 'doane', 'scott', 'stone',
        'rice', 'sturges', 'sqrt']. Default is 'fd' (Freedman Diaconis
        Estimator).
    hist_min : numpy.float64, optional
        Minimum of the distributions' values to use. This will be ignored if
        `kde=True`.
    hist_max : numpy.float64, optional
        Maximum of the distributions' values to use. This will be ignored if
        `kde=True`.

    Returns
    -------
    pdf : numpy.array or torch.Tensor
        Discrete probability distribution calculated from samples. Torch
        samples using one-dimensional KDE or an integer number of histogram
        bins stay on-device.
    """

    if method == 'kde':
        pdf = _torch_kde_pdf(samples)
        if pdf is None:
            samples_kde = stats.gaussian_kde(samples)
            npts = 10000 if len(samples) <= 10000 else len(samples)
            draw = samples_kde.resample(npts)
            pdf = samples_kde.evaluate(draw)
    elif method == 'hist':
        hist_range, hist_bins = check_hist_params(samples, hist_min,
                                                  hist_max, bins)
        pdf = _torch_histogram_pdf(samples, hist_bins, hist_range)
        if pdf is None:
            pdf, _ = numpy.histogram(samples, bins=hist_bins,
                                     range=hist_range, density=True)
    else:
        raise ValueError('Method not recognized.')

    return pdf


def entropy(pdf1, base=numpy.e):
    """ Computes the information entropy for a single parameter
    from one probability density function.

    Parameters
    ----------
    pdf1 : numpy.array or torch.Tensor
        Probability density function. Torch inputs are evaluated on their
        current device.
    base : {numpy.e, numpy.float64}, optional
        The logarithmic base to use (choose base 2 for information measured
        in bits, default is nats).

    Returns
    -------
    numpy.float64 or torch.Tensor
        The information entropy value. A Torch input returns a Torch tensor.
    """

    torch_result = _torch_entropy(pdf1, base=base)
    if torch_result is not None:
        return torch_result
    return stats.entropy(pdf1, base=base)


def kl(samples1, samples2, pdf1=False, pdf2=False, kde=False,
       bins=None, hist_min=None, hist_max=None, base=numpy.e):
    """ Computes the Kullback-Leibler divergence for a single parameter
    from two distributions.

    Parameters
    ----------
    samples1 : numpy.array or torch.Tensor
        Samples or probability density function (for the latter must also set
        `pdf1=True`). One-dimensional Torch KDE samples and Torch histogram
        samples with integer `bins` stay on-device; direct Torch PDFs always
        stay on-device.
    samples2 : numpy.array or torch.Tensor
        Samples or probability density function (for the latter must also set
        `pdf2=True`). One-dimensional Torch KDE samples and Torch histogram
        samples with integer `bins` stay on-device; direct Torch PDFs always
        stay on-device.
    pdf1 : bool
        Set to `True` if `samples1` is a probability density funtion already.
    pdf2 : bool
        Set to `True` if `samples2` is a probability density funtion already.
    kde : bool
        Set to `True` if at least one of `pdf1` or `pdf2` is `False` to
        estimate the probability density function using kernel density
        estimation (KDE).
    bins : int or str, optional
        If int, number of equal-width bins to use when calculating probability
        density function from a set of samples of the distribution. If str, it
        should be one of the methods to calculate the optimal bin width
        available in numpy.histogram: ['auto', 'fd', 'doane', 'scott', 'stone',
        'rice', 'sturges', 'sqrt']. Default is 'fd' (Freedman Diaconis
        Estimator). This option will be ignored if `kde=True`.
    hist_min : numpy.float64
        Minimum of the distributions' values to use. This will be ignored if
        `kde=True`.
    hist_max : numpy.float64
        Maximum of the distributions' values to use. This will be ignored if
        `kde=True`.
    base : numpy.float64
        The logarithmic base to use (choose base 2 for information measured
        in bits, default is nats).

    Returns
    -------
    numpy.float64 or torch.Tensor
        The Kullback-Leibler divergence value. Direct Torch PDF inputs and
        one-dimensional Torch KDE or integer-bin histogram samples return a
        Torch tensor.
    """
    if pdf1 and pdf2 and kde:
        raise ValueError('KDE can only be used when at least one of pdf1 or '
                         'pdf2 is False.')

    if pdf1 and pdf2:
        torch_result = _torch_entropy(samples1, samples2, base=base)
        if torch_result is not None:
            return torch_result

    sample_groups = {'P': (samples1, pdf1), 'Q': (samples2, pdf2)}
    pdfs = {}
    for n in sample_groups:
        samples, pdf = sample_groups[n]
        if pdf:
            pdfs[n] = samples
        else:
            method = 'kde' if kde else 'hist'
            pdfs[n] = compute_pdf(samples, method, bins, hist_min, hist_max)

    torch_result = _torch_entropy(pdfs['P'], pdfs['Q'], base=base)
    if torch_result is not None:
        return torch_result
    return stats.entropy(pdfs['P'], qk=pdfs['Q'], base=base)


def js(samples1, samples2, kde=False, bins=None, hist_min=None, hist_max=None,
       base=numpy.e):
    """ Computes the Jensen-Shannon divergence for a single parameter
    from two distributions.

    Parameters
    ----------
    samples1 : numpy.array or torch.Tensor
        Samples.
    samples2 : numpy.array or torch.Tensor
        Samples.
    kde : bool
        Set to `True` to estimate the probability density function using
        kernel density estimation (KDE).
    bins : int or str, optional
        If int, number of equal-width bins to use when calculating probability
        density function from a set of samples of the distribution. If str, it
        should be one of the methods to calculate the optimal bin width
        available in numpy.histogram: ['auto', 'fd', 'doane', 'scott', 'stone',
        'rice', 'sturges', 'sqrt']. Default is 'fd' (Freedman Diaconis
        Estimator). This option will be ignored if `kde=True`.
    hist_min : numpy.float64
        Minimum of the distributions' values to use. This will be ignored if
        `kde=True`.
    hist_max : numpy.float64
        Maximum of the distributions' values to use. This will be ignored if
        `kde=True`.
    base : numpy.float64
        The logarithmic base to use (choose base 2 for information measured
        in bits, default is nats).

    Returns
    -------
    numpy.float64 or torch.Tensor
        The Jensen-Shannon divergence value. One-dimensional Torch KDE or
        integer-bin histogram samples return a Torch tensor on the input
        device.
    """

    sample_groups = {'P': samples1, 'Q': samples2}
    pdfs = {}
    for n in sample_groups:
        samples = sample_groups[n]
        method = 'kde' if kde else 'hist'
        pdfs[n] = compute_pdf(samples, method, bins, hist_min, hist_max)

    pdfs['M'] = (1./2) * (pdfs['P'] + pdfs['Q'])

    js_div = 0
    for pdf in (pdfs['P'], pdfs['Q']):
        js_div += (1./2) * kl(pdf, pdfs['M'], pdf1=True, pdf2=True, base=base)

    return js_div
