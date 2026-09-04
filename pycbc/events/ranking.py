""" This module contains functions for calculating single-ifo ranking
statistic values
"""
from pycbc.types.backend import (
    backend_array, backend_matches_scheme, is_backend, wrap_backend_array,
)
import logging
import numpy

from pycbc.types import Array

logger = logging.getLogger('pycbc.events.ranking')


def _torch_ranking_dtype(tensor):
    """Return the highest supported ranking precision for a device."""
    import torch

    return torch.float32 if tensor.device.type == "mps" else torch.float64


def effsnr(snr, reduced_x2, fac=250.,
           **kwargs):  # pylint:disable=unused-argument
    """Calculate the effective SNR statistic. See (S5y1 paper) for definition.
    """
    tensors = _torch_ranking_tensors(
        snr, reduced_x2, coerce_host=True
    )
    if tensors is not None:
        dtype = _torch_ranking_dtype(tensors[0])
        snr_t, rchisq_t = (value.to(dtype=dtype) for value in tensors)
        values = (
            snr_t
            / (1.0 + snr_t ** 2 / fac) ** 0.25
            / rchisq_t ** 0.25
        )
        return Array(wrap_backend_array(values), copy=False)

    snr = numpy.array(snr, ndmin=1, dtype=numpy.float64)
    rchisq = numpy.array(reduced_x2, ndmin=1, dtype=numpy.float64)
    esnr = snr / (1 + snr ** 2 / fac) ** 0.25 / rchisq ** 0.25

    # If snr input is float, return a float. Otherwise return numpy array.
    if hasattr(snr, '__len__'):
        return esnr
    else:
        return esnr[0]


def _torch_newsnr(snr, reduced_x2, q, n):
    """Return newSNR on Torch when either vector already lives there."""
    tensors = _torch_ranking_tensors(
        snr, reduced_x2, coerce_host=True
    )
    if tensors is None:
        return None

    import torch

    dtype = _torch_ranking_dtype(tensors[0])
    nsnr = tensors[0].to(dtype=dtype)
    rchisq = tensors[1].to(dtype=dtype)
    reweight = (0.5 * (1.0 + rchisq ** (q / n))) ** (-1.0 / q)
    values = torch.where(rchisq > 1.0, nsnr * reweight, nsnr)
    return Array(wrap_backend_array(values), copy=False)


def _torch_ranking_tensors(*values, coerce_host=False):
    """Return compatible Torch tensors for ranking statistics."""
    if not values:
        return None

    from pycbc import scheme
    if scheme.current_prefix() != "torch":
        return None

    import torch

    data = [
        backend_array(value)
        for value in values
    ]
    torch_data = [
        value for value in data if is_backend(value, "torch")
    ]
    if not torch_data or (
        not coerce_host
        and len(torch_data) != len(data)
    ):
        return None

    first = torch_data[0]
    if not (
        all(backend_matches_scheme(value) for value in torch_data)
        and all(value.device == first.device for value in torch_data)
        and all(value.shape == first.shape for value in torch_data)
        and all(value.dtype == first.dtype for value in torch_data)
        and all(value.is_floating_point() for value in torch_data)
    ):
        return None

    tensors = []
    for value in data:
        if is_backend(value, "torch"):
            tensors.append(value)
            continue
        if isinstance(value, Array):
            return None
        try:
            host = numpy.asarray(value)
            if host.dtype.kind != "f" or host.shape != first.shape:
                return None
            tensors.append(
                torch.as_tensor(
                    host, dtype=first.dtype, device=first.device
                )
            )
        except (TypeError, ValueError, RuntimeError):
            return None

    return tuple(tensors)


def _torch_newsnr_sgveto(snr, reduced_x2, sgchisq, q, n):
    """Return the combined newSNR/sine-Gaussian statistic on Torch."""
    import torch

    dtype = _torch_ranking_dtype(snr)
    snr = snr.to(dtype=dtype)
    reduced_x2 = reduced_x2.to(dtype=dtype)
    sgchisq = sgchisq.to(dtype=dtype)
    reweight = (0.5 * (1.0 + reduced_x2 ** (q / n))) ** (-1.0 / q)
    values = torch.where(reduced_x2 > 1.0, snr * reweight, snr)
    return torch.where(
        sgchisq > 4.0,
        values / torch.sqrt(sgchisq / 4.0),
        values,
    )


def newsnr(snr, reduced_x2, q=6., n=2.,
           **kwargs):  # pylint:disable=unused-argument
    """Calculate the re-weighted SNR statistic ('newSNR') from given SNR and
    reduced chi-squared values. See http://arxiv.org/abs/1208.3491 for
    definition. Previous implementation in glue/ligolw/lsctables.py
    """
    torch_nsnr = _torch_newsnr(snr, reduced_x2, q, n)
    if torch_nsnr is not None:
        return torch_nsnr

    nsnr = numpy.array(snr, ndmin=1, dtype=numpy.float64)
    reduced_x2 = numpy.array(reduced_x2, ndmin=1, dtype=numpy.float64)

    # newsnr is only different from snr if reduced chisq > 1
    ind = numpy.where(reduced_x2 > 1.)[0]
    nsnr[ind] *= (0.5 * (1. + reduced_x2[ind] ** (q/n))) ** (-1./q)

    # If snr input is float, return a float. Otherwise return numpy array.
    if hasattr(snr, '__len__'):
        return nsnr
    else:
        return nsnr[0]


def newsnr_sgveto(snr, brchisq, sgchisq, **kwargs):
    """ Combined SNR derived from NewSNR and Sine-Gaussian Chisq"""
    tensors = _torch_ranking_tensors(
        snr, brchisq, sgchisq, coerce_host=True
    )
    if tensors is not None:
        values = _torch_newsnr_sgveto(
            *tensors,
            kwargs.get("q", 6.0),
            kwargs.get("n", 2.0),
        )
        return Array(wrap_backend_array(values), copy=False)

    nsnr = numpy.array(
        newsnr(
            snr,
            brchisq,
            **kwargs),
        ndmin=1)
    sgchisq = numpy.array(sgchisq, ndmin=1)
    t = numpy.array(sgchisq > 4, ndmin=1)
    if len(t):
        nsnr[t] = nsnr[t] / (sgchisq[t] / 4.0) ** 0.5

    # If snr input is float, return a float. Otherwise return numpy array.
    if hasattr(snr, '__len__'):
        return nsnr
    else:
        return nsnr[0]


def newsnr_sgveto_psdvar(snr, brchisq, sgchisq, psd_var_val,
                         min_expected_psdvar=0.65,
                         **kwargs):
    """ Combined SNR derived from SNR, reduced Allen chisq, sine-Gaussian chisq and
    PSD variation statistic"""
    tensors = _torch_ranking_tensors(
        snr, brchisq, sgchisq, psd_var_val, coerce_host=True
    )
    if tensors is not None:
        import torch

        snr_t, brchisq_t, sgchisq_t, psd_var_t = tensors
        psd_var_t = torch.where(
            psd_var_t < min_expected_psdvar,
            torch.ones_like(psd_var_t),
            psd_var_t,
        )
        values = _torch_newsnr_sgveto(
            snr_t * psd_var_t ** -0.5,
            brchisq_t * psd_var_t ** -1.0,
            sgchisq_t,
            kwargs.get("q", 6.0),
            kwargs.get("n", 2.0),
        )
        return Array(wrap_backend_array(values), copy=False)

    # If PSD var is lower than the 'minimum usually expected value' stop this
    # being used in the statistic. This low value might arise because a
    # significant fraction of the "short" PSD period was gated (for instance).
    psd_var_val = numpy.array(psd_var_val, copy=True)
    psd_var_val[psd_var_val < min_expected_psdvar] = 1.
    scaled_snr = snr * (psd_var_val ** -0.5)
    scaled_brchisq = brchisq * (psd_var_val ** -1.)
    nsnr = newsnr_sgveto(
        scaled_snr,
        scaled_brchisq,
        sgchisq,
        **kwargs
    )

    # If snr input is float, return a float. Otherwise return numpy array.
    if hasattr(snr, '__len__'):
        return nsnr
    else:
        return nsnr[0]


def newsnr_sgveto_psdvar_threshold(snr, brchisq, sgchisq, psd_var_val,
                                   min_expected_psdvar=0.65,
                                   brchisq_threshold=10.0,
                                   psd_var_val_threshold=10.0,
                                   **kwargs):
    """ newsnr_sgveto_psdvar with thresholds applied.

    This is the newsnr_sgveto_psdvar statistic with additional options
    to threshold on chi-squared or PSD variation.
    """
    tensors = _torch_ranking_tensors(
        snr, brchisq, sgchisq, psd_var_val, coerce_host=True
    )
    if tensors is not None:
        import torch

        snr_t, brchisq_t, sgchisq_t, psd_var_t = tensors
        bounded_psd = torch.where(
            psd_var_t < min_expected_psdvar,
            torch.ones_like(psd_var_t),
            psd_var_t,
        )
        values = _torch_newsnr_sgveto(
            snr_t * bounded_psd ** -0.5,
            brchisq_t * bounded_psd ** -1.0,
            sgchisq_t,
            kwargs.get("q", 6.0),
            kwargs.get("n", 2.0),
        )
        rejected = (
            (brchisq_t > brchisq_threshold)
            | (psd_var_t > psd_var_val_threshold)
        )
        values = torch.where(rejected, torch.ones_like(values), values)
        return Array(wrap_backend_array(values), copy=False)

    nsnr = newsnr_sgveto_psdvar(
        snr,
        brchisq,
        sgchisq,
        psd_var_val,
        min_expected_psdvar=min_expected_psdvar,
        **kwargs
    )
    nsnr = numpy.array(nsnr, ndmin=1)
    nsnr[brchisq > brchisq_threshold] = 1.
    nsnr[psd_var_val > psd_var_val_threshold] = 1.

    # If snr input is float, return a float. Otherwise return numpy array.
    if hasattr(snr, '__len__'):
        return nsnr
    else:
        return nsnr[0]


def newsnr_sgveto_psdvar_scaled(snr, brchisq, sgchisq, psd_var_val,
                                scaling=0.33, min_expected_psdvar=0.65,
                                **kwargs):
    """ Combined SNR derived from NewSNR, Sine-Gaussian Chisq and scaled PSD
    variation statistic. """
    tensors = _torch_ranking_tensors(
        snr, brchisq, sgchisq, psd_var_val, coerce_host=True
    )
    if tensors is not None:
        import torch

        snr_t, brchisq_t, sgchisq_t, psd_var_t = tensors
        values = _torch_newsnr_sgveto(
            snr_t,
            brchisq_t,
            sgchisq_t,
            kwargs.get("q", 6.0),
            kwargs.get("n", 2.0),
        )
        bounded_psd = torch.where(
            psd_var_t < min_expected_psdvar,
            torch.ones_like(psd_var_t),
            psd_var_t,
        )
        values = values / bounded_psd ** scaling
        return Array(wrap_backend_array(values), copy=False)

    nsnr = numpy.array(
        newsnr_sgveto(
            snr,
            brchisq,
            sgchisq,
            **kwargs),
        ndmin=1)
    psd_var_val = numpy.array(psd_var_val, ndmin=1, copy=True)
    psd_var_val[psd_var_val < min_expected_psdvar] = 1.

    # Default scale is 0.33 as tuned from analysis of data from O2 chunks
    nsnr = nsnr / psd_var_val ** scaling

    # If snr input is float, return a float. Otherwise return numpy array.
    if hasattr(snr, '__len__'):
        return nsnr
    else:
        return nsnr[0]


def newsnr_sgveto_psdvar_scaled_threshold(snr, bchisq, sgchisq, psd_var_val,
                                          threshold=2.0,
                                          **kwargs):
    """ Combined SNR derived from NewSNR and Sine-Gaussian Chisq, and
    scaled psd variation.
    """
    tensors = _torch_ranking_tensors(
        snr, bchisq, sgchisq, psd_var_val, coerce_host=True
    )
    if tensors is not None:
        import torch

        snr_t, bchisq_t, sgchisq_t, psd_var_t = tensors
        min_expected = kwargs.get("min_expected_psdvar", 0.65)
        scaling = kwargs.get("scaling", 0.33)
        bounded_psd = torch.where(
            psd_var_t < min_expected,
            torch.ones_like(psd_var_t),
            psd_var_t,
        )
        values = _torch_newsnr_sgveto(
            snr_t,
            bchisq_t,
            sgchisq_t,
            kwargs.get("q", 6.0),
            kwargs.get("n", 2.0),
        ) / bounded_psd ** scaling
        values = torch.where(
            bchisq_t > threshold,
            torch.ones_like(values),
            values,
        )
        return Array(wrap_backend_array(values), copy=False)

    nsnr = newsnr_sgveto_psdvar_scaled(
        snr,
        bchisq,
        sgchisq,
        psd_var_val,
        **kwargs
    )
    nsnr = numpy.array(nsnr, ndmin=1)
    nsnr[bchisq > threshold] = 1.

    # If snr input is float, return a float. Otherwise return numpy array.
    if hasattr(snr, '__len__'):
        return nsnr
    else:
        return nsnr[0]


def get_snr(trigs, **kwargs):  # pylint:disable=unused-argument
    """
    Return SNR from a trigs/dictionary object

    Parameters
    ----------
    trigs: dict of numpy.ndarrays, h5py group (or similar dict-like object)
        Dictionary-like object holding single detector trigger information.
        'snr' is a required key

    Returns
    -------
    numpy.ndarray or pycbc.types.Array
        Array of SNR values. Torch-backed trigger columns remain on their
        active device.
    """
    return _format_ranking_output(trigs['snr'][:])


def _format_ranking_output(values):
    """Preserve Torch-backed ranking values while retaining legacy output."""
    if _torch_ranking_tensors(values) is not None:
        return values.astype(numpy.float32)
    return numpy.array(values, ndmin=1, dtype=numpy.float32)


def get_newsnr(trigs, **kwargs):
    """
    Calculate newsnr ('reweighted SNR') for a trigs/dictionary object

    Parameters
    ----------
    trigs: dict of numpy.ndarrays, h5py group (or similar dict-like object)
        Dictionary-like object holding single detector trigger information.
        'chisq_dof', 'snr', and 'chisq' are required keys

    Returns
    -------
    numpy.ndarray or pycbc.types.Array
        Array of newsnr values. Torch-backed trigger columns remain on their
        active device.
    """
    dof = 2. * trigs['chisq_dof'][:] - 2.
    nsnr = newsnr(
        trigs['snr'][:],
        trigs['chisq'][:] / dof,
        **kwargs
    )
    return _format_ranking_output(nsnr)


def get_newsnr_sgveto(trigs, **kwargs):
    """
    Calculate newsnr re-weigthed by the sine-gaussian veto

    Parameters
    ----------
    trigs: dict of numpy.ndarrays, h5py group (or similar dict-like object)
        Dictionary-like object holding single detector trigger information.
        'chisq_dof', 'snr', 'sg_chisq' and 'chisq' are required keys

    Returns
    -------
    numpy.ndarray or pycbc.types.Array
        Array of newsnr values. Torch-backed trigger columns remain on their
        active device.
    """
    dof = 2. * trigs['chisq_dof'][:] - 2.
    nsnr_sg = newsnr_sgveto(
        trigs['snr'][:],
        trigs['chisq'][:] / dof,
        trigs['sg_chisq'][:],
        **kwargs
    )
    return _format_ranking_output(nsnr_sg)


def get_newsnr_sgveto_psdvar(trigs, **kwargs):
    """
    Calculate snr re-weighted by Allen chisq, sine-gaussian veto and
    psd variation statistic

    Parameters
    ----------
    trigs: dict of numpy.ndarrays
        Dictionary holding single detector trigger information.
    'chisq_dof', 'snr', 'chisq' and 'psd_var_val' are required keys

    Returns
    -------
    numpy.ndarray or pycbc.types.Array
        Array of newsnr values. Torch-backed trigger columns remain on their
        active device.
    """
    dof = 2. * trigs['chisq_dof'][:] - 2.
    nsnr_sg_psd = newsnr_sgveto_psdvar(
        trigs['snr'][:],
        trigs['chisq'][:] / dof,
        trigs['sg_chisq'][:],
        trigs['psd_var_val'][:],
        **kwargs
    )
    return _format_ranking_output(nsnr_sg_psd)


def get_newsnr_sgveto_psdvar_threshold(trigs, **kwargs):
    """
    Calculate newsnr re-weighted by the sine-gaussian veto and scaled
    psd variation statistic

    Parameters
    ----------
    trigs: dict of numpy.ndarrays
        Dictionary holding single detector trigger information.
    'chisq_dof', 'snr', 'chisq' and 'psd_var_val' are required keys

    Returns
    -------
    numpy.ndarray or pycbc.types.Array
        Array of newsnr values. Torch-backed trigger columns remain on their
        active device.
    """
    dof = 2. * trigs['chisq_dof'][:] - 2.
    nsnr_sg_psdt = newsnr_sgveto_psdvar_threshold(
        trigs['snr'][:], trigs['chisq'][:] / dof,
        trigs['sg_chisq'][:],
        trigs['psd_var_val'][:],
        **kwargs
    )
    return _format_ranking_output(nsnr_sg_psdt)


def get_newsnr_sgveto_psdvar_scaled(trigs, **kwargs):
    """
    Calculate newsnr re-weighted by the sine-gaussian veto and scaled
    psd variation statistic

    Parameters
    ----------
    trigs: dict of numpy.ndarrays
        Dictionary holding single detector trigger information.
    'chisq_dof', 'snr', 'chisq' and 'psd_var_val' are required keys

    Returns
    -------
    numpy.ndarray or pycbc.types.Array
        Array of newsnr values. Torch-backed trigger columns remain on their
        active device.
    """
    dof = 2. * trigs['chisq_dof'][:] - 2.
    nsnr_sg_psdscale = newsnr_sgveto_psdvar_scaled(
        trigs['snr'][:],
        trigs['chisq'][:] / dof,
        trigs['sg_chisq'][:],
        trigs['psd_var_val'][:],
        **kwargs
    )
    return _format_ranking_output(nsnr_sg_psdscale)


def get_newsnr_sgveto_psdvar_scaled_threshold(trigs, **kwargs):
    """
    Calculate newsnr re-weighted by the sine-gaussian veto and scaled
    psd variation statistic. A further threshold is applied to the
    reduced chisq.

    Parameters
    ----------
    trigs: dict of numpy.ndarrays
        Dictionary holding single detector trigger information.
    'chisq_dof', 'snr', 'chisq' and 'psd_var_val' are required keys

    Returns
    -------
    numpy.ndarray or pycbc.types.Array
        Array of newsnr values. Torch-backed trigger columns remain on their
        active device.
    """
    dof = 2. * trigs['chisq_dof'][:] - 2.
    nsnr_sg_psdt = newsnr_sgveto_psdvar_scaled_threshold(
        trigs['snr'][:],
        trigs['chisq'][:] / dof,
        trigs['sg_chisq'][:],
        trigs['psd_var_val'][:],
        **kwargs
    )
    return _format_ranking_output(nsnr_sg_psdt)


sngls_ranking_function_dict = {
    'snr': get_snr,
    'newsnr': get_newsnr,
    'new_snr': get_newsnr,
    'newsnr_sgveto': get_newsnr_sgveto,
    'newsnr_sgveto_psdvar': get_newsnr_sgveto_psdvar,
    'newsnr_sgveto_psdvar_threshold': get_newsnr_sgveto_psdvar_threshold,
    'newsnr_sgveto_psdvar_scaled': get_newsnr_sgveto_psdvar_scaled,
    'newsnr_sgveto_psdvar_scaled_threshold':
    get_newsnr_sgveto_psdvar_scaled_threshold,
}

# Lists of datasets required in the trigs object for each function
reqd_datasets = {}
reqd_datasets['snr'] = ['snr']
reqd_datasets['newsnr'] = reqd_datasets['snr'] + ['chisq', 'chisq_dof']
reqd_datasets['new_snr'] = reqd_datasets['newsnr']
reqd_datasets['newsnr_sgveto'] = reqd_datasets['newsnr'] + ['sg_chisq']
reqd_datasets['newsnr_sgveto_psdvar'] = \
    reqd_datasets['newsnr_sgveto'] + ['psd_var_val']
reqd_datasets['newsnr_sgveto_psdvar_threshold'] = \
    reqd_datasets['newsnr_sgveto_psdvar']
reqd_datasets['newsnr_sgveto_psdvar_scaled'] = \
    reqd_datasets['newsnr_sgveto_psdvar']
reqd_datasets['newsnr_sgveto_psdvar_scaled_threshold'] = \
    reqd_datasets['newsnr_sgveto_psdvar']


def get_sngls_ranking_from_trigs(trigs, statname, **kwargs):
    """
    Return ranking for all trigs given a statname.

    Compute the single-detector ranking for a list of input triggers for a
    specific statname.

    Parameters
    -----------
    trigs: dict of numpy.ndarrays, SingleDetTriggers or ReadByTemplate
        Dictionary holding single detector trigger information.
    statname:
        The statistic to use.
    """
    # Identify correct function
    try:
        sngl_func = sngls_ranking_function_dict[statname]
    except KeyError as exc:
        err_msg = 'Single-detector ranking {} not recognized'.format(statname)
        raise ValueError(err_msg) from exc

    # NOTE: In the sngl_funcs all the kwargs are explicitly stated, so any
    #       kwargs sent here must be known to the function.
    return sngl_func(trigs, **kwargs)
