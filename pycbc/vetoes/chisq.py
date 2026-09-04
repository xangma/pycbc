# Copyright (C) 2012  Alex Nitz
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


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
from pycbc.types.backend import (
    backend_array, wrap_backend_array,
)
import numpy, logging, math, pycbc.fft

from pycbc.types import (
    Array,
    TimeSeries,
    complex_same_precision_as,
    real_same_precision_as,
    zeros,
)
from pycbc.filter import sigmasq_series, make_frequency_series, matched_filter_core, get_cutoff_indices
from pycbc.scheme import TorchScheme, mgr, schemed
import pycbc.pnutils

BACKEND_PREFIX="pycbc.vetoes.chisq_"


def _torch_tensor(value):
    """Return the tensor stored by a Torch-backed PyCBC array, if any."""
    return backend_array(value, "torch")


def _as_torch_tensor(value, reference, dtype=None):
    """Move small trigger metadata to the device without copying bulk data."""
    import torch

    tensor = _torch_tensor(value)
    if tensor is None:
        return torch.as_tensor(
            value, device=reference.device, dtype=dtype
        )
    if dtype is None:
        return tensor.to(device=reference.device)
    return tensor.to(device=reference.device, dtype=dtype)


def _torch_array(tensor):
    """Wrap a tensor as a PyCBC Array without a host copy."""

    return Array(wrap_backend_array(tensor), copy=False)


def _chisq_zeros(corr, length):
    """Allocate chi-squared output alongside its correlation array."""
    corr_tensor = _torch_tensor(corr)
    if corr_tensor is None:
        return numpy.zeros(length, dtype=numpy.float32)

    import torch

    return _torch_array(
        torch.zeros(
            length, device=corr_tensor.device, dtype=torch.float32
        )
    )


def _chisq_dof_array(reference, dof, length):
    """Allocate chi-squared degrees of freedom beside the statistic."""
    reference_tensor = _torch_tensor(reference)
    if reference_tensor is None:
        return numpy.repeat(dof, length)

    import torch

    return _torch_array(
        torch.full(
            (length,),
            int(dof),
            device=reference_tensor.device,
            dtype=torch.int64,
        )
    )


def _copy_chisq_masked(output, mask, values):
    """Scatter selected chi-squared values without leaving Torch."""
    output_tensor = _torch_tensor(output)
    if output_tensor is None:
        output[mask] = values
        return

    import torch

    mask_tensor = _as_torch_tensor(mask, output_tensor, dtype=torch.bool)
    values_tensor = _as_torch_tensor(
        values, output_tensor, dtype=output_tensor.dtype
    )
    output_tensor[mask_tensor] = values_tensor


def _torch_multiply_and_add(array, other, factor):
    """Call the Torch backend with a device scalar coefficient."""
    from pycbc.types import array_torch

    data = array_torch.multiply_and_add(
        array, wrap_backend_array(other), factor
    )
    return array._return(data)

def power_chisq_bins_from_sigmasq_series(sigmasq_series, num_bins, kmin, kmax):
    """Returns bins of equal power for use with the chisq functions

    Parameters
    ----------

    sigmasq_series: FrequencySeries
        A frequency series containing the cumulative power of a filter template
        preweighted by a psd.
    num_bins: int
        The number of chisq bins to calculate.
    kmin: int
        DOCUMENTME
    kmax: int
        DOCUMENTME

    Returns
    -------

    bins: List of ints
        A list of the edges of the chisq bins is returned.
    """
    tensor = backend_array(sigmasq_series, "torch")
    if tensor is not None:
        import torch

        sigmasq = tensor[kmax - 1]
        edge_vec = torch.arange(
            num_bins, dtype=tensor.dtype, device=tensor.device
        ) * sigmasq / num_bins
        bins = torch.searchsorted(
            tensor[kmin:kmax], edge_vec, right=True
        ) + kmin
        return numpy.asarray(
            [*bins.to(device="cpu").tolist(), kmax], dtype=numpy.int64
        )

    sigmasq = sigmasq_series[kmax - 1]
    edge_vec = numpy.arange(0, num_bins) * sigmasq / num_bins
    bins = numpy.searchsorted(sigmasq_series[kmin:kmax], edge_vec, side='right')
    bins += kmin
    return numpy.append(bins, kmax)


def power_chisq_bins(htilde, num_bins, psd, low_frequency_cutoff=None,
                     high_frequency_cutoff=None):
    """Returns bins of equal power for use with the chisq functions

    Parameters
    ----------

    htilde: FrequencySeries
        A frequency series containing the template waveform
    num_bins: int
        The number of chisq bins to calculate.
    psd: FrequencySeries
        A frequency series containing the psd. Its length must be commensurate
        with the template waveform.
    low_frequency_cutoff: {None, float}, optional
        The low frequency cutoff to apply
    high_frequency_cutoff: {None, float}, optional
        The high frequency cutoff to apply

    Returns
    -------

    bins: List of ints
        A list of the edges of the chisq bins is returned.
    """
    sigma_vec = sigmasq_series(
        htilde, psd, low_frequency_cutoff, high_frequency_cutoff
    )
    kmin, kmax = get_cutoff_indices(low_frequency_cutoff,
                                    high_frequency_cutoff,
                                    htilde.delta_f,
                                    (len(htilde)-1)*2)
    return power_chisq_bins_from_sigmasq_series(sigma_vec, num_bins, kmin, kmax)


@schemed(BACKEND_PREFIX)
def chisq_accum_bin(chisq, q):
    err_msg = "This function is a stub that should be overridden using the "
    err_msg += "scheme. You shouldn't be seeing this error!"
    raise ValueError(err_msg)

@schemed(BACKEND_PREFIX)
def shift_sum(v1, shifts, bins):
    """ Calculate the time shifted sum of the FrequencySeries
    """
    err_msg = "This function is a stub that should be overridden using the "
    err_msg += "scheme. You shouldn't be seeing this error!"
    raise ValueError(err_msg)


def power_chisq_at_points_from_precomputed(corr, snr, snr_norm, bins, indices):
    """Calculate the chisq timeseries from precomputed values for only select points.

    This function calculates the chisq at each point by explicitly time shifting
    and summing each bin. No FFT is involved.

    Parameters
    ----------
    corr: FrequencySeries
        The product of the template and data in the frequency domain.
    snr: numpy.ndarray
        The unnormalized array of snr values at only the selected points in `indices`.
    snr_norm: float
        The normalization of the snr (EXPLAINME : refer to Findchirp paper?)
    bins: List of integers
        The edges of the equal power bins
    indices: Array
        The indices where we will calculate the chisq. These must be relative
        to the given `corr` series.

    Returns
    -------
    chisq: Array
        An array containing only the chisq at the selected points.
    """
    if isinstance(mgr.state, TorchScheme):
        from .chisq_torch import (
            power_chisq_at_points_from_precomputed as _torch_impl,
        )

        return _torch_impl(corr, snr, snr_norm, bins, indices)

    num_bins = len(bins) - 1
    chisq = shift_sum(corr, indices, bins) # pylint:disable=assignment-from-no-return
    return (chisq * num_bins - (snr.conj() * snr).real) * (snr_norm ** 2.0)

_q_l = None
_qtilde_l = None
_chisq_l = None
def power_chisq_from_precomputed(corr, snr, snr_norm, bins, indices=None, return_bins=False):
    """Calculate the chisq timeseries from precomputed values.

    This function calculates the chisq at all times by performing an
    inverse FFT of each bin.

    Parameters
    ----------

    corr: FrequencySeries
        The produce of the template and data in the frequency domain.
    snr: TimeSeries
        The unnormalized snr time series.
    snr_norm:
        The snr normalization factor (true snr = snr * snr_norm) EXPLAINME - define 'true snr'?
    bins: List of integers
        The edges of the chisq bins.
    indices: {Array, None}, optional
        Index values into snr that indicate where to calculate
        chisq values. If none, calculate chisq for all possible indices.
    return_bins: {boolean, False}, optional
        Return a list of the SNRs for each chisq bin.

    Returns
    -------
    chisq: TimeSeries
    """
    # Get workspace memory
    global _q_l, _qtilde_l, _chisq_l

    bin_snrs = []

    if _q_l is None or len(_q_l) != len(snr):
        q = zeros(len(snr), dtype=complex_same_precision_as(snr))
        qtilde = zeros(len(snr), dtype=complex_same_precision_as(snr))
        _q_l = q
        _qtilde_l = qtilde
    else:
        q = _q_l
        qtilde = _qtilde_l

    if indices is not None:
        snr = snr.take(indices)

    if _chisq_l is None or len(_chisq_l) < len(snr):
        chisq = zeros(len(snr), dtype=real_same_precision_as(snr))
        _chisq_l = chisq
    else:
        chisq = _chisq_l[0:len(snr)]
        chisq.clear()

    num_bins = len(bins) - 1

    for j in range(num_bins):
        k_min = int(bins[j])
        k_max = int(bins[j+1])

        qtilde[k_min:k_max] = corr[k_min:k_max]
        pycbc.fft.ifft(qtilde, q)
        qtilde[k_min:k_max].clear()

        if return_bins:
            bin_snrs.append(TimeSeries(q  * snr_norm *  num_bins ** 0.5,
                                      delta_t=snr.delta_t,
                                      epoch=snr.start_time))

        if indices is not None:
            chisq_accum_bin(chisq, q.take(indices))
        else:
            chisq_accum_bin(chisq, q)

    chisq = (chisq * num_bins - snr.squared_norm()) * (snr_norm ** 2.0)

    if indices is None:
        chisq = TimeSeries(chisq, delta_t=snr.delta_t, epoch=snr.start_time, copy=False)

    if return_bins:
        return chisq, bin_snrs
    else:
        return chisq


def fastest_power_chisq_at_points(corr, snr, snrv, snr_norm, bins, indices):
    """Calculate the chisq values for only selected points.

    This function looks at the number of points to be evaluated and selects
    the fastest method (FFT, or direct time shift and sum). In either case,
    only the selected points are returned.

    Parameters
    ----------
    corr: FrequencySeries
        The product of the template and data in the frequency domain.
    snr: Array
        The unnormalized snr
    snr_norm: float
        The snr normalization factor  --- EXPLAINME
    bins: List of integers
        The edges of the equal power bins
    indices: Array
        The indices where we will calculate the chisq. These must be relative
        to the given `snr` series.

    Returns
    -------
    chisq: Array
        An array containing only the chisq at the selected points.
    """
    import pycbc.scheme
    if isinstance(pycbc.scheme.mgr.state, pycbc.scheme.CPUScheme):
        # We don't have that many points so do the direct time shift.
        return power_chisq_at_points_from_precomputed(corr, snrv,
                                                      snr_norm, bins, indices)
    else:
        # We have a lot of points so it is faster to use the fourier transform
        return power_chisq_from_precomputed(corr, snr, snr_norm, bins,
                                            indices=indices)


def power_chisq(template, data, num_bins, psd,
                low_frequency_cutoff=None,
                high_frequency_cutoff=None,
                return_bins=False):
    """Calculate the chisq timeseries

    Parameters
    ----------
    template: FrequencySeries or TimeSeries
        A time or frequency series that contains the filter template.
    data: FrequencySeries or TimeSeries
        A time or frequency series that contains the data to filter. The length
        must be commensurate with the template.
        (EXPLAINME - does this mean 'the same as' or something else?)
    num_bins: int
        The number of frequency bins used for chisq. The number of statistical
        degrees of freedom ('dof') is 2*num_bins-2.
    psd: FrequencySeries
        The psd of the data.
    low_frequency_cutoff: {None, float}, optional
        The low frequency cutoff for the filter
    high_frequency_cutoff: {None, float}, optional
        The high frequency cutoff for the filter
    return_bins: {boolean, False}, optional
        Return a list of the individual chisq bins

    Returns
    -------
    chisq: TimeSeries
        TimeSeries containing the chisq values for all times.
    """
    htilde = make_frequency_series(template)
    stilde = make_frequency_series(data)

    bins = power_chisq_bins(htilde, num_bins, psd, low_frequency_cutoff,
                            high_frequency_cutoff)
    corra = zeros((len(htilde) - 1) * 2, dtype=htilde.dtype)
    total_snr, corr, tnorm = matched_filter_core(htilde, stilde, psd,
                           low_frequency_cutoff, high_frequency_cutoff,
                           corr_out=corra)

    return power_chisq_from_precomputed(corr, total_snr, tnorm, bins, return_bins=return_bins)


class SingleDetPowerChisq(object):
    """Class that handles precomputation and memory management for efficiently
    running the power chisq in a single detector inspiral analysis.
    """
    def __init__(self, num_bins=0, snr_threshold=None):
        if not (num_bins == "0" or num_bins == 0):
            self.do = True
            self.column_name = "chisq"
            self.table_dof_name = "chisq_dof"
            self.num_bins = num_bins
        else:
            self.do = False
        self.snr_threshold = snr_threshold

    @staticmethod
    def parse_option(row, arg):
        safe_dict = {'max': max, 'min': min}
        safe_dict.update(row.__dict__)
        safe_dict.update(math.__dict__)
        safe_dict.update(pycbc.pnutils.__dict__)
        return eval(arg, {"__builtins__":None}, safe_dict)

    def cached_chisq_bins(self, template, psd):
        from pycbc.opt import LimitedSizeDict

        key = id(psd)
        if not hasattr(psd, '_chisq_cached_key'):
            psd._chisq_cached_key = {}

        if not hasattr(template, '_bin_cache'):
            template._bin_cache = LimitedSizeDict(size_limit=2**2)

        if key not in template._bin_cache or id(template.params) not in psd._chisq_cached_key:
            psd._chisq_cached_key[id(template.params)] = True
            num_bins = int(self.parse_option(template, self.num_bins))

            if hasattr(psd, 'sigmasq_vec') and \
                    template.approximant in psd.sigmasq_vec:
                kmin = int(template.f_lower / psd.delta_f)
                kmax = template.end_idx
                bins = power_chisq_bins_from_sigmasq_series(
                    psd.sigmasq_vec[template.approximant],
                    num_bins,
                    kmin,
                    kmax
                )
            else:
                bins = power_chisq_bins(template, num_bins, psd, template.f_lower)
            template._bin_cache[key] = bins

        return template._bin_cache[key]

    def values(self, corr, snrv, snr_norm, psd, indices, template):
        """ Calculate the chisq at points given by indices.

        Returns
        -------
        chisq: Array
            Chisq values, one for each sample index, or zero for points below
            the specified SNR threshold

        chisq_dof: Array
            Number of statistical degrees of freedom for the chisq test
            in the given template, equal to 2 * num_bins - 2
        """
        if self.do:
            num_above = len(indices)
            dof = -100
            corr_tensor = _torch_tensor(corr)
            if self.snr_threshold:
                if corr_tensor is not None:
                    import torch

                    snrv_tensor = _as_torch_tensor(
                        snrv, corr_tensor, dtype=corr_tensor.dtype
                    )
                    index_tensor = _as_torch_tensor(
                        indices, corr_tensor, dtype=torch.long
                    )
                    above = (
                        torch.abs(snrv_tensor * snr_norm)
                        > self.snr_threshold
                    )
                    num_above = int(torch.count_nonzero(above).item())
                    above_indices = index_tensor[above]
                    above_snrv = snrv_tensor[above]
                    chisq_out = _chisq_zeros(corr, len(indices))
                else:
                    above = abs(snrv * snr_norm) > self.snr_threshold
                    num_above = above.sum()
                    above_indices = indices[above]
                    above_snrv = snrv[above]
                    chisq_out = numpy.zeros(
                        len(indices), dtype=numpy.float32
                    )
                logging.info('%s above chisq activation threshold' % num_above)
            else:
                above_indices = indices
                above_snrv = snrv

            if num_above > 0:
                bins = self.cached_chisq_bins(template, psd)
                # len(bins) is number of bin edges, num_bins = len(bins) - 1
                dof = (len(bins) - 1) * 2 - 2
                _chisq = power_chisq_at_points_from_precomputed(corr,
                                     above_snrv, snr_norm, bins, above_indices)

            if self.snr_threshold:
                if num_above > 0:
                    _copy_chisq_masked(chisq_out, above, _chisq)
            else:
                if num_above == 0:
                    chisq_out = _chisq_zeros(corr, 0)
                else:
                    chisq_out = _chisq

            return chisq_out, _chisq_dof_array(
                chisq_out, dof, len(indices)
            )
        else:
            return None, None


class SingleDetSkyMaxPowerChisq(SingleDetPowerChisq):
    """Class that handles precomputation and memory management for efficiently
    running the power chisq in a single detector inspiral analysis when
    maximizing analytically over sky location.
    """
    def __init__(self, **kwds):
        super(SingleDetSkyMaxPowerChisq, self).__init__(**kwds)
        self.template_mem = None
        self.corr_mem = None

    def calculate_chisq_bins(self, template, psd):
        """ Obtain the chisq bins for this template and PSD.
        """
        num_bins = int(self.parse_option(template, self.num_bins))
        if hasattr(psd, 'sigmasq_vec') and \
                                       template.approximant in psd.sigmasq_vec:
            kmin = int(template.f_lower / psd.delta_f)
            kmax = template.end_idx
            bins = power_chisq_bins_from_sigmasq_series(
                   psd.sigmasq_vec[template.approximant], num_bins, kmin, kmax)
        else:
            bins = power_chisq_bins(template, num_bins, psd, template.f_lower)
        return bins

    def values(self, corr_plus, corr_cross, snrv, psd,
               indices, template_plus, template_cross, u_vals,
               hplus_cross_corr, hpnorm, hcnorm):
        """ Calculate the chisq at points given by indices.

        Returns
        -------
        chisq: Array
            Chisq values, one for each sample index

        chisq_dof: Array
            Number of statistical degrees of freedom for the chisq test
            in the given template
        """
        if self.do:
            num_above = len(indices)
            dof = -100
            corr_tensor = _torch_tensor(corr_plus)
            torch_backed = corr_tensor is not None
            if self.snr_threshold:
                if torch_backed:
                    import torch

                    snrv_tensor = _as_torch_tensor(
                        snrv, corr_tensor, dtype=corr_tensor.dtype
                    )
                    index_tensor = _as_torch_tensor(
                        indices, corr_tensor, dtype=torch.long
                    )
                    u_tensor = _as_torch_tensor(
                        u_vals, corr_tensor, dtype=corr_tensor.real.dtype
                    )
                    above = torch.abs(snrv_tensor) > self.snr_threshold
                    num_above = int(torch.count_nonzero(above).item())
                    above_indices = index_tensor[above]
                    above_snrv = snrv_tensor[above]
                    above_u_vals = u_tensor[above]
                    rchisq = _chisq_zeros(corr_plus, len(indices))
                else:
                    above = abs(snrv) > self.snr_threshold
                    num_above = above.sum()
                    above_indices = indices[above]
                    above_snrv = snrv[above]
                    above_u_vals = u_vals[above]
                    rchisq = numpy.zeros(
                        len(indices), dtype=numpy.float32
                    )
                logging.info('%s above chisq activation threshold' % num_above)
            else:
                if torch_backed:
                    import torch

                    above_indices = _as_torch_tensor(
                        indices, corr_tensor, dtype=torch.long
                    )
                    above_snrv = _as_torch_tensor(
                        snrv, corr_tensor, dtype=corr_tensor.dtype
                    )
                    above_u_vals = _as_torch_tensor(
                        u_vals, corr_tensor, dtype=corr_tensor.real.dtype
                    )
                else:
                    above_indices = indices
                    above_snrv = snrv
                    above_u_vals = u_vals

            if num_above > 0:
                chisq = []
                curr_tmplt_mult_fac = 0.
                curr_corr_mult_fac = 0.
                if self.template_mem is None or \
                        (not len(self.template_mem) == len(template_plus)):
                    self.template_mem = zeros(len(template_plus),
                                dtype=complex_same_precision_as(corr_plus))
                if self.corr_mem is None or \
                                (not len(self.corr_mem) == len(corr_plus)):
                    self.corr_mem = zeros(len(corr_plus),
                                dtype=complex_same_precision_as(corr_plus))

                tmplt_data = template_cross.data
                corr_data = corr_cross.data
                self.template_mem[:] = template_cross
                self.corr_mem[:] = corr_cross
                template_cross._data = self.template_mem.data
                corr_cross._data = self.corr_mem.data

                try:
                    for lidx in range(num_above):
                        if torch_backed:
                            above_local_indices = above_indices[
                                lidx:lidx + 1
                            ]
                            above_local_snr = above_snrv[lidx:lidx + 1]
                            local_u_val = above_u_vals[lidx]
                            template = _torch_multiply_and_add(
                                template_cross,
                                template_plus,
                                local_u_val - curr_tmplt_mult_fac,
                            )
                        else:
                            above_local_indices = numpy.array(
                                [above_indices[lidx]]
                            )
                            above_local_snr = numpy.array(
                                [above_snrv[lidx]]
                            )
                            local_u_val = above_u_vals[lidx]
                            template = template_cross.multiply_and_add(
                                template_plus,
                                local_u_val - curr_tmplt_mult_fac,
                            )
                        curr_tmplt_mult_fac = local_u_val

                        template.f_lower = template_plus.f_lower
                        template.params = template_plus.params
                        # Construct the corr vector
                        norm_fac = local_u_val*local_u_val + 1
                        norm_fac += 2 * local_u_val * hplus_cross_corr
                        norm_fac = hcnorm / (norm_fac**0.5)
                        hp_fac = local_u_val * hpnorm / hcnorm
                        if torch_backed:
                            corr = _torch_multiply_and_add(
                                corr_cross,
                                corr_plus,
                                hp_fac - curr_corr_mult_fac,
                            )
                        else:
                            corr = corr_cross.multiply_and_add(
                                corr_plus,
                                hp_fac - curr_corr_mult_fac,
                            )
                        curr_corr_mult_fac = hp_fac

                        bins = self.calculate_chisq_bins(template, psd)
                        dof = (len(bins) - 1) * 2 - 2
                        curr_chisq = power_chisq_at_points_from_precomputed(
                            corr,
                            above_local_snr / norm_fac,
                            norm_fac,
                            bins,
                            above_local_indices,
                        )
                        if torch_backed:
                            chisq.append(backend_array(curr_chisq, "torch"))
                        else:
                            chisq.append(curr_chisq[0])
                    if torch_backed:
                        chisq = _torch_array(torch.cat(chisq))
                    else:
                        chisq = numpy.array(chisq)
                finally:
                    # Must reset corr and template to original values!
                    template_cross._data = tmplt_data
                    corr_cross._data = corr_data

            if self.snr_threshold:
                if num_above > 0:
                    _copy_chisq_masked(rchisq, above, chisq)
            else:
                if num_above == 0:
                    rchisq = _chisq_zeros(corr_plus, 0)
                else:
                    rchisq = chisq

            return rchisq, _chisq_dof_array(
                rchisq, dof, len(indices)
            )
        else:
            return None, None
