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
"""
This modules provides functions for matched filtering along with associated
utilities.
"""

from pycbc.types.backend import backend_array, wrap_backend_array
import logging
from math import sqrt
import os
import threading
import numpy

from pycbc.types import TimeSeries, FrequencySeries, zeros, empty, Array
from pycbc.types import complex_same_precision_as, real_same_precision_as
from pycbc.fft import fft, ifft, IFFT
from pycbc.opt import LimitedSizeDict
import pycbc.scheme
import pycbc

logger = logging.getLogger('pycbc.filter.matchedfilter')

_TORCH_CPU_NATIVE_BATCH_PEAK_GATE = (
    "PYCBC_TORCH_CPU_NATIVE_BATCH_PEAK"
)
_TORCH_CUDA_NATIVE_BATCH_PEAK_GATE = (
    "PYCBC_TORCH_CUDA_NATIVE_BATCH_PEAK"
)
_TORCH_ONDEVICE_PEAKS_GATE = (
    "PYCBC_TORCH_ONDEVICE_PEAKS"
)
_TORCH_CPU_NATIVE_BATCH_PEAK_TRUE = {"1", "true", "yes", "on"}
_TORCH_CPU_NATIVE_BATCH_PEAK_FALSE = {"0", "false", "no", "off"}
_TORCH_CPU_NATIVE_BATCH_PEAK_MAX_LENGTH = 2**32 - 1
_TORCH_ASYNC_STREAMS_GATE = "PYCBC_TORCH_ASYNC_STREAMS"


def _torch_inference_mode_context():
    """Return torch.inference_mode() or fallback to torch.no_grad()."""
    try:
        import torch
        if hasattr(torch, "inference_mode"):
            return torch.inference_mode()
        if hasattr(torch, "no_grad"):
            return torch.no_grad()
    except (ImportError, AttributeError):
        pass
    from contextlib import nullcontext
    return nullcontext()


BACKEND_PREFIX="pycbc.filter.matchedfilter_"

@pycbc.scheme.schemed(BACKEND_PREFIX)
def correlate(x, y, z):
    err_msg = "This function is a stub that should be overridden using the "
    err_msg += "scheme. You shouldn't be seeing this error!"
    raise ValueError(err_msg)


class BatchCorrelator(object):
    """ Create a batch correlation engine
    """
    def __init__(self, xs, zs, size):
        """ Correlate x and y, store in z. Arrays need not be equal length, but
        must be at least size long and of the same dtype. No error checking
        will be performed, so be careful. All dtypes must be complex64.
        Note, must be created within the processing context that it will be used in.
        """
        self.size = int(size)
        self.dtype = xs[0].dtype
        self.num_vectors = len(xs)

        # keep reference to arrays
        self.xs = xs
        self.zs = zs
        self._xs = xs
        self._zs = zs
        self._epoch = 0

        # Store each pointer as in integer array
        self.x = Array([v.ptr for v in xs], dtype=int)
        self.z = Array([v.ptr for v in zs], dtype=int)

    def mark_dirty(self):
        """Invalidate cached validation state when buffer references change."""
        self._epoch += 1

    @pycbc.scheme.schemed(BACKEND_PREFIX)
    def batch_correlate_execute(self, y):
        pass

    execute = batch_correlate_execute


@pycbc.scheme.schemed(BACKEND_PREFIX)
def _correlate_factory(x, y, z):
    err_msg = "This class is a stub that should be overridden using the "
    err_msg += "scheme. You shouldn't be seeing this error!"
    raise ValueError(err_msg)


class Correlator(object):
    """ Create a correlator engine

    Parameters
    ---------
    x : complex64
      Input pycbc.types.Array (or subclass); it will be conjugated
    y : complex64
      Input pycbc.types.Array (or subclass); it will not be conjugated
    z : complex64
      Output pycbc.types.Array (or subclass).
      It will contain conj(x) * y, element by element

    The addresses in memory of the data of all three parameter vectors
    must be the same modulo pycbc.PYCBC_ALIGNMENT
    """
    def __new__(cls, *args, **kwargs):
        real_cls = _correlate_factory(*args, **kwargs)
        return real_cls(*args, **kwargs) # pylint:disable=not-callable


# The class below should serve as the parent for all schemed classes.
# The intention is that this class serves simply as the location for
# all documentation of the class and its methods, though that is not
# yet implemented.  Perhaps something along the lines of:
#
#    http://stackoverflow.com/questions/2025562/inherit-docstrings-in-python-class-inheritance
#
# will work? Is there a better way?
class _BaseCorrelator(object):
    def correlate(self):
        """
        Compute the correlation of the vectors specified at object
        instantiation, writing into the output vector given when the
        object was instantiated. The intention is that this method
        should be called many times, with the contents of those vectors
        changing between invocations, but not their locations in memory
        or length.
        """
        pass


class MatchedFilterControl(object):
    def __init__(self, low_frequency_cutoff, high_frequency_cutoff, snr_threshold, tlen,
                 delta_f, dtype, segment_list, template_output, use_cluster,
                 downsample_factor=1, upsample_threshold=1, upsample_method='pruned_fft',
                 gpu_callback_method='none', cluster_function='symmetric'):
        """ Create a matched filter engine.

        Parameters
        ----------
        low_frequency_cutoff : {None, float}, optional
            The frequency to begin the filter calculation. If None, begin at the
            first frequency after DC.
        high_frequency_cutoff : {None, float}, optional
            The frequency to stop the filter calculation. If None, continue to the
            the nyquist frequency.
        snr_threshold : float
            The minimum snr to return when filtering
        segment_list : list
            List of FrequencySeries that are the Fourier-transformed data segments
        template_output : complex64
            Array of memory given as the 'out' parameter to waveform.FilterBank
        use_cluster : boolean
            If true, cluster triggers above threshold using a window; otherwise,
            only apply a threshold.
        downsample_factor : {1, int}, optional
            The factor by which to reduce the sample rate when doing a hierarchical
            matched filter
        upsample_threshold : {1, float}, optional
            The fraction of the snr_threshold to trigger on the subsampled filter.
        upsample_method : {pruned_fft, str}
            The method to upsample or interpolate the reduced rate filter.
        cluster_function : {symmetric, str}, optional
            Which method is used to cluster triggers over time. If 'findchirp', a
            sliding forward window; if 'symmetric', each window's peak is compared
            to the windows before and after it, and only kept as a trigger if larger
            than both.
        """
        # Assuming analysis time is constant across templates and segments, also
        # delta_f is constant across segments.
        self.tlen = tlen
        self.flen = self.tlen / 2 + 1
        self.delta_f = delta_f
        self.delta_t = 1.0/(self.delta_f * self.tlen)
        self.dtype = dtype
        self.snr_threshold = snr_threshold
        self.flow = low_frequency_cutoff
        self.fhigh = high_frequency_cutoff
        self.gpu_callback_method = gpu_callback_method
        if cluster_function not in ['symmetric', 'findchirp']:
            raise ValueError("MatchedFilter: 'cluster_function' must be either 'symmetric' or 'findchirp'")
        self.cluster_function = cluster_function
        self.segments = segment_list
        self.htilde = template_output

        if downsample_factor == 1:
            self.snr_mem = zeros(self.tlen, dtype=self.dtype)
            self.corr_mem = zeros(self.tlen, dtype=self.dtype)

            if use_cluster and (cluster_function == 'symmetric'):
                from pycbc import events

                self.matched_filter_and_cluster = self.full_matched_filter_and_cluster_symm
                # setup the threasholding/clustering operations for each segment
                self.threshold_and_clusterers = []
                for seg in self.segments:
                    thresh = events.ThresholdCluster(self.snr_mem[seg.analyze])
                    self.threshold_and_clusterers.append(thresh)
            elif use_cluster and (cluster_function == 'findchirp'):
                self.matched_filter_and_cluster = self.full_matched_filter_and_cluster_fc
            else:
                self.matched_filter_and_cluster = self.full_matched_filter_thresh_only

            # Assuming analysis time is constant across templates and segments, also
            # delta_f is constant across segments.
            self.kmin, self.kmax = get_cutoff_indices(self.flow, self.fhigh,
                                                      self.delta_f, self.tlen)

            # Set up the correlation operations for each analysis segment
            corr_slice = slice(self.kmin, self.kmax)
            self.correlators = []
            for seg in self.segments:
                corr = Correlator(self.htilde[corr_slice],
                                  seg[corr_slice],
                                  self.corr_mem[corr_slice])
                self.correlators.append(corr)

            # setup up the ifft we will do
            self.ifft = IFFT(self.corr_mem, self.snr_mem)

        elif downsample_factor >= 1:
            self.matched_filter_and_cluster = self.hierarchical_matched_filter_and_cluster
            self.downsample_factor = downsample_factor
            self.upsample_method = upsample_method
            self.upsample_threshold = upsample_threshold

            N_full = self.tlen
            N_red = N_full / downsample_factor
            self.kmin_full, self.kmax_full = get_cutoff_indices(self.flow,
                                              self.fhigh, self.delta_f, N_full)

            self.kmin_red, _ = get_cutoff_indices(self.flow,
                                                  self.fhigh, self.delta_f, N_red)

            if self.kmax_full < N_red:
                self.kmax_red = self.kmax_full
            else:
                self.kmax_red = N_red - 1

            self.snr_mem = zeros(N_red, dtype=self.dtype)
            self.corr_mem_full = FrequencySeries(zeros(N_full, dtype=self.dtype), delta_f=self.delta_f)
            self.corr_mem = Array(self.corr_mem_full[0:N_red], copy=False)
            self.inter_vec = zeros(N_full, dtype=self.dtype)

        else:
            raise ValueError("Invalid downsample factor")

    def full_matched_filter_and_cluster_symm(self, segnum, template_norm, window, epoch=None):
        """ Returns the complex snr timeseries, normalization of the complex snr,
        the correlation vector frequency series, the list of indices of the
        triggers, and the snr values at the trigger locations. Returns empty
        lists for these for points that are not above the threshold.

        Calculated the matched filter, threshold, and cluster.

        Parameters
        ----------
        segnum : int
            Index into the list of segments at MatchedFilterControl construction
            against which to filter.
        template_norm : float
            The htilde, template normalization factor.
        window : int
            Size of the window over which to cluster triggers, in samples

        Returns
        -------
        snr : TimeSeries
            A time series containing the complex snr.
        norm : float
            The normalization of the complex snr.
        correlation: FrequencySeries
            A frequency series containing the correlation vector.
        idx : Array
            List of indices of the triggers.
        snrv : Array
            The snr values at the trigger locations.
        """
        norm = (4.0 * self.delta_f) / sqrt(template_norm)
        thresh_val = self.snr_threshold / norm
        clusterer = self.threshold_and_clusterers[segnum]

        # Fast path: CUDA Graph replay if enabled/captured
        use_cuda_graph = (
            getattr(self, "_cuda_graph_enabled", False)
            or os.environ.get("PYCBC_TORCH_CUDA_GRAPH", "0") == "1"
        ) and (
            hasattr(clusterer, "series")
            and getattr(clusterer.series, "is_cuda", False)
        )
        if use_cuda_graph:
            from .matchedfilter_torch import replay_symmetric_cuda_graph
            graph_result = replay_symmetric_cuda_graph(
                self, segnum, window, template_norm, thresh_val
            )
            if graph_result is not None:
                snrv, idx = graph_result
                if len(idx) == 0:
                    return [], [], [], [], []
                logger.info("%d points above threshold", len(idx))
                snr = TimeSeries(self.snr_mem, epoch=epoch, delta_t=self.delta_t, copy=False)
                corr = FrequencySeries(self.corr_mem, delta_f=self.delta_f, copy=False)
                return snr, norm, corr, idx, snrv

        self.correlators[segnum].correlate()
        self.ifft.execute()
        snrv, idx = clusterer.threshold_and_cluster(thresh_val, window)

        if len(idx) == 0:
            return [], [], [], [], []

        logger.info("%d points above threshold", len(idx))

        snr = TimeSeries(self.snr_mem, epoch=epoch, delta_t=self.delta_t, copy=False)
        corr = FrequencySeries(self.corr_mem, delta_f=self.delta_f, copy=False)
        return snr, norm, corr, idx, snrv

    def capture_cuda_graph_symm(self, segnum, window, template_norm=1.0):
        """Ask the Torch backend to pre-record symmetric filtering."""
        clusterer = self.threshold_and_clusterers[segnum]
        if not (hasattr(clusterer, "series") and
                getattr(clusterer.series, "is_cuda", False)):
            return False
        from .matchedfilter_torch import capture_symmetric_cuda_graph
        return capture_symmetric_cuda_graph(
            self, segnum, window, template_norm
        )

    def full_matched_filter_and_cluster_fc(self, segnum, template_norm, window, epoch=None):
        """ Returns the complex snr timeseries, normalization of the complex snr,
        the correlation vector frequency series, the list of indices of the
        triggers, and the snr values at the trigger locations. Returns empty
        lists for these for points that are not above the threshold.

        Calculated the matched filter, threshold, and cluster.

        Parameters
        ----------
        segnum : int
            Index into the list of segments at MatchedFilterControl construction
            against which to filter.
        template_norm : float
            The htilde, template normalization factor.
        window : int
            Size of the window over which to cluster triggers, in samples

        Returns
        -------
        snr : TimeSeries
            A time series containing the complex snr.
        norm : float
            The normalization of the complex snr.
        correlation: FrequencySeries
            A frequency series containing the correlation vector.
        idx : Array
            List of indices of the triggers.
        snrv : Array
            The snr values at the trigger locations.
        """
        norm = (4.0 * self.delta_f) / sqrt(template_norm)
        self.correlators[segnum].correlate()
        self.ifft.execute()
        from pycbc import events

        idx, snrv = events.threshold_and_cluster_findchirp(
            self.snr_mem[self.segments[segnum].analyze],
            self.snr_threshold / norm,
            window,
        )

        if len(idx) == 0:
            return [], [], [], [], []

        logger.info("%d points above threshold", len(idx))

        snr = TimeSeries(self.snr_mem, epoch=epoch, delta_t=self.delta_t, copy=False)
        corr = FrequencySeries(self.corr_mem, delta_f=self.delta_f, copy=False)
        return snr, norm, corr, idx, snrv

    def full_matched_filter_thresh_only(self, segnum, template_norm, window=None, epoch=None):
        """ Returns the complex snr timeseries, normalization of the complex snr,
        the correlation vector frequency series, the list of indices of the
        triggers, and the snr values at the trigger locations. Returns empty
        lists for these for points that are not above the threshold.

        Calculated the matched filter, threshold, and cluster.

        Parameters
        ----------
        segnum : int
            Index into the list of segments at MatchedFilterControl construction
            against which to filter.
        template_norm : float
            The htilde, template normalization factor.
        window : int
            Size of the window over which to cluster triggers, in samples.
            This is IGNORED by this function, and provided only for API compatibility.

        Returns
        -------
        snr : TimeSeries
            A time series containing the complex snr.
        norm : float
            The normalization of the complex snr.
        correlation: FrequencySeries
            A frequency series containing the correlation vector.
        idx : Array
            List of indices of the triggers.
        snrv : Array
            The snr values at the trigger locations.
        """
        norm = (4.0 * self.delta_f) / sqrt(template_norm)
        self.correlators[segnum].correlate()
        self.ifft.execute()
        from pycbc import events

        idx, snrv = events.threshold_only(self.snr_mem[self.segments[segnum].analyze],
                                          self.snr_threshold / norm)
        logger.info("%d points above threshold", len(idx))

        snr = TimeSeries(self.snr_mem, epoch=epoch, delta_t=self.delta_t, copy=False)
        corr = FrequencySeries(self.corr_mem, delta_f=self.delta_f, copy=False)
        return snr, norm, corr, idx, snrv

    def hierarchical_matched_filter_and_cluster(self, segnum, template_norm, window):
        """ Returns the complex snr timeseries, normalization of the complex snr,
        the correlation vector frequency series, the list of indices of the
        triggers, and the snr values at the trigger locations. Returns empty
        lists for these for points that are not above the threshold.

        Calculated the matched filter, threshold, and cluster.

        Parameters
        ----------
        segnum : int
            Index into the list of segments at MatchedFilterControl construction
        template_norm : float
            The htilde, template normalization factor.
        window : int
            Size of the window over which to cluster triggers, in samples

        Returns
        -------
        snr : TimeSeries
            A time series containing the complex snr at the reduced sample rate.
        norm : float
            The normalization of the complex snr.
        correlation: FrequencySeries
            A frequency series containing the correlation vector.
        idx : Array
            List of indices of the triggers.
        snrv : Array
            The snr values at the trigger locations.
        """
        from pycbc.fft.fftw_pruned import pruned_c2cifft, fft_transpose
        htilde = self.htilde
        stilde = self.segments[segnum]

        norm = (4.0 * stilde.delta_f) / sqrt(template_norm)

        correlate(htilde[self.kmin_red:self.kmax_red],
                  stilde[self.kmin_red:self.kmax_red],
                  self.corr_mem[self.kmin_red:self.kmax_red])

        ifft(self.corr_mem, self.snr_mem)

        if not hasattr(stilde, 'red_analyze'):
            stilde.red_analyze = \
                             slice(stilde.analyze.start/self.downsample_factor,
                                   stilde.analyze.stop/self.downsample_factor)


        from pycbc import events

        idx_red, _ = events.threshold_and_cluster_findchirp(
            self.snr_mem[stilde.red_analyze],
            self.snr_threshold / norm * self.upsample_threshold,
            window / self.downsample_factor,
        )
        if len(idx_red) == 0:
            return [], None, [], [], []

        logger.info("%d points above threshold at reduced resolution",
                    len(idx_red))

        # The fancy upsampling is here
        if self.upsample_method=='pruned_fft':
            idx = (idx_red + stilde.analyze.start/self.downsample_factor)\
                   * self.downsample_factor

            idx = smear(idx, self.downsample_factor)

            # cache transposed  versions of htilde and stilde
            if not hasattr(self.corr_mem_full, 'transposed'):
                self.corr_mem_full.transposed = zeros(len(self.corr_mem_full), dtype=self.dtype)

            if not hasattr(htilde, 'transposed'):
                htilde.transposed = zeros(len(self.corr_mem_full), dtype=self.dtype)
                htilde.transposed[self.kmin_full:self.kmax_full] = htilde[self.kmin_full:self.kmax_full]
                htilde.transposed = fft_transpose(htilde.transposed)

            if not hasattr(stilde, 'transposed'):
                stilde.transposed = zeros(len(self.corr_mem_full), dtype=self.dtype)
                stilde.transposed[self.kmin_full:self.kmax_full] = stilde[self.kmin_full:self.kmax_full]
                stilde.transposed = fft_transpose(stilde.transposed)

            correlate(htilde.transposed, stilde.transposed, self.corr_mem_full.transposed)
            snrv = pruned_c2cifft(self.corr_mem_full.transposed, self.inter_vec, idx, pretransposed=True)
            idx = idx - stilde.analyze.start
            idx2, snrv = events.threshold(Array(snrv, copy=False), self.snr_threshold / norm)

            if len(idx2) > 0:
                correlate(htilde[self.kmax_red:self.kmax_full],
                          stilde[self.kmax_red:self.kmax_full],
                          self.corr_mem_full[self.kmax_red:self.kmax_full])
                idx, snrv = events.cluster_reduce(idx[idx2], snrv, window)
            else:
                idx, snrv = [], []

            logger.info("%d points at full rate and clustering", len(idx))
            return self.snr_mem, norm, self.corr_mem_full, idx, snrv
        else:
            raise ValueError("Invalid upsample method")


def _torch_data_tensor(value):
    """Return the tensor backing a Torch PyCBC array, if present."""
    return backend_array(value, "torch")


def _array_from_torch_tensor(tensor):
    """Wrap a result tensor like a PyCBC array without copying it."""
    return Array(wrap_backend_array(tensor), copy=False)


def _sky_max_threshold_locations(
        hplus, hcross, hpnorm, hcnorm, thresh, analyse_slice):
    """Find candidate locations for either sky-max statistic."""
    hplus_tensor = _torch_data_tensor(hplus)
    hcross_tensor = _torch_data_tensor(hcross)
    if hplus_tensor is not None and hcross_tensor is not None:
        import torch

        start = analyse_slice.start
        hp_analyse = hplus_tensor[analyse_slice]
        hc_analyse = hcross_tensor[analyse_slice]
        hp_thresh = torch.as_tensor(
            thresh / (2**0.5 * hpnorm),
            device=hp_analyse.device,
            dtype=hp_analyse.real.dtype,
        )
        hc_thresh = torch.as_tensor(
            thresh / (2**0.5 * hcnorm),
            device=hc_analyse.device,
            dtype=hc_analyse.real.dtype,
        )
        if hp_analyse.is_complex():
            hp_sq = torch.view_as_real(hp_analyse).square().sum(dim=-1)
        else:
            hp_sq = hp_analyse.square()
        if hc_analyse.is_complex():
            hc_sq = torch.view_as_real(hc_analyse).square().sum(dim=-1)
        else:
            hc_sq = hc_analyse.square()
        mask = (hp_sq > hp_thresh.square()) | (hc_sq > hc_thresh.square())
        indices = torch.nonzero(mask, as_tuple=False).flatten() + start
        hp_red = hplus_tensor[indices] * hpnorm
        hc_red = hcross_tensor[indices] * hcnorm
        if hp_red.is_complex():
            stat = (
                torch.view_as_real(hp_red).square().sum(dim=-1)
                + torch.view_as_real(hc_red).square().sum(dim=-1)
            )
        else:
            stat = hp_red.square() + hc_red.square()
        return indices[stat > thresh * thresh]

    from pycbc import events

    idx_p, _ = events.threshold_only(
        hplus[analyse_slice], thresh / (2**0.5 * hpnorm)
    )
    # The CPU threshold backend returns a view into reusable scratch storage.
    # Offset it immediately so the hcross call cannot overwrite these indices.
    idx_p = idx_p + analyse_slice.start
    idx_c, _ = events.threshold_only(
        hcross[analyse_slice], thresh / (2**0.5 * hcnorm)
    )
    idx_c = idx_c + analyse_slice.start

    def _locations(indices):
        hp_red = hplus[indices] * hpnorm
        hc_red = hcross[indices] * hcnorm
        stat = (
            hp_red.real**2
            + hp_red.imag**2
            + hc_red.real**2
            + hc_red.imag**2
        )
        keep = stat > thresh * thresh
        return indices[keep]

    return numpy.unique(numpy.concatenate((_locations(idx_p), _locations(idx_c))))


def compute_max_snr_over_sky_loc_stat(hplus, hcross, hphccorr,
                                                      hpnorm=None, hcnorm=None,
                                                      out=None, thresh=0,
                                                      analyse_slice=None):
    """
    Matched filter maximised over polarization and orbital phase.

    This implements the statistic derived in 1603.02444. It is encouraged
    to read that work to understand the limitations and assumptions implicit
    in this statistic before using it.

    Parameters
    -----------
    hplus : TimeSeries
        This is the IFFTed complex SNR time series of (h+, data). If not
        normalized, supply the normalization factor so this can be done!
        It is recommended to normalize this before sending through this
        function
    hcross : TimeSeries
        This is the IFFTed complex SNR time series of (hx, data). If not
        normalized, supply the normalization factor so this can be done!
    hphccorr : float
        The real component of the overlap between the two polarizations
        Re[(h+, hx)]. Note that the imaginary component does not enter the
        detection statistic. This must be normalized and is sign-sensitive.
    thresh : float
        Used for optimization. If we do not care about the value of SNR
        values below thresh we can calculate a quick statistic that will
        always overestimate SNR and then only calculate the proper, more
        expensive, statistic at points where the quick SNR is above thresh.
    hpsigmasq : float
        The normalization factor (h+, h+). Default = None (=1, already
        normalized)
    hcsigmasq : float
        The normalization factor (hx, hx). Default = None (=1, already
        normalized)
    out : TimeSeries (optional, default=None)
        If given, use this array to store the output.

    Returns
    --------
    det_stat : TimeSeries
        The SNR maximized over sky location
    """
    # NOTE: Not much optimization has been done here! This may need to be
    # Cythonized.

    if out is None:
        if _torch_data_tensor(hplus) is not None:
            out = zeros(len(hplus), dtype=real_same_precision_as(hplus))
        else:
            out = zeros(len(hplus))
        out.non_zero_locs = numpy.array([], dtype=out.dtype)
    else:
        if not hasattr(out, 'non_zero_locs'):
            # Doing this every time is not a zero-cost operation
            out.data[:] = 0
            out.non_zero_locs = numpy.array([], dtype=out.dtype)
        else:
            # Only set non zero locations to zero
            out.data[out.non_zero_locs] = 0


    # If threshold is given we can limit the points at which to compute the
    # full statistic
    if thresh:
        # This is the statistic that always overestimates the SNR...
        # It allows some unphysical freedom that the full statistic does not
        locs = _sky_max_threshold_locations(
            hplus, hcross, hpnorm, hcnorm, thresh, analyse_slice
        )

        hplus = hplus[locs]
        hcross = hcross[locs]

    hplus = hplus * hpnorm
    hcross = hcross * hcnorm


    # Calculate and sanity check the denominator
    denom = 1 - hphccorr*hphccorr
    if denom < 0:
        if hphccorr > 1:
            err_msg = "Overlap between hp and hc is given as %f. " %(hphccorr)
            err_msg += "How can an overlap be bigger than 1?"
            raise ValueError(err_msg)
        else:
            err_msg = "There really is no way to raise this error!?! "
            err_msg += "If you're seeing this, it is bad."
            raise ValueError(err_msg)
    if denom == 0:
        # This case, of hphccorr==1, makes the statistic degenerate
        # This case should not physically be possible luckily.
        err_msg = "You have supplied a real overlap between hp and hc of 1. "
        err_msg += "Ian is reasonably certain this is physically impossible "
        err_msg += "so why are you seeing this?"
        raise ValueError(err_msg)

    assert(len(hplus) == len(hcross))

    # Now the stuff where comp. cost may be a problem
    hplus_tensor = _torch_data_tensor(hplus)
    hcross_tensor = _torch_data_tensor(hcross)
    if hplus_tensor is not None and hcross_tensor is not None:
        import torch

        hplus_real = hplus_tensor.real
        hplus_imag = hplus_tensor.imag
        hcross_real = hcross_tensor.real
        hcross_imag = hcross_tensor.imag
        if hplus_tensor.is_complex():
            hplus_magsq = torch.view_as_real(hplus_tensor).square().sum(dim=-1)
            hcross_magsq = torch.view_as_real(hcross_tensor).square().sum(dim=-1)
        else:
            hplus_magsq = hplus_tensor.square()
            hcross_magsq = hcross_tensor.square()
        rho_pluscross = (
            hplus_real * hcross_real + hplus_imag * hcross_imag
        )
    else:
        hplus_magsq = numpy.real(hplus) * numpy.real(hplus) + \
                           numpy.imag(hplus) * numpy.imag(hplus)
        hcross_magsq = numpy.real(hcross) * numpy.real(hcross) + \
                           numpy.imag(hcross) * numpy.imag(hcross)
        rho_pluscross = numpy.real(hplus) * numpy.real(hcross) + \
                            numpy.imag(hplus) * numpy.imag(hcross)

    sqroot = (hplus_magsq - hcross_magsq)**2
    sqroot += 4 * (hphccorr * hplus_magsq - rho_pluscross) * \
                  (hphccorr * hcross_magsq - rho_pluscross)
    # Sometimes this can be less than 0 due to numeric imprecision, catch this.
    if hplus_tensor is not None:
        # This should not be much smaller than zero due to numeric imprecision.
        if torch.any(sqroot < -0.0001).item():
            err_msg = "Square root has become negative. Something wrong here!"
            raise ValueError(err_msg)
        sqroot = torch.clamp_min(sqroot, 0)
        sqroot = torch.sqrt(sqroot)
    else:
        if (sqroot < 0).any():
            indices = numpy.arange(len(sqroot))[sqroot < 0]
            # This should not be *much* smaller than 0 due to numeric imprecision
            if (sqroot[indices] < -0.0001).any():
                err_msg = "Square root has become negative. Something wrong here!"
                raise ValueError(err_msg)
            sqroot[indices] = 0
        sqroot = numpy.sqrt(sqroot)
    det_stat_sq = 0.5 * (hplus_magsq + hcross_magsq - \
                         2 * rho_pluscross*hphccorr + sqroot) / denom

    if hplus_tensor is not None:
        det_stat = torch.sqrt(det_stat_sq)
    else:
        det_stat = numpy.sqrt(det_stat_sq)

    if thresh:
        out_tensor = _torch_data_tensor(out)
        if out_tensor is not None:
            if isinstance(locs, torch.Tensor):
                locs_tensor = locs.to(device=out_tensor.device)
            else:
                locs_tensor = torch.as_tensor(
                    locs, device=out_tensor.device, dtype=torch.long
                )
            out_tensor[locs_tensor] = det_stat.to(dtype=out_tensor.dtype)
        else:
            out.data[locs] = det_stat
        out.non_zero_locs = locs
        return out
    elif hplus_tensor is not None:
        return _array_from_torch_tensor(det_stat)
    else:
        return Array(det_stat, copy=False)

def compute_u_val_for_sky_loc_stat(hplus, hcross, hphccorr,
                                 hpnorm=None, hcnorm=None, indices=None):
    """The max-over-sky location detection statistic maximizes over a phase,
    an amplitude and the ratio of F+ and Fx, encoded in a variable called u.
    Here we return the value of u for the given indices.

    Torch-backed inputs return device-resident PyCBC arrays.
    """
    if indices is not None:
        hplus = hplus[indices]
        hcross = hcross[indices]

    if hpnorm is not None:
        hplus = hplus * hpnorm
    if hcnorm is not None:
        hcross = hcross * hcnorm

    hplus_tensor = _torch_data_tensor(hplus)
    hcross_tensor = _torch_data_tensor(hcross)
    if hplus_tensor is not None and hcross_tensor is not None:
        import torch

        hplus_magsq = hplus_tensor.real.square() + hplus_tensor.imag.square()
        hcross_magsq = hcross_tensor.real.square() + hcross_tensor.imag.square()
        rho_pluscross = (
            hplus_tensor.real * hcross_tensor.real
            + hplus_tensor.imag * hcross_tensor.imag
        )

        a = hphccorr * hplus_magsq - rho_pluscross
        b = hplus_magsq - hcross_magsq
        c = rho_pluscross - hphccorr * hcross_magsq
        sq_root = -torch.sqrt(b * b - 4 * a * c)
        bad_lgc = a == 0
        dbl_bad_lgc = bad_lgc & (c == 0) & (b == 0)
        u = torch.zeros_like(sq_root)
        u[dbl_bad_lgc] = 1
        u[bad_lgc & ~dbl_bad_lgc] = 1e17
        normal = ~bad_lgc
        u[normal] = (-b[normal] + sq_root[normal]) / (2 * a[normal])
        coa_phase = torch.angle(hplus_tensor * u + hcross_tensor)
        return (
            _array_from_torch_tensor(u),
            _array_from_torch_tensor(coa_phase),
        )

    # Sanity checking in func. above should already have identified any points
    # which are bad, and should be used to construct indices for input here
    hplus_magsq = numpy.real(hplus) * numpy.real(hplus) + \
                       numpy.imag(hplus) * numpy.imag(hplus)
    hcross_magsq = numpy.real(hcross) * numpy.real(hcross) + \
                       numpy.imag(hcross) * numpy.imag(hcross)
    rho_pluscross = numpy.real(hplus) * numpy.real(hcross) + \
                       numpy.imag(hplus)*numpy.imag(hcross)

    a = hphccorr * hplus_magsq - rho_pluscross
    b = hplus_magsq - hcross_magsq
    c = rho_pluscross - hphccorr * hcross_magsq

    sq_root = b*b - 4*a*c
    sq_root = sq_root**0.5
    sq_root = -sq_root
    # Catch the a->0 case
    bad_lgc = (a == 0)
    dbl_bad_lgc = numpy.logical_and(c == 0, b == 0)
    dbl_bad_lgc = numpy.logical_and(bad_lgc, dbl_bad_lgc)
    # Initialize u
    u = sq_root * 0.
    # In this case u is completely degenerate, so set it to 1
    u[dbl_bad_lgc] = 1.
    # If a->0 avoid overflow by just setting to a large value
    u[bad_lgc & ~dbl_bad_lgc] = 1E17
    # Otherwise normal statistic
    u[~bad_lgc] = (-b[~bad_lgc] + sq_root[~bad_lgc]) / (2*a[~bad_lgc])

    snr_cplx = hplus * u + hcross
    coa_phase = numpy.angle(snr_cplx)

    return u, coa_phase

def compute_max_snr_over_sky_loc_stat_no_phase(hplus, hcross, hphccorr,
                                               hpnorm=None, hcnorm=None,
                                               out=None, thresh=0,
                                               analyse_slice=None):
    """
    Matched filter maximised over polarization phase.

    This implements the statistic derived in 1709.09181. It is encouraged
    to read that work to understand the limitations and assumptions implicit
    in this statistic before using it.

    In contrast to compute_max_snr_over_sky_loc_stat this function
    performs no maximization over orbital phase, treating that as an intrinsic
    parameter. In the case of aligned-spin 2,2-mode only waveforms, this
    collapses to the normal statistic (at twice the computational cost!)

    Parameters
    -----------
    hplus : TimeSeries
        This is the IFFTed complex SNR time series of (h+, data). If not
        normalized, supply the normalization factor so this can be done!
        It is recommended to normalize this before sending through this
        function
    hcross : TimeSeries
        This is the IFFTed complex SNR time series of (hx, data). If not
        normalized, supply the normalization factor so this can be done!
    hphccorr : float
        The real component of the overlap between the two polarizations
        Re[(h+, hx)]. Note that the imaginary component does not enter the
        detection statistic. This must be normalized and is sign-sensitive.
    thresh : float
        Used for optimization. If we do not care about the value of SNR
        values below thresh we can calculate a quick statistic that will
        always overestimate SNR and then only calculate the proper, more
        expensive, statistic at points where the quick SNR is above thresh.
    hpsigmasq : float
        The normalization factor (h+, h+). Default = None (=1, already
        normalized)
    hcsigmasq : float
        The normalization factor (hx, hx). Default = None (=1, already
        normalized)
    out : TimeSeries (optional, default=None)
        If given, use this array to store the output.

    Returns
    --------
    det_stat : TimeSeries
        The SNR maximized over sky location
    """
    # NOTE: Not much optimization has been done here! This may need to be
    # Cythonized.

    if out is None:
        if _torch_data_tensor(hplus) is not None:
            out = zeros(len(hplus), dtype=real_same_precision_as(hplus))
        else:
            out = zeros(len(hplus))
        out.non_zero_locs = numpy.array([], dtype=out.dtype)
    else:
        if not hasattr(out, 'non_zero_locs'):
            # Doing this every time is not a zero-cost operation
            out.data[:] = 0
            out.non_zero_locs = numpy.array([], dtype=out.dtype)
        else:
            # Only set non zero locations to zero
            out.data[out.non_zero_locs] = 0

    # If threshold is given we can limit the points at which to compute the
    # full statistic
    if thresh:
        # This is the statistic that always overestimates the SNR...
        # It allows some unphysical freedom that the full statistic does not
        #
        # For now this is copied from the max-over-phase statistic. One could
        # probably make this faster by removing the imaginary components of
        # the matched filter, as these are not used here.
        locs = _sky_max_threshold_locations(
            hplus, hcross, hpnorm, hcnorm, thresh, analyse_slice
        )

        hplus = hplus[locs]
        hcross = hcross[locs]

    hplus = hplus * hpnorm
    hcross = hcross * hcnorm


    # Calculate and sanity check the denominator
    denom = 1 - hphccorr*hphccorr
    if denom < 0:
        if hphccorr > 1:
            err_msg = "Overlap between hp and hc is given as %f. " %(hphccorr)
            err_msg += "How can an overlap be bigger than 1?"
            raise ValueError(err_msg)
        else:
            err_msg = "There really is no way to raise this error!?! "
            err_msg += "If you're seeing this, it is bad."
            raise ValueError(err_msg)
    if denom == 0:
        # This case, of hphccorr==1, makes the statistic degenerate
        # This case should not physically be possible luckily.
        err_msg = "You have supplied a real overlap between hp and hc of 1. "
        err_msg += "Ian is reasonably certain this is physically impossible "
        err_msg += "so why are you seeing this?"
        raise ValueError(err_msg)

    assert(len(hplus) == len(hcross))

    # Now the stuff where comp. cost may be a problem
    hplus_tensor = _torch_data_tensor(hplus)
    hcross_tensor = _torch_data_tensor(hcross)
    if hplus_tensor is not None and hcross_tensor is not None:
        import torch

        hplus_magsq = hplus_tensor.real.square()
        hcross_magsq = hcross_tensor.real.square()
        rho_pluscross = hplus_tensor.real * hcross_tensor.real
    else:
        hplus_magsq = numpy.real(hplus) * numpy.real(hplus)
        hcross_magsq = numpy.real(hcross) * numpy.real(hcross)
        rho_pluscross = numpy.real(hplus) * numpy.real(hcross)

    det_stat_sq = (hplus_magsq + hcross_magsq - 2 * rho_pluscross*hphccorr)

    if hplus_tensor is not None:
        det_stat = torch.sqrt(det_stat_sq / denom)
    else:
        det_stat = numpy.sqrt(det_stat_sq / denom)

    if thresh:
        out_tensor = _torch_data_tensor(out)
        if out_tensor is not None:
            if isinstance(locs, torch.Tensor):
                locs_tensor = locs.to(device=out_tensor.device)
            else:
                locs_tensor = torch.as_tensor(
                    locs, device=out_tensor.device, dtype=torch.long
                )
            out_tensor[locs_tensor] = det_stat.to(dtype=out_tensor.dtype)
        else:
            out.data[locs] = det_stat
        out.non_zero_locs = locs
        return out
    elif hplus_tensor is not None:
        return _array_from_torch_tensor(det_stat)
    else:
        return Array(det_stat, copy=False)

def compute_u_val_for_sky_loc_stat_no_phase(hplus, hcross, hphccorr,
                                 hpnorm=None , hcnorm=None, indices=None):
    """The max-over-sky location (no phase) detection statistic maximizes over
    an amplitude and the ratio of F+ and Fx, encoded in a variable called u.
    Here we return the value of u for the given indices.

    Torch-backed inputs return device-resident PyCBC arrays.
    """
    if indices is not None:
        hplus = hplus[indices]
        hcross = hcross[indices]

    if hpnorm is not None:
        hplus = hplus * hpnorm
    if hcnorm is not None:
        hcross = hcross * hcnorm

    hplus_tensor = _torch_data_tensor(hplus)
    hcross_tensor = _torch_data_tensor(hcross)
    if hplus_tensor is not None and hcross_tensor is not None:
        import torch

        rhoplusre = hplus_tensor.real
        rhocrossre = hcross_tensor.real
        denom = -rhocrossre + hphccorr * rhoplusre
        u_val = torch.where(
            denom == 0,
            torch.full_like(denom, 1e17),
            (-rhoplusre + hphccorr * rhocrossre) / denom,
        )
        coa_phase = torch.zeros(
            len(u_val), dtype=torch.float32, device=u_val.device
        )
        return (
            _array_from_torch_tensor(u_val),
            _array_from_torch_tensor(coa_phase),
        )

    rhoplusre=numpy.real(hplus)
    rhocrossre=numpy.real(hcross)
    overlap=numpy.real(hphccorr)

    denom = (-rhocrossre+overlap*rhoplusre)
    # Initialize tan_kappa array
    u_val = denom * 0.
    # Catch the denominator -> 0 case
    numpy.putmask(u_val, denom == 0, 1E17)
    # Otherwise do normal statistic
    numpy.putmask(u_val, denom != 0, (-rhoplusre+overlap*rhocrossre)/(-rhocrossre+overlap*rhoplusre))
    coa_phase = numpy.zeros(len(indices), dtype=numpy.float32)

    return u_val, coa_phase


class MatchedFilterSkyMaxControl(object):
    # FIXME: This seems much more simplistic than the aligned-spin class.
    #        E.g. no correlators. Is this worth updating?
    def __init__(self, low_frequency_cutoff, high_frequency_cutoff,
                snr_threshold, tlen, delta_f, dtype):
        """
        Create a matched filter engine.

        Parameters
        ----------
        low_frequency_cutoff : {None, float}, optional
            The frequency to begin the filter calculation. If None, begin
            at the first frequency after DC.
        high_frequency_cutoff : {None, float}, optional
            The frequency to stop the filter calculation. If None, continue
            to the nyquist frequency.
        snr_threshold : float
            The minimum snr to return when filtering
        """
        self.tlen = tlen
        self.delta_f = delta_f
        self.dtype = dtype
        self.snr_threshold = snr_threshold
        self.flow = low_frequency_cutoff
        self.fhigh = high_frequency_cutoff

        self.matched_filter_and_cluster = \
                                    self.full_matched_filter_and_cluster
        self.snr_plus_mem = zeros(self.tlen, dtype=self.dtype)
        self.corr_plus_mem = zeros(self.tlen, dtype=self.dtype)
        self.snr_cross_mem = zeros(self.tlen, dtype=self.dtype)
        self.corr_cross_mem = zeros(self.tlen, dtype=self.dtype)
        self.snr_mem = zeros(self.tlen, dtype=self.dtype)
        self.cached_hplus_hcross_correlation = None
        self.cached_hplus_hcross_hplus = None
        self.cached_hplus_hcross_hcross = None
        self.cached_hplus_hcross_psd = None


    def full_matched_filter_and_cluster(self, hplus, hcross, hplus_norm,
                                        hcross_norm, psd, stilde, window):
        """
        Return the complex snr and normalization.

        Calculated the matched filter, threshold, and cluster.

        Parameters
        ----------
        h_quantities : Various
            FILL ME IN
        stilde : FrequencySeries
            The strain data to be filtered.
        window : int
            The size of the cluster window in samples.

        Returns
        -------
        snr : TimeSeries
            A time series containing the complex snr.
        norm : float
            The normalization of the complex snr.
        correlation: FrequencySeries
            A frequency series containing the correlation vector.
        idx : Array
            List of indices of the triggers.
        snrv : Array
            The snr values at the trigger locations.
        """

        I_plus, Iplus_corr, Iplus_norm = matched_filter_core(hplus, stilde,
                                          h_norm=hplus_norm,
                                          low_frequency_cutoff=self.flow,
                                          high_frequency_cutoff=self.fhigh,
                                          out=self.snr_plus_mem,
                                          corr_out=self.corr_plus_mem)


        I_cross, Icross_corr, Icross_norm = matched_filter_core(hcross,
                                          stilde, h_norm=hcross_norm,
                                          low_frequency_cutoff=self.flow,
                                          high_frequency_cutoff=self.fhigh,
                                          out=self.snr_cross_mem,
                                          corr_out=self.corr_cross_mem)

        # The information on the complex side of this overlap is important
        # we may want to use this in the future.
        if not id(hplus) == self.cached_hplus_hcross_hplus:
            self.cached_hplus_hcross_correlation = None
        if not id(hcross) == self.cached_hplus_hcross_hcross:
            self.cached_hplus_hcross_correlation = None
        if not id(psd) == self.cached_hplus_hcross_psd:
            self.cached_hplus_hcross_correlation = None
        if self.cached_hplus_hcross_correlation is None:
            hplus_cross_corr = overlap_cplx(hplus, hcross, psd=psd,
                                           low_frequency_cutoff=self.flow,
                                           high_frequency_cutoff=self.fhigh,
                                           normalized=False)
            hplus_cross_corr = numpy.real(hplus_cross_corr)
            hplus_cross_corr = hplus_cross_corr / (hcross_norm*hplus_norm)**0.5
            self.cached_hplus_hcross_correlation = hplus_cross_corr
            self.cached_hplus_hcross_hplus = id(hplus)
            self.cached_hplus_hcross_hcross = id(hcross)
            self.cached_hplus_hcross_psd = id(psd)
        else:
            hplus_cross_corr = self.cached_hplus_hcross_correlation

        snr = self._maximized_snr(I_plus,I_cross,
                                  hplus_cross_corr,
                                  hpnorm=Iplus_norm,
                                  hcnorm=Icross_norm,
                                  out=self.snr_mem,
                                  thresh=self.snr_threshold,
                                  analyse_slice=stilde.analyze)
        # FIXME: This should live further down
        # Convert output to pycbc TimeSeries
        delta_t = 1.0 / (self.tlen * stilde.delta_f)

        snr = TimeSeries(snr, epoch=stilde.start_time, delta_t=delta_t,
                         copy=False)

        from pycbc import events

        idx, snrv = events.threshold_real_and_cluster_findchirp(
            snr[stilde.analyze], self.snr_threshold, window
        )

        if len(idx) == 0:
            return [], 0, 0, [], [], [], [], 0, 0, 0
        logger.info("%d points above threshold", len(idx))

        logger.info("%d clustered points", len(idx))
        # erased self.
        u_vals, coa_phase = self._maximized_extrinsic_params\
            (I_plus.data, I_cross.data, hplus_cross_corr,
             indices=idx+stilde.analyze.start, hpnorm=Iplus_norm,
             hcnorm=Icross_norm)



        return snr, Iplus_corr, Icross_corr, idx, snrv, u_vals, coa_phase,\
                                      hplus_cross_corr, Iplus_norm, Icross_norm

    def _maximized_snr(self, hplus, hcross, hphccorr, **kwargs):
        return compute_max_snr_over_sky_loc_stat(hplus, hcross, hphccorr,
                                                 **kwargs)

    def _maximized_extrinsic_params(self, hplus, hcross, hphccorr, **kwargs):
        return compute_u_val_for_sky_loc_stat(hplus, hcross, hphccorr,
                                              **kwargs)


class MatchedFilterSkyMaxControlNoPhase(MatchedFilterSkyMaxControl):
    # Basically the same as normal SkyMaxControl, except we use a slight
    # variation in the internal SNR functions.
    def _maximized_snr(self, hplus, hcross, hphccorr, **kwargs):
        return compute_max_snr_over_sky_loc_stat_no_phase(hplus, hcross,
                                                          hphccorr, **kwargs)

    def _maximized_extrinsic_params(self, hplus, hcross, hphccorr, **kwargs):
        return compute_u_val_for_sky_loc_stat_no_phase(hplus, hcross, hphccorr,
                                                       **kwargs)

def make_frequency_series(vec):
    """Return a frequency series of the input vector.

    If the input is a frequency series it is returned, else if the input
    vector is a real time series it is fourier transformed and returned as a
    frequency series.

    Parameters
    ----------
    vector : TimeSeries or FrequencySeries

    Returns
    -------
    Frequency Series: FrequencySeries
        A frequency domain version of the input vector.
    """
    if isinstance(vec, FrequencySeries):
        return vec
    if isinstance(vec, TimeSeries):
        N = len(vec)
        n = N // 2 + 1
        delta_f = 1.0 / N / vec.delta_t
        # The FFT backend overwrites every output element.
        vectilde = FrequencySeries(empty(n, dtype=complex_same_precision_as(vec)),
                                    delta_f=delta_f, copy=False)
        fft(vec, vectilde)
        return vectilde
    else:
        raise TypeError("Can only convert a TimeSeries to a FrequencySeries")

def sigmasq_series(htilde, psd=None, low_frequency_cutoff=None,
            high_frequency_cutoff=None):
    """Return a cumulative sigmasq frequency series.

    Return a frequency series containing the accumulated power in the input
    up to that frequency.

    Parameters
    ----------
    htilde : TimeSeries or FrequencySeries
        The input vector
    psd : {None, FrequencySeries}, optional
        The psd used to weight the accumulated power.
    low_frequency_cutoff : {None, float}, optional
        The frequency to begin accumulating power. If None, start at the beginning
        of the vector.
    high_frequency_cutoff : {None, float}, optional
        The frequency to stop considering accumulated power. If None, continue
        until the end of the input vector.

    Returns
    -------
    Frequency Series: FrequencySeries
        A frequency series containing the cumulative sigmasq.
    """
    htilde = make_frequency_series(htilde)
    N = (len(htilde)-1) * 2
    norm = 4.0 * htilde.delta_f
    kmin, kmax = get_cutoff_indices(low_frequency_cutoff,
                                   high_frequency_cutoff, htilde.delta_f, N)

    sigma_vec = FrequencySeries(zeros(len(htilde), dtype=real_same_precision_as(htilde)),
                                delta_f = htilde.delta_f, copy=False)

    mag = htilde.squared_norm()

    if psd is not None:
        mag /= psd

    sigma_vec[kmin:kmax] = mag[kmin:kmax].cumsum()

    return sigma_vec*norm


def sigmasq(htilde, psd = None, low_frequency_cutoff=None,
            high_frequency_cutoff=None):
    """Return the loudness of the waveform. This is defined (see Duncan
    Brown's thesis) as the unnormalized matched-filter of the input waveform,
    htilde, with itself. This quantity is usually referred to as (sigma)^2
    and is then used to normalize matched-filters with the data.

    Parameters
    ----------
    htilde : TimeSeries or FrequencySeries
        The input vector containing a waveform.
    psd : {None, FrequencySeries}, optional
        The psd used to weight the accumulated power.
    low_frequency_cutoff : {None, float}, optional
        The frequency to begin considering waveform power.
    high_frequency_cutoff : {None, float}, optional
        The frequency to stop considering waveform power.

    Returns
    -------
    sigmasq: float
    """
    htilde = make_frequency_series(htilde)
    N = (len(htilde)-1) * 2
    norm = 4.0 * htilde.delta_f
    kmin, kmax = get_cutoff_indices(low_frequency_cutoff,
                                   high_frequency_cutoff, htilde.delta_f, N)
    ht = htilde[kmin:kmax]

    if psd:
        try:
            numpy.testing.assert_almost_equal(ht.delta_f, psd.delta_f)
        except AssertionError:
            raise ValueError('Waveform does not have same delta_f as psd')

    if psd is None:
        sq = ht.inner(ht)
    else:
        sq = ht.weighted_inner(ht, psd[kmin:kmax])

    return sq.real * norm

def sigma(htilde, psd = None, low_frequency_cutoff=None,
        high_frequency_cutoff=None):
    """ Return the sigma of the waveform. See sigmasq for more details.

    Parameters
    ----------
    htilde : TimeSeries or FrequencySeries
        The input vector containing a waveform.
    psd : {None, FrequencySeries}, optional
        The psd used to weight the accumulated power.
    low_frequency_cutoff : {None, float}, optional
        The frequency to begin considering waveform power.
    high_frequency_cutoff : {None, float}, optional
        The frequency to stop considering waveform power.

    Returns
    -------
    sigmasq: float
    """
    return sqrt(sigmasq(htilde, psd, low_frequency_cutoff, high_frequency_cutoff))

def get_cutoff_indices(flow, fhigh, df, N):
    """
    Gets the indices of a frequency series at which to stop an overlap
    calculation.

    Parameters
    ----------
    flow: float
        The frequency (in Hz) of the lower index.
    fhigh: float
        The frequency (in Hz) of the upper index.
    df: float
        The frequency step (in Hz) of the frequency series.
    N: int
        The number of points in the **time** series. Can be odd
        or even.

    Returns
    -------
    kmin: int
    kmax: int
    """
    if flow:
        kmin = int(flow / df)
        if kmin < 0:
            err_msg = "Start frequency cannot be negative. "
            err_msg += "Supplied value and kmin {} and {}".format(flow, kmin)
            raise ValueError(err_msg)
    else:
        kmin = 1
    if fhigh:
        kmax = int(fhigh / df)
        if kmax > int((N + 1)/2.):
            kmax = int((N + 1)/2.)
    else:
        # int() truncates towards 0, so this is
        # equivalent to the floor of the float
        kmax = int((N + 1)/2.)

    if kmax <= kmin:
        err_msg = "Kmax cannot be less than or equal to kmin. "
        err_msg += "Provided values of freqencies (min,max) were "
        err_msg += "{} and {} ".format(flow, fhigh)
        err_msg += "corresponding to (kmin, kmax) of "
        err_msg += "{} and {}.".format(kmin, kmax)
        raise ValueError(err_msg)

    return kmin,kmax

def matched_filter_core(template, data, psd=None, low_frequency_cutoff=None,
                  high_frequency_cutoff=None, h_norm=None, out=None, corr_out=None):
    """ Return the complex snr and normalization.

    Return the complex snr, along with its associated normalization of the template,
    matched filtered against the data.

    Parameters
    ----------
    template : TimeSeries or FrequencySeries
        The template waveform
    data : TimeSeries or FrequencySeries
        The strain data to be filtered.
    psd : {FrequencySeries}, optional
        The noise weighting of the filter.
    low_frequency_cutoff : {None, float}, optional
        The frequency to begin the filter calculation. If None, begin at the
        first frequency after DC.
    high_frequency_cutoff : {None, float}, optional
        The frequency to stop the filter calculation. If None, continue to the
        the nyquist frequency.
    h_norm : {None, float}, optional
        The template normalization. If none, this value is calculated internally.
    out : {None, Array}, optional
        An array to use as memory for snr storage. If None, memory is allocated
        internally.
    corr_out : {None, Array}, optional
        An array to use as memory for correlation storage. If None, memory is allocated
        internally. If provided, management of the vector is handled externally by the
        caller. No zero'ing is done internally.

    Returns
    -------
    snr : TimeSeries
        A time series containing the complex snr.
    correlation: FrequencySeries
        A frequency series containing the correlation vector.
    norm : float
        The normalization of the complex snr.
    """
    htilde = make_frequency_series(template)
    stilde = make_frequency_series(data)

    if len(htilde) != len(stilde):
        raise ValueError("Length of template and data must match")

    N = (len(stilde)-1) * 2
    kmin, kmax = get_cutoff_indices(low_frequency_cutoff,
                                   high_frequency_cutoff, stilde.delta_f, N)

    if corr_out is not None:
        qtilde = corr_out
    else:
        qtilde = zeros(N, dtype=complex_same_precision_as(data))

    if out is None:
        # The inverse FFT backend overwrites every output element.
        _q = empty(N, dtype=complex_same_precision_as(data))
    elif (len(out) == N) and type(out) is Array and out.kind =='complex':
        _q = out
    else:
        raise TypeError('Invalid Output Vector: wrong length or dtype')

    correlate(htilde[kmin:kmax], stilde[kmin:kmax], qtilde[kmin:kmax])

    if psd is not None:
        if isinstance(psd, FrequencySeries):
            try:
                numpy.testing.assert_almost_equal(stilde.delta_f, psd.delta_f)
            except AssertionError:
                raise ValueError("PSD delta_f does not match data")
            qtilde[kmin:kmax] /= psd[kmin:kmax]
        else:
            raise TypeError("PSD must be a FrequencySeries")

    ifft(qtilde, _q)

    if h_norm is None:
        h_norm = sigmasq(htilde, psd, low_frequency_cutoff, high_frequency_cutoff)

    norm = (4.0 * stilde.delta_f) / sqrt( h_norm)

    return (TimeSeries(_q, epoch=stilde._epoch, delta_t=stilde.delta_t, copy=False),
           FrequencySeries(qtilde, epoch=stilde._epoch, delta_f=stilde.delta_f, copy=False),
           norm)

def smear(idx, factor):
    """
    This function will take as input an array of indexes and return every
    unique index within the specified factor of the inputs.

    E.g.: smear([5,7,100],2) = [3,4,5,6,7,8,9,98,99,100,101,102]

    Parameters
    -----------
    idx : numpy.array of ints
        The indexes to be smeared.
    factor : idx
        The factor by which to smear out the input array.

    Returns
    --------
    new_idx : numpy.array of ints
        The smeared array of indexes.
    """


    s = [idx]
    for i in range(factor+1):
        a = i - factor/2
        s += [idx + a]
    return numpy.unique(numpy.concatenate(s))

def matched_filter(template, data, psd=None, low_frequency_cutoff=None,
                  high_frequency_cutoff=None, sigmasq=None):
    """ Return the complex snr.

    Return the complex snr, along with its associated normalization of the
    template, matched filtered against the data.

    Parameters
    ----------
    template : TimeSeries or FrequencySeries
        The template waveform
    data : TimeSeries or FrequencySeries
        The strain data to be filtered.
    psd : FrequencySeries
        The noise weighting of the filter.
    low_frequency_cutoff : {None, float}, optional
        The frequency to begin the filter calculation. If None, begin at the
        first frequency after DC.
    high_frequency_cutoff : {None, float}, optional
        The frequency to stop the filter calculation. If None, continue to the
        the nyquist frequency.
    sigmasq : {None, float}, optional
        The template normalization. If none, this value is calculated
        internally.

    Returns
    -------
    snr : TimeSeries
        A time series containing the complex snr.
    """
    snr, _, norm = matched_filter_core(template, data, psd=psd,
            low_frequency_cutoff=low_frequency_cutoff,
            high_frequency_cutoff=high_frequency_cutoff, h_norm=sigmasq)
    return snr * norm

_snr = None
_snr_scheme_key = None


def match(
    vec1,
    vec2,
    psd=None,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
    v1_norm=None,
    v2_norm=None,
    subsample_interpolation=False,
    return_phase=False,
):
    """Return the match between the two TimeSeries or FrequencySeries.

    Return the match between two waveforms. This is equivalent to the overlap
    maximized over time and phase.

    The maximization is only performed with discrete time-shifts,
    or a quadratic interpolation of them if the subsample_interpolation
    option is turned on; for a more precise computation
    of the match between two waveforms, use the optimized_match function.
    The accuracy of this function is guaranteed up to the fourth decimal place.

    Parameters
    ----------
    vec1 : TimeSeries or FrequencySeries
        The input vector containing a waveform.
    vec2 : TimeSeries or FrequencySeries
        The input vector containing a waveform.
    psd : Frequency Series
        A power spectral density to weight the overlap.
    low_frequency_cutoff : {None, float}, optional
        The frequency to begin the match.
    high_frequency_cutoff : {None, float}, optional
        The frequency to stop the match.
    v1_norm : {None, float}, optional
        The normalization of the first waveform. This is equivalent to its
        sigmasq value. If None, it is internally calculated.
    v2_norm : {None, float}, optional
        The normalization of the second waveform. This is equivalent to its
        sigmasq value. If None, it is internally calculated.
    subsample_interpolation : {False, bool}, optional
        If True the peak will be interpolated between samples using a simple
        quadratic fit. This can be important if measuring matches very close to
        1 and can cause discontinuities if you don't use it as matches move
        between discrete samples. If True the index returned will be a float.
    return_phase : {False, bool}, optional
        If True, also return the phase shift that gives the match.

    Returns
    -------
    match: float
    index: int
        The number of samples to shift to get the match.
    phi: float
        Phase to rotate complex waveform to get the match, if desired.
    """

    htilde = make_frequency_series(vec1)
    stilde = make_frequency_series(vec2)

    N = (len(htilde) - 1) * 2

    global _snr, _snr_scheme_key
    scheme_key = pycbc.scheme.current_backend_key()
    if (
        _snr is None
        or _snr.dtype != htilde.dtype
        or len(_snr) != N
        or _snr_scheme_key != scheme_key
    ):
        _snr = zeros(N, dtype=complex_same_precision_as(vec1))
        _snr_scheme_key = scheme_key
    snr, _, snr_norm = matched_filter_core(
        htilde,
        stilde,
        psd,
        low_frequency_cutoff,
        high_frequency_cutoff,
        v1_norm,
        out=_snr,
    )
    maxsnr, max_id = snr.abs_max_loc()
    if v2_norm is None:
        v2_norm = sigmasq(stilde, psd, low_frequency_cutoff, high_frequency_cutoff)

    if subsample_interpolation:
        # This uses the implementation coded up in sbank. Thanks Nick!
        # The maths for this is well summarized here:
        # https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
        # We use adjacent points to interpolate, but wrap off the end if needed
        left = abs(snr[-1]) if max_id == 0 else abs(snr[max_id - 1])
        middle = maxsnr
        right = abs(snr[0]) if max_id == (len(snr) - 1) else abs(snr[max_id + 1])
        # Get derivatives
        id_shift, maxsnr = quadratic_interpolate_peak(left, middle, right)
        max_id = max_id + id_shift

    if return_phase:
        rounded_max_id = int(round(max_id))
        phi = numpy.angle(snr[rounded_max_id])
        return maxsnr * snr_norm / sqrt(v2_norm), max_id, phi
    else:
        return maxsnr * snr_norm / sqrt(v2_norm), max_id

def overlap(vec1, vec2, psd=None, low_frequency_cutoff=None,
          high_frequency_cutoff=None, normalized=True):
    """ Return the overlap between the two TimeSeries or FrequencySeries.

    Parameters
    ----------
    vec1 : TimeSeries or FrequencySeries
        The input vector containing a waveform.
    vec2 : TimeSeries or FrequencySeries
        The input vector containing a waveform.
    psd : Frequency Series
        A power spectral density to weight the overlap.
    low_frequency_cutoff : {None, float}, optional
        The frequency to begin the overlap.
    high_frequency_cutoff : {None, float}, optional
        The frequency to stop the overlap.
    normalized : {True, boolean}, optional
        Set if the overlap is normalized. If true, it will range from 0 to 1.

    Returns
    -------
    overlap: float
    """

    return overlap_cplx(vec1, vec2, psd=psd, \
            low_frequency_cutoff=low_frequency_cutoff,\
            high_frequency_cutoff=high_frequency_cutoff,\
            normalized=normalized).real

def overlap_cplx(vec1, vec2, psd=None, low_frequency_cutoff=None,
          high_frequency_cutoff=None, normalized=True):
    """Return the complex overlap between the two TimeSeries or FrequencySeries.

    Parameters
    ----------
    vec1 : TimeSeries or FrequencySeries
        The input vector containing a waveform.
    vec2 : TimeSeries or FrequencySeries
        The input vector containing a waveform.
    psd : Frequency Series
        A power spectral density to weight the overlap.
    low_frequency_cutoff : {None, float}, optional
        The frequency to begin the overlap.
    high_frequency_cutoff : {None, float}, optional
        The frequency to stop the overlap.
    normalized : {True, boolean}, optional
        Set if the overlap is normalized. If true, it will range from 0 to 1.

    Returns
    -------
    overlap: complex
    """
    htilde = make_frequency_series(vec1)
    stilde = make_frequency_series(vec2)

    kmin, kmax = get_cutoff_indices(low_frequency_cutoff,
            high_frequency_cutoff, stilde.delta_f, (len(stilde)-1) * 2)

    if psd:
        inner = (htilde[kmin:kmax]).weighted_inner(stilde[kmin:kmax], psd[kmin:kmax])
    else:
        inner = (htilde[kmin:kmax]).inner(stilde[kmin:kmax])

    if normalized:
        sig1 = sigma(vec1, psd=psd, low_frequency_cutoff=low_frequency_cutoff,
                     high_frequency_cutoff=high_frequency_cutoff)
        sig2 = sigma(vec2, psd=psd, low_frequency_cutoff=low_frequency_cutoff,
                     high_frequency_cutoff=high_frequency_cutoff)
        norm = 1 / sig1 / sig2
    else:
        norm = 1

    return 4 * htilde.delta_f * inner * norm

def quadratic_interpolate_peak(left, middle, right):
    """ Interpolate the peak and offset using a quadratic approximation

    Parameters
    ----------
    left : numpy array
        Values at a relative bin value of [-1]
    middle : numpy array
        Values at a relative bin value of [0]
    right : numpy array
        Values at a relative bin value of [1]

    Returns
    -------
    bin_offset : numpy array
        Array of bins offsets, each in the range [-1/2, 1/2]
    peak_values : numpy array
        Array of the estimated peak values at the interpolated offset
    """
    bin_offset = 1.0/2.0 * (left - right) / (left - 2 * middle + right)
    peak_value = middle - 0.25 * (left - right) * bin_offset
    return bin_offset, peak_value


def _torch_cpu_native_batch_peak_enabled():
    """Read the strict, default-off native Torch-CPU peak gate."""
    value = os.environ.get(_TORCH_CPU_NATIVE_BATCH_PEAK_GATE)
    if value is None:
        return False
    normalized = value.strip().lower()
    if normalized in _TORCH_CPU_NATIVE_BATCH_PEAK_TRUE:
        return True
    if normalized in _TORCH_CPU_NATIVE_BATCH_PEAK_FALSE:
        return False
    choices = ", ".join(sorted(
        _TORCH_CPU_NATIVE_BATCH_PEAK_TRUE
        | _TORCH_CPU_NATIVE_BATCH_PEAK_FALSE
    ))
    raise ValueError(
        f"{_TORCH_CPU_NATIVE_BATCH_PEAK_GATE} must be one of: "
        f"{choices}; got {value!r}"
    )


def _try_torch_cpu_native_batch_peak_values(
        output, tensor, template_count, template_size, segment):
    """Return exact standard-CPU peaks, or request the Torch fallback."""
    if not _torch_cpu_native_batch_peak_enabled():
        return None

    # Imports remain behind the explicit gate.  In particular, standard CPU,
    # CUDA, MPS, and gate-off Torch filtering retain their established route.
    try:
        from . import matchedfilter_cpu, matchedfilter_torch

        if template_count < 1:
            return None
        total = template_count * template_size
        if (
            total > _TORCH_CPU_NATIVE_BATCH_PEAK_MAX_LENGTH
            or template_count > _TORCH_CPU_NATIVE_BATCH_PEAK_MAX_LENGTH
            or template_size > _TORCH_CPU_NATIVE_BATCH_PEAK_MAX_LENGTH
            or not matchedfilter_torch._batch_tensor_contract(tensor, total)
        ):
            return None

        start, stop, step = segment.indices(template_size)
        if step != 1 or start >= stop:
            return None
        runtime = matchedfilter_torch._cpu_native_openmp_runtime(
            matchedfilter_cpu
        )
        if not matchedfilter_torch._cpu_native_batch_runtime_is_stable(
            runtime
        ):
            return None

        owner_tensor = tensor
        pointer = tensor.data_ptr()
        version = tensor._version
        pid = os.getpid()
        thread_id = threading.get_ident()
        values = tensor.detach().numpy()
        if values.__array_interface__["data"][0] != pointer:
            return None
        indices = numpy.empty(template_count, dtype=numpy.int64)
        peaks = numpy.empty(template_count, dtype=numpy.complex64)

        # Revalidate immediately before crossing the opaque Cython boundary.
        # The local owners keep the Tensor and its storage alive for the call.
        if (
            os.getpid() != pid
            or threading.get_ident() != thread_id
            or backend_array(output, "torch") is not owner_tensor
            or owner_tensor.data_ptr() != pointer
            or owner_tensor._version != version
            or not matchedfilter_torch._batch_tensor_contract(
                owner_tensor, total
            )
            or not matchedfilter_torch._cpu_native_batch_runtime_is_stable(
                runtime
            )
        ):
            return None

        matchedfilter_cpu._batch_abs_arg_max_complex64(
            values,
            indices,
            peaks,
            template_size,
            start,
            stop,
            template_count,
        )
        if (
            os.getpid() != pid
            or threading.get_ident() != thread_id
            or backend_array(output, "torch") is not owner_tensor
            or owner_tensor.data_ptr() != pointer
            or owner_tensor._version != version
        ):
            return None
        return indices, peaks
    except (
        AttributeError,
        ImportError,
        OSError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        # Only private result arrays can have been written.  The established
        # Torch helper below remains safe after every admission/setup failure.
        return None


def _torch_cuda_native_batch_peak_enabled():
    """Read the strict, default-off native Torch-CUDA peak gate."""
    value = os.environ.get(_TORCH_CUDA_NATIVE_BATCH_PEAK_GATE)
    if value is None:
        return False
    normalized = value.strip().lower()
    if normalized in _TORCH_CPU_NATIVE_BATCH_PEAK_TRUE:
        return True
    if normalized in _TORCH_CPU_NATIVE_BATCH_PEAK_FALSE:
        return False
    choices = ", ".join(sorted(
        _TORCH_CPU_NATIVE_BATCH_PEAK_TRUE
        | _TORCH_CPU_NATIVE_BATCH_PEAK_FALSE
    ))
    raise ValueError(
        f"{_TORCH_CUDA_NATIVE_BATCH_PEAK_GATE} must be one of: "
        f"{choices}; got {value!r}"
    )


def _try_torch_cuda_native_batch_peak_values(
        output, tensor, template_count, template_size, segment):
    """Return exact standard-CUDA peaks, or request the Torch fallback."""
    if not _torch_cuda_native_batch_peak_enabled():
        return None

    try:
        from . import matchedfilter_torch

        if template_count <= 1:
            return None
        total = template_count * template_size
        if not matchedfilter_torch._cuda_batch_tensor_contract(tensor, total):
            return None

        start, stop, step = segment.indices(template_size)
        if step != 1 or start >= stop:
            return None

        values = tensor.reshape(template_count, template_size)[:, segment]
        if values.shape[1] == 0:
            return None

        indices, peaks = matchedfilter_torch.standard_peak_tensor(values)
        return (
            indices.detach().cpu().numpy(),
            peaks.detach().cpu().numpy(),
        )
    except (
        AttributeError,
        ImportError,
        OSError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return None


def _torch_ondevice_peaks_enabled(device_type=None):
    """Read the PYCBC_TORCH_ONDEVICE_PEAKS environment flag."""
    default = False
    value = os.environ.get(_TORCH_ONDEVICE_PEAKS_GATE)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TORCH_CPU_NATIVE_BATCH_PEAK_TRUE:
        return True
    if normalized in _TORCH_CPU_NATIVE_BATCH_PEAK_FALSE:
        return False
    choices = ", ".join(sorted(
        _TORCH_CPU_NATIVE_BATCH_PEAK_TRUE
        | _TORCH_CPU_NATIVE_BATCH_PEAK_FALSE
    ))
    raise ValueError(
        f"{_TORCH_ONDEVICE_PEAKS_GATE} must be one of: {choices}; got {value!r}"
    )


def _torch_batch_peak_and_threshold_gpu(*args, **kwargs):
    """On-device peak extraction and thresholding on CUDA/MPS."""
    if len(args) == 4 or (
        len(args) >= 1 and hasattr(args[0], "ndim") and args[0].ndim == 2
    ):
        from . import matchedfilter_torch
        return matchedfilter_torch._torch_batch_peak_and_threshold_gpu(
            *args, **kwargs
        )

    output = args[0]
    template_count = int(args[1])
    template_size = int(args[2])
    segment = args[3]
    norms = args[4]
    snr_threshold = args[5]
    snr_abort_threshold = (
        args[6] if len(args) > 6 else kwargs.get("snr_abort_threshold", None)
    )

    if not isinstance(pycbc.scheme.mgr.state, pycbc.scheme.TorchScheme):
        return None

    tensor = backend_array(output, "torch")
    if tensor is None or tensor.ndim != 1:
        return None

    if not _torch_ondevice_peaks_enabled(tensor.device.type):
        return None

    if (
        template_count < 1
        or template_size < 1
        or tensor.numel() != template_count * template_size
    ):
        return None

    if segment.step not in (None, 1):
        return None

    try:
        from . import matchedfilter_torch
        values = tensor.reshape(template_count, template_size)[:, segment]
        if values.shape[1] == 0:
            return None
        return matchedfilter_torch._torch_batch_peak_and_threshold_gpu(
            values, norms, snr_threshold, snr_abort_threshold
        )
    except Exception:
        return None


def _torch_batch_peak_values(output, template_count, template_size, segment):
    """Materialize one peak index and value per contiguous Torch output.

    The batched live filter stores every template output consecutively in one
    allocation.  Reducing that allocation at once avoids two scalar device
    synchronizations per template while retaining the existing host-facing
    peak contract.  ``None`` requests the legacy per-template fallback.
    """
    if not isinstance(pycbc.scheme.mgr.state, pycbc.scheme.TorchScheme):
        return None

    tensor = backend_array(output, "torch")
    if tensor is None or tensor.ndim != 1:
        return None

    template_count = int(template_count)
    template_size = int(template_size)
    if (
        template_count < 1
        or template_size < 1
        or tensor.numel() != template_count * template_size
    ):
        return None

    if segment.step not in (None, 1):
        return None

    if tensor.device.type == "cpu":
        native = _try_torch_cpu_native_batch_peak_values(
            output,
            tensor,
            template_count,
            template_size,
            segment,
        )
        if native is not None:
            return native
    elif tensor.device.type == "cuda":
        native = _try_torch_cuda_native_batch_peak_values(
            output,
            tensor,
            template_count,
            template_size,
            segment,
        )
        if native is not None:
            return native

    import torch

    values = tensor.reshape(template_count, template_size)[:, segment]
    if values.shape[1] == 0:
        return None
    if values.is_complex():
        sq_mag = torch.view_as_real(values).square().sum(dim=-1)
    else:
        sq_mag = values.square()
    indices = torch.argmax(sq_mag, dim=-1)
    peaks = values[torch.arange(values.shape[0], device=values.device), indices]
    return (
        indices.detach().cpu().numpy(),
        peaks.detach().cpu().numpy(),
    )


def _cpu_batch_peak_values(output, template_count, template_size, segment):
    """Materialize one peak index and value per contiguous CPU output.

    Uses Cython OpenMP _batch_abs_arg_max_complex64 for direct vectorized
    peak extraction on standard CPU / NumPy memory, bypassing per-template
    slice allocations and Python-level scheme conversions.
    """
    if not _torch_cpu_native_batch_peak_enabled():
        return None

    if template_count <= 1:
        return None


    if segment.step not in (None, 1):
        return None

    raw = backend_array(output)
    if not isinstance(raw, numpy.ndarray):
        return None
    if raw.dtype != numpy.complex64:
        return None
    if not raw.flags.c_contiguous:
        return None

    template_count = int(template_count)
    template_size = int(template_size)
    if (
        template_count < 1
        or template_size < 1
        or raw.size != template_count * template_size
    ):
        return None

    start, stop, step = segment.indices(template_size)
    if step != 1 or start >= stop:
        return None

    try:
        from . import matchedfilter_cpu

        indices = numpy.empty(template_count, dtype=numpy.int64)
        peaks = numpy.empty(template_count, dtype=numpy.complex64)
        matchedfilter_cpu._batch_abs_arg_max_complex64(
            raw.ravel(),
            indices,
            peaks,
            template_size,
            start,
            stop,
            template_count,
        )
        return indices, peaks
    except Exception:
        return None


def _torch_batch_peak_magnitudes(peak_values):
    """Materialize live-batch magnitudes with scalar-loop precision.

    Peak extraction already crosses the Torch/host boundary once per batch.
    Computing the magnitudes together lets rejected templates avoid constructing
    one-element NumPy arrays while leaving normalization, thresholding, and all
    template side effects in their established per-template order.
    """
    peak_values = numpy.asarray(peak_values)
    if peak_values.ndim != 1:
        return None

    # NumPy's vector complex-absolute kernel can differ by one ULP from this
    # scalar contract.  Keep the scalar operation but amortize its surrounding
    # bookkeeping and avoid the rejected templates' one-element arrays.
    return numpy.fromiter(
        (abs(peak.item()) for peak in peak_values),
        dtype=numpy.float64,
        count=len(peak_values),
    )


# Minimum templates threshold is disabled (0) so all batch sizes use the fast
# vectorized magnitude reduction path.
_TORCH_BATCH_PEAK_THRESHOLD_MIN_TEMPLATES = 0


def _can_batch_torch_vetoes(power_chisq, sg_chisq):
    """Return whether the standard veto calculators can use bulk copies."""
    if not isinstance(pycbc.scheme.mgr.state, pycbc.scheme.TorchScheme):
        return False

    # Subclasses may rely on scalar materialization occurring before the SG
    # veto is evaluated, so only optimize the two calculators whose contracts
    # are known here.
    from pycbc.vetoes.chisq import SingleDetPowerChisq
    from pycbc.vetoes.sgchisq import SingleDetSGChisq

    return (
        type(power_chisq) is SingleDetPowerChisq
        and type(sg_chisq) is SingleDetSGChisq
    )


def _materialize_torch_veto_results(veto_values):
    """Copy standard one-trigger veto results to their public NumPy arrays.

    ``None`` indicates that a calculator returned a non-standard value and
    the caller should use the existing scalar conversion path instead.
    """
    import torch

    if not veto_values:
        return (
            numpy.zeros(0, dtype=numpy.float32),
            numpy.zeros(0, dtype=numpy.uint32),
            numpy.zeros(0, dtype=numpy.float32),
        )

    chisq_values = []
    dof_values = []
    sg_values = []
    device = None
    for chisq, dof, sg_chisq in veto_values:
        chisq_tensor = backend_array(chisq, "torch")
        dof_tensor = backend_array(dof, "torch")
        if (
            chisq_tensor is None
            or dof_tensor is None
            or tuple(chisq_tensor.shape) != (1,)
            or tuple(dof_tensor.shape) != (1,)
            or chisq_tensor.dtype != torch.float32
            or dof_tensor.dtype != torch.int64
        ):
            return None

        if device is None:
            device = chisq_tensor.device
        if chisq_tensor.device != device or dof_tensor.device != device:
            return None

        if sg_chisq is None:
            sg_tensor = torch.zeros_like(chisq_tensor)
        else:
            sg_tensor = backend_array(sg_chisq, "torch")
            if (
                sg_tensor is None
                or tuple(sg_tensor.shape) != (1,)
                or sg_tensor.dtype != torch.float32
                or sg_tensor.device != device
            ):
                return None

        chisq_values.append(chisq_tensor.reshape(1))
        dof_values.append(dof_tensor.reshape(1))
        sg_values.append(sg_tensor.reshape(1))

    dof_tensor = torch.cat(dof_values)
    float_values = torch.stack((
        torch.cat(chisq_values),
        torch.cat(sg_values),
    ))
    float_values = float_values.detach().cpu().numpy()
    dof_values = dof_tensor.detach().cpu().numpy()
    reduced_chisq = numpy.empty(len(dof_values), dtype=numpy.float32)
    public_dof = numpy.empty(len(dof_values), dtype=numpy.uint32)
    for index, (chisq, dof) in enumerate(zip(float_values[0], dof_values)):
        # Match Array scalar access: divide Python float/int values, then cast
        # into the float32 public result.  Besides preserving rounding, this
        # naturally retains ZeroDivisionError when the DOF is zero.
        reduced_chisq[index] = float(chisq) / int(dof)
        # Scalar assignment also preserves NumPy's OverflowError for negative
        # or out-of-range values instead of silently wrapping via ``astype``.
        public_dof[index] = int(dof)
    return (
        reduced_chisq,
        public_dof,
        float_values[1],
    )


def _is_cuda_scheme(templates=None):
    """Return whether the current scheme or templates use CUDA processing."""
    state = getattr(pycbc.scheme.mgr, "state", None)
    if state is not None:
        if isinstance(state, (pycbc.scheme.CUDAScheme, pycbc.scheme.CUPYScheme)):
            return True
        if isinstance(state, pycbc.scheme.TorchScheme):
            torch_dev = getattr(
                state, "torch_device", getattr(state, "device", None)
            )
            if torch_dev is not None and getattr(torch_dev, "type", None) == "cuda":
                return True
        if getattr(state, "prefix", None) in ("cuda", "cupy"):
            return True
    try:
        if pycbc.scheme.current_prefix() in ("cuda", "cupy"):
            return True
    except Exception:
        pass
    if templates and len(templates) > 0:
        t0 = templates[0]
        data = backend_array(t0)
        tensor = backend_array(t0, "torch")
        if tensor is not None and (
            getattr(tensor, "is_cuda", False)
            or getattr(getattr(tensor, "device", None), "type", None) == "cuda"
        ):
            return True
        device = getattr(data, "device", None)
        if device is not None and getattr(device, "type", None) == "cuda":
            return True
    return False


class LiveBatchMatchedFilter(object):
    """Calculate SNR and signal consistency tests in a batched progression"""

    def __init__(self, templates, snr_threshold, chisq_bins, sg_chisq,
                 maxelements=None,
                 snr_abort_threshold=None,
                 newsnr_threshold=None,
                 max_triggers_in_batch=None,
                 enable_cuda_graphs=None,
                 enable_async_streams=None):
        """Create a batched matchedfilter instance

        Parameters
        ----------
        templates: list of `FrequencySeries`
            List of templates from the FilterBank class.
        snr_threshold: float
            Minimum value to record peaks in the SNR time series.
        chisq_bins: str
            Str that determines how the number of chisq bins varies as a
            function of the template bank parameters.
        sg_chisq: pycbc.vetoes.SingleDetSGChisq
            Instance of the sg_chisq class to calculate sg_chisq with.
        maxelements: {int, None}, optional
            Maximum size in elements of a batched fourier transform. If None,
            defaults to 2**23 on CUDA (hardware-aware L2-cache resident default,
            yielding B_tile=64 templates per chunk for N=131,072) or 2**27 on
            CPU, or PYCBC_BATCH_MAXELEMENTS if specified in the environment.
        snr_abort_threshold: {float, None}
            If the SNR is above this threshold, do not record any triggers.
        newsnr_threshold: {float, None}
            Only record triggers that have a re-weighted NewSNR above this
            threshold.
        max_triggers_in_batch: {int, None}
            Record X number of the loudest triggers by SNR in each MPI
            process. Signal consistency values will also only be calculated
            for these triggers.
        enable_cuda_graphs: {bool, None}
            Enable CUDA Graph recording and replay on CUDA devices. If None,
            reads the PYCBC_ENABLE_CUDA_GRAPHS environment variable.
        enable_async_streams: {bool, None}
            Enable pipelined asynchronous stream support / double-buffering.
            If None, reads the PYCBC_TORCH_ASYNC_STREAMS environment variable.
        """
        self.snr_threshold = snr_threshold
        self.snr_abort_threshold = snr_abort_threshold
        self.newsnr_threshold = newsnr_threshold
        self.max_triggers_in_batch = max_triggers_in_batch
        if enable_cuda_graphs is None:
            env_val = os.environ.get("PYCBC_ENABLE_CUDA_GRAPHS", "")
            self.enable_cuda_graphs = env_val.strip().lower() in (
                "1", "true", "yes", "on"
            )
        else:
            self.enable_cuda_graphs = bool(enable_cuda_graphs)
        self._cuda_graphs = {}

        if enable_async_streams is None:
            env_val = os.environ.get(_TORCH_ASYNC_STREAMS_GATE, "")
            self.enable_async_streams = env_val.strip().lower() in (
                "1", "true", "yes", "on"
            )
        else:
            self.enable_async_streams = bool(enable_async_streams)
        self._async_streams = None
        self._async_prefetched = None

        if (
            "PYCBC_BATCH_MAXELEMENTS" in os.environ
            and os.environ["PYCBC_BATCH_MAXELEMENTS"].strip()
        ):
            maxelements = int(os.environ["PYCBC_BATCH_MAXELEMENTS"].strip())
        elif maxelements is None:
            from pycbc.hardware import get_optimal_batch_maxelements
            is_cuda = _is_cuda_scheme(templates)
            maxelements = get_optimal_batch_maxelements(is_cuda=is_cuda)

        from pycbc import vetoes
        self.power_chisq = vetoes.SingleDetPowerChisq(chisq_bins, None)
        self.sg_chisq = sg_chisq

        durations = numpy.array([1.0 / t.delta_f for t in templates])

        lsort = durations.argsort()
        durations = durations[lsort]
        templates = [templates[li] for li in lsort]

        # Figure out how to chunk together the templates into groups to process
        _, counts = numpy.unique(durations, return_counts=True)
        tsamples = [(len(t) - 1) * 2 for t in templates]
        unique_tsamples = numpy.unique(tsamples)
        grabs = numpy.maximum(1, maxelements // unique_tsamples)

        chunks = numpy.array([])
        num = 0
        for count, grab in zip(counts, grabs):
            chunks = numpy.append(chunks, numpy.arange(num, count + num, grab))
            chunks = numpy.append(chunks, [count + num])
            num += count
        chunks = numpy.unique(chunks).astype(numpy.uint32)

        # We now have how many templates to grab at a time.
        self.chunks = chunks[1:] - chunks[0:-1]

        self.out_mem = {}
        self.cout_mem = {}
        self.ifts = {}
        chunk_durations = [durations[i] for i in chunks[:-1]]
        self.chunk_tsamples = [tsamples[int(i)] for i in chunks[:-1]]
        samples = self.chunk_tsamples * self.chunks

        # Create workspace memory for correlate and snr
        mem_ids = [(a, b) for a, b in zip(chunk_durations, self.chunks)]
        mem_types = set(zip(mem_ids, samples))

        self.tgroups, self.mids = [], []
        for i, size in mem_types:
            dur, count = i
            self.out_mem[i] = zeros(size, dtype=numpy.complex64)
            self.cout_mem[i] = zeros(size, dtype=numpy.complex64)
            self.ifts[i] = IFFT(self.cout_mem[i], self.out_mem[i],
                                nbatch=count,
                                size=len(self.cout_mem[i]) // count)

        # Split the templates into their processing groups
        for dur, count in mem_ids:
            tgroup = templates[0:count]
            self.tgroups.append(tgroup)
            self.mids.append((dur, count))
            templates = templates[count:]

        # Associate the snr and corr memory block to each template
        self.corr = []
        for i, tgroup in enumerate(self.tgroups):
            psize = self.chunk_tsamples[i]
            s = 0
            e = psize
            mid = self.mids[i]
            for htilde in tgroup:
                htilde.out = self.out_mem[mid][s:e]
                htilde.cout = self.cout_mem[mid][s:e]
                htilde._mid = mid
                htilde._tgroup = tgroup
                s += psize
                e += psize
            self.corr.append(BatchCorrelator(tgroup, [t.cout for t in tgroup], len(tgroup[0])))

        self.power_matrices = {}
        self._psd_cache = {}
        if isinstance(pycbc.scheme.mgr.state, pycbc.scheme.TorchScheme):
            for mid, tgroup in zip(self.mids, self.tgroups):
                try:
                    p_list = []
                    for htilde in tgroup:
                        arr = getattr(htilde, 'data', htilde)
                        if hasattr(arr, 'numpy'):
                            arr = arr.numpy()
                        else:
                            arr = numpy.asarray(arr)
                        if numpy.iscomplexobj(arr):
                            power = (arr.real ** 2 + arr.imag ** 2).astype(
                                numpy.float32
                            )
                        else:
                            power = (arr ** 2).astype(numpy.float32)
                        p_list.append(power)
                    if p_list:
                        self.power_matrices[mid] = numpy.stack(p_list, axis=0)
                except Exception:
                    pass

    def set_data(self, data):
        """Set the data reader object to use"""
        self.data = data
        self.block_id = 0
        self._async_prefetched = None

    def combine_results(self, results):
        """Combine results from different batches of filtering"""
        result = {}
        for key in results[0]:
            result[key] = numpy.concatenate([r[key] for r in results])
        return result

    def process_data(self, data_reader):
        """Process the data for all of the templates"""
        self.set_data(data_reader)
        return self.process_all()

    def process_all(self):
        """Process every batch group and return as single result"""
        with _torch_inference_mode_context():
            results = []
            veto_info = []
            while 1:
                result, veto = self._process_batch()
                if result is False: return False
                if result is None: break
                results.append(result)
                veto_info += veto

            result = self.combine_results(results)

            if self.max_triggers_in_batch:
                sort = result['snr'].argsort()[::-1][:self.max_triggers_in_batch]
                for key in result:
                    result[key] = result[key][sort]

                tmp = veto_info
                veto_info = [tmp[i] for i in sort]

            result = self._process_vetoes(result, veto_info)
            return result

    def _process_vetoes(self, results, veto_info):
        """Calculate signal based vetoes"""
        chisq = numpy.array(numpy.zeros(len(veto_info)), numpy.float32, ndmin=1)
        dof = numpy.array(numpy.zeros(len(veto_info)), numpy.uint32, ndmin=1)
        sg_chisq = numpy.array(numpy.zeros(len(veto_info)), numpy.float32,
                               ndmin=1)
        results['chisq'] = chisq
        results['chisq_dof'] = dof
        results['sg_chisq'] = sg_chisq

        if self.newsnr_threshold:
            from pycbc.events import ranking

        keep = []
        batch_torch_vetoes = _can_batch_torch_vetoes(
            self.power_chisq, self.sg_chisq
        )
        veto_values = []
        for i, (snrv, norm, l, htilde, stilde) in enumerate(veto_info):
            mid = getattr(htilde, '_mid', None)
            cout_mem = self.cout_mem.get(mid) if mid is not None else None
            is_valid = (
                getattr(htilde, '_corr_valid', False)
                and getattr(htilde, '_corr_stilde', None) is stilde
                and (
                    cout_mem is None
                    or (
                        getattr(cout_mem, '_active_tgroup', None) is getattr(htilde, '_tgroup', None)
                        and getattr(cout_mem, '_active_stilde', None) is stilde
                    )
                )
            )
            if not is_valid:
                correlate(htilde, stilde, htilde.cout)
                htilde._corr_valid = True
                htilde._corr_stilde = stilde
                if cout_mem is not None:
                    cout_mem._active_tgroup = getattr(htilde, '_tgroup', None)
                    cout_mem._active_stilde = stilde
            c, d = self.power_chisq.values(htilde.cout, snrv,
                                           norm, stilde.psd, [l], htilde)
            if c is not None and d is not None:
                if not batch_torch_vetoes:
                    chisq[i] = c[0] / d[0]
                    dof[i] = d[0]

            sgv = self.sg_chisq.values(stilde, htilde, stilde.psd,
                                       snrv, norm, c, d, [l])
            if batch_torch_vetoes:
                veto_values.append((c, d, sgv))
            elif sgv is not None:
                sg_chisq[i] = sgv[0]

            if self.newsnr_threshold and not batch_torch_vetoes:
                newsnr = ranking.newsnr(results['snr'][i], chisq[i])
                if newsnr >= self.newsnr_threshold:
                    keep.append(i)

        if batch_torch_vetoes:
            materialized = _materialize_torch_veto_results(veto_values)
            if materialized is None:
                for i, (c, d, sgv) in enumerate(veto_values):
                    chisq[i] = c[0] / d[0]
                    dof[i] = d[0]
                    if sgv is not None:
                        sg_chisq[i] = sgv[0]
            else:
                chisq[:], dof[:], sg_chisq[:] = materialized

            if self.newsnr_threshold:
                for i in range(len(veto_info)):
                    newsnr = ranking.newsnr(results['snr'][i], chisq[i])
                    if newsnr >= self.newsnr_threshold:
                        keep.append(i)

        if self.newsnr_threshold:
            keep = numpy.array(keep, dtype=numpy.uint32)
            for key in results:
                results[key] = results[key][keep]

        return results

    def _try_cuda_graph_batch(self, block_id, mid, tgroup, psize, seg, stilde):
        """Execute or capture CUDA Graph for this batch block."""
        import torch

        if not torch.cuda.is_available() or not callable(
            getattr(torch.cuda, "CUDAGraph", None)
        ):
            return None

        stilde_tensor = backend_array(stilde, "torch")

        if stilde_tensor is None or stilde_tensor.device.type != "cuda":
            return None

        out_tensor = backend_array(self.out_mem[mid], "torch")

        if out_tensor is None or out_tensor.device != stilde_tensor.device:
            return None

        if not hasattr(self, "_cuda_graphs"):
            self._cuda_graphs = {}

        state = self._cuda_graphs.get(block_id)
        if state is not None:
            if (
                state["input_shape"] == tuple(stilde_tensor.shape)
                and state["input_dtype"] == stilde_tensor.dtype
                and state["psize"] == psize
                and state["seg"] == seg
                and state["device"] == stilde_tensor.device
            ):
                state["static_input"].copy_(stilde_tensor, non_blocking=True)
                state["graph"].replay()
                state["replays"] = state.get("replays", 0) + 1
                cout_mem = self.cout_mem.get(mid)
                if cout_mem is not None:
                    cout_mem._active_tgroup = tgroup
                    cout_mem._active_stilde = stilde
                for h in tgroup:
                    h._corr_valid = True
                    h._corr_stilde = stilde
                if "static_indices_cpu" in state:
                    state["static_indices_cpu"].copy_(
                        state["static_indices"], non_blocking=False
                    )
                    state["static_peaks_cpu"].copy_(
                        state["static_peaks"], non_blocking=False
                    )
                    return (
                        state["static_indices_cpu"].detach().cpu().numpy(),
                        state["static_peaks_cpu"].detach().cpu().numpy(),
                    )
                return (
                    state["static_indices"].detach().cpu().numpy(),
                    state["static_peaks"].detach().cpu().numpy(),
                )
            else:
                del self._cuda_graphs[block_id]

        try:
            device = stilde_tensor.device
            static_input = torch.zeros_like(stilde_tensor)
            static_indices = torch.zeros(
                len(tgroup), dtype=torch.int64, device=device
            )
            static_peaks = torch.zeros(
                len(tgroup), dtype=torch.complex64, device=device
            )

            origin = torch.cuda.current_stream(device)
            capture_stream = torch.cuda.Stream(device=device)
            capture_stream.wait_stream(origin)

            def _step():
                self.corr[block_id].execute(static_input)
                self.ifts[mid].execute()
                values = out_tensor.reshape(len(tgroup), psize)[:, seg]
                if values.is_complex():
                    sq_mag = torch.view_as_real(values).square().sum(dim=-1)
                    clean_mag = torch.nan_to_num(sq_mag, nan=0.0)
                    indices = torch.argmax(clean_mag, dim=-1)
                else:
                    clean_vals = torch.nan_to_num(torch.abs(values), nan=0.0)
                    indices = torch.argmax(clean_vals, dim=-1)
                peaks = values[torch.arange(values.shape[0], device=values.device), indices]
                static_indices.copy_(indices)
                static_peaks.copy_(peaks)

            with torch.cuda.stream(capture_stream), torch.no_grad():
                static_input.copy_(stilde_tensor, non_blocking=True)
                for _ in range(3):
                    _step()
                capture_stream.synchronize()
                try:
                    mempool = (
                        torch.cuda.graph_pool_handle()
                        if hasattr(torch.cuda, "graph_pool_handle")
                        else None
                    )
                except Exception:
                    mempool = None
                graph = torch.cuda.CUDAGraph()
                if mempool is not None:
                    try:
                        graph_ctx = torch.cuda.graph(
                            graph, stream=capture_stream, pool=mempool
                        )
                    except TypeError:
                        graph_ctx = torch.cuda.graph(
                            graph, stream=capture_stream
                        )
                else:
                    graph_ctx = torch.cuda.graph(graph, stream=capture_stream)
                with graph_ctx:
                    _step()

            origin.wait_stream(capture_stream)

            try:
                static_indices_cpu = torch.empty_like(
                    static_indices, device="cpu", pin_memory=True
                )
                static_peaks_cpu = torch.empty_like(
                    static_peaks, device="cpu", pin_memory=True
                )
            except Exception:
                static_indices_cpu = torch.empty_like(
                    static_indices, device="cpu"
                )
                static_peaks_cpu = torch.empty_like(
                    static_peaks, device="cpu"
                )

            self._cuda_graphs[block_id] = {
                "graph": graph,
                "static_input": static_input,
                "static_indices": static_indices,
                "static_peaks": static_peaks,
                "static_indices_cpu": static_indices_cpu,
                "static_peaks_cpu": static_peaks_cpu,
                "input_shape": tuple(stilde_tensor.shape),
                "input_dtype": stilde_tensor.dtype,
                "psize": psize,
                "seg": seg,
                "device": device,
                "capture_stream": capture_stream,
                "replays": 0,
            }

            cout_mem = self.cout_mem.get(mid)
            if cout_mem is not None:
                cout_mem._active_tgroup = tgroup
                cout_mem._active_stilde = stilde
            for h in tgroup:
                h._corr_valid = True
                h._corr_stilde = stilde

            static_indices_cpu.copy_(static_indices, non_blocking=False)
            static_peaks_cpu.copy_(static_peaks, non_blocking=False)
            return (
                static_indices_cpu.detach().cpu().numpy(),
                static_peaks_cpu.detach().cpu().numpy(),
            )
        except Exception:
            return None

    def _process_batch(self):
        """Process only a single batch group of data"""
        with _torch_inference_mode_context():
            if self.block_id == len(self.tgroups):
                return None, None

            tgroup = self.tgroups[self.block_id]
            psize = self.chunk_tsamples[self.block_id]
            mid = self.mids[self.block_id]

            out_tensor = backend_array(self.out_mem[mid], "torch")
            try:
                import torch
            except ImportError:
                torch = None

            is_cuda = (
                torch is not None
                and out_tensor is not None
                and out_tensor.device.type == "cuda"
                and getattr(self, "enable_async_streams", False)
            )

            current_stream = None
            if is_cuda:
                if getattr(self, "_async_streams", None) is None:
                    device = out_tensor.device
                    self._compute_stream = torch.cuda.Stream(device=device)
                    self._transfer_stream = torch.cuda.Stream(device=device)
                    self._transfer_event = torch.cuda.Event()
                    self._async_streams = (self._compute_stream, self._transfer_stream)

                current_stream = self._compute_stream

                if (
                    getattr(self, "_async_prefetched", None) is not None
                    and self._async_prefetched[0] == self.block_id
                ):
                    stilde, event = self._async_prefetched[1], self._async_prefetched[2]
                    self._async_prefetched = None
                    if event is not None:
                        self._compute_stream.wait_event(event)
                else:
                    stilde = self.data.overwhitened_data(tgroup[0].delta_f)
                    stilde_t = backend_array(stilde, "torch")
                    if stilde_t is not None and stilde_t.device.type == "cpu":
                        if not stilde_t.is_pinned():
                            stilde_pinned = stilde_t.pin_memory()
                        else:
                            stilde_pinned = stilde_t
                        with torch.cuda.stream(self._transfer_stream):
                            stilde_gpu = stilde_pinned.to(
                                device=out_tensor.device, non_blocking=True
                            )
                            self._transfer_event.record(self._transfer_stream)
                        self._compute_stream.wait_event(self._transfer_event)
                        stilde_psd = stilde.psd
                        if isinstance(stilde, Array):
                            stilde = stilde._return(wrap_backend_array(stilde_gpu))
                        elif hasattr(stilde, "delta_f"):
                            stilde = FrequencySeries(
                                wrap_backend_array(stilde_gpu),
                                delta_f=stilde.delta_f,
                                epoch=getattr(stilde, "_epoch", 0),
                                copy=False,
                            )
                        else:
                            stilde = stilde_gpu
                        # Series reconstruction preserves sampling metadata,
                        # but the PSD is attached separately by the data reader.
                        stilde.psd = stilde_psd

                # Pipelined prefetch for the next block (double buffering)
                next_block_id = self.block_id + 1
                if next_block_id < len(self.tgroups):
                    try:
                        next_tgroup = self.tgroups[next_block_id]
                        next_stilde = self.data.overwhitened_data(next_tgroup[0].delta_f)
                        next_stilde_t = backend_array(next_stilde, "torch")
                        if next_stilde_t is not None and next_stilde_t.device.type == "cpu":
                            if not next_stilde_t.is_pinned():
                                next_pinned = next_stilde_t.pin_memory()
                            else:
                                next_pinned = next_stilde_t
                            next_event = torch.cuda.Event()
                            with torch.cuda.stream(self._transfer_stream):
                                next_stilde_gpu = next_pinned.to(
                                    device=out_tensor.device, non_blocking=True
                                )
                                next_event.record(self._transfer_stream)
                            if isinstance(next_stilde, Array):
                                next_stilde_dev = next_stilde._return(
                                    wrap_backend_array(next_stilde_gpu)
                                )
                            elif hasattr(next_stilde, "delta_f"):
                                next_stilde_dev = FrequencySeries(
                                    wrap_backend_array(next_stilde_gpu),
                                    delta_f=next_stilde.delta_f,
                                    epoch=getattr(next_stilde, "_epoch", 0),
                                    copy=False,
                                )
                            else:
                                next_stilde_dev = next_stilde_gpu
                            next_stilde_dev.psd = next_stilde.psd
                            self._async_prefetched = (
                                next_block_id,
                                next_stilde_dev,
                                next_event,
                                next_pinned,
                            )
                        else:
                            self._async_prefetched = (next_block_id, next_stilde, None)
                    except Exception:
                        self._async_prefetched = None
            else:
                stilde = self.data.overwhitened_data(tgroup[0].delta_f)

            from contextlib import nullcontext
            stream_ctx = (
                torch.cuda.stream(current_stream)
                if (current_stream is not None and torch is not None)
                else nullcontext()
            )
            with stream_ctx:
                psd = stilde.psd

                valid_end = int(psize - self.data.trim_padding)
                valid_start = int(valid_end - self.data.blocksize * self.data.sample_rate)

                seg = slice(valid_start, valid_end)

                batch_peaks = None
                if getattr(self, "enable_cuda_graphs", False):
                    batch_peaks = self._try_cuda_graph_batch(
                        self.block_id, mid, tgroup, psize, seg, stilde
                    )

                if batch_peaks is None:
                    self.corr[self.block_id].execute(stilde)
                    self.ifts[mid].execute()

                cout_mem = getattr(self, "cout_mem", {}).get(mid, None) if hasattr(self, "cout_mem") else None
                if cout_mem is not None:
                    cout_mem._active_tgroup = tgroup
                    cout_mem._active_stilde = stilde
                for h in tgroup:
                    h._corr_valid = True
                    h._corr_stilde = stilde

                self.block_id += 1

        snr = numpy.zeros(len(tgroup), dtype=numpy.complex64)
        time = numpy.zeros(len(tgroup), dtype=numpy.float64)
        templates = numpy.zeros(len(tgroup), dtype=numpy.uint64)
        sigmasq = numpy.zeros(len(tgroup), dtype=numpy.float32)

        time[:] = self.data.start_time

        result = {}
        tkeys = tgroup[0].params.dtype.names
        for key in tkeys:
            result[key] = []

        veto_info = []

        ondevice_result = None
        if batch_peaks is None:
            out_tensor = backend_array(self.out_mem[mid], "torch")
            if (
                out_tensor is not None
                and _torch_ondevice_peaks_enabled(out_tensor.device.type)
                and len(tgroup) >= _TORCH_BATCH_PEAK_THRESHOLD_MIN_TEMPLATES
            ):
                power_matrix = getattr(self, 'power_matrices', {}).get(mid)
                sigmasqs = None
                norms = None

                if power_matrix is not None:
                    if not hasattr(self, '_psd_cache') or not isinstance(self._psd_cache, dict):
                        self._psd_cache = {}
                    mid_cache = self._psd_cache.get(mid)
                    if mid_cache is None:
                        mid_cache = LimitedSizeDict(size_limit=32)
                        self._psd_cache[mid] = mid_cache

                    psd_key = id(psd)
                    if psd_key in mid_cache:
                        sigmasqs, norms = mid_cache[psd_key]
                    else:
                        try:
                            delta_f = tgroup[0].delta_f
                            if hasattr(psd, 'numpy'):
                                psd_arr = psd.numpy()
                            else:
                                psd_arr = numpy.asarray(psd)
                            inv_psd = (4.0 * delta_f) / psd_arr.astype(numpy.float32)
                            if len(inv_psd) > power_matrix.shape[1]:
                                inv_psd = inv_psd[:power_matrix.shape[1]]
                            sigmasqs = power_matrix.dot(inv_psd)
                            norms = 4.0 * delta_f / numpy.sqrt(sigmasqs)
                            mid_cache[psd_key] = (sigmasqs, norms)
                        except Exception:
                            sigmasqs = None
                            norms = None

                if sigmasqs is None or norms is None:
                    sigmasqs = numpy.fromiter(
                        (htilde.sigmasq(psd) for htilde in tgroup),
                        dtype=numpy.float32,
                        count=len(tgroup),
                    )
                    norms = numpy.fromiter(
                        (4.0 * htilde.delta_f / (float(sgm) ** 0.5)
                         for htilde, sgm in zip(tgroup, sigmasqs)),
                        dtype=numpy.float64,
                        count=len(tgroup),
                    )

                ondevice_result = _torch_batch_peak_and_threshold_gpu(
                    self.out_mem[mid],
                    len(tgroup),
                    psize,
                    seg,
                    norms,
                    self.snr_threshold,
                    self.snr_abort_threshold,
                )

        if ondevice_result is not None:
            survivor_indices, peak_indices, peak_values, aborted = ondevice_result
            if aborted:
                self._async_prefetched = None
                logger.info("We are seeing some *really* high SNRs, let's "
                            "assume they aren't signals and just give up")
                return False, []

            i = 0
            for idx_pos, template_index in enumerate(survivor_indices):
                htilde = tgroup[template_index]
                if hasattr(htilde, 'time_offset'):
                    if 'time_offset' not in result:
                        result['time_offset'] = []

                l = int(peak_indices[idx_pos]) + valid_start
                peak = (
                    peak_values[idx_pos].item()
                    if hasattr(peak_values[idx_pos], 'item')
                    else peak_values[idx_pos]
                )
                snrv = numpy.array([peak])
                norm = norms[template_index]
                sgm = sigmasqs[template_index]

                time[i] += float(l - valid_start) / self.data.sample_rate
                veto_info.append((snrv, norm, l, htilde, stilde))

                snr[i] = snrv[0] * norm
                sigmasq[i] = sgm
                templates[i] = htilde.id
                if not hasattr(htilde, 'dict_params'):
                    htilde.dict_params = {}
                    for key in tkeys:
                        htilde.dict_params[key] = htilde.params[key]

                for key in tkeys:
                    result[key].append(htilde.dict_params[key])

                if hasattr(htilde, 'time_offset'):
                    result['time_offset'].append(htilde.time_offset)

                i += 1
        else:
            if batch_peaks is None:
                batch_peaks = _torch_batch_peak_values(
                    self.out_mem[mid], len(tgroup), psize, seg
                )
                if batch_peaks is None:
                    batch_peaks = _cpu_batch_peak_values(
                        self.out_mem[mid], len(tgroup), psize, seg
                    )

            # LiveBatch retains only one peak per template.  Materialize their
            # magnitudes in bulk for groups above the crossover threshold.
            batch_magnitudes = None
            if (
                batch_peaks is not None
                and self.snr_abort_threshold is None
                and len(tgroup) >= _TORCH_BATCH_PEAK_THRESHOLD_MIN_TEMPLATES
            ):
                batch_magnitudes = _torch_batch_peak_magnitudes(batch_peaks[1])

            i = 0
            if batch_magnitudes is not None:
                power_matrix = getattr(self, 'power_matrices', {}).get(mid)
                sigmasqs = None
                norms = None

                if power_matrix is not None:
                    if not hasattr(self, '_psd_cache') or not isinstance(self._psd_cache, dict):
                        self._psd_cache = {}
                    mid_cache = self._psd_cache.get(mid)
                    if mid_cache is None:
                        mid_cache = LimitedSizeDict(size_limit=32)
                        self._psd_cache[mid] = mid_cache

                    psd_key = id(psd)
                    if psd_key in mid_cache:
                        sigmasqs, norms = mid_cache[psd_key]
                    else:
                        try:
                            delta_f = tgroup[0].delta_f
                            if hasattr(psd, 'numpy'):
                                psd_arr = psd.numpy()
                            else:
                                psd_arr = numpy.asarray(psd)
                            inv_psd = (4.0 * delta_f) / psd_arr.astype(numpy.float32)
                            if len(inv_psd) > power_matrix.shape[1]:
                                inv_psd = inv_psd[:power_matrix.shape[1]]
                            sigmasqs = power_matrix.dot(inv_psd)
                            norms = 4.0 * delta_f / numpy.sqrt(sigmasqs)
                            mid_cache[psd_key] = (sigmasqs, norms)
                        except Exception:
                            sigmasqs = None
                            norms = None

                if sigmasqs is None or norms is None:
                    sigmasqs = numpy.fromiter(
                        (htilde.sigmasq(psd) for htilde in tgroup),
                        dtype=numpy.float32,
                        count=len(tgroup),
                    )
                    norms = numpy.fromiter(
                        (4.0 * htilde.delta_f / (float(sgm) ** 0.5)
                         for htilde, sgm in zip(tgroup, sigmasqs)),
                        dtype=numpy.float64,
                        count=len(tgroup),
                    )

                snrs = batch_magnitudes * norms
                survivor_indices = numpy.flatnonzero(~(snrs < self.snr_threshold))

                peak_indices, peak_values = batch_peaks
                for template_index in survivor_indices:
                    htilde = tgroup[template_index]
                    if hasattr(htilde, 'time_offset'):
                        if 'time_offset' not in result:
                            result['time_offset'] = []

                    l = int(peak_indices[template_index]) + valid_start
                    peak = peak_values[template_index].item()
                    snrv = numpy.array([peak])
                    norm = norms[template_index]
                    sgm = sigmasqs[template_index]

                    time[i] += float(l - valid_start) / self.data.sample_rate
                    veto_info.append((snrv, norm, l, htilde, stilde))

                    snr[i] = snrv[0] * norm
                    sigmasq[i] = sgm
                    templates[i] = htilde.id
                    if not hasattr(htilde, 'dict_params'):
                        htilde.dict_params = {}
                        for key in tkeys:
                            htilde.dict_params[key] = htilde.params[key]

                    for key in tkeys:
                        result[key].append(htilde.dict_params[key])

                    if hasattr(htilde, 'time_offset'):
                        result['time_offset'].append(htilde.time_offset)

                    i += 1
            else:
                for template_index, htilde in enumerate(tgroup):
                    if hasattr(htilde, 'time_offset'):
                        if 'time_offset' not in result:
                            result['time_offset'] = []

                    if batch_peaks is None:
                        l = htilde.out[seg].abs_arg_max()
                        peak = None
                    else:
                        peak_indices, peak_values = batch_peaks
                        l = int(peak_indices[template_index])
                        peak = peak_values[template_index].item()

                    sgm = htilde.sigmasq(psd)
                    norm = 4.0 * htilde.delta_f / (sgm ** 0.5)

                    l += valid_start
                    if peak is None:
                        peak = htilde.out[l]
                    snrv = numpy.array([peak])

                    s = abs(snrv[0]) * norm
                    if s < self.snr_threshold:
                        continue

                    time[i] += float(l - valid_start) / self.data.sample_rate

                    # We have an SNR so high that we will drop the entire analysis
                    # of this chunk of time!
                    if self.snr_abort_threshold is not None and s > self.snr_abort_threshold:
                        self._async_prefetched = None
                        logger.info("We are seeing some *really* high SNRs, let's "
                                    "assume they aren't signals and just give up")
                        return False, []

                    veto_info.append((snrv, norm, l, htilde, stilde))

                    snr[i] = snrv[0] * norm
                    sigmasq[i] = sgm
                    templates[i] = htilde.id
                    if not hasattr(htilde, 'dict_params'):
                        htilde.dict_params = {}
                        for key in tkeys:
                            htilde.dict_params[key] = htilde.params[key]

                    for key in tkeys:
                        result[key].append(htilde.dict_params[key])

                    if hasattr(htilde, 'time_offset'):
                        result['time_offset'].append(htilde.time_offset)

                    i += 1

        result['snr'] = abs(snr[0:i])
        result['coa_phase'] = numpy.angle(snr[0:i])
        result['end_time'] = time[0:i]
        result['template_id'] = templates[0:i]
        result['sigmasq'] = sigmasq[0:i]

        for key in tkeys:
            result[key] = numpy.array(result[key])

        if 'time_offset' in result:
            result['time_offset'] = numpy.array(result['time_offset'])

        return result, veto_info


def _count_louder_background(background, window, threshold):
    """Count background windows whose peak reaches ``threshold``.

    The Torch path performs the block reduction on its current device and
    transfers only the final count to the host.
    """
    nsamples = len(background) // window
    if nsamples == 0:
        return 0, 0

    tensor = backend_array(background, "torch")
    if tensor is not None:
        peaks = tensor[:nsamples * window].reshape(
            nsamples, window
        ).amax(dim=-1)
        count = (peaks >= threshold).sum().item()
    else:
        values = background.numpy()
        peaks = values[:nsamples * window].reshape(
            nsamples, window
        ).max(axis=1)
        count = (peaks >= threshold).sum()

    return int(count), nsamples


def followup_event_significance(ifo, data_reader, bank,
                                template_id, coinc_times,
                                coinc_threshold=0.005,
                                lookback=150, duration=0.095):
    """Given a detector, a template waveform and a set of candidate event
    times in different detectors, perform an on-source/off-source analysis
    to determine if the SNR in the first detector has a significant peak
    in the on-source window. The significance is given in terms of a
    p-value. See Dal Canton et al. 2021 (https://arxiv.org/abs/2008.07494)
    for details. A portion of the SNR time series around the on-source window
    is also returned for use in BAYESTAR.

    If the calculation cannot be carried out, for example because `ifo` is
    not in observing mode at the requested time, then None is returned.
    Otherwise, the dict contains the following keys. `snr_series` is a
    TimeSeries object with the SNR time series for BAYESTAR. `peak_time` is the
    time of maximum SNR in the on-source window. `pvalue` is the p-value for
    the maximum on-source SNR compared to the off-source realizations.
    `pvalue_saturated` is a bool indicating whether the p-value is limited by
    the number of off-source realizations, i.e. whether the maximum on-source
    SNR is larger than all the off-source ones. `sigma2` is the SNR
    normalization (squared) for the given template and detector.

    Parameters
    ----------
    ifo: str
        Which detector is being used for the calculation.
    data_reader: StrainBuffer
        StrainBuffer object providing the data for the given detector.
    bank: LiveFilterBank
        Template bank object providing the template related quantities.
    template_id: int
        Index of the template in the bank.
    coinc_times: dict
        Dictionary keyed by detector names reporting the coalescence times of
        a candidate measured at the different detectors. Used to define the
        on-source window of the candidate in `ifo`.
    coinc_threshold: float
        Nominal statistical uncertainty in `coinc_times`; expands the
        on-source window by twice the given amount.
    lookback: float
        Nominal amount of time to use for the calculation of the onsource and
        offsource SNR time series. The actual time may be reduced depending on
        the duration of the template and the strain buffer in the data reader
        (if so, a warning is logged).
    duration: float
        Duration of the SNR time series to be reported to BAYESTAR.

    Returns
    -------
    followup_info: dict or None
        Results of the followup calculation (see above) or None if `ifo` did
        not have usable data.
    """
    from pycbc.waveform import get_waveform_filter_length_in_time
    tmplt = bank.table[template_id]
    length_in_time = get_waveform_filter_length_in_time(tmplt['approximant'],
                                                        tmplt)

    # calculate onsource time range
    from pycbc.detector import Detector
    onsource_start = -numpy.inf
    onsource_end = numpy.inf
    fdet = Detector(ifo)

    for cifo in coinc_times:
        time = coinc_times[cifo]
        dtravel = Detector(cifo).light_travel_time_to_detector(fdet)
        if time - dtravel > onsource_start:
            onsource_start = time - dtravel
        if time + dtravel < onsource_end:
            onsource_end = time + dtravel

    # Source must be within this time window to be considered a possible
    # coincidence
    onsource_start -= coinc_threshold
    onsource_end += coinc_threshold

    # Calculate how much time is needed to calculate the significance.
    # At the minimum, we need enough time to include the lookback, plus time
    # that we will throw away because of corruption from finite-duration filter
    # responses (this is equal to the nominal padding plus the template
    # duration). Next, for efficiency, we round the resulting duration up to
    # align it with one of the frequency resolutions preferred by the template
    # bank. And finally, the resulting duration must fit into the strain buffer
    # available in the data reader, so we check that.
    trim_pad = data_reader.trim_padding * data_reader.strain.delta_t
    buffer_duration = lookback + 2 * trim_pad + length_in_time
    buffer_samples = bank.round_up(int(buffer_duration * bank.sample_rate))
    max_safe_buffer_samples = int(
        0.9 * data_reader.strain.duration * bank.sample_rate
    )
    if buffer_samples > max_safe_buffer_samples:
        buffer_samples = max_safe_buffer_samples
        new_lookback = (
            buffer_samples / bank.sample_rate - (2 * trim_pad + length_in_time)
        )
        # Require a minimum lookback time of twice the onsource window or SNR
        # time series (whichever is longer) so we have enough data for the
        # onsource window, the SNR time series, and at least a few background
        # samples
        min_required_lookback = 2 * max(onsource_end - onsource_start, duration)
        if new_lookback > min_required_lookback:
            logging.warning(
                'Strain buffer too short for a lookback time of %f s, '
                'reducing lookback to %f s',
                lookback,
                new_lookback
            )
        else:
            logging.error(
                'Strain buffer too short to compute the followup SNR time '
                'series for template %d, will not use %s for followup. '
                'Either use shorter templates, or raise --max-length.',
                template_id,
                ifo
            )
            return None
    buffer_duration = buffer_samples / bank.sample_rate

    # Require all strain be valid within lookback time
    if data_reader.state is not None:
        state_start_time = (
            data_reader.strain.end_time
            - data_reader.reduced_pad * data_reader.strain.delta_t
            - buffer_duration
        )
        if not data_reader.state.is_extent_valid(
            state_start_time, buffer_duration
        ):
            logging.info(
                '%s strain buffer contains invalid data during lookback, '
                'will not use for followup',
                ifo
            )
            return None

    # We won't require that all DQ checks be valid for now, except at
    # onsource time.
    if data_reader.dq is not None:
        dq_start_time = onsource_start - duration / 2.0
        dq_duration = onsource_end - onsource_start + duration
        if not data_reader.dq.is_extent_valid(dq_start_time, dq_duration):
            logging.info(
                '%s DQ buffer indicates invalid data during onsource window, '
                'will not use for followup',
                ifo
            )
            return None

    # Calculate SNR time series for the entire lookback duration
    htilde = bank.get_template(
        template_id, delta_f=bank.sample_rate / float(buffer_samples)
    )
    stilde = data_reader.overwhitened_data(htilde.delta_f)

    sigma2 = htilde.sigmasq(stilde.psd)
    snr, _, norm = matched_filter_core(htilde, stilde, h_norm=sigma2)

    # Find peak SNR in on-source and determine p-value
    onsrc = snr.time_slice(onsource_start, onsource_end)
    peak = onsrc.abs_arg_max()
    peak_time = peak * snr.delta_t + onsrc.start_time
    peak_value = abs(onsrc[peak])

    bstart = float(snr.start_time) + length_in_time + trim_pad
    bkg = abs(snr.time_slice(bstart, onsource_start))

    window = int((onsource_end - onsource_start) * snr.sample_rate)
    num_louder_bg, nsamples = _count_louder_background(
        bkg, window, peak_value
    )
    pvalue = (1 + num_louder_bg) / float(1 + nsamples)
    pvalue_saturated = num_louder_bg == 0

    # Return recentered source SNR for bayestar, along with p-value, and trig
    peak_full = int((peak_time - snr.start_time) / snr.delta_t)
    half_dur_samples = int(snr.sample_rate * duration / 2)
    snr_slice = slice(peak_full - half_dur_samples,
                      peak_full + half_dur_samples + 1)
    baysnr = snr[snr_slice]

    logger.info('Adding %s to candidate, pvalue %s, %s samples', ifo,
                pvalue, nsamples)

    return {
        'snr_series': baysnr * norm,
        'peak_time': peak_time,
        'pvalue': pvalue,
        'pvalue_saturated': pvalue_saturated,
        'sigma2': sigma2
    }

def compute_followup_snr_series(data_reader, htilde, trig_time,
                                duration=0.095, check_state=True,
                                coinc_window=0.05):
    """Given a StrainBuffer, a template frequency series and a trigger time,
    compute a portion of the SNR time series centered on the trigger for its
    rapid sky localization and followup.

    If the trigger time is too close to the boundary of the valid data segment
    the SNR series is calculated anyway and might be slightly contaminated by
    filter and wrap-around effects. For reasonable durations this will only
    affect a small fraction of the triggers and probably in a negligible way.

    Parameters
    ----------
    data_reader : StrainBuffer
        The StrainBuffer object to read strain data from.

    htilde : FrequencySeries
        The frequency series containing the template waveform.

    trig_time : {float, lal.LIGOTimeGPS}
        The trigger time.

    duration : float (optional)
        Duration of the computed SNR series in seconds. If omitted, it defaults
        to twice the Earth light travel time plus 10 ms of timing uncertainty.

    check_state : boolean
        If True, and the detector was offline or flagged for bad data quality
        at any point during the inspiral, then return (None, None) instead.

    coinc_window : float (optional)
        Maximum possible time between coincident triggers at different
        detectors. This is needed to properly determine data padding.

    Returns
    -------
    snr : TimeSeries
        The portion of SNR around the trigger. None if the detector is offline
        or has bad data quality, and check_state is True.
    """
    if check_state:
        # was the detector observing for the full amount of involved data?
        state_start_time = trig_time - duration / 2 - htilde.length_in_time
        state_end_time = trig_time + duration / 2
        state_duration = state_end_time - state_start_time
        if data_reader.state is not None:
            if not data_reader.state.is_extent_valid(state_start_time,
                                                     state_duration):
                return None

        # was the data quality ok for the full amount of involved data?
        dq_start_time = state_start_time - data_reader.dq_padding
        dq_duration = state_duration + 2 * data_reader.dq_padding
        if data_reader.dq is not None:
            if not data_reader.dq.is_extent_valid(dq_start_time, dq_duration):
                return None

    stilde = data_reader.overwhitened_data(htilde.delta_f)
    snr, _, norm = matched_filter_core(htilde, stilde,
                                          h_norm=htilde.sigmasq(stilde.psd))

    valid_end = int(len(snr) - data_reader.trim_padding)
    valid_start = int(valid_end - data_reader.blocksize * snr.sample_rate)

    half_dur_samples = int(snr.sample_rate * duration / 2)
    coinc_samples = int(snr.sample_rate * coinc_window)
    valid_start -= half_dur_samples + coinc_samples
    valid_end += half_dur_samples
    if valid_start < 0 or valid_end > len(snr)-1:
        raise ValueError(('Requested SNR duration ({0} s)'
                          ' too long').format(duration))

    # Onsource slice for Bayestar followup
    onsource_idx = float(trig_time - snr.start_time) * snr.sample_rate
    onsource_idx = int(round(onsource_idx))
    onsource_slice = slice(onsource_idx - half_dur_samples,
                           onsource_idx + half_dur_samples + 1)
    return snr[onsource_slice] * norm


class _BracketError(RuntimeError):
    """Internal invalid-bracket signal used by ``_brent_minimum``."""


def _bracket_minimum(
    function, xa, xb, grow_limit=110.0, max_iterations=1000
):
    """Bracket a scalar minimum using SciPy-compatible expansion."""
    golden_ratio = 1.618034
    very_small = 1e-21

    xa, xb = numpy.asarray([xa, xb])
    fa = function(xa)
    fb = function(xb)
    if fa < fb:
        xa, xb = xb, xa
        fa, fb = fb, fa

    xc = xb + golden_ratio * (xb - xa)
    fc = function(xc)
    function_calls = 3
    iteration = 0

    while fc < fb:
        tmp1 = (xb - xa) * (fb - fc)
        tmp2 = (xb - xc) * (fb - fa)
        value = tmp2 - tmp1
        if numpy.abs(value) < very_small:
            denominator = 2.0 * very_small
        else:
            denominator = 2.0 * value
        w = xb - (
            (xb - xc) * tmp2 - (xb - xa) * tmp1
        ) / denominator
        w_limit = xb + grow_limit * (xc - xb)

        if iteration > max_iterations:
            raise RuntimeError(
                "No valid bracket was found before the iteration limit "
                "was reached. Consider trying different initial points or "
                "increasing `maxiter`."
            )
        iteration += 1

        if (w - xc) * (xb - w) > 0.0:
            fw = function(w)
            function_calls += 1
            if fw < fc:
                xa = xb
                xb = w
                fa = fb
                fb = fw
                break
            if fw > fb:
                xc = w
                fc = fw
                break
            w = xc + golden_ratio * (xc - xb)
            fw = function(w)
            function_calls += 1
        elif (w - w_limit) * (w_limit - xc) >= 0.0:
            w = w_limit
            fw = function(w)
            function_calls += 1
        elif (w - w_limit) * (xc - w) > 0.0:
            fw = function(w)
            function_calls += 1
            if fw < fc:
                xb = xc
                xc = w
                w = xc + golden_ratio * (xc - xb)
                fb = fc
                fc = fw
                fw = function(w)
                function_calls += 1
        else:
            w = xc + golden_ratio * (xc - xb)
            fw = function(w)
            function_calls += 1

        xa = xb
        xb = xc
        xc = w
        fa = fb
        fb = fc
        fc = fw

    bracket_has_minimum = (
        (fb < fc and fb <= fa) or (fb < fa and fb <= fc)
    )
    bracket_is_ordered = xa < xb < xc or xc < xb < xa
    bracket_is_finite = (
        numpy.isfinite(xa)
        and numpy.isfinite(xb)
        and numpy.isfinite(xc)
    )
    if not (
        bracket_has_minimum and bracket_is_ordered and bracket_is_finite
    ):
        error = _BracketError(
            "The algorithm terminated without finding a valid bracket. "
            "Consider trying different initial points."
        )
        error.data = (xa, xb, xc, fa, fb, fc, function_calls)
        raise error

    return xa, xb, xc, fa, fb, fc


def _brent_minimum(
    function, bracket, tolerance=1.48e-8, max_iterations=500
):
    """Minimize a scalar function with SciPy-compatible Brent updates."""
    try:
        xa, xb, xc, _, fb, _ = _bracket_minimum(
            function, bracket[0], bracket[1]
        )
    except _BracketError as error:
        xa, xb, xc, fa, fb, fc, _ = error.data
        points = [xa, xb, xc]
        values = [fa, fb, fc]
        if numpy.any(numpy.isnan([points, values])):
            return numpy.nan
        return points[numpy.argmin(values)]

    x = w = v = xb
    fw = fv = fx = fb
    if xa < xc:
        lower = xa
        upper = xc
    else:
        lower = xc
        upper = xa

    delta_x = 0.0
    step = 0.0
    minimum_tolerance = 1.0e-11
    golden_section = 0.3819660

    for _ in range(max_iterations):
        tolerance_1 = tolerance * numpy.abs(x) + minimum_tolerance
        tolerance_2 = 2.0 * tolerance_1
        midpoint = 0.5 * (lower + upper)
        if numpy.abs(x - midpoint) < (
            tolerance_2 - 0.5 * (upper - lower)
        ):
            break

        if numpy.abs(delta_x) <= tolerance_1:
            if x >= midpoint:
                delta_x = lower - x
            else:
                delta_x = upper - x
            step = golden_section * delta_x
        else:
            tmp1 = (x - w) * (fx - fv)
            tmp2 = (x - v) * (fx - fw)
            parabola = (x - v) * tmp2 - (x - w) * tmp1
            tmp2 = 2.0 * (tmp2 - tmp1)
            if tmp2 > 0.0:
                parabola = -parabola
            tmp2 = numpy.abs(tmp2)
            previous_delta_x = delta_x
            delta_x = step
            if (
                parabola > tmp2 * (lower - x)
                and parabola < tmp2 * (upper - x)
                and numpy.abs(parabola)
                < numpy.abs(0.5 * tmp2 * previous_delta_x)
            ):
                step = parabola / tmp2
                candidate = x + step
                if (
                    candidate - lower < tolerance_2
                    or upper - candidate < tolerance_2
                ):
                    if midpoint - x >= 0:
                        step = tolerance_1
                    else:
                        step = -tolerance_1
            else:
                if x >= midpoint:
                    delta_x = lower - x
                else:
                    delta_x = upper - x
                step = golden_section * delta_x

        if numpy.abs(step) < tolerance_1:
            if step >= 0:
                candidate = x + tolerance_1
            else:
                candidate = x - tolerance_1
        else:
            candidate = x + step
        candidate_value = function(candidate)

        if candidate_value > fx:
            if candidate < x:
                lower = candidate
            else:
                upper = candidate
            if candidate_value <= fw or w == x:
                v = w
                w = candidate
                fv = fw
                fw = candidate_value
            elif candidate_value <= fv or v == x or v == w:
                v = candidate
                fv = candidate_value
        else:
            if candidate >= x:
                lower = x
            else:
                upper = x
            v = w
            w = x
            x = candidate
            fv = fw
            fw = fx
            fx = candidate_value

    return x


def optimized_match(
    vec1,
    vec2,
    psd=None,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
    v1_norm=None,
    v2_norm=None,
    return_phase=False,
):
    """Given two waveforms, compute their optimized match using a
    Brent scalar search.

    This function computes the same quantities as "match";
    it is more accurate and slower.

    Parameters
    ----------
    vec1 : TimeSeries or FrequencySeries
        The input vector containing a waveform.
    vec2 : TimeSeries or FrequencySeries
        The input vector containing a waveform.
    psd : FrequencySeries
        A power spectral density to weight the overlap.
    low_frequency_cutoff : {None, float}, optional
        The frequency to begin the match.
    high_frequency_cutoff : {None, float}, optional
        The frequency to stop the match.
    v1_norm : {None, float}, optional
        The normalization of the first waveform. This is equivalent to its
        sigmasq value. If None, it is internally calculated.
    v2_norm : {None, float}, optional
        The normalization of the second waveform. This is equivalent to its
        sigmasq value. If None, it is internally calculated.
    return_phase : {False, bool}, optional
        If True, also return the phase shift that gives the match.

    Returns
    -------
    match: float
    index: int
        The number of samples to shift to get the match.
    phi: float
        Phase to rotate complex waveform to get the match, if desired.
    """

    htilde = make_frequency_series(vec1)
    stilde = make_frequency_series(vec2)

    assert numpy.isclose(htilde.delta_f, stilde.delta_f)
    delta_f = stilde.delta_f

    assert numpy.isclose(htilde.delta_t, stilde.delta_t)
    delta_t = stilde.delta_t

    # a first time shift to get in the nearby region;
    # then the optimization is only used to move to the
    # correct subsample-timeshift within (-delta_t, delta_t)
    # of this
    _, max_id = match(
        htilde,
        stilde,
        psd=psd,
        low_frequency_cutoff=low_frequency_cutoff,
        high_frequency_cutoff=high_frequency_cutoff,
    )

    stilde = stilde.cyclic_time_shift(-max_id * delta_t)

    N = (len(stilde) - 1) * 2
    kmin, kmax = get_cutoff_indices(
        low_frequency_cutoff, high_frequency_cutoff, delta_f, N
    )
    mask = slice(kmin, kmax)

    htilde_tensor = backend_array(htilde, "torch")
    stilde_tensor = backend_array(stilde, "torch")
    if htilde_tensor is not None and stilde_tensor is not None:
        import torch

        # The scalar controller needs one magnitude for each decision, while
        # every waveform-sized objective and reduction remains on-device.
        waveform_1 = htilde_tensor[mask]
        waveform_2 = stilde_tensor[mask]
        frequencies = backend_array(stilde.sample_frequencies, "torch")[mask]
        if psd is None:
            psd_arr = torch.ones_like(waveform_1)
        else:
            psd_tensor = backend_array(psd, "torch")
            if psd_tensor is None:
                psd_tensor = torch.as_tensor(
                    psd.numpy(), device=waveform_1.device
                )
            else:
                psd_tensor = psd_tensor.to(device=waveform_1.device)
            psd_arr = psd_tensor[mask]

        weighted_product = (
            torch.conj(waveform_1) * waveform_2 / psd_arr
        )

        def product_offset(dt, return_phase=False):
            offset = torch.exp(
                2j * torch.pi * frequencies * dt
            )
            integral = torch.sum(weighted_product * offset) * delta_f
            magnitude = 4 * torch.abs(integral)
            if return_phase:
                return torch.stack(
                    (magnitude, torch.angle(integral))
                ).tolist()
            return magnitude.item()
    else:
        frequencies = stilde.sample_frequencies.numpy()[mask]
        waveform_1 = htilde.numpy()[mask]
        waveform_2 = stilde.numpy()[mask]

        if psd is not None:
            psd_arr = psd.numpy()[mask]
        else:
            psd_arr = numpy.ones_like(waveform_1)

        def product(a, b):
            integral = numpy.sum(numpy.conj(a) * b / psd_arr) * delta_f
            return 4 * abs(integral), numpy.angle(integral)

        def product_offset(dt, return_phase=False):
            offset = numpy.exp(2j * numpy.pi * frequencies * dt)
            magnitude, phase = product(waveform_1, waveform_2 * offset)
            if return_phase:
                return magnitude, phase
            return magnitude

    def to_minimize(dt):
        return -product_offset(dt)

    norm_1 = (
        sigmasq(htilde, psd, low_frequency_cutoff, high_frequency_cutoff)
        if v1_norm is None
        else v1_norm
    )
    norm_2 = (
        sigmasq(stilde, psd, low_frequency_cutoff, high_frequency_cutoff)
        if v2_norm is None
        else v2_norm
    )

    norm = numpy.sqrt(norm_1 * norm_2)

    time_shift = _brent_minimum(
        to_minimize, bracket=(-delta_t, delta_t)
    )
    m, angle = product_offset(time_shift, return_phase=True)

    if return_phase:
        return m / norm, time_shift / delta_t + max_id, -angle
    else:
        return m / norm, time_shift / delta_t + max_id


__all__ = ['match', 'optimized_match', 'matched_filter', 'sigmasq', 'sigma', 'get_cutoff_indices',
           'sigmasq_series', 'make_frequency_series', 'overlap',
           'overlap_cplx', 'matched_filter_core', 'correlate',
           'MatchedFilterControl', 'LiveBatchMatchedFilter',
           'MatchedFilterSkyMaxControl', 'MatchedFilterSkyMaxControlNoPhase',
           'compute_max_snr_over_sky_loc_stat_no_phase',
           'compute_max_snr_over_sky_loc_stat',
           'compute_followup_snr_series',
           'compute_u_val_for_sky_loc_stat_no_phase',
           'compute_u_val_for_sky_loc_stat',
           'followup_event_significance']
