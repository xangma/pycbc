# Copyright (C) 2020  Daniel Finstad
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
"""This module provides model classes and functions for implementing
a relative binning likelihood for parameter estimation.
"""

import itertools
import logging

import numpy
from scipy.interpolate import interp1d

from pycbc.detector import Detector
from pycbc.types import Array, TimeSeries
from pycbc.types.backend import wrap_backend_array
from pycbc.waveform import (
    FailedWaveformError,
    fd_det_sequence,
    get_fd_det_waveform_sequence,
    get_fd_waveform_sequence,
)

from .gaussian_noise import BaseGaussianNoise, catch_waveform_error
from .relbin_cpu import (
    likelihood_parts,
    likelihood_parts_det,
    likelihood_parts_det_multi,
    likelihood_parts_multi,
    likelihood_parts_multi_v,
    likelihood_parts_v,
    likelihood_parts_v_pol,
    likelihood_parts_v_pol_time,
    likelihood_parts_v_time,
    likelihood_parts_vector,
    likelihood_parts_vectorp,
    likelihood_parts_vectort,
    snr_predictor,
    snr_predictor_dom,
)
from .tools import DistMarg, _threshold_extent, _torch_tensor

_TORCH_POLARIZATION_LIKELIHOODS = (
    likelihood_parts,
    likelihood_parts_v,
    likelihood_parts_vector,
    likelihood_parts_vectort,
    likelihood_parts_vectorp,
    likelihood_parts_v_pol,
    likelihood_parts_v_time,
    likelihood_parts_v_pol_time,
)


def _numpy_value(value):
    """Return a CPU representation for the legacy relative-bin kernels."""
    tensor = _torch_tensor(value)
    if tensor is not None:
        tensor = tensor.detach()
        if tensor.is_conj():
            tensor = tensor.resolve_conj()
        return tensor.cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def _time_series_from_values(values, delta_t, epoch):
    """Wrap values as a TimeSeries without copying Torch storage."""
    tensor = _torch_tensor(values)
    if tensor is None:
        return TimeSeries(values, delta_t=delta_t, epoch=epoch)

    return TimeSeries(
        wrap_backend_array(tensor), delta_t=delta_t, epoch=epoch, copy=False
    )


def _prepare_reference_data(waveform, data, size, offset, delta_f, time_shift):
    """Place a fiducial waveform and shifted data on their active backend."""
    if any(_torch_tensor(value) is not None for value in (waveform, data)):
        from .relbin_torch import prepare_reference_data

        return prepare_reference_data(waveform, data, size, offset, delta_f, time_shift)

    waveform.resize(size)
    waveform = numpy.roll(waveform, offset)
    frequencies = numpy.arange(size, dtype=numpy.float64) * delta_f
    shift = numpy.exp(-2.0j * numpy.pi * frequencies * time_shift)
    return numpy.array(waveform), data * numpy.conjugate(shift)


def _uniform_frequency_grid(series):
    """Rebuild a frequency grid without copying its device representation."""
    tensor = _torch_tensor(series)
    dtype = (
        numpy.float32
        if tensor is not None and tensor.device.type == "mps"
        else numpy.float64
    )
    return numpy.arange(len(series), dtype=dtype) * series.delta_f


def setup_bins(
    f_full,
    f_lo,
    f_hi,
    chi=1.0,
    eps=0.1,
    gammas=None,
):
    """Construct frequency bins for use in a relative likelihood
    model. For details, see [Barak, Dai & Venumadhav 2018].

    Parameters
    ----------
    f_full : array
        The full resolution array of frequencies being used in the analysis.
    f_lo : float
        The starting frequency used in matched filtering. This will be the
        left edge of the first frequency bin.
    f_hi : float
        The ending frequency used in matched filtering. This will be the right
        edge of the last frequency bin.
    chi : float, optional
        Tunable parameter, see [Barak, Dai & Venumadhav 2018]
    eps : float, optional
        Tunable parameter, see [Barak, Dai & Venumadhav 2018]. Lower values
        result in larger number of bins.
    gammas : array, optional
        Frequency powerlaw indices to be used in computing bins.

    Returns
    -------
    nbin : int
        Number of bins.
    fbin : numpy.array of floats
        Bin edge frequencies.
    fbin_ind : numpy.array of ints
        Indices of bin edges in full frequency array.
    """
    f = numpy.linspace(f_lo, f_hi, 10000)
    # f^ga power law index
    ga = (
        gammas
        if gammas is not None
        else numpy.array([-5.0 / 3, -2.0 / 3, 1.0, 5.0 / 3, 7.0 / 3])
    )
    logging.info("Using powerlaw indices: %s", ga)
    dalp = chi * 2.0 * numpy.pi / numpy.absolute((f_lo**ga) - (f_hi**ga))
    dphi = numpy.sum(
        numpy.array(
            [numpy.sign(g) * d * (f**g) for g, d in zip(ga, dalp, strict=True)]
        ),
        axis=0,
    )
    dphi_diff = dphi - dphi[0]
    # now construct frequency bins
    nbin = int(dphi_diff[-1] / eps)
    dphi2f = interp1d(dphi_diff, f, kind="slinear", bounds_error=False, fill_value=0.0)
    dphi_grid = numpy.linspace(dphi_diff[0], dphi_diff[-1], nbin + 1)
    # frequency grid points
    fbin = dphi2f(dphi_grid)
    # indices of frequency grid points in the FFT array
    fbin_ind = numpy.searchsorted(f_full, fbin)
    for idx_fbin, idx_f_full in enumerate(fbin_ind):
        if idx_f_full == 0:
            curr_idx = 0
        elif idx_f_full == len(f_full):
            curr_idx = len(f_full) - 1
        else:
            abs1 = abs(f_full[idx_f_full] - fbin[idx_fbin])
            abs2 = abs(f_full[idx_f_full - 1] - fbin[idx_fbin])
            if abs1 > abs2:
                curr_idx = idx_f_full - 1
            else:
                curr_idx = idx_f_full
        fbin_ind[idx_fbin] = curr_idx
    fbin_ind = numpy.unique(fbin_ind)
    return fbin_ind


class Relative(DistMarg, BaseGaussianNoise):
    r"""Model that assumes the likelihood in a region around the peak
    is slowly varying such that a linear approximation can be made, and
    likelihoods can be calculated at a coarser frequency resolution. For
    more details on the implementation, see https://arxiv.org/abs/1806.08792.

    This model requires the use of a fiducial waveform whose parameters are
    near the peak of the likelihood. The fiducial waveform and all template
    waveforms used in likelihood calculation are currently generated using
    the SPAtmplt approximant.

    For more details on initialization parameters and definition of terms, see
    :py:class:`BaseGaussianNoise`.

    Parameters
    ----------
    variable_params : (tuple of) string(s)
        A tuple of parameter names that will be varied.
    data : dict
        A dictionary of data, in which the keys are the detector names and the
        values are the data (assumed to be unwhitened). All data must have the
        same frequency resolution.
    low_frequency_cutoff : dict
        A dictionary of starting frequencies, in which the keys are the
        detector names and the values are the starting frequencies for the
        respective detectors to be used for computing inner products.
    figucial_params : dict
        A dictionary of waveform parameters to be used for generating the
        fiducial waveform. Keys must be parameter names in the form
        'PARAM_ref' where PARAM is a recognized extrinsic parameter or
        an intrinsic parameter compatible with the chosen approximant.
    gammas : array of floats, optional
        Frequency powerlaw indices to be used in computing frequency bins.
    epsilon : float, optional
        Tuning parameter used in calculating the frequency bins. Lower values
        will result in higher resolution and more bins.
    earth_rotation: boolean, optional
        Default is False. If True, then vary the fp/fc polarization values
        as a function of frequency bin, using a predetermined PN approximation
        for the time offsets.
    \**kwargs :
        All other keyword arguments are passed to
        :py:class:`BaseGaussianNoise`.
    """

    name = "relative"

    def __init__(
        self,
        variable_params,
        data,
        low_frequency_cutoff,
        fiducial_params=None,
        gammas=None,
        epsilon=0.5,
        earth_rotation=False,
        earth_rotation_mode=2,
        marginalize_phase=True,
        **kwargs,
    ):
        variable_params, kwargs = self.setup_marginalization(
            variable_params, marginalize_phase=marginalize_phase, **kwargs
        )

        super(Relative, self).__init__(
            variable_params, data, low_frequency_cutoff, **kwargs
        )

        # If the waveform needs us to apply the detector response,
        # set flag to true (most cases for ground-based observatories).
        self.still_needs_det_response = False
        if self.static_params["approximant"] in fd_det_sequence:
            self.still_needs_det_response = True

        # reference waveform and bin edges
        self.f, self.df, self.end_time, self.det = {}, {}, {}, {}
        self.h00, self.h00_sparse = {}, {}
        self.fedges, self.edges = {}, {}
        self.ta, self.antenna_time = {}, {}

        # filtered summary data for linear approximation
        self.sdat = {}
        self._torch_likelihood_cache = {}
        self._torch_multi_likelihood_cache = {}

        # store fiducial waveform params
        self.fid_params = self.static_params.copy()
        self.fid_params.update(fiducial_params)

        # the flag used in `_loglr`
        self.return_sh_hh = False

        for k in self.static_params:
            if self.fid_params[k] == "REPLACE":
                self.fid_params.pop(k)

        for ifo in data:
            # store data and frequencies
            d0 = self.data[ifo]
            self.df[ifo] = d0.delta_f
            self.f[ifo] = _uniform_frequency_grid(d0)
            self.end_time[ifo] = float(d0.end_time)

            # generate fiducial waveform
            f_lo = self.kmin[ifo] * self.df[ifo]
            f_hi = self.kmax[ifo] * self.df[ifo]
            logging.info(
                "%s: Generating fiducial waveform from %s to %s Hz",
                ifo,
                f_lo,
                f_hi,
            )

            # prune low frequency samples to avoid waveform errors
            fpoints = Array(self.f[ifo].astype(numpy.float64))
            fpoints = fpoints[self.kmin[ifo] : self.kmax[ifo] + 1]

            if self.still_needs_det_response:
                wave = get_fd_det_waveform_sequence(
                    ifos=ifo, sample_points=fpoints, **self.fid_params
                )
                curr_wav = wave[ifo]
                self.ta[ifo] = 0.0
            else:
                fid_hp, fid_hc = get_fd_waveform_sequence(
                    sample_points=fpoints, **self.fid_params
                )
                # Apply detector response if not handled by
                # the waveform generator
                self.det[ifo] = Detector(ifo)
                dt = self.det[ifo].time_delay_from_earth_center(
                    self.fid_params["ra"],
                    self.fid_params["dec"],
                    self.fid_params["tc"],
                )
                self.ta[ifo] = self.fid_params["tc"] + dt
                fp, fc = self.det[ifo].antenna_pattern(
                    self.fid_params["ra"],
                    self.fid_params["dec"],
                    self.fid_params["polarization"],
                    self.fid_params["tc"],
                )
                curr_wav = fid_hp * fp + fid_hc * fc

            # check for zeros at low and high frequencies
            # make sure only nonzero samples are included in bins
            try:
                first_nonzero, last_nonzero = _threshold_extent(curr_wav, 0.0)
            except IndexError as exc:
                # Preserve the legacy ``list.index(True)`` failure for an
                # entirely zero fiducial waveform.
                raise ValueError("True is not in list") from exc
            numzeros_lo = first_nonzero
            if numzeros_lo > 0:
                new_kmin = self.kmin[ifo] + numzeros_lo
                f_lo = new_kmin * self.df[ifo]
                logging.info(
                    "WARNING! Fiducial waveform starts above "
                    "low-frequency-cutoff, initial bin frequency "
                    "will be %s Hz",
                    f_lo,
                )
            numzeros_hi = len(curr_wav) - last_nonzero - 1
            if numzeros_hi > 0:
                new_kmax = self.kmax[ifo] - numzeros_hi
                f_hi = new_kmax * self.df[ifo]
                logging.info(
                    "WARNING! Fiducial waveform terminates below "
                    "high-frequency-cutoff, final bin frequency "
                    "will be %s Hz",
                    f_hi,
                )

            self.ta[ifo] -= self.end_time[ifo]
            # Apply the time shift to the data in lieu of the reference
            # waveform. This makes target/reference comparisons simpler.
            self.h00[ifo], data_shifted = _prepare_reference_data(
                curr_wav,
                self.data[ifo],
                len(self.f[ifo]),
                self.kmin[ifo],
                self.df[ifo],
                self.ta[ifo],
            )

            logging.info("Computing frequency bins")
            fbin_ind = setup_bins(
                f_full=self.f[ifo],
                f_lo=f_lo,
                f_hi=f_hi,
                gammas=gammas,
                eps=float(epsilon),
            )
            logging.info("Using %s bins for this model", len(fbin_ind))

            self.fedges[ifo] = self.f[ifo][fbin_ind]
            self.edges[ifo] = fbin_ind
            self.init_from_frequencies(data_shifted, self.h00, fbin_ind, ifo)
            self.antenna_time[ifo] = self.setup_antenna(
                earth_rotation, int(earth_rotation_mode), self.fedges[ifo]
            )
        self.combine_layout()

    def init_from_frequencies(self, data, h00, fbin_ind, ifo):
        bins = numpy.array(
            [(fbin_ind[i], fbin_ind[i + 1]) for i in range(len(fbin_ind) - 1)]
        )

        # store low res copy of fiducial waveform
        self.h00_sparse[ifo] = h00[ifo].copy().take(fbin_ind)

        # compute summary data
        logging.info(
            "Calculating summary data at frequency resolution %s Hz",
            self.df[ifo],
        )

        a0, a1 = self.summary_product(data, h00[ifo], bins, ifo)
        b0, b1 = self.summary_product(h00[ifo], h00[ifo], bins, ifo)
        self.sdat[ifo] = {"a0": a0, "a1": a1, "b0": abs(b0), "b1": abs(b1)}

    def combine_layout(self):
        # determine the unique ifo layouts
        self.edge_unique = []
        self.ifo_map = {}
        unique_layouts = []
        for ifo in self.fedges:
            for i, layout in enumerate(unique_layouts):
                if numpy.array_equal(layout, self.fedges[ifo]):
                    self.ifo_map[ifo] = i
                    break
            else:
                self.ifo_map[ifo] = len(self.edge_unique)
                unique_layouts.append(self.fedges[ifo])
                self.edge_unique.append(Array(self.fedges[ifo]))
        logging.info("%s unique ifo layouts", len(self.edge_unique))

    def setup_antenna(self, earth_rotation, mode, fedges):
        # Calculate the times to evaluate fp/fc
        self.earth_rotation = earth_rotation
        if earth_rotation is not False:
            logging.info("Enabling frequency-dependent earth rotation")
            from pycbc.waveform.spa_tmplt import spa_length_in_time

            times = spa_length_in_time(
                phase_order=-1,
                mass1=self.fid_params["mass1"],
                mass2=self.fid_params["mass2"],
                f_lower=numpy.array(fedges) / mode * 2.0,
            )
            atimes = self.fid_params["tc"] - times
            self.lik = likelihood_parts_v
            self.mlik = likelihood_parts_multi_v
        else:
            atimes = self.fid_params["tc"]
            if self.still_needs_det_response:
                self.lik = likelihood_parts_det
                self.mlik = likelihood_parts_det_multi
            else:
                self.lik = likelihood_parts
                self.mlik = likelihood_parts_multi
        return atimes

    @property
    def likelihood_function(self):
        self.lformat = None
        if self.marginalize_vector_params:
            p = self.current_params

            vmarg = set(
                k for k in self.marginalize_vector_params if not numpy.isscalar(p[k])
            )

            if self.earth_rotation:
                if set(["tc", "polarization"]).issubset(vmarg):
                    self.lformat = "earth_time_pol"
                    return likelihood_parts_v_pol_time
                elif set(["polarization"]).issubset(vmarg):
                    self.lformat = "earth_pol"
                    return likelihood_parts_v_pol
                elif set(["tc"]).issubset(vmarg):
                    self.lformat = "earth_time"
                    return likelihood_parts_v_time
            else:
                if set(["ra", "dec", "tc"]).issubset(vmarg):
                    return likelihood_parts_vector
                elif set(["tc", "polarization"]).issubset(vmarg):
                    return likelihood_parts_vector
                elif set(["tc"]).issubset(vmarg):
                    return likelihood_parts_vectort
                elif set(["polarization"]).issubset(vmarg):
                    return likelihood_parts_vectorp

        return self.lik

    def summary_product(self, h1, h2, bins, ifo):
        """Calculate the summary values for the inner product <h1|h2>"""
        psd = self.psds[ifo]
        if any(_torch_tensor(value) is not None for value in (h1, h2, psd)):
            from .relbin_torch import summary_product

            return summary_product(h1, h2, psd, self.f[ifo], bins, self.df[ifo])

        # calculate coefficients
        h12 = numpy.conjugate(h1) * h2 / psd

        # constant terms
        a0 = numpy.array(
            [4.0 * self.df[ifo] * h12[low:high].sum() for low, high in bins]
        )

        # linear terms
        a1 = numpy.array(
            [
                4.0
                / (high - low)
                * (h12[low:high] * (self.f[ifo][low:high] - self.f[ifo][low])).sum()
                for low, high in bins
            ]
        )

        return a0, a1

    def _get_torch_likelihood_data(self, ifo, waveform):
        """Cache static likelihood inputs beside a Torch waveform."""
        tensor = _torch_tensor(waveform)
        if tensor is None:
            return None

        cache_key = (tensor.device.type, tensor.device.index, tensor.dtype)
        cache = self._torch_likelihood_cache.setdefault(ifo, {})
        if cache_key not in cache:
            from .relbin_torch import prepare_likelihood_data

            sdat = self.sdat[ifo]
            cache[cache_key] = prepare_likelihood_data(
                tensor,
                self.fedges[ifo],
                self.h00_sparse[ifo],
                sdat["a0"],
                sdat["a1"],
                sdat["b0"],
                sdat["b1"],
            )
        return cache[cache_key]

    def _get_torch_multi_likelihood_data(
        self, m1, m2, ifo, waveform, waveform2, freqs, h00, h002, a0, a1
    ):
        """Cache static multi-signal inputs beside Torch waveforms."""
        tensor = _torch_tensor(waveform)
        if tensor is None or _torch_tensor(waveform2) is None:
            return None

        pair_cache = self._torch_multi_likelihood_cache.setdefault((m1, m2, ifo), {})
        cache_key = (tensor.device.type, tensor.device.index, tensor.dtype)
        if cache_key not in pair_cache:
            from .relbin_torch import prepare_multi_likelihood_data

            pair_cache[cache_key] = prepare_multi_likelihood_data(
                tensor, freqs, h00, h002, a0, a1
            )
        return pair_cache[cache_key]

    def get_waveforms(self, params, keep_torch=False):
        """Get the waveform polarizations for each ifo"""
        if self.still_needs_det_response:
            wfs = {}
            for ifo in self.data:
                wfs.update(
                    get_fd_det_waveform_sequence(
                        ifos=ifo, sample_points=self.fedges[ifo], **params
                    )
                )
            return wfs

        wfs = []
        for edge in self.edge_unique:
            hp, hc = get_fd_waveform_sequence(sample_points=edge, **params)
            if not keep_torch or _torch_tensor(hp) is None:
                hp = hp.numpy()
                hc = hc.numpy()
            wfs.append((hp, hc))
        wf_ret = {ifo: wfs[self.ifo_map[ifo]] for ifo in self.data}

        self.wf_ret = wf_ret
        return wf_ret

    def _polarization_likelihood_parts(self, ifo, params, waveform, likelihood):
        """Evaluate one detector's polarization likelihood products."""
        hp, hc = waveform
        freqs = self.fedges[ifo]
        sdat = self.sdat[ifo]
        h00 = self.h00_sparse[ifo]
        times = self.antenna_time[ifo]
        detector = self.det[ifo]
        torch_data = None
        if likelihood in _TORCH_POLARIZATION_LIKELIHOODS:
            torch_data = self._get_torch_likelihood_data(ifo, hp)
        if torch_data is not None:
            from . import relbin_torch

            fp, fc, delay = relbin_torch.detector_response(
                detector, params["ra"], params["dec"], times, hp
            )
        else:
            fp, fc = detector.antenna_pattern(params["ra"], params["dec"], 0.0, times)
            delay = detector.time_delay_from_earth_center(
                params["ra"], params["dec"], times
            )
        earth_time = self.lformat in ("earth_time", "earth_time_pol")
        if earth_time:
            dtc = params["tc"] - self.end_time[ifo] - self.ta[ifo]
        else:
            dtc = params["tc"] + delay - self.end_time[ifo] - self.ta[ifo]
        if torch_data is not None:
            freqs_t, h00_t, a0_t, a1_t, b0_t, b1_t = torch_data
            pol_phase = relbin_torch.polarization_phase(params["polarization"], hp)
            if self.lformat == "earth_time_pol":
                filt, norm = relbin_torch.likelihood_parts_v_pol_time(
                    freqs_t,
                    fp,
                    fc,
                    delay,
                    dtc,
                    pol_phase,
                    hp,
                    hc,
                    h00_t,
                    a0_t,
                    a1_t,
                    b0_t,
                    b1_t,
                )
            elif self.lformat == "earth_pol":
                filt, norm = relbin_torch.likelihood_parts_v_pol(
                    freqs_t,
                    fp,
                    fc,
                    dtc,
                    pol_phase,
                    hp,
                    hc,
                    h00_t,
                    a0_t,
                    a1_t,
                    b0_t,
                    b1_t,
                )
            else:
                fp, fc = relbin_torch.polarized_antenna_response(fp, fc, pol_phase, hp)
                if self.lformat == "earth_time":
                    filt, norm = relbin_torch.likelihood_parts_v_time(
                        freqs_t,
                        fp,
                        fc,
                        delay,
                        dtc,
                        hp,
                        hc,
                        h00_t,
                        a0_t,
                        a1_t,
                        b0_t,
                        b1_t,
                    )
                elif likelihood in (
                    likelihood_parts_vector,
                    likelihood_parts_vectort,
                    likelihood_parts_vectorp,
                ):
                    filt, norm = relbin_torch.likelihood_parts_vector(
                        freqs_t, fp, fc, dtc, hp, hc, h00_t, a0_t, a1_t, b0_t, b1_t
                    )
                elif likelihood is likelihood_parts_v:
                    filt, norm = relbin_torch.likelihood_parts_v(
                        freqs_t, fp, fc, dtc, hp, hc, h00_t, a0_t, a1_t, b0_t, b1_t
                    )
                else:
                    filt, norm = relbin_torch.likelihood_parts(
                        freqs_t, fp, fc, dtc, hp, hc, h00_t, a0_t, a1_t, b0_t, b1_t
                    )
        else:
            hp, hc = _numpy_value(hp), _numpy_value(hc)
            pol_phase = numpy.exp(-2.0j * params["polarization"])
            if self.lformat == "earth_time_pol":
                filt, norm = likelihood(
                    freqs,
                    fp,
                    fc,
                    delay,
                    dtc,
                    pol_phase,
                    hp,
                    hc,
                    h00,
                    sdat["a0"],
                    sdat["a1"],
                    sdat["b0"],
                    sdat["b1"],
                )
            elif self.lformat == "earth_pol":
                filt, norm = likelihood(
                    freqs,
                    fp,
                    fc,
                    dtc,
                    pol_phase,
                    hp,
                    hc,
                    h00,
                    sdat["a0"],
                    sdat["a1"],
                    sdat["b0"],
                    sdat["b1"],
                )
            else:
                response = (fp + 1.0j * fc) * pol_phase
                fp = response.real.copy()
                fc = response.imag.copy()
                if self.lformat == "earth_time":
                    filt, norm = likelihood(
                        freqs,
                        fp,
                        fc,
                        delay,
                        dtc,
                        hp,
                        hc,
                        h00,
                        sdat["a0"],
                        sdat["a1"],
                        sdat["b0"],
                        sdat["b1"],
                    )
                else:
                    filt, norm = likelihood(
                        freqs,
                        fp,
                        fc,
                        dtc,
                        hp,
                        hc,
                        h00,
                        sdat["a0"],
                        sdat["a1"],
                        sdat["b0"],
                        sdat["b1"],
                    )
        return filt, norm, (fp, fc, dtc, hp, hc, h00)

    @property
    def multi_signal_support(self):
        """The list of classes that this model supports in a multi-signal
        likelihood
        """
        # Check if this model *can* be included in a multi-signal model.
        # All marginalizations must currently be disabled to work!
        if (
            self.marginalize_vector_params
            or self.marginalize_distance
            or self.marginalize_phase
        ):
            logging.info(
                "Cannot use single template model inside of"
                "multi_signal if marginalizations are enabled"
            )
        return [type(self)]

    def calculate_hihjs(self, models):
        """Pre-calculate the hihj inner products on a grid"""
        self.hihj = {}
        self._torch_multi_likelihood_cache = {}
        for m1, m2 in itertools.combinations(models, 2):
            self.hihj[(m1, m2)] = {}
            for ifo in self.data:
                h1 = m1.h00[ifo]
                h2 = m2.h00[ifo]

                # Combine the grids
                edge = numpy.unique([m1.edges[ifo], m2.edges[ifo]])

                # Remove any points where either reference is zero
                if any(_torch_tensor(value) is not None for value in (h1, h2)):
                    from .relbin_torch import active_edge_bins

                    bins, fedge = active_edge_bins(h1, h2, m1.f[ifo], edge)
                else:
                    keep = numpy.where((h1[edge] != 0) | (h2[edge] != 0))[0]
                    edge = edge[keep]
                    fedge = m1.f[ifo][edge]
                    bins = numpy.array(
                        [(edge[i], edge[i + 1]) for i in range(len(edge) - 1)]
                    )
                a0, a1 = self.summary_product(h1, h2, bins, ifo)
                self.hihj[(m1, m2)][ifo] = a0, a1, fedge

    def _multi_likelihood_parts(self, m1, m2, det):
        """Evaluate one pairwise multi-signal cross term."""
        a0, a1, fedge = self.hihj[(m1, m2)][det]

        if self.still_needs_det_response:
            dtc, channel, h00 = m1._current_wf_parts[det]
            dtc2, channel2, h002 = m2._current_wf_parts[det]
            torch_data = self._get_torch_multi_likelihood_data(
                m1, m2, det, channel, channel2, fedge, h00, h002, a0, a1
            )
            if torch_data is not None:
                from .relbin_torch import (
                    likelihood_parts_det_multi as torch_mlik,
                )

                freqs_t, h00_t, h002_t, a0_t, a1_t = torch_data
                return torch_mlik(
                    freqs_t, dtc, channel, h00_t, dtc2, channel2, h002_t, a0_t, a1_t
                )

            channel = _numpy_value(channel)
            channel2 = _numpy_value(channel2)
            return self.mlik(fedge, dtc, channel, h00, dtc2, channel2, h002, a0, a1)

        fp, fc, dtc, hp, hc, h00 = m1._current_wf_parts[det]
        fp2, fc2, dtc2, hp2, hc2, h002 = m2._current_wf_parts[det]
        torch_data = self._get_torch_multi_likelihood_data(
            m1, m2, det, hp, hp2, fedge, h00, h002, a0, a1
        )
        if torch_data is not None:
            from . import relbin_torch

            freqs_t, h00_t, h002_t, a0_t, a1_t = torch_data
            if self.mlik is likelihood_parts_multi_v:
                torch_mlik = relbin_torch.likelihood_parts_multi_v
            else:
                torch_mlik = relbin_torch.likelihood_parts_multi
            return torch_mlik(
                freqs_t,
                fp,
                fc,
                dtc,
                hp,
                hc,
                h00_t,
                fp2,
                fc2,
                dtc2,
                hp2,
                hc2,
                h002_t,
                a0_t,
                a1_t,
            )

        fp, fc = _numpy_value(fp), _numpy_value(fc)
        fp2, fc2 = _numpy_value(fp2), _numpy_value(fc2)
        hp, hc = _numpy_value(hp), _numpy_value(hc)
        hp2, hc2 = _numpy_value(hp2), _numpy_value(hc2)
        return self.mlik(
            fedge, fp, fc, dtc, hp, hc, h00, fp2, fc2, dtc2, hp2, hc2, h002, a0, a1
        )

    def multi_loglikelihood(self, models):
        """Calculate a multi-model (signal) likelihood"""
        models = [self] + models
        loglr = 0
        # handle sum[<d|h_i> - 0.5 <h_i|h_i>]
        for m in models:
            loglr += m.loglr

        if not hasattr(self, "hihj"):
            self.calculate_hihjs(models)

        # Cross terms contribute
        # -0.5 * re(<h1|h2> + <h2|h1>) = -re(<h1|h2>).
        for m1, m2 in itertools.combinations(models, 2):
            for det in self.data:
                loglr -= self._multi_likelihood_parts(m1, m2, det).real
        return loglr + self.lognl

    @catch_waveform_error
    def _loglr(self):
        r"""Computes the log likelihood ratio,
        or inner product <s|h> and <h|h> if `self.return_sh_hh` is True.

        .. math::

            \log \mathcal{L}(\Theta) = \sum_i
                \left<h_i(\Theta)|d_i\right> -
                \frac{1}{2}\left<h_i(\Theta)|h_i(\Theta)\right>,

        at the current parameter values :math:`\Theta`.

        Returns
        -------
        float
            The value of the log likelihood ratio.
        or
        tuple
            The inner product (<s|h>, <h|h>).
        """
        # get model params
        p = self.current_params
        wfs = self.get_waveforms(p, keep_torch=True)
        lik = self.likelihood_function
        norm = 0.0
        filt = 0j
        self._current_wf_parts = {}

        for ifo in self.data:
            freqs = self.fedges[ifo]
            sdat = self.sdat[ifo]
            h00 = self.h00_sparse[ifo]

            # project waveform to detector frame if waveform does not deal
            # with detector response. Otherwise, skip detector response.

            if self.still_needs_det_response:
                dtc = 0.0

                channel = wfs[ifo]
                torch_data = None
                if lik is likelihood_parts_det:
                    torch_data = self._get_torch_likelihood_data(ifo, channel)
                if torch_data is not None:
                    from .relbin_torch import likelihood_parts_det as torch_lik

                    freqs_t, h00_t, a0_t, a1_t, b0_t, b1_t = torch_data
                    filter_i, norm_i = torch_lik(
                        freqs_t, dtc, channel, h00_t, a0_t, a1_t, b0_t, b1_t
                    )
                else:
                    channel = _numpy_value(channel)
                    filter_i, norm_i = lik(
                        freqs,
                        dtc,
                        channel,
                        h00,
                        sdat["a0"],
                        sdat["a1"],
                        sdat["b0"],
                        sdat["b1"],
                    )
                self._current_wf_parts[ifo] = (dtc, channel, h00)
            else:
                filter_i, norm_i, wf_parts = self._polarization_likelihood_parts(
                    ifo, p, wfs[ifo], lik
                )
                self._current_wf_parts[ifo] = wf_parts

            filt += filter_i
            norm += norm_i

        loglr = self.marginalize_loglr(filt, norm)
        if self.return_sh_hh:
            filt_tensor = _torch_tensor(filt)
            if filt_tensor is not None and filt_tensor.ndim == 0:
                filt = filt.item()
                norm = norm.item()
            results = (filt, norm)
        else:
            results = loglr
        return results

    def _nowaveform_handler(self):
        """Returns -inf for loglr if no waveform generated.

        If `return_sh_hh` is set to True, a FailedWaveformError will be raised.
        """
        if self.return_sh_hh:
            raise FailedWaveformError(
                "Waveform failed to generate and "
                "return_sh_hh set to True! I don't know "
                "what to return in this case."
            )
        return -numpy.inf

    def write_metadata(self, fp, group=None):
        """Adds writing the fiducial parameters and epsilon to file's attrs.

        Parameters
        ----------
        fp : pycbc.inference.io.BaseInferenceFile instance
            The inference file to write to.
        group : str, optional
            If provided, the metadata will be written to the attrs specified
            by group, i.e., to ``fp[group].attrs``. Otherwise, metadata is
            written to the top-level attrs (``fp.attrs``).
        """
        super().write_metadata(fp, group=group)
        if group is None:
            attrs = fp.attrs
        else:
            attrs = fp[group].attrs
        for p, v in self.fid_params.items():
            attrs["{}_ref".format(p)] = v

    def max_curvature_from_reference(self):
        """Return the maximum change in slope between frequency bins
        relative to the reference waveform.
        """
        dmax = 0
        for ifo in self.data:
            waveform = self.wf_ret[ifo][0]
            waveform_tensor = _torch_tensor(waveform)
            if waveform_tensor is not None:
                import torch

                reference = self.h00_sparse[ifo]
                reference_tensor = _torch_tensor(reference)
                if reference_tensor is None:
                    reference_tensor = torch.as_tensor(
                        reference, device=waveform_tensor.device
                    )
                dtype = torch.promote_types(
                    waveform_tensor.dtype, reference_tensor.dtype
                )
                ratio = waveform_tensor.to(dtype=dtype) / reference_tensor.to(
                    device=waveform_tensor.device, dtype=dtype
                )
                ratio = ratio / torch.abs(ratio).min()
                curvature = ratio[2:] - 2 * ratio[1:-1] + ratio[:-2]
                d = torch.abs(curvature).max().item()
            else:
                waveform = _numpy_value(waveform)
                r = waveform / self.h00_sparse[ifo]
                d = abs(numpy.diff(r / abs(r).min(), n=2)).max()
            dmax = d if dmax < d else dmax
        return dmax

    @staticmethod
    def extra_args_from_config(cp, section, skip_args=None, dtypes=None):
        """Adds reading fiducial waveform parameters from config file."""
        # add fiducial params to skip list
        skip_args += [
            option for option in cp.options(section) if option.endswith("_ref")
        ]

        # get frequency power-law indices if specified
        # NOTE these should be supplied in units of 1/3
        gammas = None
        if cp.has_option(section, "gammas"):
            skip_args.append("gammas")
            gammas = numpy.array(
                [float(g) / 3.0 for g in cp.get(section, "gammas").split()]
            )
        args = super(Relative, Relative).extra_args_from_config(
            cp, section, skip_args=skip_args, dtypes=dtypes
        )

        # get fiducial params from config
        fid_params = {
            p.replace("_ref", ""): float(cp.get("model", p))
            for p in cp.options("model")
            if p.endswith("_ref")
        }

        # add optional params with default values if not specified
        opt_params = {
            "ra": numpy.pi,
            "dec": 0.0,
            "inclination": 0.0,
            "polarization": numpy.pi,
        }
        fid_params.update({p: opt_params[p] for p in opt_params if p not in fid_params})
        args.update({"fiducial_params": fid_params, "gammas": gammas})
        return args


class RelativeTime(Relative):
    """Heterodyne likelihood optimized for time marginalization. In addition
    it supports phase (dominant-mode), sky location, and polarization
    marginalization.
    """

    name = "relative_time"

    def __init__(self, *args, sample_rate=4096, **kwargs):
        super(RelativeTime, self).__init__(*args, **kwargs)
        self.sample_rate = float(sample_rate)
        self.setup_peak_lock(sample_rate=self.sample_rate, **kwargs)
        self.draw_ifos(self.ref_snr, **kwargs)

    @property
    def ref_snr(self):
        if not hasattr(self, "_ref_snr"):
            wfs = {
                ifo: (self.h00_sparse[ifo], self.h00_sparse[ifo])
                for ifo in self.h00_sparse
            }
            self._ref_snr = self.get_snr(wfs)
        return self._ref_snr

    def get_snr(self, wfs):
        """Return hp/hc maximized SNR time series"""
        delta_t = 1.0 / self.sample_rate
        snrs = {}
        for ifo in wfs:
            sdat = self.sdat[ifo]
            dtc = self.tstart[ifo] - self.end_time[ifo] - self.ta[ifo]
            hp, hc = wfs[ifo]
            torch_data = self._get_torch_likelihood_data(ifo, hp)
            if torch_data is not None:
                from .relbin_torch import snr_predictor as torch_predictor

                freqs, h00, a0, a1, b0, b1 = torch_data
                snr = torch_predictor(
                    freqs,
                    dtc - delta_t * 2.0,
                    delta_t,
                    self.num_samples[ifo] + 4,
                    hp,
                    hc,
                    h00,
                    a0,
                    a1,
                    b0,
                    b1,
                )
            else:
                hp, hc = _numpy_value(hp), _numpy_value(hc)
                snr = snr_predictor(
                    self.fedges[ifo],
                    dtc - delta_t * 2.0,
                    delta_t,
                    self.num_samples[ifo] + 4,
                    hp,
                    hc,
                    self.h00_sparse[ifo],
                    sdat["a0"],
                    sdat["a1"],
                    sdat["b0"],
                    sdat["b1"],
                )
            snrs[ifo] = _time_series_from_values(
                snr, delta_t, self.tstart[ifo] - delta_t * 2.0
            )
        return snrs

    @catch_waveform_error
    def _loglr(self):
        r"""Computes the log likelihood ratio,

        .. math::

            \log \mathcal{L}(\Theta) = \sum_i
                \left<h_i(\Theta)|d_i\right> -
                \frac{1}{2}\left<h_i(\Theta)|h_i(\Theta)\right>,

        at the current parameter values :math:`\Theta`.

        Returns
        -------
        float
            The value of the log likelihood ratio.
        """
        # get model params
        p = self.current_params
        wfs = self.get_waveforms(p, keep_torch=True)
        lik = self.likelihood_function
        norm = 0.0
        filt = 0j

        self.snr_draw(wfs)
        p = self.current_params

        for ifo in self.data:
            filter_i, norm_i, _ = self._polarization_likelihood_parts(
                ifo, p, wfs[ifo], lik
            )
            filt += filter_i
            norm += norm_i
        loglr = self.marginalize_loglr(filt, norm)
        return loglr

    def _nowaveform_handler(self):
        """Sets loglr values if no waveform generated."""
        return -numpy.inf


class RelativeTimeDom(RelativeTime):
    """Heterodyne likelihood optimized for time marginalization and only
    dominant-mode waveforms. This enables the ability to do inclination
    marginalization in addition to the other forms supportedy by RelativeTime.
    """

    name = "relative_time_dom"

    def get_snr(self, wfs):
        """Return hp/hc maximized SNR time series"""
        delta_t = 1.0 / self.sample_rate
        snrs = {}
        self.sh = {}
        self.hh = {}
        for ifo in wfs:
            sdat = self.sdat[ifo]
            dtc = self.tstart[ifo] - self.end_time[ifo] - self.ta[ifo]
            hp = wfs[ifo][0]
            torch_data = self._get_torch_likelihood_data(ifo, hp)
            if torch_data is not None:
                from .relbin_torch import snr_predictor_dom as torch_predictor

                freqs, h00, a0, a1, b0, b1 = torch_data
                sh, hh = torch_predictor(
                    freqs,
                    dtc - delta_t * 2.0,
                    delta_t,
                    self.num_samples[ifo] + 4,
                    hp,
                    h00,
                    a0,
                    a1,
                    b0,
                    b1,
                )
            else:
                hp = _numpy_value(hp)
                sh, hh = snr_predictor_dom(
                    self.fedges[ifo],
                    dtc - delta_t * 2.0,
                    delta_t,
                    self.num_samples[ifo] + 4,
                    hp,
                    self.h00_sparse[ifo],
                    sdat["a0"],
                    sdat["a1"],
                    sdat["b0"],
                    sdat["b1"],
                )
            snr = _time_series_from_values(
                abs(sh[2:-2]) / hh**0.5, delta_t, self.tstart[ifo]
            )
            self.sh[ifo] = _time_series_from_values(
                sh, delta_t, self.tstart[ifo] - delta_t * 2.0
            )
            self.hh[ifo] = hh
            snrs[ifo] = snr

        return snrs

    @catch_waveform_error
    def _loglr(self):
        r"""Computes the log likelihood ratio,
        or inner product <s|h> and <h|h> if `self.return_sh_hh` is True.

        .. math::

            \log \mathcal{L}(\Theta) = \sum_i
                \left<h_i(\Theta)|d_i\right> -
                \frac{1}{2}\left<h_i(\Theta)|h_i(\Theta)\right>,

        at the current parameter values :math:`\Theta`.

        Returns
        -------
        float
            The value of the log likelihood ratio.
        or
        tuple
            The inner product (<s|h>, <h|h>).
        """
        # calculate <d-h|d-h> = <h|h> - 2<h|d> + <d|d> up to a constant
        p = self.current_params

        p2 = p.copy()
        p2.pop("inclination")
        wfs = self.get_waveforms(p2, keep_torch=True)

        sh_total = hh_total = 0

        snrs = self.get_snr(wfs)
        self.snr_draw(snrs=snrs)

        for ifo in self.sh:
            if self.precalc_antenna_factors:
                fp, fc, dt = self.get_precalc_antenna_factors(ifo)
            else:
                sh_series_tensor = _torch_tensor(self.sh[ifo])
                if sh_series_tensor is not None:
                    from .relbin_torch import detector_response

                    fp, fc, dt = detector_response(
                        self.det[ifo], p["ra"], p["dec"], p["tc"], sh_series_tensor
                    )
                else:
                    dt = self.det[ifo].time_delay_from_earth_center(
                        p["ra"], p["dec"], p["tc"]
                    )
                    fp, fc = self.det[ifo].antenna_pattern(
                        p["ra"], p["dec"], 0, p["tc"]
                    )
            dts = p["tc"] + dt
            sh = self.sh[ifo].at_time(dts, interpolate="quadratic", extrapolate=0.0j)
            sh_tensor = _torch_tensor(sh)
            if sh_tensor is not None:
                from .relbin_torch import dominant_mode_projection

                htf = dominant_mode_projection(
                    fp, fc, p["polarization"], p["inclination"], sh_tensor
                )
            else:
                ic = numpy.cos(p["inclination"])
                ip = 0.5 * (1.0 + ic * ic)
                pol_phase = numpy.exp(-2.0j * p["polarization"])
                f = (fp + 1.0j * fc) * pol_phase
                # This includes complex conjugation already because the
                # stored inner products were hp* x data.
                htf = f.real * ip + 1.0j * f.imag * ic
            sh_total += sh * htf
            hh_total += self.hh[ifo] * abs(htf) ** 2.0

        loglr = self.marginalize_loglr(sh_total, hh_total)
        if self.return_sh_hh:
            sh_tensor = _torch_tensor(sh_total)
            if sh_tensor is not None and sh_tensor.ndim == 0:
                sh_total = sh_total.item()
                hh_total = hh_total.item()
            results = (sh_total, hh_total)
        else:
            results = loglr
        return results

    def _nowaveform_handler(self):
        """Sets loglr values if no waveform generated."""
        loglr = sh_total = hh_total = -numpy.inf
        if self.return_sh_hh:
            results = (sh_total, hh_total)
        else:
            results = loglr
        return results
