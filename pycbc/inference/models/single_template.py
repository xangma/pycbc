# Copyright (C) 2018 Alex Nitz
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

"""This module provides model classes that assume the noise is Gaussian.
"""

import logging
import numpy
import itertools
import numbers
from functools import lru_cache

from pycbc import filter as pyfilter
from pycbc import scheme as _scheme
from pycbc.types import TimeSeries
from pycbc.waveform import get_fd_waveform
from pycbc.detector import Detector
from pycbc.detector.ground import (
    _DETECTOR_BUILTIN_METHODS,
    _scalar_antenna_pattern_and_time_delay,
)

from .gaussian_noise import BaseGaussianNoise
from .tools import DistMarg, _torch_tensor


(
    _DETECTOR_ANTENNA_PATTERN,
    _DETECTOR_TIME_DELAY_FROM_EARTH_CENTER,
    _DETECTOR_TIME_DELAY_FROM_LOCATION,
    _DETECTOR_GMST_ESTIMATE,
    _DETECTOR_SET_GMST_REFERENCE,
) = _DETECTOR_BUILTIN_METHODS


def _host_scalar_extrinsics(parameters):
    """Return whether projection parameters are ordinary host scalars.

    Scalar inference evaluates one point at a time.  Sending those values
    through the generic Torch vector/autograd kernels adds many tiny tensor
    operations without moving the interpolated SNR series off device.  Keep
    tensors and parameter grids on that generic path so batching and gradients
    retain their existing semantics.
    """
    names = (
        'ra', 'dec', 'tc', 'polarization', 'inclination', 'distance'
    )
    if 'coa_phase' in parameters:
        names += ('coa_phase',)
    return all(
        _torch_tensor(parameters[name]) is None
        and numpy.ndim(parameters[name]) == 0
        for name in names
    )


@lru_cache(maxsize=32)
def _plain_host_scalar_types(value_types):
    """Classify the stable type signature of scalar extrinsics."""
    return all(
        issubclass(value_type, (numbers.Number, numpy.number))
        for value_type in value_types
    )


def _plain_host_scalar_extrinsics(parameters):
    """Cheaply recognize the common non-tensor scalar parameter set.

    Only the types are cached.  Tensor values, dtypes, devices and autograd
    state therefore remain visible whenever a caller changes parameters.
    """
    names = (
        'ra', 'dec', 'tc', 'polarization', 'inclination', 'distance'
    )
    if 'coa_phase' in parameters:
        names += ('coa_phase',)
    return _plain_host_scalar_types(
        tuple(type(parameters[name]) for name in names)
    )


def _torch_cpu_native_scalar_likelihood_eligible(
        model, host_storage, plain_host_scalars, skip_vector,
        sh_tensors=None):
    """Whether one scalar likelihood can finish with native CPU scalars.

    The public, non-marginalized scalar likelihood already returns a Python
    float. For the ordinary complex128 Torch-CPU storage case, finishing
    the final real scalar reduction through NumPy avoids tiny Torch kernels
    and an extra tensor conversion. Keep every dynamic, differentiable,
    lower-precision, overridden, or accelerator configuration on the
    established Torch path.
    """
    if (
        type(model) is not SingleTemplate
        or host_storage
        or not plain_host_scalars
        or skip_vector
    ):
        return False
    if getattr(
        getattr(model, 'marginalize_loglr', None), '__func__', None
    ) is not DistMarg.marginalize_loglr:
        return False
    vector_params = getattr(model, 'marginalize_vector_params', None)
    if (
        getattr(model, 'marginalize_phase', None) is not False
        or getattr(model, 'marginalize_distance', None) is not False
        or getattr(model, 'distance_marginalization', None) is not False
        or getattr(model, 'distance_interpolator', False) is not None
        or type(vector_params) is not dict
        or vector_params
        or getattr(model, 'reconstruct_phase', None) is not False
        or getattr(model, 'reconstruct_distance', None) is not False
        or getattr(model, 'reconstruct_vector', None) is not False
    ):
        return False
    if not model.sh:
        return False

    import torch
    from pycbc.types.timeseries import _torch_has_autograd_state

    for ifo, series in model.sh.items():
        tensor = (
            _torch_tensor(series)
            if sh_tensors is None else sh_tensors.get(ifo)
        )
        if (
            type(series) is not TimeSeries
            or type(model.det.get(ifo)) is not Detector
            or tensor is None
            or tensor.device.type != 'cpu'
            or tensor.layout != torch.strided
            or tensor.dtype != torch.complex128
            or tensor.ndim != 1
            or not tensor.is_contiguous()
            or tensor.is_conj()
            or tensor.is_neg()
            or _torch_has_autograd_state(tensor, torch)
            or type(model.hh.get(ifo)) not in (float, numpy.float64)
        ):
            return False
    return True


def _torch_cpu_scalar_detector_projection(detector, parameters):
    """Return one exact built-in scalar detector projection, if eligible.

    ``SingleTemplate`` ordinarily asks a detector separately for its antenna
    response and geocentric delay, repeating the same GMST and sky
    trigonometry.  The common Torch-CPU scalar likelihood can share those
    terms, but only when every method and float64 geometry involved is the
    stock built-in implementation.  ``None`` leaves subclasses, instance or
    class overrides, unusual scalar types, and mutable custom geometries on
    the established public calls.
    """
    if type(detector) is not Detector:
        return None
    if (
        Detector.antenna_pattern is not _DETECTOR_ANTENNA_PATTERN
        or (
            Detector.time_delay_from_earth_center
            is not _DETECTOR_TIME_DELAY_FROM_EARTH_CENTER
        )
        or (
            Detector.time_delay_from_location
            is not _DETECTOR_TIME_DELAY_FROM_LOCATION
        )
        or Detector.gmst_estimate is not _DETECTOR_GMST_ESTIMATE
        or Detector.set_gmst_reference is not _DETECTOR_SET_GMST_REFERENCE
        or any(name in detector.__dict__ for name in (
            'antenna_pattern', 'time_delay_from_earth_center',
            'time_delay_from_location', 'gmst_estimate',
            'set_gmst_reference',
        ))
    ):
        return None

    values = tuple(parameters[name] for name in ('ra', 'dec', 'tc'))
    if any(
        type(value) not in (float, numpy.float64)
        or not numpy.isfinite(value)
        for value in values
    ):
        return None
    info = getattr(detector, 'info', None)
    if (
        type(info) is not dict
        or type(detector.response) is not numpy.ndarray
        or detector.response.shape != (3, 3)
        or detector.response.dtype != numpy.float64
        or detector.response is not info.get('response')
        or type(detector.location) is not numpy.ndarray
        or detector.location.shape != (3,)
        or detector.location.dtype != numpy.float64
        or detector.location is not info.get('location')
        or type(detector.reference_time) not in (float, numpy.float64)
        or not numpy.isfinite(detector.reference_time)
    ):
        return None
    return _scalar_antenna_pattern_and_time_delay(
        detector, values[0], values[1], values[2]
    )


def _limit_torch_imrphenomd_generation(parameters, flen, delta_f):
    """Avoid generating IMRPhenomD bins that this model discards.

    ``SingleTemplate`` retains bins ``[0, flen)`` and resizes every generated
    waveform to that length.  IMRPhenomD treats ``f_final`` as an exclusive
    upper evaluation bound, so ``flen * delta_f`` is the smallest bound that
    still evaluates the final retained bin.  Apply the limit only to the
    Torch IMRPhenomD path: the legacy CPU path remains an unchanged benchmark
    reference, while CUDA and MPS benefit from the same shorter evaluation.
    """
    if (
        not isinstance(_scheme.mgr.state, _scheme.TorchScheme)
        or parameters.get('approximant') != 'IMRPhenomD'
    ):
        return

    # This is an optimization of the native generator, not a waveform-input
    # transformation.  Match its public dispatch guards so explicit native
    # opt-outs and unsupported parameter sets reach LAL with their original
    # ``f_final`` value intact.
    from pycbc.waveform.torch_switches import torch_native_enabled

    if not torch_native_enabled(
        'PYCBC_IMRPHENOMD_NATIVE', default=True
    ):
        return

    # Keep explicit opt-outs as lazy as the registry dispatcher: import the
    # implementation only after its switch has selected the native path.
    try:
        from pycbc.waveform.imrphenomd_torch import (
            imrphenomd_cutoff_frequency,
            imrphenomd_native_supported,
        )
    except (ImportError, ModuleNotFoundError):
        return

    if not imrphenomd_native_supported(parameters):
        return

    retained_upper_edge = float(flen) * float(delta_f)
    requested = parameters.get('f_final', 0.0)
    try:
        requested = float(requested)
    except (TypeError, ValueError):
        # Preserve the established downstream validation error.
        return
    if requested == 0.0:
        # A zero value asks IMRPhenomD to use its mass-dependent natural
        # layout.  Replace it only when that layout extends past the bins the
        # model retains; for sufficiently high masses the natural layout is
        # already shorter, and forcing the retained edge would inflate it.
        try:
            natural_upper_edge = imrphenomd_cutoff_frequency(
                parameters['mass1'], parameters['mass2']
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            # Preserve the established downstream parameter validation.
            return
        if (
            not numpy.isfinite(natural_upper_edge)
            or natural_upper_edge <= retained_upper_edge
        ):
            return
    elif (
        not numpy.isfinite(requested)
        or requested <= retained_upper_edge
    ):
        return

    parameters['f_final'] = retained_upper_edge


class SingleTemplate(DistMarg, BaseGaussianNoise):
    r"""Model that assumes we know all the intrinsic parameters.

    This model assumes we know all the intrinsic parameters, and are only
    maximizing over the extrinsic ones. We also assume a dominant mode waveform
    approximant only and non-precessing.


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
    sample_rate : int, optional
        The sample rate to use. Default is 32768.
    polarization_samples: int, optional
        Parameter to specify how finely to marginalize over polarization angle.
        If None, then polarization must be a parameter.
    \**kwargs :
        All other keyword arguments are passed to
        :py:class:`BaseGaussianNoise`; see that class for details.
    """
    name = 'single_template'

    def __init__(self, variable_params, data, low_frequency_cutoff,
                 sample_rate=32768,
                 marginalize_phase=True,
                 **kwargs):
        variable_params, kwargs = self.setup_marginalization(
                                   variable_params,
                                   marginalize_phase=marginalize_phase,
                                   **kwargs)
        super(SingleTemplate, self).__init__(
            variable_params, data, low_frequency_cutoff, **kwargs)

        sample_rate = float(sample_rate)

        # Generate template waveforms
        df = data[self.detectors[0]].delta_f
        self.df = df
        p = self.static_params.copy()
        for k in self.static_params:
            if p[k] == 'REPLACE':
                p.pop(k)
        if 'distance' in p:
            _ = p.pop('distance')
        if 'inclination' in p:
            _ = p.pop('inclination')

        flen = int(round(sample_rate / df) / 2 + 1)
        _limit_torch_imrphenomd_generation(p, flen, df)
        hp, _ = get_fd_waveform(delta_f=df, distance=1, inclination=0, **p)

        # Extend template to high sample rate
        hp.resize(flen)

        # Calculate high sample rate SNR time series
        self.sh = {}
        self.hh = {}
        self.snr = {}
        self.det = {}
        for ifo in self.data:
            flow = self.kmin[ifo] * df
            fhigh = self.kmax[ifo] * df
            # Extend data to high sample rate
            self.data[ifo].resize(flen)
            self.det[ifo] = Detector(ifo)
            snr, _, norm = pyfilter.matched_filter_core(
                hp, self.data[ifo],
                psd=self.psds[ifo],
                low_frequency_cutoff=flow,
                high_frequency_cutoff=fhigh)

            self.sh[ifo] = 4 * df * snr
            self.snr[ifo] = snr * norm

            self.hh[ifo] = pyfilter.sigmasq(
                hp, psd=self.psds[ifo],
                low_frequency_cutoff=flow,
                high_frequency_cutoff=fhigh)

        # The matched-filter series keep their storage backend for this
        # model's lifetime.  Classify it once so the legacy NumPy likelihood
        # does not repeatedly inspect every extrinsic parameter for Torch
        # tensors at every grid point.
        self._sh_storage_is_host = all(
            _torch_tensor(series) is None for series in self.sh.values()
        )

        self.waveform = hp
        self.htfs = {}  # Waveform phase / distance transformation factors
        self.dts = {}

        # Retrict to analyzing around peaks if chosen and choose what
        # ifos to draw from
        self.setup_peak_lock(snrs=self.snr,
                             sample_rate=sample_rate,
                             **kwargs)
        self.draw_ifos(self.snr)

    @property
    def multi_signal_support(self):
        """ The list of classes that this model supports in a multi-signal
        likelihood
        """
        # Check if this model *can* be included in a multi-signal model.
        # All marginalizations must currently be disabled to work!
        if (self.marginalize_vector_params or
            self.marginalize_distance or
            self.marginalize_phase):
            logging.info("Cannot use single template model inside of"
                         "multi_signal if marginalizations are enabled")
        return [type(self)]

    def calculate_hihjs(self, models):
        """ Pre-calculate the hihj inner products on a grid
        """
        self.hihj = {}
        for m1, m2 in itertools.combinations(models, 2):
            self.hihj[(m1, m2)] = {}
            h1 = m1.waveform
            h2 = m2.waveform
            for ifo in self.data:
                flow = self.kmin[ifo] * self.df
                fhigh = self.kmax[ifo] * self.df
                h1h2, _, _ = pyfilter.matched_filter_core(
                h1, h2,
                psd=self.psds[ifo],
                low_frequency_cutoff=flow,
                high_frequency_cutoff=fhigh)
                self.hihj[(m1, m2)][ifo] = 4 * self.df * h1h2

    def multi_loglikelihood(self, models):
        """ Calculate a multi-model (signal) likelihood
        """
        models = [self] + models
        loglr = 0
        # handle sum[<d|h_i> - 0.5 <h_i|h_i>]
        for m in models:
            loglr += m.loglr

        if not hasattr(self, 'hihj'):
            self.calculate_hihjs(models)

        # finally add in the lognl term from this model
        for m1, m2 in itertools.combinations(models, 2):
            for det in self.data:
                hihj_vec = self.hihj[(m1, m2)][det]
                dt = m1.dts[det] - m2.dts[det] + hihj_vec.start_time
                if dt < hihj_vec.start_time:
                    dt += hihj_vec.duration

                h1h2 = hihj_vec.at_time(dt, nearest_sample=True)
                h1h2 *= m1.htfs[det] * m2.htfs[det].conj()
                loglr += - h1h2.real # This is -0.5 * re(<h1|h2> + <h2|h1>)
        return loglr + self.lognl

    def batch_loglr(self, **params):
        """Evaluate independent extrinsic-parameter points as one batch.

        Array-valued parameters are broadcast by the detector response and
        time-series interpolation kernels.  Unlike the established vector
        marginalization path, the leading parameter grid is retained in the
        returned likelihood-ratio array.  NumPy-backed models return a NumPy
        array and Torch-backed models keep the result on their Torch device.

        Vector- and distance-marginalized models have an additional sample
        dimension whose meaning would be ambiguous with a caller-provided
        batch, so those configurations are rejected for now.  Analytic phase
        marginalization remains pointwise and is supported.
        """
        if self.marginalize_vector_params or self.marginalize_distance:
            raise ValueError(
                "batch_loglr does not support vector or distance "
                "marginalization"
            )
        if not params:
            raise ValueError("batch_loglr requires parameter arrays")
        self.update(**params)
        return self._loglr(skip_vector=True)

    def _loglr(self, skip_vector=False):
        r"""Computes the log likelihood ratio

        Returns
        -------
        float
            The value of the log likelihood ratio.
        """
        # calculate <d-h|d-h> = <h|h> - 2<h|d> + <d|d> up to a constant
        p = self.current_params

        sh_total = hh_total = 0

        if not skip_vector:
            self.snr_draw(snrs=self.snr)
        host_storage = getattr(self, '_sh_storage_is_host', None)
        sh_tensors = None
        if host_storage is None:
            # Lightweight models used by callers and tests may not have run
            # ``__init__``.  Preserve their dynamic storage behavior.
            sh_tensors = {
                ifo: _torch_tensor(series)
                for ifo, series in self.sh.items()
            }
            host_storage = all(
                tensor is None for tensor in sh_tensors.values()
            )
        elif not host_storage:
            # Unwrap each PyCBC series exactly once per likelihood call.  The
            # resulting tensors are reused only within this call, so storage
            # replacement, dtype/device changes and autograd state are all
            # re-observed on the next evaluation.
            sh_tensors = {
                ifo: _torch_tensor(series)
                for ifo, series in self.sh.items()
            }
        plain_host_scalars = _plain_host_scalar_extrinsics(p)
        plain_host_fast_path = host_storage and plain_host_scalars
        host_scalar_extrinsics = (
            plain_host_scalars or _host_scalar_extrinsics(p)
        )
        native_scalar_likelihood = (
            _torch_cpu_native_scalar_likelihood_eligible(
                self, host_storage, plain_host_scalars, skip_vector,
                sh_tensors=sh_tensors,
            )
        )
        host_projection_terms = None

        for ifo in self.sh:
            sh_series_tensor = (
                None if host_storage else sh_tensors[ifo]
            )
            # MPS keeps the generic complex64 path: multiplying its split
            # complex PyCBC representation by a NumPy complex scalar can
            # discard the imaginary component.
            use_host_scalar_extrinsics = (
                host_scalar_extrinsics
                and (
                    sh_series_tensor is None
                    or sh_series_tensor.device.type != 'mps'
                )
            )
            scalar_projection = None
            if native_scalar_likelihood:
                scalar_projection = _torch_cpu_scalar_detector_projection(
                    self.det[ifo], p
                )
            if scalar_projection is not None:
                fp, fc, dt = scalar_projection
            elif (
                sh_series_tensor is not None
                and not use_host_scalar_extrinsics
            ):
                from .relbin_torch import detector_response

                fp, fc, dt = detector_response(
                    self.det[ifo], p['ra'], p['dec'], p['tc'],
                    sh_series_tensor)
            else:
                dt = self.det[ifo].time_delay_from_earth_center(
                    p['ra'], p['dec'], p['tc'])
                fp, fc = self.det[ifo].antenna_pattern(
                    p['ra'], p['dec'], 0, p['tc'])
            dt_tensor = (
                None
                if plain_host_fast_path or scalar_projection is not None
                else _torch_tensor(dt)
            )
            if dt_tensor is not None:
                # Convert explicitly at the model boundary. NumPy 2 rejects
                # array + Torch tensor, while relying on Torch's reflected
                # addition raises a NumPy deprecation warning.
                import torch

                tc = torch.as_tensor(
                    p['tc'], device=dt_tensor.device, dtype=dt_tensor.dtype
                )
                self.dts[ifo] = dt_tensor + tc
            else:
                self.dts[ifo] = p['tc'] + dt

            sh = self.sh[ifo].at_time(self.dts[ifo], interpolate='quadratic')
            sh_tensor = (
                sh
                if native_scalar_likelihood
                else None if plain_host_fast_path else _torch_tensor(sh)
            )
            if sh_tensor is not None and not use_host_scalar_extrinsics:
                from .relbin_torch import dominant_mode_template_factor

                htf = dominant_mode_template_factor(
                    fp, fc, p['polarization'], p['inclination'],
                    p.get('coa_phase', 0.0), p['distance'], sh_tensor)
            else:
                if host_projection_terms is None:
                    phase = 1
                    if 'coa_phase' in p:
                        phase = numpy.exp(-1.0j * 2 * p['coa_phase'])
                    ic = numpy.cos(p['inclination'])
                    ip = 0.5 * (1.0 + ic * ic)
                    pol_phase = numpy.exp(-2.0j * p['polarization'])
                    host_projection_terms = phase, ic, ip, pol_phase
                else:
                    phase, ic, ip, pol_phase = host_projection_terms
                f = (fp + 1.0j * fc) * pol_phase
                # This includes complex conjugation already because the
                # stored inner products were hp* x data.
                htf = (
                    (f.real * ip + 1.0j * f.imag * ic)
                    / p['distance'] * phase
                )
            self.htfs[ifo] = htf
            sh_total += sh * htf
            hh_total += self.hh[ifo] * abs(htf) ** 2.0

        if native_scalar_likelihood:
            # Retain Torch's established complex multiplication and
            # accumulation exactly. Only the final real-valued scalar
            # reduction moves through the owning interpolation result's
            # NumPy view, avoiding two more tiny Torch kernels without
            # changing the complex-product rounding.
            return float(
                sh_total.numpy()[()].real - 0.5 * hh_total
            )

        if skip_vector:
            loglr = self.marginalize_loglr(
                sh_total, hh_total, skip_vector=True
            )
        else:
            loglr = self.marginalize_loglr(sh_total, hh_total)
        return loglr
