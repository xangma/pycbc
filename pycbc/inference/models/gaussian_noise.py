# Copyright (C) 2018  Collin Capano
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
import shlex
from abc import ABCMeta
from functools import wraps
import numpy

from pycbc import filter as pyfilter
from pycbc.waveform import (NoWaveformError, FailedWaveformError)
from pycbc.waveform import generator
from pycbc.types import FrequencySeries, TimeSeries, real_same_precision_as, zeros
from pycbc.strain import gates_from_cli
from pycbc.strain.calibration import Recalibrate
from pycbc.inject import InjectionSet
from pycbc.io import FieldArray
from pycbc.types.optparse import MultiDetOptionAction

from .base import ModelStats, _public_stat_value
from .base_data import BaseDataModel
from .data_utils import (data_opts_from_config, data_from_cli,
                         fd_data_from_strain_dict, gate_overwhitened_data)
from .tools import _fused_inner_hd_hh, _torch_tensor


def catch_waveform_error(method):
    """Decorator that will catch no waveform errors.

    This can be added to a method in an inference model. The decorator will
    call the model's `_nowaveform_return` method if either of the following
    happens when the wrapped method is executed:

      * A `NoWaveformError` is raised.
      * A `RuntimeError` or `FailedWaveformError` is raised and the model has
        an `ignore_failed_waveforms` attribute that is set to True.

    This requires the model to have a `_nowaveform_handler` method.
    """
    # the functools.wroaps decorator preserves the original method's name
    # and docstring
    @wraps(method)
    def method_wrapper(self, *args, **kwargs):
        try:
            retval = method(self, *args, **kwargs)
        except NoWaveformError:
            retval = self._nowaveform_handler()
        except (RuntimeError, FailedWaveformError) as e:
            try:
                ignore_failed = self.ignore_failed_waveforms
            except AttributeError:
                ignore_failed = False
            if ignore_failed:
                retval = self._nowaveform_handler()
            else:
                raise e
        return retval
    return method_wrapper


class BaseGaussianNoise(BaseDataModel, metaclass=ABCMeta):
    r"""Model for analyzing GW data with assuming a wide-sense stationary
    Gaussian noise model.

    This model will load gravitational wave data and calculate the log noise
    likelihood ``_lognl`` and normalization. It also implements the
    ``_loglikelihood`` function as the sum of the log likelihood ratio and the
    ``lognl``. It does not implement a log likelihood ratio function
    ``_loglr``, however, since that can differ depending on the signal model.
    Models that analyze GW data assuming it is stationary Gaussian should
    therefore inherit from this class and implement their own ``_loglr``
    function.

    For more details on the inner product used, the log likelihood of the
    noise, and the normalization factor, see :py:class:`GaussianNoise`.

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
    psds : dict, optional
        A dictionary of FrequencySeries keyed by the detector names. The
        dictionary must have a psd for each detector specified in the data
        dictionary. If provided, the inner products in each detector will be
        weighted by 1/psd of that detector.
    high_frequency_cutoff : dict, optional
        A dictionary of ending frequencies, in which the keys are the
        detector names and the values are the ending frequencies for the
        respective detectors to be used for computing inner products. If not
        provided, the minimum of the largest frequency stored in the data
        and a given waveform will be used.
    normalize : bool, optional
        If True, the normalization factor :math:`alpha` will be included in the
        log likelihood. See :py:class:`GaussianNoise` for details. Default is
        to not include it.
    static_params : dict, optional
        A dictionary of parameter names -> values to keep fixed.
    ignore_failed_waveforms : bool, optional
        If the waveform generator raises an error when it tries to generate,
        treat the point as having zero likelihood. This allows the parameter
        estimation to continue. Otherwise, an error will be raised, stopping
        the run. Default is False.
    \**kwargs :
        All other keyword arguments are passed to ``BaseDataModel``.

    Attributes
    ----------
    ignore_failed_waveforms : bool
        If True, points in parameter space that cause waveform generation to
        fail (i.e., they raise a ``FailedWaveformError``) will be treated as
        points with zero likelihood. Otherwise, such points will cause the
        model to raise a ``FailedWaveformError``.
    """

    def __init__(self, variable_params, data, low_frequency_cutoff, psds=None,
                 high_frequency_cutoff=None, normalize=False,
                 static_params=None, ignore_failed_waveforms=False,
                 no_save_data=False,
                 **kwargs):
        # set up the boiler-plate attributes
        super(BaseGaussianNoise, self).__init__(variable_params, data,
                                                static_params=static_params,
                                                no_save_data=no_save_data,
                                                **kwargs)
        self.ignore_failed_waveforms = ignore_failed_waveforms
        self.no_save_data = no_save_data
        # check if low frequency cutoff has been provided for every IFO with
        # data
        for ifo in self.data:
            if low_frequency_cutoff[ifo] is None:
                raise ValueError(
                    "A low-frequency-cutoff must be provided for every "
                    "detector for which data has been provided. If "
                    "loading the model settings from "
                    "a config file, please provide "
                    "`{DETECTOR}:low-frequency-cutoff` options for "
                    "every detector in the `[model]` section, where "
                    "`{DETECTOR} is the name of the detector,"
                    "or provide a single low-frequency-cutoff option"
                    "which will be used for all detectors")

        # check that the data sets all have the same delta fs and delta ts
        dts = numpy.array([d.delta_t for d in self.data.values()])
        dfs = numpy.array([d.delta_f for d in self.data.values()])
        if all(dts == dts[0]) and all(dfs == dfs[0]):
            self.all_ifodata_same_rate_length = True
        else:
            self.all_ifodata_same_rate_length = False
            logging.info(
                "You are using different data segment lengths or "
                "sampling rates for different IFOs")

        # store the number of samples in the time domain
        self._N = {}
        for (det, d) in self._data.items():
            self._N[det] = int(1./(d.delta_f*d.delta_t))

        # set lower/upper frequency cutoff
        if high_frequency_cutoff is None:
            high_frequency_cutoff = {ifo: None for ifo in self.data}
        self._f_upper = high_frequency_cutoff
        self._f_lower = low_frequency_cutoff

        # Set the cutoff indices
        self._kmin = {}
        self._kmax = {}

        for (det, d) in self._data.items():
            kmin, kmax = pyfilter.get_cutoff_indices(self._f_lower[det],
                                                     self._f_upper[det],
                                                     d.delta_f, self._N[det])
            self._kmin[det] = kmin
            self._kmax[det] = kmax

        # store the psd segments
        self._psd_segments = {}
        if psds is not None:
            self.set_psd_segments(psds)

        # store the psds and calculate the inner product weight
        self._psds = {}
        self._invpsds = {}
        self._weight = {}
        self._lognorm = {}
        self._det_lognls = {}
        self._whitened_data = {}

        # set the normalization state
        self._normalize = False
        self.normalize = normalize
        # store the psds and whiten the data
        self.psds = psds

        # attribute for storing the current waveforms
        self._current_wfs = None

        # Initialize NetworkGeometry when detectors are present
        if len(self._data) >= 1:
            try:
                from pycbc.detector import NetworkGeometry
                self.network_geometry = NetworkGeometry(list(self._data.keys()))
            except Exception:
                self.network_geometry = None
        else:
            self.network_geometry = None

    @property
    def high_frequency_cutoff(self):
        """The high frequency cutoff of the inner product."""
        return self._f_upper

    @property
    def low_frequency_cutoff(self):
        """The low frequency cutoff of the inner product."""
        return self._f_lower

    @property
    def kmin(self):
        """Dictionary of starting indices for the inner product.

        This is determined from the lower frequency cutoff and the ``delta_f``
        of the data using
        :py:func:`pycbc.filter.matchedfilter.get_cutoff_indices`.
        """
        return self._kmin

    @property
    def kmax(self):
        """Dictionary of ending indices for the inner product.

        This is determined from the high frequency cutoff and the ``delta_f``
        of the data using
        :py:func:`pycbc.filter.matchedfilter.get_cutoff_indices`. If no high
        frequency cutoff was provided, this will be the indice corresponding to
        the Nyquist frequency.
        """
        return self._kmax

    @property
    def psds(self):
        """Dictionary of detectors -> PSD frequency series.

        If no PSD was provided for a detector, this will just be a frequency
        series of ones.
        """
        return self._psds

    @psds.setter
    def psds(self, psds):
        """Sets the psds, and calculates the weight and norm from them.

        The data and the low and high frequency cutoffs must be set first.
        """
        # check that the data has been set
        if self._data is None:
            raise ValueError("No data set")
        if self._f_lower is None:
            raise ValueError("low frequency cutoff not set")
        if self._f_upper is None:
            raise ValueError("high frequency cutoff not set")
        # make sure the relevant caches are cleared
        self._psds.clear()
        self._invpsds.clear()
        self._weight.clear()
        self._lognorm.clear()
        self._det_lognls.clear()
        self._whitened_data.clear()
        for det, d in self._data.items():
            if psds is None:
                # No psd means assume white PSD
                p = FrequencySeries(
                    zeros(
                        int(self._N[det] / 2 + 1),
                        dtype=real_same_precision_as(d),
                    ),
                    delta_f=d.delta_f,
                    copy=False,
                )
                p.fill(1.)
            else:
                # copy for storage
                p = psds[det].copy()
            self._psds[det] = p
            # we'll store the weight to apply to the inner product
            # only set weight in band we will analyze
            kmin = self._kmin[det]
            kmax = self._kmax[det]
            invp = FrequencySeries(
                zeros(len(p), dtype=p.dtype),
                delta_f=p.delta_f,
                copy=False,
            )
            invp[kmin:kmax] = 1./p[kmin:kmax]
            self._invpsds[det] = invp
            self._weight[det] = numpy.sqrt(4 * invp.delta_f * invp)
            self._whitened_data[det] = d.copy()
            self._whitened_data[det] *= self._weight[det]
        # set the lognl and lognorm; we'll get this by just calling lognl
        _ = self.lognl

    @property
    def psd_segments(self):
        """Dictionary giving times used for PSD estimation for each detector.

        If a detector's PSD was not estimated from data, or the segment wasn't
        provided, that detector will not be in the dictionary.
        """
        return self._psd_segments

    def set_psd_segments(self, psds):
        """Sets the PSD segments from a dictionary of PSDs.

        This attempts to get the PSD segment from a ``psd_segment`` attribute
        of each detector's PSD frequency series. If that attribute isn't set,
        then that detector is not added to the dictionary of PSD segments.

        Parameters
        ----------
        psds : dict
            Dictionary of detector name -> PSD frequency series. The segment
            used for each PSD will try to be retrieved from the PSD's
            ``.psd_segment`` attribute.
        """
        for det, p in psds.items():
            try:
                self._psd_segments[det] = p.psd_segment
            except AttributeError:
                continue

    @property
    def weight(self):
        r"""Dictionary of detectors -> frequency series of inner-product
        weights.

        The weights are :math:`\sqrt{4 \Delta f / S_n(f)}`. This is set when
        the PSDs are set.
        """
        return self._weight

    @property
    def whitened_data(self):
        r"""Dictionary of detectors -> whitened data frequency series.

        The whitened data is the data multiplied by the inner-product weight.
        Note that this includes the :math:`\sqrt{4 \Delta f}` factor. This
        is set when the PSDs are set.
        """
        return self._whitened_data

    def det_lognorm(self, det):
        """The log of the likelihood normalization in the given detector.

        If ``self.normalize`` is False, will just return 0.
        """
        if not self.normalize:
            return 0.
        try:
            return self._lognorm[det]
        except KeyError:
            # hasn't been calculated yet
            p = self._psds[det]
            dt = self._whitened_data[det].delta_t
            kmin = self._kmin[det]
            kmax = self._kmax[det]
            lognorm = -float(self._N[det]*numpy.log(numpy.pi*self._N[det]*dt)/2.
                             + numpy.log(p[kmin:kmax]).sum())
            self._lognorm[det] = lognorm
            return self._lognorm[det]

    @property
    def normalize(self):
        """Determines if the loglikelihood includes the normalization term.
        """
        return self._normalize

    @normalize.setter
    def normalize(self, normalize):
        """Clears the current stats if the normalization state is changed.
        """
        if normalize != self._normalize:
            self._current_stats = ModelStats()
            self._lognorm.clear()
            self._det_lognls.clear()
        self._normalize = normalize

    @property
    def lognorm(self):
        """The log of the normalization of the log likelihood."""
        return sum(self.det_lognorm(det) for det in self._data)

    def det_lognl(self, det):
        r"""Returns the log likelihood of the noise in the given detector:

        .. math::

            \log p(d_i|n_i) = \log \alpha_i -
                \frac{1}{2} \left<d_i | d_i\right>.


        Parameters
        ----------
        det : str
            The name of the detector.

        Returns
        -------
        float :
            The log likelihood of the noise in the requested detector.
        """
        try:
            return self._det_lognls[det]
        except KeyError:
            # hasn't been calculated yet; calculate & store
            kmin = self._kmin[det]
            kmax = self._kmax[det]
            d = self._whitened_data[det]
            lognorm = self.det_lognorm(det)
            lognl = lognorm - 0.5 * d[kmin:kmax].inner(d[kmin:kmax]).real
            self._det_lognls[det] = lognl
            return self._det_lognls[det]

    def _lognl(self):
        """Computes the log likelihood assuming the data is noise.

        Since this is a constant for Gaussian noise, this is only computed once
        then stored.
        """
        return sum(self.det_lognl(det) for det in self._data)

    def update(self, **params):
        # update
        super().update(**params)
        # reset current waveforms
        self._current_wfs = None

    def _loglikelihood(self):
        r"""Computes the log likelihood of the paramaters,

        .. math::

            \log p(d|\Theta, h) = \log \alpha -\frac{1}{2}\sum_i
                \left<d_i - h_i(\Theta) | d_i - h_i(\Theta)\right>,

        at the current parameter values :math:`\Theta`.

        Returns
        -------
        float
            The value of the log likelihood evaluated at the given point.
        """
        # since the loglr has fewer terms, we'll call that, then just add
        # back the noise term that canceled in the log likelihood ratio
        return self.loglr + self.lognl

    def _parse_batched_params(self, *args, **params):
        """Parse positional or keyword batched parameter arguments."""
        if args:
            if len(args) == 1 and isinstance(args[0], dict):
                p = dict(args[0])
                p.update(params)
                params = p
            elif len(args) == 1 and hasattr(args[0], 'dtype') and args[0].dtype.names:
                p = {name: args[0][name] for name in args[0].dtype.names}
                p.update(params)
                params = p
            elif len(args) == 1 and hasattr(args[0], '_fields'):
                p = {name: getattr(args[0], name) for name in args[0]._fields}
                p.update(params)
                params = p
            else:
                raise ValueError("Unexpected positional arguments for batched evaluation")
        return params

    def batched_loglikelihood(self, *args, **params):
        r"""Computes the log likelihood of a batch of parameter samples,

        .. math::

            \log p(d|\Theta_j, h) = \log \mathcal{L}(\Theta_j) + \log p(d|n),

        for each sample :math:`\Theta_j` in the batch.

        Parameters
        ----------
        *args : dict or structured array, optional
            A single positional mapping or structured array of parameter values.
        **params : dict
            Detector-frame parameter names mapped to scalars or one-
            dimensional arrays or tensors of samples. The supported batched
            parameters are ``tc``, ``ra``, ``dec``, and ``polarization``.
            Radiation-frame waveform parameters must be scalar. Batched
            evaluation requires common detector frequency grids and does not
            currently support sampling or waveform transforms, template
            recalibration, or waveform gates.
            Absolute GPS-time arrays or tensors must use 64-bit precision.

        Returns
        -------
        logl : numpy.ndarray or torch.Tensor
            The log likelihood values evaluated across the batch.
        """
        return self.batched_loglr(*args, **params) + self.lognl

    def batched_loglr(self, *args, **params):
        """Evaluate a batch over detector-frame extrinsic parameters.

        Batched ``tc``, ``ra``, ``dec``, and ``polarization`` values are
        supported. Radiation-frame waveform parameters must be scalar.
        Sampling and waveform transforms are not supported. A waveform
        generation failure returns one negative infinity per sample, subject
        to the same ``ignore_failed_waveforms`` policy as scalar evaluation.
        """
        params = self._parse_batched_params(*args, **params)
        try:
            return self._batched_loglr(**params)
        except NoWaveformError:
            pass
        except NotImplementedError:
            raise
        except (RuntimeError, FailedWaveformError):
            if not self.ignore_failed_waveforms:
                raise

        full_params = dict(self.static_params)
        full_params.update(params)
        generator = self.waveform_generator
        batch_size = (
            _detector_frame_batch_size(generator, params, full_params)
            if hasattr(generator, 'location_args') else 1
        )
        # Prefer the data's device, as for the ordinary likelihood, and keep
        # the scalar model state untouched by this independent batch call.
        for value in (*self._data.values(), *full_params.values()):
            tensor = _torch_tensor(value)
            if tensor is not None:
                import torch

                dtype = tensor.real.dtype
                if not dtype.is_floating_point:
                    dtype = torch.get_default_dtype()
                return tensor.new_full((batch_size,), -numpy.inf, dtype=dtype)
        return numpy.full(batch_size, -numpy.inf)

    def _batched_loglr(self, *args, **params):
        """Computes the log likelihood ratio for a batch of parameter samples.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Batched loglr not implemented for this model.")

    def write_metadata(self, fp, group=None):
        """Adds writing the psds, analyzed detectors, and lognl.

        The analyzed detectors, their analysis segments, and the segments
        used for psd estimation are written as
        ``analyzed_detectors``, ``{{detector}}_analysis_segment``, and
        ``{{detector}}_psd_segment``, respectively. These are either written
        to the specified ``group``'s attrs, or to the top level attrs if
        ``group`` is None.

        The total and each detector's lognl is written to the sample group's
        ``attrs``. If a group is specified, the group name will be prependend
        to the lognl labels with ``{group}__``, with any ``/`` in the group
        path replaced with ``__``. For example, if group is ``/a/b``, the
        ``lognl`` will be written as ``a__b__lognl`` in the sample's group
        attrs.

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
        attrs = fp.getattrs(group=group)
        # write the analyzed detectors and times
        attrs['analyzed_detectors'] = self.detectors
        for det, data in self.data.items():
            key = '{}_analysis_segment'.format(det)
            attrs[key] = [float(data.start_time), float(data.end_time)]
        if self._psds is not None and not self.no_save_data:
            fp.write_psd(self._psds, group=group)
        # write the times used for psd estimation (if they were provided)
        for det in self.psd_segments:
            key = '{}_psd_segment'.format(det)
            attrs[key] = list(map(float, self.psd_segments[det]))
        # save the frequency cutoffs
        for det in self.detectors:
            attrs['{}_likelihood_low_freq'.format(det)] = self._f_lower[det]
            if self._f_upper[det] is not None:
                attrs['{}_likelihood_high_freq'.format(det)] = \
                    self._f_upper[det]
        # write the lognl to the samples group attrs
        sampattrs = fp.getattrs(group=fp.samples_group)
        # if a group is specified, prepend the lognl names with it
        if group is None or group == '/':
            prefix = ''
        else:
            prefix = group.replace('/', '__')
            if not prefix.endswith('__'):
                prefix += '__'
        sampattrs['{}lognl'.format(prefix)] = self.lognl
        # also save the lognl in each detector
        for det in self.detectors:
            sampattrs['{}{}_lognl'.format(prefix, det)] = self.det_lognl(det)

    @staticmethod
    def _fd_data_from_strain_dict(opts, strain_dict, psd_strain_dict):
        """Wrapper around :py:func:`data_utils.fd_data_from_strain_dict`."""
        return fd_data_from_strain_dict(opts, strain_dict, psd_strain_dict)

    def _nowaveform_handler(self):
        """Method that gets called if a NoWaveformError or FailedWaveformError
        is raised. See the :py:func:catch_waveform_error decorator for details.

        Here, this will just raise a NotImplementedError, since how this should
        be handled is model dependent. Models that wish to deal with this
        scenario should override this method.
        """
        raise NotImplementedError(
            f"A waveform could not be generated, but this model does not know "
            f"how to handle that. The parameters were: {self.current_params}.")

    @classmethod
    def from_config(cls, cp, data_section='data', data=None, psds=None,
                    **kwargs):
        r"""Initializes an instance of this class from the given config file.

        In addition to ``[model]``, a ``data_section`` (default ``[data]``)
        must be in the configuration file. The data section specifies settings
        for loading data and estimating PSDs. See the `online documentation
        <http://pycbc.org/pycbc/latest/html/inference.html#setting-data>`_ for
        more details.

        The following options are read from the ``[model]`` section, in
        addition to ``name`` (which must be set):

        * ``{{DET}}-low-frequency-cutoff = FLOAT`` :
          The low frequency cutoff to use for each detector {{DET}}. A cutoff
          must be provided for every detector that may be analyzed (any
          additional detectors are ignored).
        * ``{{DET}}-high-frequency-cutoff = FLOAT`` :
          (Optional) A high frequency cutoff for each detector. If not
          provided, the Nyquist frequency is used.
        * ``check-for-valid-times =`` :
          (Optional) If provided, will check that there are no data quality
          flags on during the analysis segment and the segment used for PSD
          estimation in each detector. To check for flags,
          :py:func:`pycbc.dq.query_flag` is used, with settings pulled from the
          ``dq-*`` options in the ``[data]`` section. If a detector has bad
          data quality during either the analysis segment or PSD segment, it
          will be removed from the analysis.
        * ``shift-psd-times-to-valid =`` :
          (Optional) If provided, the segment used for PSD estimation will
          automatically be shifted left or right until a continous block of
          data with no data quality issues can be found. If no block can be
          found with a maximum shift of +/- the requested psd segment length,
          the detector will not be analyzed.
        * ``err-on-missing-detectors =`` :
          Raises an error if any detector is removed from the analysis because
          a valid time could not be found. Otherwise, a warning is printed
          to screen and the detector is removed from the analysis.
        * ``normalize =`` :
          (Optional) Turn on the normalization factor.
        * ``ignore-failed-waveforms =`` :
          Sets the ``ignore_failed_waveforms`` attribute.

        Parameters
        ----------
        cp : WorkflowConfigParser
            Config file parser to read.
        data_section : str, optional
            The name of the section to load data options from.
        \**kwargs :
            All additional keyword arguments are passed to the class. Any
            provided keyword will override what is in the config file.
        """
        # get the injection file, to replace any FROM_INJECTION settings
        if 'injection-file' in cp.options('data'):
            injection_file = cp.get('data', 'injection-file')
        else:
            injection_file = None
        # update any values that are to be retrieved from the injection
        # Note: this does nothing if there are FROM_INJECTION values
        get_values_from_injection(cp, injection_file, update_cp=True)
        args = cls._init_args_from_config(cp)
        # add the injection file
        args['injection_file'] = injection_file
        # check if normalize is set
        if cp.has_option('model', 'normalize'):
            args['normalize'] = True
        if cp.has_option('model', 'ignore-failed-waveforms'):
            args['ignore_failed_waveforms'] = True
        if cp.has_option('model', 'det-frame-waveform'):
            args['det_frame_waveform'] = True
        if cp.has_option('model', 'no-save-data'):
            args['no_save_data'] = True
        # set inverse spectrum to truncate if not set
        if cp.has_option(data_section, 'invpsd-trunc-which-spectrum'):
            cp.set(data_section, 'invpsd-trunc-which-spectrum', 'invpsd')
        # get any other keyword arguments provided in the model section
        ignore_args = [
            'name',
            'normalize',
            'ignore-failed-waveforms',
            'no-save-data',
            'det-frame-waveform'
        ]
        for option in cp.options("model"):
            if option in ("low-frequency-cutoff", "high-frequency-cutoff"):
                ignore_args.append(option)
                name = option.replace('-', '_')
                args[name] = cp.get_cli_option('model', name,
                                               nargs='+', type=float,
                                               action=MultiDetOptionAction)

        if 'low_frequency_cutoff' not in args:
            raise ValueError("low-frequency-cutoff must be provided in the"
                             " model section, but is not found!")

        # data args
        bool_args = ['check-for-valid-times', 'shift-psd-times-to-valid',
                     'err-on-missing-detectors']
        data_args = {arg.replace('-', '_'): True for arg in bool_args
                     if cp.has_option('model', arg)}
        ignore_args += bool_args
        # load the data
        opts = data_opts_from_config(cp, data_section,
                                     args['low_frequency_cutoff'])
        if data is None or psds is None:
            strain_dict, psd_strain_dict = data_from_cli(opts, **data_args)
            # convert to frequency domain and get psds
            stilde_dict, psds = cls._fd_data_from_strain_dict(
                opts, strain_dict, psd_strain_dict)
            # save the psd data segments if the psd was estimated from data
            if opts.psd_estimation:
                _tdict = psd_strain_dict or strain_dict
                for det in psds:
                    psds[det].psd_segment = (_tdict[det].start_time,
                                             _tdict[det].end_time)
            # gate overwhitened if desired
            if opts.gate_overwhitened and opts.gate is not None:
                stilde_dict = gate_overwhitened_data(
                    stilde_dict, psds, opts.gate)
            data = stilde_dict
        args.update({'data': data, 'psds': psds})
        # any extra args
        args.update(cls.extra_args_from_config(cp, "model",
                                               skip_args=ignore_args))
        # get ifo-specific instances of calibration model
        if cp.has_section('calibration'):
            logging.info("Initializing calibration model")
            recalib = {
                ifo: Recalibrate.from_config(cp, ifo, section='calibration')
                for ifo in opts.instruments}
            args['recalibration'] = recalib
        # get gates for templates
        gates = gates_from_cli(opts)
        if gates:
            args['gates'] = gates
        args.update(kwargs)
        return cls(**args)


class GaussianNoise(BaseGaussianNoise):
    r"""Model that assumes data is stationary Gaussian noise.

    With Gaussian noise the log likelihood functions for signal
    :math:`\log p(d|\Theta, h)` and for noise :math:`\log p(d|n)` are given by:

    .. math::

        \log p(d|\Theta, h) &=  \log\alpha -\frac{1}{2} \sum_i
            \left< d_i - h_i(\Theta) | d_i - h_i(\Theta) \right> \\
        \log p(d|n) &= \log\alpha -\frac{1}{2} \sum_i \left<d_i | d_i\right>

    where the sum is over the number of detectors, :math:`d_i` is the data in
    each detector, and :math:`h_i(\Theta)` is the model signal in each
    detector. The (discrete) inner product is given by:

    .. math::

        \left<a_i | b_i\right> = 4\Re \Delta f
            \sum_{k=k_{\mathrm{min}}}^{k_{\mathrm{max}}}
            \frac{\tilde{a}_i^{*}[k] \tilde{b}_i[k]}{S^{(i)}_n[k]},

    where :math:`\Delta f` is the frequency resolution (given by 1 / the
    observation time :math:`T`), :math:`k` is an index over the discretely
    sampled frequencies :math:`f = k \Delta_f`, and :math:`S^{(i)}_n[k]` is the
    PSD in the given detector. The upper cutoff on the inner product
    :math:`k_{\max}` is by default the Nyquist frequency
    :math:`k_{\max} = N/2+1`, where :math:`N = \lfloor T/\Delta t \rfloor`
    is the number of samples in the time domain, but this can be set manually
    to a smaller value.

    The normalization factor :math:`\alpha` is:

    .. math::

        \alpha = \prod_{i} \frac{1}{\left(\pi T\right)^{N/2}
            \prod_{k=k_\mathrm{min}}^{k_{\mathrm{max}}} S^{(i)}_n[k]},

    where the product is over the number of detectors. By default, the
    normalization constant is not included in the log likelihood, but it can
    be turned on using the ``normalize`` keyword argument.

    Note that the log likelihood ratio has fewer terms than the log likelihood,
    since the normalization and :math:`\left<d_i|d_i\right>` terms cancel:

    .. math::

        \log \mathcal{L}(\Theta) = \sum_i \left[
            \left<h_i(\Theta)|d_i\right> -
            \frac{1}{2} \left<h_i(\Theta)|h_i(\Theta)\right> \right]

    Upon initialization, the data is whitened using the given PSDs. If no PSDs
    are given the data and waveforms returned by the waveform generator are
    assumed to be whitened.

    For more details on initialization parameters and definition of terms, see
    :py:class:`models.BaseDataModel`.

    Parameters
    ----------
    variable_params : (tuple of) string(s)
        A tuple of parameter names that will be varied.
    data : dict
        A dictionary of data, in which the keys are the detector names and the
        values are the data (assumed to be unwhitened). The list of keys must
        match the waveform generator's detectors keys, and the epoch of every
        data set must be the same as the waveform generator's epoch.
    low_frequency_cutoff : dict
        A dictionary of starting frequencies, in which the keys are the
        detector names and the values are the starting frequencies for the
        respective detectors to be used for computing inner products.
    psds : dict, optional
        A dictionary of FrequencySeries keyed by the detector names. The
        dictionary must have a psd for each detector specified in the data
        dictionary. If provided, the inner products in each detector will be
        weighted by 1/psd of that detector.
    high_frequency_cutoff : dict, optional
        A dictionary of ending frequencies, in which the keys are the
        detector names and the values are the ending frequencies for the
        respective detectors to be used for computing inner products. If not
        provided, the minimum of the largest frequency stored in the data
        and a given waveform will be used.
    normalize : bool, optional
        If True, the normalization factor :math:`alpha` will be included in the
        log likelihood. Default is to not include it.
    static_params : dict, optional
        A dictionary of parameter names -> values to keep fixed.
    det_frame_waveform : bool
        If True, the waveform will be generated directly in the detector frame
        using the
        :py:class:`~pycbc.waveform.generator.FDomainDirectDetFrameGenerator`.
        This requires the approximant be implemented in
        :py:func:`~pycbc.waveform.get_fd_det_waveform`.
        If False, the
        :py:class:`~pycbc.waveform.generator.FDomainDetFrameGenerator` will be
        used instead. Defaults to :code:`False`.
    \**kwargs :
        All other keyword arguments are passed to ``BaseDataModel``.

    Examples
    --------
    Create a signal, and set up the model using that signal:

    >>> from pycbc import psd as pypsd
    >>> from pycbc.inference.models import GaussianNoise
    >>> from pycbc.waveform.generator import (FDomainDetFrameGenerator,
    ...                                       FDomainCBCGenerator)
    >>> seglen = 4
    >>> sample_rate = 2048
    >>> N = seglen*sample_rate/2+1
    >>> fmin = 30.
    >>> static_params = {'approximant': 'IMRPhenomD', 'f_lower': fmin,
    ...                  'mass1': 38.6, 'mass2': 29.3,
    ...                  'spin1z': 0., 'spin2z': 0., 'ra': 1.37, 'dec': -1.26,
    ...                  'polarization': 2.76, 'distance': 3*500.}
    >>> variable_params = ['tc']
    >>> tsig = 3.1
    >>> generator = FDomainDetFrameGenerator(
    ...     FDomainCBCGenerator, 0., detectors=['H1', 'L1'],
    ...     variable_args=variable_params,
    ...     delta_f=1./seglen, **static_params)
    >>> signal = generator.generate(tc=tsig)
    >>> psd = pypsd.aLIGOZeroDetHighPower(N, 1./seglen, 20.)
    >>> psds = {'H1': psd, 'L1': psd}
    >>> low_frequency_cutoff = {'H1': fmin, 'L1': fmin}
    >>> model = GaussianNoise(variable_params, signal, low_frequency_cutoff,
                              psds=psds, static_params=static_params)

    Set the current position to the coalescence time of the signal:

    >>> model.update(tc=tsig)

    Now compute the log likelihood ratio and prior-weighted likelihood ratio;
    since we have not provided a prior, these should be equal to each other:

    >>> print('{:.2f}'.format(model.loglr))
    282.43
    >>> print('{:.2f}'.format(model.logplr))
    282.43

    Print all of the default_stats:

    >>> print(',\n'.join(['{}: {:.2f}'.format(s, v)
    ...                   for (s, v) in sorted(model.current_stats.items())]))
    H1_cplx_loglr: 177.76+0.00j,
    H1_optimal_snrsq: 355.52,
    L1_cplx_loglr: 104.67+0.00j,
    L1_optimal_snrsq: 209.35,
    logjacobian: 0.00,
    loglikelihood: 0.00,
    loglr: 282.43,
    logprior: 0.00

    Compute the SNR; for this system and PSD, this should be approximately 24:

    >>> from pycbc.conversions import snr_from_loglr
    >>> x = snr_from_loglr(model.loglr)
    >>> print('{:.2f}'.format(x))
    23.77

    Since there is no noise, the SNR should be the same as the quadrature sum
    of the optimal SNRs in each detector:

    >>> x = (model.det_optimal_snrsq('H1') +
    ...      model.det_optimal_snrsq('L1'))**0.5
    >>> print('{:.2f}'.format(x))
    23.77

    Toggle on the normalization constant:

    >>> model.normalize = True
    >>> model.loglikelihood
    835397.8757405131

    Using the same model, evaluate the log likelihood ratio at several points
    in time and check that the max is at tsig:

    >>> import numpy
    >>> times = numpy.linspace(tsig-1, tsig+1, num=101)
    >>> loglrs = numpy.zeros(len(times))
    >>> for (ii, t) in enumerate(times):
    ...     model.update(tc=t)
    ...     loglrs[ii] = model.loglr
    >>> print('tsig: {:.2f}, time of max loglr: {:.2f}'.format(
    ...     tsig, times[loglrs.argmax()]))
    tsig: 3.10, time of max loglr: 3.10

    Create a prior and use it (see distributions module for more details):

    >>> from pycbc import distributions
    >>> uniform_prior = distributions.Uniform(tc=(tsig-0.2,tsig+0.2))
    >>> prior = distributions.JointDistribution(variable_params, uniform_prior)
    >>> model = GaussianNoise(variable_params,
    ...     signal, low_frequency_cutoff, psds=psds, prior=prior,
    ...     static_params=static_params)
    >>> model.update(tc=tsig)
    >>> print('{:.2f}'.format(model.logplr))
    283.35
    >>> print(',\n'.join(['{}: {:.2f}'.format(s, v)
    ...                   for (s, v) in sorted(model.current_stats.items())]))
    H1_cplx_loglr: 177.76+0.00j,
    H1_optimal_snrsq: 355.52,
    L1_cplx_loglr: 104.67+0.00j,
    L1_optimal_snrsq: 209.35,
    logjacobian: 0.00,
    loglikelihood: 0.00,
    loglr: 282.43,
    logprior: 0.92

    """
    name = 'gaussian_noise'

    def __init__(self, variable_params, data, low_frequency_cutoff, psds=None,
                 high_frequency_cutoff=None, normalize=False,
                 static_params=None, det_frame_waveform=False, **kwargs):
        # set up the boiler-plate attributes
        super(GaussianNoise, self).__init__(
            variable_params, data, low_frequency_cutoff, psds=psds,
            high_frequency_cutoff=high_frequency_cutoff, normalize=normalize,
            static_params=static_params, **kwargs)
        # Determine if all data have the same sampling rate and segment length
        if det_frame_waveform:
            generator_class = generator.FDomainDirectDetFrameGenerator
        else:
            generator_class = generator.FDomainDetFrameGenerator
        if self.all_ifodata_same_rate_length:
            # create a waveform generator for all ifos
            self.waveform_generator = create_waveform_generator(
                self.variable_params, self.data,
                generator_class=generator_class,
                waveform_transforms=self.waveform_transforms,
                recalibration=self.recalibration,
                gates=self.gates, **self.static_params)
        else:
            # create a waveform generator for each ifo respestively
            self.waveform_generator = {}
            for det in self.data:
                self.waveform_generator[det] = create_waveform_generator(
                    self.variable_params, {det: self.data[det]},
                    generator_class=generator_class,
                    waveform_transforms=self.waveform_transforms,
                    recalibration=self.recalibration,
                    gates=self.gates, **self.static_params)

    @property
    def _extra_stats(self):
        """Adds ``loglr``, plus ``cplx_loglr`` and ``optimal_snrsq`` in each
        detector."""
        return ['loglr'] + \
               ['{}_cplx_loglr'.format(det) for det in self._data] + \
               ['{}_optimal_snrsq'.format(det) for det in self._data]

    def _nowaveform_handler(self):
        """Convenience function to set loglr values if no waveform generated.
        """
        for det in self._data:
            setattr(self._current_stats, 'loglikelihood', -numpy.inf)
            setattr(self._current_stats, '{}_cplx_loglr'.format(det),
                    -numpy.inf)
            # snr can't be < 0 by definition, so return 0
            setattr(self._current_stats, '{}_optimal_snrsq'.format(det), 0.)
        return -numpy.inf

    @property
    def multi_signal_support(self):
        """ The list of classes that this model supports in a multi-signal
        likelihood
        """
        return [type(self)]

    @catch_waveform_error
    def multi_loglikelihood(self, models):
        """ Calculate a multi-model (signal) likelihood
        """
        # Generate the waveforms for each submodel
        wfs = []
        for m in models + [self]:
            wfs.append(m.get_waveforms())

        # combine into a single waveform
        combine = {}
        for det in self.data:
            mlen = max([len(x[det]) for x in wfs])
            [x[det].resize(mlen) for x in wfs]
            combine[det] = sum([x[det] for x in wfs])

        self._current_wfs = combine
        loglr = self._loglr()
        self._current_wfs = None
        return loglr + self.lognl

    def get_waveforms(self):
        """The waveforms generated using the current parameters.

        If the waveforms haven't been generated yet, they will be generated.

        Returns
        -------
        dict :
            Dictionary of detector names -> FrequencySeries.
        """
        if self._current_wfs is None:
            params = self.current_params
            if self.all_ifodata_same_rate_length:
                wfs = self.waveform_generator.generate(**params)
            else:
                wfs = {}
                for det in self.data:
                    wfs.update(self.waveform_generator[det].generate(**params))
            self._current_wfs = wfs
        return self._current_wfs

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
        float or torch.Tensor
            The scalar log likelihood ratio. Torch-backed calculations retain
            a device-resident scalar tensor so likelihood composition and
            cached access remain differentiable.
        """
        wfs = self.get_waveforms()
        lr = 0.
        for det, h in wfs.items():
            # the kmax of the waveforms may be different than internal kmax
            kmax = min(len(h), self._kmax[det])
            if self._kmin[det] >= kmax:
                # if the waveform terminates before the filtering low frequency
                # cutoff, then the loglr is just 0 for this detector
                cplx_hd = 0j
                hh = 0.
            else:
                slc = slice(self._kmin[det], kmax)
                cplx_hd, hh = _fused_inner_hd_hh(
                    h[slc],
                    self._whitened_data[det][slc],
                    weight=self._weight[det][slc],
                )
            cplx_loglr = cplx_hd - 0.5 * hh
            # store
            setattr(self._current_stats, '{}_optimal_snrsq'.format(det), hh)
            setattr(self._current_stats, '{}_cplx_loglr'.format(det),
                    cplx_loglr)
            lr += cplx_loglr.real
        # also store the loglikelihood, to ensure it is populated in the
        # current stats even if loglikelihood is never called
        self._current_stats.loglikelihood = lr + self.lognl
        return float(lr) if numpy.isscalar(lr) else lr

    def det_cplx_loglr(self, det):
        """Returns the complex log likelihood ratio in the given detector.

        Parameters
        ----------
        det : str
            The name of the detector.

        Returns
        -------
        complex float :
            The complex log likelihood ratio.
        """
        # try to get it from current stats
        try:
            value = getattr(
                self._current_stats, '{}_cplx_loglr'.format(det)
            )
        except AttributeError:
            # Populate through the cached public property. Calling ``_loglr``
            # directly would leave ``current_stats.loglr`` unset, so a later
            # ``self.loglr`` access could regenerate and re-weight waveforms.
            _ = self.loglr
            # now try returning again
            value = getattr(
                self._current_stats, '{}_cplx_loglr'.format(det)
            )
        return _public_stat_value(value)

    def det_optimal_snrsq(self, det):
        """Returns the opitmal SNR squared in the given detector.

        Parameters
        ----------
        det : str
            The name of the detector.

        Returns
        -------
        float :
            The opimtal SNR squared.
        """
        # try to get it from current stats
        try:
            value = getattr(
                self._current_stats, '{}_optimal_snrsq'.format(det)
            )
        except AttributeError:
            # Populate through the cached public property; see the complex
            # detector-stat accessor above.
            _ = self.loglr
            # now try returning again
            value = getattr(
                self._current_stats, '{}_optimal_snrsq'.format(det)
            )
        return _public_stat_value(value)

    def _batched_loglr(self, *args, **params):
        r"""Computes the log likelihood ratio for a batch of parameter samples,

        .. math::

            \log \mathcal{L}(\Theta_j) = \sum_i \left[
                \left<h_i(\Theta_j)|d_i\right> -
                \frac{1}{2}\left<h_i(\Theta_j)|h_i(\Theta_j)\right> \right],

        simultaneously for all samples :math:`j` using ``NetworkGeometry`` and
        fused inner products ``_fused_inner_hd_hh``.

        Parameters
        ----------
        *args : dict or structured array, optional
            A single positional mapping or structured array of parameter values.
        **params : dict
            Detector-frame parameter names mapped to scalars or one-
            dimensional arrays or tensors of samples. Radiation-frame
            waveform parameters must be scalar. Batched evaluation requires
            common detector frequency grids and does not currently support
            template recalibration or waveform gates. Absolute GPS-time arrays
            or tensors must use 64-bit precision.

        Returns
        -------
        loglr : numpy.ndarray or torch.Tensor
            The log likelihood ratio values evaluated across the batch.
        """
        params = self._parse_batched_params(*args, **params)
        total_hd, total_hh, _, _ = _batched_waveform_inner_products(
            self, params, zero_phase=False
        )
        return total_hd.real - 0.5 * total_hh


def _batched_value_shape(value):
    """Return a parameter value's shape without moving Torch data to host."""
    tensor = _torch_tensor(value)
    return tuple(tensor.shape) if tensor is not None else numpy.shape(value)


def _detector_frame_batch_size(generator, supplied_params, full_params):
    """Validate detector-frame-only batching and return its sample count."""
    location_args = set(generator.location_args)
    unsupported = sorted(
        name
        for name, value in supplied_params.items()
        if name not in location_args and _batched_value_shape(value)
    )
    if unsupported:
        raise ValueError(
            "Batched radiation-frame waveform parameters are not supported; "
            "only tc, ra, dec, and polarization may be batched. Got: "
            + ", ".join(unsupported)
        )

    lengths = {}
    for name in location_args:
        if name not in full_params:
            continue
        shape = _batched_value_shape(full_params[name])
        if len(shape) > 1:
            raise ValueError(
                f"Batched detector-frame parameter {name!r} must be scalar "
                "or one-dimensional"
            )
        if shape:
            lengths[name] = shape[0]

    batch_size = max(lengths.values(), default=1)
    if batch_size < 1:
        raise ValueError("Batched detector-frame parameters cannot be empty")
    mismatched = {
        name: length
        for name, length in lengths.items()
        if length not in (1, batch_size)
    }
    if mismatched:
        detail = ", ".join(
            f"{name}={length}" for name, length in sorted(mismatched.items())
        )
        raise ValueError(
            "Batched detector-frame parameter lengths must be one or the "
            f"common batch length {batch_size}; got {detail}"
        )
    return batch_size


def _batched_waveform_inner_products(model, params, zero_phase=False):
    """Evaluate batched waveform frequency-domain projections and inner products
    across all detectors in a single pass using NetworkGeometry and
    _fused_inner_hd_hh.
    """
    if (getattr(model, 'sampling_transforms', None) is not None
            or getattr(model, 'waveform_transforms', None)):
        raise NotImplementedError(
            "Batched likelihood evaluation does not support sampling or "
            "waveform transforms; use scalar model evaluation"
        )
    full_params = dict(model.static_params) if model.static_params else {}
    full_params.update(params)

    gen = (
        model.waveform_generator
        if hasattr(model, 'waveform_generator')
        else None
    )
    if not getattr(model, 'all_ifodata_same_rate_length', True):
        raise NotImplementedError(
            "Batched likelihood evaluation requires all detectors to use "
            "the same sample rate and segment length"
        )
    if isinstance(gen, dict):
        raise NotImplementedError(
            "Batched likelihood evaluation does not support per-detector "
            "waveform generators"
        )

    recalibration = getattr(model, 'recalibration', None)
    if not recalibration and gen is not None:
        recalibration = getattr(gen, 'recalib', None)
    gates = getattr(model, 'gates', None)
    if not gates and gen is not None:
        gates = getattr(gen, 'gates', None)
    if recalibration:
        raise NotImplementedError(
            "Batched likelihood evaluation does not support template "
            "recalibration"
        )
    if gates:
        raise NotImplementedError(
            "Batched likelihood evaluation does not support waveform gates"
        )

    has_rframe = (
        gen is not None
        and hasattr(gen, 'rframe_generator')
        and hasattr(gen, 'location_args')
    )

    if not has_rframe:
        shaped_params = sorted(
            name for name, value in params.items()
            if _batched_value_shape(value)
        )
        if shaped_params:
            raise NotImplementedError(
                "Batched likelihood parameters require a radiation-frame "
                "waveform generator with an explicit detector-frame "
                "contract. Got: " + ", ".join(shaped_params)
            )

    if has_rframe:
        batch_size = _detector_frame_batch_size(
            gen, params, full_params
        )
        rfparams = {
            param: full_params[param]
            for param in full_params
            if param not in gen.location_args
        }
        if zero_phase:
            rfparams['coa_phase'] = 0.0
        hp, hc = gen.rframe_generator.generate(**rfparams)
        from pycbc.waveform.utils import (
            scheme_cast_series as _scheme_cast_series
        )
        hp = _scheme_cast_series(hp)
        hc = _scheme_cast_series(hc)
        if isinstance(hp, TimeSeries):
            df = full_params.get(
                'delta_f', gen.current_params.get('delta_f', hp.delta_f)
            )
            hp = hp.to_frequencyseries(delta_f=df)
            hc = hc.to_frequencyseries(delta_f=df)
            tshift = 1.0 / df - abs(hp._epoch)
        else:
            tshift = 0.0

        epoch = getattr(gen, '_epoch', float(getattr(hp, '_epoch', 0.0)))
        delta_f = hp.delta_f

        ra = full_params.get('ra', 0.0)
        dec = full_params.get('dec', 0.0)
        pol = full_params.get('polarization', 0.0)
        ref_tc = full_params.get('tc', 0.0)
        refframe = full_params.get('tc_ref_frame', 'geocentric')
        generator._validate_absolute_time_precision(ref_tc, epoch)

        total_hd = None
        total_hh = None
        det_hd = {}
        det_hh = {}
        delay_dict = None
        if (
            refframe == 'geocentric'
            and getattr(model, 'network_geometry', None) is not None
        ):
            delays = model.network_geometry.time_delay_from_earth_center(
                ra, dec, ref_tc
            )
            delay_dict = model.network_geometry.to_dict(delays)

        for detname in model._data:
            if getattr(model, 'network_geometry', None) is not None:
                det = model.network_geometry[detname]
            else:
                from pycbc.detector import Detector
                det = Detector(detname)
            if delay_dict is not None:
                offset = delay_dict[detname]
            else:
                offset = generator._detector_time_offset(
                    det, ref_tc, ra, dec, refframe
                )
            tc, dt = generator._arrival_time_and_shift(
                ref_tc, offset, epoch, tshift
            )
            fp, fc = det.antenna_pattern(ra, dec, pol, tc)

            hp_tensor = _torch_tensor(hp)
            hc_tensor = _torch_tensor(hc)
            fp_tensor = _torch_tensor(fp)
            fc_tensor = _torch_tensor(fc)
            dt_tensor = _torch_tensor(dt)

            tensors = (
                hp_tensor, hc_tensor, fp_tensor, fc_tensor, dt_tensor
            )
            if any(tensor is not None for tensor in tensors):
                import torch

                like = next(t for t in tensors if t is not None)
                device = like.device
                real_dtype = like.real.dtype
                complex_dtype = (
                    torch.complex128
                    if real_dtype == torch.float64
                    else torch.complex64
                )

                def _to_tensor(x, dev, dt):
                    if hasattr(x, '_data') and hasattr(x._data, 'tensor'):
                        return x._data.tensor.to(device=dev, dtype=dt)
                    if hasattr(x, 'numpy') and callable(x.numpy):
                        arr = x.numpy()
                    elif (
                        hasattr(x, 'data')
                        and not isinstance(x.data, memoryview)
                    ):
                        arr = numpy.asarray(x.data)
                    else:
                        arr = numpy.asarray(x)
                    return torch.as_tensor(arr, device=dev, dtype=dt)

                hp_t = (
                    hp_tensor
                    if hp_tensor is not None
                    else _to_tensor(hp, device, complex_dtype)
                )
                if hc_tensor is None:
                    hc_t = _to_tensor(hc, device, complex_dtype)
                else:
                    hc_t = hc_tensor.to(
                        device=device, dtype=complex_dtype
                    )

                if hp_t.ndim != 1 or hc_t.ndim != 1:
                    raise ValueError(
                        "batched likelihood waveform parameters must be "
                        "scalar; only detector-frame extrinsics may be "
                        "batched"
                    )

                def _sample_column(value, name):
                    value_tensor = _torch_tensor(value)
                    if value_tensor is None:
                        value_tensor = torch.as_tensor(
                            value, device=device, dtype=real_dtype
                        )
                    else:
                        value_tensor = value_tensor.to(
                            device=device, dtype=real_dtype
                        )
                    if value_tensor.ndim == 0:
                        value_tensor = value_tensor.expand(batch_size)
                    elif (
                        value_tensor.ndim == 1
                        and value_tensor.shape[0] == 1
                    ):
                        value_tensor = value_tensor.expand(batch_size)
                    elif (
                        value_tensor.ndim != 1
                        or value_tensor.shape[0] != batch_size
                    ):
                        raise ValueError(
                            f"Batched detector-frame parameter {name!r} "
                            f"must be scalar or have length {batch_size}"
                        )
                    return value_tensor.unsqueeze(-1)

                fp_t = _sample_column(fp, 'fplus')
                fc_t = _sample_column(fc, 'fcross')
                dt_t = _sample_column(dt, 'tc')

                n_freq = hp_t.shape[-1]
                freqs = (
                    torch.arange(
                        n_freq, device=device, dtype=real_dtype
                    ) * delta_f
                )

                phase = -2.0 * torch.pi * freqs * dt_t
                shift = torch.complex(torch.cos(phase), torch.sin(phase))
                h_det = (fp_t * hp_t + fc_t * hc_t) * shift
                if model.recalibration and detname in model.recalibration:
                    h_det = model.recalibration[detname].map_to_adjust(
                        h_det, **full_params
                    )
            else:
                def _numpy_data(value):
                    data = getattr(value, 'data', None)
                    if data is None or isinstance(data, memoryview):
                        data = value
                    return numpy.asarray(data)

                hp_arr = _numpy_data(hp)
                hc_arr = _numpy_data(hc)
                if hp_arr.ndim != 1 or hc_arr.ndim != 1:
                    raise ValueError(
                        "batched likelihood waveform parameters must be "
                        "scalar; only detector-frame extrinsics may be "
                        "batched"
                    )
                n_freq = hp_arr.shape[-1]
                freqs = numpy.arange(n_freq) * delta_f

                def _sample_column(value, name):
                    value_array = numpy.asarray(value)
                    if value_array.ndim == 0:
                        value_array = numpy.broadcast_to(
                            value_array, (batch_size,)
                        )
                    elif value_array.ndim == 1 and value_array.shape[0] == 1:
                        value_array = numpy.broadcast_to(
                            value_array, (batch_size,)
                        )
                    elif (
                        value_array.ndim != 1
                        or value_array.shape[0] != batch_size
                    ):
                        raise ValueError(
                            f"Batched detector-frame parameter {name!r} "
                            f"must be scalar or have length {batch_size}"
                        )
                    return value_array[:, None]

                fp_arr = _sample_column(fp, 'fplus')
                fc_arr = _sample_column(fc, 'fcross')
                dt_arr = _sample_column(dt, 'tc')
                phase = -2.0 * numpy.pi * freqs * dt_arr
                shift = numpy.exp(1j * phase)
                h_det = (fp_arr * hp_arr + fc_arr * hc_arr) * shift
                if model.recalibration and detname in model.recalibration:
                    h_det = model.recalibration[detname].map_to_adjust(
                        h_det, **full_params
                    )

            kmax = min(h_det.shape[-1], model._kmax[detname])
            kmin = model._kmin[detname]
            if kmin >= kmax:
                if _torch_tensor(h_det) is not None:
                    import torch

                    cplx_hd = torch.zeros(
                        h_det.shape[:-1], dtype=complex_dtype, device=device
                    )
                    hh = torch.zeros(
                        h_det.shape[:-1], dtype=real_dtype, device=device
                    )
                else:
                    cplx_hd = numpy.zeros(h_det.shape[:-1], dtype=complex)
                    hh = numpy.zeros(h_det.shape[:-1], dtype=float)
            else:
                slc = slice(kmin, kmax)
                w_slc = model._weight[detname][slc]
                d_slc = model._whitened_data[detname][slc]
                cplx_hd, hh = _fused_inner_hd_hh(
                    h_det[..., slc], d_slc, weight=w_slc
                )

            det_hd[detname] = cplx_hd
            det_hh[detname] = hh
            total_hd = cplx_hd if total_hd is None else total_hd + cplx_hd
            total_hh = hh if total_hh is None else total_hh + hh

        return total_hd, total_hh, det_hd, det_hh
    else:
        if gen is not None:
            wfs = gen.generate(**full_params)
        else:
            wfs = model.get_waveforms()

        total_hd = None
        total_hh = None
        det_hd = {}
        det_hh = {}
        for detname, h in wfs.items():
            kmax = min(h.shape[-1] if hasattr(h, 'shape') else len(h), model._kmax[detname])
            kmin = model._kmin[detname]
            if kmin >= kmax:
                cplx_hd = 0j
                hh = 0.0
            else:
                slc = slice(kmin, kmax)
                w_slc = model._weight[detname][slc]
                d_slc = model._whitened_data[detname][slc]
                cplx_hd, hh = _fused_inner_hd_hh(h[..., slc], d_slc, weight=w_slc)
            det_hd[detname] = cplx_hd
            det_hh[detname] = hh
            total_hd = cplx_hd if total_hd is None else total_hd + cplx_hd
            total_hh = hh if total_hh is None else total_hh + hh
        return total_hd, total_hh, det_hd, det_hh


#
# =============================================================================
#
#                               Support functions
#
# =============================================================================
#


def get_values_from_injection(cp, injection_file, update_cp=True):
    """Replaces all FROM_INJECTION values in a config file with the
    corresponding value from the injection.

    This looks for any options that start with ``FROM_INJECTION[:ARG]`` in
    a config file. It then replaces that value with the corresponding value
    from the injection file. An argument may be optionally provided, in which
    case the argument will be retrieved from the injection file. Functions of
    parameters in the injection file may be used; the syntax and functions
    available is the same as the ``--parameters`` argument in executables
    such as ``pycbc_inference_extract_samples``. If no ``ARG`` is provided,
    then the option name will try to be retrieved from the injection.

    For example,

    .. code-block:: ini

       mass1 = FROM_INJECTION

    will cause ``mass1`` to be retrieved from the injection file, while:

    .. code-block:: ini

       mass1 = FROM_INJECTION:'primary_mass(mass1, mass2)'

    will cause the larger of mass1 and mass2 to be retrieved from the injection
    file. Note that if spaces are in the argument, it must be encased in
    single quotes.

    The injection file may contain only one injection. Otherwise, a ValueError
    will be raised.

    Parameters
    ----------
    cp : ConfigParser
        The config file within which to replace values.
    injection_file : str or None
        The injection file to get values from. A ValueError will be raised
        if there are any ``FROM_INJECTION`` values in the config file, and
        injection file is None, or if there is more than one injection.
    update_cp : bool, optional
        Update the config parser with the replaced parameters. If False,
        will just retrieve the parameter values to update, without updating
        the config file. Default is True.

    Returns
    -------
    list
        The parameters that were replaced, as a tuple of section name, option,
        value.
    """
    lookfor = 'FROM_INJECTION'
    # figure out what parameters need to be set
    replace_params = []
    for sec in cp.sections():
        for opt in cp.options(sec):
            val = cp.get(sec, opt)
            splitvals = shlex.split(val)
            replace_this = []
            for ii, subval in enumerate(splitvals):
                if subval.startswith(lookfor):
                    # determine what we should retrieve from the injection
                    subval = subval.split(':', 1)
                    if len(subval) == 1:
                        subval = opt
                    else:
                        subval = subval[1]
                    replace_this.append((ii, subval))
            if replace_this:
                replace_params.append((sec, opt, splitvals, replace_this))
    if replace_params:
        # check that we have an injection file
        if injection_file is None:
            raise ValueError("One or values are set to {}, but no injection "
                             "file provided".format(lookfor))
        # load the injection file
        inj = InjectionSet(injection_file).table.view(type=FieldArray)
        # make sure there's only one injection provided
        if inj.size > 1:
            raise ValueError("One or more values are set to {}, but more than "
                             "one injection exists in the injection file."
                             .format(lookfor))
    # get the injection values to replace
    for ii, (sec, opt, splitvals, replace_this) in enumerate(replace_params):
        # replace the value in the shlex-splitted string with the value
        # from the injection
        for jj, arg in replace_this:
            splitvals[jj] = str(inj[arg][0])
        # now rejoin the string...
        # shlex will strip quotes around arguments; this can be problematic
        # when rejoining if the the argument had a space in it. In python 3.8
        # there is a shlex.join function which properly rejoins things taking
        # that into account. Since we need to continue to support earlier
        # versions of python, the following kludge tries to account for that.
        # If/when we drop support for all earlier versions of python, then the
        # following can just be replaced by:
        # replace_val = shlex.join(splitvals)
        for jj, arg in enumerate(splitvals):
            if ' ' in arg:
                arg = "'" + arg + "'"
                splitvals[jj] = arg
        replace_val = ' '.join(splitvals)
        replace_params[ii] = (sec, opt, replace_val)
    # replace in the config file
    if update_cp:
        for (sec, opt, replace_val) in replace_params:
            cp.set(sec, opt, replace_val)
    return replace_params


def create_waveform_generator(
        variable_params, data, waveform_transforms=None,
        recalibration=None, gates=None,
        generator_class=generator.FDomainDetFrameGenerator,
        **static_params):
    r"""Creates a waveform generator for use with a model.

    Parameters
    ----------
    variable_params : list of str
        The names of the parameters varied.
    data : dict
        Dictionary mapping detector names to either a
        :py:class:`<pycbc.types.TimeSeries TimeSeries>` or
        :py:class:`<pycbc.types.FrequencySeries FrequencySeries>`.
    waveform_transforms : list, optional
        The list of transforms applied to convert variable parameters into
        parameters that will be understood by the waveform generator.
    recalibration : dict, optional
        Dictionary mapping detector names to
        :py:class:`<pycbc.calibration.Recalibrate>` instances for
        recalibrating data.
    gates : dict of tuples, optional
        Dictionary of detectors -> tuples of specifying gate times. The
        sort of thing returned by :py:func:`pycbc.gate.gates_from_cli`.
    generator_class : detector-frame fdomain generator, optional
        Class to use for generating waveforms. Default is
        :py:class:`waveform.generator.FDomainDetFrameGenerator`.
    \**static_params :
        All other keyword arguments are passed as static parameters to the
        waveform generator.

    Returns
    -------
    pycbc.waveform.FDomainDetFrameGenerator
        A waveform generator for frequency domain generation.
    """
    # the waveform generator will get the variable_params + the output
    # of the waveform transforms, so we'll add them to the list of
    # parameters
    if waveform_transforms is not None:
        wfoutputs = set.union(*[t.outputs
                                for t in waveform_transforms])
    else:
        wfoutputs = set()
    variable_params = list(variable_params) + list(wfoutputs)
    # figure out what generator to use based on the approximant
    try:
        approximant = static_params['approximant']
    except KeyError:
        raise ValueError("no approximant provided in the static args")

    dm = static_params.get('preferred_domain', None)
    if isinstance(dm, str) and dm.lower() == 'none':
        dm = None

    gen_function = generator_class.select_rframe_generator(approximant, dm)
    # get data parameters; we'll just use one of the data to get the
    # values, then check that all the others are the same
    delta_f = None
    for d in data.values():
        if delta_f is None:
            delta_f = d.delta_f
            delta_t = d.delta_t
            start_time = d.start_time
        else:
            if not all([d.delta_f == delta_f, d.delta_t == delta_t,
                        d.start_time == start_time]):
                raise ValueError("data must all have the same delta_t, "
                                 "delta_f, and start_time")
    waveform_generator = generator_class(
        gen_function, epoch=start_time,
        variable_args=variable_params, detectors=list(data.keys()),
        delta_f=delta_f, delta_t=delta_t,
        recalib=recalibration, gates=gates,
        **static_params)
    return waveform_generator
