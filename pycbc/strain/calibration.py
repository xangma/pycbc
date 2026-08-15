# Copyright (C) 2018 Colm Talbot
#
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
""" Functions for adding calibration factors to waveform templates.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from abc import (ABCMeta, abstractmethod)


def _evaluate_spline(spline, sample_frequencies):
    """Evaluate a SciPy spline without copying a Torch grid to the host."""
    tensor = getattr(
        getattr(sample_frequencies, '_data', None), 'tensor', None
    )
    if tensor is None:
        return spline(sample_frequencies.numpy())

    import torch

    base_knots = np.asarray(spline.get_knots())
    coefficients = np.asarray(spline.get_coeffs())
    degree = len(coefficients) - len(base_knots) + 1
    knots = np.concatenate((
        np.repeat(base_knots[0], degree),
        base_knots,
        np.repeat(base_knots[-1], degree),
    ))

    samples = tensor.reshape(-1)
    knots = torch.as_tensor(
        knots, dtype=samples.dtype, device=samples.device
    )
    coefficients = torch.as_tensor(
        coefficients, dtype=samples.dtype, device=samples.device
    )
    spans = torch.searchsorted(knots, samples, right=True) - 1
    spans = spans.clamp(min=degree, max=len(coefficients) - 1)
    offsets = torch.arange(degree + 1, device=samples.device)
    values = coefficients[
        spans.unsqueeze(-1) - degree + offsets
    ].clone()

    # Vectorized de Boor evaluation of the fitted B-spline.
    for level in range(1, degree + 1):
        for index in range(degree, level - 1, -1):
            left = knots[spans - degree + index]
            right = knots[spans + 1 + index - level]
            weight = (samples - left) / (right - left)
            values[:, index] = (
                (1 - weight) * values[:, index - 1]
                + weight * values[:, index]
            )

    return values[:, degree].reshape(tensor.shape)


def _multiply_frequency_series(strain, correction):
    """Apply a correction while preserving the strain storage scheme."""
    tensor = getattr(getattr(strain, '_data', None), 'tensor', None)
    if tensor is None:
        return strain * correction

    from pycbc.types.array_torch import TorchArrayData
    return strain._return(TorchArrayData(tensor * correction))


def _apply_spline_calibration(strain, spline_points,
                              amplitude_parameters, phase_parameters):
    """Fit and apply the standard amplitude and phase calibration splines."""
    frequencies = strain.sample_frequencies
    amplitude_spline = UnivariateSpline(
        spline_points, amplitude_parameters
    )
    delta_amplitude = _evaluate_spline(amplitude_spline, frequencies)
    phase_spline = UnivariateSpline(spline_points, phase_parameters)
    delta_phase = _evaluate_spline(phase_spline, frequencies)

    correction = (
        (1.0 + delta_amplitude)
        * (2.0 + 1j * delta_phase)
        / (2.0 - 1j * delta_phase)
    )
    return _multiply_frequency_series(strain, correction)


class Recalibrate(metaclass=ABCMeta):
    name = None

    def __init__(self, ifo_name):
        self.ifo_name = ifo_name
        self.params = dict()

    @abstractmethod
    def apply_calibration(self, strain):
        """Apply calibration model

        This method should be overwritten by subclasses

        Parameters
        ----------
        strain : FrequencySeries
            The strain to be recalibrated.

        Return
        ------
        strain_adjusted : FrequencySeries
            The recalibrated strain.
        """
        return

    def map_to_adjust(self, strain, prefix='recalib_', **params):
        """Map an input dictionary of sampling parameters to the
        adjust_strain function by filtering the dictionary for the
        calibration parameters, then calling adjust_strain.

        Parameters
        ----------
        strain : FrequencySeries
            The strain to be recalibrated.
        prefix: str
            Prefix for calibration parameter names
        params : dict
            Dictionary of sampling parameters which includes
            calibration parameters.
        Return
        ------
        strain_adjusted : FrequencySeries
            The recalibrated strain.
        """

        self.params.update({
            key[len(prefix):]: params[key]
            for key in params if prefix in key and self.ifo_name in key})

        strain_adjusted = self.apply_calibration(strain)

        return strain_adjusted

    @classmethod
    def from_config(cls, cp, ifo, section):
        """Read a config file to get calibration options and transfer
        functions which will be used to intialize the model.

        Parameters
        ----------
        cp : WorkflowConfigParser
            An open config file.
        ifo : string
            The detector (H1, L1) for which the calibration model will
            be loaded.
        section : string
            The section name in the config file from which to retrieve
            the calibration options.
        Return
        ------
        instance
            An instance of the class.
        """
        all_params = dict(cp.items(section))
        params = {key[len(ifo)+1:]: all_params[key]
                  for key in all_params if ifo.lower() in key}
        model = params.pop('model')
        params['ifo_name'] = ifo.lower()

        return all_models[model](**params)


class CubicSpline(Recalibrate):
    name = 'cubic_spline'

    def __init__(self, minimum_frequency, maximum_frequency, n_points,
                 ifo_name):
        """
        Cubic spline recalibration

        see https://dcc.ligo.org/LIGO-T1400682/public

        This assumes the spline points follow
        np.logspace(np.log(minimum_frequency), np.log(maximum_frequency),
                    n_points)

        Parameters
        ----------
        minimum_frequency: float
            minimum frequency of spline points
        maximum_frequency: float
            maximum frequency of spline points
        n_points: int
            number of spline points
        """
        Recalibrate.__init__(self, ifo_name=ifo_name)
        minimum_frequency = float(minimum_frequency)
        maximum_frequency = float(maximum_frequency)
        n_points = int(n_points)
        if n_points < 4:
            raise ValueError(
                'Use at least 4 spline points for calibration model')
        self.n_points = n_points
        self.spline_points = np.logspace(np.log10(minimum_frequency),
                                         np.log10(maximum_frequency), n_points)

    def apply_calibration(self, strain):
        """Apply calibration model

        This applies cubic spline calibration to the strain.

        Parameters
        ----------
        strain : FrequencySeries
            The strain to be recalibrated.

        Return
        ------
        strain_adjusted : FrequencySeries
            The recalibrated strain.
        """
        amplitude_parameters =\
            [self.params['amplitude_{}_{}'.format(self.ifo_name, ii)]
             for ii in range(self.n_points)]
        phase_parameters =\
            [self.params['phase_{}_{}'.format(self.ifo_name, ii)]
             for ii in range(self.n_points)]
        return _apply_spline_calibration(
            strain, self.spline_points,
            amplitude_parameters, phase_parameters
        )


all_models = {
    CubicSpline.name: CubicSpline
}
