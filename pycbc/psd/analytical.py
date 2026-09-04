# Copyright (C) 2012-2016 Alex Nitz, Tito Dal Canton, Leo Singer
#               2022 Shichao Wu
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
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Provides reference PSDs from LALSimulation and pycbc.psd.analytical_space.

More information about how to use these ground-based detectors' PSD can be
found in the guide about :ref:`Analytic PSDs from lalsimulation`. For
space-borne ones, see `pycbc.psd.analytical_space` module.
"""

import numbers

import lal
import numpy

from pycbc import scheme as _scheme
from pycbc.psd.analytical_space import (
    analytical_psd_lisa_tdi_AE,
    analytical_psd_lisa_tdi_AE_confusion,
    analytical_psd_lisa_tdi_T,
    analytical_psd_lisa_tdi_XYZ,
    analytical_psd_taiji_tdi_AE,
    analytical_psd_taiji_tdi_AE_confusion,
    analytical_psd_taiji_tdi_T,
    analytical_psd_taiji_tdi_XYZ,
    analytical_psd_tianqin_tdi_AE,
    analytical_psd_tianqin_tdi_AE_confusion,
    analytical_psd_tianqin_tdi_T,
    analytical_psd_tianqin_tdi_XYZ,
    sh_transformed_psd_lisa_tdi_XYZ,
)
from pycbc.types import FrequencySeries
from pycbc.types.backend import wrap_backend_array

# build a list of usable PSD functions from lalsimulation
_name_prefix = "SimNoisePSD"
_name_suffix = "Ptr"
_name_blacklist = ("FromFile", "MirrorTherm", "Quantum", "Seismic", "Shot", "SuspTherm")
_psd_list = []

try:
    import lalsimulation

    for _name in lalsimulation.__dict__:
        if (
            _name != _name_prefix
            and _name.startswith(_name_prefix)
            and not _name.endswith(_name_suffix)
        ):
            _name = _name[len(_name_prefix) :]
            if _name not in _name_blacklist:
                _psd_list.append(_name)
except ImportError:
    pass

_psd_list = sorted(_psd_list)

# Torch-native models are a PyCBC capability, rather than a property of the
# installed LALSimulation module.  Keep their registry available when
# LALSimulation is absent, while retaining the historical optional-Torch
# import behavior.
try:
    from pycbc.psd.analytical_torch import TORCH_ANALYTICAL_PSD_MODELS
except (ImportError, OSError):  # pragma: no cover - optional Torch runtime
    _torch_psd_list = []
else:
    _torch_psd_list = sorted(TORCH_ANALYTICAL_PSD_MODELS)

# Add convenience functions for every ground-detector model PyCBC can expose.
for _name in sorted(set(_psd_list) | set(_torch_psd_list)):
    exec(
        """
def %s(length, delta_f, low_freq_cutoff):
    \"\"\"Return a FrequencySeries containing the %s analytical PSD.
    \"\"\"
    return from_string("%s", length, delta_f, low_freq_cutoff)
"""
        % (_name, _name, _name)
    )


def get_psd_model_list():
    """Returns a list of available reference PSD functions.

    Returns
    -------
    list
        Returns a list of names of reference PSD functions.
    """
    ground_models = set(get_lalsim_psd_list()) | set(get_torch_psd_list())
    return sorted(ground_models) + get_pycbc_psd_list()


def get_lalsim_psd_list():
    """Return a list of available reference PSD functions from LALSimulation."""
    return _psd_list


def get_torch_psd_list():
    """Return analytical PSD models implemented by the Torch backend."""
    return _torch_psd_list


def _swig_real8_compatible(value):
    """Return whether SWIG accepts ``value`` for a LAL ``REAL8`` argument."""
    if not isinstance(
        value,
        (int, float, numpy.integer, numpy.floating),
    ):
        return False
    try:
        float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def get_pycbc_psd_list():
    """Return a list of available reference PSD functions coded in PyCBC.

    Returns
    -------
    list
        Returns a list of names of all reference PSD functions coded in PyCBC.
    """
    pycbc_analytical_psd_list = pycbc_analytical_psds.keys()
    pycbc_analytical_psd_list = sorted(pycbc_analytical_psd_list)
    return pycbc_analytical_psd_list


def from_string(psd_name, length, delta_f, low_freq_cutoff, **kwargs):
    """Generate a frequency series containing a LALSimulation or
    built-in space-borne detectors' PSD specified by name.

    Parameters
    ----------
    psd_name : string
        PSD name as found in LALSimulation (minus the SimNoisePSD prefix)
        or pycbc.psd.analytical_space.
    length : int
        Length of the frequency series in samples.
    delta_f : float
        Frequency resolution of the frequency series, in hertz.
    low_freq_cutoff : float
        Frequencies below this value (in hertz) are set to zero.
    **kwargs :
        All other keyword arguments are passed to the PSD model.

    Returns
    -------
    psd : FrequencySeries
        The generated frequency series.
    """

    ground_models = set(get_lalsim_psd_list()) | set(get_torch_psd_list())

    # check if valid PSD model
    if psd_name not in get_psd_model_list():
        if lalsimulation is None:
            detail = (
                " It is not a built-in Torch model, and lalsimulation is not installed."
            )
        else:
            detail = ""
        raise ValueError(
            psd_name + " not found among analytical PSD functions." + detail
        )

    # make sure length has the right type for CreateREAL8FrequencySeries
    if not isinstance(length, numbers.Integral) or length <= 0:
        raise TypeError("length must be a positive integer")
    length = int(length)

    # if PSD model is provided by LALSimulation or the native Torch backend
    if psd_name in ground_models:
        state = _scheme.mgr.state
        native_torch = False
        if (
            isinstance(state, _scheme.TorchScheme)
            and state.torch_device.type in ("cpu", "cuda")
            and _swig_real8_compatible(delta_f)
        ):
            from pycbc.psd.analytical_torch import (
                DATA_FILE_TORCH_ANALYTICAL_MODELS,
                _DataFileNativeUnsupported,
                analytical_psd,
            )

            native_torch = psd_name in get_torch_psd_list()
            if psd_name in DATA_FILE_TORCH_ANALYTICAL_MODELS:
                native_torch = (
                    native_torch
                    and _swig_real8_compatible(low_freq_cutoff)
                    and numpy.isfinite(float(delta_f))
                    and float(delta_f) > 0.0
                    and numpy.isfinite(float(low_freq_cutoff))
                )
        if native_torch:
            try:
                psd = analytical_psd(
                    psd_name,
                    length,
                    delta_f,
                    state.torch_device,
                    low_freq_cutoff=low_freq_cutoff,
                )
            except _DataFileNativeUnsupported:
                native_torch = False
        if not native_torch:
            if lalsimulation is None:
                raise ImportError(
                    f"Analytical PSD {psd_name} requires lalsimulation "
                    "outside its supported Torch-native CPU/CUDA path. "
                    "Install LALSimulation or activate TorchScheme."
                )
            lalseries = lal.CreateREAL8FrequencySeries(
                "", lal.LIGOTimeGPS(0), 0, delta_f, lal.DimensionlessUnit, length
            )
            try:
                func = lalsimulation.__dict__[_name_prefix + psd_name + _name_suffix]
            except KeyError:
                func = lalsimulation.__dict__[_name_prefix + psd_name]
                func(lalseries, low_freq_cutoff)
            else:
                lalsimulation.SimNoisePSD(lalseries, 0, func)
            psd = FrequencySeries(lalseries.data.data, delta_f=delta_f)

    # if PSD model is coded in PyCBC
    else:
        func = pycbc_analytical_psds[psd_name]
        psd = func(length, delta_f, low_freq_cutoff, **kwargs)

    # zero-out content below low-frequency cutoff
    kmin = int(low_freq_cutoff / delta_f)
    psd.data[:kmin] = 0

    return psd


def flat_unity(length, delta_f, low_freq_cutoff):
    """Returns a FrequencySeries of ones above the low_frequency_cutoff.

    Parameters
    ----------
    length : int
        Length of output Frequencyseries.
    delta_f : float
        Frequency step for output FrequencySeries, in hertz.
    low_freq_cutoff : int
        Low-frequency cutoff for output FrequencySeries, in hertz.

    Returns
    -------
    FrequencySeries
        Returns a FrequencySeries containing the unity PSD model.
    """
    state = _scheme.mgr.state
    if isinstance(state, _scheme.TorchScheme):
        import torch

        dtype = torch.float32 if state.torch_device.type == "mps" else torch.float64
        values = torch.ones(length, dtype=dtype, device=state.torch_device)
        values[: int(low_freq_cutoff / delta_f)] = 0
        return FrequencySeries(
            wrap_backend_array(values),
            delta_f=delta_f,
            copy=False,
        )

    fseries = FrequencySeries(numpy.ones(length), delta_f=delta_f)
    kmin = int(low_freq_cutoff / fseries.delta_f)
    fseries.data[:kmin] = 0
    return fseries


# dict of analytical PSDs coded in PyCBC
pycbc_analytical_psds = {
    "flat_unity": flat_unity,
    "analytical_psd_lisa_tdi_XYZ": analytical_psd_lisa_tdi_XYZ,
    "analytical_psd_lisa_tdi_AE": analytical_psd_lisa_tdi_AE,
    "analytical_psd_lisa_tdi_T": analytical_psd_lisa_tdi_T,
    "sh_transformed_psd_lisa_tdi_XYZ": sh_transformed_psd_lisa_tdi_XYZ,
    "analytical_psd_lisa_tdi_AE_confusion": analytical_psd_lisa_tdi_AE_confusion,
    "analytical_psd_tianqin_tdi_XYZ": analytical_psd_tianqin_tdi_XYZ,
    "analytical_psd_tianqin_tdi_AE": analytical_psd_tianqin_tdi_AE,
    "analytical_psd_tianqin_tdi_T": analytical_psd_tianqin_tdi_T,
    "analytical_psd_tianqin_tdi_AE_confusion": analytical_psd_tianqin_tdi_AE_confusion,
    "analytical_psd_taiji_tdi_XYZ": analytical_psd_taiji_tdi_XYZ,
    "analytical_psd_taiji_tdi_AE": analytical_psd_taiji_tdi_AE,
    "analytical_psd_taiji_tdi_T": analytical_psd_taiji_tdi_T,
    "analytical_psd_taiji_tdi_AE_confusion": analytical_psd_taiji_tdi_AE_confusion,
}
