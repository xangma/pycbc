# Copyright (C) 2020  Collin Capano, Alex Nitz
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

"""Provides functions and utilities for generating waveforms mode-by-mode.
"""

from string import Formatter
from pycbc import lal_compat as lal

from pycbc import pnutils
from pycbc import scheme as _scheme
from pycbc.types import (TimeSeries, FrequencySeries)
from pycbc.constants import MSUN_SI, PC_SI
from .waveform import (
    _check_lal_pars,
    _lal_output_for_active_scheme,
    check_args,
    lalsimulation,
    props,
)
from . import parameters
from .torch_waveform_registry import (
    native_approximants,
    try_torch_native_waveform,
)


def _formatdocstr(docstr):
    """Utility for formatting docstrings with parameter information.
    """
    return docstr.format(
        **{_p[1]: getattr(parameters, _p[1]).docstr(
            prefix="    ", include_label=False).lstrip(' ')
           for _p in Formatter().parse(docstr) if _p[1] is not None
           })


def _formatdocstrlist(docstr, paramlist, skip_params=None):
    """Utility for formatting docstrings with parameter information.
    """
    if skip_params is None:
        skip_params = []
    pl = '\n'.join([_p.docstr(prefix="    ", include_label=False)
                    for _p in paramlist if _p not in skip_params])
    return docstr.format(params=pl)


def sum_modes(hlms, inclination, phi):
    """Applies spherical harmonics and sums modes to produce a plus and cross
    polarization.

    Parameters
    ----------
    hlms : dict
        Dictionary of ``(l, m)`` -> complex ``hlm``. The ``hlm`` may be a
        complex number or array, or complex ``TimeSeries``. All modes in the
        dictionary will be summed.
    inclination : float
        The inclination to use.
    phi : float
        The phase to use.

    Returns
    -------
    complex float or array
        The plus and cross polarization as a complex number. The real part
        gives the plus, the negative imaginary part the cross.
    """
    if hlms:
        first_hlm = next(iter(hlms.values()))
        first_tensor = getattr(
            getattr(first_hlm, "_data", None), "tensor", None
        )
        if isinstance(first_hlm, FrequencySeries) and first_tensor is not None:
            modes = list(hlms.keys())
            if all(
                2 <= ell <= 4 and abs(emm) <= ell
                for ell, emm in modes
            ):
                import torch
                from ._spherical_harmonics_torch import (
                    selected_spin_minus_two_spherical_harmonics,
                )
                from pycbc.types.array_torch import TorchArrayData

                dtype = first_tensor.real.dtype
                device = first_tensor.device
                ylm_dict = selected_spin_minus_two_spherical_harmonics(
                    inclination, phi, modes, dtype=dtype, device=device
                )
                ylm_vector = torch.stack([ylm_dict[m] for m in modes])
                hlm_matrix = torch.stack(
                    [hlms[m]._data.tensor for m in modes], dim=0
                )
                res_tensor = torch.matmul(ylm_vector, hlm_matrix)
                return FrequencySeries(
                    TorchArrayData(res_tensor),
                    delta_f=first_hlm.delta_f,
                    epoch=first_hlm.epoch,
                    copy=False,
                )

    out = None
    for mode in hlms:
        ell, m = mode
        hlm = hlms[ell, m]
        ylm = lal.SpinWeightedSphericalHarmonic(
            inclination, phi, -2, ell, m
        )
        if out is None:
            out = ylm * hlm
        else:
            out += ylm * hlm
    return out


def default_modes(approximant):
    """Returns the default modes for the given approximant.
    """
    # FIXME: this should be replaced to a call to a lalsimulation function,
    # whenever that's added
    if approximant in ['IMRPhenomXPHM', 'IMRPhenomXHM']:
        # according to arXiv:2004.06503
        ma = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)]
        # add the -m modes
        ma += [(ell, -m) for ell, m in ma]
    elif approximant in ['IMRPhenomPv3HM', 'IMRPhenomHM']:
        # according to arXiv:1911.06050
        ma = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)]
        # add the -m modes
        ma += [(ell, -m) for ell, m in ma]
    elif approximant.startswith('NRSur7dq4'):
        # according to arXiv:1905.09300
        ma = [
            (ell, m)
            for ell in [2, 3, 4]
            for m in range(-ell, ell + 1)
        ]
    elif approximant.startswith('NRHybSur3dq8'):
        # according to arXiv:1812.07865
        ma = [(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2),
              (3, 3), (4, 2), (4, 3), (4, 4), (5, 5)]
    else:
        raise ValueError("I don't know what the default modes are for "
                         "approximant {}, sorry!".format(approximant))
    return ma


def get_glm(l, m, theta):  # noqa: E741 - preserve the public keyword
    r"""The maginitude of the :math:`{}_{-2}Y_{\ell m}`.

    The spin-weighted spherical harmonics can be written as
    :math:`{}_{-2}Y_{\ell m}(\theta, \phi) = g_{\ell m}(\theta)e^{i m \phi}`.
    This returns the `g_{\ell m}(\theta)` part. Note that this is real.

    Parameters
    ----------
    l : int
        The :math:`\ell` index of the spherical harmonic.
    m : int
        The :math:`m` index of the spherical harmonic.
    theta : float
        The polar angle (in radians).

    Returns
    -------
    float :
        The amplitude of the harmonic at the given polar angle.
    """
    return lal.SpinWeightedSphericalHarmonic(theta, 0.0, -2, l, m).real


def get_nrsur_modes(**params):
    """Generates NRSurrogate waveform mode-by-mode.

    All waveform parameters should be provided as keyword arguments.
    Recognized parameters are listed below. Unrecognized arguments are ignored.

    Parameters
    ----------
    template: object
        An object that has attached properties. This can be used to substitute
        for keyword arguments. A common example would be a row in an xml table.
    approximant : str
        The approximant to generate. Must be one of the ``NRSur*`` models.
    {delta_t}
    {mass1}
    {mass2}
    {spin1x}
    {spin1y}
    {spin1z}
    {spin2x}
    {spin2y}
    {spin2z}
    {f_lower}
    {f_ref}
    {distance}
    {mode_array}

    Returns
    -------
    dict :
        Dictionary of ``(l, m)`` -> ``(h_+, -h_x)`` ``TimeSeries``.
    """
    laldict = _check_lal_pars(params)
    ret = lalsimulation.SimInspiralPrecessingNRSurModes(
        params['delta_t'],
        params['mass1']*MSUN_SI,
        params['mass2']*MSUN_SI,
        params['spin1x'], params['spin1y'], params['spin1z'],
        params['spin2x'], params['spin2y'], params['spin2z'],
        params['f_lower'], params['f_ref'],
        params['distance']*1e6*PC_SI, laldict,
        getattr(lalsimulation, params['approximant'])
    )
    hlms = {}
    while ret:
        hlm = TimeSeries(
            _lal_output_for_active_scheme(ret.mode.data.data),
            delta_t=ret.mode.deltaT,
            epoch=ret.mode.epoch,
        )
        hlms[ret.l, ret.m] = (hlm.real(), hlm.imag())
        ret = ret.next
    return hlms

def get_nrhybsur_modes(**params):
    """Generates NRHybSur3dq8 waveform mode-by-mode.

    All waveform parameters should be provided as keyword arguments.
    Recognized parameters are listed below. Unrecognized arguments are ignored.

    Parameters
    ----------
    template: object
        An object that has attached properties. This can be used to substitute
        for keyword arguments. A common example would be a row in an xml table.
    approximant : str
        The approximant to generate. Must be one of the ``NRHyb*`` models.
    {delta_t}
    {mass1}
    {mass2}
    {spin1z}
    {spin2z}
    {f_lower}
    {f_ref}
    {distance}
    {mode_array}

    Returns
    -------
    dict :
        Dictionary of ``(l, m)`` -> ``(h_+, -h_x)`` ``TimeSeries``.
    """
    laldict = _check_lal_pars(params)
    ret = lalsimulation.SimIMRNRHybSur3dq8Modes(
        params['delta_t'],
        params['mass1']*MSUN_SI,
        params['mass2']*MSUN_SI,
        params['spin1z'],
        params['spin2z'],
        params['f_lower'], params['f_ref'],
        params['distance']*1e6*PC_SI, laldict
    )
    hlms = {}
    while ret:
        hlm = TimeSeries(
            _lal_output_for_active_scheme(ret.mode.data.data),
            delta_t=ret.mode.deltaT,
            epoch=ret.mode.epoch,
        )
        hlms[ret.l, ret.m] = (hlm.real(), hlm.imag())
        ret = ret.next
    return hlms


get_nrsur_modes.__doc__ = _formatdocstr(get_nrsur_modes.__doc__)
get_nrhybsur_modes.__doc__ = _formatdocstr(get_nrhybsur_modes.__doc__)

def get_lalsimulation_approximant(approximant):
    import lalsimulation as ls
    return {
        'EOBNRv2': ls.EOBNRv2,
        'EOBNRv2HM': ls.EOBNRv2HM,
        'IMRPhenomTPHM': ls.IMRPhenomTPHM,
        'NRSur7dq2': ls.NRSur7dq2,
        'NRSur7dq4': ls.NRSur7dq4,
        'NRHybSur3dq8': ls.NRHybSur3dq8,
        'pSEOBNRv4HM_PA': ls.pSEOBNRv4HM_PA,
        'SEOBNRv4HM_PA': ls.SEOBNRv4HM_PA,
        'SEOBNRv4P': ls.SEOBNRv4P,
        'SEOBNRv4PHM': ls.SEOBNRv4PHM,
        'SpinTaylorT1': ls.SpinTaylorT1,
        'SpinTaylorT4': ls.SpinTaylorT4,
        'SpinTaylorT5': ls.SpinTaylorT5,
        'TaylorT1': ls.TaylorT1,
        'TaylorT2': ls.TaylorT2,
        'TaylorT3': ls.TaylorT3,
        'TaylorT4': ls.TaylorT4,
        }[approximant]

def get_lalsimulation_modes(**params):
    """Generates approximant waveform mode-by-mode.

    All waveform parameters should be provided as keyword arguments.
    Recognized parameters are listed below. Unrecognized arguments are ignored.

    Parameters
    ----------
    template: object
        An object that has attached properties. This can be used to substitute
        for keyword arguments. A common example would be a row in an xml table.
    approximant : str
        The approximant to generate. Must be available in ``lalsimulation``.
    {delta_t}
    {mass1}
    {mass2}
    {spin1x}
    {spin1y}
    {spin1z}
    {spin2x}
    {spin2y}
    {spin2z}
    {f_lower}
    {f_ref}
    {distance}
    {mode_array}
    {ell_max}
    {approximant}

    Returns
    -------
    dict :
        Dictionary of ``(l, m)`` -> ``(h_+, -h_x)`` ``TimeSeries``.
    """
    ell_max = 5
    if 'ell_max' in params:
        ell_max = params['ell_max']
    laldict = _check_lal_pars(params)
    ret = lalsimulation.SimInspiralChooseTDModes(
        params['coa_phase'],
        params['delta_t'],
        params['mass1']*MSUN_SI,
        params['mass2']*MSUN_SI,
        params['spin1x'],
        params['spin1y'],
        params['spin1z'],
        params['spin2x'],
        params['spin2y'],
        params['spin2z'],
        params['f_lower'], params['f_ref'],
        params['distance']*1e6*PC_SI, laldict,
        ell_max,
        get_lalsimulation_approximant(params['approximant'])
    )
    hlms = {}
    while ret:
        hlm = TimeSeries(
            _lal_output_for_active_scheme(ret.mode.data.data),
            delta_t=ret.mode.deltaT,
            epoch=ret.mode.epoch,
        )
        hlms[(ret.l, ret.m)] = (hlm.real(), hlm.imag())
        ret = ret.next
    return hlms

def get_imrphenomxh_modes(**params):
    """Generates ``IMRPhenomXHM`` waveforms mode-by-mode. """
    approx = params['approximant']
    if not approx.startswith('IMRPhenomX'):
        raise ValueError("unsupported approximant")
    mode_array = params.pop('mode_array', None)
    if mode_array is None:
        mode_array = default_modes(approx)
    if 'f_final' not in params:
        # setting to 0 will default to ringdown frequency
        params['f_final'] = 0.
    hlms = {}
    for (ell, m) in mode_array:
        params['mode_array'] = [(ell, m)]
        laldict = _check_lal_pars(params)
        hlm = lalsimulation.SimIMRPhenomXHMGenerateFDOneMode(
            float(pnutils.solar_mass_to_kg(params['mass1'])),
            float(pnutils.solar_mass_to_kg(params['mass2'])),
            float(params['spin1z']),
            float(params['spin2z']), ell, m,
            pnutils.megaparsecs_to_meters(float(params['distance'])),
            params['f_lower'], params['f_final'], params['delta_f'],
            params['coa_phase'], params['f_ref'],
            laldict)
        hlm = FrequencySeries(
            _lal_output_for_active_scheme(hlm.data.data),
            delta_f=hlm.deltaF,
            epoch=hlm.epoch,
        )
        # Plus, cross strains without Y_lm.
        # (-1)**(l) factor ALREADY included in FDOneMode
        hplm = 0.5 * hlm  # Plus strain
        hclm = 0.5j * hlm  # Cross strain
        if m > 0:
            hclm *= -1
        hlms[ell, m] = (hplm, hclm)
    return hlms


def get_imrphenomhm_modes(**params):
    """Generate ``IMRPhenomHM`` frequency-domain modes with LAL."""
    import lalsimulation as ls

    requested = params.get("mode_array")
    if requested is None:
        requested = default_modes("IMRPhenomHM")
    requested = list(dict.fromkeys(tuple(mode) for mode in requested))
    if not requested:
        return {}

    families = list(
        dict.fromkeys((ell, abs(emm)) for ell, emm in requested)
    )
    lal_params = dict(params)
    lal_params["mode_array"] = families
    laldict = _check_lal_pars(lal_params)
    bounds = lal.CreateREAL8Vector(2)
    bounds.data[:] = (
        float(params["f_lower"]),
        float(params.get("f_final", 0.0)),
    )
    node = ls.SimIMRPhenomHMGethlmModes(
        bounds,
        float(pnutils.solar_mass_to_kg(params["mass1"])),
        float(pnutils.solar_mass_to_kg(params["mass2"])),
        float(params["spin1x"]),
        float(params["spin1y"]),
        float(params["spin1z"]),
        float(params["spin2x"]),
        float(params["spin2y"]),
        float(params["spin2z"]),
        float(params["coa_phase"]),
        float(params["delta_f"]),
        float(params["f_ref"]),
        laldict,
    )

    positive_modes = {}
    while node:
        positive_modes[node.l, node.m] = FrequencySeries(
            _lal_output_for_active_scheme(node.mode.data.data),
            delta_f=node.mode.deltaF,
            epoch=node.mode.epoch,
        )
        node = node.next

    total_mass = float(params["mass1"]) + float(params["mass2"])
    amplitude_scale = (
        total_mass
        * lal.MRSUN_SI
        * total_mass
        * lal.MTSUN_SI
        / pnutils.megaparsecs_to_meters(float(params["distance"]))
    )
    modes = {}
    for ell, emm in requested:
        hlm = amplitude_scale * positive_modes[ell, abs(emm)]
        if emm < 0:
            hlm *= (-1) ** ell
        ulm = 0.5 * hlm
        vlm = (0.5j if emm > 0 else -0.5j) * hlm
        modes[ell, emm] = (ulm, vlm)
    return modes


_mode_waveform_td = {'EOBNRv2': get_lalsimulation_modes,
                     'EOBNRv2HM': get_lalsimulation_modes,
                     'IMRPhenomTPHM': get_lalsimulation_modes,
                     'NRSur7dq2': get_lalsimulation_modes,
                     'NRSur7dq4': get_nrsur_modes,
                     'NRHybSur3dq8': get_nrhybsur_modes,
                     'pSEOBNRv4HM_PA': get_lalsimulation_modes,
                     'SEOBNRv4HM_PA': get_lalsimulation_modes,
                     'SEOBNRv4P': get_lalsimulation_modes,
                     'SEOBNRv4PHM': get_lalsimulation_modes,
                     'SpinTaylorT1': get_lalsimulation_modes,
                     'SpinTaylorT4': get_lalsimulation_modes,
                     'SpinTaylorT5': get_lalsimulation_modes,
                     'TaylorT1': get_lalsimulation_modes,
                     'TaylorT2': get_lalsimulation_modes,
                     'TaylorT3': get_lalsimulation_modes,
                     'TaylorT4': get_lalsimulation_modes,
                     }
_mode_waveform_fd = {
    'IMRPhenomXHM': get_imrphenomxh_modes,
    'IMRPhenomHM': get_imrphenomhm_modes,
}
# 'IMRPhenomXPHM':get_imrphenomhm_modes needs to be implemented
# LAL function do not split strain mode by mode

def fd_waveform_mode_approximants(scheme=None):
    """Frequency domain approximants that will return separate modes."""
    if scheme is None:
        scheme = _scheme.mgr.state
    approximants = set(_mode_waveform_fd)
    if isinstance(scheme, _scheme.TorchScheme):
        approximants.update(native_approximants("fd_modes"))
    return sorted(approximants)


def td_waveform_mode_approximants(scheme=None):
    """Time domain approximants that will return separate modes."""
    if scheme is None:
        scheme = _scheme.mgr.state
    approximants = set(_mode_waveform_td)
    if isinstance(scheme, _scheme.TorchScheme):
        approximants.update(native_approximants("td_modes"))
    return sorted(approximants)


def get_fd_waveform_modes(template=None, **kwargs):
    r"""Generates frequency domain waveforms, but does not sum over the modes.

    The returned values are the frequency-domain equivalents of the real and
    imaginary parts of the complex :math:`\mathfrak{{h}}_{{\ell m}}(t)` time
    series. In other words, the returned values are equivalent to the Fourier
    Transform of the two time series returned by
    :py:func:`get_td_waveform_modes`; see that function for more details.

    Parameters
    ----------
    template: object
        An object that has attached properties. This can be used to subsitute
        for keyword arguments.

    {params}

    Returns
    -------
    modes : dict
        Dictionary mapping ``(l, m)`` mode tuples to ``(u_lm, v_lm)`` pairs.
        Each pair contains the Fourier transforms of the real and imaginary
        parts of the hlm time series, respectively, as
        :py:class:`pycbc.types.FrequencySeries` instances.
    """
    params = props(template, **kwargs)
    required = parameters.fd_required
    check_args(params, required)
    apprx = params['approximant']
    if isinstance(_scheme.mgr.state, _scheme.TorchScheme):
        native_modes = try_torch_native_waveform(
            "fd_modes",
            params,
        )
        if native_modes is not None:
            return native_modes
    if apprx not in _mode_waveform_fd:
        raise ValueError("I don't support approximant {}, sorry"
                         .format(apprx))
    return _mode_waveform_fd[apprx](**params)


get_fd_waveform_modes.__doc__ = _formatdocstrlist(
    get_fd_waveform_modes.__doc__, parameters.fd_waveform_params,
    skip_params=['inclination', 'long_asc_nodes'])


def get_td_waveform_modes(template=None, **kwargs):
    r"""Generates time domain waveforms, but does not sum over the modes.

    The returned values are the real and imaginary parts of the complex
    :math:`\mathfrak{{h}}_{{\ell m}}(t)`. These are defined such that the plus
    and cross polarizations :math:`h_{{+,\times}}` are:

    .. math::

       h_{{+,\times}}(\theta, \phi; t) = (\Re, -\Im) \sum_{{\ell m}}
        {{}}_{{-2}}Y_{{\ell m}}(\theta, \phi) \mathfrak{{h}}_{{\ell m}}(t).


    Parameters
    ----------
    template: object
        An object that has attached properties. This can be used to subsitute
        for keyword arguments.

    {params}

    Returns
    -------
    hlms : dict
        Dictionary mapping each mode tuple to a pair containing the real and
        imaginary parts of the mode as
        :py:class:`pycbc.types.TimeSeries` objects.
    """
    params = props(template, **kwargs)
    required = parameters.td_required
    check_args(params, required)
    apprx = params['approximant']
    if isinstance(_scheme.mgr.state, _scheme.TorchScheme):
        native_modes = try_torch_native_waveform(
            "td_modes",
            params,
        )
        if native_modes is not None:
            return native_modes
    if apprx not in _mode_waveform_td:
        raise ValueError("I don't support approximant {}, sorry"
                         .format(apprx))
    return _mode_waveform_td[apprx](**params)


get_td_waveform_modes.__doc__ = _formatdocstrlist(
    get_td_waveform_modes.__doc__, parameters.td_waveform_params,
    skip_params=['inclination', 'coa_phase'])
