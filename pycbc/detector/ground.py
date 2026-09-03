# -*- coding: UTF-8 -*-

# Copyright (C) 2012  Alex Nitz
#
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


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
"""This module provides utilities for calculating detector responses and timing
between ground-based observatories.
"""
import os
import logging
import numpy as np
from numpy import cos, sin

import lal
from astropy import constants, coordinates, units
from astropy.coordinates.matrix_utilities import rotation_matrix
from astropy.units.si import sday, meter

import pycbc.scheme as _scheme
import pycbc.libutils
from pycbc.types import TimeSeries
from pycbc.types.config import InterpolatingConfigParser
from pycbc.time import gmst_accurate

logger = logging.getLogger('pycbc.detector')

# Response functions are modelled after those in lalsuite and as also
# presented in https://arxiv.org/pdf/gr-qc/0008066.pdf

def _to_lal_real8_time_series(series):
    """Copy a PyCBC series to the host representation required by LAL."""
    lal_series = lal.CreateREAL8TimeSeries(
        '',
        series.start_time,
        0,
        series.delta_t,
        lal.SecondUnit,
        len(series),
    )
    lal_series.data.data[:] = series.numpy()
    return lal_series



def _scalar_antenna_pattern_and_time_delay(
        detector, right_ascension, declination, t_gps):
    """Return the built-in tensor response and geocentric delay together.

    This private helper is for the tightly guarded scalar inference path.  It
    evaluates the same long-wavelength, zero-polarization expressions as
    :meth:`Detector.antenna_pattern` and
    :meth:`Detector.time_delay_from_earth_center`, but shares their GMST and
    trigonometric terms.  The explicit scalar operations retain the public
    methods' object-dot accumulation order so their float64 results remain
    bit-for-bit identical.
    """
    gha = detector.gmst_estimate(t_gps) - right_ascension
    cosgha = cos(gha)
    singha = sin(gha)
    cosdec = cos(declination)
    sindec = sin(declination)

    # ``Detector.antenna_pattern(..., polarization=0)`` basis vectors.
    x0 = -singha
    x1 = -cosgha
    x2 = np.float64(0.0)
    y0 = -cosgha * sindec
    y1 = singha * sindec
    y2 = cosdec

    response = detector.response
    dx0 = ((response[0, 0] * x0 + response[0, 1] * x1)
           + response[0, 2] * x2)
    dx1 = ((response[1, 0] * x0 + response[1, 1] * x1)
           + response[1, 2] * x2)
    dx2 = ((response[2, 0] * x0 + response[2, 1] * x1)
           + response[2, 2] * x2)
    dy0 = ((response[0, 0] * y0 + response[0, 1] * y1)
           + response[0, 2] * y2)
    dy1 = ((response[1, 0] * y0 + response[1, 1] * y1)
           + response[1, 2] * y2)
    dy2 = ((response[2, 0] * y0 + response[2, 1] * y1)
           + response[2, 2] * y2)

    fplus = (((x0 * dx0 - y0 * dy0) + (x1 * dx1 - y1 * dy1))
             + (x2 * dx2 - y2 * dy2))
    fcross = (((x0 * dy0 + y0 * dx0) + (x1 * dy1 + y1 * dx1))
              + (x2 * dy2 + y2 * dx2))

    e0 = cosdec * cosgha
    e1 = cosdec * -singha
    e2 = sindec
    location = detector.location
    delay_dot = (((-location[0]) * e0 + (-location[1]) * e1)
                 + (-location[2]) * e2)
    delay = np.float64(delay_dot) / constants.c.value
    return fplus, fcross, delay


def _torch_antenna_pattern(response, right_ascension, declination,
                           polarization, gmst_start, phase_offsets):
    """Evaluate a tensor-polarization response on a Torch device."""
    import torch

    device = phase_offsets.device
    dtype = phase_offsets.dtype
    right_ascension = torch.as_tensor(
        right_ascension, device=device, dtype=dtype
    )
    declination = torch.as_tensor(declination, device=device, dtype=dtype)
    polarization = torch.as_tensor(
        polarization, device=device, dtype=dtype
    )

    gha_start = torch.as_tensor(
        gmst_start, device=device, dtype=dtype
    ) - right_ascension
    # Expanding the angle addition keeps sub-second sidereal changes visible
    # when the device only supports float32 (notably MPS).
    cos_start = torch.cos(gha_start)
    sin_start = torch.sin(gha_start)
    cos_offset = torch.cos(phase_offsets)
    sin_offset = torch.sin(phase_offsets)
    cosgha = cos_start * cos_offset - sin_start * sin_offset
    singha = sin_start * cos_offset + cos_start * sin_offset
    cosdec = torch.cos(declination)
    sindec = torch.sin(declination)
    cospsi = torch.cos(polarization)
    sinpsi = torch.sin(polarization)

    x = torch.stack((
        -cospsi * singha - sinpsi * cosgha * sindec,
        -cospsi * cosgha + sinpsi * singha * sindec,
        sinpsi * cosdec + torch.zeros_like(phase_offsets),
    ))
    y = torch.stack((
        sinpsi * singha - cospsi * cosgha * sindec,
        sinpsi * cosgha + cospsi * singha * sindec,
        cospsi * cosdec + torch.zeros_like(phase_offsets),
    ))

    response_is_complex = (
        response.is_complex() if isinstance(response, torch.Tensor)
        else np.iscomplexobj(response)
    )
    if response_is_complex:
        response_dtype = (
            torch.complex128 if dtype == torch.float64
            else torch.complex64
        )
    else:
        response_dtype = dtype
    response = torch.as_tensor(
        response, device=device, dtype=response_dtype
    )

    # A frequency-dependent response carries the broadcast sky/frequency
    # dimensions after its two matrix dimensions.  Add those dimensions to
    # the static response as singletons and contract without staging through
    # NumPy.
    response_grid_dims = response.ndim - 2
    vector_grid_dims = x.ndim - 1
    if response_grid_dims < vector_grid_dims:
        response = response.reshape(
            response.shape + (1,) * (vector_grid_dims - response_grid_dims)
        )
    x = x.to(dtype=response_dtype)
    y = y.to(dtype=response_dtype)
    dx = torch.einsum("ij...,j...->i...", response, x)
    dy = torch.einsum("ij...,j...->i...", response, y)
    fplus = torch.sum(x * dx - y * dy, dim=0)
    fcross = torch.sum(x * dy + y * dx, dim=0)
    return fplus, fcross


def _torch_single_arm_frequency_response(frequency, direction, arm_length):
    """Evaluate the finite-arm transfer function with Torch operations."""
    import torch

    tensor_inputs = tuple(
        value for value in (frequency, direction, arm_length)
        if isinstance(value, torch.Tensor)
    )
    anchor = tensor_inputs[0]
    if any(value.is_complex() for value in tensor_inputs):
        raise TypeError("Torch finite-arm response inputs must be real")

    dtype = None
    for value in tensor_inputs:
        value_dtype = (
            value.dtype if torch.is_floating_point(value)
            else torch.get_default_dtype()
        )
        dtype = (
            value_dtype if dtype is None
            else torch.promote_types(dtype, value_dtype)
        )
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32

    frequency = torch.as_tensor(
        frequency, device=anchor.device, dtype=dtype
    )
    direction = torch.as_tensor(
        direction, device=anchor.device, dtype=dtype
    ).clamp(-0.999, 0.999)
    arm_length = torch.as_tensor(
        arm_length, device=anchor.device, dtype=dtype
    )

    # This sinc form is algebraically equivalent to the expression in
    # ``single_arm_frequency_response`` while retaining its limit of one at
    # zero frequency and avoiding cancellation around that limit.
    phase = (
        2.0 * torch.pi * frequency * arm_length
        / float(constants.c.value)
    )
    minus = 1.0 - direction
    plus = 1.0 + direction
    return 0.5 * (
        torch.exp(-0.5j * phase * minus)
        * torch.sinc(phase * minus / (2.0 * torch.pi))
        + torch.exp(-0.5j * phase * (3.0 - direction))
        * torch.sinc(phase * plus / (2.0 * torch.pi))
    )


def _torch_time_delay(detector_location, other_location, right_ascension,
                      declination, gmst_start, phase_offsets):
    """Evaluate a detector time delay without leaving a Torch device."""
    import torch

    device = phase_offsets.device
    dtype = phase_offsets.dtype
    right_ascension = torch.as_tensor(
        right_ascension, device=device, dtype=dtype
    )
    declination = torch.as_tensor(declination, device=device, dtype=dtype)

    gha_start = torch.as_tensor(
        gmst_start, device=device, dtype=dtype
    ) - right_ascension
    # Keep the large absolute sidereal angle separate from the usually small
    # time offset. This retains sub-second changes on float32-only devices.
    cos_start = torch.cos(gha_start)
    sin_start = torch.sin(gha_start)
    cos_offset = torch.cos(phase_offsets)
    sin_offset = torch.sin(phase_offsets)
    cosgha = cos_start * cos_offset - sin_start * sin_offset
    singha = sin_start * cos_offset + cos_start * sin_offset
    cosdec = torch.cos(declination)

    ehat = torch.stack((
        cosdec * cosgha,
        -cosdec * singha,
        torch.sin(declination),
    ))
    dx = torch.as_tensor(
        np.asarray(other_location) - np.asarray(detector_location),
        device=device,
        dtype=dtype,
    )
    dx = dx.reshape((3,) + (1,) * (ehat.ndim - 1))
    return torch.sum(dx * ehat, dim=0) / float(constants.c.value)


def _torch_antenna_pattern_and_time_delay(detector_location, response,
                                          right_ascension, declination,
                                          polarization, gmst_start,
                                          phase_offsets):
    """Evaluate a tensor response and geocentric delay on a Torch device."""
    import torch

    device = phase_offsets.device
    dtype = phase_offsets.dtype
    right_ascension = torch.as_tensor(
        right_ascension, device=device, dtype=dtype
    )
    declination = torch.as_tensor(declination, device=device, dtype=dtype)
    polarization = torch.as_tensor(
        polarization, device=device, dtype=dtype
    )

    gha_start = torch.as_tensor(
        gmst_start, device=device, dtype=dtype
    ) - right_ascension
    # Expanding the angle addition keeps sub-second sidereal changes visible
    # when the device only supports float32 (notably MPS).
    cos_start = torch.cos(gha_start)
    sin_start = torch.sin(gha_start)
    cos_offset = torch.cos(phase_offsets)
    sin_offset = torch.sin(phase_offsets)
    cosgha = cos_start * cos_offset - sin_start * sin_offset
    singha = sin_start * cos_offset + cos_start * sin_offset
    cosdec = torch.cos(declination)
    sindec = torch.sin(declination)
    cospsi = torch.cos(polarization)
    sinpsi = torch.sin(polarization)

    x = torch.stack((
        -cospsi * singha - sinpsi * cosgha * sindec,
        -cospsi * cosgha + sinpsi * singha * sindec,
        sinpsi * cosdec + torch.zeros_like(phase_offsets),
    ))
    y = torch.stack((
        sinpsi * singha - cospsi * cosgha * sindec,
        sinpsi * cosgha + cospsi * singha * sindec,
        cospsi * cosdec + torch.zeros_like(phase_offsets),
    ))

    response_is_complex = (
        response.is_complex() if isinstance(response, torch.Tensor)
        else np.iscomplexobj(response)
    )
    if response_is_complex:
        response_dtype = (
            torch.complex128 if dtype == torch.float64
            else torch.complex64
        )
    else:
        response_dtype = dtype
    response = torch.as_tensor(
        response, device=device, dtype=response_dtype
    )

    response_grid_dims = response.ndim - 2
    vector_grid_dims = x.ndim - 1
    if response_grid_dims < vector_grid_dims:
        response = response.reshape(
            response.shape + (1,) * (vector_grid_dims - response_grid_dims)
        )
    x = x.to(dtype=response_dtype)
    y = y.to(dtype=response_dtype)
    dx = torch.einsum("ij...,j...->i...", response, x)
    dy = torch.einsum("ij...,j...->i...", response, y)
    fplus = torch.sum(x * dx - y * dy, dim=0)
    fcross = torch.sum(x * dy + y * dx, dim=0)

    ehat = torch.stack((
        cosdec * cosgha,
        -cosdec * singha,
        sindec + torch.zeros_like(phase_offsets),
    ))
    dx_loc = torch.as_tensor(
        -np.asarray(detector_location),
        device=device,
        dtype=dtype,
    )
    dx_loc = dx_loc.reshape((3,) + (1,) * (ehat.ndim - 1))
    delay = torch.sum(dx_loc * ehat, dim=0) / float(constants.c.value)
    return fplus, fcross, delay


def _torch_network_antenna_pattern_and_time_delay(
    detector_locations,
    responses,
    right_ascension,
    declination,
    polarization,
    gmst_start,
    phase_offsets,
):
    """Evaluate tensor responses and geocentric delays for D detectors on a Torch device."""
    import torch

    device = phase_offsets.device
    dtype = phase_offsets.dtype
    right_ascension = torch.as_tensor(
        right_ascension, device=device, dtype=dtype
    )
    declination = torch.as_tensor(declination, device=device, dtype=dtype)
    polarization = torch.as_tensor(
        polarization, device=device, dtype=dtype
    )

    gha_start = torch.as_tensor(
        gmst_start, device=device, dtype=dtype
    ) - right_ascension

    cos_start = torch.cos(gha_start)
    sin_start = torch.sin(gha_start)
    cos_offset = torch.cos(phase_offsets)
    sin_offset = torch.sin(phase_offsets)
    cosgha = cos_start * cos_offset - sin_start * sin_offset
    singha = sin_start * cos_offset + cos_start * sin_offset
    cosdec = torch.cos(declination)
    sindec = torch.sin(declination)
    cospsi = torch.cos(polarization)
    sinpsi = torch.sin(polarization)

    x = torch.stack((
        -cospsi * singha - sinpsi * cosgha * sindec,
        -cospsi * cosgha + sinpsi * singha * sindec,
        sinpsi * cosdec + torch.zeros_like(phase_offsets),
    ))
    y = torch.stack((
        sinpsi * singha - cospsi * cosgha * sindec,
        sinpsi * cosgha + cospsi * singha * sindec,
        cospsi * cosdec + torch.zeros_like(phase_offsets),
    ))

    responses_tensor = torch.as_tensor(
        responses, device=device, dtype=dtype
    )
    dx = torch.einsum("dij,j...->di...", responses_tensor, x)
    dy = torch.einsum("dij,j...->di...", responses_tensor, y)

    fplus = torch.sum(x * dx - y * dy, dim=1)
    fcross = torch.sum(x * dy + y * dx, dim=1)

    ehat = torch.stack((
        cosdec * cosgha,
        -cosdec * singha,
        sindec + torch.zeros_like(phase_offsets),
    ))
    locations_tensor = torch.as_tensor(
        detector_locations, device=device, dtype=dtype
    )
    delay = -torch.einsum("dj,j...->d...", locations_tensor, ehat) / float(constants.c.value)

    return fplus, fcross, delay


def _numpy_network_antenna_pattern_and_time_delay(
    detector_locations,
    responses,
    right_ascension,
    declination,
    polarization,
    gmst,
):
    """Evaluate tensor responses and geocentric delays for D detectors using NumPy."""
    gha = gmst - right_ascension
    cosgha = np.cos(gha)
    singha = np.sin(gha)
    cosdec = np.cos(declination)
    sindec = np.sin(declination)
    cospsi = np.cos(polarization)
    sinpsi = np.sin(polarization)

    x = np.array([
        -cospsi * singha - sinpsi * cosgha * sindec,
        -cospsi * cosgha + sinpsi * singha * sindec,
        sinpsi * cosdec + np.zeros_like(gha),
    ])
    y = np.array([
        sinpsi * singha - cospsi * cosgha * sindec,
        sinpsi * cosgha + cospsi * singha * sindec,
        cospsi * cosdec + np.zeros_like(gha),
    ])

    dx = np.einsum("dij,j...->di...", responses, x)
    dy = np.einsum("dij,j...->di...", responses, y)

    fplus = np.sum(x * dx - y * dy, axis=1)
    fcross = np.sum(x * dy + y * dx, axis=1)

    ehat = np.array([
        cosdec * cosgha,
        -cosdec * singha,
        sindec + np.zeros_like(gha),
    ])
    delay = -np.einsum("dj,j...->d...", detector_locations, ehat) / constants.c.value

    return fplus, fcross, delay


def get_available_detectors():
    """ List the available detectors """
    dets = list(_ground_detectors.keys())
    return dets

def get_available_lal_detectors():
    """Return list of detectors known in the currently sourced lalsuite.
    This function will query lalsuite about which detectors are known to
    lalsuite. Detectors are identified by a two character string e.g. 'K1',
    but also by a longer, and clearer name, e.g. KAGRA. This function returns
    both. As LAL doesn't really expose this functionality we have to make some
    assumptions about how this information is stored in LAL. Therefore while
    we hope this function will work correctly, it's possible it will need
    updating in the future. Better if lal would expose this information
    properly.
    """
    ld = lal.__dict__
    known_lal_names = [j for j in ld.keys() if "DETECTOR_PREFIX" in j]
    known_prefixes = [ld[k] for k in known_lal_names]
    known_names = [ld[k.replace('PREFIX', 'NAME')] for k in known_lal_names]
    return list(zip(known_prefixes, known_names))

_ground_detectors = {}

def add_detector_on_earth(name, longitude, latitude,
                          yangle=0, xangle=None, height=0,
                          xlength=4000, ylength=4000,
                          xaltitude=0, yaltitude=0):
    """ Add a new detector on the earth

    Parameters
    ----------

    name: str
        two-letter name to identify the detector
    longitude: float
        Longitude in radians using geodetic coordinates of the detector
    latitude: float
        Latitude in radians using geodetic coordinates of the detector
    yangle: float
        Azimuthal angle of the y-arm (angle drawn from pointing north)
    xangle: float
        Azimuthal angle of the x-arm (angle drawn from point north). If not set
        we assume a right angle detector following the right-hand rule.
    xaltitude: float
        The altitude angle of the x-arm measured from the local horizon.
    yaltitude: float
        The altitude angle of the y-arm measured from the local horizon.
    height: float
        The height in meters of the detector above the standard
        reference ellipsoidal earth
    """
    if xangle is None:
        # assume right angle detector if no separate xarm direction given
        xangle = yangle + np.pi / 2.0

    # baseline response of a single arm pointed in the -X direction
    resp = np.array([[-1, 0, 0], [0, 0, 0], [0, 0, 0]])
    rm2 = rotation_matrix(-longitude * units.rad, 'z')
    rm1 = rotation_matrix(-1.0 * (np.pi / 2.0 - latitude) * units.rad, 'y')
    
    # Calculate response in earth centered coordinates
    # by rotation of response in coordinates aligned
    # with the detector arms
    resps = []
    vecs = []
    for angle, azi in [(yangle, yaltitude), (xangle, xaltitude)]:
        rm0 = rotation_matrix(angle * units.rad, 'z')
        rmN = rotation_matrix(-azi *  units.rad, 'y')
        rm = rm2 @ rm1 @ rm0 @ rmN
        # apply rotation
        resps.append(rm @ resp @ rm.T / 2.0)
        vecs.append(rm @ np.array([-1, 0, 0]))

    full_resp = (resps[0] - resps[1])
    loc = coordinates.EarthLocation.from_geodetic(longitude * units.rad,
                                                  latitude * units.rad,
                                                  height=height*units.meter)
    loc = np.array([loc.x.value, loc.y.value, loc.z.value])
    _ground_detectors[name] = {'location': loc,
                               'response': full_resp,
                               'xresp': resps[1],
                               'yresp': resps[0],
                               'xvec': vecs[1],
                               'yvec': vecs[0],
                               'yangle': yangle,
                               'xangle': xangle,
                               'height': height,
                               'xaltitude': xaltitude,
                               'yaltitude': yaltitude,
                               'ylength': ylength,
                               'xlength': xlength,
                              }

# Notation matches
# Eq 4 of https://link.aps.org/accepted/10.1103/PhysRevD.96.084004
def single_arm_frequency_response(f, n, arm_length):
    """ The relative amplitude factor of the arm response due to
    signal delay. This is relevant where the long-wavelength
    approximation no longer applies)
    """
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and any(
            isinstance(value, torch.Tensor) for value in (f, n, arm_length)):
        return _torch_single_arm_frequency_response(f, n, arm_length)

    n = np.clip(n, -0.999, 0.999)
    phase = arm_length / constants.c.value * 2.0j * np.pi * f
    a = 1.0 / 4.0 / phase
    b = (1 - np.exp(-phase * (1 - n))) / (1 - n)
    c = np.exp(-2.0 * phase) * (1 - np.exp(phase * (1 + n))) / (1 + n)
    return a * (b - c) * 2.0  # We'll make this relative to the static resp

def load_detector_config(config_files):
    """ Add custom detectors from a configuration file

    Parameters
    ----------
    config_files: str or list of strs
        The config file(s) which specify new detectors
    """
    methods = {'earth_normal': (add_detector_on_earth,
                                ['longitude', 'latitude'])}
    conf = InterpolatingConfigParser(config_files)
    dets = conf.get_subsections('detector')
    for det in dets:
        kwds = dict(conf.items('detector-{}'.format(det)))
        try:
            method, arg_names = methods[kwds.pop('method')]
        except KeyError:
            raise ValueError("Missing or unkown method, "
                             "options are {}".format(methods.keys()))
        for k in kwds:
            kwds[k] = float(kwds[k])
        try:
            args = [kwds.pop(arg) for arg in arg_names]
        except KeyError:
            raise ValueError("missing required detector argument"
                             " {} are required".format(arg_names))
        method(det.upper(), *args, **kwds)


# prepopulate using detectors hardcoded into lalsuite
for pref, name in get_available_lal_detectors():
    lalsim = pycbc.libutils.import_optional('lalsimulation')
    lal_det = lalsim.DetectorPrefixToLALDetector(pref).frDetector
    add_detector_on_earth(pref,
                          lal_det.vertexLongitudeRadians,
                          lal_det.vertexLatitudeRadians,
                          height=lal_det.vertexElevation,
                          xangle=lal_det.xArmAzimuthRadians,
                          yangle=lal_det.yArmAzimuthRadians,
                          xlength=lal_det.xArmMidpoint * 2,
                          ylength=lal_det.yArmMidpoint * 2,
                          xaltitude=lal_det.xArmAltitudeRadians,
                          yaltitude=lal_det.yArmAltitudeRadians,
                          )
# autoload detector config files
if 'PYCBC_DETECTOR_CONFIG' in os.environ:
    load_detector_config(os.environ['PYCBC_DETECTOR_CONFIG'].split(':'))


class Detector(object):
    """A gravitational wave detector
    """
    @staticmethod
    def _apply_response(resp, v0, v1, v2):
        """Return the vector (v0, v1, v2) and the response applied to it.

        The components need not share a shape: many sky positions may share one
        time, or one position be asked about at many times. Broadcasting them
        together gives a float array whatever shapes they arrive in, so the
        product runs over the whole set at once.

        Parameters
        ----------
        resp: numpy.ndarray
            The 3x3 detector response matrix.
        v0, v1, v2: float or numpy.ndarray
            The components of the vector.

        Returns
        -------
        v: numpy.ndarray
            The components stacked along the first axis.
        dv: numpy.ndarray
            The response matrix applied to it, with the same shape.
        """
        v = np.array(np.broadcast_arrays(v0, v1, v2))
        return v, np.tensordot(resp, v, axes=(1, 0))

    def __init__(self, detector_name, reference_time=1126259462.0):
        """ Create class representing a gravitational-wave detector
        Parameters
        ----------
        detector_name: str
            The two-character detector string, i.e. H1, L1, V1, K1, I1
        reference_time: float
            Default is time of GW150914. In this case, the earth's rotation
        will be estimated from a reference time. If 'None', we will
        calculate the time for each gps time requested explicitly
        using a slower but higher precision method.
        """
        self.name = str(detector_name)
        
        if detector_name in _ground_detectors:
            self.info = _ground_detectors[detector_name]
            self.response = self.info['response']
            self.location = self.info['location']
        else:
            raise ValueError("Unkown detector {}".format(detector_name))

        loc = coordinates.EarthLocation(self.location[0],
                                        self.location[1],
                                        self.location[2],
                                        unit=meter)
        self.latitude = loc.lat.rad
        self.longitude = loc.lon.rad

        self.reference_time = reference_time
        self.sday = None
        self.gmst_reference = None

    def set_gmst_reference(self):
        if self.reference_time is not None:
            self.sday = float(sday.si.scale)
            self.gmst_reference = gmst_accurate(self.reference_time)
        else:
            raise RuntimeError("Can't get accurate sidereal time without GPS "
                               "reference time!")

    def lal(self):
        """ Return lal data type detector instance """
        import lal
        d = lal.FrDetector()
        d.vertexLongitudeRadians = self.longitude
        d.vertexLatitudeRadians = self.latitude
        d.vertexElevation = self.info['height']
        d.xArmAzimuthRadians = self.info['xangle']
        d.yArmAzimuthRadians = self.info['yangle']
        d.xArmAltitudeRadians = self.info['xaltitude']
        d.yArmAltitudeRadians = self.info['yaltitude']

        # This is somewhat abused by lalsimulation at the moment
        # to determine a filter kernel size. We set this only so that
        # value gets a similar number of samples as other detectors
        # it is used for nothing else
        d.yArmMidpoint = self.info['ylength'] / 2.0
        d.xArmMidpoint = self.info['xlength'] / 2.0

        x = lal.Detector()
        r = lal.CreateDetector(x, d, lal.LALDETECTORTYPE_IFODIFF)
        self._lal = r
        return r

    def gmst_estimate(self, gps_time):
        if self.reference_time is None:
            return gmst_accurate(gps_time)

        if self.gmst_reference is None:
            self.set_gmst_reference()
        dphase = (gps_time - self.reference_time) / self.sday * (2.0 * np.pi)
        gmst = (self.gmst_reference + dphase) % (2.0 * np.pi)
        return gmst

    def light_travel_time_to_detector(self, det):
        """ Return the light travel time from this detector
        Parameters
        ----------
        det: Detector
            The other detector to determine the light travel time to.
        Returns
        -------
        time: float
            The light travel time in seconds
        """
        d = self.location - det.location
        return float(d.dot(d)**0.5 / constants.c.value)

    def antenna_pattern(self, right_ascension, declination, polarization, t_gps,
                        frequency=0,
                        polarization_type='tensor'):
        """Return the detector response.

        Parameters
        ----------
        right_ascension: float, numpy.ndarray, or torch.Tensor
            The right ascension of the source
        declination: float, numpy.ndarray, or torch.Tensor
            The declination of the source
        polarization: float, numpy.ndarray, or torch.Tensor
            The polarization angle of the source
        polarization_type: string flag: Tensor, Vector or Scalar
            The gravitational wave polarizations. Default: 'Tensor'

        Returns
        -------
        fplus(default) or fx or fb : float, numpy.ndarray, or torch.Tensor
            The plus or vector-x or breathing polarization factor for this sky location / orientation
        fcross(default) or fy or fl : float, numpy.ndarray, or torch.Tensor
            The cross or vector-y or longitudnal polarization factor for this sky location / orientation
        """
        if isinstance(t_gps, lal.LIGOTimeGPS):
            t_gps = float(t_gps)

        # Polarization marginalizations commonly evaluate thousands of
        # angles at once. Keep that grid, and finite-arm frequency responses,
        # on their Torch device instead of passing them through NumPy.
        try:
            import torch
        except ImportError:
            torch = None
        angular_inputs = (right_ascension, declination, polarization)
        torch_inputs = angular_inputs + (frequency, t_gps)
        if torch is not None and any(
                isinstance(value, torch.Tensor) for value in torch_inputs):
            if polarization_type != 'tensor':
                raise NotImplementedError(
                    "Torch antenna patterns currently support only the "
                    "tensor response"
                )
            angular_tensors = tuple(
                value for value in angular_inputs
                if isinstance(value, torch.Tensor)
            )
            if any(
                    not torch.is_floating_point(value)
                    for value in angular_tensors):
                raise TypeError(
                    "Torch antenna-pattern angles must be floating"
                )
            tensor_inputs = tuple(
                value for value in torch_inputs
                if isinstance(value, torch.Tensor)
            )
            if any(value.is_complex() for value in tensor_inputs):
                raise TypeError(
                    "Torch antenna-pattern inputs must be real"
                )
            anchor = next(
                value for value in torch_inputs
                if isinstance(value, torch.Tensor)
            )
            dtype = None
            for value in tensor_inputs:
                value_dtype = (
                    value.dtype if torch.is_floating_point(value)
                    else torch.get_default_dtype()
                )
                dtype = (
                    value_dtype if dtype is None
                    else torch.promote_types(dtype, value_dtype)
                )
            if dtype in (torch.float16, torch.bfloat16):
                dtype = torch.float32
            if anchor.device.type == 'mps' and dtype == torch.float64:
                dtype = torch.float32

            frequency_is_tensor = isinstance(frequency, torch.Tensor)
            frequency_is_array = (
                not frequency_is_tensor and np.ndim(frequency) > 0
            )
            finite_arm = (
                frequency_is_tensor
                or frequency_is_array
                or frequency != 0
            )
            time_is_tensor = isinstance(t_gps, torch.Tensor)
            time_is_array = not time_is_tensor and np.ndim(t_gps) > 0
            time_grid = time_is_tensor or time_is_array
            values = list(angular_inputs)
            if time_grid:
                if self.reference_time is None:
                    raise NotImplementedError(
                        "Torch GPS-time grids require a detector GMST "
                        "reference time"
                    )
                if self.gmst_reference is None:
                    self.set_gmst_reference()
                if time_is_tensor:
                    relative_time = t_gps.to(
                        device=anchor.device, dtype=dtype
                    ) - float(self.reference_time)
                else:
                    # Center host times before their one-way upload.  This
                    # keeps fractional samples when the device is limited to
                    # float32 (notably MPS).
                    relative_time = torch.as_tensor(
                        np.asarray(t_gps, dtype=np.float64)
                        - float(self.reference_time),
                        device=anchor.device,
                        dtype=dtype,
                    )
                values.append(relative_time)
            if finite_arm:
                values.append(frequency)
            broadcast = torch.broadcast_tensors(*(
                torch.as_tensor(
                    value, device=anchor.device, dtype=dtype
                )
                for value in values
            ))
            angles = broadcast[:3]
            next_value = 3
            if time_grid:
                relative_time = broadcast[next_value]
                next_value += 1
                phase_offsets = (
                    relative_time / float(self.sday) * (2.0 * np.pi)
                )
                gmst_start = self.gmst_reference
            else:
                phase_offsets = torch.zeros_like(angles[0])
                gmst_start = self.gmst_estimate(t_gps)
            response = self.response
            if finite_arm:
                frequency_tensor = broadcast[next_value]
                gmst = torch.as_tensor(
                    gmst_start,
                    device=anchor.device,
                    dtype=dtype,
                ) + phase_offsets
                gha = gmst - angles[0]
                cosdec = torch.cos(angles[1])
                direction = torch.stack((
                    cosdec * torch.cos(gha),
                    -cosdec * torch.sin(gha),
                    torch.sin(angles[1]),
                ))
                grid_dims = direction.ndim - 1
                xvec = torch.as_tensor(
                    self.info['xvec'], device=anchor.device, dtype=dtype
                ).reshape((3,) + (1,) * grid_dims)
                yvec = torch.as_tensor(
                    self.info['yvec'], device=anchor.device, dtype=dtype
                ).reshape((3,) + (1,) * grid_dims)
                nx = torch.sum(direction * xvec, dim=0)
                ny = torch.sum(direction * yvec, dim=0)
                rx = _torch_single_arm_frequency_response(
                    frequency_tensor, nx, self.info['xlength']
                )
                ry = _torch_single_arm_frequency_response(
                    frequency_tensor, ny, self.info['ylength']
                )
                matrix_shape = (3, 3) + (1,) * rx.ndim
                xresp = torch.as_tensor(
                    self.info['xresp'], device=anchor.device, dtype=dtype
                ).reshape(matrix_shape)
                yresp = torch.as_tensor(
                    self.info['yresp'], device=anchor.device, dtype=dtype
                ).reshape(matrix_shape)
                response = ry * yresp - rx * xresp
            return _torch_antenna_pattern(
                response,
                angles[0],
                angles[1],
                angles[2],
                gmst_start,
                phase_offsets,
            )

        gha = self.gmst_estimate(t_gps) - right_ascension

        cosgha = cos(gha)
        singha = sin(gha)
        cosdec = cos(declination)
        sindec = sin(declination)
        cospsi = cos(polarization)
        sinpsi = sin(polarization)

        if frequency:
            e0 = cosdec * cosgha
            e1 = cosdec * -singha
            e2 = sin(declination)
            nhat = np.array([e0, e1, e2], dtype=object)

            nx = nhat.dot(self.info['xvec'])
            ny = nhat.dot(self.info['yvec'])

            rx = single_arm_frequency_response(frequency, nx,
                                               self.info['xlength'])
            ry = single_arm_frequency_response(frequency, ny,
                                               self.info['ylength'])
            resp = ry * self.info['yresp'] -  rx * self.info['xresp']
            ttype = np.complex128
        else:
            resp = self.response
            ttype = np.float64

        x0 = -cospsi * singha - sinpsi * cosgha * sindec
        x1 = -cospsi * cosgha + sinpsi * singha * sindec
        x2 =  sinpsi * cosdec

        x, dx = self._apply_response(resp, x0, x1, x2)

        y0 =  sinpsi * singha - cospsi * cosgha * sindec
        y1 =  sinpsi * cosgha + cospsi * singha * sindec
        y2 =  cospsi * cosdec

        y, dy = self._apply_response(resp, y0, y1, y2)

        if polarization_type != 'tensor':
            z0 = -cosdec * cosgha
            z1 = cosdec * singha
            z2 = -sindec
            z, dz = self._apply_response(resp, z0, z1, z2)

        if polarization_type == 'tensor':
            if hasattr(dx, 'shape'):
                fplus = (x * dx - y * dy).sum(axis=0).astype(ttype)
                fcross = (x * dy + y * dx).sum(axis=0).astype(ttype)
            else:
                fplus = (x * dx - y * dy).sum()
                fcross = (x * dy + y * dx).sum()
            return fplus, fcross

        elif polarization_type == 'vector':
            if hasattr(dx, 'shape'):
                fx = (z * dx + x * dz).sum(axis=0).astype(ttype)
                fy = (z * dy + y * dz).sum(axis=0).astype(ttype)
            else:
                fx = (z * dx + x * dz).sum()
                fy = (z * dy + y * dz).sum()

            return fx, fy

        elif polarization_type == 'scalar':
            if hasattr(dx, 'shape'):
                fb = (x * dx + y * dy).sum(axis=0).astype(ttype)
                fl = (z * dz).sum(axis=0)
            else:
                fb = (x * dx + y * dy).sum()
                fl = (z * dz).sum()
            return fb, fl

    def antenna_pattern_and_time_delay(
        self, right_ascension, declination, polarization, t_gps
    ):
        """Return antenna pattern (fp, fc) and geocentric delay together.

        Parameters
        ----------
        right_ascension : float, numpy.ndarray, or torch.Tensor
            The right ascension of the source.
        declination : float, numpy.ndarray, or torch.Tensor
            The declination of the source.
        polarization : float, numpy.ndarray, or torch.Tensor
            The polarization angle of the source.
        t_gps : float, lal.LIGOTimeGPS, numpy.ndarray, or torch.Tensor
            The GPS time.

        Returns
        -------
        fplus : float, numpy.ndarray, or torch.Tensor
            Plus polarization antenna response.
        fcross : float, numpy.ndarray, or torch.Tensor
            Cross polarization antenna response.
        delay : float, numpy.ndarray, or torch.Tensor
            Geocentric time delay.
        """
        if isinstance(t_gps, lal.LIGOTimeGPS):
            t_gps = float(t_gps)

        try:
            import torch
        except ImportError:
            torch = None

        angular_inputs = (right_ascension, declination, polarization)
        torch_inputs = angular_inputs + (t_gps,)
        if torch is not None and any(
                isinstance(value, torch.Tensor) for value in torch_inputs):
            angular_tensors = tuple(
                value for value in angular_inputs
                if isinstance(value, torch.Tensor)
            )
            if any(
                    not torch.is_floating_point(value)
                    for value in angular_tensors):
                raise TypeError(
                    "Torch antenna-pattern angles must be floating"
                )
            tensor_inputs = tuple(
                value for value in torch_inputs
                if isinstance(value, torch.Tensor)
            )
            if any(value.is_complex() for value in tensor_inputs):
                raise TypeError(
                    "Torch antenna-pattern inputs must be real"
                )
            anchor = next(
                value for value in torch_inputs
                if isinstance(value, torch.Tensor)
            )
            dtype = None
            for value in tensor_inputs:
                value_dtype = (
                    value.dtype if torch.is_floating_point(value)
                    else torch.get_default_dtype()
                )
                dtype = (
                    value_dtype if dtype is None
                    else torch.promote_types(dtype, value_dtype)
                )
            if dtype in (torch.float16, torch.bfloat16):
                dtype = torch.float32
            if anchor.device.type == 'mps' and dtype == torch.float64:
                dtype = torch.float32

            time_is_tensor = isinstance(t_gps, torch.Tensor)
            time_is_array = not time_is_tensor and np.ndim(t_gps) > 0
            time_grid = time_is_tensor or time_is_array
            values = list(angular_inputs)
            if time_grid:
                if self.reference_time is None:
                    raise NotImplementedError(
                        "Torch GPS-time grids require a detector GMST "
                        "reference time"
                    )
                if self.gmst_reference is None:
                    self.set_gmst_reference()
                if time_is_tensor:
                    relative_time = t_gps.to(
                        device=anchor.device, dtype=dtype
                    ) - float(self.reference_time)
                else:
                    # Center host times before their one-way upload.
                    relative_time = torch.as_tensor(
                        np.asarray(t_gps, dtype=np.float64)
                        - float(self.reference_time),
                        device=anchor.device,
                        dtype=dtype,
                    )
                values.append(relative_time)
            broadcast = torch.broadcast_tensors(*(
                torch.as_tensor(
                    value, device=anchor.device, dtype=dtype
                )
                for value in values
            ))
            angles = broadcast[:3]
            if time_grid:
                relative_time = broadcast[3]
                phase_offsets = (
                    relative_time / float(self.sday) * (2.0 * np.pi)
                )
                gmst_start = self.gmst_reference
            else:
                phase_offsets = torch.zeros_like(angles[0])
                gmst_start = self.gmst_estimate(t_gps)
            return _torch_antenna_pattern_and_time_delay(
                self.location,
                self.response,
                angles[0],
                angles[1],
                angles[2],
                gmst_start,
                phase_offsets,
            )

        is_scalar = (
            np.ndim(right_ascension) == 0
            and np.ndim(declination) == 0
            and np.ndim(polarization) == 0
            and np.ndim(t_gps) == 0
        )
        if is_scalar:
            fp0, fc0, delay = _scalar_antenna_pattern_and_time_delay(
                self, right_ascension, declination, t_gps
            )
            if polarization == 0 or polarization == 0.0:
                return fp0, fc0, delay
            cos2psi = np.cos(2.0 * polarization)
            sin2psi = np.sin(2.0 * polarization)
            fp = cos2psi * fp0 + sin2psi * fc0
            fc = -sin2psi * fp0 + cos2psi * fc0
            return fp, fc, delay

        right_ascension, declination, polarization, t_gps = (
            np.broadcast_arrays(
                right_ascension, declination, polarization, t_gps
            )
        )
        gha = self.gmst_estimate(t_gps) - right_ascension
        cosgha = np.cos(gha)
        singha = np.sin(gha)
        cosdec = np.cos(declination)
        sindec = np.sin(declination)
        cospsi = np.cos(polarization)
        sinpsi = np.sin(polarization)

        x0 = -cospsi * singha - sinpsi * cosgha * sindec
        x1 = -cospsi * cosgha + sinpsi * singha * sindec
        x2 = sinpsi * cosdec

        y0 = sinpsi * singha - cospsi * cosgha * sindec
        y1 = sinpsi * cosgha + cospsi * singha * sindec
        y2 = cospsi * cosdec

        x = np.array(np.broadcast_arrays(x0, x1, x2))
        y = np.array(np.broadcast_arrays(y0, y1, y2))
        dx = np.tensordot(self.response, x, axes=(1, 0))
        dy = np.tensordot(self.response, y, axes=(1, 0))
        fplus = (x * dx - y * dy).sum(axis=0).astype(np.float64)
        fcross = (x * dy + y * dx).sum(axis=0).astype(np.float64)

        e0 = cosdec * cosgha
        e1 = -cosdec * singha
        e2 = sindec
        proj = (-self.location[0] * e0
                - self.location[1] * e1
                - self.location[2] * e2)
        delay = (proj / constants.c.value).astype(np.float64)
        return fplus, fcross, delay

    antenna_pattern_and_delay = antenna_pattern_and_time_delay

    def time_delay_from_earth_center(self, right_ascension, declination, t_gps):
        """Return the time delay from the earth center
        """
        return self.time_delay_from_location(np.array([0, 0, 0]),
                                             right_ascension,
                                             declination,
                                             t_gps)

    def time_delay_from_location(self, other_location, right_ascension,
                                 declination, t_gps):
        """Return the time delay from the given location to detector for
        a signal with the given sky location
        In other words return `t1 - t2` where `t1` is the
        arrival time in this detector and `t2` is the arrival time in the
        other location.

        Parameters
        ----------
        other_location : numpy.ndarray of coordinates
            A detector instance.
        right_ascension : float, numpy.ndarray, or torch.Tensor
            The right ascension (in rad) of the signal.
        declination : float, numpy.ndarray, or torch.Tensor
            The declination (in rad) of the signal.
        t_gps : float, numpy.ndarray, or torch.Tensor
            The GPS time (in s) of the signal.

        Returns
        -------
        float, numpy.ndarray, or torch.Tensor
            The arrival time difference between the detectors.
        """
        if isinstance(t_gps, lal.LIGOTimeGPS):
            t_gps = float(t_gps)

        try:
            import torch
        except ImportError:
            torch = None
        inputs = (right_ascension, declination, t_gps)
        if torch is not None and any(
                isinstance(value, torch.Tensor) for value in inputs):
            tensors = tuple(
                value for value in inputs
                if isinstance(value, torch.Tensor)
            )
            if any(torch.is_complex(value) for value in tensors):
                raise TypeError("Torch detector-timing inputs must be real")
            anchor = tensors[0]
            dtype = anchor.dtype
            for value in tensors[1:]:
                dtype = torch.promote_types(dtype, value.dtype)
            if not dtype.is_floating_point:
                dtype = torch.get_default_dtype()
            device = anchor.device

            right_ascension = torch.as_tensor(
                right_ascension, device=device, dtype=dtype
            )
            declination = torch.as_tensor(
                declination, device=device, dtype=dtype
            )
            if isinstance(t_gps, torch.Tensor):
                if self.reference_time is None:
                    raise NotImplementedError(
                        "Torch GPS-time grids require a detector GMST "
                        "reference time"
                    )
                if self.gmst_reference is None:
                    self.set_gmst_reference()
                t_gps = torch.as_tensor(
                    t_gps, device=device, dtype=dtype
                )
                right_ascension, declination, t_gps = (
                    torch.broadcast_tensors(
                        right_ascension, declination, t_gps
                    )
                )
                phase_offsets = (
                    (t_gps - float(self.reference_time))
                    / float(self.sday)
                    * (2.0 * np.pi)
                )
                gmst_start = self.gmst_reference
            else:
                right_ascension, declination = torch.broadcast_tensors(
                    right_ascension, declination
                )
                phase_offsets = torch.zeros_like(right_ascension)
                gmst_start = self.gmst_estimate(t_gps)

            return _torch_time_delay(
                self.location,
                other_location,
                right_ascension,
                declination,
                gmst_start,
                phase_offsets,
            )

        ra_angle = self.gmst_estimate(t_gps) - right_ascension
        cosd = cos(declination)

        e0 = cosd * cos(ra_angle)
        e1 = cosd * -sin(ra_angle)
        e2 = sin(declination)

        # written out componentwise rather than stacked and dotted: the
        # stack costs more to build than the reduction saves, measured at
        # every size, and the components may have different shapes anyway
        dx = other_location - self.location
        proj = dx[0] * e0 + dx[1] * e1 + dx[2] * e2
        return proj / constants.c.value

    def time_delay_from_detector(self, other_detector, right_ascension,
                                 declination, t_gps):
        """Return the time delay from the given to detector for a signal with
        the given sky location; i.e. return `t1 - t2` where `t1` is the
        arrival time in this detector and `t2` is the arrival time in the
        other detector. Note that this would return the same value as
        `time_delay_from_earth_center` if `other_detector` was geocentric.
        Parameters
        ----------
        other_detector : detector.Detector
            A detector instance.
        right_ascension : float
            The right ascension (in rad) of the signal.
        declination : float
            The declination (in rad) of the signal.
        t_gps : float
            The GPS time (in s) of the signal.
        Returns
        -------
        float
            The arrival time difference between the detectors.
        """
        return self.time_delay_from_location(other_detector.location,
                                             right_ascension,
                                             declination,
                                             t_gps)
    
    def arrival_time(self, ref_tc, ra, dec, ref_frame='geocentric'):
        """Compute the arrival time in this detector.
        
        Parameters
        ----------
        ref_tc : {float, lal.LIGOTimeGPS, torch.Tensor}
            The coalescence time to convert, defined in ref_frame
        ra : float or torch.Tensor
            Right ascension.
        dec : float or torch.Tensor
            Declination.
        ref_frame : str (optional)
            The detector to convert from, in which ref_tc is sampled. Default
            'geocentric'.
            
        Returns
        -------
        float or torch.Tensor :
            The coalescence time converted to the current detector frame.
        """
        if ref_frame == 'geocentric':
            # from geocenter
            tc = ref_tc + \
                self.time_delay_from_earth_center(ra, dec, ref_tc)
        elif ref_frame == self.name:
            # no time shift; sampling in current det
            tc = ref_tc
        elif ref_frame in get_available_detectors():
            # from sampling det
            refdet = Detector(ref_frame)
            tc = ref_tc + \
                self.time_delay_from_detector(refdet, ra, dec, ref_tc)
        else:
            raise ValueError(f'Unrecognized ref_frame argument {ref_frame}. '
                             'Accepted arguments are: "geocentric", '
                             f'{get_available_detectors()}')
        return tc

    def project_wave(self, hp, hc, ra, dec, polarization,
                     method='lal',
                     reference_time=None):
        """Return the strain of a waveform as measured by the detector.
        Apply the time shift for the given detector relative to the assumed
        geocentric frame and apply the antenna patterns to the plus and cross
        polarizations.

        Parameters
        ----------
        hp: pycbc.types.TimeSeries
            Plus polarization of the GW
        hc: pycbc.types.TimeSeries
            Cross polarization of the GW
        ra: float
            Right ascension of source location
        dec: float
            Declination of source location
        polarization: float
            Polarization angle of the source
        method: {'lal', 'constant', 'vary_polarization'}
            The method to use for projecting the polarizations into the
            detector frame. Default is 'lal'.
        reference_time: float, Optional
            The time to use as, a reference for some methods of projection.
            Used by 'constant' and 'vary_polarization' methods. Uses average
            time if not provided.

        Notes
        -----
        Under :class:`~pycbc.scheme.TorchScheme`, ``method='lal'`` keeps the
        waveform mixing and finite-arm interpolation on the active Torch
        device. Small detector-geometry calculations continue to use LAL at
        the response update cadence; ``lalsimulation`` is not used.
        """
        # The robust and most feature-rich method which includes
        # time changing antenna patterns and doppler shifts due to the
        # earth rotation and orbit
        if method == 'lal':
            active_scheme = _scheme.mgr.state
            using_torch = isinstance(active_scheme, _scheme.TorchScheme)
            tensor = getattr(getattr(hp, '_data', None), 'tensor', None)
            if using_torch and tensor is not None:
                from pycbc.detector.ground_torch import project_wave
                ts = project_wave(self, hp, hc, ra, dec, polarization)
            else:
                import lalsimulation
                h_lal = lalsimulation.SimDetectorStrainREAL8TimeSeries(
                        _to_lal_real8_time_series(hp),
                        _to_lal_real8_time_series(hc),
                        ra, dec, polarization, self.lal())
                ts = TimeSeries(
                        h_lal.data.data, delta_t=h_lal.deltaT,
                        epoch=h_lal.epoch, dtype=np.float64,
                        copy=not isinstance(active_scheme, _scheme.CPUScheme))

        # 'constant' assume fixed orientation relative to source over the
        # duration of the signal, accurate for short duration signals
        # 'fixed_polarization' applies only time changing orientation
        # but no doppler corrections
        elif method in ['constant', 'vary_polarization']:
            tensor = None
            if reference_time is not None:
                rtime = reference_time
            else:
                # In many cases, one should set the reference time if using
                # this method as we don't know where the signal is within
                # the given time series. If not provided, we'll choose
                # the midpoint time.
                rtime = (float(hp.end_time) + float(hp.start_time)) / 2.0

            if method == 'constant':
                time = rtime
            elif method == 'vary_polarization':
                if (not isinstance(hp, TimeSeries) or
                   not isinstance(hc, TimeSeries)):
                    raise TypeError('Waveform polarizations must be given'
                                    ' as time series for this method')

                tensor = getattr(getattr(hp, '_data', None), 'tensor', None)
                if tensor is None:
                    time = hp.sample_times.numpy()
                else:
                    import torch
                    from pycbc.types.array_torch import TorchArrayData

                    # Start from an accurately evaluated scalar GMST and add
                    # only the small per-sample phase increment.  This avoids
                    # both a host copy and loss of GPS-time resolution on MPS.
                    gmst_start = float(
                        self.gmst_estimate(float(hp.start_time))
                    )
                    phase_step = (
                        hp.delta_t * 2.0 * np.pi / float(sday.si.scale)
                    )
                    phase_offsets = torch.arange(
                        len(hp),
                        device=tensor.device,
                        dtype=tensor.real.dtype,
                    ) * phase_step
                    fp, fc = _torch_antenna_pattern(
                        self.response, ra, dec, polarization,
                        gmst_start, phase_offsets
                    )
                    fp = TimeSeries(
                        TorchArrayData(fp), delta_t=hp.delta_t,
                        epoch=hp.start_time, copy=False
                    )
                    fc = TimeSeries(
                        TorchArrayData(fc), delta_t=hp.delta_t,
                        epoch=hp.start_time, copy=False
                    )

            if method == 'constant' or tensor is None:
                fp, fc = self.antenna_pattern(ra, dec, polarization, time)
            dt = self.time_delay_from_earth_center(ra, dec, rtime)
            # Keep PyCBC arrays on the left so their active backend handles
            # the scalar operations.  NumPy scalars on the left invoke
            # NumPy's ufunc machinery, which cannot wrap Torch storage
            # without a copy.
            ts = hp * fp + hc * fc
            ts.start_time = float(ts.start_time) + dt

        # add in only the correction for the time variance in the polarization
        # due to the earth's rotation, no doppler correction applied
        else:
            raise ValueError("Unkown projection method {}".format(method))
        return ts

    def optimal_orientation(self, t_gps):
        """Return the optimal orientation in right ascension and declination
           for a given GPS time.

        Parameters
        ----------
        t_gps: float
            Time in gps seconds

        Returns
        -------
        ra: float
            Right ascension that is optimally oriented for the detector
        dec: float
            Declination that is optimally oriented for the detector
        """
        ra = self.longitude + (self.gmst_estimate(t_gps) % (2.0*np.pi))
        dec = self.latitude
        return ra, dec

    def get_icrs_pos(self):
        """ Transforms GCRS frame to ICRS frame

        Returns
        ----------
        loc: numpy.ndarray shape (3,1) units: AU
             ICRS coordinates in cartesian system
        """
        loc = self.location
        loc = coordinates.SkyCoord(x=loc[0], y=loc[1], z=loc[2], unit=units.m,
                frame='gcrs', representation_type='cartesian').transform_to('icrs')
        loc.representation_type = 'cartesian'
        conv = np.float32(((loc.x.unit/units.AU).decompose()).to_string())
        loc = np.array([np.float32(loc.x), np.float32(loc.y),
                        np.float32(loc.z)])*conv
        return loc

    def effective_distance(self, distance, ra, dec, pol, time, inclination):
        """ Distance scaled to account for amplitude factors

        The effective distance of the source. This scales the distance so that
        the amplitude is equal to a source which is optimally oriented with
        respect to the detector. For fixed detector-frame intrinsic parameters
        this is a measure of the expected signal strength.

        Parameters
        ----------
        distance: float
            Source luminosity distance in megaparsecs
        ra: float
            The right ascension in radians
        dec: float
            The declination in radians
        pol: float
            Polarization angle of the gravitational wave in radians
        time: float
            GPS time in seconds
        inclination:
            The inclination of the binary's orbital plane

        Returns
        -------
        eff_dist: float
            The effective distance of the source
        """
        try:
            import torch
        except ImportError:
            torch = None
        values = (distance, ra, dec, pol, time, inclination)
        if torch is not None and any(
                isinstance(value, torch.Tensor) for value in values):
            tensor_inputs = tuple(
                value for value in values
                if isinstance(value, torch.Tensor)
            )
            if any(value.is_complex() for value in tensor_inputs):
                raise TypeError("Torch effective-distance inputs must be real")
            if any(
                    isinstance(value, torch.Tensor)
                    and not torch.is_floating_point(value)
                    for value in (ra, dec, pol)):
                raise TypeError(
                    "Torch effective-distance angles must be floating"
                )

            anchor = tensor_inputs[0]
            dtype = None
            for value in tensor_inputs:
                value_dtype = (
                    value.dtype if torch.is_floating_point(value)
                    else torch.get_default_dtype()
                )
                dtype = (
                    value_dtype if dtype is None
                    else torch.promote_types(dtype, value_dtype)
                )
            if dtype in (torch.float16, torch.bfloat16):
                dtype = torch.float32

            broadcast_values = (distance, ra, dec, pol, inclination)
            if isinstance(time, torch.Tensor):
                broadcast_values += (time,)
            broadcast = torch.broadcast_tensors(*(
                torch.as_tensor(
                    value, device=anchor.device, dtype=dtype
                )
                for value in broadcast_values
            ))
            distance, ra, dec, pol, inclination = broadcast[:5]
            if isinstance(time, torch.Tensor):
                time = broadcast[5]
            fp, fc = self.antenna_pattern(ra, dec, pol, time)
            ic = torch.cos(inclination)
            ip = 0.5 * (1.0 + ic * ic)
            scale = torch.sqrt((fp * ip).square() + (fc * ic).square())
            return distance / scale

        fp, fc = self.antenna_pattern(ra, dec, pol, time)
        ic = np.cos(inclination)
        ip = 0.5 * (1. + ic * ic)
        scale = ((fp * ip) ** 2.0 + (fc * ic) ** 2.0) ** 0.5
        return distance / scale


# Capture the built-in implementations while this module is loading.  Code
# importing ``Detector`` may monkeypatch its methods before importing an
# inference model, so model-local snapshots cannot reliably identify stock
# detector behavior.
_DETECTOR_BUILTIN_METHODS = (
    Detector.antenna_pattern,
    Detector.time_delay_from_earth_center,
    Detector.time_delay_from_location,
    Detector.gmst_estimate,
    Detector.set_gmst_reference,
)


def overhead_antenna_pattern(right_ascension, declination, polarization):
    """Return the antenna pattern factors F+ and Fx as a function of sky
    location and polarization angle for a hypothetical interferometer located
    at the north pole. Angles are in radians. Declinations of ±π/2 correspond
    to the normal to the detector plane (i.e. overhead and underneath) while
    the point with zero right ascension and declination is the direction
    of one of the interferometer arms.
    Parameters
    ----------
    right_ascension: float
    declination: float
    polarization: float
    Returns
    -------
    f_plus: float
    f_cros: float
    """
    # convert from declination coordinate to polar (angle dropped from north axis)
    theta = np.pi / 2.0 - declination

    f_plus  = - (1.0/2.0) * (1.0 + cos(theta)*cos(theta)) * \
                cos (2.0 * right_ascension) * cos (2.0 * polarization) - \
                cos(theta) * sin(2.0*right_ascension) * sin (2.0 * polarization)

    f_cross =   (1.0/2.0) * (1.0 + cos(theta)*cos(theta)) * \
                cos (2.0 * right_ascension) * sin (2.0* polarization) - \
                cos(theta) * sin(2.0*right_ascension) * cos (2.0 * polarization)

    return f_plus, f_cross


def ppdets(ifos, separator=', '):
    """Pretty-print a list (or set) of detectors: return a string listing
    the given detectors alphabetically and separated by the given string
    (comma by default).
    """
    if ifos:
        return separator.join(sorted(ifos))
    return 'no detectors'


class NetworkGeometry(object):
    """Vectorized multi-detector network geometry projection helper.

    Computes GMST, sidereal angles, and spatial projection trigonometry once
    for an entire network of detectors, contracting the 3x3 detector response
    matrices and location vectors in a single tensor operation across D detectors.

    Parameters
    ----------
    detectors : list of str or Detector
        The list or tuple of detector names (e.g. ['H1', 'L1', 'V1']) or
        Detector instances in the network.
    reference_time : float, optional
        Reference GPS time for GMST estimation. Defaults to 1126259462.0
        (time of GW150914).
    """

    def __init__(self, detectors, reference_time=1126259462.0):
        self.detectors = [
            d if isinstance(d, Detector) else Detector(
                d, reference_time=reference_time
            )
            for d in detectors
        ]
        self.detector_names = [d.name for d in self.detectors]
        self.reference_time = reference_time
        self.sday = float(sday.si.scale) if reference_time is not None else None
        self.gmst_reference = (
            gmst_accurate(self.reference_time)
            if reference_time is not None else None
        )

        self.responses = np.stack([d.response for d in self.detectors], axis=0)
        self.locations = np.stack([d.location for d in self.detectors], axis=0)

    def __len__(self):
        return len(self.detectors)

    def __getitem__(self, key):
        if isinstance(key, str):
            for d in self.detectors:
                if d.name == key:
                    return d
            raise KeyError(
                f"Detector {key} not found in network {self.detector_names}"
            )
        return self.detectors[key]

    def set_gmst_reference(self):
        if self.reference_time is not None:
            self.sday = float(sday.si.scale)
            self.gmst_reference = gmst_accurate(self.reference_time)
        else:
            raise RuntimeError(
                "Can't get accurate sidereal time without GPS reference time!"
            )

    def gmst_estimate(self, gps_time):
        if self.reference_time is None:
            return gmst_accurate(gps_time)
        if self.gmst_reference is None:
            self.set_gmst_reference()
        dphase = (gps_time - self.reference_time) / self.sday * (2.0 * np.pi)
        gmst = (self.gmst_reference + dphase) % (2.0 * np.pi)
        return gmst

    def antenna_pattern_and_time_delay(
        self, right_ascension, declination, polarization, t_gps
    ):
        """Return antenna pattern (fplus, fcross) and geocentric delay for all D detectors.

        Parameters
        ----------
        right_ascension : float, numpy.ndarray, or torch.Tensor
            The right ascension of the source.
        declination : float, numpy.ndarray, or torch.Tensor
            The declination of the source.
        polarization : float, numpy.ndarray, or torch.Tensor
            The polarization angle of the source.
        t_gps : float, lal.LIGOTimeGPS, numpy.ndarray, or torch.Tensor
            The GPS time.

        Returns
        -------
        fplus : numpy.ndarray or torch.Tensor
            Plus polarization antenna response of shape (D, ...).
        fcross : numpy.ndarray or torch.Tensor
            Cross polarization antenna response of shape (D, ...).
        delay : numpy.ndarray or torch.Tensor
            Geocentric time delay of shape (D, ...).
        """
        if isinstance(t_gps, lal.LIGOTimeGPS):
            t_gps = float(t_gps)

        try:
            import torch
        except ImportError:
            torch = None

        angular_inputs = (right_ascension, declination, polarization)
        torch_inputs = angular_inputs + (t_gps,)
        if torch is not None and any(
                isinstance(value, torch.Tensor) for value in torch_inputs):
            angular_tensors = tuple(
                value for value in angular_inputs
                if isinstance(value, torch.Tensor)
            )
            if any(
                    not torch.is_floating_point(value)
                    for value in angular_tensors):
                raise TypeError(
                    "Torch antenna-pattern angles must be floating"
                )
            tensor_inputs = tuple(
                value for value in torch_inputs
                if isinstance(value, torch.Tensor)
            )
            if any(value.is_complex() for value in tensor_inputs):
                raise TypeError(
                    "Torch antenna-pattern inputs must be real"
                )
            anchor = next(
                value for value in torch_inputs
                if isinstance(value, torch.Tensor)
            )
            dtype = None
            for value in tensor_inputs:
                value_dtype = (
                    value.dtype if torch.is_floating_point(value)
                    else torch.get_default_dtype()
                )
                dtype = (
                    value_dtype if dtype is None
                    else torch.promote_types(dtype, value_dtype)
                )
            if dtype in (torch.float16, torch.bfloat16):
                dtype = torch.float32
            if anchor.device.type == 'mps' and dtype == torch.float64:
                dtype = torch.float32

            time_is_tensor = isinstance(t_gps, torch.Tensor)
            time_is_array = not time_is_tensor and np.ndim(t_gps) > 0
            time_grid = time_is_tensor or time_is_array
            values = list(angular_inputs)
            if time_grid:
                if self.reference_time is None:
                    raise NotImplementedError(
                        "Torch GPS-time grids require a detector GMST "
                        "reference time"
                    )
                if self.gmst_reference is None:
                    self.set_gmst_reference()
                if time_is_tensor:
                    relative_time = t_gps.to(
                        device=anchor.device, dtype=dtype
                    ) - float(self.reference_time)
                else:
                    relative_time = torch.as_tensor(
                        np.asarray(t_gps, dtype=np.float64)
                        - float(self.reference_time),
                        device=anchor.device,
                        dtype=dtype,
                    )
                values.append(relative_time)
            broadcast = torch.broadcast_tensors(*(
                torch.as_tensor(
                    value, device=anchor.device, dtype=dtype
                )
                for value in values
            ))
            angles = broadcast[:3]
            if time_grid:
                relative_time = broadcast[3]
                phase_offsets = (
                    relative_time / float(self.sday) * (2.0 * np.pi)
                )
                gmst_start = self.gmst_reference
            else:
                phase_offsets = torch.zeros_like(angles[0])
                gmst_start = self.gmst_estimate(t_gps)
            return _torch_network_antenna_pattern_and_time_delay(
                self.locations,
                self.responses,
                angles[0],
                angles[1],
                angles[2],
                gmst_start,
                phase_offsets,
            )

        gmst = self.gmst_estimate(t_gps)
        right_ascension, declination, polarization, gmst = (
            np.broadcast_arrays(
                right_ascension, declination, polarization, gmst
            )
        )
        return _numpy_network_antenna_pattern_and_time_delay(
            self.locations,
            self.responses,
            right_ascension,
            declination,
            polarization,
            gmst,
        )

    def antenna_pattern(
        self, right_ascension, declination, polarization, t_gps
    ):
        """Return (fplus, fcross) antenna response for all detectors."""
        fp, fc, _ = self.antenna_pattern_and_time_delay(
            right_ascension, declination, polarization, t_gps
        )
        return fp, fc

    def time_delay_from_earth_center(
        self, right_ascension, declination, t_gps
    ):
        """Return geocentric time delays for all detectors."""
        _, _, delay = self.antenna_pattern_and_time_delay(
            right_ascension, declination, 0.0, t_gps
        )
        return delay

    def response_and_delay(
        self, right_ascension, declination, polarization, t_gps
    ):
        """Alias for antenna_pattern_and_time_delay."""
        return self.antenna_pattern_and_time_delay(
            right_ascension, declination, polarization, t_gps
        )

    def to_dict(self, values):
        """Convert a (D, ...) array/tensor into a dictionary keyed by detector name."""
        return {name: values[i] for i, name in enumerate(self.detector_names)}


__all__ = [
    'Detector',
    'NetworkGeometry',
    'get_available_detectors',
    'get_available_lal_detectors',
    'add_detector_on_earth',
    'single_arm_frequency_response',
    'ppdets',
    'overhead_antenna_pattern',
    'load_detector_config',
    '_ground_detectors',
    '_torch_network_antenna_pattern_and_time_delay',
]
