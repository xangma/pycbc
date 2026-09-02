# Copyright (C) 2023  Shichao Wu, Alex Nitz
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
This module provides coordinate transformations related to space-borne
detectors, such as coordinate transformations between space-borne detectors
and ground-based detectors. Note that current LISA orbit used in this module
is a circular orbit, need to be replaced by a more realistic and general orbit
model in the near future.

The analytic SSB/LISA transformations accept ``torch.Tensor`` inputs and
return tensors on the same device. Transformations that use Astropy remain
CPU-only.
"""

import logging
import numpy as np

from scipy.spatial.transform import Rotation
from scipy.optimize import fsolve
from astropy import units
from astropy.constants import c, au
from astropy.time import Time
from astropy.coordinates import BarycentricMeanEcliptic, PrecessedGeocentric
from astropy.coordinates import get_body_barycentric
from astropy.coordinates import SkyCoord
from astropy.coordinates.builtin_frames import ecliptic_transforms

logger = logging.getLogger('pycbc.coordinates.space')

# This constant makes sure LISA is behind the Earth by 19-23 degrees.
# Making this a stand-alone constant will also make it callable by
# the waveform plugin and PE config file. In the unit of 's'.
TIME_OFFSET_20_DEGREES = 7365189.431698299

# "rotation_matrix_ssb_to_lisa" and "lisa_position_ssb" should be
# more general for other detectors in the near future.


def _torch_module_for(*values):
    """Return Torch when any input is a tensor without importing it eagerly."""
    if not any(type(value).__module__.split(".", 1)[0] == "torch"
               for value in values):
        return None

    import torch

    if any(isinstance(value, torch.Tensor) for value in values):
        return torch
    return None


def _torch_values(*values):
    """Convert mixed scalar/tensor inputs to the first tensor's device."""
    torch = _torch_module_for(*values)
    if torch is None:
        return None, values

    reference = next(value for value in values
                     if isinstance(value, torch.Tensor))
    dtype = reference.dtype
    if not (dtype.is_floating_point or dtype.is_complex):
        dtype = torch.get_default_dtype()
    converted = tuple(
        value.to(device=reference.device, dtype=dtype)
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, device=reference.device, dtype=dtype)
        for value in values
    )
    return torch, converted


def _torch_rotation_matrix_ssb_to_lisa(alpha, torch):
    """Batched Torch form of Rz(alpha) Ry(-pi/3) Rz(-alpha)."""
    cosine = torch.cos(alpha)
    sine = torch.sin(alpha)
    zero = torch.zeros_like(alpha)
    one = torch.ones_like(alpha)

    rz_alpha = torch.stack((
        torch.stack((cosine, -sine, zero), dim=-1),
        torch.stack((sine, cosine, zero), dim=-1),
        torch.stack((zero, zero, one), dim=-1),
    ), dim=-2)
    rz_minus_alpha = torch.stack((
        torch.stack((cosine, sine, zero), dim=-1),
        torch.stack((-sine, cosine, zero), dim=-1),
        torch.stack((zero, zero, one), dim=-1),
    ), dim=-2)

    half = torch.full_like(alpha, 0.5)
    root_three_over_two = torch.full_like(alpha, np.sqrt(3.0) / 2.0)
    ry_minus_sixty = torch.stack((
        torch.stack((half, zero, -root_three_over_two), dim=-1),
        torch.stack((zero, one, zero), dim=-1),
        torch.stack((root_three_over_two, zero, half), dim=-1),
    ), dim=-2)
    return rz_alpha @ ry_minus_sixty @ rz_minus_alpha


def _validate_torch_angles(longitude, latitude, polarization, torch):
    """Apply the public angular-domain checks to tensor inputs."""
    pi = torch.as_tensor(np.pi, device=longitude.device,
                         dtype=longitude.dtype)
    if bool(torch.any((longitude < 0) | (longitude >= 2 * pi))):
        raise ValueError("Longitude should within [0, 2*pi).")
    if bool(torch.any((latitude < -pi / 2) | (latitude > pi / 2))):
        raise ValueError("Latitude should within [-pi/2, pi/2].")
    if bool(torch.any((polarization < 0) | (polarization >= 2 * pi))):
        raise ValueError("Polarization angle should within [0, 2*pi).")


def rotation_matrix_ssb_to_lisa(alpha):
    """ The rotation matrix (of frame basis) from SSB frame to LISA frame.
    This function assumes the angle between LISA plane and the ecliptic
    is 60 degrees, and the period of LISA's self-rotation and orbital
    revolution is both one year.

    Parameters
    ----------
    alpha : float or torch.Tensor
        The angular displacement of LISA in SSB frame.
        In the unit of 'radian'.

    Returns
    -------
    r_total : numpy.array or torch.Tensor
        A 3x3 rotation matrix from SSB frame to LISA frame.
    """
    torch, (alpha_t,) = _torch_values(alpha)
    if torch is not None:
        return _torch_rotation_matrix_ssb_to_lisa(alpha_t, torch)

    r = Rotation.from_rotvec([
        [0, 0, alpha],
        [0, -np.pi/3, 0],
        [0, 0, -alpha]
    ]).as_matrix()
    r_total = np.array(r[0]) @ np.array(r[1]) @ np.array(r[2])

    return r_total


def lisa_position_ssb(t_lisa, t0=TIME_OFFSET_20_DEGREES):
    """ Calculating the position vector and angular displacement of LISA
    in the SSB frame, at a given time. This function assumes LISA's barycenter
    is orbiting around a circular orbit within the ecliptic behind the Earth.
    The period of it is one year.

    Parameters
    ----------
    t_lisa : float or torch.Tensor
        The time when a GW signal arrives at the origin of LISA frame,
        or any other time you want.
    t0 : float
        The initial time offset of LISA, in the unit of 's',
        default is 7365189.431698299. This makes sure LISA is behind
        the Earth by 19-23 degrees.

    Returns
    -------
    (p, alpha) : tuple
    p : numpy.array or torch.Tensor
        The position vector of LISA in the SSB frame. In the unit of 'm'.
    alpha : float or torch.Tensor
        The angular displacement of LISA in the SSB frame.
        In the unit of 'radian'.
    """
    torch, values = _torch_values(t_lisa, t0)
    if torch is not None:
        t_lisa_t, t0_t = torch.broadcast_tensors(*values)
        omega = torch.as_tensor(
            1.99098659277e-7,
            device=t_lisa_t.device,
            dtype=t_lisa_t.dtype,
        )
        two_pi = torch.as_tensor(
            2 * np.pi,
            device=t_lisa_t.device,
            dtype=t_lisa_t.dtype,
        )
        radius = torch.as_tensor(
            au.value,
            device=t_lisa_t.device,
            dtype=t_lisa_t.dtype,
        )
        alpha = torch.remainder(omega * (t_lisa_t + t0_t), two_pi)
        p = torch.stack((
            radius * torch.cos(alpha),
            radius * torch.sin(alpha),
            torch.zeros_like(alpha),
        ), dim=-1).unsqueeze(-1)
        return p, alpha

    OMEGA_0 = 1.99098659277e-7
    R_ORBIT = au.value
    alpha = np.mod(OMEGA_0 * (t_lisa + t0), 2*np.pi)
    p = np.array([[R_ORBIT * np.cos(alpha)],
                  [R_ORBIT * np.sin(alpha)],
                  [0]], dtype=object)
    return (p, alpha)


def localization_to_propagation_vector(longitude, latitude,
                                       use_astropy=True, frame=None):
    """ Converting the sky localization to the corresponding
    propagation unit vector of a GW signal.

    Parameters
    ----------
    longitude : float or torch.Tensor
        The longitude, in the unit of 'radian'.
    latitude : float or torch.Tensor
        The latitude, in the unit of 'radian'.
    use_astropy : bool
        Using Astropy to calculate the sky localization or not.
        Default is True.
    frame : astropy.coordinates
        The frame from astropy.coordinates if use_astropy is True,
        the default is None.

    Returns
    -------
    [[x], [y], [z]] : numpy.array or torch.Tensor
        The propagation unit vector of that GW signal.
    """
    torch, values = _torch_values(longitude, latitude)
    if torch is not None and not use_astropy:
        longitude_t, latitude_t = torch.broadcast_tensors(*values)
        x = -torch.cos(latitude_t) * torch.cos(longitude_t)
        y = -torch.cos(latitude_t) * torch.sin(longitude_t)
        z = -torch.sin(latitude_t)
        vector = torch.stack((x, y, z), dim=-1).unsqueeze(-1)
        return vector / torch.linalg.vector_norm(
            vector, dim=(-2, -1), keepdim=True
        )

    if use_astropy:
        x = -frame.cartesian.x.value
        y = -frame.cartesian.y.value
        z = -frame.cartesian.z.value
    else:
        x = -np.cos(latitude) * np.cos(longitude)
        y = -np.cos(latitude) * np.sin(longitude)
        z = -np.sin(latitude)
    v = np.array([[x], [y], [z]])

    return v / np.linalg.norm(v)


def propagation_vector_to_localization(k, use_astropy=True, frame=None):
    """ Converting the propagation unit vector to the corresponding
    sky localization of a GW signal.

    Parameters
    ----------
    k : numpy.array or torch.Tensor
        The propagation unit vector of a GW signal.
    use_astropy : bool
        Using Astropy to calculate the sky localization or not.
        Default is True.
    frame : astropy.coordinates
        The frame from astropy.coordinates if use_astropy is True,
        the default is None.

    Returns
    -------
    (longitude, latitude) : tuple of float or torch.Tensor
        The sky localization of that GW signal.
    """
    torch, (k_t,) = _torch_values(k)
    if torch is not None and not use_astropy:
        latitude = torch.asin(-k_t[..., 2, 0])
        cosine_latitude = torch.cos(latitude)
        longitude = torch.atan2(
            -k_t[..., 1, 0] / cosine_latitude,
            -k_t[..., 0, 0] / cosine_latitude,
        )
        two_pi = torch.as_tensor(
            2 * np.pi,
            device=k_t.device,
            dtype=k_t.dtype,
        )
        return torch.remainder(longitude, two_pi), latitude

    if use_astropy:
        try:
            longitude = frame.lon.rad
            latitude = frame.lat.rad
        except AttributeError:
            longitude = frame.ra.rad
            latitude = frame.dec.rad
    else:
        # latitude already within [-pi/2, pi/2]
        latitude = np.float64(np.arcsin(-k[2,0]))
        longitude = np.float64(np.arctan2(-k[1,0]/np.cos(latitude),
                               -k[0,0]/np.cos(latitude)))
        # longitude should within [0, 2*pi)
        longitude = np.mod(longitude, 2*np.pi)

    return (longitude, latitude)


def polarization_newframe(polarization, k, rotation_matrix, use_astropy=True,
                          old_frame=None, new_frame=None):
    """ Converting a polarization angle from a frame to a new frame
    by using rotation matrix method.

    Parameters
    ----------
    polarization : float or torch.Tensor
        The polarization angle in the old frame, in the unit of 'radian'.
    k : numpy.array or torch.Tensor
        The propagation unit vector of a GW signal in the old frame.
    rotation_matrix : numpy.array or torch.Tensor
        The rotation matrix (of frame basis) from the old frame to
        the new frame.
    use_astropy : bool
        Using Astropy to calculate the sky localization or not.
        Default is True.
    old_frame : astropy.coordinates
        The frame from astropy.coordinates if use_astropy is True,
        the default is None.
    new_frame : astropy.coordinates
        The frame from astropy.coordinates if use_astropy is True,
        the default is None. The new frame for the new polarization
        angle.

    Returns
    -------
    polarization_new_frame : float or torch.Tensor
        The polarization angle in the new frame of that GW signal.
    """
    torch, values = _torch_values(polarization, k, rotation_matrix)
    if torch is not None and not use_astropy:
        polarization_t, k_t, rotation_t = values
        longitude, _ = propagation_vector_to_localization(
            k_t, use_astropy=False
        )
        zero = torch.zeros_like(longitude)
        u = torch.stack((
            torch.sin(longitude),
            -torch.cos(longitude),
            zero,
        ), dim=-1)
        k_vector = k_t.squeeze(-1)
        cosine = torch.cos(polarization_t).unsqueeze(-1)
        sine = torch.sin(polarization_t).unsqueeze(-1)
        p = (
            u * cosine
            + torch.linalg.cross(k_vector, u, dim=-1) * sine
            + k_vector
            * torch.sum(k_vector * u, dim=-1, keepdim=True)
            * (1 - cosine)
        ).unsqueeze(-1)
        p_newframe = rotation_t.transpose(-1, -2) @ p
        k_newframe = rotation_t.transpose(-1, -2) @ k_t
        longitude_newframe, latitude_newframe = \
            propagation_vector_to_localization(
                k_newframe, use_astropy=False
            )
        u_newframe = torch.stack((
            torch.sin(longitude_newframe),
            -torch.cos(longitude_newframe),
            torch.zeros_like(longitude_newframe),
        ), dim=-1).unsqueeze(-1)
        v_newframe = torch.stack((
            -torch.sin(latitude_newframe) * torch.cos(longitude_newframe),
            -torch.sin(latitude_newframe) * torch.sin(longitude_newframe),
            torch.cos(latitude_newframe),
        ), dim=-1).unsqueeze(-1)
        p_dot_u = torch.sum(p_newframe * u_newframe, dim=(-2, -1))
        p_dot_v = torch.sum(p_newframe * v_newframe, dim=(-2, -1))
        two_pi = torch.as_tensor(
            2 * np.pi,
            device=p_dot_u.device,
            dtype=p_dot_u.dtype,
        )
        return torch.remainder(torch.atan2(p_dot_v, p_dot_u), two_pi)

    longitude, _ = propagation_vector_to_localization(
                        k, use_astropy, old_frame)
    u = np.array([[np.sin(longitude)], [-np.cos(longitude)], [0]])
    rotation_vector = polarization * k
    rotation_polarization = Rotation.from_rotvec(rotation_vector.T[0])
    p = rotation_polarization.apply(u.T[0]).reshape(3, 1)
    p_newframe = rotation_matrix.T @ p
    k_newframe = rotation_matrix.T @ k
    longitude_newframe, latitude_newframe = \
        propagation_vector_to_localization(k_newframe, use_astropy, new_frame)
    u_newframe = np.array([[np.sin(longitude_newframe)],
                           [-np.cos(longitude_newframe)], [0]])
    v_newframe = np.array([
                    [-np.sin(latitude_newframe) * np.cos(longitude_newframe)],
                    [-np.sin(latitude_newframe) * np.sin(longitude_newframe)],
                    [np.cos(latitude_newframe)]])
    p_dot_u_newframe = np.vdot(p_newframe, u_newframe)
    p_dot_v_newframe = np.vdot(p_newframe, v_newframe)
    polarization_new_frame = np.arctan2(p_dot_v_newframe, p_dot_u_newframe)
    polarization_new_frame = np.mod(polarization_new_frame, 2*np.pi)
    # avoid the round error
    if polarization_new_frame == 2*np.pi:
        polarization_new_frame = 0

    return polarization_new_frame


def t_lisa_from_ssb(t_ssb, longitude_ssb, latitude_ssb,
                    t0=TIME_OFFSET_20_DEGREES):
    """ Calculating the time when a GW signal arrives at the barycenter
    of LISA, by using the time and sky localization in SSB frame.

    Parameters
    ----------
    t_ssb : float or torch.Tensor
        The time when a GW signal arrives at the origin of SSB frame.
        In the unit of 's'.
    longitude_ssb : float or torch.Tensor
        The ecliptic longitude of a GW signal in SSB frame.
        In the unit of 'radian'.
    latitude_ssb : float or torch.Tensor
        The ecliptic latitude of a GW signal in SSB frame.
        In the unit of 'radian'.
    t0 : float
        The initial time offset of LISA, in the unit of 's',
        default is 7365189.431698299. This makes sure LISA is behind
        the Earth by 19-23 degrees.

    Returns
    -------
    t_lisa : float or torch.Tensor
        The time when a GW signal arrives at the origin of LISA frame.
    """
    torch, values = _torch_values(
        t_ssb, longitude_ssb, latitude_ssb, t0
    )
    if torch is not None:
        t_ssb_t, longitude_t, latitude_t, t0_t = \
            torch.broadcast_tensors(*values)
        k = localization_to_propagation_vector(
            longitude_t, latitude_t, use_astropy=False
        )
        speed_of_light = torch.as_tensor(
            c.value,
            device=t_ssb_t.device,
            dtype=t_ssb_t.dtype,
        )
        omega = torch.as_tensor(
            1.99098659277e-7,
            device=t_ssb_t.device,
            dtype=t_ssb_t.dtype,
        )
        radius = torch.as_tensor(
            au.value,
            device=t_ssb_t.device,
            dtype=t_ssb_t.dtype,
        )
        t_lisa = t_ssb_t
        for _ in range(5):
            p, alpha = lisa_position_ssb(t_lisa, t0_t)
            delay = torch.sum(k * p, dim=(-2, -1)) / speed_of_light
            velocity = torch.stack((
                -radius * omega * torch.sin(alpha),
                radius * omega * torch.cos(alpha),
                torch.zeros_like(alpha),
            ), dim=-1).unsqueeze(-1)
            derivative = 1 - torch.sum(
                k * velocity, dim=(-2, -1)
            ) / speed_of_light
            residual = (t_lisa - t_ssb_t) - delay
            t_lisa = t_lisa - residual / derivative
        return t_lisa

    k = localization_to_propagation_vector(
            longitude_ssb, latitude_ssb, use_astropy=False)

    def equation(t_lisa):
        # LISA is moving, when GW arrives at LISA center,
        # time is t_lisa, not t_ssb.
        p = lisa_position_ssb(t_lisa, t0)[0]
        return t_lisa - t_ssb - np.vdot(k, p) / c.value

    return fsolve(equation, t_ssb)[0]


def t_ssb_from_t_lisa(t_lisa, longitude_ssb, latitude_ssb,
                      t0=TIME_OFFSET_20_DEGREES):
    """ Calculating the time when a GW signal arrives at the barycenter
    of SSB, by using the time in LISA frame and sky localization in SSB frame.

    Parameters
    ----------
    t_lisa : float or torch.Tensor
        The time when a GW signal arrives at the origin of LISA frame.
        In the unit of 's'.
    longitude_ssb : float or torch.Tensor
        The ecliptic longitude of a GW signal in SSB frame.
        In the unit of 'radian'.
    latitude_ssb : float or torch.Tensor
        The ecliptic latitude of a GW signal in SSB frame.
        In the unit of 'radian'.
    t0 : float
        The initial time offset of LISA, in the unit of 's',
        default is 7365189.431698299. This makes sure LISA is behind
        the Earth by 19-23 degrees.

    Returns
    -------
    t_ssb : float or torch.Tensor
        The time when a GW signal arrives at the origin of SSB frame.
    """
    torch, values = _torch_values(
        t_lisa, longitude_ssb, latitude_ssb, t0
    )
    if torch is not None:
        t_lisa_t, longitude_t, latitude_t, t0_t = \
            torch.broadcast_tensors(*values)
        k = localization_to_propagation_vector(
            longitude_t, latitude_t, use_astropy=False
        )
        p = lisa_position_ssb(t_lisa_t, t0_t)[0]
        speed_of_light = torch.as_tensor(
            c.value,
            device=t_lisa_t.device,
            dtype=t_lisa_t.dtype,
        )
        return t_lisa_t - torch.sum(
            k * p, dim=(-2, -1)
        ) / speed_of_light

    k = localization_to_propagation_vector(
            longitude_ssb, latitude_ssb, use_astropy=False)
    # LISA is moving, when GW arrives at LISA center,
    # time is t_lisa, not t_ssb.
    p = lisa_position_ssb(t_lisa, t0)[0]

    def equation(t_ssb):
        return t_lisa - t_ssb - np.vdot(k, p) / c.value

    return fsolve(equation, t_lisa)[0]


def ssb_to_lisa(t_ssb, longitude_ssb, latitude_ssb, polarization_ssb,
                t0=TIME_OFFSET_20_DEGREES):
    """ Converting the arrive time, the sky localization, and the polarization
    from the SSB frame to the LISA frame.

    Parameters
    ----------
    t_ssb : float, numpy.array, or torch.Tensor
        The time when a GW signal arrives at the origin of SSB frame.
        In the unit of 's'.
    longitude_ssb : float, numpy.array, or torch.Tensor
        The ecliptic longitude of a GW signal in SSB frame.
        In the unit of 'radian'.
    latitude_ssb : float, numpy.array, or torch.Tensor
        The ecliptic latitude of a GW signal in SSB frame.
        In the unit of 'radian'.
    polarization_ssb : float, numpy.array, or torch.Tensor
        The polarization angle of a GW signal in SSB frame.
        In the unit of 'radian'.
    t0 : float
        The initial time offset of LISA, in the unit of 's',
        default is 7365189.431698299. This makes sure LISA is behind
        the Earth by 19-23 degrees.

    Returns
    -------
    (t_lisa, longitude_lisa, latitude_lisa, polarization_lisa) : tuple
    t_lisa : float, numpy.array, or torch.Tensor
        The time when a GW signal arrives at the origin of LISA frame.
        In the unit of 's'.
    longitude_lisa : float, numpy.array, or torch.Tensor
        The longitude of a GW signal in LISA frame, in the unit of 'radian'.
    latitude_lisa : float, numpy.array, or torch.Tensor
        The latitude of a GW signal in LISA frame, in the unit of 'radian'.
    polarization_lisa : float, numpy.array, or torch.Tensor
        The polarization angle of a GW signal in LISA frame.
        In the unit of 'radian'.
    """
    torch, values = _torch_values(
        t_ssb, longitude_ssb, latitude_ssb, polarization_ssb, t0
    )
    if torch is not None:
        t_ssb_t, longitude_t, latitude_t, polarization_t, t0_t = \
            torch.broadcast_tensors(*values)
        _validate_torch_angles(
            longitude_t, latitude_t, polarization_t, torch
        )
        t_lisa = t_lisa_from_ssb(
            t_ssb_t, longitude_t, latitude_t, t0_t
        )
        k_ssb = localization_to_propagation_vector(
            longitude_t, latitude_t, use_astropy=False
        )
        alpha = lisa_position_ssb(t_lisa, t0_t)[1]
        rotation_matrix_lisa = rotation_matrix_ssb_to_lisa(alpha)
        k_lisa = rotation_matrix_lisa.transpose(-1, -2) @ k_ssb
        longitude_lisa, latitude_lisa = \
            propagation_vector_to_localization(
                k_lisa, use_astropy=False
            )
        polarization_lisa = polarization_newframe(
            polarization_t,
            k_ssb,
            rotation_matrix_lisa,
            use_astropy=False,
        )
        return (
            t_lisa,
            longitude_lisa,
            latitude_lisa,
            polarization_lisa,
        )

    if not isinstance(t_ssb, np.ndarray):
        t_ssb = np.array([t_ssb])
    if not isinstance(longitude_ssb, np.ndarray):
        longitude_ssb = np.array([longitude_ssb])
    if not isinstance(latitude_ssb, np.ndarray):
        latitude_ssb = np.array([latitude_ssb])
    if not isinstance(polarization_ssb, np.ndarray):
        polarization_ssb = np.array([polarization_ssb])
    num = len(t_ssb)
    t_lisa, longitude_lisa = np.zeros(num), np.zeros(num)
    latitude_lisa, polarization_lisa = np.zeros(num), np.zeros(num)

    for i in range(num):
        if longitude_ssb[i] < 0 or longitude_ssb[i] >= 2*np.pi:
            raise ValueError("Longitude should within [0, 2*pi).")
        if latitude_ssb[i] < -np.pi/2 or latitude_ssb[i] > np.pi/2:
            raise ValueError("Latitude should within [-pi/2, pi/2].")
        if polarization_ssb[i] < 0 or polarization_ssb[i] >= 2*np.pi:
            raise ValueError("Polarization angle should within [0, 2*pi).")
        t_lisa[i] = t_lisa_from_ssb(t_ssb[i], longitude_ssb[i],
                                    latitude_ssb[i], t0)
        k_ssb = localization_to_propagation_vector(
                    longitude_ssb[i], latitude_ssb[i], use_astropy=False)
        # Although t_lisa calculated above using the corrected LISA position
        # vector by adding t0, it corresponds to the true t_ssb, not t_ssb+t0,
        # we need to include t0 again to correct LISA position.
        alpha = lisa_position_ssb(t_lisa[i], t0)[1]
        rotation_matrix_lisa = rotation_matrix_ssb_to_lisa(alpha)
        k_lisa = rotation_matrix_lisa.T @ k_ssb
        longitude_lisa[i], latitude_lisa[i] = \
            propagation_vector_to_localization(k_lisa, use_astropy=False)
        polarization_lisa[i] = polarization_newframe(
            polarization_ssb[i], k_ssb, rotation_matrix_lisa,
            use_astropy=False)

    if num == 1:
        params_lisa = (t_lisa[0], longitude_lisa[0],
                       latitude_lisa[0], polarization_lisa[0])
    else:
        params_lisa = (t_lisa, longitude_lisa,
                       latitude_lisa, polarization_lisa)

    return params_lisa


def lisa_to_ssb(t_lisa, longitude_lisa, latitude_lisa, polarization_lisa,
                t0=TIME_OFFSET_20_DEGREES):
    """ Converting the arrive time, the sky localization, and the polarization
    from the LISA frame to the SSB frame.

    Parameters
    ----------
    t_lisa : float, numpy.array, or torch.Tensor
        The time when a GW signal arrives at the origin of LISA frame.
        In the unit of 's'.
    longitude_lisa : float, numpy.array, or torch.Tensor
        The longitude of a GW signal in LISA frame, in the unit of 'radian'.
    latitude_lisa : float, numpy.array, or torch.Tensor
        The latitude of a GW signal in LISA frame, in the unit of 'radian'.
    polarization_lisa : float, numpy.array, or torch.Tensor
        The polarization angle of a GW signal in LISA frame.
        In the unit of 'radian'.
    t0 : float
        The initial time offset of LISA, in the unit of 's',
        default is 7365189.431698299. This makes sure LISA is behind
        the Earth by 19-23 degrees.

    Returns
    -------
    (t_ssb, longitude_ssb, latitude_ssb, polarization_ssb) : tuple
    t_ssb : float, numpy.array, or torch.Tensor
        The time when a GW signal arrives at the origin of SSB frame.
        In the unit of 's'.
    longitude_ssb : float, numpy.array, or torch.Tensor
        The ecliptic longitude of a GW signal in SSB frame.
        In the unit of 'radian'.
    latitude_ssb : float, numpy.array, or torch.Tensor
        The ecliptic latitude of a GW signal in SSB frame.
        In the unit of 'radian'.
    polarization_ssb : float, numpy.array, or torch.Tensor
        The polarization angle of a GW signal in SSB frame.
        In the unit of 'radian'.
    """
    torch, values = _torch_values(
        t_lisa, longitude_lisa, latitude_lisa, polarization_lisa, t0
    )
    if torch is not None:
        t_lisa_t, longitude_t, latitude_t, polarization_t, t0_t = \
            torch.broadcast_tensors(*values)
        _validate_torch_angles(
            longitude_t, latitude_t, polarization_t, torch
        )
        k_lisa = localization_to_propagation_vector(
            longitude_t, latitude_t, use_astropy=False
        )
        alpha = lisa_position_ssb(t_lisa_t, t0_t)[1]
        rotation_matrix_lisa = rotation_matrix_ssb_to_lisa(alpha)
        k_ssb = rotation_matrix_lisa @ k_lisa
        longitude_ssb, latitude_ssb = \
            propagation_vector_to_localization(
                k_ssb, use_astropy=False
            )
        t_ssb = t_ssb_from_t_lisa(
            t_lisa_t, longitude_ssb, latitude_ssb, t0_t
        )
        polarization_ssb = polarization_newframe(
            polarization_t,
            k_lisa,
            rotation_matrix_lisa.transpose(-1, -2),
            use_astropy=False,
        )
        return (
            t_ssb,
            longitude_ssb,
            latitude_ssb,
            polarization_ssb,
        )

    if not isinstance(t_lisa, np.ndarray):
        t_lisa = np.array([t_lisa])
    if not isinstance(longitude_lisa, np.ndarray):
        longitude_lisa = np.array([longitude_lisa])
    if not isinstance(latitude_lisa, np.ndarray):
        latitude_lisa = np.array([latitude_lisa])
    if not isinstance(polarization_lisa, np.ndarray):
        polarization_lisa = np.array([polarization_lisa])
    num = len(t_lisa)
    t_ssb, longitude_ssb = np.zeros(num), np.zeros(num)
    latitude_ssb, polarization_ssb = np.zeros(num), np.zeros(num)

    for i in range(num):
        if longitude_lisa[i] < 0 or longitude_lisa[i] >= 2*np.pi:
            raise ValueError("Longitude should within [0, 2*pi).")
        if latitude_lisa[i] < -np.pi/2 or latitude_lisa[i] > np.pi/2:
            raise ValueError("Latitude should within [-pi/2, pi/2].")
        if polarization_lisa[i] < 0 or polarization_lisa[i] >= 2*np.pi:
            raise ValueError("Polarization angle should within [0, 2*pi).")
        k_lisa = localization_to_propagation_vector(
                    longitude_lisa[i], latitude_lisa[i], use_astropy=False)
        alpha = lisa_position_ssb(t_lisa[i], t0)[1]
        rotation_matrix_lisa = rotation_matrix_ssb_to_lisa(alpha)
        k_ssb = rotation_matrix_lisa @ k_lisa
        longitude_ssb[i], latitude_ssb[i] = \
            propagation_vector_to_localization(k_ssb, use_astropy=False)
        t_ssb[i] = t_ssb_from_t_lisa(t_lisa[i], longitude_ssb[i],
                                     latitude_ssb[i], t0)
        polarization_ssb[i] = polarization_newframe(
            polarization_lisa[i], k_lisa, rotation_matrix_lisa.T,
            use_astropy=False)

    if num == 1:
        params_ssb = (t_ssb[0], longitude_ssb[0],
                      latitude_ssb[0], polarization_ssb[0])
    else:
        params_ssb = (t_ssb, longitude_ssb,
                      latitude_ssb, polarization_ssb)

    return params_ssb


def rotation_matrix_ssb_to_geo(epsilon=np.deg2rad(23.439281)):
    """ The rotation matrix (of frame basis) from SSB frame to
    geocentric frame.

    Parameters
    ----------
    epsilon : float
        The Earth's axial tilt (obliquity), in the unit of 'radian'.

    Returns
    -------
    r : numpy.array
        A 3x3 rotation matrix from SSB frame to geocentric frame.
    """
    r = Rotation.from_rotvec([
        [-epsilon, 0, 0]
    ]).as_matrix()

    return np.array(r[0])


def earth_position_ssb(t_geo):
    """ Calculating the position vector and angular displacement of the Earth
    in the SSB frame, at a given time. By using Astropy.

    Parameters
    ----------
    t_geo : float
        The time when a GW signal arrives at the origin of geocentric frame,
        or any other time you want.

    Returns
    -------
    (p, alpha) : tuple
    p : numpy.array
        The position vector of the Earth in the SSB frame. In the unit of 'm'.
    alpha : float
        The angular displacement of the Earth in the SSB frame.
        In the unit of 'radian'.
    """
    t = Time(t_geo, format='gps')
    pos = get_body_barycentric('earth', t)
    # BarycentricMeanEcliptic doesn't have obstime attribute,
    # it's a good inertial frame, but ICRS is not.
    icrs_coord = SkyCoord(pos, frame='icrs', obstime=t)
    bme_coord = icrs_coord.transform_to(
                    BarycentricMeanEcliptic(equinox='J2000'))
    x = bme_coord.cartesian.x.to(units.m).value
    y = bme_coord.cartesian.y.to(units.m).value
    z = bme_coord.cartesian.z.to(units.m).value
    p = np.array([[x], [y], [z]])
    alpha = bme_coord.lon.rad

    return (p, alpha)


def t_geo_from_ssb(t_ssb, longitude_ssb, latitude_ssb,
                   use_astropy=True, frame=None):
    """ Calculating the time when a GW signal arrives at the barycenter
    of the Earth, by using the time and sky localization in SSB frame.

    Parameters
    ----------
    t_ssb : float
        The time when a GW signal arrives at the origin of SSB frame.
        In the unit of 's'.
    longitude_ssb : float
        The ecliptic longitude of a GW signal in SSB frame.
        In the unit of 'radian'.
    latitude_ssb : float
        The ecliptic latitude of a GW signal in SSB frame.
        In the unit of 'radian'.

    Returns
    -------
    t_geo : float
        The time when a GW signal arrives at the origin of geocentric frame.
    """
    k = localization_to_propagation_vector(
            longitude_ssb, latitude_ssb, use_astropy, frame)

    def equation(t_geo):
        # Earth is moving, when GW arrives at Earth center,
        # time is t_geo, not t_ssb.
        p = earth_position_ssb(t_geo)[0]
        return t_geo - t_ssb - np.vdot(k, p) / c.value

    return fsolve(equation, t_ssb)[0]


def t_ssb_from_t_geo(t_geo, longitude_ssb, latitude_ssb,
                     use_astropy=True, frame=None):
    """ Calculating the time when a GW signal arrives at the barycenter
    of SSB, by using the time in geocentric frame and sky localization
    in SSB frame.

    Parameters
    ----------
    t_geo : float
        The time when a GW signal arrives at the origin of geocentric frame.
        In the unit of 's'.
    longitude_ssb : float
        The ecliptic longitude of a GW signal in SSB frame.
        In the unit of 'radian'.
    latitude_ssb : float
        The ecliptic latitude of a GW signal in SSB frame.
        In the unit of 'radian'.

    Returns
    -------
    t_ssb : float
        The time when a GW signal arrives at the origin of SSB frame.
    """
    k = localization_to_propagation_vector(
            longitude_ssb, latitude_ssb, use_astropy, frame)
    # Earth is moving, when GW arrives at Earth center,
    # time is t_geo, not t_ssb.
    p = earth_position_ssb(t_geo)[0]

    def equation(t_ssb):
        return t_geo - t_ssb - np.vdot(k, p) / c.value

    return fsolve(equation, t_geo)[0]


def ssb_to_geo(t_ssb, longitude_ssb, latitude_ssb, polarization_ssb,
               use_astropy=True):
    """ Converting the arrive time, the sky localization, and the polarization
    from the SSB frame to the geocentric frame.

    Parameters
    ----------
    t_ssb : float or numpy.array
        The time when a GW signal arrives at the origin of SSB frame.
        In the unit of 's'.
    longitude_ssb : float or numpy.array
        The ecliptic longitude of a GW signal in SSB frame.
        In the unit of 'radian'.
    latitude_ssb : float or numpy.array
        The ecliptic latitude of a GW signal in SSB frame.
        In the unit of 'radian'.
    polarization_ssb : float or numpy.array
        The polarization angle of a GW signal in SSB frame.
        In the unit of 'radian'.
    use_astropy : bool
        Using Astropy to calculate the sky localization or not.
        Default is True.

    Returns
    -------
    (t_geo, longitude_geo, latitude_geo, polarization_geo) : tuple
    t_geo : float or numpy.array
        The time when a GW signal arrives at the origin of geocentric frame.
        In the unit of 's'.
    longitude_geo : float or numpy.array
        The longitude of a GW signal in geocentric frame.
        In the unit of 'radian'.
    latitude_geo : float or numpy.array
        The latitude of a GW signal in geocentric frame.
        In the unit of 'radian'.
    polarization_geo : float or numpy.array
        The polarization angle of a GW signal in geocentric frame.
        In the unit of 'radian'.
    """
    if not isinstance(t_ssb, np.ndarray):
        t_ssb = np.array([t_ssb])
    if not isinstance(longitude_ssb, np.ndarray):
        longitude_ssb = np.array([longitude_ssb])
    if not isinstance(latitude_ssb, np.ndarray):
        latitude_ssb = np.array([latitude_ssb])
    if not isinstance(polarization_ssb, np.ndarray):
        polarization_ssb = np.array([polarization_ssb])
    num = len(t_ssb)
    t_geo = np.full(num, np.nan)
    longitude_geo = np.full(num, np.nan)
    latitude_geo = np.full(num, np.nan)
    polarization_geo = np.full(num, np.nan)

    for i in range(num):
        if longitude_ssb[i] < 0 or longitude_ssb[i] >= 2*np.pi:
            raise ValueError("Longitude should within [0, 2*pi).")
        if latitude_ssb[i] < -np.pi/2 or latitude_ssb[i] > np.pi/2:
            raise ValueError("Latitude should within [-pi/2, pi/2].")
        if polarization_ssb[i] < 0 or polarization_ssb[i] >= 2*np.pi:
            raise ValueError("Polarization angle should within [0, 2*pi).")

        if use_astropy:
            # BarycentricMeanEcliptic doesn't have obstime attribute,
            # it's a good inertial frame, but PrecessedGeocentric is not.
            bme_coord = BarycentricMeanEcliptic(
                            lon=longitude_ssb[i]*units.radian,
                            lat=latitude_ssb[i]*units.radian,
                            equinox='J2000')
            t_geo[i] = t_geo_from_ssb(t_ssb[i], longitude_ssb[i],
                                      latitude_ssb[i], use_astropy, bme_coord)
            geo_sky = bme_coord.transform_to(PrecessedGeocentric(
                equinox='J2000', obstime=Time(t_geo[i], format='gps')))
            longitude_geo[i] = geo_sky.ra.rad
            latitude_geo[i] = geo_sky.dec.rad
            k_geo = localization_to_propagation_vector(
                        longitude_geo[i], latitude_geo[i],
                        use_astropy, geo_sky)
            k_ssb = localization_to_propagation_vector(
                        None, None, use_astropy, bme_coord)
            rotation_matrix_geo = \
                ecliptic_transforms.icrs_to_baryecliptic(
                    from_coo=None,
                    to_frame=BarycentricMeanEcliptic(equinox='J2000'))
            polarization_geo[i] = polarization_newframe(
                                    polarization_ssb[i], k_ssb,
                                    rotation_matrix_geo, use_astropy,
                                    old_frame=bme_coord,
                                    new_frame=geo_sky)
        else:
            t_geo[i] = t_geo_from_ssb(t_ssb[i], longitude_ssb[i],
                                      latitude_ssb[i], use_astropy)
            rotation_matrix_geo = rotation_matrix_ssb_to_geo()
            k_ssb = localization_to_propagation_vector(
                        longitude_ssb[i], latitude_ssb[i],
                        use_astropy)
            k_geo = rotation_matrix_geo.T @ k_ssb
            longitude_geo[i], latitude_geo[i] = \
                propagation_vector_to_localization(k_geo, use_astropy)
            polarization_geo[i] = polarization_newframe(
                                    polarization_ssb[i], k_ssb,
                                    rotation_matrix_geo, use_astropy)

        # As mentioned in LDC manual, the p,q vectors are opposite between
        # LDC and LAL conventions, see Sec 4.1.5 in <LISA-LCST-SGS-MAN-001>.
        polarization_geo[i] = np.mod(polarization_geo[i]+np.pi, 2*np.pi)

    if num == 1:
        params_geo = (t_geo[0], longitude_geo[0],
                      latitude_geo[0], polarization_geo[0])
    else:
        params_geo = (t_geo, longitude_geo,
                      latitude_geo, polarization_geo)

    return params_geo


def geo_to_ssb(t_geo, longitude_geo, latitude_geo, polarization_geo,
               use_astropy=True):
    """ Converting the arrive time, the sky localization, and the polarization
    from the geocentric frame to the SSB frame.

    Parameters
    ----------
    t_geo : float or numpy.array
        The time when a GW signal arrives at the origin of geocentric frame.
        In the unit of 's'.
    longitude_geo : float or numpy.array
        The longitude of a GW signal in geocentric frame.
        In the unit of 'radian'.
    latitude_geo : float or numpy.array
        The latitude of a GW signal in geocentric frame.
        In the unit of 'radian'.
    polarization_geo : float or numpy.array
        The polarization angle of a GW signal in geocentric frame.
        In the unit of 'radian'.
    use_astropy : bool
        Using Astropy to calculate the sky localization or not.
        Default is True.

    Returns
    -------
    (t_ssb, longitude_ssb, latitude_ssb, polarization_ssb) : tuple
    t_ssb : float or numpy.array
        The time when a GW signal arrives at the origin of SSB frame.
        In the unit of 's'.
    longitude_ssb : float or numpy.array
        The ecliptic longitude of a GW signal in SSB frame.
        In the unit of 'radian'.
    latitude_ssb : float or numpy.array
        The ecliptic latitude of a GW signal in SSB frame.
        In the unit of 'radian'.
    polarization_ssb : float or numpy.array
        The polarization angle of a GW signal in SSB frame.
        In the unit of 'radian'.
    """
    if not isinstance(t_geo, np.ndarray):
        t_geo = np.array([t_geo])
    if not isinstance(longitude_geo, np.ndarray):
        longitude_geo = np.array([longitude_geo])
    if not isinstance(latitude_geo, np.ndarray):
        latitude_geo = np.array([latitude_geo])
    if not isinstance(polarization_geo, np.ndarray):
        polarization_geo = np.array([polarization_geo])
    num = len(t_geo)
    t_ssb = np.full(num, np.nan)
    longitude_ssb = np.full(num, np.nan)
    latitude_ssb = np.full(num, np.nan)
    polarization_ssb = np.full(num, np.nan)

    for i in range(num):
        if longitude_geo[i] < 0 or longitude_geo[i] >= 2*np.pi:
            raise ValueError("Longitude should within [0, 2*pi).")
        if latitude_geo[i] < -np.pi/2 or latitude_geo[i] > np.pi/2:
            raise ValueError("Latitude should within [-pi/2, pi/2].")
        if polarization_geo[i] < 0 or polarization_geo[i] >= 2*np.pi:
            raise ValueError("Polarization angle should within [0, 2*pi).")

        if use_astropy:
            # BarycentricMeanEcliptic doesn't have obstime attribute,
            # it's a good inertial frame, but PrecessedGeocentric is not.
            geo_coord = PrecessedGeocentric(
                            ra=longitude_geo[i]*units.radian,
                            dec=latitude_geo[i]*units.radian,
                            equinox='J2000',
                            obstime=Time(t_geo[i], format='gps'))
            ssb_sky = geo_coord.transform_to(
                        BarycentricMeanEcliptic(equinox='J2000'))
            longitude_ssb[i] = ssb_sky.lon.rad
            latitude_ssb[i] = ssb_sky.lat.rad
            k_ssb = localization_to_propagation_vector(
                        longitude_ssb[i], latitude_ssb[i],
                        use_astropy, ssb_sky)
            k_geo = localization_to_propagation_vector(
                None, None, use_astropy, geo_coord)
            rotation_matrix_geo = \
                ecliptic_transforms.icrs_to_baryecliptic(
                    from_coo=None,
                    to_frame=BarycentricMeanEcliptic(equinox='J2000'))
            t_ssb[i] = t_ssb_from_t_geo(t_geo[i], longitude_ssb[i],
                                        latitude_ssb[i], use_astropy,
                                        ssb_sky)
            polarization_ssb[i] = polarization_newframe(
                                    polarization_geo[i], k_geo,
                                    rotation_matrix_geo.T,
                                    use_astropy,
                                    old_frame=geo_coord,
                                    new_frame=ssb_sky)
        else:
            rotation_matrix_geo = rotation_matrix_ssb_to_geo()
            k_geo = localization_to_propagation_vector(
                        longitude_geo[i], latitude_geo[i], use_astropy)
            k_ssb = rotation_matrix_geo @ k_geo
            longitude_ssb[i], latitude_ssb[i] = \
                propagation_vector_to_localization(k_ssb, use_astropy)
            t_ssb[i] = t_ssb_from_t_geo(t_geo[i], longitude_ssb[i],
                                        latitude_ssb[i], use_astropy)
            polarization_ssb[i] = polarization_newframe(
                                    polarization_geo[i], k_geo,
                                    rotation_matrix_geo.T, use_astropy)

        # As mentioned in LDC manual, the p,q vectors are opposite between
        # LDC and LAL conventions, see Sec 4.1.5 in <LISA-LCST-SGS-MAN-001>.
        polarization_ssb[i] = np.mod(polarization_ssb[i]-np.pi, 2*np.pi)

    if num == 1:
        params_ssb = (t_ssb[0], longitude_ssb[0],
                      latitude_ssb[0], polarization_ssb[0])
    else:
        params_ssb = (t_ssb, longitude_ssb,
                      latitude_ssb, polarization_ssb)

    return params_ssb


def lisa_to_geo(t_lisa, longitude_lisa, latitude_lisa, polarization_lisa,
                t0=TIME_OFFSET_20_DEGREES, use_astropy=True):
    """ Converting the arrive time, the sky localization, and the polarization
    from the LISA frame to the geocentric frame.

    Parameters
    ----------
    t_lisa : float or numpy.array
        The time when a GW signal arrives at the origin of LISA frame.
        In the unit of 's'.
    longitude_lisa : float or numpy.array
        The longitude of a GW signal in LISA frame, in the unit of 'radian'.
    latitude_lisa : float or numpy.array
        The latitude of a GW signal in LISA frame, in the unit of 'radian'.
    polarization_lisa : float or numpy.array
        The polarization angle of a GW signal in LISA frame.
        In the unit of 'radian'.
    t0 : float
        The initial time offset of LISA, in the unit of 's',
        default is 7365189.431698299. This makes sure LISA is behind
        the Earth by 19-23 degrees.
    use_astropy : bool
        Using Astropy to calculate the sky localization or not.
        Default is True.

    Returns
    -------
    (t_geo, longitude_geo, latitude_geo, polarization_geo) : tuple
    t_geo : float or numpy.array
        The time when a GW signal arrives at the origin of geocentric frame.
        In the unit of 's'.
    longitude_geo : float or numpy.array
        The ecliptic longitude of a GW signal in geocentric frame.
        In the unit of 'radian'.
    latitude_geo : float or numpy.array
        The ecliptic latitude of a GW signal in geocentric frame.
        In the unit of 'radian'.
    polarization_geo : float or numpy.array
        The polarization angle of a GW signal in geocentric frame.
        In the unit of 'radian'.
    """
    t_ssb, longitude_ssb, latitude_ssb, polarization_ssb = lisa_to_ssb(
        t_lisa, longitude_lisa, latitude_lisa, polarization_lisa, t0)
    t_geo, longitude_geo, latitude_geo, polarization_geo = ssb_to_geo(
        t_ssb, longitude_ssb, latitude_ssb, polarization_ssb, use_astropy)

    return (t_geo, longitude_geo, latitude_geo, polarization_geo)


def geo_to_lisa(t_geo, longitude_geo, latitude_geo, polarization_geo,
                t0=TIME_OFFSET_20_DEGREES, use_astropy=True):
    """ Converting the arrive time, the sky localization, and the polarization
    from the geocentric frame to the LISA frame.

    Parameters
    ----------
    t_geo : float or numpy.array
        The time when a GW signal arrives at the origin of geocentric frame.
        In the unit of 's'.
    longitude_geo : float or numpy.array
        The longitude of a GW signal in geocentric frame.
        In the unit of 'radian'.
    latitude_geo : float or numpy.array
        The latitude of a GW signal in geocentric frame.
        In the unit of 'radian'.
    polarization_geo : float or numpy.array
        The polarization angle of a GW signal in geocentric frame.
        In the unit of 'radian'.
    t0 : float
        The initial time offset of LISA, in the unit of 's',
        default is 7365189.431698299. This makes sure LISA is behind
        the Earth by 19-23 degrees.
    use_astropy : bool
        Using Astropy to calculate the sky localization or not.
        Default is True.

    Returns
    -------
    (t_lisa, longitude_lisa, latitude_lisa, polarization_lisa) : tuple
    t_lisa : float or numpy.array
        The time when a GW signal arrives at the origin of LISA frame.
        In the unit of 's'.
    longitude_lisa : float or numpy.array
        The longitude of a GW signal in LISA frame, in the unit of 'radian'.
    latitude_lisa : float or numpy.array
        The latitude of a GW signal in LISA frame, in the unit of 'radian'.
    polarization_geo : float or numpy.array
        The polarization angle of a GW signal in LISA frame.
        In the unit of 'radian'.
    """
    t_ssb, longitude_ssb, latitude_ssb, polarization_ssb = geo_to_ssb(
        t_geo, longitude_geo, latitude_geo, polarization_geo, use_astropy)
    t_lisa, longitude_lisa, latitude_lisa, polarization_lisa = ssb_to_lisa(
        t_ssb, longitude_ssb, latitude_ssb, polarization_ssb, t0)

    return (t_lisa, longitude_lisa, latitude_lisa, polarization_lisa)


__all__ = ['TIME_OFFSET_20_DEGREES',
           'localization_to_propagation_vector',
           'propagation_vector_to_localization', 'polarization_newframe',
           't_lisa_from_ssb', 't_ssb_from_t_lisa',
           'ssb_to_lisa', 'lisa_to_ssb',
           'rotation_matrix_ssb_to_lisa', 'rotation_matrix_ssb_to_geo',
           'lisa_position_ssb', 'earth_position_ssb',
           't_geo_from_ssb', 't_ssb_from_t_geo', 'ssb_to_geo', 'geo_to_ssb',
           'lisa_to_geo', 'geo_to_lisa',
           ]
