# Copyright (C) 2016 Christopher M. Biwer
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
"""
Base coordinate transformations, this module provides transformations between
cartesian and spherical coordinates.
"""

import logging

import numpy

from pycbc.types.backend import coerce_torch_values as _torch_values

logger = logging.getLogger("pycbc.coordinates.base")


def cartesian_to_spherical_rho(x, y, z):
    """Calculates the magnitude in spherical coordinates from Cartesian
    coordinates.

    Parameters
    ----------
    x : {numpy.array, torch.Tensor, float}
        X-coordinate.
    y : {numpy.array, torch.Tensor, float}
        Y-coordinate.
    z : {numpy.array, torch.Tensor, float}
        Z-coordinate.

    Returns
    -------
    rho : {numpy.array, torch.Tensor, float}
        The radial amplitude.
    """
    torch, values = _torch_values(x, y, z)
    if torch is not None:
        x, y, z = values
        return torch.sqrt(x**2 + y**2 + z**2)
    return numpy.sqrt(x**2 + y**2 + z**2)


def cartesian_to_spherical_azimuthal(x, y):
    """Calculates the azimuthal angle in spherical coordinates from Cartesian
    coordinates. The azimuthal angle is in [0,2*pi].

    Parameters
    ----------
    x : {numpy.array, torch.Tensor, float}
        X-coordinate.
    y : {numpy.array, torch.Tensor, float}
        Y-coordinate.

    Returns
    -------
    phi : {numpy.array, torch.Tensor, float}
        The azimuthal angle.
    """
    torch, values = _torch_values(x, y)
    if torch is not None:
        x, y = values
        phi = torch.atan2(y, x)
        two_pi = torch.as_tensor(2 * numpy.pi, device=phi.device, dtype=phi.dtype)
        return torch.remainder(phi, two_pi)

    y = float(y) if isinstance(y, int) else y
    phi = numpy.arctan2(y, x)
    return phi % (2 * numpy.pi)


def cartesian_to_spherical_polar(x, y, z):
    """Calculates the polar angle in spherical coordinates from Cartesian
    coordinates. The polar angle is in [0,pi].

    Parameters
    ----------
    x : {numpy.array, torch.Tensor, float}
        X-coordinate.
    y : {numpy.array, torch.Tensor, float}
        Y-coordinate.
    z : {numpy.array, torch.Tensor, float}
        Z-coordinate.

    Returns
    -------
    theta : {numpy.array, torch.Tensor, float}
        The polar angle.
    """
    torch, values = _torch_values(x, y, z)
    if torch is not None:
        x, y, z = values
        rho = torch.sqrt(x**2 + y**2 + z**2)
        nonzero = rho != 0
        safe_rho = torch.where(nonzero, rho, torch.ones_like(rho))
        cosine = torch.where(nonzero, z / safe_rho, torch.ones_like(rho))
        return torch.acos(cosine)

    rho = cartesian_to_spherical_rho(x, y, z)
    if numpy.isscalar(rho):
        return numpy.arccos(z / rho) if rho else 0.0
    else:
        return numpy.arccos(
            numpy.divide(z, rho, out=numpy.ones_like(z), where=rho != 0)
        )


def cartesian_to_spherical(x, y, z):
    """Maps cartesian coordinates (x,y,z) to spherical coordinates
    (rho,phi,theta) where phi is in [0,2*pi] and theta is in [0,pi].

    Parameters
    ----------
    x : {numpy.array, torch.Tensor, float}
        X-coordinate.
    y : {numpy.array, torch.Tensor, float}
        Y-coordinate.
    z : {numpy.array, torch.Tensor, float}
        Z-coordinate.

    Returns
    -------
    rho : {numpy.array, torch.Tensor, float}
        The radial amplitude.
    phi : {numpy.array, torch.Tensor, float}
        The azimuthal angle.
    theta : {numpy.array, torch.Tensor, float}
        The polar angle.
    """
    rho = cartesian_to_spherical_rho(x, y, z)
    phi = cartesian_to_spherical_azimuthal(x, y)
    theta = cartesian_to_spherical_polar(x, y, z)
    return rho, phi, theta


def spherical_to_cartesian(rho, phi, theta):
    """Maps spherical coordinates (rho,phi,theta) to cartesian coordinates
    (x,y,z) where phi is in [0,2*pi] and theta is in [0,pi].

    Parameters
    ----------
    rho : {numpy.array, torch.Tensor, float}
        The radial amplitude.
    phi : {numpy.array, torch.Tensor, float}
        The azimuthal angle.
    theta : {numpy.array, torch.Tensor, float}
        The polar angle.

    Returns
    -------
    x : {numpy.array, torch.Tensor, float}
        X-coordinate.
    y : {numpy.array, torch.Tensor, float}
        Y-coordinate.
    z : {numpy.array, torch.Tensor, float}
        Z-coordinate.
    """
    torch, values = _torch_values(rho, phi, theta)
    if torch is not None:
        rho, phi, theta = values
        x = rho * torch.cos(phi) * torch.sin(theta)
        y = rho * torch.sin(phi) * torch.sin(theta)
        z = rho * torch.cos(theta)
        return x, y, z

    x = rho * numpy.cos(phi) * numpy.sin(theta)
    y = rho * numpy.sin(phi) * numpy.sin(theta)
    z = rho * numpy.cos(theta)
    return x, y, z


__all__ = [
    "cartesian_to_spherical_rho",
    "cartesian_to_spherical_azimuthal",
    "cartesian_to_spherical_polar",
    "cartesian_to_spherical",
    "spherical_to_cartesian",
]
