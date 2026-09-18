# Copyright (C) 2016 Christopher M. Biwer
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

"""
Base coordinate transformations, this module provides transformations between
cartesian and spherical coordinates.
"""
import logging
import numpy

from pycbc.types.backend import coerce_jax_values as _jax_values

logger = logging.getLogger("pycbc.coordinates.base")


def cartesian_to_spherical_rho(x, y, z):
    """Calculates the magnitude in spherical coordinates from Cartesian
    coordinates.

    Parameters
    ----------
    x : {numpy.array, jax.Array, float}
        X-coordinate.
    y : {numpy.array, jax.Array, float}
        Y-coordinate.
    z : {numpy.array, jax.Array, float}
        Z-coordinate.

    Returns
    -------
    rho : {numpy.array, jax.Array, float}
        The radial amplitude.
    """
    jax, values = _jax_values(x, y, z)
    if jax is not None:
        x, y, z = values
        return jax.numpy.sqrt(x**2 + y**2 + z**2)
    return numpy.sqrt(x**2 + y**2 + z**2)


def cartesian_to_spherical_azimuthal(x, y):
    """Calculates the azimuthal angle in spherical coordinates from Cartesian
    coordinates. The azimuthal angle is in [0,2*pi].

    Parameters
    ----------
    x : {numpy.array, jax.Array, float}
        X-coordinate.
    y : {numpy.array, jax.Array, float}
        Y-coordinate.

    Returns
    -------
    phi : {numpy.array, jax.Array, float}
        The azimuthal angle.
    """
    jax, values = _jax_values(x, y)
    if jax is not None:
        x, y = values
        jnp = jax.numpy
        phi = jnp.atan2(y, x)
        two_pi = 2 * numpy.pi
        return jnp.remainder(phi, two_pi)

    y = float(y) if isinstance(y, int) else y
    phi = numpy.arctan2(y, x)
    return phi % (2 * numpy.pi)


def cartesian_to_spherical_polar(x, y, z):
    """Calculates the polar angle in spherical coordinates from Cartesian
    coordinates. The polar angle is in [0,pi].

    Parameters
    ----------
    x : {numpy.array, jax.Array, float}
        X-coordinate.
    y : {numpy.array, jax.Array, float}
        Y-coordinate.
    z : {numpy.array, jax.Array, float}
        Z-coordinate.

    Returns
    -------
    theta : {numpy.array, jax.Array, float}
        The polar angle.
    """
    jax, values = _jax_values(x, y, z)
    if jax is not None:
        x, y, z = values
        jnp = jax.numpy
        rho = jnp.sqrt(x**2 + y**2 + z**2)
        nonzero = rho != 0
        safe_rho = jnp.where(nonzero, rho, 1.0)
        cosine = jnp.where(nonzero, z / safe_rho, 1.0)
        return jnp.arccos(cosine)

    rho = cartesian_to_spherical_rho(x, y, z)
    if numpy.isscalar(rho):
        return numpy.arccos(z / rho) if rho else 0.0
    return numpy.arccos(
        numpy.divide(z, rho, out=numpy.ones_like(z), where=rho != 0)
    )


def cartesian_to_spherical(x, y, z):
    """Maps cartesian coordinates (x,y,z) to spherical coordinates
    (rho,phi,theta) where phi is in [0,2*pi] and theta is in [0,pi].

    Parameters
    ----------
    x : {numpy.array, jax.Array, float}
        X-coordinate.
    y : {numpy.array, jax.Array, float}
        Y-coordinate.
    z : {numpy.array, jax.Array, float}
        Z-coordinate.

    Returns
    -------
    rho : {numpy.array, jax.Array, float}
        The radial amplitude.
    phi : {numpy.array, jax.Array, float}
        The azimuthal angle.
    theta : {numpy.array, jax.Array, float}
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
    rho : {numpy.array, jax.Array, float}
        The radial amplitude.
    phi : {numpy.array, jax.Array, float}
        The azimuthal angle.
    theta : {numpy.array, jax.Array, float}
        The polar angle.

    Returns
    -------
    x : {numpy.array, jax.Array, float}
        X-coordinate.
    y : {numpy.array, jax.Array, float}
        Y-coordinate.
    z : {numpy.array, jax.Array, float}
        Z-coordinate.
    """
    jax, values = _jax_values(rho, phi, theta)
    if jax is not None:
        rho, phi, theta = values
        jnp = jax.numpy
        x = rho * jnp.cos(phi) * jnp.sin(theta)
        y = rho * jnp.sin(phi) * jnp.sin(theta)
        z = rho * jnp.cos(theta)
        return x, y, z

    x = rho * numpy.cos(phi) * numpy.sin(theta)
    y = rho * numpy.sin(phi) * numpy.sin(theta)
    z = rho * numpy.cos(theta)
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
