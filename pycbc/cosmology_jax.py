# Copyright (C) 2026 PyCBC developers
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

"""JAX implementations for cosmological distance and volume interpolation."""

import jax.numpy as jnp


def create_jax_grids(nearby_grid, faraway_grid):
    """Convert numpy interpolation grids to JAX arrays."""
    return (
        jnp.asarray(nearby_grid[0], dtype=float),
        jnp.asarray(nearby_grid[1], dtype=float),
        jnp.asarray(faraway_grid[0], dtype=float),
        jnp.asarray(faraway_grid[1], dtype=float),
    )


def get_redshift_jax(dist, grids, default_maxz):
    """Interpolate redshift from luminosity distance using JAX."""
    if jnp.issubdtype(dist.dtype, jnp.complexfloating):
        raise TypeError("distance must be real")
    if not jnp.issubdtype(dist.dtype, jnp.floating):
        dist = dist.astype(float)
    if not bool(jnp.all(jnp.isfinite(dist) & (dist >= 0))):
        raise ValueError("distance must be finite and >= 0")

    nearby_d, nearby_z, faraway_d, faraway_z = grids
    if bool(jnp.any(dist > faraway_d[-1])):
        raise ValueError(
            "JAX distances must be within the precomputed redshift "
            f"range z <= {default_maxz:g}"
        )

    shape = dist.shape
    flat_dist = dist.reshape(-1)
    nearby_result = jnp.interp(flat_dist, nearby_d, nearby_z)
    faraway_result = jnp.interp(flat_dist, faraway_d, faraway_z)
    result = jnp.where(flat_dist <= nearby_d[-1], nearby_result, faraway_result)
    return result.reshape(shape)


def get_value_from_logv_jax(logv, grids, default_maxz):
    """Interpolate quantity from log comoving volume using JAX."""
    if jnp.issubdtype(logv.dtype, jnp.complexfloating):
        raise TypeError("comoving volume must be real")
    if not jnp.issubdtype(logv.dtype, jnp.floating):
        logv = logv.astype(float)
    if not bool(jnp.all(jnp.isfinite(logv))):
        raise ValueError("comoving volume must be finite and > 0")

    nearby_logv, nearby_y, faraway_logv, faraway_y = grids
    if bool(jnp.any((logv < nearby_logv[0]) | (logv > faraway_logv[-1]))):
        raise ValueError(
            "JAX comoving volumes must be within the precomputed "
            f"redshift range 0.001 <= z <= {default_maxz:g}"
        )

    shape = logv.shape
    flat_logv = logv.reshape(-1)
    nearby_result = jnp.interp(flat_logv, nearby_logv, nearby_y)
    faraway_result = jnp.interp(flat_logv, faraway_logv, faraway_y)
    result = jnp.where(
        flat_logv <= nearby_logv[-1], nearby_result, faraway_result
    )
    return result.reshape(shape)


def get_value_from_volume_jax(volume, grids, default_maxz):
    """Interpolate quantity from comoving volume using JAX."""
    if jnp.issubdtype(volume.dtype, jnp.complexfloating):
        raise TypeError("comoving volume must be real")
    if not jnp.issubdtype(volume.dtype, jnp.floating):
        volume = volume.astype(float)
    if not bool(jnp.all(jnp.isfinite(volume) & (volume > 0))):
        raise ValueError("comoving volume must be finite and > 0")
    return get_value_from_logv_jax(jnp.log(volume), grids, default_maxz)
