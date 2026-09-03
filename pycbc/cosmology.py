# Copyright (C) 2017  Collin Capano
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
This modules provides functions for computing cosmological quantities, such as
redshift. This is mostly a wrapper around ``astropy.cosmology``.

Note: in all functions, ``distance`` is short hand for ``luminosity_distance``.
Any other distance measure is explicitly named; e.g., ``comoving_distance``.
"""

import logging
import numpy
from scipy import interpolate
import astropy.cosmology
from astropy import units
from astropy.cosmology import CosmologyError, parameters
import pycbc.conversions

logger = logging.getLogger('pycbc.cosmology')

DEFAULT_COSMOLOGY = 'Planck15'


def get_cosmology(cosmology=None, **kwargs):
    r"""Gets an astropy cosmology class.

    Parameters
    ----------
    cosmology : str or astropy.cosmology.FlatLambdaCDM, optional
        The name of the cosmology to use. For the list of options, see
        :py:attr:`astropy.cosmology.parameters.available`. If None, and no
        other keyword arguments are provided, will default to
        :py:attr:`DEFAULT_COSMOLOGY`. If an instance of
        :py:class:`astropy.cosmology.FlatLambdaCDM`, will just return that.
    \**kwargs :
        If any other keyword arguments are provided they will be passed to
        :py:attr:`astropy.cosmology.FlatLambdaCDM` to create a custom
        cosmology.

    Returns
    -------
    astropy.cosmology.FlatLambdaCDM
        The cosmology to use.

    Examples
    --------
    Use the default:

    >>> from pycbc.cosmology import get_cosmology
    >>> get_cosmology()
    FlatLambdaCDM(name="Planck15", H0=67.7 km / (Mpc s), Om0=0.307,
                  Tcmb0=2.725 K, Neff=3.05, m_nu=[0.   0.   0.06] eV,
                  Ob0=0.0486)

    Use properties measured by WMAP instead:

    >>> get_cosmology("WMAP9")
    FlatLambdaCDM(name="WMAP9", H0=69.3 km / (Mpc s), Om0=0.286, Tcmb0=2.725 K,
                  Neff=3.04, m_nu=[0. 0. 0.] eV, Ob0=0.0463)

    Create your own cosmology (see :py:class:`astropy.cosmology.FlatLambdaCDM`
    for details on the default values used):

    >>> get_cosmology(H0=70., Om0=0.3)
    FlatLambdaCDM(H0=70 km / (Mpc s), Om0=0.3, Tcmb0=0 K, Neff=3.04, m_nu=None,
                  Ob0=None)

    """
    if kwargs and cosmology is not None:
        raise ValueError("if providing custom cosmological parameters, do "
                         "not provide a `cosmology` argument")
    if isinstance(cosmology, astropy.cosmology.FlatLambdaCDM):
        # just return
        return cosmology
    if kwargs:
        cosmology = astropy.cosmology.FlatLambdaCDM(**kwargs)
    else:
        if cosmology is None:
            cosmology = DEFAULT_COSMOLOGY
        if cosmology not in parameters.available:
            raise ValueError("unrecognized cosmology {}".format(cosmology))
        cosmology = getattr(astropy.cosmology, cosmology)
    return cosmology


def z_at_value(func, fval, unit, zmax=1000., **kwargs):
    r"""Wrapper around astropy.cosmology.z_at_value to handle numpy arrays.

    Getting a z for a cosmological quantity involves numerically inverting
    ``func``. The ``zmax`` argument sets how large of a z to guess (see
    :py:func:`astropy.cosmology.z_at_value` for details). If a z is larger than
    ``zmax``, this will try a larger zmax up to ``zmax * 10**5``. If that still
    is not large enough, will just return ``numpy.inf``.

    Parameters
    ----------
    func : function or method
        A function that takes redshift as input.
    fval : float
        The value of ``func(z)``.
    unit : astropy.unit
        The unit of ``fval``.
    zmax : float, optional
        The initial maximum search limit for ``z``. Default is 1000.
    \**kwargs :
        All other keyword arguments are passed to
        :py:func:``astropy.cosmology.z_at_value``.

    Returns
    -------
    float
        The redshift at the requested values.
    """
    fval, input_is_array = pycbc.conversions.ensurearray(fval)
    # make sure fval is atleast 1D
    if fval.size == 1 and fval.ndim == 0:
        fval = fval.reshape(1)
    zs = numpy.zeros(fval.shape, dtype=float)  # the output array
    if 'method' not in kwargs:
        # workaround for https://github.com/astropy/astropy/issues/14249
        # FIXME remove when fixed in astropy/scipy
        kwargs['method'] = 'bounded'
    for (ii, val) in enumerate(fval):
        try:
            zs[ii] = astropy.cosmology.z_at_value(func, val*unit, zmax=zmax,
                                                  **kwargs)
        except CosmologyError:
            if ii == len(zs)-1:
                # if zs[ii] is less than but very close to zmax, let's say
                # zs[ii] is the last element in the [zmin, zmax],
                # `z_at_value` will also returns "CosmologyError", please
                # see (https://docs.astropy.org/en/stable/api/astropy.
                # cosmology.z_at_value.html), in order to avoid bumping up
                # zmax, just set zs equals to previous value, we assume
                # the `func` is smooth
                zs[ii] = zs[ii-1]
            else:
                # we'll get this if the z was larger than zmax; in that
                # case we'll try bumping up zmax later to get a value
                zs[ii] = numpy.inf
    # check if there were any zs > zmax
    replacemask = numpy.isinf(zs)
    # try bumping up zmax to get a result
    if replacemask.any():
        # we'll keep bumping up the maxz until we can get a result
        counter = 0  # to prevent running forever
        while replacemask.any():
            kwargs['zmin'] = zmax
            zmax = 10 * zmax
            idx = numpy.where(replacemask)
            for ii in idx:
                val = fval[ii]
                try:
                    zs[ii] = astropy.cosmology.z_at_value(
                        func, val*unit, zmax=zmax, **kwargs)
                    replacemask[ii] = False
                except CosmologyError:
                    # didn't work, try on next loop
                    pass
            counter += 1
            if counter == 5:
                # give up and warn the user
                logger.warning("One or more values correspond to a "
                               "redshift > {0:.1e}. The redshift for these "
                               "have been set to inf. If you would like "
                               "better precision, call God.".format(zmax))
                break
    return pycbc.conversions.formatreturn(zs, input_is_array)


def _redshift(distance, **kwargs):
    r"""Uses astropy to get redshift from the given luminosity distance.

    Parameters
    ----------
    distance : float
        The luminosity distance, in Mpc.
    \**kwargs :
        All other keyword args are passed to :py:func:`get_cosmology` to
        select a cosmology. If none provided, will use
        :py:attr:`DEFAULT_COSMOLOGY`.

    Returns
    -------
    float :
        The redshift corresponding to the given luminosity distance.
    """
    cosmology = get_cosmology(**kwargs)
    return z_at_value(cosmology.luminosity_distance, distance, units.Mpc)


class DistToZ(object):
    r"""Interpolates luminosity distance as a function of redshift to allow for
    fast conversion.

    The :mod:`astropy.cosmology` module provides methods for converting any
    cosmological parameter (like luminosity distance) to redshift. This can be
    very slow when operating on a large array, as it involves numerically
    inverting :math:`z(D)` (where :math:`D` is the luminosity distance). This
    class speeds that up by pre-interpolating :math:`D(z)`. It works by setting
    up a dense grid of redshifts, then using linear interpolation to find the
    inverse function.  The interpolation uses a grid linear in z for z < 1, and
    log in z for ``default_maxz`` > z > 1. This interpolater is setup the first
    time `get_redshift` is called. If a host value is requested that results
    in a z > ``default_maxz``, the class falls back to calling Astropy
    directly. Torch tensors outside the precomputed range fail closed rather
    than being copied to the host.

    Instances of this class can be called like a function on luminosity
    distances, which will return the corresponding redshifts.

    Parameters
    ----------
    default_maxz : float, optional
        The maximum z to interpolate up to before falling back to calling
        astropy directly. Default is 1000.
    numpoints : int, optional
        The number of points to use in the linear interpolation between 0 to 1
        and 1 to ``default_maxz``. Default is 10000.
    \**kwargs :
        All other keyword args are passed to :py:func:`get_cosmology` to
        select a cosmology. If none provided, will use
        :py:attr:`DEFAULT_COSMOLOGY`.
    """
    def __init__(self, default_maxz=1000., numpoints=10000, **kwargs):
        self.numpoints = int(numpoints)
        self.default_maxz = default_maxz
        self.cosmology = get_cosmology(**kwargs)
        # the interpolating functions; we'll set them to None for now, then set
        # them up when get_redshift is first called
        self.nearby_d2z = None
        self.faraway_d2z = None
        self.default_maxdist = None
        self._nearby_grid = None
        self._faraway_grid = None
        self._torch_grids = {}

    def setup_interpolant(self):
        """Initializes the z(d) interpolation."""
        # for computing nearby (z < 1) redshifts
        nearby_zs = numpy.linspace(0., 1., num=self.numpoints)
        nearby_ds = self.cosmology.luminosity_distance(nearby_zs).value
        self.nearby_d2z = interpolate.interp1d(
            nearby_ds, nearby_zs, kind='linear', bounds_error=False)
        # for computing far away (z > 1) redshifts
        faraway_zs = numpy.logspace(
            0, numpy.log10(self.default_maxz), num=self.numpoints)
        faraway_ds = self.cosmology.luminosity_distance(faraway_zs).value
        self.faraway_d2z = interpolate.interp1d(
            faraway_ds, faraway_zs, kind='linear', bounds_error=False)
        # store the default maximum distance
        self.default_maxdist = float(faraway_ds.max())
        self._nearby_grid = (nearby_ds, nearby_zs)
        self._faraway_grid = (faraway_ds, faraway_zs)
        self._torch_grids.clear()

    def _get_torch_grids(self, torch, dist):
        """Return interpolation grids cached on a tensor's device."""
        if self._nearby_grid is None or self._faraway_grid is None:
            self.setup_interpolant()
        key = (dist.device, dist.dtype)
        try:
            return self._torch_grids[key]
        except KeyError:
            grids = tuple(
                torch.as_tensor(values, device=dist.device, dtype=dist.dtype)
                for grid in (self._nearby_grid, self._faraway_grid)
                for values in grid
            )
            self._torch_grids[key] = grids
            return grids

    @staticmethod
    def _torch_linear_interp(torch, x, y, values):
        """Linearly interpolate tensor values on a monotonic grid."""
        upper = torch.searchsorted(x, values, right=False)
        upper = upper.clamp(1, x.numel() - 1)
        lower = upper - 1
        x0 = x[lower]
        x1 = x[upper]
        y0 = y[lower]
        y1 = y[upper]
        return y0 + (values - x0) * (y1 - y0) / (x1 - x0)

    def _get_redshift_torch(self, torch, dist):
        """Evaluate the precomputed inverse on a Torch device."""
        if dist.dtype.is_complex:
            raise TypeError("distance must be real")
        if not dist.dtype.is_floating_point:
            dist = dist.to(dtype=torch.get_default_dtype())
        if not bool(torch.all(torch.isfinite(dist) & (dist >= 0))):
            raise ValueError("distance must be finite and >= 0")

        nearby_d, nearby_z, faraway_d, faraway_z = (
            self._get_torch_grids(torch, dist)
        )
        if bool(torch.any(dist > faraway_d[-1])):
            raise ValueError(
                "Torch distances must be within the precomputed redshift "
                f"range z <= {self.default_maxz:g}"
            )

        shape = dist.shape
        flat_dist = dist.reshape(-1).contiguous()
        nearby_values = flat_dist.clamp(
            min=nearby_d[0], max=nearby_d[-1])
        faraway_values = flat_dist.clamp(
            min=faraway_d[0], max=faraway_d[-1])
        nearby_result = self._torch_linear_interp(
            torch, nearby_d, nearby_z, nearby_values)
        faraway_result = self._torch_linear_interp(
            torch, faraway_d, faraway_z, faraway_values)
        result = torch.where(
            flat_dist <= nearby_d[-1], nearby_result, faraway_result)
        return result.reshape(shape)

    def get_redshift(self, dist):
        """Returns the redshift for the given distance.

        Torch tensors are interpolated on their existing device. The static
        cosmology grids are constructed on the host once and cached per Torch
        device and dtype. Tensor distances beyond ``default_maxz`` fail closed
        instead of being copied to the host for Astropy inversion.
        """
        torch, values = pycbc.conversions._torch_values(dist)
        if torch is not None:
            return self._get_redshift_torch(torch, values[0])

        dist, input_is_array = pycbc.conversions.ensurearray(dist)
        try:
            zs = self.nearby_d2z(dist)
        except TypeError:
            # interpolant hasn't been setup yet
            self.setup_interpolant()
            zs = self.nearby_d2z(dist)
        # if any points had red shifts beyond the nearby, will have nans;
        # replace using the faraway interpolation
        replacemask = numpy.isnan(zs)
        if replacemask.any():
            zs[replacemask] = self.faraway_d2z(dist[replacemask])
            replacemask = numpy.isnan(zs)
        # if we still have nans, means that some distances are beyond our
        # furthest default; fall back to using astropy
        if replacemask.any():
            # well... check that the distance is positive and finite first
            if not (dist > 0.).all() and numpy.isfinite(dist).all():
                raise ValueError("distance must be finite and > 0")
            zs[replacemask] = _redshift(dist[replacemask],
                                        cosmology=self.cosmology)
        return pycbc.conversions.formatreturn(zs, input_is_array)

    def __call__(self, dist):
        return self.get_redshift(dist)


# set up D(z) interpolating classes for the standard cosmologies
_d2zs = {_c: DistToZ(cosmology=_c)
         for _c in parameters.available}


def redshift(distance, **kwargs):
    r"""Returns the redshift associated with the given luminosity distance.

    If the requested cosmology is one of the pre-defined ones in
    :py:attr:`astropy.cosmology.parameters.available`, :py:class:`DistToZ` is
    used to provide a fast interpolation. This takes a few seconds to setup
    on the first call.

    Parameters
    ----------
    distance : float, array-like, or torch.Tensor
        The luminosity distance, in Mpc. For a predefined cosmology, Torch
        tensors are evaluated on their existing device.
    \**kwargs :
        All other keyword args are passed to :py:func:`get_cosmology` to
        select a cosmology. If none provided, will use
        :py:attr:`DEFAULT_COSMOLOGY`.

    Returns
    -------
    float, numpy.ndarray, or torch.Tensor :
        The redshift corresponding to the given distance.
    """
    cosmology = get_cosmology(**kwargs)
    try:
        z = _d2zs[cosmology.name](distance)
    except KeyError:
        # not a standard cosmology, call the redshift function
        z = _redshift(distance, cosmology=cosmology)
    return z


class ComovingVolInterpolator(object):
    r"""Interpolates comoving volume to distance or redshift.

    The :mod:`astropy.cosmology` module provides methods for converting any
    cosmological parameter (like luminosity distance) to redshift. This can be
    very slow when operating on a large array, as it involves numerically
    inverting :math:`z(D)` (where :math:`D` is the luminosity distance). This
    class speeds that up by pre-interpolating :math:`D(z)`. It works by setting
    up a dense grid of redshifts, then using linear interpolation to find the
    inverse function. The interpolation uses a grid linear in z for z < 1, and
    log in z for ``default_maxz`` > z > 1. This interpolator is set up the
    first time `get_value` is called. If a host value is requested outside the
    interpolation range, the class falls back to calling Astropy directly.
    Torch tensors are evaluated on their existing device and fail closed
    outside the precomputed range instead of being copied to the host.

    Instances of this class can be called like a function on luminosity
    distances, which will return the corresponding redshifts.

    Parameters
    ----------
    parameter : {'luminosity_distance', 'redshift'}
        What parameter to interpolate.
    default_maxz : float, optional
        The maximum z to interpolate up to before falling back to calling
        astropy directly. Default is 10.
    numpoints : int, optional
    The number of points to use in the linear interpolation between 0 to 1
        and 1 to ``default_maxz``. Default is 1000.
    vol_func: function, optional
        Optionally set how the volume is calculated by providing a function
    \**kwargs :
        All other keyword args are passed to :py:func:`get_cosmology` to
        select a cosmology. If none provided, will use
        :py:attr:`DEFAULT_COSMOLOGY`.
    """
    def __init__(self, parameter, default_maxz=10., numpoints=1000,
                 vol_func=None, **kwargs):
        self.parameter = parameter
        self.numpoints = int(numpoints)
        self.default_maxz = default_maxz
        self.cosmology = get_cosmology(**kwargs)
        # the interpolating functions; we'll set them to None for now, then set
        # them up when get_redshift is first called
        self.nearby_interp = None
        self.faraway_interp = None
        self.default_maxvol = None
        self._nearby_grid = None
        self._faraway_grid = None
        self._torch_grids = {}
        if vol_func is not None:
            self.vol_func = vol_func
        else:
            self.vol_func = self.cosmology.comoving_volume
        self.vol_units = self.vol_func(0.5).unit

    def _create_interpolant(self, minz, maxz):
        minlogv = numpy.log(self.vol_func(minz).value)
        maxlogv = numpy.log(self.vol_func(maxz).value)
        logvs = numpy.linspace(minlogv, maxlogv, num=self.numpoints)

        zs = z_at_value(self.vol_func, numpy.exp(logvs), self.vol_units, maxz)

        if self.parameter != 'redshift':
            ys = cosmological_quantity_from_redshift(zs, self.parameter)
        else:
            ys = zs

        return (
            interpolate.interp1d(
                logvs, ys, kind='linear', bounds_error=False
            ),
            (logvs, numpy.asarray(ys, dtype=float)),
        )

    def setup_interpolant(self):
        """Initializes the z(d) interpolation."""
        # get VC bounds
        # for computing nearby (z < 1) redshifts
        minz = 0.001
        maxz = 1.
        self.nearby_interp, self._nearby_grid = self._create_interpolant(
            minz, maxz
        )
        # for computing far away (z > 1) redshifts
        minz = 1.
        maxz = self.default_maxz
        self.faraway_interp, self._faraway_grid = self._create_interpolant(
            minz, maxz
        )
        # store the default maximum volume
        self.default_maxvol = numpy.log(self.vol_func(maxz).value)
        self._torch_grids.clear()

    def _get_torch_grids(self, torch, logv):
        """Return interpolation grids cached on a tensor's device."""
        if self._nearby_grid is None or self._faraway_grid is None:
            self.setup_interpolant()
        key = (logv.device, logv.dtype)
        try:
            return self._torch_grids[key]
        except KeyError:
            grids = tuple(
                torch.as_tensor(values, device=logv.device, dtype=logv.dtype)
                for grid in (self._nearby_grid, self._faraway_grid)
                for values in grid
            )
            self._torch_grids[key] = grids
            return grids

    @staticmethod
    def _torch_linear_interp(torch, x, y, values):
        """Linearly interpolate tensor values on a monotonic grid."""
        upper = torch.searchsorted(x, values, right=False)
        upper = upper.clamp(1, x.numel() - 1)
        lower = upper - 1
        x0 = x[lower]
        x1 = x[upper]
        y0 = y[lower]
        y1 = y[upper]
        return y0 + (values - x0) * (y1 - y0) / (x1 - x0)

    def _get_value_from_logv_torch(self, torch, logv):
        """Evaluate the precomputed inverse on a Torch device."""
        if logv.dtype.is_complex:
            raise TypeError("comoving volume must be real")
        if not logv.dtype.is_floating_point:
            logv = logv.to(dtype=torch.get_default_dtype())
        if not bool(torch.all(torch.isfinite(logv))):
            raise ValueError("comoving volume must be finite and > 0")

        nearby_logv, nearby_y, faraway_logv, faraway_y = (
            self._get_torch_grids(torch, logv)
        )
        if bool(
            torch.any(
                (logv < nearby_logv[0]) | (logv > faraway_logv[-1])
            )
        ):
            raise ValueError(
                "Torch comoving volumes must be within the precomputed "
                f"redshift range 0.001 <= z <= {self.default_maxz:g}"
            )

        shape = logv.shape
        flat_logv = logv.reshape(-1).contiguous()
        nearby_values = flat_logv.clamp(
            min=nearby_logv[0], max=nearby_logv[-1]
        )
        faraway_values = flat_logv.clamp(
            min=faraway_logv[0], max=faraway_logv[-1]
        )
        nearby_result = self._torch_linear_interp(
            torch, nearby_logv, nearby_y, nearby_values
        )
        faraway_result = self._torch_linear_interp(
            torch, faraway_logv, faraway_y, faraway_values
        )
        result = torch.where(
            flat_logv <= nearby_logv[-1], nearby_result, faraway_result
        )
        return result.reshape(shape)

    def get_value_from_logv(self, logv):
        """Return the requested quantity for a log comoving volume.

        Torch tensors are interpolated on their existing device. Their values
        must lie within the precomputed redshift range; tensor inputs never
        fall back to the host Astropy inversion.
        """
        torch, values = pycbc.conversions._torch_values(logv)
        if torch is not None:
            return self._get_value_from_logv_torch(torch, values[0])

        logv, input_is_array = pycbc.conversions.ensurearray(logv)
        try:
            vals = self.nearby_interp(logv)
        except TypeError:
            # interpolant hasn't been setup yet
            self.setup_interpolant()
            vals = self.nearby_interp(logv)
        # if any points had red shifts beyond the nearby, will have nans;
        # replace using the faraway interpolation
        replacemask = numpy.isnan(vals)
        if replacemask.any():
            vals[replacemask] = self.faraway_interp(logv[replacemask])
            replacemask = numpy.isnan(vals)
        # if we still have nans, means that some distances are beyond our
        # furthest default; fall back to using astropy
        if replacemask.any():
            # well... check that the logv is finite first
            if not numpy.isfinite(logv).all():
                raise ValueError("comoving volume must be finite and > 0")
            zs = z_at_value(self.vol_func,
                            numpy.exp(logv[replacemask]), self.vol_units)
            if self.parameter == 'redshift':
                vals[replacemask] = zs
            else:
                vals[replacemask] = \
                    getattr(self.cosmology, self.parameter)(zs).value
        return pycbc.conversions.formatreturn(vals, input_is_array)

    def get_value(self, volume):
        """Return the requested quantity for a comoving volume."""
        torch, values = pycbc.conversions._torch_values(volume)
        if torch is not None:
            volume = values[0]
            if volume.dtype.is_complex:
                raise TypeError("comoving volume must be real")
            if not volume.dtype.is_floating_point:
                volume = volume.to(dtype=torch.get_default_dtype())
            if not bool(
                torch.all(torch.isfinite(volume) & (volume > 0))
            ):
                raise ValueError("comoving volume must be finite and > 0")
            return self._get_value_from_logv_torch(
                torch, torch.log(volume)
            )
        return self.get_value_from_logv(numpy.log(volume))

    def __call__(self, volume):
        return self.get_value(volume)


# set up D(z) interpolating classes for the standard cosmologies
_v2ds = {_c: ComovingVolInterpolator('luminosity_distance', cosmology=_c)
         for _c in parameters.available}

_v2zs = {_c: ComovingVolInterpolator('redshift', cosmology=_c)
         for _c in parameters.available}


def redshift_from_comoving_volume(vc, interp=True, **kwargs):
    r"""Returns the redshift from the given comoving volume.

    Parameters
    ----------
    vc : float, array-like, or torch.Tensor
        The comoving volume, in units of cubed Mpc. With interpolation
        enabled for a predefined cosmology, Torch tensors are evaluated on
        their existing device.
    interp : bool, optional
        If true, this will setup an interpolator between redshift and comoving
        volume the first time this function is called. This is useful when
        making many successive calls to this function (and is necessary when
        using this function in a transform when doing parameter estimation).
        However, setting up the interpolator the first time takes O(10)s of
        seconds. If you will only be making a single call to this function, or
        will only run it on an array with < ~100000 elements, it is faster to
        not use the interpolator (i.e., set ``interp=False``). Default is
        ``True``.
    \**kwargs :
        All other keyword args are passed to :py:func:`get_cosmology` to
        select a cosmology. If none provided, will use
        :py:attr:`DEFAULT_COSMOLOGY`.

    Returns
    -------
    float, numpy.ndarray, or torch.Tensor :
        The redshift at the given comoving volume.
    """
    cosmology = get_cosmology(**kwargs)
    lookup = _v2zs if interp else {}
    try:
        z = lookup[cosmology.name](vc)
    except KeyError:
        # not using interp or not a standard cosmology,
        # call the redshift function directly
        z = z_at_value(cosmology.comoving_volume, vc, units.Mpc**3)
    return z


def distance_from_comoving_volume(vc, interp=True, **kwargs):
    r"""Returns the luminosity distance from the given comoving volume.

    Parameters
    ----------
    vc : float, array-like, or torch.Tensor
        The comoving volume, in units of cubed Mpc. With interpolation
        enabled for a predefined cosmology, Torch tensors are evaluated on
        their existing device.
    interp : bool, optional
        If true, this will setup an interpolator between distance and comoving
        volume the first time this function is called. This is useful when
        making many successive calls to this function (such as when using this
        function in a transform for parameter estimation).  However, setting up
        the interpolator the first time takes O(10)s of seconds. If you will
        only be making a single call to this function, or will only run it on
        an array with < ~100000 elements, it is faster to not use the
        interpolator (i.e., set ``interp=False``). Default is ``True``.
    \**kwargs :
        All other keyword args are passed to :py:func:`get_cosmology` to
        select a cosmology. If none provided, will use
        :py:attr:`DEFAULT_COSMOLOGY`.

    Returns
    -------
    float, numpy.ndarray, or torch.Tensor :
        The luminosity distance at the given comoving volume.
    """
    cosmology = get_cosmology(**kwargs)
    lookup = _v2ds if interp else {}
    try:
        dist = lookup[cosmology.name](vc)
    except KeyError:
        # not using interp or not a standard cosmology,
        # call the redshift function directly
        z = z_at_value(cosmology.comoving_volume, vc, units.Mpc**3)
        dist = cosmology.luminosity_distance(z).value
    return dist


def cosmological_quantity_from_redshift(z, quantity, strip_unit=True,
                                        **kwargs):
    r"""Returns the value of a cosmological quantity (e.g., age) at a redshift.

    Parameters
    ----------
    z : float
        The redshift.
    quantity : str
        The name of the quantity to get. The name may be any attribute of
        :py:class:`astropy.cosmology.FlatLambdaCDM`.
    strip_unit : bool, optional
        Just return the value of the quantity, sans units. Default is True.
    \**kwargs :
        All other keyword args are passed to :py:func:`get_cosmology` to
        select a cosmology. If none provided, will use
        :py:attr:`DEFAULT_COSMOLOGY`.

    Returns
    -------
    float or astropy.units.quantity :
        The value of the quantity at the requested value. If ``strip_unit`` is
        ``True``, will return the value. Otherwise, will return the value with
        units.
    """
    cosmology = get_cosmology(**kwargs)
    val = getattr(cosmology, quantity)(z)
    if strip_unit:
        val = val.value
    return val


__all__ = ['redshift', 'redshift_from_comoving_volume',
           'distance_from_comoving_volume',
           'cosmological_quantity_from_redshift',
           ]
