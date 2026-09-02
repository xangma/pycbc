# Copyright (C) 2022 Francesco Pannarale, Andrew Williamson,
# Samuel Higginbotham
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
Innermost Stable Spherical Orbit (ISSO) solver in the Perez-Giz (PG)
formalism. See `Stone, Loeb, Berger, PRD 87, 084053 (2013)`_.

.. _Stone, Loeb, Berger, PRD 87, 084053 (2013):
    http://dx.doi.org/10.1103/PhysRevD.87.084053
"""

import numpy as np
from scipy.optimize import root_scalar


def _torch_values(*values):
    """Return broadcast Torch inputs without importing Torch eagerly."""
    if not any(
            type(value).__module__.split(".", 1)[0] == "torch"
            for value in values):
        return None, values

    import torch

    tensors = [value for value in values
               if isinstance(value, torch.Tensor)]
    if not tensors:
        return None, values
    if any(value.is_complex() for value in tensors):
        raise TypeError("Torch ISSO inputs must be real-valued")

    reference = tensors[0]
    dtype = reference.dtype
    if not dtype.is_floating_point:
        dtype = torch.get_default_dtype()
    converted = tuple(
        value.to(device=reference.device, dtype=dtype)
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, device=reference.device, dtype=dtype)
        for value in values
    )
    return torch, torch.broadcast_tensors(*converted)


def _sin(value):
    """Evaluate sine without sending Torch inputs through NumPy."""
    if type(value).__module__.split(".", 1)[0] == "torch":
        import torch
        return torch.sin(value)
    return np.sin(value)


def ISCO_solution(chi, incl):
    r"""Analytic solution of the innermost
    stable circular orbit (ISCO) for the Kerr metric.

    ..See eq. (2.21) of
    Bardeen, J. M., Press, W. H., Teukolsky, S. A. (1972)`
    https://articles.adsabs.harvard.edu/pdf/1972ApJ...178..347B

    Parameters
    -----------
    chi: float
        the BH dimensionless spin parameter
    incl: float
        inclination angle between the BH spin and the orbital angular
        momentum in radians

    Returns
    ----------
    float
    """
    torch, values = _torch_values(chi, incl)
    if torch is not None:
        chi, incl = values

        def cbrt(value):
            return torch.sign(value) * torch.abs(value).pow(1.0 / 3.0)

        chi2 = chi * chi
        sgn = torch.sign(torch.cos(incl))
        z1 = 1 + cbrt(1 - chi2) * (cbrt(1 + chi) + cbrt(1 - chi))
        z2 = torch.sqrt(3 * chi2 + z1 * z1)
        return 3 + z2 - sgn * torch.sqrt(
            (3 - z1) * (3 + z1 + 2 * z2)
        )

    chi2 = chi * chi
    sgn = np.sign(np.cos(incl))
    Z1 = 1 + np.cbrt(1 - chi2) * (np.cbrt(1 + chi) + np.cbrt(1 - chi))
    Z2 = np.sqrt(3 * chi2 + Z1 * Z1)
    return 3 + Z2 - sgn * np.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))


def ISSO_eq_at_pole(r, chi):
    r"""Polynomial that enables the calculation of the Kerr polar
    (:math:`\iota = \pm \pi / 2`) innermost stable spherical orbit
    (ISSO) radius via the roots of

    .. math::

        P(r) &= r^3 [r^2 (r - 6) + \chi^2 (3 r + 4)] \\
             &\quad + \chi^4 [3 r (r - 2) + \chi^2] \, ,

    where :math:`\chi` is the BH dimensionless spin parameter. Physical
    solutions are between 6 and
    :math:`1 + \sqrt{3} + \sqrt{3 + 2 \sqrt{3}}`.

    Parameters
    ----------
    r: float
        the radial coordinate in BH mass units
    chi: float
        the BH dimensionless spin parameter

    Returns
    -------
    float
    """
    chi2 = chi * chi
    return (
        r**3 * (r**2 * (r - 6) + chi2 * (3 * r + 4))
        + chi2 * chi2 * (3 * r * (r - 2) + chi2))


def ISSO_eq_at_pole_dr(r, chi):
    """Partial derivative of :func:`ISSO_eq_at_pole` with respect to r.

    Parameters
    ----------
    r: float
        the radial coordinate in BH mass units
    chi: float
        the BH dimensionless spin parameter

    Returns
    -------
    float
    """
    chi2 = chi * chi
    twlvchi2 = 12 * chi2
    sxchi4 = 6 * chi2 * chi2
    return (
        6 * r**5 - 30 * r**4 + twlvchi2 * r**3 + twlvchi2 * r**2 + sxchi4 * r
        + sxchi4)


def ISSO_eq_at_pole_dr2(r, chi):
    """Double partial derivative of :func:`ISSO_eq_at_pole` with
    respect to r.

    Parameters
    ----------
    r: float
        the radial coordinate in BH mass units
    chi: float
        the BH dimensionless spin parameter

    Returns
    -------
    float
    """
    chi2 = chi * chi
    return (
        30 * r**4 - 120 * r**3 + 36 * chi2 * r**2 + 24 * chi2 * r
        + 6 * chi2 * chi2)


def PG_ISSO_eq(r, chi, incl):
    r"""Polynomial that enables the calculation of a generic innermost
    stable spherical orbit (ISSO) radius via the roots in :math:`r` of

    .. math::

        S(r) &= r^8 Z(r) + \chi^2 (1 - \cos(\iota)^2) \\
             &\quad * [\chi^2 (1 - \cos(\iota)^2) Y(r) - 2 r^4 X(r)]\,,

    where

    .. math::

        X(r) &= \chi^2 (\chi^2 (3 \chi^2 + 4 r (2 r - 3)) \\
             &\quad + r^2 (15 r (r - 4) + 28)) - 6 r^4 (r^2 - 4) \, ,

    .. math::

        Y(r) &= \chi^4 (\chi^4 + r^2 [7 r (3 r - 4) + 36]) \\
             &\quad + 6 r (r - 2) \\
             &\qquad * (\chi^6 + 2 r^3
             [\chi^2 (3 r + 2) + 3 r^2 (r - 2)]) \, ,

    and :math:`Z(r) =` :func:`ISCO_eq`. Physical solutions are between
    the equatorial ISSO (i.e. the ISCO) radius (:func:`ISCO_eq`) and
    the polar ISSO radius (:func:`ISSO_eq_at_pole`).
    See `Stone, Loeb, Berger, PRD 87, 084053 (2013)`_.

    .. _Stone, Loeb, Berger, PRD 87, 084053 (2013):
        http://dx.doi.org/10.1103/PhysRevD.87.084053

    Parameters
    ----------
    r: float
        the radial coordinate in BH mass units
    chi: float
        the BH dimensionless spin parameter
    incl: float
        inclination angle between the BH spin and the orbital angular
        momentum in radians

    Returns
    -------
    float
    """
    chi2 = chi * chi
    chi4 = chi2 * chi2
    r2 = r * r
    r4 = r2 * r2
    three_r = 3 * r
    r_minus_2 = r - 2
    sin_incl2 = _sin(incl)**2

    X = (
        chi2 * (
            chi2 * (3 * chi2 + 4 * r * (2 * r - 3))
            + r2 * (15 * r * (r - 4) + 28))
        - 6 * r4 * (r2 - 4))
    Y = (
        chi4 * (chi4 + r2 * (7 * r * (three_r - 4) + 36))
        + 6 * r * r_minus_2 * (
            chi4 * chi2 + 2 * r2 * r * (
                chi2 * (three_r + 2) + 3 * r2 * r_minus_2)))
    Z = (r * (r - 6))**2 - chi2 * (2 * r * (3 * r + 14) - 9 * chi2)

    return r4 * r4 * Z + chi2 * sin_incl2 * (chi2 * sin_incl2 * Y - 2 * r4 * X)


def PG_ISSO_eq_dr(r, chi, incl):
    """Partial derivative of :func:`PG_ISSO_eq` with respect to r.

    Parameters
    ----------
    r: float
        the radial coordinate in BH mass units
    chi: float
        the BH dimensionless spin parameter
    incl: float
        inclination angle between the BH spin and the orbital angular
        momentum in radians

    Returns
    -------
    float
    """
    sini = _sin(incl)
    sin2i = sini * sini
    sin4i = sin2i * sin2i
    chi2 = chi * chi
    chi4 = chi2 * chi2
    chi6 = chi4 * chi2
    chi8 = chi4 * chi4
    chi10 = chi6 * chi4
    return (
        12 * r**11 - 132 * r**10
        + r**9 * (120 * chi2 * sin2i - 60 * chi2 + 360) - r**8 * 252 * chi2
        + 8 * r**7 * (
            36 * chi4 * sin4i - 30 * chi4 * sin2i + 9 * chi4
            - 48 * chi2 * sin2i)
        + 7 * r**6 * (120 * chi4 * sin2i - 144 * chi4 * sin4i)
        + 6 * r**5 * (
            36 * chi6 * sin4i - 16 * chi6 * sin2i + 144 * chi4 * sin4i
            - 56 * chi4 * sin2i)
        + r**4 * (120 * chi6 * sin2i - 240 * chi6 * sin4i)
        + r**3 * (84 * chi8 * sin4i - 24 * chi8 * sin2i - 192 * chi6 * sin4i)
        - 84 * r**2 * chi8 * sin4i
        + r * (12 * chi10 * sin4i + 72 * chi8 * sin4i) - 12 * chi10 * sin4i)


def PG_ISSO_eq_dr2(r, chi, incl):
    """Second partial derivative of :func:`PG_ISSO_eq` with respect to
    r.

    Parameters
    ----------
    r: float
        the radial coordinate in BH mass units
    chi: float
        the BH dimensionless spin parameter
    incl: float
        inclination angle between the BH spin and the orbital angular
        momentum in radians

    Returns
    -------
    float
    """
    sini = _sin(incl)
    sin2i = sini * sini
    sin4i = sin2i * sin2i
    chi2 = chi * chi
    chi4 = chi2 * chi2
    chi6 = chi4 * chi2
    chi8 = chi4 * chi4
    return (
        132 * r**10 - 1320 * r**9
        + 90 * r**8 * (12 * chi2 * sin2i - 6 * chi2 + 36) - 2016 * chi2 * r**7
        + 56 * r**6 * (
            36 * chi4 * sin4i - 30 * chi4 * sin2i + 9 * chi4
            - 48 * chi2 * sin2i)
        + 42 * r**5 * (120 * chi4 * sin2i - 144 * chi4 * sin4i)
        + 30 * r**4 * (
            36 * chi6 * sin4i - 16 * chi6 * sin2i + 144 * chi4 * sin4i
            - 56 * chi4 * sin2i)
        + r**3 * (480 * chi6 * sin2i - 960 * chi6 * sin4i)
        + r**2 * (
            252 * chi8 * sin4i - 72 * chi8 * sin2i - 576 * chi6 * sin4i)
        - r * 168 * chi8 * sin4i
        + 12 * chi8 * chi2 * sin4i + 72 * chi8 * sin4i)


def _PG_ISSO_extremal_eq(r, incl):
    """PG polynomial with its repeated ``(r - 1)^3`` root removed."""
    sin2i = _sin(incl)**2
    sin4i = sin2i * sin2i
    return (
        r**9 - 9 * r**8 + 12 * r**7 * sin2i
        + 36 * r**6 * sin2i
        + r**5 * (36 * sin4i - 6 * sin2i)
        + r**4 * (-36 * sin4i + 6 * sin2i)
        - 36 * r**3 * sin4i
        - 12 * r**2 * sin4i
        + 9 * r * sin4i
        - sin4i
    )


def _PG_ISSO_extremal_eq_dr(r, incl):
    """Radial derivative of :func:`_PG_ISSO_extremal_eq`."""
    sin2i = _sin(incl)**2
    sin4i = sin2i * sin2i
    return (
        9 * r**8 - 72 * r**7 + 84 * r**6 * sin2i
        + 216 * r**5 * sin2i
        + r**4 * (180 * sin4i - 30 * sin2i)
        + r**3 * (-144 * sin4i + 24 * sin2i)
        - 108 * r**2 * sin4i
        - 24 * r * sin4i
        + 9 * sin4i
    )


def _PG_ISSO_root_eq(r, chi, incl):
    """Numerically stable generic polynomial for Torch root finding."""
    import torch
    return torch.where(
        chi == 1,
        _PG_ISSO_extremal_eq(r, incl),
        PG_ISSO_eq(r, chi, incl),
    )


def _PG_ISSO_root_eq_dr(r, chi, incl):
    """Radial derivative of :func:`_PG_ISSO_root_eq`."""
    import torch
    return torch.where(
        chi == 1,
        _PG_ISSO_extremal_eq_dr(r, incl),
        PG_ISSO_eq_dr(r, chi, incl),
    )


def _torch_bisect(function, lo, hi, args, iterations=64):
    """Vectorized bisection for already-bracketed Torch roots."""
    import torch

    f_lo = function(lo, *args)
    f_hi = function(hi, *args)
    bracketed = ((f_lo < 0) & (f_hi > 0)) | (
        (f_lo > 0) & (f_hi < 0)
    )
    lo_root = f_lo == 0
    hi_root = f_hi == 0

    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        f_mid = function(mid, *args)
        same_side = ((f_lo < 0) & (f_mid < 0)) | (
            (f_lo > 0) & (f_mid > 0)
        )
        update_lo = bracketed & same_side
        lo = torch.where(update_lo, mid, lo)
        f_lo = torch.where(update_lo, f_mid, f_lo)
        hi = torch.where(bracketed & ~same_side, mid, hi)

    midpoint = 0.5 * (lo + hi)
    return torch.where(
        lo_root,
        lo,
        torch.where(hi_root, hi, midpoint),
    ), bracketed | lo_root | hi_root


def _torch_newton_refine(root, lo, hi, function, derivative, args):
    """Refine values and expose the implicit root derivative to autograd."""
    import torch

    value = function(root, *args)
    slope = derivative(root, *args)
    safe_slope = torch.where(
        torch.isfinite(slope) & (slope != 0),
        slope,
        torch.ones_like(slope),
    )
    candidate = root - value / safe_slope
    valid = (
        torch.isfinite(candidate)
        & torch.isfinite(slope)
        & (slope != 0)
        & (candidate >= lo)
        & (candidate <= hi)
    )
    return torch.where(valid, candidate, root)


def _PG_ISSO_solver_torch(torch, chi, incl):
    """Torch implementation of :func:`PG_ISSO_solver`."""
    chi = torch.abs(chi)
    r_isco = ISCO_solution(chi, incl)
    equatorial = torch.isclose(incl, torch.zeros_like(incl)) | torch.isclose(
        incl, torch.full_like(incl, torch.pi)
    )

    pole_lo = torch.full_like(chi, 5.0)
    pole_hi = torch.full_like(chi, 6.5)
    r_pole, _ = _torch_bisect(
        ISSO_eq_at_pole, pole_lo, pole_hi, (chi,)
    )
    r_pole = _torch_newton_refine(
        r_pole,
        pole_lo,
        pole_hi,
        ISSO_eq_at_pole,
        ISSO_eq_at_pole_dr,
        (chi,),
    )
    polar = torch.isclose(
        incl, torch.full_like(incl, 0.5 * torch.pi)
    )

    initial_lo = torch.minimum(r_isco, r_pole)
    initial_hi = torch.maximum(r_isco, r_pole)

    # At exactly extremal spin, PG's polynomial has a repeated r=1 root.
    # The root helper factors it out so the physical inclined root remains
    # bracketable without cancellation near the equatorial endpoint.
    generic_lo = initial_lo
    generic_hi = initial_hi
    generic, bracketed = _torch_bisect(
        _PG_ISSO_root_eq, generic_lo, generic_hi, (chi, incl)
    )
    generic = _torch_newton_refine(
        generic,
        generic_lo,
        generic_hi,
        _PG_ISSO_root_eq,
        _PG_ISSO_root_eq_dr,
        (chi, incl),
    )
    generic = torch.where(bracketed, generic, r_isco)
    return torch.where(equatorial, r_isco, torch.where(polar, r_pole, generic))


def PG_ISSO_solver(chi, incl):
    """Function that determines the radius of the innermost stable
    spherical orbit (ISSO) for a Kerr BH and a generic inclination
    angle between the BH spin and the orbital angular momentum.
    This function finds the appropriate root of :func:`PG_ISSO_eq`.

    Parameters
    ----------
    chi: {float, array}
        the BH dimensionless spin parameter
    incl: {float, array}
        the inclination angle between the BH spin and the orbital
        angular momentum in radians

    Returns
    -------
    solution: array or torch.Tensor
        the radius of the orbit in BH mass units
    """
    torch, values = _torch_values(chi, incl)
    if torch is not None:
        return _PG_ISSO_solver_torch(torch, *values)

    # Auxiliary variables
    if np.isscalar(chi):
        chi = np.array(chi, copy=False, ndmin=1)
        incl = np.array(incl, copy=False, ndmin=1)
    chi = np.abs(chi)
    # ISCO radius for the given spin magnitude
    rISCO_limit = ISCO_solution(chi, incl)
    # If the inclination is 0 or pi, just output the ISCO radius
    equatorial = np.isclose(incl, 0) | np.isclose(incl, np.pi)
    if all(equatorial):
        return rISCO_limit

    # ISSO radius for an inclination of pi/2
    # Initial guess is based on the extrema of the polar ISSO radius equation,
    # that are: r=6 (chi=1) and r=1+sqrt(3)+sqrt(3+sqrt(12))=5.274... (chi=0)
    initial_guess = [5.27451056440629 if c > 0.5 else 6 for c in chi]
    rISSO_at_pole_limit = np.array([
        root_scalar(
            ISSO_eq_at_pole, x0=g0, fprime=ISSO_eq_at_pole_dr,
            fprime2=ISSO_eq_at_pole_dr2, args=(c)).root
        for g0, c in zip(initial_guess, chi)])
    # If the inclination is pi/2, just output the ISSO radius at the pole(s)
    polar = np.isclose(incl, 0.5*np.pi)
    if all(polar):
        return rISSO_at_pole_limit

    # Otherwise, find the ISSO radius for a generic inclination
    initial_hi = np.maximum(rISCO_limit, rISSO_at_pole_limit)
    initial_lo = np.minimum(rISCO_limit, rISSO_at_pole_limit)
    brackets = [
        (bl, bh) if (c != 1 and PG_ISSO_eq(bl, c, inc) *
                     PG_ISSO_eq(bh, c, inc) < 0) else None
        for bl, bh, c, inc in zip(initial_lo, initial_hi, chi, incl)]
    solution = np.array([
        root_scalar(
            PG_ISSO_eq, x0=g0, fprime=PG_ISSO_eq_dr, bracket=bracket,
            fprime2=PG_ISSO_eq_dr2, args=(c, inc), xtol=1e-12).root
        for g0, bracket, c, inc in zip(initial_hi, brackets, chi, incl)])
    oob = (solution < 1) | (solution > 9)
    if any(oob):
        solution = np.array([
            root_scalar(
                PG_ISSO_eq, x0=g0, fprime=PG_ISSO_eq_dr, bracket=bracket,
                fprime2=PG_ISSO_eq_dr2, args=(c, inc)).root
            if ob else sol for g0, bracket, c, inc, ob, sol
            in zip(initial_lo, brackets, chi, incl, oob, solution)
            ])
        oob = (solution < 1) | (solution > 9)
        if any(oob):
            raise RuntimeError('Unable to obtain some solutions!')
    return solution
