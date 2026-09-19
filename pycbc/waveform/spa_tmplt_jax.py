# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Pure JAX implementation of the stationary phase approximation (SPA) template engine."""

import math
import numpy as np
import jax.numpy as jnp

from pycbc.constants import PI, PI_4, TWOPI, LN2
from pycbc.types import Array as PyCBCArray
from pycbc.types.array_jax import JAXArrayData, _ensure_x64, to_jax


def _jax_native_spa(
    n,
    kmin,
    delta_f,
    piM,
    pfaN,
    pfa2,
    pfa3,
    pfa4,
    pfa5,
    pfl5,
    pfa6,
    pfl6,
    pfa7,
    amp_factor,
    dtype_out=jnp.complex64,
):
    """JAX implementation of spa_tmplt kernel matching CPU/Cython precision."""
    _ensure_x64()
    f_dtype = jnp.float64 if dtype_out == jnp.complex128 else jnp.float32

    i = jnp.arange(n)
    f = (i + kmin) * delta_f
    safe_f = jnp.where(f == 0.0, 1.0, f)

    v = (piM * safe_f) ** (1.0 / 3.0)
    v = jnp.where(f == 0.0, 0.0, v)
    logv = jnp.log(v)
    log4 = 2.0 * LN2

    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v2 * v3
    v6 = v3 * v3
    v7 = v3 * v4

    # Phasing polynomial
    ph = pfa7 * v7 + (pfa6 + pfl6 * (logv + log4)) * v6 + (pfa5 + pfl5 * logv) * v5 + pfa4 * v4 + pfa3 * v3 + pfa2 * v2 + 1.0
    safe_v5 = jnp.where(v5 == 0.0, 1.0, v5)
    phasing = ph * (pfaN / safe_v5) - PI_4
    phasing = phasing - jnp.trunc(phasing / TWOPI) * TWOPI
    phasing = jnp.where(phasing < -PI, phasing + TWOPI, phasing)
    phasing = jnp.where(phasing > PI, phasing - TWOPI, phasing)

    # Fast trigonometric evaluation matching CPU kernel
    sinp = 1.273239545 * phasing - 0.405284735 * phasing * jnp.abs(phasing)
    sinp = 0.225 * (sinp * jnp.abs(sinp) - sinp) + sinp

    phs = phasing + math.pi / 2.0
    phs = jnp.where(phs > PI, phs - TWOPI, phs)
    cosp = 1.273239545 * phs - 0.405284735 * phs * jnp.abs(phs)
    cosp = 0.225 * (cosp * jnp.abs(cosp) - cosp) + cosp

    # Amplitude with f^-7/6
    amp2 = amp_factor * (safe_f ** (-7.0 / 6.0))
    amp2 = jnp.where(f == 0.0, 0.0, amp2)

    res = (cosp * amp2).astype(f_dtype) - 1j * (sinp * amp2).astype(f_dtype)
    return res.astype(dtype_out)


def spa_tmplt_engine(
    htilde,
    kmin,
    phase_order,
    delta_f,
    piM,
    pfaN,
    pfa2,
    pfa3,
    pfa4,
    pfa5,
    pfl5,
    pfa6,
    pfl6,
    pfa7,
    amp_factor,
):
    """Calculate the SPA template phase on the active JAX backend."""
    _ensure_x64()
    n = len(htilde)
    delta_f = float(delta_f)
    kmin_int = int(kmin)

    c_dtype = jnp.complex64 if getattr(htilde, "dtype", None) == np.complex64 else jnp.complex128

    out = _jax_native_spa(
        n,
        kmin_int,
        delta_f,
        float(piM),
        float(pfaN),
        float(pfa2),
        float(pfa3),
        float(pfa4),
        float(pfa5),
        float(pfl5),
        float(pfa6),
        float(pfl6),
        float(pfa7),
        float(amp_factor),
        dtype_out=c_dtype,
    )

    if hasattr(htilde, "_data") and isinstance(htilde._data, JAXArrayData):
        htilde._data.set_array(out)
    elif hasattr(htilde, "data"):
        htilde.data[:] = np.asarray(out)
    else:
        htilde[:] = np.asarray(out)
    return None


def spa_tmplt_sequence(
    sample_points,
    piM,
    pfaN,
    pfa2,
    pfa3,
    pfa4,
    pfa5,
    pfl5,
    pfa6,
    pfl6,
    pfa7,
    amp_factor,
):
    """Evaluate SPAtmplt at arbitrary frequencies in JAX."""
    _ensure_x64()
    freqs = to_jax(sample_points)
    safe_f = jnp.where(freqs <= 0.0, 1.0, freqs)

    v = (float(piM) * safe_f) ** (1.0 / 3.0)
    v = jnp.where(freqs <= 0.0, 0.0, v)
    logv = jnp.log(v)
    log4 = 2.0 * LN2

    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v2 * v3
    v6 = v3 * v3
    v7 = v3 * v4

    ph = pfa7 * v7 + (pfa6 + pfl6 * (logv + log4)) * v6 + (pfa5 + pfl5 * logv) * v5 + pfa4 * v4 + pfa3 * v3 + pfa2 * v2 + 1.0
    safe_v5 = jnp.where(v5 == 0.0, 1.0, v5)
    phasing = ph * (pfaN / safe_v5) - PI_4
    phasing = phasing - jnp.trunc(phasing / TWOPI) * TWOPI
    phasing = jnp.where(phasing < -PI, phasing + TWOPI, phasing)
    phasing = jnp.where(phasing > PI, phasing - TWOPI, phasing)

    sinp = 1.273239545 * phasing - 0.405284735 * phasing * jnp.abs(phasing)
    sinp = 0.225 * (sinp * jnp.abs(sinp) - sinp) + sinp

    phs = phasing + math.pi / 2.0
    phs = jnp.where(phs > PI, phs - TWOPI, phs)
    cosp = 1.273239545 * phs - 0.405284735 * phs * jnp.abs(phs)
    cosp = 0.225 * (cosp * jnp.abs(cosp) - cosp) + cosp

    amp2 = amp_factor * (safe_f ** (-7.0 / 6.0))
    amp2 = jnp.where(freqs <= 0.0, 0.0, amp2)

    res = cosp * amp2 - 1j * sinp * amp2
    return PyCBCArray(JAXArrayData(res), copy=False)
