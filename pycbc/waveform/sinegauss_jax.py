"""JAX implementation of the frequency-domain sine-Gaussian."""

import math
import jax.numpy as jnp

from pycbc.types import FrequencySeries


def fd_sine_gaussian(amp, quality, central_frequency, fmin, fmax, delta_f):
    kmin = int(round(fmin / delta_f))
    kmax = int(round(fmax / delta_f))
    pi = math.pi
    tau = quality / (2 * pi * central_frequency)
    f = jnp.arange(kmax, dtype=jnp.float64) * delta_f
    out = jnp.zeros((kmax,), dtype=jnp.complex128)
    cutoff = -50.0
    low = max(kmin, int((central_frequency - (-cutoff)**.5 /
                         (tau * pi)) // delta_f))
    high = min(kmax, int((central_frequency + (-cutoff)**.5 /
                          (tau * pi)) // delta_f))
    term = -(tau * pi * (f[low:high] - central_frequency)) ** 2
    aterm = amp * pi**.5 / 2 * tau
    out = out.at[low:high].set(aterm * jnp.exp(term))
    high2 = int(-cutoff / quality**2 * central_frequency // delta_f)
    if high2 > kmin:
        term2 = -quality**2 * f[kmin:high2] / central_frequency
        out = out.at[kmin:high2].multiply(1 + jnp.exp(term2))
    return FrequencySeries(out, delta_f=delta_f)
