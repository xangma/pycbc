"""Ordered point chi-square recurrence for the JAX scheme.

The phase convention, three-product complex multiplication and ordered sums
follow ``chisq_cpu.point_chisq_code``. Independent points and bins run in
parallel; frequency samples within a bin retain their recurrence dependency.
Agreement is qualified against the compiled CPU reference, not assumed to be
bitwise portable across compilers or devices.
"""

import jax
import jax.numpy as jnp


@jax.jit
def shift_sum(correlations, rows, points, bins, n_time, base_k):
    """Return bin powers on the input device without copying correlations out.

    ``correlations`` and absolute ``bins`` each have a row per template.
    Cropped correlations are zero outside their stored interval, but phase
    evolution starts at the original bin boundary, including omitted zeros.
    Both complex64/float32 and complex128/float64 arithmetic are retained.
    """
    real_dtype = correlations.real.dtype
    # Empty crops and point sets have no power. Avoid tracing a gather from
    # an empty frequency axis or a reduction over an empty point/bin axis.
    if (correlations.shape[1] == 0 or points.size == 0 or
            bins.shape[1] < 2):
        return jnp.zeros(points.shape, real_dtype)
    # Signed indices also preserve negative offsets below a cropped interval.
    edges = bins[rows].astype(jnp.int64)
    starts, ends = edges[:, :-1], edges[:, 1:]
    # The CPU first converts shifts to the correlation's real precision.
    shifts = points.astype(real_dtype).astype(jnp.float64)[:, None]
    angle = 2 * 3.141592653 * shifts / n_time
    initial = 2 * 3.141592653 * shifts * starts / n_time
    pr, pi = jnp.cos(initial).astype(real_dtype), jnp.sin(initial).astype(real_dtype)
    rr, ri = jnp.cos(angle).astype(real_dtype), jnp.sin(angle).astype(real_dtype)
    zero = jnp.zeros(starts.shape, real_dtype)
    width = correlations.shape[1]

    def step(offset, carry):
        pr, pi, outr, outi = carry
        k = starts + offset
        active = k < ends
        inside = (k >= base_k) & (k < base_k + width)
        value = correlations[rows[:, None], jnp.clip(k - base_k, 0, width - 1)]
        value = jnp.where(inside, value, 0)
        vr, vi = value.real, value.imag
        # Keep the CPU's three-product multiplication and update sequence.
        k1 = vr * (pr + pi)
        k2 = pr * (vi - vr)
        k3 = pi * (vr + vi)
        next_r = pr * rr - pi * ri
        next_i = pr * ri + pi * rr
        return (jnp.where(active, next_r, pr),
                jnp.where(active, next_i, pi),
                jnp.where(active, outr + (k1 - k3), outr),
                jnp.where(active, outi + (k1 + k2), outi))

    # Group dependent steps in one compiled loop body to reduce GPU launch
    # overhead. Each step still consumes the preceding phase and sum state.
    unroll = 16

    def group(index, carry):
        for offset in range(unroll):
            carry = step(index * unroll + offset, carry)
        return carry

    _, _, outr, outi = jax.lax.fori_loop(
        0, (jnp.max(ends - starts) + unroll - 1) // unroll,
        group, (pr, pi, zero, zero))
    powers = outr * outr + outi * outi
    # Summation over bins is ordered, too; a parallel reduction changes rounding.
    return jax.lax.fori_loop(
        0, powers.shape[1], lambda b, total: total + powers[:, b],
        jnp.zeros(points.shape, real_dtype))
