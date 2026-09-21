"""Small, explicit arithmetic controls for the two CPU chi-square notebooks.

This NumPy model follows PyCBC 2.11.0's recurrence and three-product multiply,
but is not its compiled kernel: NumPy/libm and compiler contraction can differ.
Four independent switches introduce the old pi literal, float32 phase state,
float32 term operations, and float32 accumulation/powers. Final chi-square
arithmetic is held in float64. No PyCBC library code is changed.
"""

import math

import numpy as np


FACTORS = {1: "Truncated π", 2: "Float32 phase", 4: "Float32 terms",
           8: "Float32 sums / powers"}
COLORS = {1: "#b35806", 2: "#2166ac", 4: "#7570b3", 8: "#678232"}
STYLES = {1: "--", 2: "-", 4: ":", 8: "-."}


def phases(size, point, start, length, single=False, truncated_pi=False):
    """Seed once at the bin start, then multiply by a fixed rotation.

    NumPy's cumulative product retains the requested complex dtype. These
    controls use float64 indices: index casting is a separate notebook probe.
    """
    pi = 3.141592653 if truncated_pi else math.pi
    angle = 2 * pi * float(point) / size
    seed_angle = 2 * pi * float(point) * start / size
    dtype = np.complex64 if single else np.complex128
    rotations = np.full(length, complex(math.cos(angle), math.sin(angle)),
                        dtype=dtype)
    if length:
        rotations[0] = complex(math.cos(seed_angle), math.sin(seed_angle))
    return np.multiply.accumulate(rotations, dtype=dtype)


def terms(values, phase, single=False):
    """Round each operation in PyCBC's three-product complex multiply.

    Phase operands are retained as computed, not pre-cast to the term dtype.
    Rounding each operation separately defines a controlled counterfactual;
    it does not prescribe how a mixed-precision C compiler would evaluate it.
    """
    round_to = np.float32 if single else np.float64
    vr, vi = values.real.astype(np.float64), values.imag.astype(np.float64)
    pr, pi = phase.real.astype(np.float64), phase.imag.astype(np.float64)
    vs, va = round_to(vr + vi), round_to(vi - vr)
    k1 = round_to(vr * round_to(pr + pi))
    k2, k3 = round_to(pr * va), round_to(pi * vs)
    return round_to(k1 - k3), round_to(k1 + k2)


def controlled_powers(corr, points, bins):
    """All 16 combinations; values are cumulative bin powers [point, bin].

    Bits 1/2/4/8 introduce the issues named in FACTORS. Mask 0 uses double
    precision and full pi; mask 15 introduces all four. Accumulation rounds
    terms to its dtype, adds them sequentially, and uses that dtype for bin
    powers and the cumulative sum over bins. It is not NumPy's pairwise sum.
    """
    output = {mask: np.empty((len(points), len(bins)-1), dtype=np.float64)
              for mask in range(16)}
    for column, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        for row, point in enumerate(points):
            for phase_mask in range(4):
                phase = phases(len(corr), point, int(lo), int(hi-lo),
                               bool(phase_mask & 2), bool(phase_mask & 1))
                for term_bit in (0, 4):
                    real, imag = terms(corr[lo:hi], phase, bool(term_bit))
                    for sum_bit in (0, 8):
                        dtype = np.float32 if sum_bit else np.float64
                        qr = np.cumsum(real, dtype=dtype)[-1] if len(real) else dtype(0)
                        qi = np.cumsum(imag, dtype=dtype)[-1] if len(imag) else dtype(0)
                        output[phase_mask | term_bit | sum_bit][row, column] = qr*qr + qi*qi
    for mask in output:
        dtype = np.float32 if mask & 8 else np.float64
        output[mask] = np.cumsum(output[mask], axis=1, dtype=dtype).astype(np.float64)
    return output


def decompose(errors):
    """Möbius decomposition: baseline, four main effects, all interactions.

    Signed contributions reconstruct the all-issues error exactly (up to
    arithmetic rounding). They are measured relative to mask 0, not assigned
    shares of a unique physical error budget. Interactions depend on inputs.
    """
    effects = {}
    for mask in range(16):
        effects[mask] = errors[mask].copy()
        for subset in range(mask):
            if subset & mask == subset:
                effects[mask] -= effects[subset]
    interaction = sum(value for mask, value in effects.items()
                      if mask.bit_count() >= 2)
    reconstructed = effects[0] + sum(effects[bit] for bit in FACTORS) + interaction
    np.testing.assert_allclose(reconstructed, errors[15], rtol=1e-12, atol=1e-12)
    return effects, interaction
