"""Torch-specific helpers for :mod:`pycbc.conversions`."""

import operator

import torch

from pycbc.types.backend import coerce_torch_values

_qnm_spline_cache = {}


def broadcast_values(*values):
    """Broadcast conversion inputs on the device of their first tensor."""
    module, converted = coerce_torch_values(*values)
    if module is None:
        return None, values
    return torch, torch.broadcast_tensors(*converted)


def qnm_spline(pykerr, spin, mode_l, mode_m, overtone, reim):
    """Evaluate a cached pykerr QNM spline without moving data to the CPU."""
    try:
        mode_l = operator.index(mode_l)
        mode_m = operator.index(mode_m)
        overtone = operator.index(overtone)
    except TypeError as exc:
        raise TypeError("Torch QNM mode indices must be scalar integers") from exc

    max_spin = pykerr.qnm.MAX_SPIN
    if bool(torch.any(torch.abs(spin) > max_spin)):
        raise ValueError(f"|spin| must be < {max_spin}")

    key = (
        reim,
        mode_l,
        abs(mode_m),
        overtone,
        spin.device.type,
        spin.device.index,
        spin.dtype,
    )
    try:
        knots, coefficients = _qnm_spline_cache[key]
    except KeyError:
        if reim == "re":
            cache = pykerr.qnm._reomega_splines
        else:
            cache = pykerr.qnm._imomega_splines
        spline = pykerr.qnm._getspline("omega", reim, mode_l, mode_m, overtone, cache)
        knots = torch.as_tensor(spline.x, dtype=spin.dtype, device=spin.device)
        coefficients = torch.as_tensor(spline.c, dtype=spin.dtype, device=spin.device)
        _qnm_spline_cache[key] = knots, coefficients

    points = spin.contiguous()
    indices = torch.searchsorted(knots, points) - 1
    indices = indices.clamp(0, knots.numel() - 2)
    offset = points - knots[indices]
    coeff = coefficients[:, indices]
    return ((coeff[0] * offset + coeff[1]) * offset + coeff[2]) * offset + coeff[3]


def real_cuberoot(value):
    """Return the real cube root without losing Torch autograd state."""
    nonzero = value != 0
    magnitude = torch.where(nonzero, torch.abs(value), torch.ones_like(value))
    result = torch.sign(value) * magnitude.pow(1.0 / 3.0)
    return torch.where(nonzero, result, torch.zeros_like(result))
