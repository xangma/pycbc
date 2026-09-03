# Copyright (C) 1996-2007 Brian Gough and Gerard Jungman
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

"""Deterministic GSL-compatible hybrid root solving for SEOBNRv4PHM.

The LAL initial-condition calculation uses GSL's scaled ``hybrids`` solver.
Some macOS GSL builds leave their CBLAS symbols for runtime lookup.  This
module ports the small GSL control algorithm to Python and uses either the
companion GSL CBLAS library or an exact scalar implementation of the tiny BLAS
subset, making that arithmetic stable on every supported Torch device.

The implementation follows GSL 2.8 ``multiroots/hybrid.c``, ``dogleg.c``,
``fdjac.c``, and the QR routines.  Those sources are Copyright (C) 1996-2007
Brian Gough and Gerard Jungman and are distributed under GPLv3 or later.
"""

from __future__ import annotations

import ctypes
import math
from collections.abc import Callable, Iterable, Sequence

_DOUBLE_PTR = ctypes.POINTER(ctypes.c_double)
_SQRT_DBL_EPSILON = math.sqrt(2.2204460492503131e-16)

# CBLAS enums used below.
_ROW_MAJOR = 101
_NO_TRANS = 111
_TRANS = 112
_UPPER = 121
_NON_UNIT = 131


class GslCblasUnavailable(RuntimeError):
    """Raised when no companion GSL CBLAS library can be loaded."""


class _GslCblas:
    def __init__(self, path: str):
        self.library = ctypes.CDLL(path)
        self.dnrm2 = self._bind(
            "cblas_dnrm2",
            [ctypes.c_int, _DOUBLE_PTR, ctypes.c_int],
            ctypes.c_double,
        )
        self.dscal = self._bind(
            "cblas_dscal",
            [ctypes.c_int, ctypes.c_double, _DOUBLE_PTR, ctypes.c_int],
        )
        self.ddot = self._bind(
            "cblas_ddot",
            [
                ctypes.c_int,
                _DOUBLE_PTR,
                ctypes.c_int,
                _DOUBLE_PTR,
                ctypes.c_int,
            ],
            ctypes.c_double,
        )
        self.daxpy = self._bind(
            "cblas_daxpy",
            [
                ctypes.c_int,
                ctypes.c_double,
                _DOUBLE_PTR,
                ctypes.c_int,
                _DOUBLE_PTR,
                ctypes.c_int,
            ],
        )
        self.dgemv = self._bind(
            "cblas_dgemv",
            [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_double,
                _DOUBLE_PTR,
                ctypes.c_int,
                _DOUBLE_PTR,
                ctypes.c_int,
                ctypes.c_double,
                _DOUBLE_PTR,
                ctypes.c_int,
            ],
        )
        self.dger = self._bind(
            "cblas_dger",
            [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_double,
                _DOUBLE_PTR,
                ctypes.c_int,
                _DOUBLE_PTR,
                ctypes.c_int,
                _DOUBLE_PTR,
                ctypes.c_int,
            ],
        )
        self.drot = self._bind(
            "cblas_drot",
            [
                ctypes.c_int,
                _DOUBLE_PTR,
                ctypes.c_int,
                _DOUBLE_PTR,
                ctypes.c_int,
                ctypes.c_double,
                ctypes.c_double,
            ],
        )
        self.dtrsv = self._bind(
            "cblas_dtrsv",
            [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                _DOUBLE_PTR,
                ctypes.c_int,
                _DOUBLE_PTR,
                ctypes.c_int,
            ],
        )

    def _bind(self, name, argtypes, restype=None):
        function = getattr(self.library, name)
        function.argtypes = argtypes
        function.restype = restype
        return function


class _PythonCblas:
    """Scalar BLAS subset used when a companion GSL CBLAS is unavailable.

    Keeping these operations explicit preserves the GSL hybrids control flow
    on devices and installations where loading a native CBLAS is impossible.
    The root systems here have only three variables, so a scalar implementation
    is also negligible beside the waveform residual evaluations.
    """

    @staticmethod
    def dnrm2(size, values, stride):
        # Scaled sum of squares, matching the overflow-safe BLAS norm.
        scale = 0.0
        sum_squares = 1.0
        for index in range(size):
            value = abs(values[index * stride])
            if value == 0.0:
                continue
            if scale < value:
                ratio = scale / value
                sum_squares = 1.0 + sum_squares * ratio * ratio
                scale = value
            else:
                ratio = value / scale
                sum_squares += ratio * ratio
        return scale * math.sqrt(sum_squares) if scale else 0.0

    @staticmethod
    def dscal(size, factor, values, stride):
        for index in range(size):
            offset = index * stride
            values[offset] *= factor

    @staticmethod
    def ddot(size, left, left_stride, right, right_stride):
        result = 0.0
        for index in range(size):
            result += left[index * left_stride] * right[index * right_stride]
        return result

    @staticmethod
    def daxpy(size, factor, source, source_stride, target, target_stride):
        for index in range(size):
            target[index * target_stride] += (
                factor * source[index * source_stride]
            )

    @staticmethod
    def dgemv(
        order,
        transpose,
        rows,
        columns,
        factor,
        matrix,
        leading_dimension,
        source,
        source_stride,
        target_factor,
        target,
        target_stride,
    ):
        if order != _ROW_MAJOR:
            raise ValueError("ported hybrids expects row-major BLAS matrices")
        if transpose == _TRANS:
            for column in range(columns):
                total = 0.0
                for row in range(rows):
                    total += (
                        matrix[row * leading_dimension + column]
                        * source[row * source_stride]
                    )
                target[column * target_stride] = (
                    factor * total
                    + target_factor * target[column * target_stride]
                )
        elif transpose == _NO_TRANS:
            for row in range(rows):
                total = 0.0
                for column in range(columns):
                    total += (
                        matrix[row * leading_dimension + column]
                        * source[column * source_stride]
                    )
                target[row * target_stride] = (
                    factor * total + target_factor * target[row * target_stride]
                )
        else:
            raise ValueError("unsupported BLAS transpose selector")

    @staticmethod
    def dger(
        order,
        rows,
        columns,
        factor,
        left,
        left_stride,
        right,
        right_stride,
        matrix,
        leading_dimension,
    ):
        if order != _ROW_MAJOR:
            raise ValueError("ported hybrids expects row-major BLAS matrices")
        for row in range(rows):
            scaled_left = factor * left[row * left_stride]
            for column in range(columns):
                matrix[row * leading_dimension + column] += (
                    scaled_left * right[column * right_stride]
                )

    @staticmethod
    def drot(size, left, left_stride, right, right_stride, cosine, sine):
        for index in range(size):
            left_offset = index * left_stride
            right_offset = index * right_stride
            left_value = left[left_offset]
            right_value = right[right_offset]
            left[left_offset] = cosine * left_value + sine * right_value
            right[right_offset] = cosine * right_value - sine * left_value

    @staticmethod
    def dtrsv(
        order,
        triangle,
        transpose,
        diagonal,
        size,
        matrix,
        leading_dimension,
        values,
        stride,
    ):
        if (
            order != _ROW_MAJOR
            or triangle != _UPPER
            or transpose != _NO_TRANS
            or diagonal != _NON_UNIT
        ):
            raise ValueError("unsupported triangular solve in ported hybrids")
        for row in range(size - 1, -1, -1):
            result = values[row * stride]
            for column in range(row + 1, size):
                result -= (
                    matrix[row * leading_dimension + column]
                    * values[column * stride]
                )
            values[row * stride] = result / matrix[
                row * leading_dimension + row
            ]


_CBLAS = None


def _load_cblas(candidates: Iterable[str]) -> _GslCblas | _PythonCblas:
    global _CBLAS

    if _CBLAS is not None:
        return _CBLAS
    for path in candidates:
        try:
            _CBLAS = _GslCblas(path)
        except (AttributeError, OSError):
            continue
        return _CBLAS
    _CBLAS = _PythonCblas()
    return _CBLAS


def _buffer(values: Sequence[float]):
    return (ctypes.c_double * len(values))(*values)


def _pointer(values, offset=0):
    byte_offset = offset * ctypes.sizeof(ctypes.c_double)
    return ctypes.cast(ctypes.byref(values, byte_offset), _DOUBLE_PTR)


def _enorm(values):
    total = 0.0
    for value in values:
        total += value * value
    return math.sqrt(total)


def _scaled_enorm(diag, values):
    total = 0.0
    for scale, value in zip(diag, values, strict=False):
        product = scale * value
        total += product * product
    return math.sqrt(total)


def _qr_decomp_unpack(jacobian, size, cblas):
    packed = _buffer(jacobian)
    tau = _buffer([0.0] * size)
    for i in range(size):
        length = size - i
        if length == 1:
            tau_i = 0.0
        else:
            xnorm = cblas.dnrm2(
                length - 1,
                _pointer(packed, (i + 1) * size + i),
                size,
            )
            if xnorm == 0.0:
                tau_i = 0.0
            else:
                alpha = packed[i * size + i]
                sign = 1.0 if alpha >= 0.0 else -1.0
                beta = -sign * math.hypot(alpha, xnorm)
                tau_i = (beta - alpha) / beta
                scale = 1.0 / (alpha - beta)
                cblas.dscal(
                    length - 1,
                    scale,
                    _pointer(packed, (i + 1) * size + i),
                    size,
                )
                packed[i * size + i] = beta
        tau[i] = tau_i
        if i + 1 >= size or tau_i == 0.0:
            continue

        first = packed[i * size + i]
        packed[i * size + i] = 1.0
        rows = size - i
        cols = size - i - 1
        cblas.dgemv(
            _ROW_MAJOR,
            _TRANS,
            rows,
            cols,
            1.0,
            _pointer(packed, i * size + i + 1),
            size,
            _pointer(packed, i * size + i),
            size,
            0.0,
            _pointer(tau, i + 1),
            1,
        )
        cblas.dger(
            _ROW_MAJOR,
            rows,
            cols,
            -tau_i,
            _pointer(packed, i * size + i),
            size,
            _pointer(tau, i + 1),
            1,
            _pointer(packed, i * size + i + 1),
            size,
        )
        packed[i * size + i] = first

    q = _buffer(
        [
            1.0 if i == j else 0.0
            for i in range(size)
            for j in range(size)
        ]
    )
    for i in range(size - 1, -1, -1):
        tau_i = tau[i]
        if tau_i == 0.0:
            continue
        length = size - i
        for j in range(length):
            q0 = i * size + i + j
            work = q[q0]
            if length > 1:
                work += cblas.ddot(
                    length - 1,
                    _pointer(q, (i + 1) * size + i + j),
                    size,
                    _pointer(packed, (i + 1) * size + i),
                    size,
                )
            q[q0] = q[q0] - tau_i * work
            if length > 1:
                cblas.daxpy(
                    length - 1,
                    -tau_i * work,
                    _pointer(packed, (i + 1) * size + i),
                    size,
                    _pointer(q, (i + 1) * size + i + j),
                    size,
                )

    r = _buffer([0.0] * (size * size))
    for i in range(size):
        for j in range(i, size):
            r[i * size + j] = packed[i * size + j]
    return q, r


def _givens(a, b):
    if b == 0.0:
        return 1.0, 0.0
    if abs(b) > abs(a):
        tangent = -a / b
        sine = 1.0 / math.sqrt(1.0 + tangent * tangent)
        return sine * tangent, sine
    tangent = -b / a
    cosine = 1.0 / math.sqrt(1.0 + tangent * tangent)
    return cosine, cosine * tangent


def _apply_givens(q, r, size, i, j, cosine, sine, cblas):
    cblas.drot(
        size,
        _pointer(q, i),
        size,
        _pointer(q, j),
        size,
        cosine,
        -sine,
    )
    start = min(i, j)
    cblas.drot(
        size - start,
        _pointer(r, i * size + start),
        1,
        _pointer(r, j * size + start),
        1,
        cosine,
        -sine,
    )


def _qr_update(q, r, w, v, size, cblas):
    for k in range(size - 1, 0, -1):
        cosine, sine = _givens(w[k - 1], w[k])
        wi = w[k - 1]
        wj = w[k]
        w[k - 1] = cosine * wi - sine * wj
        w[k] = sine * wi + cosine * wj
        _apply_givens(q, r, size, k - 1, k, cosine, sine, cblas)
    w0 = w[0]
    for j in range(size):
        r[j] = r[j] + w0 * v[j]
    for k in range(1, size):
        index = (k - 1) * size + k - 1
        cosine, sine = _givens(r[index], r[k * size + k - 1])
        _apply_givens(q, r, size, k - 1, k, cosine, sine, cblas)
        r[k * size + k - 1] = 0.0


def _evaluate(residual, values, size):
    result = [float(value) for value in residual(values)]
    if len(result) != size:
        raise ValueError("hybrids residual size does not match its root vector")
    return result


def _fdjac(residual, x, f):
    size = len(x)
    jacobian = [0.0] * (size * size)
    x_trial = list(x)
    for j in range(size):
        xj = x[j]
        step = _SQRT_DBL_EPSILON * abs(xj)
        if step == 0.0:
            step = _SQRT_DBL_EPSILON
        x_trial[j] = xj + step
        f_trial = _evaluate(residual, x_trial, size)
        x_trial[j] = xj
        for i in range(size):
            jacobian[i * size + j] = (f_trial[i] - f[i]) / step
    return jacobian


def _diag(jacobian, old=None):
    size = math.isqrt(len(jacobian))
    result = [] if old is None else list(old)
    for j in range(size):
        total = 0.0
        for i in range(size):
            value = jacobian[i * size + j]
            total += value * value
        if total == 0.0:
            total = 1.0
        norm = math.sqrt(total)
        if old is None:
            result.append(norm)
        elif norm > result[j]:
            result[j] = norm
    return result


def _qtf(q, f, size):
    result = []
    for j in range(size):
        total = 0.0
        for i in range(size):
            total += q[i * size + j] * f[i]
        result.append(total)
    return result


def _rdx(r, dx, size):
    result = []
    for i in range(size):
        total = 0.0
        for j in range(i, size):
            total += r[i * size + j] * dx[j]
        result.append(total)
    return result


def _dogleg(r, qtf, diag, delta, size, cblas):
    newton_buffer = _buffer(qtf)
    cblas.dtrsv(
        _ROW_MAJOR,
        _UPPER,
        _NO_TRANS,
        _NON_UNIT,
        size,
        r,
        size,
        newton_buffer,
        1,
    )
    newton = [-newton_buffer[i] for i in range(size)]
    qnorm = _scaled_enorm(diag, newton)
    if qnorm <= delta:
        return newton

    gradient = []
    for j in range(size):
        total = 0.0
        for i in range(size):
            total += r[i * size + j] * qtf[i]
        gradient.append(-total / diag[j])
    gnorm = _enorm(gradient)
    if gnorm == 0.0:
        return [(delta / qnorm) * value for value in newton]

    gradient = [
        (value / gnorm) / diag[i]
        for i, value in enumerate(gradient)
    ]
    rg = _rdx(r, gradient, size)
    temp = _enorm(rg)
    sgnorm = (gnorm / temp) / temp
    if sgnorm > delta:
        return [delta * value for value in gradient]

    bnorm = _enorm(qtf)
    bg = bnorm / gnorm
    bq = bnorm / qnorm
    dq = delta / qnorm
    dq2 = dq * dq
    sd = sgnorm / delta
    sd2 = sd * sd
    term1 = bg * bq * sd
    u = term1 - dq
    term2 = term1 - dq * sd2 + math.sqrt(
        u * u + (1.0 - dq2) * (1.0 - sd2)
    )
    alpha = dq * (1.0 - sd2) / term2
    beta = (1.0 - alpha) * sgnorm
    return [
        alpha * newton[i] + beta * gradient[i]
        for i in range(size)
    ]


def gsl_multiroot_hybrids(
    residual: Callable[[list[float]], Sequence[float]],
    guess: Sequence[float],
    *,
    epsabs: float,
    max_iter: int,
    cblas_candidates: Iterable[str],
) -> tuple[list[float], list[float]] | None:
    """Solve with GSL 2.8's scaled ``hybrids`` arithmetic and stop rule."""

    if epsabs < 0.0:
        raise ValueError("absolute residual tolerance must be non-negative")
    cblas = _load_cblas(cblas_candidates)
    size = len(guess)
    if size == 0:
        raise ValueError("hybrids requires a non-empty root vector")

    x = [float(value) for value in guess]
    f = _evaluate(residual, x, size)
    jacobian = _fdjac(residual, x, f)
    iteration = 1
    fnorm = _enorm(f)
    ncfail = ncsuc = nslow1 = nslow2 = 0
    diag = _diag(jacobian)
    scaled_xnorm = _scaled_enorm(diag, x)
    delta = 100.0 * scaled_xnorm if scaled_xnorm > 0.0 else 100.0
    q, r = _qr_decomp_unpack(jacobian, size, cblas)

    for _ in range(max_iter):
        old_fnorm = fnorm
        qtf = _qtf(q, f, size)
        dx = _dogleg(r, qtf, diag, delta, size, cblas)
        x_trial = [x[i] + dx[i] for i in range(size)]
        pnorm = _scaled_enorm(diag, dx)
        if iteration == 1 and pnorm < delta:
            delta = pnorm
        f_trial = _evaluate(residual, x_trial, size)
        df = [f_trial[i] - f[i] for i in range(size)]
        fnorm1 = _enorm(f_trial)
        if fnorm1 < old_fnorm:
            ratio_norm = fnorm1 / old_fnorm
            actred = 1.0 - ratio_norm * ratio_norm
        else:
            actred = -1.0

        rdx = _rdx(r, dx, size)
        fnorm1p = _enorm(
            [qtf[i] + rdx[i] for i in range(size)]
        )
        if fnorm1p < old_fnorm:
            ratio_norm = fnorm1p / old_fnorm
            prered = 1.0 - ratio_norm * ratio_norm
        else:
            prered = 0.0
        ratio = actred / prered if prered > 0.0 else 0.0

        if ratio < 0.1:
            ncsuc = 0
            ncfail += 1
            delta *= 0.5
        else:
            ncfail = 0
            ncsuc += 1
            if ratio >= 0.5 or ncsuc > 1:
                delta = max(delta, pnorm / 0.5)
            if abs(ratio - 1.0) <= 0.1:
                delta = pnorm / 0.5

        if ratio >= 0.0001:
            x = x_trial
            f = f_trial
            fnorm = fnorm1
            iteration += 1

        nslow1 += 1
        if actred >= 0.001:
            nslow1 = 0
        if actred >= 0.1:
            nslow2 = 0

        if ncfail == 2:
            jacobian = _fdjac(residual, x, f)
            nslow2 += 1
            if iteration == 1:
                diag = _diag(jacobian)
                scaled_xnorm = _scaled_enorm(diag, x)
                delta = (
                    100.0 * scaled_xnorm
                    if scaled_xnorm > 0.0
                    else 100.0
                )
            else:
                diag = _diag(jacobian, diag)
            q, r = _qr_decomp_unpack(jacobian, size, cblas)
        else:
            if pnorm == 0.0:
                return None
            qtdf = _qtf(q, df, size)
            w = [
                (qtdf[i] - rdx[i]) / pnorm
                for i in range(size)
            ]
            v = [
                diag[i] * diag[i] * dx[i] / pnorm
                for i in range(size)
            ]
            _qr_update(q, r, _buffer(w), _buffer(v), size, cblas)
            if nslow2 == 5 or nslow1 == 10:
                return None

        # gsl_multiroot_test_residual uses the L1 norm.
        if sum(abs(value) for value in f) < epsabs:
            return x, f

    return None


__all__ = ["GslCblasUnavailable", "gsl_multiroot_hybrids"]
