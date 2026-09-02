# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Small compatibility surface for Torch-native operation without LAL.

PyCBC still uses the real :mod:`lal` package whenever it is installed.  The
fallback provided here is deliberately narrow: scalar physical constants and
GPS metadata are sufficient for native Torch calculations and PyCBC series.
Operations that create LAL objects or resolve LAL data files remain explicit
dependencies and fail with an actionable error.

This module is not installed as ``sys.modules['lal']``.  Code outside the
bounded native surface therefore cannot accidentally mistake it for LAL.
"""

from __future__ import annotations

import math
import operator
import struct
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN


class LALDependencyError(ImportError):
    """Raised when an operation genuinely requires the core LAL package."""


try:
    import lal as _lal
except ImportError:
    _lal = None


LAL_AVAILABLE = _lal is not None


def require_lal(feature="this operation"):
    """Return the real LAL module or raise a precise dependency error."""
    if _lal is None:
        raise LALDependencyError(
            f"{feature} requires the core 'lal' Python package, which is "
            "not installed"
        )
    return _lal


_MISSING = object()


class _FallbackLIGOTimeGPS:
    """Nanosecond GPS timestamp matching LAL's common Python semantics."""

    __slots__ = ("gpsSeconds", "gpsNanoSeconds")
    _NS_PER_SECOND = 1_000_000_000
    _MIN_SECONDS = -(1 << 31)
    _MAX_SECONDS = (1 << 31) - 1

    def __init__(self, seconds=0, nanoseconds=_MISSING):
        if isinstance(seconds, _FallbackLIGOTimeGPS):
            if nanoseconds is not _MISSING:
                raise TypeError("copy construction takes one argument")
            total_ns = seconds._total_nanoseconds()
        elif hasattr(seconds, "gpsSeconds") and hasattr(
            seconds, "gpsNanoSeconds"
        ):
            if nanoseconds is not _MISSING:
                raise TypeError("copy construction takes one argument")
            total_ns = (
                operator.index(seconds.gpsSeconds) * self._NS_PER_SECOND
                + operator.index(seconds.gpsNanoSeconds)
            )
        elif nanoseconds is not _MISSING:
            whole_seconds = operator.index(seconds)
            if not self._MIN_SECONDS <= whole_seconds <= self._MAX_SECONDS:
                raise OverflowError("GPS seconds out of range")
            total_ns = (
                whole_seconds * self._NS_PER_SECOND
                + operator.index(nanoseconds)
            )
        elif isinstance(seconds, str):
            total_ns = self._parse_string(seconds)
        else:
            total_ns = self._real_to_nanoseconds(seconds)
        self._set_total_nanoseconds(total_ns)

    @classmethod
    def _parse_string(cls, value):
        # XLALStrToGPS accepts the empty string and leading whitespace, rejects
        # trailing whitespace, and rounds decimal nanosecond ties to even.
        if value == "":
            return 0
        if value.rstrip() != value:
            raise RuntimeError("invalid GPS time string")
        try:
            decimal = Decimal(value)
        except InvalidOperation as exc:
            raise RuntimeError("invalid GPS time string") from exc
        if not decimal.is_finite():
            raise RuntimeError("non-finite GPS time")
        return int(
            (decimal * cls._NS_PER_SECOND).to_integral_value(
                rounding=ROUND_HALF_EVEN
            )
        )

    @staticmethod
    def _coerce_real(value):
        # SWIG accepts Python int/float and NumPy integer/floating scalars,
        # but not other Number implementations such as Decimal or Fraction.
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(operator.index(value))
        except TypeError:
            pass
        value_type = type(value)
        if value_type.__module__.startswith("numpy") and "float" in (
            value_type.__name__.lower()
        ):
            return float(value)
        raise TypeError("GPS time must be an int, float, string, or GPS time")

    @classmethod
    def _real_to_nanoseconds(cls, value):
        """Match XLALGPSSetREAL8 without a GPS-scale precision loss."""
        try:
            integer = operator.index(value)
        except TypeError:
            pass
        else:
            if not cls._MIN_SECONDS <= integer <= cls._MAX_SECONDS:
                raise RuntimeError("GPS seconds out of range")
            return integer * cls._NS_PER_SECOND

        number = cls._coerce_real(value)
        if not math.isfinite(number):
            raise RuntimeError("non-finite GPS time")
        if abs(number) > cls._MAX_SECONDS:
            raise RuntimeError("GPS seconds out of range")
        seconds = math.floor(number)
        nanoseconds = round(
            (number - seconds) * cls._NS_PER_SECOND
        )
        return seconds * cls._NS_PER_SECOND + nanoseconds

    @classmethod
    def _from_total_nanoseconds(cls, total_ns):
        result = object.__new__(cls)
        result._set_total_nanoseconds(total_ns)
        return result

    def _set_total_nanoseconds(self, total_ns):
        total_ns = operator.index(total_ns)
        seconds = abs(total_ns) // self._NS_PER_SECOND
        if total_ns < 0:
            seconds = -seconds
        if not self._MIN_SECONDS <= seconds <= self._MAX_SECONDS:
            raise RuntimeError("GPS seconds out of range")
        nanoseconds = total_ns - seconds * self._NS_PER_SECOND
        self.gpsSeconds = seconds
        self.gpsNanoSeconds = nanoseconds

    def _total_nanoseconds(self):
        return (
            self.gpsSeconds * self._NS_PER_SECOND + self.gpsNanoSeconds
        )

    @staticmethod
    def _coerce_nanoseconds(value):
        if isinstance(value, _FallbackLIGOTimeGPS):
            return value._total_nanoseconds()
        if hasattr(value, "gpsSeconds") and hasattr(
            value, "gpsNanoSeconds"
        ):
            return (
                operator.index(value.gpsSeconds)
                * _FallbackLIGOTimeGPS._NS_PER_SECOND
                + operator.index(value.gpsNanoSeconds)
            )
        return _FallbackLIGOTimeGPS._real_to_nanoseconds(value)

    @staticmethod
    def _split_double(value):
        """Port LAL's REAL4-leading split used for GPS multiplication."""
        try:
            high = struct.unpack("=f", struct.pack("=f", value))[0]
        except OverflowError:
            high = math.copysign(math.inf, value)
        high -= 2**-23 * high
        return high, value - high

    def _multiply(self, factor):
        if not math.isfinite(factor):
            raise RuntimeError("non-finite GPS time")

        seconds = self.gpsSeconds
        nanoseconds = self.gpsNanoSeconds
        if seconds < 0 < nanoseconds:
            seconds += 1
            nanoseconds -= self._NS_PER_SECOND
        elif seconds > 0 > nanoseconds:
            seconds -= 1
            nanoseconds += self._NS_PER_SECOND

        seconds_low = abs(seconds) % (1 << 16)
        if seconds < 0:
            seconds_low = -seconds_low
        seconds_high = seconds - seconds_low
        factor_high, factor_low = self._split_double(factor)
        products = (
            seconds_low * factor_low,
            seconds_high * factor_low,
            seconds_low * factor_high,
            seconds_high * factor_high,
        )
        fractions = []
        integrals = []
        for product in products:
            fraction, integral = math.modf(product)
            fractions.append(fraction)
            integrals.append(integral)

        fractional_total = (
            fractions[0]
            + fractions[1]
            + fractions[2]
            + fractions[3]
            + nanoseconds * factor / self._NS_PER_SECOND
        )
        integral_total = (
            integrals[0]
            + integrals[1]
            + integrals[2]
            + integrals[3]
        )
        total_ns = self._real_to_nanoseconds(fractional_total)
        total_ns += self._real_to_nanoseconds(integral_total)
        return self._from_total_nanoseconds(total_ns)

    def __float__(self):
        return self.gpsSeconds + self.gpsNanoSeconds / self._NS_PER_SECOND

    def __int__(self):
        # LAL truncates toward zero.  Its signed-nanosecond representation
        # makes that exactly the stored seconds field, without a lossy float
        # round-trip for large GPS values.
        return self.gpsSeconds

    def __repr__(self):
        return (
            f"LIGOTimeGPS({self.gpsSeconds}, {self.gpsNanoSeconds})"
        )

    def __str__(self):
        total_ns = self._total_nanoseconds()
        sign = "-" if total_ns < 0 else ""
        total_ns = abs(total_ns)
        seconds, nanoseconds = divmod(total_ns, self._NS_PER_SECOND)
        if nanoseconds == 0:
            return f"{sign}{seconds}"
        fraction = f"{nanoseconds:09d}".rstrip("0")
        return f"{sign}{seconds}.{fraction}"

    def __hash__(self):
        # This is the hash used by SWIG's LIGOTimeGPS wrapper.
        return hash(self.gpsSeconds ^ self.gpsNanoSeconds)

    def __bool__(self):
        return self._total_nanoseconds() != 0

    def __eq__(self, other):
        try:
            return self._total_nanoseconds() == self._coerce_nanoseconds(other)
        except (TypeError, ValueError, OverflowError, RuntimeError):
            return False

    def __lt__(self, other):
        return self._total_nanoseconds() < self._coerce_nanoseconds(other)

    def __le__(self, other):
        return self._total_nanoseconds() <= self._coerce_nanoseconds(other)

    def __gt__(self, other):
        return self._total_nanoseconds() > self._coerce_nanoseconds(other)

    def __ge__(self, other):
        return self._total_nanoseconds() >= self._coerce_nanoseconds(other)

    def __neg__(self):
        return self._from_total_nanoseconds(-self._total_nanoseconds())

    def __abs__(self):
        return self._from_total_nanoseconds(abs(self._total_nanoseconds()))

    def __add__(self, other):
        return self._from_total_nanoseconds(
            self._total_nanoseconds() + self._coerce_nanoseconds(other)
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self._from_total_nanoseconds(
            self._total_nanoseconds() - self._coerce_nanoseconds(other)
        )

    def __rsub__(self, other):
        return self._from_total_nanoseconds(
            self._coerce_nanoseconds(other) - self._total_nanoseconds()
        )

    def __mul__(self, other):
        factor = self._coerce_real(other)
        return self._multiply(factor)

    __rmul__ = __mul__

    def __truediv__(self, other):
        divisor = self._coerce_real(other)
        if not math.isfinite(divisor):
            raise RuntimeError("non-finite GPS time")
        if divisor == 0:
            raise RuntimeError("division by zero")

        quotient = type(self)(float(self) / divisor)
        threshold = 0.5e-9 * (1 + abs(1 / divisor))
        seen = set()
        for _ in range(100):
            state = (quotient.gpsSeconds, quotient.gpsNanoSeconds)
            if state in seen:
                raise RuntimeError("GPS division did not converge")
            seen.add(state)
            workspace = quotient._multiply(divisor)
            seconds = self.gpsSeconds - workspace.gpsSeconds
            nanoseconds = self.gpsNanoSeconds - workspace.gpsNanoSeconds
            residual = (
                seconds + nanoseconds / self._NS_PER_SECOND
            ) / divisor
            quotient = quotient + residual
            if abs(residual) <= threshold:
                return quotient
        raise RuntimeError("GPS division did not converge")


if _lal is not None:
    LIGOTimeGPS = _lal.LIGOTimeGPS
else:
    LIGOTimeGPS = _FallbackLIGOTimeGPS


_FALLBACK_CONSTANTS = {
    "C_SI": 299792458.0,
    "G_SI": 6.6743e-11,
    "MSUN_SI": 1.9884098706980507e30,
    "MTSUN_SI": 4.925490947641267e-6,
    "MRSUN_SI": 1476.6250380501247,
    "PC_SI": 3.085677581491367e16,
    "PI": 3.141592653589793,
    "TWOPI": 6.283185307179586,
    "PI_2": 1.5707963267948966,
    "PI_4": 0.7853981633974483,
    "PI_180": 0.017453292519943295,
    "GAMMA": 0.5772156649015329,
    "LN2": 0.6931471805599453,
    "HBAR_SI": 1.0545718176461565e-34,
    "K_SI": 1.380649e-23,
    "YRJUL_SI": 31557600.0,
    "REARTH_SI": 6378137.0,
}

for _name, _fallback in _FALLBACK_CONSTANTS.items():
    globals()[_name] = getattr(_lal, _name, _fallback) if _lal else _fallback


def __getattr__(name):
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    if _lal is not None:
        return getattr(_lal, name)
    raise LALDependencyError(
        f"lal.{name} requires the core 'lal' Python package, which is not "
        "installed"
    )


__all__ = [
    "LAL_AVAILABLE",
    "LALDependencyError",
    "LIGOTimeGPS",
    "require_lal",
    *_FALLBACK_CONSTANTS,
]
