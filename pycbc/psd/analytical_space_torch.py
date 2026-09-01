# Copyright (C) 2026
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
"""Torch kernels for direct analytical space-detector PSD models."""

import math

import torch

from pycbc.psd.read_torch import _log_interpolate
from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData


_LIGHT_SPEED = 299792458.0
_TDI_ERROR = "The version of TDI, currently only for 1.5 or 2.0."

_TIANQIN_DURATIONS = (0.5, 1, 2, 4, 5)
_TIANQIN_CONFUSION_COEFFICIENTS = (
    (-18.6, -1.22, 0.009, -1.87, 0.65, 3.6, -4.6),
    (-18.6, -1.13, -0.945, -1.02, 4.05, -4.5, -0.5),
    (-18.6, -1.45, 0.315, -1.19, -4.48, 10.8, -9.4),
    (-18.6, -1.43, -0.687, 0.24, -0.15, -1.8, -3.2),
    (-18.6, -1.51, -0.710, -1.13, -0.83, 13.2, -19.1),
)
_TAIJI_DURATIONS = (0.5, 1, 2, 4)
_TAIJI_CONFUSION_COEFFICIENTS = (
    (-85.3498, -2.64899, -0.0699707, -0.478447, -0.334821, 0.0658353),
    (-85.4336, -2.46276, -0.183175, -0.884147, -0.427176, 0.128666),
    (-85.3919, -2.69735, -0.749294, -1.15302, -0.302761, 0.175521),
    (-85.5448, -3.23671, -1.64187, -1.14711, 0.0325887, 0.187854),
)


def _require_float64_device(device):
    if device.type == "mps":
        raise TypeError(
            "Analytical space-detector PSDs require float64; Torch MPS only "
            "supports float32, which underflows their physical dynamic range"
        )


def _frequency_grid(length, delta_f, low_freq_cutoff, device):
    return torch.linspace(
        low_freq_cutoff,
        (length - 1) * 2.0 * delta_f,
        length,
        dtype=torch.float64,
        device=device,
    )


def _as_frequency_series(values, frequencies, length, delta_f, cutoff):
    output = _log_interpolate(
        frequencies,
        values,
        length,
        delta_f,
        cutoff,
    )
    return FrequencySeries(
        TorchArrayData(output),
        delta_f=delta_f,
        copy=False,
    )


def _combine_frequency_series(first, second, second_scale=1.0):
    values = first._data.tensor + second_scale * second._data.tensor
    return FrequencySeries(
        TorchArrayData(values),
        delta_f=first.delta_f,
        copy=False,
    )


def _noise_components(
    frequencies,
    detector,
    acc_noise_level,
    oms_noise_level,
):
    """Return acceleration and optical-metrology noise on ``frequencies``."""
    angular_frequency = 2.0 * math.pi * frequencies
    frequency_over_c = angular_frequency / _LIGHT_SPEED

    if detector == "tianqin":
        acceleration_displacement = (
            acc_noise_level**2 * angular_frequency**-4 * (1.0 + 1e-4 / frequencies)
        )
        oms_displacement = oms_noise_level**2
    elif detector in ("lisa", "taiji"):
        acceleration_displacement = (
            acc_noise_level**2
            * (1.0 + (4e-4 / frequencies) ** 2)
            * (1.0 + (frequencies / 8e-3) ** 4)
            * angular_frequency**-4
        )
        oms_displacement = oms_noise_level**2 * (1.0 + (2e-3 / frequencies) ** 4)
    else:
        raise ValueError(f"Unknown space detector {detector!r}")

    return (
        frequency_over_c**2 * acceleration_displacement,
        frequency_over_c**2 * oms_displacement,
    )


def _tdi_psd(channel, omega_length, acceleration_noise, oms_noise):
    sine = torch.sin(omega_length)
    cosine = torch.cos(omega_length)

    if channel == "XYZ":
        return (
            16.0
            * sine**2
            * (oms_noise + acceleration_noise * (3.0 + torch.cos(2.0 * omega_length)))
        )
    if channel == "AE":
        return (
            8.0
            * sine**2
            * (
                4.0 * (1.0 + cosine + cosine**2) * acceleration_noise
                + (2.0 + cosine) * oms_noise
            )
        )
    if channel == "T":
        half_sine = torch.sin(omega_length / 2.0)
        return (
            32.0
            * sine**2
            * half_sine**2
            * (4.0 * acceleration_noise * half_sine**2 + oms_noise)
        )
    raise ValueError(f"Unknown TDI channel {channel!r}")


def _averaged_fplus_sq_approximated(frequencies, len_arm):
    omega_length = 2.0 * math.pi * frequencies * len_arm / _LIGHT_SPEED
    return (3.0 / 20.0) / (1.0 + 0.6 * omega_length**2)


def _linear_interpolate_extrapolate(query, x_values, y_values):
    """Linearly interpolate, matching SciPy's extrapolating ``interp1d``."""
    upper = torch.searchsorted(x_values, query, right=False)
    upper = torch.clamp(upper, 1, x_values.numel() - 1)
    lower = upper - 1
    x0 = x_values[lower]
    x1 = x_values[upper]
    y0 = y_values[lower]
    y1 = y_values[upper]
    return y0 + (query - x0) * (y1 - y0) / (x1 - x0)


def _averaged_lisa_fplus_sq_numerical(
    frequencies,
    len_arm,
    response_frequencies,
    response_values,
):
    if float(len_arm) != 2.5e9:
        raise ValueError("Currently only support 'len_arm=2.5e9'.")
    response_dtype = (
        frequencies.dtype
        if torch.is_floating_point(frequencies)
        else torch.get_default_dtype()
    )
    query = frequencies.to(dtype=response_dtype)
    response_frequencies = torch.tensor(
        response_frequencies,
        dtype=response_dtype,
        device=frequencies.device,
    )
    response_values = torch.tensor(
        response_values,
        dtype=response_dtype,
        device=frequencies.device,
    )
    return (
        _linear_interpolate_extrapolate(
            query,
            response_frequencies,
            response_values,
        )
        / 16.0
    )


def _averaged_tianqin_fplus_sq(frequencies, len_arm):
    base = _averaged_fplus_sq_approximated(frequencies, len_arm)
    omega_length = 2.0 * math.pi * frequencies * len_arm / _LIGHT_SPEED
    coefficients = (
        1.0,
        1e-4,
        2639e-4,
        231 / 5 * 1e-4,
        -2093 / 1.25 * 1e-4,
        2173e-5,
        2101e-6,
        3027 / 2 * 1e-5,
        -42373 / 5 * 1e-6,
        176087e-8,
        -8023 / 5 * 1e-7,
        5169e-9,
    )
    polynomial = torch.zeros_like(omega_length)
    for coefficient in reversed(coefficients):
        polynomial = polynomial * omega_length + coefficient
    high_frequency = torch.exp(-0.322 * torch.sin(2.0 * omega_length - 4.712) + 0.078)
    return base * torch.where(omega_length < 4.1, polynomial, high_frequency)


def _averaged_tdi_response(
    detector,
    frequencies,
    len_arm,
    tdi,
    response_frequencies=None,
    response_values=None,
):
    tdi = str(tdi)
    if tdi not in ("1.5", "2.0"):
        raise ValueError(_TDI_ERROR)

    omega_length = 2.0 * math.pi * frequencies * len_arm / _LIGHT_SPEED
    if detector == "lisa":
        fplus_sq = _averaged_lisa_fplus_sq_numerical(
            frequencies,
            len_arm,
            response_frequencies,
            response_values,
        )
    elif detector == "tianqin":
        fplus_sq = _averaged_tianqin_fplus_sq(frequencies, len_arm)
    elif detector == "taiji":
        fplus_sq = _averaged_fplus_sq_approximated(frequencies, len_arm)
    else:
        raise ValueError(f"Unknown TDI-response detector {detector!r}")

    response = (4.0 * omega_length) ** 2 * torch.sin(omega_length) ** 2 * fplus_sq
    if tdi == "2.0":
        response = response * (4.0 * torch.sin(2.0 * omega_length) ** 2)
    return response


def analytical_sensitivity_curve(
    detector,
    length,
    delta_f,
    low_freq_cutoff,
    device,
    len_arm=None,
    acc_noise_level=None,
    oms_noise_level=None,
    response_frequencies=None,
    response_values=None,
):
    """Generate an analytical strain-sensitivity curve on a Torch device."""
    _require_float64_device(device)
    frequencies = _frequency_grid(length, delta_f, low_freq_cutoff, device)

    if detector == "lisa_scird":
        displacement = 5.76e-48 * (1.0 + (4e-4 / frequencies) ** 2)
        response = 1.0 + (frequencies / 2.5e-2) ** 2
        values = (
            (10.0 / 3.0)
            * (displacement / (2.0 * math.pi * frequencies) ** 4 + 3.6e-41)
            * response
        )
    elif detector in ("lisa_semi", "tianqin", "taiji"):
        len_arm = float(len_arm)
        noise_detector = "lisa" if detector == "lisa_semi" else detector
        acceleration_noise, oms_noise = _noise_components(
            frequencies,
            noise_detector,
            float(acc_noise_level),
            float(oms_noise_level),
        )
        omega_length = 2.0 * math.pi * frequencies * len_arm / _LIGHT_SPEED
        if detector == "lisa_semi":
            fplus_sq = _averaged_lisa_fplus_sq_numerical(
                frequencies,
                len_arm,
                response_frequencies,
                response_values,
            )
        elif detector == "tianqin":
            fplus_sq = _averaged_tianqin_fplus_sq(frequencies, len_arm)
        else:
            fplus_sq = _averaged_fplus_sq_approximated(frequencies, len_arm)
        values = (
            oms_noise + acceleration_noise * (3.0 + torch.cos(2.0 * omega_length))
        ) / (2.0 * omega_length**2 * fplus_sq)
    else:
        raise ValueError(f"Unknown sensitivity-curve detector {detector!r}")

    return _as_frequency_series(
        values,
        frequencies,
        length,
        delta_f,
        low_freq_cutoff,
    )


def _duration_row(duration, durations, coefficients, detector):
    if duration not in durations:
        if detector == "tianqin":
            choices = "0.5, 1, 2, 4, and 5"
        else:
            choices = "0.5, 1, 2, and 4"
        raise Warning(
            f"Note that the results between {choices} years are extrapolated, "
            "might be non-physical."
        )
    return coefficients[durations.index(duration)]


def analytical_confusion_fit(
    detector,
    length,
    delta_f,
    low_freq_cutoff,
    duration,
    device,
):
    """Generate a Galactic-confusion sensitivity fit on a Torch device."""
    _require_float64_device(device)
    frequencies = _frequency_grid(length, delta_f, low_freq_cutoff, device)

    if detector == "lisa":
        duration = torch.as_tensor(
            duration,
            dtype=torch.float64,
            device=device,
        )
        f1 = torch.pow(10.0, -0.25 * torch.log10(duration) - 2.7)
        fk = torch.pow(10.0, -0.27 * torch.log10(duration) - 2.47)
        values = (
            0.5
            * 1.14e-44
            * frequencies ** (-7.0 / 3.0)
            * torch.exp(-((frequencies / f1) ** 1.8))
            * (1.0 + torch.tanh((fk - frequencies) / 0.31e-3))
        )
    elif detector == "tianqin":
        coefficients = _duration_row(
            duration,
            _TIANQIN_DURATIONS,
            _TIANQIN_CONFUSION_COEFFICIENTS,
            detector,
        )
        log_frequency = torch.log10(frequencies * 1e3)
        exponent = torch.zeros_like(frequencies)
        for power, coefficient in enumerate(coefficients):
            exponent = exponent + coefficient * log_frequency**power
        values = (10.0 / 3.0) * torch.pow(10.0, exponent) ** 2
        anchor = values[torch.argmin(torch.abs(frequencies - 5e-4))]
        values = torch.where(
            (frequencies > 3e-4) & (frequencies < 5e-4),
            anchor,
            values,
        )
        values = torch.where(
            (frequencies < 3e-4) | (frequencies > 1e-2),
            torch.zeros_like(values),
            values,
        )
    elif detector == "taiji":
        coefficients = _duration_row(
            duration,
            _TAIJI_DURATIONS,
            _TAIJI_CONFUSION_COEFFICIENTS,
            detector,
        )
        log_frequency = torch.log(frequencies * 1e3)
        exponent = torch.zeros_like(frequencies)
        for power, coefficient in enumerate(coefficients):
            exponent = exponent + coefficient * log_frequency**power
        values = torch.exp(exponent)
        values = torch.where(
            (frequencies < 1e-4) | (frequencies > 1e-2),
            torch.zeros_like(values),
            values,
        )
    else:
        raise ValueError(f"Unknown confusion-fit detector {detector!r}")

    return _as_frequency_series(
        values,
        frequencies,
        length,
        delta_f,
        low_freq_cutoff,
    )


def analytical_combined_sensitivity_curve(
    detector,
    length,
    delta_f,
    low_freq_cutoff,
    duration,
    device,
    len_arm=None,
    acc_noise_level=None,
    oms_noise_level=None,
    response_frequencies=None,
    response_values=None,
):
    """Generate an instrument-plus-confusion sensitivity curve on-device."""
    if detector in ("lisa_scird", "lisa_semi"):
        max_duration = 10.0
        confusion_detector = "lisa"
    elif detector == "tianqin":
        max_duration = 5.0
        confusion_detector = detector
    elif detector == "taiji":
        max_duration = 4.0
        confusion_detector = detector
    else:
        raise ValueError(f"Unknown combined sensitivity detector {detector!r}")

    base = analytical_sensitivity_curve(
        detector,
        length,
        delta_f,
        low_freq_cutoff,
        device,
        len_arm=len_arm,
        acc_noise_level=acc_noise_level,
        oms_noise_level=oms_noise_level,
        response_frequencies=response_frequencies,
        response_values=response_values,
    )
    if duration < 0 or duration > max_duration:
        raise ValueError(f"Must between 0 and {max_duration:g}.")
    confusion = analytical_confusion_fit(
        confusion_detector,
        length,
        delta_f,
        low_freq_cutoff,
        duration,
        device,
    )
    return _combine_frequency_series(base, confusion)


def analytical_confusion_tdi_psd(
    detector,
    length,
    delta_f,
    low_freq_cutoff,
    len_arm,
    duration,
    tdi,
    device,
    response_frequencies=None,
    response_values=None,
):
    """Generate a LISA, TianQin, or Taiji XYZ confusion PSD on-device."""
    _require_float64_device(device)
    frequencies = _frequency_grid(length, delta_f, low_freq_cutoff, device)
    tdi_response_values = _averaged_tdi_response(
        detector,
        frequencies,
        float(len_arm),
        tdi,
        response_frequencies=response_frequencies,
        response_values=response_values,
    )
    response = _as_frequency_series(
        tdi_response_values,
        frequencies,
        length,
        delta_f,
        low_freq_cutoff,
    )
    confusion = analytical_confusion_fit(
        detector,
        length,
        delta_f,
        low_freq_cutoff,
        duration,
        device,
    )
    values = 2.0 * confusion._data.tensor * response._data.tensor
    return FrequencySeries(
        TorchArrayData(values),
        delta_f=delta_f,
        copy=False,
    )


def analytical_ae_confusion_psd(
    detector,
    length,
    delta_f,
    low_freq_cutoff,
    len_arm,
    acc_noise_level,
    oms_noise_level,
    duration,
    tdi,
    device,
    response_frequencies=None,
    response_values=None,
):
    """Generate an instrument-plus-confusion TDI A/E PSD on-device."""
    instrument = analytical_tdi_psd(
        detector,
        "AE",
        length,
        delta_f,
        low_freq_cutoff,
        len_arm,
        acc_noise_level,
        oms_noise_level,
        tdi,
        device,
    )
    confusion = analytical_confusion_tdi_psd(
        detector,
        length,
        delta_f,
        low_freq_cutoff,
        len_arm,
        duration,
        tdi,
        device,
        response_frequencies=response_frequencies,
        response_values=response_values,
    )
    return _combine_frequency_series(instrument, confusion, second_scale=1.5)


def analytical_sh_transformed_psd(
    length,
    delta_f,
    low_freq_cutoff,
    len_arm,
    acc_noise_level,
    oms_noise_level,
    base_model,
    duration,
    tdi,
    device,
    response_frequencies,
    response_values,
):
    """Transform a LISA strain-sensitivity curve to an XYZ TDI PSD."""
    _require_float64_device(device)
    frequencies = _frequency_grid(length, delta_f, low_freq_cutoff, device)
    response_values_tdi = _averaged_tdi_response(
        "lisa",
        frequencies,
        float(len_arm),
        tdi,
        response_frequencies=response_frequencies,
        response_values=response_values,
    )
    response = _as_frequency_series(
        response_values_tdi,
        frequencies,
        length,
        delta_f,
        low_freq_cutoff,
    )
    if base_model == "semi":
        detector = "lisa_semi"
    elif base_model == "SciRD":
        detector = "lisa_scird"
    else:
        raise ValueError("Must choose from 'semi' or 'SciRD'.")
    sensitivity = analytical_combined_sensitivity_curve(
        detector,
        length,
        delta_f,
        low_freq_cutoff,
        duration,
        device,
        len_arm=len_arm,
        acc_noise_level=acc_noise_level,
        oms_noise_level=oms_noise_level,
        response_frequencies=response_frequencies,
        response_values=response_values,
    )
    values = 2.0 * sensitivity._data.tensor * response._data.tensor
    return FrequencySeries(
        TorchArrayData(values),
        delta_f=delta_f,
        copy=False,
    )


def analytical_tdi_psd(
    detector,
    channel,
    length,
    delta_f,
    low_freq_cutoff,
    len_arm,
    acc_noise_level,
    oms_noise_level,
    tdi,
    device,
):
    """Generate a direct analytical TDI PSD on a Torch CPU or CUDA device."""
    _require_float64_device(device)

    tdi = str(tdi)
    if tdi not in ("1.5", "2.0"):
        raise ValueError(_TDI_ERROR)

    frequencies = _frequency_grid(length, delta_f, low_freq_cutoff, device)
    acceleration_noise, oms_noise = _noise_components(
        frequencies,
        detector,
        float(acc_noise_level),
        float(oms_noise_level),
    )
    omega_length = 2.0 * math.pi * frequencies * float(len_arm) / _LIGHT_SPEED
    psd = _tdi_psd(channel, omega_length, acceleration_noise, oms_noise)
    if tdi == "2.0":
        psd = psd * (4.0 * torch.sin(2.0 * omega_length) ** 2)

    return _as_frequency_series(
        psd,
        frequencies,
        length,
        delta_f,
        low_freq_cutoff,
    )


def analytical_tdi_csd_xy(
    length,
    delta_f,
    low_freq_cutoff,
    len_arm,
    acc_noise_level,
    oms_noise_level,
    tdi,
    device,
):
    """Generate the LISA XY cross spectrum on a Torch CPU or CUDA device."""
    _require_float64_device(device)

    tdi = str(tdi)
    if tdi not in ("1.5", "2.0"):
        raise ValueError(_TDI_ERROR)

    frequencies = _frequency_grid(length, delta_f, low_freq_cutoff, device)
    acceleration_noise, oms_noise = _noise_components(
        frequencies,
        "lisa",
        float(acc_noise_level),
        float(oms_noise_level),
    )
    omega_length = 2.0 * math.pi * frequencies * float(len_arm) / _LIGHT_SPEED
    csd = (
        -8.0
        * torch.sin(omega_length) ** 2
        * torch.cos(omega_length)
        * (oms_noise + 4.0 * acceleration_noise)
    )
    if tdi == "2.0":
        csd = csd * (4.0 * torch.sin(2.0 * omega_length) ** 2)

    # Preserve the public CPU routine's existing log-interpolation semantics,
    # including NaNs where the signed CSD is negative.
    return _as_frequency_series(
        csd,
        frequencies,
        length,
        delta_f,
        low_freq_cutoff,
    )
