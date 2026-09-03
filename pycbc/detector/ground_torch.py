# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch implementation of the full ground-detector strain response.

The scalar detector geometry is evaluated through :mod:`lal` at the same
four-Hertz cadence as ``XLALSimDetectorStrainREAL8TimeSeries``.  Waveform
mixing, finite-arm interpolation, and output assembly remain on the active
Torch device.
"""

import math

import lal
import numpy as np
import torch

from pycbc.types import TimeSeries
from pycbc.types.array_torch import TorchArrayData


# Coefficients from the Cephes sine-integral approximation.  The real-valued
# SciPy implementation uses the same approximation.
_SN = (
    -8.39167827910303881427e-11,
    4.62591714427012837309e-8,
    -9.75759303843632795789e-6,
    9.76945438170435310816e-4,
    -4.13470316229406538752e-2,
    1.00000000000000000302,
)
_SD = (
    2.03269266195951942049e-12,
    1.27997891179943299903e-9,
    4.41827842801218905784e-7,
    9.96412122043875552487e-5,
    1.42085239326149893930e-2,
    9.99999999999999996984e-1,
)
_FN4 = (
    4.23612862892216586994,
    5.45937717161812843388,
    1.62083287701538329132,
    1.67006611831323023771e-1,
    6.81020132472518137426e-3,
    1.08936580650328664411e-4,
    5.48900223421373614008e-7,
)
_FD4 = (
    8.16496634205391016773,
    7.30828822505564552187,
    1.86792257950184183883,
    1.78792052963149907262e-1,
    7.01710668322789753610e-3,
    1.10034357153915731354e-4,
    5.48900252756255700982e-7,
)
_GN4 = (
    8.71001698973114191777e-2,
    6.11379109952219284151e-1,
    3.97180296392337498885e-1,
    7.48527737628469092119e-2,
    5.38868681462177273157e-3,
    1.61999794598934024525e-4,
    1.97963874140963632189e-6,
    7.82579040744090311069e-9,
)
_GD4 = (
    1.64402202413355338886,
    6.66296701268987968381e-1,
    9.88771761277688796203e-2,
    6.22396345441768420760e-3,
    1.73221081474177119497e-4,
    2.02659182086343991969e-6,
    7.82579218933534490868e-9,
)
_FN8 = (
    4.55880873470465315206e-1,
    7.13715274100146711374e-1,
    1.60300158222319456320e-1,
    1.16064229408124407915e-2,
    3.49556442447859055605e-4,
    4.86215430826454749482e-6,
    3.20092790091004902806e-8,
    9.41779576128512936592e-11,
    9.70507110881952024631e-14,
)
_FD8 = (
    9.17463611873684053703e-1,
    1.78685545332074536321e-1,
    1.22253594771971293032e-2,
    3.58696481881851580297e-4,
    4.92435064317881464393e-6,
    3.21956939101046018377e-8,
    9.43720590350276732376e-11,
    9.70507110881952025725e-14,
)
_GN8 = (
    6.97359953443276214934e-1,
    3.30410979305632063225e-1,
    3.84878767649974295920e-2,
    1.71718239052347903558e-3,
    3.48941165502279436777e-5,
    3.47131167084116673800e-7,
    1.70404452782044526189e-9,
    3.85945925430276600453e-12,
    3.14040098946363334640e-15,
)
_GD8 = (
    1.68548898811011640017,
    4.87852258695304967486e-1,
    4.67913194259625806320e-2,
    1.90284426674399523638e-3,
    3.68475504442561108162e-5,
    3.57043223443740838771e-7,
    1.72693748966316146736e-9,
    3.87830166023954706752e-12,
    3.14040098946363335242e-15,
)


def _polevl(x, coefficients):
    value = x * coefficients[0] + coefficients[1]
    for coefficient in coefficients[2:]:
        value = value * x + coefficient
    return value


def _p1evl(x, coefficients):
    value = x + coefficients[0]
    for coefficient in coefficients[1:]:
        value = value * x + coefficient
    return value


def _torch_sine_integral(x):
    """Evaluate the real sine integral without leaving the Torch device."""
    absolute = torch.abs(x)

    small = absolute <= 4.0
    if small.all():
        z = absolute * absolute
        result = absolute * _polevl(z, _SN) / _polevl(z, _SD)
        return torch.sign(x) * result

    large = absolute >= 8.0
    if large.all():
        z = 1.0 / (absolute * absolute)
        f = _polevl(z, _FN8) / (absolute * _p1evl(z, _FD8))
        g = z * _polevl(z, _GN8) / _p1evl(z, _GD8)
        result = (
            (math.pi / 2.0)
            - f * torch.cos(absolute)
            - g * torch.sin(absolute)
        )
        return torch.sign(x) * result

    result = torch.empty_like(absolute)
    if small.any():
        val_small = absolute[small]
        z = val_small * val_small
        result[small] = val_small * _polevl(z, _SN) / _polevl(z, _SD)

    mid = (absolute > 4.0) & (absolute < 8.0)
    if mid.any():
        val_mid = absolute[mid]
        z = 1.0 / (val_mid * val_mid)
        f = _polevl(z, _FN4) / (val_mid * _p1evl(z, _FD4))
        g = z * _polevl(z, _GN4) / _p1evl(z, _GD4)
        result[mid] = (
            (math.pi / 2.0)
            - f * torch.cos(val_mid)
            - g * torch.sin(val_mid)
        )

    if large.any():
        val_large = absolute[large]
        z = 1.0 / (val_large * val_large)
        f = _polevl(z, _FN8) / (val_large * _p1evl(z, _FD8))
        g = z * _polevl(z, _GN8) / _p1evl(z, _GD8)
        result[large] = (
            (math.pi / 2.0)
            - f * torch.cos(val_large)
            - g * torch.sin(val_large)
        )

    return torch.sign(x) * result


def _c_round(value):
    """C99 round/lround semantics for a scalar."""
    if value >= 0:
        return math.floor(value + 0.5)
    return math.ceil(value - 0.5)


def _gps_ns(gps):
    return int(gps.gpsSeconds) * 1_000_000_000 + int(gps.gpsNanoSeconds)


def _gps_offsets_ns(indices, delta_t):
    """Mirror XLALGPSAdd(base, index * delta_t) in integer nanoseconds."""
    seconds = indices.astype(np.float64) * delta_t
    integral = np.floor(seconds)
    nanoseconds = np.rint((seconds - integral) * 1.0e9)
    return (
        integral.astype(np.int64) * 1_000_000_000
        + nanoseconds.astype(np.int64)
    )


def _high_frequency_kernel(
    kernel_length, residual, arm_length_samples, arm_cosine, *, device, dtype
):
    half_length = (kernel_length - 1) // 2
    x = (
        torch.arange(
            -half_length,
            half_length + 1,
            device=device,
            dtype=dtype,
        )
        + residual
    )
    welch_factor = 1.0 / (half_length + 1.0)
    y = welch_factor * x
    first = _torch_sine_integral(
        math.pi * (x + arm_length_samples * arm_cosine)
    )
    second = _torch_sine_integral(math.pi * (x + arm_length_samples))
    third = _torch_sine_integral(math.pi * (x - arm_length_samples))
    kernel = (
        (
            (second - first) / (arm_length_samples * (1.0 - arm_cosine))
            + (first - third) / (arm_length_samples * (1.0 + arm_cosine))
        )
        * (1.0 - y * y)
        / (2.0 * math.pi)
    )
    return torch.where(torch.abs(y) < 1.0, kernel, torch.zeros_like(kernel))


def _response_parts(lal_detector, ra, dec, polarization, gps):
    gmst = lal.GreenwichMeanSiderealTime(gps)
    return lal.ComputeDetAMResponseParts(
        lal_detector, ra, dec, polarization, gmst
    )


def _mix_arm_signals(
    hp, hc, input_epoch, delta_t, interval, lal_detector, ra, dec, polarization
):
    xsignal = torch.empty_like(hp)
    ysignal = torch.empty_like(hp)
    for start in range(0, hp.numel(), interval):
        stop = min(start + interval, hp.numel())
        gps = lal.LIGOTimeGPS(input_epoch) + start * delta_t
        parts = _response_parts(lal_detector, ra, dec, polarization, gps)
        fxplus, fyplus, fxcross, fycross = parts[3:]
        xsignal[start:stop] = (
            fxplus * hp[start:stop] + fxcross * hc[start:stop]
        )
        ysignal[start:stop] = (
            fyplus * hp[start:stop] + fycross * hc[start:stop]
        )
    return xsignal, ysignal


def _output_shape_and_epoch(
    input_epoch,
    input_length,
    delta_t,
    kernel_length,
    arm_length_samples,
    lal_detector,
    ra,
    dec,
):
    end = lal.LIGOTimeGPS(input_epoch) + input_length * delta_t
    delay_start = lal.TimeDelayFromEarthCenter(
        lal_detector.location, ra, dec, input_epoch
    )
    delay_end = lal.TimeDelayFromEarthCenter(
        lal_detector.location, ra, dec, end
    )
    output_length = (
        input_length
        + kernel_length
        - 1
        + math.ceil((delay_start - delay_end) / delta_t)
        + _c_round(4.0 * arm_length_samples)
    )

    output_epoch = lal.LIGOTimeGPS(input_epoch)
    output_epoch += delay_start - (kernel_length - 1) / 2 * delta_t
    fraction = output_epoch.gpsNanoSeconds / 1.0e9
    output_epoch += _c_round(fraction / delta_t) * delta_t - fraction
    return output_length, output_epoch


def _apply_kernel_segment(
    output, output_start, centers, xsignal, ysignal, xkernel, ykernel,
    offsets=None
):
    kernel_length = xkernel.numel()
    half_length = (kernel_length - 1) // 2
    if isinstance(centers, torch.Tensor):
        center_tensor = centers
    else:
        center_tensor = torch.as_tensor(
            centers,
            device=output.device,
            dtype=torch.int64,
        )
    if offsets is None:
        offsets = torch.arange(
            -half_length,
            half_length + 1,
            device=output.device,
            dtype=torch.int64,
        )
    indices = center_tensor[:, None] + offsets[None, :]
    sig_len = xsignal.numel()
    num_centers = center_tensor.shape[0]

    if num_centers > 0 and indices[0, 0] >= 0 and indices[-1, -1] < sig_len:
        xvalues = xsignal[indices]
        yvalues = ysignal[indices]
    else:
        valid = (indices >= 0) & (indices < sig_len)
        safe_indices = torch.clamp(indices, 0, sig_len - 1)
        xvalues = xsignal[safe_indices] * valid
        yvalues = ysignal[safe_indices] * valid

    output[output_start:output_start + num_centers] = (
        torch.matmul(xvalues, xkernel) + torch.matmul(yvalues, ykernel)
    )


def _project_output(
    xsignal,
    ysignal,
    input_epoch,
    output_epoch,
    output_length,
    delta_t,
    interval,
    kernel_length,
    lal_detector,
    ra,
    dec,
    polarization,
):
    output = torch.empty(
        output_length,
        device=xsignal.device,
        dtype=xsignal.dtype,
    )
    source_ns = _gps_ns(input_epoch)
    output_ns = _gps_ns(output_epoch)
    threshold = 1.0 / (4.0 * kernel_length)
    cached_residual = 2.0
    xkernel = ykernel = None

    half_length = (kernel_length - 1) // 2
    offsets = torch.arange(
        -half_length,
        half_length + 1,
        device=output.device,
        dtype=torch.int64,
    )

    for response_start in range(0, output_length, interval):
        response_stop = min(response_start + interval, output_length)
        response_gps = lal.LIGOTimeGPS(output_epoch) + response_start * delta_t
        delay = -lal.TimeDelayFromEarthCenter(
            lal_detector.location, ra, dec, response_gps
        )
        delay_ns = _gps_ns(lal.LIGOTimeGPS(delay))
        parts = _response_parts(
            lal_detector, ra, dec, polarization, response_gps
        )
        arm_length_samples = parts[0] / (lal.C_SI * delta_t)
        xcos, ycos = parts[1:3]

        sample_indices = np.arange(
            response_start, response_stop, dtype=np.int64
        )
        geocenter_ns = (
            output_ns + _gps_offsets_ns(sample_indices, delta_t) + delay_ns
        )
        coordinates = (
            (geocenter_ns - source_ns).astype(np.float64) / 1.0e9 / delta_t
        )
        centers = np.where(
            coordinates >= 0,
            np.floor(coordinates + 0.5),
            np.ceil(coordinates - 0.5),
        ).astype(np.int64)
        residuals = centers - coordinates

        cursor = 0
        while cursor < len(centers):
            if abs(residuals[cursor] - cached_residual) >= threshold:
                cached_residual = float(residuals[cursor])
                xkernel = _high_frequency_kernel(
                    kernel_length,
                    cached_residual,
                    arm_length_samples,
                    xcos,
                    device=output.device,
                    dtype=output.dtype,
                )
                ykernel = _high_frequency_kernel(
                    kernel_length,
                    cached_residual,
                    arm_length_samples,
                    ycos,
                    device=output.device,
                    dtype=output.dtype,
                )

            changed = np.flatnonzero(
                np.abs(residuals[cursor:] - cached_residual) >= threshold
            )
            segment_stop = (
                cursor + int(changed[0]) if changed.size else len(centers)
            )
            _apply_kernel_segment(
                output,
                response_start + cursor,
                centers[cursor:segment_stop],
                xsignal,
                ysignal,
                xkernel,
                ykernel,
                offsets=offsets,
            )
            cursor = segment_stop

    return output


def project_wave(detector, hp, hc, ra, dec, polarization):
    """Project two Torch-backed polarizations with the full LAL response."""
    hp_tensor = getattr(getattr(hp, "_data", None), "tensor", None)
    hc_tensor = getattr(getattr(hc, "_data", None), "tensor", None)
    if hp_tensor is None or hc_tensor is None:
        raise TypeError("Torch detector projection requires Torch time series")
    if not isinstance(hp, TimeSeries) or not isinstance(hc, TimeSeries):
        raise TypeError("Waveform polarizations must be time series")
    if len(hp) != len(hc) or hp.delta_t != hc.delta_t:
        raise ValueError("Waveform polarizations must have matching grids")
    if not len(hp):
        raise ValueError("Waveform polarizations must not be empty")
    if hp.start_time != hc.start_time:
        raise ValueError("Waveform polarizations must have matching epochs")
    if hp_tensor.device != hc_tensor.device:
        raise ValueError("Waveform polarizations must be on the same device")
    if hp_tensor.is_complex() or hc_tensor.is_complex():
        raise TypeError("Detector projection requires real polarizations")

    output_dtype = (
        torch.float32 if hp_tensor.device.type == "mps" else torch.float64
    )
    hp_tensor = hp_tensor.to(dtype=output_dtype)
    hc_tensor = hc_tensor.to(dtype=output_dtype)
    delta_t = hp.delta_t
    input_epoch = lal.LIGOTimeGPS(hp.start_time)
    lal_detector = detector.lal()

    arm_length = (
        lal_detector.frDetector.xArmMidpoint
        + lal_detector.frDetector.yArmMidpoint
    )
    arm_length_samples = arm_length / (lal.C_SI * delta_t)
    kernel_length = 67 + 48 * _c_round(2.0 * arm_length_samples)
    interval = max(1, _c_round(0.25 / delta_t))

    xsignal, ysignal = _mix_arm_signals(
        hp_tensor,
        hc_tensor,
        input_epoch,
        delta_t,
        interval,
        lal_detector,
        ra,
        dec,
        polarization,
    )
    output_length, output_epoch = _output_shape_and_epoch(
        input_epoch,
        len(hp),
        delta_t,
        kernel_length,
        arm_length_samples,
        lal_detector,
        ra,
        dec,
    )
    output = _project_output(
        xsignal,
        ysignal,
        input_epoch,
        output_epoch,
        output_length,
        delta_t,
        interval,
        kernel_length,
        lal_detector,
        ra,
        dec,
        polarization,
    )
    return TimeSeries(
        TorchArrayData(output),
        delta_t=delta_t,
        epoch=output_epoch,
        copy=False,
    )


def project_wave_network(network, hp, hc, ra, dec, polarization):

    """Project two Torch-backed polarizations across a network of detectors."""
    from .ground import NetworkGeometry

    if not isinstance(network, NetworkGeometry):
        network = NetworkGeometry(network)
    return {
        det.name: project_wave(det, hp, hc, ra, dec, polarization)
        for det in network.detectors
    }


__all__ = [
    "project_wave",
    "project_wave_network",
]
