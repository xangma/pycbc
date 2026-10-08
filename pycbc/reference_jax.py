# Copyright (C) 2026 PyCBC contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Opt-in validation against original CPU routines in a separate process.

The original routines use scheme-dispatched arrays and FFTs. A separate CPU
process executes those routines unchanged without changing the caller's
active JAX scheme. The private pickle transport connects only this process
and its own child; it never accepts external input.
"""

import contextlib
import os
import pickle
import subprocess
import sys

import numpy as np


def cpu_reference(operation, values=None, spacing=1.0, epoch=None, **kwargs):
    ("Return original CPU values and metadata; this route is intentionally "
     "slow.")

    from pycbc import scheme
    from pycbc.fft import backend_cpu
    from pycbc.psd import estimate
    from pycbc.filter import resample

    settings = {
        "backend": backend_cpu.cpu_backend,
        "num_threads": getattr(scheme.mgr.state, "num_threads", 1),
        "welch_cache": estimate.USE_CACHING_FOR_WELCH_FFTS,
        "truncation_cache": estimate.USE_CACHING_FOR_INV_SPEC_TRUNC,
        "lfilter_cache": resample.USE_CACHING_FOR_LFILTER,
    }
    if settings["backend"] == "fftw":
        from pycbc.fft import fftw

        settings["measure_level"] = fftw.get_measure_level()
        settings["threads_backend"] = (fftw._fftw_threaded_lib
                                       if fftw._fftw_threaded_set else None)
    request = (
        operation,
        None if values is None else np.asarray(values),
        spacing,
        None if epoch is None else str(epoch),
        kwargs,
        settings,
    )
    result = subprocess.run(
        [sys.executable, "-m", "pycbc.reference_jax"],
        input=pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL),
        env=dict(os.environ, PYCBC_SCHEME="cpu"),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode:
        raise RuntimeError("CPU validation failed: " +
                           result.stderr.decode(errors="replace"))
    success, payload = pickle.loads(result.stdout)
    if not success:
        raise payload
    return payload


def _execute(request):
    from pycbc import scheme
    from pycbc.fft import backend_cpu
    from pycbc.psd import analytical, estimate
    from pycbc.types import FrequencySeries, TimeSeries

    operation, values, spacing, epoch, kwargs, settings = request
    backend_cpu.set_backend([settings["backend"]])
    if backend_cpu.cpu_backend != settings["backend"]:
        raise RuntimeError("The selected CPU FFT backend is unavailable")
    if settings["backend"] == "fftw":
        from pycbc.fft import fftw

        fftw.set_measure_level(settings["measure_level"])
        if settings["threads_backend"] is not None:
            fftw.set_threads_backend(settings["threads_backend"])
    estimate.USE_CACHING_FOR_WELCH_FFTS = settings["welch_cache"]
    estimate.USE_CACHING_FOR_INV_SPEC_TRUNC = settings["truncation_cache"]
    with scheme.CPUScheme(num_threads=settings["num_threads"]):
        if operation == "analytical_psd":
            result = analytical.from_string(**kwargs)
        elif operation == "welch":
            series = TimeSeries(values, delta_t=spacing, epoch=epoch)
            result = estimate.welch(series, **kwargs)
        elif operation in ("inverse_spectrum_truncation", "interpolate"):
            series = FrequencySeries(values, delta_f=spacing, epoch=epoch)
            result = getattr(estimate, operation)(series, **kwargs)
        elif operation in (
            "highpass",
            "lowpass",
            "lfilter",
            "fir_zero_filter",
            "resample",
        ):
            from pycbc.filter import resample

            resample.USE_CACHING_FOR_LFILTER = settings["lfilter_cache"]
            series = TimeSeries(values, delta_t=spacing, epoch=epoch)
            if operation in ("lfilter", "fir_zero_filter"):
                coefficients = kwargs.pop("coefficients")
                result = getattr(resample, operation)(
                    coefficients, series, **kwargs
                )
            else:
                name = (
                    "resample_to_delta_t"
                    if operation == "resample"
                    else operation
                )
                result = getattr(resample, name)(series, **kwargs)
            return (result.numpy(), result.delta_t,
                    None if result._epoch is None else str(result._epoch),
                    getattr(result, "corrupted_samples", None))
        elif operation in ("autocorrelation", "autocorrelation_length"):
            from pycbc.filter import autocorrelation

            is_series = kwargs.pop("is_series", True)
            series = (TimeSeries(values, delta_t=spacing, epoch=epoch)
                      if is_series else values)
            if operation == "autocorrelation_length":
                return autocorrelation.calculate_acl(series, **kwargs)
            result = autocorrelation.calculate_acf(
                series, delta_t=spacing, **kwargs
            )
            return (result.numpy(), result.delta_t,
                    None if result._epoch is None else str(result._epoch))
        else:
            raise ValueError("Unknown CPU validation operation")
        return (
            result.numpy(),
            result.delta_f,
            (None if result.epoch is None else str(result.epoch)),
        )


if __name__ == "__main__":
    request = pickle.loads(sys.stdin.buffer.read())
    try:
        with contextlib.redirect_stdout(sys.stderr):
            response = True, _execute(request)
    except Exception as error:
        response = False, error
    sys.stdout.buffer.write(
        pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
    )
