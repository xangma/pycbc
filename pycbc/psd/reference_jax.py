# Copyright (C) 2026 PyCBC contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Opt-in validation against original CPU PSD routines in a separate process.

The original routines use scheme-dispatched arrays and FFTs. A separate CPU
process executes those routines unchanged without changing the caller's
active JAX scheme. The private pickle transport connects only this process
and its own child; it never accepts external input.
"""

import contextlib
import pickle
import subprocess
import sys

import numpy as np


def cpu_psd_reference(operation, values=None, spacing=1.0, epoch=None, **kwargs):
    """Return original CPU values and metadata; this route is intentionally slow."""
    from pycbc import scheme
    from pycbc.fft import backend_cpu
    from pycbc.psd import estimate

    settings = {
        "backend": backend_cpu.cpu_backend,
        "num_threads": getattr(scheme.mgr.state, "num_threads", 1),
        "welch_cache": estimate.USE_CACHING_FOR_WELCH_FFTS,
        "truncation_cache": estimate.USE_CACHING_FOR_INV_SPEC_TRUNC,
    }
    if settings["backend"] == "fftw":
        from pycbc.fft import fftw

        settings["measure_level"] = fftw.get_measure_level()
        settings["threads_backend"] = (fftw._fftw_threaded_lib
                                       if fftw._fftw_threaded_set else None)
    request = (operation, None if values is None else np.asarray(values),
               spacing, None if epoch is None else str(epoch), kwargs, settings)
    result = subprocess.run(
        [sys.executable, "-m", "pycbc.psd.reference_jax"],
        input=pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode:
        raise RuntimeError("CPU PSD validation failed: " +
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
        else:
            raise ValueError("Unknown CPU PSD validation operation")
        return result.numpy(), result.delta_f, (None if result.epoch is None
                                               else str(result.epoch))


if __name__ == "__main__":
    request = pickle.loads(sys.stdin.buffer.read())
    try:
        with contextlib.redirect_stdout(sys.stderr):
            response = True, _execute(request)
    except Exception as error:
        response = False, error
    sys.stdout.buffer.write(pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL))
