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
        elif operation in ("gate_data", "detect_loud_glitches"):
            from pycbc.strain import strain

            series = TimeSeries(values, delta_t=spacing, epoch=epoch)
            result = getattr(strain, operation)(series, **kwargs)
            if operation == "detect_loud_glitches":
                return result
            return (result.numpy(), result.delta_t,
                    None if result._epoch is None else str(result._epoch))
        elif operation in ("newsnr", "effsnr"):
            from pycbc.events import ranking

            snr = values[()] if kwargs.pop("is_scalar", False) else values
            return getattr(ranking, operation)(snr, **kwargs)
        elif operation == "findchirp_cluster":
            from pycbc.events.eventmgr import findchirp_cluster_over_window

            return findchirp_cluster_over_window(
                kwargs.pop("times"), values, kwargs.pop("window_length"))
        elif operation == "segment_veto":
            from pycbc.events import veto

            function = (veto.indices_outside_times
                        if kwargs.pop("outside", False)
                        else veto.indices_within_times)
            return function(values, kwargs.pop("start"), kwargs.pop("end"))
        elif operation in (
                "time_coincidence", "cluster_over_time", "cluster_coincs",
                "cluster_coincs_multiifo"):
            from pycbc.events import coinc

            return getattr(coinc, operation)(values, **kwargs)
        elif operation == "quadrature_sum":
            from pycbc.events.stat import QuadratureSumStatistic

            statistic = object.__new__(QuadratureSumStatistic)
            return statistic.rank_stat_coinc(**kwargs)
        elif operation.startswith("event_"):
            from types import SimpleNamespace
            from pycbc.events.eventmgr import EventManager

            columns = [name for name in values.dtype.names
                       if name != "template_id"]
            manager = EventManager(
                SimpleNamespace(**kwargs.pop("opt", {})), columns,
                [values.dtype.fields[name][0] for name in columns],
                array_minsize=max(1, len(values)))
            manager._events[:len(values)] = values
            manager._events_size = len(values)
            manager.template_params = kwargs.pop("template_params", [])
            method = {
                "event_chisq_threshold": "chisq_threshold",
                "event_newsnr_threshold": "newsnr_threshold",
                "event_loudest": "keep_loudest_in_interval",
            }[operation]
            getattr(manager, method)(**kwargs.pop("method_options"))
            return manager.events.copy()
        elif operation == "live_selection":
            from types import SimpleNamespace
            from pycbc.filter.matchedfilter import LiveBatchMatchedFilter
            from pycbc.types import Array

            sigmasqs = kwargs.pop("sigmasqs")
            python_scalars = kwargs.pop("python_sigmasq")
            templates = []
            for row, (peak, sigmasq, python_scalar) in enumerate(zip(
                    values, sigmasqs, python_scalars, strict=True)):
                sigma = float(sigmasq) if python_scalar else sigmasq
                templates.append(SimpleNamespace(
                    id=row, delta_f=spacing,
                    params=np.zeros((), dtype=[]),
                    out=Array(np.atleast_1d(peak)),
                    sigmasq=lambda _psd, sigma=sigma: sigma))
            control = object.__new__(LiveBatchMatchedFilter)
            control.block_id = 0
            control.tgroups = [templates]
            control.chunk_tsamples = [1]
            control.mids = [0]
            spectrum = SimpleNamespace(psd=None)
            control.data = SimpleNamespace(
                overwhitened_data=lambda _df: spectrum,
                trim_padding=0, blocksize=1, sample_rate=1, start_time=0)
            control.corr = [SimpleNamespace(execute=lambda _data: None)]
            control.ifts = [SimpleNamespace(execute=lambda: None)]
            control.snr_threshold = kwargs.pop("snr_threshold")
            control.snr_abort_threshold = kwargs.pop("snr_abort_threshold", None)
            result, candidates = control._process_batch()
            return (result, np.array([candidate[3].id for candidate in candidates],
                                     dtype=np.int64),
                    np.array([candidate[1] for candidate in candidates]))
        elif operation == "power_chisq_bins":
            from pycbc.vetoes.chisq import power_chisq_bins

            psd = FrequencySeries(kwargs.pop("psd"), delta_f=spacing)
            template = FrequencySeries(values, delta_f=spacing, epoch=epoch)
            return power_chisq_bins(template, psd=psd, **kwargs)
        elif operation == "power_chisq_at_points":
            from pycbc.vetoes.chisq import (
                power_chisq_at_points_from_precomputed,
            )

            correlation = FrequencySeries(values, delta_f=spacing, epoch=epoch)
            return power_chisq_at_points_from_precomputed(
                correlation, **kwargs
            )
        elif operation == "sg_basis":
            from pycbc.waveform.sinegauss import fd_sine_gaussian

            result = fd_sine_gaussian(**kwargs)
        elif operation == "decompress":
            from pycbc.waveform.compress import fd_decompress

            output = (None if values is None else FrequencySeries(
                values, delta_f=spacing, epoch=epoch))
            result = fd_decompress(out=output, **kwargs)
        elif operation in ("td_taper", "fd_taper"):
            from pycbc.waveform import utils

            series = (TimeSeries(values, delta_t=spacing, epoch=epoch)
                      if operation == "td_taper" else
                      FrequencySeries(values, delta_f=spacing, epoch=epoch))
            result = getattr(utils, operation)(series, **kwargs)
            return (result.numpy(), spacing,
                    None if result._epoch is None else str(result._epoch))
        elif operation == "time_shift":
            from pycbc.waveform.utils import apply_fseries_time_shift

            if values.ndim > 1:
                shifts = np.broadcast_to(
                    kwargs.pop("dt"), values.shape[:-1]
                ).reshape(-1)
                result = np.empty_like(values)
                for row, output, shift in zip(
                    values.reshape(-1, values.shape[-1]),
                    result.reshape(-1, values.shape[-1]),
                    shifts,
                ):
                    series = FrequencySeries(row, delta_f=spacing, epoch=epoch)
                    output[:] = apply_fseries_time_shift(
                        series, float(shift), **kwargs
                    ).numpy()
                return result, spacing, epoch
            series = FrequencySeries(values, delta_f=spacing, epoch=epoch)
            result = apply_fseries_time_shift(series, **kwargs)
        elif operation == "sgchisq":
            from types import SimpleNamespace
            from pycbc.vetoes.sgchisq import SingleDetSGChisq

            stilde = FrequencySeries(values, delta_f=spacing, epoch=epoch)
            template = FrequencySeries(kwargs.pop("template"), delta_f=spacing,
                                       epoch=kwargs.pop("template_epoch"))
            template.f_lower = kwargs.pop("f_lower")
            template.params = SimpleNamespace(
                template_hash=kwargs.pop("template_hash")
            )
            psd = FrequencySeries(kwargs.pop("psd"), delta_f=spacing)
            calculator = SingleDetSGChisq.__new__(SingleDetSGChisq)
            calculator.do = True
            calculator.params = kwargs.pop("params")
            calculator.snr_threshold = kwargs.pop("snr_threshold")
            bins = kwargs.pop("bins")
            calculator.cached_chisq_bins = lambda template, psd: bins
            return calculator.values(stilde, template, psd, **kwargs)
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
