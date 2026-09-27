"""Benchmark warmup must compile the same precision context as timed trials."""

import numpy as np
import pytest

from tools import bench_jax_performance as benchmark


def test_filter_trials_do_not_retrace_after_warmup(monkeypatch):
    jax = pytest.importorskip("jax")
    from pycbc.filter import matchedfilter_jax

    complex_dtype = np.dtype("complex64")
    real_dtype = np.dtype("float32")
    waveform_calls = []
    filter_traces = []
    actual_filter = matchedfilter_jax.batch_matched_filter_bank

    def cheap_templates(m1, m2, flow, df, n_freq, precision):
        # A deterministic nonzero waveform avoids LAL cost without replacing
        # the actual JAX filtering, compilation, or synchronization paths.
        waveform_calls.append(bool(jax.config.jax_enable_x64))
        return np.ones((len(m1), n_freq), dtype=complex_dtype)

    def record_filter(templates, strain, psd, **kwargs):
        result = actual_filter(templates, strain, psd, **kwargs)
        filter_traces.append((templates.dtype, strain.dtype, psd.dtype,
                              result[0].dtype, result[1].dtype))
        return result

    monkeypatch.setattr(benchmark, "generate_lal_templates", cheap_templates)
    monkeypatch.setattr(matchedfilter_jax, "batch_matched_filter_bank", record_filter)
    original_x64 = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", False)
        result = benchmark.run_benchmark_arm(
            "jax_cpu_lal", n_time=128, batch_size=2, trials=3,
            flow=32.0, fhigh=512.0, cpu_dev=jax.devices("cpu")[0],
            precision="single",
        )
    finally:
        jax.config.update("jax_enable_x64", original_x64)

    assert waveform_calls == [True] * 4
    assert filter_traces == [(complex_dtype, complex_dtype, real_dtype,
                              complex_dtype, real_dtype)]
    assert result["jax_enable_x64"] is True
    assert result["jit_trace_counts"] == {"warmup": 1, "after_trials": 1}
    assert result["trials"] == 3
    assert result["sample_rate_hz"] == 2048.0
    assert result["complex_dtype"] == "complex64"
    assert result["real_dtype"] == "float32"


def test_filter_retracing_rejects_timed_trial(monkeypatch):
    jax = pytest.importorskip("jax")
    waveform_calls = []

    def changing_templates(m1, m2, flow, df, n_freq, precision):
        # An input shape change forces actual recompilation after warmup.
        count = len(m1) + bool(waveform_calls)
        waveform_calls.append(count)
        return np.ones((count, n_freq), dtype=np.complex64)

    monkeypatch.setattr(benchmark, "generate_lal_templates", changing_templates)
    original_x64 = jax.config.jax_enable_x64
    try:
        with pytest.raises(RuntimeError, match="retraced during a timed trial"):
            benchmark.run_benchmark_arm(
                "jax_cpu_lal", n_time=128, batch_size=2, trials=3,
                flow=32.0, fhigh=512.0, cpu_dev=jax.devices("cpu")[0],
                precision="single",
            )
    finally:
        jax.config.update("jax_enable_x64", original_x64)

    assert waveform_calls == [2, 3]
