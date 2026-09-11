# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch filtering regression tests."""


import types
import numpy as np
import pytest
import scipy.interpolate
import scipy.signal
import scipy.special
import scipy.stats
import pycbc
from pycbc import events, scheme
from pycbc.events import eventmgr, threshold_torch
from pycbc.filter import matchedfilter, resample, zpk
from pycbc.strain.strain import detect_loud_glitches
from pycbc.types import Array, FrequencySeries, TimeSeries
from pycbc.types.array_torch import TorchArrayData
import pycbc.vetoes.autochisq as autochisq
import pycbc.vetoes.bank_chisq as bank_chisq
import pycbc.vetoes.chisq as chisq


torch = pytest.importorskip("torch")

if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without torch support", allow_module_level=True)


@pytest.fixture
def torch_ctx():
    ctx = scheme.TorchScheme("cpu")
    try:
        yield ctx
    finally:
        # Allow other tests to construct schemes after we exit
        del ctx
        scheme.Scheme._single = None


@pytest.fixture(params=("cpu", "cuda", "mps"))
def torch_device_ctx(request):
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device unavailable")

    ctx = scheme.TorchScheme(device)
    try:
        yield ctx, device
    finally:
        del ctx
        scheme.Scheme._single = None


def test_fir_filters_stay_on_device(torch_ctx):
    t = np.arange(0, 1, 1 / 1024.0)
    sig = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 200 * t)

    with torch_ctx:
        ts = TimeSeries(sig, delta_t=1 / 1024.0)
        lp = resample.lowpass_fir(ts, 50, 128, beta=5.0)
        hp = resample.highpass_fir(ts, 50, 128, beta=5.0)

    for out in (lp, hp):
        assert isinstance(out._data.tensor, torch.Tensor)
        assert out._data.tensor.device.type == "cpu"
        assert len(out) == len(ts)


def _relative_l2(a, b):
    diff = a - b
    return np.linalg.norm(diff) / np.linalg.norm(b)


def test_power_chisq_point_indices_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    rng = np.random.default_rng(9328)
    corr_values = (
        rng.normal(size=32) + 1j * rng.normal(size=32)
    ).astype(np.complex64)
    point_values = np.array([1, 4, 9], dtype=np.int64)
    bins = np.array([1, 4, 8, 14], dtype=np.int64)
    expected = np.asarray(chisq.shift_sum(
        FrequencySeries(corr_values, delta_f=0.1), point_values, bins
    ))

    with ctx:
        corr = FrequencySeries(corr_values, delta_f=0.1)
        points = Array(point_values)

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("power chi-squared indices copied to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            # The optimized CPU path exposes a zero-copy NumPy ABI view of
            # Torch-owned storage; this is not a device transfer.  Non-CPU
            # devices must continue to avoid NumPy entirely.
            if device != "cpu":
                patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "item", reject_host_transfer)
            actual = chisq.shift_sum(corr, points, bins)

    assert isinstance(actual, Array)
    assert actual._data.tensor.device.type == device
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(),
        expected,
        rtol=2e-5,
        atol=2e-5,
    )


def test_real_threshold_stays_on_torch_device(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    values = np.array([-2.0, 0.5, 1.5, 3.0, -4.0], dtype=np.float32)

    with ctx:
        series = TimeSeries(values, delta_t=1.0)
        assert series._data.tensor.device.type == device

        def _reject_host_transfer(_self):
            raise AssertionError("full real series transferred to NumPy")

        monkeypatch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
        locations, triggers = events.threshold_real(series, 1.0)

    np.testing.assert_array_equal(locations, np.array([2, 3]))
    np.testing.assert_array_equal(triggers, values[[2, 3]])


@pytest.mark.parametrize("empty", (False, True))
def test_complex_threshold_results_stay_on_torch_device(
        torch_device_ctx, monkeypatch, empty):
    ctx, device = torch_device_ctx
    if device == "mps":
        values = np.array(
            [0.0, 2.0, 3.0, -4.0, 7.0, 0.5], dtype=np.float32
        )
    else:
        values = np.array(
            [0.0, 2.0j, 3.0, -4.0j, 7.0, 0.5j],
            dtype=np.complex64,
        )
    threshold = 20.0 if empty else 1.5
    expected_locations = np.flatnonzero(np.abs(values) > threshold)
    expected_values = values[expected_locations]

    def _reject_host_transfer(*_args, **_kwargs):
        raise AssertionError("Torch threshold results transferred to host")

    with ctx:
        series = TimeSeries(values, delta_t=1.0)
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", _reject_host_transfer)
            locations, triggers = events.threshold(series, threshold)

    assert isinstance(locations, Array)
    assert isinstance(triggers, Array)
    assert locations._data.tensor.device.type == device
    assert triggers._data.tensor.device.type == device
    assert locations.dtype == np.dtype(np.int64)
    assert triggers.dtype == np.dtype(values.dtype)
    np.testing.assert_array_equal(
        locations._data.tensor.detach().cpu().numpy(), expected_locations
    )
    np.testing.assert_array_equal(
        triggers._data.tensor.detach().cpu().numpy(), expected_values
    )


@pytest.mark.parametrize("empty", (False, True))
@pytest.mark.parametrize("interface", ("function", "engine"))
def test_symmetric_threshold_cluster_results_stay_on_torch_device(
        torch_device_ctx, monkeypatch, empty, interface):
    ctx, device = torch_device_ctx
    if device == "mps":
        values = np.array(
            [0, 2, 3, -4, 7, 5, 2, -6, 1, 0, 8, 2],
            dtype=np.float32,
        )
    else:
        values = np.array(
            [0, 2j, 3, -4j, 7, 5j, 2, -6j, 1, 0, 8j, 2],
            dtype=np.complex64,
        )
    threshold = 20.0 if empty else 1.5
    expected_locations = (
        np.array([], dtype=np.int64)
        if empty else np.array([4, 10], dtype=np.int64)
    )
    expected_values = values[expected_locations]

    def _reject_host_transfer(*_args, **_kwargs):
        raise AssertionError("Torch clustered results transferred to host")

    with ctx:
        series = TimeSeries(values, delta_t=1.0)
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", _reject_host_transfer)
            if interface == "function":
                triggers, locations = events.threshold_and_cluster(
                    series, threshold, 3
                )
            else:
                engine = events.ThresholdCluster(series)
                triggers, locations = engine.threshold_and_cluster(
                    threshold, 3
                )

    assert isinstance(locations, Array)
    assert isinstance(triggers, Array)
    assert locations._data.tensor.device.type == device
    assert triggers._data.tensor.device.type == device
    assert locations.dtype == np.dtype(np.int64)
    assert triggers.dtype == np.dtype(values.dtype)
    np.testing.assert_array_equal(
        locations._data.tensor.detach().cpu().numpy(), expected_locations
    )
    np.testing.assert_array_equal(
        triggers._data.tensor.detach().cpu().numpy(), expected_values
    )


@pytest.mark.parametrize("real", (False, True))
def test_findchirp_threshold_copies_only_torch_survivors(
        torch_device_ctx, monkeypatch, real):
    ctx, device = torch_device_ctx
    if real:
        values = np.array(
            [-8.0, 2.0, 3.0, 4.0, -7.0, 5.0, 2.0, 6.0, 1.0],
            dtype=np.float32,
        )
        candidate_mask = values > 1.5
        operation = events.threshold_real_and_cluster_findchirp
    else:
        values = np.array(
            [0.0, 2.0j, 3.0, -4.0j, 7.0, 5.0j, 2.0, -6.0j, 1.0],
            dtype=np.complex64,
        )
        candidate_mask = np.abs(values) > 1.5
        operation = events.threshold_and_cluster_findchirp

    candidate_times = np.flatnonzero(candidate_mask)
    candidate_values = values[candidate_times]
    survivor_positions = events.findchirp_cluster_over_window(
        candidate_times, candidate_values, 3
    )
    expected_times = candidate_times[survivor_positions]
    expected_values = candidate_values[survivor_positions]
    assert len(expected_times) < len(candidate_times)

    def reject_host_array(*_args, **_kwargs):
        raise AssertionError("FindChirp copied all Torch candidates to host")

    with ctx:
        series = TimeSeries(values, delta_t=1.0)
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_array)
            patch.setattr(eventmgr, "threshold", reject_host_array)
            patch.setattr(eventmgr, "threshold_real", reject_host_array)
            patch.setattr(eventmgr, "cluster_reduce", reject_host_array)
            patch.setattr(threshold_torch, "threshold", reject_host_array)
            actual_times, actual_values = operation(series, 1.5, 3)

    np.testing.assert_array_equal(actual_times, expected_times)
    np.testing.assert_array_equal(actual_values, expected_values)
    assert series._data.tensor.device.type == device


def test_power_chisq_bins_stay_on_device(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    htilde_values = np.ones(513, dtype=np.complex64)
    psd_values = np.ones(513, dtype=np.float32)
    parameters = {
        "num_bins": 8,
        "low_frequency_cutoff": 20,
        "high_frequency_cutoff": 400,
    }
    expected = chisq.power_chisq_bins(
        FrequencySeries(htilde_values, delta_f=1),
        parameters["num_bins"],
        FrequencySeries(psd_values, delta_f=1),
        parameters["low_frequency_cutoff"],
        parameters["high_frequency_cutoff"],
    )

    with ctx:
        htilde = FrequencySeries(htilde_values, delta_f=1)
        psd = FrequencySeries(psd_values, delta_f=1)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("chi-squared bins copied data to host")

            def _reject_numpy_search(*_args, **_kwargs):
                raise AssertionError("chi-squared bins used NumPy searchsorted")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(chisq.numpy, "searchsorted", _reject_numpy_search)
            actual = chisq.power_chisq_bins(
                htilde,
                parameters["num_bins"],
                psd,
                parameters["low_frequency_cutoff"],
                parameters["high_frequency_cutoff"],
            )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("interface", ["tensor", "backend_protocol"])
def test_power_chisq_bins_accepts_public_backend_storage(
        torch_device_ctx, monkeypatch, interface):
    ctx, device = torch_device_ctx

    def reject_numpy(*_args, **_kwargs):
        raise AssertionError("chi-squared bin search left Torch")

    with ctx:
        tensor = torch.tensor(
            [0, 0, 1, 2, 2, 4, 8, 8], dtype=torch.float32, device=device,
        )
        values = tensor
        if interface == "backend_protocol":
            values = types.SimpleNamespace(
                backend="torch", backend_array=tensor,
            )
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "numpy", reject_numpy)
            patch.setattr(chisq.numpy, "searchsorted", reject_numpy)
            actual = chisq.power_chisq_bins_from_sigmasq_series(
                values, num_bins=4, kmin=2, kmax=7,
            )

    np.testing.assert_array_equal(actual, [2, 5, 6, 6, 7])
    assert actual.dtype == np.int64
    assert tensor.device.type == device


def test_power_chisq_dof_stays_on_device(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    values = np.array([2.0, 3.0, 5.0], dtype=np.float32)
    indices = np.array([1, 4, 7], dtype=np.int32)

    with ctx:
        corr = Array(np.arange(8, dtype=np.float32))
        snrv = Array(values)
        trigger_indices = indices
        veto = object.__new__(chisq.SingleDetPowerChisq)
        veto.do = True
        veto.snr_threshold = None
        veto.cached_chisq_bins = lambda *_args: [0, 2, 4]

        with monkeypatch.context() as patch:
            def _reject_numpy_repeat(*_args, **_kwargs):
                raise AssertionError(
                    "power chi-squared metadata used NumPy repeat"
                )

            patch.setattr(
                chisq,
                "power_chisq_at_points_from_precomputed",
                lambda *_args: Array(values),
            )
            patch.setattr(chisq.numpy, "repeat", _reject_numpy_repeat)
            actual, actual_dof = veto.values(
                corr, snrv, 1.0, object(), trigger_indices, object()
            )

    assert actual._data.tensor.device.type == device
    assert actual_dof._data.tensor.device.type == device
    assert actual_dof.dtype == np.dtype(np.int64)
    torch.testing.assert_close(
        actual_dof._data.tensor,
        torch.full((3,), 2, device=device, dtype=torch.int64),
    )


def test_skymax_chisq_dof_stays_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    indices = np.array([1, 4, 7], dtype=np.int32)

    with ctx:
        corr_plus = Array(np.arange(8, dtype=np.float32))
        corr_cross = Array(np.arange(8, dtype=np.float32))
        snrv = Array(np.zeros(3, dtype=np.float32))
        trigger_indices = indices
        u_vals = Array(np.ones(3, dtype=np.float32))
        veto = object.__new__(chisq.SingleDetSkyMaxPowerChisq)
        veto.do = True
        veto.snr_threshold = 10.0

        with monkeypatch.context() as patch:
            def _reject_numpy_repeat(*_args, **_kwargs):
                raise AssertionError(
                    "sky-max chi-squared metadata used NumPy repeat"
                )

            patch.setattr(chisq.numpy, "repeat", _reject_numpy_repeat)
            actual, actual_dof = veto.values(
                corr_plus,
                corr_cross,
                snrv,
                object(),
                trigger_indices,
                object(),
                object(),
                u_vals,
                0.0,
                1.0,
                1.0,
            )

    assert actual._data.tensor.device.type == device
    assert actual_dof._data.tensor.device.type == device
    torch.testing.assert_close(
        actual_dof._data.tensor,
        torch.full((3,), -100, device=device, dtype=torch.int64),
    )


def test_bank_chisq_dof_stays_on_device(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    values = np.array([0.5, 1.5, 2.5], dtype=np.float32)

    with ctx:
        statistic = Array(values)
        veto = object.__new__(bank_chisq.SingleDetBankVeto)
        veto.do = True
        veto.dof = 12
        veto.cache_overlaps = lambda *_args: []
        veto.cache_segment_snrs = lambda *_args: ([], [])

        with monkeypatch.context() as patch:
            def _reject_numpy_repeat(*_args, **_kwargs):
                raise AssertionError(
                    "bank chi-squared metadata used NumPy repeat"
                )

            patch.setattr(
                bank_chisq,
                "bank_chisq_from_filters",
                lambda *_args: statistic,
            )
            patch.setattr(chisq.numpy, "repeat", _reject_numpy_repeat)
            actual, actual_dof = veto.values(
                object(), object(), object(), object(), 1.0, object()
            )

    assert actual is statistic
    assert actual_dof._data.tensor.device.type == device
    torch.testing.assert_close(
        actual_dof._data.tensor,
        torch.full((3,), 12, device=device, dtype=torch.int64),
    )


@pytest.mark.parametrize(
    "settings",
    (
        {"oneside": None, "twophase": True, "maxvalued": False},
        {"oneside": "left", "twophase": False, "maxvalued": True},
        {
            "oneside": "right", "twophase": True, "maxvalued": True,
            "real_autocorr": True,
        },
    ),
)
def test_autochisq_stays_on_device(
        torch_device_ctx, monkeypatch, settings):
    ctx, device = torch_device_ctx
    if device == "mps":
        pytest.skip("Torch MPS PyCBC arrays do not support complex dtypes")

    rng = np.random.default_rng(481)
    snr_values = (
        rng.normal(size=64) + 1j * rng.normal(size=64)
    ).astype(np.complex64)
    corr_values = (
        rng.normal(size=64) + 1j * rng.normal(size=64)
    ).astype(np.complex64)
    autocorr_values = (
        0.15 * rng.normal(size=64) + 0.15j * rng.normal(size=64)
    ).astype(np.complex64)
    autocorr_values[0] = 1
    indices = np.array([1, 17, 62], dtype=np.int32)
    settings = settings.copy()
    real_autocorr = settings.pop("real_autocorr", False)
    if real_autocorr:
        autocorr_values = autocorr_values.real.copy()
    parameters = dict(stride=2, num_points=4, **settings)
    expected_dof, expected = autochisq.autochisq_from_precomputed(
        Array(snr_values), Array(corr_values), Array(autocorr_values),
        indices, **parameters
    )

    with ctx:
        snr = Array(snr_values)
        corr = Array(corr_values)
        autocorr = Array(autocorr_values)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("auto-chi-squared copied data to host")

            def _reject_numpy_output(*_args, **_kwargs):
                raise AssertionError("auto-chi-squared allocated NumPy output")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(autochisq.np, "zeros", _reject_numpy_output)
            actual_dof, actual = autochisq.autochisq_from_precomputed(
                snr, corr, autocorr, indices, **parameters
            )

    assert actual_dof == expected_dof
    assert isinstance(actual, Array)
    assert actual._data.tensor.device.type == device
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(), expected,
        rtol=2e-5, atol=2e-5
    )


def test_single_det_autochisq_keeps_indices_on_device(torch_ctx, monkeypatch):
    rng = np.random.default_rng(581)
    snr_values = (
        rng.normal(size=64) + 1j * rng.normal(size=64)
    ).astype(np.complex64)
    autocorr_values = (
        0.15 * rng.normal(size=64) + 0.15j * rng.normal(size=64)
    ).astype(np.complex64)
    autocorr_values[0] = 1
    index_values = np.array([1, 17, 62], dtype=np.int32)
    norm = 0.75
    parameters = dict(
        stride=2, num_points=4, oneside=None, twophase=True,
        maxvalued=False,
    )
    expected_dof, expected = autochisq.autochisq_from_precomputed(
        Array(snr_values * norm), Array(snr_values * norm),
        Array(autocorr_values), index_values, **parameters
    )
    template = object()
    psd = object()

    with torch_ctx:
        snr = Array(snr_values)
        indices = Array(index_values)
        veto = autochisq.SingleDetAutoChisq(
            stride=parameters["stride"],
            num_points=parameters["num_points"],
            twophase=parameters["twophase"],
        )
        veto._autocor = Array(autocorr_values)
        veto._autocor_id = (id(template), id(psd))
        original_np_array = autochisq.np.array

        def _reject_host_transfer(_self):
            raise AssertionError("auto-chi-squared copied data to host")

        def _reject_index_copy(value, *args, **kwargs):
            if isinstance(value, Array):
                raise AssertionError("auto-chi-squared copied indices to NumPy")
            return original_np_array(value, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(autochisq, "make_frequency_series", lambda x: x)
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(autochisq.np, "array", _reject_index_copy)
            actual, actual_dof = veto.values(
                snr, indices, template, psd, norm
            )

    assert actual_dof == expected_dof
    assert isinstance(actual, Array)
    assert actual._data.tensor.device.type == "cpu"
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(), expected,
        rtol=2e-5, atol=2e-5
    )


def test_detect_loud_glitches_thresholds_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    if device == "mps":
        pytest.skip("Glitch detection requires complex PyCBC arrays on MPS")

    sample_rate = 128
    rng = np.random.default_rng(1234)
    data = rng.normal(size=sample_rate * 16)
    data[sample_rate * 6] += 80
    data[sample_rate * 11] -= 100
    parameters = dict(
        psd_duration=2,
        psd_stride=1,
        low_freq_cutoff=10,
        threshold=6,
        cluster_window=0.5,
        corrupt_time=1,
    )
    expected = detect_loud_glitches(
        TimeSeries(data, delta_t=1 / sample_rate, epoch=1000),
        **parameters,
    )

    with ctx:
        torch_data = TimeSeries(
            data, delta_t=1 / sample_rate, epoch=1000
        )
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError(
                    "Glitch detection copied the full series to host"
                )

            def _reject_host_event_stage(*_args, **_kwargs):
                raise AssertionError(
                    "Glitch detection used a host threshold/cluster stage"
                )

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(pycbc.events, "threshold_only",
                          _reject_host_event_stage)
            patch.setattr(pycbc.events, "findchirp_cluster_over_window",
                          _reject_host_event_stage)
            actual = detect_loud_glitches(torch_data, **parameters)

    assert actual == expected


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("filter_order", (5, 8), ids=("odd", "even"))
@pytest.mark.parametrize("filter_name", ("highpass", "lowpass"))
def test_butterworth_filter_torch_matches_lal_without_host_transfer(
        torch_device_ctx, monkeypatch, dtype, filter_order, filter_name):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype == np.float64:
        pytest.skip("Torch MPS does not support float64")

    sample_rate = 2048
    delta_t = 1 / sample_rate
    samples = np.arange(4099) * delta_t
    data = (
        np.sin(2 * np.pi * 31 * samples)
        + 0.2 * np.cos(2 * np.pi * 317 * samples)
    ).astype(dtype)
    filter_func = getattr(resample, filter_name)
    expected = filter_func(
        TimeSeries(data, delta_t=delta_t, epoch=654),
        97,
        filter_order=filter_order,
        attenuation=0.17,
    )

    with ctx:
        input_series = TimeSeries(data, delta_t=delta_t, epoch=654)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError(
                    "Butterworth filtering copied Torch samples to host"
                )

            def _reject_lal(*_args, **_kwargs):
                raise AssertionError("Butterworth filtering called LAL")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(TimeSeries, "lal", _reject_lal)
            patch.setattr(zpk, "_TORCH_SOS_TARGET_BLOCK_SIZE", 128)
            actual = filter_func(
                input_series,
                97,
                filter_order=filter_order,
                attenuation=0.17,
            )

    assert actual._data.tensor.device.type == device
    assert actual.dtype == np.dtype(dtype)
    assert actual.delta_t == delta_t
    assert actual.start_time == expected.start_time
    assert len(actual) == len(expected)

    if device == "mps":
        rtol, atol = 3e-4, 3e-5
    elif dtype == np.float32:
        rtol, atol = 5e-6, 5e-7
    else:
        rtol, atol = 5e-10, 5e-11
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(),
        expected.numpy(),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_array_equal(
        input_series._data.tensor.detach().cpu().numpy(), data
    )


@pytest.mark.parametrize(
    "length,num_taps,block_size,coefficient_type",
    ((63, 7, 2**18, "numpy"), (4096, 129, 512, "array")),
    ids=("single-block", "overlap-add"),
)
@pytest.mark.parametrize(
    "dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
def test_lfilter_torch_matches_scipy_without_host_transfer(
        torch_device_ctx, monkeypatch, dtype, length, num_taps, block_size,
        coefficient_type):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype != np.float32:
        pytest.skip("Torch MPS only supports float32 PyCBC arrays")

    rng = np.random.default_rng(9182)
    single_precision = dtype in (np.float32, np.complex64)
    real_dtype = np.float32 if single_precision else np.float64
    coefficients = rng.normal(size=num_taps).astype(real_dtype)
    data = rng.normal(size=length)
    if np.issubdtype(dtype, np.complexfloating):
        data = data + 1j * rng.normal(size=length)
    data = data.astype(dtype)
    expected = scipy.signal.lfilter(coefficients, 1.0, data).astype(dtype)
    fft_input_dtypes = []
    transform_name = (
        "fft" if np.issubdtype(dtype, np.complexfloating) else "rfft"
    )
    original_transform = getattr(torch.fft, transform_name)

    def recording_transform(values, *args, **kwargs):
        fft_input_dtypes.append(values.dtype)
        return original_transform(values, *args, **kwargs)

    with ctx:
        filter_coefficients = (
            Array(coefficients) if coefficient_type == "array"
            else coefficients
        )
        input_series = TimeSeries(data, delta_t=1 / 2048, epoch=456)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("lfilter copied Torch data to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(torch.fft, transform_name, recording_transform)
            patch.setattr(
                resample, "_TORCH_LFILTER_TARGET_BLOCK_SIZE", block_size
            )
            actual = resample.lfilter(filter_coefficients, input_series)

    assert isinstance(actual._data.tensor, torch.Tensor)
    assert actual._data.tensor.device.type == device
    assert actual.dtype == np.dtype(dtype)
    assert actual.delta_t == input_series.delta_t
    assert actual.start_time == input_series.start_time
    assert fft_input_dtypes
    assert all(
        call_dtype == resample._torch_lfilter_work_dtype(
            input_series._data.tensor
        )
        for call_dtype in fft_input_dtypes
    )

    actual_data = actual._data.tensor.detach().cpu().numpy()
    input_data = input_series._data.tensor.detach().cpu().numpy()
    rtol, atol = ((5e-5, 5e-5) if single_precision
                  else (1e-11, 1e-11))
    np.testing.assert_allclose(actual_data, expected, rtol=rtol, atol=atol)
    np.testing.assert_array_equal(input_data, data)


@pytest.mark.parametrize(
    "dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
@pytest.mark.parametrize("length,num_taps", ((63, 7), (4096, 129)))
def test_fir_zero_filter_torch_matches_scipy_without_host_transfer(
        torch_device_ctx, monkeypatch, dtype, length, num_taps):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype != np.float32:
        pytest.skip("Torch MPS only supports float32 PyCBC arrays")

    rng = np.random.default_rng(12043)
    single_precision = dtype in (np.float32, np.complex64)
    real_dtype = np.float32 if single_precision else np.float64
    coefficients = scipy.signal.windows.hann(num_taps).astype(real_dtype)
    coefficients /= coefficients.sum()
    data = rng.normal(size=length)
    if np.issubdtype(dtype, np.complexfloating):
        data = data + 1j * rng.normal(size=length)
    data = data.astype(dtype)
    expected = scipy.signal.lfilter(coefficients, 1.0, data).astype(dtype)
    expected[:(num_taps // 2) * 2] = 0
    expected = np.roll(expected, -num_taps // 2)

    with ctx:
        input_series = TimeSeries(data, delta_t=1 / 2048, epoch=456)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError(
                    "fir_zero_filter copied Torch data to host"
                )

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(
                resample, "_TORCH_LFILTER_TARGET_BLOCK_SIZE", 512
            )
            actual = resample.fir_zero_filter(
                coefficients, input_series
            )

    assert actual._data.tensor.device.type == device
    assert actual.dtype == np.dtype(dtype)
    assert actual.delta_t == input_series.delta_t
    assert actual.start_time == input_series.start_time

    actual_data = actual._data.tensor.detach().cpu().numpy()
    input_data = input_series._data.tensor.detach().cpu().numpy()
    rtol, atol = ((5e-5, 5e-5) if single_precision
                  else (1e-11, 1e-11))
    np.testing.assert_allclose(actual_data, expected, rtol=rtol, atol=atol)
    np.testing.assert_array_equal(input_data, data)


@pytest.mark.parametrize(
    "dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
@pytest.mark.parametrize(
    "parameters",
    (
        ([20, 30], [1, 2, 100, 120], 5e4),
        ([40], [2, 80, 160], 1e3),
    ),
    ids=("two-second-order-sections", "first-and-second-order-sections"),
)
def test_filter_zpk_torch_matches_scipy_without_host_transfer(
        torch_device_ctx, monkeypatch, dtype, parameters):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype != np.float32:
        pytest.skip("Torch MPS only supports float32 PyCBC arrays")

    rng = np.random.default_rng(1845)
    data = rng.normal(size=4097)
    if np.issubdtype(dtype, np.complexfloating):
        data = data + 1j * rng.normal(size=data.size)
    data = data.astype(dtype)
    expected = zpk.filter_zpk(
        TimeSeries(data, delta_t=1 / 2048, epoch=789), *parameters
    )

    with ctx:
        input_series = TimeSeries(data, delta_t=1 / 2048, epoch=789)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("filter_zpk copied Torch samples to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(zpk, "_TORCH_SOS_TARGET_BLOCK_SIZE", 128)
            actual = zpk.filter_zpk(input_series, *parameters)

    assert actual._data.tensor.device.type == device
    assert actual.dtype == np.dtype(dtype)
    assert actual.delta_t == input_series.delta_t
    assert actual.start_time == input_series.start_time

    actual_data = actual._data.tensor.detach().cpu().numpy()
    input_data = input_series._data.tensor.detach().cpu().numpy()
    if device == "mps":
        rtol = 2e-3
        atol = np.max(np.abs(expected.numpy())) * 1e-3
    elif dtype in (np.float32, np.complex64):
        rtol, atol = 5e-5, 5e-6
    else:
        rtol, atol = 5e-10, 5e-11
    np.testing.assert_allclose(
        actual_data, expected.numpy(), rtol=rtol, atol=atol
    )
    np.testing.assert_array_equal(input_data, data)


def test_matched_filter_torch_vs_cpu(torch_ctx):
    # Simple sine wave template/data with flat PSD
    t = np.arange(0, 1, 1 / 1024.0)
    data = np.sin(2 * np.pi * 50 * t)
    ts_cpu = TimeSeries(data, delta_t=1 / 1024.0)
    psd_cpu = FrequencySeries(np.ones(len(ts_cpu) // 2 + 1), delta_f=ts_cpu.delta_f)
    snr_cpu = matchedfilter.matched_filter(ts_cpu, ts_cpu, psd=psd_cpu)

    with torch_ctx:
        ts_t = TimeSeries(data, delta_t=1 / 1024.0)
        psd_t = FrequencySeries(np.ones(len(ts_t) // 2 + 1), delta_f=ts_t.delta_f)
        snr_t = matchedfilter.matched_filter(ts_t, ts_t, psd=psd_t)

    assert isinstance(snr_t._data.tensor, torch.Tensor)
    assert snr_t._data.tensor.device.type == "cpu"
    assert _relative_l2(snr_t.numpy(), snr_cpu.numpy()) < 0.05
