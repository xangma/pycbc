# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch live-search regression tests."""


import types
import numpy as np
import pytest
from pycbc import scheme
from pycbc.events import ranking
from pycbc.filter import matchedfilter
from pycbc.types import Array, FrequencySeries
from pycbc.types.array_torch import TorchArrayData
import pycbc


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


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_newsnr_moves_host_summary_to_torch_device(
        torch_device_ctx, monkeypatch, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype == np.float64:
        pytest.skip("Torch MPS does not support float64 PyCBC arrays")

    snr = np.array([8.0, 12.0, 20.0, 30.0], dtype=dtype)
    reduced_chisq = np.array([0.5, 1.0, 2.0, 6.0], dtype=dtype)
    expected = ranking.newsnr(snr, reduced_chisq, q=6.0, n=2.0)

    with ctx:
        torch_chisq = Array(reduced_chisq)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("newSNR copied Torch data to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = ranking.newsnr(snr, torch_chisq, q=6.0, n=2.0)

    assert isinstance(actual, Array)
    assert actual._data.tensor.device.type == device
    expected_dtype = np.float32 if device == "mps" else np.float64
    assert actual.dtype == np.dtype(expected_dtype)
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(), expected,
        rtol=2e-6 if dtype == np.float32 else 1e-12,
        atol=2e-7 if dtype == np.float32 else 0.0,
    )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_effsnr_stays_on_torch_device(
        torch_device_ctx, monkeypatch, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype == np.float64:
        pytest.skip("Torch MPS does not support float64 PyCBC arrays")

    snr = np.array([8.0, 7.5, 6.0, 5.5, 4.0], dtype=dtype)
    rchisq = np.array([0.5, 1.0, 1.5, 4.0, np.nan], dtype=dtype)
    expected = ranking.effsnr(snr, rchisq, fac=200.0)

    with ctx:
        torch_snr = Array(snr)
        torch_rchisq = Array(rchisq)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("effective SNR copied Torch data to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = ranking.effsnr(
                torch_snr, torch_rchisq, fac=200.0
            )

    assert isinstance(actual, Array)
    assert actual._data.tensor.device.type == device
    expected_dtype = np.float32 if device == "mps" else np.float64
    assert actual.dtype == np.dtype(expected_dtype)
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(), expected,
        rtol=2e-6 if dtype == np.float32 else 1e-12,
        atol=2e-7 if dtype == np.float32 else 0.0,
        equal_nan=True,
    )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_newsnr_stays_on_torch_device(
        torch_device_ctx, monkeypatch, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype == np.float64:
        pytest.skip("Torch MPS does not support float64 PyCBC arrays")

    snr = np.array([8.0, 7.5, 6.0, 5.5, 4.0], dtype=dtype)
    rchisq = np.array([-0.5, 1.0, 1.5, 4.0, np.nan], dtype=dtype)
    expected = ranking.newsnr(snr, rchisq, q=8.0, n=3.0)

    with ctx:
        torch_snr = Array(snr)
        torch_rchisq = Array(rchisq)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("newSNR copied Torch data to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = ranking.newsnr(
                torch_snr, torch_rchisq, q=8.0, n=3.0
            )

    assert isinstance(actual, Array)
    assert actual._data.tensor.device.type == device
    expected_dtype = np.float32 if device == "mps" else np.float64
    assert actual.dtype == np.dtype(expected_dtype)
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(), expected,
        rtol=2e-6 if dtype == np.float32 else 1e-12,
        atol=2e-7 if dtype == np.float32 else 0.0,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "function_name, include_psd, kwargs",
    (
        ("newsnr_sgveto", False, {"q": 8.0, "n": 3.0}),
        (
            "newsnr_sgveto_psdvar",
            True,
            {"min_expected_psdvar": 0.65, "q": 8.0, "n": 3.0},
        ),
        (
            "newsnr_sgveto_psdvar_threshold",
            True,
            {
                "min_expected_psdvar": 0.65,
                "brchisq_threshold": 10.0,
                "psd_var_val_threshold": 10.0,
                "q": 8.0,
                "n": 3.0,
            },
        ),
        (
            "newsnr_sgveto_psdvar_scaled",
            True,
            {
                "scaling": 0.33,
                "min_expected_psdvar": 0.65,
                "q": 8.0,
                "n": 3.0,
            },
        ),
        (
            "newsnr_sgveto_psdvar_scaled_threshold",
            True,
            {
                "threshold": 2.0,
                "scaling": 0.33,
                "min_expected_psdvar": 0.65,
                "q": 8.0,
                "n": 3.0,
            },
        ),
    ),
)
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_newsnr_veto_family_stays_on_torch_device(
        torch_device_ctx, monkeypatch, function_name, include_psd,
        kwargs, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype == np.float64:
        pytest.skip("Torch MPS does not support float64 PyCBC arrays")

    snr = np.array([8.0, 7.5, 6.0, 5.5, 4.0, 9.0], dtype=dtype)
    brchisq = np.array([-0.5, 1.0, 1.5, 4.0, 12.0, np.nan], dtype=dtype)
    sgchisq = np.array([2.0, 4.0, 8.0, 16.0, np.nan, 5.0], dtype=dtype)
    psd_var = np.array([0.5, 0.65, 1.0, 4.0, 12.0, np.nan], dtype=dtype)
    function = getattr(ranking, function_name)
    args = (snr, brchisq, sgchisq)
    if include_psd:
        args += (psd_var,)
    expected = function(*args, **kwargs)

    with ctx:
        torch_args = tuple(Array(value) for value in args)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError(
                    f"{function_name} copied Torch data to host"
                )

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = function(*torch_args, **kwargs)

    assert isinstance(actual, Array)
    assert actual._data.tensor.device.type == device
    expected_dtype = np.float32 if device == "mps" else np.float64
    assert actual.dtype == np.dtype(expected_dtype)
    tolerance = 2e-6 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(), expected,
        rtol=tolerance, atol=0.0, equal_nan=True,
    )


def test_live_batch_cuda_graph_flag_from_environment(monkeypatch):
    template = FrequencySeries(np.ones(33, dtype=np.complex64), delta_f=0.25)
    template.id = 1
    template.params = np.array(
        [(10.0,)], dtype=[("mass1", np.float32)]
    )[0]

    monkeypatch.delenv("PYCBC_ENABLE_CUDA_GRAPHS", raising=False)
    batch = matchedfilter.LiveBatchMatchedFilter(
        [template],
        snr_threshold=5.0,
        chisq_bins=None,
        sg_chisq=None,
        maxelements=64,
    )
    assert batch.enable_cuda_graphs is False
    assert batch._cuda_graphs == {}

    monkeypatch.setenv("PYCBC_ENABLE_CUDA_GRAPHS", "1")
    batch = matchedfilter.LiveBatchMatchedFilter(
        [template],
        snr_threshold=5.0,
        chisq_bins=None,
        sg_chisq=None,
        maxelements=64,
    )
    assert batch.enable_cuda_graphs is True


@pytest.mark.parametrize("enable_cuda_graphs", (True, False))
def test_live_batch_matched_filter_cuda_graphs_pipeline(torch_device_ctx, enable_cuda_graphs):
    ctx, device = torch_device_ctx
    size = 64
    fsize = size // 2 + 1
    rng = np.random.default_rng(999)
    template_values = (
        rng.normal(size=(2, fsize)) + 1j * rng.normal(size=(2, fsize))
    ).astype(np.complex64)
    data_values = (
        rng.normal(size=fsize) + 1j * rng.normal(size=fsize)
    ).astype(np.complex64)

    with ctx:
        templates = []
        for index, values in enumerate(template_values):
            template = FrequencySeries(values, delta_f=1.0 / size)
            template.id = 100 + index
            template.params = np.array(
                [(15.0 + index,)], dtype=[("mass1", np.float32)]
            )[0]
            template.sigmasq = lambda _psd: 1.0
            templates.append(template)

        batch = matchedfilter.LiveBatchMatchedFilter(
            templates,
            snr_threshold=0.0,
            chisq_bins=None,
            sg_chisq=types.SimpleNamespace(values=lambda *args: None),
            maxelements=len(templates) * size,
            enable_cuda_graphs=enable_cuda_graphs,
        )
        batch.power_chisq.values = lambda *args: (
            np.array([0.0], dtype=np.float32),
            np.array([1], dtype=np.uint32),
        )
        assert batch.enable_cuda_graphs is enable_cuda_graphs

        stilde = FrequencySeries(data_values, delta_f=1.0 / size)
        stilde.psd = FrequencySeries(np.ones(fsize, dtype=np.float32), delta_f=1.0 / size)

        batch.set_data(types.SimpleNamespace(
            overwhitened_data=lambda _delta_f: stilde,
            trim_padding=0,
            blocksize=size,
            sample_rate=1,
            start_time=100.0,
        ))

        res1 = batch.process_data(types.SimpleNamespace(
            overwhitened_data=lambda _delta_f: stilde,
            trim_padding=0,
            blocksize=size,
            sample_rate=1,
            start_time=100.0,
        ))
        assert "snr" in res1
        assert len(res1["snr"]) == 2

        if device == "cuda" and enable_cuda_graphs and torch.cuda.is_available():
            assert 0 in batch._cuda_graphs
            entry = batch._cuda_graphs[0]
            replays = entry.get("replays", getattr(entry["graph"], "replay_count", 0))
            assert replays == 0

            # Second run replays graph
            res2 = batch.process_data(types.SimpleNamespace(
                overwhitened_data=lambda _delta_f: stilde,
                trim_padding=0,
                blocksize=size,
                sample_rate=1,
                start_time=100.0,
            ))
            replays = entry.get("replays", getattr(entry["graph"], "replay_count", 0))
            assert replays == 1
            np.testing.assert_allclose(res1["snr"], res2["snr"])
