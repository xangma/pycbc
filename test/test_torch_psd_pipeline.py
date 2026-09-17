# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch psd pipeline regression tests."""

import lal
import numpy as np
import pytest
from pycbc import scheme
from pycbc.psd import (
    analytical as analytical_psd,
    analytical_space,
    estimate as psd_estimate,
    read as psd_read,
    variation,
    welch,
)
from pycbc.types import Array, FrequencySeries, TimeSeries
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


def _relative_l2(a, b):
    diff = a - b
    return np.linalg.norm(diff) / np.linalg.norm(b)


_DIRECT_SPACE_CURVES = (
    ("sensitivity_curve_lisa_SciRD", {}),
    (
        "sensitivity_curve_tianqin_analytical",
        {
            "len_arm": 1.8e8,
            "acc_noise_level": 1.2e-15,
            "oms_noise_level": 1.1e-12,
        },
    ),
    (
        "sensitivity_curve_taiji_analytical",
        {
            "len_arm": 3.1e9,
            "acc_noise_level": 2.8e-15,
            "oms_noise_level": 8.4e-12,
        },
    ),
    ("confusion_fit_lisa", {"duration": 2.0}),
    ("confusion_fit_tianqin", {"duration": 4.0}),
    ("confusion_fit_taiji", {"duration": 2.0}),
)


_COMBINED_SPACE_CURVES = (
    (
        "sensitivity_curve_lisa_confusion",
        {"base_model": "SciRD", "duration": 2.0},
    ),
    (
        "sensitivity_curve_tianqin_confusion",
        {
            "duration": 2.0,
            "len_arm": 1.8e8,
            "acc_noise_level": 1.2e-15,
            "oms_noise_level": 1.1e-12,
        },
    ),
    (
        "sensitivity_curve_taiji_confusion",
        {
            "duration": 2.0,
            "len_arm": 3.1e9,
            "acc_noise_level": 2.8e-15,
            "oms_noise_level": 8.4e-12,
        },
    ),
)


_CONFUSION_SPACE_PSDS = (
    (
        "analytical_psd_tianqin_confusion_noise",
        {"duration": 2.0, "len_arm": 1.8e8},
    ),
    (
        "analytical_psd_taiji_confusion_noise",
        {"duration": 2.0, "len_arm": 3.1e9},
    ),
    (
        "analytical_psd_tianqin_tdi_AE_confusion",
        {
            "duration": 2.0,
            "len_arm": 1.8e8,
            "acc_noise_level": 1.2e-15,
            "oms_noise_level": 1.1e-12,
        },
    ),
    (
        "analytical_psd_taiji_tdi_AE_confusion",
        {
            "duration": 2.0,
            "len_arm": 3.1e9,
            "acc_noise_level": 2.8e-15,
            "oms_noise_level": 8.4e-12,
        },
    ),
)


_SYNTHETIC_LISA_RESPONSE = (
    np.array((1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 2.0)),
    np.array((2.35, 2.28, 2.04, 1.21, 0.12, 0.0012712348970728724)),
)


_DIRECT_SPACE_RESPONSE_HELPERS = (
    ("averaged_lisa_fplus_sq_numerical", {}),
    ("averaged_tianqin_fplus_sq_numerical", {}),
    ("averaged_response_lisa_tdi", {"tdi": "2.0"}),
    ("averaged_response_tianqin_tdi", {"tdi": "2.0"}),
    ("averaged_response_taiji_tdi", {"tdi": "2.0"}),
)


_LISA_RESPONSE_MODELS = (
    ("sensitivity_curve_lisa_semi_analytical", {}),
    (
        "sensitivity_curve_lisa_confusion",
        {"base_model": "semi", "duration": 2.0},
    ),
    (
        "sh_transformed_psd_lisa_tdi_XYZ",
        {"base_model": "semi", "duration": 2.0, "tdi": "2.0"},
    ),
    (
        "sh_transformed_psd_lisa_tdi_XYZ",
        {"base_model": "SciRD", "duration": 2.0, "tdi": "1.5"},
    ),
    (
        "semi_analytical_psd_lisa_confusion_noise",
        {"duration": 2.0, "tdi": "2.0"},
    ),
    (
        "analytical_psd_lisa_tdi_AE_confusion",
        {"duration": 2.0, "tdi": "2.0"},
    ),
)


_ALIGO_TORCH_100_HZ_PINS = {
    "aLIGOQuantumNoSRMLowPower": 2.1931470245995445e-47,
    "aLIGOQuantumNoSRMHighPower": 3.1834841378930625e-48,
    "aLIGOQuantumZeroDetLowPower": 4.6779675124272452e-47,
    "aLIGOQuantumZeroDetHighPower": 8.4358809516525952e-48,
    "aLIGOQuantumNSNSOpt": 4.5169379541865925e-48,
    "aLIGOQuantumBHBH20Deg": 2.9004873157206200e-47,
    "aLIGOQuantumHighFrequency": 1.9787138993885476e-46,
    "aLIGONoSRMLowPower": 2.8728520233912104e-47,
    "aLIGONoSRMHighPower": 9.9805341258097216e-48,
    "aLIGOZeroDetLowPower": 5.3576725112189112e-47,
    "aLIGOZeroDetHighPower": 1.5232930939569254e-47,
    "aLIGONSNSOpt": 1.1313987942103253e-47,
    "aLIGOBHBH20Deg": 3.5801923145122860e-47,
    "aLIGOHighFrequency": 2.0466843992677141e-46,
    "aLIGOThermal": 6.7970499879166597e-48,
}


_ALIGO_TORCH_CONFIGURATIONS = (
    "NoSRMLowPower",
    "NoSRMHighPower",
    "ZeroDetLowPower",
    "ZeroDetHighPower",
    "NSNSOpt",
    "BHBH20Deg",
    "HighFrequency",
)


_ILIGO_TORCH_100_HZ_PINS = {
    "iLIGOSRD": 1.7323979166300358e-45,
    "iLIGOSeismic": 2.0903586303906188e-54,
    "iLIGOThermal": 2.3207710766653360e-46,
    "iLIGOShot": 2.7077800641050937e-46,
    "eLIGOShot": 9.025933547016977e-47,
    "iLIGOModel": 5.0285511616740164e-46,
    "eLIGOModel": 3.2233644522706198e-46,
}


_GROUND_FIT_TORCH_100_HZ_PINS = {
    "Virgo": 2.9764052287135322e-45,
    "GEO": 6.1707070707089880e-45,
    "GEOHF": 1.4534562248455390e-44,
    "TAMA": 8.1417187500000004e-42,
    "KAGRA": 9.1062488995400101e-48,
    "AdvVirgo": 2.1112328638054695e-47,
}


def test_analytical_space_psd_public_dispatch_and_devices(
    torch_device_ctx, monkeypatch
):
    ctx, device = torch_device_ctx
    parameters = dict(
        length=257,
        delta_f=1e-4,
        low_freq_cutoff=3.5e-4,
        len_arm=2.4e9,
        acc_noise_level=3.2e-15,
        oms_noise_level=13e-12,
        tdi="2.0",
    )
    expected = analytical_space.analytical_psd_lisa_tdi_AE(**parameters).numpy()

    with ctx:
        with monkeypatch.context() as patch:

            def _reject_host_path(*_args, **_kwargs):
                raise AssertionError("Torch analytical PSD used NumPy/SciPy")

            patch.setattr(analytical_space.np, "linspace", _reject_host_path)
            patch.setattr(analytical_space, "from_numpy_arrays", _reject_host_path)
            if device == "mps":
                with pytest.raises(TypeError, match="require float64"):
                    analytical_psd.from_string(
                        "analytical_psd_lisa_tdi_AE", **parameters
                    )
                return
            actual = analytical_psd.from_string(
                "analytical_psd_lisa_tdi_AE", **parameters
            )

    tensor = actual._data.tensor
    assert tensor.device.type == device
    assert tensor.dtype == torch.float64
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=0.0)


def test_lisa_response_psd_public_devices(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    parameters = dict(
        length=257,
        delta_f=5e-5,
        low_freq_cutoff=3.7e-4,
        duration=2.0,
        tdi="2.0",
    )
    monkeypatch.setattr(
        analytical_space,
        "_load_lisa_averaged_response_data",
        lambda: _SYNTHETIC_LISA_RESPONSE,
    )
    expected = analytical_space.analytical_psd_lisa_tdi_AE_confusion(
        **parameters
    ).numpy()

    with ctx:
        with monkeypatch.context() as patch:

            def _reject_host_path(*_args, **_kwargs):
                raise AssertionError("Torch LISA response model used NumPy/SciPy")

            patch.setattr(analytical_space.np, "linspace", _reject_host_path)
            patch.setattr(analytical_space, "from_numpy_arrays", _reject_host_path)
            patch.setattr(analytical_space, "interp1d", _reject_host_path)
            if device == "mps":
                with pytest.raises(TypeError, match="require float64"):
                    analytical_space.analytical_psd_lisa_tdi_AE_confusion(**parameters)
                return
            actual = analytical_space.analytical_psd_lisa_tdi_AE_confusion(**parameters)

    tensor = actual._data.tensor
    assert tensor.device.type == device
    assert tensor.dtype == torch.float64
    assert _relative_l2(actual.numpy(), expected) < 5e-12


def test_flat_unity_psd_stays_on_device(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    with ctx:
        with monkeypatch.context() as patch:

            def _reject_numpy_ones(*_args, **_kwargs):
                raise AssertionError("Torch flat_unity PSD used NumPy")

            patch.setattr(analytical_psd.numpy, "ones", _reject_numpy_ones)
            actual = analytical_psd.from_string("flat_unity", 17, 0.25, 0.75)

    tensor = actual._data.tensor
    assert tensor.device.type == device
    expected_dtype = torch.float32 if device == "mps" else torch.float64
    assert tensor.dtype == expected_dtype
    assert torch.equal(
        tensor,
        torch.tensor(
            [0.0, 0.0, 0.0] + [1.0] * 14,
            dtype=expected_dtype,
            device=device,
        ),
    )


def test_ground_detector_fit_family_torch_matches_lalsimulation(
    torch_device_ctx, monkeypatch
):
    ctx, device = torch_device_ctx
    parameters = dict(
        length=1031,
        delta_f=0.375,
        low_freq_cutoff=9.1,
    )

    if device == "mps":
        with ctx:
            for model_name in _GROUND_FIT_TORCH_100_HZ_PINS:
                with pytest.raises(
                    TypeError,
                    match="MPS backend only supports",
                ):
                    analytical_psd.from_string(
                        model_name,
                        **parameters,
                    )
        return

    expected = {
        model_name: analytical_psd.from_string(
            model_name,
            **parameters,
        ).numpy()
        for model_name in _GROUND_FIT_TORCH_100_HZ_PINS
    }

    def reject_lal_path(*_args, **_kwargs):
        raise AssertionError("Torch ground PSD fit called LALSimulation")

    with ctx:
        with monkeypatch.context() as patch:
            patch.setattr(
                analytical_psd.lal,
                "CreateREAL8FrequencySeries",
                reject_lal_path,
            )
            patch.setattr(
                analytical_psd.lalsimulation,
                "SimNoisePSD",
                reject_lal_path,
            )
            actual = {
                model_name: analytical_psd.from_string(
                    model_name,
                    **parameters,
                )
                for model_name in _GROUND_FIT_TORCH_100_HZ_PINS
            }
            wrapper_actual = analytical_psd.Virgo(**parameters)

    for model_name, result in actual.items():
        tensor = result._data.tensor
        assert tensor.device.type == device
        assert tensor.dtype == torch.float64
        assert result.epoch == lal.LIGOTimeGPS(0)
        assert result.delta_f == parameters["delta_f"]
        assert tensor[0] == 0
        assert tensor[-1] == 0
        np.testing.assert_allclose(
            tensor.detach().cpu().numpy(),
            expected[model_name],
            rtol=2e-14,
            atol=0.0,
        )
    torch.testing.assert_close(
        wrapper_actual._data.tensor,
        actual["Virgo"]._data.tensor,
    )


def test_ground_detector_fit_family_layout_cutoff_and_source_pins(torch_ctx):
    with torch_ctx:
        actual = {
            model_name: analytical_psd.from_string(
                model_name,
                length=1002,
                delta_f=1.0,
                low_freq_cutoff=10.9,
                ignored_lalsimulation_keyword=True,
            )._data.tensor
            for model_name in _GROUND_FIT_TORCH_100_HZ_PINS
        }
        one_bin = analytical_psd.Virgo(1, 1.0, 0.0)
        two_bins = analytical_psd.Virgo(2, 1.0, 0.0)
        negative_cutoff = analytical_psd.Virgo(8, 1.0, -1.0)

    for values in actual.values():
        assert torch.count_nonzero(values[:10]) == 0
        assert values[10] != 0
        assert values[-1] == 0
    torch.testing.assert_close(
        torch.stack([values[100] for values in actual.values()]),
        torch.tensor(
            list(_GROUND_FIT_TORCH_100_HZ_PINS.values()),
            dtype=torch.float64,
        ),
        rtol=2e-15,
        atol=0.0,
    )
    assert torch.count_nonzero(one_bin._data.tensor) == 0
    assert torch.count_nonzero(two_bins._data.tensor) == 0
    assert torch.count_nonzero(negative_cutoff._data.tensor) == 0


def test_psd_array_reader_interpolates_on_device(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    frequencies = np.array([0.5, 0.9, 1.8, 3.2, 5.0])
    noise = np.array([4.0, 2.0, 0.75, 1.5, 3.0])
    parameters = dict(
        length=25,
        delta_f=0.25,
        low_freq_cutoff=0.75,
    )
    expected = psd_read.from_numpy_arrays(
        frequencies,
        noise,
        **parameters,
    ).numpy()

    with ctx:
        with monkeypatch.context() as patch:

            def _reject_host_path(*_args, **_kwargs):
                raise AssertionError("Torch PSD reader used NumPy/SciPy")

            patch.setattr(
                psd_read.scipy.interpolate,
                "interp1d",
                _reject_host_path,
            )
            patch.setattr(psd_read.numpy, "zeros", _reject_host_path)
            patch.setattr(psd_read.numpy, "arange", _reject_host_path)
            if device == "mps":
                with pytest.raises(TypeError, match="requires float64"):
                    psd_read.from_numpy_arrays(
                        frequencies,
                        noise,
                        **parameters,
                    )
                return
            actual = psd_read.from_numpy_arrays(
                frequencies,
                noise,
                **parameters,
            )

    tensor = actual._data.tensor
    assert tensor.device.type == device
    assert tensor.dtype == torch.float64
    assert len(actual) == len(expected) == 21
    assert torch.count_nonzero(tensor[:3]) == 0
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=0)


def test_psd_text_reader_parses_host_and_interpolates_on_device(
    torch_ctx, monkeypatch, tmp_path
):
    frequencies = np.array([0.5, 0.9, 1.8, 3.2, 5.0])
    noise = np.array([4.0, 2.0, 0.75, 1.5, 3.0])
    filename = tmp_path / "asd.txt"
    np.savetxt(filename, np.column_stack((frequencies, np.sqrt(noise))))
    parameters = dict(
        filename=filename,
        length=17,
        delta_f=0.25,
        low_freq_cutoff=0.75,
    )
    expected = psd_read.from_txt(**parameters).numpy()

    with torch_ctx:
        with monkeypatch.context() as patch:

            def _reject_host_path(*_args, **_kwargs):
                raise AssertionError("Torch PSD reader used NumPy/SciPy")

            patch.setattr(
                psd_read.scipy.interpolate,
                "interp1d",
                _reject_host_path,
            )
            patch.setattr(psd_read.numpy, "zeros", _reject_host_path)
            patch.setattr(psd_read.numpy, "arange", _reject_host_path)
            actual = psd_read.from_txt(**parameters)

    tensor = actual._data.tensor
    assert tensor.device.type == "cpu"
    assert tensor.dtype == torch.float64
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=0)


def test_psd_interpolation_stays_on_device_and_clamps_band_edge(
    torch_device_ctx, monkeypatch
):
    ctx, device = torch_device_ctx
    values = np.array([4.0, 1.0, 9.0, 16.0])
    parameters = dict(delta_f=0.5, length=8)
    expected = psd_estimate.interpolate(
        FrequencySeries(values, delta_f=0.75),
        **parameters,
    ).numpy()

    dtype = torch.float32 if device == "mps" else torch.float64
    with ctx:
        source = FrequencySeries(
            TorchArrayData(torch.tensor(values, dtype=dtype, device=device)),
            delta_f=0.75,
            copy=False,
        )
        with monkeypatch.context() as patch:

            def _reject_host_path(*_args, **_kwargs):
                raise AssertionError("Torch PSD interpolation used the host")

            patch.setattr(psd_estimate.numpy, "interp", _reject_host_path)
            patch.setattr(torch.Tensor, "cpu", _reject_host_path)
            patch.setattr(torch.Tensor, "numpy", _reject_host_path)
            actual = psd_estimate.interpolate(source, **parameters)

    tensor = actual._data.tensor
    assert tensor.device.type == device
    assert tensor.dtype == dtype
    assert actual.delta_f == parameters["delta_f"]
    assert tensor[-3:].tolist() == [16.0, 16.0, 16.0]
    np.testing.assert_allclose(
        tensor.detach().tolist(),
        expected,
        rtol=2e-6 if dtype == torch.float32 else 1e-12,
        atol=0,
    )


@pytest.mark.parametrize(
    "model_name",
    (
        "analytical_csd_lisa_tdi_XY",
        "analytical_psd_taiji_tdi_T",
        "analytical_psd_taiji_confusion_noise",
        "analytical_psd_taiji_tdi_AE_confusion",
    ),
)
def test_analytical_space_psd_rejects_invalid_tdi(torch_ctx, model_name):
    with torch_ctx:
        with pytest.raises(ValueError, match="currently only for 1.5 or 2.0"):
            getattr(analytical_space, model_name)(17, 1e-4, 3.5e-4, tdi="3.0")


@pytest.mark.parametrize("avg_method", ["mean", "median", "median-mean"])
@pytest.mark.parametrize("num_segments", [7, 8])
def test_psd_welch_torch_matches_cpu(torch_ctx, avg_method, num_segments):
    # Short deterministic signal
    rng = np.random.default_rng(1234)
    seg_len = 256
    data = rng.standard_normal(num_segments * seg_len)
    ts_cpu = TimeSeries(data, delta_t=1 / 1024.0)
    psd_cpu = welch(ts_cpu, seg_len, seg_stride=seg_len, avg_method=avg_method)

    with torch_ctx:
        ts_t = TimeSeries(data, delta_t=1 / 1024.0)
        psd_t = welch(ts_t, seg_len, seg_stride=seg_len, avg_method=avg_method)

    assert isinstance(psd_t._data.tensor, torch.Tensor)
    assert psd_t._data.tensor.device.type == "cpu"
    assert psd_t.kind == psd_cpu.kind == "real"
    np.testing.assert_allclose(psd_t.numpy(), psd_cpu.numpy(), rtol=1e-12, atol=1e-14)


def test_psd_welch_builds_work_buffers_on_torch_device(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    rng = np.random.default_rng(8129)
    seg_len = 32
    seg_stride = 16
    values = rng.standard_normal(64).astype(np.float32)
    expected = welch(
        TimeSeries(values, delta_t=1 / 256.0),
        seg_len,
        seg_stride=seg_stride,
        avg_method="mean",
    )

    with ctx:
        timeseries = TimeSeries(values, delta_t=1 / 256.0)

        def reject_numpy_window(*args, **kwargs):
            raise AssertionError("Welch created its Hann window with NumPy")

        def reject_numpy_zeros(*args, **kwargs):
            raise AssertionError("Welch created its FFT buffer with NumPy")

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("Welch copied Torch data back to the host")

        with monkeypatch.context() as patch:
            patch.setattr(np, "hanning", reject_numpy_window)
            patch.setattr(psd_estimate.numpy, "zeros", reject_numpy_zeros)
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            actual = welch(
                timeseries,
                seg_len,
                seg_stride=seg_stride,
                avg_method="mean",
            )

        assert actual._data.tensor.device.type == device

    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("batch_segments", [1, 3, 4, 9])
def test_psd_welch_respects_temporary_budget(torch_ctx, monkeypatch, batch_segments):
    rng = np.random.default_rng(9137)
    seg_len = 32
    seg_stride = 16
    num_segments = 9
    values = rng.standard_normal(seg_len + (num_segments - 1) * seg_stride).astype(
        np.float32
    )
    expected = welch(
        TimeSeries(values, delta_t=1 / 256.0),
        seg_len,
        seg_stride=seg_stride,
        avg_method="median",
    )

    real_bytes = values.dtype.itemsize
    bytes_per_segment = seg_len * real_bytes + (seg_len // 2 + 1) * 2 * real_bytes
    fft_batch_sizes = []
    original_rfft = torch.fft.rfft

    def tracked_rfft(values, *args, **kwargs):
        fft_batch_sizes.append(values.shape[0])
        return original_rfft(values, *args, **kwargs)

    with torch_ctx:
        with monkeypatch.context() as patch:
            patch.setattr(
                psd_estimate,
                "_TORCH_WELCH_TEMPORARY_BYTES",
                batch_segments * bytes_per_segment,
            )
            patch.setattr(torch.fft, "rfft", tracked_rfft)
            actual = welch(
                TimeSeries(values, delta_t=1 / 256.0),
                seg_len,
                seg_stride=seg_stride,
                avg_method="median",
            )

    full_batches, remainder = divmod(num_segments, batch_segments)
    expected_batch_sizes = [batch_segments] * full_batches
    if remainder:
        expected_batch_sizes.append(remainder)
    assert fft_batch_sizes == expected_batch_sizes
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("psd_duration", [2, 8])
def test_psd_variation_bandpass_response_stays_on_device(
    torch_device_ctx, monkeypatch, psd_duration
):
    ctx, device = torch_device_ctx
    sample_rate = 64
    low_freq = 5
    high_freq = 20

    coefficients = variation.sig.firwin(
        4 * sample_rate,
        [low_freq, high_freq],
        pass_zero=False,
        window="hann",
        fs=sample_rate,
    )
    coefficients.resize(psd_duration * sample_rate)
    expected = np.abs(np.fft.rfft(coefficients)).astype(np.float32)

    with ctx:
        reference = FrequencySeries(
            np.ones(expected.size, dtype=np.float32), delta_f=0.25
        )

        def reject_host_path(*_args, **_kwargs):
            raise AssertionError("bandpass response used its host path")

        with monkeypatch.context() as patch:
            patch.setattr(variation.sig, "firwin", reject_host_path)
            patch.setattr(variation, "rfft", reject_host_path)
            patch.setattr(TorchArrayData, "numpy", reject_host_path)
            actual = variation._torch_bandpass_response(
                reference, sample_rate, low_freq, high_freq, psd_duration
            )

    assert isinstance(actual, torch.Tensor)
    assert actual.device.type == device
    assert actual.dtype == torch.float32
    np.testing.assert_allclose(
        actual.detach().cpu().numpy(), expected, rtol=5e-5, atol=5e-6
    )


def test_live_psd_variation_stays_on_device(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    sample_rate = 64
    rng = np.random.default_rng(9182)
    strain_values = rng.standard_normal(30 * sample_rate).astype(np.float32)
    psd_values = np.linspace(1.0, 2.0, 129, dtype=np.float32)

    cpu_strain = TimeSeries(strain_values, delta_t=1 / sample_rate, epoch=100)
    cpu_psd = FrequencySeries(psd_values, delta_f=0.25)
    expected_filter = variation.live_create_filter(
        cpu_psd, 4, sample_rate, low_freq=5, high_freq=20
    )
    expected_series = variation.live_calc_psd_variation(
        cpu_strain, expected_filter, 4, data_trim=0.5
    )
    trigger_times = np.array(
        [
            float(expected_series.start_time) - 1,
            float(expected_series.start_time) + 0.5,
            float(expected_series.end_time) - 1,
            float(expected_series.end_time) + 1,
        ]
    )
    expected_values = variation.live_find_var_value(
        {"end_time": trigger_times}, expected_series
    )

    with ctx:
        torch_strain = TimeSeries(strain_values, delta_t=1 / sample_rate, epoch=100)
        torch_psd = FrequencySeries(psd_values, delta_f=0.25)
        with monkeypatch.context() as patch:

            def _reject_host_transfer(_self):
                raise AssertionError("PSD variation copied a full PyCBC array to host")

            def _reject_scipy(*_args, **_kwargs):
                raise AssertionError("PSD variation used its SciPy data path")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(variation.sig, "firwin", _reject_scipy)
            patch.setattr(variation, "rfft", _reject_scipy)
            patch.setattr(variation.sig, "fftconvolve", _reject_scipy)
            patch.setattr(variation, "interp1d", _reject_scipy)
            actual_filter = variation.live_create_filter(
                torch_psd, 4, sample_rate, low_freq=5, high_freq=20
            )
            actual_series = variation.live_calc_psd_variation(
                torch_strain, actual_filter, 4, data_trim=0.5
            )
            actual_values = variation.live_find_var_value(
                {"end_time": trigger_times}, actual_series
            )

    assert isinstance(actual_filter, torch.Tensor)
    assert actual_filter.device.type == device
    assert actual_series._data.tensor.device.type == device
    assert isinstance(actual_values, Array)
    assert actual_values._data.tensor.device.type == device
    assert actual_series.dtype == np.dtype(np.float32)
    assert actual_series.start_time == expected_series.start_time
    np.testing.assert_allclose(
        actual_filter.detach().cpu().numpy(),
        expected_filter,
        rtol=5e-5,
        atol=5e-6,
    )
    np.testing.assert_allclose(
        actual_series._data.tensor.detach().cpu().numpy(),
        expected_series.numpy(),
        rtol=5e-5,
        atol=5e-6,
    )
    np.testing.assert_allclose(
        actual_values._data.tensor.detach().cpu().numpy(),
        expected_values,
        rtol=5e-5,
        atol=5e-6,
    )


def test_offline_psd_variation_stays_on_device(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    if device == "mps":
        pytest.skip("Welch requires complex PyCBC arrays, unsupported on MPS")
    sample_rate = 64
    rng = np.random.default_rng(1729)
    strain_values = rng.standard_normal(30 * sample_rate).astype(np.float32)
    parameters = (4, 0.25, 24, 2, 1, "median", 5, 20)

    cpu_strain = TimeSeries(strain_values, delta_t=1 / sample_rate, epoch=100)
    expected = variation.calc_filt_psd_variation(cpu_strain, *parameters)
    trigger_times = np.array(
        [
            float(expected.start_time) - 1,
            float(expected.start_time) + 0.5,
            float(expected.end_time) - 1,
            float(expected.end_time) + 1,
        ]
    )
    trigger_indices = (trigger_times - 100) * sample_rate
    expected_values = variation.find_trigger_value(
        expected, trigger_indices, 100, sample_rate
    )

    with ctx:
        torch_strain = TimeSeries(strain_values, delta_t=1 / sample_rate, epoch=100)
        with monkeypatch.context() as patch:

            def _reject_host_transfer(_self):
                raise AssertionError("PSD variation copied a full PyCBC array to host")

            def _reject_scipy(*_args, **_kwargs):
                raise AssertionError("PSD variation used its SciPy data path")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(variation.sig, "firwin", _reject_scipy)
            patch.setattr(variation, "rfft", _reject_scipy)
            patch.setattr(variation.sig, "fftconvolve", _reject_scipy)
            patch.setattr(variation, "interp1d", _reject_scipy)
            actual = variation.calc_filt_psd_variation(torch_strain, *parameters)
            actual_values = variation.find_trigger_value(
                actual, trigger_indices, 100, sample_rate
            )

    assert actual._data.tensor.device.type == device
    assert isinstance(actual_values, Array)
    assert actual_values._data.tensor.device.type == device
    assert actual.dtype == np.dtype(np.float32)
    assert actual.start_time == expected.start_time
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(),
        expected.numpy(),
        rtol=1e-4,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        actual_values._data.tensor.detach().cpu().numpy(),
        expected_values,
        rtol=1e-4,
        atol=1e-5,
    )


def test_psd_variation_interpolation_preserves_torch_autograd(
    torch_device_ctx, monkeypatch
):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        values = torch.tensor(
            [2.0, 4.0, 8.0, 16.0],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        positions = torch.tensor(
            [-1.0, 0.5, 2.25, 4.0],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        series = TimeSeries(TorchArrayData(values), delta_t=1.0, copy=False)

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("PSD interpolation copied data to the host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            actual = variation._torch_interpolate_positions(series, positions)

        result = actual._data.tensor
        assert result.device.type == device
        assert result.requires_grad
        result.sum().backward()

    tolerance = 2e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(
        result.detach().cpu(),
        torch.tensor([1.0, 3.0, 10.0, 1.0], dtype=dtype),
        rtol=tolerance,
        atol=tolerance,
    )
    torch.testing.assert_close(
        values.grad.detach().cpu(),
        torch.tensor([0.5, 0.5, 0.75, 0.25], dtype=dtype),
        rtol=tolerance,
        atol=tolerance,
    )
    torch.testing.assert_close(
        positions.grad.detach().cpu(),
        torch.tensor([0.0, 2.0, 8.0, 0.0], dtype=dtype),
        rtol=tolerance,
        atol=tolerance,
    )
