"""Torch regression tests for inverse-spectrum cutoff validation."""

import numpy as np
import pytest

import pycbc
from pycbc import scheme
from pycbc.psd import inverse_spectrum_truncation
from pycbc.psd import estimate as estimate_module
from pycbc.strain.strain import StrainBuffer
from pycbc.types import FrequencySeries, TimeSeries
import pycbc.types.frequencyseries as frequencyseries_module


torch = pytest.importorskip("torch")

if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without Torch support", allow_module_level=True)


@pytest.fixture
def torch_ctx():
    ctx = scheme.TorchScheme("cpu")
    try:
        yield ctx
    finally:
        del ctx
        scheme.Scheme._single = None


def _reject_host_boundary(*_args, **_kwargs):
    raise AssertionError("cutoff validation crossed a Torch host boundary")


def _exception_signature(call):
    try:
        call()
    except Exception as exc:  # noqa: BLE001 - compare the public exception
        return type(exc), str(exc)
    raise AssertionError("call did not raise")


def _poison_coordinate_grid_and_item(patch):
    patch.setattr(
        frequencyseries_module, "_regular_grid", _reject_host_boundary
    )
    patch.setattr(torch.Tensor, "item", _reject_host_boundary)


def _make_uncached_strain_buffer(data, sample_rate):
    """Build the state needed to exercise both PSD truncations."""
    buffer = object.__new__(StrainBuffer)
    buffer.strain = TimeSeries(
        data, delta_t=1 / sample_rate, epoch=100
    )
    buffer.sample_rate = sample_rate
    buffer.reduced_pad = 0
    buffer.trim_padding = 0
    buffer.psd_inverse_length = 0.125
    buffer.low_frequency_cutoff = 10.0
    buffer.segments = {}
    buffer.psds = {}
    buffer.psd = FrequencySeries(
        np.linspace(1.0, 2.0, sample_rate // 2 + 1), delta_f=1.0
    )
    return buffer


def test_inverse_spectrum_cutoff_validation_has_no_host_boundary(
    torch_ctx, monkeypatch
):
    values = np.linspace(1.0, 4.0, 257)
    expected = inverse_spectrum_truncation(
        FrequencySeries(values, delta_f=1.0),
        64,
        low_frequency_cutoff=1.0,
        trunc_method="hann",
    )

    with torch_ctx:
        psd = FrequencySeries(values, delta_f=1.0)
        with monkeypatch.context() as patch:
            _poison_coordinate_grid_and_item(patch)
            actual = inverse_spectrum_truncation(
                psd,
                64,
                low_frequency_cutoff=1.0,
                trunc_method="hann",
            )

    assert actual._data.tensor.device.type == "cpu"
    assert actual.delta_f == expected.delta_f
    assert actual.epoch == expected.epoch
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(),
        expected.numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "cutoff",
    (
        pytest.param(-1.0, id="negative"),
        pytest.param(np.nextafter(256.0, np.inf), id="above-band"),
        pytest.param(256.0, id="upper-edge"),
        pytest.param(np.inf, id="positive-infinity"),
        pytest.param(-np.inf, id="negative-infinity"),
        pytest.param(np.nan, id="nan"),
    ),
)
def test_inverse_spectrum_cutoff_error_parity(
    torch_ctx, monkeypatch, cutoff
):
    values = np.linspace(1.0, 4.0, 257)
    expected = _exception_signature(
        lambda: inverse_spectrum_truncation(
            FrequencySeries(values, delta_f=1.0),
            64,
            low_frequency_cutoff=cutoff,
        )
    )

    with torch_ctx:
        psd = FrequencySeries(values, delta_f=1.0)
        with monkeypatch.context() as patch:
            _poison_coordinate_grid_and_item(patch)
            actual = _exception_signature(
                lambda: inverse_spectrum_truncation(
                    psd, 64, low_frequency_cutoff=cutoff
                )
            )

    assert actual == expected


@pytest.mark.parametrize(
    ("delta_f", "cutoff", "enters_body"),
    (
        pytest.param(
            0.1,
            float(np.float32(256) * np.float32(0.1)),
            True,
            id="rounded-grid-above-metadata-bound",
        ),
        pytest.param(
            0.7,
            256 * 0.7,
            False,
            id="rounded-grid-below-metadata-bound",
        ),
    ),
)
def test_inverse_spectrum_mps_uses_legacy_grid_boundary(
    monkeypatch, delta_f, cutoff, enters_body
):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")

    class ReachedInverseSpectrumBody(Exception):
        pass

    def reached_body(*_args, **_kwargs):
        raise ReachedInverseSpectrumBody

    with scheme.TorchScheme("mps"):
        psd = FrequencySeries(
            np.ones(257, dtype=np.float32), delta_f=delta_f
        )
        with monkeypatch.context() as patch:
            _poison_coordinate_grid_and_item(patch)
            patch.setattr(estimate_module, "zeros", reached_body)
            if enters_body:
                with pytest.raises(ReachedInverseSpectrumBody):
                    inverse_spectrum_truncation(
                        psd, 64, low_frequency_cutoff=cutoff
                    )
            else:
                with pytest.raises(
                    ValueError,
                    match="low_frequency_cutoff must be within the bandwidth",
                ):
                    inverse_spectrum_truncation(
                        psd, 64, low_frequency_cutoff=cutoff
                    )


def test_inverse_spectrum_grid_dtype_follows_active_mps_scheme(monkeypatch):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")

    psd = FrequencySeries(np.ones(257), delta_f=0.1)
    with scheme.TorchScheme("mps"):
        with monkeypatch.context() as patch:
            _poison_coordinate_grid_and_item(patch)
            actual = estimate_module._inverse_spectrum_max_frequency(psd)

    expected = float(np.float32(256) * np.float32(0.1))
    assert actual == expected


def test_inverse_spectrum_grid_dtype_follows_non_torch_scheme():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")

    context = scheme.TorchScheme("mps")
    with context:
        psd = FrequencySeries(
            np.ones(257, dtype=np.float32), delta_f=0.1
        )
    del context

    expected = psd.sample_frequencies[-1]
    actual = estimate_module._inverse_spectrum_max_frequency(psd)
    assert expected == actual == 25.6


@pytest.mark.parametrize(
    "delta_f",
    (
        pytest.param(1, id="python-int"),
        pytest.param(0.1, id="python-float"),
        pytest.param(np.float32(0.1), id="numpy-float32"),
        pytest.param(np.float64(0.1), id="numpy-float64"),
        pytest.param(np.longdouble("0.1"), id="numpy-longdouble"),
        pytest.param(np.inf, id="positive-infinity"),
    ),
)
def test_inverse_spectrum_torch_cpu_bound_uses_float64_grid(
    torch_ctx, monkeypatch, delta_f
):
    with torch_ctx:
        psd = FrequencySeries(np.ones(257), delta_f=delta_f)
        with monkeypatch.context() as patch:
            _poison_coordinate_grid_and_item(patch)
            actual = estimate_module._inverse_spectrum_max_frequency(psd)

    with np.errstate(invalid="ignore", over="ignore"):
        expected = float(np.float64(256) * np.float64(delta_f))
    if np.isnan(expected):
        assert np.isnan(actual)
    else:
        assert actual == expected


@pytest.mark.parametrize(
    "cutoff", (None, 1.0, np.inf, -np.inf, np.nan)
)
def test_inverse_spectrum_infinite_delta_f_error_parity(
    torch_ctx, monkeypatch, cutoff
):
    values = np.ones(9)
    with np.errstate(invalid="ignore"):
        expected = _exception_signature(
            lambda: inverse_spectrum_truncation(
                FrequencySeries(values, delta_f=np.inf),
                4,
                low_frequency_cutoff=cutoff,
            )
        )

    with torch_ctx:
        psd = FrequencySeries(values, delta_f=np.inf)
        with monkeypatch.context() as patch:
            _poison_coordinate_grid_and_item(patch)
            with np.errstate(invalid="ignore"):
                actual = _exception_signature(
                    lambda: inverse_spectrum_truncation(
                        psd, 4, low_frequency_cutoff=cutoff
                    )
                )

    assert actual == expected


@pytest.mark.parametrize(
    "cutoff",
    (
        pytest.param(None, id="none"),
        pytest.param(0.0, id="zero-edge"),
        pytest.param(np.nextafter(0.0, np.inf), id="above-band"),
        pytest.param(np.nan, id="nan"),
    ),
)
def test_inverse_spectrum_one_bin_error_parity(
    torch_ctx, monkeypatch, cutoff
):
    expected = _exception_signature(
        lambda: inverse_spectrum_truncation(
            FrequencySeries(np.ones(1), delta_f=1.0),
            1,
            low_frequency_cutoff=cutoff,
        )
    )

    with torch_ctx:
        psd = FrequencySeries(np.ones(1), delta_f=1.0)
        with monkeypatch.context() as patch:
            _poison_coordinate_grid_and_item(patch)
            actual = _exception_signature(
                lambda: inverse_spectrum_truncation(
                    psd, 1, low_frequency_cutoff=cutoff
                )
            )

    assert actual == expected


def test_timeseries_whiten_cutoff_validation_stays_on_device(
    torch_ctx, monkeypatch
):
    rng = np.random.default_rng(5678)
    data = rng.standard_normal(4096)
    expected = TimeSeries(data, delta_t=1 / 2048.0).whiten(
        2, 1, low_frequency_cutoff=20.0
    )

    with torch_ctx:
        series = TimeSeries(data, delta_t=1 / 2048.0)
        with monkeypatch.context() as patch:
            _poison_coordinate_grid_and_item(patch)
            actual = series.whiten(
                2, 1, low_frequency_cutoff=20.0
            )

    assert actual._data.tensor.device.type == "cpu"
    assert len(actual) == len(expected)
    difference = actual._data.tensor.detach().cpu().numpy() - expected.numpy()
    assert np.linalg.norm(difference) / np.linalg.norm(expected) < 0.1


def test_strain_buffer_cutoff_validation_stays_on_device(
    torch_ctx, monkeypatch
):
    sample_rate = 256
    rng = np.random.default_rng(2468)
    data = rng.standard_normal(sample_rate)
    expected = _make_uncached_strain_buffer(
        data, sample_rate
    ).overwhitened_data(1.0)

    with torch_ctx:
        buffer = _make_uncached_strain_buffer(data, sample_rate)
        with monkeypatch.context() as patch:
            _poison_coordinate_grid_and_item(patch)
            actual = buffer.overwhitened_data(1.0)

    assert actual._data.tensor.device.type == "cpu"
    assert actual.delta_f == expected.delta_f == 1.0
    assert actual.start_time == expected.start_time
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(),
        expected.numpy(),
        rtol=1e-11,
        atol=1e-11,
    )
