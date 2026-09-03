"""Torch-specific tests for shared waveform utilities."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme  # noqa: E402
from pycbc.types import FrequencySeries, TimeSeries  # noqa: E402
from pycbc.waveform import utils  # noqa: E402


@pytest.fixture
def torch_cpu_ctx():
    ctx = scheme.TorchScheme("cpu")
    try:
        yield ctx
    finally:
        del ctx
        scheme.Scheme._single = None


@pytest.mark.parametrize("series_type", (TimeSeries, FrequencySeries))
def test_scheme_cast_series_preserves_metadata(torch_cpu_ctx, series_type):
    delta_name = "delta_t" if series_type is TimeSeries else "delta_f"
    source = series_type(
        np.arange(16, dtype=np.float64),
        epoch=123,
        **{delta_name: 0.25},
    )

    with torch_cpu_ctx:
        actual = utils.scheme_cast_series(source)

    assert actual._data.tensor.device.type == "cpu"
    np.testing.assert_array_equal(actual.numpy(), source.numpy())
    if series_type is TimeSeries:
        assert actual.start_time == source.start_time
    else:
        assert actual.epoch == source.epoch
    assert getattr(actual, delta_name) == getattr(source, delta_name)


@pytest.mark.parametrize(
    ("values", "expected_dtype"),
    (
        (np.arange(8, dtype=np.float64), torch.float32),
        (np.arange(8, dtype=np.complex128), torch.complex64),
    ),
)
@pytest.mark.parametrize("torch_backed", (False, True))
def test_scheme_cast_series_uses_mps_supported_dtype(
    values, expected_dtype, torch_backed
):
    if not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device unavailable")
    source = TimeSeries(values, delta_t=0.25)
    if torch_backed:
        cpu_ctx = scheme.TorchScheme("cpu")
        with cpu_ctx:
            source = TimeSeries(values, delta_t=0.25)
        del cpu_ctx
    ctx = scheme.TorchScheme("mps")
    try:
        with ctx:
            actual = utils.scheme_cast_series(source)
    finally:
        del ctx
        scheme.Scheme._single = None

    assert actual._data.tensor.device.type == "mps"
    assert actual._data.tensor.dtype == expected_dtype


def test_apply_fseries_time_shift_matches_cpu(torch_cpu_ctx):
    samples = np.sin(np.arange(64) / 7.0)
    source = TimeSeries(samples, delta_t=1 / 1024, epoch=10)
    reference = utils.apply_fd_time_shift(
        source.to_frequencyseries(), 10.125
    )

    with torch_cpu_ctx:
        torch_source = TimeSeries(samples, delta_t=1 / 1024, epoch=10)
        actual = utils.apply_fd_time_shift(
            torch_source.to_frequencyseries(), 10.125
        )

    torch.testing.assert_close(
        actual._data.tensor,
        torch.as_tensor(reference.numpy()),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("side", ("left", "right"))
def test_td_taper_matches_cpu(torch_cpu_ctx, side):
    samples = np.linspace(1.0, 2.0, 64)
    source = TimeSeries(samples, delta_t=0.25, epoch=-4)
    reference = utils.td_taper(source, -2, 2, side=side)

    with torch_cpu_ctx:
        torch_source = TimeSeries(samples, delta_t=0.25, epoch=-4)
        actual = utils.td_taper(torch_source, -2, 2, side=side)

    np.testing.assert_allclose(actual.numpy(), reference.numpy(), rtol=1e-14)


@pytest.mark.parametrize("side", ("left", "right"))
def test_fd_taper_matches_cpu(torch_cpu_ctx, side):
    samples = np.linspace(1.0, 2.0, 64).astype(np.complex128)
    source = FrequencySeries(samples, delta_f=0.25)
    reference = utils.fd_taper(source, 2, 6, side=side)

    with torch_cpu_ctx:
        torch_source = FrequencySeries(samples, delta_f=0.25)
        actual = utils.fd_taper(torch_source, 2, 6, side=side)

    np.testing.assert_allclose(actual.numpy(), reference.numpy(), rtol=1e-14)
