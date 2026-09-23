"""Live strain whitening uses the padded FFT grid before trimming."""

from types import SimpleNamespace

import numpy as np
import pytest

from pycbc import scheme
from pycbc.strain.strain import StrainBuffer
from pycbc.types import FrequencySeries, TimeSeries


@pytest.fixture(params=["cpu", "cuda:0"])
def torch_context(request):
    torch = pytest.importorskip("torch")
    if request.param.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return scheme.TorchScheme(request.param, num_threads=1)


def make_buffer(values, reduced_pad, truncate_psd):
    # Only the rolling buffer state used by overwhitened_data is needed.
    return SimpleNamespace(
        strain=TimeSeries(values, delta_t=1 / 1024, epoch=1234567890),
        sample_rate=1024, reduced_pad=reduced_pad, trim_padding=256,
        psd_inverse_length=0.25, low_frequency_cutoff=20,
        psd=FrequencySeries(
            (1 + (np.arange(513, dtype=np.float64) / 128) ** 2
             if truncate_psd else np.full(513, 2.0)),
            delta_f=1, dtype=values.dtype,
        ),
        psds={}, segments={},
    )


@pytest.mark.parametrize("dtype,truncate_psd", [
    (np.float32, False), (np.float64, True),
])
@pytest.mark.parametrize("reduced_pad", [0, 897])
def test_overwhitened_data_padded_grid(
    torch_context, monkeypatch, dtype, truncate_psd, reduced_pad
):
    if not truncate_psd:
        # Isolate the grid and FFT scaling from FP32 PSD estimation errors.
        # The FP64 case below also exercises real spectrum truncation.
        monkeypatch.setattr(
            "pycbc.psd.inverse_spectrum_truncation",
            lambda psd, *args, **kwargs: psd,
        )
    values = np.random.default_rng(872).normal(size=12288).astype(dtype)
    delta_f = 1 / 8
    with scheme.CPUScheme(1):
        reference_buffer = make_buffer(values, reduced_pad, truncate_psd)
        reference = StrainBuffer.overwhitened_data(reference_buffer, delta_f)
        expected = reference.numpy().copy()
        expected_epoch = reference.start_time
        expected_psdt = reference.psd.psdt.numpy().copy()

    with torch_context:
        buffer = make_buffer(values, reduced_pad, truncate_psd)
        result = StrainBuffer.overwhitened_data(buffer, delta_f)
        assert StrainBuffer.overwhitened_data(buffer, delta_f) is result
        assert result.backend == "torch"
        assert result.dtype == (np.complex64 if dtype == np.float32 else
                                np.complex128)
        assert len(result) == 4097
        assert result.delta_f == delta_f
        assert result.start_time == expected_epoch
        assert (result.start_time ==
                buffer.strain.end_time - 8 - reduced_pad / 1024)
        assert result.psd is buffer.psds[delta_f]
        assert result.psd.delta_f == delta_f
        assert len(result.psd) == len(result)
        padded_length = 8192 + 2 * reduced_pad
        assert len(result.psd.psdt) == padded_length // 2 + 1
        assert result.psd.psdt.delta_f == 1024 / padded_length
        tolerance = 3e-6 if dtype == np.float32 else 2e-12
        np.testing.assert_allclose(
            result.psd.psdt.numpy(), expected_psdt,
            rtol=tolerance, atol=tolerance,
        )
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=tolerance, atol=tolerance,
        )
        np.testing.assert_array_equal(buffer.strain.numpy(), values)
