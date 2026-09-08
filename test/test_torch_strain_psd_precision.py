"""Single-precision search conditioning must reproduce the original CPU."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import welch as scipy_welch

from pycbc import psd, scheme
from pycbc.strain.strain import StrainSegments
from pycbc.types import TimeSeries


@pytest.fixture(params=["torch-cpu", "torch-cuda"])
def context(request):
    torch = pytest.importorskip("torch")
    device = request.param.removeprefix("torch-")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return scheme.TorchScheme(device, num_threads=1)


def line_rich_data(dtype):
    time = np.arange(65536) / 512
    values = np.random.default_rng(9172).normal(0, 0.001, len(time))
    for frequency, amplitude in [(63.25, 32), (91.125, 11), (123.5, 7)]:
        values += amplitude * np.sin(2 * np.pi * frequency * time)
    # The numerical reference starts from these same rounded input samples.
    return values.astype(dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_strain_segments_retain_weak_frequencies(context, dtype):
    values = line_rich_data(dtype)
    with scheme.CPUScheme(1):
        reference_segments = StrainSegments(
            TimeSeries(values, delta_t=1 / 512, epoch=1234567890),
            segment_length=64, segment_start_pad=8, segment_end_pad=8,
            trigger_start=1234567890, trigger_end=1234568018,
            allow_zero_padding=True,
        )
        cpu_outputs = [s.numpy().copy() for s in reference_segments.fourier_segments()]
    with context:
        strain = TimeSeries(values, delta_t=1 / 512, epoch=1234567890)
        segments = StrainSegments(
            strain, segment_length=64, segment_start_pad=8, segment_end_pad=8,
            trigger_start=1234567890, trigger_end=1234568018,
            allow_zero_padding=True,
        )
        result = segments.fourier_segments()
        assert segments.fourier_segments() is result
        assert result[0].seg_slice.start < 0
        assert result[-1].seg_slice.stop > len(values)
        for output, segment, analyze, cpu_output in zip(
            result, segments.segment_slices, segments.analyze_slices, cpu_outputs
        ):
            chunk = np.zeros(32768, dtype=np.float64)
            start, stop = max(segment.start, 0), min(segment.stop, len(values))
            chunk[start - segment.start:stop - segment.start] = values[start:stop]
            reference = np.fft.rfft(chunk) / 512
            expected_dtype = np.complex64 if dtype == np.float32 else np.complex128
            assert output.dtype == expected_dtype
            assert output.delta_f == 1 / 64
            assert float(output.epoch) == 1234567890 + segment.start / 512
            assert output.analyze == analyze
            assert output.cumulative_index == segment.start + analyze.start
            assert output.seg_slice == segment
            if dtype == np.float32:
                np.testing.assert_array_equal(output.numpy(), cpu_output)
            else:
                np.testing.assert_allclose(
                    output.numpy(), reference, rtol=2e-11, atol=2e-11,
                )
        np.testing.assert_array_equal(strain.numpy(), values)


def psd_options(inverse_length):
    return SimpleNamespace(
        psd_model=None, psd_file=None, asd_file=None, psd_estimation="median",
        psd_low_frequency_cutoff=None, psd_segment_length=4,
        psd_segment_stride=2, psd_num_segments=63,
        psd_inverse_length=inverse_length, invpsd_trunc_method="hann",
        invpsd_trunc_which_spectrum="invasd", psd_output=None,
    )


def reference_psd(values, inverse_length):
    # SciPy's independent Welch estimator includes the even/odd median bias.
    frequency, estimate = scipy_welch(
        values.astype(np.float64), fs=512, window=np.hanning(2048),
        nperseg=2048, noverlap=1024, detrend=False, average="median",
    )
    interpolated = np.interp(np.arange(16385) / 64, frequency, estimate)
    if not inverse_length:
        return interpolated
    inverse_asd = np.zeros(16385, dtype=np.complex128)
    inverse_asd[30 * 64:-1] = interpolated[30 * 64:-1] ** -0.5
    impulse = np.fft.irfft(inverse_asd)
    half = inverse_length * 512 // 2
    window = np.hanning(inverse_length * 512)
    impulse[:half] *= window[-half:]
    impulse[-half:] *= window[:half]
    impulse[half:-half] = 0
    with np.errstate(divide="ignore"):
        return 1 / np.abs(np.fft.rfft(impulse)) ** 2


@pytest.mark.parametrize("dtype,precision", [
    (np.float32, None), (np.float32, "double"),
    (np.float64, None), (np.float64, "single"),
])
@pytest.mark.parametrize("inverse_length", [0, 2])
def test_estimated_psd_retains_weak_frequencies(
    context, dtype, precision, inverse_length
):
    values = line_rich_data(dtype)
    expected = reference_psd(values, inverse_length)
    with scheme.CPUScheme(1):
        cpu_output = psd.from_cli(
            psd_options(inverse_length), 16385, 1 / 64, 30,
            strain=TimeSeries(values, delta_t=1 / 512, epoch=1234567890),
            precision=precision,
        ).numpy().copy()
    output_dtype = (dtype if precision is None else
                    np.float64 if precision == "double" else np.float32)
    with context:
        strain = TimeSeries(values, delta_t=1 / 512, epoch=1234567890)
        actual = psd.from_cli(
            psd_options(inverse_length), 16385, 1 / 64, 30,
            strain=strain, precision=precision,
        )
        assert actual.dtype == output_dtype
        assert actual.delta_f == 1 / 64
        if dtype == np.float32:
            np.testing.assert_array_equal(actual.numpy(), cpu_output)
        else:
            np.testing.assert_allclose(
                actual.numpy()[30 * 64:-1], expected[30 * 64:-1],
                rtol=3e-7 if output_dtype == np.float32 else 2e-9, atol=0,
            )
        np.testing.assert_array_equal(strain.numpy(), values)
