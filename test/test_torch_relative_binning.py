"""Focused tests for the Torch relative-binning kernels and dispatch."""

import numpy
import pytest


torch = pytest.importorskip("torch")

from pycbc.inference.models import relbin_torch  # noqa: E402


@pytest.fixture
def summary_data():
    generator = torch.Generator().manual_seed(8182)
    frequency_count = 129
    frequencies = torch.linspace(
        20.0, 500.0, frequency_count, dtype=torch.float64
    )
    edge_indices = torch.arange(17, dtype=torch.int64) * 8
    bins = torch.stack((edge_indices[:-1], edge_indices[1:]), dim=1)
    hp = torch.randn(
        frequency_count, dtype=torch.complex128, generator=generator
    )
    hc = torch.randn(
        frequency_count, dtype=torch.complex128, generator=generator
    )
    reference = torch.randn(
        frequency_count, dtype=torch.complex128, generator=generator
    )
    psd = torch.rand(
        frequency_count, dtype=torch.float64, generator=generator
    ) + 0.1
    a0, a1 = relbin_torch.summary_product(
        hp, reference, psd, frequencies, bins, 0.25
    )
    b0, b1 = relbin_torch.summary_product(
        reference, reference, psd, frequencies, bins, 0.25
    )
    return {
        "frequencies": frequencies[edge_indices],
        "hp": hp[edge_indices],
        "hc": hc[edge_indices],
        "reference": reference[edge_indices],
        "a0": a0,
        "a1": a1,
        "b0": b0.real,
        "b1": b1.real,
    }


def test_batched_likelihood_matches_scalar_evaluation(summary_data):
    sample_count = 12
    fplus = torch.linspace(0.1, 1.0, sample_count, dtype=torch.float64)
    fcross = torch.linspace(-0.5, 0.5, sample_count, dtype=torch.float64)
    delays = torch.linspace(-0.01, 0.01, sample_count, dtype=torch.float64)
    args = (
        summary_data["frequencies"],
        summary_data["hp"],
        summary_data["hc"],
        summary_data["reference"],
        summary_data["a0"],
        summary_data["a1"],
        summary_data["b0"],
        summary_data["b1"],
    )

    actual = relbin_torch.likelihood_parts(
        args[0], fplus, fcross, delays, *args[1:]
    )
    expected = tuple(
        torch.stack(values)
        for values in zip(*(
            relbin_torch.likelihood_parts(
                args[0], fplus[index], fcross[index], delays[index],
                *args[1:]
            )
            for index in range(sample_count)
        ))
    )

    assert actual[0].shape == (sample_count,)
    assert actual[1].shape == (sample_count,)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_public_detector_fallback_uses_tensors_and_keeps_gradients():
    calls = []

    class PublicDetector:
        @staticmethod
        def antenna_pattern(ra, dec, polarization, times):
            calls.append(("antenna", ra, dec, polarization, times))
            return ra.cos() + polarization, dec.sin() - polarization

        @staticmethod
        def time_delay_from_earth_center(ra, dec, times):
            calls.append(("delay", ra, dec, times))
            return 0.01 * (ra - dec)

    like = torch.ones(8, dtype=torch.complex128)
    ra = torch.tensor(1.1, dtype=torch.float64, requires_grad=True)
    dec = torch.tensor(-0.4, dtype=torch.float64, requires_grad=True)
    fplus, fcross, delay = relbin_torch.detector_response(
        PublicDetector(), ra, dec, 100.0, like
    )

    assert [call[0] for call in calls] == ["antenna", "delay"]
    assert all(value.device == like.device for value in (fplus, fcross, delay))
    (fplus + fcross + delay).backward()
    assert ra.grad is not None
    assert dec.grad is not None


def test_earth_rotation_response_matches_public_detector_geometry():
    from pycbc.detector import Detector

    reference_time = 1126259462.0
    detector = Detector("H1", reference_time=reference_time)
    times = reference_time + numpy.asarray([0.0, 0.25, 1.5, 8.0])
    ra = 1.37
    dec = -1.26
    like = torch.ones(8, dtype=torch.complex128)

    actual = relbin_torch.detector_response(
        detector, ra, dec, times, like
    )
    expected = tuple(numpy.asarray(values) for values in zip(*(
        (
            *detector.antenna_pattern(ra, dec, 0.0, time),
            detector.time_delay_from_earth_center(ra, dec, time),
        )
        for time in times
    )))

    for result, reference in zip(actual, expected):
        numpy.testing.assert_allclose(
            result.detach().numpy(), reference, rtol=2e-12, atol=2e-12
        )
    assert not torch.equal(actual[0][0], actual[0][-1])


def test_summary_product_keeps_batch_dimensions():
    frequency_count = 33
    batch_count = 5
    frequencies = torch.arange(frequency_count, dtype=torch.float64) * 0.25
    edges = torch.arange(0, frequency_count, 4, dtype=torch.int64)
    bins = torch.stack((edges[:-1], edges[1:]), dim=1)
    first = torch.randn(
        batch_count, frequency_count, dtype=torch.complex128,
        requires_grad=True,
    )
    second = torch.randn(frequency_count, dtype=torch.complex128)
    psd = torch.ones(frequency_count, dtype=torch.float64)

    a0, a1 = relbin_torch.summary_product(
        first, second, psd, frequencies, bins, 0.25
    )
    assert a0.shape == (batch_count, len(bins))
    assert a1.shape == (batch_count, len(bins))
    (a0.real.sum() + a1.imag.sum()).backward()
    assert first.grad is not None
