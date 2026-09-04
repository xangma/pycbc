"""Focused tests for Gaussian-noise batched likelihood evaluation."""

import builtins
import types
from unittest import mock

import numpy
import pytest

from pycbc.inference.models.gaussian_noise import (
    BaseGaussianNoise,
    GaussianNoise,
    _detector_frame_batch_size,
)
from pycbc.types import FrequencySeries


def _minimal_numpy_batch_model(frequency_count):
    reference_epoch = 1.0
    hp = FrequencySeries(
        numpy.ones(frequency_count, dtype=numpy.complex128),
        delta_f=1.0,
        epoch=reference_epoch,
    )
    hc = FrequencySeries(
        numpy.zeros(frequency_count, dtype=numpy.complex128),
        delta_f=1.0,
        epoch=reference_epoch,
    )

    class RadiationFrameGenerator:
        @staticmethod
        def generate(**_params):
            return hp.copy(), hc.copy()

    detector = types.SimpleNamespace(antenna_times=[])

    def antenna_pattern(ra, _dec, polarization, arrival_time):
        detector.antenna_times.append(numpy.asarray(arrival_time).copy())
        return (
            numpy.cos(polarization) + 0.001 * arrival_time + 0.0 * ra,
            0.0 * arrival_time,
        )

    detector.antenna_pattern = antenna_pattern

    class Geometry:
        def __getitem__(self, _name):
            return detector

        @staticmethod
        def time_delay_from_earth_center(ra, _dec, _time):
            return numpy.asarray([numpy.zeros_like(ra) + 0.125])

        @staticmethod
        def antenna_pattern_and_time_delay(ra, dec, pol, time):
            delay = numpy.asarray([numpy.zeros_like(ra) + 0.125])
            fplus = numpy.asarray(
                [numpy.cos(pol) + 0.001 * time + 0.0 * dec]
            )
            return fplus, numpy.zeros_like(fplus), delay

        @staticmethod
        def to_dict(values):
            return {"H1": values[0]}

    generator = types.SimpleNamespace(
        location_args={"tc", "ra", "dec", "polarization"},
        rframe_generator=RadiationFrameGenerator(),
        _epoch=reference_epoch,
    )
    model = types.SimpleNamespace(
        static_params={},
        waveform_generator=generator,
        network_geometry=Geometry(),
        all_ifodata_same_rate_length=True,
        _data={"H1": None},
        _kmin={"H1": 0},
        _kmax={"H1": frequency_count},
        _weight={"H1": numpy.ones(frequency_count)},
        _whitened_data={
            "H1": numpy.ones(frequency_count, dtype=numpy.complex128)
        },
        recalibration={},
        gates=None,
    )
    model._batched_loglr = types.MethodType(
        GaussianNoise._batched_loglr, model
    )
    model._parse_batched_params = types.MethodType(
        BaseGaussianNoise._parse_batched_params, model
    )
    return model, detector


def test_numpy_batch_axis_is_explicit_when_it_matches_frequency_length():
    frequency_count = 4
    model, detector = _minimal_numpy_batch_model(frequency_count)
    ra = numpy.linspace(0.1, 0.4, frequency_count)
    dec = numpy.linspace(-0.2, 0.2, frequency_count)
    polarization = numpy.linspace(0.0, 0.3, frequency_count)
    tc = numpy.linspace(2.0, 2.3, frequency_count)

    actual = model._batched_loglr(
        ra=ra, dec=dec, polarization=polarization, tc=tc
    )

    arrival = tc + 0.125
    fplus = numpy.cos(polarization) + 0.001 * arrival
    frequencies = numpy.arange(frequency_count)
    projected = fplus[:, None] * numpy.exp(
        -2.0j * numpy.pi
        * (arrival - 1.0)[:, None]
        * frequencies
    )
    expected = numpy.sum(projected.conj(), axis=-1).real
    expected -= 0.5 * numpy.sum(numpy.abs(projected) ** 2, axis=-1)

    assert actual.shape == (frequency_count,)
    numpy.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    numpy.testing.assert_allclose(detector.antenna_times, [arrival])


def test_numpy_batch_path_does_not_import_torch():
    model, _ = _minimal_numpy_batch_model(3)
    original_import = builtins.__import__

    def reject_torch_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("NumPy batch path imported Torch")
        return original_import(name, *args, **kwargs)

    values = numpy.linspace(0.1, 0.3, 3)
    with mock.patch("builtins.__import__", side_effect=reject_torch_import):
        result = model._batched_loglr(
            ra=values,
            dec=values,
            polarization=values,
            tc=values + 2.0,
        )
    assert result.shape == (3,)


def test_batched_parameters_require_a_common_sample_length():
    model, _ = _minimal_numpy_batch_model(4)
    with pytest.raises(ValueError, match="common batch length"):
        model._batched_loglr(
            ra=numpy.asarray([0.2, 0.3]),
            dec=numpy.asarray([0.1, 0.2, 0.3]),
            polarization=0.2,
            tc=2.0,
        )


def test_detector_frame_batch_contract_accepts_torch_parameters():
    torch = pytest.importorskip("torch")
    generator = types.SimpleNamespace(
        location_args={"tc", "ra", "dec", "polarization"}
    )
    supplied = {
        "ra": torch.linspace(0.1, 0.4, 4),
        "dec": torch.tensor(0.2),
        "polarization": torch.linspace(0.0, 0.3, 4),
        "tc": torch.tensor(2.0),
    }
    assert _detector_frame_batch_size(generator, supplied, supplied) == 4

    invalid = dict(supplied, mass1=torch.ones(4))
    with pytest.raises(ValueError, match="radiation-frame waveform"):
        _detector_frame_batch_size(generator, invalid, invalid)
