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


@pytest.fixture(params=["numpy", "cpu", "cuda"])
def real_model(request):
    """Use real TaylorF2 waveforms, detector geometry, and data containers."""
    from pycbc import scheme
    from pycbc.inference.models.marginalized_gaussian_noise import (
        MarginalizedPhaseGaussianNoise,
    )

    backend = request.param
    if backend == "numpy":
        context = scheme.CPUScheme()
    else:
        torch = pytest.importorskip("torch")
        if backend == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is unavailable")
        context = scheme.TorchScheme(backend)
    try:
        with context:
            data = {
                det: FrequencySeries(
                    numpy.full(129, 1e-23 + 1e-23j),
                    delta_f=2.0, epoch=1126259460.0,
                )
                for det in ("H1", "L1")
            }
            psds = {
                det: FrequencySeries(numpy.full(129, 1e-46), delta_f=2.0)
                for det in data
            }
            static = dict(
                approximant="TaylorF2", mass1=30.0, mass2=20.0,
                distance=500.0, inclination=0.4, f_lower=20.0,
                coa_phase=0.2,
            )
            params = {
                name: numpy.asarray(values, dtype=numpy.float64)
                for name, values in dict(
                    tc=[1126259462.0, 1126259462.001],
                    ra=[1.1, 1.2], dec=[-0.3, -0.2],
                    polarization=[0.2, 0.4],
                ).items()
            }
            if backend != "numpy":
                params = {
                    name: torch.tensor(
                        value, dtype=torch.float64, device=backend,
                        requires_grad=(name == "ra"),
                    )
                    for name, value in params.items()
                }
            models = [
                cls(tuple(params), data, {"H1": 20.0, "L1": 24.0},
                    psds=psds, static_params=static)
                for cls in (GaussianNoise, MarginalizedPhaseGaussianNoise)
            ]
            yield models, params, backend
    finally:
        del context
        scheme.Scheme._single = None


def test_real_batch_matches_scalar_likelihood_and_gradients(real_model):
    models, params, backend = real_model
    for model in models:
        expected = []
        for index in range(2):
            model.update(**{name: value[index]
                            for name, value in params.items()})
            expected.append(model.loglr)
        old_params = model._current_params
        old_stats = model._current_stats
        actual = model.batched_loglr(**params)
        assert model._current_params is old_params
        assert model._current_stats is old_stats
        if backend == "numpy":
            numpy.testing.assert_allclose(actual, expected, atol=1e-9)
            numpy.testing.assert_allclose(
                model.batched_loglikelihood(**params), actual + model.lognl
            )
            continue

        import torch

        assert actual.device == params["ra"].device
        torch.testing.assert_close(actual, torch.stack(expected), atol=1e-9,
                                   rtol=1e-10)
        torch.testing.assert_close(
            model.batched_loglikelihood(**params), actual + model.lognl
        )
        gradient = torch.autograd.grad(actual.sum(), params["ra"])[0]
        scalar_gradient = torch.autograd.grad(
            torch.stack(expected).sum(), params["ra"]
        )[0]
        torch.testing.assert_close(gradient, scalar_gradient)
        # An independent numerical derivative detects finite but wrong
        # gradients, including loss of parameter dependence inside projection.
        step = 1e-5
        finite_difference = []
        for index in range(2):
            sample = {name: value[index].detach().item()
                      for name, value in params.items()}
            values = []
            for offset in (-step, step):
                model.update(**dict(sample, ra=sample["ra"] + offset))
                values.append(float(model.loglr))
            finite_difference.append((values[1] - values[0]) / (2 * step))
        torch.testing.assert_close(
            gradient, gradient.new_tensor(finite_difference),
            rtol=1e-5, atol=1e-5,
        )


@pytest.mark.parametrize("failure", ["no_waveform", "failed", "runtime"])
@pytest.mark.parametrize("ignore_failed", [False, True])
def test_real_batch_waveform_failure_matches_scalar_policy(
        real_model, monkeypatch, failure, ignore_failed):
    from pycbc.waveform import NoWaveformError, FailedWaveformError

    models, params, backend = real_model
    error_type = {
        "no_waveform": NoWaveformError,
        "failed": FailedWaveformError,
        "runtime": RuntimeError,
    }[failure]

    def fail(**_params):
        raise error_type("waveform generation failed")

    for model in models:
        model.ignore_failed_waveforms = ignore_failed
        monkeypatch.setattr(model.waveform_generator.rframe_generator,
                            "generate", fail)
        model.update(**{name: value[0] for name, value in params.items()})
        if failure != "no_waveform" and not ignore_failed:
            with pytest.raises(error_type):
                _ = model.loglr
            with pytest.raises(error_type):
                model.batched_loglr(**params)
            continue
        assert model.loglr == -numpy.inf
        old_stats = model._current_stats
        actual = model.batched_loglr(**params)
        assert actual.shape == (2,)
        assert model._current_stats is old_stats
        if backend == "numpy":
            assert numpy.isneginf(actual).all()
            assert numpy.isneginf(model.batched_loglikelihood(**params)).all()
        else:
            import torch

            assert actual.device == params["ra"].device
            assert torch.isneginf(actual).all()
            assert torch.isneginf(model.batched_loglikelihood(**params)).all()


@pytest.mark.parametrize("attribute", ["sampling_transforms",
                                       "waveform_transforms"])
def test_real_batch_rejects_configured_transforms(real_model, attribute):
    models, params, _ = real_model
    for model in models:
        setattr(model, attribute, [object()])
        # Unsupported configurations must remain errors even when waveform
        # failures would normally be interpreted as zero likelihood.
        model.ignore_failed_waveforms = True
        with pytest.raises(NotImplementedError, match="transforms"):
            model.batched_loglr(**params)
