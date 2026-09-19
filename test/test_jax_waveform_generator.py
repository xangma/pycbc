"""Focused JAX tests for waveform generation and projection helpers."""

import importlib
import sys
import types

import numpy
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

import pycbc  # noqa: E402
from pycbc import scheme  # noqa: E402
from pycbc.types import FrequencySeries  # noqa: E402
from pycbc.types.array_jax import JAXArrayData  # noqa: E402
from pycbc.types.backend import backend_array  # noqa: E402


if not getattr(pycbc, "HAVE_JAX", False):
    pytest.skip("PyCBC built without JAX support", allow_module_level=True)


@pytest.fixture
def jax_context():
    context = scheme.JAXScheme()
    try:
        yield context
    finally:
        del context
        scheme.Scheme._single = None


def _load_generator(monkeypatch):
    fake_strain = types.ModuleType("pycbc.strain")
    fake_strain.apply_gates_to_fd = lambda series, gates: series
    monkeypatch.setitem(sys.modules, "pycbc.strain", fake_strain)
    monkeypatch.delitem(sys.modules, "pycbc.waveform.generator", raising=False)
    return importlib.import_module("pycbc.waveform.generator")


class _StaticCPUGenerator:
    def __init__(self, variable_args=(), **frozen_params):
        self.variable_args = tuple(variable_args)
        self.frozen_params = frozen_params

    def generate(self, **kwargs):
        old_state = scheme.mgr.state
        old_single = scheme.Scheme._single
        scheme.Scheme._single = None
        scheme.mgr.state = scheme.CPUScheme()
        try:
            hp = FrequencySeries(
                numpy.ones(8, dtype=numpy.complex128), delta_f=0.25
            )
            hc = FrequencySeries(
                1j * numpy.ones(8, dtype=numpy.complex128), delta_f=0.25
            )
        finally:
            scheme.mgr.state = old_state
            scheme.Scheme._single = old_single
        return hp, hc


class _StaticJAXGenerator:
    def __init__(self, variable_args=(), **frozen_params):
        self.variable_args = tuple(variable_args)
        self.frozen_params = frozen_params

    def generate(self, **kwargs):
        hp = FrequencySeries(
            JAXArrayData(jnp.ones(8, dtype=jnp.complex128)),
            delta_f=0.25,
            copy=False,
        )
        hc = FrequencySeries(
            JAXArrayData(1j * jnp.ones(8, dtype=jnp.complex128)),
            delta_f=0.25,
            copy=False,
        )
        return hp, hc


def test_detector_frame_generator_moves_output_to_jax(monkeypatch, jax_context):
    generator = _load_generator(monkeypatch)

    with jax_context:
        detgen = generator.FDomainDetFrameGenerator(
            _StaticCPUGenerator,
            epoch=0.0,
            detectors=None,
            delta_f=0.25,
        )
        result = detgen.generate()["RF"]

    assert isinstance(result._data, JAXArrayData)
    numpy.testing.assert_allclose(
        numpy.asarray(result._data.array),
        numpy.ones(8, dtype=numpy.complex128),
    )


def test_time_shift_uses_an_explicit_sample_axis(jax_context):
    from pycbc.waveform.utils_jax import apply_fseries_time_shift

    with jax_context:
        batch_size = frequency_count = 4
        raw = jnp.ones(
            (batch_size, frequency_count), dtype=jnp.complex128
        )
        series = FrequencySeries(
            JAXArrayData(raw),
            delta_f=0.5,
            epoch=0.0,
            copy=False,
        )
        shifts = jnp.linspace(0.01, 0.04, batch_size)
        actual = apply_fseries_time_shift(
            series, shifts, copy=False
        )._data.array

    frequencies = jnp.arange(frequency_count) * 0.5
    expected = raw * jnp.exp(
        -2j * jnp.pi * shifts[:, None] * frequencies
    )
    numpy.testing.assert_allclose(numpy.asarray(actual), numpy.asarray(expected), rtol=1e-12)


def test_fused_projection_matches_individual_detector_results():
    from pycbc.waveform.utils_jax import fused_detector_strain_fd_jax

    key = jax.random.PRNGKey(99)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    hp = jax.random.normal(k1, (32,)) + 1j * jax.random.normal(k2, (32,))
    hc = jax.random.normal(k3, (32,)) + 1j * jax.random.normal(k4, (32,))
    fplus = [0.8, -0.4]
    fcross = [0.2, 0.7]
    delays = [0.002, -0.005]
    delta_f = 0.25

    actual = fused_detector_strain_fd_jax(
        hp, hc, fplus, fcross, delays, delta_f, kmin=2
    )
    assert actual.shape == (2, 32)
    frequencies = jnp.arange(2, 32, dtype=jnp.float64) * delta_f
    for index in range(2):
        expected = fplus[index] * hp + fcross[index] * hc
        shift = jnp.exp(-2j * jnp.pi * delays[index] * frequencies)
        expected_shifted = jnp.concatenate((expected[:2], expected[2:] * shift))
        numpy.testing.assert_allclose(numpy.asarray(actual[index]), numpy.asarray(expected_shifted), rtol=1e-12)


def test_arrival_time_is_centered_before_float32_offsets_are_applied():
    from pycbc.waveform.generator import _arrival_time_and_shift

    epoch = 1126259460.0
    reference_time = 1126259462.125
    offset = jnp.array(0.003, dtype=jnp.float32)
    arrival, relative = _arrival_time_and_shift(
        reference_time, offset, epoch
    )

    assert float(arrival) == pytest.approx(reference_time + 0.003, rel=1e-7)
    assert float(relative) == pytest.approx(2.128, rel=1e-7)


def test_arrival_time_accepts_public_backend_values():
    from pycbc.waveform.generator import _arrival_time_and_shift

    class BackendValue:
        backend = "jax"

        def __init__(self, arr):
            self.backend_array = arr

    epoch = 1126259460.0
    reference = jnp.array(epoch + 2.125, dtype=jnp.float64)
    offset = jnp.array(0.003, dtype=jnp.float32)
    arrival, relative = _arrival_time_and_shift(
        BackendValue(reference), BackendValue(offset), epoch
    )
    numpy.testing.assert_allclose(float(arrival), epoch + 2.128, rtol=1e-7)
    numpy.testing.assert_allclose(float(relative), 2.128, rtol=1e-7)


def test_detector_projection_preserves_parameter_gradients(
        monkeypatch, jax_context):
    generator = _load_generator(monkeypatch)

    class DifferentiableDetector:
        def arrival_time(self, ref_tc, ra, dec, _reference_frame):
            return ref_tc + 0.02 * ra - 0.03 * dec

        @staticmethod
        def antenna_pattern(ra, dec, polarization, arrival_time):
            return (
                jnp.cos(2.0 * polarization)
                + 0.01 * ra
                + 0.001 * arrival_time,
                jnp.sin(2.0 * polarization)
                + 0.01 * dec
                - 0.001 * arrival_time,
            )

    with jax_context:
        detgen = generator.FDomainDetFrameGenerator(
            _StaticJAXGenerator,
            epoch=0.0,
            detectors=["H1"],
            variable_args=["tc", "ra", "dec", "polarization"],
            delta_f=0.25,
        )
        detgen.detectors = {"H1": DifferentiableDetector()}

        def loss_fn(p_vec):
            p = {
                "tc": p_vec[0],
                "ra": p_vec[1],
                "dec": p_vec[2],
                "polarization": p_vec[3],
            }
            res = detgen.generate(**p)["H1"]
            arr = backend_array(res, "jax")
            return jnp.sum(jnp.real(arr)) + jnp.sum(jnp.imag(arr))

        p_init = jnp.array([2.0, 1.1, -0.4, 0.3])
        grad = jax.grad(loss_fn)(p_init)

    assert grad is not None
    assert grad.shape == (4,)
    assert numpy.all(numpy.isfinite(numpy.asarray(grad)))
