"""Analytical PSD grids preserve native contracts and validation routes."""

import os
import subprocess
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
lal = pytest.importorskip("lal")

from pycbc import scheme  # noqa: E402
from pycbc.psd import analytical_jax, reference_jax  # noqa: E402
from pycbc.psd.analytical import from_string  # noqa: E402
from pycbc.types.array_jax import JAXArrayData  # noqa: E402


MODELS = ["flat_unity", "Virgo", "iLIGOThermal", "aLIGOZeroDetHighPower",
          "aLIGOQuantumBHBH20Deg", "aLIGODesignSensitivityP1200087"]


@pytest.fixture
def device_spec():
    specification = os.environ.get("PYCBC_TEST_SCHEME", "jax:cpu")
    if specification == "jax":
        return "cpu"
    if not specification.startswith("jax:"):
        pytest.skip("analytical PSD tests require a JAX PYCBC_TEST_SCHEME")
    return specification.split(':', 1)[1]


def _reject_cpu(*args, **kwargs):
    raise AssertionError("default analytical PSD selected the CPU worker")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("cutoff", [-1., 0., 10.25, 10.5])
def test_default_analytical_psd_preserves_native_grid_mask(
        monkeypatch, device_spec, model, cutoff):
    with scheme.CPUScheme():
        expected = from_string(model, 128, .5, cutoff).numpy()
    monkeypatch.setattr(reference_jax, "cpu_psd_reference", _reject_cpu)
    with scheme.JAXScheme(device_spec) as ctx:
        actual = from_string(model, 128, .5, cutoff)
        assert isinstance(actual._data, JAXArrayData)
        assert actual._data.device == ctx.jax_device
        np.testing.assert_array_equal(actual.numpy() == 0, expected == 0)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-9, atol=0)


@pytest.mark.parametrize("cutoff", [1., 8.5, 9.])
def test_table_psd_preserves_native_low_frequency_extrapolation(
        monkeypatch, device_spec, cutoff):
    model = "aLIGODesignSensitivityP1200087"
    with scheme.CPUScheme():
        expected = from_string(model, 128, .5, cutoff).numpy()
    monkeypatch.setattr(reference_jax, "cpu_psd_reference", _reject_cpu)
    with scheme.JAXScheme(device_spec):
        actual = from_string(model, 128, .5, cutoff).numpy()
    np.testing.assert_array_equal(actual == 0, expected == 0)
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=0)


def test_table_psd_preserves_native_high_frequency_extrapolation(
        monkeypatch, device_spec):
    model = "aLIGODesignSensitivityP1200087"
    with scheme.CPUScheme():
        expected = from_string(model, 100, 100., 10.).numpy()
    monkeypatch.setattr(reference_jax, "cpu_psd_reference", _reject_cpu)
    with scheme.JAXScheme(device_spec):
        actual = from_string(model, 100, 100., 10.).numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=0)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("api", [from_string, analytical_jax.analytical_psd])
def test_analytical_reference_matches_original_cpu_bytes(
        monkeypatch, device_spec, model, api):
    with scheme.CPUScheme():
        expected = from_string(model, 128, .5, 10.25)
        expected_values = expected.numpy().copy()

    def reject_jax(*args, **kwargs):
        raise AssertionError("analytical reference evaluated a JAX model")

    monkeypatch.setattr(analytical_jax, "analytical_psd_jax", reject_jax)
    monkeypatch.setattr(analytical_jax, "_data_file_psd", reject_jax)
    with scheme.JAXScheme(
            device_spec, reference_operations=("analytical_psd",)) as ctx:
        result = api(model, 128, .5, 10.25)
        assert scheme.mgr.state is ctx
        assert result._data.device == ctx.jax_device
        assert result.delta_f == expected.delta_f
        assert result.epoch == expected.epoch
        assert result.dtype == expected.dtype
        assert result.numpy().tobytes() == expected_values.tobytes()


def test_reduced_planck_constant_matches_native_precision():
    assert analytical_jax._HBAR_SI == lal.HBAR_SI


@pytest.mark.parametrize("length", [0, -1, 3.5, "16"])
def test_analytical_series_rejects_invalid_lengths(length):
    with pytest.raises(TypeError, match="positive integer"):
        analytical_jax.analytical_psd("flat_unity", length, .5)


def test_flat_unity_from_string_keeps_native_keyword_errors(device_spec):
    with scheme.JAXScheme(device_spec):
        with pytest.raises(TypeError):
            from_string("flat_unity", 128, .5, 10., unrelated_argument=True)


def test_raw_frequency_helper_keeps_jit_semantics_with_reference_selected(
        monkeypatch, device_spec):
    monkeypatch.setattr(reference_jax, "cpu_psd_reference", _reject_cpu)
    with scheme.JAXScheme(
            device_spec, reference_operations=("analytical_psd",)):
        values = jax.numpy.asarray([1., 10., 20.])
        calculate = jax.jit(lambda f: analytical_jax.analytical_psd_jax(
            "flat_unity", f, low_freq_cutoff=10.))
        np.testing.assert_array_equal(calculate(values), [0., 1., 1.])


def test_explicit_analytical_device_applies_to_each_model_family():
    env = os.environ.copy()
    env.pop("PYCBC_SCHEME", None)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    result = subprocess.run([sys.executable, "-c", """
import jax
from pycbc import scheme
from pycbc.psd.analytical_jax import analytical_psd
device = jax.devices('cpu')[1]
with scheme.JAXScheme('cpu'):
    for model in ('flat_unity', 'Virgo', 'aLIGODesignSensitivityP1200087'):
        result = analytical_psd(model, 128, .5, 10., device=device)
        assert result._data.device == device
print('device-ok')
"""], env=env, capture_output=True, check=True, text=True, timeout=45)
    assert result.stdout.strip() == "device-ok"
