"""Native template normalization used by live JAX filtering."""

import types

import numpy as np
import pytest
import jax

from pycbc import scheme
from pycbc.filter.matchedfilter import _live_template_norms
from pycbc.filter.matchedfilter_jax import (
    batch_template_matrix_jax,
    batch_template_power_jax,
    live_template_norms_jax,
)
from pycbc.types import FrequencySeries
from pycbc import waveform
from pycbc.waveform.bank import sigma_cached


def _jax_devices():
    devices = ["cpu"]
    try:
        jax.devices("gpu")
    except RuntimeError:
        return devices
    return devices + ["cuda"]


def _templates(seed=0):
    n = 4097  # 2048 Hz, four seconds, complex64 frequency series
    rng = np.random.default_rng(seed)
    templates = []
    for f_lower, end_frequency in ((20, 700), (31, 900), (47, 1000)):
        t = FrequencySeries(
            (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64),
            delta_f=0.25,
        )
        t.approximant = "TaylorF2"
        t.f_lower = f_lower
        t.min_f_lower = f_lower - 7 if f_lower == 31 else f_lower
        t.end_frequency = end_frequency
        t.end_idx = int(end_frequency / t.delta_f)
        t.params = types.SimpleNamespace()
        t.sigma_scale = 1.0
        t.sigmasq = types.MethodType(sigma_cached, t)
        templates.append(t)
    return templates


@pytest.mark.parametrize("device", _jax_devices())
def test_live_template_norms_matches_original_sigma_cached(device):
    candidate_templates = _templates(1)
    psd = FrequencySeries(
        np.linspace(1, 2, 4097, dtype=np.float32), delta_f=0.25
    )
    candidate_psd = FrequencySeries(np.asarray(psd).copy(), delta_f=0.25)

    with scheme.CPUScheme():
        expected = np.asarray([template.sigmasq(psd)
                               for template in candidate_templates])

    with scheme.JAXScheme(device):
        state = scheme.mgr.state
        device_norms = _live_template_norms(candidate_templates, candidate_psd)
        got = np.asarray(device_norms).copy()
        assert scheme.mgr.state is state
    np.testing.assert_allclose(got, expected, rtol=2e-6)
    assert type(device_norms).__module__.startswith("jax")
    assert device_norms.dtype == np.dtype(np.float64)

    psd2 = FrequencySeries(
        np.linspace(2, 3, 4097, dtype=np.float32), delta_f=0.25
    )
    candidate_psd2 = FrequencySeries(np.asarray(psd2).copy(), delta_f=0.25)
    with scheme.CPUScheme():
        expected2 = np.asarray([template.sigmasq(psd2)
                                for template in candidate_templates])
    with scheme.JAXScheme(device):
        got2 = np.asarray(
            _live_template_norms(candidate_templates, candidate_psd2)
        ).copy()
    np.testing.assert_allclose(got2, expected2, rtol=2e-6)
    assert np.any(got2 != got)


def test_live_template_norms_restores_context_on_exception():
    templates = _templates()
    psd = FrequencySeries(np.ones(4097, dtype=np.float32), delta_f=0.25)
    templates[1].f_lower = "invalid"
    templates[1].min_f_lower = "invalid"
    with scheme.JAXScheme("cpu"):
        state = scheme.mgr.state
        with pytest.raises((TypeError, ValueError)):
            _live_template_norms(templates, psd)
        assert scheme.mgr.state is state


def test_live_template_norms_cpu_path_is_unchanged():
    templates = _templates()
    psd = FrequencySeries(np.ones(4097, dtype=np.float32), delta_f=0.25)
    with scheme.CPUScheme():
        expected = np.asarray([t.sigmasq(psd) for t in templates]).copy()
        got = np.asarray(_live_template_norms(templates, psd)).copy()
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("device", _jax_devices())
def test_cached_template_power_preserves_live_norms(device):
    templates = _templates(7)
    psd = FrequencySeries(
        np.linspace(0.8, 2.4, 4097, dtype=np.float32), delta_f=0.25)
    with scheme.JAXScheme(device):
        matrix = batch_template_matrix_jax(templates, len(templates[0]))
        power = batch_template_power_jax(matrix, templates[0].delta_f)
        expected = live_template_norms_jax(
            templates, psd, template_matrix=matrix)
        actual = live_template_norms_jax(
            templates, psd, template_matrix=matrix,
            template_power=power)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("psd_dtype", [np.float32, np.float64])
def test_spa_norm_matches_native_float32_stages(device, psd_dtype):
    length = 4097
    delta_f = 0.3
    f_lower = 1.2
    psd = FrequencySeries(
        np.linspace(0.7, 2.1, length, dtype=psd_dtype), delta_f=delta_f
    )
    with scheme.CPUScheme():
        expected = np.asarray(waveform.get_waveform_filter_norm(
            "SPAtmplt", psd, length, delta_f, f_lower))
    with scheme.JAXScheme(device):
        got_jax = waveform.get_waveform_filter_norm(
            "SPAtmplt", psd, length, delta_f, f_lower)
        got = np.asarray(got_jax)
    assert got_jax.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(got, expected, rtol=3e-6, atol=1e-12)


@pytest.mark.parametrize("device", _jax_devices())
def test_live_spa_norm_computes_missing_amplitude_scale(device):
    def template():
        result = _templates()[0]
        result.approximant = "SPAtmplt"
        result.params = types.SimpleNamespace(mass1=12.0, mass2=9.0)
        del result.sigma_scale
        return result

    reference, candidate = template(), template()
    psd = FrequencySeries(np.linspace(0.7, 2.1, 4097, dtype=np.float32),
                          delta_f=0.25)
    candidate_psd = FrequencySeries(psd.numpy().copy(), delta_f=0.25)
    with scheme.CPUScheme():
        expected = reference.sigmasq(psd)
    with scheme.JAXScheme(device):
        actual = _live_template_norms([candidate], candidate_psd)
        assert actual.dtype == np.dtype(np.float64)
        assert actual.devices() == {scheme.mgr.state.jax_device}
    np.testing.assert_allclose(np.asarray(actual)[0], expected, rtol=2e-6)
