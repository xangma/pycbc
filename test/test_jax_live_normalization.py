"""Native template normalization used by live JAX filtering."""

import types

import numpy as np
import pytest
import jax

from pycbc import scheme
from pycbc.filter.matchedfilter_jax import (
    batch_template_matrix_jax,
    batch_template_power_jax,
    live_template_norms_jax,
)
from pycbc.types import FrequencySeries
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
        device_norms = live_template_norms_jax(candidate_templates, candidate_psd)
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
            live_template_norms_jax(candidate_templates, candidate_psd2)
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
            live_template_norms_jax(templates, psd)
        assert scheme.mgr.state is state


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


def test_cached_generic_norm_dispatch_does_not_grow_per_template(monkeypatch):
    """Warm generic normalization must not dispatch one scalar read per row."""
    from jax._src import dispatch

    counts = []
    original = dispatch.apply_primitive
    with scheme.JAXScheme("cpu"):
        psd = FrequencySeries(np.ones(4097, np.float32), delta_f=0.25)
        for size in (8, 128):
            templates = (_templates(19) * ((size + 2) // 3))[:size]
            matrix = batch_template_matrix_jax(templates, 4097)
            power = batch_template_power_jax(matrix, 0.25)
            live_template_norms_jax(templates, psd, template_power=power
                                    ).block_until_ready()
            calls = []

            def counted(*args, **kwargs):
                calls.append(args[0])
                return original(*args, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(dispatch, "apply_primitive", counted)
                actual = live_template_norms_jax(
                    templates, psd, template_power=power)
                actual.block_until_ready()
            assert actual.shape == (size,)
            counts.append(len(calls))
    assert counts[1] <= counts[0] + 2


def _norm_correlator(templates, immutable=True):
    from pycbc.filter.matchedfilter import BatchCorrelator
    from pycbc.types import zeros

    return BatchCorrelator(
        templates, [zeros(len(template), dtype=np.complex64)
                    for template in templates], len(templates[0]),
        immutable_templates=immutable)


@pytest.mark.parametrize("device", _jax_devices())
def test_live_norm_cache_reuses_alternating_detector_psds_and_groups(
        device, monkeypatch):
    from pycbc.filter import matchedfilter_jax as module

    with scheme.JAXScheme(device):
        groups = [_templates(51), _templates(52)[:1]]
        correlators = [_norm_correlator(group) for group in groups]
        psds = [FrequencySeries(
            np.linspace(start, start + 1, 4097, dtype=np.float32),
            delta_f=.25) for start in (1, 2)]
        expected = {}
        for psd_index, psd in enumerate(psds):
            for group_index, (group, corr) in enumerate(zip(groups, correlators)):
                native = live_template_norms_jax(
                    group, psd, template_power=corr._jax_template_power)
                expected[psd_index, group_index] = (
                    native, (4.0 * group[0].delta_f) / jax.numpy.sqrt(native))

        original = module.live_template_norms_jax
        calls = []

        def normalize(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs)

        monkeypatch.setattr(module, "live_template_norms_jax", normalize)
        loaded = {}
        for repetition in range(2):
            for psd_index, psd in enumerate(psds):
                for group_index, (group, corr) in enumerate(
                        zip(groups, correlators)):
                    actual = module._live_cached_template_norms_jax(
                        corr, group, psd)
                    for value, reference in zip(
                            actual, expected[psd_index, group_index]):
                        assert value.dtype == reference.dtype
                        np.testing.assert_array_equal(value, reference)
                    key = psd_index, group_index
                    if repetition:
                        assert all(value is previous for value, previous in
                                   zip(actual, loaded[key]))
                    else:
                        loaded[key] = actual
            assert len(calls) == 4


@pytest.mark.parametrize("device", _jax_devices())
@pytest.mark.parametrize("replacement", ["matrix", "power", "psd_array",
                                         "psd", "templates"])
def test_live_norm_cache_invalidates_replaced_source(device, replacement,
                                                   monkeypatch):
    from pycbc.filter import matchedfilter_jax as module

    with scheme.JAXScheme(device):
        templates = _templates(53)
        corr = _norm_correlator(templates)
        psd = FrequencySeries(np.ones(4097, np.float32), delta_f=.25)
        first = module._live_cached_template_norms_jax(corr, templates, psd)
        if replacement == "matrix":
            corr._jax_template_matrix = corr._jax_template_matrix * 1
        elif replacement == "power":
            corr._jax_template_power = corr._jax_template_power * 2
        elif replacement == "psd_array":
            psd *= 2
        elif replacement == "psd":
            psd = FrequencySeries(np.full(4097, 2, np.float32), delta_f=.25)
        else:
            # A replacement group can retain the identical waveform matrix
            # while carrying different support metadata. Ownership follows
            # the correlator's new list, rather than its previous cache entry.
            templates = _templates(53)
            templates[0].min_f_lower += 8
            corr.xs = templates
        native = live_template_norms_jax(
            templates, psd, template_matrix=corr._jax_template_matrix,
            template_power=corr._jax_template_power)
        expected = (native, (4.0 * templates[0].delta_f)
                    / jax.numpy.sqrt(native))
        original = module.live_template_norms_jax
        calls = []

        def normalize(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs)

        monkeypatch.setattr(module, "live_template_norms_jax", normalize)
        actual = module._live_cached_template_norms_jax(corr, templates, psd)
        repeated = module._live_cached_template_norms_jax(corr, templates, psd)
        assert len(calls) == 1
        for value, reference, previous, retained in zip(
                actual, expected, first, repeated):
            np.testing.assert_array_equal(value, reference)
            assert value is not previous
            assert value is retained


def test_live_norm_cache_leaves_mutable_callers_uncached():
    from pycbc.filter import matchedfilter_jax as module

    with scheme.JAXScheme("cpu"):
        templates = _templates(54)
        corr = _norm_correlator(templates, immutable=False)
        psd = FrequencySeries(np.ones(4097, np.float32), delta_f=.25)
        first = module._live_cached_template_norms_jax(corr, templates, psd)
        templates[0].min_f_lower += 8
        expected = live_template_norms_jax(templates, psd)
        actual = module._live_cached_template_norms_jax(corr, templates, psd)
        np.testing.assert_array_equal(actual[0], expected)
        assert np.asarray(actual[0])[0] != np.asarray(first[0])[0]
        assert not hasattr(psd, "_jax_live_template_norms")


def test_live_norm_cache_releases_retired_correlators():
    import gc
    import weakref
    from pycbc.filter import matchedfilter_jax as module

    with scheme.JAXScheme("cpu"):
        templates = _templates(55)
        corr = _norm_correlator(templates)
        psd = FrequencySeries(np.ones(4097, np.float32), delta_f=.25)
        module._live_cached_template_norms_jax(corr, templates, psd)
        assert len(psd._jax_live_template_norms) == 1
        retired = weakref.ref(corr)
        del corr
        gc.collect()
        assert retired() is None
        assert not psd._jax_live_template_norms
