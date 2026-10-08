"""Live vetoes must not rewrite the immutable JAX batch workspace."""

from types import SimpleNamespace

import numpy as np
import pytest

from pycbc import scheme
from pycbc.filter.matchedfilter import LiveBatchMatchedFilter
from pycbc.types import Array, FrequencySeries
from pycbc.vetoes.chisq import power_chisq_at_points_from_precomputed
from pycbc.vetoes.chisq_cpu import point_chisq_code

jax = pytest.importorskip("jax")


def test_live_veto_bin_cache_retains_source_and_selected_row_indices(monkeypatch):
    from pycbc.filter.matchedfilter_jax import _cache_live_veto_bins_jax
    from pycbc.vetoes import chisq_jax

    rows = jax.numpy.asarray(np.arange(24, dtype=np.float32).reshape(3, 8))
    templates = [SimpleNamespace() for _ in range(2)]
    psd = object()
    stilde = SimpleNamespace(psd=psd)
    calls = []
    monkeypatch.setattr(
        chisq_jax,
        "cache_batch_power_chisq_bins_jax",
        lambda power_chisq, batch, batch_psd, **kwargs: calls.append(
            (power_chisq, batch, batch_psd)
        ),
    )

    veto_info = [
        (None, None, None, templates[0], stilde, rows, 2),
        (None, None, None, templates[1], stilde, rows, 0),
    ]
    power_chisq = object()

    def reject_dense_transfer(*args, **kwargs):
        raise AssertionError("live veto setup must not transfer dense rows")

    with monkeypatch.context() as patch:
        patch.setattr(jax, "device_get", reject_dense_transfer)
        _cache_live_veto_bins_jax(power_chisq, veto_info)

    assert len(calls) == 1
    assert calls[0][0] is power_chisq
    assert list(calls[0][1]) == templates
    assert calls[0][2] is psd
    assert calls[0][1]._host_batch_tensor is None
    assert hasattr(calls[0][1]._batch_tensor, "devices")
    assert calls[0][1]._batch_tensor is rows
    np.testing.assert_array_equal(calls[0][1]._batch_positions, [2, 0])


@pytest.mark.parametrize("invalidate", ["params", "psd_entries", "psd"])
@pytest.mark.parametrize("dev_name", ["cpu", "cuda:0"])
def test_live_veto_bin_cache_skips_warm_gathers_and_rebuilds_stale_rows(
        monkeypatch, invalidate, dev_name):
    """Only PSD/parameter-valid bins may bypass the device source gather."""
    from pycbc.filter import matchedfilter_jax
    from pycbc.vetoes import chisq_jax
    from pycbc.vetoes.chisq import SingleDetPowerChisq, power_chisq_bins

    if dev_name == "cuda:0":
        try:
            jax.devices("gpu")
        except RuntimeError:
            pytest.skip("CUDA JAX device unavailable")
    rng = np.random.default_rng(119)
    raw = (rng.normal(size=(3, 129))
           + 1j * rng.normal(size=(3, 129))).astype(np.complex64)
    # A line spectrum exercises repeated edges and empty frequency bins.
    raw[0] = 0
    raw[0, 80] = 1 + 2j
    psd_values = np.linspace(.5, 3, 129, dtype=np.float32)
    positions, lowers = [2, 0], [10.0, 43.0]
    with scheme.CPUScheme():
        reference_psd = FrequencySeries(psd_values, delta_f=1.0)
        expected = [np.asarray(power_chisq_bins(
            FrequencySeries(raw[row], delta_f=1.0), 4, reference_psd,
            lower)) for row, lower in zip(positions, lowers)]

    calls = []
    original = chisq_jax.cache_batch_power_chisq_bins_jax

    def record_misses(power, batch, psd, **kwargs):
        calls.append((list(batch), list(batch._batch_positions)))
        return original(power, batch, psd, **kwargs)

    monkeypatch.setattr(chisq_jax, "cache_batch_power_chisq_bins_jax",
                        record_misses)
    with scheme.JAXScheme(dev_name):
        source = jax.numpy.asarray(raw)
        psd = FrequencySeries(psd_values, delta_f=1.0)
        stilde = SimpleNamespace(psd=psd)
        templates = []
        for row, lower in zip(positions, lowers):
            template = FrequencySeries(raw[row], delta_f=1.0)
            template.f_lower = lower
            template.params = SimpleNamespace()
            templates.append(template)
        infos = [(None, None, None, template, stilde, source, row)
                 for template, row in zip(templates, positions)]
        power = SingleDetPowerChisq("4")
        matchedfilter_jax._cache_live_veto_bins_jax(power, infos)
        cached = [template._bin_cache[id(psd)] for template in templates]
        for actual, reference in zip(cached, expected):
            np.testing.assert_array_equal(actual, reference)
        assert calls == [(templates, positions)]

        original_to_jax = matchedfilter_jax.to_jax

        def reject_source_gather(value, *args, **kwargs):
            if value is source:
                raise AssertionError("warm bin cache gathered source rows")
            return original_to_jax(value, *args, **kwargs)

        def reject_transfer(*args, **kwargs):
            raise AssertionError("warm bin cache synchronized device data")

        with monkeypatch.context() as patch:
            patch.setattr(matchedfilter_jax, "to_jax", reject_source_gather)
            patch.setattr(jax, "device_get", reject_transfer)
            matchedfilter_jax._cache_live_veto_bins_jax(power, infos)
        assert len(calls) == 1
        assert all(template._bin_cache[id(psd)] is bins
                   for template, bins in zip(templates, cached))

        if invalidate == "params":
            templates[0].params = SimpleNamespace()
            stale, stale_positions = templates[:1], positions[:1]
        elif invalidate == "psd_entries":
            psd._chisq_cached_key.clear()
            stale, stale_positions = templates, positions
        else:
            psd = FrequencySeries(psd_values, delta_f=1.0)
            stilde.psd = psd
            stale, stale_positions = templates, positions
        matchedfilter_jax._cache_live_veto_bins_jax(power, infos)
        assert calls[-1] == (stale, stale_positions)
        assert len(calls) == 2
        for template, reference in zip(templates, expected):
            assert id(template.params) in psd._chisq_cached_key
            np.testing.assert_array_equal(template._bin_cache[id(psd)],
                                          reference)


@pytest.mark.parametrize("lazy_candidates", [False, True])
def test_live_veto_scratch_preserves_batch_and_reuses_matching_lengths(
        monkeypatch, lazy_candidates):
    from pycbc.filter.matchedfilter_jax import _LiveVetoCandidate

    rng = np.random.default_rng(342)
    captures = []
    references = []
    cache_calls = []
    bins = np.array([40, 100, 600, 1400], dtype=np.uint32)

    def values(corr, snrv, norm, psd, indices, template):
        captures.append((corr, np.asarray(corr).copy()))
        chisq = power_chisq_at_points_from_precomputed(
            corr, snrv, norm, bins, indices
        )
        return np.asarray(chisq), np.array([4], dtype=np.uint32)

    control = object.__new__(LiveBatchMatchedFilter)
    control.power_chisq = SimpleNamespace(
        do=True,
        cached_chisq_bins=lambda *args: bins,
        values=values,
    )
    control.sg_chisq = SimpleNamespace(values=lambda *args: None)
    control.newsnr_threshold = None

    from pycbc.vetoes import chisq_jax
    monkeypatch.setattr(
        chisq_jax,
        "cache_batch_power_chisq_bins_jax",
        lambda power_chisq, templates, psd, **kwargs: cache_calls.append(
            (power_chisq, list(templates), psd)
        ),
    )

    with scheme.JAXScheme("cpu"):
        parents, vetoes = [], []
        for n in (4096, 6144, 4096):
            flen = n // 2 + 1
            data = (rng.normal(size=(2, flen)) +
                    1j * rng.normal(size=(2, flen))).astype(np.complex64)
            template = FrequencySeries(data[0], delta_f=2048 / n)
            strain = FrequencySeries(data[1], delta_f=2048 / n)
            strain.psd = None
            parent = Array(np.full(3 * n, 7 + 2j, dtype=np.complex64))
            template.cout = parent[n:2 * n]
            parents.append(parent)
            snrv, norm, index = np.array([2 + 1j]), 0.25, 2000
            if lazy_candidates:
                vetoes.append(_LiveVetoCandidate(
                    jax.numpy.asarray(snrv), jax.numpy.asarray([norm]), 0,
                    index, template, strain, None, None))
            else:
                vetoes.append((snrv, norm, index, template, strain))
            expected = np.zeros(n, dtype=np.complex64)
            expected[:flen] = np.conj(data[0]) * data[1]
            references.append(expected)

        result = control._process_vetoes({}, vetoes)
        for parent in parents:
            np.testing.assert_array_equal(np.asarray(parent), 7 + 2j)
        assert captures[0][0] is captures[2][0]
        assert captures[0][0] is not captures[1][0]
        for (corr, got), expected in zip(captures, references):
            assert corr._data.parent is None
            np.testing.assert_allclose(got, expected, rtol=2e-6, atol=2e-6)

    # Small CPU batches retain scalar exact scans; CUDA batches are populated
    # together to reduce executable loads and dispatch overhead.
    assert cache_calls == []

    # Use the captured device correlation as input to the independent native
    # CPU point-chi-square kernel. This compares the veto reduction against
    # the same product while avoiding a second implementation of the phase
    # recurrence in this test.
    expected_chisq = []
    for _, corr in captures:
        corr = np.asarray(corr, dtype=np.complex64)
        point = np.zeros(1, dtype=np.float32)
        point_chisq_code(
            point, corr, 1, len(corr), np.array([2000], dtype=np.float32),
            bins, len(bins) - 1,
        )
        expected_chisq.append((point[0] * 3 - 5) * 0.25**2 / 4)
    np.testing.assert_allclose(result['chisq'], expected_chisq, rtol=2e-5)
    assert isinstance(result['chisq'], jax.Array)
    assert isinstance(result['chisq_dof'], jax.Array)
    assert isinstance(result['sg_chisq'], jax.Array)
    np.testing.assert_array_equal(result['chisq_dof'], 4)
    np.testing.assert_array_equal(result['sg_chisq'], 0)
