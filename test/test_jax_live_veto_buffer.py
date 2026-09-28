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


def test_live_veto_scratch_preserves_batch_and_reuses_matching_lengths(
        monkeypatch):
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
        lambda power_chisq, templates, psd: cache_calls.append(
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
