"""Focused CPU dispatch tests for the JAX batched chi-square path."""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _select_direct_phase(monkeypatch):
    import pycbc.vetoes.chisq_jax as chisq_jax
    # The FFT parity tests intentionally cover the explicit direct-phase mode.
    monkeypatch.setattr(chisq_jax, "_chisq_mode", lambda: "direct-phase")


class _LazyCorr:
    def __init__(self, tensor, pos, kmin, tlen):
        self._batch_tensor = tensor
        self._batch_pos = pos
        self._kmin = kmin
        self._tlen = tlen


def _fft_oracle(row, points, bins, ntime, base_k=0):
    output = np.zeros(len(points), dtype=np.float64)
    for start, end in zip(bins[:-1], bins[1:]):
        full = np.zeros(ntime, dtype=np.complex128)
        full[start:end] = row[start-base_k:end-base_k]
        values = (np.fft.ifft(full)*ntime)[points % ntime]
        output += abs(values)**2
    return output


def test_cpu_point_chisq_cropped_fft_parity():
    from pycbc.vetoes.chisq_jax import _point_chisq_cpu

    rng = np.random.default_rng(7)
    kmin = 113
    tlen = 4096
    row = (rng.normal(size=701) + 1j * rng.normal(size=701)).astype(np.complex64)
    bins = np.array([kmin, kmin + 17, kmin + 389, kmin + len(row) - 10], dtype=np.uint32)
    points = np.array([3, 127, 2048, 7777], dtype=np.int32)

    got = _point_chisq_cpu(row, points, bins, tlen, base_k=kmin)
    expected = _fft_oracle(row, points, bins, tlen, kmin)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_cpu_point_chisq_periodic_points():
    import pycbc.vetoes.chisq_jax as chisq_jax

    rng = np.random.default_rng(11)
    row = (rng.normal(size=32) + 1j * rng.normal(size=32)).astype(np.complex64)
    bins = np.array([0, 9, 32], dtype=np.uint32)
    points = np.array([1, 4097], dtype=np.int32)
    got = chisq_jax._point_chisq_cpu(row, points, bins, 128)
    expected = _fft_oracle(row, points, bins, 128)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_lazy_shift_sum_cpu_matches_fft():
    import pycbc.vetoes.chisq_jax as chisq_jax
    rng = np.random.default_rng(15)
    kmin, tlen = 73, 1024
    row = (rng.normal(size=211) + 1j * rng.normal(size=211)).astype(np.complex64)
    tensor = np.stack([row, row * (1 + 0.2j)])
    bins = np.array([kmin, kmin + 19, kmin + 180, kmin + len(row)], dtype=np.uint32)
    points = np.array([2, 77, 2049], dtype=np.int32)
    corr = _LazyCorr(tensor, 1, kmin, tlen)
    got = chisq_jax.shift_sum(corr, points, bins)
    expected = _fft_oracle(tensor[1], points, bins, tlen, kmin)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_lazy_shift_sum_gpu_branch_keeps_row_core(monkeypatch):
    import pycbc.vetoes.chisq_jax as chisq_jax
    tensor = np.ones((2, 32), dtype=np.complex64)
    corr = _LazyCorr(tensor, 1, 0, 64)
    calls = []

    def fake_core(*args, **kwargs):
        calls.append(args[1])
        return np.array([3.0] * len(args[3]))

    monkeypatch.setattr(chisq_jax, "_array_is_cpu", lambda _: False)
    monkeypatch.setattr(chisq_jax, "_shift_sum_gpu_core_row", fake_core)
    got = chisq_jax.shift_sum(corr, np.array([1, 2]), np.array([0, 16, 32]))
    assert calls == [1]
    np.testing.assert_array_equal(got, [3.0, 3.0])


def test_batch_empty_and_threshold_gated_candidates():
    from pycbc.vetoes.chisq_jax import batch_power_chisq_jax

    class Corr:
        _kmin = 0
        _tlen = 64

    class Template:
        pass

    psd = object()
    tmpl = Template()
    tmpl._bin_cache = {id(psd): np.array([0, 8, 16], dtype=np.uint32)}
    empty = np.array([], dtype=np.uint32)
    gated = np.array([1], dtype=np.uint32)
    results = [
        (None, 1.0, Corr(), empty, empty.astype(np.complex64)),
        (None, 1.0, Corr(), gated, np.array([1 + 0j], dtype=np.complex64)),
    ]
    got = batch_power_chisq_jax(
        np.ones((2, 16), dtype=np.complex64), results, [tmpl, tmpl], psd, 0,
        snr_threshold=5.0
    )
    assert got[0][0].size == 0
    np.testing.assert_array_equal(got[1][0], [0.0])


def test_batch_cpu_jax_dispatch_groups_candidates_and_thresholds(monkeypatch):
    from pycbc.vetoes.chisq_jax import batch_power_chisq_jax

    class Corr:
        _kmin = 97
        _tlen = 2048

    class Template:
        pass

    rng = np.random.default_rng(9)
    kmin, nfreq, ntemplates = 97, 313, 2
    tensor = (rng.normal(size=(ntemplates, nfreq)) +
              1j * rng.normal(size=(ntemplates, nfreq))).astype(np.complex64)
    bins = np.array([kmin, kmin + 31, kmin + 177, kmin + nfreq], dtype=np.uint32)
    psd = object()
    templates = []
    for _ in range(ntemplates):
        tmpl = Template()
        tmpl._bin_cache = {id(psd): bins}
        templates.append(tmpl)
    indices = np.array([2, 61, 511], dtype=np.uint32)
    snrv = np.array([10 + 0j, 2 + 0j, 40 + 0j], dtype=np.complex64)
    results = [(None, 1.0, Corr(), indices, snrv) for _ in templates]

    calls = []
    original = __import__("pycbc.vetoes.chisq_jax", fromlist=["_point_chisq_cpu"])._point_chisq_cpu

    def wrapped(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr("pycbc.vetoes.chisq_jax._point_chisq_cpu", wrapped)
    got = batch_power_chisq_jax(
        tensor, results, templates, psd, 0, snr_threshold=5.0
    )
    assert calls == []
    assert all(value[0].devices() for value in got.values())
    for row, (chisq, dof) in zip(tensor, got.values()):
        assert len(chisq) == len(indices)
        assert np.all(dof == 4)
        above = np.array([True, False, True])
        expected_shift = _fft_oracle(row, indices[above], bins, 2048, kmin)
        expected = np.zeros(3, dtype=np.float32)
        expected[above] = expected_shift * 3 - np.abs(snrv[above]) ** 2
        np.testing.assert_allclose(chisq, expected, rtol=3e-5, atol=3e-5)
