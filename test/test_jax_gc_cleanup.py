"""Cache eviction releases bank ownership while preserving returned templates."""

import gc
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

from pycbc import scheme
from pycbc.waveform.bank import FilterBank
from pycbc.waveform.bank_jax import LazyFrequencySeries

jax = pytest.importorskip('jax')


@pytest.mark.parametrize('collect', [False, True])
def test_clear_batch_cache_preserves_caller_owned_views(monkeypatch, collect):
    calls = []
    monkeypatch.setattr(gc, 'collect', lambda: calls.append(True))
    with scheme.JAXScheme('cpu'):
        values = jax.numpy.asarray(np.arange(16, dtype=np.complex64).reshape(2, 8))
        templates = [LazyFrequencySeries(values, index, .25) for index in range(2)]
        bank = object.__new__(FilterBank)
        bank._template_cache = dict(enumerate(templates))
        bank._last_batch_tensor = values
        bank._last_batch_indices = (0, 1)
        before = np.asarray(templates[0]).copy()
        bank.clear_batch_cache([0], collect=collect)
        assert 0 not in bank._template_cache
        assert bank._template_cache[1] is templates[1]
        assert bank._last_batch_tensor is None
        assert bank._last_batch_indices is None
        assert np.asarray(templates[0]).tobytes() == before.tobytes()
        assert templates[0].shape == (8,)
    assert calls == ([True] if collect else [])


def test_prefetch_is_discarded_when_reference_selection_changes(monkeypatch):
    from pycbc.waveform import bank_jax, diffgw_jax

    with scheme.JAXScheme('cpu', reference_operations=('decompress',)):
        old = jax.numpy.ones((1, 8), dtype=jax.numpy.complex64)
        future = Future()
        future.set_result(((0,), None, old,
                           {0: LazyFrequencySeries(old, 0, .25)}))
        bank = SimpleNamespace(
            _template_cache={},
            _template_cache_backend_key=scheme.current_backend_key(),
            _prefetch_indices=(0,), _prefetch_future=future,
            _prefetch_backend_key=scheme.current_backend_key(),
            has_compressed_waveforms=False, enable_compressed_waveforms=False)
    with scheme.JAXScheme('cpu'):
        current = jax.numpy.full((1, 8), 2, dtype=jax.numpy.complex64)
        monkeypatch.setattr(diffgw_jax, 'generate_batch', lambda *args: (
            current, [LazyFrequencySeries(current, 0, .25)]))
        result = bank_jax.get_batch_jax(bank, [0])
        np.testing.assert_array_equal(np.asarray(result[0]), current[0])
