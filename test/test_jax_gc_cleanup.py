"""Tests for bounded JAX batch-cache garbage-collection cleanup."""

import gc
from types import SimpleNamespace

from pycbc.waveform.bank import FilterBank


def _bank_with_cached_templates():
    bank = object.__new__(FilterBank)
    bank._last_batch_tensor = object()
    bank._last_batch_indices = (1, 2)
    bank._template_cache = {
        index: SimpleNamespace(
            sigma_view=object(),
            _batch_tensor=object(),
            _data_inst=object(),
            _sigmasq={"psd": 1.0},
        )
        for index in (1, 2)
    }
    return bank


def test_clear_batch_cache_collects_by_default(monkeypatch):
    calls = []
    monkeypatch.setattr(gc, "collect", lambda: calls.append(True))

    bank = _bank_with_cached_templates()
    evicted = bank._template_cache[1]
    bank.clear_batch_cache([1])

    assert 1 not in bank._template_cache
    assert 2 in bank._template_cache
    assert not hasattr(evicted, "sigma_view")
    assert evicted._batch_tensor is None
    assert evicted._data_inst is None
    assert evicted._sigmasq == {}
    assert bank._last_batch_tensor is None
    assert bank._last_batch_indices is None
    assert calls == [True]


def test_clear_batch_cache_can_defer_collection(monkeypatch):
    calls = []
    monkeypatch.setattr(gc, "collect", lambda: calls.append(True))

    bank = _bank_with_cached_templates()
    evicted = bank._template_cache[1]
    bank.clear_batch_cache([1], collect=False)

    assert 1 not in bank._template_cache
    assert 2 in bank._template_cache
    assert not hasattr(evicted, "sigma_view")
    assert evicted._batch_tensor is None
    assert evicted._data_inst is None
    assert evicted._sigmasq == {}
    assert bank._last_batch_tensor is None
    assert bank._last_batch_indices is None
    assert calls == []
