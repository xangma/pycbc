# Copyright (C) 2026
# This program is free software under the GNU General Public License,
# version 3 or (at your option) any later version.
"""Integration checks for the provider-owned TorchWave source identity."""

import pytest

from pycbc.waveform import torchwave as adapter


pytest.importorskip('torchwave')
from torchwave import provenance


@pytest.fixture(autouse=True)
def reset_identity():
    adapter.provider_identity.cache_clear()
    yield
    adapter.provider_identity.cache_clear()


def test_absent_optional_provider(monkeypatch):
    monkeypatch.setattr(adapter.importlib.util, 'find_spec', lambda name: None)
    assert adapter.provider_identity() is None


def test_relocated_provider_identity(monkeypatch):
    assert adapter.provider_identity() == provenance.taylorf2_source_identity()
    adapter.provider_identity.cache_clear()
    monkeypatch.setattr(provenance, 'taylorf2_source_identity',
                        lambda: 'provider-owned-value')
    assert adapter.provider_identity() == 'provider-owned-value'


def test_wrapper_owns_cache(monkeypatch):
    values = iter(['first', 'second'])
    monkeypatch.setattr(provenance, 'taylorf2_source_identity',
                        lambda: next(values))
    assert adapter.provider_identity() == 'first'
    assert adapter.provider_identity() == 'first'
    adapter.provider_identity.cache_clear()
    assert adapter.provider_identity() == 'second'


def test_provider_error_is_not_hidden(monkeypatch):
    def broken():
        raise OSError('provider source unavailable')

    monkeypatch.setattr(provenance, 'taylorf2_source_identity', broken)
    with pytest.raises(OSError, match='provider source unavailable'):
        adapter.provider_identity()


def test_missing_provider_api_is_an_error(monkeypatch):
    monkeypatch.delattr(provenance, 'taylorf2_source_identity')
    with pytest.raises(ImportError, match='taylorf2_source_identity'):
        adapter.provider_identity()
