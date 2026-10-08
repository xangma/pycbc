"""Setup contracts of the dedicated JAX inspiral search."""

from types import SimpleNamespace

import pytest

from pycbc import scheme, strain
from pycbc.filter.inspiral_jax import batch_size_from_cli, prepare_data


def _error(message):
    raise ValueError(message)


@pytest.mark.parametrize('platform,requested,expected', [
    ('cpu', None, 16), ('gpu', None, 64), ('cpu', 8, 8), ('gpu', 1, 1)])
def test_batch_size_defaults_and_explicit_single(platform, requested, expected):
    options = SimpleNamespace(batch_size=requested,
                              multiprocessing_nprocesses=None)
    context = SimpleNamespace(jax_device=SimpleNamespace(platform=platform))
    assert batch_size_from_cli(options, context,
                               SimpleNamespace(error=_error)) == expected


@pytest.mark.parametrize('count,processes,message', [
    (0, None, 'positive'), (-1, None, 'positive'),
    (8, 2, 'multiprocessing')])
def test_invalid_execution_geometry(count, processes, message):
    options = SimpleNamespace(batch_size=count,
                              multiprocessing_nprocesses=processes)
    with pytest.raises(ValueError, match=message):
        batch_size_from_cli(options, None, SimpleNamespace(error=_error))


def test_conditioning_runs_on_selected_scheme(monkeypatch):
    events = []
    context = scheme.JAXScheme('cpu')
    options, rejection, series, segments = (object() for _ in range(4))

    def load(received, **kwargs):
        assert received is options
        assert kwargs == {'dyn_range_fac': 2., 'inj_filter_rejector': rejection}
        assert scheme.mgr.state is context
        events.append('load')
        return series

    def segment(received, data):
        assert received is options and data is series
        assert scheme.mgr.state is context
        events.append('segments')
        return segments

    monkeypatch.setattr(strain, 'from_cli', load)
    monkeypatch.setattr(strain.StrainSegments, 'from_cli', segment)
    prior = scheme.mgr.state
    assert prepare_data(options, context, 2., rejection) == (series, segments)
    assert scheme.mgr.state is prior
    assert events == ['load', 'segments']
