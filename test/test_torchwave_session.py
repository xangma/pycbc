"""Persistent provider cache ownership and content invalidation."""
import h5py
import numpy as np
import pytest

from pycbc import scheme
from pycbc.types import FrequencySeries
from pycbc.waveform.bank import FilterBank
from pycbc.filter.gpu_search.inspiral_session import InspiralSession

torch = pytest.importorskip('torch')
pytest.importorskip('torchwave')
pytestmark = pytest.mark.skipif(
    not hasattr(FilterBank, 'get_batch_tensor'), reason='Provider in later stack branch')


@pytest.fixture
def bank(tmp_path):
    path = tmp_path / 'bank.hdf'
    with h5py.File(path, 'w') as f:
        for name, values in dict(mass1=[10., 12.], mass2=[2., 3.],
                                 spin1z=[0., 0.], spin2z=[0., 0.],
                                 f_lower=[30., 30.]).items():
            f[name] = values
    return FilterBank(str(path), 1025, .5, approximant='TaylorF2',
                      dtype=np.complex64, enable_torchwave=True)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_owned_batch_cache_changes_parameters_precision_and_budget(bank, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    session = InspiralSession(cache_bytes=1000000)
    session.bind_bank(bank, {})
    with scheme.TorchScheme(device=device):
        a, metadata = session.template_batch(bank, [1, 0], device=device)
        expected = a.clone()
        a.zero_()
        b, second = session.template_batch(bank, [1, 0], device=device)
        assert session.stats['batch_hits'] == 1
        assert torch.equal(b, expected)
        assert second[0].data.tensor.data_ptr() == b[0].data_ptr()
        assert metadata[0] is not second[0]
        bank.extra_args['coa_phase'] = .3
        c, _ = session.template_batch(bank, [1, 0], device=device)
        assert session.stats['batch_misses'] == 2
        assert not torch.equal(c, b)
        d, _ = session.template_batch(bank, [1, 0], device=device,
                                      dtype=torch.complex128)
        assert d.dtype == torch.complex128
        assert session.stats['batch_misses'] == 3
        assert session.current_bytes <= session.cache_bytes
        small = InspiralSession(cache_bytes=1)
        small.template_batch(bank, [0], device=device)
        small.template_batch(bank, [0], device=device)
        assert small.current_bytes == 0 and small.stats['batch_hits'] == 0


@pytest.mark.parametrize('budget', [0, 1000000])
def test_psd_mutation_with_same_object_recomputes_norm(bank, budget):
    session = InspiralSession(cache_bytes=budget)
    session.bind_bank(bank, {})
    _, templates = session.template_batch(bank, [0])
    psd = FrequencySeries(np.ones(1025, dtype=np.float32), delta_f=.5)
    first = session.sigmasq(templates[0], 0, psd)
    psd *= 2
    second = session.sigmasq(templates[0], 0, psd)
    assert second == pytest.approx(first / 2, rel=2e-7)


def test_batch_norms_hash_psd_once_and_recheck_next_call(bank, monkeypatch):
    session = InspiralSession(cache_bytes=1000000)
    _, templates = session.template_batch(bank, [0, 1])
    psd = FrequencySeries(np.ones(1025, dtype=np.float32), delta_f=.5)
    calls = []
    original = session._psd_digest
    def digest(value):
        calls.append(value)
        return original(value)
    monkeypatch.setattr(session, '_psd_digest', digest)
    first = session.sigmasq_batch(templates, [0, 1], psd)
    assert len(calls) == 1
    psd *= 2
    second = session.sigmasq_batch(templates, [0, 1], psd)
    assert len(calls) == 2
    np.testing.assert_allclose(second, np.asarray(first)/2)
    with pytest.raises(TypeError):
        session.template_batch(bank, [0.9])
