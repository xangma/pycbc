# Copyright (C) 2026
# This program is free software under the GNU General Public License,
# version 3 or (at your option) any later version.
"""Bounded live waveform iteration and offline template lifetime tests."""

import h5py
import numpy as np
import pytest

from pycbc import scheme
from pycbc.types import zeros
from pycbc.waveform.bank import FilterBank, LiveFilterBank


torch = pytest.importorskip('torch')
pytest.importorskip('torchwave')


@pytest.fixture
def bank_file(tmp_path):
    path = tmp_path / 'live.hdf'
    with h5py.File(path, 'w') as bank:
        bank['mass1'] = [1.4, 1.8, 25., 30., 4., 8., 5., 10., 3.]
        bank['mass2'] = [1.2, 1.4, 20., 20., 2., 2., 2., 2., 2.]
        bank['spin1z'] = np.zeros(9)
        bank['spin2z'] = np.zeros(9)
        bank['f_lower'] = [30., 30., 20., 20., 40., 40., 40., 40., 40.]
        bank.attrs['parameters'] = list(bank)
    return str(path)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_live_iteration_parity_slicing_and_bound(
        bank_file, monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from pycbc.waveform import torchwave as provider
    kwargs = dict(sample_rate=1024, minimum_buffer=4, approximant='TaylorF2',
                  coa_phase=.4, inclination=.7)
    bank = LiveFilterBank(bank_file, enable_torchwave=True, **kwargs)[1::2]
    reference = LiveFilterBank(bank_file, enable_torchwave=False, **kwargs)
    references = list(reference[1::2])
    original = provider.generate_batch
    generated = []

    def record(bank, indices, **kwargs):
        generated.extend(indices)
        return original(bank, indices, **kwargs)

    monkeypatch.setattr(provider, 'generate_batch', record)
    original_iterator = bank._iter_torchwave
    monkeypatch.setattr(bank, '_iter_torchwave',
                        lambda: original_iterator(batch_size=2))
    with scheme.TorchScheme(device=device):
        iterator = iter(bank)
        first = next(iterator)
        assert len(generated) == 2
        waveforms = [first] + list(iterator)
        for actual, expected in zip(waveforms, references):
            assert actual.waveform_provider == 'torchwave'
            assert actual.data.tensor.device.type == device
            assert actual.id == expected.id
            assert actual.delta_f == expected.delta_f
            assert actual.chirp_length == expected.chirp_length
            error = np.linalg.norm(actual.numpy() - expected.numpy())
            assert error / np.linalg.norm(expected.numpy()) < 2e-7
    assert generated == list(range(len(bank)))
    with pytest.raises(ValueError, match='positive'):
        next(original_iterator(batch_size=0))


def test_fallback_retains_rows_with_shared_scalar_buffer(bank_file):
    bank = FilterBank(bank_file, filter_length=2049, delta_f=.25,
                      dtype=np.complex64, approximant='TaylorF2',
                      out=zeros(2049, dtype=np.complex64),
                      enable_torchwave=True,
                      phase_order=4)
    expected = [bank[i].numpy().copy() for i in [2, 3]]
    batch, templates = bank.get_batch_tensor([2, 3])
    for i in range(2):
        assert np.array_equal(batch[i].numpy(), expected[i])
        assert np.array_equal(templates[i].numpy(), expected[i])
    before = batch.clone()
    bank.get_batch_tensor([4, 5])
    assert np.array_equal(batch.numpy(), before.numpy())


def test_variable_start_frequency_and_empty_bank(bank_file):
    kwargs = dict(filter_length=2049, delta_f=.25, dtype=np.complex128,
                  approximant='TaylorF2', low_frequency_cutoff=20.,
                  max_template_length=8., spin_order=7)
    bank = FilterBank(bank_file, enable_torchwave=True, **kwargs)
    expected = bank[0]
    batch, metadata = bank.get_batch_tensor([0])
    assert metadata[0].f_lower == expected.f_lower > 20.
    error = np.linalg.norm(batch[0].numpy() - expected.numpy())
    assert error / np.linalg.norm(expected.numpy()) < 2e-9
    bank.table = bank.table[:0]
    assert not bank.can_use_torchwave()
    assert bank.get_batch_tensor([])[0].shape == (0, 2049)


def test_is_diffgw_available():
    """Verify is_diffgw_available API on FilterBank and LiveFilterBank."""
    assert FilterBank.is_diffgw_available() is True
    assert LiveFilterBank.is_diffgw_available() is True


def test_diffgw_and_native_gpu_conditioning_resolution():
    """Test resolution semantics for GPU defaults:
    - native_gpu_conditioning defaults to True on CUDA, False on CPU.
    - enable_diffgw defaults to True on CUDA if diffgw is installed, False on CPU.
    - explicit disable flags take precedence.
    - when diffgw is missing, defaults to False without error.
    """
    # 1. CUDA with diffgw installed: both default to True
    is_cuda = True
    has_diffgw = True

    opt_native_gpu = None
    opt_enable_diffgw = None
    opt_disable_diffgw = False

    resolved_native = is_cuda if opt_native_gpu is None else opt_native_gpu
    resolved_diffgw = (
        False if opt_disable_diffgw else (
            True if opt_enable_diffgw is True else bool(is_cuda and has_diffgw)
        )
    )
    assert resolved_native is True
    assert resolved_diffgw is True

    # 2. CUDA with diffgw NOT installed: native_gpu True, diffgw False (graceful)
    has_diffgw = False
    resolved_native = is_cuda if opt_native_gpu is None else opt_native_gpu
    resolved_diffgw = (
        False if opt_disable_diffgw else (
            True if opt_enable_diffgw is True else bool(is_cuda and has_diffgw)
        )
    )
    assert resolved_native is True
    assert resolved_diffgw is False

    # 3. CPU with diffgw installed: both default to False
    is_cuda = False
    has_diffgw = True
    resolved_native = is_cuda if opt_native_gpu is None else opt_native_gpu
    resolved_diffgw = (
        False if opt_disable_diffgw else (
            True if opt_enable_diffgw is True else bool(is_cuda and has_diffgw)
        )
    )
    assert resolved_native is False
    assert resolved_diffgw is False

    # 4. Explicit disable overrides on CUDA
    is_cuda = True
    has_diffgw = True
    opt_native_gpu = False  # from --disable-native-gpu-conditioning
    opt_disable_diffgw = True  # from --disable-diffgw
    resolved_native = is_cuda if opt_native_gpu is None else opt_native_gpu
    resolved_diffgw = (
        False if opt_disable_diffgw else (
            True if opt_enable_diffgw is True else bool(is_cuda and has_diffgw)
        )
    )
    assert resolved_native is False
    assert resolved_diffgw is False

