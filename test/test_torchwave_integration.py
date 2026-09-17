# Copyright (C) 2026
# This program is free software under the GNU General Public License,
# version 3 or (at your option) any later version.
"""Independent complex-waveform and dispatch tests for the bank provider."""

import h5py
import numpy as np
import pytest

from pycbc import scheme
from pycbc.types import FrequencySeries
from pycbc.waveform.bank import FilterBank


torch = pytest.importorskip('torch')
pytest.importorskip('torchwave')
DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def write_bank(path, rows):
    """Write each waveform parameter, including per-row options, to a bank."""
    with h5py.File(path, 'w') as bank:
        for name in rows[0]:
            values = [r[name] for r in rows]
            if isinstance(values[0], str):
                values = np.array(values, dtype=h5py.string_dtype())
            bank[name] = values
        bank.attrs['parameters'] = list(rows[0])
    return str(path)


def row(mass1=1.4, mass2=1.2, **options):
    return dict(mass1=mass1, mass2=mass2, spin1z=0., spin2z=0.,
                f_lower=30.11, f_final=400.11, approximant='TaylorF2',
                **options)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('storage', [torch.complex64, torch.complex128])
def test_bank_complex_parity_and_metadata(tmp_path, device, storage):
    # Finite sampled coverage, including limits of the runtime allowlist.
    # These fixtures do not validate every point in the admitted domain.
    rows = []
    for i, (m1, m2, s1, s2, flow, fend) in enumerate([
            (1.4, 1.2, 0., 0., 30.11, 400.11),
            (25., 20., .2, -.2, 20., 400.),
            (1., 100., -.9, .9, 10., 4096.),
            (100., 1., .99, -.99, 11.03, 512.),
            (2., 1., .02, -.02, 40., 80.11)]):
        p = row(m1, m2, coa_phase=.3 * i, inclination=.4 * i)
        p.update(spin1z=s1, spin2z=s2, f_lower=flow, f_final=fend)
        rows.append(p)
    filename = write_bank(tmp_path / 'bank.hdf', rows)
    kwargs = dict(filter_length=16385, delta_f=.25, dtype=np.complex128)
    reference_bank = FilterBank(filename, enable_torchwave=False, **kwargs)
    order = [4, 0, 2, 1, 3, 0]
    references = [reference_bank[i] for i in order]
    samples = [r.numpy().copy() for r in references]
    bank = FilterBank(filename, enable_torchwave=True, **kwargs)
    with scheme.TorchScheme(device=device):
        batch, templates = bank.get_batch_tensor(order, device, storage)
        assert batch.dtype == storage
        assert batch.device.type == device
        for i, (template, reference, expected) in enumerate(
                zip(templates, references, samples)):
            assert template.waveform_provider == 'torchwave'
            assert template.data.tensor.data_ptr() == batch[i].data_ptr()
            actual = batch[i].detach().cpu().numpy()
            rel = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
            assert rel < (2e-7 if storage == torch.complex64 else 2e-9)
            assert np.array_equal(actual != 0, expected != 0)
            for name in ('f_lower', 'min_f_lower', 'end_idx', 'chirp_length',
                         'length_in_time', 'approximant', 'end_frequency',
                         'epoch'):
                assert getattr(template, name) == getattr(reference, name)
            assert (template.params.template_hash
                    == reference.params.template_hash)
    # Exercise the actual norm consumer after the intentional CPU boundary.
    psd_dtype = np.float32 if storage == torch.complex64 else np.float64
    psd_values = np.linspace(1., 3., kwargs['filter_length'], dtype=psd_dtype)
    psd = FrequencySeries(psd_values, delta_f=kwargs['delta_f'])
    reference_psd = FrequencySeries(psd_values.astype(np.float64),
                                    delta_f=kwargs['delta_f'])
    for template, reference in zip(templates, references):
        assert template.sigmasq(psd) == pytest.approx(
            reference.sigmasq(reference_psd), rel=3e-7)


@pytest.mark.parametrize('device', DEVICES)
def test_inspiral_cli_bank_arguments_use_native_provider(tmp_path, device):
    filename = write_bank(tmp_path / 'cli.hdf', [row(24., 18.), row(20., 15.)])
    # Mirror bin/pycbc_inspiral's FilterBank construction for a tiled search.
    kwargs = dict(filter_length=2049, delta_f=.25,
                  low_frequency_cutoff=30., dtype=np.complex64,
                  phase_order=-1, taper=None, approximant=['TaylorF2'],
                  out=None, max_template_length=None,
                  enable_compressed_waveforms=False,
                  waveform_decompression_method=None)
    reference = FilterBank(filename, enable_torchwave=False, **kwargs)
    expected = [reference[i].numpy().copy() for i in range(len(reference))]
    without_taper = dict(kwargs)
    without_taper.pop('taper')
    untapered = FilterBank(filename, enable_torchwave=False, **without_taper)
    for i, samples in enumerate(expected):
        np.testing.assert_array_equal(samples, untapered[i].numpy())

    bank = FilterBank(filename, enable_torchwave=True, **kwargs)
    assert bank.can_use_torchwave()
    with scheme.TorchScheme(device=device):
        batch, templates = bank.get_batch_tensor([0, 1], device=device)
        assert batch.dtype == torch.complex64
        assert batch.device.type == device
        for i, template in enumerate(templates):
            assert template.waveform_provider == 'torchwave'
            actual = batch[i].detach().cpu().numpy()
            assert np.linalg.norm(actual - expected[i]) / np.linalg.norm(
                expected[i]) < 2e-7
    for taper in ('start', 'end', 'startend'):
        bank.extra_args['taper'] = taper
        assert not bank.can_use_torchwave()
        for decision in bank.torchwave_diagnostics(device=device):
            assert decision['provider'] == 'reference'
            assert 'taper' in decision['reason']


def test_global_precedence_and_heterogeneous_fallback(tmp_path):
    rows = []
    for i, (approx, phase_order, f_ref, tidal) in enumerate([
            ('TaylorF2', -1, 0., 0.), ('IMRPhenomPv2', -1, 0., 0.),
            ('TaylorF2', 4, 0., 0.), ('TaylorF2', -1, 30., 0.),
            ('TaylorF2', -1, 0., 100.)]):
        p = row(8. + i, 2., coa_phase=.1, inclination=.2,
                phase_order=phase_order, f_ref=f_ref, lambda1=tidal)
        p['approximant'] = approx
        rows.append(p)
    filename = write_bank(tmp_path / 'mixed.hdf', rows)
    kwargs = dict(filter_length=2049, delta_f=.25, dtype=np.complex128,
                  coa_phase=.9, inclination=1.1)
    reference = FilterBank(filename, enable_torchwave=False, **kwargs)
    bank = FilterBank(filename, enable_torchwave=True, **kwargs)
    order = [3, 0, 2, 4, 1, 0]
    batch, metadata = bank.get_batch_tensor(order)
    assert [t.waveform_provider for t in metadata] == [
        'reference', 'torchwave', 'reference', 'reference', 'reference',
        'torchwave']
    for i, index in enumerate(order):
        expected = reference[index].numpy()
        assert np.linalg.norm(batch[i].numpy() - expected) / np.linalg.norm(
            expected) < 2e-9
    # Global options override row values, including provider eligibility.
    bank.extra_args['phase_order'] = 4
    assert all(d['provider'] == 'reference'
               for d in bank.torchwave_diagnostics())


def test_opt_in_dtype_empty_and_option_diagnostics(tmp_path):
    filename = write_bank(tmp_path / 'one.hdf', [row()])
    args = dict(filter_length=2049, delta_f=.25, dtype=np.complex128)
    bank = FilterBank(filename, **args)
    assert not bank.can_use_torchwave()
    bank.enable_torchwave = False
    assert not bank.can_use_torchwave()
    bank.enable_torchwave = True
    assert bank.can_use_torchwave()
    empty, templates = bank.get_batch_tensor([])
    assert empty.shape == (0, 2049) and empty.dtype == torch.complex128
    assert templates == []
    with pytest.raises(TypeError, match='storage dtype'):
        bank.get_batch_tensor([], dtype=torch.float64)
    with pytest.raises(IndexError):
        bank.get_batch_tensor([1])
    with pytest.raises(ValueError, match='matching TorchScheme'):
        bank.get_batch_tensor([0], device='cuda')
    for options, reason in [({'mode_array': [(2, 2)]}, 'mode_array'),
                            ({'spin1x': .1}, 'spin1x'),
                            ({'unknown_option': 1}, 'unknown_option'),
                            ({'eccentricity': .1}, 'eccentricity')]:
        bank.extra_args = options
        decision = bank.torchwave_diagnostics()[0]
        assert decision['provider'] == 'reference'
        assert reason in decision['reason']
    bank.extra_args = {}
    bank.has_compressed_waveforms = True
    assert 'compressed' in bank.torchwave_diagnostics()[0]['reason']


def test_missing_optional_provider_uses_reference(tmp_path, monkeypatch):
    from pycbc.waveform import torchwave as provider
    filename = write_bank(tmp_path / 'one.hdf', [row()])
    bank = FilterBank(filename, filter_length=2049, delta_f=.25,
                      dtype=np.complex64, enable_torchwave=True)
    monkeypatch.setattr(provider.importlib.util, 'find_spec', lambda _: None)
    assert not bank.can_use_torchwave()
    batch, templates = bank.get_batch_tensor([0])
    assert templates[0].waveform_provider == 'reference'
    assert np.array_equal(batch[0].numpy(), bank[0].numpy())


def test_batch_identity_and_cached_metadata_views(tmp_path):
    from pycbc.waveform.torchwave import template_metadata
    filename = write_bank(tmp_path / 'cache.hdf', [row(), row(2., 1.3)])
    bank = FilterBank(filename, filter_length=2049, delta_f=.25,
                      dtype=np.complex64, enable_torchwave=True)
    key = bank.waveform_batch_key([0, 1])
    with scheme.TorchScheme(device='cpu'):
        samples, templates = bank.get_batch_tensor([0, 1])
        assert key == bank.waveform_batch_key([0, 1])
        records = template_metadata(templates)
        cloned = samples.clone()
        views = bank.wrap_batch_tensor([0, 1], cloned, records)
        for position, view in enumerate(views):
            assert view.data.tensor.data_ptr() == cloned[position].data_ptr()
            assert view is not templates[position]
            assert view._sigmasq is not templates[position]._sigmasq
            assert view.params.template_hash == templates[
                position].params.template_hash
        assert all(not isinstance(v, (torch.Tensor, FrequencySeries))
                   for record in records for v in record.values())
        # A later shard constructs a fresh bank before reusing cached samples.
        rebound = FilterBank(filename, filter_length=2049, delta_f=.25,
                             dtype=np.complex64, enable_torchwave=True)
        assert np.all(rebound.table.template_duration == 0)
        rebound_views = rebound.wrap_batch_tensor([0, 1], cloned, records)
        for view, original in zip(rebound_views, templates):
            assert view.params.template_duration == (
                original.params.template_duration)
            assert view.chirp_length == original.chirp_length
        views[0][:] = 0
        assert torch.count_nonzero(samples[0]) > 0
    bank.extra_args['coa_phase'] = 1.
    assert key != bank.waveform_batch_key([0, 1])
    bank.extra_args = {}
    bank.table[0].spin1z = .2
    assert key != bank.waveform_batch_key([0, 1])
    with pytest.raises(ValueError, match='must agree'):
        bank.wrap_batch_tensor([0], cloned, records)
