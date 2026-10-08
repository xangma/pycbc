# Copyright (C) 2026 PyCBC contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""diffGW dispatch and exact, independent original waveform validation."""

import gc
import os
import subprocess
import sys

import numpy as np
import pytest

from pycbc import scheme
from pycbc.types import Array, zeros
from pycbc.types.array_jax import to_jax
from pycbc.waveform import waveform

pytest.importorskip('jax')

DEVICE = os.environ.get('PYCBC_TEST_SCHEME', 'jax:cpu').removeprefix('jax:')
PARAMETERS = dict(mass1=30., mass2=20., spin1z=.1, spin2z=-.2,
                  distance=100., coa_phase=.2, inclination=.3,
                  delta_f=2., f_lower=20., f_final=128.)


def _evaluate(function, approximant):
    parameters = dict(PARAMETERS, approximant=approximant)
    if function == 'get_fd_waveform_sequence':
        parameters['sample_points'] = np.array([21., 25.5, 70.])
    if function == 'get_td_waveform':
        parameters['delta_t'] = 1 / 512
    if function == 'get_waveform_filter':
        return (waveform.get_waveform_filter(
            zeros(65, dtype=np.complex64), **parameters),)
    return getattr(waveform, function)(**parameters)


@pytest.mark.parametrize('function,approximant', [
    ('get_fd_waveform', 'TaylorF2'),
    ('get_fd_waveform', 'IMRPhenomD'),
    ('get_fd_waveform', 'IMRPhenomXAS'),
    ('get_fd_waveform_sequence', 'TaylorF2'),
    ('get_fd_waveform_sequence', 'IMRPhenomD'),
    ('get_fd_waveform_sequence', 'IMRPhenomXAS'),
    ('get_td_waveform', 'IMRPhenomD'),
    ('get_waveform_filter', 'SPAtmplt'),
])
def test_original_waveform_is_exact_and_resident(function, approximant, monkeypatch):
    with scheme.CPUScheme():
        expected = _evaluate(function, approximant)
    from pycbc.waveform import diffgw_jax
    monkeypatch.setattr(diffgw_jax, '_native_polarizations',
                        lambda *a, **k: pytest.fail('native diffGW was selected'))
    if function in ('get_fd_waveform', 'get_fd_waveform_sequence'):
        import pycbc.reference_jax as reference
        monkeypatch.setattr(reference, 'cpu_reference',
                            lambda *a, **k: pytest.fail('pure FD route used worker'))
    with scheme.JAXScheme(DEVICE, reference_operations=('waveform',)) as context:
        state = scheme.mgr.state
        actual = _evaluate(function, approximant)
        assert scheme.mgr.state is state
        for result, reference in zip(actual, expected, strict=True):
            assert to_jax(result).devices() == {context.jax_device}
            assert result.dtype == reference.dtype
            assert result.shape == reference.shape
            assert np.asarray(result).tobytes() == reference.numpy().tobytes()
            for name in ('delta_f', 'delta_t', 'epoch', 'chirp_length',
                         'length_in_time', 'time_offset'):
                if hasattr(reference, name):
                    assert getattr(result, name) == getattr(reference, name)


@pytest.mark.parametrize('approximant', ['TaylorF2', 'IMRPhenomD', 'IMRPhenomXAS'])
def test_default_calls_public_diffgw_only(approximant, monkeypatch):
    provider = pytest.importorskip('diffgw')
    import pycbc.reference_jax as reference
    calls = []
    original = provider.get_fd_waveform

    def generate(*args, **kwargs):
        calls.append(kwargs['backend'])
        return original(*args, **kwargs)

    monkeypatch.setattr(provider, 'get_fd_waveform', generate)
    monkeypatch.setattr(reference, 'cpu_reference',
                        lambda *a, **k: pytest.fail('CPU waveform worker was selected'))
    with scheme.JAXScheme(DEVICE) as context:
        hp, hc = _evaluate('get_fd_waveform', approximant)
        assert to_jax(hp).devices() == {context.jax_device}
        assert to_jax(hc).devices() == {context.jax_device}
        assert np.any(np.asarray(hp))
    assert calls == ['jax']


@pytest.mark.parametrize('approximant', ['IMRPhenomD', 'IMRPhenomXAS'])
@pytest.mark.parametrize('f_lower', [20., 22.])
def test_default_phenom_preserves_original_reference_phase(approximant, f_lower):
    pytest.importorskip('diffgw')
    parameters = dict(PARAMETERS, approximant=approximant, f_lower=f_lower)
    with scheme.CPUScheme():
        expected = waveform.get_fd_waveform(**parameters)
    with scheme.JAXScheme(DEVICE):
        actual = waveform.get_fd_waveform(**parameters)
        for result, reference in zip(actual, expected, strict=True):
            mask = (reference.numpy() != 0) & (result.numpy() != 0)
            ratio = result.numpy()[mask] / reference.numpy()[mask]
            assert np.max(np.abs(np.angle(ratio))) < 1e-9
            assert np.max(np.abs(np.abs(ratio) - 1)) < 1e-9


@pytest.mark.parametrize('approximant', ['IMRPhenomD', 'IMRPhenomXAS'])
def test_default_sequence_preserves_original_reference_phase(approximant):
    pytest.importorskip('diffgw')
    with scheme.CPUScheme():
        expected = _evaluate('get_fd_waveform_sequence', approximant)
    with scheme.JAXScheme(DEVICE):
        actual = _evaluate('get_fd_waveform_sequence', approximant)
        for result, reference in zip(actual, expected, strict=True):
            ratio = result.numpy() / reference.numpy()
            assert np.max(np.abs(np.angle(ratio))) < 1e-9
            assert np.max(np.abs(np.abs(ratio) - 1)) < 1e-9


@pytest.mark.parametrize('first_import', ['numpy', 'pycbc'])
def test_original_waveform_preserves_cpu_import_order(first_import):
    """The worker must share the original native-library loading boundary."""
    source = f"""
import {first_import}
import numpy as np
from pycbc import scheme
from pycbc.waveform import get_fd_waveform
parameters = {dict(PARAMETERS, approximant='IMRPhenomXAS')!r}
with scheme.CPUScheme():
    expected = get_fd_waveform(**parameters)
with scheme.JAXScheme('cpu', reference_operations=('waveform',)):
    actual = get_fd_waveform(**parameters)
    for result, reference in zip(actual, expected, strict=True):
        assert result.numpy().tobytes() == reference.numpy().tobytes()
"""
    subprocess.run([sys.executable, '-c', source], check=True,
                   capture_output=True, text=True)


@pytest.mark.parametrize('options', [dict(approximant='SPAtmplt'),
                                    dict(approximant='IMRPhenomPv2'),
                                    dict(approximant='TaylorF2', f_ref=30.),
                                    dict(approximant='TaylorF2', phase_order=0)])
def test_unsupported_request_never_substitutes(options):
    pytest.importorskip('diffgw')
    with scheme.JAXScheme(DEVICE), pytest.raises(ValueError, match='does not support request'):
        waveform.get_fd_waveform(**dict(PARAMETERS, **options))


def test_filter_preserves_output_view_and_input_parameters():
    pytest.importorskip('diffgw')
    parameters = dict(PARAMETERS, approximant='TaylorF2')
    with scheme.JAXScheme(DEVICE):
        storage = Array(np.full(69, 3 + 4j, dtype=np.complex64))
        output = storage[2:-2]
        result = waveform.get_waveform_filter(output, **parameters)
        assert result._data is output._data
        assert np.asarray(storage)[:2].tobytes() == np.full(2, 3 + 4j, np.complex64).tobytes()
        assert np.asarray(storage)[-2:].tobytes() == np.full(2, 3 + 4j, np.complex64).tobytes()
        assert np.any(np.asarray(result))
    assert parameters == dict(PARAMETERS, approximant='TaylorF2')



@pytest.mark.parametrize('function', ['get_waveform_filter',
                                     'get_two_pol_waveform_filter'])
@pytest.mark.parametrize('approximant', ['TaylorF2', 'IMRPhenomD', 'IMRPhenomXAS'])
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_original_fd_filter_uses_caller_environment(function, approximant, dtype, monkeypatch):
    parameters = dict(PARAMETERS, approximant=approximant)

    def evaluate():
        parents = [Array(np.full(69, 3 + 4j, dtype=dtype))
                   for _ in range(1 if function == 'get_waveform_filter' else 2)]
        outputs = [parent[2:-2] for parent in parents]
        if function == 'get_waveform_filter':
            results = (waveform.get_waveform_filter(outputs[0], **parameters),)
        else:
            results = waveform.get_two_pol_waveform_filter(*outputs, None, **parameters)
        return parents, outputs, results

    with scheme.CPUScheme():
        _, _, expected = evaluate()
    import pycbc.reference_jax as reference
    monkeypatch.setattr(reference, 'cpu_reference',
                        lambda *a, **k: pytest.fail('pure FD filter used worker'))
    with scheme.JAXScheme(DEVICE, reference_operations=('waveform',)) as context:
        state = scheme.mgr.state
        parents, outputs, actual = evaluate()
        assert scheme.mgr.state is state
        for parent, output, result, original in zip(parents, outputs, actual, expected, strict=True):
            assert result._data is output._data
            assert to_jax(result).devices() == {context.jax_device}
            assert result.dtype == original.dtype
            assert result.numpy().tobytes() == original.numpy().tobytes()
            assert result._epoch == original._epoch
            assert result.delta_f == original.delta_f
            assert result.chirp_length == original.chirp_length
            assert result.length_in_time == original.length_in_time
            assert parent[:2].numpy().tobytes() == np.full(2, 3 + 4j, dtype).tobytes()
            assert parent[-2:].numpy().tobytes() == np.full(2, 3 + 4j, dtype).tobytes()


@pytest.mark.parametrize('native', [False, True])
def test_two_pol_filter_preserves_template_inclination_alias(native):
    from types import SimpleNamespace
    pytest.importorskip('diffgw')
    parameters = dict(PARAMETERS, approximant='TaylorF2')
    parameters.pop('inclination')
    operations = ('waveform',) if native else ()
    with scheme.JAXScheme(DEVICE, reference_operations=operations):
        actual = waveform.get_two_pol_waveform_filter(
            zeros(65, dtype=np.complex128), zeros(65, dtype=np.complex128),
            SimpleNamespace(alpha3=.3), **parameters)
        expected = waveform.get_two_pol_waveform_filter(
            zeros(65, dtype=np.complex128), zeros(65, dtype=np.complex128),
            None, inclination=.3, **parameters)
        for result, original in zip(actual, expected, strict=True):
            assert result.numpy().tobytes() == original.numpy().tobytes()


def test_availability_matches_selected_generation_provider():
    pytest.importorskip('diffgw')
    from pycbc.waveform import waveform_modes
    cpu = scheme.CPUScheme()
    with cpu:
        expected = (waveform.fd_approximants(), waveform.td_approximants(),
                    waveform.filter_approximants(),
                    waveform_modes.fd_waveform_mode_approximants(),
                    waveform_modes.td_waveform_mode_approximants())
    for native in (False, True):
        context = scheme.JAXScheme(DEVICE, reference_operations=('waveform',) if native else ())
        explicit = (waveform.fd_approximants(context), waveform.td_approximants(context),
                    waveform.filter_approximants(context))
        supported = (['TaylorF2', 'IMRPhenomD', 'IMRPhenomXAS'], [],
                     ['TaylorF2', 'IMRPhenomD', 'IMRPhenomXAS'])
        assert explicit == (expected[:3] if native else supported)
        with context:
            actual = (waveform.fd_approximants(), waveform.td_approximants(),
                      waveform.filter_approximants(),
                      waveform_modes.fd_waveform_mode_approximants(),
                      waveform_modes.td_waveform_mode_approximants())
            assert actual == (expected if native else (*explicit, [], []))
            assert waveform.fd_approximants(cpu) == expected[0]
            assert waveform.td_approximants(cpu) == expected[1]
            assert waveform.filter_approximants(cpu) == expected[2]


@pytest.mark.parametrize('function', ['get_fd_waveform', 'get_fd_waveform_sequence',
                                     'get_waveform_filter', 'get_two_pol_waveform_filter'])
def test_original_fd_dispatch_preserves_scheme_shared_state(function, monkeypatch):
    with scheme.JAXScheme(DEVICE, reference_operations=('waveform',)):
        state = scheme.mgr.state
        marker = object()
        monkeypatch.setattr(scheme.Scheme, '_single', marker)
        parameters = dict(PARAMETERS, approximant='IMRPhenomXAS')
        if function == 'get_two_pol_waveform_filter':
            waveform.get_two_pol_waveform_filter(
                zeros(65, dtype=np.complex128), zeros(65, dtype=np.complex128),
                None, **parameters)
        else:
            _evaluate(function, 'IMRPhenomXAS')
        gc.collect()
        assert scheme.mgr.state is state
        assert scheme.Scheme._single is marker


def test_cpu_waveform_import_does_not_require_optional_backends():
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'jax', 'jaxwave', 'diffgw'}:
        raise AssertionError('CPU waveform import loaded optional backend: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import pycbc.waveform
assert callable(pycbc.waveform.get_fd_waveform)
from pycbc.waveform import waveform_modes
assert pycbc.waveform.fd_approximants()
assert pycbc.waveform.td_approximants()
assert pycbc.waveform.filter_approximants()
assert waveform_modes.fd_waveform_mode_approximants()
assert waveform_modes.td_waveform_mode_approximants()
"""
    subprocess.run([sys.executable, '-c', code], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
