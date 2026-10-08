# Copyright (C) 2026 PyCBC developers
# SPDX-License-Identifier: GPL-3.0-or-later
"""Original domain kernels remain independently selectable for validation."""
import os
import subprocess
import sys
from pathlib import Path

import numpy
import pytest

from pycbc import boundaries, conversions, cosmology, scheme, transforms
from pycbc.coordinates import base as coordinates

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')
jax.config.update('jax_enable_x64', True)
DEVICE = os.environ.get('PYCBC_TEST_SCHEME', 'jax:cpu').removeprefix('jax:')


def _exact(actual, expected):
    if isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        for value, target in zip(actual, expected):
            _exact(value, target)
        return
    actual = numpy.asarray(actual)
    expected = numpy.asarray(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize('dtype', [numpy.float32, numpy.float64])
@pytest.mark.parametrize('name,values,kwargs', [
    ('mchirp_from_mass1_mass2', ([30., 1.4], [20., 1.3]), {}),
    ('mass2_from_mchirp_mass1', ([16., 1.1], [30., 1.4]), {}),
    ('mass_from_knownmass_eta', ([30., 1.4], [.1, .24]), {}),
    ('lambda_tilde', ([30., 1.4], [20., 1.3], [300., 400.], [200., 250.]), {}),
    ('chi_eff_from_spherical', ([30., 1.4], [20., 1.3], [.2, .3], [.1, .2], [.4, .5], [.3, .6]), {}),
    ('hypertriangle', ([.1, .2], [.3, .4], [.6, .7]), {}),
    ('snr_from_loglr', ([.1, 1., -1., numpy.nan],), {}),
])
def test_conversion_reference_is_original(name, values, kwargs, dtype):
    function = getattr(conversions, name)
    inputs = tuple(numpy.asarray(value, dtype=dtype) for value in values)
    expected = function(*inputs, **kwargs)
    with scheme.JAXScheme(DEVICE, reference_operations=['conversions.' + name]) as context:
        inputs = tuple(jax.device_put(value, context.jax_device) for value in inputs)
        actual = function(*inputs, **kwargs)
        _exact(actual, expected)
        outputs = actual if isinstance(actual, (list, tuple)) else (actual,)
        assert all(value.device == context.jax_device for value in outputs)


@pytest.mark.parametrize('name,values', [
    ('cartesian_to_spherical', ([.1, 1., -.4], [.2, -.3, .5], [.4, .1, -.2])),
    ('spherical_to_cartesian', ([1., 2., 3.], [.1, .5, 1.], [.2, .6, .8])),
])
def test_coordinate_reference_is_original(name, values):
    function = getattr(coordinates, name)
    inputs = tuple(numpy.asarray(value) for value in values)
    expected = function(*inputs)
    with scheme.JAXScheme(DEVICE, reference_operations=['coordinates.' + name]):
        _exact(function(*(jnp.asarray(value) for value in inputs)), expected)


def test_reference_switch_is_independent(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('original root solver unexpectedly called')
    monkeypatch.setattr(conversions, '_mass2_from_mchirp_mass1_numpy', forbidden)
    with scheme.JAXScheme(DEVICE, reference_operations=['conversions.hypertriangle']):
        actual = conversions.mass2_from_mchirp_mass1(jnp.array([16.]), jnp.array([30.]))
        assert actual.shape == (1,)
    with scheme.JAXScheme(DEVICE, reference_operations=['conversions.mass2_from_mchirp_mass1']):
        with pytest.raises(AssertionError, match='original root solver'):
            conversions.mass2_from_mchirp_mass1(jnp.array([16.]), jnp.array([30.]))


def test_cosmology_reference_preserves_original_interpolator():
    converter = cosmology.DistToZ(numpoints=32)
    distances = numpy.array([10., 100., 1000., 10000.])
    expected = converter(distances)
    with scheme.JAXScheme(DEVICE, reference_operations=['cosmology.DistToZ.get_redshift']):
        _exact(converter(jnp.asarray(distances)), expected)
    volume_converter = cosmology.ComovingVolInterpolator('redshift', numpoints=32)
    volume_converter.setup_interpolant()
    volumes = volume_converter.cosmology.comoving_volume(numpy.array([.01, .5, 2.])).value
    expected = volume_converter(volumes)
    with scheme.JAXScheme(DEVICE, reference_operations=['cosmology']):
        _exact(volume_converter(jnp.asarray(volumes)), expected)


def test_transform_reference_preserves_input_identity_and_dtype():
    transform = transforms.Log('x', 'logx')
    values = numpy.array([.125, .75, 2.5], dtype=numpy.float32)
    expected = transform.transform({'x': values})['logx']
    labels = numpy.array(['first', 'second', 'third'])
    with scheme.JAXScheme(DEVICE, reference_operations=['transforms.Log.transform']):
        original = jnp.asarray(values)
        result = transform.transform({'x': original, 'labels': labels})
        assert result['x'] is original
        assert result['labels'] is labels
        _exact(result['logx'], expected)


def test_qnm_reference_retains_installed_original_contract():
    mass, spin = numpy.array([30., 60.]), numpy.array([.1, .2])
    try:
        expected = conversions.get_lm_f0tau(mass, spin, 2, 2)
    except AttributeError as error:
        # Some installed pykerr versions use the removed numpy.float alias.
        # Preserve that original failure; never repair it in a JAX branch.
        assert "float" in str(error)
        with scheme.JAXScheme(DEVICE, reference_operations=['conversions.get_lm_f0tau']):
            with pytest.raises(type(error)) as caught:
                conversions.get_lm_f0tau(jnp.asarray(mass), jnp.asarray(spin), 2, 2)
            assert str(caught.value) == str(error)
    else:
        with scheme.JAXScheme(DEVICE, reference_operations=['conversions.get_lm_f0tau']):
            _exact(conversions.get_lm_f0tau(jnp.asarray(mass), jnp.asarray(spin), 2, 2), expected)


def _tov(tmp_path):
    path = tmp_path / 'table.txt'
    numpy.savetxt(path, [[1., 1000.], [1.2, 600.], [1.6, 200.], [2., 50.]])
    return transforms.LambdaFromTOVFile(mass_param='mass1', lambda_param='lambda1',
                                       mass_lambda_file=path, redshift_mass=False)


def test_vector_tov_reference_replays_original_scalar_api(tmp_path):
    transform = _tov(tmp_path)
    masses = numpy.array([[.8, 1.1, 1.4], [1.6, 2., 3.]], dtype=numpy.float32)
    with pytest.raises(ValueError, match='truth value'):
        transform.transform({'mass1': masses})
    expected = numpy.asarray([transform.transform({'mass1': mass})['lambda1']
                              for mass in masses.flat]).reshape(masses.shape)
    with scheme.JAXScheme(DEVICE, reference_operations=['transforms.LambdaFromTOVFile.transform']):
        actual = transform.transform({'mass1': jnp.asarray(masses)})['lambda1']
        _exact(actual, expected)


def test_tov_first_compilation_does_not_leak_cached_tracers(tmp_path):
    transform = _tov(tmp_path)
    evaluate = lambda mass: transform.transform({'mass1': mass})['lambda1']
    with scheme.JAXScheme(DEVICE):
        masses = jnp.array([1.1, 1.4, 1.8])
        compiled = jax.jit(evaluate)(masses)
        eager = evaluate(masses)
        _exact(eager, compiled)
        assert all(not isinstance(value, jax.core.Tracer)
                   for values in transform._jax_data_cache.values() for value in values)


def test_static_domain_tables_follow_raw_device_after_first_compilation(tmp_path):
    devices = jax.devices('cpu')
    if len(devices) < 2:
        pytest.skip('requires two CPU devices')
    transform = _tov(tmp_path)
    converter = cosmology.DistToZ(numpoints=32)
    with scheme.JAXScheme('cpu:0'):
        initial = jax.device_put(numpy.array([1.1, 1.4]), devices[0])
        jax.jit(lambda mass: transform.transform({'mass1': mass})['lambda1'])(initial)
        transform.transform({'mass1': initial})
        converter(jax.device_put(numpy.array([100.]), devices[0]))
        raw = jax.device_put(numpy.array([1.1, 1.4]), devices[1])
        result = transform.transform({'mass1': raw})['lambda1']
        assert result.device == devices[1]
        result = converter(jax.device_put(numpy.array([100.]), devices[1]))
        assert result.device == devices[1]
        assert all(value.device == devices[1]
                   for value in transform._jax_data_cache[(raw.dtype, raw.sharding)])
        assert all(value.device == devices[1]
                   for value in converter._jax_grids[(raw.dtype, raw.sharding)])


def test_conditioned_membership_keeps_cpu_error_and_replays_scalar_original():
    bounds = boundaries.Bounds(0., 1., cyclic=True)
    values = numpy.array([-.2, .1, 1.2])
    with pytest.raises(ValueError, match='truth value'):
        bounds.contains_conditioned(values)
    expected = numpy.asarray([bounds.contains_conditioned(value) for value in values])
    with scheme.JAXScheme(DEVICE, reference_operations=['boundaries.Bounds.contains_conditioned']):
        _exact(bounds.contains_conditioned(jnp.asarray(values)), expected)


def test_reference_controls_reject_tracers_explicitly():
    with scheme.JAXScheme(DEVICE, reference_operations=['conversions']):
        with pytest.raises(TypeError, match='concrete JAX inputs'):
            jax.jit(conversions.mchirp_from_mass1_mass2)(jnp.array([30.]), jnp.array([20.]))


def test_real_mass_validation_retains_mixed_complex_inputs():
    with scheme.JAXScheme(DEVICE):
        with pytest.raises(TypeError, match='real'):
            conversions.mass2_from_mchirp_mass1(jnp.array([16.]), jnp.array([30.+1.j]))


def test_reference_rejects_precision_truncation():
    with scheme.JAXScheme(DEVICE):
        with jax.enable_x64(False):
            chirp, mass = jnp.array([16.], dtype=jnp.float32), jnp.array([30.], dtype=jnp.float32)
            assert conversions.mass2_from_mchirp_mass1(chirp, mass).dtype == jnp.float32
    with scheme.JAXScheme(DEVICE, reference_operations=['conversions.mass2_from_mchirp_mass1']):
        with jax.enable_x64(False):
            with pytest.raises(RuntimeError, match='jax_enable_x64=True'):
                conversions.mass2_from_mchirp_mass1(chirp, mass)


@pytest.mark.parametrize('direction', ['transform', 'inverse_transform'])
def test_precession_array_order_is_original(direction):
    transform = transforms.PrecessionMassSpinToCartesianSpin()
    maps = {'mass1': numpy.array([5., 10., 4.]), 'mass2': numpy.array([8., 6., 7.])}
    if direction == 'transform':
        maps.update(xi1=numpy.array([.1, .2, .3]), xi2=numpy.array([.3, .2, .1]),
                    phi_a=numpy.array([.1, .2, .3]), phi_s=numpy.array([.4, .5, .6]))
        outputs = ('spin1x', 'spin1y', 'spin2x', 'spin2y')
    else:
        maps.update(spin1x=numpy.array([.1, .2, .3]), spin2x=numpy.array([.3, .2, .1]),
                    spin1y=numpy.array([.4, .5, .6]), spin2y=numpy.array([.1, .2, .3]))
        outputs = ('xi1', 'xi2')
    function = getattr(transform, direction)
    expected = function(maps)
    with scheme.JAXScheme(DEVICE):
        actual = function({name: jnp.asarray(value) for name, value in maps.items()})
        for name in outputs:
            numpy.testing.assert_allclose(actual[name], expected[name], rtol=1e-14)
    with scheme.JAXScheme(DEVICE, reference_operations=['transforms.PrecessionMassSpinToCartesianSpin.' + direction]):
        actual = function({name: jnp.asarray(value) for name, value in maps.items()})
        for name in outputs:
            _exact(actual[name], expected[name])


def test_precession_scalar_equal_mass_tie_is_original():
    transform = transforms.PrecessionMassSpinToCartesianSpin()
    maps = dict(mass1=5., mass2=5., xi1=.1, xi2=.3, phi_a=0., phi_s=0.)
    expected = transform.transform(maps)
    assert expected['spin1x'] == .3
    with scheme.JAXScheme(DEVICE):
        maps['xi1'] = jnp.asarray(maps['xi1'])
        actual = transform.transform(maps)
        assert float(actual['spin1x']) == expected['spin1x']


def test_public_logit_compilation_preserves_valid_gradients_and_eager_errors():
    transform = transforms.Logit('x', 'logitx')
    values = numpy.array([.1, .3, .7])
    expected = numpy.asarray([transform.jacobian({'x': value}) for value in values])
    with scheme.JAXScheme(DEVICE):
        raw = jnp.asarray(values)
        loss = lambda value: jnp.sum(transform.transform({'x': value})['logitx'])
        numpy.testing.assert_allclose(jax.jit(jax.grad(loss))(raw), expected, rtol=1e-14)
        numpy.testing.assert_allclose(jax.jit(lambda value: transform.jacobian({'x': value}))(raw),
                                      expected, rtol=1e-14)
        with pytest.raises(ValueError, match='in bounds'):
            transform.transform({'x': jnp.array([-1., .5])})
        with pytest.raises(ValueError, match='in bounds'):
            transform.jacobian({'x': jnp.array([-1., .5])})
    with pytest.raises(ValueError, match='truth value'):
        transform.jacobian({'x': values})
    with scheme.JAXScheme(DEVICE, reference_operations=['transforms.Logit.jacobian']):
        _exact(transform.jacobian({'x': jnp.asarray(values)}), expected)


def test_cpu_domain_imports_do_not_require_optional_jax():
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run([sys.executable, '-c',
        'import sys; from pycbc import scheme, boundaries, conversions, cosmology, transforms; '
        'assert "jax" not in sys.modules'], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
